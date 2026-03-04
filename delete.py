#!/usr/bin/env python3
"""
delete.py — List and selectively delete resources created by deploy.py

Usage:
    python3 delete.py                          # auto-discovers all deployment JSONs
    python3 delete.py --file my-server.json    # target a specific deployment file
    python3 delete.py --region eu-west-2       # filter by region
    python3 delete.py --yes                    # skip confirmation prompts
"""
import argparse
import glob
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3


DELETION_LOG = Path("deletion-log.json")


def parse_args():
    p = argparse.ArgumentParser(description="Delete MCP server AgentCore resources")
    p.add_argument("--file", help="Specific deployment JSON file")
    p.add_argument("--region", help="Filter deployments by region")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    return p.parse_args()


def load_deployments(file_arg, region_filter):
    files = [file_arg] if file_arg else sorted(glob.glob("*-deployment.json"))
    deployments = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
            if region_filter and d.get("region") != region_filter:
                continue
            d["_file"] = f
            deployments.append(d)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")
    return deployments


def find_shared_resources(deployments):
    """Return sets of resource IDs that appear in more than one deployment."""
    from collections import Counter
    pool_counts = Counter(d.get("cognito_pool_id") for d in deployments if d.get("cognito_pool_id"))
    gateway_counts = Counter(d.get("gateway_id") for d in deployments if d.get("gateway_id"))
    shared_pools = {k for k, v in pool_counts.items() if v > 1}
    shared_gateways = {k for k, v in gateway_counts.items() if v > 1}
    return shared_pools, shared_gateways


def print_resources(d, shared_pools, shared_gateways):
    print(f"\n  Server:       {d.get('server_name')}  ({d.get('region')})")
    print(f"  File:         {d.get('_file')}")
    print()

    def tag(val, shared_set):
        return f"{val}  ⚠ SHARED — will be skipped" if val in shared_set else val

    print(f"  AgentCore:")
    print(f"    Gateway:        {tag(d.get('gateway_id', '-'), shared_gateways)}")
    print(f"    Gateway Target: {d.get('gateway_target_id', '-')}")
    print(f"    Runtime:        {d.get('runtime_id', '-')}")
    print(f"    OAuth Provider: {d.get('oauth_provider_arn', '-')}")
    print(f"  ECR Repo:     {d.get('ecr_repo_name', '-')}")
    print(f"  IAM Roles:    {d.get('iam_role_name', '-')}, {d.get('iam_gateway_role_name', '-')}")
    print(f"  Cognito Pool: {tag(d.get('cognito_pool_id', '-'), shared_pools)}")


def log_deletion(server_name, region, resource_type, resource_id, status, detail=""):
    """Append a deletion record to deletion-log.json."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_name": server_name,
        "region": region,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "status": status,
    }
    if detail:
        record["detail"] = detail

    existing = []
    if DELETION_LOG.exists():
        try:
            existing = json.loads(DELETION_LOG.read_text())
        except Exception:
            existing = []
    existing.append(record)
    DELETION_LOG.write_text(json.dumps(existing, indent=2))


def delete_resources(d, shared_pools, shared_gateways):
    region = d["region"]
    server = d["server_name"]
    ac = boto3.client("bedrock-agentcore-control", region_name=region)
    iam = boto3.client("iam", region_name=region)
    ecr = boto3.client("ecr", region_name=region)
    cognito = boto3.client("cognito-idp", region_name=region)

    deleted, skipped, failed = [], [], []

    def attempt(label, resource_id, fn):
        try:
            fn()
            deleted.append(f"  ✓ {label}: {resource_id}")
            log_deletion(server, region, label, resource_id, "deleted")
        except Exception as e:
            failed.append(f"  ✗ {label}: {resource_id} — {e}")
            log_deletion(server, region, label, resource_id, "failed", str(e))

    def skip(label, resource_id, reason):
        skipped.append(f"  ⚠ {label}: {resource_id} — {reason}")
        log_deletion(server, region, label, resource_id, "skipped", reason)

    # Gateway Target (always server-specific)
    if d.get("gateway_id") and d.get("gateway_target_id"):
        gw_id, tgt_id = d["gateway_id"], d["gateway_target_id"]
        attempt("Gateway Target", tgt_id,
                lambda: ac.delete_gateway_target(gatewayIdentifier=gw_id, targetId=tgt_id))
        time.sleep(5)

    # Gateway (skip if shared)
    if d.get("gateway_id"):
        gw_id = d["gateway_id"]
        if gw_id in shared_gateways:
            skip("Gateway", gw_id, "shared with other servers")
        else:
            attempt("Gateway", gw_id, lambda: ac.delete_gateway(gatewayIdentifier=gw_id))

    # OAuth Provider (always server-specific)
    if d.get("oauth_provider_arn"):
        name = d["oauth_provider_arn"].split("/")[-1]
        attempt("OAuth Provider", name,
                lambda: ac.delete_oauth2_credential_provider(name=name))

    # Runtime (always server-specific)
    if d.get("runtime_id"):
        rid = d["runtime_id"]
        attempt("Runtime", rid, lambda: ac.delete_agent_runtime(agentRuntimeId=rid))

    # IAM roles (always server-specific)
    for role_name in [d.get("iam_role_name"), d.get("iam_gateway_role_name")]:
        if not role_name:
            continue
        def _delete_role(r=role_name):
            for p in iam.list_role_policies(RoleName=r)["PolicyNames"]:
                iam.delete_role_policy(RoleName=r, PolicyName=p)
            iam.delete_role(RoleName=r)
        attempt(f"IAM Role", role_name, _delete_role)

    # ECR repo (always server-specific)
    if d.get("ecr_repo_name"):
        repo = d["ecr_repo_name"]
        attempt("ECR Repo", repo,
                lambda: ecr.delete_repository(repositoryName=repo, force=True))

    # Cognito pool (skip if shared)
    if d.get("cognito_pool_id"):
        pool_id = d["cognito_pool_id"]
        if pool_id in shared_pools:
            skip("Cognito Pool", pool_id, "shared with other servers")
        else:
            def _delete_pool():
                domain = d.get("cognito_domain")
                if domain:
                    cognito.delete_user_pool_domain(Domain=domain, UserPoolId=pool_id)
                cognito.delete_user_pool(UserPoolId=pool_id)
            attempt("Cognito Pool", pool_id, _delete_pool)

    # Print summary
    print("\n  Deleted:")
    for r in deleted or ["    (none)"]:
        print(r)
    if skipped:
        print("\n  Skipped (shared resources):")
        for r in skipped:
            print(r)
    if failed:
        print("\n  Failed:")
        for r in failed:
            print(r)

    # Remove deployment file if fully deleted (no failures, no skips)
    if not failed and not skipped:
        try:
            Path(d["_file"]).unlink()
            print(f"\n  ✓ Removed {d['_file']}")
        except Exception:
            pass
    elif deleted:
        # Partially deleted — update the deployment JSON to remove deleted resources
        _remove_deleted_from_json(d, deleted)
        print(f"\n  ✓ Updated {d['_file']} (removed deleted resources)")

    print(f"\n  Deletion log appended to: {DELETION_LOG}")


def _remove_deleted_from_json(d, deleted_lines):
    """Remove deleted resource fields from the deployment JSON."""
    # Map resource type labels to JSON keys
    type_to_keys = {
        "Gateway Target":  ["gateway_target_id"],
        "Gateway":         ["gateway_id", "gateway_url"],
        "OAuth Provider":  ["oauth_provider_arn"],
        "Runtime":         ["runtime_id", "runtime_arn"],
        "IAM Role":        [],  # handled by role name matching below
        "ECR Repo":        ["ecr_uri", "ecr_repo_name"],
        "Cognito Pool":    ["cognito_pool_id", "cognito_domain", "token_url",
                            "client_id", "client_secret"],
    }
    updated = dict(d)
    updated.pop("_file", None)

    for line in deleted_lines:
        # line format: "  ✓ ResourceType: resource_id"
        parts = line.strip().lstrip("✓ ").split(":", 1)
        rtype = parts[0].strip()
        for key in type_to_keys.get(rtype, []):
            updated.pop(key, None)
        # IAM roles: match by role name
        if rtype == "IAM Role" and len(parts) > 1:
            role_name = parts[1].strip()
            if updated.get("iam_role_name") == role_name:
                updated.pop("iam_role_name", None)
                updated.pop("iam_role_arn", None)
                updated.pop("iam_policy_names", None)
            if updated.get("iam_gateway_role_name") == role_name:
                updated.pop("iam_gateway_role_name", None)

    Path(d["_file"]).write_text(json.dumps(updated, indent=2))


def main():
    args = parse_args()
    deployments = load_deployments(args.file, args.region)

    if not deployments:
        print("No deployment JSON files found.")
        sys.exit(0)

    shared_pools, shared_gateways = find_shared_resources(deployments)

    print(f"\nFound {len(deployments)} deployment(s):\n")
    for i, d in enumerate(deployments, 1):
        shared_note = ""
        if d.get("cognito_pool_id") in shared_pools or d.get("gateway_id") in shared_gateways:
            shared_note = "  ⚠ has shared resources"
        print(f"  [{i}] {d.get('server_name')} ({d.get('region')}){shared_note}")
    print(f"  [{len(deployments) + 1}] Delete ALL")
    print(f"  [0] Exit")

    while True:
        choice = input("\nSelect deployment to delete: ").strip()
        if choice == "0":
            print("Exiting.")
            sys.exit(0)
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(deployments):
                to_delete = [deployments[idx - 1]]
                break
            elif idx == len(deployments) + 1:
                to_delete = deployments
                break
        print("  Invalid selection.")

    for d in to_delete:
        print_resources(d, shared_pools, shared_gateways)
        if not args.yes:
            ans = input("\n  Proceed with deletion? [y/N]: ").strip().lower()
            if ans != "y":
                print("  Skipped.")
                continue
        print("\n  Deleting...")
        delete_resources(d, shared_pools, shared_gateways)

    print("\nDone.")


if __name__ == "__main__":
    main()
