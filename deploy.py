"""
AgentCore MCP Deployer
======================
Single-file Python CLI script that automates the full deployment lifecycle of any
AWS Labs MCP server to Amazon Bedrock AgentCore Runtime via an AgentCore Gateway.

Usage:
    python deploy.py [--server <server-name>] [--region <region>] [--verbose]

DeploymentOutput JSON schema:
{
  "server_name": "eks-mcp-server",
  "region": "us-west-2",
  "gateway_url": "https://...",
  "client_id": "...",
  "client_secret": "...",
  "token_url": "https://...",
  "runtime_id": "...",
  "ecr_uri": "...",
  "iam_role_arn": "...",
  "iam_policy_names": ["ECRPullPolicy", "ServicePolicy-eks"],
  "gateway_id": "...",
  "oauth_provider_arn": "...",
  "cognito_pool_id": "...",
  "resources_reused": ["gateway", "cognito_pool", "oauth_provider"],
  "resources_created": ["ecr_repo", "runtime", "gateway_target", "iam_role", "cognito_app_client"]
}
"""

from __future__ import annotations

import argparse
import base64
import logging
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
import requests


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DeploymentError(Exception):
    """Raised by any stage to abort the pipeline with a descriptive message."""

    def __init__(self, stage: str, message: str, details: str = "") -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.details = details


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    index: int           # 1-based display number
    directory_name: str  # e.g., "eks-mcp-server"
    display_name: str    # e.g., "Amazon EKS MCP Server"
    path: Path           # full path to server directory


@dataclass
class DeploymentContext:
    # CLI inputs
    server_name: str | None = None
    region: str = ""
    verbose: bool = False

    @property
    def safe_name(self) -> str:
        """For runtime: [a-zA-Z][a-zA-Z0-9_]{0,47} — hyphens replaced with underscores."""
        return (self.server_name or "server").replace("-", "_")[:48]

    @property
    def hyphen_name(self) -> str:
        """For gateway/target/oauth: ([0-9a-zA-Z][-]?){1,100} — hyphens allowed, no underscores."""
        return (self.server_name or "server")[:100]

    # Repository
    repo_dir: Path = None
    server_dir: Path = None

    # Catalog
    catalog: list[CatalogEntry] = field(default_factory=list)

    # Pattern detection
    pattern: str = ""        # "A" or "B"
    pattern_file: Path = None  # file containing the FastMCP import

    # Container
    image_tag: str = ""
    ecr_uri: str = ""

    # IAM
    iam_role_arn: str = ""
    iam_role_name: str = ""
    iam_policy_names: list[str] = field(default_factory=list)
    detected_services: list[str] = field(default_factory=list)

    # Infrastructure reuse
    reuse_existing: bool = False
    existing_gateway_id: str = ""
    existing_gateway_name: str = ""
    existing_gateway_role_arn: str = ""

    # Cognito
    cognito_pool_id: str = ""
    cognito_domain: str = ""
    resource_server_id: str = "mcp-gateway"
    client_id: str = ""
    client_secret: str = ""

    # OAuth provider
    oauth_provider_arn: str = ""
    oauth_provider_name: str = ""
    oauth_provider_client_id: str = ""

    # Runtime
    runtime_id: str = ""
    runtime_arn: str = ""

    # Gateway
    gateway_id: str = ""
    gateway_url: str = ""
    gateway_name: str = ""

    # Gateway Target
    gateway_target_id: str = ""

    # Account
    account_id: str = ""


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy an AWS Labs MCP server to Amazon Bedrock AgentCore Runtime"
    )
    parser.add_argument(
        "--server",
        help="MCP server directory name (e.g. eks-mcp-server). If omitted, shows interactive catalog.",
        default=None,
    )
    parser.add_argument(
        "--region",
        help="AWS region. Falls back to AWS CLI default, then prompts if not set.",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class StageLogger(logging.LoggerAdapter):
    """Logging adapter that injects a ``stage`` field into every log record."""

    def process(self, msg, kwargs):
        return msg, {**kwargs, "extra": {**self.extra, **(kwargs.get("extra") or {})}}


def configure_logging(args, server_name: str = "deploy") -> StageLogger:
    """Configure root logger with stderr + file handlers and return a StageLogger.

    The formatter produces lines like:
        [2026-02-26 10:15:03] [INFO] [CLONE] Cloning repository...

    Args:
        args: Parsed argparse namespace; ``args.verbose`` controls log level.
        server_name: Used to name the log file ``<server_name>-deploy.log``.

    Returns:
        A :class:`StageLogger` wrapping the root logger with stage set to ``"INIT"``.
    """
    level = logging.DEBUG if args.verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter by level

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(stage)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class _StageFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "stage"):
                record.stage = "-"
            return True

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)
    stderr_handler.addFilter(_StageFilter())

    file_handler = logging.FileHandler(f"{server_name}-deploy.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_StageFilter())

    root.addHandler(stderr_handler)
    root.addHandler(file_handler)

    return StageLogger(root, {"stage": "INIT"})


# Module-level logger (replaced by configure_logging() at runtime)
logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

def run_pipeline(ctx: DeploymentContext) -> None:
    """Execute all deployment stages sequentially."""
    stages: list[tuple[str, Any]] = [
        ("CATALOG",    fetch_catalog),
        ("SELECT",     select_server),
        ("CLONE",      sparse_download),
        ("DETECT",     detect_pattern),
        ("PATCH",      patch_server_code),
        ("DOCKERFILE", modify_dockerfile),
        ("FINCH",      install_finch),
        ("BUILD",      build_container),
        ("VERIFY",     verify_container),
        ("ECR",        push_ecr),
        ("IAM",        create_iam),
        ("INFRA",      detect_infrastructure),
        ("COGNITO",    setup_cognito),
        ("RUNTIME",    create_runtime),
        ("GATEWAY",    create_gateway),
        ("OAUTH",      create_oauth_provider),
        ("TARGET",     create_gateway_target),
        ("OUTPUT",     output_results),
    ]

    for stage_name, stage_fn in stages:
        try:
            stage_fn(ctx)
        except DeploymentError as err:
            logger.debug(
                "Stage %s failed: %s\nDetails: %s\n%s",
                err.stage, err.message, err.details, traceback.format_exc(),
                extra={"stage": stage_name},
            )
            logger.error(
                "Deployment failed at stage [%s]: %s", err.stage, err.message,
                extra={"stage": stage_name},
            )
            sys.exit(1)
        except Exception as exc:
            logger.debug(
                "Unexpected error in stage %s:\n%s", stage_name, traceback.format_exc(),
                extra={"stage": stage_name},
            )
            logger.error(
                "Unexpected error in stage [%s]: %s", stage_name, exc,
                extra={"stage": stage_name},
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True, stage: str = "CMD") -> subprocess.CompletedProcess:
    """Run a shell command, log it, and return the result."""
    logger.debug("Running command: %s", " ".join(cmd), extra={"stage": stage})
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    logger.debug("Exit code: %d", result.returncode, extra={"stage": stage})
    if result.stdout:
        logger.debug("stdout: %s", result.stdout.strip(), extra={"stage": stage})
    if result.stderr:
        logger.debug("stderr: %s", result.stderr.strip(), extra={"stage": stage})
    if check and result.returncode != 0:
        raise DeploymentError(
            stage=stage,
            message=f"Command failed with exit code {result.returncode}: {' '.join(cmd)}",
            details=result.stderr or result.stdout,
        )
    return result


def _finch(*args: str) -> list[str]:
    """Return a finch command prefixed with sudo on Linux (containerd requires root)."""
    prefix = ["sudo"] if _platform.system().lower() == "linux" else []
    return prefix + ["finch", *args]


def github_api_get(path: str, stage: str = "CATALOG") -> Any:
    """Fetch JSON from the GitHub API with rate-limit handling."""
    url = f"https://api.github.com/repos/awslabs/mcp/contents/{path}"
    logger.debug("GitHub API GET: %s", url, extra={"stage": stage})
    resp = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"}, timeout=30)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise DeploymentError(
            stage=stage,
            message="GitHub API rate limit exceeded. Try again later or set GITHUB_TOKEN env var.",
            details=resp.text,
        )
    resp.raise_for_status()
    return resp.json()


def poll_status(
    get_fn: Any,
    ready_status: str = "READY",
    failed_status: str = "FAILED",
    timeout: int = 300,
    interval: int = 10,
    stage: str = "POLL",
) -> dict:
    """Poll a resource status function until READY, FAILED, or timeout.

    Args:
        get_fn: Callable that returns a dict with a 'status' key (and optionally 'statusReasons').
        ready_status: Status string indicating success.
        failed_status: Status string indicating failure.
        timeout: Maximum seconds to wait.
        interval: Seconds between polls.
        stage: Stage name for logging.

    Returns:
        The final response dict when status is ready_status.

    Raises:
        DeploymentError: If status reaches failed_status or timeout is exceeded.
    """
    deadline = time.time() + timeout
    status = ""
    while time.time() < deadline:
        response = get_fn()
        status = response.get("status", "")
        logger.debug("Poll status: %s", status, extra={"stage": stage})
        if status == ready_status:
            return response
        if status == failed_status:
            reasons = response.get("statusReasons", [])
            raise DeploymentError(
                stage=stage,
                message=f"Resource reached {failed_status} status.",
                details=str(reasons),
            )
        time.sleep(interval)
    raise DeploymentError(
        stage=stage,
        message=f"Timed out after {timeout}s waiting for {ready_status} status.",
        details=f"Last status: {status}",
    )


def find_free_port() -> int:
    """Find an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_aws_account_id(region: str) -> str:
    """Return the AWS account ID for the current credentials."""
    sts = boto3.client("sts", region_name=region)
    identity = sts.get_caller_identity()
    logger.debug("AWS account ID: %s", identity["Account"], extra={"stage": "INIT"})
    return identity["Account"]


# ---------------------------------------------------------------------------
# Stage: Fetch Catalog
# ---------------------------------------------------------------------------

def fetch_catalog(ctx: DeploymentContext) -> None:
    """Fetch the list of MCP servers from GitHub and build the catalog."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "CATALOG"})
    log.info("Fetching MCP server catalog from GitHub...")

    try:
        entries = github_api_get("src", stage="CATALOG")
        dirs = [e for e in entries if e.get("type") == "dir"]

        catalog = []
        for dir_entry in dirs:
            dir_name = dir_entry["name"]
            display_name = _extract_display_name_from_api(dir_name, log)
            catalog.append((dir_name, display_name))

        catalog.sort(key=lambda x: x[1])
        ctx.catalog = [
            CatalogEntry(index=i + 1, directory_name=dn, display_name=disp, path=Path(dn))
            for i, (dn, disp) in enumerate(catalog)
        ]
        ctx.repo_dir = Path(tempfile.mkdtemp(prefix="mcp-deployer-"))
        log.info("Found %d MCP servers.", len(ctx.catalog))

    except Exception as exc:
        log.warning("GitHub API failed (%s), falling back to shallow clone...", exc)
        _fetch_catalog_fallback(ctx, log)


def _extract_display_name_from_api(dir_name: str, log: Any) -> str:
    """Try to get a human-readable name from README.md or pyproject.toml via API."""
    # Try README.md
    try:
        readme_data = github_api_get(f"src/{dir_name}/README.md", stage="CATALOG")
        content = base64.b64decode(readme_data["content"]).decode("utf-8", errors="replace")
        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except Exception:
        pass

    # Try pyproject.toml
    try:
        toml_data = github_api_get(f"src/{dir_name}/pyproject.toml", stage="CATALOG")
        content = base64.b64decode(toml_data["content"]).decode("utf-8", errors="replace")
        m = re.search(r'^name\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception:
        pass

    return dir_name


def _fetch_catalog_fallback(ctx: DeploymentContext, log: Any) -> None:
    """Fallback: shallow clone the repo and build catalog from local filesystem."""
    tmpdir = Path(tempfile.mkdtemp(prefix="mcp-deployer-"))
    ctx.repo_dir = tmpdir
    run_cmd(
        ["git", "clone", "--depth", "1", "https://github.com/awslabs/mcp.git", str(tmpdir)],
        stage="CATALOG",
    )
    src_dir = tmpdir / "src"
    catalog = []
    for server_dir in sorted(src_dir.iterdir()):
        if not server_dir.is_dir():
            continue
        dir_name = server_dir.name
        display_name = _extract_display_name_from_local(server_dir, dir_name)
        catalog.append((dir_name, display_name))

    catalog.sort(key=lambda x: x[1])
    ctx.catalog = [
        CatalogEntry(index=i + 1, directory_name=dn, display_name=disp, path=src_dir / dn)
        for i, (dn, disp) in enumerate(catalog)
    ]
    log.info("Found %d MCP servers (from local clone).", len(ctx.catalog))


def _extract_display_name_from_local(server_dir: Path, dir_name: str) -> str:
    """Extract display name from local README.md or pyproject.toml."""
    readme = server_dir / "README.md"
    if readme.exists():
        for line in readme.read_text(errors="replace").splitlines():
            if line.startswith("# "):
                return line[2:].strip()

    pyproject = server_dir / "pyproject.toml"
    if pyproject.exists():
        m = re.search(r'^name\s*=\s*["\']([^"\']+)["\']', pyproject.read_text(), re.MULTILINE)
        if m:
            return m.group(1)

    return dir_name


def _match_catalog_entry(catalog: list[CatalogEntry], query: str) -> CatalogEntry | None:
    """Match a catalog entry by number, directory name, or partial display name."""
    # Try numeric index
    if query.isdigit():
        idx = int(query)
        for entry in catalog:
            if entry.index == idx:
                return entry
        return None

    # Try exact directory name
    for entry in catalog:
        if entry.directory_name == query:
            return entry

    # Try partial display name (case-insensitive)
    query_lower = query.lower()
    for entry in catalog:
        if query_lower in entry.display_name.lower():
            return entry

    return None


def select_server(ctx: DeploymentContext) -> None:
    """Display the catalog and let the user select a server."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "SELECT"})

    def print_catalog():
        print("\nAvailable MCP Servers:")
        for entry in ctx.catalog:
            print(f"  [{entry.index}] {entry.display_name} ({entry.directory_name})")
        print()

    # If --server was provided, try to match it
    if ctx.server_name:
        match = _match_catalog_entry(ctx.catalog, ctx.server_name)
        if match:
            log.info("Selected server: %s (%s)", match.display_name, match.directory_name)
            ctx.server_name = match.directory_name
            ctx.server_dir = match.path
            return
        else:
            print(f"Server '{ctx.server_name}' not found in catalog.")

    # Interactive selection
    print_catalog()
    while True:
        try:
            query = input("Select a server (number, name, or partial match): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise DeploymentError("SELECT", "User cancelled server selection.")

        if not query:
            continue

        match = _match_catalog_entry(ctx.catalog, query)
        if match:
            log.info("Selected server: %s (%s)", match.display_name, match.directory_name)
            ctx.server_name = match.directory_name
            ctx.server_dir = match.path
            return
        else:
            print(f"Invalid selection '{query}'. Please try again.")
            print_catalog()


# ---------------------------------------------------------------------------
# Stage: Sparse Download
# ---------------------------------------------------------------------------

def sparse_download(ctx: DeploymentContext) -> None:
    """Download only the selected server's directory using git sparse-checkout."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "CLONE"})

    # If we already have a full clone from the fallback, just use it
    if ctx.repo_dir and (ctx.repo_dir / ".git").exists():
        log.info("Reusing existing repository at %s, running git pull...", ctx.repo_dir)
        run_cmd(["git", "pull"], cwd=ctx.repo_dir, stage="CLONE")
    else:
        if not ctx.repo_dir:
            ctx.repo_dir = Path(tempfile.mkdtemp(prefix="mcp-deployer-"))

        log.info("Sparse-checking out %s into %s...", ctx.server_name, ctx.repo_dir)
        run_cmd(
            ["git", "clone", "--filter=blob:none", "--sparse",
             "https://github.com/awslabs/mcp.git", str(ctx.repo_dir)],
            stage="CLONE",
        )
        run_cmd(
            ["git", "sparse-checkout", "set", f"src/{ctx.server_name}"],
            cwd=ctx.repo_dir,
            stage="CLONE",
        )

    ctx.server_dir = ctx.repo_dir / "src" / ctx.server_name
    if not ctx.server_dir.exists():
        raise DeploymentError(
            stage="CLONE",
            message=f"Server directory not found after download: {ctx.server_dir}",
            details=f"Server name: {ctx.server_name}",
        )
    log.info("Server directory ready: %s", ctx.server_dir)


# ---------------------------------------------------------------------------
# Stage: Detect FastMCP Pattern
# ---------------------------------------------------------------------------

PATTERN_A_RE = re.compile(r'from\s+mcp\.server\.fastmcp\s+import\s+[^\n]*FastMCP')
PATTERN_B_RE = re.compile(r'from\s+fastmcp\s+import\s+[^\n]*FastMCP')


def detect_pattern(ctx: DeploymentContext) -> None:
    """Scan server Python files to detect which FastMCP pattern is used."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "DETECT"})
    log.info("Scanning for FastMCP pattern in %s...", ctx.server_dir)

    py_files = list(ctx.server_dir.rglob("*.py"))
    log.debug("Found %d Python files to scan.", len(py_files))

    for py_file in py_files:
        try:
            content = py_file.read_text(errors="replace")
        except OSError as exc:
            log.debug("Could not read %s: %s", py_file, exc)
            continue

        if PATTERN_A_RE.search(content):
            ctx.pattern = "A"
            ctx.pattern_file = py_file
            log.info("Detected Pattern A (mcp.server.fastmcp) in %s", py_file)
            return

        if PATTERN_B_RE.search(content):
            ctx.pattern = "B"
            ctx.pattern_file = py_file
            log.info("Detected Pattern B (fastmcp) in %s", py_file)
            return

    scanned = [str(f) for f in py_files]
    raise DeploymentError(
        stage="DETECT",
        message="Could not detect FastMCP pattern. Neither Pattern A nor Pattern B import found.",
        details=f"Scanned {len(py_files)} files: {', '.join(scanned[:10])}{'...' if len(scanned) > 10 else ''}",
    )


# ---------------------------------------------------------------------------
# Stage: Patch Server Code
# ---------------------------------------------------------------------------

_PATCH_MARKER = "/invocations"
_TRANSPORT_MARKER = "MCP_TRANSPORT"

_PATTERN_A_TRANSPORT_BLOCK = '''

# --- AgentCore streamable-HTTP transport (injected by deploy.py) ---
import os as _os
if _os.environ.get("MCP_TRANSPORT") == "streamable-http":
    import sys as _sys
    from starlette.requests import Request as _Request
    from starlette.responses import JSONResponse as _JSONResponse
    from starlette.routing import Route as _Route

    _module = _sys.modules[__name__]
    # Find the FastMCP instance by checking known names, then scanning module globals
    _app_instance = getattr(_module, "mcp", None) or getattr(_module, "app", None)
    if _app_instance is None:
        from mcp.server.fastmcp import FastMCP as _FastMCP
        for _v in vars(_module).values():
            if isinstance(_v, _FastMCP):
                _app_instance = _v
                break

    if _app_instance is not None:
        async def _ping(request: _Request) -> _JSONResponse:
            return _JSONResponse({"status": "healthy"})

        _app_instance._custom_starlette_routes.append(_Route("/ping", _ping, methods=["GET"]))
        _app_instance.settings.streamable_http_path = "/invocations"
        _app_instance.settings.host = "0.0.0.0"
        _app_instance.settings.port = 8080
        _app_instance.settings.stateless_http = True
        _app_instance.settings.json_response = True
        _app_instance.settings.transport_security.enable_dns_rebinding_protection = False
        # Run uvicorn directly with log_config=None to avoid isatty() crash in detached containers
        import anyio as _anyio
        import uvicorn as _uvicorn
        async def _run_http():
            _starlette_app = _app_instance.streamable_http_app()
            _config = _uvicorn.Config(
                _starlette_app,
                host="0.0.0.0",
                port=8080,
                log_level="info",
                log_config=None,
            )
            await _uvicorn.Server(_config).serve()
        _anyio.run(_run_http)
'''

_PATTERN_B_TRANSPORT_BLOCK = '''

# --- AgentCore streamable-HTTP transport (injected by deploy.py) ---
import os as _os
if _os.environ.get("MCP_TRANSPORT") == "streamable-http":
    import uvicorn as _uvicorn
    from starlette.applications import Starlette as _Starlette
    from starlette.requests import Request as _Request
    from starlette.responses import JSONResponse as _JSONResponse
    from starlette.routing import Mount as _Mount
    from starlette.routing import Route as _Route
    from mcp.server.streamable_http import StreamableHTTPSessionManager as _StreamableHTTPSessionManager
    from mcp.server.streamable_http_manager import StreamableHTTPASGIApp as _StreamableHTTPASGIApp
    import sys as _sys

    _module = _sys.modules[__name__]
    _mcp_instance = getattr(_module, "mcp", None)

    async def _ping(request: _Request) -> _JSONResponse:
        return _JSONResponse({"status": "healthy"})

    if _mcp_instance is not None:
        _session_manager = _StreamableHTTPSessionManager(
            app=_mcp_instance._mcp_server,
            event_store=None,
            json_response=True,
            stateless=True,
        )
        _mcp_asgi = _StreamableHTTPASGIApp(session_manager=_session_manager)
        _transport_app = _Starlette(routes=[
            _Route("/ping", _ping, methods=["GET"]),
            _Mount("/invocations", app=_mcp_asgi),
        ])
        _uvicorn.run(_transport_app, host="0.0.0.0", port=8080)
'''


def patch_server_code(ctx: DeploymentContext) -> None:
    """Patch the server's main Python file to add streamable HTTP transport."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "PATCH"})

    target_file = ctx.pattern_file
    content = target_file.read_text(errors="replace")

    # Idempotency check
    if _PATCH_MARKER in content or _TRANSPORT_MARKER in content:
        log.info("Server code already patched (found marker in %s), skipping.", target_file)
        return

    if ctx.pattern == "A":
        log.info("Applying Pattern A transport patch to %s", target_file)
        patched = content + _PATTERN_A_TRANSPORT_BLOCK
    elif ctx.pattern == "B":
        log.info("Applying Pattern B transport patch to %s", target_file)
        patched = content + _PATTERN_B_TRANSPORT_BLOCK
    else:
        raise DeploymentError(
            stage="PATCH",
            message=f"Unknown pattern '{ctx.pattern}'. Cannot patch server code.",
        )

    target_file.write_text(patched)
    log.info("Successfully patched %s", target_file)

# ---------------------------------------------------------------------------
# Stage: Modify Dockerfile
# ---------------------------------------------------------------------------

_AGENTCORE_INSTALL_LINE = "RUN uv pip install --python /app/.venv/bin/python bedrock-agentcore"
_REQUIRED_ENV_VARS = {
    "MCP_TRANSPORT": "streamable-http",
    "ALLOW_WRITE": "true",
    "ALLOW_SENSITIVE_DATA_ACCESS": "true",
    "PYTHONUNBUFFERED": "1",
}


def modify_dockerfile(ctx: DeploymentContext) -> None:
    """Modify the Dockerfile to install bedrock-agentcore and set required ENV vars."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "DOCKERFILE"})

    dockerfile = ctx.server_dir / "Dockerfile"
    if not dockerfile.exists():
        # Generate a standard Dockerfile from pyproject.toml
        pyproject = ctx.server_dir / "pyproject.toml"
        if not pyproject.exists():
            raise DeploymentError(
                stage="DOCKERFILE",
                message=f"Neither Dockerfile nor pyproject.toml found in {ctx.server_dir}",
            )
        # Extract entry point from [project.scripts]
        import re as _re_df
        scripts_match = _re_df.search(r'\[project\.scripts\][^\[]*?"[^"]*"\s*=\s*"([^"]+)"', pyproject.read_text(), _re_df.DOTALL)
        entrypoint = scripts_match.group(1).split(":")[0].replace(".", "/") if scripts_match else None
        package_name = _re_df.search(r'^name\s*=\s*"([^"]+)"', pyproject.read_text(), _re_df.MULTILINE)
        pkg = package_name.group(1) if package_name else ctx.server_name

        dockerfile.write_text(f"""FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY . .
RUN uv sync --frozen --no-dev
RUN uv pip install --python $(uv python find) bedrock-agentcore
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "awslabs.{ctx.server_name.replace('-', '_')}.server"]
""")
        log.info("Generated Dockerfile for %s", ctx.server_name)
        log.info("Dockerfile modified successfully at %s", dockerfile)
        return  # generated Dockerfile already has all required content

    lines = dockerfile.read_text().splitlines(keepends=True)

    # Find last uv sync line
    last_uv_sync_idx = None
    for i, line in enumerate(lines):
        if "uv sync" in line:
            last_uv_sync_idx = i

    if last_uv_sync_idx is None:
        log.warning("No 'uv sync' found in Dockerfile; appending bedrock-agentcore install at end.")
        insert_idx = len(lines)
    else:
        insert_idx = last_uv_sync_idx + 1

    # Insert bedrock-agentcore install line
    install_line = _AGENTCORE_INSTALL_LINE + "\n"
    lines.insert(insert_idx, install_line)

    # Verify no uv sync after install line
    for line in lines[insert_idx + 1:]:
        if "uv sync" in line:
            raise DeploymentError(
                stage="DOCKERFILE",
                message="Found 'uv sync' after bedrock-agentcore install line. Cannot safely modify Dockerfile.",
                details="Manual Dockerfile editing required.",
            )

    # Build ENV block
    env_block = "ENV " + " \\\n    ".join(f"{k}={v}" for k, v in _REQUIRED_ENV_VARS.items()) + "\n"

    # Find first CMD or ENTRYPOINT line to insert ENV block before it
    last_cmd_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("CMD") or stripped.startswith("ENTRYPOINT"):
            last_cmd_idx = i
            break  # use first CMD/ENTRYPOINT

    if last_cmd_idx is not None:
        lines.insert(last_cmd_idx, env_block)
    else:
        lines.append(env_block)

    dockerfile.write_text("".join(lines))
    log.info("Dockerfile modified successfully at %s", dockerfile)

# ---------------------------------------------------------------------------
# Stage: Install Finch and Build Container
# ---------------------------------------------------------------------------

import platform as _platform
import shutil as _shutil


def install_finch(ctx: DeploymentContext) -> None:
    """Ensure Finch is installed; auto-install if missing."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "FINCH"})

    if _shutil.which("finch"):
        if _platform.system().lower() == "darwin":
            log.info("Finch is already installed. Checking VM status...")
            vm_result = run_cmd(["finch", "vm", "status"], check=False, stage="FINCH")
            if "running" not in vm_result.stdout.lower():
                log.info("Finch VM not running. Initializing...")
                run_cmd(["finch", "vm", "init"], check=False, stage="FINCH")
                run_cmd(["finch", "vm", "start"], check=False, stage="FINCH")
        else:
            log.info("Finch is already installed (Linux — no VM required).")
        return

    system = _platform.system().lower()
    log.info("Finch not found. Attempting auto-install on %s...", system)

    if system == "darwin":
        run_cmd(["brew", "install", "--cask", "finch"], stage="FINCH")
        run_cmd(["finch", "vm", "init"], stage="FINCH")
    elif system == "linux":
        # Detect distro
        distro_id = ""
        try:
            distro_id = _platform.freedesktop_os_release().get("ID", "")
        except AttributeError:
            pass
        if distro_id in ("ubuntu", "debian"):
            import subprocess as _sp
            _ver = _sp.run(["curl", "-s", "https://api.github.com/repos/runfinch/finch/releases/latest"],
                           capture_output=True, text=True).stdout
            import json as _j
            _finch_ver = _j.loads(_ver)["tag_name"].lstrip("v")
            _arch = "arm64" if _platform.machine() == "aarch64" else "amd64"
            run_cmd(["curl", "-sLo", "/tmp/finch.deb",
                     f"https://github.com/runfinch/finch/releases/download/v{_finch_ver}/runfinch-finch_{_finch_ver}_{_arch}.deb"],
                    stage="FINCH")
            run_cmd(["sudo", "apt-get", "install", "-y", "/tmp/finch.deb"], stage="FINCH")
        elif distro_id in ("amzn", "fedora", "rhel", "centos"):
            run_cmd(["sudo", "dnf", "install", "-y", "finch"], stage="FINCH")
        else:
            _finch_install_failure(log)
            return
    elif system == "windows":
        run_cmd(["winget", "install", "-e", "--id", "FinchContainers.Finch"], stage="FINCH")
    else:
        _finch_install_failure(log)
        return

    if not _shutil.which("finch"):
        _finch_install_failure(log)


def _finch_install_failure(log: Any) -> None:
    print(
        "\nFinch auto-installation failed. Please install Finch manually:\n"
        "  macOS:   brew install --cask finch && finch vm init\n"
        "  Linux:   https://github.com/runfinch/finch/releases\n"
        "  Windows: winget install -e --id FinchContainers.Finch\n"
    )
    raise DeploymentError(
        stage="FINCH",
        message="Finch is not installed and auto-installation failed.",
        details="See manual install instructions above.",
    )


def build_container(ctx: DeploymentContext) -> None:
    """Build the container image with Finch for linux/arm64."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "BUILD"})
    image_tag = f"{ctx.server_name}:arm64"
    ctx.image_tag = image_tag
    log.info("Building container image %s...", image_tag)
    run_cmd(
        _finch("build", "--platform", "linux/arm64", "-t", image_tag, "."),
        cwd=ctx.server_dir,
        stage="BUILD",
    )
    log.info("Container image built: %s", image_tag)


# ---------------------------------------------------------------------------
# Stage: Verify Container Locally
# ---------------------------------------------------------------------------

import json as _json_mod


def verify_container(ctx: DeploymentContext) -> None:
    """Run the container locally and verify /ping and /invocations endpoints."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "VERIFY"})
    port = find_free_port()
    container_id = None

    log.info("Starting container %s on port %d for local verification...", ctx.image_tag, port)
    result = run_cmd(
        _finch("run", "-d", "-p", f"{port}:8080",
               "-e", "MCP_TRANSPORT=streamable-http",
               ctx.image_tag),
        stage="VERIFY",
    )
    container_id = result.stdout.strip()

    try:
        base_url = f"http://localhost:{port}"
        deadline = time.time() + 30

        # Wait for /ping
        ping_ok = False
        while time.time() < deadline:
            try:
                resp = requests.get(f"{base_url}/ping", timeout=5)
                if resp.status_code == 200 and resp.json().get("status") == "healthy":
                    ping_ok = True
                    log.info("/ping returned healthy.")
                    break
            except Exception:
                pass
            time.sleep(2)

        if not ping_ok:
            _dump_container_logs(container_id, log)
            raise DeploymentError(
                stage="VERIFY",
                message="Container /ping health check failed after 30s timeout.",
            )

        # Check /invocations with tools/list
        payload = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        try:
            resp = requests.post(
                f"{base_url}/invocations",
                json=payload,
                headers={"Accept": "application/json, text/event-stream"},
                timeout=15,
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                # Parse SSE: extract first data: line
                body = None
                for line in resp.text.splitlines():
                    if line.startswith("data:"):
                        body = _json_mod.loads(line[len("data:"):].strip())
                        break
                if body is None:
                    raise ValueError("No data line found in SSE response")
            else:
                body = resp.json()
            if "error" in body and "result" not in body:
                raise ValueError(f"JSON-RPC error: {body['error']}")
            log.info("/invocations tools/list succeeded.")
        except Exception as exc:
            _dump_container_logs(container_id, log)
            raise DeploymentError(
                stage="VERIFY",
                message=f"Container /invocations check failed: {exc}",
            )

    finally:
        if container_id:
            log.info("Stopping and removing test container %s...", container_id[:12])
            run_cmd(_finch("stop", container_id), check=False, stage="VERIFY")
            run_cmd(_finch("rm", container_id), check=False, stage="VERIFY")


def _dump_container_logs(container_id: str, log: Any) -> None:
    result = run_cmd(_finch("logs", container_id), check=False, stage="VERIFY")
    log.error("Container logs:\n%s", result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# Stage: Push to ECR
# ---------------------------------------------------------------------------

def push_ecr(ctx: DeploymentContext) -> None:
    """Create ECR repo if needed, authenticate, tag, and push the image."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "ECR"})

    if not ctx.account_id:
        ctx.account_id = get_aws_account_id(ctx.region)

    ecr = boto3.client("ecr", region_name=ctx.region)
    repo_name = ctx.server_name

    # Create repo if it doesn't exist
    try:
        ecr.create_repository(repositoryName=repo_name)
        log.info("Created ECR repository: %s", repo_name)
    except ecr.exceptions.RepositoryAlreadyExistsException:
        log.info("Reusing existing ECR repository: %s", repo_name)

    import time as _time_ecr
    image_version = f"v{int(_time_ecr.time())}"
    ecr_uri = f"{ctx.account_id}.dkr.ecr.{ctx.region}.amazonaws.com/{repo_name}:{image_version}"
    ctx.ecr_uri = ecr_uri
    log.info("ECR URI: %s", ecr_uri)

    # Authenticate finch to ECR
    log.info("Authenticating to ECR...")
    token_result = run_cmd(
        ["aws", "ecr", "get-login-password", "--region", ctx.region],
        stage="ECR",
    )
    ecr_host = f"{ctx.account_id}.dkr.ecr.{ctx.region}.amazonaws.com"

    # Write credentials directly to finch config.json (osxkeychain is inaccessible from Lima VM)
    import base64 as _base64
    finch_config_path = Path.home() / ".finch" / "config.json"
    try:
        finch_cfg = _json_mod.loads(finch_config_path.read_text()) if finch_config_path.exists() else {}
    except Exception:
        finch_cfg = {}
    finch_cfg.setdefault("auths", {})[ecr_host] = {
        "auth": _base64.b64encode(f"AWS:{token_result.stdout.strip()}".encode()).decode()
    }
    finch_config_path.write_text(_json_mod.dumps(finch_cfg, indent="\t"))
    log.info("ECR credentials written to finch config for %s", ecr_host)

    # Tag and push
    run_cmd(_finch("tag", ctx.image_tag, ecr_uri), stage="ECR")
    log.info("Pushing image to ECR...")
    run_cmd(_finch("push", ecr_uri), stage="ECR")
    log.info("Image pushed successfully: %s", ecr_uri)


# ---------------------------------------------------------------------------
# Stage: Create IAM Role and Policies
# ---------------------------------------------------------------------------

_BOTO3_CLIENT_RE = re.compile(r'boto3\.(?:client|resource)\(["\']([a-zA-Z0-9_-]+)["\']')

_AGENTCORE_TRUST_POLICY = _json_mod.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
})

_ECR_PULL_POLICY = _json_mod.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": [
            "ecr:GetDownloadUrlForLayer",
            "ecr:BatchGetImage",
            "ecr:GetAuthorizationToken",
            "ecr:BatchCheckLayerAvailability",
        ],
        "Resource": "*",
    }],
})


def infer_boto3_services(server_dir: "Path") -> list[str]:
    """Scan Python source files for boto3.client/resource calls and return unique service names."""
    services: set[str] = set()
    for py_file in server_dir.rglob("*.py"):
        try:
            content = py_file.read_text(errors="replace")
            for match in _BOTO3_CLIENT_RE.finditer(content):
                services.add(match.group(1))
        except OSError:
            pass
    return sorted(services)


def create_iam(ctx: DeploymentContext) -> None:
    """Create IAM role with trust policy, ECR pull policy, and service-specific policies."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "IAM"})
    iam = boto3.client("iam", region_name=ctx.region)
    role_name = f"agentcore-{ctx.server_name}-role"
    ctx.iam_role_name = role_name

    # Create or reuse role
    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=_AGENTCORE_TRUST_POLICY,
            Description=f"AgentCore runtime role for {ctx.server_name}",
        )
        ctx.iam_role_arn = resp["Role"]["Arn"]
        log.info("Created IAM role: %s", ctx.iam_role_arn)
    except iam.exceptions.EntityAlreadyExistsException:
        ctx.iam_role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        log.info("Reusing existing IAM role: %s", ctx.iam_role_arn)

    # Attach ECR pull policy
    ecr_policy_name = "ECRPullPolicy"
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=ecr_policy_name,
        PolicyDocument=_ECR_PULL_POLICY,
    )
    log.info("Attached ECR pull policy to role.")
    ctx.iam_policy_names = [ecr_policy_name]

    # Infer boto3 services and create service policy
    services = infer_boto3_services(ctx.server_dir)
    ctx.detected_services = services
    log.info("Detected boto3 services: %s", services)

    if services:
        service_policy_name = f"ServicePolicy-{ctx.server_name}"
        service_policy_doc = _json_mod.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": [f"{svc}:*" for svc in services],
                "Resource": "*",
            }],
        })
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=service_policy_name,
            PolicyDocument=service_policy_doc,
        )
        log.info("Attached service policy for: %s", services)
        ctx.iam_policy_names.append(service_policy_name)

    import time as _time_iam
    _time_iam.sleep(15)  # IAM eventual consistency — allow role + policies to propagate


# ---------------------------------------------------------------------------
# Stage: Detect Existing Infrastructure
# ---------------------------------------------------------------------------

def detect_infrastructure(ctx: DeploymentContext) -> None:
    """Check for existing AgentCore Gateways and prompt user to reuse or create new."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "INFRA"})
    ac = boto3.client("bedrock-agentcore-control", region_name=ctx.region)

    log.info("Checking for existing AgentCore Gateways...")
    try:
        resp = ac.list_gateways()
        gateways = resp.get("items", [])
    except Exception as exc:
        log.warning("Could not list gateways (%s); proceeding with new infrastructure.", exc)
        ctx.reuse_existing = False
        return

    if not gateways:
        log.info("No existing Gateways found. Creating new infrastructure.")
        ctx.reuse_existing = False
        return

    print("\nExisting AgentCore Gateways:")
    for i, gw in enumerate(gateways, 1):
        print(f"  [{i}] {gw.get('name', 'unnamed')} ({gw['gatewayId']})")
    print(f"  [{len(gateways) + 1}] Create new infrastructure")
    print()

    while True:
        try:
            choice = input("Select a Gateway to reuse, or create new: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise DeploymentError("INFRA", "User cancelled infrastructure selection.")

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(gateways):
                selected = gateways[idx - 1]
                ctx.reuse_existing = True
                ctx.existing_gateway_id = selected["gatewayId"]
                ctx.existing_gateway_name = selected.get("name", "")
                log.info("Reusing Gateway: %s (%s)", ctx.existing_gateway_name, ctx.existing_gateway_id)
                _retrieve_reuse_details(ctx, ac, log)
                return
            elif idx == len(gateways) + 1:
                ctx.reuse_existing = False
                log.info("Creating new infrastructure.")
                return
        print("Invalid selection. Please try again.")


def _retrieve_reuse_details(ctx: DeploymentContext, ac: Any, log: Any) -> None:
    """Retrieve Cognito and OAuth provider details from an existing Gateway."""
    try:
        gw = ac.get_gateway(gatewayIdentifier=ctx.existing_gateway_id)
        auth_config = gw.get("authorizerConfiguration", {}).get("customJWTAuthorizer", {})
        discovery_url = auth_config.get("discoveryUrl", "")
        # Extract pool ID from discovery URL:
        # https://cognito-idp.<region>.amazonaws.com/<pool_id>/.well-known/openid-configuration
        m = re.search(r"cognito-idp\.[^/]+\.amazonaws\.com/([^/]+)", discovery_url)
        if m:
            ctx.cognito_pool_id = m.group(1)
            log.info("Found Cognito pool ID: %s", ctx.cognito_pool_id)

        # Try to get execution role ARN
        ctx.existing_gateway_role_arn = gw.get("roleArn", "")
    except Exception as exc:
        log.warning("Could not retrieve full Gateway details: %s", exc)


# ---------------------------------------------------------------------------
# Stage: Setup Cognito OAuth
# ---------------------------------------------------------------------------

import secrets as _secrets


def setup_cognito(ctx: DeploymentContext) -> None:
    """Create Cognito User Pool, Domain, Resource Server, and App Clients."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "COGNITO"})
    cognito = boto3.client("cognito-idp", region_name=ctx.region)

    # Let user pick an existing pool or create new
    if not ctx.cognito_pool_id:
        existing_pools = [
            p for p in cognito.list_user_pools(MaxResults=60)["UserPools"]
            if "mcp-gateway" in p.get("Name", "")
        ]
        if existing_pools:
            print("\nExisting Cognito User Pools:")
            for i, p in enumerate(existing_pools, 1):
                print(f"  [{i}] {p['Name']} ({p['Id']})")
            print(f"  [{len(existing_pools) + 1}] Create new User Pool")
            print()
            while True:
                try:
                    choice = input("Select a User Pool: ").strip()
                except (EOFError, KeyboardInterrupt):
                    raise DeploymentError("COGNITO", "User cancelled pool selection.")
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(existing_pools):
                        selected = existing_pools[idx - 1]
                        ctx.cognito_pool_id = selected["Id"]
                        pool_detail = cognito.describe_user_pool(UserPoolId=ctx.cognito_pool_id)
                        ctx.cognito_domain = pool_detail["UserPool"].get("Domain", "")
                        log.info("Reusing Cognito User Pool: %s", ctx.cognito_pool_id)
                        break
                    elif idx == len(existing_pools) + 1:
                        break  # fall through to create new
                print("Invalid selection. Please try again.")

    if not ctx.reuse_existing and not ctx.cognito_pool_id:
        # Create User Pool
        pool_resp = cognito.create_user_pool(
            PoolName=f"mcp-gateway-pool-{ctx.server_name}",
            AdminCreateUserConfig={"AllowAdminCreateUserOnly": True},
        )
        ctx.cognito_pool_id = pool_resp["UserPool"]["Id"]
        log.info("Created Cognito User Pool: %s", ctx.cognito_pool_id)

        # Create Domain
        domain_suffix = _secrets.token_hex(6)
        ctx.cognito_domain = f"mcp-gateway-{domain_suffix}"
        cognito.create_user_pool_domain(
            Domain=ctx.cognito_domain,
            UserPoolId=ctx.cognito_pool_id,
        )
        log.info("Created Cognito Domain: %s", ctx.cognito_domain)

        # Create Resource Server
        cognito.create_resource_server(
            UserPoolId=ctx.cognito_pool_id,
            Identifier="mcp-gateway",
            Name="MCP Gateway",
            Scopes=[{"ScopeName": "invoke", "ScopeDescription": "Invoke MCP server"}],
        )
        log.info("Created Resource Server: mcp-gateway")

        # Create initial App Client
        initial_client = cognito.create_user_pool_client(
            UserPoolId=ctx.cognito_pool_id,
            ClientName="mcp-gateway-initial-client",
            GenerateSecret=True,
            AllowedOAuthFlows=["client_credentials"],
            AllowedOAuthScopes=["mcp-gateway/invoke"],
            AllowedOAuthFlowsUserPoolClient=True,
        )
        ctx.oauth_provider_client_id = initial_client["UserPoolClient"]["ClientId"]
        log.info("Created initial App Client: %s", ctx.oauth_provider_client_id)
    elif ctx.cognito_pool_id and not ctx.oauth_provider_client_id:
        # Reusing existing pool — find the initial client from the previous deployment JSON
        prev_file = Path(f"{ctx.server_name}-deployment.json")
        if prev_file.exists():
            prev = _json_mod.loads(prev_file.read_text())
            # Use the previous client_id as the oauth_provider_client_id placeholder
            ctx.oauth_provider_client_id = prev.get("client_id", "")

    # Always create a server-specific App Client
    server_client = cognito.create_user_pool_client(
        UserPoolId=ctx.cognito_pool_id,
        ClientName=f"{ctx.server_name}-client",
        GenerateSecret=True,
        AllowedOAuthFlows=["client_credentials"],
        AllowedOAuthScopes=["mcp-gateway/invoke"],
        AllowedOAuthFlowsUserPoolClient=True,
    )
    ctx.client_id = server_client["UserPoolClient"]["ClientId"]
    ctx.client_secret = server_client["UserPoolClient"]["ClientSecret"]
    log.info("Created server App Client: %s", ctx.client_id)

    # Update Gateway allowedClients if reusing
    if ctx.reuse_existing and ctx.existing_gateway_id:
        _update_gateway_allowed_clients(ctx, log)


def _update_gateway_allowed_clients(ctx: DeploymentContext, log: Any) -> None:
    """Add the new client ID to the existing Gateway's allowedClients list."""
    ac = boto3.client("bedrock-agentcore-control", region_name=ctx.region)
    try:
        gw = ac.get_gateway(gatewayIdentifier=ctx.existing_gateway_id)
        jwt_config = gw.get("authorizerConfiguration", {}).get("customJWTAuthorizer", {})
        existing_clients = list(jwt_config.get("allowedClients", []))
        if ctx.client_id not in existing_clients:
            existing_clients.append(ctx.client_id)
        ac.update_gateway(
            gatewayIdentifier=ctx.existing_gateway_id,
            name=gw["name"],
            roleArn=gw["roleArn"],
            protocolType=gw["protocolType"],
            authorizerType=gw["authorizerType"],
            authorizerConfiguration={
                "customJWTAuthorizer": {**jwt_config, "allowedClients": existing_clients}
            },
        )
        log.info("Updated Gateway allowedClients with new client ID: %s", ctx.client_id)
    except Exception as exc:
        log.warning("Could not update Gateway allowedClients: %s", exc)


# ---------------------------------------------------------------------------
# Stage: Create AgentCore Runtime
# ---------------------------------------------------------------------------

def create_runtime(ctx: DeploymentContext) -> None:
    """Create an AgentCore Runtime and poll until READY."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "RUNTIME"})
    ac = boto3.client("bedrock-agentcore-control", region_name=ctx.region)

    discovery_url = (
        f"https://cognito-idp.{ctx.region}.amazonaws.com/{ctx.cognito_pool_id}"
        "/.well-known/openid-configuration"
    )
    allowed_clients = list({ctx.client_id, ctx.oauth_provider_client_id} - {""})

    log.info("Creating AgentCore Runtime for %s...", ctx.server_name)
    try:
        resp = ac.create_agent_runtime(
            agentRuntimeName=f"{ctx.safe_name}_runtime",
            agentRuntimeArtifact={
                "containerConfiguration": {
                    "containerUri": ctx.ecr_uri,
                }
            },
            roleArn=ctx.iam_role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
            environmentVariables={"MCP_TRANSPORT": "streamable-http"},
            authorizerConfiguration={
                "customJWTAuthorizer": {
                    "discoveryUrl": discovery_url,
                    "allowedClients": allowed_clients,
                }
            },
        )
        ctx.runtime_id = resp["agentRuntimeId"]
        ctx.runtime_arn = resp.get("agentRuntimeArn", "")
        log.info("Runtime created: %s", ctx.runtime_id)
    except ac.exceptions.ConflictException:
        # Runtime already exists — find it by name
        runtimes = ac.list_agent_runtimes().get("agentRuntimes", [])
        existing = next((r for r in runtimes if r.get("agentRuntimeName") == f"{ctx.safe_name}_runtime"), None)
        if existing is None:
            raise DeploymentError("RUNTIME", "Runtime already exists but could not be found by name.")
        ctx.runtime_id = existing["agentRuntimeId"]
        ctx.runtime_arn = existing.get("agentRuntimeArn", "")
        log.info("Reusing existing Runtime: %s", ctx.runtime_id)
        # Update allowedClients with new Cognito client IDs AND new container image
        allowed_clients = list({ctx.client_id, ctx.oauth_provider_client_id} - {""})
        runtime_detail = ac.get_agent_runtime(agentRuntimeId=ctx.runtime_id)
        ac.update_agent_runtime(
            agentRuntimeId=ctx.runtime_id,
            agentRuntimeArtifact={"containerConfiguration": {"containerUri": ctx.ecr_uri}},
            roleArn=runtime_detail["roleArn"],
            networkConfiguration=runtime_detail["networkConfiguration"],
            authorizerConfiguration={
                "customJWTAuthorizer": {
                    "discoveryUrl": discovery_url,
                    "allowedClients": allowed_clients,
                }
            },
            environmentVariables=runtime_detail.get("environmentVariables", {}),
        )
        log.info("Updated Runtime image to %s and allowedClients: %s", ctx.ecr_uri, allowed_clients)

    # Poll until READY
    poll_status(
        get_fn=lambda: ac.get_agent_runtime(agentRuntimeId=ctx.runtime_id),
        ready_status="READY",
        failed_status="FAILED",
        timeout=300,
        stage="RUNTIME",
    )
    log.info("Runtime is READY: %s", ctx.runtime_id)


# ---------------------------------------------------------------------------
# Stage: Create AgentCore Gateway
# ---------------------------------------------------------------------------

def create_gateway(ctx: DeploymentContext) -> None:
    """Create an AgentCore Gateway with MCP protocol and CUSTOM_JWT authorizer. Skip if reusing."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "GATEWAY"})

    if ctx.reuse_existing:
        ctx.gateway_id = ctx.existing_gateway_id
        log.info("Reusing existing Gateway: %s", ctx.gateway_id)
        # Retrieve gateway URL
        ac = boto3.client("bedrock-agentcore-control", region_name=ctx.region)
        try:
            gw = ac.get_gateway(gatewayIdentifier=ctx.gateway_id)
            ctx.gateway_url = gw.get("gatewayUrl", "")
            ctx.existing_gateway_role_arn = gw.get("roleArn", "")
        except Exception as exc:
            log.warning("Could not retrieve Gateway URL: %s", exc)
        return

    ac = boto3.client("bedrock-agentcore-control", region_name=ctx.region)
    iam = boto3.client("iam", region_name=ctx.region)

    # Create Gateway execution role
    gw_role_name = f"agentcore-gateway-{ctx.server_name}-role"
    try:
        gw_role = iam.create_role(
            RoleName=gw_role_name,
            AssumeRolePolicyDocument=_AGENTCORE_TRUST_POLICY,
            Description=f"AgentCore Gateway execution role for {ctx.server_name}",
        )
        gw_role_arn = gw_role["Role"]["Arn"]
        log.info("Created Gateway execution role: %s", gw_role_arn)
    except iam.exceptions.EntityAlreadyExistsException:
        gw_role_arn = iam.get_role(RoleName=gw_role_name)["Role"]["Arn"]
        log.info("Reusing Gateway execution role: %s", gw_role_arn)
    ctx.existing_gateway_role_arn = gw_role_arn

    # Attach InvokeAgentRuntime policy to Gateway role
    iam.put_role_policy(
        RoleName=gw_role_name,
        PolicyName="InvokeRuntimePolicy",
        PolicyDocument=_json_mod.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["bedrock-agentcore:InvokeAgentRuntime"],
                "Resource": "*",
            }],
        }),
    )
    log.info("Attached InvokeAgentRuntime policy to Gateway role.")

    discovery_url = (
        f"https://cognito-idp.{ctx.region}.amazonaws.com/{ctx.cognito_pool_id}"
        "/.well-known/openid-configuration"
    )
    allowed_clients = list({ctx.client_id, ctx.oauth_provider_client_id} - {""})

    log.info("Creating AgentCore Gateway...")
    try:
        resp = ac.create_gateway(
            name=f"{ctx.hyphen_name}-gateway",
            protocolType="MCP",
            roleArn=gw_role_arn,
            authorizerType="CUSTOM_JWT",
            authorizerConfiguration={
                "customJWTAuthorizer": {
                    "discoveryUrl": discovery_url,
                    "allowedClients": allowed_clients,
                }
            },
        )
        ctx.gateway_id = resp["gatewayId"]
        log.info("Gateway created: %s", ctx.gateway_id)
    except ac.exceptions.ConflictException:
        gateways = ac.list_gateways().get("items", [])
        existing = next((g for g in gateways if g.get("name") == f"{ctx.hyphen_name}-gateway"), None)
        if existing is None:
            raise DeploymentError("GATEWAY", "Gateway already exists but could not be found by name.")
        ctx.gateway_id = existing["gatewayId"]
        log.info("Reusing existing Gateway: %s", ctx.gateway_id)

    final = poll_status(
        get_fn=lambda: ac.get_gateway(gatewayIdentifier=ctx.gateway_id),
        ready_status="READY",
        failed_status="FAILED",
        timeout=300,
        stage="GATEWAY",
    )
    ctx.gateway_url = final.get("gatewayUrl", "")
    log.info("Gateway is READY. URL: %s", ctx.gateway_url)


# ---------------------------------------------------------------------------
# Stage: Create OAuth Provider
# ---------------------------------------------------------------------------

def create_oauth_provider(ctx: DeploymentContext) -> None:
    """Create an OAuth2 credential provider in AgentCore Token Vault. Skip if reusing."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "OAUTH"})

    if ctx.reuse_existing:
        if not ctx.oauth_provider_arn:
            # Look up the existing OAuth provider for this server
            ac = boto3.client("bedrock-agentcore-control", region_name=ctx.region)
            providers = ac.list_oauth2_credential_providers().get("credentialProviders", [])
            existing = next((p for p in providers if p.get("name") == f"{ctx.hyphen_name}-oauth-provider"), None)
            if existing:
                ctx.oauth_provider_arn = existing["credentialProviderArn"]
                log.info("Found existing OAuth provider: %s", ctx.oauth_provider_arn)
            else:
                log.info("No existing OAuth provider found, will create one.")
                ctx.reuse_existing = False  # fall through to create
        else:
            log.info("Reusing existing OAuth provider: %s", ctx.oauth_provider_arn)
        if ctx.reuse_existing:
            return

    ac = boto3.client("bedrock-agentcore-control", region_name=ctx.region)
    token_url = f"https://{ctx.cognito_domain}.auth.{ctx.region}.amazoncognito.com/oauth2/token"
    auth_url = f"https://{ctx.cognito_domain}.auth.{ctx.region}.amazoncognito.com/oauth2/authorize"
    issuer_url = f"https://cognito-idp.{ctx.region}.amazonaws.com/{ctx.cognito_pool_id}"

    log.info("Creating OAuth2 credential provider...")
    try:
        resp = ac.create_oauth2_credential_provider(
            name=f"{ctx.hyphen_name}-oauth-provider",
            credentialProviderVendor="CustomOauth2",
            oauth2ProviderConfigInput={
                "customOauth2ProviderConfig": {
                    "oauthDiscovery": {
                        "discoveryUrl": (
                            f"https://cognito-idp.{ctx.region}.amazonaws.com/{ctx.cognito_pool_id}"
                            "/.well-known/openid-configuration"
                        ),
                    },
                    "clientId": ctx.client_id,
                    "clientSecret": ctx.client_secret,
                }
            },
        )
        ctx.oauth_provider_arn = resp.get("credentialProviderArn", "")
        ctx.oauth_provider_name = resp.get("name", "")
        log.info("OAuth provider created: %s", ctx.oauth_provider_arn)
        import time as _time_oauth_wait
        _time_oauth_wait.sleep(10)  # Allow Secrets Manager secret to propagate
    except ac.exceptions.ValidationException as exc:
        if "already exists" in str(exc):
            providers = ac.list_oauth2_credential_providers().get("credentialProviders", [])
            existing = next((p for p in providers if p.get("name") == f"{ctx.hyphen_name}-oauth-provider"), None)
            if existing is None:
                raise DeploymentError("OAUTH", "OAuth provider already exists but could not be found by name.")
            ctx.oauth_provider_arn = existing["credentialProviderArn"]
            ctx.oauth_provider_name = existing["name"]

            # Check if the existing provider uses the current Cognito pool
            current_discovery_url = (
                f"https://cognito-idp.{ctx.region}.amazonaws.com/{ctx.cognito_pool_id}"
                "/.well-known/openid-configuration"
            )
            try:
                detail = ac.get_oauth2_credential_provider(name=f"{ctx.hyphen_name}-oauth-provider")
                existing_url = (
                    detail.get("oauth2ProviderConfigOutput", {})
                    .get("customOauth2ProviderConfig", {})
                    .get("oauthDiscovery", {})
                    .get("discoveryUrl", "")
                )
            except Exception:
                existing_url = ""

            if existing_url == current_discovery_url:
                log.info("Reusing OAuth provider (same Cognito pool): %s", ctx.oauth_provider_arn)
            else:
                # Pool changed — delete and recreate
                log.info("Cognito pool changed, recreating OAuth provider...")
                ac.delete_oauth2_credential_provider(name=f"{ctx.hyphen_name}-oauth-provider")
                import time as _time_oauth_del
                _time_oauth_del.sleep(15)  # Allow Secrets Manager secret to be fully deleted
                resp = ac.create_oauth2_credential_provider(
                    name=f"{ctx.hyphen_name}-oauth-provider",
                    credentialProviderVendor="CustomOauth2",
                    oauth2ProviderConfigInput={
                        "customOauth2ProviderConfig": {
                            "oauthDiscovery": {"discoveryUrl": current_discovery_url},
                            "clientId": ctx.client_id,
                            "clientSecret": ctx.client_secret,
                        }
                    },
                )
                ctx.oauth_provider_arn = resp.get("credentialProviderArn", "")
                ctx.oauth_provider_name = resp.get("name", "")
                log.info("Recreated OAuth provider: %s", ctx.oauth_provider_arn)
                import time as _time_oauth_new
                _time_oauth_new.sleep(10)  # Allow new secret to propagate
        else:
            raise


# ---------------------------------------------------------------------------
# Stage: Create Gateway Target
# ---------------------------------------------------------------------------

def create_gateway_target(ctx: DeploymentContext) -> None:
    """Create a Gateway Target linking the Gateway to the Runtime."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "TARGET"})
    ac = boto3.client("bedrock-agentcore-control", region_name=ctx.region)

    # URL-encode the Runtime ARN
    encoded_arn = ctx.runtime_arn.replace(":", "%3A").replace("/", "%2F")
    endpoint_url = (
        f"https://bedrock-agentcore.{ctx.region}.amazonaws.com"
        f"/runtimes/{encoded_arn}/invocations"
    )

    log.info("Creating Gateway Target for runtime %s...", ctx.runtime_id)
    try:
        resp = ac.create_gateway_target(
            gatewayIdentifier=ctx.gateway_id,
            name=f"{ctx.hyphen_name}-target",
            targetConfiguration={
                "mcp": {
                    "mcpServer": {
                        "endpoint": endpoint_url,
                    }
                }
            },
            credentialProviderConfigurations=[{
                "credentialProviderType": "OAUTH",
                "credentialProvider": {
                    "oauthCredentialProvider": {
                        "providerArn": ctx.oauth_provider_arn,
                        "scopes": ["mcp-gateway/invoke"],
                    }
                },
            }],
        )
        ctx.gateway_target_id = resp["targetId"]
        log.info("Gateway Target created: %s", ctx.gateway_target_id)
    except ac.exceptions.ConflictException:
        targets = ac.list_gateway_targets(gatewayIdentifier=ctx.gateway_id).get("items", [])
        existing = next((t for t in targets if t.get("name") == f"{ctx.hyphen_name}-target"), None)
        if existing is None:
            raise DeploymentError("TARGET", "Gateway Target already exists but could not be found by name.")
        if existing.get("status") == "FAILED":
            log.info("Existing Gateway Target is FAILED, deleting and recreating...")
            ac.delete_gateway_target(gatewayIdentifier=ctx.gateway_id, targetId=existing["targetId"])
            import time as _time
            _time.sleep(3)
            resp = ac.create_gateway_target(
                gatewayIdentifier=ctx.gateway_id,
                name=f"{ctx.hyphen_name}-target",
                targetConfiguration={"mcp": {"mcpServer": {"endpoint": endpoint_url}}},
                credentialProviderConfigurations=[{
                    "credentialProviderType": "OAUTH",
                    "credentialProvider": {
                        "oauthCredentialProvider": {
                            "providerArn": ctx.oauth_provider_arn,
                            "scopes": ["mcp-gateway/invoke"],
                        }
                    },
                }],
            )
            ctx.gateway_target_id = resp["targetId"]
            log.info("Gateway Target recreated: %s", ctx.gateway_target_id)
        else:
            ctx.gateway_target_id = existing["targetId"]
            log.info("Reusing existing Gateway Target: %s", ctx.gateway_target_id)

    try:
        poll_status(
            get_fn=lambda: ac.get_gateway_target(
                gatewayIdentifier=ctx.gateway_id, targetId=ctx.gateway_target_id
            ),
            ready_status="READY",
            failed_status="FAILED",
            timeout=300,
            stage="TARGET",
        )
        log.info("Gateway Target is READY.")
    except DeploymentError as err:
        if "Unable to connect" in err.details or "FAILED" in err.message:
            print(
                "\nGateway Target failed. Common causes:\n"
                "  1. Runtime JWT authorizer missing allowedClients entry\n"
                "  2. Container not listening on port 8080\n"
                "  3. Missing MCP_TRANSPORT=streamable-http env var in container\n"
                "  4. OAuth provider ARN or scope mismatch\n"
            )
        raise


# ---------------------------------------------------------------------------
# Stage: Output Deployment Results
# ---------------------------------------------------------------------------

def output_results(ctx: DeploymentContext) -> None:
    """Write deployment JSON file and print human-readable summary."""
    log = StageLogger(logging.getLogger(__name__), {"stage": "OUTPUT"})

    # Resolve cognito_domain from pool if not already set (e.g. when reusing existing gateway)
    if not ctx.cognito_domain and ctx.cognito_pool_id:
        try:
            _cognito = boto3.client("cognito-idp", region_name=ctx.region)
            _pool = _cognito.describe_user_pool(UserPoolId=ctx.cognito_pool_id)
            ctx.cognito_domain = _pool["UserPool"].get("Domain", "")
        except Exception:
            pass

    token_url = (
        f"https://{ctx.cognito_domain}.auth.{ctx.region}.amazoncognito.com/oauth2/token"
        if ctx.cognito_domain else ""
    )

    output: dict[str, Any] = {
        "server_name": ctx.server_name,
        "region": ctx.region,
        # Connection details
        "gateway_url": ctx.gateway_url,
        "client_id": ctx.client_id,
        "client_secret": ctx.client_secret,
        "token_url": token_url,
        # AgentCore resources
        "gateway_id": ctx.gateway_id,
        "gateway_target_id": ctx.gateway_target_id,
        "runtime_id": ctx.runtime_id,
        "runtime_arn": ctx.runtime_arn,
        "oauth_provider_arn": ctx.oauth_provider_arn,
        # ECR
        "ecr_uri": ctx.ecr_uri,
        "ecr_repo_name": ctx.server_name,
        # IAM
        "iam_role_arn": ctx.iam_role_arn,
        "iam_role_name": ctx.iam_role_name,
        "iam_gateway_role_name": f"agentcore-gateway-{ctx.server_name}-role",
        "iam_policy_names": ctx.iam_policy_names,
        # Cognito
        "cognito_pool_id": ctx.cognito_pool_id,
        "cognito_domain": ctx.cognito_domain,
    }

    if ctx.reuse_existing:
        output["resources_reused"] = ["gateway", "cognito_pool", "oauth_provider"]
        output["resources_created"] = [
            "ecr_repo", "runtime", "gateway_target", "iam_role", "cognito_app_client"
        ]

    output_file = Path(f"{ctx.server_name}-deployment.json")
    output_file.write_text(_json_mod.dumps(output, indent=2))
    log.info("Deployment details written to %s", output_file)

    print("\n" + "=" * 60)
    print(f"  Deployment Complete: {ctx.server_name}")
    print("=" * 60)
    print(f"  Gateway URL:    {ctx.gateway_url}")
    print(f"  Client ID:      {ctx.client_id}")
    print(f"  Client Secret:  {ctx.client_secret}")
    print(f"  Token URL:      {token_url}")
    print(f"  Runtime ID:     {ctx.runtime_id}")
    print(f"  ECR URI:        {ctx.ecr_uri}")
    print(f"  IAM Role ARN:   {ctx.iam_role_arn}")
    print(f"  Output file:    {output_file}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def _resolve_region(args_region: str | None) -> str:
    """Resolve AWS region: CLI arg → AWS config → prompt."""
    if args_region:
        return args_region
    # Try AWS config
    try:
        result = subprocess.run(
            ["aws", "configure", "get", "region"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    # Prompt
    while True:
        region = input("Enter AWS region (e.g. us-east-1): ").strip()
        if region:
            return region


def main() -> None:
    args = parse_args()
    region = _resolve_region(args.region)
    server_name = args.server or None

    global logger
    logger = configure_logging(args, server_name=server_name or "deploy")

    ctx = DeploymentContext(
        server_name=server_name,
        region=region,
        verbose=args.verbose,
    )
    run_pipeline(ctx)


if __name__ == "__main__":
    main()
