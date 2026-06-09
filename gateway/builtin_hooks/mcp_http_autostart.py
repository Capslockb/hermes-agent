"""mcp_http_autostart — boot the Hermes MCP server on streamable-HTTP when the gateway starts.

This is the first built-in gateway hook in the codebase. The hook reads
``config.yaml`` keys under ``mcp_serve:`` and spawns the HTTP transport
of the existing ``mcp_serve.py`` module in a daemon thread. The gateway
itself stays free to handle messaging traffic.

Config schema (``~/.hermes/config.yaml``):

    mcp_serve:
      enabled: true                 # default: false
      transport: http               # default: http (only http is wired here; stdio is opt-in)
      host: 127.0.0.1               # default
      port: 18950                   # default
      no_auth: false                # default false. Set true ONLY for local dev.
      log_level: info               # default

Env-var equivalents (override config when set):
    HERMES_MCP_TRANSPORT, HERMES_MCP_HOST, HERMES_MCP_PORT, HERMES_MCP_API_KEY

Failure modes are silent: if anything blows up we log to gateway.log and
move on, so a broken MCP autostart never blocks gateway startup.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.gateway.builtin.mcp_http_autostart")

# Single shared loop so the autostart hook and any later hook can talk to
# the running MCP server via the same task queue.
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_started: bool = False
_start_lock = threading.Lock()


def _read_config() -> Dict[str, Any]:
    """Pull the ``mcp_serve`` block out of config.yaml.

    Uses ``hermes_cli.config.cfg_get`` to stay consistent with the rest of
    the gateway config loaders. Missing block → empty dict.
    """
    try:
        from hermes_cli.config import cfg_get  # type: ignore[import-not-found]

        block = cfg_get("mcp_serve", default={})
        return block if isinstance(block, dict) else {}
    except Exception as exc:  # pragma: no cover - hermes_cli may not be importable here
        logger.debug("cfg_get failed: %s; falling back to env-only", exc)
        return {}


def _resolve() -> Dict[str, Any]:
    """Compute final transport settings: config first, then env, then defaults."""
    cfg = _read_config()
    if not cfg.get("enabled", False) and not os.getenv("HERMES_MCP_PORT"):
        return {"enabled": False}

    return {
        "enabled": True,
        "transport": os.getenv("HERMES_MCP_TRANSPORT", cfg.get("transport", "http")),
        "host": os.getenv("HERMES_MCP_HOST", cfg.get("host", "127.0.0.1")),
        "port": int(os.getenv("HERMES_MCP_PORT", str(cfg.get("port", 18950)))),
        "api_key": os.getenv("HERMES_MCP_API_KEY", cfg.get("api_key")),
        "no_auth": bool(cfg.get("no_auth", False)) or os.getenv("HERMES_MCP_NO_AUTH") == "1",
        "log_level": os.getenv("HERMES_MCP_LOG_LEVEL", cfg.get("log_level", "info")),
    }


def _server_thread_target(settings: Dict[str, Any]) -> None:
    """Run mcp_serve.run_mcp_server() in its own asyncio loop on a daemon thread."""
    global _loop
    try:
        # Make sure we can import the upstream module from the gateway process.
        # The repo root is on sys.path when running as a module, but the gateway
        # may have been launched with a different cwd, so add it defensively.
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        import mcp_serve  # type: ignore[import-not-found]
    except Exception as exc:
        logger.error("mcp_http_autostart: cannot import mcp_serve: %s", exc)
        return

    # Force the argv our _parse_transport_args expects, then delegate.
    sys.argv = [
        "mcp_serve.py",
        "--transport", settings["transport"],
        "--host", settings["host"],
        "--port", str(settings["port"]),
    ]
    if settings.get("api_key"):
        sys.argv += ["--api-key", settings["api_key"]]
    if settings.get("no_auth"):
        sys.argv += ["--no-auth"]

    logger.info(
        "mcp_http_autostart: starting MCP server on http://%s:%d (auth=%s)",
        settings["host"], settings["port"],
        "off" if settings["no_auth"] else "bearer",
    )

    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    try:
        mcp_serve.run_mcp_server(verbose=False)
    except SystemExit as exc:  # mcp_serve.py exits with code 2 on bad config
        logger.warning("mcp_http_autostart: mcp_serve exited: %s", exc)
    except Exception as exc:
        logger.exception("mcp_http_autostart: mcp_serve crashed: %s", exc)
    finally:
        try:
            _loop.close()
        except Exception:  # pragma: no cover
            pass
        _loop = None


def register(gateway_runner: Any) -> None:
    """Called by ``GatewayHooks._register_builtin_hooks()``.

    Spawns the daemon thread that hosts the MCP HTTP server. Idempotent:
    repeated calls are a no-op once the server is up.
    """
    global _started, _thread
    with _start_lock:
        if _started:
            return

        settings = _resolve()
        if not settings.get("enabled"):
            logger.info("mcp_http_autostart: disabled (set mcp_serve.enabled: true in config.yaml)")
            return

        if not settings.get("api_key") and not settings.get("no_auth"):
            logger.warning(
                "mcp_http_autostart: HTTP transport requires auth. Either set "
                "mcp_serve.api_key in config.yaml (or HERMES_MCP_API_KEY env) "
                "or set mcp_serve.no_auth: true for local-only testing. "
                "Autostart will NOT start until this is fixed."
            )
            return

        _thread = threading.Thread(
            target=_server_thread_target,
            args=(settings,),
            name="mcp-http-autostart",
            daemon=True,
        )
        _thread.start()
        _started = True
        logger.info("mcp_http_autostart: thread launched (PID lives in the gateway process)")


def get_status() -> Dict[str, Any]:
    """Introspection helper for /healthz or future /diagnostic commands."""
    return {
        "enabled": _started,
        "thread_alive": bool(_thread and _thread.is_alive()),
        "loop_running": _loop is not None and not _loop.is_closed(),
    }
