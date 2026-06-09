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

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.gateway.builtin.mcp_http_autostart")

# Background thread bookkeeping for the autostarted MCP HTTP server.
# The actual event loop is owned/managed by asyncio.run/uvicorn; we don't
# attempt to track or expose it here to avoid misleading status.  Earlier
# versions kept a ``_loop`` reference assigned from inside the thread
# target, but uvicorn never actually used it — so ``get_status()`` would
# happily report ``loop_running=False`` on a perfectly healthy server.
_thread: Optional[threading.Thread] = None
_started: bool = False
_start_lock = threading.Lock()
_uvicorn_server: Optional[Any] = None
_uvicorn_lock = threading.Lock()


def _read_config() -> Dict[str, Any]:
    """Pull the ``mcp_serve`` block out of config.yaml.

    The actual signature of ``hermes_cli.config.cfg_get`` is
    ``cfg_get(cfg: Optional[Dict], *keys, default=...)`` — the *first*
    positional arg is the loaded config dict, not a key path. The previous
    implementation called ``cfg_get("mcp_serve", default={})``, which
    short-circuited through the ``isinstance(cfg, dict)`` guard and
    silently returned the default every time. That meant the
    ``mcp_serve:`` block in ``config.yaml`` was dead weight: the autostart
    hook would only ever fire when ``HERMES_MCP_PORT`` (or another env
    var) was set.

    Fix: load the config dict first, then pass it as the first positional
    arg with ``"mcp_serve"`` as the key path.
    """
    try:
        from hermes_cli.config import load_config, cfg_get  # type: ignore[import-not-found]

        cfg = load_config()
        block = cfg_get(cfg, "mcp_serve", default={})
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
    """Run ``mcp_serve.run_mcp_server()`` in a daemon thread.

    We pass the resolved transport settings directly through
    ``MCPServerSettings`` rather than mutating ``sys.argv`` — the latter
    would leak the MCP transport args into the rest of the gateway process
    (any code that reads ``sys.argv`` for logging or CLI helpers would see
    them).  The thread is daemon, so the gateway process can exit cleanly
    even if uvicorn is mid-handshake.
    """
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

    logger.info(
        "mcp_http_autostart: starting MCP server on http://%s:%d (auth=%s)",
        settings["host"], settings["port"],
        "off" if settings["no_auth"] else "bearer",
    )

    try:
        mcp_serve.run_mcp_server(
            settings=mcp_serve.MCPServerSettings(
                transport=settings["transport"],
                host=settings["host"],
                port=settings["port"],
                path="/mcp",
                api_key=settings.get("api_key"),
                no_auth=settings.get("no_auth", False),
                verbose=False,
            )
        )
    except SystemExit as exc:  # mcp_serve.py exits with code 2 on bad config
        logger.warning("mcp_http_autostart: mcp_serve exited: %s", exc)
    except Exception as exc:
        logger.exception("mcp_http_autostart: mcp_serve crashed: %s", exc)
    finally:
        with _uvicorn_lock:
            global _uvicorn_server
            _uvicorn_server = None


def register_uvicorn_server(server: Any) -> None:
    """Optional hook for ``mcp_serve`` to register the live uvicorn server.

    ``mcp_serve`` doesn't currently call this, but the public hook is here
    so a future refactor (e.g. exposing the uvicorn ``Server`` from
    ``_run_http``) can wire it without a cross-module API change.  When
    the server reference is set, ``get_status()`` will use its
    ``started``/``should_exit`` flags instead of just thread liveness.
    """
    with _uvicorn_lock:
        global _uvicorn_server
        _uvicorn_server = server


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
    """Introspection helper for ``/healthz`` or future ``/diagnostic`` commands.

    Status is derived from:
      * ``thread_alive`` — the daemon thread that hosts the MCP server.
      * ``server_started`` — if a uvicorn ``Server`` has been registered
        via :func:`register_uvicorn_server`, its ``started`` flag.  This
        is the load-bearing signal; a healthy uvicorn reports ``True``
        after the bind completes, and ``False`` if it crashed or hasn't
        bound yet.

    Earlier revisions exposed a ``loop_running`` field that mirrored a
    ``_loop`` variable assigned in the thread target — but uvicorn owns
    the real event loop, so that field could report ``False`` on a
    healthy server.  Removed.
    """
    with _uvicorn_lock:
        server_started: Optional[bool] = None
        if _uvicorn_server is not None:
            started_flag = getattr(_uvicorn_server, "started", None)
            if isinstance(started_flag, bool):
                server_started = started_flag
            should_exit = getattr(_uvicorn_server, "should_exit", None)
            if isinstance(should_exit, bool) and should_exit:
                server_started = False
    return {
        "enabled": _started,
        "thread_alive": bool(_thread and _thread.is_alive()),
        "server_started": server_started,
    }
