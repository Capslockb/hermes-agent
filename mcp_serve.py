"""
Hermes MCP Server — expose messaging conversations as MCP tools.

Starts a stdio MCP server that lets any MCP client (Claude Code, Cursor, Codex,
etc.) list conversations, read message history, send messages, poll for live
events, and manage approval requests across all connected platforms.

Matches OpenClaw's 9-tool MCP channel bridge surface:
  conversations_list, conversation_get, messages_read, attachments_fetch,
  events_poll, events_wait, messages_send, permissions_list_open,
  permissions_respond

Plus: channels_list (Hermes-specific extra)

Usage:
    hermes mcp serve
    hermes mcp serve --verbose

MCP client config (e.g. claude_desktop_config.json):
    {
        "mcpServers": {
            "hermes": {
                "command": "hermes",
                "args": ["mcp", "serve"]
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("hermes.mcp_serve")

# ---------------------------------------------------------------------------
# Lazy MCP SDK import
# ---------------------------------------------------------------------------

_MCP_SERVER_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP

    _MCP_SERVER_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sessions_dir() -> Path:
    """Return the sessions directory using HERMES_HOME."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "sessions"
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "sessions"


def _get_session_db():
    """Get a SessionDB instance for reading message transcripts."""
    try:
        from hermes_state import SessionDB
        return SessionDB()
    except Exception as e:
        logger.debug("SessionDB unavailable: %s", e)
        return None


def _load_sessions_index() -> dict:
    """Load the gateway sessions.json index directly.

    Returns a dict of session_key -> entry_dict with platform routing info.
    This avoids importing the full SessionStore which needs GatewayConfig.
    """
    sessions_file = _get_sessions_dir() / "sessions.json"
    if not sessions_file.exists():
        return {}
    try:
        with open(sessions_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to load sessions.json: %s", e)
        return {}


def _load_channel_directory() -> dict:
    """Load the cached channel directory for available targets."""
    try:
        from hermes_constants import get_hermes_home
        directory_file = get_hermes_home() / "channel_directory.json"
    except ImportError:
        directory_file = Path(
            os.environ.get("HERMES_HOME", Path.home() / ".hermes")
        ) / "channel_directory.json"

    if not directory_file.exists():
        return {}
    try:
        with open(directory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to load channel_directory.json: %s", e)
        return {}


def _coerce_int(
    value,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Coerce value to int with fallback and clamping.

    Used at MCP tool boundaries to handle invalid types from external clients.
    Returns default if value cannot be converted to int.
    """
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(coerced, maximum))


def _extract_message_content(msg: dict) -> str:
    """Extract text content from a message, handling multi-part content."""
    content = msg.get("content", "")
    if isinstance(content, list):
        text_parts = [
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(text_parts)
    return str(content) if content else ""


def _extract_attachments(msg: dict) -> List[dict]:
    """Extract non-text attachments from a message.

    Finds: multi-part image/file content blocks, MEDIA: tags in text,
    image URLs, and file references.
    """
    attachments = []
    content = msg.get("content", "")

    # Multi-part content blocks (image_url, file, etc.)
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "image_url":
                url = part.get("image_url", {}).get("url", "") if isinstance(part.get("image_url"), dict) else ""
                if url:
                    attachments.append({"type": "image", "url": url})
            elif ptype == "image":
                url = part.get("url", part.get("source", {}).get("url", ""))
                if url:
                    attachments.append({"type": "image", "url": url})
            elif ptype not in {"text",}:
                # Unknown non-text content type
                attachments.append({"type": ptype, "data": part})

    # MEDIA: tags in text content
    text = _extract_message_content(msg)
    if text:
        media_pattern = re.compile(r'MEDIA:\s*(\S+)')
        for match in media_pattern.finditer(text):
            path = match.group(1)
            attachments.append({"type": "media", "path": path})

    return attachments


# ---------------------------------------------------------------------------
# Event Bridge — polls SessionDB for new messages, maintains event queue
# ---------------------------------------------------------------------------

QUEUE_LIMIT = 1000
POLL_INTERVAL = 0.2  # seconds between DB polls (200ms)


@dataclass
class QueueEvent:
    """An event in the bridge's in-memory queue."""
    cursor: int
    type: str  # "message", "approval_requested", "approval_resolved"
    session_key: str = ""
    data: dict = field(default_factory=dict)


class EventBridge:
    """Background poller that watches SessionDB for new messages and
    maintains an in-memory event queue with waiter support.

    This is the Hermes equivalent of OpenClaw's WebSocket gateway bridge.
    Instead of WebSocket events, we poll the SQLite database for changes.
    """

    def __init__(self):
        self._queue: List[QueueEvent] = []
        self._cursor = 0
        self._lock = threading.Lock()
        self._new_event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_poll_timestamps: Dict[str, float] = {}  # session_key -> unix timestamp
        # In-memory approval tracking (populated from events)
        self._pending_approvals: Dict[str, dict] = {}
        # mtime cache — skip expensive work when files haven't changed
        self._sessions_json_mtime: float = 0.0
        self._state_db_mtime: float = 0.0
        self._cached_sessions_index: dict = {}

    def start(self):
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.debug("EventBridge started")

    def stop(self):
        """Stop the background polling thread."""
        self._running = False
        self._new_event.set()  # Wake any waiters
        if self._thread:
            self._thread.join(timeout=5)
        logger.debug("EventBridge stopped")

    def poll_events(
        self,
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        """Return events since after_cursor, optionally filtered by session_key."""
        with self._lock:
            events = [
                e for e in self._queue
                if e.cursor > after_cursor
                and (not session_key or e.session_key == session_key)
            ][:limit]

        next_cursor = events[-1].cursor if events else after_cursor
        return {
            "events": [
                {"cursor": e.cursor, "type": e.type,
                 "session_key": e.session_key, **e.data}
                for e in events
            ],
            "next_cursor": next_cursor,
        }

    def wait_for_event(
        self,
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> Optional[dict]:
        """Block until a matching event arrives or timeout expires."""
        deadline = time.monotonic() + (timeout_ms / 1000.0)

        while time.monotonic() < deadline:
            with self._lock:
                for e in self._queue:
                    if e.cursor > after_cursor and (
                        not session_key or e.session_key == session_key
                    ):
                        return {
                            "cursor": e.cursor, "type": e.type,
                            "session_key": e.session_key, **e.data,
                        }

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._new_event.clear()
            self._new_event.wait(timeout=min(remaining, POLL_INTERVAL))

        return None

    def list_pending_approvals(self) -> List[dict]:
        """List approval requests observed during this bridge session."""
        with self._lock:
            return sorted(
                self._pending_approvals.values(),
                key=lambda a: a.get("created_at", ""),
            )

    def respond_to_approval(self, approval_id: str, decision: str) -> dict:
        """Resolve a pending approval (best-effort without gateway IPC)."""
        with self._lock:
            approval = self._pending_approvals.pop(approval_id, None)

        if not approval:
            return {"error": f"Approval not found: {approval_id}"}

        self._enqueue(QueueEvent(
            cursor=0,  # Will be set by _enqueue
            type="approval_resolved",
            session_key=approval.get("session_key", ""),
            data={"approval_id": approval_id, "decision": decision},
        ))

        return {"resolved": True, "approval_id": approval_id, "decision": decision}

    def _enqueue(self, event: QueueEvent) -> None:
        """Add an event to the queue and wake any waiters."""
        with self._lock:
            self._cursor += 1
            event.cursor = self._cursor
            self._queue.append(event)
            # Trim queue to limit
            while len(self._queue) > QUEUE_LIMIT:
                self._queue.pop(0)
        self._new_event.set()

    def _poll_loop(self):
        """Background loop: poll SessionDB for new messages."""
        db = _get_session_db()
        if not db:
            logger.warning("EventBridge: SessionDB unavailable, event polling disabled")
            return

        while self._running:
            try:
                self._poll_once(db)
            except Exception as e:
                logger.debug("EventBridge poll error: %s", e)
            time.sleep(POLL_INTERVAL)

    def _poll_once(self, db):
        """Check for new messages across all sessions.

        Uses mtime checks on sessions.json and state.db to skip work
        when nothing has changed — makes 200ms polling essentially free.
        """
        # Check if sessions.json has changed (mtime check is ~1μs)
        sessions_file = _get_sessions_dir() / "sessions.json"
        try:
            sj_mtime = sessions_file.stat().st_mtime if sessions_file.exists() else 0.0
        except OSError:
            sj_mtime = 0.0

        if sj_mtime != self._sessions_json_mtime:
            self._sessions_json_mtime = sj_mtime
            self._cached_sessions_index = _load_sessions_index()

        # Check if state.db has changed
        try:
            from hermes_constants import get_hermes_home
            db_file = get_hermes_home() / "state.db"
        except ImportError:
            db_file = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "state.db"

        try:
            db_mtime = db_file.stat().st_mtime if db_file.exists() else 0.0
        except OSError:
            db_mtime = 0.0

        if db_mtime == self._state_db_mtime and sj_mtime == self._sessions_json_mtime:
            return  # Nothing changed since last poll — skip entirely

        self._state_db_mtime = db_mtime
        entries = self._cached_sessions_index

        for session_key, entry in entries.items():
            session_id = entry.get("session_id", "")
            if not session_id:
                continue

            last_seen = self._last_poll_timestamps.get(session_key, 0.0)

            try:
                messages = db.get_messages(session_id)
            except Exception:
                continue

            if not messages:
                continue

            # Normalize timestamps to float for comparison
            def _ts_float(ts) -> float:
                if isinstance(ts, (int, float)):
                    return float(ts)
                if isinstance(ts, str) and ts:
                    try:
                        return float(ts)
                    except ValueError:
                        # ISO string — parse to epoch
                        try:
                            from datetime import datetime
                            return datetime.fromisoformat(ts).timestamp()
                        except Exception:
                            return 0.0
                return 0.0

            # Find messages newer than our last seen timestamp
            new_messages = []
            for msg in messages:
                ts = _ts_float(msg.get("timestamp", 0))
                role = msg.get("role", "")
                if role not in {"user", "assistant"}:
                    continue
                if ts > last_seen:
                    new_messages.append(msg)

            for msg in new_messages:
                content = _extract_message_content(msg)
                if not content:
                    continue
                self._enqueue(QueueEvent(
                    cursor=0,
                    type="message",
                    session_key=session_key,
                    data={
                        "role": msg.get("role", ""),
                        "content": content[:500],
                        "timestamp": str(msg.get("timestamp", "")),
                        "message_id": str(msg.get("id", "")),
                    },
                ))

            # Update last seen to the most recent message timestamp
            all_ts = [_ts_float(m.get("timestamp", 0)) for m in messages]
            if all_ts:
                latest = max(all_ts)
                if latest > last_seen:
                    self._last_poll_timestamps[session_key] = latest


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

def create_mcp_server(event_bridge: Optional[EventBridge] = None) -> "FastMCP":
    """Create and return the Hermes MCP server with all tools registered."""
    if not _MCP_SERVER_AVAILABLE:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            f"Install with: {sys.executable} -m pip install 'mcp'"
        )

    mcp = FastMCP(
        "hermes",
        instructions=(
            "Hermes Agent messaging bridge. Use these tools to interact with "
            "conversations across Telegram, Discord, Slack, WhatsApp, Signal, "
            "Matrix, and other connected platforms."
        ),
    )

    bridge = event_bridge or EventBridge()

    # -- conversations_list ------------------------------------------------

    @mcp.tool()
    def conversations_list(
        platform: Optional[str] = None,
        limit: int = 50,
        search: Optional[str] = None,
    ) -> str:
        """List active messaging conversations across connected platforms.

        Returns conversations with their session keys (needed for messages_read),
        platform, chat type, display name, and last activity time.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
            limit: Maximum number of conversations to return (default 50)
            search: Optional text to filter conversations by name
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        entries = _load_sessions_index()
        conversations = []

        for key, entry in entries.items():
            origin = entry.get("origin", {})
            entry_platform = entry.get("platform") or origin.get("platform", "")

            if platform and entry_platform.lower() != platform.lower():
                continue

            display_name = entry.get("display_name", "")
            chat_name = origin.get("chat_name", "")
            if search:
                search_lower = search.lower()
                if (search_lower not in display_name.lower()
                        and search_lower not in chat_name.lower()
                        and search_lower not in key.lower()):
                    continue

            conversations.append({
                "session_key": key,
                "session_id": entry.get("session_id", ""),
                "platform": entry_platform,
                "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
                "display_name": display_name,
                "chat_name": chat_name,
                "user_name": origin.get("user_name", ""),
                "updated_at": entry.get("updated_at", ""),
            })

        conversations.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        conversations = conversations[:limit]

        return json.dumps({
            "count": len(conversations),
            "conversations": conversations,
        }, indent=2)

    # -- conversation_get --------------------------------------------------

    @mcp.tool()
    def conversation_get(session_key: str) -> str:
        """Get detailed info about one conversation by its session key.

        Args:
            session_key: The session key from conversations_list
        """
        entries = _load_sessions_index()
        entry = entries.get(session_key)

        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})

        origin = entry.get("origin", {})
        return json.dumps({
            "session_key": session_key,
            "session_id": entry.get("session_id", ""),
            "platform": entry.get("platform") or origin.get("platform", ""),
            "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
            "display_name": entry.get("display_name", ""),
            "user_name": origin.get("user_name", ""),
            "chat_name": origin.get("chat_name", ""),
            "chat_id": origin.get("chat_id", ""),
            "thread_id": origin.get("thread_id"),
            "updated_at": entry.get("updated_at", ""),
            "created_at": entry.get("created_at", ""),
            "input_tokens": entry.get("input_tokens", 0),
            "output_tokens": entry.get("output_tokens", 0),
            "total_tokens": entry.get("total_tokens", 0),
        }, indent=2)

    # -- messages_read -----------------------------------------------------

    @mcp.tool()
    def messages_read(
        session_key: str,
        limit: int = 50,
    ) -> str:
        """Read recent messages from a conversation.

        Returns the message history in chronological order with role, content,
        and timestamp for each message.

        Args:
            session_key: The session key from conversations_list
            limit: Maximum number of messages to return (default 50, most recent)
        """
        limit = _coerce_int(limit, default=50, minimum=1, maximum=200)
        entries = _load_sessions_index()
        entry = entries.get(session_key)
        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})

        session_id = entry.get("session_id", "")
        if not session_id:
            return json.dumps({"error": "No session ID for this conversation"})

        db = _get_session_db()
        if not db:
            return json.dumps({"error": "Session database unavailable"})

        try:
            all_messages = db.get_messages(session_id)
        except Exception as e:
            return json.dumps({"error": f"Failed to read messages: {e}"})

        filtered = []
        for msg in all_messages:
            role = msg.get("role", "")
            if role in {"user", "assistant"}:
                content = _extract_message_content(msg)
                if content:
                    filtered.append({
                        "id": str(msg.get("id", "")),
                        "role": role,
                        "content": content[:2000],
                        "timestamp": msg.get("timestamp", ""),
                    })

        messages = filtered[-limit:]

        return json.dumps({
            "session_key": session_key,
            "count": len(messages),
            "total_in_session": len(filtered),
            "messages": messages,
        }, indent=2)

    # -- attachments_fetch -------------------------------------------------

    @mcp.tool()
    def attachments_fetch(
        session_key: str,
        message_id: str,
    ) -> str:
        """List non-text attachments for a message in a conversation.

        Extracts images, media files, and other non-text content blocks
        from the specified message.

        Args:
            session_key: The session key from conversations_list
            message_id: The message ID from messages_read
        """
        entries = _load_sessions_index()
        entry = entries.get(session_key)
        if not entry:
            return json.dumps({"error": f"Conversation not found: {session_key}"})

        session_id = entry.get("session_id", "")
        if not session_id:
            return json.dumps({"error": "No session ID for this conversation"})

        db = _get_session_db()
        if not db:
            return json.dumps({"error": "Session database unavailable"})

        try:
            all_messages = db.get_messages(session_id)
        except Exception as e:
            return json.dumps({"error": f"Failed to read messages: {e}"})

        # Find the target message
        target_msg = None
        for msg in all_messages:
            if str(msg.get("id", "")) == message_id:
                target_msg = msg
                break

        if not target_msg:
            return json.dumps({"error": f"Message not found: {message_id}"})

        attachments = _extract_attachments(target_msg)

        return json.dumps({
            "message_id": message_id,
            "count": len(attachments),
            "attachments": attachments,
        }, indent=2)

    # -- events_poll -------------------------------------------------------

    @mcp.tool()
    def events_poll(
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """Poll for new conversation events since a cursor position.

        Returns events that have occurred since the given cursor. Use the
        returned next_cursor value for subsequent polls.

        Event types: message, approval_requested, approval_resolved

        Args:
            after_cursor: Return events after this cursor (0 for all)
            session_key: Optional filter to one conversation
            limit: Maximum events to return (default 20)
        """
        after_cursor = _coerce_int(after_cursor, default=0, minimum=0, maximum=10**18)
        limit = _coerce_int(limit, default=20, minimum=1, maximum=200)
        result = bridge.poll_events(
            after_cursor=after_cursor,
            session_key=session_key,
            limit=limit,
        )
        return json.dumps(result, indent=2)

    # -- events_wait -------------------------------------------------------

    @mcp.tool()
    def events_wait(
        after_cursor: int = 0,
        session_key: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> str:
        """Wait for the next conversation event (long-poll).

        Blocks until a matching event arrives or the timeout expires.
        Use this for near-real-time event delivery without polling.

        Args:
            after_cursor: Wait for events after this cursor
            session_key: Optional filter to one conversation
            timeout_ms: Maximum wait time in milliseconds (default 30000)
        """
        after_cursor = _coerce_int(after_cursor, default=0, minimum=0, maximum=10**18)
        timeout_ms = _coerce_int(
            timeout_ms,
            default=30000,
            minimum=0,
            maximum=300000,
        )  # Cap at 5 minutes
        event = bridge.wait_for_event(
            after_cursor=after_cursor,
            session_key=session_key,
            timeout_ms=timeout_ms,
        )
        if event:
            return json.dumps({"event": event}, indent=2)
        return json.dumps({"event": None, "reason": "timeout"}, indent=2)

    # -- messages_send -----------------------------------------------------

    @mcp.tool()
    def messages_send(
        target: str,
        message: str,
    ) -> str:
        """Send a message to a platform conversation.

        The target format is "platform:chat_id" — same format used by the
        channels_list tool. You can also use human-friendly channel names
        that will be resolved automatically.

        Examples:
            target="telegram:6308981865"
            target="discord:#general"
            target="slack:#engineering"

        Args:
            target: Platform target in "platform:identifier" format
            message: The message text to send
        """
        if not target or not message:
            return json.dumps({"error": "Both target and message are required"})

        try:
            from tools.send_message_tool import send_message_tool
            result_str = send_message_tool(
                {"action": "send", "target": target, "message": message}
            )
            return result_str
        except ImportError:
            return json.dumps({"error": "Send message tool not available"})
        except Exception as e:
            return json.dumps({"error": f"Send failed: {e}"})

    # -- channels_list -----------------------------------------------------

    @mcp.tool()
    def channels_list(platform: Optional[str] = None) -> str:
        """List available messaging channels and targets across platforms.

        Returns channels that you can send messages to. The target strings
        returned here can be used directly with the messages_send tool.

        Args:
            platform: Filter by platform name (telegram, discord, slack, etc.)
        """
        directory = _load_channel_directory()
        if not directory:
            entries = _load_sessions_index()
            targets = []
            seen = set()
            for key, entry in entries.items():
                origin = entry.get("origin", {})
                p = entry.get("platform") or origin.get("platform", "")
                chat_id = origin.get("chat_id", "")
                if not p or not chat_id:
                    continue
                if platform and p.lower() != platform.lower():
                    continue
                target_str = f"{p}:{chat_id}"
                if target_str in seen:
                    continue
                seen.add(target_str)
                targets.append({
                    "target": target_str,
                    "platform": p,
                    "name": entry.get("display_name") or origin.get("chat_name", ""),
                    "chat_type": entry.get("chat_type", origin.get("chat_type", "")),
                })
            return json.dumps({"count": len(targets), "channels": targets}, indent=2)

        channels = []
        for plat, entries_list in directory.get("platforms", {}).items():
            if platform and plat.lower() != platform.lower():
                continue
            if isinstance(entries_list, list):
                for ch in entries_list:
                    if isinstance(ch, dict):
                        chat_id = ch.get("id", ch.get("chat_id", ""))
                        channels.append({
                            "target": f"{plat}:{chat_id}" if chat_id else plat,
                            "platform": plat,
                            "name": ch.get("name", ch.get("display_name", "")),
                            "chat_type": ch.get("type", ""),
                        })

        return json.dumps({"count": len(channels), "channels": channels}, indent=2)

    # -- permissions_list_open ---------------------------------------------

    @mcp.tool()
    def permissions_list_open() -> str:
        """List pending approval requests observed during this bridge session.

        Returns exec and plugin approval requests that the bridge has seen
        since it started. Approvals are live-session only — older approvals
        from before the bridge connected are not included.
        """
        approvals = bridge.list_pending_approvals()
        return json.dumps({
            "count": len(approvals),
            "approvals": approvals,
        }, indent=2)

    # -- permissions_respond -----------------------------------------------

    @mcp.tool()
    def permissions_respond(
        id: str,
        decision: str,
    ) -> str:
        """Respond to a pending approval request.

        Args:
            id: The approval ID from permissions_list_open
            decision: One of "allow-once", "allow-always", or "deny"
        """
        if decision not in {"allow-once", "allow-always", "deny"}:
            return json.dumps({
                "error": f"Invalid decision: {decision}. "
                         f"Must be allow-once, allow-always, or deny"
            })

        result = bridge.respond_to_approval(id, decision)
        return json.dumps(result, indent=2)

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public settings surface — lets callers (e.g. mcp_http_autostart) pass
# transport settings directly without touching process-global ``sys.argv``.
# Keeping the CLI parser as the user-facing entrypoint and the keyword args
# as the programmatic one means in-process embedders (the autostart hook)
# don't mutate argv and risk side effects elsewhere in the gateway.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPServerSettings:
    """Programmatic transport settings for :func:`run_mcp_server`."""

    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 18950
    path: str = "/mcp"
    api_key: Optional[str] = None
    no_auth: bool = False
    verbose: bool = False


def _settings_from_argv(argv: List[str]) -> MCPServerSettings:
    """Bridge the CLI parser into the programmatic settings object."""
    (
        transport,
        host,
        port,
        path,
        api_key,
        no_auth,
        verbose,
    ) = _parse_transport_args(argv)
    return MCPServerSettings(
        transport=transport,
        host=host,
        port=port,
        path=path,
        api_key=api_key,
        no_auth=no_auth,
        verbose=verbose,
    )


def run_mcp_server(
    verbose: bool = False,
    *,
    settings: Optional[MCPServerSettings] = None,
) -> None:
    """Start the Hermes MCP server (stdio or streamable-HTTP).

    Transports:
      - ``stdio`` (default): one MCP client per process, speaking JSON-RPC on
        stdin/stdout. Same behaviour as the original `hermes mcp serve`.
      - ``http``: serve the FastMCP ASGI app on host:port. Multiple clients
        can connect; requires a bearer token (or the explicit ``--no-auth``
        flag for local-only testing).

    Selection: pass ``--transport http`` (CLI), set
    ``HERMES_MCP_TRANSPORT=http`` (env), set ``HERMES_MCP_PORT`` to enable
    HTTP at the given port with stdio as the fallback default, or call
    ``run_mcp_server(settings=MCPServerSettings(...))`` directly (the path
    in-process embedders like the gateway autostart hook should use — it
    avoids mutating the host process's ``sys.argv``).
    """
    if not _MCP_SERVER_AVAILABLE:
        print(
            "Error: MCP server requires the 'mcp' package.\n"
            f"Install with: {sys.executable} -m pip install 'mcp'",
            file=sys.stderr,
        )
        sys.exit(1)

    if settings is None:
        # CLI / __main__ path — parse the user's argv. The ``verbose`` kwarg
        # is honored as a fallback for callers that don't go through
        # settings but do want to override argv-based detection.
        settings = _settings_from_argv(sys.argv[1:])
        if verbose and not settings.verbose:
            object.__setattr__(settings, "verbose", True)
            settings = replace(settings, verbose=True)

    if settings.verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # ---- transport selection ---------------------------------------------
    transport = settings.transport
    http_host = settings.host
    http_port = settings.port
    http_path = settings.path
    api_key = settings.api_key
    no_auth = settings.no_auth

    bridge = EventBridge()
    bridge.start()

    server = create_mcp_server(event_bridge=bridge)

    # ---- optional ari.* tool registration --------------------------------
    # Always import-and-try; skip silently if the ari tools file isn't present
    # (allows the stdio entrypoint to keep working when only the messaging
    # server is bundled).
    try:
        from mcp_serve_ari_tools import register_ari_tools  # type: ignore[import-not-found]

        register_ari_tools(server)
        logger.info("Registered ari.* MCP tools (asterisk ARI)")
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to register ari.* tools: %s", exc)

    import asyncio

    async def _run() -> None:
        try:
            if transport == "stdio":
                await server.run_stdio_async()
            else:  # "http"
                await _run_http(server, http_host, http_port, http_path, api_key, no_auth)
        finally:
            bridge.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        bridge.stop()


# ---------------------------------------------------------------------------
# Transport plumbing
# ---------------------------------------------------------------------------


def _parse_transport_args(argv: List[str]) -> tuple:
    """Parse ``[--transport, --host, --port, --path, --api-key, --no-auth, --verbose]``.

    Returns: ``(transport, host, port, path, api_key, no_auth, verbose)``
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="hermes-mcp-serve",
        description=(
            "Expose Hermes as an MCP server. "
            "Default transport is stdio (one client per process). "
            "Use --transport http to expose the streamable-HTTP ASGI app."
        ),
        add_help=True,
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default=os.getenv("HERMES_MCP_TRANSPORT", "stdio"),
        help="MCP transport. stdio (default) or http.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HERMES_MCP_HOST", "127.0.0.1"),
        help="HTTP bind host (default 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("HERMES_MCP_PORT", "0") or 0),
        help=(
            "HTTP bind port. If 0 (default) the stdio transport is used. "
            "Set HERMES_MCP_PORT=18950 to expose HTTP at the canonical port."
        ),
    )
    parser.add_argument(
        "--path",
        default=os.getenv("HERMES_MCP_PATH", "/mcp"),
        help="HTTP mount path (default /mcp).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("HERMES_MCP_API_KEY"),
        help=(
            "Bearer token required on HTTP requests. Falls back to "
            "HERMES_MCP_API_KEY env. If unset, HTTP refuses to start unless "
            "--no-auth is given."
        ),
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable bearer-token auth on the HTTP transport (LOCAL ONLY).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging (debug).",
    )
    # When --port is provided non-zero, force transport=http
    args = parser.parse_args(argv)
    if args.port and args.transport == "stdio":
        args.transport = "http"
    if args.transport == "http" and not args.api_key and not args.no_auth:
        print(
            "Error: HTTP transport requires --api-key or HERMES_MCP_API_KEY "
            "(or pass --no-auth for local-only testing).",
            file=sys.stderr,
        )
        sys.exit(2)
    return (
        args.transport,
        args.host,
        args.port or 18950,
        args.path,
        args.api_key,
        args.no_auth,
        args.verbose,
    )


async def _run_http(
    server: "FastMCP",
    host: str,
    port: int,
    path: str,
    api_key: Optional[str],
    no_auth: bool,
) -> None:
    """Serve FastMCP over streamable-HTTP, optionally behind a bearer-token ASGI middleware.

    Also serves:
      - ``GET /.well-known/mcp.json``   — MCP discovery document (spec §3.1).
        Conforming MCP clients (Claude Desktop >=0.7, Cursor >=0.40, recent
        Codex) auto-detect this and prompt to connect.
      - ``GET /healthz``                — liveness probe (always 200, no auth).
    """
    try:
        # FastMCP ships the ASGI app directly (mcp>=1.9)
        mcp_asgi = server.streamable_http_app()  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        # Older FastMCP: use run_streamable_http_async instead
        await server.run_streamable_http_async()  # type: ignore[attr-defined]
        return

    # Build the parent app: discovery + healthz on the root, MCP at /mcp.
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    async def _wellknown_mcp(_request):
        """MCP discovery document per modelcontextprotocol.io spec §3.1.

        The ``transports`` block advertises the streamable-HTTP endpoint so
        clients know where to connect without manual config.
        """
        return JSONResponse(
            {
                "mcp_version": "2025-03-26",
                "server": {
                    "name": "hermes",
                    "version": "1.26.0",
                    "description": (
                        "Hermes Agent messaging bridge + Asterisk ARI call control. "
                        "Use these tools to read/write conversations across connected "
                        "platforms and to control live phone calls via ari.* tools."
                    ),
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "transports": {
                    "streamable-http": {
                        "endpoint": "/mcp",
                        "auth": {"type": "bearer"} if not no_auth else None,
                    }
                },
                "tools_hint": [
                    "conversations_list",
                    "messages_read",
                    "messages_send",
                    "channels_list",
                    "events_poll",
                    "ari.answer",
                    "ari.hangup",
                    "ari.play_tts",
                    "ari.transfer",
                    "ari.dial",
                ],
            },
            headers={"Cache-Control": "no-store"},
        )

    async def _healthz(_request):
        return JSONResponse({"status": "ok", "transport": "http", "ari_tools": _has_ari_tools()})

    # If the user's path is "/mcp" (default), mount the MCP app under /mcp and
    # expose discovery at the root. If they picked a custom path, mount it there.
    mount_path = path if path.startswith("/") else f"/{path}"
    # Strip trailing slash so Mount("/mcp", ...) and Mount("/mcp/", ...) both work.
    mount_path = mount_path.rstrip("/") or "/mcp"

    parent = Starlette(
        routes=[
            Route("/.well-known/mcp.json", _wellknown_mcp),
            Route("/.well-known/mcp", _wellknown_mcp),  # alias some clients look for
            Route("/healthz", _healthz),
            Mount(mount_path, app=mcp_asgi),
        ]
    )

    wrapped = _BearerAuthMiddleware(parent, expected_token=api_key, disabled=no_auth)

    try:
        import uvicorn  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        print(
            f"Error: HTTP transport requires 'uvicorn': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    config = uvicorn.Config(
        app=wrapped,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    uvi = uvicorn.Server(config)
    logger.info(
        "Hermes MCP server listening on http://%s:%d (auth=%s, ari_tools=%s, "
        "discovery at /.well-known/mcp.json, mcp at %s)",
        host,
        port,
        "off" if no_auth else "bearer",
        _has_ari_tools(),
        mount_path,
    )
    await uvi.serve()


def _has_ari_tools() -> bool:
    try:
        import mcp_serve_ari_tools  # type: ignore[import-not-found]

        return hasattr(mcp_serve_ari_tools, "register_ari_tools")
    except ImportError:
        return False


class _BearerAuthMiddleware:
    """ASGI middleware that requires ``Authorization: Bearer <token>`` on HTTP transports.

    Stdio and local-loopback tests can opt out with ``--no-auth``. The token
    comparison is constant-time to avoid timing oracles.
    """

    def __init__(self, app, *, expected_token: Optional[str], disabled: bool) -> None:
        self._app = app
        self._expected = (expected_token or "").encode("utf-8") if expected_token else b""
        self._disabled = disabled

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            # WebSocket / lifespan scopes: pass through unchanged
            await self._app(scope, receive, send)
            return
        # Public, unauthenticated endpoints. The spec requires discovery to
        # be reachable without credentials, and healthz must work behind a
        # load balancer / k8s probe that doesn't have our bearer token.
        path = scope.get("path", "")
        if path in ("/healthz", "/.well-known/mcp.json", "/.well-known/mcp"):
            await self._app(scope, receive, send)
            return
        if self._disabled or not self._expected:
            await self._app(scope, receive, send)
            return
        # Extract Authorization header (case-insensitive)
        auth_value: Optional[bytes] = None
        for k, v in scope.get("headers", []):
            if k.lower() == b"authorization":
                auth_value = v
                break
        if not auth_value or not auth_value.startswith(b"Bearer "):
            await self._reject(send, status=401, reason="Missing bearer token")
            return
        presented = auth_value[len(b"Bearer "):]
        if not _constant_time_eq(presented, self._expected):
            await self._reject(send, status=403, reason="Invalid bearer token")
            return
        await self._app(scope, receive, send)

    @staticmethod
    async def _reject(send, *, status: int, reason: str) -> None:
        body = json.dumps({"error": reason}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _constant_time_eq(a: bytes, b: bytes) -> bool:
    """Constant-time bytes comparison (avoid timing oracles)."""
    import hmac

    return hmac.compare_digest(a, b)


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow direct invocation: `python mcp_serve.py [--transport http|stdio ...]`
    # The `hermes mcp serve` subcommand in hermes_cli invokes run_mcp_server() directly.
    run_mcp_server(verbose=False)
