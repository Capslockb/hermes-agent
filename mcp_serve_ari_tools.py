from __future__ import annotations

import os
import json
import logging
import httpx
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("hermes.mcp_ari")

# --- Configuration ---
# Defaults to a localhost Asterisk ARI listener. Override via env vars in
# production. Default creds are *only* the standard ARI default — operators
# should override in their .env.
ARI_URL = os.environ.get("ASTERISK_ARI_URL", "http://127.0.0.1:8088")
ARI_USER = os.environ.get("ASTERISK_ARI_USER", "admin")
ARI_PASS = os.environ.get("ASTERISK_ARI_PASS", "admin")

# Optional external TTS endpoint used by ``ari.play_tts``. POST the ``text`` to
# ``{TTS_URL}`` and expect an audio file back (Content-Type audio/*) or a JSON
# body with a ``media_uri`` field. If unset, ``ari.play_tts`` requires the
# caller to pass ``media_uri`` directly. This keeps ARI dependency-free of any
# particular TTS provider while still letting tools synthesize speech.
TTS_URL = os.environ.get("ASTERISK_TTS_URL")  # e.g. http://127.0.0.1:8089/synth


def _ari_request(method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Internal helper to call the Asterisk ARI REST API.

    ARI HTTP semantics:
      * Path parameters (``{channelId}``, ``{playbackId}``) belong in the URL.
      * Operation parameters (e.g. ``media`` for ``/play``) are query string
        params on GET/POST, not JSON bodies. ARI is form-encoded in practice.
      * ``/play`` (NOT ``/playback`` — that's a sub-resource) is the channel
        operation that creates a new Playback object and starts streaming
        media to the channel. See Asterisk docs: "Almost all media is played
        to a channel using the POST /channels/{channel_id}/play operation."
    """
    url = f"{ARI_URL}/ari{endpoint}"
    # ARI expects operation params on the query string, not the body.
    merged_params: Dict[str, Any] = {}
    if params:
        merged_params.update(params)
    if data:
        # If the caller passed a body (e.g. originate ``variables``), send it
        # as form data; otherwise, the per-operation spec wants query params.
        if method.upper() in ("POST", "PUT", "PATCH") and endpoint in ("/channels",):
            merged_params.update(data)
        else:
            merged_params.update(data)
    try:
        with httpx.Client(auth=(ARI_USER, ARI_PASS), timeout=5.0) as client:
            response = client.request(method, url, params=merged_params)
            response.raise_for_status()
            return {
                "status": "success",
                "message": f"ARI {method} {endpoint} successful",
                "data": response.json() if response.content else {},
            }
    except httpx.HTTPStatusError as e:
        return {
            "status": "error",
            "message": f"ARI API returned {e.response.status_code}",
            "error": e.response.text,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"ARI request failed: {str(e)}",
        }


def _synthesize_tts(text: str, voice: str) -> Dict[str, Any]:
    """Best-effort TTS via an external HTTP endpoint.

    Two response shapes are accepted:
      1. ``Content-Type: audio/*`` — the response body is the audio file
         and we return it as a ``data:`` URI ARI can play directly.
      2. ``application/json`` with a ``media_uri`` field — the upstream
         service already stored the audio and returns an ``http:`` or
         ``sound:`` URI.

    Returns a dict with keys ``ok``, ``media_uri``, and (on failure) ``error``.
    """
    # Re-read the env var on every call (instead of using the
    # module-level TTS_URL constant) so operators can flip the service
    # on or off at runtime, and so test fixtures can toggle it without
    # re-importing the module.
    tts_url = os.environ.get("ASTERISK_TTS_URL")
    if not tts_url:
        return {"ok": False, "error": "ASTERISK_TTS_URL not set; pass media_uri explicitly"}
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(tts_url, params={"text": text, "voice": voice})
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if ct.startswith("audio/"):
                import base64
                encoded = base64.b64encode(r.content).decode("ascii")
                return {"ok": True, "media_uri": f"data:{ct};base64,{encoded}"}
            # Try JSON shape
            try:
                body = r.json()
            except Exception:
                return {"ok": False, "error": f"TTS returned non-audio, non-JSON: {ct}"}
            uri = body.get("media_uri") or body.get("uri")
            if not uri:
                return {"ok": False, "error": "TTS JSON missing media_uri/uri"}
            return {"ok": True, "media_uri": uri}
    except Exception as e:
        return {"ok": False, "error": f"TTS request failed: {e}"}


def register_ari_tools(mcp: FastMCP):
    """Register ARI-related tools to the provided FastMCP server."""

    def ari_answer(channel_id: str) -> str:
        """Answer a ringing channel in Asterisk."""
        res = _ari_request("POST", f"/channels/{channel_id}/answer")
        return json.dumps(res)

    def ari_hangup(channel_id: str) -> str:
        """Hang up a channel in Asterisk.

        Optional ``reason_code`` query param (defaults to 16 = normal clearing).
        """
        res = _ari_request("POST", f"/channels/{channel_id}/hangup", params={"reason_code": 16})
        return json.dumps(res)

    def ari_play_tts(channel_id: str, text: str, voice: str = "alice", media_uri: Optional[str] = None) -> str:
        """Play text-to-speech on a channel.

        ARI has no native TTS — ``POST /channels/{id}/play`` only accepts a
        pre-existing ``media`` URI. This tool:

          1. If the caller passes ``media_uri`` directly, plays it verbatim.
          2. Otherwise, calls the external TTS endpoint at ``ASTERISK_TTS_URL``
             and uses its returned ``media_uri`` (or its raw audio body,
             base64-encoded into a ``data:`` URI).
          3. Falls back to a clear error if neither is available.

        Args:
            channel_id: The ARI channel id (e.g. from ``ari.dial``).
            text: The text to speak. Ignored if ``media_uri`` is set.
            voice: TTS voice hint passed to the upstream service.
            media_uri: Skip TTS and play this URI directly. Must be an ARI-
                compatible scheme (``sound:``, ``http(s):``, ``data:``).
        """
        if not media_uri:
            synth = _synthesize_tts(text, voice)
            if not synth.get("ok"):
                return json.dumps({
                    "status": "error",
                    "message": "TTS synthesis failed; pass media_uri to skip TTS",
                    "error": synth.get("error"),
                })
            media_uri = synth["media_uri"]
        # /play (not /playback — /playback is a sub-resource of a Playback obj)
        res = _ari_request(
            "POST",
            f"/channels/{channel_id}/play",
            params={"media": media_uri},
        )
        return json.dumps(res)

    def ari_stop_playback(playback_id: str) -> str:
        """Stop an in-progress media playback.

        The complementary operation to ``ari.play_tts``: once a Playback
        object exists, ``DELETE /playbacks/{id}`` stops the stream. Exposed
        separately because the caller needs the playback id returned from
        the play call, not the channel id.
        """
        res = _ari_request("DELETE", f"/playbacks/{playback_id}")
        return json.dumps(res)

    def ari_transfer(channel_id: str, target: str) -> str:
        """Transfer a channel to another target (extension or context/exten)."""
        res = _ari_request(
            "POST",
            f"/channels/{channel_id}/redirect",
            params={"endpoint": target},
        )
        return json.dumps(res)

    def ari_dial(target: str, variables: Optional[Dict[str, str]] = None, app: str = "bridge", caller_id: Optional[str] = None) -> str:
        """Originate a call to a target.

        Args:
            target: ARI endpoint string (e.g. ``PJSIP/1001``, ``SIP/peer``).
            variables: Channel variables to set at originate time
                (e.g. ``{"CALLERID(name)": "Hermes"}``).
            app: Stasis application the new channel will route into. Defaults
                to ``bridge`` — override to your dialplan app if you have one.
            caller_id: Optional caller-id name to present.
        """
        params: Dict[str, Any] = {"endpoint": target, "app": app}
        if caller_id:
            params["callerId"] = caller_id
        if variables:
            for k, v in variables.items():
                params[f"variable_{k}"] = v
        res = _ari_request("POST", "/channels", params=params)
        return json.dumps(res)

    def ari_list_channels() -> str:
        """List currently active channels (debugging / dashboard use)."""
        res = _ari_request("GET", "/channels")
        return json.dumps(res)

    mcp.add_tool(ari_answer, name="ari.answer", description="Answer a ringing channel in Asterisk.")
    mcp.add_tool(ari_hangup, name="ari.hangup", description="Hang up a channel in Asterisk (reason_code=16).")
    mcp.add_tool(ari_play_tts, name="ari.play_tts", description="Play TTS on a channel. ARI has no native TTS — pass media_uri or set ASTERISK_TTS_URL.")
    mcp.add_tool(ari_stop_playback, name="ari.stop_playback", description="Stop a running Playback by its id.")
    mcp.add_tool(ari_transfer, name="ari.transfer", description="Redirect a channel to another endpoint (uses ARI /redirect).")
    mcp.add_tool(ari_dial, name="ari.dial", description="Originate a call to an ARI endpoint.")
    mcp.add_tool(ari_list_channels, name="ari.list_channels", description="List currently active ARI channels.")
