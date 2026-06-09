from __future__ import annotations

import os
import json
import logging
import httpx
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("hermes.mcp_ari")

# --- Configuration ---
ARI_URL = os.environ.get("ASTERISK_ARI_URL", "http://127.0.0.1:8088")
ARI_USER = os.environ.get("ASTERISK_ARI_USER", "admin")
ARI_PASS = os.environ.get("ASTERISK_ARI_PASS", "admin")

def _ari_request(method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Internal helper to call the Asterisk ARI REST API."""
    url = f"{ARI_URL}/ari{endpoint}"
    try:
        with httpx.Client(auth=(ARI_USER, ARI_PASS), timeout=5.0) as client:
            response = client.request(method, url, params=params, json=data)
            response.raise_for_status()
            return {
                "status": "success",
                "message": f"ARI {method} {endpoint} successful",
                "data": response.json() if response.content else {}
            }
    except httpx.HTTPStatusError as e:
        return {
            "status": "error",
            "message": f"ARI API returned {e.response.status_code}",
            "error": e.response.text
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"ARI request failed: {str(e)}"
        }

def register_ari_tools(mcp: FastMCP):
    """Register ARI-related tools to the provided FastMCP server."""

    def ari_answer(channel_id: str) -> str:
        """Answer a ringing channel in Asterisk."""
        res = _ari_request("POST", f"/channels/{channel_id}/answer")
        return json.dumps(res)

    def ari_hangup(channel_id: str) -> str:
        """Hang up a channel in Asterisk."""
        res = _ari_request("POST", f"/channels/{channel_id}/hangup")
        return json.dumps(res)

    def ari_play_tts(channel_id: str, text: str, voice: str = "alice") -> str:
        """Play text-to-speech on a channel."""
        res = _ari_request("POST", f"/channels/{channel_id}/playback", data={"text": text, "voice": voice})
        return json.dumps(res)

    def ari_transfer(channel_id: str, target: str) -> str:
        """Transfer a channel to another target."""
        res = _ari_request("POST", f"/channels/{channel_id}/transfer", data={"target": target})
        return json.dumps(res)

    def ari_dial(target: str, variables: Optional[Dict[str, str]] = None) -> str:
        """Originate a call to a target."""
        res = _ari_request("POST", f"/channels", data={"endpoint": target, "variables": variables})
        return json.dumps(res)

    mcp.add_tool(ari_answer, name="ari.answer", description="Answer a ringing channel in Asterisk.")
    mcp.add_tool(ari_hangup, name="ari.hangup", description="Hang up a channel in Asterisk.")
    mcp.add_tool(ari_play_tts, name="ari.play_tts", description="Play text-to-speech on a channel.")
    mcp.add_tool(ari_transfer, name="ari.transfer", description="Transfer a channel to another target.")
    mcp.add_tool(ari_dial, name="ari.dial", description="Originate a call to a target.")
