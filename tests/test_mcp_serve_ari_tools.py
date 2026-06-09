"""
Tests for ``mcp_serve_ari_tools``.

Covers the ARI bugs the original PR draft had:

  1. ``ari.play_tts`` must post to ``/channels/{id}/play`` with a
     ``media`` query parameter, NOT ``/channels/{id}/playback`` with a
     JSON body.  ARI has no native TTS — ``/play`` is a streaming-media
     operation, ``/playback`` is a sub-resource of an existing Playback
     object.  The ARI docs are unambiguous: "Almost all media is played
     to a channel using the POST /channels/{channel_id}/play operation."

  2. ``ari.transfer`` must post to ``/channels/{id}/redirect``, not
     ``/channels/{id}/transfer`` (which does not exist in ARI).

  3. ``ari.dial`` must put operation parameters (``app``, ``endpoint``,
     ``callerId``) on the **query string**, and flatten
     ``variables={k: v}`` into per-variable ``variable_<k>=v`` query
     parameters.  ARI is form/query-encoded, not JSON-bodied, for these
     fields.  Sending a JSON body to ``POST /channels`` is rejected.

  4. ``ari.stop_playback`` must use ``DELETE /playbacks/{id}``.

  5. ``ari.play_tts`` with an explicit ``media_uri`` must skip the
     TTS round-trip and post directly to ``/play`` with the URI.

  6. When TTS isn't configured and no ``media_uri`` is passed,
     ``ari.play_tts`` must return a clear error rather than silently
     404-ing on a non-existent ``/playback`` endpoint.

  7. The optional external TTS endpoint may return audio in two shapes:
     a ``Content-Type: audio/*`` response (we base64-wrap into a
     ``data:`` URI) or an ``application/json`` body with a
     ``media_uri`` field.  Both must work.

The tools are closures registered on the FastMCP server, so the test
fixture builds a real ``FastMCP`` instance, calls
``register_ari_tools()`` on it, and looks up the resulting tool
functions by their registered name (``ari.answer``, ``ari.dial``,
etc.).  This pins the public surface that downstream MCP clients see.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

import mcp_serve_ari_tools as ari_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_response(json_body=None):
    resp = MagicMock()
    resp.status_code = 200
    resp.content = b"" if json_body is None else json.dumps(json_body).encode()
    resp.json.return_value = json_body or {}
    resp.raise_for_status = MagicMock()
    return resp


def _err_response(status, body):
    resp = MagicMock()
    resp.status_code = status
    resp.text = body
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=resp,
        )
    )
    return resp


@pytest.fixture
def tools():
    """Build a FastMCP server, register the ARI tools, return the
    public tool functions keyed by their ``ari.*`` name.
    """
    mcp = FastMCP("test")
    ari_mod.register_ari_tools(mcp)
    return {name: tool.fn for name, tool in mcp._tool_manager._tools.items()}


@pytest.fixture
def mock_ari_post(monkeypatch):
    """Capture every ``httpx.Client.request`` invocation.  A per-test
    router can override this with a more elaborate mock that also
    handles the TTS endpoint.
    """
    calls = []

    def fake_request(self, method, url, params=None, **kwargs):
        calls.append({
            "method": method,
            "url": url,
            "params": params or {},
            "kwargs": kwargs,
        })
        return _ok_response({"id": "playback-abc123"})

    monkeypatch.setattr("httpx.Client.request", fake_request)
    return calls


# ---------------------------------------------------------------------------
# 1. play_tts: /play (not /playback) with media on the query string
# ---------------------------------------------------------------------------

class TestPlayTTS:
    def test_calls_play_not_playback(self, tools, monkeypatch):
        """The headline bug from the PR review: play_tts was hitting
        the wrong endpoint.  Pinned here against regression.
        """
        monkeypatch.setenv("ASTERISK_TTS_URL", "http://tts.local/synth")
        calls = []

        def router(self, method, url, params=None, **kwargs):
            calls.append({"method": method, "url": url, "params": params or {}, "kwargs": kwargs})
            if "tts.local" in url:
                r = MagicMock()
                r.status_code = 200
                r.headers = {"content-type": "application/json"}
                r.content = b'{"media_uri": "http://tts.local/out/abc.wav"}'
                r.json.return_value = {"media_uri": "http://tts.local/out/abc.wav"}
                r.raise_for_status = MagicMock()
                return r
            return _ok_response({"id": "pb1"})

        monkeypatch.setattr("httpx.Client.request", router)
        tools["ari.play_tts"](channel_id="ch-1", text="hello world")

        ari_calls = [c for c in calls if c["url"].endswith("/play")]
        assert len(ari_calls) == 1, f"Expected 1 /play call, got: {calls}"
        call = ari_calls[0]
        assert call["method"] == "POST"
        # The /playback sub-resource is the wrong endpoint.
        assert "/playback" not in call["url"], (
            "REGRESSION: play_tts is hitting /playback. ARI has no such "
            "channel operation; /playback is a sub-resource of a Playback. "
            f"URL was: {call['url']}"
        )
        # Media must be a query param, not in a JSON body.
        assert "media" in call["params"]
        assert call["params"]["media"] == "http://tts.local/out/abc.wav"

    def test_explicit_media_uri_skips_tts(self, tools, mock_ari_post, monkeypatch):
        """If the operator already has a sound: or http: URI, the tool
        must NOT make a TTS call.
        """
        monkeypatch.setenv("ASTERISK_TTS_URL", "http://should-not-be-hit.example/synth")
        tools["ari.play_tts"](channel_id="ch-1", text="ignored", media_uri="sound:custom")
        assert len(mock_ari_post) == 1
        assert mock_ari_post[0]["params"]["media"] == "sound:custom"

    def test_no_tts_no_uri_returns_error(self, tools, mock_ari_post, monkeypatch):
        """Without TTS_URL or media_uri we must return an error dict and
        NOT hit ARI at all (the original draft would 404 on /playback).
        """
        monkeypatch.delenv("ASTERISK_TTS_URL", raising=False)
        out = tools["ari.play_tts"](channel_id="ch-1", text="hello world")
        assert mock_ari_post == [], (
            f"REGRESSION: ari.play_tts made ARI calls without a media URI. "
            f"Calls: {mock_ari_post}"
        )
        body = json.loads(out)
        assert body["status"] == "error"

    def test_tts_audio_response_base64_wrapped(self, tools, monkeypatch):
        monkeypatch.setenv("ASTERISK_TTS_URL", "http://tts.local/synth")
        calls = []

        def router(self, method, url, params=None, **kwargs):
            calls.append({"method": method, "url": url, "params": params or {}, "kwargs": kwargs})
            if "tts.local" in url:
                r = MagicMock()
                r.status_code = 200
                r.headers = {"content-type": "audio/wav"}
                r.content = b"RIFFFAKE"
                r.raise_for_status = MagicMock()
                return r
            return _ok_response({"id": "pb1"})

        monkeypatch.setattr("httpx.Client.request", router)
        tools["ari.play_tts"](channel_id="ch-1", text="hi")
        ari_calls = [c for c in calls if c["url"].endswith("/play")]
        assert len(ari_calls) == 1
        # Raw audio body is base64-wrapped into a data: URI ARI can play.
        assert ari_calls[0]["params"]["media"].startswith("data:audio/wav;base64,")

    def test_tts_json_response_uri_passthrough(self, tools, monkeypatch):
        monkeypatch.setenv("ASTERISK_TTS_URL", "http://tts.local/synth")
        calls = []

        def router(self, method, url, params=None, **kwargs):
            calls.append({"method": method, "url": url, "params": params or {}, "kwargs": kwargs})
            if "tts.local" in url:
                r = MagicMock()
                r.status_code = 200
                r.headers = {"content-type": "application/json"}
                r.content = b'{"media_uri": "http://tts.local/out/x.wav"}'
                r.json.return_value = {"media_uri": "http://tts.local/out/x.wav"}
                r.raise_for_status = MagicMock()
                return r
            return _ok_response({"id": "pb1"})

        monkeypatch.setattr("httpx.Client.request", router)
        tools["ari.play_tts"](channel_id="ch-1", text="hi")
        ari_calls = [c for c in calls if c["url"].endswith("/play")]
        assert ari_calls[0]["params"]["media"] == "http://tts.local/out/x.wav"


# ---------------------------------------------------------------------------
# 2. transfer must use /redirect, not /transfer
# ---------------------------------------------------------------------------

class TestTransfer:
    def test_calls_redirect_not_transfer(self, tools, mock_ari_post):
        tools["ari.transfer"](channel_id="ch-1", target="PJSIP/1001")
        assert len(mock_ari_post) == 1
        call = mock_ari_post[0]
        assert call["url"].endswith("/ari/channels/ch-1/redirect"), (
            "ARI redirect must POST to /channels/{id}/redirect. "
            f"/channels/{{id}}/transfer does not exist. Got: {call['url']}"
        )
        assert call["params"]["endpoint"] == "PJSIP/1001"


# ---------------------------------------------------------------------------
# 3. dial: app, endpoint, callerId on query; variables as variable_* prefix
# ---------------------------------------------------------------------------

class TestDial:
    def test_app_on_query(self, tools, mock_ari_post):
        tools["ari.dial"](target="PJSIP/1001", app="stasis-bridge")
        assert mock_ari_post[0]["params"]["app"] == "stasis-bridge"
        assert mock_ari_post[0]["params"]["endpoint"] == "PJSIP/1001"
        assert mock_ari_post[0]["url"].endswith("/ari/channels")

    def test_variables_become_variable_prefix(self, tools, mock_ari_post):
        tools["ari.dial"](
            target="PJSIP/1001",
            variables={"CALLERID(name)": "Hermes", "CHANNEL(language)": "en"},
        )
        p = mock_ari_post[0]["params"]
        # ARI's convention: each variable becomes its own query param.
        assert p["variable_CALLERID(name)"] == "Hermes"
        assert p["variable_CHANNEL(language)"] == "en"
        # The raw key must NOT appear in the params (the original draft
        # tried to send variables as a JSON body, which ARI rejects).
        assert "CALLERID(name)" not in p
        assert "variables" not in p
        # Make sure we didn't smuggle a JSON body in either.
        assert "json" not in mock_ari_post[0]["kwargs"]

    def test_caller_id_passed_through(self, tools, mock_ari_post):
        tools["ari.dial"](target="PJSIP/1001", caller_id="+31xxxxxxxxx")
        assert mock_ari_post[0]["params"]["callerId"] == "+31xxxxxxxxx"


# ---------------------------------------------------------------------------
# 4. stop_playback: DELETE /playbacks/{id}
# ---------------------------------------------------------------------------

class TestStopPlayback:
    def test_delete_playbacks(self, tools, mock_ari_post):
        tools["ari.stop_playback"](playback_id="pb-abc")
        assert len(mock_ari_post) == 1
        call = mock_ari_post[0]
        assert call["method"] == "DELETE"
        assert call["url"].endswith("/ari/playbacks/pb-abc")


# ---------------------------------------------------------------------------
# 5. answer / hangup
# ---------------------------------------------------------------------------

class TestAnswer:
    def test_answer_url(self, tools, mock_ari_post):
        tools["ari.answer"](channel_id="ch-1")
        assert mock_ari_post[0]["method"] == "POST"
        assert mock_ari_post[0]["url"].endswith("/ari/channels/ch-1/answer")


class TestHangup:
    def test_hangup_with_reason_code(self, tools, mock_ari_post):
        tools["ari.hangup"](channel_id="ch-1")
        call = mock_ari_post[0]
        assert call["method"] == "POST"
        assert call["url"].endswith("/ari/channels/ch-1/hangup")
        assert call["params"]["reason_code"] == 16


# ---------------------------------------------------------------------------
# 6. list_channels
# ---------------------------------------------------------------------------

class TestListChannels:
    def test_get_channels(self, tools, mock_ari_post):
        tools["ari.list_channels"]()
        assert mock_ari_post[0]["method"] == "GET"
        assert mock_ari_post[0]["url"].endswith("/ari/channels")


# ---------------------------------------------------------------------------
# 7. Error handling: ARI returns 404 → tool reports error, not crash
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_ari_404_returns_error_dict(self, tools, monkeypatch):
        def boom(self, method, url, params=None, **kwargs):
            return _err_response(404, "Not Found")
        monkeypatch.setattr("httpx.Client.request", boom)
        out = tools["ari.play_tts"](channel_id="ch-1", text="hi", media_uri="sound:foo")
        body = json.loads(out)
        assert body["status"] == "error"
        assert "404" in body["message"]
