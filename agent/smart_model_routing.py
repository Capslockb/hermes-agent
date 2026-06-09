"""
Smart Model Routing — auto-route simple conversational turns to a cheap/fast
model and reserve the main model for complex work.

The classifier is intentionally cheap: it inspects the user message locally,
never makes an LLM call. This keeps the "fast tier" actually fast — adding
a classifier inference would defeat the purpose.

Design choices, with rationale:
  - Keyword regex over a hardcoded English list. Catches ~80% of obvious
    complexity signals (code blocks, file paths, error patterns, action verbs).
    Users can extend via `complex_keywords_extra` in config.
  - Char/word thresholds as a second gate. A 5,000-character paste of a
    stacktrace is not "simple" even if no keyword matches.
  - Tool-call detection. A user message starting with `/` or matching
    Hermes' slash-command prefix is never routed to a tool-stripped cheap tier.
  - Dry-run mode. When `dry_run: true`, route() returns the original model
    but logs what WOULD have happened. Lets you validate the classifier
    against real traffic before flipping the switch.
  - Conservative defaults. `enabled: false` by default. Even when enabled,
    `max_simple_chars: 240` and `max_simple_words: 40` keep the fast tier
    scoped to short conversational exchanges.

Integration point: ``gateway/run.py::_resolve_turn_agent_config`` calls
:classify_and_route` once per turn before building the AIAgent. The hook
mutates the route dict (model, runtime, request_overrides) and the per-turn
toolsets/skills.

What gets stripped on the cheap tier:
  - toolsets → empty list (no tool calls, no /command dispatch)
  - skill loading → disabled for the turn
  - memory recall → skipped (cheap tier won't use it)
  - context files (AGENTS.md etc.) → not prepended
  - reasoning effort → forced to "low" regardless of session default

What's preserved:
  - Conversation history (the user expects continuity)
  - Display/timestamp/user identity (the message still looks like a Hermes turn)
  - Streaming/callbacks (so the Discord reply renders normally)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.smart_model_routing")


# Default English keyword set. Conservative — biased toward *not* misrouting
# complex work. Users can extend with `complex_keywords_extra` in config.
DEFAULT_COMPLEX_KEYWORDS = frozenset({
    # Action verbs that imply work
    "debug", "fix", "implement", "refactor", "build", "deploy", "install",
    "configure", "setup", "migrate", "patch", "review", "audit", "analyze",
    "compare", "benchmark", "optimize", "profile", "trace", "investigate",
    "design", "architect", "plan", "estimate", "scope",
    # Object nouns that imply technical content
    "code", "script", "function", "class", "module", "package", "library",
    "api", "endpoint", "route", "handler", "schema", "query", "migration",
    "dockerfile", "compose", "yaml", "json", "regex", "cron",
    "stacktrace", "traceback", "error", "exception", "panic", "segfault",
    "log", "metrics", "telemetry", "trace",
    # Tooling / infra
    "docker", "kubernetes", "k8s", "terraform", "ansible", "helm",
    "nginx", "apache", "systemd", "iptables", "ufw",
    "ssh", "rsync", "scp", "tar", "curl", "wget",
    "git", "github", "gitlab", "pr", "merge", "rebase", "cherry-pick",
    "pytest", "unittest", "jest", "junit", "mocha",
    "redis", "postgres", "mysql", "sqlite", "mongodb", "qdrant",
    "ollama", "openai", "anthropic", "openrouter", "litellm",
    # Output verbs
    "write", "create", "generate", "produce", "scaffold", "draft",
    "summarize", "summarise", "explain", "describe", "document",
    "translate", "convert", "transform", "encode", "decode",
    "search", "find", "lookup", "query", "grep", "rg",
    "test", "verify", "validate", "check",
    "delete", "remove", "drop", "destroy", "wipe", "purge", "prune",
    "rename", "move", "copy", "upload", "download", "fetch",
    "list", "show", "print", "display", "render",
})

# Patterns that strongly imply complexity regardless of length.
_COMPLEX_PATTERNS = [
    re.compile(r"```"),                                  # code block
    re.compile(r"\$[A-Z_]{2,}|%[A-Z_]+%"),               # env vars / template tokens
    re.compile(r"^/[a-z][a-z0-9-]{1,40}\b", re.M),       # slash commands
    re.compile(r"^>>>|^<<<"),                            # tool delimiters
    re.compile(r"Traceback \(most recent call last\):"),  # python stacktrace
    re.compile(r"\b[A-Z][a-zA-Z]+Error:\s"),             # Error: pattern
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),  # IPv4
    re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}/"),  # URL-ish
    re.compile(r"^\s*def\s+\w+\(|^\s*class\s+\w+\(", re.M),  # python def/class
    re.compile(r"^\s*(?:import|from)\s+\w+", re.M),       # import statements
    re.compile(r"--[a-z][a-z0-9-]+|\b-[a-zA-Z]\b"),       # CLI flags
    re.compile(r"=~|=>|->|::|\$\(|`[^`]+`"),             # code-ish punctuation
    re.compile(r"\b(?:TODO|FIXME|XXX|HACK|NOTE)\b", re.I),  # code annotations
    re.compile(r"\b(?:hermes|docker|kubectl|npm|pip|uv|poetry)\s+[a-z]"),
    re.compile(r"[<{}\[\]]"),                            # JSON/bracket structure
]

# Orchestrator-eligible patterns — clarification/follow-up questions that
# the cheap model CAN handle well. These OVERRIDE keyword forcing.
# Order matters: more specific patterns first.
_ORCHESTRATOR_PATTERNS = [
    # Clarification requests
    re.compile(r"\b(?:what do you mean|what do u mean)\b", re.I),
    re.compile(r"\b(?:can you clarify|could you clarify|clarify that)\b", re.I),
    re.compile(r"\b(?:in simple terms|in plain english|explain like)\b", re.I),
    re.compile(r"\b(?:rephrase|put differently|say it again)\b", re.I),
    re.compile(r"\b(?:what does .+ mean|what is .+\?)\b", re.I),  # "what does X mean?"
    # Follow-up / continuation
    re.compile(r"^\s*(?:and then|also|additionally|one more thing|also,?)\b", re.I),
    re.compile(r"^\s*(?:so |then |btw |also )\b", re.I),
    # Low-risk questions (cheap tier excels at these)
    re.compile(r"^\s*(?:how (?:do|can|to|would) .+\?)\b", re.I),   # "how do I...?"
    re.compile(r"^\s*(?:what (?:is|are|was|would) .+\?)\b", re.I), # "what is...?"
    re.compile(r"^\s*(?:why (?:is|are|does|did|would) .+\?)", re.I),
    re.compile(r"^\s*(?:can you|could you|would you) .+\?", re.I),
    re.compile(r"^\s*(?:give me an?|show me an?|example of)\b", re.I),
    re.compile(r"^\s*(?:summarize|summarise|tldr)\b", re.I),
    re.compile(r"^\s*(?:list|show|describe).+\b", re.I),           # "list the files", "show me..."
]

# Greetings/thanks/short acknowledgements — even if a keyword accidentally
# matches, these force-route to the main model. Conservative.
_SOCIAL_PHRASES = frozenset({
    "hi", "hey", "hello", "yo", "sup", "morning", "evening", "night",
    "thanks", "thank you", "thx", "ty", "cheers", "ta", "🙏", "lol",
    "ok", "okay", "k", "kk", "yep", "yup", "nope", "nah", "yes", "no",
    "lmao", "haha", "heh", "😂", "👍", "👌", "❤️", "🔥", "🎉",
    "bye", "cya", "gn", "ttyl", "ciao", "👋",
    "?", "??", "???",
})


@dataclass
class RoutingConfig:
    """Effective routing settings after defaults + user overrides."""
    enabled: bool = False
    dry_run: bool = True
    max_simple_chars: int = 240
    max_simple_words: int = 40
    complex_keywords: frozenset = field(default_factory=lambda: DEFAULT_COMPLEX_KEYWORDS)
    complex_keywords_extra: tuple = ()
    complex_patterns: tuple = field(default_factory=lambda: tuple(_COMPLEX_PATTERNS))
    social_phrases: frozenset = field(default_factory=lambda: _SOCIAL_PHRASES)

    # Cheap-tier model slot — shape mirrors the top-level `model:` block
    cheap_model: Dict[str, Any] = field(default_factory=dict)
    # Strip these toolsets when cheap tier handles the turn
    strip_toolsets: tuple = ("terminal", "file", "browser", "code_execution", "delegation", "cronjob", "messaging", "kanban", "image_gen", "video", "tts", "homeassistant", "rl", "moa", "debugging")
    # Cheap tier reasoning effort override
    cheap_reasoning_effort: str = "low"

    # Platform gate: cheap routing ONLY applies to these platforms.
    # Default: hermes channels (discord, telegram, whatsapp, terminal).
    # NOT web-ui — the user explicitly said "dont route cheap to web ui
    # only in hermes". The list is matched case-insensitively against
    # source.platform.value; empty list = no platforms (effectively disables).
    # ``None`` (the default) means "all hermes channels" — covers any platform
    # the gateway is connected to. To restrict further, set explicitly.
    allowed_platforms: frozenset = field(default_factory=lambda: frozenset({
        "discord", "telegram", "whatsapp", "terminal", "cli",
    }))


def _normalize_config(raw: Optional[Dict[str, Any]]) -> RoutingConfig:
    """Merge user config over defaults. Tolerates missing keys, bad types."""
    raw = raw or {}
    if not isinstance(raw, dict):
        logger.warning("smart_model_routing is not a dict (got %s); using defaults", type(raw).__name__)
        raw = {}

    cfg = RoutingConfig(
        enabled=bool(raw.get("enabled", False)),
        dry_run=bool(raw.get("dry_run", True)),  # dry-run by default — opt in to live
        max_simple_chars=int(raw.get("max_simple_chars", 240)),
        max_simple_words=int(raw.get("max_simple_words", 40)),
        cheap_model=dict(raw.get("cheap_model") or {}),
        strip_toolsets=tuple(raw.get("strip_toolsets") or RoutingConfig().strip_toolsets),
        cheap_reasoning_effort=str(raw.get("cheap_reasoning_effort", "low")),
    )

    # Optional explicit override of allowed_platforms
    plat_raw = raw.get("allowed_platforms")
    if plat_raw is None:
        pass  # keep the default frozenset
    elif isinstance(plat_raw, (list, tuple, set, frozenset)):
        cfg.allowed_platforms = frozenset(str(p).lower().strip() for p in plat_raw if p)
        if not cfg.allowed_platforms:
            logger.warning("smart_model_routing.allowed_platforms is empty — cheap tier effectively disabled")
    else:
        logger.warning("allowed_platforms must be a list; using defaults")

    # Keyword merge: extra list extends the default set.
    extras = raw.get("complex_keywords_extra") or []
    if isinstance(extras, (list, tuple, set)):
        cfg.complex_keywords_extra = tuple(str(x) for x in extras if x)
        cfg.complex_keywords = DEFAULT_COMPLEX_KEYWORDS | frozenset(cfg.complex_keywords_extra)
    else:
        logger.warning("complex_keywords_extra must be a list; ignoring")

    # Optional full override of the keyword set
    override = raw.get("complex_keywords")
    if isinstance(override, (list, tuple, set)) and override:
        cfg.complex_keywords = frozenset(str(x) for x in override)
        cfg.complex_keywords_extra = tuple(sorted(cfg.complex_keywords - DEFAULT_COMPLEX_KEYWORDS))

    return cfg


# Pre-built word boundary for the keyword regex (avoids recompiling per call)
def _build_keyword_regex(keywords: frozenset) -> re.Pattern:
    if not keywords:
        return re.compile(r"(?!)")  # never matches
    escaped = "|".join(re.escape(k) for k in sorted(keywords))
    return re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE)


@dataclass
class RoutingDecision:
    """Result of one classification call. Carry through to logging + telemetry."""
    tier: str                       # "main" | "cheap" | "social"
    reason: str                     # why this tier
    matched_keyword: Optional[str] = None
    matched_pattern: Optional[str] = None
    char_count: int = 0
    word_count: int = 0
    eligible: bool = False          # would route to cheap if enabled
    dry_run: bool = False

    def to_log_dict(self) -> dict:
        return {
            "tier": self.tier,
            "reason": self.reason,
            "matched_keyword": self.matched_keyword,
            "matched_pattern": self.matched_pattern,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "eligible": self.eligible,
            "dry_run": self.dry_run,
        }


def classify(
    user_message: str,
    cfg: RoutingConfig,
    platform: Optional[str] = None,
) -> RoutingDecision:
    """Inspect a user message and return a routing decision.

    Does NOT decide whether to apply — that's :route_message`'s job, which
    respects ``enabled`` and ``dry_run``. This function is pure: same input
    always gives same decision.

    ``platform`` is checked against ``cfg.allowed_platforms``. If the source
    platform is not in the allowed set, the classifier returns tier="main"
    with reason "platform-not-allowed" — the cheap tier is not applied even
    if the message is otherwise simple. Web UI, REST API, and any other
    non-hermes surface is excluded by default.
    """
    # Platform gate — runs FIRST so platform-mismatched messages don't even
    # burn the rest of the classification logic.
    if platform is not None:
        plat_norm = str(platform).lower().strip()
        if cfg.allowed_platforms and plat_norm not in cfg.allowed_platforms:
            return RoutingDecision(
                tier="main", reason=f"platform-not-allowed:{plat_norm}",
                char_count=0, word_count=0,
            )

    if not isinstance(user_message, str):
        return RoutingDecision(tier="main", reason="non-string input")

    text = user_message.strip()
    char_count = len(text)
    word_count = len(text.split())

    # Empty / whitespace-only — main model (nothing to "fast-reply" to)
    if char_count == 0:
        return RoutingDecision(tier="main", reason="empty", char_count=0, word_count=0)

    # Length gates first (cheapest check, catches most long-paste cases)
    if char_count > cfg.max_simple_chars:
        return RoutingDecision(
            tier="main", reason=f"len>{cfg.max_simple_chars}ch",
            char_count=char_count, word_count=word_count,
        )
    if word_count > cfg.max_simple_words:
        return RoutingDecision(
            tier="main", reason=f"len>{cfg.max_simple_words}w",
            char_count=char_count, word_count=word_count,
        )

    # Pattern match (catches code blocks, URLs, stacktraces, etc.)
    for pat in _COMPLEX_PATTERNS:
        m = pat.search(text)
        if m:
            return RoutingDecision(
                tier="main", reason=f"pattern:{pat.pattern[:32]}",
                matched_pattern=pat.pattern[:64],
                char_count=char_count, word_count=word_count,
            )

    # Orchestrator gate — clarification/follow-up questions the cheap tier
    # handles well. Runs BEFORE keyword check so it OVERRIDES broad keywords
    # like "explain", "describe", "list", "show", "search" that appear in
    # normal questioning rounds.
    for pat in _ORCHESTRATOR_PATTERNS:
        m = pat.search(text)
        if m:
            return RoutingDecision(
                tier="cheap", reason=f"orchestrator:{pat.pattern[:32]}",
                matched_pattern=pat.pattern[:64],
                char_count=char_count, word_count=word_count,
                eligible=True,
            )

    # Keyword match
    kw_re = _build_keyword_regex(cfg.complex_keywords)
    m = kw_re.search(text)
    if m:
        return RoutingDecision(
            tier="main", reason=f"keyword:{m.group(0).lower()}",
            matched_keyword=m.group(0).lower(),
            char_count=char_count, word_count=word_count,
        )

    # Social / pure-acknowledgement gate
    normalized = text.lower().strip(" .,!?:;")
    if normalized in cfg.social_phrases:
        return RoutingDecision(
            tier="social", reason=f"social:{normalized[:16]}",
            char_count=char_count, word_count=word_count,
            eligible=True,
        )

    # No complexity signals, no social phrase — eligible for cheap tier
    return RoutingDecision(
        tier="cheap", reason="no-complexity-signals",
        char_count=char_count, word_count=word_count, eligible=True,
    )


def build_cheap_route(
    base_route: Dict[str, Any],
    decision: RoutingDecision,
    cfg: RoutingConfig,
) -> Dict[str, Any]:
    """Mutate a per-turn route dict to use the cheap tier.

    ``base_route`` is the dict returned by ``_resolve_turn_agent_config`` —
    shape is ``{"model": str, "runtime": dict, "signature": tuple, "request_overrides": dict}``.

    Returns the same dict (mutated in place is fine, but we also return it
    for chaining).
    """
    cheap = cfg.cheap_model or {}
    if not cheap:
        logger.warning("smart_model_routing eligible but cheap_model not configured; skipping")
        return base_route

    new_model = cheap.get("model") or cheap.get("default")
    if not new_model:
        logger.warning("cheap_model block missing 'model'/'default' key; skipping")
        return base_route

    base_route["model"] = new_model

    # Provider/base_url/api_key — if cheap_model block provides them, use those;
    # otherwise inherit from the runtime so the call works against a custom
    # OpenAI-compatible endpoint (like your ollama-cloud setup).
    rt = base_route.setdefault("runtime", {})
    if cheap.get("provider"):
        rt["provider"] = cheap["provider"]
    if cheap.get("base_url") is not None:
        rt["base_url"] = cheap["base_url"]
    if cheap.get("api_key"):
        rt["api_key"] = cheap["api_key"]
    if cheap.get("api_mode"):
        rt["api_mode"] = cheap["api_mode"]

    # Reasoning effort override (cheap models often don't support 'medium'+)
    base_route.setdefault("request_overrides", {})["reasoning_effort"] = cfg.cheap_reasoning_effort

    # Update signature so the gateway's de-dup cache (if any) sees the new model
    sig = base_route.get("signature")
    if isinstance(sig, tuple) and len(sig) >= 6:
        base_route["signature"] = (
            new_model, rt.get("provider"), rt.get("base_url"),
            rt.get("api_mode"), rt.get("command"), rt.get("args") or (),
        )

    # Mark the route with metadata so the agent can apply turn-time context strip
    base_route.setdefault("routing_metadata", {})["smart_routing"] = {
        "tier": decision.tier,
        "reason": decision.reason,
        "strip_toolsets": list(cfg.strip_toolsets),
    }

    return base_route


def route_message(
    user_message: str,
    base_route: Dict[str, Any],
    raw_config: Optional[Dict[str, Any]],
    platform: Optional[str] = None,
) -> Tuple[Dict[str, Any], RoutingDecision]:
    """Top-level entry point. Returns (route, decision).

    Honors ``enabled`` and ``dry_run``:
      - disabled → return (base_route, decision) unchanged
      - dry_run → return (base_route, decision) so the classifier still logs
        what WOULD have happened, but no model swap is applied
      - enabled + eligible → apply the cheap route

    ``platform`` is passed to ``classify`` for the allowed_platforms gate.
    """
    cfg = _normalize_config(raw_config)

    decision = classify(user_message, cfg, platform=platform)
    decision.dry_run = cfg.dry_run

    if not cfg.enabled:
        decision.reason = "disabled:" + decision.reason
        return base_route, decision

    if not decision.eligible:
        # Not eligible (already routed to main or marked as complex).
        # Dry-run still logs the decision.
        return base_route, decision

    if cfg.dry_run:
        decision.reason = "dry-run:" + decision.reason
        return base_route, decision

    cheap_route = build_cheap_route(base_route, decision, cfg)
    return cheap_route, decision


def strip_turn_toolsets(
    enabled: List[str],
    cfg: RoutingConfig,
) -> List[str]:
    """Return the toolsets list with cheap-tier-disabled entries removed.

    Called by the gateway after route_message() has decided the cheap tier
    applies. The cheap tier can ONLY chat — no terminal, no files, no browser,
    no delegation, no cron, no messaging. This is what makes it "lightweight"
    in practice, not just the model swap.
    """
    if not enabled:
        return enabled
    strip = set(cfg.strip_toolsets)
    return [t for t in enabled if t not in strip]


# ---------------------------------------------------------------------------
# Per-turn telemetry log
# ---------------------------------------------------------------------------
# The gateway calls ``log_decision`` after every turn with the routing
# decision, the model that actually ran, prompt-size info, and timing. We
# append one JSON line per turn to ``~/.hermes/logs/smart_routing.jsonl``.
# The smart-routing-cost plugin reads this log to power its analytics
# tools (cost_breakdown, recent_decisions, tier_comparison, savings_report).
#
# This is intentionally lightweight and always-on — no Langfuse dependency,
# no async, no extra DB. A 10k-turn day is ~1.5MB of JSONL.

DECISIONS_LOG_PATH = os.environ.get(
    "HERMES_SMART_ROUTING_LOG",
    str(Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "logs" / "smart_routing.jsonl"),
)
_decisions_log_lock = threading.Lock()


def log_decision(
    *,
    session_id: Optional[str] = None,
    platform: Optional[str] = None,
    user_id: Optional[str] = None,
    user_message_preview: str = "",
    char_count: int = 0,
    word_count: int = 0,
    decision: "RoutingDecision",
    dry_run: bool,
    main_model: str,
    cheap_model: Optional[str] = None,
    cheap_provider: Optional[str] = None,
    enabled_toolsets_before: Optional[List[str]] = None,
    enabled_toolsets_after: Optional[List[str]] = None,
    main_prompt_chars: Optional[int] = None,
    cheap_prompt_chars: Optional[int] = None,
    latency_ms: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a single JSONL record describing one turn's routing decision.

    Safe to call from any thread; the file write is lock-guarded and
    errors are swallowed (telemetry must never break a turn).
    """
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "platform": platform,
            "user_id": user_id,
            "user_message_preview": user_message_preview[:120],
            "char_count": char_count,
            "word_count": word_count,
            "tier": decision.tier,
            "reason": decision.reason,
            "matched_keyword": decision.matched_keyword,
            "matched_pattern": decision.matched_pattern,
            "eligible": decision.eligible,
            "dry_run": dry_run,
            "main_model": main_model,
            "cheap_model": cheap_model,
            "cheap_provider": cheap_provider,
            "applied": (
                decision.tier in ("cheap", "social")
                and decision.eligible
                and not dry_run
            ),
            "enabled_toolsets_before": enabled_toolsets_before or [],
            "enabled_toolsets_after": enabled_toolsets_after or [],
            "main_prompt_chars": main_prompt_chars,
            "cheap_prompt_chars": cheap_prompt_chars,
            "prompt_chars_saved": (
                (main_prompt_chars or 0) - (cheap_prompt_chars or 0)
                if (main_prompt_chars is not None and cheap_prompt_chars is not None
                    and decision.tier in ("cheap", "social") and decision.eligible)
                else None
            ),
            "latency_ms": latency_ms,
        }
        if extra:
            record["extra"] = extra

        path = Path(DECISIONS_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _decisions_log_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        # Never let telemetry break a turn.
        logger.debug("smart_routing telemetry log failed: %s", e)


def tail_recent_decisions(n: int = 50) -> List[Dict[str, Any]]:
    """Read the last ``n`` decision records (most recent first).

    Used by the smart-routing-cost plugin's ``recent_decisions`` tool.
    """
    path = Path(DECISIONS_LOG_PATH)
    if not path.exists():
        return []
    try:
        with _decisions_log_lock:
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        records = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        records.reverse()
        return records
    except Exception as e:
        logger.debug("smart_routing tail_recent_decisions failed: %s", e)
        return []


def aggregate_decisions(
    *,
    since: Optional[str] = None,
    platform: Optional[str] = None,
) -> Dict[str, Any]:
    """Walk the full log and aggregate per-tier stats.

    Returns a dict shaped like::

        {
            "total_turns": int,
            "by_tier": {"cheap": int, "social": int, "main": int},
            "by_model": {"minimax-m3": int, "gemma4:31b": int, ...},
            "by_platform": {"discord": int, ...},
            "applied_count": int,            # cheap tier actually swapped
            "dry_run_count": int,
            "complex_reasons": {"keyword:implement": int, ...},
            "total_prompt_chars_saved": int,
            "avg_prompt_chars_saved": float,
            "period": {"from": iso, "to": iso},
        }
    """
    path = Path(DECISIONS_LOG_PATH)
    if not path.exists():
        return {"total_turns": 0, "by_tier": {}, "by_model": {},
                "by_platform": {}, "applied_count": 0, "dry_run_count": 0,
                "complex_reasons": {}, "total_prompt_chars_saved": 0,
                "avg_prompt_chars_saved": 0.0, "period": {"from": since, "to": None}}

    from_ts = None
    if since:
        try:
            from_ts = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            from_ts = None

    by_tier: Dict[str, int] = {}
    by_model: Dict[str, int] = {}
    by_platform: Dict[str, int] = {}
    complex_reasons: Dict[str, int] = {}
    applied_count = 0
    dry_run_count = 0
    total_saved = 0
    saved_count = 0
    total = 0
    earliest_ts = None
    latest_ts = None

    try:
        with _decisions_log_lock:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts_str = rec.get("ts")
                    ts = None
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except ValueError:
                            pass
                    if from_ts and ts and ts < from_ts:
                        continue
                    if platform and rec.get("platform") and rec.get("platform") != platform:
                        continue

                    total += 1
                    tier = rec.get("tier", "unknown")
                    by_tier[tier] = by_tier.get(tier, 0) + 1
                    # The model that ACTUALLY ran (cheap_model if applied,
                    # else main_model). Counts tell you which model really
                    # served the user.
                    actual_model = rec.get("cheap_model") if rec.get("applied") else rec.get("main_model")
                    if actual_model:
                        by_model[actual_model] = by_model.get(actual_model, 0) + 1
                    plat = rec.get("platform") or "unknown"
                    by_platform[plat] = by_platform.get(plat, 0) + 1
                    if rec.get("applied"):
                        applied_count += 1
                    if rec.get("dry_run"):
                        dry_run_count += 1
                    if tier == "main" and rec.get("reason"):
                        # Bucket complex reasons for the "what's making
                        # messages fall back to main" view
                        key = rec["reason"].split(":", 1)[0]
                        complex_reasons[key] = complex_reasons.get(key, 0) + 1
                    saved = rec.get("prompt_chars_saved")
                    if isinstance(saved, (int, float)) and saved > 0:
                        total_saved += int(saved)
                        saved_count += 1
                    if ts:
                        if earliest_ts is None or ts < earliest_ts:
                            earliest_ts = ts
                        if latest_ts is None or ts > latest_ts:
                            latest_ts = ts
    except Exception as e:
        logger.debug("smart_routing aggregate_decisions failed: %s", e)

    return {
        "total_turns": total,
        "by_tier": dict(sorted(by_tier.items(), key=lambda x: -x[1])),
        "by_model": dict(sorted(by_model.items(), key=lambda x: -x[1])),
        "by_platform": dict(sorted(by_platform.items(), key=lambda x: -x[1])),
        "applied_count": applied_count,
        "dry_run_count": dry_run_count,
        "complex_reasons": dict(sorted(complex_reasons.items(), key=lambda x: -x[1])),
        "total_prompt_chars_saved": total_saved,
        "avg_prompt_chars_saved": (total_saved / saved_count) if saved_count else 0.0,
        "period": {
            "from": earliest_ts.isoformat() if earliest_ts else since,
            "to": latest_ts.isoformat() if latest_ts else None,
        },
    }


__all__ = [
    "RoutingConfig",
    "RoutingDecision",
    "classify",
    "build_cheap_route",
    "route_message",
    "strip_turn_toolsets",
    "log_decision",
    "tail_recent_decisions",
    "aggregate_decisions",
    "DECISIONS_LOG_PATH",
    "DEFAULT_COMPLEX_KEYWORDS",
]
