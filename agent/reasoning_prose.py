"""Reasoning-prose stripping — public surface for the regex primitives.

Some chat-tuned reasoning models (notably minimax-m3 and the kimi-k2.5
family) emit their chain-of-thought as natural-language sentences
directly in the visible ``content`` field, instead of using XML-style
``<think>…</think>`` tags.  We strip that prose in two places:

* **Final-response chokepoint** — ``agent.agent_runtime_helpers.strip_reasoning_prose``
  walks the entire response and removes leading (and optionally trailing)
  reasoning sentences.  This module owns the regexes that function uses.
* **Stream-time** — ``gateway.stream_consumer.GatewayStreamConsumer`` strips
  the leading reasoning sentence on the *first* delta of a turn, so users
  never see the chain-of-thought flash by mid-stream.  It needs the same
  opener regex.

Originally both call sites imported the regexes as the private
``_REASONING_PROSE_OPENERS_RE`` and ``_SENTENCE_END`` symbols from
``agent.agent_runtime_helpers``.  That created a tight, non-obvious
cross-module coupling — renaming the private symbol would silently break
the gateway's stream-time stripper, with no obvious link in either file
explaining the dependency.

This module is the supported import surface.  Both call sites now use
the public ``REASONING_PROSE_OPENERS_RE`` and ``SENTENCE_END_RE`` names
imported from here, so the dependency is discoverable via the import
graph and grep finds it without needing to know the leading underscore
is load-bearing.

Pattern-design notes (kept here so future maintainers don't repeat the
mistake):

The opener patterns are *tuned* for chat-tuned reasoning models.  A naive
list of "thinking verbs" (let me, I think, so, actually, wait) is too
broad — those words show up in normal user-facing answers all the time
("I'll be there at 5", "wait for it", "the first thing to try is X").
The mistake we hit before: matching ``\\bi'll\\b`` or ``\\bso[,\\s]`` as
opener patterns *anywhere* in the text causes legitimate mid-sentence
phrases to be silently deleted from the user-visible reply.

So every pattern here is **start-of-message-anchored in practice** by
the consumer (``strip_reasoning_prose`` and the stream-time stripper
both walk from position 0 and stop as soon as a non-opener is hit), and
each pattern is **verb-shaped**: it must be a meta-cognitive opener, not
a content word.  "Let me check" is meta; "I'll be there" is not — the
distinction is that the meta opener comes with a thinking verb
attached (``let me VERB``, ``I think I VERB``, ``so let me``), and a
trailing verb is what separates it from the conversational use.
"""

from __future__ import annotations

import re


# ── Source of truth: the opener patterns ────────────────────────────────
# Each entry is a substring of a single regex alternative.  They match
# English chain-of-thought openers that some chat-tuned reasoning models
# emit directly in the assistant's visible content field.
#
# Tightness rules (read before adding new patterns):
#   1. A pattern must require a *thinking verb* attached.  Bare ``\\bso\\b``,
#      bare ``\\bfirst\\b``, bare ``\\bi'll\\b``, etc. will match in normal
#      answers and silently delete content.
#   2. A pattern must be *anchored* by the consumer to start-of-message.
#      Don't try to catch mid-message reasoning here — the whole function
#      stops at the first non-opener, so any opener it matches must be
#      the *first* thing in the message.
#   3. Keep the list short.  Every entry has a false-positive cost.  If
#      you're tempted to add ``\\bactually\\b`` remember "actually works"
#      is a legitimate reply, not reasoning.
_REASONING_PROSE_OPENERS: tuple[str, ...] = (
    # "Let me / Let's" + (optional adverb) + thinking verb.
    # The verb list is the *only* thing that separates "Let me check"
    # (reasoning) from "Let's meet at 5" (content).  Don't strip a verb.
    r"\blet['']?s\b\s+(?:also\s+|just\s+|first\s+|actually\s+|quickly\s+"
    r"|now\s+|try\s+to\s+)?"
    r"(?P<verb>think|check|look|trace|find|examine|reason|verify|"
    r"recall|consider|review|recheck|re-?verify|re-?check|test|push|"
    r"step\s+back|backtrack|skip|read|run|do|go|see|open|close|"
    r"inspect|investigate|walk|drill|dig|break|split|cross|"
    r"poke|grep|search|scan|hit|re-?read|cross-?check|take|put|"
    r"move|kill|restart|rebuild|recompile|rerun|reapply|revert|"
    r"apply|patch|fix|compare|diff|map|reconstruct|retrace|"
    r"simulate|isolate|identify|enumerate|summarise|summarize|"
    r"elaborate|expand|recap|give|hand|pass|set|tell|show|try|"
    r"attempt|see|head|jump|dive)\b",
    # "Now let me / Now I can see / Now I understand / Now the real"
    r"\bnow\s+(?:let['']?s\s+me|let\s+me|i\s+can\s+see|i\s+see|"
    r"i\s+have|i\s+understand|it['']?s\s+clear|the\s+real|"
    r"everything|the\s+full|here|we\s+have)\b",
    # "Found it" / "Found the bug" — punctuated and unpunctuated
    r"\bfound\s+(?:it|the|that|an?|one|two|three|my|our|another)\b",
    # "Aha" / "Aha —" (insight beat)
    r"\baha\b\s*[:—\-]?",
    # "I see the X" / "I can see the X" — needs a *noun phrase* after,
    # not "I see what you mean" (which is content).  The
    # required-following-article distinguishes them.
    r"\bi\s+(?:see|can\s+see)\s+(?:the|a|an|my|our|this|that|these|those)\b",
    # "Smoking gun" / "this is the root cause" / "this is the bug"
    r"\bthis\s+is\s+the\s+(?:smoking\s+gun|root\s+cause|bug|"
    r"real\s+issue|actual\s+issue|real\s+bug|actual\s+bug|"
    r"core\s+issue|key\s+issue)\b",
    # Trailing-realization beats (used for the trailing-sentence strip).
    # "Got it." / "Got it —" / "Right," / "Right —" / "OK so" /
    # "Okay so" / "Confirmed:".  These are LESS risky to match in the
    # leading pass too, because they only appear as sentence openers
    # when the model is announcing its own conclusion.  A user-facing
    # answer doesn't start with "Got it."
    r"\bgot\s+it\b\s*[:—\-.]?",
    r"\bright\s*[,—\-]",
    r"\b(?:ok|okay)\s+so\b",
    r"\bconfirmed\s*[:.]",
)

# Compile once at import.
# The boundary lookbehind ``(?:^|(?<=[\.\!\?]\s)|(?<=\n))`` matches the
# opener at the *start* of the message OR right after a sentence-ending
# punctuation + whitespace.  That's what lets the consumer walk sentence
# by sentence from the top.
REASONING_PROSE_OPENERS_RE: re.Pattern[str] = re.compile(
    r"(?i)(?:^|(?<=[\.\!\?]\s)|(?<=\n))"
    r"\s*"
    r"(?:" + "|".join(_REASONING_PROSE_OPENERS) + r")",
    flags=re.UNICODE,
)

# A "starts with reasoning opener" pattern anchored to position 0.
# Used by the stream-time consumer, which only ever looks at the start
# of the message.  This is the SAFE form — it can never match mid-sentence
# because it requires position 0.
STARTS_WITH_OPENER_RE: re.Pattern[str] = re.compile(
    r"\s*(?:" + "|".join(_REASONING_PROSE_OPENERS) + r")",
    flags=re.UNICODE | re.IGNORECASE,
)

# Sentence-end characters used to find the end of the preamble.
# The lookahead ``(?=[A-Z"'`(\[]|\*\*[A-Z])`` requires the *next*
# sentence to start with a capital / quote / parenthesis / bolded
# capital — that keeps "Wait —" or "Right —" from matching their
# own em-dash as a sentence boundary, and it keeps comma-followed
# clauses from being treated as separate sentences.
SENTENCE_END_RE: re.Pattern[str] = re.compile(
    r"(?<=[\.\!\?])\s+(?=[A-Z\"'`\(\[]|\*\*[A-Z])|$|\n\s*\n",
    re.UNICODE,
)
