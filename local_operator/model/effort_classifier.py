"""Tiny local prompt-complexity model -> reasoning effort.

OMP can delegate this to a small language model or a local sub-2B classifier.
Local Operator's constraint is stricter: the classifier exists to SAVE time
and output tokens, so paying another provider round trip before every user
message would erase the win on the short prompts most likely to classify low.
This module is therefore a tiny deterministic LINEAR model over cheap prompt
features — sub-millisecond, no network, no tokenizer, no install extra.

The output is coarse (lo / med / hi), then mapped onto the active model's own
``reasoning_efforts`` ladder. The mapping never chooses ``minimal`` for lo
when ``low`` exists, and hi defaults one rung below ``max`` (an operator can
set ``values.effort.allowMax: true``). The selection freezes for the whole
user-message tool loop: changing effort between tool calls would bust the
provider cache and make one task reason at several depths.

Disabled by default (``values.effort.auto: false``) so upgrading the harness
does not silently change an operator's model spend. Enable it explicitly:

    values:
      effort:
        auto: true
        allowMax: false
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_CODE_MARKERS = ("```", "def ", "class ", "function ", "=>", "traceback", "stack trace")
_COMPLEX_VERBS = {
    "implement",
    "build",
    "refactor",
    "debug",
    "design",
    "architect",
    "migrate",
    "review",
    "investigate",
    "compare",
    "optimize",
    "deploy",
    "release",
}
_SIMPLE_WORDS = {
    "hello",
    "hi",
    "thanks",
    "yes",
    "no",
    "continue",
    "retry",
    "status",
    "list",
    "show",
    "read",
    "find",
    "search",
}


@dataclass(frozen=True)
class Classification:
    tier: str
    score: float


class PromptEffortClassifier:
    """Eight-feature linear classifier. Weights are deliberately readable:
    future tuning can pin a behaviour in tests without downloading a model or
    changing an opaque embedding."""

    def classify(self, prompt: str) -> Classification:
        text = (prompt or "").strip()
        words = _WORD_RE.findall(text.lower())
        word_set = set(words)
        lines = text.count("\n") + 1 if text else 0
        score = 0.0
        # Length grows smoothly rather than one cliff: 200 words ≈ +2.
        score += min(len(words) / 100.0, 3.0)
        score += min(lines / 12.0, 2.0)
        score += 1.6 if any(marker in text.lower() for marker in _CODE_MARKERS) else 0.0
        score += min(len(word_set & _COMPLEX_VERBS) * 0.7, 2.8)
        score += (
            0.8
            if any(ch.isdigit() for ch in text) and ("error" in word_set or "test" in word_set)
            else 0.0
        )
        score += 0.8 if sum(text.count(ch) for ch in ("- ", "* ", "1.")) >= 3 else 0.0
        score += 0.7 if sum(text.count(sep) for sep in (" and ", " then ", ";")) >= 3 else 0.0
        if len(words) <= 12 and word_set & _SIMPLE_WORDS and not (word_set & _COMPLEX_VERBS):
            score -= 1.5
        if score < 0.8:
            tier = "lo"
        elif score < 4.0:
            tier = "med"
        else:
            tier = "hi"
        return Classification(tier, score)


def map_tier_to_effort(
    tier: str, efforts: tuple[str, ...], *, allow_max: bool = False
) -> str | None:
    """Map lo/med/hi onto one model's supported effort ladder."""
    if not efforts:
        return None
    ladder = list(efforts)
    if tier == "lo":
        # Never underthink at provider-specific "minimal" when low exists.
        return ladder[1] if ladder[0] == "minimal" and len(ladder) > 1 else ladder[0]
    if tier == "med":
        return ladder[len(ladder) // 2]
    # hi: protect spend/latency by stopping one rung below max by default.
    if not allow_max and ladder[-1] == "max" and len(ladder) > 1:
        return ladder[-2]
    return ladder[-1]


def auto_effort_for(
    prompt: str,
    efforts: tuple[str, ...],
    settings: Mapping[str, Any] | None,
) -> tuple[str | None, Classification | None]:
    """Configured effort + classification; (None, None) when disabled."""
    cfg = settings.get("effort", {}) if isinstance(settings, Mapping) else {}
    if not isinstance(cfg, Mapping) or not bool(cfg.get("auto", False)):
        return None, None
    result = PromptEffortClassifier().classify(prompt)
    effort = map_tier_to_effort(
        result.tier,
        efforts,
        allow_max=bool(cfg.get("allowMax", cfg.get("allow_max", False))),
    )
    return effort, result
