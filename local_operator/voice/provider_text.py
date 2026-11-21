"""Provider text that is about to be shown to the app or written to the log.

The voice routes quote what the speech provider said (a 429's sentence, a 502's
clause) and log it. The HTTP clients already mask the WHOLE key and the shapes of
a credential, but a provider may echo something in between: the first characters
of the key, a masked ``abcd...wxyz`` form, or the key with whitespace or hyphens
put through it. :func:`redact_provider_text` removes those, for every secret the
request could have used, and :func:`secrets_in_play` finds them.

Import-light (``voice`` is read on the feed's poll cadence); the store and the
Radient resolver are imported when a failure is being reported, not before.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from local_operator.redaction_shapes import REDACTION_MARKER, scrub_secrets

logger = logging.getLogger(__name__)

#: A secret shorter than this is skipped: a 6-character window of it would be most
#: of the secret, and common words would be "leaks" of it.
MIN_SECRET_CHARS = 8

#: Consecutive characters of one secret that make a fragment worth removing.
RUN_CHARS = 6

#: Seconds one secret lookup may take on the failure path.
_LOOKUP_TIMEOUT_S = 5.0

#: What a masking provider puts between the head and the tail it keeps.
_MASK_RUN = r"[.\u2026*\u2022\u00b7\s]{1,8}"
#: Whitespace and hyphens: what splits a key across a wrapped line or a "sk-"-style
#: grouping without being part of what was sent.
_SEPARATORS = re.compile(r"[\s\-]")


def _spans_of_runs(text: str, secret: str, *, fold: bool) -> list[tuple[int, int]]:
    """``[start, end)`` spans of ``text`` that spell :data:`RUN_CHARS`+ of ``secret``.

    ``fold`` ignores whitespace and hyphens on both sides, so a key split by them
    still matches; a span then runs from the first matched character to the last,
    and covers the separators between them.
    """
    if fold:
        positions = [i for i, ch in enumerate(text) if not _SEPARATORS.match(ch)]
        haystack = "".join(text[i] for i in positions).casefold()
        needle = _SEPARATORS.sub("", secret).casefold()
    else:
        positions = list(range(len(text)))
        haystack = text.casefold()
        needle = secret.casefold()
    if len(haystack) != len(positions) or len(needle) < RUN_CHARS:
        # ``casefold`` can change a string's length (``ß``); positions would then
        # point at the wrong characters, so fall back to the plain lower-case.
        haystack, needle = haystack.lower(), needle.lower()
        if len(haystack) != len(positions) or len(needle) < RUN_CHARS:
            return []
    windows = {needle[i : i + RUN_CHARS] for i in range(len(needle) - RUN_CHARS + 1)}
    covered = [False] * len(haystack)
    for i in range(len(haystack) - RUN_CHARS + 1):
        if haystack[i : i + RUN_CHARS] in windows:
            for j in range(i, i + RUN_CHARS):
                covered[j] = True
    spans: list[tuple[int, int]] = []
    j = 0
    while j < len(covered):
        if not covered[j]:
            j += 1
            continue
        k = j
        while k + 1 < len(covered) and covered[k + 1]:
            k += 1
        spans.append((positions[j], positions[k] + 1))
        j = k + 1
    return spans


def _spans_of_masks(text: str, secret: str) -> list[tuple[int, int]]:
    """Spans of the masked form: the first four characters, a mask, the last four."""
    pattern = re.compile(
        re.escape(secret[:4]) + _MASK_RUN + re.escape(secret[-4:]), flags=re.IGNORECASE
    )
    return [m.span() for m in pattern.finditer(text)]


def redact_provider_text(text: str, secrets: Iterable[Optional[str]]) -> str:
    """``text`` with every fragment of ``secrets`` replaced by the redaction marker.

    Removes, per secret: its whole value and the credential shapes (the same pass
    the clients run, kept as the first line), any run of :data:`RUN_CHARS` or more
    consecutive characters of it (so a prefix, a suffix or a middle slice), the
    same run with whitespace or hyphens inserted, and the first-4...last-4 masked
    form. The rule is per secret: a word is only removed when it is part of a
    secret that was actually in play. Secrets under :data:`MIN_SECRET_CHARS` are
    skipped, and an empty list returns ``text`` untouched.

    Args:
        text: A provider's sentence or an error message.
        secrets: Every credential the request could have sent.

    Returns:
        The text, safe to put in a response body or a log line.
    """
    live = sorted(
        {s for s in secrets if isinstance(s, str) and len(s) >= MIN_SECRET_CHARS},
        key=len,
        reverse=True,
    )
    if not live or not text:
        return text
    text = scrub_secrets(text, live)
    spans: list[tuple[int, int]] = []
    for secret in live:
        spans += _spans_of_runs(text, secret, fold=False)
        spans += _spans_of_runs(text, secret, fold=True)
        spans += _spans_of_masks(text, secret)
    if not spans:
        return text
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: list[str] = []
    cursor = 0
    for start, end in merged:
        out.append(text[cursor:start])
        # An existing marker (the exact-value pass ran first) must not be doubled.
        out.append(REDACTION_MARKER)
        cursor = end
    out.append(text[cursor:])
    return "".join(out).replace(REDACTION_MARKER * 2, REDACTION_MARKER)


async def _read(label: str, read: Any) -> list[str]:
    """One lookup, never raising: an unreadable row hides itself and nothing else."""
    try:
        # Bounded: a refresh behind the Radient lookup must not hold the refusal.
        value = await asyncio.wait_for(read(), timeout=_LOOKUP_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - redaction must not take the answer down
        # The type only: the message of a store failure is not ours to quote.
        logger.debug("voice secret lookup for %s failed", label)
        return []
    if value is None:
        return []
    revealed = getattr(value, "get_secret_value", None)
    value = revealed() if callable(revealed) else value
    return [value] if isinstance(value, str) and value else []


async def secrets_in_play(
    store: Any,
    *,
    config_dir: Path | None,
    base_url: str,
    session_id: str | None,
) -> list[str]:
    """Every key a voice request could have sent, for :func:`redact_provider_text`.

    Read-only lookups of what the cascades resolve: the ElevenLabs key, the Radient
    credential, and the OpenAI and Groq speech keys the walk may reach. Called
    only once a failure is being reported, so the success path pays nothing.
    The values go to the redactor and nowhere else: never logged, never returned
    to a client. Never raises; a missing or unreadable store gives what could be
    read, which may be nothing.
    """
    from local_operator.providers.radient_credentials import resolve_radient_credential
    from local_operator.stt.cascade import (
        GROQ_STT_NAMESPACE,
        OPENAI_STT_KINDS,
        OPENAI_STT_NAMESPACE,
    )
    from local_operator.tts.cascade import (
        GROQ_TTS_NAMESPACE,
        OPENAI_TTS_KINDS,
        OPENAI_TTS_NAMESPACE,
    )

    if store is None:
        return []
    found: list[str] = []
    found += await _read(
        "elevenlabs", lambda: store.get_api_key("elevenlabs", session_id, read_only=True)
    )
    found += await _read(
        "radient", lambda: resolve_radient_credential(config_dir, base_url, store=store)
    )
    for namespace, kinds in (
        (OPENAI_STT_NAMESPACE, OPENAI_STT_KINDS),
        (OPENAI_TTS_NAMESPACE, OPENAI_TTS_KINDS),
        (GROQ_STT_NAMESPACE, OPENAI_STT_KINDS),
        (GROQ_TTS_NAMESPACE, OPENAI_TTS_KINDS),
    ):
        found += await _read(
            namespace,
            lambda namespace=namespace, kinds=kinds: store.get_persisted_api_key(
                namespace, session_id, kinds=kinds
            ),
        )
    return list(dict.fromkeys(found))
