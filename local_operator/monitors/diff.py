"""Normalization, hashing and bounded line deltas (contract §7).

Pure functions. The scheduler owns the file state; this module owns the
comparison semantics:

- :func:`normalize` — line endings, trailing whitespace per line, optional
  timestamp scrubbing, per-monitor ignore regexes, optional line-multiset
  ordering. The result is the ONLY thing hashed or stored, so equality is
  defined by this output.
- :func:`content_hash` — ``sha256`` over the FULL normalized text. The
  counters file carries it as the equality fast-path; because it covers the
  full text it is exact regardless of the stored snapshot cap (§7.2).
- :func:`render_delta` / :func:`beyond_window_text` — the bounded delta a
  delivery carries. One delta is ≤ ``deltaMaxChars`` by construction.

Timestamp scrubbing is the single highest-yield noise class: an ``updated_at``
rewrite would otherwise be a "change" on every tick of any source that carries
one. The epoch floor/ceiling keep the rule away from ordinary numbers (ports,
row counts, short ids); 10-digit tokens in ``[1e9, 4_102_444_800]`` are
epoch-seconds between 2001 and 2100, 13-digit tokens in the millisecond
window the same.
"""

from __future__ import annotations

import difflib
import hashlib
import re

#: Where one preview line is clipped. A code constant, not a setting: the
#: total message budget (``deltaMaxChars``) is the contract's bound, and this
#: only stops one monstrous line from eating it whole.
DELTA_PREVIEW_CHARS = 200

#: The marker word for a scrubbed timestamp.
_TS_MARKER = "<ts>"

_ISO_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_EPOCH_MS_RE = re.compile(r"\b\d{13}\b")
_EPOCH_S_RE = re.compile(r"\b\d{10}\b")

_EPOCH_S_FLOOR = 1_000_000_000  # 2001-09-09
_EPOCH_S_CEIL = 4_102_444_800  # 2100-01-01
_EPOCH_MS_FLOOR = 1_000_000_000_000
_EPOCH_MS_CEIL = 4_102_444_800_000


def normalize(
    text: str,
    *,
    sort_lines: bool = False,
    ignore: tuple[str, ...] | list[str] = (),
    normalize_timestamps: bool = True,
) -> str:
    """The stored/comparable form of one tool result (§7.1, applied in order)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    # A trailing newline is formatting, not content: the final empty element
    # (and any all-whitespace tail) drops. Documented rather than incidental —
    # without it every source that sometimes ends in a newline would flap.
    while lines and not lines[-1]:
        lines.pop()
    if normalize_timestamps:
        lines = [_strip_timestamps(line) for line in lines]
    if ignore:
        patterns = [re.compile(pattern) for pattern in ignore]
        lines = [line for line in lines if not any(p.search(line) for p in patterns)]
    if sort_lines:
        # Sorting the STORED form is what makes equality the multiset compare
        # §7.1 asks for: the hash and the diff both see line sets.
        lines = sorted(lines)
    return "\n".join(lines)


def _strip_timestamps(line: str) -> str:
    if not line:
        return line
    line = _ISO_TS_RE.sub(_TS_MARKER, line)

    def _epoch_ms(match: re.Match[str]) -> str:
        value = int(match.group(0))
        return _TS_MARKER if _EPOCH_MS_FLOOR <= value <= _EPOCH_MS_CEIL else match.group(0)

    def _epoch_s(match: re.Match[str]) -> str:
        value = int(match.group(0))
        return _TS_MARKER if _EPOCH_S_FLOOR <= value <= _EPOCH_S_CEIL else match.group(0)

    line = _EPOCH_MS_RE.sub(_epoch_ms, line)
    return _EPOCH_S_RE.sub(_epoch_s, line)


def content_hash(normalized: str) -> str:
    """``sha256:…`` over the full normalized text (the counters fast-path)."""
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8", "replace")).hexdigest()


def has_line_difference(old: str, new: str) -> bool:
    """Whether two normalized texts differ as line sequences."""
    return old.split("\n") != new.split("\n")


def count_changes(old: str, new: str) -> tuple[int, int]:
    """``(added, removed)`` changed lines between two normalized texts."""
    matcher = difflib.SequenceMatcher(None, old.split("\n"), new.split("\n"), autojunk=False)
    added = removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


def render_delta(
    old: str,
    new: str,
    *,
    max_delta_lines: int,
    delta_max_chars: int,
) -> tuple[str, int]:
    """``(delta_text, changed_lines)`` — counts + bounded previews (§7.3).

    The input is bounded by the snapshot cap (~1k lines worst case), which
    keeps the quadratic matcher cheap — the unit suite pins that bound.
    """
    matcher = difflib.SequenceMatcher(None, old.split("\n"), new.split("\n"), autojunk=False)
    opcodes = matcher.get_opcodes()
    added = removed = 0
    previews: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
        if tag == "equal":
            continue
        old_lines = old.split("\n")[i1:i2]
        new_lines = new.split("\n")[j1:j2]
        for line in old_lines:
            previews.append("- " + _clip(line))
        for line in new_lines:
            previews.append("+ " + _clip(line))

    changed = added + removed
    header = f"+{added}/-{removed} changed lines"
    shown = previews[: max(0, max_delta_lines)]
    dropped = changed - len(shown)
    while True:
        marker = f"… and {dropped} more changed lines" if dropped > 0 else ""
        body_lines = [header, *shown]
        if marker:
            body_lines.append(marker)
        text = "\n".join(body_lines)
        if len(text) <= delta_max_chars or not shown:
            return text, changed
        shown.pop()
        dropped += 1


def beyond_window_text(new_hash: str) -> tuple[str, int]:
    """The honest delta when there is no diff input (§7.2 loss semantics)."""
    return (
        f"change beyond the stored snapshot window (current checksum {new_hash})",
        1,
    )


def _clip(line: str) -> str:
    if len(line) <= DELTA_PREVIEW_CHARS:
        return line
    return line[:DELTA_PREVIEW_CHARS] + "…"
