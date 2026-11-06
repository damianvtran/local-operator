"""Normalization, hashing and bounded line deltas (contract §7).

Pure functions. The scheduler owns the file state; this module owns the
comparison semantics:

- :func:`unpack_records` — the record split that runs BEFORE everything
  else: a JSON envelope whose string field is a RECORD LOG
  (``{"name": "…"}`` / ``{"results": "…"}``) becomes one line per record.
  A monitor watches RECORDS, not an envelope, and :func:`is_record_log`
  says whether a result took that path.
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
import json
import re
from dataclasses import dataclass

#: Where one preview line is clipped. A code constant, not a setting: the
#: total message budget (``deltaMaxChars``) is the contract's bound, and this
#: only stops one monstrous line from eating it whole.
DELTA_PREVIEW_CHARS = 200

#: How much text a differing-line window keeps BEFORE the first difference.
#: A replace hunk whose lines are longer than a preview used to render two
#: identical prefix clips (the live Slack-thread failure: a 13.7 KB single-line
#: JSON whose only visible difference was the spill footer's char count), so a
#: long pair is now shown around the character where the two lines diverge.
DELTA_WINDOW_LEAD_CHARS = 40

#: The marker word for a scrubbed timestamp.
_TS_MARKER = "<ts>"

_ISO_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_EPOCH_MS_RE = re.compile(r"\b\d{13}\b")
#: ``\d{10}`` NOT followed by a dotted six-digit fraction. A dotted pair
#: (``1793908014.147179``) is a Slack message id, i.e. an IDENTIFIER, not a clock
#: reading: scrubbing it destroyed the one field that lets a reader call
#: ``mcp__slack_read_thread`` straight from the wake, which is why it is exempt
#: here (design D1). Every bare 10-digit epoch is still scrubbed.
_EPOCH_S_RE = re.compile(r"\b\d{10}\b(?!\.\d{6})")

_EPOCH_S_FLOOR = 1_000_000_000  # 2001-09-09
_EPOCH_S_CEIL = 4_102_444_800  # 2100-01-01
_EPOCH_MS_FLOOR = 1_000_000_000_000
_EPOCH_MS_CEIL = 4_102_444_800_000


#: Blank lines separate the BLOCKS of an envelope's payload. Whether a block
#: starts a new record is decided by the repeated record lead, not by the blank
#: line itself — a message body may contain blank lines (see ``_record_lines``).
_ENVELOPE_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n+")

#: What joins a record's own fields (and a continuation block) back up, so one
#: record stays ONE comparable line.
_RECORD_FIELD_SEPARATOR = " · "

#: A payload must contain a blank line before it is treated as a record log at
#: all. Two newlines in a row is that blank line; a single-newline string field
#: is prose (a pagination sentence) and stays a ``key: value`` line.
_RECORD_LOG_MARKER = "\n\n"

#: Envelope fields that are PAGE STATE, never content. A cursor changes whenever
#: the page window moves, so comparing it makes every tick a change (measured:
#: it churned a line on every message in the 2026-11-05 QA run). Dropped from
#: the compared text entirely rather than rendered and then ignored.
_ENVELOPE_BOOKKEEPING_KEYS = frozenset(
    {
        "pagination_info",
        "pagination",
        "next_cursor",
        "cursor",
        "next_page",
        "has_more",
        "has_next",
    }
)

#: A record's OWN clock, as a source renders it (``2026-11-05 14:46:54 EST`` or
#: ``2026-11-05T14:46:54Z``). Rewritten to ``2026/11/05 …`` inside a record line
#: before timestamp scrubbing runs, because for a record log the clock is
#: CONTENT: "when was this said" is the first thing a reader of a wake needs,
#: and the operator's own report was a timing report ("Victor replied at 20:17").
#: The slash form is deliberately outside ``_ISO_TS_RE``, so the global churn
#: rule keeps applying to every other timestamp in every other source.
_RECORD_CLOCK_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})([T ])(?=\d{2}:\d{2})")


def is_record_log(text: str) -> bool:
    """Whether THIS RESULT is a JSON envelope holding a record log.

    The trigger is the shape of the result — the same one
    :func:`unpack_records` uses — not the provenance of the source: a result
    counts while it parses as an object with a blank-line-separated payload, and
    a source that stops returning that shape stops counting on the next tick.
    The decision is what tells the scheduler to judge the delta structurally
    instead of by the materiality gate (see :func:`unpack_records`).

    Consequence, stated where it bites: a plain-text source that STARTS
    returning a record shape changes comparison basis mid-stream, so its next
    tick skips the gate and delivers the whole payload once rather than a delta.
    """
    return _record_log_envelope(text) is not None


def unpack_records(text: str) -> str:
    """One line per record when ``text`` is a JSON envelope over a record log.

    The shape this exists for: an MCP read tool returns its result as
    ``{"messages": "<a whole channel log in ONE string>"}`` (or
    ``{"results": …}``, ``{"thread": …}``), where the payload string carries
    real newlines once decoded. Read literally the envelope is ONE line of
    escaped JSON, so a new message can only ever REPLACE it: the delta the
    materiality gate then sees is a window of escaped bookkeeping
    (``Message TS: …``, a cursor) instead of the message, and the gate answers
    ``non-material-metadata``. Measured 2026-11-05 on a
    ``mcp__slack_read_channel`` watch: two real messages delivered nothing,
    ``suppressed.non_material_metadata: 2``. The same single line hid thread
    replies (a ``Thread: N replies (latest: …)`` field buried inside it) and
    made the watcher's own posts impossible to ``ignore``.

    A monitor watches RECORDS, not an envelope, so one line per record is the
    unit everything downstream already assumes: a new message becomes an
    ADDED line, ``ignore`` drops a record line whole, and a thread's reply
    count changing is a changed line that reads as news instead of as part of
    one long escaped line.

    Trigger: the WHOLE text parses as a JSON object AND at least one value is
    a ``str`` holding a BLANK LINE (two newlines) — i.e. the envelope was
    JSON-escaped around a multi-line log. Everything else is returned byte for
    byte unchanged: plain text, a compact JSON document whose strings are
    single-line (``{"messages": [{…}]}`` is a list of objects), an array, a
    prose field such as ``pagination_info``, and a TRUNCATED result that no
    longer parses. It never raises: a tick may not fail because a source
    produced something odd, and the fallback is exactly the previous
    behaviour.

    Deterministic and idempotent: the envelope's own key order is honoured
    (nested containers are dumped with sorted keys; a TOP-LEVEL key reorder
    does reorder the lines, which is why the key order is read from the source
    and never re-sorted), and the unpacked text no longer parses as JSON, so
    re-normalizing it — which the scheduler's baseline reproduction does — is
    a no-op rather than a second, different transform.
    """
    envelope = _record_log_envelope(text)
    if envelope is None:
        return text
    lines: list[str] = []
    for key, value in envelope.items():
        if key in _ENVELOPE_BOOKKEEPING_KEYS:
            continue
        if isinstance(value, str) and _is_record_payload(value):
            lines.extend(_record_lines(value))
        else:
            lines.append(f"{key}: {_envelope_scalar(value)}")
    return "\n".join(lines)


def _record_log_envelope(text: str) -> dict[str, object] | None:
    """``text`` parsed as an envelope over a record log, or ``None``."""
    candidate = text.strip()
    # Cheap reject before the parse: only an object can be this envelope, and
    # most tool results are not JSON at all.
    if not candidate.startswith("{"):
        return None
    try:
        data = json.loads(candidate)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    if not any(_is_record_payload(value) for value in data.values()):
        return None
    return data


def _is_record_payload(value: object) -> bool:
    """A payload string holding blank-line-separated records."""
    return isinstance(value, str) and _RECORD_LOG_MARKER in value


def _record_lines(payload: str) -> list[str]:
    """One line per record, a multi-block body joined into its own record.

    The record boundary is the REPEATED LEAD, not the blank line: Slack renders
    each message as ``=== Message from … ===`` / ``Message TS: …`` / body, and a
    body with a paragraph break is still one message. Splitting on every blank
    line fragmented a real message into a header plus four unattributed
    fragments (measured: ``+2/-7`` where the record was one new message), which
    also defeated ``ignore`` — the fragment carrying the author id was dropped
    and the prose survived.

    So the blocks are classified: a block whose first token repeats across at
    least two blocks is a record lead; every block starting with it opens a new
    record, and every other block continues the record it follows (joined with
    the field separator, so the author, the id and the time stay on ONE line).
    Blocks before the first record (a channel header) stay lines of their own.
    With no repeated lead the payload is not structured this way and keeps the
    plain block-per-line behaviour.
    """
    blocks = _payload_blocks(payload)
    lead = _record_lead(blocks)

    def starts_record(block: list[str]) -> bool:
        if lead is not None:
            return _block_lead(block) == lead
        # No repeated lead: a ONE-record payload still has a record header, and
        # the fallback below is for payloads with no record structure at all, not
        # for a single-record log whose body happens to have blank lines.
        return _is_record_header_block(block)

    if not any(starts_record(block) for block in blocks):
        return [_RECORD_FIELD_SEPARATOR.join(_record_fields(block)) for block in blocks]
    lines: list[str] = []
    current: list[str] | None = None
    for block in blocks:
        if starts_record(block):
            if current is not None:
                lines.append(_RECORD_FIELD_SEPARATOR.join(current))
            current = _record_fields(block)
        elif current is not None:
            current.extend(_record_fields(block))
        else:
            lines.append(_RECORD_FIELD_SEPARATOR.join(_record_fields(block)))
    if current is not None:
        lines.append(_RECORD_FIELD_SEPARATOR.join(current))
    return lines


def _payload_blocks(payload: str) -> list[list[str]]:
    """The payload's blank-line-separated blocks, as lists of field lines."""
    text = payload.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[list[str]] = []
    for raw in _ENVELOPE_BLOCK_SPLIT.split(text):
        # rstripped, not stripped: leading indentation is content (an
        # indentation-only change in a line-mode payload must stay visible;
        # normalize() rstrips every line anyway, so nothing trailing survives).
        fields = [field.rstrip() for field in raw.split("\n")]
        kept = [field for field in fields if field.strip()]
        if kept:
            blocks.append(kept)
    return blocks


def _block_lead(block: list[str]) -> str:
    """The block's first token: what a repeated record prefix repeats."""
    first = block[0].lstrip()
    return first.split(" ", 1)[0]


def _record_lead(blocks: list[list[str]]) -> str | None:
    """The token that opens a record, or ``None`` when no token repeats.

    First-appearance order decides a tie, so the choice is deterministic for a
    given payload rather than dependent on hash ordering.
    """
    counts: dict[str, int] = {}
    for block in blocks:
        if not block:
            continue
        token = _block_lead(block)
        counts[token] = counts.get(token, 0) + 1
    for token, count in counts.items():
        if count >= 2:
            return token
    return None


#: The fields a record's rendering applies to: the ones that say WHO, WHEN and
#: WHICH message. Another field — the message BODY — is content the source wrote,
#: so its timestamps keep the global scrub (a body ``updated_at`` is exactly the
#: churn class that rule exists for; review R2-MIN-A).
_RECORD_HEADER_FIELD_RE = re.compile(
    r"^(?:===.*===|Message TS:|Time:|Sent:|Date:|Sent at:)(?:\s|$)"
)


def _is_record_header_field(field: str) -> bool:
    return bool(_RECORD_HEADER_FIELD_RE.match(field.strip()))


def _is_record_header_block(block: list[str]) -> bool:
    """Whether a block opens a record even without a repeated lead (R2-MIN-D).

    The Slack rendering's lead is ``=== Message from … ===``. A ONE-record
    payload has no repeated lead, so the lead rule cannot see it; the header
    shape can.
    """
    return bool(block) and _is_record_header_field(block[0])


def _record_fields(block: list[str]) -> list[str]:
    """A block's fields with only its HEADER clock made readable.

    A header's own clock and message id are the fields a reader of the wake
    needs (``at 2026/11/05 20:37 EST``, ``Message TS: 1793908014.147179``).
    Everything else in the record is body text: its timestamps stay masked, so a
    changing body stamp cannot ride out ungated.
    """
    fields: list[str] = []
    for field in block:
        if _is_record_header_field(field):
            fields.append(_RECORD_CLOCK_RE.sub(r"\1/\2/\3 ", field))
        else:
            # A dotted epoch pair would otherwise survive the global scrub; in a
            # BODY it is just numbers that can churn.
            fields.append(_LEGACY_DOTTED_EPOCH_RE.sub(_TS_MARKER, field))
    return fields


#: The pre-record-view MASK, applied to BOTH sides of a basis-change diff
#: (:func:`legacy_masked`). A stored baseline written before the record view was
#: already masked — ``at <ts> EST``, ``Message TS: <ts>.147170`` — and no
#: restatement can unmask it, so comparing it against today's UNMASKED rendering
#: reports every record as changed even when the source did not move (measured by
#: review round 2 as a ``+2/-2`` phantom delta, and ``+25/-25`` on a 20-record
#: window). Masking both sides keeps only real content changes.
#: The readable clock form the record view renders, masked back to the shape the
#: old pipeline stored. A trailing ``Z``/offset belongs to the clock (the ISO
#: pattern swallowed it), so it is masked with it; a `` EST``-style zone was
#: never part of the ISO match and stays.
_LEGACY_SLASH_CLOCK_RE = re.compile(
    r"\b\d{4}/\d{2}/\d{2} \d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?"
)
_LEGACY_DOTTED_EPOCH_RE = re.compile(r"\b\d{10}(?=\.\d{6})")
_LEGACY_EPOCH_S_RE = re.compile(r"\b\d{10}\b")


def legacy_masked(text: str) -> str:
    """``text`` with the pre-record-view timestamp MASK applied to every line.

    The basis-change diff compares in this space: the stored baseline is already
    masked, and today's rendering is not, so masking both sides makes the two
    comparable without undo-masking anything (the unmasked rendering is kept for
    everything else — review MAJ-3 stays closed). Only real changes survive it.
    """
    masked: list[str] = []
    for line in text.split("\n"):
        line = _ISO_TS_RE.sub(_TS_MARKER, line)
        line = _LEGACY_SLASH_CLOCK_RE.sub(_TS_MARKER, line)
        line = _LEGACY_DOTTED_EPOCH_RE.sub(_TS_MARKER, line)
        line = _EPOCH_MS_RE.sub(_mask_epoch_ms, line)
        masked.append(_LEGACY_EPOCH_S_RE.sub(_mask_epoch_s, line))
    return "\n".join(masked)


def _envelope_scalar(value: object) -> str:
    """``key: value`` rendering for an envelope field that is not a record log."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value)


def normalize(
    text: str,
    *,
    sort_lines: bool = False,
    ignore: tuple[str, ...] | list[str] = (),
    normalize_timestamps: bool = True,
) -> str:
    """The stored/comparable form of one tool result (§7.1, applied in order)."""
    text = unpack_records(text)
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


def _mask_epoch_ms(match: re.Match[str]) -> str:
    value = int(match.group(0))
    return _TS_MARKER if _EPOCH_MS_FLOOR <= value <= _EPOCH_MS_CEIL else match.group(0)


def _mask_epoch_s(match: re.Match[str]) -> str:
    value = int(match.group(0))
    return _TS_MARKER if _EPOCH_S_FLOOR <= value <= _EPOCH_S_CEIL else match.group(0)


def _strip_timestamps(line: str) -> str:
    if not line:
        return line
    line = _ISO_TS_RE.sub(_TS_MARKER, line)
    line = _EPOCH_MS_RE.sub(_mask_epoch_ms, line)
    return _EPOCH_S_RE.sub(_mask_epoch_s, line)


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


#: A ``read`` line-number gutter (``12| ``) — presentation the tool adds, not
#: content, so it must not make a lone timestamp line look like a record.
_GUTTER_RE = re.compile(r"^\s*\d+\|\s?")


def _strip_gutters(lines: list[str]) -> list[str]:
    """``read``'s ``N| `` line-number gutter removed from every line."""
    return [_GUTTER_RE.sub("", line) for line in lines]


def is_pure_addition(old: str, new: str) -> bool:
    """Whether ``new`` only ADDS content to ``old``: no line removed or changed.

    The shape of an append-only log (a build-progress file, an audit trail) and
    of a listing that gained a row. It is the one delta shape the classifier
    gate must not judge: the gate's suppress classes are about bookkeeping that
    CHANGED without meaning (timestamps, ordering, volatile ids), and a model
    asked about ``+ 4| === c2 done rc=0 <ts> ===`` answered
    ``non-material-metadata`` in the 2026-10-09 regression. An added line is a
    new record, which is what the gate's own rubric calls MATERIAL.

    Guard: at least one added line must carry content beyond the line-number
    gutter and the ``<ts>`` scrub marker, so a delta that adds only blank or
    timestamp-only lines still goes to the gate.

    An EMPTY ``old`` (the watched command printed nothing last time) counts as
    zero lines, so ``"" -> "ERROR x"`` (the ``grep``/``ls`` first-match shape)
    is an addition; ``"".split("\\n")`` would otherwise be ``[""]`` and make it
    a replace. This is not baseline establishment: that path has no snapshot
    and never reaches a delta.

    Callers must not use this on a TRUNCATED snapshot: a tail edit beyond the
    stored window reads as a pure insert there (the scheduler checks).

    ``read``'s gutter is stripped from BOTH sides before comparing (QA round 1,
    Q1): crossing 9→10 lines re-pads every line number (``9| `` -> ``10| ``),
    which without this reads as a full replace and would gate an append that
    only gained a line. A content change that merely alters a gutter-like
    prefix (``3| x`` -> ``4| x``) then shows no opcodes at all and returns
    False — it keeps going to the gate, which is the conservative direction.
    """
    old_lines = _strip_gutters(old.split("\n")) if old else []
    new_lines = _strip_gutters(new.split("\n"))
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    carries_content = False
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            return False
        if tag == "insert":
            for line in new_lines[j1:j2]:
                body = line.replace(_TS_MARKER, "")
                if any(ch.isalnum() for ch in body):
                    carries_content = True
    return carries_content


@dataclass(frozen=True)
class DeltaRender:
    """One rendered delta plus what its previews cannot show (§7.3, §8.4).

    ``hidden`` is the gate's fail-open signal: True when a changed line's
    RENDERED preview equals its counterpart's while the raw lines differ — the
    preview hides the difference, so the materiality gate must not be asked to
    judge evidence it cannot see. :func:`render_delta` structures every long
    replace pair as a character window precisely so this stays False; the flag
    is the guard that keeps a future renderer change from silently restoring
    the 2026-10-20 blind suppression.
    """

    text: str
    changed_lines: int
    hidden: bool = False


def render_delta(
    old: str,
    new: str,
    *,
    max_delta_lines: int,
    delta_max_chars: int,
) -> DeltaRender:
    """Counts + bounded previews, and whether a preview hides its change (§7.3).

    The input is bounded by the snapshot cap (~1k lines worst case), which
    keeps the quadratic matcher cheap — the unit suite pins that bound.
    """
    matcher = difflib.SequenceMatcher(None, old.split("\n"), new.split("\n"), autojunk=False)
    opcodes = matcher.get_opcodes()
    added = removed = 0
    previews: list[tuple[str, bool]] = []
    hidden = False
    old_split = old.split("\n")
    new_split = new.split("\n")
    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
        if tag == "equal":
            continue
        old_lines = old_split[i1:i2]
        new_lines = new_split[j1:j2]
        # Preview ORDER is the pre-window contract: every ``-`` line of a hunk,
        # then every ``+`` line. The gate reads this text, so reordering a
        # multi-line replace would change its input for diffs that never needed
        # the character window — short-line hunks stay byte-identical.
        minus: list[str] = []
        plus: list[str] = []
        if tag == "replace":
            # Pair the two sides positionally: a replaced line's information is
            # the CHARACTERS that differ, and a prefix clip of a long JSON line
            # shows only the shared header, so both sides read identically (the
            # live Slack monitor suppressed seven new replies this way). One
            # window anchored at the first differing character keeps the change
            # inside the preview and its context around it.
            paired = min(len(old_lines), len(new_lines))
            for index in range(paired):
                old_preview, new_preview = _pair_previews(old_lines[index], new_lines[index])
                minus.append(old_preview)
                plus.append(new_preview)
                if old_preview == new_preview:
                    hidden = True
            for line in old_lines[paired:]:
                minus.append(_clip(line))
            for line in new_lines[paired:]:
                plus.append(_clip(line))
        else:
            # An ADDED line is not clipped here: for a record log the added line
            # IS the new content, and a 200-char prefix of a real message cut
            # the body off mid-sentence (measured: an 823-char record delivered
            # as a link preamble and a ``…``, design D1/MAJ-1). It is bounded by
            # the delivery budget below instead. A REMOVED line keeps the clip:
            # its job is to identify what went, not to be read in full.
            minus = [_clip(line) for line in old_lines]
            plus = list(new_lines)
        previews.extend(("- " + preview, False) for preview in minus)
        previews.extend(("+ " + preview, True) for preview in plus)

    changed = added + removed
    header = f"+{added}/-{removed} changed lines"
    # The budget is spent in preview order, so what a reader sees first is what
    # the diff considered first. An added line that would overflow what is left
    # is cut there (with the ellipsis marking it) rather than dropped whole —
    # a count with no content is not a wake. Every other line keeps the old
    # behaviour: it is shown, or it is dropped and counted in the marker.
    shown: list[str] = []
    used = len(header) + 1
    for text_preview, is_addition in previews:
        if len(shown) >= max(0, max_delta_lines):
            break
        room = delta_max_chars - used
        if room <= 2:
            break
        if len(text_preview) + 1 > room:
            if not is_addition:
                break
            text_preview = text_preview[: max(0, room - 2)] + "…"
        shown.append(text_preview)
        used += len(text_preview) + 1
    dropped = changed - len(shown)
    while True:
        marker = f"… and {dropped} more changed lines" if dropped > 0 else ""
        body_lines = [header, *shown]
        if marker:
            body_lines.append(marker)
        text = "\n".join(body_lines)
        if len(text) <= delta_max_chars or not shown:
            return DeltaRender(text=text, changed_lines=changed, hidden=hidden)
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


def _common_prefix_length(left: str, right: str) -> int:
    """Characters the two lines share before the first difference."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _window_preview(line: str, start: int) -> str:
    """One preview line from ``start``, marked where it was cut."""
    head = "…" if start > 0 else ""
    # One char of headroom for the trailing marker, so a window never renders
    # longer than the plain clip it replaces.
    budget = DELTA_PREVIEW_CHARS - len(head) - 1
    body = line[start : start + budget]
    tail = "…" if start + len(body) < len(line) else ""
    return head + body + tail


def _pair_previews(old_line: str, new_line: str) -> tuple[str, str]:
    """Two previews of one replaced pair, windowed on their first difference.

    Both windows start at the same character — ``DELTA_WINDOW_LEAD_CHARS``
    before the divergence — so the reader sees the text that changed, not two
    copies of a shared 200-char header. Short lines are clipped exactly as
    before; only a pair with a line past :data:`DELTA_PREVIEW_CHARS` is
    windowed, which keeps every existing delta byte-identical.
    """
    if len(old_line) <= DELTA_PREVIEW_CHARS and len(new_line) <= DELTA_PREVIEW_CHARS:
        return _clip(old_line), _clip(new_line)
    start = max(0, _common_prefix_length(old_line, new_line) - DELTA_WINDOW_LEAD_CHARS)
    return _window_preview(old_line, start), _window_preview(new_line, start)
