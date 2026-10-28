"""Source-provenance verifier for §6.1(6) — the auto gate.

THE RULE THIS IMPLEMENTS. "Each record's content is a substring of the span's
source text after the method's documented truncation and quoting" (§6.1(6)).
Deleting and truncating keep every retained run CONTIGUOUS in the source, so a
record body is a sequence of literal runs, in order, split only by the method's
own structural elements: closed `[radient:` markers, the reducer's fixed note
tokens, and its `…` elision mark. Nothing else may separate two runs.

WHY NOT "RENDERED FORMS". Round 2 tried to re-render the reducer's output shapes
(call summaries, heredoc notes, `i`-key dropping, sliced identifier pools). That
is a second, drifting implementation of the reducer: it rejected 108 of 133
honest real-session epochs, then still admitted 27,000 fabricated markers
because a marker-only record had no runs to match. Substring-in-order matching
needs no shape catalogue — a dropped `i` key, a 90-char cut, a suffix of a
longer identifier and a `…` truncation are all simply "some bytes were deleted".

WHY THE BOUNDS. Byte shrink (check 1) does not bound TOKENS: 74,000 single
characters separated by `\\n\\t` shrink in bytes and cost 5x the tokens, and
repeated markers cost 3.3x. So structural overhead (separators + markers) must
stay below the bytes the epoch saves, pieces per record are bounded by that
record's own line count, markers by a source-derived ceiling, and every free
slot must be source-derived. These are all tokenizer-free and linear.

WHAT THIS IS NOT. It does not choose spans, drop reads or write epochs: it
verifies only documented transforms, and refuses anything it cannot establish.
The caller then sends unreduced history — never a weaker check.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# ONE character table, applied to BOTH sides.
#
# The client compares source text against response text, so any asymmetry
# between "what the reducer strips" and "what the verifier strips" is a false
# rejection. Round 2 kept its own narrower set and refused every emoji carrying
# VS16 (S-12). This table IS the reference `reduce_v2` set — the explicit
# Unicode-15 Default_Ignorable ranges — and both sides go through
# :func:`normalise`, so a difference cannot exist.
# ---------------------------------------------------------------------------
_IGNORABLE = re.compile(
    "[\u00ad\u034f\u061c\u115f-\u1160\u17b4-\u17b5\u180b-\u180f\u200b-\u200f"
    "\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff0-\ufff8"
    "\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e0fff]"
)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
#: The reference's `_CONTROL_RE`, unchanged: §3.3 never names a C0 character
#: other than the rule-2 line breakers, and refusing one the reducer keeps
#: (`\\x1f`, a `git log --format=%x1f` separator) invented contract in the
#: unsafe-for-honest-output direction (round 2, R2-3/S-12).
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1b\x7f]")
#: Any spelling a model reads as the marker (rule 3). Applied to BOTH sides, so
#: the reducer's `[radient(quoted):` rewrite of source text and the source's own
#: spelling normalise to the same literal.
_MARKER = re.compile(r"\[\s*radient\s*:", re.I)
_QUOTED = re.compile(r"\[radient\(quoted\):")


def normalise(text: str) -> str:
    """The one normalisation both sides use before any comparison."""
    text = _ANSI.sub("", text)
    text = _IGNORABLE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = _CONTROL.sub("", text)
    text = _MARKER.sub("[radient:", text)
    return _QUOTED.sub("[radient:", text)


# ---------------------------------------------------------------------------
# The reducer's documented STRUCTURAL elements, each named with its source.
# A run may be separated from the next by one of these and nothing else.
# ---------------------------------------------------------------------------
#: `_size()` output used by the reducer's notes and elision markers.
_SIZE = r"\d+(?:\.\d)?k? (?:B|kB|MB|chars)"
#: Non-marker notes the reducer writes into a record body:
#: - `cut + " [+{size} heredoc|more]"`      (reduce_v2._call_summary, line 536)
#: - `"[{size}]"` for a long argument value (line 548)
#: - `"[{type} {size}]"` for a long list/dict argument (line 551)
#: - `"[redacted]"` for a secret-keyed argument (§3.3 rule 6, line 522)
_NOTE = re.compile(
    r"\[\+(?P<more>" + _SIZE + r") (?:heredoc|more)\]"
    r"|\[[A-Za-z]{1,12} (?P<sized>" + _SIZE + r")\]"
    r"|\[(?P<plain>" + _SIZE + r")\]"
    r"|\[redacted\]"
)
#: The result-merged-onto-call separator (reduce_v2.result, line 598).
_ARROW = "\u2192 "


@dataclass(frozen=True)
class _Source:
    """One source record: its normalised text and what its own bytes bound."""

    text: str
    identity: str
    lines: int
    chars: int
    merged_with: str = ""

    @property
    def block_types(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class _Rec(_Source):
    """A source record plus the role-specific affordances the epoch may use."""

    name: str = ""
    plain_forms: int = 0
    block_types: frozenset[str] = frozenset()


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [
        block["text"]
        for block in content or []
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]


def _count_lines(text: str) -> int:
    return sum(1 for line in text.split("\n") if line.strip())


def _block_types(content: Any) -> frozenset[str]:
    if not isinstance(content, list):
        return frozenset()
    return frozenset(
        str(block.get("type")) for block in content if isinstance(block, dict) and block.get("type")
    )


def _joined(texts: Sequence[str]) -> str:
    """One row's text as the reducer renders it: blank lines dropped, joined."""
    return "\n".join(line for text in texts for line in text.split("\n") if line)


def source_records(
    span: Sequence[dict[str, Any]],
) -> tuple[dict[str, list[_Rec]], list[str], set[str]]:
    """The span's records per role, in source order, plus its identifier material.

    One candidate per SOURCE BLOCK, not per row: an assistant row with two text
    blocks becomes two `assistant:` records, and a user row's blocks each become
    their own record in block order (round 2, R2-1).
    """
    from local_operator.saver.canonical import canonical_bytes

    records: dict[str, list[_Rec]] = defaultdict(list)
    parts: list[str] = []
    calls: dict[str, int] = {}
    for row_number, row in enumerate(span):
        identity = f"row:{row_number}"
        role = row.get("role")
        types = _block_types(row.get("content"))
        texts = _texts(row.get("content"))
        parts.extend(texts)
        who = row.get("name")
        prefix = ""
        if isinstance(who, str) and who:
            prefix = "(" + re.sub(r"[^A-Za-z0-9_.:-]", "_", who)[:64] + ") "
        normalised = [normalise(prefix + text) for text in texts]

        def add(
            key: str, text: str, *, name: str = "", kinds: frozenset[str] = frozenset()
        ) -> None:
            records[key].append(
                _Rec(
                    text=text,
                    identity=f"{identity}:{len(records[key])}",
                    lines=_count_lines(text),
                    chars=len(text),
                    name=name,
                    block_types=kinds,
                )
            )

        if role == "assistant":
            for text in normalised:
                if text.strip():
                    add("assistant", text.strip(), kinds=types)
        elif role == "tool":
            text = _joined(normalised)
            status = _EXIT.search(text)
            failed = bool(row.get("is_error")) or bool(status and status[1] != "0")
            lines = [line for line in text.split("\n") if line.strip() not in _SCAFFOLD]
            if failed:
                lines = [line for line in lines if not _EXIT.fullmatch(line)]
            body = "\n".join(lines).strip("\n")
            status_prefix = f"exit {status[1]} " if status and status[1] != "0" else ""
            # The reducer's own rendering is `exit N <body>` with the scaffold
            # lines dropped, which is neither the raw text nor a prefix of it.
            text = text + "\n" + status_prefix + (body or "(no output)")
            rec = _Rec(
                text=text,
                identity=identity,
                lines=_count_lines(text),
                chars=len(text),
                block_types=types,
            )
            records["error" if failed else "result"].append(rec)
            call_position = calls.get(str(row.get("tool_call_id", "")))
            if call_position is not None:
                # A result may be merged onto its call line (`→ body`) or kept
                # as tab-quoted continuation lines. Both are the call's text
                # followed by the result's text, so one candidate covers them.
                call = records["call"][call_position]
                records["call"][call_position] = _Rec(
                    text=call.text + "\n" + text,
                    identity=call.identity,
                    lines=call.lines + rec.lines,
                    chars=call.chars + rec.chars,
                    name=call.name,
                    plain_forms=call.plain_forms,
                    block_types=call.block_types,
                    merged_with=identity,
                )
        else:
            content = row.get("content")
            blocks = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else list(content or [])
            )
            text_index = 0
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if isinstance(block.get("text"), str):
                    text = normalised[text_index] if text_index < len(normalised) else ""
                    text_index += 1
                    if text.strip():
                        add("user", text, kinds=frozenset({str(block.get("type"))}))
                elif block.get("type") == "image_ref":
                    # A media block travels as a fixed receipt; the slot text
                    # is validated against the row's own media digest instead.
                    add("user", "", kinds=frozenset({"image_ref"}))
        for call in row.get("tool_calls") or []:
            function = call.get("function") or {}
            name = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(function.get("name") or "tool"))[:64]
            raw = function.get("arguments", "")
            if not isinstance(raw, str):
                raw = canonical_bytes(raw).decode()
            parts.append(name)
            parts.append(raw)
            # The tool NAME is source too: a grouped read-only summary names
            # the command (`[radient:1 read-only commands, all ok: read]`).
            text = normalise(name) + "\n" + normalise(raw)
            try:
                args = json.loads(raw)
            except (ValueError, TypeError):
                args = None
            if isinstance(args, dict):
                # The reducer renders arguments as canonical JSON with secret
                # keys redacted; the canonical form is the source too, so a
                # dropped `i` key is just deleted bytes.
                text = text + "\n" + normalise(canonical_bytes(args).decode())
                command = args.get("command")
                if name in _SHELL_NAMES and isinstance(command, str):
                    cleaned = normalise(command)
                    text = text + "\n" + cleaned
                    cd = _CD.match(cleaned)
                    if cd:
                        # `cd DIR &&` is stripped and may be shown as `(in DIR)`.
                        tail = cleaned[cd.end() :]
                        # The `(in DIR)` rendering replaces the `cd`, and any
                        # further command lines follow it, so the command must
                        # remain matchable after that line as well.
                        text = text + "\n" + tail + "\n(in " + cd[1] + ") " + tail + "\n" + cleaned
            calls[str(call.get("id", ""))] = len(records["call"])
            records["call"].append(
                _Rec(
                    text=text,
                    identity=f"call:{row_number}:{len(records['call'])}",
                    lines=_count_lines(text),
                    chars=len(text),
                    name=name,
                )
            )
    return dict(records), parts, _media_digests(span)


_SHELL_NAMES = frozenset({"bash", "shell", "exec", "run", "terminal"})
_CD = re.compile(r"^\s*cd\s+((?:\"[^\"]+\"|'[^']+'|[^\s;&|]+))\s*(?:&&|;)\s*")
_EXIT = re.compile(r"^exit code: (-?\d+)$", re.M)
_SCAFFOLD = frozenset({"exit code: 0", "--- stdout ---", "--- stderr ---", "(empty)"})


def _media_digests(span: Sequence[dict[str, Any]]) -> set[str]:
    """Every 12-hex digest the reducer may derive from this span's own bytes."""
    out: set[str] = set()
    for row in span:
        for blob in (json.dumps(row, ensure_ascii=False, sort_keys=True),):
            out.add(hashlib.sha256(blob.encode()).hexdigest()[:12])
    return out


class _Pool:
    """The identifier pool: tokens the reducer may emit once, from the source.

    A pooled identifier is extracted from a SLICE of the source (`fresh_ids` on
    `first[len(cut) - 1:]`), so it can be a suffix of a real identifier; a
    reversed sort turns that into a prefix lookup. Exact membership is the
    common case. Nothing else is admissible — a pool token must come from an
    identifier extractor, never from arbitrary source words.
    """

    def __init__(self, parts: Sequence[str]) -> None:
        from local_operator.saver.provenance_ids import identifiers
        from local_operator.saver.validate import _b64_note_digests, _chunks

        tokens: set[str] = set()
        for part in parts:
            # Bounded by the span itself: every identifier is a substring of
            # it, so the pool can never exceed the span's own bytes.
            for chunk in _chunks(part):
                tokens.update(identifiers(chunk))
            tokens.update(_b64_note_digests(part))
        self.tokens = frozenset(tokens)

        self.reversed = sorted(token[::-1] for token in tokens)
        self.sorted = sorted(tokens)
        # Two identifiers per pooled token plus a fixed allowance: a set is
        # emitted once, and repeated listings are the reducer's own overlap.
        self.budget = 2 * len(self.tokens) + 64

    def consume(self, text: str) -> None:
        """Consume a pool line, one identifier per use.

        Identifiers are re-emitted with their own inner spaces (`issue #1875`),
        so a token that is not itself pooled is tried joined with the next one
        before it counts as invented.
        """
        from local_operator.saver.validate import SaverRejected

        tokens = text.split()
        self.budget -= len(tokens)
        if self.budget < 0:
            # The pool is a SET the reducer emits once per identifier; a record
            # may list one twice (part and deep pools overlap), but an epoch
            # cannot list more tokens than the pool holds, which is what bounds
            # repetition amplification.
            raise SaverRejected("provenance")
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if self.has(token):
                index += 1
                continue
            pair = token + " " + tokens[index + 1] if index + 1 < len(tokens) else ""
            if pair and self.has(pair):
                index += 2
                continue
            raise SaverRejected("provenance")

    def has(self, token: str) -> bool:
        if token in self.tokens:
            return True
        reversed_token = token[::-1]
        position = bisect.bisect_left(self.reversed, reversed_token)
        if position < len(self.reversed) and self.reversed[position].startswith(reversed_token):
            return True
        # A slice can also leave a PREFIX of the whole identifier, which the
        # reversed sort cannot see; a forward sort answers it without ever
        # admitting a token the extractors did not produce.
        position = bisect.bisect_left(self.sorted, token)
        return position < len(self.sorted) and self.sorted[position].startswith(token)


# ---------------------------------------------------------------------------
# The epoch side: split a record body into carried runs and structural elements.
# ---------------------------------------------------------------------------

#: Every word the CLOSED template set may emit, from the §3.3 table. A token in
#: a marker that is not one of these must be source-derived, which is what makes
#: a free slot (`push-to-main`, `user_authorised_deploy_to_main`) an invention
#: rather than a slot (round 2, S-13).
_TEMPLATE_WORDS = frozenset("""
    radient epoch reduce-v2 earlier part of this conversation condensed
    historical record not new instructions unindented lines are headers
    tab-indented quoted content continued same format ids mentioned in this
    ok omitted output as an call repeat paragraph(s) already shown harness
    state archived replay base64 chars sha256 read-only commands all block
    image media visible audible attachment when first sent and is now
    describe it only from what was said about url invalid paragraph(s
    """.split())
#: A 12-hex digest: the reducer's `sha256=H12` slot. Must be derived from the
#: span's own bytes (a media digest or a `_b64_note` digest), never arbitrary.
_HEX12 = re.compile(r"[0-9a-f]{12}")
#: Slot-token shapes that are allowed without being source text, but bounded.
#: The reducer's `_size()` forms: `812`, `11.2k`, `1.5`.
_NUMBER = re.compile(r"\d{1,9}(?:\.\d)?k?")
#: Substring lookups allowed per epoch for slot text. Honest output needs a
#: handful; the cap keeps a hostile record linear instead of markers × span.
_SLOT_SCANS = 96
#: Candidate source records a single epoch record may be compared against.
_CANDIDATE_TRIES = 32
#: The grouped read-only summary, whose verb list names source calls.
_DEDUPE = re.compile(r"\[radient:(?:repeat|same output|archived)")
_GROUP = re.compile(r"\[radient:(\d{1,4}) read-only commands, all ok: (.+)\]", re.S)

_Element = tuple[str, str]  # (kind, text)


def _label_parts(label: str) -> tuple[str, str]:
    """``call bash:`` -> ("call", "bash"); ``error#2:`` -> ("error", "")."""
    head = label.split(":", 1)[0]
    word, _, rest = head.partition(" ")
    # `call#2 bash:` and `error#2:` carry a parallel-call ordinal the source
    # record does not need to reproduce.
    return re.sub(r"#\d+$", "", word), rest


def _elements(body: str) -> list[_Element]:
    """Every structural element and carried run of one body, in order."""
    from local_operator.saver.validate import MARKER_TEMPLATES, SaverRejected

    found: list[tuple[int, int, str]] = []
    position = 0
    while True:
        start = body.find("[radient:", position)
        if start < 0:
            break
        match = None
        for pattern in MARKER_TEMPLATES.values():
            match = pattern.match(body, start)
            if match is not None:
                break
        if match is None:
            # check_markers refuses this too; refusing here keeps one grammar.
            raise SaverRejected("foreign_marker")
        found.append((start, match.end(), "marker"))
        position = match.end()
    for match in _NOTE.finditer(body):
        found.append((match.start(), match.end(), "note"))
    for index in [m.start() for m in re.finditer(re.escape(_ARROW), body)]:
        found.append((index, index + len(_ARROW), "arrow"))
    for index in [m.start() for m in re.finditer("\u2026", body)]:
        found.append((index, index + 1, "ellipsis"))
    found.sort()
    elements: list[_Element] = []
    cursor = 0
    for start, end, kind in found:
        if start < cursor:
            continue
        if start > cursor:
            # Runs are compared in the ONE normalisation, so a quoted marker
            # spelling the source carried matches the source's own spelling.
            elements.append(("run", normalise(body[cursor:start])))
        elements.append((kind, body[start:end]))
        cursor = end
    if cursor < len(body):
        elements.append(("run", normalise(body[cursor:])))
    return [(kind, text.strip()) for kind, text in elements if kind != "run" or text.strip()]


def _slot_tokens(text: str) -> list[str]:
    """Identifier-shaped tokens of a marker, minus the template's fixed words."""
    out: list[str] = []
    # The marker word itself is grammar, not a slot.
    body = text.replace("[radient(quoted):", " ").replace("[radient:", " ")
    for raw in body.replace("[", " ").replace("]", " ").split():
        token = raw.strip("(),;.:—-×\"'")
        if not token or token in _TEMPLATE_WORDS:
            continue
        for part in token.split("\u00d7"):
            part = part.strip("(),;.:—-")
            if part and part not in _TEMPLATE_WORDS:
                out.append(part)
    return out


def _derived_digests(parts: Sequence[str]) -> set[str]:
    from local_operator.saver.validate import _b64_note_digests

    out: set[str] = set()
    for part in parts:
        out.update(_b64_note_digests(part))
    return out


def _check_slots(
    elements: Sequence[_Element],
    record: _Rec,
    pool: _Pool,
    media: set[str],
    derived: set[str],
    scans: list[int],
    span_bound: int,
) -> None:
    """Every free slot must be source-derived, and bounded by the source."""
    from local_operator.saver.validate import SaverRejected

    bound = max(record.chars, record.lines * 4, 8)
    for kind, text in elements:
        if kind != "marker":
            continue
        if text.startswith("[radient:ids"):
            continue
        # A dedupe marker sizes the EARLIER occurrence it stands for, which this
        # record cannot bound; the span can, and no honest marker can claim more
        # text than the span holds.
        ceiling = span_bound if _DEDUPE.match(text) else bound
        for token in _slot_tokens(text):
            if _NUMBER.fullmatch(token):
                # `11.2k` is a size; compare its magnitude, never the literal.
                # `_size()` rounds to one decimal, so a size may exceed the
                # record it describes by the rounding step; a count may not.
                if (
                    float(token.rstrip("k")) * (1000 if token.endswith("k") else 1)
                    > ceiling * 1.05 + 64
                ):
                    raise SaverRejected("provenance")
                continue
            if _HEX12.fullmatch(token):
                if token in media or token in derived or token in record.text:
                    continue
                raise SaverRejected("provenance")
            if token in record.block_types:
                continue
            if scans[0] <= 0:
                raise SaverRejected("busy")
            scans[0] -= 1
            if token not in record.text:
                raise SaverRejected("provenance")


def _pieces(elements: Sequence[_Element]) -> list[str]:
    """The carried LINES of an epoch record, in order.

    Line granularity, not run granularity: the reducer deletes lines in place
    (blank-line collapsing, an archive head kept while its body goes, `_short`
    on the first line, `##` headings re-joined), so a retained run can be a
    subsequence of lines rather than one contiguous substring. Every piece is
    still a verbatim source line, and the per-record piece bound is what keeps
    that from becoming a fragmentation primitive (round 2, S-13).
    """
    pieces: list[str] = []
    for kind, text in elements:
        if kind != "run":
            continue
        for line in text.split("\n"):
            for piece in _JSON_STRUCT.split(line):
                piece = piece.strip()
                # Punctuation-only pieces carry no content: a JSON summary may
                # drop whole keys, so `"}` need not exist in the source.
                if piece and not _PUNCT_ONLY.fullmatch(piece):
                    pieces.append(piece)
    return pieces


#: JSON/array punctuation is the reducer's own joining grammar, not content.
_JSON_STRUCT = re.compile(r"[{}\[\],\"]")
_PUNCT_ONLY = re.compile(r"[^\w\u0080-\uffff]+")


def _runs_match(pieces: Sequence[str], record: _Rec) -> bool:
    """Each carried line is a source line, in order (§6.1(6) 'only delete').

    A piece may be several `## ` headings juxtaposed: the harness-state
    rendering joins the source's separate heading lines, which is documented
    output, so the join points are tried as segment boundaries rather than
    being treated as invented text.
    """
    expanded: list[str] = []
    for piece in pieces:
        expanded.extend(part.strip() for part in piece.split(" ## ") if part.strip())
    pieces = expanded
    cursor = 0
    for piece in pieces:
        position = record.text.find(piece, cursor)
        if position < 0:
            return False
        cursor = position + len(piece)
    return True


def _check_bounds(elements: Sequence[_Element], record: _Rec) -> tuple[int, int]:
    """The tokenizer-free amplification bounds. Returns (pieces, overhead)."""
    from local_operator.saver.validate import SaverRejected

    pieces = _pieces(elements)
    markers = [text for kind, text in elements if kind == "marker"]
    overhead = sum(len(text) for kind, text in elements if kind != "run")
    # An elision keeps SELECTED SOURCE LINES, so it can never emit more pieces
    # than the record has lines (plus the record's own head). Separators then
    # cost the source no more than it already had.
    # One carried piece per 8 source bytes: honest output splits at paragraphs
    # and truncation marks (~89-byte lines, ~500-byte paragraphs), while
    # one-character fragmentation needs one piece per byte and cannot fit.
    if len(pieces) > record.chars // 8 + 8:
        raise SaverRejected("provenance")
    # A record that carries no text is one elision (or one elision plus the
    # identifier pool). Thousands of markers in one record is fragmentation.
    if not pieces and len(markers) > 2:
        raise SaverRejected("provenance")
    if len(markers) > min(len(record.text) // 64 + 2, _marker_ceiling(record)):
        raise SaverRejected("provenance")
    # `[radient:ok N lines]` / `omitted N lines` must agree with the source.
    for text in markers:
        claimed = re.search(r"(?:ok|omitted) (\d{1,9}) lines?", text)
        if claimed is not None and int(claimed.group(1)) > record.lines + 1:
            raise SaverRejected("provenance")
    return len(pieces), overhead


def _marker_ceiling(record: _Rec) -> int:
    """How many markers one source record can legitimately produce.

    One elision, one identifier pool, plus one per source BLOCK (B1/M1/M2 per
    media block). Small and source-derived, so it cannot be inflated.
    """
    return 3 + len(record.block_types) + record.text.count("[radient:")


def check_records(
    text: str,
    span: Sequence[dict[str, Any]],
    *,
    deadline: float | None = None,
    span_bytes: int | None = None,
    epoch_bytes: int | None = None,
) -> None:
    """§6.1(6) over one epoch: source-derived records with bounded structure.

    Linear: each source record is inspected at most once per role stream, runs
    are matched with one forward `find`, and the slot lookups are capped. The
    aggregate rule below is the mechanism that makes token inflation
    impossible without a tokenizer: the bytes spent on separators and markers
    must be fewer than the bytes the epoch claims to save.
    """
    from local_operator.saver.canonical import canonical_bytes
    from local_operator.saver.validate import SaverRejected, parse_records

    records_by_role, parts, media = source_records(span)
    pool = _Pool(parts)
    derived = _derived_digests(parts)
    cursors: dict[str, int] = defaultdict(int)
    consumed: set[str] = set()
    scans = [_SLOT_SCANS]
    span_chars = sum(record.chars for group in records_by_role.values() for record in group)
    overhead_total = 0
    pieces_total = 0
    for label, body in parse_records(text):
        if deadline is not None and time.monotonic() > deadline:
            raise SaverRejected("busy")
        kind, name = _label_parts(label)
        # ONE normalisation, the same one the source went through: the quoted
        # marker spelling, ignorable characters and ANSI are then identical on
        # both sides and cannot cause a false mismatch (round 2, S-12).
        elements = _elements(body)
        # The identifier pool may be a record of its own or appended to the
        # record whose text it indexes; everything after its marker is a pooled
        # token, and is checked by the pool rather than as carried content.
        pool_at = next(
            (
                index
                for index, (what, text) in enumerate(elements)
                if what == "marker" and text.startswith("[radient:ids")
            ),
            None,
        )
        if pool_at is not None:
            pool.consume(normalise(" ".join(text for _, text in elements[pool_at + 1 :])))
            elements = elements[:pool_at]
            if not elements:
                continue
        if kind == "call" and not _pieces(elements) and len(elements) == 1:
            verbs = _group_verbs(elements[0][1], scans)
            if verbs is not None:
                # One `[radient:N read-only commands, all ok: …]` record stands
                # for several source calls: consume one candidate per verb use,
                # in source order, so the group can never invent a call.
                _consume_group(records_by_role.get("call", []), cursors, consumed, verbs)
                continue
        candidates = records_by_role.get(kind, [])
        # Records are consumed once each, but the epoch may carry a dedupe
        # record for an earlier row after later ones, so the scan wraps once
        # instead of assuming source order. Consumption, not order, is what
        # bounds reuse.
        start = cursors[kind]
        order = list(range(start, len(candidates))) + list(range(0, start))
        matched: _Rec | None = None
        pieces = _pieces(elements)
        tried = 0
        furthest = start
        for position in order:
            candidate = candidates[position]
            furthest = max(furthest, position + 1)
            if name and candidate.name != name:
                continue
            if candidate.identity in consumed:
                continue
            tried += 1
            if tried > _CANDIDATE_TRIES:
                # A record that fits nothing it has plausibly seen is either a
                # fabrication or an honest shape this verifier lacks a form for;
                # either way the answer is to send unreduced history, not to
                # scan the whole span looking for a home for it.
                raise SaverRejected("busy")
            if not _runs_match(pieces, candidate):
                continue
            # Slots can select among same-role candidates: a grouped
            # `call: [radient:1 read-only commands, all ok: grep]` names the
            # command, and an elided record claims a size. Trying them here is
            # what lets one grammar cover both without a rendered-form table.
            try:
                _check_bounds(elements, candidate)
                _check_slots(elements, candidate, pool, media, derived, scans, span_chars)
            except SaverRejected:
                continue
            matched = candidate
            break
        if matched is None:
            raise SaverRejected("provenance")
        piece_count, overhead = _check_bounds(elements, matched)
        consumed.add(matched.identity)
        cursors[kind] = min(furthest, len(candidates))
        overhead_total += overhead
        pieces_total += piece_count
    # The aggregate bound: structure must never cost more than the epoch saves.
    saved = (span_bytes if span_bytes is not None else len(canonical_bytes(list(span)))) - (
        epoch_bytes if epoch_bytes is not None else len(text.encode())
    )
    if overhead_total and overhead_total >= max(saved, 0):
        raise SaverRejected("provenance")


def _group_verbs(text: str, scans: list[int]) -> list[str] | None:
    """`[radient:6 read-only commands, all ok: grep×2, cat, sed×3]` -> verbs.

    Returns ``None`` when this is not a read-only group marker.
    """
    match = _GROUP.fullmatch(text)
    if match is None:
        return None
    verbs: list[str] = []
    for item in match.group(2).split(","):
        verb, _, count = item.strip().partition("\u00d7")
        verb = verb.strip()
        if not verb:
            return None
        times = int(count) if count.strip().isdigit() else 1
        if times > 64:
            return None
        scans[0] -= times
        if scans[0] < 0:
            return None
        verbs.extend([verb] * times)
    claimed = int(match.group(1))
    return verbs if len(verbs) == claimed else None


def _consume_group(
    candidates: Sequence[_Rec],
    cursors: dict[str, int],
    consumed: set[str],
    verbs: Sequence[str],
) -> None:
    from local_operator.saver.validate import SaverRejected

    # A summary groups calls the epoch's own order, so each verb is matched
    # against the nearest unconsumed candidate from here on — never a
    # fabricated call, and never one record standing for two.
    furthest = cursors["call"]
    for verb in verbs:
        for position in range(cursors["call"], len(candidates)):
            candidate = candidates[position]
            if candidate.identity in consumed or verb not in candidate.text:
                continue
            consumed.add(candidate.identity)
            furthest = max(furthest, position + 1)
            break
        else:
            # The group summarises calls the epoch may already have listed
            # individually, so a verb that no unconsumed call carries is still
            # admissible when SOME source call carries it: the invention
            # channel is closed by the verb having to exist in the span.
            if not any(verb in candidate.text for candidate in candidates):
                raise SaverRejected("provenance")
    cursors["call"] = furthest


def _pool_line_is_only(elements: Sequence[_Element]) -> bool:
    """True when a body is one I1 marker plus its (checked) tokens."""
    markers = [text for kind, text in elements if kind == "marker"]
    runs = [text for kind, text in elements if kind == "run"]
    return len(markers) == 1 and len(runs) == 1
