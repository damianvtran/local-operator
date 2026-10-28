"""Source-provenance verifier for §6.1(6) — THE AUTO GATE, AND ITS INVARIANT.

THE INVARIANT THIS MODULE ENFORCES. An epoch is accepted only if **every
emitted content line maps injectively and in order to a distinct line of its
source span, and the emitted content of each line is a contiguous whole,
prefix, or suffix of that source line**. Separately, **the epoch's total line
count must be strictly less than the span's total line count**.

WHAT THAT BUYS, BY CONSTRUCTION RATHER THAN BY MORE CAPS.
- No free-form text as content: a line that is not a source line, or a prefix
  or suffix of one, is refused. Summarisation prose, invented note payloads and
  marker-slot text all die here, because none of them is a piece of a source
  line.
- No re-segmentation: one source line maps to AT MOST ONE emitted line, so a
  record cannot be shredded into one emitted line per character, and the `\n\t`
  separator cost can never exceed what was deleted (round 3, S-15/R3-2).
- Token boundedness WITHOUT a tokenizer: emitted content is a subset of the
  span's own lines, fragments of them, each used once. Tokens(epoch content)
  <= tokens(span content) follows from that injectivity, not from a byte
  budget — which is why this check no longer charges separators in bytes. A
  byte budget cannot bound a token cost: span bytes-per-token is unbounded
  above (a repeated-character span tokenizes at ~10 B/token), and round 3's
  18.35x and 2.65x counter-examples were both reached under one.
- Total lines strictly fewer: disclosure lines (markers, the reducer's own
  notes) are EXTRA LINES, not content, so each one has to be paid for by real
  deletions elsewhere in the span. The line count is the guarantee.

WHAT IT DOES NOT DO. It does not choose spans, drop reads or write epochs: it
verifies only documented transforms and refuses anything it cannot establish.
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
#: `[redacted]` is deliberately NOT a note: the projection redacted the row
#: before upload, so the literal is CONTENT the source really contains.
_NOTE = re.compile(
    r"\[\+(?P<more>" + _SIZE + r") (?:heredoc|more)\]"
    r"|\[[A-Za-z]{1,12} (?P<sized>" + _SIZE + r")\]"
    r"|\[(?P<plain>" + _SIZE + r")\]"
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
    #: The record's canonical SEGMENTS: its lines, each split at the punctuation
    #: the reducer paraphrases around. Derived from ``text`` alone, so every
    #: segment is a piece of the row's own bytes and nothing is invented.
    segments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.segments and self.text:
            object.__setattr__(self, "segments", _segments(self.text))


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
#: The reducer's fixed note vocabulary (`[+5.8k chars heredoc]`, `[1.4k chars]`,
#: `[dict 300 chars]`, `[redacted]`) — everything else in a note is a slot.
_NOTE_WORDS = frozenset({"chars", "heredoc", "more", "redacted", "list"})
_SIZE_TOKEN = re.compile(r"\d+(?:\.\d)?k?")
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


def _note_tokens(text: str) -> list[str]:
    """The name slots of a reducer note, minus its fixed words and sizes."""
    out: list[str] = []
    for raw in text.replace("[", " ").replace("]", " ").split():
        token = raw.strip("()+\u2014\u2026,;.:")
        if not token or token in _NOTE_WORDS or _SIZE_TOKEN.fullmatch(token):
            continue
        out.append(token)
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
        if kind == "note":
            # A note is a disclosure line, never content: its number must be a
            # count and its name a token of the source. `[1.4k chars]` passes;
            # `[approveall 999 chars]` is an invention and does not (round 3,
            # R3-1 raised exactly this channel).
            for token in _note_tokens(text):
                if _SIZE_TOKEN.fullmatch(token):
                    continue
                if token not in record.text:
                    raise SaverRejected("provenance")
            continue
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


def _split_line(line: str) -> list[str]:
    """One line's content pieces: whole, or split at JSON punctuation.

    Only an OBJECT-shaped line splits. An argument summary is a JSON object; a
    line that merely starts with `[` is content (`[redacted]`), and splitting it
    would stop an honest record from being a prefix of its own line.
    """
    # A JSON-ish line is one carrying a key/value separator; argument summaries
    # are paraphrased around exactly that punctuation. A line that merely starts
    # with `[` is content (`[redacted]`), and splitting it would stop an honest
    # record being a contiguous piece of its own line.
    pieces = _JSON_STRUCT.split(line) if '":"' in line or line.lstrip().startswith("{") else [line]
    out: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        # Punctuation-only pieces carry no content: a JSON summary may drop
        # whole keys, so `"}` need not exist in the source.
        if piece and not _PUNCT_ONLY.fullmatch(piece):
            out.append(piece)
    return out


def _segments(text: str) -> tuple[str, ...]:
    """The text's canonical segments: its lines, split at paraphrase punctuation.

    Splitting the SOURCE at the same characters the epoch side splits at is what
    keeps an honest argument summary (a canonical JSON line with keys dropped)
    compatible while still making it out of nothing but the row's own tokens.
    """
    out: list[str] = []
    for line in text.split("\n"):
        out.extend(_split_line(line))
    return tuple(out)


def _pieces(elements: Sequence[_Element]) -> list[str]:
    """The carried fragments of an epoch record, in order.

    Fragments, not lines: a line may carry a marker inline. Every fragment must
    then be a whole, prefix or suffix of ONE source segment, and no segment may
    serve twice — that is the invariant, and :func:`_map_fragments` enforces it.
    """
    pieces: list[str] = []
    for kind, text in elements:
        if kind != "run":
            continue
        for line in text.split("\n"):
            # The harness-state rendering joins the source's separate `## `
            # heading lines onto one epoch line; each heading is still a source
            # line, so split them back out before mapping.
            for part in line.split(" ## ") if " ## " in line else [line]:
                pieces.extend(_split_line(part))
    return pieces


_JSON_STRUCT = re.compile(r"[{}\[\],\"]")
_PUNCT_ONLY = re.compile(r"[^\w\u0080-\uffff]+")


#: JSON/array punctuation is the reducer's own joining grammar, not content.
_JSON_STRUCT = re.compile(r"[{}\[\],\"]")
_PUNCT_ONLY = re.compile(r"[^\w\u0080-\uffff]+")
#: How far ahead of the last consumed segment a non-exact fragment may match.
#: Exact fragments (the overwhelming majority) are a dict lookup, so the forward
#: window only pays for prefixes and suffixes, and it keeps the whole mapping
#: linear in the epoch's size.
#: Slack over a source record's own line count: a record may gain a head line
#: and a merged result line. Measured on the 20-snapshot real-session replay:
#: honest records sit at p99 +3, with one JSON-bearing outlier.
_PIECE_SLACK = 8


def _map_fragments(pieces: Sequence[str], record: _Rec) -> bool:
    """The invariant's first half: in-order, non-overlapping, contiguous pieces.

    Each emitted fragment must be a contiguous piece of this record's text and
    the fragments must advance monotonically, so the emitted content is what is
    left after deleting: the reducer may not reorder, may not invent, and may
    not compose text across source records. The second half of the invariant —
    at most one emitted line per source line — is enforced by the piece bound in
    :func:`_check_bounds`, which is what actually stops re-segmentation
    (round 3: S-15 shredded one 27 KB line into 3,420 emitted lines).
    """
    cursor = 0
    for fragment in pieces:
        found = record.text.find(fragment, cursor)
        if found < 0:
            return False
        cursor = found + len(fragment)
    return True


def _check_bounds(elements: Sequence[_Element], record: _Rec) -> tuple[int, int]:
    """The disclosure and re-segmentation budget, both source-derived.

    NO BYTE BUDGET. Round 3 shipped ``chars // 8 + 8`` and an aggregate
    overhead-vs-saved rule; security measured 2.65x tokens under them on plain
    English prose, because span bytes-per-token is unbounded above. The bound
    that carries the guarantee is the LINE COUNT: an emitted record may not
    carry more content lines than the source record it came from has lines (plus
    a small fixed slack for a head and a merged result). One source line can
    therefore appear as at most one emitted line, which is the manager's
    "no re-segmentation" stated as a number the reducer's own output satisfies.
    """
    from local_operator.saver.validate import SaverRejected

    markers = [text for kind, text in elements if kind == "marker"]
    notes = [text for kind, text in elements if kind == "note"]
    overhead = sum(len(text) for kind, text in elements if kind != "run")
    pieces = _pieces(elements)
    if len(pieces) > record.lines + _PIECE_SLACK:
        raise SaverRejected("provenance")
    # A record that carries no text is one elision (or one elision plus the
    # identifier pool). Thousands of disclosure elements is fragmentation.
    if not pieces and len(markers) + len(notes) > 2:
        raise SaverRejected("provenance")
    if len(markers) + len(notes) > min(len(record.text) // 64 + 2, _marker_ceiling(record)):
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


def _check_line_budget(records: Sequence[tuple[str, str]], span_lines: int) -> None:
    """The invariant's second half: the epoch is strictly FEWER lines than the span.

    Disclosure lines count, so every marker and note is paid for by real
    deletions. This is the rule that makes note multiplication unprofitable
    (round 3, R3-1: 230 notes cost 18.35x the span's tokens).
    """
    from local_operator.saver.validate import SaverRejected

    emitted = sum(1 + body.count("\n") for _label, body in records)
    if emitted >= max(span_lines, 1):
        raise SaverRejected("provenance")


def check_records(
    text: str,
    span: Sequence[dict[str, Any]],
    *,
    deadline: float | None = None,
    span_bytes: int | None = None,
    epoch_bytes: int | None = None,
) -> None:
    """The invariant over one epoch: injective, in-order line mapping.

    The module docstring states the contract and why a byte budget was removed
    in its favour (round 3: S-15, R3-1, R3-2).
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
    records = parse_records(text)
    # THE SECOND HALF OF THE INVARIANT: the epoch must be strictly FEWER lines
    # than the span. Disclosure lines count, so every marker and note has to be
    # paid for by real deletions — that is their whole budget, and it is why
    # they cannot be multiplied into a token amplifier (round 3: S-15, R3-1).
    span_lines = sum(record.lines for group in records_by_role.values() for record in group)
    _check_line_budget(records, span_lines)
    for label, body in records:
        if deadline is not None and time.monotonic() > deadline:
            raise SaverRejected("busy")
        kind, name = _label_parts(label)
        elements = _elements(body)
        # The identifier pool may be a record of its own or appended to the
        # record whose text it indexes: everything after its marker is a pooled
        # token, checked by the pool rather than as carried content.
        pool_at = next(
            (
                index
                for index, (what, item) in enumerate(elements)
                if what == "marker" and item.startswith("[radient:ids")
            ),
            None,
        )
        if pool_at is not None:
            pool.consume(normalise(" ".join(item for _, item in elements[pool_at + 1 :])))
            elements = elements[:pool_at]
            if not elements:
                continue
        pieces = _pieces(elements)
        if kind == "call" and not pieces and len(elements) == 1:
            verbs = _group_verbs(elements[0][1], scans)
            if verbs is not None:
                _consume_group(records_by_role.get("call", []), {"call": 0}, consumed, verbs)
                continue
        candidates = records_by_role.get(kind, [])
        # Consumption, not source order, is what bounds reuse: the epoch may
        # carry a dedupe record for an earlier row after later ones, so one
        # wrap is allowed. Each source record is spent once.
        start = cursors[kind]
        order = list(range(start, len(candidates))) + list(range(0, start))
        matched: _Rec | None = None
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
                # either way the answer is to send unreduced history.
                raise SaverRejected("busy")
            if not _map_fragments(pieces, candidate):
                continue
            # Slots can select among same-role candidates: a grouped
            # `call: [radient:1 read-only commands, all ok: grep]` names the
            # command, and an elided record claims a size.
            try:
                _check_bounds(elements, candidate)
                _check_slots(elements, candidate, pool, media, derived, scans, span_chars)
            except SaverRejected:
                continue
            matched = candidate
            break
        if matched is None:
            raise SaverRejected("provenance")
        consumed.add(matched.identity)
        cursors[kind] = min(furthest, len(candidates))
    _ = (canonical_bytes, span_bytes, epoch_bytes)


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
