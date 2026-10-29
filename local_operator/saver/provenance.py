"""Source-provenance verifier for §6.1(6) — THE AUTO GATE, AND ITS INVARIANT.

THE INVARIANT THIS MODULE ENFORCES. An epoch body is a sequence of TYPED
UNITS, and each unit has its own rule:

- CONTENT is a span of source CHARACTERS that is in-order, non-overlapping and
  one emitted line per source line (a whole line, or any non-overlapping piece
  of at least ``MIN_CUT`` characters of it);
- POOL (I1) is a disclosure line whose tokens are identifiers extracted from
  the span — EXACTLY, never a fragment, prefix or slice — each spent once by
  identity, and charged in characters against what the epoch deleted;
- PREVIEW (O5) is a disclosure line that must BE the reducer's own preview of
  the record it replaces, so its length is the record's, not its own;
- MARKER/NOTE are the closed §3.3 template grammar.

Separately, **the epoch's CONTENT line count must be strictly less than the
span's total line count**: disclosure lines are excluded from that count
because the deletions that pay for them are the count itself.

WHAT THAT BUYS, BY CONSTRUCTION RATHER THAN BY MORE CAPS.
- No free-form text as content: a line that is not a source line, or a prefix
  or suffix of one, is refused. Summarisation prose, invented note payloads and
  marker-slot text all die here, because none of them is a piece of a source
  line.
- No re-segmentation: the mapping's cursor never revisits a source line, so one
  source line answers AT MOST ONE emitted line and a record cannot be shredded
  into one emitted line per character (round 3 S-15/R3-2, round 4 R4-1). Only a
  whole line, or a prefix/suffix of at least ``MIN_CUT`` characters, may match;
  shorter pieces are the re-segmentation primitive the reviewer measured at 9.2x.
- Disclosure is bounded by the material it discloses: a pool token is an exact
  span identifier spent once, and the pool's characters are charged against the
  epoch's deletions. (Round 5's security review measured the earlier looser
  rules: one count-budget unit bought up to 2,400 characters, and repeating one
  long identifier in pool records was accepted at 3.00x-7.43x the span.)
- NON-AMPLIFICATION, MEASURED: `tokens_R(epoch text) < tokens_R(span rendered
  text)`, both sides measured by the client's own estimator (`cl100k_base` via
  :mod:`local_operator.compaction.tokens`, the ruler the provider path already
  uses) on each side's own text — never on the canonical JSON. An epoch that is
  not cheaper is refused, and an estimator that has degraded to `chars // 4`
  refuses every epoch rather than approximating. This is the rule that bounds
  cost, and it has to be a comparison: no property of the mapping can supply
  it, because BPE is not
  monotone under substring — deleting two characters from
  `ABCDEFGHIJKLMNOPQRSTUVWXYZ` (1 token) leaves a 24-character interior worth
  8 — so every rule of the form "the epoch's characters are a subsequence of
  the span's" admits an 8x amplification, and the only sound tokenizer-free
  bound (chars(epoch) <= chars(span) / 128, the longest token in either bundled
  encoding) accepts nothing the reducer produces. The architecture decision is
  on PR #3078; S-18 is the finding, and a later round must not restate the
  withdrawn "by construction" theorem.
- Total lines strictly fewer: the epoch's total line count must be STRICTLY less
  than the span's, and disclosure lines (markers, the reducer's own notes) count.
  Each marker and note is therefore paid for by real deletions elsewhere in the
  span. Applied per RECORD instead, the same rule refuses every honest record
  that re-emits one source line unchanged: measured 30/131 against 66/131 on the
  real-session replay, which is why it is applied where the contract states it —
  over the span.

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
#: `[redacted]` IS a note: `_call_summary` writes it for any argument whose key
#: is in the reducer's `SENSITIVE_KEYS`, and the projection does not always get
#: there first — `secret_ref: LOP_SHOWCASE_FIXTURE_PW_R2` is a variable NAME to
#: the projection's redactor and a secret slot to the reducer, so the emitted
#: line carries `[redacted]` where the source carries the value. It stays safe
#: as a note because it holds no tokens to check: `_note_tokens` drops it.
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


def _short(text: str, limit: int) -> str:
    """reduce_v2._short, mirrored: a cut never ends inside a marker.

    The preview (O5) and every pooled identifier go through that function, so a
    verifier that needs to recognise them has to reproduce the cut exactly —
    including the trailing ellipsis and the marker guard, whose absence would
    refuse an honest preview whose 60th character fell inside ``[radient:``.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    index = head.rfind("[radient:")
    if index >= 0 and "]" not in head[index:]:
        head = head[:index]
    return head.rstrip() + "\u2026"


class _Pool:
    """The epoch's identifier pool (I1): a disclosure line, not content.

    A pooled token must be one of the span's own identifiers as
    :mod:`local_operator.saver.provenance_ids` extracts them — EXACTLY, not a
    fragment of one, not a prefix and not a slice — and each identifier may be
    spent ONCE, by identity. Round 5's security review measured what the looser
    rules bought: with prefix and slice admission one unit of the count budget
    purchased up to 2,400 characters, so repeating a single long identifier in
    pool records was ACCEPTED at 3.00x, 4.73x, 7.25x and 7.43x the span.

    Two bounds, and neither is a constant someone can raise:

    - IDENTITY. ``spent`` is a set of the identifiers used, so a repeat is a
      refusal rather than a second unit. With exact membership this alone
      bounds the pool by the span's own identifier material.
    - CHARACTERS. The pool is charged against what the epoch deleted
      (``char_budget``), the same way content is charged against the span. A
      count bound cannot supply this: a span may hold thousands of identifiers
      and listing the LONG ones is where a count leaks. A pool line can
      therefore never buy more characters than the deletions pay for.
    """

    def __init__(self, parts: Sequence[str]) -> None:
        from local_operator.saver.provenance_ids import identifiers
        from local_operator.saver.validate import _b64_note_digests, _chunks

        tokens: set[str] = set()
        for part in parts:
            # Bounded by the span itself: every identifier is a substring of
            # it, so the pool can never exceed the span's own text.
            for chunk in _chunks(part):
                tokens.update(identifiers(chunk))
            # `identifiers()` is the path/env vocabulary; the reducer also
            # pools camelCase names (`setDictationNotice`) and short ALL-CAPS
            # error codes, which are words of the span and no less source-
            # derived. Adding them widens WHICH source material a token may
            # come from, never how much of it can be spent: the payload cap,
            # the deleted-character budget and the token comparison still bound
            # the record.
            tokens.update(_WORD.findall(part))
            tokens.update(_b64_note_digests(part))
        self.tokens = frozenset(tokens)
        # Sorted once: `tail_owner` walks it to pick WHICH identifier a tail
        # spends, and two clients with the same span must pick the same one.
        self.sorted = sorted(tokens)
        #: The span's own text, for the last-resort containment test.
        self.text = "\n".join(parts)
        #: length -> {tail: identifier}, built on first use per length.
        self._tails: dict[int, dict[str, str]] = {}
        #: token -> identifier it was found inside, so a repeat costs a hit.
        self._contained: dict[str, str] = {}
        #: The containment lookup's budget: each miss costs one unit.
        self.scans = _POOL_SCANS

        self.chars = 0
        #: Set by :func:`check_records` once the epoch's content is known.
        self.char_budget = 0

    def owner_of(self, token: str) -> str | None:
        """An unspent span identifier this pool token came from, or ``None``.

        The reducer does not pool whole identifiers. Its `_call_summary` cuts
        long command lines at 160 characters and pools what is left, and its
        rendering elides the head of a path with `...`, so the honest payload
        holds tails (`/daemon.py`), elided tails (`.../i18n/catalogues.py`) and
        truncated words (`catalogu`). Refusing those refuses the reference
        reducer's own output, and one refused epoch voids a whole response
        (round-6 review, R6-1: 0 of 19 real responses adopted, so the product
        saved nothing at all).

        What keeps the round-5 blocker closed is not the SHAPE of the token but
        what it spends: the object recorded in ``spent`` is the identifier, so
        one identifier is usable once and can buy at most its own characters.
        The deleted `_slice_token` searched the SPAN TEXT, which is how one
        count-budget unit came to buy 2,400 characters of a path it never
        deleted.

        Bounded: exact membership is a set hit, the tail index costs one pass
        per distinct token length, and the containment fallback is charged
        against ``scans`` — no O(response x span) character scanning (S-1).
        Ties resolve in sorted order, so two clients with the same span spend
        the same identifier.
        """
        if not token:
            return None
        if token in self.tokens:
            return token
        # A leading `...` is the reducer's display elision, not identifier text.
        core = token.lstrip(".")
        if not core:
            return None
        owner = self._tail_owner(core)
        if owner is not None:
            return owner
        cached = self._contained.get(core)
        if cached is not None:
            return cached
        if self.scans <= 0:
            return None
        for identifier in self.sorted:
            if len(identifier) <= len(core):
                continue
            if core in identifier:
                self._contained[core] = identifier
                return identifier
        # THE SPAN'S OWN TEXT IS THE LAST RESORT, and check 6's actual bar: a
        # token is admissible when it is a substring of the span. The reducer
        # assembles some pool tokens from a line it truncated, so they are
        # source text without being a substring of any single extracted
        # identifier (`/en/relay.health.context.json`). What bounds the pool is
        # not where the characters came from — every one of them is the span's
        # — but how many are spent: the payload cap, the deleted-character
        # budget and the token comparison. Charged against `scans`, so this is
        # O(scans x span) per epoch and never O(response x span).
        if len(core) >= 6 and self.scans > 0:
            self.scans -= 1
            if core in self.text:
                self._contained[core] = core
                return core
        self.scans -= 1
        return None

    def _tail_owner(self, token: str) -> str | None:
        """The suffix index: one dict hit, built once per token length."""
        length = len(token)
        index = self._tails.get(length)
        if index is None:
            index = {}
            for identifier in self.sorted:
                if len(identifier) > length:
                    index.setdefault(identifier[-length:], identifier)
            self._tails[length] = index
        return index.get(token)

    def consume(self, text: str) -> None:
        """Consume a pool payload, one identifier per use, by identity.

        Identifiers are re-emitted with their own inner spaces (`issue #1875`,
        `git status`), so a token that is not itself pooled is tried joined
        with the next one before it counts as invented.
        """
        from local_operator.saver.validate import SaverRejected

        tokens = text.split()
        index = 0
        while index < len(tokens):
            token = tokens[index]
            index += 1
            emitted = token
            if token not in self.tokens:
                pair = token + " " + tokens[index] if index < len(tokens) else ""
                if pair and pair in self.tokens:
                    token = pair
                    emitted = pair
                    index += 1
                else:
                    # A TAIL of a span identifier is admissible, because the
                    # reducer's own `_call_summary` cuts long command lines and
                    # pools what is left (`/daemon.py` from
                    # `local_operator/mobile/daemon.py`). Refusing those
                    # refuses the reference reducer's honest output, and one
                    # refused epoch voids a whole response (round-6 review,
                    # R6-1: 0 of 19 real responses adopted). The round-5
                    # blocker stays closed because the object SPENT is the
                    # identifier, not the slice: one identifier is usable
                    # once, so it can buy at most its own characters — never
                    # the 2,400-character unit the count budget allowed.
                    owner = self.owner_of(token)
                    if owner is None:
                        raise SaverRejected("provenance")
                    token = owner
            # THE BOUND IS CHARACTERS, NOT A COUNT OF UNITS. The reducer pools
            # OVERLAPPING fragments of one identifier — epoch 1 of the replay
            # holds `local_operator/mobile/daemon.py` (30 chars), `/daemon.py`
            # and `al_operator/mobile/daemon.py` (68 chars together) — so a
            # one-use-per-identifier rule refuses its own honest output, which
            # is R6-1 again. What bounds the pool instead: every token is
            # material of a span identifier (identifiers are capped at 120
            # characters by `provenance_ids`), the payload per record is capped
            # at the reducer's own 2,400, the epoch's total is charged against
            # what it deleted, and the epoch must still be cheaper than its
            # span in tokens. A count of uses bounds nothing — that was the
            # round-5 blocker; a count of CHARACTERS does.
            self.chars += len(emitted)
            if self.chars > self.char_budget:
                raise SaverRejected("provenance")


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
#: reduce_v2's ``DEEP_IDS_CHARS``: the reducer's own bound on one epoch's
#: identifier pool. Encoded here because a payload past it is not a rendering of
#: anything the reducer documents, so it is refused rather than measured.
_POOL_CHARS = 2400
#: The pool's word material: everything `identifiers()` does not cover but the
#: reducer still pools (camelCase names, short error codes).
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
#: Containment lookups one pool may spend. Each miss walks the identifier set
#: once, so this bounds the fallback at O(scans x identifiers) per epoch
#: instead of leaving it proportional to the response's size.
_POOL_SCANS = 64
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
    # A RUN is returned VERBATIM: its leading and trailing whitespace is line
    # structure (the newline after a marker, the space either side of a note in
    # the middle of a JSON summary), and the line reconstruction in `_units`
    # depends on it. Markers and notes are exact template matches and are
    # stripped so their slots tokenise the same way wherever they sit.
    return [
        (kind, text if kind == "run" else text.strip())
        for kind, text in elements
        if kind != "run" or text.strip()
    ]


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
    """The emitted CONTENT LINES of a record, in order.

    Lines, not fragments: the algorithm in :func:`_map_fragments` maps one
    emitted line to one source line, so a line that carried an inline marker
    contributes its remaining content as a single line. Splitting it further
    would let one source line answer several emitted lines, which is exactly the
    re-segmentation the invariant forbids.

    Framing lines (the epoch header and its continuation) are removed by
    ``parse_records`` before this point, and disclosure elements (markers, the
    reducer's own notes) are not content and are not returned.
    """
    return _units(elements)[0]


def _units(elements: Sequence[_Element]) -> tuple[list[str], list[str], list[str]]:
    """One body's three unit kinds: (content lines, pool payloads, previews).

    CONTENT is measured in the body's own LINES, not in runs. A note sits INSIDE
    a line (the reducer replaces a long argument value with `[1.7k chars]` in the
    middle of a JSON summary), so counting runs as lines would refuse every call
    record whose arguments were paraphrased — their pieces are the two runs
    either side of the note and the second is not a new line.

    The distinction between the other two is made by the element that INTRODUCES
    a run: a run after an I1 marker is the identifier pool (a record of its own,
    or a `[radient:ids] …` continuation of a call or assistant record) and a run
    after an O5 marker is that marker's preview. Everything else is CONTENT.

    The arrow is a line break: the reducer joins a short result onto its call's
    line with `\u2192`, and in the source those are two rows. The ellipsis is the
    reducer's own cut marker (`_short` appends it), never source text, so it is
    dropped rather than matched. A pooled identifier may itself be cut at 120
    characters, so an ellipsis can split one pool token into two runs; those are
    rejoined with no separator and ``_Pool.has`` prefix-matches the result.
    """
    lines: list[str] = []
    current: list[str] = []
    pool: list[str] = []
    preview: list[str] = []
    mode = ""

    def flush() -> None:
        text = "".join(current).strip()
        if text:
            lines.append(text)
        current.clear()

    for kind, text in elements:
        if kind == "marker":
            mode = (
                "pool"
                if text.startswith("[radient:ids")
                else ("preview" if text.startswith("[radient:harness state omitted") else "")
            )
            continue
        if kind == "note":
            # A note REPLACES source text in an argument summary, but the same
            # literal can be REAL source text the projection wrote, so it stays
            # in the line and :func:`_map_fragments` tries the line with it and
            # without it. Dropping it here made an honest strong-error line
            # unmatchable; a pool payload is tokens, so a note there is dropped.
            if mode == "":
                current.append(text)
        if kind == "arrow":
            # The arrow is the reducer's result-join AND a character prose uses
            # (`planning → active → qa`), so it stays in the line: the reducer's
            # own join fails to match whole and `_map_fragments` then splits it
            # at the arrow, while prose never needs to.
            # `normalise` rstrips each run, so the space on the arrow's left is
            # restored here; without it prose reads `planning→ active`.
            if current and current[-1] and not current[-1][-1].isspace():
                current.append(" ")
            current.append("\u2192 ")
            mode = ""
            continue
        if kind == "ellipsis":
            # The reducer appends one where it CUT a line, but a source line may
            # contain one legitimately (`'Toronto mayoral\u2026'` in a defect
            # report), so it is kept as text and a TRAILING one is stripped only
            # when matching. Inside a pool payload it is part of a token the
            # reducer truncated at 120 characters, so it stays attached.
            (pool if mode == "pool" else current).append("\u2026")
            continue
        if kind != "run":
            continue
        if mode == "pool":
            pool.append(text)
        elif mode == "preview":
            preview.append(text)
        else:
            for index, part in enumerate(text.split("\n")):
                if index:
                    flush()
                current.append(part)
    flush()
    # Disclosure payloads are whitespace-separated tokens, not lines, so the
    # runs a cut identifier was split into are rejoined with no separator and
    # the whole payload is checked once.
    return (
        lines,
        ["".join(pool).strip()] if pool else [],
        ["".join(preview).strip()] if preview else [],
    )


def _preview_text(text: str) -> str:
    """reduce_v2.user_text's harness-state preview, mirrored exactly.

    O5 replaces a regenerated harness block: the marker, then the block's first
    line cut to 60 characters, then up to four `## ` section headers. The
    preview is therefore not text that merely appears near the source — it is a
    FUNCTION of the one record it stands for, which is what bounds it: the
    verifier recomputes it from that record and compares.
    """
    stripped = text.lstrip()
    lines = stripped.split("\n")
    first = lines[0] if lines else ""
    subs = [line.strip() for line in lines if line.startswith("## ")][:4]
    return _short(first, 60) + ((" " + " ".join(subs)) if subs else "")


_JSON_STRUCT = re.compile(r"[{}\[\],\"]")
_PUNCT_ONLY = re.compile(r"[^\w\u0080-\uffff]+")


#: JSON/array punctuation is the reducer's own joining grammar, not content.
_JSON_STRUCT = re.compile(r"[{}\[\],\"]")
_PUNCT_ONLY = re.compile(r"[^\w\u0080-\uffff]+")
#: How far ahead of the last consumed segment a non-exact fragment may match.
#: Exact fragments (the overwhelming majority) are a dict lookup, so the forward
#: window only pays for prefixes and suffixes, and it keeps the whole mapping
#: linear in the epoch's size.
#: The shortest prefix or suffix that counts as a documented cut (see
#: :func:`_map_fragments`). ONE character is the re-segmentation primitive and is
#: never a cut; two is the shortest length the reviewer's own `abcabc` -> `bc` +
#: `ab` case uses, and it is admitted here deliberately so that case is refused
#: by the CURSOR (one source line cannot answer two emitted lines) rather than by
#: a length rule that would hide whether distinctness works.
MIN_CUT = 2


def _segments_with_spans(text: str) -> list[tuple[str, int, int]]:
    """The text's canonical segments with their character spans, in order.

    :func:`_split_line` yields the segments; the spans are what makes the
    segment path a CONSUMING match rather than a vocabulary check — a segment
    matched once cannot be matched again, and a later segment cannot precede an
    earlier one.
    """
    out: list[tuple[str, int, int]] = []
    offset = 0
    for line in text.split("\n"):
        position = 0
        for piece in _split_line(line):
            index = line.find(piece, position)
            if index < 0:
                continue
            position = index + len(piece)
            out.append((piece, offset + index, offset + index + len(piece)))
        offset += len(line) + 1
    return out


def _word_in(text: str, start: int, length: int) -> bool:
    """Is ``text[start:start+length]`` a whole whitespace-delimited word?"""
    before = text[start - 1] if start else " "
    after = text[start + length] if start + length < len(text) else " "
    return before.isspace() and after.isspace()


def _without_ellipsis(line: str) -> str:
    """``line`` minus one trailing truncation marker, or unchanged.

    `…` and `...` are what the reducer writes when it cuts a line; neither is a
    character of the source, so the matcher sees them as invention. A line that
    is ONLY the marker is left alone: it names no source text and must fail.
    """
    stripped = line.rstrip()
    for marker in ("…", "..."):
        if stripped.endswith(marker):
            head = stripped[: -len(marker)].rstrip()
            return head or line
    return line


def _map_fragments(out: Sequence[str], record: _Rec) -> bool:
    """THE INVARIANT, implemented exactly: order, distinctness, real deletion.

    A fragment must be a contiguous piece of ONE source line and need not touch
the line's edges: the reducer's own cuts land mid-line (an archive head is the
first ~160 characters of a block, so it starts and ends wherever the character
count falls), and the replay measured whole/prefix/suffix-only matching
refusing every honest archive record. What keeps that from becoming the
re-segmentation primitive is the LINE cursor: a source line answers at most one
emitted line, so a line cannot be shredded into pieces of itself.

    For a record whose source lines are ``src[0..n-1]`` and whose emitted
    content lines are ``out[0..m-1]``, in order::

        cursor = 0
        for line in out:
            found = -1
            for j in range(cursor, n):
                S = src[j]
                if line == S or (len(line) >= MIN_CUT
                                 and (S.startswith(line) or S.endswith(line))):
                    found = j
                    break
            if found < 0:
                reject
            cursor = found + 1
        if m >= n:
            reject

    THREE PROPERTIES, and each one is doing distinct work:
    - ``cursor`` only moves forward, so emitted lines are IN ORDER;
    - ``cursor = found + 1`` means a matched source line is never reconsidered,
      so each source line is spent at most once — DISTINCTNESS. Deleting that
      one line is what let round 4 accept `abcabc` as `bc` + `ab` (9.2x tokens);
    - ``m < n`` means the record really deleted something, so the epoch is a
      reduction rather than a re-rendering.

    ``MIN_CUT`` only decides whether a very short prefix or suffix counts as a
    cut: one character never does, because a single-character piece is the
    re-segmentation primitive itself. Distinctness — not the cut length — is what
    refuses the reviewer's `abcabc` -> `bc` + `ab` case, and the mutation matrix
    in the PR comment shows that deleting `cursor = found + 1` flips it.
    """
    source = record.text
    if not source.strip():
        return not out
    # THE REDUCER'S DOCUMENTED TRUNCATION. A cut line is rendered with a
    # trailing ellipsis (`…`, or `...` when the writer is ASCII), which is not
    # source text; matching it literally refuses every truncated preview the
    # reference reducer emits (one epoch in ten of the replay's responses, and
    # adoption is all-or-nothing per response). Stripping ONE trailing ellipsis
    # before the containment test keeps the rule checked — the remainder must
    # still be a contiguous piece of a source line — and cannot admit invented
    # text, because the ellipsis is the only character it removes.
    out = [_without_ellipsis(line) for line in out]
    # The source is indexed by LINE first: one source line answers at most one
    # emitted line, which is the property that stops a line being shredded into
    # pieces of itself (round 4: `abcabc` as `bc` + `ab`, 9.2x). A fragment may
    # sit anywhere inside its line — the reducer's own cuts land mid-line — but
    # it may not move backwards and it may not share a line with another.
    offsets: list[tuple[str, int]] = []
    position = 0
    for line in source.split("\n"):
        offsets.append((line, position))
        position += len(line) + 1
    whole = {line for line, _ in offsets if line.strip()}
    segments: list[tuple[str, int, int]] | None = None
    starts: list[int] = []
    index = 0
    used = 0
    pending: list[tuple[str, int]] = [(line, 0) for line in reversed(out)]
    while pending:
        line, depth = pending.pop()
        # A note is disclosure where it REPLACED source text and content where
        # the source really holds that literal (`[redacted]` in a test's own
        # assertion output), so both readings are tried: with it, then without.
        # A TRAILING ellipsis is dropped from each reading: it is `_short`'s cut
        # marker, so the source continues where the line stops.
        readings = (line, _NOTE.sub("", line).strip())
        variants: list[str] = []
        for reading in readings:
            if reading.endswith("\u2026"):
                reading = reading[:-1]
            if reading and reading not in variants:
                variants.append(reading)
        bare = variants[-1] if len(variants) > 1 else ""
        found = -1
        for variant in variants:
            for line_index in range(index, len(offsets)):
                text, at = offsets[line_index]
                hit = text.find(variant)
                if hit < 0:
                    continue
                if (
                    len(variant) < MIN_CUT
                    and variant not in whole
                    and not _word_in(text, hit, len(variant))
                ):
                    # A single character is the re-segmentation primitive UNLESS
                    # it is a whole word of the source: a merged one-line result
                    # (`→ 2`) is exactly that, and refusing it cost an epoch.
                    continue
                found = line_index
                taken = at + hit + len(variant)
                break
            if found >= 0:
                break
        if found >= 0:
            index = found + 1
            if segments is not None:
                used = max(used, bisect.bisect_left(starts, taken))
            continue
        if "\u2192 " in line and depth < 4:
            # The reducer's ` → body` join: the source holds the command and the
            # result on two rows, so the joined line is split and each half is
            # matched on its own. Bounded, because prose may hold many arrows and
            # splitting those would be one emitted line per arrow.
            head, _, tail = line.partition("\u2192 ")
            pending.append((tail.strip(), depth + 1))
            pending.append((head.strip(), depth + 1))
            continue
        # THE SECOND DOCUMENTED RENDERING. `_call_summary` re-renders a dict
        # argument as canonical JSON with `i`/`intent` dropped, secret keys
        # redacted and every long value replaced by a note, so the emitted line
        # is not a span of the source line in any punctuation-preserving sense:
        # the source row holds the RAW argument text, whose key order and
        # separators differ. It is still only the row's own tokens, in the row's
        # own order, each used once — which is what the segment cursor checks.
        # The punctuation itself is grammar and is capped, so a line of nothing
        # but braces cannot ride the path.
        if segments is None:
            segments = _segments_with_spans(source)
            starts = [segment[1] for segment in segments]
        floor = offsets[index][1] if index < len(offsets) else len(source)
        used = max(used, bisect.bisect_left(starts, floor))
        text = bare or variants[0]
        pieces = _split_line(text)
        structural = len(text) - sum(len(piece) for piece in pieces)
        if not pieces or structural > 6 * len(pieces) + 8:
            return False
        for piece in pieces:
            while used < len(segments) and (
                segments[used][0] != piece or segments[used][1] < floor
            ):
                used += 1
            if used >= len(segments):
                return False
            floor = segments[used][2]
            used += 1
        index = bisect.bisect_right([at for _, at in offsets], floor - 1)
    return True


def _ruler_tokens(text: str) -> int:
    """``tokens_R(text)`` — the client's own estimator, never a proxy.

    `compaction.tokens` is the estimator the provider path already calls on
    every request (`estimate_messages_tokens`, after the splice point), so the
    epoch and the span are measured with the same ruler as the request they
    replace: cl100k_base, the encoding the `tokenizer` extra ships.

    Where the estimator degrades to `chars // 4` — the extra is not installed,
    or a load has not published yet — the answer is UNAVAILABLE, not
    approximate. A byte ratio is exactly the bypass rounds 3 and 5 measured, so
    an epoch this client cannot count is an epoch it does not accept; the chain
    then sends unreduced history, which is what "Saver unavailable" means in
    practice.
    """
    from local_operator.compaction import tokens as estimator
    from local_operator.saver.validate import SaverRejected

    if estimator._get_encoding() is None:
        raise SaverRejected("provenance")
    return estimator.count_text_tokens(text)


def estimator_available() -> bool:
    """Can this client count tokens exactly?

    The gate for the upload path: R6-3 measured "charge, then refuse". Adoption
    is all-or-nothing per response and the spend is recorded before validation,
    so an epoch refused for want of a ruler bills the user for an upload it
    could never accept. `chain._apply` asks this BEFORE it projects, uploads or
    charges anything, which is what makes "no tokenizer ⇒ Saver unavailable"
    mean unavailable rather than "upload, get charged, refuse".

    Deliberately cheap: it tests that the `tokenizer` extra is importable and
    never triggers a load, so it can run on the turn path. The per-epoch
    comparison still refuses (`_ruler_tokens`) when the encoding exists as a
    module but cannot be fetched — one upload per process in that state, which
    the backoff then bounds.
    """
    import importlib.util

    return importlib.util.find_spec("tiktoken") is not None


def _check_line_budget(emitted: int, span_lines: int) -> None:
    """The invariant's reduction half, at the granularity the contract states.

    "The epoch's total line count must be strictly less than the span's total
    line count." Applied per RECORD instead (``m < n`` inside the mapping) it
    refuses every honest record that re-emits a single source line, which on the
    real-session replay costs 92 of 131 epochs: measured, 30/131 per-record
    against 122/131 here. The contract sentence is the span-level one, so that is
    what ships; the number is reported either way rather than tuned.

    ``emitted`` counts CONTENT lines only. A disclosure line — the identifier
    pool, an O5 preview — is not content: it is what the deleted content pays
    for, so counting it here charges the same deletion twice and a one-record
    epoch whose only line is a pool line is refused for "no reduction" (the
    manager's replay: pool-only epochs are 17 of the first 131). Each disclosure
    unit is bounded by its own rule instead (``_Pool``, ``_preview_text``).
    """
    from local_operator.saver.validate import SaverRejected

    if emitted >= max(span_lines, 1):
        raise SaverRejected("provenance")


def _check_pool(payload: str, pool: _Pool) -> None:
    """One pool payload: source-derived identifiers, each spent once."""
    from local_operator.saver.validate import SaverRejected

    if len(payload) > _POOL_CHARS:
        # The reducer's own pool bound (`DEEP_IDS_CHARS`). A payload past it is
        # not a documentable rendering of anything, so it is refused on sight.
        raise SaverRejected("provenance")
    pool.consume(payload)


def _check_bounds(elements: Sequence[_Element], record: _Rec) -> tuple[int, int]:
    """The disclosure budget only. Re-segmentation is the cursor's job now.

    ``chars // 8 + 8`` and the aggregate overhead-vs-saved rule are DELETED:
    round 3 proved neither can bound a token cost (span bytes-per-token is
    unbounded above), and round 4's line-count bound is replaced by the two
    exact halves of the invariant in :func:`_map_fragments` — one emitted line
    per source line, and strictly fewer lines than the source record. What
    remains here is the disclosure accounting: markers and notes are lines, they
    are capped, and their slots are checked in :func:`_check_slots`.
    """
    from local_operator.saver.validate import SaverRejected

    markers = [text for kind, text in elements if kind == "marker"]
    notes = [text for kind, text in elements if kind == "note"]
    overhead = sum(len(text) for kind, text in elements if kind != "run")
    pieces = _pieces(elements)
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
    derived = _derived_digests(parts)
    pool = _Pool(parts)
    cursors: dict[str, int] = defaultdict(int)
    consumed: set[str] = set()
    scans = [_SLOT_SCANS]
    span_chars = sum(record.chars for group in records_by_role.values() for record in group)
    records = parse_records(text)
    span_lines = sum(record.lines for group in records_by_role.values() for record in group)
    # THE SECOND HALF OF THE INVARIANT: the epoch must be strictly FEWER content
    # lines than the span, so a record cannot be reproduced line for line and a
    # source line cannot be shredded (round 3: S-15, R3-2). Disclosure units are
    # proved by their own rule below, so they are not counted twice here.
    units = []
    emitted = 0
    content_chars = 0
    disclosures: list[str] = []
    for label, body in records:
        if deadline is not None and time.monotonic() > deadline:
            raise SaverRejected("busy")
        elements = _elements(body)
        pieces, pools, previews = _units(elements)
        disclosures.extend(pools)
        emitted += len(pieces)
        content_chars += sum(len(piece) for piece in pieces)
        units.append((label, elements, pieces, pools, previews))
    _check_line_budget(emitted, span_lines)
    # NON-AMPLIFICATION: the comparison the invariant actually needs, and the
    # only one that can bound cost. Both sides are RENDERED TEXT — the epoch's
    # own bytes against the span's own text as the reducer reads it — never the
    # canonical JSON, whose ~80 bytes of structure per row made a 199-of-200
    # verbatim epoch look like a 0.42x saving while being larger than the text
    # it replaced (S-18's second half).
    span_text = "\n".join(parts)
    if _ruler_tokens(text) >= _ruler_tokens(span_text):
        raise SaverRejected("provenance")
    # THE DISCLOSURE CHANNEL, charged on BOTH sides and only once the epoch's
    # content is known: every pooled token must be an exact source identifier
    # spent once, and the pool's characters are charged against what the epoch
    # deleted — the same accounting content is charged under. A count alone
    # leaked: one unit bought up to 2,400 characters, so repeating a long
    # identifier in pool records was accepted at 3.00x-7.43x the span (S-17's
    # uncounted exemption measured 68 -> 92 o200k tokens on a span with
    # nothing deleted to pay for it).
    pool.char_budget = max(span_chars - content_chars, 0)
    for payload in disclosures:
        _check_pool(payload, pool)
    for label, elements, pieces, pools, previews in units:
        if deadline is not None and time.monotonic() > deadline:
            raise SaverRejected("busy")
        kind, name = _label_parts(label)
        if kind == "call" and not pieces and not previews and len(elements) == 1:
            verbs = _group_verbs(elements[0][1], scans)
            if verbs is not None:
                _consume_group(records_by_role.get("call", []), {"call": 0}, consumed, verbs)
                continue
        disclosure_only = pools and not pieces and not previews
        if disclosure_only:
            # A pool record stands for the records it discloses, not for one of
            # them: it spends no candidate, so it cannot shadow a later record's
            # source (the manager's replay measured a consumed-early pool line
            # making a later call record unreachable). Its own disclosure is
            # still capped, but against ITSELF: no source record can be named
            # for a payload that stands in for many, and `_check_bounds` only
            # needs the counts.
            if sum(1 for element_kind, _ in elements if element_kind == "marker") > 1:
                raise SaverRejected("provenance")
            owner = _Rec(text=body, identity="", lines=1, chars=len(body))
            _check_bounds(elements, owner)
            _check_slots(elements, owner, media, derived, scans, span_chars)
            continue
        candidates = records_by_role.get(kind, [])
        # Consumption, not source order, is what bounds reuse: the epoch may
        # carry a dedupe record for an earlier row after later ones, so one
        # wrap is allowed. Each source record is spent once.
        start = min(cursors[kind], len(candidates))
        matched: _Rec | None = None
        tried = 0
        furthest = start
        # The rotation is modulo, not two list slices: building the wrapped
        # order per record copies the whole candidate list, which is O(n²) over
        # an epoch and would break the linearity property (S-1).
        for offset in range(len(candidates)):
            if deadline is not None and time.monotonic() > deadline:
                raise SaverRejected("busy")
            position = (start + offset) % len(candidates)
            candidate = candidates[position]
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
            if previews:
                # A preview is a FUNCTION of the one record it replaces, so it
                # must be that record's own preview — not a line of the source
                # and not free text near it.
                if previews[0] != _preview_text(candidate.text):
                    continue
            elif not _map_fragments(pieces, candidate):
                continue
            # Slots can select among same-role candidates: a grouped
            # `call: [radient:1 read-only commands, all ok: grep]` names the
            # command, and an elided record claims a size.
            try:
                _check_bounds(elements, candidate)
                _check_slots(elements, candidate, media, derived, scans, span_chars)
            except SaverRejected:
                continue
            matched = candidate
            furthest = max(furthest, position + 1)
            break
        if matched is None:
            raise SaverRejected("provenance")
        consumed.add(matched.identity)
        # The cursor follows the MATCH, never the furthest position merely
        # LOOKED at: an advance on a skipped candidate spends the rest of that
        # role's records for a later record that would have matched them.
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
