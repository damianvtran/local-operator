"""Fail-closed record provenance for auto: §6.1(6), not token membership.

A vocabulary check cannot prove extraction: one source token could be repeated
55,000 times into a byte-smaller but 2.87x-token epoch. We instead match complete
record fragments against client-derived source forms. Fragments are separated
ONLY by the method's fixed markers; quoting and terminal ellipsis are undone,
not arbitrary words or punctuation. Each source record is consumed at most
once, and fragments within it consume disjoint, ordered character ranges.

This is a verifier, not a client-side reducer. It does not choose spans, drop
reads, summarise, or generate epochs. Only documented formatting transforms
(cleaning, call argument rendering, exit status and identifier metadata) are
recognised. An output using a transform we cannot establish is refused: the
caller sends unreduced history, never falls back to the former coarse check.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

# reduce-v2.clean: the transforms quoted by §6.1(6). Rule 4's full normative
# range is used even where the reference currently strips a narrower subset.
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_INVISIBLE = re.compile(
    "[\u00ad\u061c\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]"
)
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1b\x7f]")
_MARKER = re.compile(r"\[\s*radient\s*:", re.I)
_CD = re.compile(r"^\s*cd\s+((?:\"[^\"]+\"|'[^']+'|[^\s;&|]+))\s*(?:&&|;)\s*")
_EXIT = re.compile(r"^exit code: (-?\d+)$", re.M)
_SCAFFOLD = frozenset({"exit code: 0", "--- stdout ---", "--- stderr ---", "(empty)"})


def clean(text: str) -> str:
    text = _ANSI.sub("", text)
    text = _INVISIBLE.sub("", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return _MARKER.sub("[radient(quoted):", _CONTROL.sub("", text))


@dataclass(frozen=True)
class _Record:
    name: str
    forms: tuple[str, ...]
    identity: str = ""
    result_identity: str = ""
    plain_count: int = 0


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [
        block["text"]
        for block in content or []
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]


def _source(span: Sequence[dict[str, Any]]) -> tuple[dict[str, list[_Record]], list[str]]:
    """Source records in order per role; alternate forms NEVER duplicate a record."""
    from local_operator.saver.canonical import canonical_bytes

    records: dict[str, list[_Record]] = defaultdict(list)
    parts: list[str] = []
    calls: dict[str, int] = {}
    for row_number, row in enumerate(span):
        identity = f"row:{row_number}"
        role = row.get("role")
        texts = [
            "\n".join(line for line in clean(t).split("\n") if line)
            for t in _texts(row.get("content"))
        ]
        parts.extend(texts)
        who = row.get("name")
        if isinstance(who, str) and who:
            prefix = "(" + re.sub(r"[^A-Za-z0-9_.:-]", "_", who)[:64] + ") "
            texts = [prefix + text if text.strip() else text for text in texts]
        if role == "assistant":
            text = "\n".join(texts).strip()
            if text:
                records["assistant"].append(_Record("", (text,), identity))
        elif role == "tool":
            text = "\n".join(texts)
            status = _EXIT.search(text)
            failed = bool(row.get("is_error")) or bool(status and status[1] != "0")
            lines = [line for line in text.split("\n") if line.strip() not in _SCAFFOLD]
            if failed:
                lines = [line for line in lines if not _EXIT.fullmatch(line)]
            body = "\n".join(lines).strip("\n")
            prefix = f"exit {status[1]} " if status and status[1] != "0" else ""
            forms = tuple(dict.fromkeys((text, prefix + (body or "(no output)"))))
            records["error" if failed else "result"].append(_Record("", forms, identity))
            call_position = calls.get(str(row.get("tool_call_id", "")))
            if call_position is not None and not failed:
                call = records["call"][call_position]
                joined = tuple(f"{form}\n{body}" for form in call.forms)
                inline = (
                    tuple(f"{form} → {body}" for form in call.forms) if "\n" not in body else ()
                )
                records["call"][call_position] = replace(
                    call,
                    forms=(*call.forms, *joined, *inline),
                    result_identity=identity,
                    plain_count=len(call.forms),
                )
        else:
            # Keep the original block order (a receipt can precede text), and
            # give each block its own occurrence identity. Sharing a row id
            # made a valid second text block look like duplicated content.
            content = row.get("content")
            blocks = (
                [{"type": "text", "text": content}] if isinstance(content, str) else content or []
            )
            text_index = 0
            for block_index, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                block_id = f"{identity}:block:{block_index}"
                if isinstance(block.get("text"), str):
                    text = texts[text_index]
                    text_index += 1
                    if text.strip():
                        records["user"].append(_Record("", (text,), block_id))
                elif block.get("type") == "image_ref":
                    records["user"].append(_Record("", ("",), block_id))
        for call in row.get("tool_calls") or []:
            function = call.get("function") or {}
            name = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(function.get("name") or "tool"))[:64]
            raw = function.get("arguments", "")
            if not isinstance(raw, str):
                raw = canonical_bytes(raw).decode()
            forms = [clean(raw)]
            parts.extend((name, clean(raw)))
            try:
                args = json.loads(raw)
                forms.append(clean(canonical_bytes(args).decode()))
            except (ValueError, TypeError):
                args = None
            if isinstance(args, dict):
                # The shell-call display transform removes ``cd DIR &&`` and
                # may render its cwd as ``(in DIR)``. No shell is executed.
                command = args.get("command")
                if name in {"bash", "shell", "exec", "run", "terminal"} and isinstance(
                    command, str
                ):
                    command = clean(command)
                    parts.append(command)
                    cd = _CD.match(command)
                    forms.append(command)
                    if cd:
                        tail = command[cd.end() :]
                        forms.extend((tail, f"(in {cd[1]}) " + tail))
            calls[str(call.get("id", ""))] = len(records["call"])
            records["call"].append(
                _Record(
                    name, tuple(dict.fromkeys(forms)), f"call:{row_number}:{len(records['call'])}"
                )
            )
    return dict(records), parts


class _Identifiers:
    """Exact reducer identifier extraction, once-only metadata, not word membership."""

    def __init__(self, parts: Sequence[str]) -> None:
        self.parts = parts
        self.tokens: set[str] = set()
        self.used: set[str] = set()
        self.built = False

    def _build(self) -> None:
        from local_operator.saver.provenance_ids import identifiers
        from local_operator.saver.validate import (
            SaverRejected,
            _b64_note_digests,
            _chunks,
        )

        for part in self.parts:
            for chunk in _chunks(part):
                self.tokens.update(identifiers(chunk))
                if len(self.tokens) > 20_000:
                    raise SaverRejected("busy")
            self.tokens.update(_b64_note_digests(part))
        self.built = True

    def consume(self, text: str) -> None:
        from local_operator.saver.validate import SaverRejected

        if not self.built:
            self._build()
        words = text.split()
        index = 0
        while index < len(words):
            token = words[index]
            if index + 1 < len(words) and token + " " + words[index + 1] in self.tokens:
                token += " " + words[index + 1]
                index += 1
            if token not in self.tokens or token in self.used:
                raise SaverRejected("provenance")
            self.used.add(token)
            index += 1


def _fragments(body: str, identifiers: _Identifiers) -> tuple[list[str], bool]:
    """Split only at closed method markers; identifier lists are separately checked."""
    from local_operator.saver.validate import MARKER_TEMPLATES, SaverRejected

    fragments: list[str] = []
    at = 0
    result_lines = False
    pool_only = body.strip().startswith("[radient:ids mentioned in this part]")
    while True:
        mark = body.find("[radient:", at)
        if mark < 0:
            fragments.append(body[at:])
            break
        fragments.append(body[at:mark])
        matched = next(
            (
                m
                for pattern in MARKER_TEMPLATES.values()
                if (m := pattern.match(body, mark)) is not None
            ),
            None,
        )
        if matched is None:
            raise SaverRejected("provenance")
        at = matched.end()
        if matched[0] in {"[radient:ids]", "[radient:ids mentioned in this part]"}:
            end = body.find("\n", at)
            end = len(body) if end < 0 else end
            identifiers.consume(body[at:end])
            at = end
    # ``_short`` appends an ellipsis; it is not permission to delete an
    # arbitrary interior substring. Do not split on ellipses inside content.
    # Result elision explicitly keeps selected source lines, not a contiguous
    # paragraph (reduce-v2.result and merged _strong_errors). Only that marker
    # authorises line-wise matching; ordinary user/assistant prose stays whole.
    result_lines = "[radient:ok " in body or "[radient:omitted " in body
    pieces = (
        [line for piece in fragments for line in piece.split("\n")] if result_lines else fragments
    )
    normalized = [piece.strip().removesuffix("…").rstrip() for piece in pieces]
    return [piece for piece in normalized if piece], pool_only


def _matches(fragments: Sequence[str], forms: Sequence[str], work: list[int]) -> int | None:
    for index, form in enumerate(forms):
        # A fixed linear allowance across the WHOLE epoch, including alternate
        # source forms. Exhaustion fails open instead of becoming O(R × S) on
        # adversarial near-matches. The normal case advances source cursors.
        work[0] -= len(form) + sum(map(len, fragments))
        if work[0] < 0:
            from local_operator.saver.validate import SaverRejected

            raise SaverRejected("busy")
        cursor = 0
        for fragment in fragments:
            position = form.find(fragment, cursor)
            if position < 0:
                break
            cursor = position + len(fragment)
        else:
            return index
    return None


def check_records(
    text: str, span: Sequence[dict[str, Any]], *, deadline: float | None = None
) -> None:
    """Verify source-derived records with monotonically advancing source cursors.

    Each source record is inspected at most once in its role stream, has a
    fixed small number of formatting alternatives, and is consumed once on a
    match. Fragments consume disjoint ordered ranges within a form. One failed
    search exhausts that candidate; exhausting the role stream rejects the
    epoch immediately. There is no independent whole-span scan per output
    token, no vocabulary-only fallback and no multiplicity amplification.
    """
    from local_operator.saver.validate import SaverRejected, parse_records

    source, parts = _source(span)
    identifiers = _Identifiers(parts)
    cursors: dict[str, int] = defaultdict(int)
    pool_seen = False
    consumed: set[str] = set()
    work = [32 * (len(text) + sum(map(len, parts)) + 1)]
    for label, body in parse_records(text):
        if deadline is not None and time.monotonic() > deadline:
            raise SaverRejected("busy")
        fragments, pool_only = _fragments(body, identifiers)
        if pool_only:
            if label != "user:" or pool_seen or any(fragments):
                raise SaverRejected("provenance")
            pool_seen = True
            continue
        kind = label.split("#", 1)[0].split(" ", 1)[0].rstrip(":")
        name = label.split(" ", 1)[1].rstrip(":") if kind == "call" and " " in label else ""
        candidates = source.get(kind, [])
        position = cursors[kind]
        while position < len(candidates):
            candidate = candidates[position]
            position += 1
            if name and candidate.name != name or candidate.identity in consumed:
                continue
            forms = candidate.forms
            if candidate.result_identity and candidate.result_identity in consumed:
                forms = forms[: candidate.plain_count]
            match = _matches(fragments, forms, work)
            if match is not None:
                consumed.add(candidate.identity)
                if candidate.result_identity and match >= candidate.plain_count:
                    consumed.add(candidate.result_identity)
                cursors[kind] = position
                break
        else:
            raise SaverRejected("provenance")
