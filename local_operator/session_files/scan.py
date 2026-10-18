"""Derive a session's file inventory from its transcript journal. Pure: bytes in, records out.

WHY THIS EXISTS. The desktop Files view used to build its list in the renderer by scanning the
transcript for path-shaped strings. That cannot see a generated image or video at all (the
tool result carries a content-addressed DIGEST, never a path), cannot see a file a subagent
wrote, stops after ~2,400 messages, and loses anything compaction blanked. This module is the
authoritative half: it reads the journal once, remembers what each file-producing tool call
actually did, and hands the service a record per file.

WHAT COUNTS AS A FILE THE SESSION PRODUCED (design §3.1/§3.3), each on EVIDENCE, never on
resemblance:

* ``write`` / ``edit`` results -> ``details.path``  (origin ``written`` / ``edited``);
* ``bash`` / ``eval`` calls -> output paths parsed by ``supplements.candidates`` (the same
  grammar the Highlights feature uses; one parser, not two), resolved against the session cwd
  (origin ``command``). Paired with the RESULT by ``tool_call_id`` so a failed command, which
  wrote nothing, is not evidence;
* ``browser`` screenshots and downloads -> ``details.path`` / ``details.files[]``;
* ``generate_image`` / ``generate_video`` / ``video_status`` / video ``job_result`` rows ->
  ``details.artifacts[]`` (new rows), else ``AttachmentContent`` blocks in ``content`` (old
  rows), else ``details.digest(s)`` (older video receipts). These are DIGEST-ONLY artifacts
  (``ref.artifact``): no filesystem path exists for them until the service materialises one;
* ``supplement_v1`` ("Highlights") rows -> kept as per-anchor callouts, newest version per
  anchor; the SERVICE overlays them (``highlighted: true``) at read time, because a newer
  version of an anchor must be able to retract an older version's callout.

COST DISCIPLINE (the reason this is line-oriented and not message-oriented). Journal rows
reach ~900 KB and a heavy session has 50k of them, but only a few percent are file-producing
tool rows. :meth:`FileScan.feed` therefore decides from a few cheap ``in`` / one regex probes on
the RAW BYTES whether a row could matter, and calls ``json.loads`` only for those. A
``bash`` RESULT row is not parsed at all: its call (parsed once, when it was an assistant row
carrying the command) already told us what it would write, and the only fact needed from the
result - did it error - is a byte probe. ``tests/unit/session_files/test_scan.py`` pins this
as a structural invariant (parse counts), which cannot flake the way a wall-clock bound does.

NO I/O, NO CLOCK. The caller supplies bytes, the cwd and home, and an optional ``deny``
predicate (the sensitive-path denylist, which stats the filesystem and so lives in the
caller). Every rule here is therefore unit-testable against a list of lines, and the
persisted shape (:meth:`FileScan.to_payload`) is a plain JSON-able dict.

STDLIB PLUS TWO LEAVES: ``local_operator.ansi`` (control-sequence stripping for the call
intent) and ``local_operator.supplements.{candidates,contract}`` (the output grammar and the
reader rule for callouts). Nothing here imports the session runtime.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterable, Mapping

from local_operator.ansi import sanitize_prompt_line

# The bash/eval output grammar is the Highlights feature's; importing its private helpers is the
# point (ONE parser for "what did this command write"), not an accident.
from local_operator.supplements.candidates import _bash_outputs, _eval_outputs, kind_of
from local_operator.supplements.contract import (
    SUPPLEMENT_CUSTOM_TYPE,
    is_digest,
    reader_disposition,
)

#: Bumped on an incompatible change to the record shape OR to what the scanner admits (a new
#: tool, a changed rule): a cache written under another version is discarded and rebuilt, so a
#: rule change is never half-applied to an old session. Also covers the denylist's verdicts,
#: which are baked into the stored records.
SCAN_SCHEMA: Final = 3

#: ``origin`` vocabulary, strongest evidence first. A file seen under several origins keeps the
#: strongest (``write`` then a later ``bash > same`` is still "written"), so the filter chips are
#: stable under re-touching.
ORIGINS: Final = (
    "generated",
    "written",
    "screenshot",
    "download",
    "edited",
    "command",
    "highlight",
)
_ORIGIN_RANK: Final = {name: index for index, name in enumerate(ORIGINS)}

#: Hard bound on records per session. The cap keeps the cache file and the in-memory view finite
#: for a pathological session (a loop that writes 100k files); overflow is COUNTED and reported
#: (``dropped``/``truncated``), never silent.
MAX_RECORDS: Final = 50_000

#: Calls remembered while their result has not arrived yet. A call whose result never lands
#: (interrupted turn) must not grow the state without bound.
MAX_PENDING: Final = 512

#: Output paths kept per bash/eval call. A command that names hundreds of ``-o`` flags is not a
#: file-producing call in any sense a reader cares about.
MAX_CALL_OUTPUTS: Final = 16

#: Callouts kept per anchor (the contract itself bounds ``files`` at 12 and ``more`` at 20).
MAX_ANCHOR_PATHS: Final = 40

_INTENT_MAX_CHARS: Final = 80

#: Tool names whose RESULT rows can carry a file fact.
_RESULT_TOOLS: Final = frozenset(
    {
        "write",
        "edit",
        "bash",
        "eval",
        "browser",
        "generate_image",
        "generate_voiceover",
        "generate_sound_effect",
        "generate_video",
        "video_status",
    }
)
#: Tool names whose CALL rows are worth parsing: the ones that pair with a result above.
_CALL_TOOLS: Final = frozenset({"write", "edit", "bash", "eval"})
_ARTIFACT_TOOLS: Final = frozenset(
    {
        "generate_image",
        "generate_video",
        "video_status",
        "generate_voiceover",
        "generate_sound_effect",
    }
)
_ARTIFACT_KINDS: Final = frozenset({"image", "video", "audio"})

# -- byte probes ------------------------------------------------------------------------------
# Rows are written compact (``TranscriptEntry.to_json``), but the probes tolerate whitespace so
# a hand-built fixture or a future pretty-printer cannot silently blind the scanner.
_CALL_NAME_PROBE: Final = re.compile(rb'"name"\s*:\s*"(?:write|edit|bash|eval)"')
_TOOL_NAME_PROBE: Final = re.compile(rb'"tool_name"\s*:\s*"([a-z_]{1,32})"')
_CUSTOM_PROBE: Final = re.compile(
    rb'"custom_type"\s*:\s*"(supplement_v1|job_result|audio_settle_v1)"'
)
_CALL_ID_PROBE: Final = re.compile(rb'"tool_call_id"\s*:\s*"([^"\\]{1,128})"')
_IS_ERROR_PROBE: Final = re.compile(rb'"is_error"\s*:\s*true')
#: A bash result's own first line, ``exit code: N``. A non-zero exit is not ``is_error`` (the tool
#: reports it as a normal result), yet ``cmd > out.png`` that failed leaves a truncated or empty
#: file the operator did not ask for. Only matches while the content is unpruned; a pruned result
#: keeps no text, and then the call's outputs are admitted on the ``is_error`` evidence alone.
_NONZERO_EXIT_PROBE: Final = re.compile(
    rb'"text"\s*:\s*"(?:TIMEOUT[^"]{0,120}?\\n)?exit code: (?!0\b)\d+'
)
_BROWSER_FACT_PROBE: Final = re.compile(rb'"details"\s*:\s*\{\s*"(?:path|files)"')
_ROW_TS_PROBE: Final = re.compile(rb'^\{\s*"id"\s*:\s*"[^"]*"\s*,\s*"ts"\s*:\s*([0-9.eE+\-]+)')
_DIGEST_FIELD_PROBE: Final = re.compile(rb'"(?:digest|digests|artifacts)"')


@dataclass(slots=True)
class FileRecord:
    """One file (or digest-only artifact) the session produced. Mutated only by its scan."""

    key: str  # "p:<absolute path>" | "a:<digest>"
    path: str | None
    digest: str | None
    name: str
    kind: str
    content_type: str | None
    size: int | None
    width: int | None
    height: int | None
    duration_s: float | None
    origin: str
    tool: str
    call_id: str | None
    intent: str | None
    first_seq: int
    last_seq: int
    first_ts: float
    last_ts: float
    touches: int = 1


#: Positional persisted layout of :class:`FileRecord`. Positional (a list per record, not a
#: dict) because a 15k-record cache is read on a warm path and key names would triple its size;
#: the layout is covered by :data:`SCAN_SCHEMA`.
_RECORD_FIELDS: Final = (
    "key",
    "path",
    "digest",
    "name",
    "kind",
    "content_type",
    "size",
    "width",
    "height",
    "duration_s",
    "origin",
    "tool",
    "call_id",
    "intent",
    "first_seq",
    "last_seq",
    "first_ts",
    "last_ts",
    "touches",
)


@dataclass(slots=True)
class PendingCall:
    """A write/edit/bash/eval call whose result has not been read yet."""

    name: str
    intent: str | None
    outputs: tuple[str, ...]


@dataclass(slots=True)
class Anchor:
    """The newest ``supplement_v1`` version of one anchor, reduced to its file callouts.

    ``files`` entries are ``{path, name, kind, size, why}`` with ``path`` in the contract's
    display form (session-relative or ``~/``); ``more`` is the bare ``more[]`` path list.
    """

    version: int
    ts: float
    seq: int
    files: list[dict[str, Any]] = field(default_factory=list)
    more: list[str] = field(default_factory=list)


def _row(line: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _row_ts(line: bytes) -> float:
    match = _ROW_TS_PROBE.match(line)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    row = _row(line)
    ts = row.get("ts") if row else None
    return float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else 0.0


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def resolve_path(raw: str, *, cwd: str, home: str) -> str | None:
    """An absolute, normalised path for ``raw`` as a tool call spelled it, or ``None``.

    ``~`` expands against ``home`` (not the process's, so a reader can pin it), a relative
    path joins ``cwd`` (``None`` when the session's cwd is unknown: a guess would list a file
    that is not there), and ``normpath`` only collapses ``..``/``.`` textually - no
    ``realpath``, so this stays free of syscalls and the stored spelling is what the tool saw.
    """
    if not raw or "\x00" in raw:
        return None
    if raw == "~" or raw.startswith("~/"):
        raw = os.path.join(home, raw[2:]) if home else ""
    elif not os.path.isabs(raw):
        raw = os.path.join(cwd, raw) if cwd else ""
    if not raw or not os.path.isabs(raw):
        return None
    return os.path.normpath(raw)


class FileScan:
    """The scanner's whole state: feed it journal lines, read records and callouts off it.

    ``rows`` is the ordinal of the NEXT line (equivalently, how many complete lines have been
    fed), which is what makes a scan RESUMABLE: the ledger persists this object and continues
    numbering from it after an append or a deadline-cut partial scan.
    """

    def __init__(
        self,
        *,
        cwd: str = "",
        home: str = "",
        deny: Callable[[str], bool] | None = None,
    ) -> None:
        self.cwd = cwd
        self.home = home
        self._deny = deny
        self.rows = 0
        self.records: dict[str, FileRecord] = {}
        self.pending: dict[str, PendingCall] = {}
        self.anchors: dict[str, Anchor] = {}
        # Legacy roster rows share this bounded/resumable journal pass. Persisting absence
        # avoids a whole-journal backward fallback on every warm Files request.
        self.roster: dict[str, Any] | None = None
        self.truncated = False
        self.dropped = 0
        self._denied: set[str] = set()

    # -- feeding -----------------------------------------------------------------------------

    def feed_all(self, lines: Iterable[bytes]) -> None:
        for line in lines:
            self.feed(self.rows, line)

    def feed(self, seq: int, line: bytes) -> None:
        """Consume one COMPLETE journal line (ordinal ``seq``) and advance :attr:`rows`."""
        self.rows = seq + 1
        # ``in`` is a memchr: the three probes below cost nothing on the ~95% of rows that are
        # prose, state snapshots and spend ledgers.
        if b'"tool_calls"' in line:
            if _CALL_NAME_PROBE.search(line):
                self._on_call_row(seq, line)
            return
        if b'"tool_name"' in line:
            match = _TOOL_NAME_PROBE.search(line)
            if match is not None:
                name = match.group(1).decode("ascii", "replace")
                if name in _RESULT_TOOLS:
                    self._on_result_row(seq, line, name)
            return
        if b'"custom_type"' in line:
            if b"subagent_roster" in line:
                row = _row(line)
                payload = row.get("payload") if row else None
                if (
                    row is not None
                    and isinstance(payload, dict)
                    and payload.get("custom_type") == "subagent_roster"
                ):
                    self.roster = {"ts": row.get("ts", 0), "details": payload.get("details", {})}
                return
            match = _CUSTOM_PROBE.search(line)
            if match is None:
                return
            if match.group(1) == b"supplement_v1":
                self._on_supplement_row(seq, line)
            elif _DIGEST_FIELD_PROBE.search(line):
                self._on_job_result_row(seq, line)

    # -- assistant rows: remember what a call WILL write --------------------------------------

    def _on_call_row(self, seq: int, line: bytes) -> None:
        row = _row(line)
        payload = row.get("payload") if row else None
        calls = payload.get("tool_calls") if isinstance(payload, dict) else None
        if not isinstance(calls, list):
            return
        for call in calls:
            if not isinstance(call, dict):
                continue
            name, call_id = call.get("name"), call.get("id")
            if name not in _CALL_TOOLS or not isinstance(call_id, str) or not call_id:
                continue
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            if not isinstance(args, dict):
                continue
            intent = sanitize_prompt_line(str(args.get("i") or ""), limit=_INTENT_MAX_CHARS)
            outputs: list[str] = []
            if name == "bash":
                outputs = _bash_outputs(str(args.get("command") or ""))
            elif name == "eval":
                outputs = _eval_outputs(str(args.get("code") or ""))
            if len(self.pending) >= MAX_PENDING:
                self.pending.pop(next(iter(self.pending)))
            self.pending[call_id] = PendingCall(
                name, intent or None, tuple(outputs[:MAX_CALL_OUTPUTS])
            )

    # -- tool result rows ---------------------------------------------------------------------

    def _on_result_row(self, seq: int, line: bytes, tool: str) -> None:
        if tool in ("bash", "eval"):
            # No parse: the call told us the outputs, and an error is a byte probe. A result
            # with no pending call (its call row was before a compaction, or it wrote nothing
            # we could name) has nothing to add.
            id_match = _CALL_ID_PROBE.search(line)
            call = self.pending.pop(id_match.group(1).decode(), None) if id_match else None
            if (
                call is None
                or not call.outputs
                or _IS_ERROR_PROBE.search(line)
                or _NONZERO_EXIT_PROBE.search(line)
            ):
                return
            ts = _row_ts(line)
            for raw in call.outputs:
                path = resolve_path(raw, cwd=self.cwd, home=self.home)
                if path:
                    self._note_path(
                        path,
                        "command",
                        call.name,
                        id_match.group(1).decode() if id_match else None,
                        call.intent,
                        seq,
                        ts,
                        None,
                    )
            return
        if tool == "browser" and not _BROWSER_FACT_PROBE.search(line):
            return
        row = _row(line)
        payload = row.get("payload") if row else None
        if not isinstance(payload, dict) or payload.get("is_error"):
            return
        ts = _float(row.get("ts")) or 0.0 if row else 0.0
        call_id = _str(payload.get("tool_call_id"))
        call = self.pending.pop(call_id, None) if call_id else None
        intent = call.intent if call else None
        provider = payload.get("provider_payload")
        details = provider.get("details") if isinstance(provider, dict) else None
        details = details if isinstance(details, dict) else {}
        if tool in ("write", "edit"):
            path = resolve_path(str(details.get("path") or ""), cwd="", home=self.home)
            if path:
                self._note_path(
                    path,
                    "written" if tool == "write" else "edited",
                    tool,
                    call_id,
                    intent,
                    seq,
                    ts,
                    None,
                )
        elif tool == "browser":
            self._on_browser_facts(seq, ts, call_id, details)
        else:
            self._on_artifact_facts(seq, ts, tool, call_id, details, payload.get("content"))

    def _on_browser_facts(
        self, seq: int, ts: float, call_id: str | None, details: Mapping[str, Any]
    ) -> None:
        files = details.get("files")
        if isinstance(files, list) and "directory" in details:
            # A DOWNLOAD (``directory`` is its marker; ``files`` alone is the upload-attach
            # receipt, which names files the model SENT, not ones it produced).
            for fact in files:
                if isinstance(fact, dict):
                    path = resolve_path(str(fact.get("path") or ""), cwd="", home=self.home)
                    if path:
                        self._note_path(
                            path,
                            "download",
                            "browser",
                            call_id,
                            None,
                            seq,
                            ts,
                            _int(fact.get("bytes")),
                        )
            return
        path = resolve_path(str(details.get("path") or ""), cwd="", home=self.home)
        if path:
            self._note_path(
                path, "screenshot", "browser", call_id, None, seq, ts, _int(details.get("bytes"))
            )

    # -- digest-only artifacts ----------------------------------------------------------------

    def _on_job_result_row(self, seq: int, line: bytes) -> None:
        """Background video/audio receipts share the same durable artifact facts."""
        row = _row(line)
        payload = row.get("payload") if row else None
        if not isinstance(payload, dict):
            return
        details = payload.get("details")
        if not isinstance(details, dict):
            return
        ts = _float(row.get("ts")) or 0.0 if row else 0.0
        self._on_artifact_facts(
            seq,
            ts,
            (
                (_str(details.get("tool")) or "generate_voiceover")
                if payload.get("custom_type") == "audio_settle_v1"
                else "generate_video"
            ),
            _str(details.get("request_id") or details.get("job_id")),
            details,
            None,
        )

    def _on_artifact_facts(
        self,
        seq: int,
        ts: float,
        tool: str,
        call_id: str | None,
        details: Mapping[str, Any],
        content: Any,
    ) -> None:
        """Register digest-only artifacts from, in order of preference:

        1. ``details.artifacts[]`` - new rows. Durable by construction: compaction and pruning
           blank ``content`` but keep ``provider_payload.details`` (``transcript._pruned_entry``),
           so this is the only source that survives them.
        2. ``AttachmentContent`` blocks in ``content`` - old image rows, while not pruned.
        3. ``details.digest`` / ``digests`` - old video receipts.

        The first source that yields anything wins, so one result is never counted twice.
        """
        found = 0
        artifacts = details.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if isinstance(item, dict):
                    found += self._note_artifact(
                        item.get("digest"),
                        item.get("kind"),
                        item.get("content_type"),
                        item.get("name"),
                        item,
                        tool,
                        call_id,
                        seq,
                        ts,
                    )
        if not found and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("kind") in _ARTIFACT_KINDS:
                    found += self._note_artifact(
                        block.get("attachment"),
                        block.get("kind"),
                        block.get("content_type"),
                        block.get("name"),
                        block,
                        tool,
                        call_id,
                        seq,
                        ts,
                    )
        if not found:
            digests = details.get("digests")
            digests = digests if isinstance(digests, list) else [details.get("digest")]
            for digest in digests:
                self._note_artifact(
                    digest,
                    "video",
                    details.get("content_type"),
                    None,
                    {"duration_s": details.get("duration_s")},
                    tool,
                    call_id,
                    seq,
                    ts,
                )

    def _note_artifact(
        self,
        digest: Any,
        kind: Any,
        content_type: Any,
        name: Any,
        facts: Mapping[str, Any],
        tool: str,
        call_id: str | None,
        seq: int,
        ts: float,
    ) -> int:
        if not is_digest(digest):
            return 0
        assert isinstance(digest, str)
        resolved_kind = kind if kind in _ARTIFACT_KINDS else "image"
        label = _str(name) or f"{resolved_kind}-{digest[:8]}"
        self._upsert(
            f"a:{digest}",
            lambda: FileRecord(
                key=f"a:{digest}",
                path=None,
                digest=digest,
                name=label,
                kind=resolved_kind,
                content_type=_str(content_type),
                size=_int(facts.get("size_bytes")),
                width=_int(facts.get("width")),
                height=_int(facts.get("height")),
                duration_s=_float(facts.get("duration_s")),
                origin="generated",
                tool=tool,
                call_id=call_id,
                intent=None,
                first_seq=seq,
                last_seq=seq,
                first_ts=ts,
                last_ts=ts,
            ),
            "generated",
            tool,
            call_id,
            None,
            seq,
            ts,
            _int(facts.get("size_bytes")),
        )
        return 1

    # -- Highlights -----------------------------------------------------------------------------

    def _on_supplement_row(self, seq: int, line: bytes) -> None:
        row = _row(line)
        payload = row.get("payload") if row else None
        details = payload.get("details") if isinstance(payload, dict) else None
        if not isinstance(details, dict):
            return
        anchor, version = details.get("anchor"), details.get("version")
        if not isinstance(anchor, str) or not isinstance(version, int) or isinstance(version, bool):
            return  # the contract's reader rule: a malformed row is skipped, never coerced
        held = self.anchors.get(anchor)
        if held is not None and version < held.version:
            return  # newest version wins; a tie keeps the later row, which is this one
        ts = _float(details.get("at")) or (_float(row.get("ts")) if row else None) or 0.0
        entry = Anchor(version=version, ts=ts, seq=seq)
        if reader_disposition(details, job_live=False) != "nothing":
            for item in (details.get("files") or [])[:MAX_ANCHOR_PATHS]:
                path = item.get("path") if isinstance(item, dict) else None
                if isinstance(path, str) and path:
                    entry.files.append(
                        {
                            "path": path,
                            "name": _str(item.get("name")) or os.path.basename(path),
                            "kind": _str(item.get("kind")),
                            "size": _int(item.get("size_bytes")),
                            "why": _str(item.get("why")),
                        }
                    )
            more = details.get("more")
            entry.more = [
                m for m in (more if isinstance(more, list) else []) if isinstance(m, str)
            ][:MAX_ANCHOR_PATHS]
        self.anchors[anchor] = entry

    # -- record bookkeeping -----------------------------------------------------------------------

    def _denied_path(self, path: str) -> bool:
        if self._deny is None:
            return False
        if path in self._denied:
            return True
        if self._deny(path):
            if len(self._denied) < MAX_RECORDS:
                self._denied.add(path)
            return True
        return False

    def _note_path(
        self,
        path: str,
        origin: str,
        tool: str,
        call_id: str | None,
        intent: str | None,
        seq: int,
        ts: float,
        size: int | None,
    ) -> None:
        key = f"p:{path}"
        if key not in self.records and self._denied_path(path):
            return
        name = os.path.basename(path) or path
        self._upsert(
            key,
            lambda: FileRecord(
                key=key,
                path=path,
                digest=None,
                name=name,
                kind=kind_of(name),
                content_type=None,
                size=size,
                width=None,
                height=None,
                duration_s=None,
                origin=origin,
                tool=tool,
                call_id=call_id,
                intent=intent,
                first_seq=seq,
                last_seq=seq,
                first_ts=ts,
                last_ts=ts,
            ),
            origin,
            tool,
            call_id,
            intent,
            seq,
            ts,
            size,
        )

    def _upsert(
        self,
        key: str,
        build: Callable[[], FileRecord],
        origin: str,
        tool: str,
        call_id: str | None,
        intent: str | None,
        seq: int,
        ts: float,
        size: int | None,
    ) -> None:
        record = self.records.get(key)
        if record is None:
            if len(self.records) >= MAX_RECORDS:
                self.truncated = True
                self.dropped += 1
                return
            self.records[key] = build()
            return
        record.touches += 1
        record.last_seq = seq
        record.last_ts = ts
        if size is not None:
            record.size = size
        if _ORIGIN_RANK[origin] < _ORIGIN_RANK[record.origin]:
            record.origin, record.tool, record.call_id, record.intent = (
                origin,
                tool,
                call_id,
                intent,
            )

    # -- persistence ------------------------------------------------------------------------

    def clone(self) -> "FileScan":
        """A copy a scan may mutate while readers keep using the original.

        Published scans are treated as immutable (the service sorts them on other threads); an
        incremental refresh therefore works on a clone and publishes it whole.
        """
        other = FileScan(cwd=self.cwd, home=self.home, deny=self._deny)
        other.rows = self.rows
        other.records = {
            key: FileRecord(*(getattr(rec, name) for name in _RECORD_FIELDS))
            for key, rec in self.records.items()
        }
        other.pending = dict(self.pending)
        other.roster = self.roster
        other.anchors = {
            key: Anchor(a.version, a.ts, a.seq, list(a.files), list(a.more))
            for key, a in self.anchors.items()
        }
        other.truncated, other.dropped = self.truncated, self.dropped
        other._denied = self._denied
        return other

    def to_payload(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "roster": self.roster,
            "truncated": self.truncated,
            "dropped": self.dropped,
            "records": [
                [getattr(rec, name) for name in _RECORD_FIELDS] for rec in self.records.values()
            ],
            "pending": {
                key: [call.name, call.intent, list(call.outputs)]
                for key, call in self.pending.items()
            },
            "anchors": {
                key: [a.version, a.ts, a.seq, a.files, a.more] for key, a in self.anchors.items()
            },
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        cwd: str = "",
        home: str = "",
        deny: Callable[[str], bool] | None = None,
    ) -> "FileScan":
        """Rebuild a scan from :meth:`to_payload`. Raises ``ValueError`` on a malformed one,
        which the ledger turns into "rescan" (a cache is derived, never authoritative)."""
        scan = cls(cwd=cwd, home=home, deny=deny)
        try:
            scan.rows = int(payload["rows"])
            scan.roster = payload.get("roster") if isinstance(payload.get("roster"), dict) else None
            scan.truncated = bool(payload.get("truncated"))
            scan.dropped = int(payload.get("dropped") or 0)
            for values in payload["records"]:
                if len(values) != len(_RECORD_FIELDS):
                    raise ValueError("record arity")
                record = FileRecord(*values)
                scan.records[record.key] = record
            for key, (name, intent, outputs) in payload.get("pending", {}).items():
                scan.pending[key] = PendingCall(name, intent, tuple(outputs))
            for key, (version, ts, seq, files, more) in payload.get("anchors", {}).items():
                scan.anchors[key] = Anchor(version, ts, seq, list(files), list(more))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed session-files scan payload: {error}") from error
        return scan


__all__ = [
    "Anchor",
    "FileRecord",
    "FileScan",
    "MAX_RECORDS",
    "ORIGINS",
    "PendingCall",
    "SCAN_SCHEMA",
    "SUPPLEMENT_CUSTOM_TYPE",
    "resolve_path",
]
