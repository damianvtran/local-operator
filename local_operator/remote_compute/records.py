"""The local record of remote runs this device submitted: a JSONL file, not sqlite.

WHY THIS EXISTS AT ALL. A remote run outlives the process that submitted it and
the network that was up when it did. The desktop panel that lists runs has to
answer "what did I send, when, and where is it now?" while Radient is
unreachable, so the answers cannot live only in the server's database or in a
client object's memory: they are written here, on this machine, as they happen.

WHY JSONL RATHER THAN SQLITE. Every write here is ONE observation of ONE run —
"submitted", then "status was running", then "status was stopped" — and the
newest observation of a run id wins. That is an append and a fold, which a
line-per-record text file does natively; sqlite would buy transactions, a binary
format, a schema migration path and a second thing that can be locked against
the config writers, for a table whose expected size is a few hundred rows. The
format also stays support-readable: a run that will not list is one `cat` away
from being understood, and a torn final line (the one crash a plain append can
leave) is skipped by the fold rather than corrupting the store.

WHAT IS NOT HERE. No prompt text, no bearer, no run token, no tunnel token, no
response body: a repository URL and commit, a SCRUBBED one-line summary of the
prompt, the size, the API origin, the run id, the network id, the status the
server reported and two timestamps. The summary goes through the repo's
credential-shape scrubber before it is stored (see :func:`prompt_summary`),
because an operator's prompt can quote a secret and this file is on disk.

CONCURRENCY IS WHOLE-LINE, AND ONLY THAT. A writer appends one line (under a
kilobyte) with ``O_APPEND``; two processes therefore interleave at line
boundaries rather than mid-record on POSIX, and on any platform a line that
fails to parse is skipped. Compaction (the only rewrite) is a same-directory
temp file plus ``os.replace``, so a reader sees the old file or the new one and
never a half-written one.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

#: Where the record lives, under the app's config directory.
RECORD_SUBDIRECTORY = "remote-compute"
RECORD_FILENAME = "runs.jsonl"

#: How much of a prompt is stored. Enough to recognise the run in a list, short
#: enough that the summary cannot become a copy of the prompt.
SUMMARY_CHARS = 200

#: How many runs the file keeps once it is compacted. The panel lists recent
#: runs; the server remains the authority for anything older.
KEPT_RUNS = 500

#: The size at which the file is rewritten to one line per kept run.
#:
#: SIZE, NOT A LINE COUNT, because size is the thing being bounded and every
#: append already costs a ``stat``. A stored line measures ~490 bytes in these
#: tests (a 40-character SHA, an https origin, a capped summary), so 512 KB is
#: roughly a thousand observations of ~500 runs — about a month of writes for an
#: active panel — and the rewrite brings it back to about :data:`KEPT_RUNS`
#: lines. Reading the whole file on EVERY append to count lines
#: instead would make the store quadratic in the number of runs.
MAX_FILE_BYTES = 512_000

#: Per-process serialisation of the append. Cross-process safety comes from
#: ``O_APPEND`` plus the fold; this lock keeps two threads of one process (a
#: status poll beside a submit) from interleaving their own writes.
_APPEND_LOCK = threading.Lock()

_WHITESPACE = re.compile(r"\s+")


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """One run as this device last recorded it.

    ``created_at`` and ``updated_at`` are ISO-8601 UTC with a ``Z`` suffix, so a
    reader needs no locale or timezone guess. ``status`` is the server's word,
    stored as given and never interpreted here: the panel and the CLI already
    have the vocabulary for what a status means.
    """

    run_id: str
    repo_url: str
    repo_sha: str
    prompt_summary: str
    size: str
    created_at: str
    api_origin: str
    network_id: str | None = None
    status: str | None = None
    updated_at: str | None = None
    deleted: bool = False

    def to_json(self) -> dict[str, Any]:
        """The stored shape: every field, so a record is readable on its own."""
        return {
            "run_id": self.run_id,
            "repo_url": self.repo_url,
            "repo_sha": self.repo_sha,
            "prompt_summary": self.prompt_summary,
            "size": self.size,
            "created_at": self.created_at,
            "api_origin": self.api_origin,
            "network_id": self.network_id,
            "status": self.status,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
        }

    @classmethod
    def from_json(cls, payload: object) -> "RunRecord | None":
        """A record from a stored line, or ``None`` when the line is not one.

        EVERY FIELD IS CHECKED, and a line that fails is skipped rather than
        repaired: this file can be hand-edited, truncated by a crash, or written
        by an older build, and none of those is a reason to stop listing the
        runs that did parse.
        """
        if not isinstance(payload, dict):
            return None
        required = ("run_id", "repo_url", "repo_sha", "prompt_summary", "size", "created_at")
        values: dict[str, str] = {}
        for field_name in required:
            value = payload.get(field_name)
            if not isinstance(value, str) or not value:
                return None
            values[field_name] = value
        api_origin = payload.get("api_origin")
        if not isinstance(api_origin, str) or not api_origin:
            return None
        network_id = payload.get("network_id")
        status = payload.get("status")
        updated_at = payload.get("updated_at")
        return cls(
            run_id=values["run_id"],
            repo_url=values["repo_url"],
            repo_sha=values["repo_sha"],
            prompt_summary=values["prompt_summary"],
            size=values["size"],
            created_at=values["created_at"],
            api_origin=api_origin,
            network_id=network_id if isinstance(network_id, str) and network_id else None,
            status=status if isinstance(status, str) and status else None,
            updated_at=updated_at if isinstance(updated_at, str) and updated_at else None,
            deleted=payload.get("deleted") is True,
        )


def records_path(config_dir: Path | None = None) -> Path:
    """``<config dir>/remote-compute/runs.jsonl``.

    The directory is resolved per call, not captured at import: the config
    directory is overridable, and a module constant would freeze whatever the
    first importer saw — including a test session's temporary root.
    """
    if config_dir is None:
        from local_operator.paths import config_dir as default_config_dir

        config_dir = default_config_dir()
    return config_dir / RECORD_SUBDIRECTORY / RECORD_FILENAME


def prompt_summary(prompt: str) -> str:
    """The stored form of a prompt: one line, capped, and shape-scrubbed.

    THE SCRUB IS NOT DECORATION. A prompt is free text an operator typed, so it
    can quote an environment variable, a DSN or a token — and this summary is
    written to a file, read by a panel and copied into bug reports. The scrubber
    is the repo's existing credential-SHAPE pass, imported lazily because it is a
    large stdlib-only module and nothing on the package's light import path
    (``is_enabled``) should pay for it.
    """
    from local_operator.redaction_shapes import scrub_shapes

    collapsed = _WHITESPACE.sub(" ", prompt).strip()
    scrubbed = scrub_shapes(collapsed)
    return scrubbed[:SUMMARY_CHARS]


def _now() -> str:
    """The timestamp format every field uses: ISO-8601 UTC, second precision."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load(config_dir: Path | None = None) -> list[RunRecord]:
    """Every recorded run, newest first, folded to one row per run id.

    NEWEST IS DECIDED BY ``created_at``, then by position in the file for ties:
    two runs submitted in the same second are still ordered, because the fold
    preserves the file's order and a later line is a later observation. A
    ``deleted`` newest observation wins too — the run is kept, marked, so the
    panel can say it was deleted instead of silently losing the row.
    """
    return _fold(_read_lines(records_path(config_dir)))


def _read_lines(path: Path) -> list[dict[str, Any]]:
    """Every parsable object in the file, in file order; a missing file is empty."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        # An unreadable record is not a reason to fail a panel: it lists nothing.
        return []
    lines: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            lines.append(payload)
    return lines


def _fold(payloads: list[dict[str, Any]]) -> list[RunRecord]:
    """One record per run id, newest run first.

    ORDERED BY ``(created_at, file position)``, both descending. ``created_at``
    has second precision, so two runs submitted inside one second tie; without
    the file position as a tiebreak the fold would list the OLDER of the two
    first, and a panel that submits a batch would show it backwards.
    """
    latest: dict[str, RunRecord] = {}
    positions: dict[str, int] = {}
    for position, payload in enumerate(payloads):
        record = RunRecord.from_json(payload)
        if record is None:
            continue
        latest[record.run_id] = record
        positions[record.run_id] = position
    return sorted(
        latest.values(),
        key=lambda record: (record.created_at, positions[record.run_id]),
        reverse=True,
    )


def record_submission(
    *,
    run_id: str,
    repo_url: str,
    repo_sha: str,
    prompt: str,
    size: str,
    api_origin: str,
    network_id: str | None = None,
    status: str | None = None,
    config_dir: Path | None = None,
) -> RunRecord:
    """Write the submission observation for ``run_id`` and return it.

    ``api_origin`` is recorded because a run answered by a non-production Radient
    is a different run: the panel has to be able to say where a row came from,
    and the client's base is resolved per process rather than stored globally.
    """
    record = RunRecord(
        run_id=run_id,
        repo_url=repo_url,
        repo_sha=repo_sha,
        prompt_summary=prompt_summary(prompt),
        size=size,
        created_at=_now(),
        api_origin=api_origin,
        network_id=network_id,
        status=status,
        updated_at=None,
    )
    _append(record, config_dir=config_dir)
    return record


def record_status(
    payload: dict[str, Any], *, api_origin: str, config_dir: Path | None = None
) -> RunRecord | None:
    """Record a status read of a run the file already knows, else ``None``.

    A READ OF A RUN THIS DEVICE DID NOT SUBMIT IS NOT RECORDED: the panel lists
    this device's runs, and the server's own listing is the place for everything
    else. The run's id and status are the only values taken from ``payload`` —
    the rest of a run object belongs to the server, not to this file.
    """
    identifier = payload.get("id")
    status = payload.get("status")
    if not isinstance(identifier, str) or not identifier:
        return None
    existing = _latest(identifier, config_dir=config_dir)
    if existing is None:
        return None
    updated = dataclasses.replace(
        existing,
        status=status if isinstance(status, str) and status else existing.status,
        updated_at=_now(),
    )
    _append(updated, config_dir=config_dir)
    return updated


def mark_deleted(run_id: str, *, config_dir: Path | None = None) -> RunRecord | None:
    """Mark a run deleted in the record, keeping the row.

    KEPT, NOT REMOVED. A row that vanished would look like this device forgot a
    run it submitted; a row marked deleted says what happened, and the server's
    record is gone either way.
    """
    existing = _latest(run_id, config_dir=config_dir)
    if existing is None:
        return None
    updated = dataclasses.replace(existing, deleted=True, updated_at=_now())
    _append(updated, config_dir=config_dir)
    return updated


def _latest(run_id: str, *, config_dir: Path | None = None) -> RunRecord | None:
    """The newest recorded observation of one run, or ``None``."""
    found: RunRecord | None = None
    for payload in _read_lines(records_path(config_dir)):
        record = RunRecord.from_json(payload)
        if record is not None and record.run_id == run_id:
            found = record
    return found


def _append(record: RunRecord, *, config_dir: Path | None = None) -> None:
    """Append one observation, compacting first when the file has grown."""
    path = records_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
    with _APPEND_LOCK:
        _compact_if_grown(path)
        with open(path, "ab") as handle:
            handle.write(line.encode("utf-8"))


def _compact_if_grown(path: Path) -> None:
    """Rewrite the file as one line per kept run once it passes its size bound.

    THE COMPACTION IS THE BOUND on the file: appends are unbounded on their own,
    so this is what keeps a long-lived install from growing a status log forever.
    A reader either sees the old file or the replaced one (``os.replace``), and
    the temp file is in the same directory so the replace is a rename.

    The size is read with ``stat`` on every append, so the common path costs one
    syscall; the file is parsed only when it is about to be rewritten.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= MAX_FILE_BYTES:
        return
    kept = _fold(_read_lines(path))[:KEPT_RUNS]
    body = "".join(
        json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True) + "\n" for record in kept
    )
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".runs-", suffix=".jsonl")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


__all__ = [
    "KEPT_RUNS",
    "MAX_FILE_BYTES",
    "RECORD_FILENAME",
    "RECORD_SUBDIRECTORY",
    "RunRecord",
    "SUMMARY_CHARS",
    "load",
    "mark_deleted",
    "prompt_summary",
    "record_status",
    "record_submission",
    "records_path",
]
