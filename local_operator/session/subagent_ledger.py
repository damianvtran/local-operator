"""Durable per-lane evidence for subagent children.

WHY THIS MODULE EXISTS. A subagent child is an in-process ``Session`` with its
own transcript directory and **no pid**, so none of the runtime-death
instrumentation (``runtime-signal.json``, ``runtime-stop.json``, the stall
watchdog's per-pid files) can describe it. Two shapes therefore left no record
naming what happened:

* a child that **stalls** — progress existed only as the transient string
  ``latest_details["progress"]``, never persisted, carrying no timestamp; and
* a child that **ends without an actor or a reason** — the comms record kept
  ``outcome``/``cut_off_cause`` but no field said WHO stopped it, and the
  caller's own words were explicitly discarded (``del reason``).

This module owns the two durable artifacts that close those shapes. Both are
small JSON files written into the CHILD's transcript directory — the one
directory a child owns for its whole life, including after its parent is gone:

* ``subagent-lane-<job_id>.v1.json`` — the **launch receipt**, written when the
  child attaches and withdrawn when its runner settles. A lane receipt present
  with no stop receipt beside it is the reading "a lane ran here and never
  settled" — the only artifact that survives a hard parent death, because the
  dead parent writes nothing.
* ``subagent-stop-<job_id>.v1.json`` — the **stop receipt**, staged BEFORE a
  deliberate stop is issued, exactly as ``runtime-stop.json`` is staged before
  a runtime kill. It is the acting party's attestation: at the moment the stop
  is issued the child may be wedged and cannot record anything itself.

Both are keyed by ``job_id`` (one file per ATTEMPT), because a resume mints a
new job id while reusing the same transcript directory — a single fixed name
would be overwritten by the next attempt and erase the "stalled twice" history.

IMPORT GRAPH — STDLIB + ``local_operator.paths`` ONLY, and that is a
constraint rather than a preference. Any heavyweight writer (``subagent.py``,
``comms.py``, ``session.py``) must be able to import this without widening its
own import closure, so this module imports nothing from ``harness`` or
``session`` and nothing that pulls pydantic or asyncio. The staged write below
is a deliberate local copy of ``session.runtime.registry._staged_write`` rather
than an import of it: the registry pulls ``local_operator.procstate`` and the
run-plane namespace, which is exactly the weight this module exists to avoid.
``tests/unit/test_import_graph.py`` pins the closure with a fresh-subprocess
assertion so a future contributor cannot widen it silently.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from local_operator.paths import config_dir

#: Payload schema version, stamped into every artifact so a later reader can
#: tell which shape it is looking at. Bump only on an incompatible change.
LEDGER_VERSION = 1

#: The ``artifact`` discriminator on each payload. Distinct from the filename
#: prefix so a reader that has only the parsed dict (a spill, a log line) can
#: still tell the two apart.
LANE_ARTIFACT = "subagent-lane"
STOP_ARTIFACT = "subagent-stop"

#: Filename prefixes/suffix. The full name is
#: ``<prefix><job_id><suffix>`` — one file per attempt, see the module docstring.
LANE_PREFIX = "subagent-lane-"
STOP_PREFIX = "subagent-stop-"
RECEIPT_SUFFIX = ".v1.json"

#: The per-lane stall bound, in seconds, recorded on the lane receipt and
#: rendered beside the idle reading once the idle time exceeds it. It is
#: ADVISORY in v1 — nothing acts on it — and deliberately per-lane rather than
#: the runtime's process-global watchdog: the runtime timer is one bound per
#: pid, and a single wedged lane must not be able to silence a CI stage's only
#: bound. 900 s discriminates the measured lane events (the shortest was ~45
#: minutes); the known false positive is a legitimately long single tool call,
#: which is why the surfaces print the bound they applied instead of pretending
#: the reading is exact.
SUBAGENT_LANE_BOUND_S = 900.0

#: ``bound_source`` values: where the bound above came from.
LANE_BOUND_SOURCE_DEFAULT = "lane-default"
LANE_BOUND_SOURCE_ENV = "env-override"

#: Environment variable an operator can set to override the advisory lane bound
#: for the current runtime, without a code change (the value is recorded on the
#: receipt so a post-mortem reads the bound that ACTUALLY applied).
LANE_BOUND_ENV = "LOCAL_OPERATOR_SUBAGENT_LANE_BOUND_S"

#: NOTE — there is deliberately NO ``PROGRESS_PERSIST_S`` here. The design's
#: contested option was a throttled durable progress write; it is REFUSED by an
#: existing pinned invariant (progress is not a field the resume projection
#: schedules persistence for — see ``_notify_transient_job_change`` and
#: ``test_task_notification_callers_classify_transient_and_durable_mutations``),
#: so the progress stamp ships LIVE-ONLY. Durability comes from the roster moves
#: that already persist and, for the hard-death shape, from the lane receipt.

#: The ``detail`` note a reconcile pass attaches to a record whose lane receipt
#: survived with no stop receipt and no recorded outcome. This is the reading
#: that replaces "a row stuck running with nothing to say" after a hard parent
#: death (SIGKILL/crash).
LANE_NEVER_SETTLED_DETAIL = "a lane was recorded here and never settled"


def lane_bound_s() -> float:
    """The advisory lane bound, honouring the env override when it parses.

    Read at WRITE time (not import time) so a test or an operator can move the
    bound for one run, and so the receipt records the value that actually
    applied rather than whatever was set when the module first loaded.
    """
    raw = os.environ.get(LANE_BOUND_ENV)
    if raw:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return SUBAGENT_LANE_BOUND_S


def lane_bound_source() -> str:
    """``bound_source`` for the receipt: ``lane-default`` or ``env-override``."""
    raw = os.environ.get(LANE_BOUND_ENV)
    if raw:
        try:
            if float(raw) > 0:
                return LANE_BOUND_SOURCE_ENV
        except (TypeError, ValueError):
            pass
    return LANE_BOUND_SOURCE_DEFAULT


def _receipt_name(prefix: str, job_id: str) -> str:
    return f"{prefix}{job_id}{RECEIPT_SUFFIX}"


def lane_receipt_path(child_dir: Path, job_id: str) -> Path:
    """Where one attempt's lane receipt lives."""
    return Path(child_dir) / _receipt_name(LANE_PREFIX, job_id)


def stop_receipt_path(child_dir: Path, job_id: str) -> Path:
    """Where one attempt's stop receipt lives."""
    return Path(child_dir) / _receipt_name(STOP_PREFIX, job_id)


def _staged_write(target: Path, payload: Any) -> None:
    """Write ``payload`` as JSON to ``target`` staged, 0600.

    The one write shape this artifact family uses (and a deliberate local copy
    of ``registry._staged_write`` — see the module docstring): a temp file in
    the SAME directory, ``json.dump``, ``chmod 0600``, ``os.replace`` — so a
    reader sees either the old bytes or the new ones and never a torn file. The
    target's directory is NOT created here: a writer that invented a directory
    would conjure a conversation directory that every session listing then shows
    as an empty session.

    NO FSYNC, and the guarantee is stated at that strength: what is promised is
    PROCESS-durability (the bytes are in the page cache; every reader sees one
    whole version), which is the same strength every runtime artifact here
    already carries.
    """
    directory = target.parent
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --- lane receipt (a3) -------------------------------------------------------


def build_lane_payload(
    *,
    job_id: str,
    label: str = "",
    agent_role: str = "",
    child_session_id: str = "",
    parent_session_id: str = "",
    parent_job_id: str | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    """The launch receipt's payload — the facts a post-mortem needs to name the
    lane and the bound that applied to it."""
    return {
        "version": LEDGER_VERSION,
        "artifact": LANE_ARTIFACT,
        "job_id": job_id,
        "label": label,
        "agent_role": agent_role,
        "child_session_id": child_session_id,
        "parent_session_id": parent_session_id,
        "parent_job_id": parent_job_id,
        "started_at": time.time() if started_at is None else started_at,
        "bound_s": lane_bound_s(),
        "bound_source": lane_bound_source(),
        "written_at": time.time(),
    }


def write_lane_receipt(child_dir: Path, payload: Mapping[str, Any]) -> Path | None:
    """Stage-write a lane receipt. Best-effort: returns the path, or ``None``.

    A launch must never fail because evidence could not be written — the child
    is what matters, the receipt is what a later reader gets — so this swallows
    any write failure (a missing directory, a read-only mount, a full disk) and
    reports it by return value rather than by raising.
    """
    target = lane_receipt_path(Path(child_dir), str(payload.get("job_id") or ""))
    try:
        _staged_write(target, dict(payload))
    except Exception:  # noqa: BLE001 — a launch must not fail over evidence
        return None
    return target


def withdraw_lane_receipt(child_dir: Path, job_id: str) -> None:
    """Take one attempt's lane receipt back on settle. Best-effort.

    A settled lane is not evidence of a stall, so the receipt goes as soon as
    the runner settles; a leftover file is exactly the "ran here and never
    settled" reading the boot reconcile pass looks for.
    """
    try:
        lane_receipt_path(Path(child_dir), job_id).unlink()
    except OSError:
        pass


def read_lane_receipts(child_dir: Path) -> list[dict[str, Any]]:
    """Every lane receipt in one child directory, oldest first.

    Tolerant by design (mirrors ``registry.read_stop_marker``): an unreadable or
    malformed receipt means "no usable evidence", never an exception on a boot
    path.
    """
    return _read_receipts(Path(child_dir), LANE_PREFIX)


# --- stop receipt (b2) -------------------------------------------------------


def build_stop_payload(
    *,
    job_id: str,
    label: str = "",
    child_session_id: str = "",
    parent_session_id: str = "",
    at: float | None = None,
    deliberate: bool = True,
    actor: str = "",
    mechanism: str = "cancel",
    reason: str = "",
) -> dict[str, Any]:
    """The stop receipt's payload — modelled field-for-field on
    ``control._stop_marker_payload``, the family's one documented writer.

    ``deliberate`` is True for every child stop the harness issues through the
    attested paths (the same reading a user-requested runtime stop gets): the
    acting side is stating that this child was stopped ON PURPOSE, which is the
    fact that lets a later reader tell it from a casual death.
    """
    argv0 = os.path.basename(sys.argv[0] or "") or sys.executable
    command = " ".join(sys.argv)
    return {
        "version": LEDGER_VERSION,
        "artifact": STOP_ARTIFACT,
        "job_id": job_id,
        "label": label,
        "child_session_id": child_session_id,
        "parent_session_id": parent_session_id,
        "at": time.time() if at is None else at,
        "deliberate": bool(deliberate),
        "actor": actor,
        "mechanism": mechanism,
        "reason": reason,
        "killer": {"pid": os.getpid(), "argv0": argv0, "command": command},
    }


def write_stop_receipt(child_dir: Path, payload: Mapping[str, Any]) -> Path | None:
    """Stage-write a stop receipt. Best-effort: returns the path, or ``None``.

    The stop MUST NOT fail because evidence could not be written (mirroring
    ``registry.write_stop_marker``'s caller guard), so failure is reported by
    return value.
    """
    target = stop_receipt_path(Path(child_dir), str(payload.get("job_id") or ""))
    try:
        _staged_write(target, dict(payload))
    except Exception:  # noqa: BLE001 — the stop must not fail over evidence
        return None
    return target


def withdraw_stop_receipt(child_dir: Path, job_id: str) -> None:
    """Take one attempt's stop receipt back when the stop it attested did NOT
    happen. Best-effort.

    Mirrors ``control._withdraw_staged_stop_marker``: a staged receipt for a
    stop that was refused (the job was already settled or unknown) is a lie, and
    would make a later involuntary death read as the caller's own act.
    """
    try:
        stop_receipt_path(Path(child_dir), job_id).unlink()
    except OSError:
        pass


def read_stop_receipts(child_dir: Path) -> list[dict[str, Any]]:
    """Every stop receipt in one child directory, oldest first. Tolerant."""
    return _read_receipts(Path(child_dir), STOP_PREFIX)


def _read_receipts(child_dir: Path, prefix: str) -> list[dict[str, Any]]:
    """Parse every ``<prefix>*<RECEIPT_SUFFIX>`` file, malformed ones skipped."""
    try:
        entries = sorted(child_dir.glob(f"{prefix}*{RECEIPT_SUFFIX}"))
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for entry in entries:
        try:
            data = json.loads(entry.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


# --- boot reconcile (2.5) ----------------------------------------------------


@dataclass(frozen=True)
class LaneEvidence:
    """What the on-disk receipts say about ONE record, resolved at boot.

    Returned by :func:`reconcile_lane_evidence` and applied by the caller
    (``SubagentComms.apply_lane_evidence``) so this module stays free of any
    dependency on the comms record type.
    """

    job_id: str
    #: Recovered attribution from a stop receipt: the actor token and the
    #: caller's own words. Empty when no stop receipt named one.
    ended_by: str = ""
    cancel_reason: str = ""
    #: True when a lane receipt survived with no stop receipt and no recorded
    #: outcome — "a lane ran here and never settled".
    never_settled: bool = False


def reconcile_lane_evidence(
    records: Iterable[Mapping[str, Any]],
    root: Path | None = None,
) -> list[LaneEvidence]:
    """Read every record's lane/stop receipts and say what they can recover.

    ``records`` are the durable record dicts (``SubagentComms.snapshot()``
    shape): each must carry ``job_id``, ``session_dir`` and optionally
    ``outcome``. Read-only and pure over dicts plus the filesystem — it changes
    no status and mints no state, which is what keeps it off every status
    consumer. ``root`` is only a fallback base for a RELATIVE ``session_dir``
    (an absolute one is used as given); it defaults to the config root.

    The two recoveries, and why each is load-bearing:

    * a **stop receipt** present -> the actor and reason are recovered onto the
      record even when the sidecar never captured them (a crash between the
      stamp and the persist), which is what makes (b2) durable rather than
      merely live;
    * a **lane receipt** present with no stop receipt and no recorded outcome ->
      the record is flagged ``never_settled``: the parent died before the lane
      could settle and nothing attests to how, which is the hard-kill shape no
      other artifact can describe.
    """
    base = Path(root) if root is not None else None
    out: list[LaneEvidence] = []
    for record in records:
        job_id = str(record.get("job_id") or "")
        if not job_id:
            continue
        raw_dir = record.get("session_dir")
        if not raw_dir:
            continue
        child_dir = Path(str(raw_dir))
        if not child_dir.is_absolute() and base is not None:
            child_dir = base / child_dir
        stops = read_stop_receipts(child_dir)
        outcome = record.get("outcome")
        if stops:
            # Newest stop wins: a resume mints a new job id, but a record folded
            # across attempts could in principle see more than one.
            latest = max(stops, key=lambda item: float(item.get("at") or 0.0))
            out.append(
                LaneEvidence(
                    job_id=job_id,
                    ended_by=str(latest.get("actor") or ""),
                    cancel_reason=str(latest.get("reason") or ""),
                )
            )
            continue
        if outcome is None and read_lane_receipts(child_dir):
            out.append(LaneEvidence(job_id=job_id, never_settled=True))
    return out


def default_config_root() -> Path:
    """The config root :func:`reconcile_lane_evidence` falls back to."""
    return config_dir()


# --- surface rendering -------------------------------------------------------


def format_duration(seconds: float) -> str:
    """A compact, human duration: ``4s``, ``47m``, ``3h``.

    Deliberately coarse. The reading it feeds is advisory (see
    :data:`SUBAGENT_LANE_BOUND_S`), so a reader comparing it against a bound
    needs magnitude, not precision, and a compact form keeps the roster line
    from shearing its column. Negatives clamp to ``0s``.
    """
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.0f}h"
    return f"{seconds / 86400:.0f}d"


def idle_clause(
    last_progress_at: float | None,
    now: float,
    *,
    bound_s: float | None = None,
) -> str | None:
    """The idle reading for a RUNNING row, or ``None`` when there is none.

    THE reader-semantics rule, stated here because it is load-bearing: an idle
    clause is emitted only when ``last_progress_at`` is a real stamp. A job that
    has never reported progress is "no progress recorded" — a different fact
    from "silent for a while" — and must not read as stalled, so ``None`` in
    returns ``None`` out.

    When idle exceeds ``bound_s`` the clause says ``no progress for 47m (bound
    15m)`` rather than a bare ``idle 47m``: the bound is what makes the reading
    honest after the config moves, and printing it is the operator's only cue
    that a reading is advisory and a long tool call can trip it.
    """
    if last_progress_at is None:
        return None
    idle = max(now - float(last_progress_at), 0.0)
    if bound_s is not None and idle > float(bound_s):
        return f"no progress for {format_duration(idle)} (bound {format_duration(bound_s)})"
    return f"idle {format_duration(idle)}"
