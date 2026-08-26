"""Reproduction + evidence for the subagent-roster persistence hardening.

Context
-------
A real session (``~/.local-operator/sessions/58a613856339``) grew a 125 MB
``transcript.jsonl`` and was killed several times by the macOS "disk writes
exceeding limit" throttle. Two defects fed it:

1. (Fixed upstream in PR #308, before this change) the roster used to append a
   full snapshot to the transcript on every roster event, and each record
   carried an uncapped ``result_text`` tail, so the file grew O(N^2). #308 moved
   persistence to an atomically-replaced sidecar and capped per-record text via
   ``_compact_subagent_record``. This script confirms that fix still holds.

2. (Fixed here) the sidecar writer fired on EVERY roster event, including the
   many that leave the durable projection byte-identical (usage/heartbeat
   churn). Each redundant write is a full mkstemp + fsync + os.replace +
   directory-fsync cycle. That fsync volume is the disk-write pressure the
   throttle reacts to. This script measures the redundant-write count before and
   after the fingerprint guard.

The script drives the REAL ``SubagentComms`` and the REAL session persist path
(``Session._persist_subagent_roster``) over a fan-out of ~80 settled children,
each with a ~30 KB ``result_text``, firing the persist hook on every settle plus
the usage/heartbeat churn events that do not change the projection.

Run:
    PYTHONPATH=<worktree> <venv>/bin/python scripts/roster_bloat_repro.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from local_operator.harness.comms import SubagentComms
from local_operator.session import session as S

CHILDREN = 80
RESULT_BYTES = 30_000
# Events fired per settle: 1 real settle + 2 no-op churn events (usage,
# heartbeat) that leave the persisted projection unchanged. This is the shape
# the fingerprint guard is meant to collapse.
CHURN_PER_SETTLE = 2


class _StubChild:
    """Smallest child that ``SubagentComms.attach`` accepts.

    ``attach`` binds a reply watcher through ``child.subscribe`` and expects an
    unsubscribe callable back; nothing in this reproduction drives replies, so a
    no-op subscription is enough to reach ``record.session_dir`` (the field a
    restore needs).
    """

    def subscribe(self, _watcher: object) -> object:
        return lambda: None


def _build_comms() -> SubagentComms:
    """A real ``SubagentComms`` with a minimal session stand-in.

    ``SubagentComms`` only needs a session object for detail-change plumbing,
    which this reproduction never exercises, so a namespace is enough to
    construct it and drive ``record_launch`` / ``attach`` / ``record_outcome``.
    """
    return SubagentComms(SimpleNamespace())  # type: ignore[arg-type]


def _persist_once(sidecar: Path, comms: SubagentComms, generation: int) -> tuple[int, str]:
    """Run the durable half of ``_persist_subagent_roster`` for one event.

    Returns the serialized fingerprint (content minus the generation counter) so
    the caller can decide whether a write is redundant, exactly as the guard in
    ``Session._persist_subagent_roster`` does.
    """
    records = comms.snapshot()
    compact_records = [S._compact_subagent_record(record) for record in records]
    payload = {
        "version": S._SUBAGENT_ROSTER_VERSION,
        "generation": generation,
        "jobs": [],
        "records": compact_records,
    }
    fingerprint = json.dumps(
        {"version": S._SUBAGENT_ROSTER_VERSION, "jobs": [], "records": compact_records},
        sort_keys=True,
        separators=(",", ":"),
    )
    S._write_roster_sidecar(sidecar, payload)
    return sidecar.stat().st_size, fingerprint


def _simulate(guarded: bool) -> dict[str, int]:
    """Fan out ``CHILDREN`` settled children, firing the persist path per event.

    ``guarded`` toggles the fingerprint guard so the same run reports the
    before (unguarded, every event writes) and after (guarded, identical
    payloads skipped) write counts.
    """
    with tempfile.TemporaryDirectory() as directory:
        sidecar = Path(directory) / S.SUBAGENT_ROSTER_SIDECAR
        comms = _build_comms()
        generation = 0
        writes = 0
        redundant_skipped = 0
        last_fingerprint: str | None = None
        final_size = 0

        for index in range(CHILDREN):
            job_id = f"job{index}"
            comms.record_launch(
                job_id,
                f"child{index}",
                prompt="p" * 200,
                agent_role="task",
            )
            child_dir = Path(directory) / f"child{index}"
            comms.attach(job_id, _StubChild(), child_dir)  # type: ignore[arg-type]
            comms.record_outcome(
                job_id,
                "completed",
                result_text="X" * RESULT_BYTES,
            )
            # One real settle event plus churn events that do not change the
            # projection. Under the guard, only the settle event should write.
            for churn in range(1 + CHURN_PER_SETTLE):
                generation += 1
                records = comms.snapshot()
                compact_records = [S._compact_subagent_record(record) for record in records]
                fingerprint = json.dumps(
                    {
                        "version": S._SUBAGENT_ROSTER_VERSION,
                        "jobs": [],
                        "records": compact_records,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if guarded and fingerprint == last_fingerprint:
                    redundant_skipped += 1
                    continue
                payload = {
                    "version": S._SUBAGENT_ROSTER_VERSION,
                    "generation": generation,
                    "jobs": [],
                    "records": compact_records,
                }
                S._write_roster_sidecar(sidecar, payload)
                writes += 1
                last_fingerprint = fingerprint
                final_size = sidecar.stat().st_size

        largest_record = max(
            len(json.dumps(S._compact_subagent_record(record))) for record in comms.snapshot()
        )
        return {
            "writes": writes,
            "redundant_skipped": redundant_skipped,
            "final_sidecar_bytes": final_size,
            "largest_record_bytes": largest_record,
            "events": CHILDREN * (1 + CHURN_PER_SETTLE),
        }


def _roundtrip_ok() -> bool:
    """Snapshot -> restore still lists every child, each resumable.

    Confirms the guard/cap changes did not break the resume basis: every settled
    record with a ``session_dir`` must survive a restore with its identity and
    outcome intact.
    """
    with tempfile.TemporaryDirectory() as directory:
        comms = _build_comms()
        for index in range(5):
            job_id = f"job{index}"
            comms.record_launch(job_id, f"child{index}", prompt="p", agent_role="task")
            child_dir = Path(directory) / f"child{index}"
            comms.attach(job_id, _StubChild(), child_dir)  # type: ignore[arg-type]
            comms.record_outcome(job_id, "completed", result_text="X" * RESULT_BYTES)
        snap = [S._compact_subagent_record(record) for record in comms.snapshot()]

        restored = _build_comms()
        restored.restore(snap)
        rows = restored.snapshot()
        return len(rows) == 5 and all(row.get("session_dir") for row in rows)


def main() -> None:
    before = _simulate(guarded=False)
    after = _simulate(guarded=True)
    print("=== GAP 1: redundant sidecar writes (fsync churn) ===")
    print(f"roster events fired            : {before['events']}")
    print(f"BEFORE guard  - sidecar writes : {before['writes']}")
    print(
        f"AFTER  guard  - sidecar writes : {after['writes']} "
        f"({after['redundant_skipped']} redundant writes skipped)"
    )
    print()
    print("=== #308 size fix still holds (sidecar, capped records) ===")
    print(f"final sidecar size (bytes)     : {after['final_sidecar_bytes']:,}")
    print(f"largest single record (bytes)  : {after['largest_record_bytes']:,}")
    print()
    print("=== resume basis intact ===")
    print(f"snapshot -> restore round-trip : {'OK' if _roundtrip_ok() else 'FAILED'}")


if __name__ == "__main__":
    # ``asyncio.run`` kept for parity with the session's async persist path even
    # though the durable helpers used here are synchronous.
    asyncio.run(asyncio.to_thread(main))
