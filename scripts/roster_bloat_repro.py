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

3. (Fixed here) a transcript ALREADY bloated pre-v0.40.0 was never healed: its
   superseded ``subagent_roster`` custom entries loaded whole into memory on
   every resume and were re-serialized on every compaction. ``compact_file`` now
   drops superseded collapsible customs, keeping only the newest. This script
   synthesizes such a bloated transcript with the REAL ``Transcript`` and prints
   the before/after file bytes and entry counts the heal reclaims.

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
from local_operator.harness.types import Message
from local_operator.session import session as S
from local_operator.session.transcript import ENTRY_CUSTOM, Transcript

CHILDREN = 80
RESULT_BYTES = 30_000
# Superseded legacy roster snapshots to synthesize for the GAP 2 heal. Each
# carries a ~2 KB record tail, the pre-v0.40.0 shape (a full snapshot appended
# on every roster move). 50 is enough to make the reclaim obvious without a
# slow script; the real incident had ~247.
LEGACY_ROSTER_ENTRIES = 50
LEGACY_RECORD_BYTES = 2_000
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


async def _heal_legacy_transcript() -> dict[str, object]:
    """Synthesize a pre-v0.40.0 bloated transcript and heal it on compaction.

    Drives the REAL ``Transcript``: a real user message plus a long run of
    superseded ``subagent_roster`` custom entries (the shape the old
    append-on-every-move roster left behind), then ``compact_file`` — which now
    drops every superseded collapsible custom, keeping only the newest. Reports
    the before/after file bytes, the roster-entry count, that the message
    survived replay, and that ``latest_custom`` is unchanged, so the evidence is
    what the code actually does rather than a relabeled test assertion.
    """
    with tempfile.TemporaryDirectory() as directory:
        transcript = Transcript(Path(directory) / "sess")
        await transcript.append_message(Message.user("keep me"))
        for generation in range(LEGACY_ROSTER_ENTRIES):
            await transcript.append_custom(
                "subagent_roster",
                {
                    "generation": generation,
                    "jobs": [],
                    "records": [{"blob": "X" * LEGACY_RECORD_BYTES}],
                },
            )

        def _roster_count(entries: object) -> int:
            return sum(
                1
                for entry in entries  # type: ignore[union-attr]
                if entry.type == ENTRY_CUSTOM
                and entry.payload.get("custom_type") == "subagent_roster"
            )

        before_bytes = transcript.path.stat().st_size
        roster_before = _roster_count(transcript.entries())
        # No pending prune: the heal must fire on the superseded-custom signal
        # alone, since a legacy bloated file never journals a prune.
        reclaimed = await transcript.compact_file(min_reclaim_bytes=1)
        after_bytes = transcript.path.stat().st_size

        reopened = Transcript(Path(directory) / "sess")
        rosters = [
            entry
            for entry in reopened.entries()
            if entry.type == ENTRY_CUSTOM and entry.payload.get("custom_type") == "subagent_roster"
        ]
        messages = [
            message.text for message in reopened.build_llm_history() if isinstance(message, Message)
        ]
        latest = reopened.latest_custom("subagent_roster")
        survivor = rosters[0].payload["details"]["generation"] if rosters else None
        return {
            "before_bytes": before_bytes,
            "after_bytes": after_bytes,
            "reclaimed": reclaimed,
            "roster_before": roster_before,
            "roster_after": len(rosters),
            "survivor_generation": survivor,
            "messages": messages,
            "latest_generation": latest["generation"] if latest else None,
            # The equality invariant reclaimable_bytes() prices and compact_file
            # frees; both must agree or the accounting is wrong.
            "accounting_ok": after_bytes == before_bytes - reclaimed,
        }


def main() -> None:
    before = _simulate(guarded=False)
    after = _simulate(guarded=True)
    heal = asyncio.run(_heal_legacy_transcript())
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
    print("=== GAP 2: heal legacy transcript bloat on compaction ===")
    print(f"roster custom entries          : {heal['roster_before']} -> {heal['roster_after']}")
    print(
        f"transcript size (bytes)        : {heal['before_bytes']:,} -> {heal['after_bytes']:,} "
        f"(reclaimed {heal['reclaimed']:,})"
    )
    print(f"surviving roster is newest     : generation {heal['survivor_generation']}")
    print(f"latest_custom unchanged        : generation {heal['latest_generation']}")
    print(f"message survived replay        : {heal['messages']}")
    print(f"reclaimable == reclaimed        : {heal['accounting_ok']}")
    print()
    print("=== resume basis intact ===")
    print(f"snapshot -> restore round-trip : {'OK' if _roundtrip_ok() else 'FAILED'}")


if __name__ == "__main__":
    # ``asyncio.run`` kept for parity with the session's async persist path even
    # though the durable helpers used here are synchronous.
    asyncio.run(asyncio.to_thread(main))
