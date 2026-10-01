"""State carry-over across a move: the rebuild, the prune, and the ORDER.

The design note's §5.3 is the contract under test here, and its two hard words
are "AFTER a successful commit" (F5: the prune runs only once the source's
directory is really gone) and "the no-consume property" (a refused engage must
not advance ``fired_count``/``next_due_at``, or the destination's one legal
fire would be the second one). The cells:

* the local literals this module keeps (it is stdlib-only by design) still
  equal the canonical constants they shadow;
* the backward transcript scan answers exactly what
  ``session.transcript.read_latest_custom_entry`` answers for the custom types
  a carry-over reads;
* ``rebuild_indexes`` writes the carried rows AS STORED, is a no-op on a second
  run, and never materialises device-local monitor state;
* ``prune_after_commit`` removes the three derived things and is idempotent;
* ON A REAL MOVE over two relays: the destination's indexes exist after the
  promote while the source's are pruned — and the prune is observed with the
  session directory already gone, which is the F5 order itself.
* a REFUSED ENGAGE does not consume the row (OQ14): the sweep reaches
  ``engage_runtime``, the handoff guard refuses it, and
  ``next_due_at``/``fired_count`` are untouched, so the row still fires.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import carry
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures and helpers
    SESSION,
    Devices,
    _move,
    _owned_session,
    pair,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — the fixture `pair` reaches for
    _pair_settled,
    devices,
)

WAKE_ROW = {
    "id": "w1",
    "message": "check the deploy",
    # DUE IN AN HOUR, and computed for the same reason a constant will not do:
    # the value must be future on every run. A PAST due time is actively
    # rewritten by whichever runtime opens the session (the open-time catch-up
    # consumes an overdue occurrence and advances the row), so an e2e that arms
    # with one cannot then assert "carried as stored" against its own arm — the
    # e2e jumps the clock at its fire cell instead (``now_ms``), which is what
    # keeps "fires once at its due time" deterministic rather than a race with
    # the source's own scheduler.
    "next_due_at": int(time.time() * 1000) + 3_600_000,
    "every_ms": 86_400_000,
    "fired_count": 2,  # carried as stored — the number the source last wrote
    "created_at": 1,
    "notify": False,
}
MONITOR_ROW = {
    "id": "m1",
    "name": "queue depth",
    "tool": "bash",
    "arguments": {"command": "true"},
    "every_ms": 600_000,
    "created_at": 1,
}


def _custom_entry(custom_type: str, details: dict[str, Any], *, ts: float = 2.0) -> str:
    return json.dumps(
        {
            "id": f"e-{custom_type}",
            "ts": ts,
            "type": "custom",
            "payload": {"custom_type": custom_type, "details": details},
        },
        separators=(",", ":"),
    )


def _session_with_state(
    root: Path, session_id: str = SESSION, *, rows: int = 3, extra_lines: list[str] | None = None
) -> Path:
    """A session directory whose transcript carries wake + monitor snapshots."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"id": f"e{index}", "ts": 1.0, "type": "message", "payload": {}})
        for index in range(rows)
    ]
    lines.append(_custom_entry("wake_schedules", {"schedules": [dict(WAKE_ROW)]}))
    lines.append(_custom_entry("monitor_schedules", {"monitors": [dict(MONITOR_ROW)]}))
    lines.extend(extra_lines or [])
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


class TestLocalLiterals:
    def test_the_shadowed_constants_still_equal_their_sources(self) -> None:
        from local_operator.harness.wake import WAKE_SCHEDULES_CUSTOM_TYPE
        from local_operator.monitors.spec import MONITOR_SCHEDULES_CUSTOM_TYPE
        from local_operator.session.retention import DESKTOP_MARKER_NAME
        from local_operator.session.transcript import TRANSCRIPT_FILENAME

        assert carry.TRANSCRIPT_FILENAME == TRANSCRIPT_FILENAME
        assert carry.WAKE_SCHEDULES_CUSTOM_TYPE == WAKE_SCHEDULES_CUSTOM_TYPE
        assert carry.MONITOR_SCHEDULES_CUSTOM_TYPE == MONITOR_SCHEDULES_CUSTOM_TYPE
        assert carry.DESKTOP_MARKER_NAME == DESKTOP_MARKER_NAME

    def test_the_scanner_answers_what_the_reference_reader_answers(self, tmp_path: Path) -> None:
        from local_operator.session.transcript import read_latest_custom_entry

        directory = _session_with_state(tmp_path)
        for custom_type in ("wake_schedules", "monitor_schedules", "never_written"):
            reference = read_latest_custom_entry(directory, custom_type)
            expected = (
                dict(reference.payload.get("details") or {}) if reference is not None else None
            )
            assert carry.latest_custom_details(directory, custom_type) == expected

    def test_the_scanner_survives_the_shapes_a_live_journal_produces(self, tmp_path: Path) -> None:
        random.seed(4)
        directory = tmp_path / "sessions" / "scanner"
        directory.mkdir(parents=True)
        for trial in range(50):
            lines: list[str] = []
            for index in range(random.randint(0, 24)):
                dice = random.random()
                if dice < 0.2:
                    lines.append(
                        _custom_entry("wake_schedules", {"schedules": [dict(WAKE_ROW)]}, ts=index)
                    )
                elif dice < 0.3:
                    lines.append("{not json")
                elif dice < 0.4:
                    lines.append("")
                else:
                    lines.append(
                        json.dumps(
                            {
                                "id": f"e{index}",
                                "ts": 1.0,
                                "type": "message",
                                "payload": {"pad": "x" * random.choice([0, 300, 70_000])},
                            }
                        )
                    )
            trailer = random.choice(["", "\n"])
            (directory / "transcript.jsonl").write_text(
                "\n".join(lines) + trailer, encoding="utf-8"
            )
            from local_operator.session.transcript import read_latest_custom_entry

            reference = read_latest_custom_entry(directory, "wake_schedules")
            expected = (
                dict(reference.payload.get("details") or {}) if reference is not None else None
            )
            assert carry.latest_custom_details(directory, "wake_schedules") == expected, trial


class TestRebuild:
    def test_the_rows_are_carried_as_stored_and_the_second_run_is_a_no_op(
        self, tmp_path: Path
    ) -> None:
        from local_operator.wakes import store as wake_store

        directory = _session_with_state(tmp_path)
        tapped: list[Path] = []
        original = carry.ensure_supervisor
        try:
            carry.ensure_supervisor = (  # type: ignore[assignment]
                lambda root: tapped.append(Path(root)) or "stub"
            )
            report = carry.rebuild_indexes(tmp_path, SESSION, session_dir=directory)
        finally:
            carry.ensure_supervisor = original  # type: ignore[assignment]
        # REBUILD IS PURE FILE WORK: the supervisor install belongs to the
        # PROMOTE call site (and shells out to launchctl/systemd), so a test — or
        # a relay that only wanted the indexes — must never pay it here.
        assert tapped == []
        assert report["wakes"] == 1 and report["monitors"] == 1
        entry = wake_store.read_entry(tmp_path, SESSION)
        assert entry is not None
        rows = entry["schedules"]
        assert rows[0]["next_due_at"] == WAKE_ROW["next_due_at"]
        assert rows[0]["fired_count"] == WAKE_ROW["fired_count"]
        assert rows[0]["every_ms"] == WAKE_ROW["every_ms"]
        # IDEMPOTENT: a second rebuild rewrites nothing (the mtime is the proof —
        # an identical rewrite would be visible here even though the bytes match).
        before = wake_store.entry_path(tmp_path, SESSION).stat().st_mtime_ns
        carry.rebuild_indexes(tmp_path, SESSION, session_dir=directory)
        assert wake_store.entry_path(tmp_path, SESSION).stat().st_mtime_ns == before
        after = wake_store.read_entry(tmp_path, SESSION)
        assert after is not None and len(after["schedules"]) == 1

    def test_monitor_state_is_never_materialised(self, tmp_path: Path) -> None:
        from local_operator.monitors import store as monitor_store

        directory = _session_with_state(tmp_path)
        carry.rebuild_indexes(tmp_path, SESSION, session_dir=directory)
        entry = monitor_store.read_entry(tmp_path, SESSION)
        assert entry is not None and len(entry["monitors"]) == 1
        row = entry["monitors"][0]
        # Fresh counters, no due promise: the destination has no counters file, and
        # a zeroed due time cannot claim an instant it never computed (the silent
        # re-baseline, OQ8).
        assert row["next_due_at"] is None
        assert row["checks"] == 0 and row["deliveries"] == 0 and row["last_check_at"] == 0
        assert not (tmp_path / "monitors" / "state" / SESSION).exists()

    def test_a_transcript_with_no_snapshot_removes_the_index(self, tmp_path: Path) -> None:
        from local_operator.wakes import store as wake_store

        _session_with_state(tmp_path)
        carry.rebuild_indexes(tmp_path, SESSION)
        assert wake_store.read_entry(tmp_path, SESSION) is not None
        # The latest snapshot is the full list: a session whose transcript no
        # longer carries schedules has none, which the store spells "no file".
        directory = tmp_path / "sessions" / SESSION
        (directory / "transcript.jsonl").write_text("", encoding="utf-8")
        carry.rebuild_indexes(tmp_path, SESSION)
        assert wake_store.read_entry(tmp_path, SESSION) is None


class TestPrune:
    def test_prune_removes_the_three_derived_things_and_is_idempotent(self, tmp_path: Path) -> None:
        from local_operator.monitors import store as monitor_store
        from local_operator.wakes import store as wake_store

        _session_with_state(tmp_path)
        carry.rebuild_indexes(tmp_path, SESSION)
        state = tmp_path / "monitors" / "state" / SESSION
        state.mkdir(parents=True, exist_ok=True)
        (state / "m1.json").write_text("{}", encoding="utf-8")
        report = carry.prune_after_commit(tmp_path, SESSION)
        assert report["wakes"] and report["monitors"]
        assert wake_store.read_entry(tmp_path, SESSION) is None
        assert monitor_store.read_entry(tmp_path, SESSION) is None
        assert not state.exists()
        # Idempotent: the second prune finds nothing and removes nothing.
        again = carry.prune_after_commit(tmp_path, SESSION)
        assert not again["wakes"] and not again["monitors"]


class TestOrderingOnARealMove:
    def test_the_prune_runs_after_the_directory_is_gone_and_only_on_a_move(
        self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F5, observed on the wire: prune AFTER a successful commit, never on keep."""
        from local_operator.network import carry as carry_mod
        from local_operator.wakes import store as wake_store

        server_a, server_b = request.getfixturevalue("pair")[:2]
        _pair_settled(request.getfixturevalue("pair"), monkeypatch, role="admin")
        _owned_session(server_a, SESSION)
        source = server_a.root / "sessions" / SESSION
        # Give the source real carried state: the transcript snapshot plus the
        # derived indexes the move is about to carry and prune.
        with (source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(_custom_entry("wake_schedules", {"schedules": [dict(WAKE_ROW)]}) + "\n")
        carry_mod.rebuild_indexes(server_a.root, SESSION)

        observed: list[tuple[Path, bool]] = []
        real_prune = carry_mod.prune_after_commit

        def watching(root: Path, session_id: str) -> dict[str, Any]:
            observed.append((Path(root), (Path(root) / "sessions" / session_id).exists()))
            return real_prune(root, session_id)

        monkeypatch.setattr(carry_mod, "prune_after_commit", watching)
        # The supervisor installer shells out to launchctl/systemd; a test must
        # not install a real unit. The promote still takes the §5.3 `ensure`
        # step — it just lands on this stub — and the installed-ness itself is
        # covered where it belongs (``tests/unit/wakes/test_install.py``).
        monkeypatch.setattr(carry_mod, "ensure_supervisor", lambda root: "stubbed")

        result = _move(server_b, SESSION, monkeypatch=monkeypatch)
        assert result.get("ok"), result
        # THE ORDER ITSELF: at the instant the prune ran on the source, the source
        # directory was ALREADY GONE — the commit had landed. (Prune-first would
        # have observed True here, and would strand the source on a failed copy.)
        assert observed == [(server_a.root, False)], observed
        assert wake_store.read_entry(server_a.root, SESSION) is None
        # AND THE DESTINATION HAS THE INDEX — rebuilt from the copied transcript at
        # the promote, with the row's stored numbers intact.
        destination_entry = wake_store.read_entry(server_b.root, SESSION)
        assert destination_entry is not None
        assert destination_entry["schedules"][0]["fired_count"] == WAKE_ROW["fired_count"]
        assert destination_entry["schedules"][0]["next_due_at"] == WAKE_ROW["next_due_at"]
        # A second rebuild at the destination is the no-op the engage-on-arrival
        # open will rely on.
        before = wake_store.entry_path(server_b.root, SESSION).stat().st_mtime_ns
        carry_mod.rebuild_indexes(server_b.root, SESSION)
        assert wake_store.entry_path(server_b.root, SESSION).stat().st_mtime_ns == before

        # KEEP COPIES DO NOT PRUNE: the source still holds its conversation, so its
        # derived state must stay exactly where it was.
        _owned_session(server_a, "aabbccddeeff")
        keep_source = server_a.root / "sessions" / "aabbccddeeff"
        with (keep_source / "transcript.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(_custom_entry("wake_schedules", {"schedules": [dict(WAKE_ROW)]}) + "\n")
        carry_mod.rebuild_indexes(server_a.root, "aabbccddeeff")
        observed.clear()
        kept = _move(server_b, "aabbccddeeff", keep=True, monkeypatch=monkeypatch)
        assert kept.get("ok"), kept
        assert observed == [], "a keep copy pruned its source's derived state"
        assert wake_store.read_entry(server_a.root, "aabbccddeeff") is not None


class TestRefusedEngageDoesNotConsume:
    """OQ14 (accepted default, note §8): a refused engage must not eat the row.

    The guard predicate itself is covered in ``test_mobility``; what THIS cell
    adds is the supervisor's own path over it — ``fire_due_wakes`` →
    ``_engage_one`` → ``engage_runtime`` → ``handoff_guard_refusal`` — and the
    F5 property the design names: the refused attempt leaves
    ``next_due_at``/``fired_count`` untouched, so the destination still fires
    the row exactly once.

    The refused attempt drives the REAL ``engage_runtime`` (the refusal comes
    before any spawn), with only the stale-entry recovery stubbed: recovering a
    genuinely stale entry is its own covered behaviour, and this cell is about
    the refusal's arithmetic.
    """

    def test_a_refused_engage_leaves_the_row_untouched_and_the_wake_fireable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        from local_operator.session import placement
        from local_operator.session.runtime import launch as launch_mod
        from local_operator.wakes import store as wake_store
        from local_operator.wakes.supervisor import fire_due_wakes

        root = tmp_path / "device"
        directory = root / "sessions" / SESSION
        directory.mkdir(parents=True)
        due = int(time.time() * 1000) - 60_000
        rows = [dict(WAKE_ROW, next_due_at=due)]
        (directory / "transcript.jsonl").write_text(
            _custom_entry("wake_schedules", {"schedules": rows}) + "\n", encoding="utf-8"
        )
        carry.rebuild_indexes(root, SESSION)
        assert wake_store.read_entry(root, SESSION) is not None

        # A HANDOFF IN FLIGHT: the journal entry the guard reads.
        placement.write_handoff_entry(
            root,
            SESSION,
            {
                "role": "source",
                "phase": placement.HANDOFF_PHASE_PREPARED,
                "to_device": "d_other",
            },
        )
        monkeypatch.setattr(launch_mod, "recover_stale_handoff", lambda *_a, **_k: None)
        attempted: list[str] = []
        real_engage = launch_mod.engage_runtime

        async def counting_engage(*args: Any, **kwargs: Any) -> Any:
            attempted.append(str(args[0]))
            return await real_engage(*args, **kwargs)

        monkeypatch.setattr(launch_mod, "engage_runtime", counting_engage)

        fired = asyncio.run(fire_due_wakes(root, now_ms=due + 1000))
        # THE ENGAGE WAS REACHED (not skipped as live/ghost/wedged — any of
        # which would pass a weaker assertion without ever consulting the
        # guard), and the guard refused it: nothing started.
        assert attempted == [SESSION], "the sweep never reached the engage"
        assert fired == 0, "a move in flight must refuse the engage"
        after = wake_store.read_entry(root, SESSION)
        assert after is not None
        assert after["schedules"][0]["next_due_at"] == due
        assert after["schedules"][0]["fired_count"] == rows[0]["fired_count"]
        # AND THE STORED TRANSCRIPT IS UNTOUCHED, which is the truth the index
        # is derived from (a consume would show here even if the index were
        # rewritten back).
        stored = carry.latest_custom_details(directory, "wake_schedules") or {}
        assert stored["schedules"][0]["next_due_at"] == due
        assert stored["schedules"][0]["fired_count"] == rows[0]["fired_count"]

        # AND THE FIRE IS STILL OWED, NOT EATEN: clear the handoff and the same
        # row engages. ``later`` is past any owed-failure backoff (bounded by
        # ``deliveries.RETRY_CAP_S`` <= 1 h), because the refusal durably owes
        # the attempt — that record is the retry's bookkeeping, deliberately
        # NOT the schedule's numbers. A recorder stands in for the spawn here:
        # what this half asserts is the sweep's arithmetic, not a runtime.
        assert placement.clear_handoff_entry(root, SESSION) is True
        engaged: list[str] = []

        async def fake_engage(session_id: str, cwd: str, work: Any, **kwargs: Any) -> None:
            engaged.append(session_id)

        monkeypatch.setattr(launch_mod, "engage_runtime", fake_engage)
        later = due + 2 * 3600 * 1000
        assert asyncio.run(fire_due_wakes(root, now_ms=later)) == 1
        assert engaged == [SESSION]
