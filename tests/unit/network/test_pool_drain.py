"""The pre-spindown drain (R22): the flush, the barrier, and both of its bounds.

``mesh-compute-pool.md`` §6.1–6.5, exercised through the production paths rather than
through doubles where the real thing is reachable:

* the FLUSH is ``sync.sync_from`` — a real plan/fetch/verify round trip against a real
  source directory, with only the transport frame handler supplied in process (the
  shape ``test_sync_copy_set`` established). That is what makes the compaction cell
  meaningful: the second flush runs against a transcript ``Transcript.compact_file``
  rewrote, so a byte-offset cursor would splice at the wrong row boundary and the test
  would see it.
* the BARRIER is the coordinator, driven with a clock the test owns, so "the grace
  expired" and "the deadline expired" are facts rather than sleeps.
* the LIFECYCLE and the AUDIT ROWS are the product's own (``store.mutate`` and
  ``audit.jsonl``), read back from disk.

What is deliberately NOT doubled: the cursor, the safe-region boundary, the stamp, the
copy set. A test that invented any of those would be testing its own model of the
protocol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import pool_drain, store, sync
from local_operator.network.types import MemberRecord, NetworkRecord
from local_operator.session import placement
from local_operator.session.placement import SessionPlacement, read_stamp, write_stamp
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import (
    PRE_SPINDOWN_DRAIN_CAUSE,
    SIGNAL_DRAIN_CAUSE,
)
from local_operator.session.transcript import Transcript

SESSION_ID = "b7d1c0ffee42"
OTHER_ID = "b7d1c0ffee43"
MEMBER = "dev_pod_1"
HOME = "dev_home_1"
NETWORK = "n_test0002"


# ---------------------------------------------------------------------------
# The two ends, built from the product's writers
# ---------------------------------------------------------------------------


def _rows(count: int = 4) -> bytes:
    return b"".join(
        json.dumps({"id": f"e{index}", "type": "message", "content": f"row {index}"}).encode()
        + b"\n"
        for index in range(count)
    )


def _session(root: Path, session_id: str = SESSION_ID, payload: bytes | None = None) -> Path:
    """A session directory at ``root`` holding a transcript and the §7.2 sidecars."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_bytes(_rows() if payload is None else payload)
    (directory / "title.json").write_text(json.dumps({"title": "pooled"}), encoding="utf-8")
    (directory / "runtime-stop.json").write_text(json.dumps({"stopped_at": 1.0}), encoding="utf-8")
    return directory


def _pool_stamp(root: Path, session_id: str = SESSION_ID, *, mode: str = "pool") -> None:
    write_stamp(
        root,
        placement.MeshStamp(
            session_id=session_id,
            network_id=NETWORK,
            home_device=HOME,
            placement=SessionPlacement(
                mode=mode,  # type: ignore[arg-type]
                network_id=NETWORK,
                home_device=HOME,
                stamp_revision=1,
            ),
            pool_member=MEMBER if mode == "pool" else "",
        ),
    )


def _network(root: Path, lifecycle: str = "active") -> None:
    store.save(
        NetworkRecord(
            network_id=NETWORK,
            name="probe",
            self_device_id=MEMBER,
            members=[
                MemberRecord(
                    device_id=MEMBER,
                    kind="pool",
                    lifecycle=lifecycle,  # type: ignore[arg-type]
                    name="pod",
                ),
                MemberRecord(device_id=HOME, kind="device", lifecycle="active", name="home"),
            ],
        ),
        root,
    )


def _lifecycle(root: Path, member: str = MEMBER) -> str:
    return str(store.load(NETWORK, root).member(member).lifecycle)  # type: ignore[union-attr]


def _events(root: Path) -> list[tuple[str, dict[str, Any]]]:
    path = root / "network" / "audit.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [(str(row.get("event") or ""), dict(row.get("detail") or {})) for row in rows]


def _ask(source: Path) -> Any:
    """The owner's half of ``net_sync``, in process (``test_sync_copy_set``'s shape)."""

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        session_id = str(frame.get("session_id") or "")
        if phase == "plan":
            return sync.build_manifest(source, session_id, have=frame.get("have") or {})
        if phase == "fetch":
            return sync.serve_fetch(
                source,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                offset=int(frame.get("offset") or 0),
                limit=int(frame.get("limit") or sync.SYNC_CHUNK_BYTES),
            )
        if phase == "verify":
            return sync.serve_verify(
                source,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                prefix_bytes=int(frame.get("prefix_bytes") or 0),
                prefix_digest=str(frame.get("prefix_digest") or ""),
            )
        raise AssertionError(f"unexpected phase {phase!r}")

    return ask


class _Clock:
    """A monotonic clock the test advances, so both bounds expire deterministically."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


# ---------------------------------------------------------------------------
# The flush (R22), at home
# ---------------------------------------------------------------------------


def test_a_flush_brings_home_to_the_frontier_and_keeps_the_id(tmp_path: Path) -> None:
    member, home = tmp_path / "member", tmp_path / "home"
    source = _session(member)
    home.mkdir()
    _pool_stamp(home)

    result = pool_drain.flush_session_home(home, SESSION_ID, ask=_ask(member), member_device=MEMBER)

    copied = home / "sessions" / SESSION_ID
    assert (copied / "transcript.jsonl").read_bytes() == (source / "transcript.jsonl").read_bytes()
    # THE SIDECARS TRAVEL (§7.2), not merely the transcript.
    assert (copied / "title.json").read_bytes() == (source / "title.json").read_bytes()
    assert (copied / "runtime-stop.json").is_file()
    # THE SAME ID, at the path ``/resume`` already looks in (§6.5 rule 1).
    assert copied.is_dir()
    assert result["session_id"] == SESSION_ID
    # THE CURSOR IS THE GENERATION/DIGEST ONE, never a bare offset.
    cursor = sync.read_replica_cursor(home, SESSION_ID)
    assert cursor["owner_device"] == MEMBER
    assert cursor["cursor"]["prefix_digest"] == sync.sha256_bytes(_rows())
    # AND THE PLACEMENT IS REWRITTEN TO LOCAL (§6.5 rule 4), audited with the drain reason.
    stamp = read_stamp(home, SESSION_ID)
    assert stamp is not None
    assert stamp.placement.mode == "local"
    assert stamp.pool_member == ""
    assert stamp.placement.stamp_revision == 2
    events = dict(_events(home))
    assert events["session_sync_completed"]["reason"] == "pre_spindown"
    assert events["session_placement_changed"]["reason"] == "drain"
    assert events["session_placement_changed"]["placement_before"] == "pool"
    assert events["session_placement_changed"]["placement_after"] == "local"


def test_the_guard_lifts_once_the_flush_has_landed(tmp_path: Path) -> None:
    """The end of the whole story: a home copy that was refused is now resumable."""
    member, home = tmp_path / "member", tmp_path / "home"
    _session(member)
    home.mkdir()
    _pool_stamp(home)
    _network(home)

    assert placement.pool_guard_refusal(home, SESSION_ID), "a live member must refuse first"

    pool_drain.flush_session_home(home, SESSION_ID, ask=_ask(member), member_device=MEMBER)

    assert placement.pool_guard_refusal(home, SESSION_ID) == ""


@pytest.mark.asyncio
async def test_a_flush_after_a_compaction_replaces_rather_than_splices(tmp_path: Path) -> None:
    """THE CURSOR'S WHOLE REASON (mobility §7.3).

    ``Transcript.compact_file`` rewrites the file with ``os.replace``, so the bytes a
    cursor's digest covered are gone. The second flush must be told ``replace`` and land
    a byte-identical copy — a byte-offset cursor would append at an offset that now
    falls inside another row.
    """
    member, home = tmp_path / "member", tmp_path / "home"
    directory = member / "sessions" / SESSION_ID
    transcript = Transcript(directory)
    big = _big_message()
    await transcript.append_message(big)
    await transcript.append_message(_small_message())
    await transcript.append_prune(big.id, "[pruned]")
    home.mkdir()
    _pool_stamp(home)

    first = pool_drain.flush_session_home(home, SESSION_ID, ask=_ask(member), member_device=MEMBER)
    before = sync.read_replica_cursor(home, SESSION_ID)["cursor"]
    assert first["mode"] == "replace"
    assert before["prefix_bytes"] == len((directory / "transcript.jsonl").read_bytes())

    reclaimed = await transcript.compact_file()
    assert reclaimed > 0
    rewritten = (directory / "transcript.jsonl").read_bytes()
    assert len(rewritten) < before["prefix_bytes"]

    second = pool_drain.flush_session_home(home, SESSION_ID, ask=_ask(member), member_device=MEMBER)

    assert second["mode"] == "replace", "the old prefix is gone; only a replace is safe here"
    copied = (home / "sessions" / SESSION_ID / "transcript.jsonl").read_bytes()
    assert copied == rewritten
    cursor = sync.read_replica_cursor(home, SESSION_ID)["cursor"]
    assert cursor["prefix_digest"] == "sha256:" + hashlib.sha256(rewritten).hexdigest()
    assert cursor["prefix_bytes"] == len(rewritten)


def test_a_flush_of_a_session_home_never_placed_leaves_the_body_untouched(tmp_path: Path) -> None:
    """No stamp at home means no placement was ever written, and none is invented."""
    member, home = tmp_path / "member", tmp_path / "home"
    _session(member)
    home.mkdir()

    pool_drain.flush_session_home(home, SESSION_ID, ask=_ask(member), member_device=MEMBER)

    assert (home / "sessions" / SESSION_ID / "transcript.jsonl").is_file()
    assert read_stamp(home, SESSION_ID) is None
    assert [event for event, _ in _events(home)] == ["session_sync_completed"]


def _big_message() -> Any:
    from local_operator.session.transcript import Message, TextContent

    return Message(
        role="tool",
        content=[TextContent(text="x" * 400_000)],
        tool_call_id="c1",
        tool_name="bash",
    )


def _small_message() -> Any:
    from local_operator.session.transcript import Message

    return Message.user("keep me")


# ---------------------------------------------------------------------------
# The cadence (§6.3)
# ---------------------------------------------------------------------------


def test_the_tick_flushes_a_dirty_session_and_skips_an_unchanged_one(tmp_path: Path) -> None:
    _session(tmp_path)
    calls: list[str] = []

    def flush(session_id: str) -> bool:
        calls.append(session_id)
        return True

    assert pool_drain.pool_sync_tick(tmp_path, [SESSION_ID], flush=flush) == (SESSION_ID,)
    assert calls == [SESSION_ID]

    # UNCHANGED: the digest the ack recorded still describes the file.
    assert pool_drain.pool_sync_tick(tmp_path, [SESSION_ID], flush=flush) == ()
    assert calls == [SESSION_ID], "an unchanged session must not cost a transfer"

    # APPENDED: dirty again, so the next tick carries the new bytes.
    directory = tmp_path / "sessions" / SESSION_ID
    with (directory / "transcript.jsonl").open("ab") as handle:
        handle.write(json.dumps({"id": "e9", "type": "message", "content": "new"}).encode() + b"\n")
    assert pool_drain.pool_sync_tick(tmp_path, [SESSION_ID], flush=flush) == (SESSION_ID,)
    assert calls == [SESSION_ID, SESSION_ID]


def test_the_tick_does_not_record_an_ack_that_never_came(tmp_path: Path) -> None:
    """A marker written over a FAILED flush would make the next tick skip lost bytes."""
    _session(tmp_path)

    assert pool_drain.pool_sync_tick(tmp_path, [SESSION_ID], flush=lambda _id: False) == ()
    assert pool_drain.flushed_digest(tmp_path, SESSION_ID) == ""
    assert pool_drain.unsynced(tmp_path, SESSION_ID) is True


def test_a_session_with_no_transcript_is_never_dirty(tmp_path: Path) -> None:
    """A freshly-created draft has nothing to carry, and a tick must not chase it."""
    (tmp_path / "sessions" / SESSION_ID).mkdir(parents=True)

    assert pool_drain.unsynced(tmp_path, SESSION_ID) is False


# ---------------------------------------------------------------------------
# The barrier (§6.2), both outcomes
# ---------------------------------------------------------------------------


def _coordinator(
    tmp_path: Path, clock: _Clock, **overrides: Any
) -> tuple[pool_drain.PoolDrainCoordinator, Path]:
    member, home = tmp_path / "member", tmp_path / "home"
    source = member / "sessions" / SESSION_ID
    kwargs: dict[str, Any] = {
        "config_dir": member,
        "member_id": MEMBER,
        "network_id": NETWORK,
        "sessions": lambda: [SESSION_ID],
        "flush": pool_drain.make_flush(
            home, ask_for=lambda _session_id: _ask(member), member_device=MEMBER
        ),
        "clock": clock,
        "sleep": clock.sleep,
        "poll_s": 1.0,
    }
    kwargs.update(overrides)
    return pool_drain.PoolDrainCoordinator(**kwargs), source


def _both_ends(tmp_path: Path) -> tuple[Path, Path]:
    member, home = tmp_path / "member", tmp_path / "home"
    _session(member)
    home.mkdir()
    _pool_stamp(home)
    _network(member)
    _network(home)
    return member, home


def test_a_reachable_home_drains_expires_and_flushes_everything(tmp_path: Path) -> None:
    member, home = _both_ends(tmp_path)
    clock = _Clock()
    signalled: list[tuple[str, str]] = []
    coordinator, _source = _coordinator(
        tmp_path,
        clock,
        signal=lambda session_id, cause: signalled.append((session_id, cause)),
        busy=lambda _id: False,
    )

    outcome = coordinator.drain()

    assert outcome.drained is True
    assert outcome.pending == ()
    assert outcome.cut == ()
    assert signalled == [(SESSION_ID, PRE_SPINDOWN_DRAIN_CAUSE)]
    assert _lifecycle(member) == "expired"
    # The bytes are at home, under the same id.
    assert (home / "sessions" / SESSION_ID / "transcript.jsonl").read_bytes() == (
        member / "sessions" / SESSION_ID / "transcript.jsonl"
    ).read_bytes()
    assert read_stamp(home, SESSION_ID).placement.mode == "local"  # type: ignore[union-attr]
    # Every row of the story, in order, on the one log.
    events = [event for event, _detail in _events(member)]
    assert events[:1] == ["drain_started"]
    assert "pool_member_expired" in events
    detail = dict(_events(home))["session_sync_completed"]
    assert detail["reason"] == "pre_spindown"


def test_an_in_flight_turn_is_waited_for_inside_the_grace(tmp_path: Path) -> None:
    """A turn that finishes within the grace is NOT cut — the grace is a wait, not a kill."""
    member, _home = _both_ends(tmp_path)
    clock = _Clock()
    polls = {"n": 0}

    def busy(_session_id: str) -> bool:
        polls["n"] += 1
        return polls["n"] <= 3

    cuts: list[str] = []
    coordinator, _source = _coordinator(
        tmp_path,
        clock,
        busy=busy,
        cut=cuts.append,
        drain_grace_s=120.0,
    )

    outcome = coordinator.drain()

    assert outcome.cut == ()
    assert cuts == []
    assert clock.now < 1000.0 + 120.0, "the wait must end when the turn does"
    assert outcome.drained is True


def test_a_turn_still_running_at_the_grace_is_cut_and_journalled(tmp_path: Path) -> None:
    member, _home = _both_ends(tmp_path)
    directory = member / "sessions" / SESSION_ID
    registry.write_turn_journal(
        directory,
        {
            "session_id": SESSION_ID,
            "pid": 4242,
            "parent_pid": 4241,
            "turn_seq": 3,
            "command_id": "cmd-3",
            "started_at": 990.0,
            "ended_at": None,
            "open": True,
            "end_cause": "",
            "exit_cause": "",
            "still_open_at_exit": False,
            "last_boundary": "bash",
            "build": {},
            "install_root": "",
            "updated_at": 990.0,
        },
    )
    clock = _Clock()
    cuts: list[str] = []
    coordinator, _source = _coordinator(
        tmp_path,
        clock,
        busy=lambda _id: True,
        cut=cuts.append,
        drain_grace_s=30.0,
    )

    outcome = coordinator.drain()

    assert outcome.cut == (SESSION_ID,)
    assert cuts == [SESSION_ID]
    assert clock.now >= 1000.0 + 30.0, "the grace is spent before the cut, never skipped"
    row = registry.read_turn_journal(directory)
    assert row is not None
    assert row["open"] is False
    assert row["ended_at"] is not None
    # THE RUNTIME'S OWN SIGNAL TOKEN, so every surface that renders ``end_cause``
    # through ``incidents.CUT_OFF_CAUSES`` names it (a new token would need a
    # translated sentence in every locale — see ``_record_drain_cut``).
    assert row["end_cause"] == SIGNAL_DRAIN_CAUSE
    # The pool trigger is stated where this slice owns the vocabulary: the audit rows.
    rows = dict(_events(member))
    assert rows["drain_started"]["trigger"] == "user"
    assert rows["pool_member_expired"]["cut"] == [SESSION_ID]
    assert rows["drain_started"]["sessions"] == [SESSION_ID]


def test_an_unreachable_home_exits_on_the_deadline_without_hanging(tmp_path: Path) -> None:
    member, _home = _both_ends(tmp_path)
    clock = _Clock()
    coordinator, _source = _coordinator(
        tmp_path,
        clock,
        flush=lambda _session_id: False,
        busy=lambda _id: False,
        sync_drain_deadline_s=900.0,
    )

    outcome = coordinator.drain()

    assert outcome.drained is False
    assert outcome.barrier_expired is True
    assert outcome.pending == (SESSION_ID,)
    # THE BOUND IS A BOUND: the clock advanced no further than the deadline plus one poll.
    assert clock.now <= 1000.0 + 900.0 + 1.0
    # The member stays ``draining`` — it is about to power off as ``lost`` (§6.6).
    assert _lifecycle(member) == "draining"
    rows = dict(_events(member))
    assert rows["drain_barrier_timeout"]["pending"] == [SESSION_ID]
    assert rows["drain_barrier_timeout"]["pending_count"] == 1
    assert rows["drain_barrier_timeout"]["deadline_s"] == 900.0
    assert "pool_member_expired" not in rows


def test_one_unacked_session_does_not_expire_the_member(tmp_path: Path) -> None:
    """The barrier is ALL acks or the deadline — a partial copy is not a drained pod."""
    member, home = _both_ends(tmp_path)
    _session(member, OTHER_ID, payload=_rows(2))
    _pool_stamp(home, OTHER_ID)
    clock = _Clock()
    coordinator, _source = _coordinator(
        tmp_path,
        clock,
        sessions=lambda: [SESSION_ID, OTHER_ID],
        flush=lambda session_id: session_id == SESSION_ID,
        busy=lambda _id: False,
        sync_drain_deadline_s=10.0,
    )

    outcome = coordinator.drain()

    assert outcome.flushed == (SESSION_ID,)
    assert outcome.pending == (OTHER_ID,)
    assert _lifecycle(member) == "draining"


def test_a_non_pool_session_is_untouched_by_a_flush(tmp_path: Path) -> None:
    """The regression guard for every install that never joins a mesh."""
    member, home = tmp_path / "member", tmp_path / "home"
    _session(member)
    home.mkdir()
    _pool_stamp(home, mode="local")
    before = (home / "sessions" / SESSION_ID / "mesh.json").read_text()

    pool_drain.flush_session_home(home, SESSION_ID, ask=_ask(member), member_device=MEMBER)

    assert (home / "sessions" / SESSION_ID / "mesh.json").read_text() == before
    assert "session_placement_changed" not in [event for event, _ in _events(home)]


# ---------------------------------------------------------------------------
# The drain ladder (§3.6): one more trigger, no second vocabulary
# ---------------------------------------------------------------------------


def test_the_pool_trigger_rides_the_existing_leaving_ladder() -> None:
    """A draining pool session speaks the runtime's words, not a mesh dialect.

    §3.6: "A viewer sees the existing leaving notice, with the peer's words; it does
    not get a mesh-specific error." So the trigger resolves to the SIGNAL phrase —
    the pool drain IS a signalled drain, and its grace is the same 2-minute bound the
    phrase names — and no fourth phrase is minted. A fourth phrase would be a fourth
    trigger every one of the four consumer tables has to be taught
    (``tests/unit/session/runtime/test_leaving_vocabulary.py``).
    """
    from local_operator.session.runtime import types

    assert types.PRE_SPINDOWN_DRAIN_CAUSE == "pre_spindown"
    assert types.PRE_SPINDOWN_DRAIN_CAUSE != types.SIGNAL_DRAIN_CAUSE
    assert types.leaving_phrase_for_frame("pre_spindown") == types.LEAVING_ON_SIGNAL
    assert types.leaving_phrase_for_frame("shutdown-drain") == types.LEAVING_ON_SIGNAL
    assert (
        types.drain_phrase_for_frame({"reason": types.PRE_SPINDOWN_DRAIN_CAUSE})
        == types.LEAVING_ON_SIGNAL
    )
    # A frame carrying its OWN phrase still wins, and an unrecognised trigger still
    # inherits nothing.
    assert (
        types.drain_phrase_for_frame(
            {"reason": types.PRE_SPINDOWN_DRAIN_CAUSE, "leaving": types.LEAVING_FOR_BUILD}
        )
        == types.LEAVING_FOR_BUILD
    )
    assert types.drain_phrase_for_frame({"reason": "something-new"}) == ""
    # The bound in the phrase is the pool's default grace, so the sentence is true.
    assert pool_drain.DEFAULT_DRAIN_GRACE_S == types.SIGNAL_DRAIN_S
