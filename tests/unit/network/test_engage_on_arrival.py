"""``--engage-on-arrival``: a conversation that comes up live where it lands.

WHAT EACH CELL PINS, and each is a requirement of the decision rather than a nicety of
the implementation:

* **Default off.** A move that does not ask for the engage starts no runtime and
  carries no ``engagement`` member: the cold arrival §6.3 step 17 decided on, which
  every caller that existed before the flag still gets, on both routes.
* **On, it engages THE ADOPTED ID once**, and "live" here is the product's own answer
  to "is a runtime up": a record ``registry.scan`` reads as ``live``, published by a
  real ``RuntimeServer`` — so the spawned PROCESS is the only thing stood in for.
* **A failed engage is not a failed move.** The receipt stays successful, the phases
  are the move's own, ``engagement.engaged`` is ``False`` with the engage's own
  sentence, and the conversation is on the destination's disk with its journal
  settled — nothing half-applied, nothing rolled back.
* **It does not fight the lease.** Two engagements that genuinely contend — the
  arrival engage and the destination's own, both driven through the REAL
  ``launch.engage_runtime`` and the real transcript lease — end with ONE runtime, and
  the contender that came second is JOINED onto it rather than refused. The MOVE is a
  successful move whichever of the two won.
* **The busy refusal is untouched.** A source with a turn in flight refuses ``busy``
  with the ``--wait`` remedy whether or not the flag is set, and nothing moves — the
  operator kept that refusal deliberately.

Driven through ``mobility.request_move`` — the CLI's own path through this device's
relay — on the two-relay rig ``test_mobility`` builds: two real config roots, two
identities, real loopback TCP, real control sockets, real protocol frames.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import local_operator.session.runtime.launch as launch_module
from local_operator.mobile import attach_client
from local_operator.network import mobility
from local_operator.session.runtime.launch import RuntimeStartupError
from local_operator.session.runtime.types import SessionRecord
from local_operator.session_lease import SessionLeaseHeldError, acquire_session_lease
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures and helpers
    SESSION,
    Devices,
    _move,
    _owned_session,
    pair,
)
from tests.unit.network.test_mobility_busy_wait import _BusyClient
from tests.unit.network.test_mobility_busy_wait import (  # noqa: F401 — the owner's stubs
    _Record as _BusyRecord,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — the fixture `pair` reaches for
    _pair_settled,
    devices,
)
from tests.unit.session.runtime.test_server import FakeHandle

#: A REFUSAL THE ENGAGE LAYER REALLY PRODUCES, as a literal: the curated sentence a
#: ``RuntimeStartupError`` carries when no candidate for the conversation could be
#: started, which ``_engage_locally`` turns into ``engaged: False``.
#:
#: IT IS NOT WHAT LOSING A LEASE RACE LOOKS LIKE, and the name it used to carry
#: (``LOST_THE_LEASE``) said it was — which made the file's contention cell assert a
#: sentence the product never emits for that event (review round 1, R1-1). Contention
#: JOINS; the cell that covers it drives the real arbitration, and this literal is used
#: only where the subject is the refusal's CONSEQUENCE (a failed engage is not a failed
#: move), never its cause.
#:
#: A literal rather than the module's own renderer, because a stub that called into the
#: code under test would make a red-on-old run fail for the wrong reason.
ENGAGE_REFUSED = "the runtime for this conversation could not be started, so nothing was started"


class _Handle(FakeHandle):
    """``FakeHandle`` for a session id this test mints, not its hardcoded one."""

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self._projection = replace(self._projection, session_id=session_id)
        from local_operator.session.frontend_state import FrontendStateStore

        self._frontend = FrontendStateStore(
            self._frontend.state.model_copy(update={"session_id": session_id})
        )


def _wait_live(root: Path, session_id: str, timeout: float = 20.0) -> bool:
    """Is a runtime for ``session_id`` published as LIVE on ``root``?

    ``registry.scan`` is the scanner a viewer, a listing and the relay's own dial all
    read, so this is the product's answer to "is that conversation live here" and not
    this file's.
    """
    from local_operator.session.runtime import registry

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(
            record.session_id == session_id and status == "live"
            for record, status in registry.scan(root)
        ):
            return True
        time.sleep(0.01)
    return False


def _journal(root: Path, session_id: str) -> dict[str, Any]:
    from local_operator.session.placement import read_handoff_journal

    return dict(read_handoff_journal(root).get(session_id) or {})


class _Arrivals:
    """``launch.engage_runtime`` as the arrival path reaches it.

    THE SPAWNED PROCESS IS THE ONLY THING STOOD IN FOR (the shape
    ``test_session_plane._serve`` established): a call starts a real ``RuntimeServer``
    that publishes a real registry record, so everything downstream of the engage is
    the production path.

    ``fail_with`` makes every call lose the way a refused engage loses.

    IT CANNOT STAND IN FOR THE ARBITRATION, and must not be asked to. A stub owns the
    winner/loser answer, so a cell built on one passes whatever the real lease does —
    the defect review round 1 found in this file's contention cell (R1-1), whose fix is
    ``_LeasedFleet`` below. What this is for is the SPAWNED PROCESS: everything
    downstream of an engage that has already been decided.
    """

    def __init__(
        self,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        fail_with: str = "",
    ) -> None:
        self.root = root
        self.monkeypatch = monkeypatch
        self.fail_with = fail_with
        self.calls: list[str] = []
        self.served: list[tuple[str, Any]] = []
        self._lock = threading.Lock()
        monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", self)

    def wait_for(self, count: int, timeout: float = 30.0) -> bool:
        """Has the destination made ``count`` engage calls yet?

        NEEDED ON THE OFFLOAD ROUTE, and it is the design's own shape rather than a
        test's convenience: the inviter settles on its OWN durable progress (``_offload``)
        and answers as soon as the destination's ``move.done`` lands, which is BEFORE that
        device engages. So a caller that returns from ``_move`` has a destination still
        working, and only a shared observation can order them.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.calls) >= count:
                    return True
            time.sleep(0.01)
        return False

    async def __call__(self, session_id: str, cwd: str, work: Any, **kwargs: Any) -> Any:
        # The runtime publishes into the AMBIENT config root, and the relay calls this
        # from its own thread: the fake pins that root here rather than trusting
        # whatever the test last set, or the record lands in the wrong device's store
        # (the same trap ``test_session_plane._serve`` documents).
        self.monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(self.root))
        with self._lock:
            self.calls.append(session_id)
        if self.fail_with:
            raise RuntimeStartupError(
                self.fail_with,
                # ``actionable`` is the channel a curated runtime sentence reaches a user
                # through (``_engage_failure_detail`` prefers it over the raw message),
                # and this stands in for exactly such a refusal — so the cell can assert
                # that the refusal's OWN words reach the receipt.
                actionable=self.fail_with,
            )
        self._start(session_id)
        return None

    def _start(self, session_id: str) -> None:
        from local_operator.session.runtime.server import RuntimeServer

        if any(served == session_id for served, _runtime in self.served):
            return
        runtime = RuntimeServer(_Handle(session_id), kind="tui")
        runtime.start()
        self.served.append((session_id, runtime))
        assert _wait_live(
            self.root, session_id
        ), "the runtime this engage started never published a live record"

    def stop(self) -> None:
        for _session_id, runtime in self.served:
            try:
                runtime.close()
            except Exception:  # noqa: BLE001 — teardown must not mask a failure
                pass
        self.served.clear()


def _arrivals(
    request: pytest.FixtureRequest, *, root: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> _Arrivals:
    arrivals = _Arrivals(root, monkeypatch, **kwargs)
    request.addfinalizer(arrivals.stop)
    return arrivals


def _pair_and_own(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Devices:
    """Two paired relays, A holding one session. The destination is B on every route.

    THE PAIRING IS SETTLED BEFORE THE CELL'S FIRST MOVE, exactly as the mobility
    cells are: the join link outlives the ceremony by the length of
    ``PeerLink.close``'s flush window, and a move driven inside it resolves the dying
    link instead of dialling the reachable peer -- CI run 36518398815 failed
    ``test_the_offload_carries_the_flag_to_the_destination`` with that refusal
    (``unreachable (no answer on its link)``). ``_pair_settled`` docs carry the
    measurement; every cell here goes through this one helper.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    return both


# ---------------------------------------------------------------------------
# Default off
# ---------------------------------------------------------------------------


def test_a_move_without_the_flag_still_arrives_cold(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The behaviour every existing caller has, asserted against the runtime itself."""
    server_a, server_b, _host, _port = _pair_and_own(request, monkeypatch)
    arrivals = _arrivals(request, root=server_b.root, monkeypatch=monkeypatch)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert result["phase"] == "done"
    # NO ENGAGE, and no member telling a reader to look for one: the receipt a caller
    # parses is the one it always got.
    assert arrivals.calls == [], arrivals.calls
    assert "engagement" not in result, result
    assert (server_b.root / "sessions" / SESSION).is_dir()
    assert not (server_a.root / "sessions" / SESSION).exists()


def test_the_offload_route_leaves_the_flag_off_too(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--to <peer>`` without the flag: the invite carries nothing, nothing engages."""
    server_a, server_b, _host, _port = _pair_and_own(request, monkeypatch)
    arrivals = _arrivals(request, root=server_b.root, monkeypatch=monkeypatch)

    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    # A NEGATIVE WITH A BOUND, not a race of its own: the destination engages AFTER its
    # own ``move.done``, so a bare equality could pass before a wrongly-carried flag had
    # the chance to show itself.
    assert not arrivals.wait_for(1, timeout=3.0), arrivals.calls
    assert "engagement" not in result, result
    assert (server_b.root / "sessions" / SESSION).is_dir()


def test_a_frame_without_the_field_engages_nothing(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The READER's tolerance, driven through the handler that owns it.

    A client older than the field sends no key at all; a hand-rolled one may send a
    null. Both are OFF, and the move itself is untouched — the alternative reading
    ("not true" as a direction to engage) would have every existing caller's move start
    a runtime on a device nobody asked to touch.
    """
    _server_a, server_b, _host, _port = _pair_and_own(request, monkeypatch)
    arrivals = _arrivals(request, root=server_b.root, monkeypatch=monkeypatch)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))

    handler = mobility.local_move_handler(server_b)
    result = handler({"session_id": SESSION, "to": "local", "engage_on_arrival": None})

    assert result["ok"] is True, result
    assert arrivals.calls == [], arrivals.calls
    assert "engagement" not in result, result
    assert (server_b.root / "sessions" / SESSION).is_dir()


# ---------------------------------------------------------------------------
# On: the conversation comes up live
# ---------------------------------------------------------------------------


def test_the_flag_engages_the_adopted_id_and_the_receipt_says_so(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE FEATURE, on the route whose receipt is the destination's own.

    Landing and becoming live are asserted separately and in that order: the bytes are
    on B's disk, and a runtime for the adopted id is published as live — with no prompt
    and nothing else asked of this device.
    """
    server_a, server_b, _host, _port = _pair_and_own(request, monkeypatch)
    arrivals = _arrivals(request, root=server_b.root, monkeypatch=monkeypatch)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch, engage_on_arrival=True)

    assert result["ok"] is True, result
    assert arrivals.calls == [SESSION], arrivals.calls
    engagement = result["engagement"]
    assert engagement["engaged"] is True, engagement
    assert engagement["session_id"] == SESSION, engagement
    assert engagement["detail"], engagement
    assert _wait_live(server_b.root, SESSION), "the conversation did not come up live"
    assert (server_b.root / "sessions" / SESSION).is_dir()
    assert not (server_a.root / "sessions" / SESSION).exists()


def test_the_offload_carries_the_flag_to_the_destination(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--to <peer>``: the device that ADOPTS the conversation is the one that engages.

    The runtime is started on B, not on the device that typed the command — which is
    the whole reason the flag has to cross the link with the invite. B's own answer is
    what a reader checks there (the inviter settles on its own durable progress by
    design, ``_offload``), so this cell asserts the runtime, and the inviter's receipt
    is asserted to be exactly the receipt it always was.
    """
    server_a, server_b, _host, _port = _pair_and_own(request, monkeypatch)
    arrivals = _arrivals(request, root=server_b.root, monkeypatch=monkeypatch)

    result = _move(
        server_a,
        SESSION,
        to=server_b.identity.device_id,
        monkeypatch=monkeypatch,
        engage_on_arrival=True,
    )

    assert result["ok"] is True, result
    # THE DESTINATION'S OWN WORK, OBSERVED WHERE IT HAPPENS. The inviter's receipt is the
    # receipt it always was and does not carry the engage, so the evidence that the flag
    # crossed the link is B's engage call — WAITED FOR, because that device engages after
    # its ``move.done`` and the inviter returns at that frame (`_offload` settles on its
    # own durable progress).
    assert arrivals.wait_for(1), "the destination never engaged the conversation it adopted"
    assert arrivals.calls == [SESSION], arrivals.calls
    assert _wait_live(server_b.root, SESSION), "the conversation did not come up live"
    assert (server_b.root / "sessions" / SESSION).is_dir()
    assert not (server_a.root / "sessions" / SESSION).exists()
    # THE INVITER'S RECEIPT IS NOT THE DESTINATION'S: it reports the transfer it made
    # and does not carry the other device's post-move work, so the member is absent
    # rather than a claim this side cannot observe.
    assert "engagement" not in result, result


# ---------------------------------------------------------------------------
# A failed engage is not a failed move
# ---------------------------------------------------------------------------


def test_a_refused_engage_leaves_a_successful_move(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE HONESTY REQUIREMENT, with the move's own guarantees re-asserted after it."""
    server_a, server_b, _host, _port = _pair_and_own(request, monkeypatch)
    _arrivals(request, root=server_b.root, monkeypatch=monkeypatch, fail_with=ENGAGE_REFUSED)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch, engage_on_arrival=True)

    assert result["ok"] is True, result
    assert result["phase"] == "done"
    assert [item["phase"] for item in result["phases"]] == [
        "prepared",
        "handing_off",
        "committed",
        "done",
    ]
    engagement = result["engagement"]
    assert engagement["engaged"] is False, engagement
    assert ENGAGE_REFUSED in engagement["detail"], engagement
    # NOTHING HALF-APPLIED: the conversation is here, whole, and the journal that
    # named it in transit is settled.
    assert (server_b.root / "sessions" / SESSION).is_dir()
    assert (server_b.root / "sessions" / SESSION / "transcript.jsonl").is_file()
    assert _journal(server_b.root, SESSION) == {}
    assert not (server_a.root / "sessions" / SESSION).exists()


def test_an_unexpected_engage_failure_cannot_break_the_move(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engage is an OPTIONAL step on a move that has already committed.

    ``_engage_locally`` turns the failures a runtime start can produce into a sentence,
    but it catches a fixed tuple: anything else would otherwise propagate out of the
    arrival path and destroy the answer to a transfer that has already happened. This
    drives that shape — a bare ``ValueError`` — and requires the receipt to survive it,
    named.
    """
    server_a, server_b, _host, _port = _pair_and_own(request, monkeypatch)
    # THE HANDLE IS NOT KEPT: ``_arrivals`` installs the fake and registers its own
    # teardown, and this cell replaces the callable it installed a line later.
    _arrivals(request, root=server_b.root, monkeypatch=monkeypatch)

    async def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("a bug in the engage path")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", explode)

    result = _move(server_b, SESSION, monkeypatch=monkeypatch, engage_on_arrival=True)

    assert result["ok"] is True, result
    assert result["phase"] == "done"
    engagement = result["engagement"]
    assert engagement["engaged"] is False, engagement
    assert "does not recognise" in engagement["detail"], engagement
    assert (server_b.root / "sessions" / SESSION).is_dir()
    # AND NOTHING CAME UP, which is what the receipt says: the failure is reported, not
    # covered by a runtime that quietly started anyway.
    assert not _wait_live(server_b.root, SESSION, timeout=0.5)


# ---------------------------------------------------------------------------
# The race: two engagements, one lease
# ---------------------------------------------------------------------------


class _LeasedFleet:
    """``launch._spawn_runtime`` as the arrival path reaches it — THE LEASE IS REAL.

    THE SPAWNED PROCESS IS THE ONLY THING STOOD IN FOR, and that is the whole reason
    this exists beside ``_Arrivals`` (review round 1, R1-1). Who runs is decided HERE by
    the REAL transcript lease, contended for exactly as ``process.py`` contends for it:
    ``acquire_session_lease`` admits one writer and raises for every other candidate,
    which is what ``process.py`` logs and turns into ``return 0``. What stays production
    is everything that READS that decision — ``engage_runtime``'s loop, its
    ``_lease_holder`` probe, the record scan, and the single entry point both routes
    reach an engage through (``RelayServer.engage_session``).

    THAT IS THE DIFFERENCE THAT MATTERS. A stub owns the winner/loser answer, so a cell
    built on one stays green whatever the real lease does — measured on the previous
    round's cell, which passed 4/4 with the real ``engage_runtime`` replaced by a raise.
    A cell built on this goes red the moment the lease stops deciding.

    ``hold`` PARKS THE WINNER BETWEEN ITS LEASE AND ITS RECORD. That window is real —
    construction takes time — and it is exactly the window a second contender must wait
    THROUGH rather than spawn into, which is the case the contention claim is about.
    """

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        #: Set by the test once the winner holds the lease — an event rather than a
        #: sleep, because the fact it waits for is a file's content, not a duration.
        self.hold = threading.Event()
        #: Set the instant a candidate HAS the lease, so the cell can start the second
        #: contender at the one moment that makes it a contender (the lease is taken)
        #: rather than at a moment in time.
        self.taken = threading.Event()
        self.spawns = 0
        self.winners = 0
        self.losers = 0
        self.records = 0
        self._lease: Any = None

    def spawn(
        self,
        session_id: str,
        cwd: str,
        *,
        defer_materialise: bool,
        warm: bool = False,
        initial_model: Any = None,
        model_selection_override: bool = False,
    ) -> None:
        """``launch._spawn_runtime``'s own signature, so the real caller is unchanged."""
        self.spawns += 1
        directory = self.config_dir / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            self._lease = acquire_session_lease(directory)
        except SessionLeaseHeldError:
            # The designed outcome for every candidate but one.
            self.losers += 1
            return
        self.winners += 1
        self.taken.set()
        self.hold.wait(timeout=90)
        self.publish(session_id)

    def publish(self, session_id: str) -> None:
        """The record a runtime publishes once it is through construction."""
        from local_operator.session.runtime import registry

        registry.publish(
            SessionRecord(
                pid=os.getpid(),
                kind="daemon",
                session_id=session_id,
                conversation_name="fake",
                cwd=str(self.config_dir),
                model_label="test/model",
                control_port=1,
                control_key="k" * 16,
            ),
            self.config_dir,
        )
        # The liveness marker every owner lookup reads (``resume.live_runtime_pid``),
        # written by the runtime itself in production.
        (self.config_dir / "sessions" / session_id / ".session.pid").write_text(
            str(os.getpid()), encoding="utf-8"
        )
        self.records += 1

    def close(self) -> None:
        """Release everything this fake took, in either order of arrival."""
        self.hold.set()
        if self._lease is not None:
            self._lease.release()
            self._lease = None


def test_two_engagements_contend_on_the_real_lease_and_both_join(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two real engagements, one real lease: ONE runtime, and both callers are JOINED.

    The two contenders are the arrival engage (inside the move) and the destination's
    own first use — the pair the design's single-entry-point rule exists for. Both go
    through ``RelayServer.engage_session`` and the REAL ``launch.engage_runtime``; only
    the spawned process is a fake, so the lease decides who runs.

    WHAT THE SECOND CONTENDER GETS IS A JOIN, NOT A REFUSAL (review round 1, R1-1). The
    loop probes ``_lease_holder`` before spawning precisely so a doomed candidate is
    never created: it waits for the winner's record and then delivers to it, so both
    callers are answered with the layer's own join state and exactly one runtime exists.
    A refusal is the OTHER event this layer produces — "no candidate could be started" —
    and asserting it here asserted a sentence the runtime layer never emits for losing a
    race.

    THE SECOND CONTENDER IS HELD OUT OF ITS OWN SPAWN BY THE WINNER'S LEASE, and the
    cell waits for the loop to SAY SO (``is starting under pid …; waiting``, emitted on
    exactly the branch that found the lease held) rather than for a sleep to elapse. A
    release on a clock would be a race this test won by accident — and a cell that wins
    by accident cannot fail when the arbitration breaks, which is what went wrong here.
    """
    _server_a, server_b, _host, _port = _pair_and_own(request, monkeypatch)
    fleet = _LeasedFleet(server_b.root)
    request.addfinalizer(fleet.close)
    monkeypatch.setattr("local_operator.session.runtime.launch._spawn_runtime", fleet.spawn)

    # THE LOOP'S OWN SENTENCE, TAKEN FROM THE LOOP'S OWN LOGGER. Seeing it is evidence
    # the second contender reached the LEASE rather than the winner's record — the
    # difference between contending and queueing — and it is the product's observable,
    # so nothing in the arbitration is patched to produce it.
    waiting = threading.Event()

    class _Waiting(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 — one hook
            if record.getMessage().startswith(f"engage: {SESSION} is starting under pid"):
                waiting.set()

    launch_logger = logging.getLogger(launch_module.__name__)
    handler = _Waiting()
    launch_logger.addHandler(handler)
    previous_level = launch_logger.level
    launch_logger.setLevel(logging.DEBUG)

    def _restore_logger() -> None:
        launch_logger.removeHandler(handler)
        launch_logger.setLevel(previous_level)

    request.addfinalizer(_restore_logger)

    theirs: list[dict[str, Any]] = []
    moves: list[dict[str, Any]] = []
    failures: list[BaseException] = []

    def _their_engage() -> None:
        # THE DESTINATION'S OWN FIRST USE, on another thread, over the same id: a
        # viewer's first message reaches this method by the same route.
        try:
            theirs.append(server_b.engage_session(SESSION))
        except BaseException as exc:  # noqa: BLE001 — asserted below, not swallowed
            failures.append(exc)

    def _the_move() -> None:
        try:
            moves.append(_move(server_b, SESSION, monkeypatch=monkeypatch, engage_on_arrival=True))
        except BaseException as exc:  # noqa: BLE001 — asserted below, not swallowed
            failures.append(exc)

    mover = threading.Thread(target=_the_move, daemon=True)
    contender = threading.Thread(target=_their_engage, daemon=True)
    mover.start()
    # The arrival's own candidate must have TAKEN THE LEASE before a second contender can
    # be told it lost — a file fact, waited on as one.
    assert fleet.taken.wait(timeout=60), "the arrival engage never took the lease"
    contender.start()
    try:
        assert waiting.wait(timeout=45), (
            "the second engage never reached the lease probe, so nothing contended: the "
            "arrival's engage is the only actor this cell drove"
        )
    finally:
        # ALWAYS RELEASED, even on that assertion: the winner is parked inside the spawn
        # and the two threads below would otherwise be joined against a standing still.
        fleet.hold.set()
    contender.join(timeout=120)
    mover.join(timeout=120)

    assert failures == [], failures
    assert len(moves) == 1, moves
    result = moves[0]
    assert result["ok"] is True, result
    assert result["phase"] == "done", result
    assert len(theirs) == 1, theirs
    mine, ours = result["engagement"], theirs[0]
    # BOTH WERE JOINED. ``runtime joining`` is ``RelayServer.engage_session``'s own state
    # for a contended engage — the single entry point both routes reach — so neither
    # caller is handed a story the layer did not tell.
    assert mine["engaged"] is True, mine
    assert mine["session_id"] == SESSION, mine
    assert mine["detail"] == "runtime joining", mine
    assert ours["engaged"] is True, ours
    assert ours["session_id"] == SESSION, ours
    assert ours["detail"] == "runtime joining", ours
    # ONE RUNTIME, BECAUSE ONE LEASE — and the contender that lost never spawned a
    # candidate to lose, which is what the probe before the spawn is for.
    assert fleet.winners == 1, fleet.winners
    assert fleet.records == 1, fleet.records
    assert fleet.spawns == 1, "the second contender spawned instead of waiting on the lease"
    assert fleet.losers == 0, fleet.losers
    from local_operator.session.runtime import registry

    live = [
        record
        for record, status in registry.scan(server_b.root)
        if record.session_id == SESSION and status == "live"
    ]
    assert len(live) == 1, live
    assert (server_b.root / "sessions" / SESSION).is_dir()


# ---------------------------------------------------------------------------
# The refusal the operator kept
# ---------------------------------------------------------------------------


def test_the_busy_refusal_is_unchanged_with_the_flag_set(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn in flight still refuses, and the flag buys no exemption.

    ``--wait`` remains the remedy, nothing moves, and no engage is attempted: there is
    no arrival to engage. This is the decision the operator kept, asserted with the flag
    on so that a future change that routed the engage around the refusal fails here.
    """
    server_a, server_b, _host, _port = _pair_and_own(request, monkeypatch)
    arrivals = _arrivals(request, root=server_b.root, monkeypatch=monkeypatch)
    # THE OWNER'S OWN RUNTIME ANSWERS, through the REAL renderer: a local client whose
    # ``retire_now`` says the bare reason word, which ``_retire_local_runtime`` turns into
    # the sentence a person reads (the stubs are ``test_mobility_busy_wait``'s, so the
    # exact behaviour the neighbour cell pins is what this cell builds on).
    monkeypatch.setattr(attach_client, "find_runtime_record", lambda root, sid: (_BusyRecord(), 0))
    monkeypatch.setattr(attach_client, "AttachClient", _BusyClient)

    result = _move(
        server_a,
        SESSION,
        to=server_b.identity.device_id,
        monkeypatch=monkeypatch,
        engage_on_arrival=True,
    )

    assert result["ok"] is False, result
    assert result["code"] == "busy", result
    message = str(result["message"])
    assert message != "busy", "the reason token reached the user as the whole message"
    assert "--wait" in message, message
    assert arrivals.calls == [], "the flag engaged a conversation that never arrived"
    assert (server_a.root / "sessions" / SESSION).is_dir()
    assert not (server_b.root / "sessions" / SESSION).exists()
