"""The record carries this runtime's subagent trajectory counts.

**This file exists because ``/info`` could not answer "how many agent
trajectories are running on this machine".** ``SessionRecord`` published no
subagent data at all, so the screen could only ever describe the window it was
opened in — on a host running a dozen sessions with children in most of them,
the section said "none running".

The counts ride the record rather than a new control-socket op for the reason
the option was rejected: ``/info`` is opened when something is ALREADY wrong,
which is exactly when peers are wedged, and a diagnostic screen that fans out
TCP dials to a broken fleet is a diagnostic screen that hangs. The record is
already scanned, already 0600, and already carries live state on the same
purely additive contract.

Two properties are load-bearing and both are asserted here:

* **``None`` is not ``0``.** A handle that cannot answer publishes ``None``, and
  a reader must be able to tell "did not report" from "reported none". The
  fleet total's honesty caveat is derived from exactly that difference.
* **The floor and the transition publisher read the SAME predicate.** This code
  has had that bug once already, in the ``busy`` bit: a 15 s heartbeat that
  disagreed with the event-driven publisher does not merely go stale, it
  OVERWRITES the correct value on its next tick.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.types import RUNNING_SUBAGENT_STATUSES
from tests.e2e.harness import ScriptedStream, build_session, text_turn


async def _rig(directory: Path) -> tuple[Any, OwnedSessionHandle, RuntimeServer]:
    """A real Session under the production handle and server.

    The same rig as ``test_busy_settles`` and ``test_activity_vs_residency``:
    the seam under test is the one production runs, so only the provider stream
    is scripted.
    """
    directory.mkdir(parents=True, exist_ok=True)
    session = build_session(directory, ScriptedStream([text_turn("reply")]))
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    handle.subscribe(server._schedule_push)
    return session, handle, server


class _Node:
    """One roster entry, in the shape ``SubagentComms.nodes()`` returns.

    A SHAPE stub for the wiring tests below (dedupe, floor/transition
    agreement) and nothing more. It cannot prove the FILTER is right, because
    it sets ``status`` from the same vocabulary the filter tests against — an
    assertion against itself. Review round 1 (R1) found exactly that: the real
    ``node()`` derived a status from a disjoint vocabulary and the tally was a
    hard 0 while every test here stayed green. The real-object test at the
    bottom of this file is the one that pins the filter.
    """

    def __init__(self, job_id: str, status: str, parent_job_id: str | None = None) -> None:
        self.job_id = job_id
        self.status = status
        self.parent_job_id = parent_job_id
        self.label = job_id
        self.agent_role = ""
        self.effort = ""
        self.session_id = None
        self.live = status in RUNNING_SUBAGENT_STATUSES


class _Comms:
    def __init__(self, nodes: list[_Node]) -> None:
        self._nodes = nodes

    def nodes(self) -> list[_Node]:
        return list(self._nodes)


def _with_roster(session: Any, comms: Any) -> None:
    """Install a roster the real ``subagent_comms`` property will hand back.

    Assigned to the PRIVATE slot, not the public name: ``subagent_comms`` is a
    property on the class, which an instance attribute cannot shadow. Writing
    the public name would silently leave the real (empty) roster in place and
    the test would pass against nothing.
    """
    session._subagent_comms = comms


@pytest.mark.asyncio
async def test_the_probe_counts_running_and_queued_off_one_flat_roster(tmp_path: Path) -> None:
    """``nodes()`` already contains every nested descendant.

    So the count is a ``len()`` over a filter. Adding a recursive child walk on
    top — the obvious-looking way to "include nested subagents" — would count
    every node below depth 0 twice. The roster here is three deep for that
    reason: a recursive implementation would report more than three.
    """
    session, handle, _ = await _rig(tmp_path / "s")
    _with_roster(
        session,
        _Comms(
            [
                _Node("a", "running"),
                _Node("b", "starting", parent_job_id="a"),
                _Node("c", "pausing", parent_job_id="b"),
                _Node("d", "queued"),
                _Node("e", "completed"),
            ]
        ),
    )
    assert handle.subagent_counts() == (3, 1)


@pytest.mark.asyncio
async def test_a_session_event_publishes_the_counts_onto_the_record(tmp_path: Path) -> None:
    """The transition path: ``_notify`` → ``_publish_busy`` → ``set_subagents``.

    Driven from the same seam as the busy bit because a subagent launching or
    settling IS a session event, which is what keeps the number current
    sub-second rather than up to a heartbeat stale.
    """
    session, handle, server = await _rig(tmp_path / "s")
    _with_roster(session, _Comms([_Node("a", "running"), _Node("b", "queued")]))
    handle._publish_busy()
    assert (server._subagents_running, server._subagents_queued) == (1, 1)


@pytest.mark.asyncio
async def test_an_unchanged_count_does_not_rewrite_the_record(tmp_path: Path) -> None:
    """Deduped like ``set_busy``: this runs on EVERY session event.

    Without the comparison the steady-state cost would be a staged write and
    rename per event rather than per transition.
    """
    _, _, server = await _rig(tmp_path / "s")
    writes: list[dict[str, Any]] = []

    class _Publisher:
        def heartbeat(self, **updates: Any) -> None:
            writes.append(updates)

    server._publisher = _Publisher()  # type: ignore[assignment]
    server.set_subagents(2, 1)
    assert len(writes) == 1, "a change publishes"
    assert writes[0]["subagents_running"] == 2
    assert writes[0]["subagents_queued"] == 1
    server.set_subagents(2, 1)
    assert len(writes) == 1, "an identical count must not rewrite the record"
    server.set_subagents(2, 0)
    assert len(writes) == 2, "a change in EITHER count publishes"


@pytest.mark.asyncio
async def test_a_handle_without_the_probe_publishes_none_and_does_not_raise(
    tmp_path: Path,
) -> None:
    """An older or narrower handle must degrade, not explode or fabricate a zero.

    ``TuiSessionHandle`` implements neither ``is_busy`` nor
    ``is_conversationally_active``, which is the precedent: a handle that cannot
    answer a probe leaves the field unreported. ``None`` is the honest value —
    the reader counts it under ``subagents_unreported`` rather than as a session
    with no children.
    """
    _, _, server = await _rig(tmp_path / "s")

    class _Bare:
        pass

    server._handle = _Bare()  # type: ignore[assignment]
    counts = getattr(server._handle, "subagent_counts", None)
    assert counts is None, "the probe is optional by getattr, exactly like is_busy"
    assert server._subagents_running is None, "unreported, never 0"


@pytest.mark.asyncio
async def test_an_unreadable_roster_publishes_none_rather_than_zero(tmp_path: Path) -> None:
    """A roster that RAISES is not a roster of zero.

    ``_publish_busy`` runs on every session event and must never take a turn
    down, so the failure is swallowed — but what it publishes afterwards has to
    be "I could not tell", not a confident zero that quietly drops this
    runtime's children out of the fleet total.
    """
    session, handle, server = await _rig(tmp_path / "s")

    class _Exploding:
        def nodes(self) -> list[_Node]:
            raise RuntimeError("roster unavailable")

    _with_roster(session, _Exploding())
    assert handle.subagent_counts() == (None, None)
    handle._publish_busy()  # must not raise
    assert server._subagents_running is None


@pytest.mark.asyncio
async def test_the_heartbeat_floor_agrees_with_the_transition_publisher(
    tmp_path: Path,
) -> None:
    """The §5 failure mode, which this codebase has already shipped once.

    The floor exists to bound how stale a MISSED publish can get. If it read a
    different predicate from the transition publisher it would not bound
    anything — it would overwrite the correct value within one heartbeat,
    forever. Both must call ``subagent_counts``, so driving a divergence and
    then running the floor's read must restore agreement rather than clobber it.
    """
    session, handle, server = await _rig(tmp_path / "s")
    _with_roster(
        session, _Comms([_Node("a", "running"), _Node("b", "running"), _Node("c", "queued")])
    )

    # A wrong value, as a missed publish would leave behind.
    server.set_subagents(99, 99)
    assert server._subagents_running == 99

    # Exactly what the heartbeat loop does.
    probe = getattr(server._handle, "subagent_counts", None)
    assert callable(probe), "the floor's probe must exist on the production handle"
    reported = probe()
    assert isinstance(reported, tuple) and len(reported) == 2
    server.set_subagents(reported[0], reported[1])

    assert (server._subagents_running, server._subagents_queued) == (2, 1)
    assert (server._subagents_running, server._subagents_queued) == handle.subagent_counts()


@pytest.mark.asyncio
async def test_the_tui_handle_answers_the_same_probe(tmp_path: Path) -> None:
    """A ``kind="tui"`` runtime must contribute to the fleet tally too.

    It is the most common kind of window on a developer host, so leaving it
    unimplemented would report most of the fleet as non-reporting. It reads the
    shared status predicate rather than a private copy — two spellings of
    "running" is how the record and the tree beneath it come to disagree.
    """
    from local_operator.mobile.tui_handle import TuiSessionHandle

    session, _, _ = await _rig(tmp_path / "s")
    _with_roster(session, _Comms([_Node("a", "running"), _Node("b", "queued")]))

    handle = TuiSessionHandle.__new__(TuiSessionHandle)
    handle._session = lambda: session  # type: ignore[method-assign]
    assert handle.subagent_counts() == (1, 1)

    # And a session that is not ready yet reports UNREPORTED, not zero.
    def _not_started() -> Any:
        raise RuntimeError("session is still starting")

    handle._session = _not_started  # type: ignore[method-assign]
    assert handle.subagent_counts() == (None, None)


@pytest.mark.asyncio
async def test_the_probe_counts_a_REAL_SubagentComms_roster(tmp_path: Path) -> None:
    """THE R1 REGRESSION. The stub above cannot catch this; only the real object can.

    ``subagent_counts`` filters ``comms.nodes()`` on
    ``RUNNING_SUBAGENT_STATUSES = {running, starting, pausing}``, but
    ``SubagentComms.node()`` used to derive status privately as
    ``paused | <outcome> | cancelled | gone`` — a vocabulary with an EMPTY
    intersection with the filter. A live running child was reported ``gone``,
    so every owner runtime published a *measured* ``0``: it landed in
    ``reporting``, suppressed the lower-bound caveat, and made the header
    assert "none running" over a roster full of live children. Strictly worse
    than the bug the feature exists to fix, and invisible to a stub that sets
    ``status`` from the counting vocabulary itself.

    ``node()`` now derives status through ``_describe`` — the same collapse
    ``roster()`` uses — so the node and the roster cannot disagree about the
    same child. This test asserts the COUNT against a real registry, and the
    node/roster agreement that keeps it true.
    """
    from local_operator.harness.comms import SubagentComms
    from tests.unit.harness.test_comms import FakeChild, FakeJobs, FakeParent

    session, handle, _ = await _rig(tmp_path / "s")
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]

    # Distinct session dirs: ``attach`` folds a record whose dir matches a
    # settled one, which would silently collapse the roster under test.
    running_dir = tmp_path / "running"
    running_dir.mkdir()
    settled_dir = tmp_path / "settled"
    settled_dir.mkdir()

    jobs.add("j1", status="running")
    comms.record_launch("j1", "reviewer", prompt="review")
    comms.attach("j1", FakeChild(), running_dir)  # type: ignore[arg-type]

    # Queued: the job row still says ``running`` and the split lives in the
    # separate ``queued`` flag, which is exactly the distinction R2 covers.
    jobs.add("j2", status="running")
    jobs.jobs["j2"].queued = True
    comms.record_launch("j2", "qa-tester", prompt="qa")

    jobs.add("j3", status="running")
    comms.record_launch("j3", "coder", prompt="code")
    comms.attach("j3", FakeChild(), settled_dir)  # type: ignore[arg-type]
    comms.record_outcome("j3", "completed")

    _with_roster(session, comms)

    statuses = {node.job_id: node.status for node in comms.nodes()}
    assert statuses == {"j1": "running", "j2": "queued", "j3": "completed"}, statuses
    assert statuses == {
        row.job_id: row.status for row in comms.roster()
    }, "node() and roster() must agree; two derivations of one fact is how they drift"
    assert handle.subagent_counts() == (1, 1), "a live child is not 'gone'"


@pytest.mark.asyncio
async def test_a_real_roster_reaches_the_record_as_a_nonzero_count(tmp_path: Path) -> None:
    """End to end on the owner path: real registry -> probe -> published record.

    R1's damage was not the count in isolation, it was that a fabricated
    *measured* zero reaches ``SessionRecord`` and is then indistinguishable
    from a runtime that genuinely has no children — which is precisely the
    distinction the whole feature is built on.
    """
    from local_operator.harness.comms import SubagentComms
    from tests.unit.harness.test_comms import FakeChild, FakeJobs, FakeParent

    session, handle, server = await _rig(tmp_path / "s")
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    jobs.add("j1", status="running")
    comms.record_launch("j1", "reviewer", prompt="review")
    comms.attach("j1", FakeChild(), child_dir)  # type: ignore[arg-type]
    _with_roster(session, comms)

    handle._publish_busy()
    assert server._subagents_running == 1, "the record must carry the live child"
    assert server._subagents_queued == 0
