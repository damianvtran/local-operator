"""The retention predicate must not clone canonical state to answer one boolean.

``SessionInteraction.retained_for_auto_work`` is evaluated for every canonical
delta of every LEASED source, and for the session being viewed, because it is
what lets an auto-approving viewer notice a background turn finishing. It used
to answer its ``any(job.status == "running")`` half through
``self.session.frontend_state`` — a full deep copy of canonical state, on the
event loop, once per delta per source — which the coupling audit measured at
0.19-0.58 ms per call (~90 % of a whole scalar delta at the lean shape).

The fix is the session-side sibling of ``pending_gate``/``epoch``: a store-level
predicate over the already-frozen roster, exposed as ``has_running_job`` on both
session classes and read through a duck-probe with a fallback.

The assertions here are STRUCTURAL — how many times the clone happened, and what
the predicate answered — never durations, per AGENTS.md "Timing, flakes".
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
    JobState,
)
from local_operator.tui.session_interaction import SessionInteraction


def _state(*statuses: str) -> FrontendSessionState:
    return FrontendSessionState(
        session_id="retention",
        epoch="e1",
        jobs=[
            JobState(
                id=f"child-{index}",
                type="task",
                status=status,
                trajectory=[{"type": "value", "tool_call_id": f"call_{index}"}],
            )
            for index, status in enumerate(statuses)
        ],
    )


class _StoreBackedViewer:
    """A session exposing the two doors the predicate can use.

    Built on the real classes' shape rather than a mock, ``has_running_job`` is a
    PROPERTY returning a bool on both real session classes (pinned by
    ``test_the_accessor_is_a_property_rather_than_a_method``), and
    ``frontend_state`` is the clone-paying read it replaces.
    """

    def __init__(self, store: FrontendStateStore, *, narrow: bool = True) -> None:
        self._store = store
        self._narrow = narrow
        self.is_streaming = False

    @property
    def has_running_job(self) -> bool:
        if not self._narrow:
            raise AttributeError("has_running_job")
        return self._store.has_running_job()

    @property
    def frontend_state(self) -> FrontendSessionState:
        return self._store.state


def _source(viewer: Any, *, approve: bool = True) -> SessionInteraction:
    source = SessionInteraction(viewer)
    source.draft.approve_all = approve
    return source


def test_the_predicate_answers_without_reading_the_cloned_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect, as a count: one clone per delta per source for one boolean."""
    store = FrontendStateStore(_state("running", "completed"))
    source = _source(_StoreBackedViewer(store))
    reads = 0
    original = FrontendStateStore.state.fget

    def counted(self: FrontendStateStore) -> FrontendSessionState:
        nonlocal reads
        reads += 1
        assert original is not None
        return original(self)

    monkeypatch.setattr(FrontendStateStore, "state", property(counted))

    assert source.retained_for_auto_work is True
    assert reads == 0, "the retention predicate cloned canonical state for a boolean"


def test_the_predicate_answers_from_the_roster_it_is_asked_about() -> None:
    """Not vacuously true, and not vacuously false."""
    settled_store = FrontendStateStore(_state("completed"))
    running = _source(_StoreBackedViewer(FrontendStateStore(_state("running"))))
    settled = _source(_StoreBackedViewer(settled_store))
    empty = _source(_StoreBackedViewer(FrontendStateStore(_state())))

    assert running.retained_for_auto_work is True
    assert settled.retained_for_auto_work is False
    assert empty.retained_for_auto_work is False
    # PRECONDITION for the two False cases: the roster really holds no running
    # child, so they are not passing on a raised or absent read.
    assert [job.status for job in settled_store.state.jobs] == ["completed"]


def test_a_session_without_the_narrow_accessor_keeps_the_old_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback is live: a reduced facade must not lose the clause."""
    store = FrontendStateStore(_state("running"))
    source = _source(_StoreBackedViewer(store, narrow=False))
    reads = 0
    original = FrontendStateStore.state.fget

    def counted(self: FrontendStateStore) -> FrontendSessionState:
        nonlocal reads
        reads += 1
        assert original is not None
        return original(self)

    monkeypatch.setattr(FrontendStateStore, "state", property(counted))

    assert source.retained_for_auto_work is True
    assert reads == 1, "the fallback no longer reads the state it falls back to"


def test_an_unsynchronized_store_still_raises_through_the_predicate() -> None:
    """Whatever the read is, it must not turn \"no state yet\" into \"not busy\".

    The read sits ahead of ``approve_all`` for this reason, and moving the cheap
    clause in front of it would silently swallow the raise.
    """
    from local_operator.session.attached import AttachedSession

    class _Unsynced(AttachedSession):
        def __init__(self) -> None:  # deliberately not AttachedSession.__init__
            self._frontend_store = None

    source = SessionInteraction(_Unsynced())
    source.draft.approve_all = False

    with pytest.raises(RuntimeError, match="frontend state has not synchronized"):
        _ = source.retained_for_auto_work


def test_the_accessor_is_a_property_rather_than_a_method() -> None:
    """A method would be truthy at the probe, i.e. permanently "busy".

    The probe reads the accessor as a VALUE (``bool(accessor)``), so a session
    class that grew this member as a method would answer "a child is running" for
    every source, forever, with nothing raising. The protocol declares it as a
    property, which is what lets pyright hold both classes to that; this pins the
    shape at runtime as well, because the probe is blind to it.
    """
    import inspect

    from local_operator.session.attached import AttachedSession
    from local_operator.session.session import Session

    for cls in (AttachedSession, Session):
        assert isinstance(inspect.getattr_static(cls, "has_running_job"), property), (
            f"{cls.__name__}.has_running_job must be a property: the probe reads it "
            "as a value, and a bound method is always truthy"
        )


def test_the_store_predicate_agrees_with_the_state_it_replaces() -> None:
    """Equivalence with the read this replaces, over the shapes that reach it."""
    for statuses in ((), ("running",), ("completed",), ("running", "completed"), ("failed",)):
        store = FrontendStateStore(_state(*statuses))
        expected = any(job.status == "running" for job in store.state.jobs)
        assert store.has_running_job() is expected
