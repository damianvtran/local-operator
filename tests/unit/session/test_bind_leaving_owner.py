"""A bind that engaged a RETIRING owner must engage its successor once the record goes.

The race (CI witness: #3823's gate run 43431981085, job 131924458441,
``test_runtime_refresh_e2e.py::...[reset-before-frame]``): a retiring owner
marks its record ``leaving`` and only unpublishes it after its dispose. A
refresh-cold viewer's bind engaged in that window, so ``engage_runtime``
answered ``LEAVING_DETAIL`` and spawned nothing; milliseconds later the
bind's locate found no record and raised "could not start a runtime for this
session". Nothing retried, and the viewer stayed cold until the next keystroke.

These tests drive ``_bind_under_lock`` against a fake owner seam, so the
interleaving is scripted rather than timed: no sleeps, no real runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attached import AttachedSession
from local_operator.session.owner import LOCAL_PLACEMENT
from local_operator.session.runtime.launch import LEAVING_DETAIL, EngageOutcome
from tests.unit.session.test_remote_cold import _never, _seed_transcript

SESSION = "s1"
SUCCESSOR = object()  # the successor's record; only its identity is checked


def _outcome(detail: str) -> EngageOutcome:
    return EngageOutcome(session_id=SESSION, detail=detail)


class _ScriptedOwner:
    """An owner seam whose engage answers and locate results are scripted in order."""

    placement = LOCAL_PLACEMENT

    def __init__(self, outcomes: list[Any], locates: list[tuple[Any, Any]]) -> None:
        self._outcomes = list(outcomes)
        self._locates = list(locates)
        self.engages: list[dict[str, Any]] = []

    def locate(self) -> tuple[Any, Any]:
        return self._locates.pop(0) if self._locates else (None, None)

    async def engage(self, **kwargs: Any) -> Any:
        self.engages.append(kwargs)
        return self._outcomes.pop(0) if self._outcomes else _outcome("runtime ready")

    def make_client(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("_bind_to is stubbed; no client is built")


async def _viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: _ScriptedOwner
) -> tuple[AttachedSession, list[Any]]:
    _seed_transcript(tmp_path, SESSION)
    viewer = await AttachedSession.cold(
        SESSION,
        config_dir=tmp_path,
        cwd=str(tmp_path),
        takeover_factory=_never,
        owner=owner,
    )
    bound: list[Any] = []

    async def fake_bind_to(record: Any, **kwargs: Any) -> None:
        bound.append(record)

    monkeypatch.setattr(viewer, "_bind_to", fake_bind_to)
    return viewer, bound


@pytest.mark.asyncio
async def test_a_leaving_owner_whose_record_vanishes_gets_its_successor_engaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The witness's interleaving: LEAVING answer, record gone, then one re-engage binds."""
    owner = _ScriptedOwner(
        outcomes=[
            _outcome(LEAVING_DETAIL),
            EngageOutcome(SESSION, "runtime ready", spawned=True),
        ],
        locates=[(None, None), (SUCCESSOR, 4321)],
    )
    viewer, bound = await _viewer(tmp_path, monkeypatch, owner)
    try:
        await viewer._bind_under_lock(foreground=False)
        assert bound == [SUCCESSOR], "the viewer must bind the successor, not stay cold"
        assert len(owner.engages) == 2, "exactly one re-engage after the LEAVING answer"
        # The re-engage is clamped to what is left of the bind's own budget, so
        # it never widens a deadline; the first engage carries no deadline at all.
        assert "deadline_s" not in owner.engages[0]
        from local_operator.session.attached import _BACKGROUND_BIND_BUDGET_S

        assert 0 < owner.engages[1]["deadline_s"] <= _BACKGROUND_BIND_BUDGET_S
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_vanished_record_without_a_leaving_answer_keeps_the_old_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: no LEAVING answer, no re-engage, and the error wording is unchanged."""
    owner = _ScriptedOwner(outcomes=[_outcome("runtime ready")], locates=[(None, None)])
    viewer, bound = await _viewer(tmp_path, monkeypatch, owner)
    try:
        with pytest.raises(ConnectionError, match="^could not start a runtime for this session$"):
            await viewer._bind_under_lock(foreground=False)
        assert bound == []
        assert len(owner.engages) == 1
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_an_owner_seam_that_returns_nothing_keeps_the_old_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote owner's engage returns ``None``: that is not a LEAVING answer."""
    owner = _ScriptedOwner(outcomes=[None], locates=[(None, None)])
    viewer, bound = await _viewer(tmp_path, monkeypatch, owner)
    try:
        with pytest.raises(ConnectionError, match="could not start a runtime"):
            await viewer._bind_under_lock(foreground=False)
        assert len(owner.engages) == 1
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_the_re_engage_happens_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successor engage that ALSO answers LEAVING and finds nothing must not loop."""
    owner = _ScriptedOwner(
        outcomes=[
            _outcome(LEAVING_DETAIL),
            _outcome(LEAVING_DETAIL),
            _outcome(LEAVING_DETAIL),
        ],
        locates=[(None, None), (None, None), (None, None), (None, None)],
    )
    viewer, bound = await _viewer(tmp_path, monkeypatch, owner)
    try:
        with pytest.raises(ConnectionError, match="could not start a runtime"):
            await viewer._bind_under_lock(foreground=False)
        assert bound == []
        assert len(owner.engages) == 2, "the leaving-owner retry is bounded to one"
    finally:
        await viewer.dispose()


# -- the UPDATE-WINDOW retirement: no ``leaving`` mark (QA r1 Q1 on #3939) ------
#
# The witness's idle owner retires through the update window, which publishes
# ``updating`` and never ``leaving``. So the REAL engage answers ``runtime
# ready`` against its still-published record and spawns nothing; the owner then
# writes ``runtime-exit.json``, releases its lease and unpublishes, in that
# order (``process.py``'s exit path). These cells run the real ``LocalOwner``
# and the real ``engage_runtime`` against a real published record and a real
# lease held by this process; only the re-engage is scripted, so no runtime is
# ever spawned.

REAL_SESSION = "updwindow1"
_UPDATING_PAIR = "0.102.3@aaaaaaa → 0.102.4@f4a70b9"


class _RetiringAfterEngage:
    """Wraps the real ``LocalOwner``: the first engage is REAL, then the owner retires.

    ``exit`` is what the owner leaves behind: ``None`` for a crash (no record),
    else ``(reason, requested, started_at)`` for ``journal.write_exit_record``.
    """

    def __init__(self, real: Any, *, config: Path, lease: Any, record: Any, exit: Any) -> None:
        self._real = real
        self.placement = real.placement
        self._config = config
        self._lease = lease
        self._record = record
        self._exit = exit
        self.engages: list[dict[str, Any]] = []
        self.first_outcome: Any = None
        self.successor: Any = None

    def locate(self) -> tuple[Any, Any]:
        return self._real.locate()

    def make_client(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("_bind_to is stubbed; no client is built")

    async def engage(self, **kwargs: Any) -> Any:
        import dataclasses
        import os

        from local_operator.session.runtime import journal, registry
        from local_operator.session_lease import acquire_session_lease

        self.engages.append(kwargs)
        session_dir = self._config / "sessions" / REAL_SESSION
        if len(self.engages) == 1:
            outcome = await self._real.engage(**kwargs)
            self.first_outcome = outcome
            # The retirement, in the real owner's order: exit record, lease,
            # then the unpublish.
            if self._exit is not None:
                reason, requested, started_at = self._exit
                assert journal.write_exit_record(
                    REAL_SESSION,
                    conversation_dir=session_dir,
                    pid=self._record.pid,
                    started_at=started_at,
                    reason=reason,
                    requested=requested,
                )
            self._lease.release()
            registry.unpublish(self._record.pid, self._config)
            return outcome
        # The scripted successor: a fresh run of this process, published the
        # way a real successor publishes (lease first, then the record).
        self._lease = acquire_session_lease(session_dir, pid=os.getpid())
        self.successor = dataclasses.replace(
            self._record, started_at=self._record.started_at + 1.0, updating=""
        )
        registry.publish(self.successor, self._config)
        return EngageOutcome(REAL_SESSION, "runtime ready", spawned=True)

    def cleanup(self) -> None:
        from local_operator.session.runtime import registry

        registry.unpublish(self._record.pid, self._config)
        self._lease.release()


async def _update_window_rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, exit: Any
) -> tuple[AttachedSession, _RetiringAfterEngage, list[Any]]:
    import os
    import time

    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import PROTOCOL_VERSION, SessionRecord
    from local_operator.session_lease import acquire_session_lease

    config = tmp_path / ".local-operator"
    _seed_transcript(config, REAL_SESSION)
    session_dir = config / "sessions" / REAL_SESSION
    session_dir.mkdir(parents=True, exist_ok=True)
    lease = acquire_session_lease(session_dir, pid=os.getpid())
    record = SessionRecord(
        pid=os.getpid(),
        kind="tui",
        session_id=REAL_SESSION,
        conversation_name="update-window",
        cwd=str(tmp_path),
        model_label="test/model",
        control_port=1,
        control_key="k" * 16,
        protocol=PROTOCOL_VERSION,
        capabilities=["tui_state_v1"],
        started=True,
        started_at=time.time(),
        # The update window's own mark, and NO leaving mark: the state QA
        # froze a real owner in (record present, ``leaving=""``).
        updating=_UPDATING_PAIR,
    )
    registry.publish(record, config)
    viewer = await AttachedSession.cold(
        REAL_SESSION, config_dir=config, cwd=str(tmp_path), takeover_factory=_never
    )
    owner = _RetiringAfterEngage(
        viewer._owner, config=config, lease=lease, record=record, exit=exit
    )
    viewer._owner = owner
    # The run this viewer was bound to before the refresh sent it cold.
    viewer._runtime_record = record
    bound: list[Any] = []

    async def fake_bind_to(rec: Any, **kwargs: Any) -> None:
        bound.append(rec)

    monkeypatch.setattr(viewer, "_bind_to", fake_bind_to)
    return viewer, owner, bound


@pytest.mark.asyncio
async def test_an_update_window_retirement_gets_its_successor_engaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA r1 Q1: a ``runtime ready`` answer from a record with no leaving mark, then gone."""
    viewer, owner, bound = await _update_window_rig(tmp_path, monkeypatch, exit=None)
    run = viewer._runtime_record
    assert run is not None
    owner._exit = ("retiring for 0.102.4@f4a70b9", False, run.started_at)
    try:
        await viewer._bind_under_lock(foreground=False)
        # The precondition is the REAL engage's answer, not a scripted one.
        assert owner.first_outcome is not None
        assert owner.first_outcome.detail == "runtime ready"
        assert owner.first_outcome.spawned is False
        assert len(owner.engages) == 2, "exactly one re-engage after the retirement"
        assert "deadline_s" not in owner.engages[0]
        from local_operator.session.attached import _BACKGROUND_BIND_BUDGET_S

        assert 0 < owner.engages[1]["deadline_s"] <= _BACKGROUND_BIND_BUDGET_S
        assert [r.started_at for r in bound] == [owner.successor.started_at]
    finally:
        await viewer.dispose()
        owner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit",
    [
        # ``lop stop``: a requested exit is a stop, never a refresh.
        pytest.param(("stopped", True, None), id="deliberate-stop"),
        # The requested flag alone must hold, whatever the reason says.
        pytest.param(("retiring for 0.102.4@f4a70b9", True, None), id="requested-retiring"),
        # A crash / SIGKILL writes no exit record at all.
        pytest.param(None, id="crash-no-exit-record"),
        # An EARLIER run's retirement cannot explain this run's disappearance.
        pytest.param(("retiring for 0.102.4@f4a70b9", False, -1.0), id="earlier-run"),
    ],
)
async def test_a_vanished_record_that_was_not_retired_keeps_the_old_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit: Any
) -> None:
    """Negative controls on the same real rig: no re-engage, today's exact wording."""
    viewer, owner, bound = await _update_window_rig(tmp_path, monkeypatch, exit=None)
    if exit is not None:
        reason, requested, started_at = exit
        run = viewer._runtime_record
        assert run is not None
        run_started = run.started_at
        owner._exit = (
            reason,
            requested,
            run_started if started_at is None else run_started + started_at,
        )
    try:
        with pytest.raises(ConnectionError, match="^could not start a runtime for this session$"):
            await viewer._bind_under_lock(foreground=False)
        assert owner.first_outcome is not None and owner.first_outcome.spawned is False
        assert bound == []
        assert len(owner.engages) == 1, "a stop or a crash must not be silently restarted"
    finally:
        await viewer.dispose()
        owner.cleanup()
