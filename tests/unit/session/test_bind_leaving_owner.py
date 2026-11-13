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
