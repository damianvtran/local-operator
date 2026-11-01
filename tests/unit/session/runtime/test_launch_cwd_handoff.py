"""The engage guard that honours a session's OWN recorded cwd move.

A session that moved itself with nobody attached writes its destination to the
move marker and stops (``cwd_move.accept_handoff``). Every later engage passes
through ``engage_runtime``, which reads that record and starts the session in
the destination — unless the caller names a third directory, which is an
explicit choice and wins.

These cells drive the decision itself (``_engage_cwd_for_recorded_move``) and
then ONE engage, far enough to observe the directory the spawn receives. The real
spawn is the e2e cell's job (``tests/e2e/test_cwd_move_e2e.py``); a unit test that
forked a runtime would be paying for a race, not for this decision.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import cwd_move
from local_operator.session.runtime import launch
from local_operator.session.runtime.launch import (
    RuntimeStartupError,
    _engage_cwd_for_recorded_move,
    engage_runtime,
)

OWN = "engage000000001"
CONFIG = "cfg"


def _handoff(root: Path, *, source: Path, destination: Path) -> cwd_move.MoveIntent:
    """The state a viewer-less move leaves behind: recorded, not yet observed."""
    intent = cwd_move.enqueue_self_move(
        root, OWN, requested_cwd=str(destination), effective_cwd=str(source), owner_pid=1
    )
    handoff = cwd_move.accept_handoff(root, OWN, intent)
    assert handoff is not None and handoff.stage == cwd_move.STAGE_HANDOFF
    return handoff


@pytest.fixture()
def tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(config root, source directory, destination directory)."""
    config = tmp_path / CONFIG
    source = tmp_path / "src"
    destination = tmp_path / "dest"
    for path in (config, source, destination):
        path.mkdir()
    return config, source, destination


def test_no_record_means_the_callers_directory_is_used(tree: tuple[Path, Path, Path]) -> None:
    """The ordinary engage pays one `stat` and changes nothing."""
    config, source, _destination = tree
    assert _engage_cwd_for_recorded_move(config, OWN, str(source)) == str(source)
    assert _engage_cwd_for_recorded_move(config, OWN, "") == ""
    assert not cwd_move.marker_path(config, OWN).exists()


def test_a_queued_request_is_not_a_recorded_move(tree: tuple[Path, Path, Path]) -> None:
    """Only ``handoff`` says where the next process starts.

    A ``queued`` request belongs to the runtime that is still running: it has not
    recorded anything as the session's truth yet, so an engage must not act on it
    — the runtime's own idle point does.
    """
    config, source, destination = tree
    cwd_move.enqueue_self_move(
        config, OWN, requested_cwd=str(destination), effective_cwd=str(source), owner_pid=1
    )
    assert _engage_cwd_for_recorded_move(config, OWN, str(source)) == str(source)
    assert _engage_cwd_for_recorded_move(config, OWN, "") == ""


def test_a_caller_naming_the_source_gets_the_destination(
    tree: tuple[Path, Path, Path],
) -> None:
    """THE ORDINARY CASE: a wake, a message, or a shell in the old directory."""
    config, source, destination = tree
    _handoff(config, source=source, destination=destination)
    assert _engage_cwd_for_recorded_move(config, OWN, str(source)) == str(destination)


def test_a_caller_naming_nothing_gets_the_destination(tree: tuple[Path, Path, Path]) -> None:
    """An empty cwd is "wherever this session lives", which the record owns."""
    config, source, destination = tree
    _handoff(config, source=source, destination=destination)
    assert _engage_cwd_for_recorded_move(config, OWN, "") == str(destination)


def test_a_symlinked_spelling_of_the_source_still_gets_the_destination(
    tree: tuple[Path, Path, Path],
) -> None:
    """Both sides realpath-ed: ``/var`` vs ``/private/var`` is one directory."""
    config, source, destination = tree
    link = source.parent / "src-link"
    link.symlink_to(source)
    _handoff(config, source=source, destination=destination)
    assert _engage_cwd_for_recorded_move(config, OWN, str(link)) == str(destination)


def test_a_caller_naming_a_third_directory_keeps_it(tree: tuple[Path, Path, Path]) -> None:
    """AN EXPLICIT CHOICE WINS, and the record is settled by the boot that follows.

    No substitution and no error here: the runtime this engage starts reconciles
    the record against where it actually booted, which turns the handoff into
    ``superseded`` — the one place that decision is made.
    """
    config, source, destination = tree
    third = source.parent / "third"
    third.mkdir()
    _handoff(config, source=source, destination=destination)
    assert _engage_cwd_for_recorded_move(config, OWN, str(third)) == str(third)
    assert cwd_move.read_move_intent(config, OWN).stage == cwd_move.STAGE_HANDOFF
    # The boot that lands there settles it, once.
    settled = cwd_move.reconcile_move(config, OWN, effective_cwd=str(third))
    assert settled is not None and settled.stage == cwd_move.STAGE_SUPERSEDED
    # And a later engage is the caller's again, with nothing left to honour.
    assert _engage_cwd_for_recorded_move(config, OWN, str(source)) == str(source)


def test_a_caller_naming_the_destination_gets_the_destination(
    tree: tuple[Path, Path, Path],
) -> None:
    config, source, destination = tree
    _handoff(config, source=source, destination=destination)
    assert _engage_cwd_for_recorded_move(config, OWN, str(destination)) == str(destination)


def test_an_unreadable_record_refuses_the_engage(tree: tuple[Path, Path, Path]) -> None:
    """FAIL CLOSED: a wrong spawn is worse than a sentence.

    The document exists and cannot be parsed, so the destination is unknown.
    Proceeding would start the session in the directory the move was leaving;
    the engage raises instead, naming the file so it can be repaired or deleted.
    """
    config, source, _destination = tree
    marker = cwd_move.marker_path(config, OWN)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{not json")
    with pytest.raises(RuntimeStartupError) as refused:
        _engage_cwd_for_recorded_move(config, OWN, str(source))
    assert str(marker) in str(refused.value)
    assert "cannot be read" in str(refused.value)


def test_a_committed_move_is_not_replayed(tree: tuple[Path, Path, Path]) -> None:
    """Once OBSERVED, the record stops redirecting: the session is there already."""
    config, source, destination = tree
    _handoff(config, source=source, destination=destination)
    cwd_move.reconcile_move(config, OWN, effective_cwd=str(destination))
    assert _engage_cwd_for_recorded_move(config, OWN, str(source)) == str(source)


@pytest.mark.asyncio
async def test_the_spawn_receives_the_destination_not_the_callers_directory(
    tree: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """END TO END AT THE SEAM THAT MATTERS: what ``_spawn_runtime`` is handed.

    Everything between the guard and the fork is the engine's own arbitration, so
    the fake spawn stops the engage the moment it is asked to fork. The assertion
    is the argument, not a record: the record would be written by the child that
    never ran.
    """
    config, source, destination = tree
    _handoff(config, source=source, destination=destination)
    seen: list[str] = []

    def _refuse_spawn(session_id: str, cwd: str, **_: Any) -> Any:
        seen.append(cwd)
        raise RuntimeStartupError("spawn refused by the test")

    from local_operator import memory_guard

    monkeypatch.setattr(launch, "_spawn_runtime", _refuse_spawn)
    monkeypatch.setattr(
        memory_guard,
        "admission_check",
        lambda *_a, **_k: type("A", (), {"admitted": True, "reason": ""})(),
    )
    monkeypatch.setattr(launch, "find_runtime_record", lambda *_a, **_k: None, raising=False)
    errand = launch.WakeErrand(schedule_id="", occurrence_ms=0, command_id="c1")
    with pytest.raises((RuntimeStartupError, TimeoutError)):
        await asyncio.wait_for(
            engage_runtime(OWN, str(source), errand, config_dir=config, deadline_s=5.0),
            timeout=30,
        )
    assert seen == [str(destination)], "the fork is handed the recorded destination"
