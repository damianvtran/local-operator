"""What directory a RESPAWN starts in when the caller has no opinion.

A wake, a bridge delivery and a peer note all reach the fork boundary with the
empty cwd sentinel. Before this cell existed the boundary answered HOME, so a
session that lives in ``~/workspace`` restarted in ``~``: block 0 then loses the
repository guidance of the cwd's ancestors, which is a NEW ``system_prefix``
epoch and a full prompt-cache rewrite for every restart (measured on this fleet
at roughly USD 50 a week for one chief-of-staff session).

The rule these cells pin, in order: a recorded move, then a caller's explicit
cwd (HOME included), then the session's own durable record, then HOME.

The spawn is faked, exactly as ``test_launch_cwd_handoff`` does: what matters is
the argument the fork route receives, and a real fork would pay for a race
rather than for this decision.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator.session import cwd_move
from local_operator.session.runtime import launch
from local_operator.session.runtime.launch import (
    RuntimeStartupError,
    _durable_cwd_for_restart,
    _spawn_runtime,
    engage_runtime,
)

OWN = "durablecwd000001"
CONFIG = "cfg"


def _tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(config root, the session's own directory, a different directory)."""
    config = tmp_path / CONFIG
    own = tmp_path / "own"
    other = tmp_path / "other"
    for path in (config, own, other):
        path.mkdir()
    (config / "sessions" / OWN).mkdir(parents=True)
    return config, own, other


def _record_cwd(config: Path, cwd: Path, session_id: str = OWN) -> None:
    """Write the marker the desktop door leaves at create."""
    directory = config / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "desktop.json").write_text(
        json.dumps({"version": 1, "cwd": str(cwd)}), encoding="utf-8"
    )


@contextlib.contextmanager
def _captured_spawn() -> Iterator[list[str]]:
    """Capture the cwd the fork route is handed, then restore the real one."""
    import local_operator.session.runtime.launch as launch_module

    seen: list[str] = []
    original = launch_module._spawn_runtime_local
    launch_module._spawn_runtime_local = (  # type: ignore[assignment]
        lambda _sid, cwd, **_k: seen.append(cwd)
    )
    try:
        yield seen
    finally:
        launch_module._spawn_runtime_local = original  # type: ignore[assignment]


@pytest.fixture()
def spawn_seen(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the cwd of an ENGAGE's spawn, and stop the engage there."""
    from local_operator import memory_guard
    from local_operator.session.runtime import session_host

    seen: list[str] = []

    def _refuse_spawn(session_id: str, cwd: str, **_: Any) -> Any:
        seen.append(cwd)
        raise RuntimeStartupError("spawn refused by the test")

    monkeypatch.setattr(session_host, "spawn_authority", lambda: session_host.AUTHORITY_LOCAL)
    monkeypatch.setattr(launch, "_spawn_runtime_local", _refuse_spawn)
    monkeypatch.setattr(
        memory_guard,
        "admission_check",
        lambda *_a, **_k: type("A", (), {"admitted": True, "reason": ""})(),
    )
    monkeypatch.setattr(launch, "find_runtime_record", lambda *_a, **_k: None, raising=False)
    return seen


# --------------------------------------------------------------------------
# The record itself
# --------------------------------------------------------------------------


def test_the_marker_is_the_sessions_own_directory(tmp_path: Path) -> None:
    config, own, _other = _tree(tmp_path)
    _record_cwd(config, own)
    assert _durable_cwd_for_restart(config, OWN) == str(own)


def test_a_session_with_no_record_has_no_opinion(tmp_path: Path) -> None:
    """A SUBAGENT-shaped session (no marker, no ledger, no wake index) keeps HOME."""
    config, _own, _other = _tree(tmp_path)
    assert _durable_cwd_for_restart(config, OWN) == ""


def test_the_wake_index_is_the_second_source(tmp_path: Path) -> None:
    """No desktop marker (a terminal-created session): the ledger still answers."""
    from local_operator.harness.wake_types import WakeSchedule
    from local_operator.wakes.store import write_entry

    config, own, _other = _tree(tmp_path)
    row = WakeSchedule(id="w1", message="tick", next_due_at=4_000_000_000_000)
    written = write_entry(config, OWN, cwd=str(own), schedules=[row])
    assert written is not None
    assert _durable_cwd_for_restart(config, OWN) == str(own)


def test_a_recorded_directory_that_is_gone_is_skipped(tmp_path: Path) -> None:
    """A deleted tree must degrade to HOME, never to a fork that cannot start."""
    config, own, _other = _tree(tmp_path)
    _record_cwd(config, own)
    own.rmdir()
    assert _durable_cwd_for_restart(config, OWN) == ""


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b"])
def test_a_session_id_that_is_not_a_plain_name_is_refused(tmp_path: Path, bad: str) -> None:
    config, _own, _other = _tree(tmp_path)
    assert _durable_cwd_for_restart(config, bad) == ""


# --------------------------------------------------------------------------
# The sentinel at the fork boundary
# --------------------------------------------------------------------------


def test_an_empty_cwd_starts_in_the_sessions_own_directory(tmp_path: Path) -> None:
    config, own, _other = _tree(tmp_path)
    _record_cwd(config, own)
    with _captured_spawn() as seen:
        _spawn_runtime(OWN, "", defer_materialise=False, config_dir=config)
    assert seen == [str(own)]


def test_an_explicit_cwd_still_wins_over_the_record(tmp_path: Path) -> None:
    """A caller naming a directory is never second-guessed."""
    config, own, other = _tree(tmp_path)
    _record_cwd(config, own)
    with _captured_spawn() as seen:
        _spawn_runtime(OWN, str(other), defer_materialise=False, config_dir=config)
    assert seen == [str(other)]


def test_an_explicit_home_wins_over_the_record(tmp_path: Path) -> None:
    """HOME IS A VALID EXPLICIT OVERRIDE: it is not the omission sentinel."""
    config, own, _other = _tree(tmp_path)
    _record_cwd(config, own)
    with _captured_spawn() as seen:
        _spawn_runtime(OWN, str(Path.home()), defer_materialise=False, config_dir=config)
    assert seen == [str(Path.home())]


def test_a_recorded_move_outranks_the_record(tmp_path: Path) -> None:
    """The move destination is the session's address; the marker is older news."""
    config, own, other = _tree(tmp_path)
    _record_cwd(config, own)
    intent = cwd_move.enqueue_self_move(
        config, OWN, requested_cwd=str(other), effective_cwd=str(own), owner_pid=1
    )
    handoff = cwd_move.accept_handoff(config, OWN, intent)
    assert handoff is not None and handoff.stage == cwd_move.STAGE_HANDOFF
    with _captured_spawn() as seen:
        _spawn_runtime(OWN, "", defer_materialise=False, config_dir=config)
    assert seen == [str(other)]


def test_a_caller_naming_the_move_source_follows_the_move(tmp_path: Path) -> None:
    """PRE-EXISTING, UNCHANGED: naming the source IS the move's documented case.

    ``cwd_move.spawn_cwd_for_session`` treats a request for the directory a move
    left as a request for the session, so a wake or a peer note still standing in
    the old tree lands in the destination. That rule is the fix for a move that
    survived exactly one start, and it keeps priority over this PR's "an explicit
    cwd wins" rule — the two collide only here, and this cell pins which one wins.
    """
    config, own, other = _tree(tmp_path)
    _record_cwd(config, other)
    intent = cwd_move.enqueue_self_move(
        config, OWN, requested_cwd=str(other), effective_cwd=str(own), owner_pid=1
    )
    handoff = cwd_move.accept_handoff(config, OWN, intent)
    assert handoff is not None and handoff.stage == cwd_move.STAGE_HANDOFF
    with _captured_spawn() as seen:
        _spawn_runtime(OWN, str(own), defer_materialise=False, config_dir=config)
    assert seen == [str(other)]


def test_no_record_and_no_move_falls_back_to_home(tmp_path: Path) -> None:
    config, _own, _other = _tree(tmp_path)
    with _captured_spawn() as seen:
        _spawn_runtime(OWN, "", defer_materialise=False, config_dir=config)
    assert seen == [str(Path.home())]


# --------------------------------------------------------------------------
# The two real callers that pass the sentinel: a wake and a peer note
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("errand_kind", ["wake", "peer", "warm"])
async def test_a_cold_wake_a_cold_peer_note_and_a_warm_engage_start_in_the_sessions_directory(
    tmp_path: Path, spawn_seen: list[str], errand_kind: str
) -> None:
    config, own, _other = _tree(tmp_path)
    _record_cwd(config, own)
    errand: Any
    if errand_kind == "wake":
        errand = launch.WakeErrand(schedule_id="", occurrence_ms=0, command_id="c1")
    elif errand_kind == "peer":
        errand = launch.PeerMessageErrand(
            text="hello", mode="mailbox", wake=True, sender={"pid": 0}
        )
    else:
        errand = launch.WarmErrand()
    with pytest.raises((RuntimeStartupError, TimeoutError)):
        await asyncio.wait_for(
            engage_runtime(OWN, "", errand, config_dir=config, deadline_s=5.0),
            timeout=30,
        )
    assert spawn_seen == [str(own)], "the respawn keeps the session in its own directory"


def test_the_hosted_fork_route_is_handed_the_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host route forks the child, but the DIRECTORY is decided here."""
    from local_operator.session.runtime import session_host

    config, own, _other = _tree(tmp_path)
    _record_cwd(config, own)
    seen: list[str] = []
    monkeypatch.setattr(session_host, "spawn_authority", lambda: session_host.AUTHORITY_HOST)
    monkeypatch.setattr(launch, "_spawn_runtime_hosted", lambda _sid, cwd, **_k: seen.append(cwd))
    _spawn_runtime(OWN, "", defer_materialise=False, config_dir=config)
    assert seen == [str(own)]
