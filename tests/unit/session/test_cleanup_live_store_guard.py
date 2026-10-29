"""A test process may not remove from the operator's LIVE store. Fail closed.

WHY THIS TEST EXISTS. On 2026-10-29 a lane's pytest run executed against the
real ``$HOME`` and a real session directory was gone afterwards. Every guard in
``session/cleanup.py`` asks whether the TARGET is a legitimate store — marked,
directly under ``sessions/``, inside the config dir the process was given — and
a test process handed the operator's real config dir passes all of them, because
on paper it is doing something legal. Nothing asked whether THIS PROCESS should
be touching the live store at all.

The guard is exercised on a store that IS live as far as the code can tell: the
live-store set (derived from the passwd home, so it is the operator's real tree
and not a redirected ``$HOME``) is monkeypatched to name this test's ``tmp_path``
store. That is the only way to prove the refusal without pointing a test at
``~/.local-operator`` — a test that proved it by deleting from the real store
would be the incident, not the proof.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from local_operator.session import cleanup
from local_operator.session.cleanup import (
    ALLOW_TEST_STORE_DELETIONS_ENV,
    CLEANUP_LOG_NAME,
    STORE_MARKER_NAME,
    mark_store,
    remove_session_dir,
)


@pytest.fixture(autouse=True)
def _no_inherited_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this file starts with the opt-in ABSENT.

    THE HAZARD, in the ambient-isolation suite's own words: an inherited
    ESCAPE HATCH "makes every cell take the allow path while looking like it
    tested the refusal". ``ALLOW_TEST_STORE_DELETIONS_ENV`` is registered in
    ``tests/conftest.py::_AMBIENT_VARS`` so the whole run scrubs it; this
    fixture states the same guarantee beside the cells that depend on it, so
    the file stays honest if that entry is ever removed or this file is run
    somewhere the global fixture does not apply. The cells that DO mean the
    opt-in set it through ``monkeypatch`` in their own body.
    """
    monkeypatch.delenv(ALLOW_TEST_STORE_DELETIONS_ENV, raising=False)


def _store(root: Path, session_id: str = "abc") -> Path:
    """A marked store with one real-looking session directory; returns the target."""
    sessions = root / "sessions"
    target = sessions / session_id
    target.mkdir(parents=True)
    (target / "transcript.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
    mark_store(sessions)
    return target


def _live(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Make ``root``'s store the one this process believes is the LIVE store."""
    monkeypatch.setattr(cleanup, "_live_session_stores", lambda: ((root / "sessions").resolve(),))


def test_pytest_is_detected_as_a_test_run() -> None:
    """The detection itself, before any store is involved: the suite IS a test run.

    Pinned as a test so the ambient signal is not assumed. ``PYTEST_CURRENT_TEST``
    is set by pytest for the duration of every test, and the module check is the
    fallback that covers collection and teardown; either is enough.
    """
    assert cleanup._test_run_reason() is not None


def test_the_opt_in_cannot_be_inherited_from_the_environment() -> None:
    """A shell that exported the opt-in cannot waive these refusal cells.

    CI found the gap this pins: the variable was read by production code but
    declared nowhere, so an operator (or a rig) with it exported made every cell
    in this file take the ALLOW path while looking like it tested the refusal —
    a fail-closed guard proven by nothing. It is now scrubbed for the whole run
    (``tests/conftest.py::_AMBIENT_VARS``) and again by the autouse fixture above;
    this asserts the second one is actually in force.
    """
    assert ALLOW_TEST_STORE_DELETIONS_ENV not in os.environ


def test_a_test_run_is_refused_on_the_live_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The reproduction: a live store + a test process = refused, nothing removed.

    The refusal must land BEFORE the receipt is written, so a refused removal
    leaves no row claiming it happened — ``.cleanup-log.jsonl`` must not exist
    here at all.
    """
    target = _store(tmp_path)
    _live(monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING, logger="local_operator.session.cleanup"):
        done = remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r", actor="test")

    assert done is False
    assert target.is_dir(), "the live store's session directory must survive"
    assert (target / "transcript.jsonl").is_file()
    assert not (
        tmp_path / "sessions" / CLEANUP_LOG_NAME
    ).exists(), "a refused removal must not leave a receipt claiming it happened"
    refusal = [r.getMessage() for r in caplog.records if "REFUSED" in r.getMessage()]
    assert refusal, caplog.text
    sentence = refusal[0]
    # The message names what was refused AND the way to opt in; a refusal an
    # operator cannot act on is the kind that gets deleted instead of understood.
    assert str(target) in sentence
    assert "LIVE store" in sentence
    assert ALLOW_TEST_STORE_DELETIONS_ENV in sentence


def test_the_explicit_opt_in_permits_the_same_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same store, same test process, one env var: the removal goes through.

    The escape hatch has to be usable, or the legitimate cases (proving a
    removal's own behaviour against a live-shaped store) cannot run at all.
    """
    target = _store(tmp_path)
    _live(monkeypatch, tmp_path)
    monkeypatch.setenv(ALLOW_TEST_STORE_DELETIONS_ENV, "1")

    done = remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r", actor="test")

    assert done is True
    assert not target.exists()
    rows = (tmp_path / "sessions" / CLEANUP_LOG_NAME).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1, rows


@pytest.mark.parametrize("value", ["0", "false", "no", "", "maybe"])
def test_a_value_that_is_not_a_yes_does_not_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """``LOCAL_OPERATOR_ALLOW_TEST_STORE_DELETIONS=0`` is not a permission (fail closed).

    The variable is read only to find an ENABLED override; anything else leaves
    the guard exactly as it was. A guard that treats "set at all" as consent is
    one an inherited environment variable can switch off silently.
    """
    target = _store(tmp_path)
    _live(monkeypatch, tmp_path)
    monkeypatch.setenv(ALLOW_TEST_STORE_DELETIONS_ENV, value)

    assert (
        remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r", actor="test")
        is False
    )
    assert target.is_dir()


def test_a_temporary_store_is_not_refused_under_a_test_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control: the guard is inert for every rig's ``tmp_path``.

    Without this, a guard that refused everything would pass the tests above and
    take the whole suite down with it.
    """
    target = _store(tmp_path)
    # NOT patched: the real live-store set is in force, and this store is not in it.
    assert (tmp_path / "sessions").resolve() not in cleanup._live_session_stores()

    assert (
        remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r", actor="test")
        is True
    )
    assert not target.exists()


def test_a_redirected_home_is_not_the_live_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HOME=<scratch>`` keeps the guard out of an isolated rig, by construction.

    This is the isolation recipe every rig here uses, so it must NOT read as the
    operator's store: the live store is derived from the passwd home, never from
    ``$HOME``. Pinning it stops a future "just add Path.home()" from turning
    every isolated run into a refusal.
    """
    scratch_home = tmp_path / "scratch-home"
    (scratch_home / cleanup.DEFAULT_CONFIG_DIRNAME).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(scratch_home))
    target = _store(scratch_home / cleanup.DEFAULT_CONFIG_DIRNAME)

    assert (
        remove_session_dir(
            target,
            config_dir=scratch_home / cleanup.DEFAULT_CONFIG_DIRNAME,
            policy="p",
            reason="r",
            actor="test",
        )
        is True
    )
    assert not target.exists()


def test_every_destructive_path_is_gated_not_just_the_directory_remover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, m-1: the gate reaches all three non-``sessions/`` removers.

    ``_live_store_refusal`` is consulted from ``_refusal``, which carries
    ``remove_session_dir`` and ``hand_off_session_dir``. Three other paths delete
    store data without going through it, and each is asked the same question here:
    a pad's content, a just-minted directory, and the aged mesh-handoff
    quarantine. None is reachable by a test process on the live store any more.
    """
    sessions = tmp_path / "sessions"
    target = sessions / "abc"
    scratch = target / "scratchpad"
    scratch.mkdir(parents=True)
    (target / "transcript.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
    pad_file = scratch / "build.bin"
    pad_file.write_bytes(b"y" * 64)
    fresh = sessions / "fresh"
    fresh.mkdir()
    quarantine = tmp_path / "network" / cleanup.HANDED_AWAY_DIRNAME / "abc.1.deadbeef"
    quarantine.mkdir(parents=True)
    (quarantine / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    os.utime(quarantine, (0, 0))
    mark_store(sessions)
    _live(monkeypatch, tmp_path)

    # 1. a pad's content (the delegated class's content phase)
    assert (
        cleanup.remove_scratchpad_entry(pad_file, config_dir=tmp_path, sessions_dir=sessions)
        is False
    )
    assert pad_file.is_file()
    # 2. a just-minted directory (wake-rollback / failed-create)
    assert (
        cleanup.remove_fresh_session_dir(
            fresh, config_dir=tmp_path, policy="failed-create", reason="r", actor="test"
        )
        is False
    )
    assert fresh.is_dir()
    # 3. the aged handoff quarantine (mesh move housekeeping)
    assert cleanup.sweep_handed_away(tmp_path) == []
    assert (quarantine / "transcript.jsonl").is_file()


@pytest.mark.parametrize(
    "remover",
    ["remove_scratchpad_entry", "remove_fresh_session_dir", "sweep_handed_away"],
)
def test_the_opt_in_also_permits_the_other_destructive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remover: str
) -> None:
    """The escape hatch reaches every gated path, or the gate is a trap.

    A deliberate run against a live-shaped store (the legitimate case: proving one
    of these removers' own behaviour) must be able to proceed with one variable.
    """
    sessions = tmp_path / "sessions"
    target = sessions / "abc"
    scratch = target / "scratchpad"
    scratch.mkdir(parents=True)
    pad_file = scratch / "build.bin"
    pad_file.write_bytes(b"y" * 64)
    fresh = sessions / "fresh"
    fresh.mkdir()
    quarantine = tmp_path / "network" / cleanup.HANDED_AWAY_DIRNAME / "abc.1.deadbeef"
    quarantine.mkdir(parents=True)
    (quarantine / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    os.utime(quarantine, (0, 0))
    mark_store(sessions)
    _live(monkeypatch, tmp_path)
    monkeypatch.setenv(ALLOW_TEST_STORE_DELETIONS_ENV, "1")

    if remover == "remove_scratchpad_entry":
        assert cleanup.remove_scratchpad_entry(pad_file, config_dir=tmp_path, sessions_dir=sessions)
        assert not pad_file.exists()
    elif remover == "remove_fresh_session_dir":
        assert cleanup.remove_fresh_session_dir(
            fresh, config_dir=tmp_path, policy="failed-create", reason="r", actor="test"
        )
        assert not fresh.exists()
    else:
        assert cleanup.sweep_handed_away(tmp_path) == [quarantine.name]
        assert not quarantine.exists()


def test_an_unmarked_live_store_still_reports_the_marker_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering: the store-identity refusal wins over the test-run sentence.

    Both are true for an unmarked path that is also live, and the marker reason is
    the more specific one — it says the path is not a store at all. Pinned so a
    later reordering cannot make every refusal read "you are a test run".
    """
    sessions = tmp_path / "sessions"
    target = sessions / "abc"
    target.mkdir(parents=True)
    _live(monkeypatch, tmp_path)

    assert (
        remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r", actor="test")
        is False
    )
    assert target.is_dir()
    assert not (sessions / STORE_MARKER_NAME).exists()
