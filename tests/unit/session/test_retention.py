"""The retention module after the incident: liveness claims only, no deletion.

What used to be here tested a sweep that reaped "empty" directories and, in
its last form, "unused" ones. Both are gone, and the central property of
this file has inverted: it is no longer "only empty directories go", it is
**nothing goes** — not from the retention module (which has no remover
any more), and not from the startup maintenance pass that used to call it,
whatever shape a directory has and whatever the config says short of the
user explicitly enabling ``session.cleanup``.
"""

from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

import pytest

import local_operator.session.retention as retention
from local_operator.session.retention import (
    CLAIM_TRUST_S,
    LIVE_MARKER_NAME,
    _is_claimed,
    _process_alive,
    claim_session,
    release_session,
)

# ---------------------------------------------------------------------------
# The module has no remover
# ---------------------------------------------------------------------------


def test_the_retention_module_cannot_delete_anything() -> None:
    """No ``rmtree``, ``rmdir``, ``rename``/``replace``/``move`` and no
    ``shutil`` import at all: the module's whole vocabulary is claim, release
    and probe. ``unlink`` appears exactly once, on the liveness MARKER."""
    source = inspect.getsource(retention)
    for forbidden in ("rmtree", "rmdir", "os.rename", "os.replace", "shutil.move", "import shutil"):
        assert forbidden not in source, forbidden
    assert source.count(".unlink(") == 1
    for name in ("sweep_sessions", "sweep_from_config", "reap_unused_sessions", "_holds_content"):
        assert not hasattr(retention, name), f"{name} came back"


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def test_claim_creates_the_directory_and_names_this_process(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "abc"
    claim_session(session)
    assert session.is_dir()
    assert (session / LIVE_MARKER_NAME).read_text() == str(os.getpid())
    assert _is_claimed(session, time.time())


def test_release_drops_the_marker(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "abc"
    claim_session(session)
    release_session(session)
    assert not (session / LIVE_MARKER_NAME).exists()
    assert session.is_dir(), "release must never remove the directory"


def test_release_leaves_a_leased_directory_alone(tmp_path: Path) -> None:
    """A leased session drops its mirror through the lease's own hook; an
    unconditional unlink here could erase a successor's mirror."""
    session = tmp_path / "sessions" / "abc"
    claim_session(session)
    (session / ".execution-lease").write_text("{}")
    release_session(session)
    assert (session / LIVE_MARKER_NAME).exists()


def test_claims_are_confined_to_the_sessions_store(tmp_path: Path) -> None:
    """An agent directory is exported and published; a marker there would
    escape into the bundle."""
    agent = tmp_path / "agents" / "abc"
    agent.mkdir(parents=True)
    claim_session(agent)
    assert not (agent / LIVE_MARKER_NAME).exists()
    (agent / LIVE_MARKER_NAME).write_text("1")
    release_session(agent)
    assert (agent / LIVE_MARKER_NAME).exists()


def test_a_dead_pid_is_not_a_claim(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "abc"
    session.mkdir(parents=True)
    (session / LIVE_MARKER_NAME).write_text("999999999")
    assert not _is_claimed(session, time.time())


@pytest.mark.parametrize("raw", ["", "not-a-pid", "-1", "0"])
def test_an_unparseable_or_impossible_marker_is_not_a_claim(tmp_path: Path, raw: str) -> None:
    session = tmp_path / "sessions" / "abc"
    session.mkdir(parents=True)
    (session / LIVE_MARKER_NAME).write_text(raw)
    assert not _is_claimed(session, time.time())


def test_process_alive_probe() -> None:
    assert _process_alive(os.getpid())
    assert not _process_alive(999999999)
    assert not _process_alive(0)
    assert not _process_alive(-5)


def test_on_an_unverifiable_platform_a_stale_claim_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where liveness cannot be probed (Windows) a claim is bounded by
    ``CLAIM_TRUST_S`` from the later of the marker and the last write."""
    monkeypatch.setattr(retention, "_PLATFORM", "win32")
    monkeypatch.setattr(retention, "_LIVENESS_IS_VERIFIABLE", False)
    session = tmp_path / "sessions" / "stale"
    session.mkdir(parents=True)
    marker = session / LIVE_MARKER_NAME
    marker.write_text("12345")
    old = time.time() - CLAIM_TRUST_S - 3600
    os.utime(marker, (old, old))
    assert not _is_claimed(session, time.time())


def test_on_an_unverifiable_platform_a_fresh_claim_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention, "_PLATFORM", "win32")
    monkeypatch.setattr(retention, "_LIVENESS_IS_VERIFIABLE", False)
    session = tmp_path / "sessions" / "fresh"
    session.mkdir(parents=True)
    (session / LIVE_MARKER_NAME).write_text("12345")
    assert _is_claimed(session, time.time())


def test_on_an_unverifiable_platform_activity_extends_a_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long quiet session whose marker is old but whose transcript was
    written recently is still claimed: the bound is on activity, not age."""
    monkeypatch.setattr(retention, "_PLATFORM", "win32")
    monkeypatch.setattr(retention, "_LIVENESS_IS_VERIFIABLE", False)
    session = tmp_path / "sessions" / "busy"
    session.mkdir(parents=True)
    marker = session / LIVE_MARKER_NAME
    marker.write_text("12345")
    old = time.time() - CLAIM_TRUST_S - 3600
    os.utime(marker, (old, old))
    (session / "transcript.jsonl").write_text("row\n")  # written now
    assert _is_claimed(session, time.time())


# ---------------------------------------------------------------------------
# The startup maintenance pass removes nothing
# ---------------------------------------------------------------------------


def _mixed_store(root: Path) -> set[str]:
    """Every shape the old reapers targeted, plus the ones they spared."""
    sessions = root / "sessions"
    sessions.mkdir(parents=True)
    ancient = time.time() - 400 * 86400
    shapes: dict[str, dict[str, str]] = {
        "empty": {},
        "marker-only-dead": {LIVE_MARKER_NAME: "999999999"},
        "lease-only": {".execution-lease": '{"generation": "g", "pid": 999999999}'},
        "sidecar-only": {"attachment.json": "{}"},
        "origin-only": {"origin.json": '{"origin": "subagent"}'},
        "zero-transcript": {"transcript.jsonl": ""},
        "machine-only-transcript": {
            "transcript.jsonl": '{"type": "model_route", "payload": {}}\n'
            '{"type": "incident", "payload": {"role": "system"}}\n'
        },
        "real-transcript": {
            "transcript.jsonl": '{"type": "message", "payload": {"role": "user"}}\n'
        },
    }
    for name, files in shapes.items():
        directory = sessions / name
        directory.mkdir()
        for filename, body in files.items():
            (directory / filename).write_text(body, encoding="utf-8")
        for entry in (directory, *directory.iterdir()):
            os.utime(entry, (ancient, ancient))
    return set(shapes)


@pytest.mark.asyncio
async def test_store_maintenance_removes_nothing_from_a_mixed_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the real maintenance coroutine over a store holding every
    reapable shape of the old policies, with a default config. Every
    directory must survive, and the cleanup log must not exist."""
    from local_operator import session_factory
    from local_operator.config import ConfigManager
    from local_operator.session.cleanup import CLEANUP_LOG_NAME

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    expected = _mixed_store(tmp_path)
    manager = ConfigManager(tmp_path)

    async def no_wait() -> None:
        return None

    monkeypatch.setattr(session_factory, "_wait_for_store_maintenance_idle_window", no_wait)
    await session_factory._run_store_maintenance(manager, tmp_path, live_dir=None)

    survivors = {p.name for p in (tmp_path / "sessions").iterdir() if p.is_dir()}
    assert survivors == expected
    assert not (tmp_path / "sessions" / CLEANUP_LOG_NAME).exists()


@pytest.mark.asyncio
async def test_store_maintenance_ignores_aggressive_limits_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator import session_factory
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    expected = _mixed_store(tmp_path)
    manager = ConfigManager(tmp_path)
    manager.update_config(
        {
            "session": {
                "cleanup": {
                    "enabled": False,
                    "max_sessions": 1,
                    "max_inactive_days": 1,
                    "max_total_bytes": 1,
                    "remove_empty": True,
                }
            }
        }
    )

    async def no_wait() -> None:
        return None

    monkeypatch.setattr(session_factory, "_wait_for_store_maintenance_idle_window", no_wait)
    await session_factory._run_store_maintenance(ConfigManager(tmp_path), tmp_path, live_dir=None)

    survivors = {p.name for p in (tmp_path / "sessions").iterdir() if p.is_dir()}
    assert survivors == expected


# ---------------------------------------------------------------------------
# The activity clock
# ---------------------------------------------------------------------------


def test_sidecar_list_names_every_bookkeeping_file_the_harness_writes() -> None:
    """``_SIDECAR_NAMES`` spells the names as literals (import weight); this
    pins them to the canonical constants so a renamed sidecar cannot quietly
    start counting as activity again."""
    from local_operator import resume
    from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR
    from local_operator.session_lease import LEASE_NAME, MIRROR_NAME, RECOVERY_LOCK_NAME

    expected = {
        MIRROR_NAME,
        LEASE_NAME,
        RECOVERY_LOCK_NAME,
        resume.ATTACHMENT_SIDECAR_NAME,
        resume.ORIGIN_NAME,
        resume.TITLE_SIDECAR_NAME,
        resume.ORIGIN_SCAN_SENTINEL_NAME,
        resume.TITLE_SCAN_SENTINEL_NAME,
        resume.ORIGIN_CACHE_NAME,
        SUBAGENT_ROSTER_SIDECAR,
    }
    assert retention._SIDECAR_NAMES == frozenset(expected)


def test_activity_is_the_transcript_or_spool_and_nothing_else(tmp_path: Path) -> None:
    from local_operator.session.retention import _activity_mtime

    session = tmp_path / "sessions" / "abc"
    session.mkdir(parents=True)
    old = time.time() - 40 * 86400
    (session / "transcript.jsonl").write_text("row\n")
    os.utime(session / "transcript.jsonl", (old, old))
    # Every sidecar, an unknown file, a nested output file and the directory
    # itself are all "now" — none may move the clock.
    for name in sorted(retention._SIDECAR_NAMES) + ["notes.unknown"]:
        (session / name).write_text("x")
    (session / "attachments").mkdir()
    (session / "attachments" / "shot.png").write_bytes(b"png")
    assert abs(_activity_mtime(session, 0.0) - old) < 1.0
    # The spool DOES count: an unread note is activity waiting for the user.
    (session / "inbox.jsonl").write_text('{"from":"peer"}\n')
    assert _activity_mtime(session, 0.0) > time.time() - 5
    # Neither file: the fallback.
    bare = tmp_path / "sessions" / "bare"
    bare.mkdir()
    assert _activity_mtime(bare, 12345.0) == 12345.0
