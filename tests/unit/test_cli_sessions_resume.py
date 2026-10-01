"""``lop sessions resume``: the arg→set mapping, exit codes and the preview.

The command's own contract is small — map flags to the shared selector, run
the shared batch runner, render, return the documented code — so these tests
double the SELECTOR and the RUNNER at their module boundary and assert the
wiring: what the command asks for, what it prints, and what it returns. The
runner's own behaviour (bounded parallelism, failure isolation, follow-up)
belongs to ``tests/unit/session/test_bulk_resume.py``; the real end-to-end
path is exercised by the CLI itself against a synthetic store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli as cli_module
from local_operator import helpers
from local_operator.session import bulk_resume
from local_operator.session.bulk_resume import ResumeOutcome, ResumeSelection


def _args(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        paused=False,
        failed=False,
        all_sessions=False,
        dry_run=False,
        limit=None,
        message=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _seed_session(root: Path, session_id: str, name: str) -> None:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
    (directory / "title.json").write_text(json.dumps({"title": name}), encoding="utf-8")


def _selection(rows: list[tuple[str, float]], matched: int | None = None) -> ResumeSelection:
    return ResumeSelection(
        sessions=tuple(rows), matched=len(rows) if matched is None else matched, kinds=frozenset()
    )


def _no_setup(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def spy() -> None:
        calls.append(1)

    monkeypatch.setattr(helpers, "setup_cross_platform_environment", spy)
    return calls


# --- the parser -------------------------------------------------------------


def test_the_parser_maps_every_flag() -> None:
    parser = cli_module.build_cli_parser()
    parsed = parser.parse_args(
        [
            "sessions",
            "resume",
            "--paused",
            "--failed",
            "--limit",
            "7",
            "--dry-run",
            "--message",
            "go",
        ]
    )
    assert parsed.sessions_command == "resume"
    assert parsed.paused is True
    assert parsed.failed is True
    assert parsed.all_sessions is False
    assert parsed.limit == 7
    assert parsed.dry_run is True
    assert parsed.message == "go"
    # And the default shape: no set, no cap.
    bare = parser.parse_args(["sessions", "resume", "--all"])
    assert bare.all_sessions is True and bare.paused is False and bare.limit is None


# --- refusal and previews ---------------------------------------------------


def test_no_set_is_misuse_with_exit_2(store: Path, capsys: Any) -> None:
    assert cli_module.sessions_resume_command(_args()) == 2
    assert "choose a set to resume" in capsys.readouterr().err


def test_a_blank_message_is_misuse(store: Path, capsys: Any) -> None:
    assert cli_module.sessions_resume_command(_args(paused=True, message="   ")) == 2
    assert "--message must be a non-empty message" in capsys.readouterr().err


def test_a_dry_run_lists_the_set_and_spawns_nothing(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _seed_session(store, "sess-a01", "paused five A")
    _seed_session(store, "sess-c01", "failed set A")
    captured: dict[str, Any] = {}

    def fake_select(root: Path, **kwargs: Any) -> ResumeSelection:
        captured.update(kwargs)
        return _selection([("sess-a01", 1.0), ("sess-c01", 2.0)])

    async def must_not_run(*args: Any, **kwargs: Any) -> list[ResumeOutcome]:  # pragma: no cover
        raise AssertionError("a dry run must not start anything")

    setup_calls = _no_setup(monkeypatch)
    monkeypatch.setattr(bulk_resume, "live_session_ids", lambda root: {"live-id"})
    monkeypatch.setattr(bulk_resume, "select_resume_candidates", fake_select)
    monkeypatch.setattr(bulk_resume, "resume_sessions", must_not_run)

    assert cli_module.sessions_resume_command(_args(paused=True, failed=True, dry_run=True)) == 0
    out = capsys.readouterr().out
    assert 'sess-a01  "paused five A"' in out
    assert 'sess-c01  "failed set A"' in out
    assert "nothing resumed (dry run)" in out
    assert captured["paused"] is True and captured["failed"] is True
    assert captured["exclude_ids"] == {"live-id"}
    # The PATH prime is for spawning commands only.
    assert setup_calls == []


def test_the_run_primes_the_path_once_and_maps_the_set(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _seed_session(store, "sess-a01", "one")
    captured: dict[str, Any] = {}

    def fake_select(root: Path, **kwargs: Any) -> ResumeSelection:
        captured.update(kwargs)
        return _selection([("sess-a01", 1.0)], matched=3)

    async def fake_run(rows: Any, **kwargs: Any) -> list[ResumeOutcome]:
        captured["rows"] = list(rows)
        captured["message"] = kwargs["message"]
        return [
            ResumeOutcome(session_id="sess-a01", name="one", ok=True, status="running", job_id="j1")
        ]

    setup_calls = _no_setup(monkeypatch)
    monkeypatch.setattr(bulk_resume, "live_session_ids", lambda root: set())
    monkeypatch.setattr(bulk_resume, "select_resume_candidates", fake_select)
    monkeypatch.setattr(bulk_resume, "resume_sessions", fake_run)

    rc = cli_module.sessions_resume_command(_args(all_sessions=True, limit=10, message="go on"))
    out = capsys.readouterr().out
    assert rc == 0
    assert setup_calls == [1]
    assert captured["all_sessions"] is True and captured["limit"] == 10
    assert captured["rows"] == [("sess-a01", "one")]
    assert captured["message"] == "go on"
    assert "1 session(s): 1 ok, 0 failed, 0 unresolved" in out
    assert "newest 1 of 3 matched" in out


def test_the_default_message_is_the_continuation(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(rows: Any, **kwargs: Any) -> list[ResumeOutcome]:
        captured["message"] = kwargs["message"]
        return []

    _no_setup(monkeypatch)
    monkeypatch.setattr(bulk_resume, "live_session_ids", lambda root: set())
    monkeypatch.setattr(
        bulk_resume, "select_resume_candidates", lambda root, **kwargs: _selection([])
    )
    monkeypatch.setattr(bulk_resume, "resume_sessions", fake_run)

    # An empty set never reaches the runner: it is answered directly.
    assert cli_module.sessions_resume_command(_args(paused=True)) == 0

    _seed_session(store, "sess-a01", "one")
    monkeypatch.setattr(
        bulk_resume,
        "select_resume_candidates",
        lambda root, **kwargs: _selection([("sess-a01", 1.0)]),
    )
    cli_module.sessions_resume_command(_args(paused=True))
    assert captured["message"] == bulk_resume.DEFAULT_RESUME_MESSAGE


def test_all_wins_in_the_header_and_the_selection(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """m3 (review round 1): `--paused --all` resolves to all and SAYS all.

    The header used to print "paused+all" while the selector intersected to
    the narrower set — a claim the behaviour contradicted. One story now:
    the widest reading wins, and the header names it.
    """
    _seed_session(store, "sess-a01", "one")
    captured: dict[str, Any] = {}

    def fake_select(root: Path, **kwargs: Any) -> ResumeSelection:
        captured.update(kwargs)
        return _selection([("sess-a01", 1.0)])

    async def fake_run(rows: Any, **kwargs: Any) -> list[ResumeOutcome]:
        return [
            ResumeOutcome(session_id="sess-a01", name="one", ok=True, status="running", job_id="j1")
        ]

    _no_setup(monkeypatch)
    monkeypatch.setattr(bulk_resume, "live_session_ids", lambda root: set())
    monkeypatch.setattr(bulk_resume, "select_resume_candidates", fake_select)
    monkeypatch.setattr(bulk_resume, "resume_sessions", fake_run)

    rc = cli_module.sessions_resume_command(_args(paused=True, all_sessions=True))
    out = capsys.readouterr().out
    assert rc == 0
    # The full-widening flags reach the shared selector (which defines
    # all-wins); the header prints the selection that was actually made.
    assert captured["all_sessions"] is True and captured["paused"] is True
    assert "(all, newest first)" in out
    assert "paused+all" not in out


def test_any_failed_session_makes_the_exit_code_nonzero(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _seed_session(store, "sess-ok", "fine")
    _seed_session(store, "sess-bad", "broken")

    async def fake_run(rows: Any, **kwargs: Any) -> list[ResumeOutcome]:
        return [
            ResumeOutcome(
                session_id="sess-ok", name="fine", ok=True, status="running", job_id="j1"
            ),
            ResumeOutcome(
                session_id="sess-bad",
                name="broken",
                ok=False,
                status="failed",
                job_id="j2",
                detail="session sess-bad is already open in another process (pid 42)",
            ),
        ]

    _no_setup(monkeypatch)
    monkeypatch.setattr(bulk_resume, "live_session_ids", lambda root: set())
    monkeypatch.setattr(
        bulk_resume,
        "select_resume_candidates",
        lambda root, **kwargs: _selection([("sess-ok", 2.0), ("sess-bad", 1.0)]),
    )
    monkeypatch.setattr(bulk_resume, "resume_sessions", fake_run)

    rc = cli_module.sessions_resume_command(_args(paused=True))
    out = capsys.readouterr().out
    assert rc == 1
    assert "1 ok, 1 failed, 0 unresolved" in out
    assert "already open in another process" in out


def test_an_unresolved_session_also_fails_the_exit_code(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _seed_session(store, "sess-slow", "slow")

    async def fake_run(rows: Any, **kwargs: Any) -> list[ResumeOutcome]:
        return [
            ResumeOutcome(
                session_id="sess-slow",
                name="slow",
                ok=False,
                status="unresolved",
                job_id="j9",
                detail="still starting after 30s — check `lop exec --status j9`",
            )
        ]

    _no_setup(monkeypatch)
    monkeypatch.setattr(bulk_resume, "live_session_ids", lambda root: set())
    monkeypatch.setattr(
        bulk_resume,
        "select_resume_candidates",
        lambda root, **kwargs: _selection([("sess-slow", 1.0)]),
    )
    monkeypatch.setattr(bulk_resume, "resume_sessions", fake_run)

    rc = cli_module.sessions_resume_command(_args(paused=True))
    out = capsys.readouterr().out
    assert rc == 1
    assert "0 ok, 0 failed, 1 unresolved" in out
    assert "--status j9" in out
