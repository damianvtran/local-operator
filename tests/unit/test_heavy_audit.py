"""The heavy-job lease audit trail (``local_operator/heavy_audit.py``).

Why these cells exist: the trail is only worth having if it can PROVE two heavy
jobs serialized, survives a holder killed with SIGKILL, stays bounded, and never
leaks a command's arguments or environment. So the cells that matter drive real
processes (the real ``lop services heavy run`` CLI, a real holder killed with
SIGKILL) against an isolated config dir and read the jsonl file back.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator import heavy_audit, heavy_cli, heavy_lease

pytestmark = pytest.mark.skipif(not heavy_lease.supported(), reason="flock is POSIX-only")

ROOT = Path(__file__).resolve().parents[2]

#: A fake credential: it must appear in the command's argv and env and NEVER
#: in the trail. Built at runtime so no scanner mistakes this file for a leak.
FAKE_TOKEN = "ghp_" + "FAKEtoken0123456789abcdef"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, "1")
    monkeypatch.setenv("LOP_I18N", "0")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    return {**clean, "PYTHONPATH": str(ROOT)}


def _events(directory: Path) -> list[dict[str, Any]]:
    path = heavy_audit.audit_path(directory)
    return [json.loads(line) for line in path.read_text().splitlines()]


def _lop_run(env: dict[str, str], *argv: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "local_operator.cli", "services", "heavy", "run", "--", *argv],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _reap(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)


def _wait_for(condition, timeout: float = 20.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.02)


# -- serialization -------------------------------------------------------------


def test_two_real_runs_log_non_overlapping_intervals(env: dict[str, str]) -> None:
    """take→release→take: the second run waited, and its interval starts after the first ends."""
    directory = heavy_lease.lease_dir()
    first = _lop_run(env, "sh", "-c", "sleep 1.5")
    try:
        _wait_for(lambda: heavy_lease.read_holders(directory))
        second = _lop_run(env, "sh", "-c", "exit 3")
        try:
            assert first.wait(timeout=30) == 0
            assert second.wait(timeout=30) == 3
        finally:
            _reap(second)
    finally:
        _reap(first)
    events = _events(directory)
    assert [e["event"] for e in events] == ["take", "release", "take", "release"]
    take1, rel1, take2, rel2 = events
    assert take1["pid"] == rel1["pid"] == first.pid
    assert take2["pid"] == rel2["pid"] == second.pid
    assert {e["slot"] for e in events} == {0}
    assert take1["label"] == "sh" and rel1["exit"] == 0 and rel2["exit"] == 3
    assert take2["waited_s"] > 0.5, take2
    # Interval proof: first held [take1.ts, rel1.ts - small], second took after it.
    first_end = rel1["ts"]
    second_start = take2["ts"]
    assert rel1["held_s"] >= 1.4
    # The release line is written just after the unlock, so the next take can
    # precede it by a hair; the held interval itself never overlaps.
    first_unlocked = take1["ts"] + rel1["held_s"]
    assert second_start >= first_unlocked - 0.05, (first_unlocked, second_start)
    assert first_end - second_start < 0.5


def test_take_records_waited_and_memory(tmp_path: Path) -> None:
    lease = heavy_lease.try_acquire(
        tmp_path,
        1,
        session_id="sess1",
        command="pnpm build --filter web",
        waited_s=2.25,
        memory_mb=4096,
    )
    assert lease is not None
    lease.release(exit_code=0)
    take, release = _events(tmp_path)
    assert take == {
        "ts": take["ts"],
        "event": "take",
        "slot": 0,
        "waited_s": 2.25,
        "pid": os.getpid(),
        "session": "sess1",
        "label": "pnpm build",
        "memory_mb": 4096,
    }
    assert release["event"] == "release" and release["exit"] == 0
    assert release["session"] == "sess1" and release["held_s"] >= 0


def test_timeout_is_logged(tmp_path: Path) -> None:
    holder = heavy_lease.try_acquire(tmp_path, 1, command="make -j8")
    assert holder is not None
    try:

        async def _no_sleep(_s: float) -> None:
            await asyncio.sleep(0)

        result = asyncio.run(
            heavy_lease.acquire(
                tmp_path, 1, wait_seconds=0.05, reason="r", command="cargo test", sleep=_no_sleep
            )
        )
    finally:
        holder.release()
    assert isinstance(result, heavy_lease.Refusal)
    timeout = [e for e in _events(tmp_path) if e["event"] == "timeout"]
    assert len(timeout) == 1
    assert timeout[0]["label"] == "cargo test" and timeout[0]["waited_s"] >= 0.05
    assert timeout[0]["holders"] == [os.getpid()]
    assert "slot" not in timeout[0]


# -- stale reclaim -------------------------------------------------------------

_HOLDER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from local_operator import heavy_lease
    lease = heavy_lease.try_acquire(Path(sys.argv[1]), 1, session_id="dead1", command="make test")
    assert lease is not None
    Path(sys.argv[2]).write_text("1")
    time.sleep(60)
""")


def test_killed_holder_is_logged_as_stale_reclaim(env: dict[str, str], tmp_path: Path) -> None:
    directory = heavy_lease.lease_dir()
    marker = tmp_path / "held"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(directory), str(marker)],
        env=env,
        start_new_session=True,
    )
    try:
        _wait_for(marker.exists)
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        _reap(proc)
    lease = heavy_lease.try_acquire(directory, 1, command="pytest -q")
    assert lease is not None
    lease.release()
    events = _events(directory)
    assert [e["event"] for e in events] == ["take", "stale-reclaim", "take", "release"]
    stale = events[1]
    assert stale["pid"] == proc.pid and stale["session"] == "dead1"
    assert stale["label"] == "make test" and stale["reclaimed_by"] == os.getpid()
    assert stale["slot"] == 0 and stale["held_s"] >= 0


def test_clean_release_is_not_a_stale_reclaim(tmp_path: Path) -> None:
    for _ in range(2):
        lease = heavy_lease.try_acquire(tmp_path, 1, command="make")
        assert lease is not None
        lease.release()
    assert [e["event"] for e in _events(tmp_path)] == ["take", "release", "take", "release"]


# -- rotation and best effort ---------------------------------------------------


def test_rotates_at_the_cap_and_keeps_bounded_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(heavy_audit, "AUDIT_MAX_BYTES", 400)
    path = heavy_audit.audit_path(tmp_path)
    for index in range(60):
        heavy_audit.record(tmp_path, "take", slot=0, pid=index, label="make")
    generations = sorted(
        p.name for p in tmp_path.glob("heavy-lease.jsonl*") if not p.name.endswith(".lock")
    )
    assert generations == [
        "heavy-lease.jsonl",
        "heavy-lease.jsonl.1",
        "heavy-lease.jsonl.2",
        "heavy-lease.jsonl.3",
    ]
    # Each file stays near the cap: a file rotates once it reaches it, so it
    # can exceed it by at most one line.
    for name in generations:
        assert (tmp_path / name).stat().st_size < 400 + 200
    # Newest events are in the live file; tail reads across the boundary.
    assert _events(tmp_path)[-1]["pid"] == 59
    assert [e["pid"] for e in heavy_audit.tail(tmp_path, 5)] == [55, 56, 57, 58, 59]
    assert path.stat().st_size < 400 + 200


def test_write_failure_never_breaks_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = os.open

    def _fail(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path).endswith(heavy_audit.AUDIT_FILENAME):
            raise OSError(28, "No space left on device")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", _fail)
    lease = heavy_lease.try_acquire(tmp_path, 1, command="make")
    assert lease is not None and lease.held
    lease.release(exit_code=1)
    assert not lease.held


# -- privacy -------------------------------------------------------------------


def test_trail_holds_no_argv_or_env_secret(env: dict[str, str]) -> None:
    """A real run with a token in argv and env logs only argv[0] + subcommand."""
    directory = heavy_lease.lease_dir()
    secret_env = {**env, "GITHUB_TOKEN": FAKE_TOKEN}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "local_operator.cli",
            "services",
            "heavy",
            "run",
            "--",
            # echo, not a real client: the cell is about what the trail keeps,
            # and nothing should act on these arguments.
            "echo",
            "deploy",
            f"--token={FAKE_TOKEN}",
            f"https://x:{FAKE_TOKEN}@example.invalid",
            FAKE_TOKEN,
        ],
        env=secret_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        proc.communicate(timeout=30)
    finally:
        _reap(proc)
    raw = heavy_audit.audit_path(directory).read_text()
    assert FAKE_TOKEN not in raw
    assert "example.invalid" not in raw and "--token" not in raw
    labels = {e["label"] for e in _events(directory)}
    assert labels == {"echo deploy"}


@pytest.mark.parametrize(
    ("command", "label"),
    [
        (f"env GITHUB_TOKEN={FAKE_TOKEN} make test", "make test"),
        (f"GITHUB_TOKEN={FAKE_TOKEN} pnpm build", "pnpm build"),
        ("cd web && pnpm build --filter x", "pnpm build"),
        (f"tool --token {FAKE_TOKEN.lower()} deploy", "tool"),
        (f"curl https://u:{FAKE_TOKEN}@host", "curl"),
        (f"tool {FAKE_TOKEN}", "tool"),
        ("/usr/bin/python3 -m pytest", "python3"),
        ("uv run pytest tests/unit", "pytest"),
        # A leased no-op is still named (it once hid behind a setup-word skip).
        ("true", "true"),
        ("cd web", "?"),
        ("", "?"),
        # Everyday subcommands survive the tighter rule (PR #4264 R2/Q2).
        ("make test", "make test"),
        ("npm run build", "npm run"),
        ("cargo build --release", "cargo build"),
        ("pnpm install", "pnpm install"),
        ("aws s3 ls", "aws s3"),
        ("npm run-script lint", "npm run-script"),
        # Lowercase secret-shaped argv[1] is dropped (PR #4264 R2/Q2).
        ("deploy " + "a1b2c3d4" * 4, "deploy"),
        ("deploy a1b2c3d4e5f6", "deploy"),
        ("mytool sk-live-abcdef", "mytool"),
        ("mytool sk_test_abc", "mytool"),
        ("mytool glpat-abcdefg", "mytool"),
        ("mytool xoxb-abc-def", "mytool"),
        ("job build20261119", "job"),
        ("job x1234", "job"),
        ("tool abcdefghijklmnopqrstu", "tool"),
        ("tool script.js", "tool"),
        ("tool trailing-", "tool"),
    ],
)
def test_command_label_keeps_only_program_and_subcommand(command: str, label: str) -> None:
    assert heavy_audit.command_label(command) == label


# -- read surfaces ---------------------------------------------------------------


def test_status_prints_holder_then_last_five_events(env: dict[str, str]) -> None:
    directory = heavy_lease.lease_dir()
    for _ in range(3):
        done = heavy_lease.try_acquire(directory, 1, session_id="s1", command="make check")
        assert done is not None
        done.release(exit_code=0)
    lease = heavy_lease.try_acquire(directory, 1, session_id="s1", command="make check")
    assert lease is not None
    out = io.StringIO()
    try:
        assert heavy_cli.status_command(out=out) == 0
    finally:
        lease.release()
    lines = out.getvalue().splitlines()
    assert lines[0] == "heavy jobs: 1/1 slots held: s1/%d make check (0m)" % os.getpid()
    assert lines[1].startswith(f"  slot 0: s1/{os.getpid()}, running ")
    path = heavy_audit.audit_path(directory)
    assert lines[2] == f"  recent lease events (log: {path}):"
    events = lines[3:]
    assert len(events) == heavy_cli.STATUS_EVENTS == 5
    # 3 take/release pairs plus the live take = 7 events; the last 5 shown.
    assert [line.split()[1] for line in events] == ["take", "release", "take", "release", "take"]
    newest = events[-1]
    assert newest.startswith("    20") and "Z take slot=0 pid=" in newest
    assert 'session=s1 label="make check" waited_s=0.0' in newest


def test_status_with_no_events_says_so(env: dict[str, str]) -> None:
    out = io.StringIO()
    assert heavy_cli.status_command(out=out) == 0
    path = heavy_audit.audit_path(heavy_lease.lease_dir())
    assert out.getvalue().splitlines() == [
        "heavy jobs: 0/1 slots in use",
        f"  no lease events recorded yet (log: {path})",
    ]


def test_services_status_audit_row_is_one_line(env: dict[str, str]) -> None:
    from local_operator import services

    directory = heavy_lease.lease_dir()
    # No trail yet: no row (a fresh install's status stays as it was).
    assert services._heavy_job_lines() == ["heavy jobs: 0/1 slots in use"]
    lease = heavy_lease.try_acquire(directory, 1, command="make test", memory_mb=2048)
    assert lease is not None
    lease.release(exit_code=2)
    lines = services._heavy_job_lines()
    assert len(lines) == 2 and lines[0] == "heavy jobs: 0/1 slots in use"
    row = lines[1]
    assert row.startswith(f"heavy jobs audit: last release slot=0 pid={os.getpid()} ")
    assert 'label="make test"' in row and "exit=2 memory_mb=2048" in row
    assert row.endswith(f" ago (log: {heavy_audit.audit_path(directory)})")
    assert "\n" not in row


# -- PR #4264 round-1 remediation ------------------------------------------------


def test_wait_timeout_is_logged(env: dict[str, str]) -> None:
    """Q1: `heavy wait` giving up leaves the same timeout line `heavy run` does."""
    directory = heavy_lease.lease_dir()
    holder = heavy_lease.try_acquire(directory, 1, command="make -j8")
    assert holder is not None
    ticks = iter([0.0, 2.5])
    try:
        code = heavy_cli.wait_command(
            1.0, out=io.StringIO(), sleep=lambda _s: None, clock=lambda: next(ticks)
        )
    finally:
        holder.release()
    assert code == heavy_cli.EXIT_TIMED_OUT
    timeout = [e for e in _events(directory) if e["event"] == "timeout"]
    assert len(timeout) == 1
    assert timeout[0]["label"] == heavy_cli.WAIT_LABEL == "heavy wait"
    assert timeout[0]["waited_s"] == 2.5 and timeout[0]["pid"] == os.getpid()
    assert timeout[0]["holders"] == [os.getpid()]
    assert "slot" not in timeout[0]


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    return proc.pid


def test_corrupt_stale_record_never_leaves_the_slot_locked(tmp_path: Path) -> None:
    """R1: an overflowing ``started_at`` must not escape try_acquire's audit."""
    huge = "1" + "0" * 400  # a JSON int float() cannot represent
    (tmp_path / "slot-0.json").write_text(
        f'{{"slot": 0, "pid": {_dead_pid()}, "command": "make", "started_at": {huge}}}'
    )
    lease = heavy_lease.try_acquire(tmp_path, 1, command="make test")
    assert lease is not None
    lease.release()
    again = heavy_lease.try_acquire(tmp_path, 1, command="make test")
    assert again is not None, "slot stayed locked after the audit raised"
    again.release()


def test_signal_exit_is_logged_as_128_plus_n(tmp_path: Path) -> None:
    """R3: the bash tool's raw ``-9`` and `heavy run`'s 137 log the same code."""
    for code in (-9, 137, 0, 2):
        lease = heavy_lease.try_acquire(tmp_path, 1, command="make")
        assert lease is not None
        lease.release(exit_code=code)
    exits = [e.get("exit") for e in _events(tmp_path) if e["event"] == "release"]
    assert exits == [137, 137, 0, 2]
    assert heavy_lease.shell_exit_code(None) is None
    assert heavy_cli._exit_code(-15) == heavy_lease.shell_exit_code(-15) == 143


_TOKEN_HOLDER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    Path(sys.argv[1]).write_text("1")
    time.sleep(60)
""")


def _leaking_surfaces(token: str) -> list[str]:
    """Which holder surfaces print ``token``: record file, heavy status, services status."""
    from local_operator import services

    directory = heavy_lease.lease_dir()
    leaks = []
    records = "".join(p.read_text() for p in directory.glob("slot-*.json"))
    if token in records:
        leaks.append("slot record")
    out = io.StringIO()
    heavy_cli.status_command(out=out)
    if token in out.getvalue():
        leaks.append("heavy status")
    if token in "\n".join(services._heavy_job_lines()):
        leaks.append("services status")
    return leaks


def test_holder_surfaces_hold_no_argv_or_env_secret(env: dict[str, str], tmp_path: Path) -> None:
    """Q3: a live holder's record and both status rows show the label only."""
    directory = heavy_lease.lease_dir()
    marker = tmp_path / "held"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "local_operator.cli",
            "services",
            "heavy",
            "run",
            "--",
            sys.executable,
            "-c",
            _TOKEN_HOLDER,
            str(marker),
            f"--token={FAKE_TOKEN}",
            f"https://x:{FAKE_TOKEN}@example.invalid",
            FAKE_TOKEN,
        ],
        env={**env, "GITHUB_TOKEN": FAKE_TOKEN},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _wait_for(marker.exists)
        record = json.loads((directory / "slot-0.json").read_text())
        assert record["command"] == os.path.basename(sys.executable)
        assert _leaking_surfaces(FAKE_TOKEN) == []
        assert _leaking_surfaces("example.invalid") == []
        holders = heavy_lease.read_holders(directory)
        assert [h.command for h in holders] == [os.path.basename(sys.executable)]
    finally:
        _reap(proc)


def test_older_raw_preview_record_is_relabelled_on_read(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q3 compatibility: a pre-upgrade holder's raw preview is never printed.

    Includes the negative control: with the old pass-through read restored the
    same surfaces DO leak, so the clean assertion is not vacuous.
    """
    directory = heavy_lease.lease_dir()
    directory.mkdir(parents=True, exist_ok=True)
    preview = f"deploy --token={FAKE_TOKEN} prod"[: heavy_lease.PREVIEW_CHARS]
    (directory / "slot-0.json").write_text(
        json.dumps(
            {
                "slot": 0,
                "pid": os.getpid(),
                "session_id": "old1",
                "command": preview,
                "started_at": time.time(),
                "pgid": None,
            }
        )
    )
    assert [h.command for h in heavy_lease.read_holders(directory)] == ["deploy"]
    assert _leaking_surfaces(FAKE_TOKEN) == ["slot record"]  # the old file itself
    assert "old1/" in heavy_lease.status_line(directory, 1)
    monkeypatch.setattr(heavy_lease, "_holder_label", lambda command: str(command or ""))
    assert _leaking_surfaces(FAKE_TOKEN) == ["slot record", "heavy status", "services status"]
