"""The machine-wide heavy-job lease (``local_operator/heavy_lease.py``).

Why these cells exist: on 2026-11-16 parallel builds from several sessions
OOMed a 16 GB node while every command stayed under its own memory-guard
ceiling. The lease is the node-wide limit, so the cells that matter most are
the REAL cross-process ones: two OS processes, one flock'd slot, critical
sections that must not overlap, and a holder killed by exact pid freeing it.
Every cell uses an isolated config dir and a small wait so it stays fast.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from local_operator import heavy_lease

pytestmark = pytest.mark.skipif(not heavy_lease.supported(), reason="flock is POSIX-only")

ROOT = Path(__file__).resolve().parents[3]


# -- classification -----------------------------------------------------------

HEAVY = [
    ("pnpm build", "pnpm build"),
    ("pnpm run build", "pnpm build"),
    ("npm run build", "npm build"),
    ("yarn build", "yarn build"),
    ("cd ui && pnpm build", "pnpm build"),
    ("cd ui && NODE_OPTIONS=--max-old-space-size=4096 pnpm -C app run build:web", "pnpm build:web"),
    ("pnpm --filter desktop test", "pnpm test"),
    ("npx electron-vite build", "electron-vite build"),
    ("vite build --mode production", "vite build"),
    ("tsc --noEmit -p .", "tsc"),
    ("npx tsc -b", "tsc"),
    ("esbuild src/index.ts --bundle --outfile=out.js", "esbuild"),
    ("vitest run", "vitest"),
    ("pnpm exec vitest run src", "vitest"),
    ("jest --ci", "jest"),
    ("node --test test/", "node --test"),
    ("pytest -n 4 tests/unit", "pytest -n"),
    ("python -m pytest -nauto -q", "pytest -n"),
    ("uv run pytest -n 4", "pytest -n"),
    ("uv run --with x pytest --numprocesses=8", "pytest -n"),
    ("make test-affected", "make test"),
    ("timeout 600 make test", "make test"),
    ("cargo build --release", "cargo build"),
    ("go test ./...", "go test"),
    ("systemd-run --user --scope -p MemoryMax=4G -- pnpm build", "pnpm build"),
    ("nice -n 10 timeout -k 5 900 vitest run", "vitest"),
    ("env CI=1 FOO=bar jest", "jest"),
    ("git status && pnpm build 2>&1 | tail -20", "pnpm build"),
]

NOT_HEAVY = [
    "ls -la",
    "git status",
    "pytest tests/unit/x.py -q",
    "pytest -n 0 tests/x.py",
    "node -e 'console.log(1)'",
    "tsc --version",
    "npx esbuild --version",
    "vite",
    "electron-vite dev",
    "pnpm install",
    "npm build",
    "cargo fmt",
    "make lint",
    "echo 'pnpm build'",
    "grep -rn vitest src/",
    "",
]


@pytest.mark.parametrize("command,reason", HEAVY)
def test_heavy_commands_are_classified(command: str, reason: str) -> None:
    heavy = heavy_lease.is_heavy(command)
    assert heavy is not None, command
    assert heavy.reason == reason


@pytest.mark.parametrize("command", NOT_HEAVY)
def test_short_commands_are_not_heavy(command: str) -> None:
    assert heavy_lease.is_heavy(command) is None


def test_a_large_memory_request_is_heavy_on_its_own() -> None:
    assert heavy_lease.is_heavy("printf x", 1024) == heavy_lease.HeavyClass("memory_mb=1024")
    assert heavy_lease.is_heavy("printf x", 1023) is None
    assert heavy_lease.is_heavy("printf x", None) is None


# -- slot sizing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "total_mb,available_mb,expected",
    [
        (16 * 1024, 15 * 1024, 1),  # node2-sized: one slot even with lots free
        (15_700, 13_500, 1),
        (32 * 1024, 30 * 1024, 3),
        (32 * 1024, 7 * 1024, 1),  # crowded big host: never zero
        (128 * 1024, 120 * 1024, 4),  # capped
        (None, None, 1),  # unmeasurable: the safe answer
        (64 * 1024, None, 1),
    ],
)
def test_auto_slot_count(total_mb: int | None, available_mb: int | None, expected: int) -> None:
    assert heavy_lease.auto_slot_count(total_mb, available_mb) == expected


def test_env_overrides_set_slots_and_wait(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, "3")
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_WAIT_SECONDS_ENV, "7.5")
    settings = heavy_lease.configured_settings()
    assert (settings.enabled, settings.slots, settings.wait_seconds) == (True, 3, 7.5)
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_WAIT_SECONDS_ENV, "garbage")
    assert heavy_lease.configured_settings().wait_seconds == 600


# -- real cross-process serialisation --------------------------------------------

_HOLDER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from local_operator import heavy_lease
    directory, log, hold = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    while True:
        lease = heavy_lease.try_acquire(directory, 1, session_id=sys.argv[4], command="pnpm build")
        if lease is not None:
            break
        time.sleep(0.02)
    log.write_text(f"{time.time()}\\n")
    (log.parent / (log.name + ".held")).write_text("1")
    time.sleep(hold)
    end = time.time()
    lease.release()
    log.write_text(log.read_text() + f"{end}\\n")
""")


def _spawn_holder(directory: Path, log: Path, hold: float, session: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(directory), str(log), str(hold), session],
        env=env,
        start_new_session=True,
    )


def _wait_for(path: Path, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)


def _reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)


def test_two_processes_never_overlap_with_one_slot(tmp_path: Path) -> None:
    directory = tmp_path / "heavy-jobs"
    logs = [tmp_path / "a.log", tmp_path / "b.log"]
    procs = [_spawn_holder(directory, log, 0.4, f"s{i}") for i, log in enumerate(logs)]
    try:
        for proc in procs:
            assert proc.wait(timeout=15) == 0
    finally:
        for proc in procs:
            _reap(proc)
    spans = sorted(tuple(float(x) for x in log.read_text().split()) for log in logs)
    assert all(len(span) == 2 for span in spans)
    (a_start, a_end), (b_start, b_end) = spans
    assert a_end <= b_start, f"critical sections overlapped: {spans}"


def test_holder_death_frees_the_slot(tmp_path: Path) -> None:
    directory = tmp_path / "heavy-jobs"
    log = tmp_path / "holder.log"
    holder = _spawn_holder(directory, log, 60, "victim")
    try:
        _wait_for(tmp_path / "holder.log.held")
        assert heavy_lease.try_acquire(directory, 1) is None
        os.killpg(holder.pid, signal.SIGKILL)  # exact pid we spawned, its own group
        holder.wait(timeout=5)
        lease = asyncio.run(heavy_lease.acquire(directory, 1, wait_seconds=5, reason="t"))
        assert isinstance(lease, heavy_lease.Lease)
        lease.release()
    finally:
        _reap(holder)
    # The dead holder's record is still on disk but must not be reported.
    assert heavy_lease.read_holders(directory, 1) == []


def test_timeout_refusal_names_the_holder(tmp_path: Path) -> None:
    directory = tmp_path / "heavy-jobs"
    held = heavy_lease.try_acquire(directory, 1, session_id="sess-holder", command="pnpm build")
    assert held is not None
    lines: list[str] = []
    try:
        result = asyncio.run(
            heavy_lease.acquire(
                directory, 1, wait_seconds=0.3, reason="vitest", on_wait=lines.append
            )
        )
    finally:
        held.release()
    assert isinstance(result, heavy_lease.Refusal)
    assert "sess-holder (pnpm build)" in result.message
    assert "bash.heavy_jobs.enabled" in result.message
    assert lines and lines[0].startswith("queued behind sess-holder (pnpm build), waited")


def test_status_line_renders_holders_and_ignores_stale_records(tmp_path: Path) -> None:
    directory = tmp_path / "heavy-jobs"
    directory.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=5)
    (directory / "slot-0.json").write_text(
        f'{{"slot": 0, "pid": {dead.pid}, "session_id": "gone", "command": "x",'
        f' "started_at": {time.time()}}}'
    )
    assert heavy_lease.status_line(directory, 1) == "heavy jobs: 0/1 slots in use"
    lease = heavy_lease.try_acquire(directory, 1, session_id="abc123", command="tsc -b")
    assert lease is not None
    try:
        line = heavy_lease.status_line(directory, 1, now=time.time() + 12 * 60 + 5)
    finally:
        lease.release()
    assert line == f"heavy jobs: 1/1 slots held: abc123/{os.getpid()} tsc -b (12m)"
    assert not (directory / "slot-0.json").exists()  # release removes our record


def test_services_status_shows_the_queue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from local_operator import services

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, "1")
    assert services._heavy_job_lines() == ["heavy jobs: 0/1 slots in use"]


# -- the bash tool wiring ----------------------------------------------------------


@pytest.fixture
def lease_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, "1")
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_WAIT_SECONDS_ENV, "0.5")
    return heavy_lease.lease_dir()


def _text(result) -> str:  # noqa: ANN001 - ToolResult
    return "".join(getattr(block, "text", "") for block in result.content)


async def _until_free(directory: Path) -> None:
    deadline = time.monotonic() + 5
    while True:
        lease = heavy_lease.try_acquire(directory, 1)
        if lease is not None:
            lease.release()
            return
        assert time.monotonic() < deadline, "slot never released"
        await asyncio.sleep(0.02)


def test_non_heavy_command_never_waits(lease_env: Path) -> None:
    from local_operator.tools import builtin

    held = heavy_lease.try_acquire(lease_env, 1, session_id="other", command="pnpm build")
    assert held is not None
    try:
        started = time.monotonic()
        result = asyncio.run(builtin.execute_bash("t", {"command": "printf quick"}))
        elapsed = time.monotonic() - started
    finally:
        held.release()
    assert "quick" in _text(result)
    assert "HEAVY JOB" not in _text(result)
    # Structural, not a timing claim: the configured wait is 0.5 s and a queued
    # command would have been refused, not run.
    assert not result.is_error, elapsed


def test_heavy_command_is_refused_when_the_slot_stays_held(lease_env: Path) -> None:
    from local_operator.tools import builtin

    held = heavy_lease.try_acquire(lease_env, 1, session_id="sess-xyz", command="pnpm build")
    assert held is not None
    try:
        result = asyncio.run(
            builtin.execute_bash("t", {"command": "printf ran", "memory_mb": 2048})
        )
    finally:
        held.release()
    text = _text(result)
    assert result.is_error
    assert "HEAVY JOB QUEUE FULL" in text and "sess-xyz (pnpm build)" in text
    assert "ran" not in text.split("Held by")[0]


def test_heavy_command_queues_then_runs_and_releases(lease_env: Path, tmp_path: Path) -> None:
    from local_operator.tools import builtin

    os.environ[heavy_lease.HEAVY_JOBS_WAIT_SECONDS_ENV] = "10"
    log = tmp_path / "h.log"
    # The queued command prints its own start time, so "it ran only after the
    # holder released" is checked against the holder's recorded end.
    started_cmd = f"{sys.executable} -c 'import time; print(\"started\", time.time())'"
    holder = _spawn_holder(lease_env, log, 1.2, "first-session")
    updates: list = []
    try:
        _wait_for(tmp_path / "h.log.held")

        async def run() -> object:
            result = await builtin.execute_bash(
                "t", {"command": started_cmd, "memory_mb": 2048}, None, updates.append
            )
            await _until_free(lease_env)
            return result

        result = asyncio.run(run())
        assert holder.wait(timeout=10) == 0
    finally:
        _reap(holder)
    text = _text(result)
    assert not result.is_error and "started" in text, text
    queued = [u.details.get("heavy_job_queue") for u in updates if u.details]
    assert any(q and q.startswith("queued behind first-session (pnpm build)") for q in queued)
    command_start = float(text.split("started", 1)[1].split()[0])
    holder_end = float(log.read_text().split()[1])
    assert command_start >= holder_end


def test_kill_switch_disables_the_lease(lease_env: Path) -> None:
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.paths import config_dir
    from local_operator.tools import builtin

    settings_io.write_setting(
        ConfigManager(config_dir()), settings_io.BY_KEY["bash.heavy_jobs.enabled"], False
    )
    held = heavy_lease.try_acquire(lease_env, 1, session_id="other", command="pnpm build")
    assert held is not None
    try:
        result = asyncio.run(
            builtin.execute_bash("t", {"command": "printf ran", "memory_mb": 2048})
        )
    finally:
        held.release()
    assert not result.is_error and "ran" in _text(result)
