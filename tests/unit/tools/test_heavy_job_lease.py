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
    # R1: pnpm/yarn run a local binary when no script has that name, as exec.
    ("pnpm vitest run", "vitest"),
    ("pnpm tsc --noEmit", "tsc"),
    ("pnpm esbuild src/x.ts --bundle", "esbuild"),
    ("pnpm electron-vite build", "electron-vite build"),
    ("pnpm dlx vitest", "vitest"),
    ("yarn exec jest --ci", "jest"),
    ("npm exec -- vitest run", "vitest"),
    # R1: wrappers that run the rest of the line.
    ("bash -c 'pnpm build'", "pnpm build"),
    ('sh -c "npm run build"', "npm build"),
    ("bash -lc 'cd x && pnpm build'", "pnpm build"),
    ("scripts/run_bounded.py -- pnpm build", "pnpm build"),
    ("python3 scripts/run_bounded.py --memory-mb 4096 -- pnpm build", "pnpm build"),
    ("xvfb-run -a pnpm test", "pnpm test"),
    ("taskset -c 0-3 pnpm build", "pnpm build"),
    ("chrt -f 10 pnpm build", "pnpm build"),
    ("prlimit --as=4000000000 pnpm build", "pnpm build"),
    ("corepack pnpm build", "pnpm build"),
    ("next build", "next build"),
    ("nx run web:build", "nx build"),
    ("nx run-many -t test", "nx test"),
    ("node node_modules/typescript/bin/tsc -p .", "tsc"),
    # R8 control: `-v` is verbose for pytest, not a version query.
    ("pytest -v -n 4", "pytest -n"),
]

#: Tonight's (2026-11-16) incident shapes, each pinned as its own row
#: (manager addition (c) on PR #4104). `timeout 900 pnpm test` is how the mac
#: proof runs were spelled: timeout -> node -> node-22.
INCIDENT = [
    ("pnpm vitest run", "vitest"),
    ("pnpm tsc --noEmit", "tsc"),
    ("node node_modules/.bin/esbuild src/main.ts --bundle --outfile=out/main.js", "esbuild"),
    ("timeout 900 pnpm test", "pnpm test"),
    ("bash -c 'cd x && pnpm build'", "pnpm build"),
]

#: Watchers and dev servers: they run until stopped, so they must never hold
#: the slot (R2 / QA Q2). One row per QA row 5 shape plus the reviewer's.
LONG_LIVED = [
    "tsc --watch",
    "tsc -w",
    "tsc -b -w",
    "vue-tsc --watch",
    "vitest --watch",
    "vitest -w",
    "vitest watch",
    "vitest dev",
    "jest --watch",
    "jest --watchAll",
    "node --test --watch test/",
    "esbuild --watch src/x.ts",
    "esbuild --serve src/x.ts",
    "webpack --watch",
    "webpack serve",
    "rollup -w",
    "vite build --watch",
    "pnpm test -- --watch",
    "pnpm test --watch",
    "npm test -- --watchAll",
    "pnpm test:watch",
    "pnpm build:watch",
    "pnpm watch",
    "pnpm dev",
    "pnpm serve",
    "pnpm start",
    "npm start",
    "npm run dev",
    "yarn dev",
    "next dev",
    "nx serve web",
    "nx run web:serve",
    "turbo watch build",
    "turbo run dev",
    "bash -c 'pnpm dev'",
    # A command line that also starts a watcher lives as long as the watcher.
    "pnpm build && pnpm dev",
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
    # R8: informational flags anywhere in the argv.
    "pnpm build --help",
    "vitest run --version",
    "pnpm lint",
    "pnpm add -D vitest",
    "bash -c 'ls -la'",
    "bash script.sh",
    "taskset -c 0-3 ls",
    "nx run web:lint",
]


@pytest.mark.parametrize("command,reason", HEAVY)
def test_heavy_commands_are_classified(command: str, reason: str) -> None:
    heavy = heavy_lease.is_heavy(command)
    assert heavy is not None, command
    assert heavy.reason == reason


@pytest.mark.parametrize("command", NOT_HEAVY)
def test_short_commands_are_not_heavy(command: str) -> None:
    assert heavy_lease.is_heavy(command) is None


@pytest.mark.parametrize("command,reason", INCIDENT)
def test_incident_shapes_are_heavy(command: str, reason: str) -> None:
    heavy = heavy_lease.is_heavy(command)
    assert heavy is not None and heavy.reason == reason, command


@pytest.mark.parametrize("command", LONG_LIVED)
def test_watchers_and_dev_servers_never_take_the_slot(command: str) -> None:
    assert heavy_lease.is_heavy(command) is None, command


def _ui_repo(root: Path) -> Path:
    """A package.json shaped like local-operator-ui's (electron-vite + vitest)."""
    repo = root / "local-operator-ui"
    (repo / "src").mkdir(parents=True)
    (repo / "package.json").write_text(
        '{"scripts": {"build": "electron-vite build", "dev": "electron-vite dev",'
        ' "test": "vitest run", "test:watch": "vitest", "lint": "eslint .",'
        ' "compile": "tsc -b", "loop": "pnpm loop", "start": "electron-vite preview"}}'
    )
    return repo


def test_package_scripts_resolve_through_package_json(tmp_path: Path) -> None:
    repo = _ui_repo(tmp_path)
    cwd = str(repo)
    # The incident's own shapes in the ui repo.
    assert heavy_lease.is_heavy("pnpm build", cwd=cwd) == heavy_lease.HeavyClass("pnpm build")
    assert heavy_lease.is_heavy("pnpm electron-vite build", cwd=cwd) == heavy_lease.HeavyClass(
        "electron-vite build"
    )
    # A script whose NAME says nothing, resolved by its body.
    assert heavy_lease.is_heavy("pnpm compile", cwd=cwd) == heavy_lease.HeavyClass("pnpm compile")
    assert heavy_lease.is_heavy("npm run compile", cwd=cwd) == heavy_lease.HeavyClass("npm compile")
    # From a subdirectory (nearest package.json up the tree) and after a `cd`.
    assert heavy_lease.is_heavy("pnpm compile", cwd=str(repo / "src")) is not None
    assert heavy_lease.is_heavy(f"cd {repo} && pnpm compile", cwd=str(tmp_path)) is not None
    assert heavy_lease.is_heavy("bash -c 'cd local-operator-ui && pnpm compile'", cwd=str(tmp_path))
    # Not heavy: a non-heavy body, watcher names, and a self-recursive script.
    for command in ("pnpm lint", "pnpm dev", "pnpm test:watch", "pnpm start", "pnpm loop"):
        assert heavy_lease.is_heavy(command, cwd=cwd) is None, command
    # Without a readable package.json the name heuristics still decide.
    assert heavy_lease.is_heavy("pnpm compile", cwd=str(tmp_path / "missing")) is None
    assert heavy_lease.is_heavy("pnpm build", cwd=str(tmp_path / "missing")) is not None
    (repo / "package.json").write_text("{not json")
    assert heavy_lease.is_heavy("pnpm build", cwd=cwd) is not None


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
    assert heavy_lease.configured_settings().wait_seconds == 900


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
    assert f"sess-holder/{os.getpid()} (pnpm build)" in result.message
    # Q3: the refusal names the bound that refused it, the queue limit.
    assert "the queue limit (LOCAL_OPERATOR_HEAVY_JOB_WAIT_S=0.3s)" in result.message
    assert "bash.heavy_jobs.enabled" in result.message
    # R7: the holder is named `<session>/<pid>`, as the request specified.
    assert lines and lines[0].startswith(
        f"queued behind sess-holder/{os.getpid()} (pnpm build), waited"
    )


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
    # 1 s is the smallest accepted queue limit (R4 clamps lower values back to
    # the 900 s default, which would hang these cells).
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_WAIT_SECONDS_ENV, "1")
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
    assert "HEAVY JOB QUEUE FULL" in text and f"sess-xyz/{os.getpid()} (pnpm build)" in text
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
    assert any(
        q and q.startswith(f"queued behind first-session/{holder.pid} (pnpm build)") for q in queued
    )
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


# -- round-1 remediation (PR #4104) ---------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [("inf", 900), ("-inf", 900), ("nan", 900), ("-5", 900), ("0", 900), ("0.2", 900)]
    + [("1e9", 900), ("14401", 900), ("1", 1), ("14400", 14400), ("75.5", 75.5)],
)
def test_wait_override_is_finite_and_clamped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str, expected: float
) -> None:
    """R4: a bad queue limit falls back to 900 s instead of crashing or hanging."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, "1")
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_WAIT_SECONDS_ENV, raw)
    assert heavy_lease.configured_settings().wait_seconds == expected


@pytest.mark.parametrize("raw", ["inf", "nan", "-1", "1e3", "5", "99999999999999999999"])
def test_bad_slot_override_means_auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str
) -> None:
    """R4: `inf` once raised OverflowError out of every heavy bash call."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(heavy_lease, "auto_slot_count", lambda total, avail: 1)
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, raw)
    assert heavy_lease.configured_settings().slots == 1
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, "4")
    assert heavy_lease.configured_settings().slots == 4


def test_status_counts_holders_above_the_current_slot_count(tmp_path: Path) -> None:
    """R5: auto sizing can shrink while builds run; no live holder may vanish."""
    directory = tmp_path / "heavy-jobs"
    first = heavy_lease.try_acquire(directory, 2, session_id="s0", command="pnpm build")
    second = heavy_lease.try_acquire(directory, 2, session_id="s1", command="tsc -b")
    assert first is not None and second is not None and second.slot == 1
    try:
        shrunk = heavy_lease.status_line(directory, 1)
        same = heavy_lease.status_line(directory, 2)
    finally:
        first.release()
        second.release()
    assert shrunk.startswith("heavy jobs: 2 held (1 slot now): s0/")
    assert "s1/" in shrunk and "tsc -b" in shrunk
    assert same.startswith("heavy jobs: 2/2 slots held: ")


def test_release_frees_the_slot_even_while_a_child_holds_a_duplicate(tmp_path: Path) -> None:
    """Q1's other half: the harness's normal release must not wait on the child.

    A daemon a build leaves behind (nx/turbo) inherits the descriptor; the
    explicit LOCK_UN in release() frees the slot for every duplicate.
    """
    directory = tmp_path / "heavy-jobs"
    lease = heavy_lease.try_acquire(directory, 1, command="pnpm build")
    assert lease is not None and lease.fileno() is not None
    child = subprocess.Popen(["sleep", "30"], pass_fds=(lease.fileno(),), start_new_session=True)
    try:
        assert heavy_lease.try_acquire(directory, 1) is None
        lease.release()
        after = heavy_lease.try_acquire(directory, 1)
        assert after is not None, "release left the slot locked by the child's copy"
        after.release()
    finally:
        _reap(child)


_BASH_HARNESS = textwrap.dedent("""
    import asyncio, sys
    from local_operator.tools import builtin
    asyncio.run(builtin.execute_bash("t", {"command": sys.argv[1], "memory_mb": 2048,
                                           "timeout": 60}))
""")


def test_killed_harness_keeps_the_slot_until_its_command_exits(
    lease_env: Path, tmp_path: Path
) -> None:
    """Q1: SIGKILL the harness mid-build; a waiter must not start beside the orphan.

    The real ``execute_bash`` runs in a child "harness" process. Its command
    (its own session, as in production) marks start, sleeps, marks end. After
    the harness is SIGKILLed by exact pid, the slot must stay held until the
    command exits, then free on its own.
    """
    started, done = tmp_path / "started", tmp_path / "done"
    command = f"touch {started}; sleep 2; touch {done}"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    harness = subprocess.Popen(
        [sys.executable, "-c", _BASH_HARNESS, command], env=env, start_new_session=True
    )
    pgid = None
    try:
        _wait_for(started, timeout=30)
        record = (lease_env / "slot-0.json").read_text()
        pgid = __import__("json").loads(record)["pgid"]
        os.kill(harness.pid, signal.SIGKILL)
        harness.wait(timeout=5)
        assert not done.exists(), "command finished before the check; widen the sleep"
        assert heavy_lease.try_acquire(lease_env, 1) is None, "slot freed beside the orphan"
        # The dead harness's record still names the live build to a waiter.
        holders = heavy_lease.read_holders(lease_env)
        assert [h.pgid for h in holders] == [pgid]
        lease = asyncio.run(heavy_lease.acquire(lease_env, 1, wait_seconds=10, reason="t"))
        assert isinstance(lease, heavy_lease.Lease)
        acquired_at = time.time()
        lease.release()
        assert done.exists() and done.stat().st_mtime <= acquired_at
    finally:
        _reap(harness)
        if pgid:
            with __import__("contextlib").suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)


def test_queue_wait_is_not_capped_by_the_command_timeout(lease_env: Path, tmp_path: Path) -> None:
    """K2: the queue limit is separate from `timeout`, which starts at spawn."""
    from local_operator.tools import builtin

    os.environ[heavy_lease.HEAVY_JOBS_WAIT_SECONDS_ENV] = "10"
    log = tmp_path / "k2.log"
    holder = _spawn_holder(lease_env, log, 2.5, "long-build")
    try:
        _wait_for(tmp_path / "k2.log.held")
        started = time.monotonic()
        result = asyncio.run(
            builtin.execute_bash(
                "t", {"command": "printf ran-after-queue", "memory_mb": 2048, "timeout": 1}
            )
        )
        elapsed = time.monotonic() - started
        assert holder.wait(timeout=10) == 0
    finally:
        _reap(holder)
    assert not result.is_error and "ran-after-queue" in _text(result), _text(result)
    assert elapsed > 1.0  # it queued past its own 1 s timeout and still ran


def test_heavy_command_cannot_drop_its_memory_cap_per_call(lease_env: Path) -> None:
    """R3: `memory_mb=0` on a heavy command is treated as unset."""
    from local_operator.tools import builtin

    probe = "echo mem=$LOCAL_OPERATOR_RESOLVED_MEMORY_MB"
    heavy = asyncio.run(
        builtin.execute_bash("t", {"command": f"{probe}; false && pnpm build", "memory_mb": 0})
    )
    light = asyncio.run(builtin.execute_bash("t", {"command": probe, "memory_mb": 0}))
    assert "mem=0" in _text(light)  # control: a normal command may still opt out
    assert "mem=" in _text(heavy) and "mem=0" not in _text(heavy), _text(heavy)


def test_operator_memory_switch_still_wins_and_is_announced(lease_env: Path) -> None:
    """R3 decision: bash.memory.enabled=false is the operator's master switch."""
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.paths import config_dir
    from local_operator.tools import builtin

    settings_io.write_setting(
        ConfigManager(config_dir()), settings_io.BY_KEY["bash.memory.enabled"], False
    )
    updates: list = []
    result = asyncio.run(
        builtin.execute_bash(
            "t",
            {"command": "echo mem=$LOCAL_OPERATOR_RESOLVED_MEMORY_MB; false && pnpm build"},
            None,
            updates.append,
        )
    )
    assert "mem=0" in _text(result)
    lines = [u.details.get("heavy_job_queue") for u in updates if u.details]
    assert any(
        line and line.startswith("heavy-job slot acquired (pnpm build); memory cap off")
        for line in lines
    ), lines


def test_spawn_passes_the_lease_descriptor(lease_env: Path) -> None:
    """Q1 wiring: the command sees the lock description as an open descriptor."""
    from local_operator.tools import builtin

    lock = lease_env / "slot-0.lock"
    # A probe that finds an inherited fd pointing at the slot lock.
    probe = (
        f"{sys.executable} -c \"import os,sys; p=os.path.realpath('{lock}');"
        " fds=[f for f in range(3,256) if os.path.exists(f'/dev/fd/{f}')"
        " and os.path.realpath(f'/proc/self/fd/{f}' if os.path.exists('/proc/self/fd')"
        " else f'/dev/fd/{f}')==p]; print('inherited', len(fds))\"; false && pnpm build"
    )
    if not Path("/proc/self/fd").exists():
        pytest.skip("fd-to-path probe needs /proc")
    result = asyncio.run(builtin.execute_bash("t", {"command": probe}))
    assert "inherited 1" in _text(result), _text(result)
