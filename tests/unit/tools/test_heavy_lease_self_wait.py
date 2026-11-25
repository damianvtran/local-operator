"""A heavy command must not poll the heavy-job lease it is holding.

2026-11-22, 21:09-21:31Z: one bash call ``for i in ..; do lop services status |
grep "1/1 slots held"; sleep 50; done; vitest ...`` was leased as a whole, took
the only slot, and waited ~21 min for that same slot to free. Two other
sessions hit the 900 s queue limit behind it. The bash tool now refuses such a
line before taking a slot (``HeavyClass.polls_lease``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from local_operator import heavy_lease

if TYPE_CHECKING:
    from local_operator.harness.types import ToolResult

pytestmark = pytest.mark.skipif(not heavy_lease.supported(), reason="flock is POSIX-only")

INCIDENT = (
    'for i in $(seq 1 30); do lop services status | grep "1/1 slots held"; sleep 50; done;'
    " cd app && vitest run"
)

POLLING = [
    INCIDENT,
    "while lop services status | grep -q 'slots held'; do sleep 5; done; vitest run",
    "until ! lo services status | grep -q held; do sleep 5; done && pnpm build",
    "while sleep 5; do local-operator services status; done; tsc --noEmit -p .",
    "for i in 1 2 3; do lop services heavy status; sleep 9; done; pnpm test",
    "while lop services heavy status | grep -q held; do sleep 5; done; vitest run",
    "for i in 1 2; do /usr/local/bin/lop services status; sleep 9; done; vitest run",
    "bash -c 'while lop services status | grep -q held; do sleep 5; done; vitest run'",
    "while true; do\n  lop services status\n  sleep 5\ndone; vitest run",
    "cd app\nfor i in 1 2; do lop services status; sleep 9; done; pnpm build",
    "vitest run\nwhile lop services status | grep held; do sleep 5; done",
    "{ while lop services status; do sleep 1; done; }; vitest run",
    "if true; then :; else while lop services status; do sleep 1; done; fi; vitest run",
    # The heavy command on its own line was not seen as heavy at all (#4477 follow-up).
    "while lop services status | grep '1/1 slots held'; do sleep 50; done\nvitest run",
    "while lop services status | grep -q held; do sleep 5; done\ncd app\npnpm build",
]

# Heavy, but NOT a poll: one status read, no loop, or a loop with no lease read.
ALLOWED = [
    "lop services status; vitest run",
    "lop services status\nvitest run",
    "echo hi\nlop services status\npytest -n 4 tests/unit",
    "lop services status && pytest -n 4 tests/unit",
    "for f in a b; do echo $f; done; vitest run",
    "vitest run; while sleep 5; do echo waiting; done",
    "lop services status; for p in a b; do pnpm --filter $p test; done",
    "git commit -m 'docs: lop services status'; for f in a b; do echo $f; done; vitest run",
]


@pytest.mark.parametrize("command", POLLING)
def test_a_heavy_line_that_polls_the_lease_is_flagged(command: str) -> None:
    heavy = heavy_lease.is_heavy(command)
    assert heavy is not None and heavy.polls_lease


@pytest.mark.parametrize("command", ALLOWED)
def test_a_heavy_line_without_a_polling_loop_is_not_flagged(command: str) -> None:
    heavy = heavy_lease.is_heavy(command)
    assert heavy is None or not heavy.polls_lease


@pytest.fixture
def lease_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_SLOTS_ENV, "1")
    monkeypatch.setenv(heavy_lease.HEAVY_JOBS_WAIT_SECONDS_ENV, "1")
    return heavy_lease.lease_dir()


def _text(result: ToolResult) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


def _bash(command: str) -> ToolResult:
    from local_operator.tools import builtin

    async def call() -> ToolResult:
        return await builtin.execute_bash("t", {"command": command, "timeout": 5}, None, None, None)

    return asyncio.run(call())


def test_the_bash_tool_refuses_the_incident_line_and_takes_no_slot(lease_env: Path) -> None:
    result = _bash(INCIDENT)
    text = _text(result)
    assert result.is_error
    assert "HEAVY JOB NOT STARTED" in text
    assert "run the heavy command on its own" in text.lower()
    assert "lop services heavy run --" in text
    # Refused before the lease: no slot was taken, so nobody queued behind it.
    assert heavy_lease.read_holders(lease_env) == []
    assert heavy_lease.try_acquire(lease_env, 1) is not None


def test_the_bash_tool_refuses_a_newline_split_poll_and_takes_no_slot(lease_env: Path) -> None:
    result = _bash(
        "while lop services status | grep '1/1 slots held'; do sleep 50; done\nvitest run"
    )
    assert result.is_error
    assert "HEAVY JOB NOT STARTED" in _text(result)
    assert heavy_lease.read_holders(lease_env) == []
    assert heavy_lease.try_acquire(lease_env, 1) is not None


def test_one_status_read_on_its_own_line_beside_a_heavy_command_is_not_refused(
    lease_env: Path,
) -> None:
    result = _bash("lop services status\ntrue")
    assert "HEAVY JOB NOT STARTED" not in _text(result)
    heavy = heavy_lease.is_heavy("lop services status\nvitest run")
    assert heavy is not None and not heavy.polls_lease
