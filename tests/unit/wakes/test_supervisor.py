"""The wake supervisor: what it fires, what it refuses to fire, when it retires.

The property under test throughout is that the supervisor STARTS runtimes and
never delivers wakes. A session fires its own overdue wakes on load
(``WakeScheduler.load`` re-arms them to ``now + LOAD_GRACE_MS``), so the
supervisor's whole contribution is making a runtime exist — and any attempt to
also deliver would double-fire every wake it touched.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_operator.wakes.store import write_entry
from local_operator.wakes.supervisor import fire_due_wakes, serve

NOW_MS = int(time.time() * 1000)


def _schedule(due_ms: int, wake_id: str = "w1") -> dict[str, object]:
    return {"id": wake_id, "message": "check the deploy", "next_due_at": due_ms, "created_at": 1}


@pytest.fixture
def engagements(monkeypatch):  # noqa: ANN201
    """Record every engage the supervisor makes, without starting a process."""
    calls: list[dict[str, object]] = []

    async def fake_engage(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        calls.append({"session_id": session_id, "cwd": cwd, "work": work})
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
    return calls


@pytest.fixture
def no_live_runtimes(monkeypatch):  # noqa: ANN201
    async def _none(config_dir, session_id):  # noqa: ANN001
        return False

    monkeypatch.setattr("local_operator.wakes.supervisor._has_live_runtime", _none)


@pytest.mark.asyncio
async def test_a_due_wake_on_a_cold_session_starts_a_runtime(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    write_entry(tmp_path, "sessioncold1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])

    fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert fired == 1
    assert [call["session_id"] for call in engagements] == ["sessioncold1"]
    assert engagements[0]["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_the_errand_delivers_nothing(tmp_path: Path, engagements, no_live_runtimes) -> None:
    """The correction that supersedes the spec's ``wake_fire`` op.

    A ``WakeErrand`` carries no text and no message: the session's own
    scheduler delivers the occurrence when it loads. An errand that also
    delivered would append every wake twice.
    """
    from local_operator.session.runtime.launch import WakeErrand

    write_entry(tmp_path, "sessioncold1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 1_000)])
    await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    work = engagements[0]["work"]
    assert isinstance(work, WakeErrand)
    assert not hasattr(work, "text"), "a wake errand must carry no message to deliver"


@pytest.mark.asyncio
async def test_a_live_session_fires_its_own_wakes(tmp_path: Path, engagements, monkeypatch) -> None:
    """The no-live-record rule.

    A session with a live record is already running and its scheduler owns its
    wakes. Engaging there would be a second opinion about a schedule the live
    session is actively advancing.
    """

    async def _always_live(config_dir, session_id):  # noqa: ANN001
        return True

    monkeypatch.setattr("local_operator.wakes.supervisor._has_live_runtime", _always_live)
    write_entry(tmp_path, "sessionlive1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])

    fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert fired == 0
    assert engagements == []


@pytest.mark.asyncio
async def test_a_future_wake_is_not_fired_early(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    write_entry(
        tmp_path, "sessionlater", cwd=str(tmp_path), schedules=[_schedule(NOW_MS + 600_000)]
    )

    assert await fire_due_wakes(tmp_path, now_ms=NOW_MS) == 0
    assert engagements == []


@pytest.mark.asyncio
async def test_a_dormant_session_is_left_alone(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    """A /stop stamps ``stopped_at``; its wakes stay armed but must not fire.

    Firing here would resurrect a session the kill switch deliberately ended —
    PR 3's contract is that reopening the session is what re-arms it.
    """
    write_entry(
        tmp_path,
        "sessionstopd",
        cwd=str(tmp_path),
        schedules=[_schedule(NOW_MS - 5_000)],
        preserve={"stopped_at": NOW_MS - 10_000},
    )

    assert await fire_due_wakes(tmp_path, now_ms=NOW_MS) == 0
    assert engagements == []


@pytest.mark.asyncio
async def test_a_long_overdue_wake_is_left_to_the_session_catchup(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    """Nothing is gained by racing to start a runtime for a week-old wake.

    The session's resume catch-up handles arbitrarily-old overdue schedules
    when the user next opens it, and that is the surface where a stale reminder
    belongs.
    """
    ancient = NOW_MS - int((8 * 24 * 3600) * 1000)
    write_entry(tmp_path, "sessionstale", cwd=str(tmp_path), schedules=[_schedule(ancient)])

    assert await fire_due_wakes(tmp_path, now_ms=NOW_MS) == 0


@pytest.mark.asyncio
async def test_one_session_failing_does_not_stop_the_others(
    tmp_path: Path, no_live_runtimes, monkeypatch
) -> None:
    """A sweep is not all-or-nothing: the schedule is untouched, so it retries."""
    seen: list[str] = []

    async def flaky(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        seen.append(session_id)
        if session_id == "sessionbadaa":
            raise TimeoutError("no runtime")
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", flaky)
    write_entry(tmp_path, "sessionbadaa", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 9_000)])
    write_entry(tmp_path, "sessiongood1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 8_000)])

    fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert set(seen) == {"sessionbadaa", "sessiongood1"}
    assert fired == 1, "the healthy session's wake still fired"


@pytest.mark.asyncio
async def test_the_supervisor_retires_when_the_index_empties(tmp_path: Path) -> None:
    """Exit 0, so ``KeepAlive: {SuccessfulExit: False}`` leaves it down.

    This is the whole reason the supervisor is not an always-on cost: a machine
    with no wakes left runs no supervisor at all.
    """
    assert await serve(tmp_path) == 0


@pytest.mark.asyncio
async def test_a_malformed_entry_costs_one_session_not_the_sweep(
    tmp_path: Path, engagements, no_live_runtimes
) -> None:
    """The index is written by other processes; one bad file must not be fatal."""
    write_entry(tmp_path, "sessiongood2", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 3_000)])
    from local_operator.wakes.store import entry_path

    bad = entry_path(tmp_path, "sessionbroken")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json", encoding="utf-8")

    fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert fired == 1
    assert [call["session_id"] for call in engagements] == ["sessiongood2"]


# --- The login-shell PATH bootstrap ------------------------------------------
#
# These two run the entry point in a CHILD INTERPRETER, not in-process, and
# that is the whole point. The property under test is a mutation of the
# process's own `os.environ`, so an in-process test would have to monkeypatch
# `os.environ` and then assert its own mutation — passing whether or not the
# entry point does anything. Only a real process started with a bare PATH can
# observe the bootstrap actually running.
#
# `SHELL` pointed at a two-line script is the seam that makes them hermetic:
# the bootstrap shells out to `$SHELL -l -c 'echo "$PATH"'`, so a fake shell
# echoing a known marker directory gives an identical result on CI and on a
# laptop, with no dependence on the developer's rc files or on Homebrew being
# installed.

_PATH_DRIVERS = {
    "supervisor": (
        "import os\n"
        "from local_operator.wakes.supervisor import main\n"
        "main(['--once'])\n"
        "print('OBSERVED_PATH=' + os.environ['PATH'])\n"
    ),
    "cli": (
        "import os, sys\n"
        "sys.argv = ['lop', 'wake', 'serve', '--once']\n"
        "from local_operator.cli import main\n"
        "main()\n"
        "print('OBSERVED_PATH=' + os.environ['PATH'])\n"
    ),
}


def _observe_entry_point_path(driver: str, tmp_path: Path) -> str:
    """Run one wake entry point under a bare PATH; return the PATH it ended with.

    The config dir handed to the child is EMPTY, so ``serve()`` reads an empty
    index and retires 0 on its first pass: no runtime is spawned, no launchd
    domain is addressed, and no real session is touched. ``HOME`` is redirected
    alongside ``LOCAL_OPERATOR_CONFIG_DIR`` because the config variable alone
    does not redirect the cache root.
    """
    marker = tmp_path / "marker-bin"
    marker.mkdir()
    shell = tmp_path / "fake-login-shell"
    shell.write_text(f'#!/bin/sh\necho "{marker}:/usr/bin:/bin"\n', encoding="utf-8")
    shell.chmod(0o755)
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", _PATH_DRIVERS[driver]],
        env={
            # launchd's bare default, reproduced exactly.
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(tmp_path),
            "SHELL": str(shell),
            "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
            "PYTHONPATH": str(repo_root),
        },
        capture_output=True,
        text=True,
        # A HANG BACKSTOP ONLY — never an assertion. Nothing on the success
        # path compares elapsed time (AGENTS.md, "Timing, flakes, and how to
        # assert that something is fast"); this exists so a wedged child fails
        # the run instead of holding a CI slot.
        timeout=180,
    )
    assert result.returncode == 0, f"child failed:\n{result.stdout}\n{result.stderr}"
    observed = [
        line.removeprefix("OBSERVED_PATH=")
        for line in result.stdout.splitlines()
        if line.startswith("OBSERVED_PATH=")
    ]
    assert observed, f"driver printed no PATH:\n{result.stdout}\n{result.stderr}"
    return observed[-1]


@pytest.mark.skipif(os.name != "posix", reason="the login-shell seam is POSIX-only")
def test_the_launchagent_entry_primes_the_login_shell_path(tmp_path: Path) -> None:
    """Detects the removal of ``setup_cross_platform_environment`` from ``main``.

    launchd gives the supervisor ``/usr/bin:/bin:/usr/sbin:/sbin`` and nothing
    else, and every runtime it engages is spawned with ``dict(os.environ)``
    (``session/runtime/launch.py``) — so without this call each wake-driven
    turn runs without kubectl, Homebrew, nvm, cargo, pyenv or Nix. Mutating
    the call away leaves the child reporting launchd's bare PATH verbatim.
    """
    observed = _observe_entry_point_path("supervisor", tmp_path)

    marker = str(tmp_path / "marker-bin")
    # Membership, never equality and never index 0: the bootstrap PREPENDS the
    # Electron ``Local Operator/bin`` directory when one exists, so the marker
    # is not reliably first and the string is not reliably ours alone.
    assert marker in observed.split(os.pathsep), observed


@pytest.mark.skipif(os.name != "posix", reason="the login-shell seam is POSIX-only")
def test_wake_serve_primes_the_login_shell_path(tmp_path: Path) -> None:
    """Detects ``"wake"`` being dropped from ``cli._SUBPROCESS_SUBCOMMANDS``.

    ``lop wake serve`` is a second, documented entry into the same ``serve()``
    for anyone running the supervisor under their own supervisor, so it spawns
    runtimes on the identical path. ``cli.main`` primes the PATH only for
    subcommands on that frozenset; removing the name leaves this child with
    whatever PATH it inherited.
    """
    observed = _observe_entry_point_path("cli", tmp_path)

    marker = str(tmp_path / "marker-bin")
    assert marker in observed.split(os.pathsep), observed
