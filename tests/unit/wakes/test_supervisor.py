"""The wake supervisor: what it fires, what it refuses to fire, when it retires.

The property under test throughout is that the supervisor STARTS runtimes and
never delivers wakes. A session fires its own overdue wakes on load
(``WakeScheduler.load`` re-arms them to ``now + LOAD_GRACE_MS``), so the
supervisor's whole contribution is making a runtime exist — and any attempt to
also deliver would double-fire every wake it touched.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
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


def _make_session(config_dir: Path, session_id: str) -> None:
    """Give ``session_id`` a transcript, so the ghost guard sees a real session.

    The supervisor now refuses to engage an index entry whose session has no
    transcript on disk (such an entry can never be engaged successfully and
    used to burn a full deadline on every pass, forever). These tests are
    about firing DECISIONS, so they need their sessions to exist.
    """
    from local_operator.resume import TRANSCRIPT_NAME

    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TRANSCRIPT_NAME).write_text("", encoding="utf-8")


@pytest.fixture(autouse=True)
def _sessions_exist(monkeypatch):  # noqa: ANN201
    """Every ``write_entry`` in this file also lays down a transcript.

    Wrapping the writer rather than annotating twenty call sites: the ghost
    guard is asserted directly in its own test, and everywhere else a session
    having a transcript is background, not the property under test.
    """
    import local_operator.wakes.store as store_mod

    real = store_mod.write_entry

    def writing(config_dir, session_id, **kwargs):  # noqa: ANN001, ANN202
        _make_session(Path(config_dir), session_id)
        return real(config_dir, session_id, **kwargs)

    monkeypatch.setattr(store_mod, "write_entry", writing)
    monkeypatch.setattr("tests.unit.wakes.test_supervisor.write_entry", writing)


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


# --- Bounded lateness, concurrency, and the dead-ends that used to be silent --


@pytest.fixture
def real_sessions(monkeypatch):  # noqa: ANN201
    """Treat every session id as having a transcript.

    The ghost guard is asserted on its own below; the other tests are about
    firing decisions and would otherwise all have to build session dirs.
    """
    monkeypatch.setattr(
        "local_operator.wakes.supervisor._session_exists", lambda _config, _session: True
    )


@pytest.mark.asyncio
async def test_the_sweep_fires_due_sessions_concurrently(
    tmp_path: Path, no_live_runtimes, real_sessions, monkeypatch
) -> None:
    """The serial loop was the lateness amplifier, and this is what proves the fix.

    Before: `for ... await engage_runtime` meant a session waited behind every
    earlier session's FULL deadline — measured at ~63 s per pass for two cold
    sessions. Since a wake's deadline is now sized for a cold start (180 s),
    staying serial would have made lateness worse, not better.

    Asserted structurally (overlap actually happened), never on wall-clock
    duration: each engage blocks on an event that only a LATER engage can set,
    so a serial implementation deadlocks its way to a timeout failure while a
    concurrent one completes. No sleep, no timing assertion.
    """
    import asyncio

    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    order: list[str] = []

    async def overlapping(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        order.append(session_id)
        if session_id == "sessionfirst":
            first_entered.set()
            # Only satisfiable if the SECOND engage runs while this one is
            # still in flight.
            await second_entered.wait()
        else:
            await first_entered.wait()
            second_entered.set()
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", overlapping)

    from local_operator.wakes import supervisor as mod

    started_order: list[str] = []
    real_engage = mod._Sweeper.engage

    def recording_engage(self, config_dir, session_id, cwd, due_ms, moment):  # noqa: ANN001, ANN202
        started_order.append(session_id)
        return real_engage(self, config_dir, session_id, cwd, due_ms, moment)

    monkeypatch.setattr(mod._Sweeper, "engage", recording_engage)
    write_entry(tmp_path, "sessionfirst", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 9_000)])
    write_entry(tmp_path, "sessionsecnd", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 8_000)])

    fired = await asyncio.wait_for(fire_due_wakes(tmp_path, now_ms=NOW_MS), timeout=10)

    assert fired == 2
    # Oldest-first is preserved in the order engagements are STARTED, which is
    # the property the sweep controls. Which one wins the race into
    # `engage_runtime` afterwards is the event loop's business, so asserting on
    # arrival order there would be asserting on the scheduler.
    assert started_order[0] == "sessionfirst"
    assert set(order) == {"sessionfirst", "sessionsecnd"}


@pytest.mark.asyncio
async def test_the_sweep_bounds_its_concurrency(
    tmp_path: Path, no_live_runtimes, real_sessions, monkeypatch
) -> None:
    """Bounded, because each engage may cold-start a full harness process.

    An unbounded gather over a backlog would be a thundering herd against the
    resource contention that makes cold starts slow in the first place.
    """
    import asyncio

    from local_operator.wakes.supervisor import _MAX_CONCURRENT_ENGAGES

    live = 0
    peak = 0

    async def counting(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)  # a real await point, so overlap is possible
        live -= 1
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", counting)
    for n in range(6):
        write_entry(
            tmp_path, f"sessionbulk{n}", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)]
        )

    await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert peak <= _MAX_CONCURRENT_ENGAGES, f"{peak} engages ran at once"


@pytest.mark.asyncio
async def test_a_wake_errand_gets_a_cold_start_budget(
    tmp_path: Path, no_live_runtimes, real_sessions, monkeypatch
) -> None:
    """The 30 s default is sized for a USER waiting at a prompt.

    Nobody waits on a wake, and 344 of 682 engages (50.4%) timed out at 30 s
    on the machine this was diagnosed on. The shared default must be left
    alone for prompt/steer; only the wake path takes the longer budget.
    """
    from local_operator.session.runtime.launch import DEFAULT_DEADLINE_S
    from local_operator.wakes.supervisor import WAKE_DEADLINE_S

    seen: list[float] = []

    async def record_deadline(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        seen.append(deadline_s)
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", record_deadline)
    write_entry(tmp_path, "sessioncold9", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])

    await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert seen == [WAKE_DEADLINE_S]
    assert WAKE_DEADLINE_S >= 120, "a cold start on a loaded box needs more than two minutes"
    assert DEFAULT_DEADLINE_S == 30.0, "the USER-facing deadline must not have moved"


@pytest.mark.asyncio
async def test_a_ghost_index_entry_is_skipped_and_logged(
    tmp_path: Path, engagements, no_live_runtimes, caplog
) -> None:
    """An entry whose session has no transcript can never be engaged.

    The live log shows two such ids accumulating 30 s timeouts with zero
    successful starts, ever — each one consuming a sweep slot on every pass.
    """
    import shutil

    write_entry(tmp_path, "sessionghost", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])
    # THE GHOST: the index entry stays, the session it describes does not.
    # That is the real shape on disk — a reap or a hand-deleted directory
    # leaves the entry behind, and nothing prunes it.
    shutil.rmtree(tmp_path / "sessions" / "sessionghost")

    with caplog.at_level("WARNING"):
        fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert fired == 0
    assert engagements == [], "a session with no transcript must not be engaged"
    assert any("ghost" in record.getMessage() for record in caplog.records), [
        record.getMessage() for record in caplog.records
    ]


@pytest.mark.asyncio
async def test_the_stale_skip_is_logged_with_how_overdue_it_is(
    tmp_path: Path, engagements, no_live_runtimes, real_sessions, caplog
) -> None:
    """Behaviour unchanged, silence removed.

    A wake past STALE_AFTER_S is still left to the session's own catch-up —
    but skipping it without a word is how a wedged-runtime starvation becomes
    permanent and invisible.
    """
    ancient = NOW_MS - int((9 * 24 * 3600) * 1000)
    write_entry(tmp_path, "sessionstal2", cwd=str(tmp_path), schedules=[_schedule(ancient)])

    with caplog.at_level("WARNING"):
        fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert fired == 0, "the skip BEHAVIOUR is deliberately unchanged"
    message = " ".join(record.getMessage() for record in caplog.records)
    assert "sessionstal2" in message
    assert "9.0 days overdue" in message, message


@pytest.mark.asyncio
async def test_the_live_session_skip_reports_how_overdue_it_is(
    tmp_path: Path, engagements, real_sessions, monkeypatch, caplog
) -> None:
    """A WEDGED runtime looks exactly like a healthy one here.

    Its record is live, so the supervisor skips it on every pass while the
    wake never fires. At debug level that case was invisible; a repeating
    "already running" line with a GROWING overdue figure is what makes it
    findable.
    """

    async def _always_live(config_dir, session_id):  # noqa: ANN001
        return True

    monkeypatch.setattr("local_operator.wakes.supervisor._has_live_runtime", _always_live)
    write_entry(tmp_path, "sessionwedge", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 90_000)])

    with caplog.at_level("INFO"):
        await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    message = " ".join(record.getMessage() for record in caplog.records)
    assert "sessionwedge" in message
    assert "90.0s overdue" in message, message


@pytest.mark.asyncio
async def test_a_wake_written_during_the_sleep_is_seen_within_a_slice(
    tmp_path: Path, monkeypatch
) -> None:
    """THE LATENESS BOUND. A single computed sleep could hide a new wake for an hour.

    The loop used to compute `delay` once from a snapshot and sleep it whole,
    so a wake persisted while it slept was invisible until that sleep expired
    (up to MAX_SLEEP_S = 3600 s). Here the supervisor starts out sleeping
    toward a wake three hours away and a nearer one is written behind its
    back; the sliced re-read must notice.

    Time is driven by a fake `asyncio.sleep`, so this asserts on the ORDER of
    events and never on the clock — no real waiting, no timing flake.
    """
    import asyncio

    from local_operator.wakes import supervisor as mod

    real_sleep = asyncio.sleep
    slept: list[float] = []
    fired_for: list[str] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        # The new wake lands DURING the first slice, which is exactly the race
        # the old single-sleep could not see.
        if len(slept) == 1:
            write_entry(
                tmp_path,
                "sessionurgent",
                cwd=str(tmp_path),
                schedules=[_schedule(int(time.time() * 1000) - 1_000, "w2")],
            )
        # Yield for real. The loop is cooperative, and a "sleep" that never
        # reaches the scheduler turns any await in serve() into a spin.
        # `real_sleep` is bound before the patch: `mod.asyncio` IS the asyncio
        # module, so calling asyncio.sleep here would re-enter this fake.
        await real_sleep(0)

    passes = 0

    def fake_sweep(self, config_dir, index, moment):  # noqa: ANN001, ANN202
        # Hooked at the SWEEP, the seam serve() actually drives now that
        # engagements no longer block the loop. The first pass finds nothing
        # due and returns, which is what lets serve() reach its sleep — the
        # window the defect lived in. The second records what the re-read saw
        # and ends the loop.
        nonlocal passes
        passes += 1
        if passes == 1:
            return 0
        fired_for.extend(sorted(index))
        raise _StopServing

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(mod._Sweeper, "sweep", fake_sweep)
    # The far-away wake the sleep is sized for: three hours out, so the old
    # code would have slept MAX_SLEEP_S before looking again.
    write_entry(
        tmp_path,
        "sessiondistnt",
        cwd=str(tmp_path),
        schedules=[_schedule(int(time.time() * 1000) + 3 * 3600_000)],
    )

    with pytest.raises(_StopServing):
        await serve(tmp_path)

    # First pass fires (nothing due), then sleeps in SLICES rather than one
    # 3600 s block, and the second sweep sees the urgent wake.
    assert slept, "the supervisor did not sleep at all"
    assert max(slept) <= mod.SLICE_S, (
        f"slept {max(slept)}s in one uninterrupted block; a wake written during that "
        "window is invisible for its whole duration (the defect this closes)"
    )
    assert "sessionurgent" in fired_for


class _StopServing(Exception):
    """Breaks out of ``serve``'s infinite loop from inside a patched callee."""


@pytest.mark.parametrize(
    ("name", "build", "reason"),
    [
        ("dormant", "dormant", "every session is stopped"),
        ("stale", "stale", "every wake is past the staleness bound"),
        ("ghost", "ghost", "no session on disk owns the entry"),
    ],
)
@pytest.mark.asyncio
async def test_an_index_with_nothing_fireable_retires(
    tmp_path: Path, name: str, build: str, reason: str, monkeypatch
) -> None:
    """ "Nothing fireable" is not the same question as "non-empty".

    Each of these three entry kinds satisfies "a non-dormant entry carrying
    schedules" (dormant excepted) while being work this process can NEVER do:
    a stopped session re-arms only on reopen, a >7-day wake only gets older,
    and a ghost never grows a transcript. Counting them as fireable made the
    supervisor immortal and re-logged the refusal every slice forever
    (round 1, R2/Q4).

    Asserted on the RETIREMENT ITSELF, not on `serve(once=True) == 0`. Round 1
    (R4): `once=True` returns 0 from the retirement branch and from the
    post-sweep branch alike, so the old assertion held whatever the predicate
    did — it could not fail, and a verbatim restoration of the pre-fix
    behaviour kept the suite green. This drives the real loop and requires the
    retirement log line, which only the retirement branch emits.
    """
    import asyncio

    from local_operator.wakes import supervisor as mod

    if build == "dormant":
        write_entry(
            tmp_path,
            "sessiondorm1",
            cwd=str(tmp_path),
            schedules=[_schedule(NOW_MS - 5_000)],
            preserve={"stopped_at": NOW_MS - 10_000},
        )
    elif build == "stale":
        write_entry(
            tmp_path,
            "sessionstal3",
            cwd=str(tmp_path),
            schedules=[_schedule(NOW_MS - int(9 * 24 * 3600 * 1000))],
        )
    else:
        import shutil

        write_entry(
            tmp_path, "sessionghos2", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)]
        )
        shutil.rmtree(tmp_path / "sessions" / "sessionghos2")

    real_sleep = asyncio.sleep

    async def fast_sleep(seconds: float) -> None:
        # The retirement grace is a real await; collapse it so the test does
        # not spend SLICE_S of wall time proving a decision.
        await real_sleep(0)

    monkeypatch.setattr(mod.asyncio, "sleep", fast_sleep)

    records: list[str] = []
    monkeypatch.setattr(
        mod.logger,
        "info",
        lambda msg, *args, **kw: records.append(str(msg) % args if args else msg),
    )

    assert await asyncio.wait_for(serve(tmp_path), timeout=10) == 0
    assert any("retiring" in line for line in records), (
        f"the supervisor did not retire on an index where {reason}; it stayed up with "
        f"nothing it could ever fire. Log was: {records}"
    )


@pytest.mark.asyncio
async def test_retirement_re_reads_after_a_grace_before_exiting(
    tmp_path: Path, monkeypatch
) -> None:
    """The RACE half of the never-restarted-supervisor defect.

    `_persist_wake_schedules` writes the index entry and THEN calls the
    install hook, so the supervisor can read an empty index while a session is
    part-way through arming its first wake. Retiring inside that window exits
    0, the hook that follows sees a job launchd still knows about, and the
    wake is left armed with nothing running.
    """
    import asyncio

    from local_operator.wakes import supervisor as mod

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        # The entry lands during the retirement grace.
        write_entry(
            tmp_path, "sessionraced", cwd=str(tmp_path), schedules=[_schedule(NOW_MS + 600_000)]
        )
        await real_sleep(0)

    def fake_sweep(self, config_dir, index, moment):  # noqa: ANN001, ANN202
        raise _StopServing

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(mod._Sweeper, "sweep", fake_sweep)

    # Reaching the sweep at all means it did NOT retire on the empty first read.
    with pytest.raises(_StopServing):
        await serve(tmp_path)

    assert asyncio.get_event_loop_policy() is not None  # loop intact; no hidden teardown


# --- Round 1 remediation: non-blocking sweep, throttled skips, wedged runtimes


@pytest.mark.asyncio
async def test_a_slow_engagement_does_not_hold_the_serve_loop(
    tmp_path: Path, no_live_runtimes, real_sessions, monkeypatch
) -> None:
    """R3. The loop must keep slicing while an engagement runs.

    An awaited sweep put head-of-line blocking back where the slice re-read
    had removed it: six all-timing-out sessions at concurrency 2 and a 180 s
    deadline block for 540 s, and NOTHING re-reads the index for the whole of
    it. Here one engagement never finishes; the loop must still observe a wake
    written afterwards.

    Asserted structurally — the second index read happens while the first
    engagement is provably still in flight — so there is no timing assertion.
    """
    import asyncio

    from local_operator.wakes import supervisor as mod

    engaged = asyncio.Event()
    release = asyncio.Event()
    seen_later: list[str] = []

    async def never_finishes(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        engaged.set()
        await release.wait()  # the 180 s deadline, modelled as "not yet"
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", never_finishes)

    real_sleep = asyncio.sleep

    async def fast_sleep(seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(mod.asyncio, "sleep", fast_sleep)

    passes = 0
    real_sweep = mod._Sweeper.sweep

    def watching_sweep(self, config_dir, index, moment):  # noqa: ANN001, ANN202
        # Stop on the CONDITION, not on a pass count: serve() spins freely
        # here because sleeps are collapsed, so "pass 2" would race the test's
        # write rather than prove anything about it.
        nonlocal passes
        passes += 1
        launched = real_sweep(self, config_dir, index, moment)
        if "sessionlater" in index:
            seen_later.extend(sorted(index))
            raise _StopServing
        if passes > 500:  # a hang backstop, never an assertion
            raise _StopServing
        return launched

    monkeypatch.setattr(mod._Sweeper, "sweep", watching_sweep)

    write_entry(tmp_path, "sessionslow1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])

    async def drive() -> None:
        with pytest.raises(_StopServing):
            await serve(tmp_path)

    task = asyncio.create_task(drive())
    await asyncio.wait_for(engaged.wait(), timeout=10)
    # Written while the engagement is stuck: the old awaited sweep could not
    # see this until the engagement finished.
    write_entry(tmp_path, "sessionlater", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 1_000)])
    await asyncio.wait_for(task, timeout=10)

    # Asserted BEFORE releasing: the engagement is provably still blocked at
    # this point, so a later pass having seen the new entry can only mean the
    # loop kept running alongside it.
    assert not release.is_set(), "the engagement finished; this proved nothing about blocking"
    assert "sessionlater" in seen_later, (
        "the serve loop did not re-read the index while an engagement was in flight; "
        "a wake armed during a slow sweep waits out the whole sweep"
    )
    release.set()


@pytest.mark.asyncio
async def test_the_same_occurrence_is_never_engaged_twice(
    tmp_path: Path, no_live_runtimes, real_sessions, monkeypatch
) -> None:
    """The in-flight set is what makes background engagements safe.

    A pass every SLICE_S over a wake that takes WAKE_DEADLINE_S to engage
    would otherwise start ~18 engagements for one occurrence.
    """
    import asyncio

    from local_operator.wakes import supervisor as mod

    attempts: list[str] = []
    release = asyncio.Event()

    async def slow(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        attempts.append(session_id)
        await release.wait()
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", slow)
    write_entry(tmp_path, "sessiondupe1", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])

    sweeper = mod._Sweeper()
    from local_operator.wakes.store import read_index

    index = read_index(tmp_path)
    # Three passes over the same due occurrence, as the slice loop would do.
    launched = [sweeper.sweep(tmp_path, index, NOW_MS) for _ in range(3)]
    await asyncio.sleep(0)

    assert launched == [1, 0, 0], f"the same occurrence was engaged more than once: {launched}"
    release.set()
    await sweeper.drain()
    assert attempts == ["sessiondupe1"]


@pytest.mark.asyncio
async def test_a_permanent_skip_stops_flooding_the_log(
    tmp_path: Path, engagements, no_live_runtimes, caplog
) -> None:
    """R2/Q4. A ghost or stale skip repeats every slice and never self-clears.

    Unthrottled that is ~8,640 lines/day/entry into a file nothing rotates,
    drowning the signal this observability exists to create. The first
    occurrences carry the information; after that a heartbeat is enough.
    """
    from local_operator.wakes import supervisor as mod

    mod._skip_log = mod._SkipLog()  # a fresh process's worth of state
    write_entry(tmp_path, "sessionflood", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])
    import shutil

    shutil.rmtree(tmp_path / "sessions" / "sessionflood")

    with caplog.at_level("WARNING"):
        for _ in range(20):
            await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    lines = [r for r in caplog.records if "sessionflood" in r.getMessage()]
    assert len(lines) == mod._SKIP_LOG_BURST, (
        f"{len(lines)} lines for 20 passes of a permanently-unfireable entry; at the "
        f"shipped SLICE_S that is {len(lines) / 20 * 8640:.0f} lines/day"
    )


def test_a_changed_skip_reason_speaks_up_again() -> None:
    """The throttle is keyed by reason, so a state CHANGE is not swallowed.

    An entry going from "a live runtime owns it" to "stale" is new
    information, and inheriting the old key's silence would hide it.
    """
    from local_operator.wakes.supervisor import _SkipLog

    log = _SkipLog()
    for _ in range(10):
        log.should_log("sessionx", "live")

    assert log.should_log("sessionx", "stale") is True


def test_the_skip_throttle_resumes_after_the_heartbeat_interval() -> None:
    """Throttled is not silenced: the condition stays visible, hourly."""
    from local_operator.wakes.supervisor import (
        _SKIP_HEARTBEAT_S,
        _SKIP_LOG_BURST,
        _SkipLog,
    )

    log = _SkipLog()
    for _ in range(_SKIP_LOG_BURST):
        assert log.should_log("sessiony", "ghost", now=0.0) is True
    assert log.should_log("sessiony", "ghost", now=1.0) is False
    assert log.should_log("sessiony", "ghost", now=_SKIP_HEARTBEAT_S + 1) is True


@pytest.mark.asyncio
async def test_a_wedged_runtime_is_named_rather_than_timing_out(
    tmp_path: Path, engagements, real_sessions, monkeypatch, caplog
) -> None:
    """The last silent dead-end.

    A runtime whose pid is alive but whose heartbeat is stale is invisible to
    `_has_live_runtime` (so the supervisor engages) AND holds the transcript
    lease (so the engage cannot spawn), burning the whole deadline and
    reporting an ordinary timeout. Naming it is the difference between "this
    wake is slow" and "this wake cannot fire until pid N recovers".
    """

    async def _no_live_record(config_dir, session_id):  # noqa: ANN001
        return False

    monkeypatch.setattr("local_operator.wakes.supervisor._has_live_runtime", _no_live_record)
    monkeypatch.setattr(
        "local_operator.wakes.supervisor.wedged_runtime",
        lambda _config, _session: (4242, 312.0),
    )
    write_entry(tmp_path, "sessionwedg2", cwd=str(tmp_path), schedules=[_schedule(NOW_MS - 5_000)])

    with caplog.at_level("WARNING"):
        fired = await fire_due_wakes(tmp_path, now_ms=NOW_MS)

    assert fired == 0
    assert engagements == [], "a wedged session must not burn an engage deadline"
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "wedged" in message and "4242" in message and "312" in message, message


def test_the_wedged_probe_reads_the_registry_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read from `registry.scan`, never re-derived from `heartbeat_at` here.

    The registry already owns live/wedged/stale; a second opinion about the
    same record is how two surfaces come to disagree about one process.
    """
    from local_operator.session.runtime.types import SessionRecord
    from local_operator.wakes.supervisor import wedged_runtime

    record = SessionRecord(
        pid=777,
        kind="tui",
        session_id="sessionwedg3",
        conversation_name="wedged",
        cwd="/tmp",
        model_label="m",
        control_port=0,
        control_key="k",
    )
    record.heartbeat_at = time.time() - 300
    monkeypatch.setattr(
        "local_operator.session.runtime.registry.scan",
        lambda _root=None: [(record, "wedged")],
    )

    found = wedged_runtime(tmp_path, "sessionwedg3")
    assert found is not None
    pid, age = found
    assert pid == 777
    assert age >= 299

    # A LIVE record of the same shape is not wedged: the classification is the
    # registry's, not this module's.
    monkeypatch.setattr(
        "local_operator.session.runtime.registry.scan",
        lambda _root=None: [(record, "live")],
    )
    assert wedged_runtime(tmp_path, "sessionwedg3") is None


# --- Round 2: per-schedule staleness, and the guards the round added ---------
#
# Round 2's reviewer observed that 3 of round 1's 4 new guards survived being
# reverted while the whole suite stayed green — the delta's own guards were its
# least-tested code, which is how R7 and R8 passed a green CI. Every test below
# is written to FAIL on a specific reverted line, and each was mutation-checked
# by actually reverting that line.


def _mixed_entry(now_ms: int) -> dict[str, object]:
    """One entry: a >7-day stale one-shot beside a live recurring watch.

    The R7 shape. `next_due_at(entry)` answers with the EARLIEST schedule, so
    an entry-level staleness test classifies this whole entry as stale — and
    once that predicate drove retirement, the live 20-minute watch stopped
    firing for good with no trace.
    """
    return {
        "schema": 1,
        "session_id": "mixedentry01",
        "cwd": "/tmp",
        "updated_at": now_ms,
        "schedules": [
            {"id": "w1", "message": "ancient one-shot", "next_due_at": now_ms - 9 * 86400_000},
            {
                "id": "w2",
                "message": "20-minute watch",
                "next_due_at": now_ms - 5_000,
                "every_ms": 1_200_000,
            },
        ],
    }


def test_a_stale_schedule_does_not_speak_for_its_live_siblings(tmp_path: Path) -> None:
    """R7 (BLOCKER): mixed entry ⇒ still fireable, and the LIVE wake is the one due.

    Mutation-checked: restoring the entry-level test (`_is_stale` answering on
    `next_due_at(entry)`) fails this with
    `the supervisor retired on an entry holding a live 20-minute watch`.
    """
    from local_operator.wakes.supervisor import (
        _due_sessions,
        _fireable_due_ms,
        _has_fireable_wakes,
        _is_stale,
    )

    now_ms = int(time.time() * 1000)
    entry = _mixed_entry(now_ms)
    index = {"mixedentry01": entry}
    (tmp_path / "sessions" / "mixedentry01").mkdir(parents=True)
    (tmp_path / "sessions" / "mixedentry01" / "transcript.jsonl").write_text("{}\n")

    assert not _is_stale(entry, now_ms), "an entry with a live schedule is not stale"
    assert _has_fireable_wakes(
        index, config_dir=tmp_path, now_ms=now_ms
    ), "the supervisor retired on an entry holding a live 20-minute watch"
    # And the wake it engages is the live one, not the week-old one-shot: the
    # occurrence key is built from this figure, so engaging on the stale row
    # would also mis-key the in-flight dedupe.
    assert _fireable_due_ms(entry, now_ms) == now_ms - 5_000
    assert [due for _, _, due in _due_sessions(index, now_ms)] == [now_ms - 5_000]


def test_an_entry_whose_every_schedule_is_stale_is_still_stale(tmp_path: Path) -> None:
    """The other half of R7: `all(...)` must not become "never stale"."""
    from local_operator.wakes.supervisor import _has_fireable_wakes, _is_stale

    now_ms = int(time.time() * 1000)
    entry = {
        "schema": 1,
        "session_id": "allstale0001",
        "cwd": "/tmp",
        "schedules": [
            {"id": "w1", "message": "old", "next_due_at": now_ms - 9 * 86400_000},
            {"id": "w2", "message": "older", "next_due_at": now_ms - 30 * 86400_000},
        ],
    }
    (tmp_path / "sessions" / "allstale0001").mkdir(parents=True)
    (tmp_path / "sessions" / "allstale0001" / "transcript.jsonl").write_text("{}\n")

    assert _is_stale(entry, now_ms)
    assert not _has_fireable_wakes({"allstale0001": entry}, config_dir=tmp_path, now_ms=now_ms)


@pytest.mark.asyncio
async def test_serve_keeps_running_for_a_live_watch_beside_a_stale_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R7 end to end: `serve()` must not retire, and must engage the live wake.

    The reviewer's probe asserted on the process, not the predicate, because
    the two disagreed: the contract-level test passed while `serve()` exited 0
    in 0.2 s. This runs the real loop.
    """
    from local_operator.wakes import supervisor as mod

    (tmp_path / "wakes").mkdir()
    (tmp_path / "sessions" / "mixedentry01").mkdir(parents=True)
    (tmp_path / "sessions" / "mixedentry01" / "transcript.jsonl").write_text("{}\n")
    now_ms = int(time.time() * 1000)
    (tmp_path / "wakes" / "mixedentry01.json").write_text(json.dumps(_mixed_entry(now_ms)))

    engaged: list[str] = []

    async def _fake_engage(session_id: str, *_args: object, **_kwargs: object) -> object:
        engaged.append(session_id)
        return object()

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", _fake_engage)
    monkeypatch.setattr(mod, "SLICE_S", 0.05)

    task = asyncio.create_task(mod.serve(tmp_path))
    await asyncio.sleep(0.6)
    still_running = not task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert still_running, "serve() retired on an entry that holds a live recurring watch"
    # The fake engage does not advance the schedule (the real session owns that
    # write), so the same occurrence stays due and is re-engaged each slice —
    # what matters is that it fired AT ALL, which it did not before this fix.
    assert engaged, "the live watch never fired"
    assert set(engaged) == {"mixedentry01"}, engaged
    assert "the supervisor is retiring" not in caplog.text


@pytest.mark.asyncio
async def test_the_retirement_line_says_which_state_it_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """D15/R7: a stale-only store retires BEFORE the sweep, so the retirement
    line is the only line the operator gets — it must carry the reason.

    Mutation-checked: dropping `_retirement_reason` from the log call fails
    this with `the retirement line does not say why`.
    """
    from local_operator.wakes import supervisor as mod

    (tmp_path / "wakes").mkdir()
    (tmp_path / "sessions" / "staleonly001").mkdir(parents=True)
    (tmp_path / "sessions" / "staleonly001" / "transcript.jsonl").write_text("{}\n")
    now_ms = int(time.time() * 1000)
    (tmp_path / "wakes" / "staleonly001.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "session_id": "staleonly001",
                "cwd": "/tmp",
                "schedules": [
                    {"id": "w1", "message": "forgotten", "next_due_at": now_ms - 9 * 86400_000}
                ],
            }
        )
    )
    monkeypatch.setattr(mod, "SLICE_S", 0.05)

    with caplog.at_level(logging.INFO):
        rc = await mod.serve(tmp_path)

    assert rc == 0
    assert "the supervisor is retiring" in caplog.text
    assert (
        "past the 7d staleness bound" in caplog.text
    ), f"the retirement line does not say why: {caplog.text}"
    assert "delivered when their sessions are next opened" in caplog.text


@pytest.mark.asyncio
async def test_a_runtimeerror_from_engage_is_throttled_and_the_wake_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R8 (MAJOR): `engage_runtime` documents RuntimeError for "every candidate
    died"; unhandled, it escaped a detached task as a 13-line asyncio traceback
    per slice — 8,640/day into an unrotated log, bypassing the throttle.

    Mutation-checked: removing `RuntimeError` from the `except` fails this with
    `a RuntimeError escaped the engage as an unretrieved task exception`.
    """
    from local_operator.wakes import supervisor as mod

    (tmp_path / "wakes").mkdir()
    (tmp_path / "sessions" / "alwaysfail01").mkdir(parents=True)
    (tmp_path / "sessions" / "alwaysfail01" / "transcript.jsonl").write_text("{}\n")
    now_ms = int(time.time() * 1000)
    (tmp_path / "wakes" / "alwaysfail01.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "session_id": "alwaysfail01",
                "cwd": "/tmp",
                "schedules": [
                    {
                        "id": "w1",
                        "message": "doomed",
                        "next_due_at": now_ms - 5_000,
                        "every_ms": 1_200_000,
                    }
                ],
            }
        )
    )

    attempts = 0

    async def _always_raises(*_args: object, **_kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("every candidate died: no credential")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", _always_raises)
    monkeypatch.setattr(mod, "SLICE_S", 0.05)

    loop_exceptions: list[dict[str, object]] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: loop_exceptions.append(context)
    )

    with caplog.at_level(logging.WARNING):
        task = asyncio.create_task(mod.serve(tmp_path))
        await asyncio.sleep(1.2)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await asyncio.sleep(0)

    assert (
        not loop_exceptions
    ), f"a RuntimeError escaped the engage as an unretrieved task exception: {loop_exceptions}"
    # Retried, not lost: the schedule is untouched, so later passes try again.
    assert attempts > 1, f"the wake was not retried after the failure (attempts={attempts})"
    # THROTTLED: many attempts, few lines. The burst bound is _SKIP_BURST.
    # Matched on the `failed:` prefix rather than the wrapper's old prose: the
    # subject now comes from the exception itself, which already names the
    # session (round 3, D22).
    lines = [rec for rec in caplog.records if rec.message.startswith("failed:")]
    assert (
        len(lines) <= mod._SKIP_LOG_BURST
    ), f"{attempts} failures produced {len(lines)} log lines; the throttle was bypassed"
    assert lines, "the engage failure was never reported"
    # THE WRAPPER ADDS ONLY TIMING (round 3, D22). The real `engage_runtime`
    # raises "could not start a runtime for session <id>: <cause>", so the
    # prefix must not restate the subject — asserted as "the session id
    # appears at most once", which holds whatever the cause says. (This test's
    # fake raises a bare message, so asserting on the phrase itself would pin
    # the fixture rather than the format.)
    assert lines[0].message.count("alwaysfail01") <= 1, lines[0].message
    assert lines[0].message.startswith("failed: after "), lines[0].message


@pytest.mark.asyncio
async def test_an_unanticipated_raise_from_an_engage_does_not_flood_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R8's belt: `_engage_one` is a detached task, so ANY unhandled raise lands
    in the same hole. A type nobody anticipated must still degrade to one
    throttled line, not a traceback per slice.

    Mutation-checked: removing the `except Exception` belt from `_run` fails
    this with `an unanticipated raise escaped the detached task`.
    """
    from local_operator.wakes import supervisor as mod

    (tmp_path / "wakes").mkdir()
    (tmp_path / "sessions" / "weirdfail001").mkdir(parents=True)
    (tmp_path / "sessions" / "weirdfail001" / "transcript.jsonl").write_text("{}\n")
    now_ms = int(time.time() * 1000)
    (tmp_path / "wakes" / "weirdfail001.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "session_id": "weirdfail001",
                "cwd": "/tmp",
                "schedules": [
                    {
                        "id": "w1",
                        "message": "doomed",
                        "next_due_at": now_ms - 5_000,
                        "every_ms": 1_200_000,
                    }
                ],
            }
        )
    )

    async def _raises_unexpectedly(*_args: object, **_kwargs: object) -> object:
        raise ValueError("nobody planned for this")

    monkeypatch.setattr(
        "local_operator.session.runtime.launch.engage_runtime", _raises_unexpectedly
    )
    monkeypatch.setattr(mod, "SLICE_S", 0.05)

    loop_exceptions: list[dict[str, object]] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: loop_exceptions.append(context)
    )

    with caplog.at_level(logging.WARNING):
        task = asyncio.create_task(mod.serve(tmp_path))
        await asyncio.sleep(1.0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await asyncio.sleep(0)

    assert (
        not loop_exceptions
    ), f"an unanticipated raise escaped the detached task: {loop_exceptions}"
    errors = [rec for rec in caplog.records if rec.message.startswith("error:")]
    assert errors, "the belt swallowed the failure without a word"
    assert len(errors) <= mod._SKIP_LOG_BURST, f"{len(errors)} lines; the throttle was bypassed"
    assert "ValueError" in errors[0].message
