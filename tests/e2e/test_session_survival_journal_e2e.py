"""A killed runtime must leave a readable account of the turn it was running.

WHY THIS FILE IS E2E AND NOT A UNIT TEST. Every piece of the instrumentation has
unit coverage (``tests/unit/session/runtime/test_turn_journal.py``), but the
pieces are then wired into the REAL runtime by ``process.amain`` at boot and by
the real turn pipeline — and tonight's incident was precisely a case where the
wiring was the thing nobody could see. So this cell boots production
``process.py`` in a subprocess, parks a real turn in the real ``bash`` tool via
the mock provider's ``[bash:N]`` marker, SIGKILLs that pid, and then reads what
it left behind from ANOTHER process:

* ``<session>/turn-journal.json`` — an OPEN row naming the turn that was in
  flight, the build the runtime was running and its install root;
* ``run/host/<pid>.json`` — the boot record, still there, which is exactly the
  file that says "this pid existed and stopped without exiting cleanly";
* and the successor's classification, which now NAMES the cause from that
  evidence instead of inferring it from a missing record.

Isolation: ``headless_tui_env`` redirects the config dir and the root conftest
redirects ``HOME``; the child environment is rebuilt from the cut-off suite's
helper, which removes EVERY ``CMUX_*`` / ``LOP_MOBILE_CHILD_*`` /
``LOP_RUNTIME_*`` variable, because a runtime that inherited a workspace id could
address the operator's live window (#648). Nothing here signals a process this
file did not spawn, and every signal is :meth:`Popen.kill` on that exact pid — a
pattern kill on a box holding dozens of live agent sessions is the hazard this
whole design exists to remove.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.session.attention import _classify_orphaned_run
from local_operator.session.runtime import journal, registry
from tests.e2e.test_cut_off_turns_e2e import (
    _attach,
    _incidents,
    _park_a_turn,
    _reap,
    _seed,
    _spawn,
    _successor_boot,
)
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_a_killed_runtime_leaves_an_open_journal_row_and_a_boot_record(
    headless_tui_env: Path,
) -> None:
    """The 19:41 shape, end to end: a hard death that says what it was doing."""
    config = headless_tui_env
    session_id = "journalkill01"
    directory = _seed(config, session_id)
    child = _spawn(config, session_id)
    viewer = None
    try:
        with bounded(180, "session survival: killed runtime journal"):
            viewer = await _attach(config, session_id)
            await _park_a_turn(viewer, directory, seconds=20)

            # -- while the runtime is ALIVE, the row says a turn is in flight ---
            live = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
            assert live is not None, "the booted runtime wrote no turn journal row"
            assert live.open is True
            assert live.pid == child.pid
            assert live.session_id == session_id
            assert live.turn_seq == 1
            assert live.command_id, "the row must name the command it was running"
            assert live.build_stamp() is not None, "the row must record the build it loaded"
            assert live.install_root
            # A LIVE owner with an open row is NOT a death — the direction that
            # would turn every running session into a reported interruption.
            assert journal.open_row_after_death(directory) is None

            # The boot record is published before the control socket listens, so
            # it is readable while the runtime is still serving.
            record = journal.read_boot_record(child.pid, root=config)
            assert record is not None, "no boot record for a listening runtime"
            assert record.session_id == session_id
            assert record.parent_pid > 0

            # -- the case study's death: SIGKILL, no exit path, no marker -------
            killed_pid = child.pid
            child.kill()
            child.wait(timeout=10)
            await asyncio.sleep(0.2)

            # The turn really was interrupted, not completed.
            assert "from the mock provider" not in (directory / "transcript.jsonl").read_text(
                encoding="utf-8"
            ), "the turn completed, so nothing was cut off"

            # -- what survived the corpse ------------------------------------
            row = journal.open_row_after_death(directory)
            assert row is not None, "a killed runtime's row must read as an unfinished turn"
            assert row.pid == killed_pid
            assert row.ended_at is None and row.end_cause == ""

            # A clean exit withdraws this file, so its survival IS the evidence
            # that this pid stopped without running its own exit ordering.
            assert journal.read_boot_record(killed_pid, root=config) is not None

            # -- and the successor NAMES it ----------------------------------
            kind, cause, reason = _classify_orphaned_run(directory)
            assert (kind, cause) == ("error", "runtime-killed"), (kind, cause, reason)
            assert "turn 1 in flight" in reason, reason
            assert f"pid {killed_pid}" in reason, reason

            session = await _successor_boot(directory)
            try:
                incidents = _incidents(directory)
                assert len(incidents) == 1, f"expected one incident, got {len(incidents)}"
                raw = str((incidents[0].payload.get("details") or {}).get("raw") or "")
                assert "in flight" in raw, raw
            finally:
                await session.dispose()
            # A second boot of the same directory narrates nothing new.
            second = await _successor_boot(directory)
            try:
                assert len(_incidents(directory)) == 1, "re-opening re-narrated the interruption"
            finally:
                await second.dispose()
    finally:
        if viewer is not None:
            try:
                await viewer.dispose()
            except Exception:  # noqa: BLE001 — teardown of a killed owner
                pass
        if child.poll() is None:
            _reap(child, config)


@pytest.mark.asyncio
async def test_a_clean_idle_exit_withdraws_its_boot_record(headless_tui_env: Path) -> None:
    """The other side of the asymmetry, on the real exit path.

    A boot record that SURVIVES says "this pid stopped without running its own
    exit ordering". The converse has to hold on a real clean exit or the file
    means nothing: every runtime that ever lived would leave one behind, and
    ``run/host`` would fill with records of sessions that ended politely.

    No turn is started, so there is no journal row either — the other half of
    the same claim: the journal records TURNS, not process lifetime.
    """
    import subprocess
    import sys

    from tests.e2e.test_cut_off_turns_e2e import _child_env

    config = headless_tui_env
    session_id = "journalidle01"
    directory = _seed(config, session_id)
    # A short grace so the idle exit lands inside this cell. The residency policy
    # is untouched (3.0 s by default); this is the suite's existing seam for
    # moving the deadline rather than the behaviour.
    env = _child_env(config, session_id)
    env["LOP_SESSION_GRACE_S"] = "3"
    child = subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        with bounded(120, "session survival: clean idle exit"):
            # Wait for the record FIRST, so "it exited" cannot be satisfied by a
            # runtime that never booted.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 30
            while loop.time() < deadline:
                if journal.read_boot_record(child.pid, root=config) is not None:
                    break
                await asyncio.sleep(0.05)
            assert (
                journal.read_boot_record(child.pid, root=config) is not None
            ), "the boot record must exist before the exit, or this cell proves nothing"
            await asyncio.to_thread(child.wait, 60)
            assert child.returncode == 0, child.returncode
    finally:
        if child.poll() is None:
            _reap(child, config)

    assert (
        journal.read_boot_record(child.pid, root=config) is None
    ), "a clean exit must withdraw its boot record"
    assert registry.read_turn_journal(directory) is None, "no turn ran, so no row"
