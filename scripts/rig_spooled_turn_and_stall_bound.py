#!/usr/bin/env python3
"""Reproduce the two 2026-09-21 deaths of a session runtime, in isolation.

WHAT THIS REPRODUCES, AND WHY THE TWO HALVES SHARE ONE FILE
============================================================

Both halves are "a runtime went away and the session it served was left worse
off", and both were measured on this machine on 2026-09-21:

**A. A death with a name, recorded as one with none.** Pids 57975, 4698 and
79757 each left ``logs/runtime-stall-<pid>.log`` carrying ``Timeout (`` — the
marker ``stall_watchdog`` writes when the runtime's OWN 300 s no-progress bound
fires, from a C thread — and the builds that shipped before 2026-09-23 then took
the process out of that same C thread. Each was nevertheless narrated to its
successor as ``runtime-killed`` with ``(unattributed, ...)``: nothing on the
death path had ever read the file, so the one artifact that named the act was
invisible to every surface.

PRODUCTION EXPIRY IS DUMP-ONLY NOW, so this half asserts the contract THAT holds
rather than the one the incident was measured under: the bound FIRES, dumps every
thread beside its log, and the process that armed it is STILL ALIVE afterwards —
while no surface narrates it as gone. The child below is deliberately the smallest
thing that can arm a bound and survive a fire: a bare interpreter that calls
``stall_watchdog.arm(seconds=2)`` and then parks in ``time.sleep(60)``. It holds
no work and serves no turn, so it is a LIVENESS control rather than the shape the
incident had — the runtime holding a turn in flight is the serving-plane cell's
child, not this one, and this half's job is the naming rung plus the fire's
aftermath. A rig that waited for the child to die would wait
forever on this build and pass only against the source it replaced, which is
exactly what this half did before the convergence round (MAJOR-1): ``--phase a``
returned ``RC=1`` on the head and ``RC=0`` on the pre-change tree.

THE RUNG THE INCIDENT WAS ABOUT IS STILL CHECKED, against an artifact this half
writes itself: ``incidents.STALL_BOUND_CAUSE = "runtime-stall-bound"`` (read by
``journal._stall_bound_leg``) is what a LEGACY header that states its own
convention earns, so the fire's own stacks are re-read under a legacy header and
the name must still come back. That keeps the rig an INDEPENDENT end-to-end check
of the rung without restoring any fatal behaviour to make an old assertion pass.
The control still prints the measured pre-change verdict (``runtime-killed`` /
``unattributed``) for the no-artifact case.

**B. A handover that ended with no successor.** A session retired for a newer
build with rows in its spool, and no runtime was ever raised for them. The
sender's receipt said "held for the next runtime — it runs it", and no process
owned that promise: the spool is drained only BY a runtime, the draining runtime
cannot start one (it holds the transcript lease until it exits), and the wake
supervisor fired only from the schedule index, which a spooled peer message
never enters.

Half B prints a CONTROL first, which is the defect itself: with the spooled
message present and no obligation recorded, the supervisor starts nothing and the
message waits — exactly the measured incident. It then records the obligation the
spool writer now records, and the same call raises a real successor that really
delivers the message.

ISOLATION, AND HOW TO READ THE OUTPUT
=====================================

Every cell runs under a private ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR`` created
under one ``mktemp -d`` root, with ``CMUX_*``/``LOP_*`` absent by construction
(the child env is built from scratch, not inherited). Nothing here reads or
writes the operator's live store, and the one process half B leaves behind is
reaped by exact pid — never by program name.

    .venv/bin/python scripts/rig_spooled_turn_and_stall_bound.py

Exit code 0 means both halves asserted what they claim — for half A that is the
DUMP-ONLY contract: the bound fired, the runtime SURVIVED it, no surface calls that
fire a death, and a legacy artifact still earns the class name. Any other value means
one of them did not, and the printed line says which.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

#: Seconds the rig's own child waits for its bound to fire. Deliberately far below
#: the production 300: the bound is a parameter of ``arm``, so the rig exercises
#: the real mechanism at a speed a test can watch.
RIG_STALL_S = 2.0

#: How long half A waits for its child's own bound to fire. A BACKSTOP, not the
#: assertion: the event is ``faulthandler``'s fired marker appearing in the dump, and
#: this only bounds the rig when the bound never fires at all. Comfortably above the
#: bound it waits on, so a loaded host reordering a 2 s timer and a poll cannot turn a
#: working mechanism into a red rig.
RIG_FIRE_DEADLINE_S = 30.0

#: How long half B waits for the successor to boot, drain and clear.
SUCCESSOR_DEADLINE_S = 60.0


def _child_env(root: Path) -> dict[str, str]:
    """A store-private environment, built rather than filtered.

    Built so that a variable this file has never heard of cannot arrive: the
    ``LOP_*`` family decides what a child runtime is (``process.py``), and a
    ``lop`` parent's ``CMUX_*`` renames real workspaces. Neither belongs in a
    cell. ``PATH`` is copied because a spawn shells out for ``ps``.
    """
    return {
        "HOME": str(root),
        "LOCAL_OPERATOR_CONFIG_DIR": str(root / ".local-operator"),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TERM": "xterm-256color",
        # THE NOTIFICATION GATE, and it is not decoration: this rig spawns real
        # ``local-operator`` children, and a child that can raise a notification
        # puts whatever this rig (or a mock session it drives) says onto the
        # operator's lock screen. ``tests/unit/test_notification_isolation.py``
        # sweeps for builders that hand a child an environment and forget this;
        # the whole reason is that a bespoke mapping is built rather than
        # inherited, so inheriting the gate by accident is not the property.
        "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1",
    }


def _run_phase(phase: str, root: Path) -> int:
    return subprocess.call(
        [sys.executable, str(Path(__file__).resolve()), "--phase", phase, "--root", str(root)],
        env=_child_env(root),
    )


def _phase_a(root: Path) -> int:  # noqa: C901 — a script, and the print IS the report
    """A real bound firing on a real child, then the verdict a successor reaches.

    THE CONTRACT WITH A LIVE RUNTIME IS DUMP-ONLY (2026-09-23 convergence round,
    MAJOR-1), so this half WAITS ON THE ARTIFACT rather than on the child's exit: a
    fire no longer ends the child, and a wait for an exit is a wait that never
    completes on this build. In order, it asserts:

    1. the bound FIRED — the dump carries ``Timeout (`` — and the child was still
       alive at that moment;
    2. the child is still alive AFTER the fire, so the fire ended nothing;
    3. :func:`stall_watchdog.fire_outcome` answers FIRE_SURVIVED, read off the header
       this build writes rather than inferred from what the file does not say;
    4. the verdict for that row is ``runtime-killed`` carrying the did-not-end clause
       and ``(unattributed, ...)`` — nothing narrates a live runtime as a bound death;
    5. THE SAME FIRE, re-read under a LEGACY header this half writes itself, still
       names :data:`incidents.STALL_BOUND_CAUSE` — the rung the incident was about,
       exercised end to end without any fatal arm in this tree;
    6. with the artifact removed, the verdict is the measured pre-change state.
    """
    import time

    from local_operator.incidents import (
        KILL_CAUSE,
        KILL_UNATTRIBUTED,
        STALL_BOUND_CAUSE,
    )
    from local_operator.paths import config_dir
    from local_operator.session.runtime import journal, registry, stall_watchdog

    cfg = config_dir()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time;"
            "from local_operator.session.runtime import stall_watchdog as w;"
            f" w.arm(seconds={RIG_STALL_S});"
            " print('armed', flush=True); time.sleep(60)",
        ],
        env={**os.environ, "HOME": str(root), "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1"},
        stdout=subprocess.PIPE,
    )
    print(f"child pid {child.pid}")
    failures = 0
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == b"armed"
        # THE EVENT IS FAULTHANDLER'S OWN MARKER; the deadline is only the backstop for
        # a rig whose bound never fired at all. The dump file itself exists from arm
        # time (the header is written before the timer is armed), so its existence is
        # NOT the event — the fired line is.
        dump = stall_watchdog.dump_path(child.pid)
        deadline = time.monotonic() + RIG_FIRE_DEADLINE_S
        text = ""
        while time.monotonic() < deadline:
            try:
                text = dump.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if stall_watchdog.FIRED_MARKER in text:
                break
            time.sleep(0.05)
        fired = stall_watchdog.FIRED_MARKER in text
        alive_at_fire = child.poll() is None
        print(f"dump: {dump}")
        print(f"fired marker present: {fired}")
        print(f"child alive at the fire: {alive_at_fire} (returncode {child.poll()})")
        if not fired:
            print(f"FAIL A: the bound never fired within {RIG_FIRE_DEADLINE_S:g} s")
            return 1
        if not alive_at_fire:
            print("FAIL A: the fire ended the child, and production expiry is dump-only")
            return 1

        outcome = stall_watchdog.fire_outcome(child.pid, dump.parent)
        print(f"fire_outcome: {outcome}")
        if outcome != stall_watchdog.FIRE_SURVIVED:
            print(
                "FAIL A: the artifact does not prove the fire was survivable "
                f"({outcome!r} != {stall_watchdog.FIRE_SURVIVED!r})"
            )
            failures += 1

        # THE ROW IS STAMPED WITH THE INSTALL THAT IS REAL HERE, not a fabricated
        # one (review round 1, R1-9). With a made-up stamp the tear rung
        # (``install-mid-update``) answers for the dump-less case instead of the
        # unattributed one, and the control then shows "the artifact mattered"
        # rather than the state the real successors were actually told — which is
        # ``runtime-killed`` with ``(unattributed, ...)``.
        from local_operator import buildwatch
        from local_operator.update import installed_build

        try:
            stamp = installed_build(buildwatch.build_prefix())
            build_fields = {"version": stamp.version, "source_ref": stamp.source_ref}
        except Exception:  # noqa: BLE001 — an unstamped tree is itself a real shape
            build_fields = {}
        print(f"row build stamp: {build_fields or '(none — an unstamped tree)'}")

        session = "rig-stall"
        directory = cfg / "sessions" / session
        directory.mkdir(parents=True, exist_ok=True)
        now = time.time()
        registry.write_turn_journal(
            directory,
            {
                "session_id": session,
                "pid": child.pid,
                "parent_pid": os.getpid(),
                "turn_seq": 1,
                "command_id": "cmd-rig",
                "started_at": now - 5,
                "ended_at": None,
                "open": True,
                "end_cause": "",
                "exit_cause": "",
                "still_open_at_exit": False,
                "last_boundary": "bash",
                "build": build_fields,
                "install_root": "/tmp/rig-install",
                "updated_at": now - 1,
            },
        )
        row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
        assert row is not None

        kind, cause, reason = journal.death_verdict(row)
        print(f"verdict on the live runtime's row: {kind} / {cause}")
        print(f"reason: {reason}")
        if cause != KILL_CAUSE or STALL_BOUND_CAUSE in reason:
            print(
                "FAIL A: a dump-only fire on a live runtime was narrated as a bound "
                f"death ({cause})"
            )
            failures += 1
        elif journal.HELD_BOUND_LEAD not in reason or KILL_UNATTRIBUTED not in reason:
            print("FAIL A: the fire is not on the row as the did-not-end clause")
            failures += 1

        # THE RUNG, against the artifact that can support it: the same fire's stacks
        # under a header stating the LEGACY convention. Written by this phase, never
        # produced by an arm in this tree — the point is that the reader still names the
        # class for the dumps that exist on disk today.
        fire_body = text[text.index(stall_watchdog.FIRED_MARKER) :]
        dump.write_text(
            f"{stall_watchdog.ARM_MARKER}pid {child.pid} armed for {RIG_STALL_S:g}s\n"
            f"{stall_watchdog.LEGACY_FATAL_POLICY_PHRASES[1]}\n{fire_body}",
            encoding="utf-8",
        )
        legacy_outcome = stall_watchdog.fire_outcome(child.pid, dump.parent)
        _kind, legacy_cause, legacy_reason = journal.death_verdict(row)
        print(f"legacy artifact: fire_outcome={legacy_outcome} verdict={legacy_cause}")
        print(f"legacy reason: {legacy_reason}")
        if legacy_outcome != stall_watchdog.FIRE_LEGACY_FATAL or legacy_cause != STALL_BOUND_CAUSE:
            print(f"FAIL A: a legacy artifact no longer names {STALL_BOUND_CAUSE}")
            failures += 1

        # The counterfactual, taken away and re-asked: with no artifact the verdict
        # falls to the rungs that existed before this change — and for a row whose
        # install has not moved, that is exactly what the real successors were told:
        # ``runtime-killed`` carrying ``(unattributed, ...)``.
        dump.unlink()
        _kind, without, without_reason = journal.death_verdict(row)
        print(f"without the dump: {without} | {without_reason}")
        if without != KILL_CAUSE or KILL_UNATTRIBUTED not in without_reason:
            print("FAIL A: the dump-less verdict is not the measured pre-change state")
            failures += 1

        # THE RUNTIME IS STILL THERE, which is the whole property this half exists to
        # assert: the fire was an observation and the process carried on holding work.
        still_alive = child.poll() is None
        print(f"child alive after the whole sequence: {still_alive}")
        if not still_alive:
            print("FAIL A: the runtime did not survive its own bound")
            failures += 1
    finally:
        # Reaped by EXACT PID, never by program name, and graduated like phase B.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                child.send_signal(sig)
            except ProcessLookupError:
                break
            try:
                child.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue

    if failures:
        print(f"FAIL A: {failures} assertion(s) above did not hold")
        return 1
    print(f"RESULT A: DUMP-ONLY FIRE, RUNTIME SURVIVED, LEGACY RUNG NAMED ({STALL_BOUND_CAUSE})")
    return 0


def _phase_b(root: Path) -> int:  # noqa: C901 — see above
    """The control, then the circuit: a spooled turn raises a real successor."""
    import asyncio
    import json
    import time

    from local_operator.config import ConfigManager
    from local_operator.harness.types import Message
    from local_operator.paths import config_dir
    from local_operator.session.runtime.inbox import InboxLine, append_inbox, peek_inbox
    from local_operator.session.transcript import Transcript
    from local_operator.wakes import spooled
    from local_operator.wakes.supervisor import fire_due_wakes

    cfg = config_dir()
    # A spawn cannot construct a session without a hosting/model pair. Nothing
    # here calls a model (the delivery is one spooled peer row, and the drain
    # runs before the socket listens), but the child refuses to boot without one.
    manager = ConfigManager(cfg)
    manager.set_config_value("hosting", "deepseek")
    manager.set_config_value("model_name", "deepseek/deepseek-flash")

    session = "rig-spool"
    directory = cfg / "sessions" / session
    directory.mkdir(parents=True, exist_ok=True)
    # Durable history, written through the product's own encoder: a peer row is
    # delivered into a conversation, and the boot drain deliberately defers one
    # destined for a session whose owner has never typed.
    asyncio.run(Transcript(directory).append_messages([Message.user("an earlier turn")]))

    append_inbox(
        directory,
        InboxLine(text="rig: run the census", wake=True, sender={"session_id": "rig"}),
    )
    print(f"spool rows after the message: {len(peek_inbox(directory))}")

    # THE DEFECT, measured by the same call the fix relies on.
    control = asyncio.run(fire_due_wakes(cfg))
    print(f"control (spool only, no record): engagements started = {control}")
    print(f"control: spool rows left = {len(peek_inbox(directory))}")

    spooled.note_spooled_turn(cfg, session, cwd=str(root))
    print(f"recorded: {spooled.read_spooled_turn(cfg, session)}")
    started = asyncio.run(fire_due_wakes(cfg))
    print(f"engagements started with the record = {started}")

    def pid_of(sid: str) -> int | None:
        for path in (cfg / "run" / "host").glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if data.get("session_id") == sid:
                return int(data["pid"])
        return None

    deadline = time.time() + SUCCESSOR_DEADLINE_S
    pid: int | None = None
    drained = False
    while time.time() < deadline:
        pid = pid or pid_of(session)
        if not peek_inbox(directory):
            drained = True
            break
        time.sleep(0.5)

    print(f"successor runtime pid: {pid}")
    print(f"spool drained by the successor: {drained}")
    print(f"record now: {spooled.read_spooled_turn(cfg, session)}")
    log = cfg / "logs" / "runtime.log"
    if log.exists():
        print("--- isolated runtime.log (tail) ---")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]:
            print("   ", line[:150])

    # Reaped by exact pid, and graduated: SIGTERM, a bounded wait, then SIGKILL.
    if pid:
        for sig in (15, 9):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.25)

    if control != 0:
        print("FAIL B: the control also engaged, so the cell proves nothing")
        return 1
    if not pid or not drained:
        print("FAIL B: no successor delivered the spooled message")
        return 1
    if spooled.read_spooled_turn(cfg, session) is not None:
        print("FAIL B: the record outlived the spool it described")
        return 1
    print("RESULT B: SUCCESSOR RAISED, MESSAGE DELIVERED")
    return 0


def main(argv: list[str]) -> int:
    if "--phase" in argv:
        phase = argv[argv.index("--phase") + 1]
        root = Path(argv[argv.index("--root") + 1])
        return _phase_a(root) if phase == "a" else _phase_b(root)

    root = Path(tempfile.mkdtemp(prefix="lop-rig-"))
    print(f"ISO={root}")
    try:
        failures = 0
        for phase, title in (
            ("a", "A. a bound that fired, and whether the verdict says so"),
            ("b", "B. a spooled turn with no runtime: does a successor appear?"),
        ):
            print("\n" + "=" * 66)
            print(title)
            print("=" * 66)
            failures += _run_phase(phase, root) != 0
        print(f"\nrig done; isolated root: {root}")
        return 1 if failures else 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
