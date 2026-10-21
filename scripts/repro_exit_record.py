"""THE REAL PATH: a runtime leaves, and the store can say when, why and who asked.

WHAT THIS REPRODUCES, AND THE REPORT IT ANSWERS. On 2026-10-21 the operator asked
"when did it stop" about session ``709291bc6e49``. It had left through the
runtime's OWN residency policy — the 3 s idle drain — which logged one INFO line
and withdrew its boot record; the store showed the session as ``stored`` with no
time and no reason anywhere, indistinguishable from a SIGKILL. This script spawns
real runtime processes under a scratch config root and drives the four paths a
session can leave by, then reads the store the way an operator does:

  1. IDLE — before: ``lop sessions --all`` lists the session LIVE; after: it is
     STORED with a STOPPED age and the reason ``idle-exit`` read from
     ``sessions/<id>/runtime-exit.json``. The record's ``at`` must be within a
     second of the exit this script observed, no ``runtime-stop.json`` may exist
     (nobody asked for it, so ``stop_requested`` is False), and a second seeded
     session with NO record must still read ``—``/unknown.
  2. SOCKET STOP — ``lop stop`` through the real ladder. The record must say
     ``reason: socket-stop`` and ``requested: true``, and the ladder's own
     ``runtime-stop.json`` (rung ``socket``, ``deliberate: true``) must exist.
     This is the cell review round 1 made a blocker: the ladder stages its marker
     AFTER the target's ack, so a runtime reading the marker while starting its
     exit saw nothing and reported ``false`` 3/3.
  3. UNATTRIBUTED SIGTERM — a signal nobody staged. The record must say
     ``SIGTERM`` and ``requested: false``: the entire point of the field is that
     the 2026-09-30 sweep and a deliberate stop are no longer the same row.
  4. RESUME → SIGKILL — run A idles out (leaving its record), the session is
     resumed and run B is killed. The listing must read ``—``/unknown, never A's
     ``idle-exit`` at A's time (review round 1, MAJOR-1). The run record written
     at B's boot is the evidence the listing uses.

ISOLATION. A fresh HOME and a fresh ``LOCAL_OPERATOR_CONFIG_DIR``: AGENTS.md is
explicit that the config dir alone is not enough (the cache and agent home derive
from HOME independently), so both are redirected and every child inherits neither
the operator's ``CMUX_*``/``LOP_*`` families nor notification permission (through
``agent_shell.harness_child_env``). Nothing here can touch a live session.

USAGE
=====

    cd ~/local-operator/.worktrees/<worktree>
    ISO=$(mktemp -d) && env -i HOME="$ISO" PATH="$PATH" TERM=xterm-256color \\
      PYTHONPATH=$PWD .venv/bin/python scripts/repro_exit_record.py

The script prints each command it ran and that command's ACTUAL output, then
asserts on it and exits non-zero if any acceptance condition fails. The ``env -i``
wrapper is this suite's own convention for a real-path run: it keeps the run out
of the operator's store, and the script additionally redirects HOME and the config
dir itself so it is safe under a plain invocation too.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: The idle-path session: the one whose departure the listing must explain.
SESSION_ID = "exitrec00001"
#: A session that never had a runtime: the "unknown, not a fabricated time" control.
QUIET_ID = "exitrec00002"
#: The socket-stop session (``lop stop`` through the ladder).
STOP_ID = "exitrec00003"
#: The unattributed-SIGTERM session.
TERM_ID = "exitrec00004"
#: Run A of the resume→kill cell (idles out cleanly).
RESUME_ID = "exitrec00005"

#: The idle window the children are given. The shipped value is 3.0 s; this is the
#: suite's existing seam for moving the deadline rather than the behaviour
#: (``tests/e2e/test_session_survival_journal_e2e.py`` uses the same variable).
GRACE_S = "3"


def _say(step: str) -> None:
    print(f"\n=== {step} ===", flush=True)


def _run(args: list[str], *, env: dict[str, str]) -> tuple[int, str]:
    completed = subprocess.run(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout


def _cli(env: dict[str, str], *args: str) -> tuple[int, str]:
    """The REAL command an operator runs, against the scratch root."""
    return _run(
        [sys.executable, "-c", "import sys; from local_operator.cli import main; sys.exit(main())"]
        + list(args),
        env=env,
    )


def _child_env(config: Path, home: Path, session_id: str) -> dict[str, str]:
    """A harness-declared child environment pointed at one scratch session.

    ``harness_child_env`` is the product's ONE carrier for "a script driving the
    real CLI": it sets the notification gate (a scratch-home child's mock turn
    must never reach the operator's lock screen) and the nested-session allowance.
    The families are stripped first because an inherited ``LOP_RUNTIME_*`` or
    ``CMUX_*`` would point the child at the operator's own session.
    """
    from local_operator.agent_shell import harness_child_env
    from local_operator.tui.resume_click import DESKTOP_LAUNCH_REFUSED_ENV

    env = harness_child_env(
        {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    )
    env.update(
        {
            "HOME": str(home),
            "LOCAL_OPERATOR_CONFIG_DIR": str(config),
            "LOP_MOBILE_CHILD_CWD": str(config),
            "LOP_MOBILE_CHILD_RESUME": session_id,
            "LOP_SESSION_GRACE_S": GRACE_S,
            # Not a notification switch: this refuses a DESKTOP LAUNCH, so the
            # child cannot spawn a second product instance against the scratch home.
            DESKTOP_LAUNCH_REFUSED_ENV: "1",
        }
    )
    return env


def _seed(config: Path, session_id: str, text: str) -> Path:
    """A session with one durable row, on the mock provider."""
    directory = config / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        f'"role": "user", "content": [{{"type": "text", "text": "{text}"}}]}}\n',
        encoding="utf-8",
    )
    return directory


def _spawn(env: dict[str, str]) -> subprocess.Popen[bytes]:
    child = subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"pid {child.pid}")
    return child


def _wait_live(config: Path, session_id: str, timeout: float = 30.0) -> None:
    """Wait for the runtime's own live record, so "it stopped" cannot be vacuous."""
    from local_operator.session.runtime import registry

    deadline = time.time() + timeout
    while time.time() < deadline:
        for record, state in registry.scan(config):
            if getattr(record, "session_id", "") == session_id and state == "live":
                return
        time.sleep(0.05)
    raise SystemExit(f"runtime for {session_id} never published a live record")


def _wait_gone(child: subprocess.Popen[bytes], timeout: float = 120.0) -> float:
    """Wait out the process and return the moment it was observed gone."""
    try:
        child.wait(timeout=timeout)
    finally:
        if child.poll() is None:
            child.kill()
    return time.time()


def _json_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _session_dir(config: Path, session_id: str) -> Path:
    return config / "sessions" / session_id


def _idle_scenario(config: Path, home: Path, failures: list[str]) -> None:
    from local_operator.session.runtime import journal

    env = _child_env(config, home, SESSION_ID)
    _say("IDLE / STEP 0 — the store before any runtime: no records anywhere")
    print(_cli(env, "sessions", "--all")[1])

    _say("IDLE / spawn the real runtime process")
    child = _spawn(env)
    _wait_live(config, SESSION_ID)

    _say("IDLE / BEFORE — the same command with the runtime LIVE")
    before_code, before_out = _cli(env, "sessions", "--all")
    print(before_out)
    if before_code != 0:
        raise SystemExit(f"sessions exited {before_code}")

    exit_at = _wait_gone(child)
    print(f"runtime exited rc={child.returncode} at {exit_at:.3f}")

    _say("IDLE / AFTER — `lop sessions --all` with the session STORED")
    after_code, after_out = _cli(env, "sessions", "--all")
    print(after_out)
    if after_code != 0:
        failures.append(f"`lop sessions --all` exited {after_code}")
    if "STOPPED" not in after_out or "STOP_REASON" not in after_out:
        failures.append("the stored listing has no STOPPED/STOP_REASON columns")
    # The CONVERSATION column shows the session's title; for a seed with no
    # title sidecar that is the id, so the row is found by id.
    stopped_lines = [line for line in after_out.splitlines() if SESSION_ID in line]
    if not stopped_lines:
        failures.append("the stopped session is missing from the listing")
    elif "idle-exit" not in stopped_lines[0]:
        failures.append(f"the listing does not name idle-exit: {stopped_lines[0]!r}")

    _say("IDLE / THE ARTIFACT — sessions/<id>/runtime-exit.json")
    record_path = _session_dir(config, SESSION_ID) / "runtime-exit.json"
    if not record_path.exists():
        raise SystemExit(f"no exit record at {record_path}")
    raw = json.loads(record_path.read_text())
    print(json.dumps(raw, indent=2))

    parsed = journal.read_exit_record(_session_dir(config, SESSION_ID))
    stop_marker = _session_dir(config, SESSION_ID) / "runtime-stop.json"
    if parsed is None:
        failures.append("the exit record does not parse")
    else:
        if parsed.reason != "idle-exit":
            failures.append(f"reason is {parsed.reason!r}, not 'idle-exit'")
        if parsed.detail != "no work, no viewer, no wake within 90s":
            failures.append(f"detail is {parsed.detail!r}")
        if parsed.requested:
            failures.append("a policy exit was reported as a requested stop")
        if abs(parsed.at - exit_at) > 1.0:
            failures.append(f"record at={parsed.at:.3f} is not within 1s of exit {exit_at}")
        if parsed.pid != child.pid:
            failures.append(f"record pid {parsed.pid} != runtime pid {child.pid}")
    if stop_marker.exists():
        failures.append(f"a stop marker exists at {stop_marker}")

    # THE MACHINE-READ HALF: the published JSON row, where the STOPPED age is an
    # exact epoch rather than a rendered duration.
    json_code, json_out = _cli(env, "sessions", "--all", "--json")
    if json_code != 0:
        failures.append(f"`lop sessions --all --json` exited {json_code}")
    else:
        published = {row["session_id"]: row for row in json.loads(json_out)}
        row = published.get(SESSION_ID)
        if row is None:
            failures.append("the stopped session is missing from the JSON listing")
        else:
            stopped_at_s = row["stopped_at_s"]
            if stopped_at_s is None:
                failures.append("stopped_at_s is null for a session with an exit record")
            elif not (0.0 <= time.time() - float(stopped_at_s) < 30.0):
                failures.append(f"stopped_at_s is not a recent real time: {stopped_at_s!r}")
            if row["stop_reason"] != "idle-exit":
                failures.append(f"stop_reason is {row['stop_reason']!r}")
            if row["stop_requested"] is not False:
                failures.append(f"stop_requested is {row['stop_requested']!r}")
        quiet_row = published.get(QUIET_ID)
        if quiet_row is None:
            failures.append("the quiet control is missing from the JSON listing")
        elif quiet_row["stopped_at_s"] is not None or quiet_row["stop_reason"]:
            failures.append(f"a session with no record fabricated a stop: {quiet_row!r}")


def _socket_stop_scenario(config: Path, home: Path, failures: list[str]) -> None:
    """``lop stop`` through the real ladder: a client asked, so requested is True."""
    from local_operator.session.runtime import journal

    env = _child_env(config, home, STOP_ID)
    _say("STOP / spawn the real runtime, then `lop stop <session>`")
    child = _spawn(env)
    _wait_live(config, STOP_ID)
    stop_code, stop_out = _cli(env, "stop", STOP_ID)
    print(f"stop rc={stop_code}: {stop_out.strip()}")
    if stop_code != 0:
        failures.append(f"`lop stop` exited {stop_code}: {stop_out.strip()[:200]}")
    exit_at = _wait_gone(child)
    print(f"runtime exited rc={child.returncode} at {exit_at:.3f}")

    record = journal.read_exit_record(_session_dir(config, STOP_ID))
    marker = _json_file(_session_dir(config, STOP_ID) / "runtime-stop.json")
    _say("STOP / the exit record and the ladder's marker")
    print(json.dumps(_json_file(_session_dir(config, STOP_ID) / "runtime-exit.json"), indent=2))
    print(json.dumps(marker, indent=2))

    if record is None:
        failures.append("the socket-stop exit record is missing")
        return
    if record.reason != "socket-stop":
        failures.append(f"socket-stop reason is {record.reason!r}")
    if record.requested is not True:
        failures.append("a socket `stop` was recorded as not requested")
    if marker is None:
        failures.append("the ladder's runtime-stop.json is missing")
    elif marker.get("rung") != "socket" or marker.get("deliberate") is not True:
        failures.append(f"the marker is not the socket rung's: {marker!r}")
    elif marker.get("pid") != child.pid:
        failures.append(f"the marker names pid {marker.get('pid')}, not {child.pid}")


def _sigterm_scenario(config: Path, home: Path, failures: list[str]) -> None:
    """A signal nobody staged stays ``requested: false`` — the field's whole point."""
    from local_operator.session.runtime import journal

    env = _child_env(config, home, TERM_ID)
    _say("SIGTERM / spawn the real runtime, then signal it with no staged stop")
    child = _spawn(env)
    _wait_live(config, TERM_ID)
    os.kill(child.pid, signal.SIGTERM)
    print(f"sent SIGTERM to {child.pid}")
    exit_at = _wait_gone(child)
    print(f"runtime exited rc={child.returncode} at {exit_at:.3f}")

    record = journal.read_exit_record(_session_dir(config, TERM_ID))
    marker = _session_dir(config, TERM_ID) / "runtime-stop.json"
    _say("SIGTERM / the exit record")
    print(json.dumps(_json_file(_session_dir(config, TERM_ID) / "runtime-exit.json"), indent=2))

    if record is None:
        failures.append("the SIGTERM exit record is missing")
        return
    if record.reason != "SIGTERM":
        failures.append(f"SIGTERM reason is {record.reason!r}")
    if record.requested is not False:
        failures.append("an unattributed SIGTERM was recorded as a requested stop")
    if marker.exists():
        failures.append(f"a stop marker exists for the unattributed SIGTERM: {marker}")


def _resume_kill_scenario(config: Path, home: Path, failures: list[str]) -> None:
    """Run A idles out; the resume is killed. The row must NOT wear A's record."""
    env = _child_env(config, home, RESUME_ID)
    _say("RESUME→KILL / run A: spawn and let it idle out")
    run_a = _spawn(env)
    _wait_live(config, RESUME_ID)
    _wait_gone(run_a)
    record_a = _json_file(_session_dir(config, RESUME_ID) / "runtime-exit.json")
    print(json.dumps(record_a, indent=2))
    if record_a is None or record_a.get("reason") != "idle-exit":
        failures.append("run A did not leave the idle-exit record this cell needs")
        return

    _say("RESUME→KILL / run B: respawn the same session, then SIGKILL it")
    run_b = _spawn(env)
    _wait_live(config, RESUME_ID)
    os.kill(run_b.pid, signal.SIGKILL)
    _wait_gone(run_b)

    # THE FIRST LISTING AFTER A KILL STILL SEES THE DEAD RECORD and may render the
    # row ``stale`` — the scan removes the record as it reads it, so the STORED
    # shape (and the STOPPED cells) arrive on the NEXT pass. That is also why the
    # cell is listed twice: the second listing is the one an operator sees a
    # second later, and it is the one the fence has to hold on.
    _cli(env, "sessions", "--all")
    _say("RESUME→KILL / the listing must say unknown, not run A's idle-exit")
    code, out = _cli(env, "sessions", "--all")
    print(out)
    if code != 0:
        failures.append(f"`lop sessions --all` exited {code}")
    lines = [line for line in out.splitlines() if RESUME_ID in line]
    if not lines:
        failures.append("the resume→kill session is missing from the listing")
    elif "idle-exit" in lines[0]:
        failures.append(f"the killed run wore run A's record: {lines[0]!r}")
    elif lines[0].split()[:1] != ["stored"]:
        failures.append(f"the resume→kill row is not stored: {lines[0]!r}")
    json_code, json_out = _cli(env, "sessions", "--all", "--json")
    if json_code == 0:
        published = {row["session_id"]: row for row in json.loads(json_out)}
        row = published.get(RESUME_ID)
        if row is None:
            failures.append("the resume→kill session is missing from the JSON listing")
        elif row["stopped_at_s"] is not None or row["stop_reason"]:
            failures.append(f"a killed run borrowed its predecessor's record: {row!r}")


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="exit-record-repro-"))
    home = scratch / "home"
    config = scratch / "config"
    home.mkdir()
    config.mkdir()
    print(f"scratch root: {scratch}")

    for session_id, text in (
        (SESSION_ID, "seed"),
        (QUIET_ID, "quiet"),
        (STOP_ID, "stopped"),
        (TERM_ID, "signalled"),
        (RESUME_ID, "resume"),
    ):
        _seed(config, session_id, text)
    (config / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )

    failures: list[str] = []
    _idle_scenario(config, home, failures)
    _socket_stop_scenario(config, home, failures)
    _sigterm_scenario(config, home, failures)
    _resume_kill_scenario(config, home, failures)

    _say("RESULT")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: idle -> stored with age+reason; socket stop requested; SIGTERM not")
    print("      requested; a resumed-and-killed run reads unknown")
    print(f"scratch root kept for inspection: {scratch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
