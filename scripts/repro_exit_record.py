"""THE REAL PATH: a runtime idles out, and the store can say when and why.

WHAT THIS REPRODUCES, AND THE REPORT IT ANSWERS. On 2026-10-21 the operator asked
"when did it stop" about session ``709291bc6e49``. It had left through the
runtime's OWN residency policy — the 3 s idle drain — which logged one INFO line
and withdrew its boot record; the store showed the session as ``stored`` with no
time and no reason anywhere, indistinguishable from a SIGKILL. This script spawns
a real runtime process under a scratch config root, lets the real idle path end
it, and then reads the store the way an operator does:

  1. BEFORE: ``lop sessions --all`` lists the session LIVE (a real record).
  2. AFTER: the same command lists it STORED, with a STOPPED age and the reason
     ``idle-exit`` read from ``sessions/<id>/runtime-exit.json``.
  3. The record's ``at`` is within a second of the exit this script observed.
  4. No ``runtime-stop.json`` exists — nobody asked for this departure, so
     ``stop_requested`` is False (not a fabricated "the user stopped it").
  5. A second seeded session with NO exit record still reads ``—``/unknown: the
     absent artifact must not be dressed up as a time.

ISOLATION. A fresh HOME and a fresh ``LOCAL_OPERATOR_CONFIG_DIR``: AGENTS.md is
explicit that the config dir alone is not enough (the cache and agent home derive
from HOME independently), so both are redirected and the child inherits neither
the operator's ``CMUX_*``/``LOP_*`` families nor notification permission. Nothing
here can touch a live session.

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
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SESSION_ID = "exitrec00001"
QUIET_ID = "exitrec00002"

#: The idle window the child is given. The shipped value is 3.0 s; this is the
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


def _sessions(env: dict[str, str], *extra: str) -> tuple[int, str]:
    """The REAL command an operator runs, against the scratch root."""
    code, output = _run(
        [sys.executable, "-c", "import sys; from local_operator.cli import main; sys.exit(main())"]
        + ["sessions", "--all", *extra],
        env=env,
    )
    return code, output


def _seed(config: Path) -> None:
    """A session with one durable row, on the mock provider."""
    directory = config / "sessions" / SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "seed"}]}}\n',
        encoding="utf-8",
    )
    # A session with NO runtime and no exit record: the "unknown" control.
    quiet = config / "sessions" / QUIET_ID
    quiet.mkdir(parents=True, exist_ok=True)
    (quiet / "transcript.jsonl").write_text(
        '{"id": "seed2", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "quiet"}]}}\n',
        encoding="utf-8",
    )
    (config / "config.yml").write_text(
        "values:\n  hosting: test\n  model_name: mock\n  tool_approval_mode: auto\n",
        encoding="utf-8",
    )


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="exit-record-repro-"))
    home = scratch / "home"
    config = scratch / "config"
    home.mkdir()
    config.mkdir()
    print(f"scratch root: {scratch}")

    # THE CHILD'S ENVIRONMENT: the operator's families stripped, then the two
    # roots and the session this spawn serves set explicitly.
    child_env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    # THE SUITE'S OWN KILL SWITCHES, imported rather than spelled: a child spawned
    # from this script must not post a desktop notification or try to focus the
    # operator's app (its env is a scratch HOME, and a launch there would be a
    # second product instance, not a test).
    from local_operator.tui.notify import ENV_DISABLE, ENV_DISABLE_VALUE
    from local_operator.tui.resume_click import DESKTOP_LAUNCH_REFUSED_ENV

    child_env.update(
        {
            "HOME": str(home),
            "LOCAL_OPERATOR_CONFIG_DIR": str(config),
            "LOP_MOBILE_CHILD_CWD": str(config),
            "LOP_MOBILE_CHILD_RESUME": SESSION_ID,
            "LOP_SESSION_GRACE_S": GRACE_S,
            ENV_DISABLE: ENV_DISABLE_VALUE,
            DESKTOP_LAUNCH_REFUSED_ENV: "1",
        }
    )
    _seed(config)

    _say("STEP 0 — the store before any runtime: two stored sessions, no records")
    print(_sessions(child_env)[1])

    _say("spawn the real runtime process")
    child = subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"pid {child.pid}")

    # WAIT FOR THE RECORD FIRST so "it stopped" cannot be satisfied by a runtime
    # that never booted.
    from local_operator.session.runtime import registry

    deadline = time.time() + 30
    live = False
    while time.time() < deadline:
        for record, state in registry.scan(config):
            if getattr(record, "session_id", "") == SESSION_ID and state == "live":
                live = True
        if live:
            break
        time.sleep(0.05)
    if not live:
        child.kill()
        raise SystemExit("the runtime never published a live record")

    _say("BEFORE — the same command with the runtime LIVE")
    before_code, before_out = _sessions(child_env)
    print(before_out)
    if before_code != 0:
        raise SystemExit(f"sessions exited {before_code}")

    exit_at = None
    try:
        child.wait(timeout=120)
        exit_at = time.time()
    finally:
        if child.poll() is None:
            child.kill()
    print(f"runtime exited rc={child.returncode} at {exit_at:.3f}")

    _say("AFTER — `lop sessions --all` with the session STORED")
    after_code, after_out = _sessions(child_env)
    print(after_out)

    _say("THE ARTIFACT — sessions/<id>/runtime-exit.json")
    record_path = config / "sessions" / SESSION_ID / "runtime-exit.json"
    if not record_path.exists():
        raise SystemExit(f"no exit record at {record_path}")
    record = json.loads(record_path.read_text())
    print(json.dumps(record, indent=2))

    _say("THE NEGATIVE CONTROL — sessions/<quiet>/ (no runtime ever, no record)")
    quiet_record = config / "sessions" / QUIET_ID / "runtime-exit.json"
    print(f"{quiet_record} exists: {quiet_record.exists()}")

    from local_operator.session.runtime import journal

    parsed = journal.read_exit_record(config / "sessions" / SESSION_ID)
    stop_marker = config / "sessions" / SESSION_ID / "runtime-stop.json"

    failures: list[str] = []
    if after_code != 0:
        failures.append(f"`lop sessions --all` exited {after_code}")
    if "STOPPED" not in after_out or "STOP_REASON" not in after_out:
        failures.append("the stored listing has no STOPPED/STOP_REASON columns")
    stopped_lines = [line for line in after_out.splitlines() if "seed" in line]
    if not stopped_lines:
        failures.append("the stopped session is missing from the listing")
    elif "idle-exit" not in stopped_lines[0]:
        failures.append(f"the listing does not name idle-exit: {stopped_lines[0]!r}")

    # THE MACHINE-READ HALF: the published JSON row, which is what a script
    # consumes and where the STOPPED age is an exact epoch rather than a rendered
    # duration. Its key order is pinned by the extraction test; this asserts the
    # three values a caller can use.
    json_code, json_out = _sessions(child_env, "--json")
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
    if parsed is None:
        failures.append("the exit record does not parse")
    else:
        if parsed.reason != "idle-exit":
            failures.append(f"reason is {parsed.reason!r}, not 'idle-exit'")
        if parsed.detail != "no work, no viewer, no wake within 90s":
            failures.append(f"detail is {parsed.detail!r}")
        if parsed.requested:
            failures.append("a policy exit was reported as a requested stop")
        if exit_at is None or abs(parsed.at - exit_at) > 1.0:
            failures.append(f"record at={parsed.at:.3f} is not within 1s of exit {exit_at}")
        if parsed.pid != child.pid:
            failures.append(f"record pid {parsed.pid} != runtime pid {child.pid}")
    if stop_marker.exists():
        failures.append(f"a stop marker exists at {stop_marker}")
    quiet_lines = [line for line in after_out.splitlines() if "quiet" in line]
    if not quiet_lines:
        failures.append("the quiet control session is missing from the listing")
    elif "—" not in quiet_lines[0]:
        failures.append(f"the quiet control does not read unknown: {quiet_lines[0]!r}")

    _say("RESULT")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: live -> stored, runtime-exit.json names idle-exit, listing shows age + reason")
    print(f"scratch root kept for inspection: {scratch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
