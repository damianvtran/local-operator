"""How many spares N consoles warm on ONE config root, and what the losers pay.

WHY A SEPARATE SCRIPT. ``bench_standby_engage.py`` measures one console's engage
with and without a spare. This measures the COUNT: N consoles, one config root,
and how many standby interpreters exist afterwards. It exists because the count
was wrong — QA measured three consoles on one root warming three spares at ~145 MB
each (review round 3), which is the memory cost the operator's constraint existed
to avoid — and because the fix for it cannot be "share one spare" (a spare is a
private descriptor to a child of exactly one console; serving another console's
engage needs a rendezvous path, which is escalation R1-1). The fix is a cap: one
``flock`` slot per root per kind of console, so the count is bounded no matter how
many consoles run, and a console that cannot take its slot spawns cold.

WHAT IT REPORTS

* ``consoles`` — how many warming consoles were started on the one root.
* ``spares`` — live ``[standby]`` interpreters after all of them warmed.
* ``spare_rss_mb`` — their total resident size.
* per console — whether it held a slot, whether it got a ready spare, and then its
  own REAL engage (``AttachedSession.cold`` → ``bind_runtime``), so "the losers go
  cold" is a measurement rather than an inference from ``adopted: false``.

TWO MODES. ``--mode capped`` is this head: each console asks for its slot.
``--mode legacy`` is the pre-cap behaviour, which the SAME harness can still
measure because the slot is not an argument the old ``enable_warming`` took — run
it with the old module checked out (``git checkout <pre-cap-sha> --
local_operator/session/runtime/standby.py``) to get the "before" row of the body's
table on the same machine, minutes apart, rather than quoting a different rig.

ISOLATION. A fresh ``HOME`` and ``LOCAL_OPERATOR_CONFIG_DIR`` per run, as
``AGENTS.md`` requires (the config dir alone is not enough: the cache and the agent
home derive from ``HOME``). The root is created once and ALL consoles share it,
which is the whole point of the measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from scripts import bench_tree  # noqa: E402

#: The console process. Phase 1 warms and reports; phase 2 waits for the bench's
#: ``go`` file, engages for real, reports that, and exits.
_CONSOLE = textwrap.dedent("""
    import asyncio, json, os, sys, time, uuid
    from pathlib import Path

    repo, root_arg, slot, out_arg, mode, budget = sys.argv[1:7]
    root, out = Path(root_arg), Path(out_arg)
    sys.path.insert(0, repo)
    os.environ.pop("LOP_RUNTIME_STANDBY_DISABLED", None)
    from local_operator.session.runtime import standby

    if mode == "legacy":
        standby.enable_warming(root)
        won = True
    else:
        standby.enable_warming(root, daemon=(slot == standby.SLOT_DAEMON))
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and slot not in standby._SLOTS:
            time.sleep(0.05)
        won = slot in standby._SLOTS

    warmed = False
    if won:
        deadline = time.monotonic() + float(budget)
        while time.monotonic() < deadline:
            warm = standby._WARM[0]
            warmed = bool(warm is not None and warm.alive() and standby.adoption_possible())
            if warmed:
                break
            time.sleep(0.2)
    spare = standby._WARM[0]
    out.write_text(json.dumps({
        "pid": os.getpid(),
        "slot": slot,
        "slot_held": bool(won),
        "warmed": bool(warmed),
        "spare_pid": spare.proc.pid if spare is not None else None,
    }))

    # Phase 2: the bench has counted the spares and says go.
    go = out.with_suffix(".go")
    while not go.exists():
        time.sleep(0.1)
    from scripts.bench_standby_engage import _engage

    result = asyncio.run(_engage(root, uuid.uuid4().hex[:12]))
    # ``adopted`` by the harness convention ``bench_standby_engage`` already uses:
    # the runtime that answered is THIS console's spare, not a fresh cold child.
    result["adopted"] = bool(spare is not None and result.get("runtime_pid") == spare.proc.pid)
    (out.parent / (out.stem + ".engage.json")).write_text(json.dumps(result))
""")


#: The module a spare is exec'd as, so ``ps`` can be read without a rendezvous path
#: (there is none — that is the R1-1 property).
STANDBY_MODULE = "local_operator.session.runtime.standby"


def _spare_processes() -> list[tuple[int, int, float, str]]:
    """Every live spare as ``(pid, ppid, rss_mb, tail-of-command)``.

    ``rss_mb`` is a float: it is rounded to a tenth, and annotating it as an int is
    what ``pyright`` refused (whole-tree, CI).
    """
    out = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,rss=,command="], capture_output=True, text=True
    ).stdout
    found: list[tuple[int, int, float, str]] = []
    for line in out.splitlines():
        fields = line.split(None, 3)
        if len(fields) < 4:
            continue
        pid, ppid, rss, command = fields
        # The ``-m`` form only: a CONSOLE's argv also contains the words
        # "session.runtime.standby" (its ``-c`` payload imports it), and counting a
        # console as a spare is exactly the mistake this script exists to avoid —
        # it read 4 spares for 2 slots before this line was restricted.
        if f"-m {STANDBY_MODULE}" in command:
            found.append((int(pid), int(ppid), round(int(rss) / 1024, 1), _tail(command)))
    return found


def _tail(command: str) -> str:
    """The words after the module, which is all a spare's row says about itself."""
    at = command.find(f"-m {STANDBY_MODULE}")
    return command[at:] if at >= 0 else ""


def _one_run(consoles: int, mode: str, warm_wait: float) -> dict[str, Any]:
    from local_operator.config import ConfigManager

    base = Path(tempfile.mkdtemp(prefix="lopsa-cap-"))
    root = base / ".local-operator"
    root.mkdir(parents=True)
    ConfigManager(config_dir=root).update_config({"hosting": "test", "model_name": "test-model"})
    out_dir = base / "reports"
    out_dir.mkdir()
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_", "HERDR_"))},
        "HOME": str(base),
        "LOCAL_OPERATOR_CONFIG_DIR": str(root),
        "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1",
        "LOCAL_OPERATOR_NO_DESKTOP_LAUNCH": "1",
    }
    # Which slot each console asks for. ``legacy`` asks for nothing (the old
    # ``enable_warming`` took no slot), so every console warms its own and the count
    # grows with N — that is the "before" this script exists to measure. Capped mode
    # gives the first console the daemon's slot and every other one the shared TUI
    # slot, so the count is 2 however large N gets.
    if mode == "legacy":
        slots = ["tui"] * consoles
    else:
        slots = (["daemon"] + ["tui"] * consoles)[:consoles]
    procs: list[subprocess.Popen[bytes]] = []
    try:
        for index, slot in enumerate(slots):
            report = out_dir / f"console-{index}.json"
            procs.append(
                subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                    [
                        sys.executable,
                        "-P",
                        "-c",
                        _CONSOLE,
                        str(HERE),
                        str(root),
                        slot,
                        str(report),
                        mode,
                        str(warm_wait),
                    ],
                    cwd=str(HERE),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            )
        reports = []
        deadline = time.monotonic() + warm_wait + 120.0
        for index in range(len(slots)):
            path = out_dir / f"console-{index}.json"
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(0.25)
            assert path.exists(), f"console {index} never reported"
            reports.append(json.loads(path.read_text()))
        spares = _spare_processes()
        console_pids = {proc.pid for proc in procs}
        # Phase 2: everyone engages now that the count is known.
        for index in range(len(slots)):
            (out_dir / f"console-{index}.go").write_text("go")
        engages = []
        deadline = time.monotonic() + 600.0
        for index in range(len(slots)):
            path = out_dir / f"console-{index}.engage.json"
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(0.25)
            engages.append(json.loads(path.read_text()) if path.exists() else None)
        row = {
            "consoles": len(slots),
            "mode": mode,
            "spares": len(spares),
            # Attributed, not just counted: a spare is this root's if it is a child
            # of a console this run started (a replacement after an engage is the
            # same slot's spare, which is why the count can exceed the slot count
            # without the cap being broken).
            "spares_of_this_run": sum(1 for item in spares if item[1] in console_pids),
            "spare_rss_mb": round(sum(item[2] for item in spares), 1),
            "spare_details": [
                {"pid": pid, "ppid": ppid, "rss_mb": rss, "mine": ppid in console_pids}
                for pid, ppid, rss, _tail in spares
            ],
            "load1": round(os.getloadavg()[0], 1),
            "per_console": [
                {
                    "slot": report["slot"],
                    "slot_held": report["slot_held"],
                    "warmed": report["warmed"],
                    "adopted": (engage or {}).get("adopted"),
                    "bind_ms": (engage or {}).get("bind_ms"),
                }
                for report, engage in zip(reports, engages)
            ],
        }
        adopted = [c["bind_ms"] for c in row["per_console"] if c["adopted"]]
        cold = [c["bind_ms"] for c in row["per_console"] if c["adopted"] is False]
        if adopted:
            row["adopted_bind_ms_median"] = round(statistics.median(adopted), 1)
        if cold:
            row["cold_bind_ms_median"] = round(statistics.median(cold), 1)
        return row
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
        for pid, _ppid, _rss, _tail in _spare_processes():
            # Only the spares whose parent we started; never by name.
            if any(proc.pid == _ppid for proc in procs):
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consoles", type=int, default=3)
    parser.add_argument("--mode", choices=("capped", "legacy"), default="capped")
    parser.add_argument("--warm-wait", type=float, default=60.0)
    parser.add_argument("--json", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--measured-tree", default="")
    args = parser.parse_args()

    tree = bench_tree.describe(args.measured_tree)
    row = _one_run(args.consoles, args.mode, args.warm_wait)
    row.update(label=args.label, tree=tree)
    print(
        f"  consoles={row['consoles']} mode={row['mode']} spares={row['spares']} "
        f"spare_rss_mb={row['spare_rss_mb']} load1={row['load1']}"
    )
    for console in row["per_console"]:
        print(
            f"    slot={console['slot']:<7} slot_held={console['slot_held']!s:<5} "
            f"warmed={console['warmed']!s:<5} adopted={console['adopted']} "
            f"bind_ms={console['bind_ms']}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(row, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
