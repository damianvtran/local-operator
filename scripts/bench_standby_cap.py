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
* ``spares`` — live spares that are CHILDREN of this run's consoles. The host-wide
  count and the foreign remainder are reported beside it and are never the number
  to quote: a sibling session's spare was counted as ours before agent review round
  3 (M3-2), which turned an uncontaminated "2 spares" into "4 spares / 424 MB".
* ``spare_footprint_mb_range`` — per-spare ``footprint(1)`` sampled before and
  after the engages, reported as a range. NOT a sum of ``ps rss``: RSS is neither
  additive across processes (they share mapped modules) nor stable within a run
  (one spare measured 178.1 MB, then 65.9 MB two minutes later on a 161.5 MB
  footprint).
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
import shutil
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
        # ASK, do not poll: the claim is idempotent for a slot this process already
        # holds, so this answers "did I get the slot?" synchronously whatever the
        # warming thread is doing — and it cannot be broken by the registry's key
        # shape, which a ``slot in _SLOTS`` poll silently was when the key became
        # ``(root, slot)`` (every console then reported slot_held=False).
        won = bool(standby._take_slot(root, slot))

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

    def _publish(path, payload):
        # ATOMIC (QA round 3, Q4-4): a plain write lets a reader observe a
        # half-written file, which crashed this bench in one of three N=5 runs.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)

    _publish(out, {
        "pid": os.getpid(),
        "slot": slot,
        "slot_held": bool(won),
        "warmed": bool(warmed),
        "spare_pid": spare.proc.pid if spare is not None else None,
    })

    # Phase 2: the bench has counted the spares and says go.
    go = out.with_suffix(".go")
    while not go.exists():
        time.sleep(0.1)
    from scripts.bench_standby_engage import _engage

    result = asyncio.run(_engage(root, uuid.uuid4().hex[:12]))
    # ``adopted`` by the harness convention ``bench_standby_engage`` already uses:
    # the runtime that answered is THIS console's spare, not a fresh cold child.
    result["adopted"] = bool(spare is not None and result.get("runtime_pid") == spare.proc.pid)
    _publish(out.parent / (out.stem + ".engage.json"), result)
""")


#: The module a spare is exec'd as, so ``ps`` can be read without a rendezvous path
#: (there is none — that is the R1-1 property).
STANDBY_MODULE = "local_operator.session.runtime.standby"

#: The module a runtime runs as (cold or adopted). Only the cleanup needs it: it is
#: what keeps the root from being written while this script removes it.
RUNTIME_MODULE = "local_operator.session.runtime.process"


def _read_report(path: Path, wait_s: float = 120.0) -> dict[str, Any] | None:
    """A report JSON, or ``None`` if it never became readable.

    Tolerant on purpose (QA round 3, Q4-4): the writers publish atomically now, and
    this side still does not assume it — a ``JSONDecodeError`` here used to abort a
    whole run on a file that was one ``os.replace`` from being complete.
    """
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                time.sleep(0.2)
                continue
            if isinstance(loaded, dict):
                return loaded
        time.sleep(0.25)
    return None


def _runtime_processes() -> list[tuple[int, int, float, str]]:
    """Live RUNTIME children (``-m ...runtime.process``), for the cleanup only.

    An engage spawns one, and a runtime still writing into the root is what made
    ``rmtree`` fail (QA round 3, Q4-3).
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
        if f"-m {RUNTIME_MODULE}" in command:
            found.append((int(pid), int(ppid), round(int(rss) / 1024, 1), _tail(command)))
    return found


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


def _footprint_mb(pid: int) -> float | None:
    """macOS ``footprint(1)`` for one pid, in MB — the honest per-process figure.

    WHY NOT THE SUM OF ``ps rss``: RSS is not additive across processes (they share
    the runtime's mapped modules) and it is not even stable within one run — the
    round-3 review measured one spare at 178.1 MB and then 65.9 MB two minutes
    later on a 161.5 MB footprint. So the per-spare footprint is sampled and
    reported as a RANGE, and no total is presented as the cost.
    """
    try:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True, timeout=25)
    except subprocess.TimeoutExpired:
        # LOUD, not silent (QA round 3, Q4-2): ``footprint`` walks a process's
        # mappings and can exceed its bound on a loaded host, which returned None in
        # three of four runs and read as "no data" with no reason given.
        print(f"    (footprint for {pid} timed out after 25 s; no footprint reported)", flush=True)
        return None
    except (OSError, subprocess.SubprocessError) as error:
        print(f"    (footprint for {pid} failed: {error})", flush=True)
        return None
    if out.returncode != 0:
        detail = (out.stderr or "").strip().splitlines()
        print(
            f"    (footprint for {pid} exited {out.returncode}: "
            f"{detail[0] if detail else 'no stderr'})",
            flush=True,
        )
        return None
    for line in out.stdout.splitlines():
        at = line.find("Footprint:")
        if at >= 0:
            parts = line[at:].split()
            for index, word in enumerate(parts):
                if word.endswith("KB") and index and parts[index - 1].replace(",", "").isdigit():
                    return round(int(parts[index - 1].replace(",", "")) / 1024, 1)
                if word.endswith("MB") and index and parts[index - 1].replace(",", "").isdigit():
                    return round(float(parts[index - 1].replace(",", "")), 1)
    return None


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
    row: dict[str, Any] | None = None
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
        for index in range(len(slots)):
            report = _read_report(out_dir / f"console-{index}.json", warm_wait + 120.0)
            assert report is not None, f"console {index} never reported a readable report"
            reports.append(report)
        spares = _spare_processes()
        console_pids = {proc.pid for proc in procs}
        # ATTRIBUTED before anything is reported (agent review round 3, M3-2): the
        # headline used to be every ``-m ...standby`` process on the machine, so one
        # sibling session's spare turned an uncontaminated "2 spares" into "4
        # spares / 424 MB". Only this run's consoles' children are ours.
        mine = [item for item in spares if item[1] in console_pids]
        foreign = [item for item in spares if item[1] not in console_pids]
        # Phase 2: everyone engages now that the count is known.
        for index in range(len(slots)):
            (out_dir / f"console-{index}.go").write_text("go")
        engages = [
            _read_report(out_dir / f"console-{index}.engage.json", 600.0)
            for index in range(len(slots))
        ]
        before = [
            {"pid": pid, "rss_mb": rss, "footprint_mb": _footprint_mb(pid)}
            for pid, _ppid, rss, _tail in mine
        ]
        row = {
            "consoles": len(slots),
            "mode": mode,
            # THE HEADLINE IS THIS RUN'S. ``host_wide_spares`` is kept for the
            # contamination check and is never the number to quote.
            "spares": len(mine),
            "host_wide_spares": len(spares),
            "foreign_spares": len(foreign),
            "spares_of_this_run": len(mine),
            "spare_footprint_mb": [
                {"pid": sample["pid"], "footprint_mb": sample["footprint_mb"]} for sample in before
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
        # Sampled AGAIN after the engages: a spare's footprint shrinks once its
        # imports settle, which is why the range is reported and not one reading.
        after = [
            {"pid": pid, "rss_mb": rss, "footprint_mb": _footprint_mb(pid)}
            for pid, _ppid, rss, _tail in _spare_processes()
            if _ppid in console_pids
        ]
        row["spare_samples"] = {"before": before, "after": after}
        footprints = [
            sample["footprint_mb"]
            for sample in before + after
            if sample.get("footprint_mb") is not None
        ]
        if footprints:
            row["spare_footprint_mb_range"] = [min(footprints), max(footprints)]
        adopted = [c["bind_ms"] for c in row["per_console"] if c["adopted"]]
        cold = [c["bind_ms"] for c in row["per_console"] if c["adopted"] is False]
        if adopted:
            row["adopted_bind_ms_median"] = round(statistics.median(adopted), 1)
        if cold:
            row["cold_bind_ms_median"] = round(statistics.median(cold), 1)
        return row
    finally:
        # CHILDREN FIRST, WHILE THEIR PARENTS ARE STILL ALIVE (QA round 3, Q4-3).
        # Phase 2's engages spawn real runtimes (``-m ...runtime.process``) as
        # children of the consoles, and killing the consoles first orphans them: a
        # runtime still writing into the root made ``rmtree`` fail ENOTEMPTY, and
        # ``ignore_errors=True`` hid it — measured as 6 of 8 runs leaving the root
        # behind even after the first attempt at this fix. So: collect the children
        # (runtimes and spares) attributable to THIS run's consoles, kill those, then
        # the consoles, sweep once more for a kill that raced a fork, remove the root,
        # and CHECK that it is gone.
        console_pids = {proc.pid for proc in procs}
        for _pass in range(2):
            ours = [
                item
                for item in _runtime_processes() + _spare_processes()
                if item[1] in console_pids
            ]
            if not ours:
                break
            for pid, _ppid, _rss, _tail in ours:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
            if _pass == 0:
                for proc in procs:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=30)
        shutil.rmtree(base, ignore_errors=True)
        # NOT silence: the removal is part of what this script claims to do, so a
        # survivor is printed and recorded rather than swallowed by ``ignore_errors``.
        root_removed = not base.exists()
        if row is not None:
            # In the artefact as well as the log: "this script removes its root" is a
            # claim, and a claim that only prints when it fails is not checkable.
            row["root_removed"] = root_removed
        if not root_removed:
            print(
                f"  WARNING: {base} was not removed; still present: "
                f"{[str(item) for item in sorted(base.rglob('*'))[:5]]}",
                flush=True,
            )


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
        f"(host-wide {row['host_wide_spares']}, foreign {row['foreign_spares']}) "
        f"footprint_mb={row.get('spare_footprint_mb_range')} load1={row['load1']}"
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
