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

from local_operator.session.runtime.reclaim import proc_environ_text  # noqa: E402
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
            warm = standby._POOL[0] if standby._POOL else None
            warmed = bool(warm is not None and warm.alive() and standby.adoption_possible())
            if warmed:
                break
            time.sleep(0.2)
    spare = standby._POOL[0] if standby._POOL else None

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


def _read_report(path: Path, wait_s: float) -> dict[str, Any] | None:
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


def _process_table() -> dict[int, tuple[int, str]]:
    """``pid -> (ppid, command)`` for every live process, one ``ps``."""
    # ``-eww`` IS NOT OPTIONAL: unlimited width. procps truncates the ``command``
    # column to the terminal width (80 columns when stdout is not a terminal), so on
    # Linux a long interpreter path is cut before the module name and a substring test
    # matches nothing — reading as "no processes" instead of as an error. macOS does
    # not truncate, which is why this is invisible on the host the whole PR was
    # measured on (agent review round 7; the same trap failed a CI shard in the suite).
    out = subprocess.run(
        ["ps", "-eww", "-o", "pid=,ppid=,command="], capture_output=True, text=True
    ).stdout
    table: dict[int, tuple[int, str]] = {}
    for line in out.splitlines():
        fields = line.split(None, 2)
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            table[int(fields[0])] = (int(fields[1]), fields[2] if len(fields) > 2 else "")
    return table


#: Read ONCE and patchable by name, as ``tools/group_reaper`` and ``secrets/peer``
#: do, so the arm this host does not have can still be exercised from it.
_IS_LINUX = sys.platform.startswith("linux")


def _census_with_environment() -> str:
    """``ps``'s whole table, with each process's environment, on EITHER platform.

    ``-E`` IS A BSD/macOS OPTION AND PROCPS HAS NO ``-E`` (the whole story is in
    ``reclaim.pid_environment``). On Linux ``ps -Eeww`` is an invalid option: non-zero,
    and NOTHING on stdout — which both readers below would read as "no writers", the
    comfortable zero ``_env_census_diagnostics`` exists to prevent, reached here by a
    dead instrument instead of a thin one. So Linux appends ``/proc/<pid>/environ`` to
    the same row (a file read, no fork), and the parsers below see the shape they
    already parse. ``-eww`` selects the whole table on both families; only the
    environment's source differs.
    """
    if not _IS_LINUX:
        return subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["ps", "-Eeww", "-o", "pid=,ppid=,command="], capture_output=True, text=True
        ).stdout
    base = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["ps", "-eww", "-o", "pid=,ppid=,command="], capture_output=True, text=True
    ).stdout
    rows: list[str] = []
    for line in base.splitlines():
        pid_text, _, rest = line.strip().partition(" ")
        env = proc_environ_text(int(pid_text)) if pid_text.isdigit() else ""
        rows.append(f"{pid_text} {rest} {env}".strip())
    return "\n".join(rows)


def _root_census(root: Path, exclude: set[int]) -> dict[int, str]:
    """``pid -> command tail`` for every live process whose ENVIRONMENT names ``root``.

    ATTRIBUTION IS BY ENVIRONMENT BECAUSE THAT IS WHERE THE ROOT IS (agent review round
    5, R5-2). A real runtime's argv is ``Local Operator [session] id=… -P -m
    local_operator.session.runtime.process --operator-fd N`` — the config root does not
    ride in argv at all, it rides in ``LOCAL_OPERATOR_CONFIG_DIR`` — so an argv-shaped
    predicate matched a probe built to be matched and missed every process that
    actually occurs: two adopted runtimes alive at ppid 1, recorded as
    ``descendants_recorded: []``.

    NOT UNIVERSAL, AND THAT MATTERS (round 7): ``ps -Ee`` prints an environment for the
    processes that EXPOSE one, which is not all of them — a Python or Node child does,
    ``/bin/sleep`` and stripped copies of it do not, measured in the same second from the
    same parent with the same environment. Where it is exposed this sees a process that
    inherited the root, including one re-parented to ppid 1 after its console exited; the
    caller therefore reads ``_env_census_diagnostics`` as well, so a census that cannot
    see environments fails the run instead of reporting "no writers".

    Shape-checked, not a bare substring (round 5, M-3): the needle is the variable NAME
    together with this run's session-unique value, so a command that merely mentions the
    path — a ``grep`` in someone else's census, a shell pipeline — is not a candidate.

    Cost: one ``ps`` over the whole table, measured at 59-75 ms for ~900 processes.
    """
    needle = f"LOCAL_OPERATOR_CONFIG_DIR={root}"
    out = _census_with_environment()
    found: dict[int, str] = {}
    for line in out.splitlines():
        fields = line.split(None, 1)
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        if pid in exclude or needle not in line:
            continue
        found[pid] = fields[1][-120:]
    return found


def _env_census_diagnostics() -> tuple[dict[str, int], list[int]]:
    """What the environment census can read, and the shape it must not miss.

    A DEAD INSTRUMENT MUST NOT RETURN A COMFORTABLE READING (agent review round 7). The
    environment is exposed per PROCESS, not by ``ps``: measured on this host, in the same
    second, from the same parent and with the same environment, Python and Node children
    show it while ``/bin/sleep``, a plain copy of ``/bin/sleep``, a signature-stripped
    copy of that and a copy of ``/bin/cat`` do not (the discriminator was not isolated —
    the dyld cache and the code signature were both ruled out). So the census cannot
    promise to see every future spawner, and its failure mode is silent in the direction
    that matters: an unreadable environment looks exactly like "no writers".

    Hence this: read how many processes the census can see an environment for at all, and
    name the shape it must not miss — a live process whose command names a standby or
    runtime module but which came back with NO environment. That is a candidate writer
    the census cannot attribute, and the caller warns and fails on it rather than
    reporting a comfortable zero.
    """
    out = _census_with_environment()
    seen = 0
    with_environment = 0
    unreadable: list[int] = []
    for line in out.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        seen += 1
        rest = fields[2] if len(fields) > 2 else ""
        # A real environment is many ``NAME=value`` words; two is a generous floor that a
        # command line of its own cannot plausibly reach.
        assignments = sum(
            1
            for word in rest.split()
            if "=" in word and word.split("=", 1)[0].replace("_", "").isalnum()
        )
        if assignments >= 2:
            with_environment += 1
        elif f"-m {STANDBY_MODULE}" in rest or f"-m {RUNTIME_MODULE}" in rest:
            unreadable.append(int(fields[0]))
    return {"processes": seen, "with_environment": with_environment}, unreadable


def _descendants(roots: set[int], table: dict[int, tuple[int, str]]) -> set[int]:
    """Every live pid strictly below ``roots``, however deep.

    Recursive on purpose (agent review round 4, R4-1): the cleanup's old predicate
    compared a candidate's ppid against a LIVE console, so once a console was killed
    every survivor was ppid 1 and unreachable by construction — which is how eight of
    eight runs reported ``root_removed: true`` while an orphaned runtime put the root
    back. A console's engage children are grandchildren or deeper of this script, so
    the walk cannot stop one level down.
    """
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _command) in table.items():
        kids.setdefault(ppid, []).append(pid)
    found: set[int] = set()
    stack = list(roots)
    while stack:
        for child in kids.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def _runtime_processes() -> list[tuple[int, int, float, str]]:
    """Live RUNTIME children (``-m ...runtime.process``), for the cleanup only.

    An engage spawns one, and a runtime still writing into the root is what made
    ``rmtree`` fail (QA round 3, Q4-3).
    """
    out = subprocess.run(
        ["ps", "-eww", "-o", "pid=,ppid=,rss=,command="], capture_output=True, text=True
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
        ["ps", "-eww", "-o", "pid=,ppid=,rss=,command="], capture_output=True, text=True
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

    # WHERE THE ROOT GOES. ``tempfile`` falls back to ``/tmp`` when ``TMPDIR`` is
    # absent, and an ``env -i`` invocation strips it — which is how 27 of this
    # session's own roots ended up in ``/tmp`` while every check for leftovers looked
    # in ``$TMPDIR`` and reported zero (agent review round 5, R5-1). So: prefer
    # ``TMPDIR``, fall back to the session's scratchpad, and never silently ``/tmp``.
    scratch = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD", "")
    parent = os.environ.get("TMPDIR") or scratch or None
    base = Path(tempfile.mkdtemp(prefix="lopsa-cap-", dir=parent or None))
    if parent in (None, "", "/tmp", "/private/tmp"):
        # Say it out loud rather than leaving a root in a directory shared with every
        # other session on this host and a claim that it was cleaned up.
        print(
            f"  (note: no TMPDIR/LOCAL_OPERATOR_SCRATCHPAD in this environment; the "
            f"isolated root is in a shared temp directory: {base})",
            flush=True,
        )
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
        reports_deadline = time.monotonic() + warm_wait + 120.0
        for index in range(len(slots)):
            report = _read_report(
                out_dir / f"console-{index}.json",
                max(0.0, reports_deadline - time.monotonic()),
            )
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
        # SAMPLED BEFORE THE ENGAGES (agent review round 4, R4-2): an adopted engage
        # consumes its console's spare and the console exits, so sampling afterwards
        # measured pids that no longer existed — no footprint at all in five of eight
        # runs — or a spare already replaced by its successor (the reviewer read
        # 163/167 MB and 170/172 MB against this body's 129-136 MB for that reason).
        before = [
            {"pid": pid, "rss_mb": rss, "footprint_mb": _footprint_mb(pid)}
            for pid, _ppid, rss, _tail in mine
        ]
        # Phase 2: everyone engages now that the count is known.
        for index in range(len(slots)):
            (out_dir / f"console-{index}.go").write_text("go")
        # ONE deadline for the set, not one per console (agent review round 4, R4-3):
        # N x 600 s is not a bound.
        engages_deadline = time.monotonic() + 600.0
        engages = [
            _read_report(
                out_dir / f"console-{index}.engage.json",
                max(0.0, engages_deadline - time.monotonic()),
            )
            for index in range(len(slots))
        ]
        row = {
            "consoles": len(slots),
            "mode": mode,
            # Recorded so a leftover can be found from the artefact rather than from
            # a guess about which temp directory this run used.
            "root": str(base),
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
        # ATTRIBUTION, AND WHY IT IS BY ENVIRONMENT (agent review round 5, R5-2). The
        # survivor that puts the root back is a RUNTIME, and its config root rides in
        # the ENVIRONMENT rather than in argv — see ``_root_census``. Two predicates,
        # both re-evaluated on every sweep:
        #   * every descendant of this run's consoles, however deep — attributable only
        #     while the consoles are alive, which is why the children go first;
        #   * every live process whose ENVIRONMENT names this run's session-unique root,
        #     which still finds an orphan after its parent is gone and after the console
        #     that forked it has exited.
        console_pids = {proc.pid for proc in procs}
        self_pid = os.getpid()

        def _mine() -> dict[int, str]:
            """Live writers and descendants of this run: pid -> command tail.

            The tail comes from the plain ``ps`` table (command only), and the census is
            only a FALLBACK — it reads ``ps -Eeww``, whose line is the command PLUS the
            environment, so letting it overwrite the table's entry recorded an environment
            fragment for every census-attributed pid, the reverse of what N-1 asked for
            (round 7). The census is therefore the fallback, and a pid it finds that the
            TREE cannot reach — the detached-runtime shape, ppid 1, which is the whole
            reason the census exists — is labelled with the route that found it instead
            of passing an environment fragment off as a command.
            """
            table = _process_table()
            found = {
                pid: cmd[-160:]
                for pid, (_ppid, cmd) in table.items()
                if pid in _descendants(console_pids, table)
            }
            for pid, tail in _root_census(base, exclude=console_pids | {self_pid}).items():
                found.setdefault(pid, "[detached: env-only] " + tail[-100:])
            for pid in list(found):
                if pid in console_pids or pid == self_pid:
                    del found[pid]
            return found

        recorded = dict(_mine())
        killed: dict[int, str] = {}
        for _round in range(10):  # a console can fork while its child is being killed
            live = _mine()
            if not live:
                break
            for pid, tail in sorted(live.items()):
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
                killed[pid] = tail
            time.sleep(0.4)
        # Children are quiescent, so the consoles can go now; then sweep again by the
        # environment alone, for anything a console forked as it died.
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
        for _round in range(6):
            live = _root_census(base, exclude={self_pid})
            if not live:
                break
            for pid, tail in live.items():
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
                killed[pid] = tail
            time.sleep(0.4)

        # IS THE INSTRUMENT WORKING? Asked before the writer set is believed (round 7):
        # an unreadable environment looks exactly like "no writers", so a census that
        # cannot read one is a failure to report, not a zero to trust.
        census_diagnostics, unreadable_writers = _env_census_diagnostics()
        census_blind = bool(unreadable_writers) or census_diagnostics["with_environment"] == 0
        if census_blind:
            print(
                f"  WARNING: the environment census may be blind — "
                f"{census_diagnostics['with_environment']}/{census_diagnostics['processes']} "
                f"processes exposed an environment, and these module-named processes "
                f"exposed none: {unreadable_writers}",
                flush=True,
            )

        # IS ANYTHING LEFT TO PUT IT BACK? Recorded, because "we removed it" is only a
        # sound claim when the writer set is empty at the moment of removal (R5-1).
        writers_at_removal = _root_census(base, exclude={self_pid})

        # LISTED BEFORE REMOVING (round 5, M-2): a listing taken after the removal can
        # be legitimately empty while the root is present again, which reads as "nothing
        # survived" beside a WARNING saying something did.
        listing_before_removal = (
            [str(item) for item in sorted(base.rglob("*"))[:8]] if base.exists() else []
        )

        # Remove, and PRINT THE REASON when it fails (round 4, R4-4): the failure that
        # actually lands here is ENOTEMPTY from a writer putting the tree back, and
        # ``ignore_errors`` threw away the only evidence of why.
        removal_error = ""
        for _round in range(4):
            try:
                shutil.rmtree(base)
            except FileNotFoundError:
                break
            except OSError as error:
                removal_error = f"{type(error).__name__}: {error}"
                print(f"  (removal attempt {_round + 1} failed: {removal_error})", flush=True)
                time.sleep(0.5)
                continue
            break

        # SETTLE, AND NAME A REAPPEARANCE RATHER THAN PASS IT OFF (round 5, R5-1). With
        # the writer set above empty a single read would be sound; the settle stays as
        # the second line of defence, and a root that comes back is recorded — the
        # mechanism is PRODUCT behaviour on the clean-exit path
        # (``journal.clear_boot_record`` → ``registry.unpublish`` → ``record_path`` /
        # ``run_dir`` doing ``mkdir(parents=True, exist_ok=True)`` unconditionally, then
        # unlinking the record), so it is disclosed rather than hidden by a retry.
        root_removed = False
        root_reappeared = False
        # A TRANSIENT failure here is not the same fact as "the removal failed" (round
        # 7, NIT): it used to land in ``removal_error`` and sit beside
        # ``root_removed: true`` as an apparent contradiction. Recorded separately, so
        # ``removal_error`` names only what a failed final state failed on.
        settle_removal_errors: list[str] = []
        clean_reads = 0
        for _round in range(6):
            time.sleep(1.0)
            if base.exists():
                root_reappeared = True
                clean_reads = 0
                try:
                    shutil.rmtree(base)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    detail = f"{type(error).__name__}: {error}"
                    settle_removal_errors.append(detail)
                    print(f"  (settle removal failed: {detail})", flush=True)
            else:
                clean_reads += 1
                if clean_reads >= 2:
                    root_removed = True
                    break
        # NIT(b) (round 7): a non-empty writer set was recorded and then ignored, which
        # is the shape where a reading exists and nothing acts on it. It IS the failure
        # this cleanup exists to prevent — a live writer is a process that can put the
        # root back after this script has exited — so it warns AND fails the run.
        cleanup_ok = root_removed and not writers_at_removal and not census_blind
        if writers_at_removal:
            print(
                f"  WARNING: {len(writers_at_removal)} writer(s) still name this root at "
                f"the moment of removal: "
                f"{[f'{pid}:{tail[:40]}' for pid, tail in sorted(writers_at_removal.items())]}",
                flush=True,
            )
        if row is not None:
            # In the artefact as well as the log, and each entry carries a command tail
            # so an audit is possible from the JSON (round 5, N-1).
            row["root_removed"] = root_removed
            row["cleanup_ok"] = cleanup_ok
            row["root_reappeared_after_removal"] = root_reappeared
            row["cleanup"] = {
                "descendants_recorded": {str(pid): tail for pid, tail in sorted(recorded.items())},
                "pids_killed": {str(pid): tail for pid, tail in sorted(killed.items())},
                "writers_at_removal": {
                    str(pid): tail for pid, tail in sorted(writers_at_removal.items())
                },
                "listing_before_removal": listing_before_removal,
                "removal_error": removal_error,
                "settle_removal_errors": settle_removal_errors,
                "census": census_diagnostics,
                "cleanup_ok": cleanup_ok,
            }
        if not root_removed:
            print(
                f"  WARNING: {base} survived the settle; listed before removal: "
                f"{listing_before_removal or [str(base)]}",
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
    if row.get("cleanup_ok") is False:
        # A surviving root, a surviving writer, or a census that could not read any
        # environment is a FAILURE of this script's own cleanup or of its instrument —
        # not a note (rounds 4, 5 and 7). Both root claims are settle-checked and the
        # writer set is acted on, so a failure is asserted here rather than reported as
        # a success for someone to discover in the temp directory later.
        print("  FAILED: the isolated root survived, or a writer survived, or the census was blind")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
