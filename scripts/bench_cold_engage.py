"""Measure the cold engage — from "the user pressed send" to "the session is bound".

WHAT THIS MEASURES, AND WHY IT IS THE RIGHT THING
=================================================
The desktop app's first send in a NEW conversation sits on "starting…" until
``AttachedSession._ensure_bound()`` returns, so the number the user feels is the
wall time of:

    AttachedSession.cold()      install a cold frontend, no runtime yet
    attach_existing()           engage_runtime(): spawn the child, poll until
                                it publishes its record, then bind
      └─ _spawn_runtime()         fork/exec `python -m ...runtime.process`
      └─ child: interpreter + import graph
      └─ child: spawn_owned_session() -> create_session()  build the session
      └─ child: RuntimeServer.start_in_process()  bind + publish the record
      └─ parent poll grid         find_runtime_record() on a backoff grid
      └─ dial + frontend sync + history load
    admit_prompt()              hand the first message to the runtime

The parent's own added latency is small: ``scripts/bench_runtime_attach.py``
measured it at 1.6% of the total, so most of what this script reports is the
child. That is why it reports the phases SEPARATELY — a single aggregate cannot
distinguish "the child's import graph got faster" from "the parent's poll grid
got lucky", and those want different fixes.

The MCP marks exist because of one specific defect they falsify: a runtime that
declares an MCP server pays the MCP SDK import inside its pre-publication
window, so the wiring cannot ride the record as its own design intends.
``mcp_entered_ms_before_publish`` and ``mcp_exit_ms_after_publish`` are the two
numbers that show whether that is still true.

ISOLATION
=========
Every run gets a fresh ``HOME`` *and* a fresh ``LOCAL_OPERATOR_CONFIG_DIR``.
``AGENTS.md`` is explicit that the config dir alone is not enough — the model
catalogue cache derives its root from the home directory independently, so a run
with only the config dir redirected reads the operator's real cache and reports
a warm number as a cold one.

Every ``LOP_*`` and ``CMUX_*`` variable is stripped from the environment first
and the child is given synthetic ids only. That is not tidiness: an inherited
``CMUX_WORKSPACE_ID`` has already let a headless boot rename the operator's live
cmux workspaces, and an inherited ``LOP_*`` makes the run attach to a real
conversation. The child is also killed between runs, and the whole temporary
root is removed at the end — a benchmark that leaves its runtimes resident
measures the next variant against a machine it degraded itself.

USAGE
=====
    .venv/bin/python scripts/bench_cold_engage.py --pairs 5
    .venv/bin/python scripts/bench_cold_engage.py --pairs 5 --json out.json

``--pairs N`` runs N interleaved passes over the variants, so a load spike on
this host lands on every variant rather than on whichever one ran during it.
Report the MEDIAN, not the mean: the distribution has a long right tail.

The ``idle anchor`` column rescales every median by ``1146 / total(base)``,
where 1146 ms is the repo's own measured cold-engage figure on an idle machine
(``server/utils/desktop_sessions.py``). This host runs at a load average of
50-90 on 14 cores, which inflates absolute milliseconds by roughly 2.5-3.5x; the
SHARES are the durable result and the anchored column is how to read them as a
user would on an idle machine. It is a projection, not a measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# A benchmark under scripts/ must read the tree it lives in, not whatever tree
# the venv was installed from (AGENTS.md, "Every feature worktree owns its own
# venv"). Without this a benchmark run in a worktree silently measures main.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SITE_DIR = Path(__file__).resolve().parent / "cold_engage_site"
MCP_STUB = SITE_DIR / "mcp_stub.py"

#: The repo's own idle cold-engage figure, for the anchored projection.
IDLE_ANCHOR_MS = 1146.0

#: Every variable a `lop` parent (or a sibling agent session) may have exported
#: that the CHILD PRODUCT reads. Stripped before every run.
_STRIPPED_PREFIXES = ("LOP_", "CMUX_")

#: variant -> how the run's `.mcp.json` declares its one server. `base` declares
#: none; the two others separate two different costs of "this machine has an MCP
#: server configured" (see `_write_mcp_config`).
_VARIANTS = ("base", "mcpx", "mcp")


def _strip_inherited() -> None:
    for key in list(os.environ):
        if key.startswith(_STRIPPED_PREFIXES):
            del os.environ[key]


def _seed_config(config_dir: Path) -> None:
    """The minimum config a runtime needs to construct (test provider).

    Same fixture as ``bench_runtime_attach._seed_config`` and for the same
    reason: every real provider would put a network client construction, and
    possibly a live credential check, inside the span being measured.
    """
    from local_operator.config import ConfigManager

    manager = ConfigManager(config_dir=config_dir)
    manager.update_config({"hosting": "test", "model_name": "test-model"})


def _write_mcp_config(root: Path, variant: str) -> None:
    """A `.mcp.json` declaring one server, for the `mcp` and `mcpx` variants.

    Two transports, because they separate two different costs:

    * ``mcpx`` — HTTP on a closed port. Nothing is spawned and the refusal is
      immediate, so what is left is the SDK import and the config parse: the
      cost ANY declared server pays before the 250 ms startup gate can help.
    * ``mcp`` — a stdio stub that never answers ``initialize``, which adds the
      spawn and the handshake attempt on top.
    """
    if variant == "mcpx":
        entry: dict[str, Any] = {"type": "http", "url": "http://127.0.0.1:1/mcp"}
    else:
        entry = {"command": sys.executable, "args": ["-u", str(MCP_STUB)]}
    payload = {"mcpServers": {"bench-slow": entry}}
    (root / ".mcp.json").write_text(json.dumps(payload), encoding="utf-8")


class Timeline:
    """Wall-clock marks for the parent, plus a reader for the child's marks."""

    def __init__(self) -> None:
        self.marks: list[tuple[str, int, float, Any]] = []

    def mark(self, label: str, detail: Any = None) -> None:
        self.marks.append((label, time.time_ns(), time.perf_counter(), detail))

    def wall(self, label: str) -> int | None:
        for name, wall, _perf, _detail in self.marks:
            if name == label:
                return wall
        return None


def _read_child_marks(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _child_wall(marks: list[dict[str, Any]], label_suffix: str) -> int | None:
    for entry in marks:
        if entry.get("label", "").endswith(label_suffix):
            return entry.get("wall_ns")
    return None


def _wait_for_child_label(path: Path, label: str, timeout: float) -> bool:
    """Wait for one child mark to land, so a *missing* mark is not read as time.

    Needed because the change this benchmark exists to measure leaves the MCP
    wiring running PAST the end of the parent's engage — which is the point of
    it. Killing the child at that instant would delete the very mark
    (``child.wire_mcp.exit``) that proves when the wiring ran, and an absent
    mark is indistinguishable from "it never happened". Nothing measured is
    affected: every parent span was already recorded, and the child's own spans
    are its perf_counter pairs, which this wait sits outside.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for entry in _read_child_marks(path):
            if entry.get("label") == label:
                return True
        time.sleep(0.1)
    return False


def _child_span_ms(marks: list[dict[str, Any]], label: str) -> float | None:
    start = end = None
    for entry in marks:
        if entry.get("label") == label + ".enter":
            start = entry.get("perf_s")
        elif entry.get("label") == label + ".exit" and start is not None:
            end = entry.get("perf_s")
            break
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 1)


class RecordWatcher(threading.Thread):
    """Samples the record directory independently of the parent's poll grid.

    A busy-poll thread rather than an inotify/fsevents subscription: the
    publication is an ``os.replace`` of ``<pid>.json``, and a 1 ms sample bounds
    the detection error well below every other span here.
    """

    def __init__(self, run_dir: Path) -> None:
        super().__init__(daemon=True)
        self.run_dir = run_dir
        self.seen: list[tuple[str, int]] = []
        self._halt = threading.Event()
        self._baseline: set[str] = set()
        self.ready = threading.Event()

    def run(self) -> None:
        try:
            self._baseline = {p.name for p in self.run_dir.iterdir()}
        except OSError:
            self._baseline = set()
        self.ready.set()
        while not self._halt.is_set():
            try:
                for path in self.run_dir.iterdir():
                    if path.name in self._baseline or not path.name.endswith(".json"):
                        continue
                    self._baseline.add(path.name)
                    self.seen.append((path.name, time.time_ns()))
            except OSError:
                pass
            time.sleep(0.001)

    def stop(self) -> None:
        self._halt.set()


#: The timeline the parent's wrappers write into. Replaced per run, because the
#: wrappers are installed once against a stable indirection rather than
#: re-installed (and re-wrapped) for every run.
_CURRENT: list[Timeline] = []


def _mark(label: str, detail: Any = None) -> None:
    if _CURRENT:
        _CURRENT[0].mark(label, detail)


def _wrap_attr(target: Any, attr: str, label: str) -> None:
    """Wrap an attribute (module function or classmethod) so its span is timed.

    Safe against the function-local imports the runtime uses, because those
    re-read the module attribute at call time, so the wrapper is what they see.
    """
    import functools
    import inspect

    static = inspect.getattr_static(target, attr, None)
    if static is None:
        return
    is_classmethod = isinstance(static, classmethod)
    original = getattr(target, attr, None)
    if original is None or getattr(original, "_lo_bench_wrapped", False):
        return

    def _make(fn):  # type: ignore[no-untyped-def]
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                _mark(label + ".enter")
                try:
                    return await fn(*args, **kwargs)
                finally:
                    _mark(label + ".exit")

            async_wrapper._lo_bench_wrapped = True  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            _mark(label + ".enter")
            try:
                return fn(*args, **kwargs)
            finally:
                _mark(label + ".exit")

        wrapper._lo_bench_wrapped = True  # type: ignore[attr-defined]
        return wrapper

    if is_classmethod:
        setattr(target, attr, classmethod(_make(static.__func__)))
    else:
        setattr(target, attr, _make(original))


def _instrument_parent() -> None:
    """Wrap every parent-side phase boundary in this process's modules."""
    from local_operator.mobile import attach_client
    from local_operator.session import attached
    from local_operator.session.runtime import launch

    _wrap_attr(launch, "_spawn_runtime", "parent.popen")
    _wrap_attr(launch, "_lease_holder", "parent.lease_probe")
    _wrap_attr(attached.AttachedSession, "cold", "parent.cold")
    _wrap_attr(attached.AttachedSession, "attach_existing", "parent.attach_existing")
    _wrap_attr(attached.AttachedSession, "bind_runtime", "parent.bind_runtime")
    _wrap_attr(attached.AttachedSession, "admit_prompt", "parent.admit_prompt")
    _wrap_attr(attached.AttachedSession, "_dial", "parent.dial")
    _wrap_attr(attached.AttachedSession, "_await_frontend", "parent.frontend_sync")
    _wrap_attr(attached.AttachedSession, "_load_frontend_history", "parent.load_history")
    _wrap_attr(attached.AttachedSession, "_ensure_bound", "parent.ensure_bound")

    # The record scan gets its own wrapper rather than the generic one: whether
    # a scan FOUND the record is what separates the parent's own poll-grid dead
    # time from the child's construction. `engage_runtime` imports this name
    # function-locally on every pass, so the module attribute is what it calls.
    original_find = attach_client.find_runtime_record
    if not getattr(original_find, "_lo_bench_wrapped", False):

        def find_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            _mark("parent.find_record.enter")
            result = original_find(*args, **kwargs)
            found = None
            if isinstance(result, tuple) and result:
                found = result[0] is not None
            _mark("parent.find_record.exit", {"found": found})
            return result

        find_wrapper._lo_bench_wrapped = True  # type: ignore[attr-defined]
        attach_client.find_runtime_record = find_wrapper  # type: ignore[assignment]


async def _one_run(variant: str, index: int) -> dict[str, Any]:
    """One cold engage in a fresh HOME + config dir, with every mark recorded."""
    from local_operator.mobile.attach_client import find_runtime_record
    from local_operator.session.attached import AttachedSession
    from local_operator.session.runtime.registry import run_dir

    root = Path(tempfile.mkdtemp(prefix=f"lop-bench-{variant}-"))
    config_dir = root / ".local-operator"
    config_dir.mkdir(parents=True, exist_ok=True)
    _seed_config(config_dir)
    if variant != "base":
        _write_mcp_config(root, variant)

    saved = {k: os.environ.get(k) for k in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR", "PYTHONPATH")}
    _strip_inherited()
    os.environ["HOME"] = str(root)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    # The child's instrument. PYTHONPATH survives `-P` (which strips only the
    # implicit cwd entry), so `sitecustomize` import time is the earliest point
    # we can mark inside the child.
    os.environ["PYTHONPATH"] = str(SITE_DIR)
    os.environ["LOP_BENCH_ROLE"] = "child"
    os.environ["LOP_BENCH_TIMELINE"] = str(root / "child.jsonl")

    session_id = f"bench-{uuid.uuid4().hex[:12]}"
    cwd = str(root)
    timeline = Timeline()
    _CURRENT[:] = [timeline]
    watcher = RecordWatcher(run_dir(config_dir))
    watcher.start()
    watcher.ready.wait(2.0)
    timeline.mark("parent.watcher_ready")

    # Sampled per run because this benchmark is routinely run on a shared host
    # where the load average moves by 5x inside one campaign, and a median with
    # no load beside it cannot be compared against another campaign's median.
    loadavg = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    result: dict[str, Any] = {
        "variant": variant,
        "run": index,
        "session_id": session_id,
        "loadavg": round(loadavg, 1),
    }
    record = None
    remote = None
    try:

        async def _no_takeover() -> None:
            raise RuntimeError("benchmark has no takeover")

        timeline.mark("parent.cold_begin")
        remote = await AttachedSession.cold(
            session_id,
            config_dir=config_dir,
            cwd=cwd,
            takeover_factory=_no_takeover,
            surface="desktop",
        )
        timeline.mark("parent.cold_end")

        bound = await remote.attach_existing()
        timeline.mark("parent.attach_existing_end", {"bound": bound})

        timeline.mark("parent.send_begin")
        await remote.bind_runtime()
        timeline.mark("parent.bind_end")

        ack_error = ""
        try:
            ack = await remote.admit_prompt(
                "bench: one word", command_id=uuid.uuid4().hex, images=[]
            )
            timeline.mark("parent.ack_end", {"ack": list(ack) if ack else None})
        except Exception as exc:  # noqa: BLE001 — the bind number is the one that matters
            ack_error = f"{type(exc).__name__}: {exc}"
            timeline.mark("parent.ack_end", {"error": ack_error})

        timeline.mark("parent.done")
        record, _owner = await asyncio.to_thread(find_runtime_record, config_dir, session_id)
    finally:
        time.sleep(0.05)
        watcher.stop()
        watcher.join(timeout=1.0)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # Dispose the viewer BEFORE the loop closes. Every mark above is already
    # recorded, so nothing measured is affected; without this the session's
    # own recovery task wakes after ``asyncio.run`` has shut its executor down
    # and dies with "Executor shutdown has been called" — noise that reads like
    # a failed run and buries the exit code of the last variant.
    if remote is not None:
        try:
            await remote.dispose()
        except Exception:  # noqa: BLE001 — teardown noise is not a measurement
            pass

    # Let the wiring settle BEFORE the kill, if this variant has one, so its
    # marks land (see _wait_for_child_label).
    if variant != "base":
        _wait_for_child_label(root / "child.jsonl", "child.wire_mcp.exit", timeout=120.0)

    # Kill the child before the next run, so runs do not measure a machine they
    # degraded themselves (and so the host stays under pressure, not over it).
    if record is not None:
        pid = getattr(record, "pid", None)
        if isinstance(pid, int) and pid > 0:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
            for _ in range(200):
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.01)
            try:
                os.kill(pid, 9)
            except OSError:
                pass
    # The MCP stub is a grandchild; sweep anything matching this run's root.
    os.system(f"pkill -f {root} >/dev/null 2>&1")  # noqa: S605 — fixed argv, tempdir path

    marks = _read_child_marks(root / "child.jsonl")
    result["parent"] = [
        {"label": label, "wall_ns": wall, "perf_s": perf, "detail": detail}
        for label, wall, perf, detail in timeline.marks
    ]
    result["child"] = marks
    result["record_seen"] = watcher.seen
    result["found"] = record is not None
    result["root"] = str(root)
    result["derived"] = _derive(timeline, marks, watcher, result)
    return result


def _derive(
    timeline: Timeline, marks: list[dict[str, Any]], watcher: RecordWatcher, result: dict[str, Any]
) -> dict[str, Any]:
    """Phase buckets and the MCP pre/post-publication marks (milliseconds)."""

    def span(a: str, b: str) -> float | None:
        aw, bw = timeline.wall(a), timeline.wall(b)
        if aw is None or bw is None:
            return None
        return round((bw - aw) / 1e6, 1)

    def parent_span(label: str) -> float | None:
        start = end = None
        for name, _wall, perf, _detail in timeline.marks:
            if name == label + ".enter" and start is None:
                start = perf
            elif name == label + ".exit" and start is not None:
                end = perf
                break
        if start is None or end is None:
            return None
        return round((end - start) * 1000.0, 1)

    derived: dict[str, Any] = {}
    derived["total_ms"] = span("parent.cold_begin", "parent.done")
    derived["cold_ms"] = parent_span("parent.cold")
    derived["attach_existing_ms"] = parent_span("parent.attach_existing")
    derived["ensure_bound_ms"] = parent_span("parent.ensure_bound")
    derived["bind_ms"] = span("parent.send_begin", "parent.bind_end")
    derived["send_to_ack_ms"] = span("parent.send_begin", "parent.ack_end")
    derived["popen_ms"] = parent_span("parent.popen")
    derived["dial_ms"] = parent_span("parent.dial")
    derived["frontend_sync_ms"] = parent_span("parent.frontend_sync")
    derived["load_history_ms"] = parent_span("parent.load_history")

    send_ns = timeline.wall("parent.send_begin")
    spawn_ns = timeline.wall("parent.popen.enter")
    bind_ns = timeline.wall("parent.bind_end")
    if spawn_ns and send_ns:
        derived["engage_pre_spawn_ms"] = round((spawn_ns - send_ns) / 1e6, 1)
    if bind_ns and spawn_ns:
        derived["spawn_to_bind_ms"] = round((bind_ns - spawn_ns) / 1e6, 1)

    child_ready = _child_wall(marks, "child.record_write.enter") if marks else None
    if child_ready and spawn_ns:
        derived["child_total_to_publish_ms"] = round((child_ready - spawn_ns) / 1e6, 1)
    record_file = watcher.seen[0][1] if watcher.seen else None
    if record_file and child_ready:
        derived["publish_signal_cross_check_ms"] = round((record_file - child_ready) / 1e6, 1)

    # The parent's own poll-grid dead time: the gap between the child's own
    # publication instant and the first find_runtime_record that returned it.
    found_ns = None
    for name, wall, _perf, detail in timeline.marks:
        if name == "parent.find_record.exit" and isinstance(detail, dict):
            if detail.get("found"):
                found_ns = wall
                break
    if record_file and found_ns:
        derived["poll_grid_dead_ms"] = round((found_ns - record_file) / 1e6, 1)

    # Child internal spans, straight from its own perf_counter.
    for label, key in (
        ("child.sitecustomize", "child_sitecustomize_ms"),
        ("child.amain", "child_amain_ms"),
        ("child.spawn_owned_session", "child_spawn_owned_ms"),
        ("child.create_session", "child_create_session_ms"),
        ("child.prepare", "child_prepare_ms"),
        ("child.store_maintenance", "child_store_maintenance_ms"),
        ("child.drain_inbox", "child_drain_inbox_ms"),
        ("child.start_in_process", "child_start_in_process_ms"),
        ("child.serve", "child_serve_ms"),
        ("child.mcp_gate_open", "child_mcp_gate_open_ms"),
        ("child.RecordPublisher_init", "child_publisher_init_ms"),
        ("child.record_write", "child_record_write_ms"),
        ("child.wire_mcp", "child_wire_mcp_ms"),
        ("child.lease_acquire", "child_lease_acquire_ms"),
    ):
        value = _child_span_ms(marks, label)
        if value is not None:
            derived[key] = value

    # Interpreter start -> amain: runpy, the composition-root import and main()'s
    # own log setup, i.e. everything before the first phase the child marks.
    site = amain = None
    for entry in marks:
        if entry["label"] == "child.sitecustomize":
            site = entry["wall_ns"]
        if entry["label"] == "child.amain.enter" and site is not None and amain is None:
            amain = entry["wall_ns"]
    if site and amain:
        derived["child_boot_to_amain_ms"] = round((amain - site) / 1e6, 1)

    # ---- the MCP gate: the two marks that falsify "wiring rides the record" --
    mcp_enter = _child_wall(marks, "child.wire_mcp.enter")
    mcp_exit = _child_wall(marks, "child.wire_mcp.exit")
    publish_ns = _child_wall(marks, "child.record_write.enter")
    gate_ns = _child_wall(marks, "child.mcp_gate_open.enter")
    if mcp_enter and publish_ns:
        derived["mcp_entered_ms_before_publish"] = round((publish_ns - mcp_enter) / 1e6, 1)
    if mcp_exit and publish_ns:
        derived["mcp_exit_ms_after_publish"] = round((mcp_exit - publish_ns) / 1e6, 1)
    if gate_ns and publish_ns:
        derived["mcp_gate_ms_after_publish"] = round((gate_ns - publish_ns) / 1e6, 1)
    if mcp_enter and gate_ns:
        derived["mcp_start_ms_after_gate"] = round((mcp_enter - gate_ns) / 1e6, 1)

    # ---- the ordered chain, as gaps between consecutive boundaries ----------
    chain = [
        ("parent.popen.exit", "child.sitecustomize", "g_exec_and_interpreter_ms"),
        ("child.sitecustomize", "child.amain.enter", "g_runpy_and_moduleimport_ms"),
        ("child.amain.enter", "child.spawn_owned_session.enter", "g_amain_pre_construction_ms"),
        ("child.spawn_owned_session.enter", "child.create_session.enter", "g_preamble_ms"),
        ("child.create_session.enter", "child.prepare.enter", "g_create_session_preamble_ms"),
        ("child.prepare.enter", "child.prepare.exit", "g_prepare_ms"),
        ("child.prepare.exit", "child.create_session.exit", "g_session_object_ms"),
        ("child.create_session.exit", "child.record_write.enter", "g_drain_serve_publish_ms"),
    ]
    for start, end, key in chain:
        end_ns = _child_wall(marks, end)
        start_ns = _child_wall(marks, start) or timeline.wall(start)
        if start_ns and end_ns:
            derived[key] = round((end_ns - start_ns) / 1e6, 1)

    return derived


def _summarize(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [
        r["derived"][key]
        for r in rows
        if key in r.get("derived", {}) and r["derived"][key] is not None
    ]
    if not values:
        return {}
    return {
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "n": len(values),
    }


#: The buckets the summary prints, in cold-engage order. Split by bucket so a
#: change can be shown to move the phase it claims to move.
_BUCKETS: list[tuple[str, list[str]]] = [
    (
        "parent: before the child",
        ["cold_ms", "engage_pre_spawn_ms", "popen_ms"],
    ),
    (
        "child: interpreter + import graph",
        [
            "g_exec_and_interpreter_ms",
            "child_boot_to_amain_ms",
            "g_runpy_and_moduleimport_ms",
            "g_amain_pre_construction_ms",
        ],
    ),
    (
        "child: composition root + session construction",
        [
            "child_spawn_owned_ms",
            "g_create_session_preamble_ms",
            "child_prepare_ms",
            "g_session_object_ms",
            "child_create_session_ms",
            "child_lease_acquire_ms",
            "child_store_maintenance_ms",
        ],
    ),
    (
        "child: drain -> serve -> publish",
        [
            "g_drain_serve_publish_ms",
            "child_drain_inbox_ms",
            "child_start_in_process_ms",
            "child_serve_ms",
            "child_publisher_init_ms",
            "child_record_write_ms",
        ],
    ),
    (
        "child: MCP wiring (the gated work)",
        [
            "child_wire_mcp_ms",
            "mcp_entered_ms_before_publish",
            "mcp_gate_ms_after_publish",
            "mcp_start_ms_after_gate",
            "mcp_exit_ms_after_publish",
        ],
    ),
    (
        "parent: publication -> bound session",
        [
            "poll_grid_dead_ms",
            "spawn_to_bind_ms",
            "dial_ms",
            "frontend_sync_ms",
            "load_history_ms",
            "bind_ms",
            "send_to_ack_ms",
        ],
    ),
    (
        "totals",
        [
            "total_ms",
            "child_total_to_publish_ms",
            "attach_existing_ms",
            "ensure_bound_ms",
        ],
    ),
]


def _print_paired(rows: list[dict[str, Any]], variants: list[str]) -> None:
    """The load-matched statistic: every variant minus `base` IN THE SAME PASS.

    Absolute milliseconds on a shared, heavily loaded host move by 4x between
    passes (observed range for one variant in one campaign: 5.5 s to 23.5 s),
    so a comparison of two medians taken minutes apart cannot separate the
    change from the load. The paired delta can: the variants in one pass run
    seconds apart, so they see the same machine.
    """
    if "base" not in variants:
        return
    base_by_pass = {r["run"]: r for r in rows if r["variant"] == "base"}
    print("\n--- paired deltas vs `base`, same pass (median over passes) ---")
    for variant in variants:
        if variant == "base":
            continue
        deltas: dict[str, list[float]] = {
            "total_ms": [],
            "child_total_to_publish_ms": [],
            "dial+sync+hist_ms": [],
        }
        for row in rows:
            if row["variant"] != variant:
                continue
            base = base_by_pass.get(row["run"])
            if base is None:
                continue
            for key in ("total_ms", "child_total_to_publish_ms"):
                a, b = row["derived"].get(key), base["derived"].get(key)
                if a is not None and b is not None:
                    deltas[key].append(a - b)
            tail = 0.0
            tail_base = 0.0
            complete = True
            for key in ("dial_ms", "frontend_sync_ms", "load_history_ms"):
                a, b = row["derived"].get(key), base["derived"].get(key)
                if a is None or b is None:
                    complete = False
                    break
                tail += a
                tail_base += b
            if complete:
                deltas["dial+sync+hist_ms"].append(round(tail - tail_base, 1))

        def med(values: list[float]) -> str:
            if not values:
                return "n/a"
            median = round(statistics.median(values), 1)
            low = round(min(values), 1)
            high = round(max(values), 1)
            return f"{median:>9} (min {low}, max {high}, n={len(values)})"

        print(f"  {variant} - base:")
        for key in ("child_total_to_publish_ms", "total_ms", "dial+sync+hist_ms"):
            print(f"     {key:<28} {med(deltas[key])}")

    # The load-invariant structural fact: is the MCP wiring finishing BEFORE the
    # record exists (the defect) or AFTER it (the gate holding)?
    for variant in variants:
        if variant == "base":
            continue
        values = [
            r["derived"]["mcp_exit_ms_after_publish"]
            for r in rows
            if r["variant"] == variant and "mcp_exit_ms_after_publish" in r["derived"]
        ]
        if not values:
            print(f"\n  {variant}: no mcp_exit_ms_after_publish reading")
            continue
        positive = sum(1 for v in values if v > 0)
        print(
            f"\n  {variant}: mcp_exit_ms_after_publish positive in {positive}/{len(values)} "
            f"runs (before the change this is 0/N by construction: the wiring ran to "
            f"completion inside the pre-publication window)"
        )


def _print_summary(rows: list[dict[str, Any]], variants: list[str]) -> dict[str, Any]:
    """Print the per-bucket medians for every variant, plus the anchored column."""
    keys = [key for _title, group in _BUCKETS for key in group]
    summary: dict[str, Any] = {"variants": {}}
    base_total = None
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        stats = {key: _summarize(subset, key) for key in keys}
        summary["variants"][variant] = stats
        if variant == "base" and stats.get("total_ms"):
            base_total = stats["total_ms"]["median"]
    scale = (IDLE_ANCHOR_MS / base_total) if base_total else 1.0

    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        stats = summary["variants"][variant]
        print(
            f"\n=== variant={variant}  (n={len(subset)}, record found "
            f"{sum(1 for r in subset if r['found'])}/{len(subset)}) ==="
        )
        for title, group in _BUCKETS:
            print(f"  -- {title}")
            for key in group:
                stat = stats.get(key)
                if not stat:
                    continue
                anchor = round(stat["median"] * scale, 1)
                print(
                    f"     {key:<34} median={stat['median']:>9} "
                    f"min={stat['min']:>9} max={stat['max']:>9} n={stat['n']:<2} "
                    f"@1146ms≈{anchor:>7}"
                )
    print(
        f"\n  (the '@1146ms' column is every median rescaled by "
        f"{IDLE_ANCHOR_MS:.0f}/{base_total:.0f} = {scale:.3f}; a projection, "
        f"not a measurement)"
    )
    summary["anchor_scale"] = round(scale, 4)
    return summary


def _print_thresholds(summary: dict[str, Any]) -> None:
    """The two §5A falsification thresholds, as numbers rather than prose.

    mechanism: `child_total_to_publish` must drop by >=90% of the MCP span, and
    `mcp_exit_ms_after_publish` must be clearly positive.
    delivery: `total_ms` must fall by >=70% of the publication saving.
    """
    print("\n--- §5A falsification thresholds (base vs mcpx) ---")
    base = summary["variants"].get("base", {})
    mcpx = summary["variants"].get("mcpx", {})

    def median(stat: dict[str, Any], key: str) -> float | None:
        entry = stat.get(key)
        return entry["median"] if entry else None

    mcpx_mcp = median(mcpx, "child_wire_mcp_ms")
    mcpx_total = median(mcpx, "total_ms")
    # The MCP span never runs inside the pre-publication window once the gate
    # takes effect, so `child_wire_mcp_ms` still measures it but
    # `child_total_to_publish` is where the saving shows up.
    print(f"  mcpx mcp span (child_wire_mcp_ms)      = {mcpx_mcp} ms")
    print(
        f"  mcpx child_total_to_publish            = {median(mcpx, 'child_total_to_publish_ms')} ms"
    )
    print(
        f"  mcpx mcp_exit_ms_after_publish         = {median(mcpx, 'mcp_exit_ms_after_publish')} ms"
    )
    print(
        f"  mcpx mcp_gate_ms_after_publish         = {median(mcpx, 'mcp_gate_ms_after_publish')} ms"
    )
    print(
        f"  mcpx mcp_start_ms_after_gate           = {median(mcpx, 'mcp_start_ms_after_gate')} ms"
    )
    print(f"  mcpx total_ms                          = {mcpx_total} ms")
    print(f"  base total_ms                          = {median(base, 'total_ms')} ms")
    print(
        "  mechanism: mcp_exit_ms_after_publish must be POSITIVE and\n"
        "             child_total_to_publish must lose >=90% of the MCP span.\n"
        "  delivery:  total_ms(mcpx) must lose >=70% of the publication saving\n"
        "             (byte-for-byte: base total - base child_to_publish - mcp span)."
    )


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=int,
        default=5,
        help="interleaved passes over every variant (default 5)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="base,mcpx,mcp",
        help="comma-separated variants to interleave (base, mcpx, mcp)",
    )
    parser.add_argument("--json", type=str, default="", help="write raw results here")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for variant in variants:
        if variant not in _VARIANTS:
            parser.error(f"unknown variant {variant!r}; choose from {', '.join(_VARIANTS)}")

    # Warm the parent's own import graph BEFORE the first timed run: the desktop
    # backend server is long-lived, so its parent-side imports are already paid
    # and counting them as per-send latency would be wrong.
    import local_operator.providers.registry  # noqa: F401
    import local_operator.session.session  # noqa: F401
    import local_operator.session_factory  # noqa: F401
    from local_operator.session import attached  # noqa: F401
    from local_operator.session.runtime import launch  # noqa: F401

    try:
        import local_operator.server.utils.desktop_sessions  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — the benchmark must still run
        print(f"  (desktop_sessions pre-warm skipped: {type(exc).__name__}: {exc})")

    _instrument_parent()

    rows: list[dict[str, Any]] = []
    for index in range(args.pairs):
        for variant in variants:
            try:
                row = await _one_run(variant, index)
            except Exception as exc:  # noqa: BLE001 — a loaded host is not a result
                # A run that cannot complete (this host reaches a load average
                # of 200+, where a cold engage can time out on its own) is
                # DROPPED and printed, never allowed to kill the campaign: a
                # benchmark that dies on the first bad run reports nothing, and
                # one that silently retries flatters whichever variant got the
                # retry. The dropped count is part of the result.
                print(
                    f"  [{variant} {index + 1}/{args.pairs}] DROPPED: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            rows.append(row)
            d = row["derived"]
            print(
                f"  [{variant} {index + 1}/{args.pairs}] total={d.get('total_ms')} "
                f"cold={d.get('cold_ms')} bind={d.get('bind_ms')} "
                f"child_publish={d.get('child_total_to_publish_ms')} "
                f"poll_dead={d.get('poll_grid_dead_ms')} "
                f"dial={d.get('dial_ms')} sync={d.get('frontend_sync_ms')} "
                f"hist={d.get('load_history_ms')} "
                f"mcp={d.get('child_wire_mcp_ms')} "
                f"mcp_exit_after_pub={d.get('mcp_exit_ms_after_publish')} "
                f"found={row['found']}",
                flush=True,
            )

    summary = _print_summary(rows, variants)
    loads = [r["loadavg"] for r in rows if isinstance(r.get("loadavg"), float)]
    if loads:
        print(
            f"\n  host load average during this campaign (1 min, per run): "
            f"median {statistics.median(loads):.0f}, min {min(loads):.0f}, max {max(loads):.0f}"
        )
    _print_paired(rows, variants)
    _print_thresholds(summary)

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")

    for row in rows:
        shutil.rmtree(row["root"], ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
