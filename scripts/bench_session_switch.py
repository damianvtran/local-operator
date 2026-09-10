"""Measure sidebar session-switch latency: click-to-usable-frame, and the jitter around it.

The operator's report is three symptoms of one switch: the composer resizes, the
transcript reflows and "shuffles" after it is already visible, and the scroll
position bounces before it settles. So this harness does not report a single
latency number. It reports the latency AND the counts of the three visible
artefacts, because on a box under load average ~36 a wall-clock number is noise
and a **count of frames, layouts, mounts and distinct scroll positions is not** —
a count is a fact about how much work one switch costs, and it is identical on an
idle laptop and a wedged CI runner. See AGENTS.md "Timing, flakes, and how to
assert that something is fast": structural invariants first, CPU before wall,
never calibrate a ceiling from this laptop.

Examples (no live configuration, sessions, providers or sockets are used)::

    # Baseline on the tree this script lives in.
    .venv/bin/python scripts/bench_session_switch.py --output /tmp/switch-bench/baseline.json

    # Interleaved A/B: the SAME harness alternates between two trees within one
    # run, group by group, so before and after share the same load weather.
    .venv/bin/python scripts/bench_session_switch.py \
        --source-root /tmp/lo-bench-base --output /tmp/switch-bench/ab.json

WHY A SEPARATE SCRIPT AND NOT ``sidebar_performance.py --mode switch``.
That harness measures the sidebar WIDGET's render cost; its matrix axes are
(session count, sidebar open, streaming) and its cell result is a render count.
This measures the SWITCH, whose axes are (transcript size, terminal size, cache
pattern) and whose result is a per-switch record. Grafting the second matrix into
the first script's ``main()`` would give one file two unrelated cell shapes and
two unrelated result schemas. It follows every convention of the neighbouring
``bench_*.py`` scripts instead — module-level parser, ``--source-root``,
``probe_isolation`` before the package import, CMUX scrubbing, provenance in the
JSON — which is the house style those files already establish.

TWO PROCESSES, ONE HARNESS. An editable install resolves ONE source root, so a
single process cannot hold two trees. The driver therefore alternates *worker
subprocesses*, each running under its own tree's venv, group by group and with
the order flipped every round. That is what makes an A/B credible here: the two
trees are sampled seconds apart under the same load, not in two runs minutes
apart. ``--source-root`` names the OTHER tree; the tree this file lives in is
always side "A".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent

#: One switch is measured from the ``Selected`` message to the readiness future.
#: Everything after that future is the SETTLE window: the frames the user is
#: already looking at. It is bounded in loop turns rather than milliseconds
#: because a turn count survives contention that a wall-clock budget does not
#: (AGENTS.md, "Wait on the event, never on the clock").
SETTLE_PUMPS = 12

#: How many switches make up one cell's sample. Small enough that the whole
#: matrix stays inside a few minutes on a loaded box, large enough that a median
#: and a p90 mean something.
DEFAULT_SAMPLES = 8

#: Distinct sessions the cold pattern rotates through. Must exceed
#: ``RETAINED_PRESENTATIONS`` -- which is **12**, not the 4 an earlier draft of
#: this comment claimed (agent review round 1, R4) -- by a clear margin, or the
#: "cold" cell silently starts measuring cache hits instead of cold switches.
#:
#: At 12 the rotation was exactly the retain budget, i.e. no margin at all: a
#: presentation evicted only just before its turn came round again is one
#: scheduling accident away from still being resident. 20 restores the headroom
#: the sentence above promises, and the cost is only a few more seeded
#: transcripts per worker.
COLD_POOL = 20

TRANSCRIPTS = {"small": 7, "large": 134}  # turns; a turn renders 3 blocks
SIZES = {"120x36": (120, 36), "200x50": (200, 50)}
PATTERNS = ("cold", "warm", "alternating")


def groups() -> list[str]:
    return [f"{t}@{s}" for t in TRANSCRIPTS for s in SIZES]


PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument(
    "--source-root",
    type=Path,
    default=None,
    help="a second worktree to interleave against; omit for a single-tree baseline",
)
PARSER.add_argument("--output", type=Path, required=True, help="destination JSON file")
PARSER.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
PARSER.add_argument("--rounds", type=int, default=3, help="interleaved passes over the matrix")
PARSER.add_argument("--group", default="", help=argparse.SUPPRESS)  # worker-only
PARSER.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)


# ---------------------------------------------------------------------------
# Worker: measures one group (transcript size x terminal size, all patterns).
# ---------------------------------------------------------------------------


def _run_worker(args: argparse.Namespace) -> None:
    # Multiplexer variables are independent of HOME/config. Even a headless
    # pilot must not inherit identifiers that could rename the operator's real
    # workspace. Scrubbed before ANY import, including this file's own.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            del os.environ[key]

    import scripts.probe_isolation  # noqa: F401  -- re-homes HOME/config on import

    # isort: split
    # Isolation must precede even the package root import.
    from unittest.mock import patch

    import local_operator
    from local_operator.harness.types import Message, ToolCall
    from local_operator.resume import SessionRow
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.session_catalog import CatalogEntry
    from local_operator.tui.session_interaction import SessionInteraction
    from local_operator.tui.widgets.session_sidebar import SessionSidebar
    from local_operator.tui.widgets.transcript import TranscriptView
    from tests.unit.tui.test_app_pilot import _factory
    from tests.unit.tui.test_sidebar_swap_reset import SidebarRemote

    transcript_name, _, size_name = args.group.partition("@")
    turns = TRANSCRIPTS[transcript_name]
    size = SIZES[size_name]

    def history(count: int) -> list[Message]:
        """A conversation shaped like the ones the complaint is about.

        Every turn carries a tool call AND its result, so the replay path mounts
        a ``ToolCard`` per turn rather than only prose blocks — the operator's
        report is specifically about tool traces shuffling, and a prose-only
        fixture would not exercise the card projection at all.
        """
        out: list[Message] = []
        for i in range(count):
            out.append(
                Message.user(
                    f"turn {i}: please look into the switch latency and report back. " * 2,
                    id=f"u{i}",
                )
            )
            call = ToolCall(
                id=f"c{i}",
                name="bash",
                arguments={"command": f"rg -n 'switch' local_operator/tui/app.py | sed -n '{i}p'"},
            )
            out.append(
                Message.assistant(
                    f"Reading the sidebar path, step {i}. Here is what the trace shows so far.",
                    id=f"a{i}",
                    tool_calls=[call],
                )
            )
            out.append(
                Message(
                    role="tool",
                    content=[],
                    tool_call_id=f"c{i}",
                    tool_name="bash",
                    id=f"t{i}",
                )
            )
        return out

    ids = [f"{i:04x}00000000" for i in range(COLD_POOL + 2)]
    home = SidebarRemote("home00000000", history=history(2))
    remotes = {sid: SidebarRemote(sid, history=history(turns)) for sid in ids}
    entries = [
        CatalogEntry(SessionRow(sid, 1788700000 - i * 60, f"Switch bench session {i}"))
        for i, sid in enumerate(ids)
    ]

    app = OperatorApp(lambda: _factory(home))

    async def lease(session_id: str, *, speculative: bool = False) -> SessionInteraction:
        """Stand in for the disk/socket lease, and NOTHING else.

        The real lease reads an owner record and opens a socket, neither of
        which may touch the operator's machine here. Everything the measurement
        is about — prepare, mount, park, commit, adopt, reveal, the readiness
        gate — stays production code.

        ``_interactions[id(remote)]`` is populated exactly as the real lease
        does: ``_adopt_session`` looks the source up by session identity, and a
        fake that skips this registration makes the app adopt a DIFFERENT
        interaction than the one the readiness gate is bound to, so the gate can
        never be satisfied and every cold switch reads as a 15 s timeout.
        """
        source = app._sidebar_sources.get(session_id)
        if source is None:
            remote = remotes[session_id]
            source = SessionInteraction(remote)
            source.display_only = remote.is_cold
            # A DRAFT PER SESSION, and a multi-line one. The composer is
            # `height: auto` between 1 and 8 rows, so it can only be seen to
            # resize if the sessions being switched between disagree about how
            # tall it should be. With every draft empty the composer is pinned
            # at one row and the harness would report "no resize" for a switch
            # the operator watches shrink and grow. Lengths are staggered so
            # consecutive targets in every pattern want different heights.
            lines = 1 + (int(session_id[:4], 16) % 4)
            source.draft.text = "\n".join(
                f"draft line {n} for {session_id[:4]}" for n in range(lines)
            )
            app._interactions[id(remote)] = source
            app._sidebar_sources[session_id] = source
        source.preparations += 1
        return source

    # ---- instrumentation. All of it lives here; product code is untouched. ----
    state: dict[str, Any] = {
        "armed": False,
        "revealed": False,
        "displays": 0,
        "after_refresh": 0,
        "layouts": 0,
        "reveal_displays": 0,
        "reveal_layouts": 0,
        "reveal_mounts": 0,
        "scrolls": [],
        "editor_heights": [],
        "settle": False,
        "settle_displays": 0,
        "settle_layouts": 0,
        "settle_mounts": 0,
        "settle_scrolls": [],
    }

    def observe() -> None:
        """Sample the VISIBLE surfaces on a display pass.

        Scroll position and composer height are read per painted frame rather
        than once at the end: the complaint is that they move between frames,
        and a single end-state reading cannot see motion. Distinct-value counts
        of these sequences are the load-independent evidence — one distinct
        value means the user saw no movement, whatever the wall clock said.
        """
        try:
            editor = app._editor()
        except Exception:  # noqa: BLE001 - a mid-swap query can find nothing
            return
        # The COMPOSER is sampled on every armed frame, including the ones
        # before reveal. The operator's report is that it "shrinks while in the
        # loading state and then grows back": that shrink happens during the
        # pending window, so a probe that only looked after reveal would miss
        # the exact frames complained about and report zero resizes.
        state["editor_heights"].append(int(editor.outer_size.height))
        if not state["revealed"]:
            return
        # SCROLL is only meaningful once the incoming view IS the visible one:
        # before that, `_transcript_view()` returns the outgoing transcript and
        # its scroll position says nothing about the switch.
        try:
            view = app._transcript_view()
        except Exception:  # noqa: BLE001
            return
        bucket = "settle_scrolls" if state["settle"] else "scrolls"
        state[bucket].append(float(view.scroll_y))

    real_hook = app.post_display_hook

    def post_display_hook() -> None:
        if state["armed"]:
            state["displays"] += 1
            if state["settle"]:
                state["settle_displays"] += 1
            elif state["revealed"]:
                state["reveal_displays"] += 1
            observe()
        real_hook()

    real_after_refresh = app.call_after_refresh

    def call_after_refresh(callback: Any, *a: Any, **kw: Any) -> bool:
        if state["armed"]:
            state["after_refresh"] += 1
        return real_after_refresh(callback, *a, **kw)

    real_commit = app._commit_sidebar_session

    def commit(session_id: str, prepared: Any, generation: int) -> Any:
        # Reveal is the instant the prepared view becomes the visible one. Every
        # layout, mount and scroll change AFTER this point is work the user is
        # looking at, which is exactly the "shuffle" being counted.
        started = time.perf_counter()
        out = real_commit(session_id, prepared, generation)
        state["commit_ms"] = (time.perf_counter() - started) * 1000
        state["revealed"] = True
        observe()
        return out

    real_prepare = app._prepare_sidebar_session

    async def prepare(session_id: str, **kw: Any) -> Any:
        started = time.perf_counter()
        out = await real_prepare(session_id, **kw)
        state["prepare_ms"] = (time.perf_counter() - started) * 1000
        return out

    ready: dict[str, Any] = {}
    real_await = app._await_sidebar_frame

    def await_frame(source: Any, generation: int) -> Any:
        future = real_await(source, generation)
        ready["future"] = future
        return future

    def install_layout_probe() -> None:
        # Bound only once the pilot has pushed a screen: `app.screen` raises
        # ScreenStackError before `run_test` starts, and the layout pass being
        # counted belongs to the live screen instance, not to the class.
        screen = app.screen
        real_layout = screen._refresh_layout

        def refresh_layout(*a: Any, **kw: Any) -> None:
            if state["armed"]:
                state["layouts"] += 1
                if state["settle"]:
                    state["settle_layouts"] += 1
                elif state["revealed"]:
                    state["reveal_layouts"] += 1
            return real_layout(*a, **kw)

        screen._refresh_layout = refresh_layout  # type: ignore[method-assign]

    real_append = TranscriptView.append_block

    def append_block(view: Any, block: Any) -> None:
        # Only mounts onto the ALREADY VISIBLE view count: rows appended to a
        # parked offscreen replay are the preparation doing its job, while rows
        # appended after reveal are rows that appear under the reader's eyes.
        if state["armed"] and state["revealed"] and view is app._transcript_view():
            if state["settle"]:
                state["settle_mounts"] += 1
            else:
                state["reveal_mounts"] += 1
        return real_append(view, block)

    async def measure(pilot: Any, target: str) -> dict[str, Any]:
        warm = target in app._sidebar_presentations
        for key, value in (
            ("revealed", False),
            ("settle", False),
            ("displays", 0),
            ("after_refresh", 0),
            ("layouts", 0),
            ("reveal_displays", 0),
            ("reveal_layouts", 0),
            ("reveal_mounts", 0),
            ("settle_displays", 0),
            ("settle_layouts", 0),
            ("settle_mounts", 0),
        ):
            state[key] = value
        state["scrolls"] = []
        state["settle_scrolls"] = []
        state["editor_heights"] = []
        state["prepare_ms"] = None
        state["commit_ms"] = None
        ready.clear()

        editor_before = app._editor().outer_size.height
        state["armed"] = True
        wall0, cpu0 = time.perf_counter(), time.thread_time()
        app.post_message(SessionSidebar.Selected(target))
        # Drive the loop, not a sleep: readiness is published by the app, and
        # the bound below exists only so a genuine wedge fails the run.
        future = None
        for _ in range(4000):
            await pilot.pause()
            future = ready.get("future")
            if future is not None and future.done():
                break
        wall = (time.perf_counter() - wall0) * 1000
        cpu = (time.thread_time() - cpu0) * 1000
        settled = future is not None and future.done() and not future.cancelled()
        # Narrowed for the type checker as well as for the reader: `settled`
        # already implies a non-None future, but only the explicit test proves
        # it at the call site.
        ok = settled and future is not None and future.exception() is None
        if future is not None and not future.done():
            future.cancel()

        # The settle window: frames the user is already looking at. Anything
        # that moves here is the reported "shuffle after it loads".
        state["settle"] = True
        for _ in range(SETTLE_PUMPS):
            await pilot.pause()
        state["armed"] = False

        view = app._transcript_view()
        heights = state["editor_heights"]
        return {
            "target": target,
            "warm": warm,
            "ok": ok,
            "wall_ms": wall,
            "loop_cpu_ms": cpu,
            "prepare_ms": state["prepare_ms"],
            "commit_ms": state["commit_ms"],
            "displays": state["displays"],
            "call_after_refresh": state["after_refresh"],
            "layouts": state["layouts"],
            "reveal_displays": state["reveal_displays"],
            "reveal_layouts": state["reveal_layouts"],
            "reveal_mounts": state["reveal_mounts"],
            "scroll_settle": len(set(state["scrolls"])),
            "scroll_sequence": state["scrolls"][:8],
            "settle_displays": state["settle_displays"],
            "settle_layouts": state["settle_layouts"],
            "settle_mounts": state["settle_mounts"],
            "settle_scroll_settle": len(set(state["settle_scrolls"])),
            "composer_heights": sorted(set([editor_before] + heights)),
            "composer_resizes": len(set([editor_before] + heights)) - 1,
            "blocks_at_ready": len(view.blocks()),
            "scroll_y_at_ready": float(view.scroll_y),
        }

    async def run() -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with (
            patch("local_operator.session.remote.RemoteSession", SidebarRemote),
            patch("local_operator.tui.session_catalog.load_catalog", return_value=entries),
            # Prewarm reads owner records off disk; the cache states it would
            # produce are modelled explicitly by the `warm` pattern instead.
            patch.object(OperatorApp, "_prewarm_sidebar", lambda _s, _e: None),
            patch.object(OperatorApp, "_check_for_update", lambda _s: None),
            patch.object(OperatorApp, "_start_terminal_title", lambda _s: None),
            patch.object(OperatorApp, "_start_multiplexer_broadcast", lambda _s: None),
            patch.object(OperatorApp, "_start_herdr_reporter", lambda _s: None),
            patch.object(TranscriptView, "append_block", append_block),
        ):
            async with app.run_test(size=size) as pilot:
                app._lease_sidebar_source = lease  # type: ignore[method-assign]
                app._prepare_sidebar_session = prepare  # type: ignore[method-assign]
                app._commit_sidebar_session = commit  # type: ignore[method-assign]
                app._await_sidebar_frame = await_frame  # type: ignore[method-assign]
                app.post_display_hook = post_display_hook  # type: ignore[method-assign]
                app.call_after_refresh = call_after_refresh  # type: ignore[method-assign]
                # SessionNavigation captures BOUND methods in `__init__`, so
                # rebinding the attributes above is invisible to it — the
                # coordinator would keep calling the originals and the phase
                # timings would silently read `null` while everything else
                # looked correct. Re-point the coordinator's own slots too.
                app._sidebar_navigation._prepare = prepare
                app._sidebar_navigation._commit = commit
                install_layout_probe()
                for _ in range(40):
                    await pilot.pause()
                app._session_sidebar.set_entries(entries)
                await pilot.pause()

                # COLD first and in a strict forward rotation: each target is
                # visited once, so nothing it measures can be a cache hit. The
                # pool is larger than the retain budget, so by the time the
                # rotation wraps the early entries have been evicted.
                for i in range(args.samples):
                    record = await measure(pilot, ids[i % COLD_POOL])
                    record["pattern"] = "cold"
                    results.append(record)

                # WARM: a two-session ping-pong, both retained. Every switch
                # here takes the `_sidebar_presentations` hit path.
                warm_pair = ids[COLD_POOL], ids[COLD_POOL + 1]
                for sid in warm_pair:
                    await measure(pilot, sid)
                for i in range(args.samples):
                    record = await measure(pilot, warm_pair[i % 2])
                    record["pattern"] = "warm"
                    results.append(record)

                # ALTERNATING over three, which is what a user actually does and
                # what exercises the LRU's recency reinsertion rather than a
                # degenerate two-key swap.
                trio = ids[0], ids[1], ids[2]
                for sid in trio:
                    await measure(pilot, sid)
                for i in range(args.samples):
                    record = await measure(pilot, trio[i % 3])
                    record["pattern"] = "alternating"
                    results.append(record)
        return results

    records = asyncio.run(run())
    for record in records:
        record["group"] = args.group
        record["transcript"] = transcript_name
        record["size"] = size_name
        record["history_messages"] = turns * 3
    print(
        json.dumps({"source": str(local_operator.__file__), "records": records}),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Driver: alternates worker subprocesses between trees.
# ---------------------------------------------------------------------------


def _provenance(root: Path) -> dict[str, Any]:
    def git(*a: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *a], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "root": str(root),
        "sha": git("rev-parse", "HEAD"),
        "describe": git("describe", "--always", "--dirty"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _interpreter(root: Path) -> tuple[str, dict[str, str]]:
    """Each worktree owns its own venv; a wrong one silently measures main.

    AGENTS.md is explicit that an editable install resolves ONE hard-coded
    source root, so running tree B's harness under tree A's interpreter would
    import A and report it as B. The venv is preferred and ``PYTHONPATH`` is set
    regardless, so the tree under test wins even if a venv is missing or stale;
    the worker echoes the resolved ``local_operator.__file__`` back and the
    driver refuses a mismatch.
    """
    env = dict(os.environ, PYTHONPATH=str(root))
    candidate = root / ".venv/bin/python"
    return (str(candidate) if candidate.exists() else sys.executable), env


def _worker(root: Path, group: str, samples: int) -> list[dict[str, Any]]:
    python, env = _interpreter(root)
    env.pop("NO_COLOR", None)
    env["TERM"] = "xterm-256color"
    proc = subprocess.run(
        [
            python,
            str(root / "scripts/bench_session_switch.py"),
            "--worker",
            "--group",
            group,
            "--samples",
            str(samples),
            "--output",
            os.devnull,
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker for {root} group {group} failed:\n{proc.stderr[-3000:]}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    resolved = Path(payload["source"]).resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError(f"worker for {root} imported {resolved} — wrong tree, refusing")
    return payload["records"]


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Median/p90/IQR, never a bare mean.

    A mean on this box is dominated by whichever sample happened to land inside
    a scheduling hole. The dispersion is reported alongside so a reader can see
    whether a difference between two trees is larger than the run's own spread.
    """

    def stat(key: str) -> dict[str, Any]:
        values = sorted(float(r[key]) for r in records if r.get(key) is not None)
        if not values:
            return {"n": 0}
        quantiles = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
        return {
            "n": len(values),
            "median": round(statistics.median(values), 2),
            "p90": round(values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))], 2),
            "iqr": round(quantiles[2] - quantiles[0], 2),
            "min": round(values[0], 2),
            "max": round(values[-1], 2),
        }

    keys = (
        "wall_ms",
        "loop_cpu_ms",
        "prepare_ms",
        "commit_ms",
        "displays",
        "call_after_refresh",
        "layouts",
        "reveal_displays",
        "reveal_layouts",
        "reveal_mounts",
        "scroll_settle",
        "settle_displays",
        "settle_layouts",
        "settle_mounts",
        "settle_scroll_settle",
        "composer_resizes",
        "blocks_at_ready",
    )
    return {
        "samples": len(records),
        "failures": sum(1 for r in records if not r["ok"]),
        **{key: stat(key) for key in keys},
    }


def _run_driver(args: argparse.Namespace) -> None:
    trees: list[tuple[str, Path]] = [("A", HERE)]
    if args.source_root is not None:
        trees.append(("B", args.source_root.resolve()))

    collected: dict[str, list[dict[str, Any]]] = {side: [] for side, _ in trees}
    started = time.time()
    for round_index in range(args.rounds):
        for group in groups():
            # Flip the order every round so neither tree systematically owns the
            # quieter half of a load wave. Without this an A/B on a busy box
            # measures scheduling order as much as it measures code.
            order = trees if round_index % 2 == 0 else list(reversed(trees))
            for side, root in order:
                records = _worker(root, group, args.samples)
                for record in records:
                    record["round"] = round_index
                    record["side"] = side
                collected[side].extend(records)
                print(
                    f"round {round_index} {group} side {side}: {len(records)} switches",
                    file=sys.stderr,
                    flush=True,
                )

    payload: dict[str, Any] = {
        "harness": "bench_session_switch",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s": round(time.time() - started, 1),
        "rounds": args.rounds,
        "samples_per_cell_per_round": args.samples,
        "settle_pumps": SETTLE_PUMPS,
        "trees": {side: _provenance(root) for side, root in trees},
        "sides": {},
    }
    for side, records in collected.items():
        cells: dict[str, Any] = {}
        for group in groups():
            for pattern in PATTERNS:
                subset = [r for r in records if r["group"] == group and r["pattern"] == pattern]
                if subset:
                    cells[f"{group}/{pattern}"] = _summarise(subset)
        payload["sides"][side] = {"overall": _summarise(records), "cells": cells}
    payload["records"] = {side: records for side, records in collected.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    overall = {side: payload["sides"][side]["overall"] for side in payload["sides"]}
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    if ARGS.samples < 1:
        PARSER.error("--samples must be positive")
    sys.path.insert(0, str(HERE))
    if ARGS.worker:
        if ARGS.group not in groups():
            PARSER.error(f"--group must be one of {groups()}")
        _run_worker(ARGS)
    else:
        _run_driver(ARGS)
