"""Growth harness: what accumulates in-process across sidebar uptime and switches.

Drives the REAL assembled ``OperatorApp`` headlessly (stylesheet loaded, real
``RuntimeServer`` owners in-process, real ``RemoteSession`` viewers over real
loopback sockets) and records, at checkpoints, every quantity that could
explain "the TUI gets slower until /reload":

* process RSS, ``gc`` object counts by type (top growers vs the baseline),
* live asyncio tasks, Textual ``Timer`` objects (app + every widget), workers,
* DOM node count (``len(app.query('*'))``), transcript views mounted,
* sizes of every app/sidebar/session container that is keyed per session,
* ``FrontendStateStore`` / ``RemoteSession`` subscriber list lengths,
* wall + loop-CPU for one switch and one sidebar poll cycle,
* tracemalloc top allocation deltas between the first and last checkpoint.

Usage (from the worktree root; isolation happens on import, see
``scripts/probe_isolation``)::

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/diag/tui_growth.py \\
        --mode switches --switches 5 20 50 --sessions 12 --output /tmp/growth

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/diag/tui_growth.py \\
        --mode uptime --polls 60 --sessions 30 --output /tmp/growth-uptime

``--mode switches``: opens the sidebar, then performs K switches round-robin
across ``--sessions`` live owners, checkpointing after each K in ``--switches``.

``--mode uptime``: opens the sidebar over ``--sessions`` live owners (busy ones
so the spinner ticks) and drives the 2 s catalog poll + spinner manually for
``--polls`` cycles, checkpointing every ``--checkpoint-every`` polls.

Output: ``<output>/checkpoints.json`` (all numbers), ``<output>/report.md``
(tables), ``<output>/tracemalloc.txt`` (top deltas), and a
``<output>/profile-{early,late}.prof`` pair in switches mode.

Numbers here are EVIDENCE, not CI ceilings. The structural assertions a
regression test should make are the ones that read the same at K=5 and K=50
(timers, tasks, subscribers, DOM nodes, container sizes).
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import gc
import json
import os
import resource
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
PARSER.add_argument("--output", type=Path, required=True)
PARSER.add_argument("--mode", choices=("switches", "uptime"), default="switches")
PARSER.add_argument("--sessions", type=int, default=12)
PARSER.add_argument(
    "--cycle",
    type=int,
    default=0,
    help="switch round-robin over only the first N sessions (0 = all); N<=4 stays inside the LRU",
)
PARSER.add_argument("--switches", type=int, nargs="+", default=[5, 20, 50])
PARSER.add_argument("--polls", type=int, default=60)
PARSER.add_argument("--checkpoint-every", type=int, default=10)
PARSER.add_argument("--history", type=int, default=40, help="messages seeded per session")
PARSER.add_argument(
    "--idle-seconds",
    type=float,
    default=3.0,
    help="seconds of IDLE pumping measured at each checkpoint",
)
PARSER.add_argument(
    "--keystrokes",
    type=int,
    default=20,
    help="keypresses timed at each checkpoint (perceived latency)",
)
PARSER.add_argument(
    "--approve-all",
    action="store_true",
    help="viewers default to approve_all (the operator's `auto` tool_approval_mode). "
    "With busy owners this arms SessionInteraction.retained_for_auto_work, which "
    "is the clause `_sidebar_source_releasable` honours on the idle path.",
)
PARSER.add_argument(
    "--retained",
    type=int,
    default=0,
    help="override RETAINED_PRESENTATIONS (0 = leave the shipped value of 12)",
)
PARSER.add_argument("--tracemalloc", action="store_true")
PARSER.add_argument("--profile", action="store_true")
PARSER.add_argument(
    "--no-cleanup-timer",
    action="store_true",
    help="stub OperatorApp._report_startup_cleanup (experiment: is the per-adopt timer chain "
    "the growth?)",
)
PARSER.add_argument(
    "--no-prewarm",
    action="store_true",
    help="stub OperatorApp._prewarm_sidebar (experiment: is prewarm churn the steady-state cost?)",
)
ARGS = PARSER.parse_args()
ARGS.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ARGS.source_root.resolve()))
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        del os.environ[_key]

import scripts.probe_isolation as isolation  # noqa: E402

# isort: split
import local_operator  # noqa: E402
from local_operator.session.remote import RemoteSession  # noqa: E402
from local_operator.session.runtime.owned import OwnedSessionHandle  # noqa: E402
from local_operator.session.runtime.server import RuntimeServer  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.e2e.harness import (  # noqa: E402
    ScriptedStream,
    assistant_message,
    build_session,
    seed_transcript,
    user_message,
    wait_for_adoption,
)

CONFIG = isolation.SANDBOX / "config"


def _publish_per_session() -> None:
    """Give every in-process owner its own discovery record.

    ``registry.publish`` keys the record file by PID because production runs
    one session per process. Every owner in this harness shares one PID, so
    without this the N servers overwrite one file, the catalog sees ONE live
    row and prewarm never engages — a fixture that silently tests the cold
    path. Keying by ``(pid, session_id)`` keeps ``scan`` (glob *.json +
    pid_alive) truthful for all of them.
    """
    import json
    import os
    import tempfile
    import time

    from local_operator.session.runtime import registry

    def publish(record: Any, root: Path | None = None) -> Path:
        directory = registry.run_dir(root)
        record.heartbeat_at = time.time()
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".x.", suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(record.to_json(), handle)
        target = directory / f"{record.pid}-{record.session_id}.json"
        os.replace(tmp, target)
        return target

    registry.publish = publish  # type: ignore[assignment]
    registry.unpublish = lambda pid, root=None: None  # type: ignore[assignment]


_publish_per_session()

#: Every switch / poll timing in order, so trends are visible rather than
#: three noisy single samples.
SERIES: dict[str, list[float]] = {
    "switch_total_wall_ms": [],
    "switch_loop_cpu_ms": [],
    "poll_wall_ms": [],
    "poll_loop_cpu_ms": [],
    "prepare_loop_cpu_ms": [],
}

#: Work counters, bumped by wrappers installed in ``instrument()``; the
#: checkpoint records the running total so per-interval deltas are readable.
COUNTS: Counter[str] = Counter()


def instrument() -> None:
    import local_operator.tui.app as app_mod

    if ARGS.retained:
        app_mod.RETAINED_PRESENTATIONS = ARGS.retained

    def wrap(cls: Any, name: str, key: str) -> None:
        original = getattr(cls, name)
        if asyncio.iscoroutinefunction(original):

            async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
                COUNTS[key] += 1
                return await original(*args, **kwargs)

            setattr(cls, name, wrapped_async)
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            COUNTS[key] += 1
            return original(*args, **kwargs)

        setattr(cls, name, wrapped)

    # Count cache HITS: the `cached is not None and _sidebar_presentation_current`
    # branch of _prepare_sidebar_session is the whole point of the LRU, so a
    # switch that misses it paid a cold rebuild.
    original_current = app_mod.OperatorApp._sidebar_presentation_current

    def counted_current(cached: Any, source: Any, gate: Any) -> bool:
        ok = original_current(cached, source, gate)
        COUNTS["presentation_hit" if ok else "presentation_stale"] += 1
        return ok

    app_mod.OperatorApp._sidebar_presentation_current = staticmethod(counted_current)

    original_prepare = app_mod.OperatorApp._prepare_sidebar_session

    async def timed_prepare(self: Any, *args: Any, **kwargs: Any) -> Any:
        COUNTS["prepare_sidebar_session"] += 1
        if kwargs.get("speculative"):
            COUNTS["prepare(speculative)"] += 1
        elif kwargs.get("refresh"):
            COUNTS["prepare(refresh)"] += 1
        else:
            COUNTS["prepare(navigation)"] += 1
        cpu0 = time.thread_time()
        try:
            return await original_prepare(self, *args, **kwargs)
        finally:
            SERIES["prepare_loop_cpu_ms"].append(round((time.thread_time() - cpu0) * 1000, 1))

    app_mod.OperatorApp._prepare_sidebar_session = timed_prepare  # type: ignore[method-assign]
    wrap(app_mod.OperatorApp, "_lease_sidebar_source", "lease_sidebar_source")
    wrap(app_mod.OperatorApp, "_release_sidebar_source", "release_sidebar_source(called)")
    wrap(app_mod.OperatorApp, "_release_sidebar_preparation", "release_sidebar_preparation")
    wrap(app_mod.OperatorApp, "_report_startup_cleanup", "report_startup_cleanup")
    if ARGS.no_cleanup_timer:
        app_mod.OperatorApp._report_startup_cleanup = (  # type: ignore[method-assign]
            lambda self, **kw: None
        )
    if ARGS.no_prewarm:
        app_mod.OperatorApp._prewarm_sidebar = (  # type: ignore[method-assign]
            lambda self, entries: None
        )
    wrap(RemoteSession, "dispose", "RemoteSession.dispose")
    original_connect = RemoteSession.connect.__func__  # type: ignore[attr-defined]

    async def connect(cls: Any, *args: Any, **kwargs: Any) -> Any:
        COUNTS["RemoteSession.connect"] += 1
        return await original_connect(cls, *args, **kwargs)

    RemoteSession.connect = classmethod(connect)  # type: ignore[method-assign]


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def _all_widgets(app: OperatorApp) -> list[Any]:
    out: list[Any] = []
    for screen in app.screen_stack:
        out.append(screen)
        out.extend(screen.walk_children(with_self=False))
    return out


def timers_snapshot(app: OperatorApp) -> dict[str, Any]:
    """Every Textual Timer reachable from every message pump, by owner type."""
    by_owner: Counter[str] = Counter()
    by_callback: Counter[str] = Counter()
    running = 0
    total = 0
    for pump in [app, *_all_widgets(app)]:
        timers = getattr(pump, "_timers", None)
        if not timers:
            continue
        for timer in list(timers):
            total += 1
            by_owner[type(pump).__name__] += 1
            cb = getattr(timer, "_callback", None)
            inner = getattr(cb, "args", None)
            target = inner[0] if inner else cb
            by_callback[getattr(target, "__qualname__", None) or repr(target)[:60]] += 1
            task = getattr(timer, "_task", None)
            if task is not None and not task.done():
                running += 1
    return {
        "total": total,
        "running": running,
        "by_owner": dict(by_owner),
        "by_callback": dict(by_callback),
    }


def tasks_snapshot() -> dict[str, Any]:
    names: Counter[str] = Counter()
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    for task in tasks:
        coro = task.get_coro()
        name = getattr(coro, "__qualname__", None) or getattr(coro, "__name__", None) or repr(coro)
        names[str(name)] += 1
    return {"total": len(tasks), "by_coro": dict(names.most_common(25))}


def containers_snapshot(app: OperatorApp) -> dict[str, int]:
    """Sizes of every per-session / per-switch container we could find."""
    sizes: dict[str, int] = {}

    def size(name: str, value: Any) -> None:
        try:
            sizes[name] = len(value)
        except TypeError:
            sizes[name] = -1

    size("app._sidebar_sources", app._sidebar_sources)
    size("app._sidebar_presentations", app._sidebar_presentations)
    size("app._sidebar_unretainable", app._sidebar_unretainable)
    size("app._sidebar_prior_workers", app._sidebar_prior_workers)
    size("app._interactions", app._interactions)
    size("app._event_sources", app._event_sources)
    size("app._sidebar_navigation._tasks", app._sidebar_navigation._tasks)
    size("app._sidebar_drafts._memory", app._sidebar_drafts._memory)
    size("app._resume_results", app._resume_results)
    size("app._resume_mounted_ids", app._resume_mounted_ids)
    size("app._tool_cards", app._tool_cards)
    size("app._composing_cards", app._composing_cards)
    size("app._superseded_steer_controllers", app._superseded_steer_controllers)
    size("app.workers", list(app.workers))
    sidebar = app._session_sidebar
    size("sidebar._painted_lines", sidebar._painted_lines)
    sizes["app._sidebar_ready_frame_pending"] = int(app._sidebar_ready_frame is not None)
    displayed = getattr(app, "_sidebar_displayed_frame", None)
    sizes["app._sidebar_displayed_frame.painted"] = len(displayed[4]) if displayed else 0
    sizes["app._message_queue"] = app._message_queue.qsize()
    sizes["screen._message_queue"] = app.screen._message_queue.qsize()
    sizes["app._callbacks"] = len(getattr(app, "_next_callbacks", ()))
    import threading

    sizes["threads"] = threading.active_count()
    names: Counter[str] = Counter()
    for t in threading.enumerate():
        base = t.name.split("_")[0] if t.name.startswith("asyncio") else t.name
        names[base.rstrip("0123456789-") or t.name] += 1
    sizes.update({f"thread:{k}": v for k, v in names.items()})
    compositor = app.screen._compositor
    for name in ("_full_map", "_visible_widgets", "_layers", "_layers_visible", "map"):
        value = getattr(compositor, name, None)
        if value is not None and hasattr(value, "__len__"):
            sizes[f"compositor.{name}"] = len(value)
    from textual.widget import Widget as _Widget

    sizes["gc.Widget_total"] = sum(1 for o in gc.get_objects() if isinstance(o, _Widget))
    size("sidebar.entries", sidebar.entries)
    # Per-source structures that a leak would multiply.
    subscribers = 0
    handlers = 0
    retired_sources = 0
    connection_tasks = 0
    seen: dict[int, Any] = {}
    for source in [*app._sidebar_sources.values(), *app._interactions.values()]:
        seen[id(source)] = source
    sizes["sources.distinct"] = len(seen)
    for source in seen.values():
        if source.retired:
            retired_sources += 1
        session = source.session
        store = getattr(session, "_frontend_store", None)
        if store is not None:
            subscribers += len(getattr(store, "_subscribers", ()))
        handlers += len(getattr(session, "_handlers", ()))
        task = source.connection_task
        if task is not None and not task.done():
            connection_tasks += 1
    sizes["sources.frontend_subscribers"] = subscribers
    sizes["sources.event_handlers"] = handlers
    sizes["sources.retired_but_referenced"] = retired_sources
    sizes["sources.live_connection_tasks"] = connection_tasks
    # Owner side (in-process here; a separate process in production, so any
    # growth here would NOT be cured by /reload — recorded to eliminate it).
    owner_clients = 0
    owner_subs = 0
    for server in FIXTURE.servers.values():
        owner_clients += len(server._clients)
        session = getattr(server._handle, "_session", None)
        store = getattr(session, "_frontend_state_store", None)
        if store is not None:
            owner_subs += len(getattr(store, "_subscribers", ()))
        owner_subs += len(getattr(session, "_handlers", ()))
    sizes["owner.server._clients"] = owner_clients
    sizes["owner.session.subscribers+handlers"] = owner_subs
    # Module-level caches.
    from local_operator.session import catalog as catalog_mod
    from local_operator.session import frontend_state as fs_mod

    size("catalog._ROW_CACHE", catalog_mod._ROW_CACHE)
    size("frontend_state._DERIVED_OWNERSHIP", fs_mod._DERIVED_OWNERSHIP)
    size("frontend_state._STATE_FIELD_ADAPTERS", fs_mod._STATE_FIELD_ADAPTERS)
    return sizes


def dom_snapshot(app: OperatorApp) -> dict[str, Any]:
    from local_operator.tui.widgets.transcript import TranscriptView

    widgets = _all_widgets(app)
    by_type: Counter[str] = Counter(type(w).__name__ for w in widgets)
    views = [w for w in widgets if isinstance(w, TranscriptView)]
    return {
        "nodes": len(widgets),
        "query_all": len(app.query("*")),
        "transcript_views": len(views),
        "transcript_blocks_total": sum(len(v.blocks()) for v in views),
        "screen_stack": len(app.screen_stack),
        "top_types": dict(by_type.most_common(15)),
    }


def gc_snapshot() -> Counter[str]:
    gc.collect()
    return Counter(type(o).__name__ for o in gc.get_objects())


def gc_delta(base: Counter[str], now: Counter[str], top: int = 25) -> list[tuple[str, int, int]]:
    rows = []
    for name, count in now.items():
        delta = count - base.get(name, 0)
        if delta > 0:
            rows.append((name, delta, count))
    rows.sort(key=lambda r: -r[1])
    return rows[:top]


class Fixture:
    def __init__(self, count: int, history: int, busy: bool) -> None:
        self.count = count
        self.history = history
        self.busy = busy
        self.servers: dict[str, RuntimeServer] = {}
        self.ids: list[str] = []

    async def start(self) -> None:
        for i in range(self.count):
            sid = f"growth{i:03d}"
            self.ids.append(sid)
            directory = CONFIG / "sessions" / sid
            messages = []
            for j in range(self.history // 2):
                messages.append(user_message(f"{sid} question {j}: " + "context words " * 12))
                messages.append(assistant_message(f"{sid} answer {j}: " + "reply words " * 25))
            await seed_transcript(directory, messages)
            owner = build_session(directory, ScriptedStream([]), cwd=CONFIG)
            handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(CONFIG))
            server = RuntimeServer(handle, kind="daemon")
            await server.start_in_process()
            if self.busy:
                # Flip the published record to busy so the sidebar's spinner
                # ticks for these rows; the owner itself stays idle.
                server._busy = True
                server._republish()
            self.servers[sid] = server

    def find(self, _directory: Path, sid: str) -> tuple[Any, Any]:
        server = self.servers.get(sid)
        return (server._record, server._record.pid) if server else (None, None)

    async def resume(self, sid: str | None) -> RemoteSession:
        async def never() -> Any:
            raise AssertionError("view navigation must never take execution ownership")

        assert sid is not None
        return await RemoteSession.connect(
            self.servers[sid]._record,
            sid,
            config_dir=CONFIG,
            takeover_factory=never,
            display_window=True,
        )

    async def close(self) -> None:
        for server in self.servers.values():
            await server.aclose()


PREV_GC: list[Counter[str]] = []


def per_source_snapshot(app: OperatorApp) -> dict[str, int]:
    """Max over sources of every per-source list/dict that could accumulate."""
    out: Counter[str] = Counter()
    seen: dict[int, Any] = {}
    for source in [*app._sidebar_sources.values(), *app._interactions.values()]:
        seen[id(source)] = source
    for source in seen.values():
        d = source.draft
        out["draft.recoveries"] = max(out["draft.recoveries"], len(d.recoveries))
        out["draft.notices"] = max(out["draft.notices"], len(d.notices))
        out["draft.attachments"] = max(out["draft.attachments"], len(d.attachments))
        out["turn.pending_echoes"] = max(
            out["turn.pending_echoes"], len(source.turn.pending_echoes)
        )
        out["turn.settled_child_ids"] = max(
            out["turn.settled_child_ids"], len(source.turn.settled_child_ids)
        )
        out["accounting.child_costs"] = max(
            out["accounting.child_costs"], len(source.accounting.child_costs)
        )
        session = source.session
        for name in (
            "_buffered_events",
            "_pending_frontend",
            "_snapshot_clients",
            "_handlers",
            "_frontend_updates",
            "_display_history",
        ):
            value = getattr(session, name, None)
            if value is not None and hasattr(value, "__len__"):
                out[f"session.{name}"] = max(out[f"session.{name}"], len(value))
        store = getattr(session, "_frontend_store", None)
        if store is not None:
            out["store._subscribers"] = max(out["store._subscribers"], len(store._subscribers))
    for presentation in app._sidebar_presentations.values():
        view = presentation.replay.view
        out["view.blocks"] = max(out["view.blocks"], len(view.blocks()))
        for name in ("_blocks", "_pending", "_gap_rows", "_anchors"):
            value = getattr(view, name, None)
            if value is not None and hasattr(value, "__len__"):
                out[f"view.{name}"] = max(out[f"view.{name}"], len(value))
    return dict(out)


async def checkpoint(
    app: OperatorApp, label: str, base_gc: Counter[str] | None, extra: dict[str, Any]
) -> dict[str, Any]:
    now_gc = gc_snapshot()
    record: dict[str, Any] = {
        "label": label,
        "rss_mb": round(rss_mb(), 1),
        "gc_objects": sum(now_gc.values()),
        "gc_counts": list(gc.get_count()),
        "timers": timers_snapshot(app),
        "tasks": tasks_snapshot(),
        "dom": dom_snapshot(app),
        "containers": containers_snapshot(app),
        "counts": dict(COUNTS),
        "frontend_objects": {
            k: now_gc.get(k, 0)
            for k in (
                "FrontendSync",
                "FrontendSessionState",
                "RemoteSession",
                "AttachClient",
                "SessionInteraction",
                "SessionPresentation",
                "PreparedReplay",
                "EventController",
                "StylesCache",
                "Style",
                "TranscriptView",
            )
        },
        **extra,
    }
    record["per_source"] = per_source_snapshot(app)
    if base_gc is not None:
        record["gc_top_growers"] = gc_delta(base_gc, now_gc)
    if PREV_GC:
        record["gc_top_growers_prev"] = gc_delta(PREV_GC[-1], now_gc, top=15)
    PREV_GC.append(now_gc)
    record["_gc"] = now_gc
    return record


async def idle_cost(app: OperatorApp, pilot: Any) -> dict[str, float]:
    """What the app burns doing NOTHING — the honest measure of "it feels slow".

    Loop-thread CPU over a fixed idle window (AGENTS.md "measure CPU, not wall
    time"): timers, polls and background tasks are the only things that can
    consume it, so a number that rises with switch count is background work
    the app acquired and never released.
    """
    seconds = ARGS.idle_seconds
    if seconds <= 0:
        return {}
    cpu0, wall0 = time.thread_time(), time.monotonic()
    while time.monotonic() - wall0 < seconds:
        await pilot.pause(0.02)
    elapsed = time.monotonic() - wall0
    cpu = time.thread_time() - cpu0
    result = {
        "idle_cpu_ms_per_s": round(cpu / elapsed * 1000, 1),
        "idle_window_s": round(elapsed, 2),
    }
    SERIES.setdefault("idle_cpu_ms_per_s", []).append(result["idle_cpu_ms_per_s"])
    return result


async def keystroke_cost(app: OperatorApp, pilot: Any) -> dict[str, float]:
    """Perceived latency: how long one keypress takes to be fully serviced."""
    n = ARGS.keystrokes
    if n <= 0:
        return {}
    editor = app._editor()
    editor.focus()
    await pilot.pause()
    samples: list[float] = []
    for _ in range(n):
        start = time.monotonic()
        await pilot.press("x")
        samples.append((time.monotonic() - start) * 1000)
    samples.sort()
    editor.text = ""
    await pilot.pause()
    result = {
        "key_median_ms": round(samples[len(samples) // 2], 2),
        "key_p90_ms": round(samples[int(len(samples) * 0.9)], 2),
        "key_max_ms": round(samples[-1], 2),
    }
    SERIES.setdefault("key_median_ms", []).append(result["key_median_ms"])
    return result


async def one_switch(app: OperatorApp, pilot: Any, sid: str) -> dict[str, float]:
    hit_before = COUNTS["presentation_hit"]
    nav_before = COUNTS["prepare(navigation)"]
    cpu0, wall0 = time.thread_time(), time.monotonic()
    task = app._sidebar_navigation.select(sid)
    await asyncio.wait_for(task, 30)
    committed = time.monotonic()
    # Let the post-commit connection task (saved view -> canonical) settle so
    # the switch is fully complete before the next one starts.
    source = app._interaction
    if source.connection_task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(source.connection_task), 30)
        except Exception:
            pass
    await pilot.pause()
    result = {
        "switch_commit_wall_ms": round((committed - wall0) * 1000, 1),
        "switch_total_wall_ms": round((time.monotonic() - wall0) * 1000, 1),
        "switch_loop_cpu_ms": round((time.thread_time() - cpu0) * 1000, 1),
    }
    result["switch_hit"] = COUNTS["presentation_hit"] - hit_before
    result["switch_navigation_prepares"] = COUNTS["prepare(navigation)"] - nav_before
    SERIES.setdefault("switch_hit", []).append(result["switch_hit"])
    SERIES["switch_total_wall_ms"].append(result["switch_total_wall_ms"])
    SERIES["switch_loop_cpu_ms"].append(result["switch_loop_cpu_ms"])
    return result


async def one_poll(app: OperatorApp, pilot: Any) -> dict[str, float]:
    """One catalog poll + prewarm + a spinner tick, timed on the loop thread."""
    cpu0, wall0 = time.thread_time(), time.monotonic()
    app._refresh_sidebar()
    # Wait for the worker that _refresh_sidebar launched.
    for _ in range(400):
        if not app._sidebar_refresh_pending:
            break
        await pilot.pause()
    prefetch = app._sidebar_prefetch
    if prefetch is not None:
        try:
            await asyncio.wait_for(asyncio.shield(prefetch.wait()), 30)
        except Exception:
            pass
    app._session_sidebar._advance_spinner()
    await pilot.pause()
    result = {
        "poll_wall_ms": round((time.monotonic() - wall0) * 1000, 1),
        "poll_loop_cpu_ms": round((time.thread_time() - cpu0) * 1000, 1),
    }
    SERIES["poll_wall_ms"].append(result["poll_wall_ms"])
    SERIES["poll_loop_cpu_ms"].append(result["poll_loop_cpu_ms"])
    return result


def write_report(records: list[dict[str, Any]], path: Path) -> None:
    lines = ["# TUI growth checkpoints", ""]
    keys = [
        ("rss_mb", lambda r: r["rss_mb"]),
        ("gc_objects", lambda r: r["gc_objects"]),
        ("timers", lambda r: r["timers"]["total"]),
        ("timers_running", lambda r: r["timers"]["running"]),
        ("tasks", lambda r: r["tasks"]["total"]),
        ("workers", lambda r: r["containers"]["app.workers"]),
        ("dom_nodes", lambda r: r["dom"]["nodes"]),
        ("transcript_views", lambda r: r["dom"]["transcript_views"]),
        ("blocks_total", lambda r: r["dom"]["transcript_blocks_total"]),
        ("sidebar_sources", lambda r: r["containers"]["app._sidebar_sources"]),
        ("presentations", lambda r: r["containers"]["app._sidebar_presentations"]),
        ("interactions", lambda r: r["containers"]["app._interactions"]),
        ("event_sources", lambda r: r["containers"]["app._event_sources"]),
        ("frontend_subs", lambda r: r["containers"]["sources.frontend_subscribers"]),
        ("event_handlers", lambda r: r["containers"]["sources.event_handlers"]),
        ("retired_refd", lambda r: r["containers"]["sources.retired_but_referenced"]),
        ("prior_workers", lambda r: r["containers"]["app._sidebar_prior_workers"]),
        ("painted_lines", lambda r: r["containers"]["sidebar._painted_lines"]),
        ("drafts_mem", lambda r: r["containers"]["app._sidebar_drafts._memory"]),
        ("row_cache", lambda r: r["containers"]["catalog._ROW_CACHE"]),
        ("owner_clients", lambda r: r["containers"]["owner.server._clients"]),
        ("owner_subs", lambda r: r["containers"]["owner.session.subscribers+handlers"]),
        ("app_queue", lambda r: r["containers"]["app._message_queue"]),
        ("displayed_frame_ids", lambda r: r["containers"]["app._sidebar_displayed_frame.painted"]),
        ("gc_widgets", lambda r: r["containers"]["gc.Widget_total"]),
        ("compositor_map", lambda r: r["containers"].get("compositor._full_map", "")),
        ("threads", lambda r: r["containers"]["threads"]),
    ]
    header = "| metric | " + " | ".join(r["label"] for r in records) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(records) + 1))
    for name, fn in keys:
        lines.append(f"| {name} | " + " | ".join(str(fn(r)) for r in records) + " |")
    timing_keys = sorted(
        {k for r in records for k in r if k.endswith("_ms")},
    )
    for name in timing_keys:
        lines.append(f"| {name} | " + " | ".join(str(r.get(name, "")) for r in records) + " |")
    lines.append("")
    lines.append("## Work counters (cumulative)")
    count_keys = sorted({k for r in records for k in r["counts"]})
    lines.append("| counter | " + " | ".join(r["label"] for r in records) + " |")
    lines.append("|" + "---|" * (len(records) + 1))
    for k in count_keys:
        lines.append(f"| {k} | " + " | ".join(str(r["counts"].get(k, 0)) for r in records) + " |")
    lines.append("")
    lines.append("## Object counts")
    lines.append("| type | " + " | ".join(r["label"] for r in records) + " |")
    lines.append("|" + "---|" * (len(records) + 1))
    for k in records[0]["frontend_objects"]:
        lines.append(
            f"| {k} | " + " | ".join(str(r["frontend_objects"][k]) for r in records) + " |"
        )
    lines.append("")
    lines.append("## Timers by owner / callback")
    for r in records:
        lines.append(f"- {r['label']}: {r['timers']['by_owner']}")
        lines.append(f"  - {r['timers']['by_callback']}")
    lines.append("")
    lines.append("## Tasks by coroutine (last checkpoint)")
    for name, n in records[-1]["tasks"]["by_coro"].items():
        lines.append(f"- {n} × {name}")
    lines.append("")
    lines.append("## DOM top types (first vs last)")
    first, last = records[0]["dom"]["top_types"], records[-1]["dom"]["top_types"]
    for name in sorted(set(first) | set(last), key=lambda n: -(last.get(n, 0) - first.get(n, 0))):
        lines.append(f"- {name}: {first.get(name, 0)} -> {last.get(name, 0)}")
    lines.append("")
    lines.append("## threads by name")
    tkeys = sorted({k for r in records for k in r["containers"] if k.startswith("thread:")})
    lines.append("| thread | " + " | ".join(r["label"] for r in records) + " |")
    lines.append("|" + "---|" * (len(records) + 1))
    for k in tkeys:
        lines.append(
            f"| {k} | " + " | ".join(str(r["containers"].get(k, 0)) for r in records) + " |"
        )
    lines.append("")
    lines.append("## per-source / per-view container maxima")
    ps_keys = sorted({k for r in records for k in r["per_source"]})
    lines.append("| container | " + " | ".join(r["label"] for r in records) + " |")
    lines.append("|" + "---|" * (len(records) + 1))
    for k in ps_keys:
        lines.append(
            f"| {k} | " + " | ".join(str(r["per_source"].get(k, "")) for r in records) + " |"
        )
    lines.append("")
    lines.append("## gc growers vs PREVIOUS checkpoint")
    for r in records[1:]:
        lines.append(
            f"- {r['label']}: "
            + ", ".join(f"{n}+{d}" for n, d, _ in r.get("gc_top_growers_prev", [])[:12])
        )
    lines.append("")
    lines.append("## gc top growers vs baseline (last checkpoint)")
    for name, delta, total in records[-1].get("gc_top_growers", []):
        lines.append(f"- {name}: +{delta} (now {total})")
    path.write_text("\n".join(lines) + "\n")


async def run_switches(fixture: Fixture) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    app = OperatorApp(lambda: fixture.resume(fixture.ids[0]), resume_factory=fixture.resume)
    profiler_early = cProfile.Profile() if ARGS.profile else None
    profiler_late = cProfile.Profile() if ARGS.profile else None
    with (
        patch("local_operator.mobile.attach_client.find_runtime_record", fixture.find),
        patch.object(OperatorApp, "_check_for_update", lambda self: None),
    ):
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for_adoption(app, pilot)
            await pilot.pause()
            if ARGS.approve_all:
                app._approvals_default_auto = True
                app._set_approve_all(True)
            app._set_sidebar_open(True)
            # Let the first catalog poll + prewarm land so the baseline
            # includes steady-state sidebar structures.
            await one_poll(app, pilot)
            for _ in range(20):
                await pilot.pause()
            if ARGS.tracemalloc:
                tracemalloc.start(25)
            base_gc = gc_snapshot()
            snap0 = tracemalloc.take_snapshot() if ARGS.tracemalloc else None
            base_extra = await one_poll(app, pilot)
            base_extra.update(await idle_cost(app, pilot))
            base_extra.update(await keystroke_cost(app, pilot))
            records.append(await checkpoint(app, "K=0", None, base_extra))
            done = 0
            targets = ARGS.switches
            total = max(targets)
            pool = fixture.ids[: ARGS.cycle] if ARGS.cycle else fixture.ids
            for k in range(1, total + 1):
                sid = pool[k % len(pool)]
                if sid == getattr(app._session, "session_id", ""):
                    sid = pool[(k + 1) % len(pool)]
                prof = None
                if profiler_early is not None and k == 5:
                    prof = profiler_early
                if profiler_late is not None and k == total:
                    prof = profiler_late
                if prof is not None:
                    prof.enable()
                timing = await one_switch(app, pilot, sid)
                if prof is not None:
                    prof.disable()
                done = k
                if k in targets:
                    for _ in range(10):
                        await pilot.pause()
                    poll = await one_poll(app, pilot)
                    idle = await idle_cost(app, pilot)
                    keys = await keystroke_cost(app, pilot)
                    records.append(
                        await checkpoint(app, f"K={k}", base_gc, {**timing, **poll, **idle, **keys})
                    )
            if snap0 is not None:
                snap1 = tracemalloc.take_snapshot()
                stats = snap1.compare_to(snap0, "traceback")
                out = []
                for stat in stats[:20]:
                    out.append(f"{stat.size_diff / 1024:+.1f} KiB, {stat.count_diff:+d} blocks")
                    out.extend("    " + line for line in stat.traceback.format()[-6:])
                (ARGS.output / "tracemalloc.txt").write_text("\n".join(out) + "\n")
                tracemalloc.stop()
            if profiler_early is not None:
                profiler_early.dump_stats(str(ARGS.output / "profile-early.prof"))
            if profiler_late is not None:
                profiler_late.dump_stats(str(ARGS.output / "profile-late.prof"))
            print(f"switches done: {done}", flush=True)
    return records


async def run_uptime(fixture: Fixture) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    app = OperatorApp(lambda: fixture.resume(fixture.ids[0]), resume_factory=fixture.resume)
    with (
        patch("local_operator.mobile.attach_client.find_runtime_record", fixture.find),
        patch.object(OperatorApp, "_check_for_update", lambda self: None),
    ):
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for_adoption(app, pilot)
            await pilot.pause()
            if ARGS.approve_all:
                app._approvals_default_auto = True
                app._set_approve_all(True)
            app._set_sidebar_open(True)
            assert app._sidebar_timer is not None
            # The harness drives polls explicitly so cycles are countable.
            app._sidebar_timer.pause()
            await one_poll(app, pilot)
            for _ in range(20):
                await pilot.pause()
            if ARGS.tracemalloc:
                tracemalloc.start(25)
            base_gc = gc_snapshot()
            snap0 = tracemalloc.take_snapshot() if ARGS.tracemalloc else None
            base_extra = await one_poll(app, pilot)
            base_extra.update(await idle_cost(app, pilot))
            base_extra.update(await keystroke_cost(app, pilot))
            records.append(await checkpoint(app, "P=0", None, base_extra))
            for p in range(1, ARGS.polls + 1):
                timing = await one_poll(app, pilot)
                # Extra spinner ticks between polls: the 2 s poll sees ~16
                # spinner frames at 120 ms.
                for _ in range(15):
                    app._session_sidebar._advance_spinner()
                    await pilot.pause()
                if p % ARGS.checkpoint_every == 0 or p == ARGS.polls:
                    idle = await idle_cost(app, pilot)
                    keys = await keystroke_cost(app, pilot)
                    records.append(
                        await checkpoint(app, f"P={p}", base_gc, {**timing, **idle, **keys})
                    )
            if snap0 is not None:
                snap1 = tracemalloc.take_snapshot()
                stats = snap1.compare_to(snap0, "traceback")
                out = []
                for stat in stats[:20]:
                    out.append(f"{stat.size_diff / 1024:+.1f} KiB, {stat.count_diff:+d} blocks")
                    out.extend("    " + line for line in stat.traceback.format()[-6:])
                (ARGS.output / "tracemalloc.txt").write_text("\n".join(out) + "\n")
                tracemalloc.stop()
    return records


FIXTURE: Fixture


async def main() -> None:
    global FIXTURE
    instrument()
    fixture = Fixture(ARGS.sessions, ARGS.history, busy=(ARGS.mode == "uptime" or ARGS.approve_all))
    FIXTURE = fixture
    await fixture.start()
    try:
        records = (
            await run_switches(fixture) if ARGS.mode == "switches" else await run_uptime(fixture)
        )
    finally:
        await fixture.close()
    for r in records:
        r.pop("_gc", None)
    (ARGS.output / "checkpoints.json").write_text(json.dumps(records, indent=1, default=str) + "\n")
    (ARGS.output / "series.json").write_text(json.dumps(SERIES, indent=1) + "\n")
    write_report(records, ARGS.output / "report.md")
    print((ARGS.output / "report.md").read_text())
    for name, values in SERIES.items():
        if not values:
            continue
        n = len(values)
        q = max(1, n // 4)
        quartiles = [
            round(sum(values[i : i + q]) / len(values[i : i + q]), 1) for i in range(0, n, q)
        ]
        print(
            f"series {name}: n={n} quartile-means={quartiles} "
            f"first5={values[:5]} last5={values[-5:]}"
        )
    print("source:", local_operator.__file__)


if __name__ == "__main__":
    asyncio.run(main())
