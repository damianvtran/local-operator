"""Measure TIME TO FIRST TOKEN on every path a user actually lives on, and gate it.

WHAT THIS MEASURES, AND WHY IT IS THE NUMBER THAT MATTERS
========================================================
The complaint is not throughput or total turn time — it is the pause between
hitting Enter and seeing the first streamed character. That pause has a different
shape on every front end, and they do not share a fix:

* ``tui``        — in-process ``Session``: submit to the first event the front end
  can paint. The floor: no HTTP, no runtime process, no bridge.
* ``desktop``    — the app's first message in a session with no runtime (cold: the
  POST carries the engage — spawn, import, construct, bind) versus a later message
  on the same session (warm).
* ``exec``       — the one-shot console entry point, ``--json``, read line by line.
* ``mobile``     — the phone's real routes: the relay's projection frames.
* ``sse-jobs``   — ``POST /v1/chat/async`` then ``GET /v1/sse/jobs/{id}``.

WHAT IS FAKED, AND WHY THAT IS THE POINT
========================================
The PROVIDER is a loopback OpenAI-compatible endpoint this harness serves itself
(``scripts/ttft/loopback.py``) and points the real ``openai-compatible`` local
provider at through the config the app already supports. Everything measured is
therefore local-operator's own overhead — imports, prompt construction, engage,
IPC, JSON, SSE — which is the only part this repository can move. A live provider
would bury the signal under its own time-to-first-byte (measured: 355-514 ms on a
102-token prompt, 1.55 s on the operator's 227k-token cached p50 prompt — see
``scripts/probe_provider_ttfb.py``) and make before/after incomparable.

The loopback endpoint is used instead of the built-in ``test`` mock for one
reason: the mock emits text and NOTHING else, so a mock-provider bench has no
reasoning delta to be dropped and cannot see the wait the operator is describing.
``--provider test`` still selects the mock, for continuity with the numbers that
predate this harness.

Everything else is real: real uvicorn daemon, real HTTP + SSE, real spawned
``python -m local_operator.session.runtime.process`` children, real console entry
point, real phone routes, real transcript.

WHAT IS REPORTED AND WHAT IS ASSERTED
=====================================
Only :data:`scripts.ttft.metrics.FIRST_EVENT` against a 300 ms budget, only on the
cells whose floor this repository owns end to end, and only on the p50. The
PROVIDER columns are reported and budgeted, never asserted — read
``metrics.BUDGET_MS`` before changing that; it explains, with the measured numbers,
why asserting a sub-300 ms first provider token would be a lie.

ISOLATION
=========
Every run gets a fresh ``HOME``, ``LOCAL_OPERATOR_CONFIG_DIR`` and ``TMPDIR``, and
every inherited ``CMUX_*``/``LOP_*`` variable is stripped from this process before
anything is spawned. The operator's live sessions are never touched and a measured
child can never attach to a real conversation. See ``scripts/ttft/isolate.py``.

USAGE
=====
    .venv/bin/python scripts/bench_ttft.py --runs 7
    .venv/bin/python scripts/bench_ttft.py --channels tui,desktop --runs 7 --concurrency 1,4,8
    .venv/bin/python scripts/bench_ttft.py --channels tui --provider test --runs 7
    .venv/bin/python scripts/bench_ttft.py --channels desktop --provider-reasoning-ms 400

Report percentiles, never a single run. The first run in a process pays for cold
page cache on the interpreter and site-packages, and the distribution has a long
right tail.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# A benchmark under scripts/ must read the tree it lives in, not whatever tree the
# venv was installed from (AGENTS.md, "Every feature worktree owns its own venv").
# Without this a benchmark run in a worktree silently measures main.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ttft import metrics as M  # noqa: E402
from scripts.ttft.channels import (  # noqa: E402
    CHANNELS,
    ChannelConfig,
    drive_channel,
    tui_child_main,
)
from scripts.ttft.isolate import (  # noqa: E402
    make_run,
    pin_shared_caches,
    strip_inherited_runtime_env,
)
from scripts.ttft.loopback import LoopbackProvider  # noqa: E402
from scripts.ttft.report import render_report  # noqa: E402

#: Minimum runs before a cell is worth reading. The task's own bar; a shorter run
#: is allowed for a smoke test and says so in the notes.
MIN_RUNS = 7

#: The built-in test provider. Kept as a selectable arm for continuity with the
#: numbers that predate this harness; it cannot show a reasoning phase.
MOCK_HOSTING = "test"
MOCK_MODEL = "mock"


def _prime_bytecode_cache() -> None:
    """Populate the run's bytecode cache through the production entry point.

    Exists so "before" and "after" can be compared under the desktop app's real
    arrangement: an interpreter that REFUSES bytecode writes, and a cache that
    something long-lived populates once for the children that follow. Run as a
    subprocess because that is how the daemon does it. Absent on a tree without this
    module — reported, not fatal, so the same script can measure both arms.
    """
    env = dict(os.environ)
    env["LOP_TTFT_REPO"] = str(Path(__file__).resolve().parents[1])
    code = (
        "import os, sys\n"
        "sys.path.insert(0, os.environ['LOP_TTFT_REPO'])\n"
        "try:\n"
        "    from local_operator.bytecode import warm_bytecode_cache_in_background\n"
        "except ImportError:\n"
        "    print('NO_MODULE')\n"
        "    raise SystemExit(0)\n"
        "thread = warm_bytecode_cache_in_background()\n"
        "if thread is None:\n"
        "    print('ALREADY_WARM')\n"
        "else:\n"
        "    thread.join(600)\n"
        "    print('PRIMED')\n"
    )
    completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", code],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=900,
        check=False,
    )
    print(f"  bytecode prime: {completed.stdout.decode().strip() or 'FAILED'}", flush=True)


def _host_context() -> dict[str, Any]:
    """The machine the numbers were taken on, because an absolute ms needs it.

    This host is shared with ~25 concurrent agent sessions and swings between load
    20 and 600, so a table without its load is not evidence of anything.
    """
    try:
        load = list(os.getloadavg())
    except (OSError, AttributeError):
        load = []
    return {
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "load_at_start": load,
        "python": sys.version.split()[0],
    }


async def _one_run(
    *,
    channel: str,
    arms: tuple[str, ...],
    concurrency: int,
    config: ChannelConfig,
    diagnostics: dict[str, Any],
    root_prefix: str,
) -> list[dict[str, Any]]:
    """One isolated run of one channel: fresh root, then the channel's turns."""
    run = make_run(prefix=root_prefix)
    try:
        run.activate()
        run.seed(
            hosting=config.hosting,
            model=config.model,
            base_url=config.provider.base_url if config.provider is not None else None,
        )
        os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = secrets.token_hex(32)
        return await drive_channel(
            channel,
            run,
            config=config,
            concurrency=concurrency,
            arms=arms,
            diagnostics=diagnostics,
        )
    finally:
        run.teardown()


def _reduce_cells(
    samples_by_cell: dict[tuple[str, str, int], list[dict[str, Any]]], runs: int
) -> list[dict[str, Any]]:
    """Reduce every cell's pooled samples into percentiles and a budget verdict."""
    cells: list[dict[str, Any]] = []
    for (channel, arm, concurrency), samples in sorted(samples_by_cell.items()):
        stats = {
            metric: M.reduce_samples(
                [float(sample[metric]) for sample in samples if metric in sample]
            )
            for metric in M.METRICS
        }
        verdict = M.judge((channel, arm), stats, concurrency=concurrency)
        warnings: list[str] = []
        # INSTRUMENT SELF-CHECK: the provider must have put text on the wire
        # BEFORE any front end could receive it. The reverse is physically
        # impossible, so it can only mean the provider column is paired to the
        # wrong request — a harness defect that would otherwise be read as a
        # product result.
        provider_text = stats.get(M.PROVIDER_TEXT) or {}
        first_text = stats.get(M.FIRST_TEXT) or {}
        if (
            provider_text.get("n")
            and first_text.get("n")
            and float(provider_text["p50"]) > float(first_text["p50"]) + 5
        ):
            warnings.append(
                f"provider_text p50 {provider_text['p50']} > first_text p50 "
                f"{first_text['p50']} — provider stamps mis-paired, numbers not usable"
            )
        # THE FINDING, stated per cell: reasoning existed on the wire and no front
        # end received it.
        produced = stats.get(M.PROVIDER_REASONING) or {}
        surfaced = stats.get(M.FIRST_REASONING) or {}
        if produced.get("n") and not surfaced.get("n"):
            warnings.append(
                f"provider emitted reasoning at p50 {produced['p50']} ms and NO front end "
                "received it: the turn's first visible event is its first TEXT"
            )
        cells.append(
            {
                "channel": channel,
                "arm": arm,
                "concurrency": concurrency,
                "runs": runs,
                "samples": len(samples),
                "stats": stats,
                "warnings": warnings,
                "verdict": {
                    "status": verdict.status,
                    "reason": verdict.reason,
                    "budget_ms": M.BUDGET_MS,
                },
            }
        )
    return cells


async def _amain(args: argparse.Namespace) -> int:
    channels = [name.strip() for name in args.channels.split(",") if name.strip()]
    for name in channels:
        if name not in CHANNELS:
            raise SystemExit(f"unknown channel {name!r}; known: {', '.join(CHANNELS)}")
    concurrencies = [int(value) for value in args.concurrency.split(",") if value.strip()]
    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())
    for arm in arms:
        if arm not in ("cold", "warm"):
            raise SystemExit(f"unknown arm {arm!r}; known: cold, warm")

    from scripts import bench_tree

    tree = bench_tree.describe(args.measured_tree)
    notes: list[str] = []
    if args.runs < MIN_RUNS:
        notes.append(
            f"{args.runs} runs is below the {MIN_RUNS} this harness calls enough; "
            "percentiles here are indicative only"
        )
    stripped = strip_inherited_runtime_env()
    if stripped:
        notes.append(f"stripped inherited {', '.join(sorted(stripped))}")
    if not args.prime_bytecode:
        notes.append(
            "bytecode cache not primed (--prime-bytecode): every measured child pays "
            "compilation the desktop app's daemon pays once for it"
        )

    pycache = Path(args.pycache_prefix or tempfile.mkdtemp(prefix="lop-ttft-bench-pycache-"))
    # A STABLE tokenizer cache across invocations when one is given: tiktoken
    # downloads its BPE table unless it finds the file, so a fresh directory per
    # invocation makes the first measured child pay a TLS round trip (413 ms of
    # SSLSocket.read, measured). Pinning it per CAMPAIGN rather than per invocation
    # is what lets a five-channel sweep be compared without that cost in run 1.
    tiktoken = Path(args.tiktoken_cache or tempfile.mkdtemp(prefix="lop-ttft-bench-tiktoken-"))
    # What THIS invocation created, it removes. A cache directory that outlives the
    # run that made it is a leak on a shared disk, and the two here are large (a
    # bytecode cache per campaign, a tokenizer table); a caller-pinned one is the
    # caller's to keep. The roots under /tmp are removed per run by
    # IsolatedRun.teardown; these two live for the whole invocation instead.
    created_caches = [
        path
        for path, given in ((pycache, args.pycache_prefix), (tiktoken, args.tiktoken_cache))
        if not given
    ]
    pin_shared_caches(pycache, tiktoken)

    provider: LoopbackProvider | None = None
    if args.provider == "loopback":
        provider = LoopbackProvider(
            model=args.model or "bench-loopback",
            prefill_ms=args.provider_prefill_ms,
            reasoning_ms=args.provider_reasoning_ms,
        )
        await provider.start()
        config = ChannelConfig(
            hosting=args.hosting or "openai-compatible",
            model=provider.model,
            provider=provider,
        )
    else:
        config = ChannelConfig(hosting=MOCK_HOSTING, model=MOCK_MODEL, provider=None)
        notes.append(
            "provider=test: the built-in mock emits text only, so the reasoning "
            "columns are unobservable rather than absent"
        )

    report: dict[str, Any] = {
        "tree": tree,
        "host": _host_context(),
        "provider": {
            "kind": args.provider,
            "hosting": config.hosting,
            "model": config.model,
            "emulated_prefill_ms": args.provider_prefill_ms,
            "emulated_reasoning_ms": args.provider_reasoning_ms,
            "note": (
                "emulated provider floors; 0 means the numbers are local-operator's "
                "own overhead, which is the gate configuration"
            ),
        },
        "runs": args.runs,
        "concurrency": concurrencies,
        "arms": list(arms),
        "notes": notes,
        "cells": [],
    }

    samples_by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    printed_banner = False
    try:
        for channel in channels:
            for concurrency in concurrencies:
                for run_index in range(args.runs):
                    diagnostics: dict[str, Any] = {}
                    samples = await _one_run(
                        channel=channel,
                        arms=arms,
                        concurrency=concurrency,
                        config=config,
                        diagnostics=diagnostics,
                        root_prefix=args.root_prefix,
                    )
                    for sample in samples:
                        key = (channel, str(sample.get("arm", "")), concurrency)
                        samples_by_cell.setdefault(key, []).append(sample)
                    shown = " ".join(
                        f"{metric.replace('_ms', '')}={_p50(samples, metric):.0f}"
                        for metric in (M.FIRST_EVENT, M.FIRST_REASONING, M.FIRST_TEXT)
                    )
                    print(
                        f"  {channel} {concurrency}x run {run_index + 1}/{args.runs}: {shown}",
                        flush=True,
                    )
                    if diagnostics and not printed_banner:
                        print(f"  instrumentation: {json.dumps(diagnostics)[:400]}", flush=True)
                        printed_banner = True
                # Persist after every cell, so a run that dies at cell 30 still
                # hands over the 29 cells it measured (`ALREADY_WARM`-style honesty
                # about what was and was not taken).
                report["cells"] = _reduce_cells(samples_by_cell, args.runs)
                report["finished_cells"] = len(report["cells"])
                _write_json(args.json, report)
        report["cells"] = _reduce_cells(samples_by_cell, args.runs)
        report["load_at_end"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else []
        if provider is not None:
            # The provider's request log ships with the report: a mis-paired
            # provider column is then diagnosable from the artefact instead of
            # only by re-running the cell.
            report["provider"]["request_log"] = provider.log[-200:]
            report["provider"]["requests"] = provider.requests
        text = render_report(report)
        print("\n" + text)
        if args.table:
            Path(args.table).write_text(text + "\n", encoding="utf-8")
            print(f"\nwrote {args.table}")
    finally:
        if provider is not None:
            await provider.stop()
        for created in created_caches:
            _remove_quietly(created)
        _write_json(args.json, report)

    failures = [cell for cell in report["cells"] if cell["verdict"]["status"] == "FAIL"]
    if failures and args.assert_budget:
        print("\nBUDGET FAILURES:", file=sys.stderr)
        for cell in failures:
            print(
                f"  {cell['channel']}/{cell['arm']}@{cell['concurrency']}: "
                f"{cell['verdict']['reason']}",
                file=sys.stderr,
            )
        return 2
    return 0


def _render_only(paths: list[str], table: str) -> int:
    """Render one table from already-written report JSONs.

    Exists because the full sweep is a long run on a loaded host and is therefore
    taken one channel at a time: this folds those artefacts back into the single
    table a reviewer reads, without re-measuring anything. Percentiles are NOT
    recomputed — they are carried as they were reduced — so a combined table can
    never disagree with the run that produced it.
    """
    merged: dict[str, Any] = {"cells": [], "notes": []}
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        merged["cells"].extend(report.get("cells") or [])
        for key in ("tree", "host", "provider", "runs", "concurrency", "arms"):
            merged.setdefault(key, report.get(key))
        merged["load_at_end"] = report.get("load_at_end", merged.get("load_at_end"))
        for note in report.get("notes") or []:
            merged["notes"].append(f"{Path(path).name}: {note}")
    text = render_report(merged)
    print(text)
    if table:
        Path(table).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {table}")
    return 0


def _p50(samples: list[dict[str, Any]], metric: str) -> float:
    values = [
        float(sample[metric])
        for sample in samples
        if metric in sample and float(sample[metric]) != M.UNAVAILABLE
    ]
    return statistics.median(values) if values else -1.0


def _remove_quietly(path: Path, attempts: int = 3) -> None:
    """Remove a directory, retrying the way a bytecode cache needs.

    ``shutil.rmtree`` walks the tree and fails with "Directory not empty" if a file
    lands in a directory it already visited — measured on the bytecode cache, where
    a child that is still finishing its first import writes a ``.pyc`` under a
    prefix this process is sweeping. Retrying is the whole fix; the alternative
    (leaving it) is 9 MB per invocation of the largest cache this harness makes.
    """
    for attempt in range(attempts):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return
        time.sleep(0.2 * (attempt + 1))


def _write_json(path: str, report: dict[str, Any]) -> None:
    if not path:
        return
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels",
        type=str,
        default="tui,desktop,exec,mobile,sse-jobs",
        help=f"comma-separated channels; known: {', '.join(CHANNELS)}",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="cold,warm",
        help="comma-separated arms (cold, warm); a channel that cannot separate them "
        "reports what it has",
    )
    parser.add_argument("--runs", type=int, default=MIN_RUNS, help="runs per cell")
    parser.add_argument(
        "--concurrency",
        type=str,
        default="1,4,8",
        help="comma-separated numbers of concurrent turns",
    )
    parser.add_argument("--json", type=str, default="", help="write the raw report here")
    parser.add_argument(
        "--root-prefix",
        type=str,
        default="lop-ttft-bench-",
        help="prefix for this run's isolated roots under TMPDIR; distinct from the "
        "older copies of bench_ttft.py other worktrees run, so a leftover root can "
        "be attributed",
    )
    parser.add_argument(
        "--render-json",
        type=str,
        default="",
        help="comma-separated report JSONs to fold into ONE table instead of measuring; "
        "percentiles are carried, never recomputed",
    )
    parser.add_argument("--table", type=str, default="", help="write the rendered table here")
    parser.add_argument(
        "--measured-tree",
        type=str,
        default="",
        help="the rev whose local_operator/ subtree this run measures; VERIFIED against "
        "disk (scripts/bench_tree.py), so the artefact can never name a commit the run "
        "did not measure",
    )
    parser.add_argument(
        "--provider",
        choices=("loopback", "test"),
        default="loopback",
        help="loopback = the harness's own OpenAI-compatible endpoint (can show a "
        "reasoning phase); test = the built-in mock (text only)",
    )
    parser.add_argument(
        "--provider-prefill-ms",
        type=float,
        default=0.0,
        help="emulated provider time to the first reasoning token (0 = measure local "
        "overhead only, which is the gate configuration)",
    )
    parser.add_argument(
        "--provider-reasoning-ms",
        type=float,
        default=0.0,
        help="emulated additional provider time to the first TEXT token; a nonzero "
        "value demonstrates the invisible reasoning phase",
    )
    parser.add_argument("--hosting", type=str, default="", help="override the provider id")
    parser.add_argument("--model", type=str, default="", help="override the model id")
    parser.add_argument(
        "--pycache-prefix",
        type=str,
        default="",
        help="bytecode cache every process runs under (default: one temp dir per invocation)",
    )
    parser.add_argument(
        "--tiktoken-cache",
        type=str,
        default="",
        help="tokenizer cache directory (default: one temp dir per invocation); pin it "
        "per campaign so run 1 does not pay a download",
    )
    parser.add_argument(
        "--prime-bytecode",
        action="store_true",
        help="populate that cache once before measuring, as the daemon does",
    )
    parser.add_argument(
        "--no-assert-budget",
        dest="assert_budget",
        action="store_false",
        help="report only; do not exit non-zero on a budget failure",
    )
    parser.add_argument(
        "--child-tui",
        action="store_true",
        help="(internal) run the TUI arm in this process and report JSON",
    )
    parser.add_argument(
        "--concurrency-child",
        type=int,
        default=1,
        help="(internal) sessions the TUI child hosts concurrently",
    )
    parser.set_defaults(assert_budget=True)
    args = parser.parse_args()

    # THE PROCESS HALF OF THE NOTIFICATION GATE, and it is here rather than only on
    # the children because this process drives sessions itself: the desktop and
    # sse-jobs arms run their daemon in-process and the phone arm runs the mobile
    # daemon in-process, so the switch ``child_env`` puts on spawned children never
    # reaches them. A bench is a throwaway driver whose sessions nobody is watching,
    # and an ungated turn in it can put a session's own reply on the operator's lock
    # screen — the failure this repository has recorded 17 times from drive-by rigs.
    # Function-local import on purpose: ``tui.notify`` pulls the terminal and
    # settings modules in with it, and the shape is the one ``agent_shell`` uses for
    # the same reason.
    from local_operator.tui.notify import suppress_notifications_for_process

    suppress_notifications_for_process()

    if args.render_json:
        return _render_only(
            [path.strip() for path in args.render_json.split(",") if path.strip()], args.table
        )
    if args.child_tui:
        return asyncio.run(
            tui_child_main(
                concurrency=args.concurrency_child,
                arms=tuple(arm.strip() for arm in args.arms.split(",") if arm.strip()),
                hosting=args.hosting or "openai-compatible",
                model=args.model or "bench-loopback",
            )
        )
    if args.prime_bytecode:
        _prime_bytecode_cache()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
