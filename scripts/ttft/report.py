"""Render the harness's report as the table a reviewer reads on the PR.

TWO TABLES, AND WHY THE SECOND ONE EXISTS
========================================
The first table is the headline the ticket asks for: per channel, per warmth, per
concurrency, the ``p50/p95/p99`` of ``first_event``, ``first_reasoning`` and
``first_text``, with the run count and the sample count. Those three percentiles
are printed as one ``p50/p95/p99`` triple because the cell and the metric are what
a reader scans for, and a three-times-wider table stops being scannable.

The second table is the LEVEL DECOMPOSITION at p50: provider, runtime, front end,
and the daemon-side emission between them. It is what turns "it is slow" into
"this much of it is the model's, this much is ours, this much is the transport",
and it is where a reader sees the reasoning delta exist at the runtime level while
the front-end column stays ``-``.

A ``-`` is not a zero and never prints as one: it means the channel cannot observe
that metric here, and rendering it as ``0`` would report the opposite finding.
"""

from __future__ import annotations

from typing import Any, Mapping

from scripts.ttft import metrics as M

#: The channels in the order a reader wants them: cheapest first, then the ones
#: that pay a process.
CHANNEL_ORDER = ("tui", "desktop", "exec", "mobile", "sse-jobs")

#: The decomposition columns, in the order the wait accumulates.
LEVEL_METRICS = (
    M.PROVIDER_REASONING,
    M.RUNTIME_REASONING,
    M.STREAM_ENTERED,
    M.ADMITTED,
    M.FIRST_EVENT,
    M.RUNTIME_EMIT,
    M.PROVIDER_TEXT,
    M.FIRST_TEXT,
    M.TURN,
)


def _order(entry: Mapping[str, Any]) -> tuple[int, str, int]:
    channel = str(entry.get("channel", ""))
    index = CHANNEL_ORDER.index(channel) if channel in CHANNEL_ORDER else len(CHANNEL_ORDER)
    return (index, str(entry.get("arm", "")), int(entry.get("concurrency", 0)))


def _triple(stats: Mapping[str, Any] | None) -> str:
    """``p50/p95/p99`` for one metric, or ``-`` when it was never observed."""
    if not stats or not stats.get("n"):
        return "-"
    return f"{stats['p50']:.0f}/{stats['p95']:.0f}/{stats['p99']:.0f}"


def _plain(stats: Mapping[str, Any] | None) -> str:
    if not stats or not stats.get("n"):
        return "-"
    return f"{stats['p50']:.0f}"


def render_report(report: Mapping[str, Any]) -> str:
    """The whole rendered report: context, headline table, levels, verdicts, notes."""
    lines: list[str] = []
    tree = report.get("tree") or {}
    lines.append(
        f"measured tree: {str(tree.get('rev', 'unknown'))[:9]}  "
        f"(worktree HEAD {str(tree.get('worktree_head', 'unknown'))[:9]}, "
        f"subtree {tree.get('subtree', '?')}, verified={tree.get('verified')})"
    )
    host = report.get("host") or {}
    lines.append(
        f"host: {host.get('platform', '?')} / {host.get('cpus', '?')} cpus / "
        f"load {host.get('load_at_start', [])} -> {report.get('load_at_end', [])}"
    )
    provider = report.get("provider") or {}
    lines.append(
        f"provider: {provider.get('kind')} ({provider.get('hosting')}/"
        f"{provider.get('model')}) emulated prefill="
        f"{provider.get('emulated_prefill_ms')} ms reasoning="
        f"{provider.get('emulated_reasoning_ms')} ms"
    )
    lines.append(
        f"budget: {M.BUDGET_MS:.0f} ms on {M.FIRST_EVENT}, asserted on the p50 at "
        f"concurrency {list(M.ASSERTED_CONCURRENCY)} for {sorted(ENFORCED_NAMES)}; "
        "every other cell is measured and printed against the same budget"
    )
    lines.append(
        f"runs: {report.get('runs')} per cell; concurrency {report.get('concurrency')}; "
        f"arms {report.get('arms')}"
    )
    lines.append("")

    cells = list(report.get("cells") or [])

    header = (
        f"{'channel':<9} {'arm':<5} {'conc':>4} {'runs':>4} {'n':>4}  "
        + "  ".join(f"{name.replace('_ms', ''):>19}" for name in M.TABLE_METRICS)
        + "  verdict"
    )
    lines.append("HEADLINE — submit -> first event the front end can paint")
    lines.append(header)
    lines.append("-" * len(header))
    for entry in sorted(cells, key=_order):
        stats = entry.get("stats") or {}
        n = (stats.get(M.FIRST_EVENT) or {}).get("n", 0)
        columns = "  ".join(_triple(stats.get(metric)).rjust(19) for metric in M.TABLE_METRICS)
        lines.append(
            f"{str(entry.get('channel')):<9} {str(entry.get('arm')):<5} "
            f"{int(entry.get('concurrency', 0)):>4} {int(entry.get('runs', 0)):>4} {n:>4}  "
            f"{columns}  {(entry.get('verdict') or {}).get('status', '?')}"
        )

    lines.append("")
    lines.append("LEVELS — p50 of each stage, so the wait can be attributed")
    level_header = f"{'channel':<9} {'arm':<5} {'conc':>4}  " + "  ".join(
        f"{name.replace('_ms', ''):>12}" for name in LEVEL_METRICS
    )
    lines.append(level_header)
    lines.append("-" * len(level_header))
    for entry in sorted(cells, key=_order):
        stats = entry.get("stats") or {}
        columns = "  ".join(_plain(stats.get(metric)).rjust(12) for metric in LEVEL_METRICS)
        lines.append(
            f"{str(entry.get('channel')):<9} {str(entry.get('arm')):<5} "
            f"{int(entry.get('concurrency', 0)):>4}  {columns}"
        )

    lines.append("")
    lines.append("VERDICTS")
    for entry in sorted(cells, key=_order):
        verdict = entry.get("verdict") or {}
        lines.append(
            f"  {str(verdict.get('status', '?')):<8} {entry.get('channel')}/"
            f"{entry.get('arm')}@{entry.get('concurrency')}: {verdict.get('reason', '')}"
        )
        for warning in entry.get("warnings") or []:
            lines.append(f"           ! {warning}")
    notes = report.get("notes") or []
    if notes:
        lines.append("")
        lines.append("NOTES")
        lines.extend(f"  - {note}" for note in notes)
    return "\n".join(lines)


#: Read off the budget table so the banner cannot drift from the rule it describes.
ENFORCED_NAMES = tuple(f"{channel}/{arm}" for channel, arm in sorted(M.ENFORCED_CELLS))
