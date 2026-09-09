#!/usr/bin/env python3
"""Cost of a single-field frontend-state read: whole-state clone vs `read_field`.

WHY THIS EXISTS. ``FrontendStateStore.state`` deep-copies the ENTIRE session
state on every read so a caller cannot mutate the store's own instance. That
protection is correct and must stay, but its cost scales with how much the
session has accumulated — jobs, usage components, live events, catalogue rows —
so the per-frame accessors in ``RemoteSession`` that need ONE immutable scalar
got slower the longer a session ran. That is the mechanism behind "the TUI
degrades on long sessions": the band repaints at the same rate, but each paint's
state reads cost more every hour.

``FrontendStateStore.read_field`` is the existing fast path for exactly this,
restricted by ``_SHAREABLE_STATE_FIELDS`` to deeply immutable scalars. This
script measures the gap the conversion closes, in two ways:

  1. WALL TIME per read, at several roster sizes (0/5/20/60/120 jobs), reported
     as MEDIANS of many reps. Wall time on a shared developer box is
     contaminated by whatever else is running, so the load average is printed
     with every table and the numbers are CEILINGS, not claims.
  2. ``copy.deepcopy`` CALL COUNTS for a representative status-band frame. A
     call count is contention-proof — it is identical on an idle box and a
     box at load 200 — which makes it the honest headline number.

INTERLEAVED BY CONSTRUCTION. Both arms are measured in one process, alternating
rep by rep at each size, so a scheduling burst that inflates one arm inflates
the other equally instead of manufacturing a win. There is no "before tree" to
check out: the two paths coexist in the same build, which is what makes the
comparison reproducible by anyone at any commit.

Run:
    PYTHONPATH=. .venv/bin/python scripts/bench_state_field_reads.py
    PYTHONPATH=. .venv/bin/python scripts/bench_state_field_reads.py --json out.json

``PYTHONPATH=.`` matters for the reason ``bench/README.md`` gives: the script
must import THIS checkout rather than whatever an editable install resolves to.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    FrontendStateStore,
    FrontendUsage,
    JobState,
)
from local_operator.session.remote import RemoteSession

#: Roster sizes to sweep. The point of the sweep is the SHAPE, not any single
#: cell: a cost that is flat in roster size would mean long sessions do not
#: degrade and this change is pointless, so the growth is the finding.
ROSTER_SIZES = (0, 5, 20, 60, 120)

#: The fields the per-frame accessors read. Every one is on
#: ``_SHAREABLE_STATE_FIELDS``; a name that is not raises from ``read_field``,
#: which is the guard working rather than a bench failure.
FRAME_FIELDS = (
    "conversation_title",
    "goal",
    "active_agent",
    "active_team",
    "epoch",
    "history_generation",
    "streaming",
)

#: The ``RemoteSession`` accessors one status-band paint calls, in the order
#: ``app.py`` calls them (``_effective_label`` first, then the profile/team
#: segments and the conversation title). This is the frame the user actually
#: pays for, as opposed to the store-level microbenchmark above.
BAND_ACCESSORS = (
    "effective_model_label",
    "model_label",
    "active_agent",
    "active_team_name",
    "conversation_name",
    "goal",
    "epoch",
)


def build_state(n_jobs: int) -> FrontendSessionState:
    """A state shaped like a long-running session, not an empty one.

    Deliberately populated on every field the clone has to walk — trajectories,
    per-job usage, usage components, live events, a catalogue — because a state
    carrying only its scalars measures the clone's floor and would understate
    the cost the accessors actually pay in a real session.
    """
    jobs = [
        JobState(
            id=f"job-{i}",
            type="task",
            status="running" if i % 3 else "completed",
            label=f"child agent {i}",
            agent="coder",
            intent="implementing a bounded slice",
            result_text="x" * 400,
            trajectory=[{"kind": "tool", "text": "y" * 80} for _ in range(6)],
            usage=FrontendUsage(input_tokens=1000 + i, output_tokens=500 + i),
        )
        for i in range(n_jobs)
    ]
    return FrontendSessionState(
        session_id="bench-session",
        epoch="epoch-1",
        conversation_title="a long-running conversation",
        goal="ship the measured change",
        active_agent="coder",
        active_team="lopdev",
        streaming=False,
        history_generation=7,
        jobs=jobs,
        usage_components=[FrontendUsage(input_tokens=100, output_tokens=50) for _ in range(n_jobs)],
        live_events=[
            {"type": "tool_execution_end", "text": "z" * 200} for _ in range(min(n_jobs, 40))
        ],
        selected_model=FrontendModelSpec(provider="anthropic", model_id="claude-opus-4"),
        effective_model=FrontendModelSpec(provider="anthropic", model_id="claude-sonnet-4"),
        last_usage=FrontendUsage(input_tokens=120_000, output_tokens=8_000),
        model_catalogue=[
            {"provider": "anthropic", "model_id": f"model-{i}", "name": f"Model {i}"}
            for i in range(60)
        ],
    )


def median_ms(fn: Callable[[], Any], reps: int) -> float:
    """Median wall time of ``reps`` calls, in milliseconds.

    Median rather than mean: on a loaded host the distribution has a long right
    tail from descheduling, and a mean reports the tail rather than the work.
    """
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def deepcopy_calls(fn: Callable[[], Any]) -> int:
    """How many ``deepcopy`` invocations one operation makes, recursion included.

    The contention-proof metric: identical on an idle box and on one at load
    200, which is why it and not wall time is the headline.

    BOTH entry points are patched, and missing the second one silently
    undercounts to near zero. ``copy.deepcopy`` covers the recursive descent,
    because ``copy.py``'s internal ``_deepcopy_dict``/``_deepcopy_list`` resolve
    ``deepcopy`` as a module global on every element. But a pydantic model
    defines ``__deepcopy__``, and ``pydantic.main`` bound ``deepcopy`` into its
    own namespace with ``from copy import deepcopy`` at import time — so the
    per-model hop that walks each job and usage row never reads
    ``copy.deepcopy`` at all. Patching only the first reported 0 calls for a
    clone that in fact makes thousands.

    The real implementation is still called, so the measured operation does
    exactly what it does in production.
    """
    import pydantic.main

    real = copy.deepcopy
    count = 0

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        return real(*args, **kwargs)

    with patch("copy.deepcopy", counting), patch.object(pydantic.main, "deepcopy", counting):
        fn()
    return count


def build_remote(state: FrontendSessionState) -> RemoteSession:
    """A viewer bound to a real store and nothing else.

    The accessors under measurement read only the store, so a socket would add
    setup without changing a single number. ``find_owner_record`` is patched to
    find nothing so construction does no disk work.
    """

    async def _never() -> Any:
        raise AssertionError("the benchmark never takes a session over")

    with (
        patch("local_operator.session.remote.find_owner_record", lambda *a, **k: (None, None)),
        tempfile.TemporaryDirectory() as config_dir,
    ):
        remote = RemoteSession(
            config_dir=Path(config_dir),
            session_id=state.session_id,
            takeover_factory=_never,
        )
    remote._frontend_store = FrontendStateStore(state)
    return remote


def band_frame_after(remote: RemoteSession) -> None:
    """One band paint through the accessors AS THEY ARE NOW."""
    for name in BAND_ACCESSORS:
        getattr(remote, name)


def band_frame_before(remote: RemoteSession) -> None:
    """One band paint through the WHOLE-STATE CLONE each accessor used to take.

    Hand-rolled rather than checked out from the parent commit deliberately:
    running both arms in ONE process is what makes the comparison interleaved,
    and a separate `git checkout` arm would measure two different processes on
    a machine whose load moves between them. Each line here is the exact body
    the corresponding accessor had before the conversion, so the arm is the old
    code rather than an approximation of it.
    """
    state = remote.frontend_state
    state.effective_model_label
    remote.frontend_state.model_label
    remote.frontend_state.active_agent
    remote.frontend_state.active_team
    remote.frontend_state.conversation_title
    remote.frontend_state.goal
    remote.frontend_state.epoch


def load_average() -> str:
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:  # pragma: no cover - not available on every platform
        return "unavailable"
    return f"{one:.2f} {five:.2f} {fifteen:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200, help="reps per arm per size")
    parser.add_argument("--json", dest="json_path", default="", help="write results as JSON")
    args = parser.parse_args()

    print(f"load average (start): {load_average()}")
    print()
    print(
        f"Per-read wall time, medians of {args.reps} interleaved reps " "(CEILINGS: host is shared)"
    )
    print()
    header = f"{'jobs':>5}  {'state clone':>14}  {'read_field':>14}  {'ratio':>10}"
    print(header)
    print("-" * len(header))

    rows: list[dict[str, Any]] = []
    for size in ROSTER_SIZES:
        store = FrontendStateStore(build_state(size))
        # Interleaved: alternate the two arms rep by rep so a scheduling burst
        # lands on both arms rather than on whichever is measured second.
        clone_samples: list[float] = []
        fast_samples: list[float] = []
        for _ in range(args.reps):
            clone_samples.append(median_ms(lambda: store.state.conversation_title, 1))
            fast_samples.append(median_ms(lambda: store.read_field("conversation_title"), 1))
        clone_ms = statistics.median(clone_samples)
        fast_ms = statistics.median(fast_samples)
        ratio = clone_ms / fast_ms if fast_ms else float("inf")
        print(f"{size:>5}  {clone_ms:>11.4f} ms  {fast_ms:>11.6f} ms  {ratio:>9.0f}x")
        rows.append(
            {
                "jobs": size,
                "state_clone_ms": clone_ms,
                "read_field_ms": fast_ms,
                "ratio": ratio,
            }
        )

    print()
    print(f"load average (after wall-time sweep): {load_average()}")
    print()
    print(
        "deepcopy calls for one status-band frame "
        f"({len(FRAME_FIELDS)} single-field reads), contention-proof:"
    )
    print()
    frame_header = f"{'jobs':>5}  {'via state':>12}  {'via read_field':>15}  {'removed':>10}"
    print(frame_header)
    print("-" * len(frame_header))

    frames: list[dict[str, Any]] = []
    for size in ROSTER_SIZES:
        store = FrontendStateStore(build_state(size))

        def via_state() -> None:
            for name in FRAME_FIELDS:
                getattr(store.state, name)

        def via_read_field() -> None:
            for name in FRAME_FIELDS:
                store.read_field(name)

        before = deepcopy_calls(via_state)
        after = deepcopy_calls(via_read_field)
        print(f"{size:>5}  {before:>12,}  {after:>15,}  {before - after:>10,}")
        frames.append({"jobs": size, "via_state": before, "via_read_field": after})

    print()
    print(f"load average (before band-frame sweep): {load_average()}")
    print()
    print("REAL status-band frame on a RemoteSession — the accessors the band calls:")
    print(f"  {', '.join(BAND_ACCESSORS)}")
    print()
    band_header = (
        f"{'jobs':>5}  {'before (clone)':>16}  {'after':>14}  "
        f"{'deepcopy before':>16}  {'after':>7}"
    )
    print(band_header)
    print("-" * len(band_header))

    bands: list[dict[str, Any]] = []
    for size in ROSTER_SIZES:
        remote = build_remote(build_state(size))
        before_ms_samples: list[float] = []
        after_ms_samples: list[float] = []
        for _ in range(args.reps):
            before_ms_samples.append(median_ms(lambda: band_frame_before(remote), 1))
            after_ms_samples.append(median_ms(lambda: band_frame_after(remote), 1))
        before_ms = statistics.median(before_ms_samples)
        after_ms = statistics.median(after_ms_samples)
        before_calls = deepcopy_calls(lambda: band_frame_before(remote))
        after_calls = deepcopy_calls(lambda: band_frame_after(remote))
        print(
            f"{size:>5}  {before_ms:>13.4f} ms  {after_ms:>11.4f} ms  "
            f"{before_calls:>16,}  {after_calls:>7,}"
        )
        bands.append(
            {
                "jobs": size,
                "before_ms": before_ms,
                "after_ms": after_ms,
                "deepcopy_before": before_calls,
                "deepcopy_after": after_calls,
            }
        )

    print()
    print(f"load average (end): {load_average()}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reps": args.reps,
                    "load_average": load_average(),
                    "wall_time": rows,
                    "deepcopy_frame": frames,
                    "frame_fields": list(FRAME_FIELDS),
                    "band_frame": bands,
                    "band_accessors": list(BAND_ACCESSORS),
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
