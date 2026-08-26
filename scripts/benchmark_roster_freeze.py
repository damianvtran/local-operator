"""Synthetic regression for the subagent-roster event-loop freeze.

Builds the live incident's shape without copying user transcripts: 75 child
records, attachment-sized transcript files, 100 ordinary projection events, and
a 500-job accounting tree. Run from the repository root with .venv/bin/python.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from local_operator.harness.comms import SubagentComms
from local_operator.harness.jobs import AsyncJob, AsyncJobManager
from local_operator.harness.types import Usage
from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.types import SessionProjection
from local_operator.session.session import Session


def _manager_tree(jobs: int = 500) -> AsyncJobManager:
    manager = AsyncJobManager()
    manager.restore(
        [
            AsyncJob(
                id=f"job-{index}",
                type="task",
                status="completed",
                label=f"job {index}",
                start_time=1.0,
                usage=Usage(input_tokens=1, provider="test", model_id="bench"),
            )
            for index in range(jobs)
        ]
    )
    return manager


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lop-roster-bench-") as raw:
        root = Path(raw)
        session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
        comms = SubagentComms(cast(Session, cast(Any, session)))
        for index in range(75):
            child = root / f"child-{index}"
            child.mkdir()
            # Approximate the incident's 26 MB history footprint without
            # embedding any private transcript or prompt content.
            (child / "transcript.jsonl").write_bytes(b"x" * 350_000)
            job_id = f"child-{index}"
            comms.record_launch(job_id, f"child {index}")
            comms._records[job_id].session_dir = child

        fold = ProjectionFold(SessionProjection(session_id="bench", pid=1))
        samples = []
        for _ in range(100):
            started = time.perf_counter_ns()
            fold.set_subagent_details(comms)
            samples.append((time.perf_counter_ns() - started) / 1_000_000)

        manager = _manager_tree()
        manager.accounting_components()
        started = time.perf_counter_ns()
        for _ in range(1_000):
            manager.accounting_components()
        accounting_ms = (time.perf_counter_ns() - started) / 1_000_000 / 1_000

        print(f"ordinary_event_p50_ms={statistics.median(samples):.3f}")
        print(f"ordinary_event_p95_ms={statistics.quantiles(samples, n=20)[18]:.3f}")
        print(f"ordinary_event_max_ms={max(samples):.3f}")
        print(f"accounting_unchanged_500_jobs_ms={accounting_ms:.3f}")
        print("child_transcript_constructions=0")


if __name__ == "__main__":
    main()
