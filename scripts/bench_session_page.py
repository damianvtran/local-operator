"""Measure the session page-read and one-row metadata paths against a real store.

The operator's complaint is that opening one conversation makes the whole desktop
unusable, so this harness reports the two reads at the seam where that cost lives
and reports them ONE AT A TIME:

* the page reads (``read_transcript_page``: tail, ``through_id``, ``before_id``)
  and ``read_replay_suffix``, which is the whole of the path an open and a resume
  take;
* ``Transcript(...)`` construction and the one-row metadata read built on it
  (``Transcript(d).latest_custom(TYPE)``), which is what ``locate()`` paid on
  every cold open;
* the new one-row scan (``read_latest_custom``) and the cached façade
  (``load_transcript_page``: a cold miss and the warm hit beside it), when the
  tree under test has them. A tree that does not gets ``absent`` rather than a
  fabricated number.

Each row carries a STRUCTURAL column beside its milliseconds: the number of
journal rows the operation DECODED. That column is the fact — it is identical on
an idle laptop and on a wedged CI runner — while the wall clock on this machine
is weather, because it runs at load average 150-270. Every run reports the host
load average it was taken under, and no figure here is a CI limit.

Examples::

    # Against the operator's real store, on the tree this file lives in.
    .venv/bin/python scripts/bench_session_page.py --output /tmp/page-bench/head.json

    # Interleaved A/B: the same harness alternates two worktrees, session by
    # session, flipping the order every round so before and after share the
    # same load weather rather than two runs minutes apart.
    .venv/bin/python scripts/bench_session_page.py \\
        --source-root ~/workspace/repos/lo-session-load-base \\
        --output /tmp/page-bench/ab.json

READ-ONLY, AND THAT IS A REQUIREMENT RATHER THAN A CONVENTION. ``--store``
defaults to the operator's live sessions directory and the harness never opens a
journal for writing: every read goes through the product's own readers (opened
``"rb"``), and the ``Transcript`` is constructed with ``defer_materialise=True``
so not even a directory is created. ``--output`` is REFUSED inside the store, so
a mistyped path cannot drop a JSON file into a session directory.

TWO PROCESSES, ONE HARNESS, and the same reason as ``bench_session_switch.py``:
an editable install resolves ONE source root, so a single process cannot hold two
trees. The driver alternates worker subprocesses and the worker echoes back the
tree its ``local_operator`` actually resolved from; the driver refuses a mismatch
rather than reporting the wrong tree's numbers under the right tree's label.
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
from typing import Any, Callable

HERE = Path(__file__).resolve().parent.parent

#: The real store, resolved BEFORE any isolation: ``scripts.probe_isolation``
#: re-homes ``HOME`` on import, so if this is computed after it the harness would
#: measure an empty sandbox and report it as the operator's store.
_REAL_HOME = Path(os.environ.get("HOME", "~")).expanduser()
DEFAULT_STORE = _REAL_HOME / ".local-operator" / "sessions"

#: Rows per page. The desktop open asks for 100 (`/history`'s default), so this
#: is the number the complaint is about rather than a round one.
PAGE_LIMIT = 100

#: Sessions measured by default when none are named: the largest journals, by
#: size, which is the order the defect scales in.
DEFAULT_TOP = 4

#: One operation's sample count. Small on purpose — each sample on the 261 MB
#: conversation costs seconds — and enough for a median.
DEFAULT_SAMPLES = 3

DEFAULT_ROUNDS = 2

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument(
    "--store",
    type=Path,
    default=DEFAULT_STORE,
    help="sessions directory to read (never written to)",
)
PARSER.add_argument(
    "--session",
    action="append",
    default=[],
    help="session id to measure; repeatable, defaults to the largest journals",
)
PARSER.add_argument("--top", type=int, default=DEFAULT_TOP)
PARSER.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
PARSER.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
PARSER.add_argument(
    "--source-root",
    type=Path,
    default=None,
    help="the OTHER tree to interleave against; omit for a single-tree run",
)
PARSER.add_argument("--output", type=Path, required=True)
PARSER.add_argument("--tree", default="", help=argparse.SUPPRESS)  # worker-only
PARSER.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)


def _sessions(args: argparse.Namespace) -> list[str]:
    """The session ids to measure, in the order they will be reported."""
    store = args.store.expanduser()
    if args.session:
        return list(args.session)
    sized: list[tuple[int, str]] = []
    for entry in store.iterdir():
        journal = entry / "transcript.jsonl"
        if entry.is_dir() and journal.is_file():
            sized.append((journal.stat().st_size, entry.name))
    sized.sort(reverse=True)
    return [session_id for _size, session_id in sized[: args.top]]


def _journal_bytes(store: Path, session_id: str) -> int:
    journal = store / session_id / "transcript.jsonl"
    return journal.stat().st_size if journal.is_file() else 0


# ---------------------------------------------------------------------------
# Worker: every operation the load path is made of, on one tree, one session at
# a time. Runs as a subprocess so an A/B never mixes two source roots.
# ---------------------------------------------------------------------------


class _DecodedRows:
    """The structural column: journal rows this process DECODED.

    Installed as a spy on ``TranscriptEntry.from_json`` — the single place both
    readers turn bytes into rows — so a row count is a fact about the work an
    operation did, not about how busy the box was. The mutation it exposes is the
    one this whole change is about: a forward whole-file scan decodes the journal
    where a backward read decodes the page.
    """

    def __init__(self, transcript_module: Any) -> None:
        self.rows = 0
        self._real = transcript_module.TranscriptEntry.from_json
        counter = self

        def counting(line: str) -> Any:
            counter.rows += 1
            return counter._real(line)

        transcript_module.TranscriptEntry.from_json = staticmethod(counting)

    def take(self) -> int:
        rows, self.rows = self.rows, 0
        return rows


async def _measure(call: Callable[[], Any], rows: _DecodedRows) -> tuple[float, int]:
    """``(milliseconds, rows decoded)`` for one call, sync or async."""
    rows.rows = 0
    start = time.perf_counter()
    result = call()
    if asyncio.iscoroutine(result):
        await result
    return (time.perf_counter() - start) * 1000, rows.take()


async def _operation_rows(
    store: Path,
    session_id: str,
    samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Every measured operation for one session, plus the cache tally."""
    from local_operator.session import transcript as transcript_module
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE

    directory = store / session_id
    rows = _DecodedRows(transcript_module)
    (directory / "transcript.jsonl").stat()  # fail LOUDLY on an absent journal

    # Cursors come from the journal itself: a tail page tells us the oldest row
    # a backward walk still has to reach (the deep cursor) and the newest
    # (the shallow one), so the two cursor rows bracket the real cost range.
    tail = transcript_module.read_transcript_page(directory, limit=PAGE_LIMIT)
    deepest = tail.entries[0].id if tail.entries else None
    newest = tail.entries[-1].id if tail.entries else None
    if deepest is None:
        return [], None

    named: list[tuple[str, Callable[[], Any]]] = [
        ("tail_page", lambda: transcript_module.read_transcript_page(directory, limit=PAGE_LIMIT)),
        (
            "through_id_page",
            lambda: transcript_module.read_transcript_page(
                directory, through_id=newest, limit=PAGE_LIMIT
            ),
        ),
        (
            "before_id_page",
            lambda: transcript_module.read_transcript_page(
                directory, before_id=deepest, limit=PAGE_LIMIT
            ),
        ),
        ("replay_suffix", lambda: transcript_module.read_replay_suffix(directory)),
        (
            "transcript_construct",
            lambda: transcript_module.Transcript(directory, defer_materialise=True),
        ),
        (
            "metadata_row_via_transcript",
            lambda: transcript_module.Transcript(directory, defer_materialise=True).latest_custom(
                FRONTEND_CHECKPOINT_CUSTOM_TYPE
            ),
        ),
    ]

    # The after-tree's own readers, if this tree has them. A tree without them is
    # reported as ``absent`` rather than as a zero, so the table cannot be read
    # as "this cost nothing here".
    scan = getattr(transcript_module, "read_latest_custom", None)
    if scan is not None:
        named.append(
            (
                "metadata_row_scan",
                lambda: scan(directory, FRONTEND_CHECKPOINT_CUSTOM_TYPE),
            )
        )

    page_cache: Any = None
    facade: Any = None
    try:
        from local_operator.session import page_cache as page_cache_module

        facade = getattr(page_cache_module, "load_transcript_page", None)
        page_cache = page_cache_module
    except ImportError:
        # A tree that predates the cache: the two cached rows are absent from its
        # table, which is the honest answer rather than a zero.
        facade = None
    if facade is not None:
        named.append(
            (
                "cached_page_miss",
                lambda: _reset_then(page_cache, facade, directory),
            )
        )
        named.append(("cached_page_hit", lambda: facade(directory, limit=PAGE_LIMIT)))

    records: list[dict[str, Any]] = []
    for name, call in named:
        timings: list[float] = []
        counts: list[int] = []
        for _sample in range(samples):
            wall, decoded = await _measure(call, rows)
            timings.append(wall)
            counts.append(decoded)
        records.append(
            {
                "session": session_id,
                "operation": name,
                "median_ms": round(statistics.median(timings), 2),
                "min_ms": round(min(timings), 2),
                "max_ms": round(max(timings), 2),
                "rows_decoded": int(statistics.median(counts)),
                "samples": samples,
            }
        )

    tally: dict[str, Any] | None = None
    if page_cache is not None:
        cache = page_cache.page_cache()
        tally = {
            "hits": cache.hits,
            "misses": cache.misses,
            "oversize": cache.oversize,
            "entries": cache.entry_count,
            "accounted_bytes": cache.accounted_bytes,
        }
    return records, tally


async def _reset_then(page_cache_module: Any, facade: Any, directory: Path) -> Any:
    """A COLD load: the cache is emptied first, so this is the miss it claims to be.

    Without the reset the second sample would be a hit and the "cold" row would
    measure the cache rather than the read.
    """
    page_cache_module.reset_page_cache()
    return await facade(directory, limit=PAGE_LIMIT)


def _run_worker(args: argparse.Namespace) -> None:
    # Multiplexer variables are independent of HOME/config, and a headless run
    # must not inherit identifiers that could rename the operator's real workspaces.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            del os.environ[key]

    import scripts.probe_isolation  # noqa: F401  -- re-homes HOME/config on import

    # isort: split
    import local_operator

    tree = Path(args.tree).resolve()
    resolved = Path(local_operator.__file__).resolve()
    store = args.store.expanduser()
    records: list[dict[str, Any]] = []
    tallies: list[dict[str, Any]] = []
    load_before = os.getloadavg()
    for session_id in _sessions(args):
        session_records, tally = asyncio.run(_operation_rows(store, session_id, args.samples))
        for record in session_records:
            record["bytes"] = _journal_bytes(store, session_id)
            records.append(record)
        if tally is not None:
            tallies.append({"session": session_id, **tally})
    payload = {
        "tree": str(tree),
        "source": str(resolved),
        "load_before": [round(value, 2) for value in load_before],
        "load_after": [round(value, 2) for value in os.getloadavg()],
        "records": records,
        "cache": tallies,
    }
    print(json.dumps(payload))


# ---------------------------------------------------------------------------
# Driver: alternates the trees so before and after share one load weather.
# ---------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance is a nicety, never a failure
        return ""


def _provenance(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "sha": _git(root, "rev-parse", "HEAD"),
        "describe": _git(root, "describe", "--always", "--dirty"),
        "dirty": bool(_git(root, "status", "--porcelain")),
    }


def _interpreter(root: Path) -> tuple[str, dict[str, str]]:
    """The tree's OWN venv, with ``PYTHONPATH`` as a second line of defence.

    AGENTS.md is explicit that an editable install resolves one hard-coded source
    root, so measuring tree B under tree A's interpreter imports A and reports it
    as B. The worker echoes the tree it resolved and the driver refuses a
    mismatch, which is what makes the defence verifiable rather than assumed.
    """
    env = dict(os.environ, PYTHONPATH=str(root))
    candidate = root / ".venv/bin/python"
    return (str(candidate) if candidate.exists() else sys.executable), env


def _worker(root: Path, session_id: str, args: argparse.Namespace) -> dict[str, Any]:
    python, env = _interpreter(root)
    script = root / "scripts/bench_session_page.py"
    if not script.exists():
        # The tree predates this harness (that is the point of the A/B), so run
        # this file's copy with the target tree first on sys.path.
        script = Path(__file__).resolve()
    proc = subprocess.run(
        [
            python,
            str(script),
            "--worker",
            "--tree",
            str(root),
            "--session",
            session_id,
            "--samples",
            str(args.samples),
            "--store",
            str(args.store),
            "--output",
            os.devnull,
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker for {root} session {session_id} failed:\n{proc.stderr[-3000:]}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    resolved = Path(payload["source"]).resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError(f"worker for {root} imported {resolved} — refusing to report it")
    return payload


def _table(
    records: dict[str, list[dict[str, Any]]], sessions: list[str], sizes: dict[str, float]
) -> str:
    """The table a reviewer reads: one row per (session, operation), both trees."""
    sides = list(records)
    header = (
        ["session", "size MB", "operation"]
        + [f"{side} ms" for side in sides]
        + [f"{side} rows decoded" for side in sides]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for session_id in sessions:
        operations: list[str] = []
        for side in sides:
            for record in records[side]:
                if record["session"] == session_id and record["operation"] not in operations:
                    operations.append(record["operation"])
        for operation in operations:
            row = [session_id, f"{sizes.get(session_id, 0.0):.1f}", operation]
            found: dict[str, dict[str, Any] | None] = {
                side: next(
                    (
                        item
                        for item in records[side]
                        if item["session"] == session_id and item["operation"] == operation
                    ),
                    None,
                )
                for side in sides
            }

            def cell(side: str, key: str, found: Any = found) -> str:
                record = found[side]
                if record is None:
                    return "absent"
                value = record[key]
                return f"{value:.1f}" if isinstance(value, float) else str(value)

            row.extend(cell(side, "median_ms") for side in sides)
            row.extend(cell(side, "rows_decoded") for side in sides)
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _run_driver(args: argparse.Namespace) -> None:
    store = args.store.expanduser()
    if str(args.output.resolve()).startswith(str(store.resolve())):
        PARSER.error("--output must not be inside the store: it is read-only here")
    sessions = _sessions(args)

    trees: list[tuple[str, Path]] = [("head", HERE)]
    if args.source_root is not None:
        trees.append(("base", args.source_root.resolve()))

    records: dict[str, list[dict[str, Any]]] = {side: [] for side, _ in trees}
    cache: dict[str, list[dict[str, Any]]] = {side: [] for side, _ in trees}
    sizes = {session_id: _journal_bytes(store, session_id) / 1e6 for session_id in sessions}
    started = time.time()
    for round_index in range(args.rounds):
        for session_id in sessions:
            # Flip the order every round so neither tree systematically owns the
            # quieter half of a load wave.
            order = trees if round_index % 2 == 0 else list(reversed(trees))
            for side, root in order:
                payload = _worker(root, session_id, args)
                for record in payload["records"]:
                    record["round"] = round_index
                    record["side"] = side
                records[side].extend(payload["records"])
                cache[side].extend(payload["cache"])
                print(
                    f"round {round_index} session {session_id} side {side}: "
                    f"{len(payload['records'])} rows, load {payload['load_before'][0]}",
                    file=sys.stderr,
                    flush=True,
                )

    document: dict[str, Any] = {
        "harness": "bench_session_page",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s": round(time.time() - started, 1),
        "rounds": args.rounds,
        "samples_per_cell": args.samples,
        "store": str(store),
        "sessions": sessions,
        "load_average_at_end": [round(value, 2) for value in os.getloadavg()],
        "trees": {side: _provenance(root) for side, root in trees},
        "records": records,
        "cache": cache,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(
        f"\nhost load average at end: "
        f"{' '.join(f'{value:.2f}' for value in os.getloadavg())} "
        f"(this machine runs at 150-270; the milliseconds below are weather, the "
        f"rows-decoded column is not)\n"
    )
    print(_table(records, sessions, sizes))
    print()
    for side in cache:
        if cache[side]:
            print(f"{side} cache tally: {json.dumps(cache[side])}")


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    if ARGS.samples < 1 or ARGS.rounds < 1:
        PARSER.error("--samples and --rounds must be positive")
    if ARGS.worker and not ARGS.tree:
        PARSER.error("--worker requires --tree")
    sys.path.insert(0, str(HERE))
    if ARGS.worker:
        sys.path.insert(0, str(Path(ARGS.tree).resolve()))
        _run_worker(ARGS)
    else:
        _run_driver(ARGS)
