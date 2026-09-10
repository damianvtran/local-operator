"""Benchmark the sidebar's catalog poll: wall time AND syscalls per poll.

Usage:

    PYTHONPATH=. .venv/bin/python scripts/bench_catalog_scan.py ladder \
        [--sizes 100,500,1000,2000,4000] [--json out.json]
    PYTHONPATH=. .venv/bin/python scripts/bench_catalog_scan.py store-axis \
        [--sizes 100,500,1000,2000,4000,8000] [--users 50] [--json out.json]
    PYTHONPATH=. .venv/bin/python scripts/bench_catalog_scan.py users-axis \
        [--users 10,50,200] [--dirs 4000] [--json out.json]
    PYTHONPATH=. .venv/bin/python scripts/bench_catalog_scan.py unmarked-axis \
        [--counts 0,500,2000,8000] [--users 50] [--hidden 500] [--json out.json]
    PYTHONPATH=. .venv/bin/python scripts/bench_catalog_scan.py real \
        [--store ~/.local-operator] [--json out.json]

``PYTHONPATH=.`` matters for the reason ``bench_resume_picker.py`` documents:
the script must import THIS checkout, not whatever an editable install
resolves to.

**Why this benchmark counts syscalls and not just milliseconds.** The cost it
exists to measure is paid by the TUI every 2 seconds for as long as the sidebar
is open, on a machine that routinely runs a dozen agent sessions at once. Wall
time there is dominated by whatever else is scheduled — samples on the
reporting machine varied 192-642 ms for the same work at load average 48 — so a
wall-clock delta alone cannot distinguish a real improvement from a quiet
minute. The syscall count is the honest invariant: it is a property of the
algorithm, it does not move with load, and it is what the poll actually asks
the filesystem for. Wall time is reported alongside it as corroboration, always
with the load average that produced it.

The ladder measures SCALING, which is the actual defect: the poll's cost grew
with the total number of session directories ever created rather than with the
user's own sessions, so it degraded permanently as the store accumulated.

**``store-axis`` is the decisive experiment, and the ladder is not.** The ladder
grows users and hidden directories together, so a cost that tracks EITHER rises
and the two hypotheses are indistinguishable — which is exactly how PR #867
shipped a title bigger than its measurement and had to be retitled mid-review.
``store-axis`` holds the USER population FIXED and varies only the total store,
so a design that is O(user sessions) holds a FLAT column while one that is
O(total directories) rises linearly. ``users-axis`` is its complement: it fixes
the store and varies users, showing the remaining per-user cost is linear and
small rather than the design being O(1) by answering the wrong question. Any
performance claim about this poll must be backed by both, not by the ladder.

**``unmarked-axis`` is where the claim STOPS, and it is not optional either.**
Both axes above grow the store with directories the skip can arm on, so both
report a flat column while a third population costs linearly forever: a
directory with neither an origin marker nor any activity is in neither the
listing nor the hidden set, never arms the skip, and pays ~4 syscalls per poll
for the life of the store while never being listable. Measured by this mode
with users fixed at 50 and hidden at 500: 266 / 2,266 / 8,266 / 32,266 over
0 / 500 / 2,000 / 8,000, a slope of exactly 4.0 (agent review round 1, R1,
which measured the same slope at 268 -> 32,268 with a counter that also hooks
``open`` and ``listdir`` — the slope is the claim, the constant is the
instrument). The honest statement of the result is
therefore **O(user sessions + directories that are neither listed nor
cached-hidden)**, and this mode is what keeps that qualifier attached to a
number instead of to a memory. A flat ``store-axis`` alone is precisely the
evidence that let #867 overstate its scaling.

The ``real`` mode measures the operator's own store. It is strictly READ-ONLY:
it never writes to the store it measures, so it is safe to point at a live
``~/.local-operator`` while sessions are running. (The one write the scan can
normally make — the origin verdict cache — is neutralised; see ``_no_writes``.)
"""

from __future__ import annotations

import collections
import contextlib
import gc
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from local_operator.resume import _recent_sessions_with_origin
from local_operator.session.catalog import load_catalog

REPS = 7


@contextlib.contextmanager
def _counted() -> Iterator[collections.Counter[str]]:
    """Count the filesystem syscalls issued inside the block.

    Patches the ``os`` entry points rather than sampling with ``dtrace``/
    ``strace``: those need privileges this script must not require, and the
    Python-level count is the one that maps back to a line of code. The
    counters are installed around a single call and removed immediately, so
    nothing else in the process pays for them.

    ``os.stat`` is patched on the ``os`` module, which is what application code
    calls. ``pathlib`` traffic IS counted too, on the supported interpreters:
    ``Path.stat``/``Path.resolve``/``Path.glob`` delegate through ``os.*``
    rather than reaching the C ``posix`` module directly, verified on the
    pinned 3.12 (``Path.stat()`` -> one counted ``stat``; ``Path.resolve()`` ->
    9 counted ``lstat``; ``Path.glob("*/x")`` -> a counted ``scandir``). That
    is why the old ``glob("*/desktop.json")`` shows up here as 1,946 scandirs
    at all — an uncounted implementation would have reported zero.
    ``--profile`` still attributes calls to their call site, which the raw
    totals cannot.

    An earlier version of this docstring claimed the opposite (pathlib
    uncounted, therefore the BEFORE number understated). It was wrong, and the
    correction moves in the safe direction: the published BEFORE figures are
    MORE complete than claimed, not less, so no number in the PR shrinks.
    """
    counts: collections.Counter[str] = collections.Counter()
    originals = {name: getattr(os, name) for name in ("stat", "lstat", "scandir", "open")}

    def wrap(name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def counting(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return real(*args, **kwargs)

        return counting

    for name, real in originals.items():
        setattr(os, name, wrap(name, real))
    try:
        yield counts
    finally:
        for name, real in originals.items():
            setattr(os, name, real)


def _timed(fn: Callable[[], Any], reps: int = REPS) -> dict[str, float]:
    """min/median/max milliseconds over ``reps`` runs.

    The median is the headline and the min is kept alongside, because this
    machine runs several agents' suites at once: a slow sample measures
    contention rather than the code. ``gc.collect()`` before each sample so a
    collection triggered by the previous run is not billed to this one.
    """
    samples = []
    for _ in range(reps):
        gc.collect()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "min_ms": round(min(samples), 2),
        "median_ms": round(statistics.median(samples), 2),
        "max_ms": round(max(samples), 2),
    }


@contextlib.contextmanager
def _no_writes(store: Path) -> Iterator[None]:
    """Make the measured call provably read-only against ``store``.

    The scan persists an origin-verdict cache under ``<store>/cache``. That is
    the only write on this path, and writing it into the operator's LIVE store
    from a benchmark would both mutate what is being measured and race the real
    sessions using it. Redirecting the cache is not enough on its own to prove
    the point, so this also asserts that nothing opened a file for writing
    anywhere under the store while the block ran.
    """
    from local_operator import resume

    real_save = resume._save_origin_cache
    written: list[str] = []

    def refuse(path: Path, entries: dict[str, Any]) -> None:
        written.append(str(path))

    resume._save_origin_cache = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        resume._save_origin_cache = real_save  # type: ignore[assignment]
    inside = [p for p in written if str(store) in p]
    if inside:
        raise AssertionError(f"benchmark attempted to write inside the store: {inside}")


def _measure(store: Path, *, read_only: bool) -> dict[str, Any]:
    """Cold and warm figures for one store, plus syscalls for one warm poll."""
    guard = _no_writes(store) if read_only else contextlib.nullcontext()
    with guard:
        gc.collect()
        start = time.perf_counter()
        rows = load_catalog(store)
        cold_ms = (time.perf_counter() - start) * 1000

        catalog = _timed(lambda: load_catalog(store))
        scan = _timed(lambda: _recent_sessions_with_origin(store))

        # Counted on a WARM store: the steady state is what the sidebar
        # actually pays every 2 s, and a cold count would measure the first
        # poll after launch instead.
        with _counted() as catalog_counts:
            load_catalog(store)
        with _counted() as scan_counts:
            _recent_sessions_with_origin(store)

    return {
        "rows": len(rows),
        # Kept so ``compare`` can assert the listing is byte-identical before
        # and after. Cost without the answer proves nothing.
        "listing": [entry.id for entry in rows],
        "cold_ms": round(cold_ms, 2),
        "load_catalog": catalog,
        "scan": scan,
        "load_catalog_syscalls": dict(catalog_counts),
        "load_catalog_syscalls_total": sum(catalog_counts.values()),
        "scan_syscalls": dict(scan_counts),
        "scan_syscalls_total": sum(scan_counts.values()),
    }


def _build_store(root: Path, total: int) -> None:
    """A synthetic store shaped like the real one.

    The proportions matter more than the absolute size. On the reporting
    machine 1,785 of 1,946 directories (92%) are SUBAGENT sessions that the
    picker never lists, which is precisely the population the poll used to pay
    full price for, so a synthetic store of only user sessions would measure a
    case that does not occur and would hide the effect entirely.

    The awkward shapes are represented on purpose, because they are where an
    equivalence bug would hide: directories with no transcript at all, ones
    carrying only an inbox spool, and forked sessions with an origin marker
    that IS user-visible.
    """
    sessions = root / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for i in range(total):
        sid = f"{i:012x}"
        d = sessions / sid
        d.mkdir(exist_ok=True)
        kind = i % 100
        if kind < 92:  # subagent: has a transcript, never listed
            (d / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
            (d / "origin.json").write_text('{"origin":"subagent"}', encoding="utf-8")
        elif kind < 95:  # fork: user-visible despite carrying a marker
            (d / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
            (d / "origin.json").write_text('{"origin":"fork"}', encoding="utf-8")
        elif kind < 97:  # plain user session
            (d / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
        elif kind < 99:  # activity is the mail spool only
            (d / "inbox.jsonl").write_text('{"from":"peer"}\n', encoding="utf-8")
        else:  # no activity at all: never a row
            (d / "notes.txt").write_text("x", encoding="utf-8")
        stamp = now - (i % 5000)
        for name in ("transcript.jsonl", "inbox.jsonl"):
            with contextlib.suppress(OSError):
                os.utime(d / name, (stamp, stamp))


def _build_axis_store(root: Path, *, users: int, hidden: int, unmarked: int = 0) -> None:
    """A store with the three populations controlled INDEPENDENTLY.

    ``_build_store`` mixes them at a fixed ratio, which is right for modelling
    the real store but useless for separating "cost tracks the store" from
    "cost tracks the user's sessions" — both rise together there. Here the
    caller pins one and varies the other.

    ``unmarked`` is the THIRD population, and it is the one that bounds the
    claim. A directory with neither an origin marker nor any activity is in
    neither the listing nor the hidden set, so it can never arm the
    zero-syscall skip and pays its stats on every poll forever while never
    being listable. ``store-axis`` and ``users-axis`` both hold it at 0, which
    is exactly how a cost that is linear in it stays invisible to them — the
    same shape of blind spot that let #867 overstate its scaling. Defaulting to
    0 keeps those two experiments unchanged; ``unmarked-axis`` varies it.

    Names are prefixed rather than sequential hex so a reader of a failing
    assertion can see at a glance which population a directory belongs to. The
    shapes are exactly what ``mark_session_origin`` and a real session write.
    """
    sessions = root / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for index in range(users):
        directory = sessions / f"user{index:08x}"
        directory.mkdir(exist_ok=True)
        (directory / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
        stamp = now - index
        os.utime(directory / "transcript.jsonl", (stamp, stamp))
    for index in range(hidden):
        directory = sessions / f"sub{index:09x}"
        directory.mkdir(exist_ok=True)
        (directory / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
        (directory / "origin.json").write_text(
            json.dumps({"origin": "subagent", "label": "reviewer"}), encoding="utf-8"
        )
        stamp = now - 100000 - index
        os.utime(directory / "transcript.jsonl", (stamp, stamp))
    for index in range(unmarked):
        # No marker and no transcript: the shape an idle open-and-quit launch
        # leaves behind. Deliberately not given an origin marker — that would
        # make it hidden and therefore skippable, which is the population this
        # one exists to be distinguished from.
        (sessions / f"idle{index:08x}").mkdir(exist_ok=True)


def _steady_state(root: Path) -> dict[str, Any]:
    """Syscalls for one STEADY-STATE poll, plus the listing it produced.

    Warmed twice before measuring, not once: the first call builds the verdict
    cache and the second is the first poll able to use it, so measuring at the
    third is the first sample that represents what the sidebar pays minute after
    minute. Measuring earlier reports a cold poll and flatters nothing.

    The listing is recorded so ``compare`` can assert byte-identical ``(id,
    order)`` between a before run and an after run. A cheaper poll that lists
    differently is not an optimisation, and the only way to know is to keep the
    answer beside the cost.
    """
    load_catalog(root)
    load_catalog(root)
    with _counted() as catalog_counts:
        rows = load_catalog(root)
    with _counted() as scan_counts:
        _recent_sessions_with_origin(root)
    return {
        "rows": len(rows),
        "listing": [entry.id for entry in rows],
        "syscalls": sum(catalog_counts.values()),
        "syscalls_detail": dict(catalog_counts),
        "scan_syscalls": sum(scan_counts.values()),
    }


def _store_axis(sizes: list[int], users: int) -> list[dict[str, Any]]:
    """THE DECISIVE EXPERIMENT: users fixed, total directories varied."""
    results = []
    for size in sizes:
        hidden = max(size - users, 0)
        root = Path(tempfile.mkdtemp(prefix=f"lo-axis-{size}-"))
        try:
            _build_axis_store(root, users=users, hidden=hidden)
            row = {"dirs": size, "users": users, "hidden": hidden, **_steady_state(root)}
            results.append(row)
            print(
                f"  {size:>5} dirs ({hidden:>5} hidden): rows={row['rows']:<4} "
                f"syscalls/poll={row['syscalls']:>6}  scan={row['scan_syscalls']:>6}",
                flush=True,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
    if results:
        growth = results[-1]["syscalls"] / max(results[0]["syscalls"], 1)
        span = results[-1]["dirs"] / max(results[0]["dirs"], 1)
        print(f"  growth over a {span:.0f}x store: {growth:.1f}x")
    return results


def _users_axis(user_counts: list[int], dirs: int) -> list[dict[str, Any]]:
    """The complementary axis: store fixed, user population varied."""
    results = []
    for users in user_counts:
        hidden = max(dirs - users, 0)
        root = Path(tempfile.mkdtemp(prefix=f"lo-users-{users}-"))
        try:
            _build_axis_store(root, users=users, hidden=hidden)
            row = {"dirs": dirs, "users": users, "hidden": hidden, **_steady_state(root)}
            row["per_user"] = round(row["syscalls"] / max(users, 1), 2)
            results.append(row)
            print(
                f"  {users:>4} users of {dirs} dirs: rows={row['rows']:<4} "
                f"syscalls/poll={row['syscalls']:>6}  per user={row['per_user']:>6.2f}",
                flush=True,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return results


def _unmarked_axis(counts: list[int], users: int, hidden: int) -> list[dict[str, Any]]:
    """THE LIMIT OF THE CLAIM: both listed populations fixed, never-active varied.

    ``store-axis`` proves the poll is flat as the store grows, but it grows the
    store with HIDDEN directories only — the population the skip can arm on. A
    directory that is neither listed nor hidden is in neither set, so it never
    arms and never stops costing. This axis is the same experimental design run
    against the one input the other two do not vary, so the linear term is
    RECORDED rather than discovered later by whoever inherits the claim.

    Reported as syscalls per never-active directory, because the honest way to
    state a linear cost is its slope: a flat column here would be a genuinely
    O(user sessions) poll, and it is not one.
    """
    results = []
    for count in counts:
        root = Path(tempfile.mkdtemp(prefix=f"lo-unmarked-{count}-"))
        try:
            _build_axis_store(root, users=users, hidden=hidden, unmarked=count)
            row = {
                "dirs": users + hidden + count,
                "users": users,
                "hidden": hidden,
                "unmarked": count,
                **_steady_state(root),
            }
            results.append(row)
            print(
                f"  {count:>5} never-active (+{users} users, {hidden} hidden): "
                f"rows={row['rows']:<4} syscalls/poll={row['syscalls']:>6}",
                flush=True,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
    if len(results) > 1:
        span = results[-1]["unmarked"] - results[0]["unmarked"]
        rise = results[-1]["syscalls"] - results[0]["syscalls"]
        if span:
            print(
                f"  slope: {rise / span:.1f} syscalls per never-active directory per poll "
                "(a flat column here would mean O(user sessions); it is not flat)"
            )
    return results


def _compare(before_path: Path, after_path: Path) -> int:
    """Assert the listings are byte-identical between two runs of this script.

    The equivalence check that makes a performance number meaningful. Compares
    ``(id, order)`` — the list as-is, so a reordering fails as loudly as a
    missing row — for every size in every axis both files carry.
    """
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    failures = 0

    def _label(row: dict[str, Any]) -> str:
        return f"{row.get('dirs')}/{row.get('users')}"

    for section in ("store_axis", "users_axis", "ladder"):
        rows_before = {_label(r): r for r in before.get(section, [])}
        for row in after.get(section, []):
            label = _label(row)
            other = rows_before.get(label)
            if other is None or "listing" not in row or "listing" not in other:
                continue
            same = row["listing"] == other["listing"]
            gain = other["syscalls"] / max(row["syscalls"], 1)
            print(
                f"  {section:>10} {label:>12}: listing identical={same}  "
                f"{other['syscalls']} -> {row['syscalls']} syscalls ({gain:.1f}x)"
            )
            if not same:
                failures += 1
    for section in ("real",):
        row = after.get(section)
        other = before.get(section)
        if isinstance(row, dict) and isinstance(other, dict) and "listing" in row:
            same = row["listing"] == other.get("listing")
            print(f"  {section:>10}: listing identical={same}")
            if not same:
                failures += 1
    print("EQUIVALENT" if not failures else f"{failures} LISTING MISMATCHES")
    return 1 if failures else 0


def _mirror_real_store(source: Path, destination: Path) -> dict[str, int]:
    """Copy the facts the poll consults out of a LIVE store, reading only.

    Why a mirror rather than measuring the store in place. The steady state this
    benchmark reports is the one where the verdict cache is warm, and warming it
    means WRITING it — into a store a dozen other ``lop`` processes are using
    right now. ``_no_writes`` correctly refuses that, which leaves an in-place
    ``real`` run permanently cold and measuring the wrong thing. Mirroring gives
    a writable copy whose syscall counts transfer exactly, because the poll
    consults only four facts: which directories exist, which carry an origin
    marker, what that marker says, and the activity mtimes.

    Transcripts are recreated EMPTY with the real mtimes. The scan stats them
    and never reads them, so their 1.7 GB of content changes no count while
    copying it would take minutes and fill the disk.

    Every operation against ``source`` here is a read: ``scandir``, ``stat``,
    ``read_bytes``. Nothing writes a byte inside it.
    """
    sessions = destination / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    stats = {"dirs": 0, "marked": 0, "with_transcript": 0}
    with os.scandir(source / "sessions") as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            stats["dirs"] += 1
            target = sessions / entry.name
            target.mkdir(exist_ok=True)
            for name in ("origin.json", "desktop.json"):
                try:
                    payload = (Path(entry.path) / name).read_bytes()
                except OSError:
                    continue
                (target / name).write_bytes(payload)
                if name == "origin.json":
                    stats["marked"] += 1
            for name in ("transcript.jsonl", "inbox.jsonl"):
                try:
                    stamp = (Path(entry.path) / name).stat().st_mtime
                except OSError:
                    continue
                (target / name).write_bytes(b"")
                os.utime(target / name, (stamp, stamp))
                if name == "transcript.jsonl":
                    stats["with_transcript"] += 1
    return stats


def _ladder(sizes: list[int]) -> list[dict[str, Any]]:
    results = []
    for size in sizes:
        root = Path(tempfile.mkdtemp(prefix=f"lo-bench-{size}-"))
        try:
            _build_store(root, size)
            # A fresh store's first call also builds the verdict cache, so the
            # cache is warmed once before measuring the steady state.
            load_catalog(root)
            row = {"dirs": size, **_measure(root, read_only=False)}
            results.append(row)
            print(
                f"  {size:>5} dirs: rows={row['rows']:<4} "
                f"catalog={row['load_catalog']['median_ms']:>8.2f} ms  "
                f"scan={row['scan']['median_ms']:>8.2f} ms  "
                f"syscalls={row['load_catalog_syscalls_total']:>6}",
                flush=True,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return results


def main() -> int:
    argv = sys.argv[1:]
    mode = argv[0] if argv else "ladder"
    out_json = None
    if "--json" in argv:
        out_json = argv[argv.index("--json") + 1]

    payload: dict[str, Any] = {"loadavg": os.getloadavg(), "mode": mode}
    print(f"load average: {os.getloadavg()}")

    if mode == "ladder":
        sizes = [100, 500, 1000, 2000, 4000]
        if "--sizes" in argv:
            sizes = [int(x) for x in argv[argv.index("--sizes") + 1].split(",")]
        print("synthetic ladder (92% subagents, mirroring the real store's shape):")
        payload["ladder"] = _ladder(sizes)
    elif mode == "store-axis":
        sizes = [100, 500, 1000, 2000, 4000, 8000]
        if "--sizes" in argv:
            sizes = [int(x) for x in argv[argv.index("--sizes") + 1].split(",")]
        users = 50
        if "--users" in argv:
            users = int(argv[argv.index("--users") + 1])
        print(f"DECISIVE EXPERIMENT: users FIXED at {users}, total directories varied")
        payload["store_axis"] = _store_axis(sizes, users)
    elif mode == "users-axis":
        user_counts = [10, 50, 200]
        if "--users" in argv:
            user_counts = [int(x) for x in argv[argv.index("--users") + 1].split(",")]
        dirs = 4000
        if "--dirs" in argv:
            dirs = int(argv[argv.index("--dirs") + 1])
        print(f"complementary axis: store FIXED at {dirs} directories, users varied")
        payload["users_axis"] = _users_axis(user_counts, dirs)
    elif mode == "unmarked-axis":
        counts = [0, 500, 2000, 8000]
        if "--counts" in argv:
            counts = [int(x) for x in argv[argv.index("--counts") + 1].split(",")]
        users = 50
        if "--users" in argv:
            users = int(argv[argv.index("--users") + 1])
        hidden = 500
        if "--hidden" in argv:
            hidden = int(argv[argv.index("--hidden") + 1])
        print(
            f"LIMIT OF THE CLAIM: users FIXED at {users}, hidden FIXED at {hidden}, "
            "never-active directories varied"
        )
        payload["unmarked_axis"] = _unmarked_axis(counts, users, hidden)
    elif mode == "compare":
        # Not a measurement: an assertion over two JSON files this script wrote.
        if "--before" not in argv or "--after" not in argv:
            print("compare needs --before <json> --after <json>")
            return 2
        return _compare(
            Path(argv[argv.index("--before") + 1]),
            Path(argv[argv.index("--after") + 1]),
        )
    elif mode == "real":
        store = Path(os.path.expanduser("~/.local-operator"))
        if "--store" in argv:
            store = Path(os.path.expanduser(argv[argv.index("--store") + 1]))
        mirror = Path(tempfile.mkdtemp(prefix="lo-bench-real-"))
        try:
            shape = _mirror_real_store(store, mirror)
            print(
                f"real store: {store} — mirrored READ-ONLY "
                f"({shape['dirs']} directories, {shape['marked']} marked, "
                f"{shape['with_transcript']} with a transcript)"
            )
            result = _steady_state(mirror)
            payload["real"] = {"users": result["rows"], "store": str(store), **shape, **result}
            print(
                f"  syscalls/poll={result['syscalls']}  scan={result['scan_syscalls']}  "
                f"rows={result['rows']}  "
                f"per listed session={result['syscalls'] / max(result['rows'], 1):.2f}"
            )
        finally:
            shutil.rmtree(mirror, ignore_errors=True)
    else:
        print(__doc__)
        return 2

    payload["loadavg_after"] = os.getloadavg()
    if out_json:
        Path(out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
