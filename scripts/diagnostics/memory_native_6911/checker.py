"""Pure data oracle for ONE bounded diagnostic allocation, never a product fix.

No product, SQLite, SDK or subprocess imports: safe for pre-dispatch negatives.
A green result says only 48 tiny connection setups were unreproduced, not cured.
"""
from copy import deepcopy

MODES = ("sequential", "same-db", "separate-db")


def check_case(row, expected):
    """Reject missing lifecycle/source evidence instead of inferring it from rc0."""
    if row.get("rc") != 0:
        raise ValueError("signal or nonzero child exit")
    if row.get("timed_out") or not row.get("reaped"):
        raise ValueError("timeout or missing reap")
    if row.get("child_pgid") != row.get("controller_pgid"):
        raise ValueError("child escaped owned process group")
    events = row.get("events", [])
    if [e.get("index") for e in events] != list(range(len(events))):
        raise ValueError("event ordering")
    starts = [e for e in events if e.get("phase") == "before_product_import"]
    if len(starts) != 1 or starts[0].get("source") != expected:
        raise ValueError("wrong source identity")
    start = starts[0]
    if (start.get("pid"), start.get("ppid"), start.get("pgid")) != (
        row.get("child_pid"), row.get("controller_pid"), row.get("controller_pgid")
    ):
        raise ValueError("process lineage")
    loaded = [e for e in events if e.get("phase") == "before_workers"]
    if len(loaded) != 1 or loaded[0].get("loaded_source") != expected:
        raise ValueError("imported source identity")
    if not any(e.get("phase") == "metadata_sqlite_done" for e in events):
        raise ValueError("missing SQLite metadata")
    workers = 1 if row["mode"] == "sequential" else 2
    pairs = {(slot, i) for slot in range(workers) for i in range(16 // workers)}
    for phase in ("before_connection", "connection_entered", "connection_closed"):
        observed = [(e.get("slot"), e.get("iteration")) for e in events
                    if e.get("phase") == phase]
        if len(observed) != 16 or set(observed) != pairs:
            raise ValueError("partial or duplicate connection completion")
    for slot, iteration in pairs:
        ordered = [next(e["index"] for e in events if e.get("phase") == phase
                        and e.get("slot") == slot and e.get("iteration") == iteration)
                   for phase in ("before_connection", "connection_entered", "connection_closed")]
        if ordered != sorted(ordered):
            raise ValueError("connection lifecycle ordering")
    done = [e for e in events if e.get("phase") == "case_done"]
    if len(done) != 1 or done[0].get("completions") != 16:
        raise ValueError("missing done or partial completion")
    if done[0].get("threads_joined") is not True or events[-1] != done[0]:
        raise ValueError("missing physical thread settlement")


def check_all(rows, expected):
    if [r.get("mode") for r in rows] != list(MODES):
        raise ValueError("missing extra or duplicate case")
    for row in rows:
        check_case(row, expected)
    return "UNREPRODUCED in three small cases; no root-cause or cure claim"


def fixture(mode="sequential"):
    """Synthetic protocol only, not a mock of MemoryStore or a native PASS."""
    source = {"head": "a" * 40, "memory": "b" * 64}
    events = [dict(phase="before_product_import", source=source, pid=11, ppid=10,
                   pgid=10), dict(phase="metadata_sqlite_done"),
              dict(phase="before_workers", loaded_source=source)]
    workers = 1 if mode == "sequential" else 2
    for slot in range(workers):
        for i in range(16 // workers):
            events.extend(dict(phase=p, slot=slot, iteration=i) for p in
                          ("before_connection", "connection_entered", "connection_closed"))
    events.append(dict(phase="case_done", completions=16, threads_joined=True))
    for index, event in enumerate(events):
        event["index"] = index
    return source, dict(mode=mode, rc=0, timed_out=False, reaped=True,
                       child_pid=11, child_pgid=10, controller_pid=10,
                       controller_pgid=10, events=events)


def negative_controls():
    """Run before native dispatch: every mutation must hit its named oracle."""
    source, good = fixture()
    for name, mutate, message in (
        ("wrong source", lambda r: r["events"][0].update(source={}), "source identity"),
        ("missing done", lambda r: r["events"].pop(), "missing done"),
        ("signal", lambda r: r.update(rc=-7), "signal"),
        ("timeout", lambda r: r.update(timed_out=True), "timeout"),
        ("partial", lambda r: r["events"][-1].update(completions=15), "partial"),
    ):
        row = deepcopy(good)
        mutate(row)
        try:
            check_case(row, source)
        except ValueError as exc:
            if message not in str(exc):
                raise AssertionError(f"{name}: wrong oracle") from exc
        else:
            raise AssertionError(f"{name}: accepted invalid evidence")
    check_case(good, source)
    rows = [fixture(m)[1] for m in MODES]
    check_all(rows, source)
    rows[2] = deepcopy(rows[1])
    try:
        check_all(rows, source)
    except ValueError as exc:
        if "duplicate case" not in str(exc):
            raise AssertionError("wrong duplicate oracle") from exc
    else:
        raise AssertionError("duplicate case accepted")
