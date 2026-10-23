"""Pure data oracle for ONE bounded diagnostic allocation, never a product fix.

No product, SQLite, SDK or subprocess imports: safe for pre-dispatch negatives.
A green result says only 48 tiny connection setups were unreproduced, not cured.
"""
from copy import deepcopy

MODES = ("sequential", "same-db", "separate-db")


TARGET_DEPS = {"pydantic": "2.14.1", "pydantic-core": "2.50.1",
               "annotated-types": "0.8.0", "typing-extensions": "4.16.0",
               "typing-inspection": "0.4.4"}
PHASES = {"before_product_import", "metadata_sqlite_start", "metadata_sqlite_done",
          "before_workers", "before_connection", "connection_entered",
          "connection_closed", "after_workers", "case_done"}


def own_process(process, observe_pgid, wait, snapshot):
    """Injected lifecycle only: no kernel calls here except through supplied fakes/API.

    Observation cannot bypass wait. Failed final wait leaves reaped=False; callers
    must preserve the sandbox, not let TemporaryDirectory delete a live database.
    """
    row = dict(child_pid=process.pid, child_pgid="UNKNOWN", rc="UNKNOWN",
               timed_out=False, reaped=False, lifecycle_errors=[])
    try:
        try:
            row["child_pgid"] = observe_pgid(process.pid)
        except Exception as exc:
            row["lifecycle_errors"].append(type(exc).__name__)
        try:
            wait(10)
        except TimeoutError:
            row["timed_out"] = True
            row["final_live_snapshot"] = snapshot(process.pid)
        except Exception as exc:
            row["lifecycle_errors"].append(type(exc).__name__)
    finally:
        try:
            if process.poll() is None:
                process.kill()
        except Exception as exc:
            row["lifecycle_errors"].append(type(exc).__name__)
        try:
            row["rc"] = wait(5)
            row["reaped"] = type(row["rc"]) is int
        except Exception as exc:
            row["lifecycle_errors"].append(type(exc).__name__)
    return row


def settled_for_cleanup(row):
    return isinstance(row, dict) and row.get("reaped") is True


def setup_proof(events, expected, mode):
    """Mandatory provenance/lifecycle fields, with UNKNOWN telemetry kept honest."""
    if any(e.get("phase") not in PHASES for e in events):
        raise ValueError("unknown phase")
    unique = {}
    for phase in ("before_product_import", "metadata_sqlite_start", "metadata_sqlite_done",
                  "before_workers", "after_workers", "case_done"):
        found = [e for e in events if e.get("phase") == phase]
        if len(found) != 1:
            raise ValueError("missing mandatory phase: " + phase)
        unique[phase] = found[0]
    if list(e["index"] for e in unique.values()) != sorted(e["index"] for e in unique.values()):
        raise ValueError("setup phase ordering")
    start = unique["before_product_import"]
    runtime = expected.get("runtime", {})
    for field, value in (("executable", runtime.get("executable")),
                         ("executable_sha256", runtime.get("executable_sha256"))):
        if not isinstance(start.get(field), str) or not value or start[field] != value:
            raise ValueError("missing or wrong executable identity")
    digest = start["executable_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("executable hash shape")
    if start.get("python_info") != [3, 12, 15] or not start.get("python", "").startswith("3.12.15"):
        raise ValueError("wrong Python target")
    if start.get("dependencies") != TARGET_DEPS:
        raise ValueError("missing or wrong dependencies")
    if type(start.get("core_limit")) is not int or start["core_limit"] != 0:
        raise ValueError("core limit must be disabled")
    for key in ("platform", "arch", "ImageOS", "ImageVersion", "image_equivalence"):
        if not isinstance(start.get(key), str) or not start[key]:
            raise ValueError("missing setup metadata")
    for key in ("statvfs_bavail", "statvfs_frsize"):
        if type(start.get(key)) is not int or start[key] < 0:
            raise ValueError("missing disk metadata")
    meta = unique["metadata_sqlite_done"]
    if not all(isinstance(meta.get(k), str) and meta[k] for k in
               ("version", "binary", "binary_sha256", "linked_libraries")):
        raise ValueError("missing SQLite metadata")
    if len(meta["binary_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in meta["binary_sha256"]):
        raise ValueError("SQLite hash")
    if type(meta.get("threadsafety")) is not int or meta["threadsafety"] not in (0, 1, 3):
        raise ValueError("SQLite threadsafety")
    options = meta.get("compile_options")
    if not isinstance(options, list) or not 0 < len(options) <= 128 or not all(isinstance(x, str) and 0 < len(x) <= 512 for x in options):
        raise ValueError("SQLite compile metadata")
    if not isinstance(meta.get("external_origins"), dict) or not meta["external_origins"]:
        raise ValueError("missing external origins")
    for phase in ("before_product_import", "before_workers", "after_workers"):
        telemetry = unique[phase].get("telemetry")
        if not isinstance(telemetry, dict) or not {"pid", "status", "vma_count"} <= telemetry.keys():
            raise ValueError("missing phase telemetry")
    allowed = {"slot-0", "slot-1"} if mode == "separate-db" else {"shared"}
    for phase in ("before_workers", "after_workers"):
        rows = unique[phase].get("synthetic_files")
        if not isinstance(rows, list) or len(rows) != 3 * len(allowed) or {r.get("store_key") for r in rows} != allowed:
            raise ValueError("missing synthetic store attribution")
        if {(r.get("store_key"), r.get("basename")) for r in rows} != {(k, n) for k in allowed for n in ("memory.db", "memory.db-wal", "memory.db-shm")}:
            raise ValueError("duplicate synthetic inode attribution")
    before = unique["before_workers"]["index"]
    after = unique["after_workers"]["index"]
    if any(not before < e["index"] < after for e in events if e["phase"] in ("before_connection", "connection_entered", "connection_closed")):
        raise ValueError("worker phase ordering")


def check_case(row, expected):
    """Reject missing lifecycle/source evidence instead of inferring it from rc0."""
    if row.get("lifecycle_errors"):
        raise ValueError("lifecycle observation or settlement error")
    if row.get("rc") != 0:
        raise ValueError("signal or nonzero child exit")
    if row.get("timed_out") or not row.get("reaped"):
        raise ValueError("timeout or missing reap")
    if row.get("child_pgid") != row.get("controller_pgid"):
        raise ValueError("child escaped owned process group")
    events = row.get("events", [])
    if [e.get("index") for e in events] != list(range(len(events))):
        raise ValueError("event ordering")
    setup_proof(events, expected, row["mode"])
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
    """Synthetic complete protocol, never measured native metadata."""
    source = {"head": "a" * 40, "memory": "b" * 64,
              "runtime": {"executable": "/synthetic/python", "executable_sha256": "c" * 64}}
    telemetry = dict(pid=11, status="UNKNOWN", vma_count="UNKNOWN")
    keys = ("slot-0", "slot-1") if mode == "separate-db" else ("shared",)
    files = [dict(store_key=k, basename=n, exists=False) for k in keys
             for n in ("memory.db", "memory.db-wal", "memory.db-shm")]
    events = [dict(phase="before_product_import", source=source, pid=11, ppid=10,
                   pgid=10, **source["runtime"], python="3.12.15 synthetic",
                   python_info=[3, 12, 15], dependencies=dict(TARGET_DEPS), core_limit=0,
                   platform="synthetic", arch="synthetic", ImageOS="UNKNOWN",
                   ImageVersion="UNKNOWN", image_equivalence="UNKNOWN", statvfs_bavail=1,
                   statvfs_frsize=1, telemetry=telemetry),
              dict(phase="metadata_sqlite_start"),
              dict(phase="metadata_sqlite_done", version="synthetic", threadsafety=3,
                   binary="/synthetic/sqlite", binary_sha256="d" * 64,
                   linked_libraries="synthetic", compile_options=["synthetic"],
                   external_origins={"pydantic": "/synthetic"}),
              dict(phase="before_workers", loaded_source=source,
                   telemetry=telemetry, synthetic_files=files)]
    workers = 1 if mode == "sequential" else 2
    for slot in range(workers):
        for i in range(16 // workers):
            events.extend(dict(phase=p, slot=slot, iteration=i) for p in
                          ("before_connection", "connection_entered", "connection_closed"))
    events.extend([dict(phase="after_workers", telemetry=telemetry, synthetic_files=files),
                   dict(phase="case_done", completions=16, threads_joined=True)])
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
        ("missing done", lambda r: r["events"].pop(), "case_done"),
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
