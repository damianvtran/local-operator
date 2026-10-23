"""Pure data oracle for ONE bounded diagnostic allocation, never a product fix.

No product, SQLite, SDK or subprocess imports: safe for pre-dispatch negatives.
A green result says only 48 tiny connection setups were unreproduced, not cured.
"""
from copy import deepcopy
import json
import re

MODES = ("same-db", "separate-db", "same-db-serialized")


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


def setup_proof(events, expected, mode, complete=True):
    """Mandatory provenance/lifecycle fields, with UNKNOWN telemetry kept honest."""
    if any(e.get("phase") not in PHASES for e in events):
        raise ValueError("unknown phase")
    unique = {}
    for phase in ("before_product_import", "metadata_sqlite_start", "metadata_sqlite_done",
                  "before_workers", "after_workers", *( ("case_done",) if complete else () )):
        found = [e for e in events if e.get("phase") == phase]
        if len(found) != 1:
            raise ValueError("missing mandatory phase: " + phase)
        unique[phase] = found[0]
    if list(e["index"] for e in unique.values()) != sorted(e["index"] for e in unique.values()):
        raise ValueError("setup phase ordering")
    start = unique["before_product_import"]
    if start.get("ImageOS") != "ubuntu26" or start.get("ImageVersion") != NATIVE_IMAGE:
        raise ValueError("native image nonequivalence STOP")
    if start.get("executable_sha256") != NATIVE_EXE:
        raise ValueError("native executable nonequivalence STOP")
    for phase in ("before_product_import", "before_workers", "after_workers"):
        check_resources(unique[phase].get("resources"))

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
    if meta.get("version") != "3.46.1" or meta.get("binary_sha256") != NATIVE_SQLITE:
        raise ValueError("SQLite native identity nonequivalence STOP")
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


def identity_proof(row, expected, complete=True):
    """Reject missing lifecycle/source evidence instead of inferring it from rc0."""
    if type(row.get("rc")) is not int or row.get("signal") is not None or row.get("memory_breach"):
        raise ValueError("signal, memory breach or malformed exit STOP")
    if row.get("lifecycle_errors"):
        raise ValueError("lifecycle observation or settlement error")
    if row.get("timed_out") or not row.get("reaped"):
        raise ValueError("timeout or missing reap")
    if row.get("child_pgid") != row.get("controller_pgid"):
        raise ValueError("child escaped owned process group")
    events = row.get("events", [])
    if [e.get("index") for e in events] != list(range(len(events))):
        raise ValueError("event ordering")
    setup_proof(events, expected, row["mode"], complete)
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


def check_case(row, expected):
    if row.get("exception_summary") is not None:
        check_exception(row["exception_summary"])
    if row.get("rc") != 0:
        check_exception(row.get("exception_summary")) if row.get("rc") == 1 else None
        raise ValueError("signal or nonzero child exit")
    identity_proof(row, expected)
    events = row["events"]
    workers = 2
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
    passes = [observation(row, expected) for row in rows]
    if not all(passes):
        return "FAIL: qualified ordinary exception observed; no root-cause or cure claim"
    return "UNREPRODUCED in three small cases; no root-cause or cure claim"


def fixture(mode="same-db"):
    """Synthetic complete protocol, never measured native metadata."""
    source = {"head": "a" * 40, "memory": "b" * 64,
              "runtime": {"executable": "/synthetic/python", "executable_sha256": NATIVE_EXE}}
    telemetry = dict(pid=11, status="UNKNOWN", vma_count="UNKNOWN")
    keys = ("slot-0", "slot-1") if mode == "separate-db" else ("shared",)
    files = [dict(store_key=k, basename=n, exists=False) for k in keys
             for n in ("memory.db", "memory.db-wal", "memory.db-shm")]
    events = [dict(phase="before_product_import", source=source, pid=11, ppid=10,
                   pgid=10, **source["runtime"], python="3.12.15 synthetic",
                   python_info=[3, 12, 15], dependencies=dict(TARGET_DEPS), core_limit=0,
                   platform="synthetic", arch="synthetic", ImageOS="ubuntu26",
                   ImageVersion=NATIVE_IMAGE, image_equivalence="UNKNOWN", statvfs_bavail=1,
                   statvfs_frsize=1, telemetry=telemetry),
              dict(phase="metadata_sqlite_start"),
              dict(phase="metadata_sqlite_done", version="3.46.1", threadsafety=3,
                   binary="/synthetic/sqlite", binary_sha256=NATIVE_SQLITE,
                   linked_libraries="synthetic", compile_options=["synthetic"],
                   external_origins={"pydantic": "/synthetic"}),
              dict(phase="before_workers", loaded_source=source,
                   telemetry=telemetry, synthetic_files=files)]
    workers = 2
    for slot in range(workers):
        for i in range(16 // workers):
            events.extend(dict(phase=p, slot=slot, iteration=i) for p in
                          ("before_connection", "connection_entered", "connection_closed"))
    events.extend([dict(phase="after_workers", telemetry=telemetry, synthetic_files=files),
                   dict(phase="case_done", completions=16, threads_joined=True)])
    limits = {k: (-1, -1) for k in ("AS", "DATA", "STACK")}
    limits["CORE"] = (0, 0)
    for event in events:
        if event["phase"] in ("before_product_import", "before_workers", "after_workers"):
            event["resources"] = resource_data("linux", 42, limits)
    for index, event in enumerate(events):
        event["index"] = index
    return source, dict(mode=mode, rc=0, timed_out=False, reaped=True,
                       child_pid=11, child_pgid=10, controller_pid=10,
                       controller_pgid=10, events=events, case_root_removed=True, signal=None)


def negative_controls():
    """Run before native dispatch: every mutation must hit its named oracle."""
    # Validate attribution privacy/cause/continuation BEFORE any product dispatch.
    source, protocol = fixture()
    frame = dict(domain="repo", file="local_operator/memory.py", function="connection", line=960)
    entry = dict(type_module="builtins", type_qualname="ValueError", relation="root",
                 link="none", frames=[frame], errno=None, sqlite_errorcode=None,
                 sqlite_errorname=None)
    summary = dict(chain=[entry], truncated=False)
    check_exception(summary)
    dirty = deepcopy(protocol)
    dirty["exception_summary"] = {"message": "PRIVATE"}
    try:
        check_case(dirty, source)
    except ValueError as error:
        if "privacy" not in str(error):
            raise AssertionError("privacy wrong oracle") from error
    else:
        raise AssertionError("privacy false green")
    failed = deepcopy(protocol)
    failed.update(rc=1, exception_summary=summary)
    failed["events"] = [e for e in failed["events"] if e["phase"] not in
                        ("before_connection", "connection_entered", "connection_closed", "case_done")]
    failed["events"][-1].update(threads_joined=True, worker_failures=[
        dict(slot=0, iteration=0, phase="before_context", exception=summary)])
    for index, event in enumerate(failed["events"]):
        event["index"] = index
    if observation(failed, source) is not False:
        raise AssertionError("ordinary failure must not pass")
    if not check_all([failed] + [fixture(m)[1] for m in MODES[1:]], source).startswith("FAIL:"):
        raise AssertionError("continuation must retain final failure")
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

# These helpers inspect only Python exception/data objects; they never import a
# product, SQLite, resource module or process API. Missing attribution is STOP.
def qualified_exception(exc, root):
    chain = []
    seen = set()
    relation = "root"
    while exc is not None and len(chain) < 3 and id(exc) not in seen:
        seen.add(id(exc))
        tail, repo = [], []
        tb = exc.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            path = code.co_filename.replace("\\", "/")
            prefix = root.rstrip("/") + "/"
            domain = "repo" if path.startswith(prefix) else "other"
            name = path[len(prefix):] if domain == "repo" else path.rsplit("/", 1)[-1]
            name = name if re.fullmatch(r"[A-Za-z0-9_./-]{1,140}", name) else "UNKNOWN"
            fn = code.co_name
            fn = fn if re.fullmatch(r"[A-Za-z0-9_.<>-]{1,80}", fn) else "UNKNOWN"
            frame = dict(domain=domain, file=name, function=fn, line=tb.tb_lineno)
            tail = (tail + [frame])[-4:]
            if domain == "repo":
                repo = (repo + [frame])[-3:]
            tb = tb.tb_next
        frames = repo + ([tail[-1]] if tail and tail[-1] not in repo else []) if repo else tail
        cls = type(exc)
        module, qualname = cls.__module__, cls.__qualname__
        module = module if isinstance(module, str) and len(module) <= 100 and re.fullmatch(IDENTIFIER, module) else "UNKNOWN"
        qualname = qualname if isinstance(qualname, str) and len(qualname) <= 100 and re.fullmatch(IDENTIFIER, qualname) else "UNKNOWN"
        values = {}
        for field in ("errno", "sqlite_errorcode", "sqlite_errorname"):
            value = getattr(exc, field, None)
            if field == "sqlite_errorname":
                valid = value is None or isinstance(value, str) and re.fullmatch(r"SQLITE_[A-Z0-9_]{1,60}", value)
            else:
                valid = value is None or type(value) is int and -(2**31) <= value < 2**31
            values[field] = value if valid else "UNKNOWN"
        next_exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
        link = "cause" if exc.__cause__ is not None else "context" if next_exc is not None else "none"
        chain.append(dict(type_module=module, type_qualname=qualname, relation=relation,
                          link=link, frames=frames, **values))
        exc, relation = next_exc, link
    result = dict(chain=chain, truncated=exc is not None)
    # Keep each exception's innermost frame even under the total JSON cap.
    while len(json.dumps(result).encode()) > 4096:
        largest = max(chain, key=lambda e: len(e["frames"]))
        if len(largest["frames"]) <= 1:
            raise ValueError("qualified exception output bound")
        largest["frames"].pop(0)
    return result


IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,10}"
WORKER_PHASES = {"before_emit", "before_barrier", "after_barrier", "before_context",
                 "entered_context", "before_sql", "after_sql", "before_exit", "closed"}
NATIVE_EXE = "6c8fa2fe8c89050f2bfcb012d82996604564d296081e5892bd64c14ee243c8a7"
NATIVE_SQLITE = "bab68aaf9734c6aac70d83d317bfdbe93f2f2f62ba5102bb4c265eedca9ffba9"
NATIVE_IMAGE = "20261012.163.1"


def check_exception(summary):
    if not isinstance(summary, dict) or set(summary) != {"chain", "truncated"}:
        raise ValueError("qualified exception privacy/schema")
    if summary["truncated"] is not False or len(json.dumps(summary).encode()) > 4096:
        raise ValueError("qualified exception bound/truncated")
    chain = summary["chain"]
    if not isinstance(chain, list) or not 1 <= len(chain) <= 3:
        raise ValueError("qualified exception cause bound")
    count = 0
    for i, entry in enumerate(chain):
        if not isinstance(entry, dict) or set(entry) != {"type_module", "type_qualname", "relation", "link", "frames", "errno", "sqlite_errorcode", "sqlite_errorname"}:
            raise ValueError("qualified exception privacy/fields")
        for field in ("type_module", "type_qualname"):
            value = entry[field]
            if not isinstance(value, str) or len(value) > 160 or value == "UNKNOWN" or len(value) > 100 or not re.fullmatch(IDENTIFIER, value):
                raise ValueError("qualified exception type identity")
        if entry["relation"] != ("root" if i == 0 else chain[i-1]["link"]):
            raise ValueError("qualified exception cause order")
        if entry["link"] != ("none" if i == len(chain)-1 else chain[i+1]["relation"]) or entry["link"] not in {"cause", "context", "none"}:
            raise ValueError("qualified exception cause missing")
        if entry["type_module"] == "local_operator.memory" and entry["type_qualname"] == "MemoryError" and entry["link"] == "none":
            raise ValueError("qualified custom MemoryError cause missing")
        for key in ("errno", "sqlite_errorcode"):
            value = entry[key]
            if value is not None and (type(value) is not int or not -(2**31) <= value < 2**31):
                raise ValueError("qualified numeric enum")
        name = entry["sqlite_errorname"]
        if name is not None and (not isinstance(name, str) or not re.fullmatch(r"SQLITE_[A-Z0-9_]{1,60}", name)):
            raise ValueError("qualified SQLite enum")
        if (name is None) != (entry["sqlite_errorcode"] is None):
            raise ValueError("qualified SQLite enum pair")
        frames = entry["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError("qualified exception missing frames")
        count += len(frames)
        for frame in frames:
            if not isinstance(frame, dict) or set(frame) != {"domain", "file", "function", "line"} or frame["domain"] not in {"repo", "other"}:
                raise ValueError("qualified frame privacy")
            name = frame["file"]
            if not isinstance(name, str) or name == "UNKNOWN" or not re.fullmatch(r"[A-Za-z0-9_./-]{1,140}", name) or name.startswith("/") or ".." in name.split("/") or frame["domain"] == "other" and "/" in name:
                raise ValueError("qualified frame path privacy")
            if not isinstance(frame["function"], str) or not re.fullmatch(r"[A-Za-z0-9_.<>-]{1,80}", frame["function"]) or type(frame["line"]) is not int or frame["line"] <= 0:
                raise ValueError("qualified frame location")
    if count > 12:
        raise ValueError("qualified frame bound")


def resource_data(platform, highwater, limits):
    """Data conversion only. ru_maxrss is lifetime high-water, NOT PG/current RSS."""
    return dict(platform=platform, ru_maxrss=highwater,
                units="KiB" if platform == "linux" else "bytes" if platform == "darwin" else "UNKNOWN",
                limits={name: dict(soft=None if pair[0] == -1 else pair[0],
                                   hard=None if pair[1] == -1 else pair[1],
                                   soft_unlimited=pair[0] == -1, hard_unlimited=pair[1] == -1)
                        for name, pair in limits.items()})


def check_resources(value):
    if not isinstance(value, dict) or value.get("platform") != "linux" or value.get("units") != "KiB" or type(value.get("ru_maxrss")) not in (int, float) or not 0 <= value["ru_maxrss"] < 2**60:
        raise ValueError("resource highwater/units")
    limits = value.get("limits", {})
    if set(limits) != {"AS", "DATA", "STACK", "CORE"}:
        raise ValueError("resource limits missing")
    for fields in limits.values():
        if set(fields) != {"soft", "hard", "soft_unlimited", "hard_unlimited"}:
            raise ValueError("resource limits shape")
        for side in ("soft", "hard"):
            v, unlimited = fields[side], fields[side+"_unlimited"]
            if type(unlimited) is not bool or (v is None) != unlimited or v is not None and (type(v) is not int or v < 0):
                raise ValueError("resource unlimited marker")
    if limits["CORE"]["soft"] != 0 or limits["CORE"]["hard"] != 0:
        raise ValueError("resource core must remain disabled")


def observation(row, expected):
    """A qualified ordinary failure can continue, but can NEVER become case_pass."""
    if row.get("exception_summary") is not None:
        check_exception(row["exception_summary"])
    if row.get("rc") == 0:
        if row.get("exception_summary") is not None:
            raise ValueError("unexpected exception in successful case")
        check_case(row, expected)
        return True
    if row.get("rc") != 1:
        raise ValueError("signal or nonordinary child exit STOP")
    check_exception(row.get("exception_summary"))
    identity_proof(row, expected, complete=False)
    after = next(e for e in row["events"] if e["phase"] == "after_workers")
    if after.get("threads_joined") is not True or not row.get("case_root_removed"):
        raise ValueError("unsettled failure or case cleanup STOP")
    failed = after.get("worker_failures")
    if not isinstance(failed, list) or not 1 <= len(failed) <= 2:
        raise ValueError("qualified worker attribution missing")
    if len({e.get("slot") for e in failed}) != len(failed):
        raise ValueError("duplicate worker failure")
    for entry in failed:
        if set(entry) != {"slot", "iteration", "phase", "exception"} or type(entry["slot"]) is not int or entry["slot"] not in (0, 1) or type(entry["iteration"]) is not int or not 0 <= entry["iteration"] < 8 or entry["phase"] not in WORKER_PHASES:
            raise ValueError("qualified worker phase")
        check_exception(entry["exception"])
    return False
