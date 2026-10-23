#!/usr/bin/env python3
"""Diagnostic-only, one allocation of three bounded real connection-setup cases.

Imports stay stdlib-only until child()'s explicit product boundary. Children inherit
run_bounded's owned PGID so an outer timeout can reap every descendant. No runtime,
capture.persist, copied schema, stress, retry, or intentional signal is involved.
"""
import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from checker import (MODES, check_all, observation, negative_controls, qualified_exception,
                     resource_data, NATIVE_IMAGE, NATIVE_EXE, NATIVE_SQLITE, own_process, settled_for_cleanup)

PINNED = {"local_operator/memory.py": "a11dbc59357fb103a79c38086fa68e18bcced685",
          "local_operator/memory_capture.py": "86f5072abf06e8ec4d6d4acd8d10982d68782b3c"}
DEPENDENCIES = {"pydantic": "2.14.1", "pydantic-core": "2.50.1",
                "annotated-types": "0.8.0", "typing-extensions": "4.16.0",
                "typing-inspection": "0.4.4"}
ALLOWED = {"pydantic", "pydantic_core", "annotated_types", "typing_extensions",
           "typing_inspection"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(root, head):
    result = {"head": head, "runtime": {
        "executable": str(Path(sys.executable).resolve()),
        "executable_sha256": sha(sys.executable),
    }}
    for rel, expected in PINNED.items():
        data = (root / rel).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != expected:
            raise ValueError("pinned producer changed")
        result[rel] = {"blob": blob, "sha256": hashlib.sha256(data).hexdigest(),
                       "path": str((root / rel).resolve())}
    return result


def proc(pid):
    """Bounded phase snapshots; absence/truncation is UNKNOWN, never zero."""
    result = {"pid": pid, "vma_count": "UNKNOWN", "status": "UNKNOWN"}
    try:
        keys = {"VmRSS", "VmSize", "RssAnon", "RssFile", "RssShmem", "Threads"}
        result["status"] = {k: "UNKNOWN" for k in keys}
        with open(f"/proc/{pid}/status", encoding="ascii") as stream:
            for line in stream:
                key, _, value = line.partition(":")
                if key in keys:
                    result["status"][key] = value.strip()
        size = count = 0
        with open(f"/proc/{pid}/maps", encoding="utf8") as stream:
            for line in stream:
                size += len(line.encode())
                if size > 1024 * 1024:
                    break
                count += 1
            else:
                result["vma_count"] = count
    except OSError:
        pass
    return result


def files(paths):
    rows = []
    for store_key, path in paths.items():
        for suffix in ("", "-wal", "-shm"):
            item = Path(str(path) + suffix)
            row = {"store_key": store_key, "basename": item.name, "exists": False}
            try:
                stat = item.lstat()
                row.update(exists=True, device=stat.st_dev, inode=stat.st_ino,
                           size=stat.st_size, nlink=stat.st_nlink)
            except FileNotFoundError:
                pass
            rows.append(row)
    return rows


def resources():
    return resource_data(sys.platform, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         {k: resource.getrlimit(getattr(resource, "RLIMIT_" + k))
                          for k in ("AS", "DATA", "STACK", "CORE")})


def child(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = Path(args.source_root).resolve()
    work = Path(args.work)
    source = identity(root, args.head)
    if (source["runtime"]["executable"] != args.expected_exe or
            source["runtime"]["executable_sha256"] != args.expected_exe_hash or
            sys.version_info[:3] != (3, 12, 15) or
            args.image_os != "ubuntu26" or args.image_version != NATIVE_IMAGE or
            source["runtime"]["executable_sha256"] != NATIVE_EXE):
        raise ValueError("unexpected child executable or Python target")
    lock = threading.Lock()
    count = size = 0

    def emit(phase, **fields):
        nonlocal count, size
        with lock:
            text = json.dumps(dict(index=count, monotonic=time.monotonic(),
                                   thread_id=threading.get_ident(), phase=phase, **fields))
            if count >= 200 or len(text) > 16384 or size + len(text) > 131072:
                raise ValueError("receipt output bound")
            print(text, flush=True)
            count += 1
            size += len(text)

    with open(work / "fatal.raw", "wb", buffering=0) as dump:
        faulthandler.enable(file=dump, all_threads=True)
        versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
        if versions != DEPENDENCIES:
            raise ValueError("dependency versions changed")
        stat = os.statvfs(work)
        emit("before_product_import", source=source, pid=os.getpid(), ppid=os.getppid(),
             pgid=os.getpgrp(), executable=str(Path(sys.executable).resolve()),
             executable_sha256=sha(sys.executable), python=sys.version,
             platform=platform.platform(), arch=platform.machine(), dependencies=versions,
             python_info=list(sys.version_info[:3]),
             ImageOS=args.image_os, ImageVersion=args.image_version,
             image_equivalence="UNKNOWN: family pinned, exact image version may differ",
             statvfs_bavail=stat.f_bavail, statvfs_frsize=stat.f_frsize,
             telemetry=proc(os.getpid()), core_limit=resource.getrlimit(resource.RLIMIT_CORE)[0], resources=resources())
        sys.path.insert(0, str(root))
        from local_operator.memory import MemoryIdentity, MemoryStore
        import local_operator.memory as memory
        import local_operator.memory_capture as capture
        loaded = {"head": args.head, "runtime": source["runtime"]}
        for rel, module in (("local_operator/memory.py", memory),
                            ("local_operator/memory_capture.py", capture)):
            path = Path(module.__file__).resolve()
            if path != (root / rel).resolve() or sha(path) != source[rel]["sha256"]:
                raise ValueError("imported source mismatch")
            loaded[rel] = source[rel]
        external = {}
        for name, module in tuple(sys.modules.items()):
            path = str(getattr(module, "__file__", "") or "")
            if "/site-packages/" in path or "/dist-packages/" in path:
                family = name.split(".")[0]
                if family not in ALLOWED:
                    raise ValueError("unexpected external import family")
                external[family] = path
        if MemoryStore.__module__ != "local_operator.memory":
            raise ValueError("constructor provenance")
        # These are native SQLite calls on a DIFFERENT in-memory DB. Label them
        # honestly; they do not initialize or warm up a candidate database.
        import sqlite3
        import _sqlite3
        if sqlite3.sqlite_version != "3.46.1" or sha(_sqlite3.__file__) != NATIVE_SQLITE:
            raise ValueError("native SQLite identity changed; no equivalence override")
        emit("metadata_sqlite_start")
        metadata_db = sqlite3.connect(":memory:")
        try:
            options = [r[0] for r in metadata_db.execute("PRAGMA compile_options")]
            if not options or len(options) > 128:
                raise ValueError("compile options bound")
        finally:
            metadata_db.close()
        linked = subprocess.run(["/usr/bin/ldd", _sqlite3.__file__], check=True,
                                capture_output=True, text=True, timeout=2)
        if len(linked.stdout) > 8192:
            raise ValueError("linked library evidence bound")
        emit("metadata_sqlite_done", version=sqlite3.sqlite_version,
             threadsafety=sqlite3.threadsafety, compile_options=options,
             binary=_sqlite3.__file__, binary_sha256=sha(_sqlite3.__file__),
             linked_libraries=linked.stdout, external_origins=external)
        workers = 2
        barrier = threading.Barrier(2, timeout=3) if workers == 2 else None
        stores = [MemoryStore(config=work / (f"separate-{i}" if args.mode == "separate-db"
                                             else "shared"),
                              identity=MemoryIdentity(session_id="native-probe"),
                              session_dir=None, origin="d_probe") for i in range(workers)]
        paths = {(f"slot-{i}" if args.mode == "separate-db" else "shared"): store.path
                 for i, store in enumerate(stores)}
        emit("before_workers", loaded_source=loaded, telemetry=proc(os.getpid()),
             synthetic_files=files(paths), resources=resources())

        state = {}
        failures = []
        serialize = threading.Lock()

        def phase(slot, iteration, label):
            with lock:
                state[slot] = dict(slot=slot, iteration=iteration, phase=label)

        def worker(slot):
            try:
                for iteration in range(8):
                    phase(slot, iteration, "before_emit")
                    emit("before_connection", slot=slot, iteration=iteration)
                    phase(slot, iteration, "before_barrier")
                    barrier.wait()  # BEFORE harness lock, never inside it.
                    phase(slot, iteration, "after_barrier")
                    with serialize if args.mode == "same-db-serialized" else nullcontext():
                        phase(slot, iteration, "before_context")
                        with stores[slot].connection() as db:
                            phase(slot, iteration, "entered_context")
                            emit("connection_entered", slot=slot, iteration=iteration)
                            token = f"probe-{slot}-{iteration}"
                            phase(slot, iteration, "before_sql")
                            db.execute("INSERT INTO captures(session,compaction,status,n) "
                                       "VALUES(?,?,?,?)", ("native-probe", token, "ok", 0))
                            db.commit()
                            row = db.execute("SELECT status,n FROM captures "
                                             "WHERE session=? AND compaction=?",
                                             ("native-probe", token)).fetchone()
                            phase(slot, iteration, "after_sql")
                            if tuple(row) != ("ok", 0):
                                raise ValueError("synthetic row oracle")
                            phase(slot, iteration, "before_exit")
                        phase(slot, iteration, "closed")
                        emit("connection_closed", slot=slot, iteration=iteration)
            except BaseException as exc:
                with lock:
                    failures.append(dict(state[slot], exception=qualified_exception(exc, str(root))))
                barrier.abort()
                raise

        try:
            # Context manager PHYSICALLY joins both workers, including peer abort.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(worker, range(workers)))
        finally:
            emit("after_workers", telemetry=proc(os.getpid()), synthetic_files=files(paths),
                 resources=resources(), threads_joined=True, worker_failures=failures)
        emit("case_done", completions=16, threads_joined=True)
        faulthandler.disable()


def control(args):
    negative_controls()  # No product/SQLite import or child before this passes.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.source_root).resolve()
    expected = identity(root, args.head)
    rows = []
    tmp = tempfile.mkdtemp(prefix="native-6911-", dir=args.temp_root)
    active = None
    try:
        for mode in MODES:
            work = Path(tmp) / mode
            work.mkdir()
            env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH")
                   if k in os.environ}
            env.update(HOME=str(work), XDG_CONFIG_HOME=str(work / "config"),
                       XDG_CACHE_HOME=str(work / "cache"),
                       LOCAL_OPERATOR_CONFIG_DIR=str(work / "config"),
                       PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--child", "--mode", mode, "--work", str(work),
                       "--source-root", str(root), "--head", args.head,
                       "--image-os", args.image_os, "--image-version", args.image_version,
                       "--expected-exe", expected["runtime"]["executable"],
                       "--expected-exe-hash", expected["runtime"]["executable_sha256"]]
            # No start_new_session: child MUST inherit bounded wrapper's PGID.
            with open(work / "stdout", "wb") as stdout, open(work / "stderr", "wb") as stderr:
                # Assignment is inside the lifetime try: any subsequent failure
                # leaves active non-None, preventing deletion of a live DB root.
                active = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                def bounded_wait(seconds):
                    try:
                        return active.wait(timeout=seconds)
                    except subprocess.TimeoutExpired as exc:
                        raise TimeoutError("child wait bound") from exc
                row = own_process(active, os.getpgid, bounded_wait, proc)
                row.update(mode=mode, controller_pid=os.getpid(), controller_pgid=os.getpgrp(),
                           events=[], native_fault_address="UNKNOWN", bus_subcode="UNKNOWN",
                           child_wait_code="UNKNOWN: Popen.wait does not expose waitid code")
                if settled_for_cleanup(row):
                    active = None
                else:
                    rows.append(row)
                    (out / "receipts.json").write_text(json.dumps(rows, indent=2))
                    raise RuntimeError("child unsettled; synthetic root retained")
            with (work / "stdout").open("rb") as stream:
                data = stream.read(131073)
            if len(data) > 131072:
                raise ValueError("child output overflow")
            try:
                row["events"] = [json.loads(line) for line in data.splitlines()]
            except (ValueError, UnicodeError) as exc:
                row["receipt_parse_error"] = type(exc).__name__
            with (work / "stderr").open("rb") as stream:
                error = stream.read(4096)
            try:
                row["exception_summary"] = json.loads(error) if error else None
            except (ValueError, UnicodeError):
                row["exception_summary"] = "UNKNOWN: non-protocol stderr (not uploaded)"
            row["signal"] = -row["rc"] if isinstance(row["rc"], int) and row["rc"] < 0 else None
            fatal = work / "fatal.raw"
            row["dump_bytes"] = fatal.stat().st_size if fatal.exists() else None
            if fatal.exists():
                with fatal.open("rb") as stream:
                    (out / f"{mode}-fatal.txt").write_bytes(stream.read(32768))
            row["stderr_bytes"] = (work / "stderr").stat().st_size
            rows.append(row)
            (out / "receipts.json").write_text(json.dumps(rows, indent=2))
            # Clean THIS case only after physical child reap, before any next mode.
            import shutil
            shutil.rmtree(work)
            row["case_root_removed"] = not work.exists()
            row["case_pass"] = observation(row, expected)
            (out / "receipts.json").write_text(json.dumps(rows, indent=2))
            # Only qualified ordinary failure can reach the next DISTINCT mode;
            # setup/signal/bound/privacy/settlement errors raise and stop here.
        result = check_all(rows, expected)
    finally:
        all_settled = active is None and all(settled_for_cleanup(row) for row in rows)
        if all_settled:
            # No TemporaryDirectory finalizer: failed final wait must retain DBs
            # until the outer wrapper physically terminates the inherited group.
            import shutil
            shutil.rmtree(tmp)
        (out / "cleanup.json").write_text(json.dumps({
            "all_children_settled": all_settled,
            "synthetic_root_removed": not Path(tmp).exists(),
            "reaped_cases": [r["mode"] for r in rows if r.get("reaped")],
            "no_success_inferred_from_cleanup": True,
        }))
    (out / "result.json").write_text(json.dumps(dict(result=result, synthetic_cleanup=True)))
    print(result)
    if result.startswith("FAIL:"):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out")
    parser.add_argument("--temp-root")
    parser.add_argument("--image-os", default="UNKNOWN")
    parser.add_argument("--image-version", default="UNKNOWN")
    parser.add_argument("--expected-exe")
    parser.add_argument("--expected-exe-hash")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--work")
    args = parser.parse_args()
    if args.child:
        try:
            child(args)
        except BaseException as exc:
            # Actual traceback STRUCTURE only; never messages/args/locals/SQL.
            print(json.dumps(qualified_exception(exc, str(Path(args.source_root).resolve()))),
                  file=sys.stderr)
            raise SystemExit(1) from None
    else:
        control(args)


if __name__ == "__main__":
    main()
