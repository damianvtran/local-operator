"""Pure stdlib protocol, not app/native proof. Unexpected or missing evidence stops."""
import json

# Per-case budget. Derived from the OBSERVED whole first S2 case wall of 10.3 s
# (run 39921114526) + the 5 s SlowListing test-fixture worker closure + the 3 s
# physical reap allowance (1 s healthy wait + 2 s final wait in finish_child).
# This is a CONSERVATIVE ceiling, not a measured future latency, and it leaves
# the original CPU500/wall2000/noise/calibration bars and every other policy
# (job 5 min, setup 120 s, 1024 MiB, 250 records, 256 KiB output, 4 serial
# cases) untouched. 4 x (20 + 3) = 92 s < 100 s outer; no automatic further raise.
CHILD_SECONDS = 20
REAP_SECONDS = 3
CASE_COUNT = 4
OUTER_SECONDS = 100
OUTER_GRACE_SECONDS = 5

MODES = ("s2-original", "s3-split", "s3-held", "synthetic")


def judge(row):
    if row.get("reaped") is not True or row.get("timed_out") or row.get("signal") or row.get("cleanup_errors"):
        raise ValueError("lifecycle STOP")
    if row.get("identity_ok") is not True:
        raise ValueError("source/runtime identity STOP")
    data = row.get("data")
    def private(value):
        if isinstance(value, dict):
            return any(k in {"locals", "args", "environment", "environ", "sourceText", "message", "sql", "token", "content"} or private(v) for k, v in value.items())
        return isinstance(value, list) and any(private(v) for v in value)
    if private(data):
        raise ValueError("privacy STOP")
    if not isinstance(data, dict) or len(json.dumps(data).encode()) > 200000:
        raise ValueError("missing/bounded protocol STOP")
    mode = row["mode"]
    if data.get("mode") != mode or mode not in MODES:
        raise ValueError("wrong mode STOP")
    if mode == "synthetic":
        controls = data.get("controls", [])
        if len(controls) != 2 or [c.get("mode") for c in controls] != ["sync", "offload"]:
            raise ValueError("missing clock control STOP")
        if not all(c.get("ready") is True and c.get("trailing") is True for c in controls):
            raise ValueError("probe baseline/trailing capture STOP")
        if controls[0].get("cpu_trip") is not True or controls[0].get("worst_cpu_ms", 0) < 50 or controls[1].get("cpu_trip") is not False or controls[1].get("worst_cpu_ms", 999) >= 50 or row.get("rc") != 0:
            raise ValueError("strict CPU oracle STOP")
        if controls[1].get("worker_cpu_ms", 0) < 200 or controls[1].get("worker_thread") == controls[1].get("loop_thread"):
            raise ValueError("matched offload CPU/identity STOP")
        return "qualified clock controls"
    if not data.get("setup_ok") or data.get("call") not in ("pass", "fail"):
        raise ValueError("setup/call missing STOP")
    records = data.get("records", [])
    clean = [r for r in records if r.get("kind") == "cleanup"]
    if len(records) > 250 or any(r.get("kind") == "overflow" for r in records):
        raise ValueError("measurement record capacity overflow STOP")
    known = RecordBudget.CRITICAL | RecordBudget.OPTIONAL | {"scope"}
    if any(r.get("kind") not in known for r in records):
        raise ValueError("unknown measurement record STOP")
    scopes = [r for r in records if r.get("kind") == "scope"]
    outcomes = [r for r in records if r.get("kind") == "outcome"]
    if len(scopes) != 1 or scopes[0].get("supplemental_gc") != "NOT_COLLECTED" or len(outcomes) != 1:
        raise ValueError("missing critical scope/outcome proof STOP")
    if outcomes[0].get("call") != data.get("call") or outcomes[0].get("exitstatus") != row.get("rc"):
        raise ValueError("critical outcome mismatch STOP")
    if len(clean) != 1:
        raise ValueError("missing cleanup proof STOP; child reap is separate")
    if not clean[0].get("worker_function_completed") or not clean[0].get("task_settled"):
        raise ValueError("observed incomplete worker function/task STOP")
    summaries = [r for r in records if r.get("kind") == "recorder"]
    if len(summaries) != 1:
        raise ValueError("missing original recorder STOP")
    if mode == "s2-original":
        callbacks = summaries[0].get("summary", {}).get("callbacks")
        if not isinstance(callbacks, dict) or "loop_thread_id" not in callbacks or not all("gc" in w.get("callbacks", {}) for key in ("worst_cpu_windows", "worst_wall_windows") for w in callbacks.get(key, [])):
            raise ValueError("S2 M15/thread GC missing STOP")
        for phase in ("s2_idle", "s2_emit_burst", "s2_settle"):
            if len([r for r in records if r.get("kind") == phase]) != 1:
                raise ValueError("missing S2 phase STOP")
        if data["call"] == "fail" and not str(data.get("assertion", "")).startswith("event loop "):
            raise ValueError("wrong S2 failure oracle STOP")
    else:
        harvests = [r for r in records if r.get("kind") == "harvest" and r.get("manual") and r.get("measured")]
        pauses = [r for r in records if r.get("kind") == "pause" and r.get("manual") and r.get("measured")]
        if len(harvests) != 3 or len(pauses) != 3:
            raise ValueError("incomplete observed S3 window STOP")
        window = pauses[-1]["end_wall"] - harvests[0]["start_wall"]
        parts = sum(r["wall_ms"] for r in harvests+pauses)/1000
        if window + 1e-9 < parts:
            raise ValueError("observed-window residual STOP")
        if mode == "s3-split":
            delay = [r for r in records if r.get("kind") == "controlled_async_delay"]
            if len(delay) != 1 or delay[0]["wall_ms"] < 600 or data["call"] != "fail" or row.get("rc") != 1 or not str(data.get("assertion", "")).startswith("harvest blocked for "):
                raise ValueError("expected OLD500 wall oracle STOP")
        else:
            witnesses = [r for r in records if r.get("kind") == "held_witness"]
            workers = [r for r in records if r.get("kind") == "worker_entry"]
            if len(witnesses) != 1 or not workers or witnesses[0].get("loop_ticks", 0) < 3 or witnesses[0].get("requests", 0) < 1 or any(w["thread"] == witnesses[0]["loop_thread"] for w in workers):
                raise ValueError("missing real offloaded worker/LOOP witness STOP")
            if data["call"] != "pass" or row.get("rc") != 0:
                raise ValueError("held original assertions failed STOP")
    if row.get("rc") not in (0, 1) or (data["call"] == "pass") != (row["rc"] == 0):
        raise ValueError("unexpected pytest outcome STOP")
    return "PASS original" if row["rc"] == 0 else "RAW FAIL retained; qualified expected assertion"


def final(rows):
    if [r.get("mode") for r in rows] != list(MODES):
        raise ValueError("missing/duplicate control STOP")
    verdicts = [judge(r) for r in rows]
    return dict(diagnostic_complete=True, original_failure=any(r["rc"] != 0 for r in rows),
                verdicts=verdicts, product_cause="UNKNOWN", source_contract_unchanged=True)


def finish_child(child, observe, kill_owned, parse, remove, fault=False):
    """Injected orchestration: EOF is not exit; final wait cannot be bypassed."""
    errors = []
    rc = None
    try:
        if not fault:
            try:
                rc = child.wait(timeout=1)
            except TimeoutError:
                pass
            except Exception:
                errors.append("healthy-wait-error")
    finally:
        try:
            if child.poll() is None:
                kill_owned()
        except Exception:
            errors.append("owned-kill-error")
            try:
                child.kill()
            except Exception:
                errors.append("direct-kill-error")
        finally:
            try:
                rc = child.wait(timeout=2)
            except Exception:
                errors.append("final-wait-error")
    try:
        if observe():
            errors.append("owned-unsettled")
    except Exception:
        errors.append("observation-error")
    data = None
    if type(rc) is int and not errors:
        try:
            data = project(parse())
            remove()
        except Exception:
            errors.append("proof-or-cleanup-error")
    return dict(rc=rc, reaped=type(rc) is int, cleanup_errors=errors, data=data)


def restore_all(patches, release, settle_task, complete):
    """Always attempt every owned cleanup, even partial installation/teardown."""
    errors = []
    for label, action in (("release", release), ("ticker", settle_task), ("work-completion", complete)):
        try:
            action()
        except BaseException:
            errors.append(label)
    for owner, name, previous in reversed(patches):
        try:
            setattr(owner, name, previous)
        except BaseException:
            errors.append("restore")
    return errors


def span_record(kind, start, end, thread, **values):
    return dict(kind=kind, cpu_ms=(end[0]-start[0])*1000,
                wall_ms=(end[1]-start[1])*1000, start_wall=start[1],
                end_wall=end[1], thread=thread, **values)


def held_record(ticks, requests, thread):
    return dict(kind="held_witness", loop_ticks=ticks, requests=requests,
                loop_thread=thread, origin="loop-witness")


def project(value):
    """Structural allowlist only; raw pytest logs NEVER leave the child pipe."""
    import math
    import re
    allowed = {"supplemental_gc", "excluded_gc", "dropped_count", "critical_dropped", "record_limit", "rc", "signal", "reaped", "timed_out", "cleanup_errors", "identity_ok", "data", "pid", "pgid", "verdict", "mode", "exitstatus", "records", "identity", "call", "assertion", "setup_ok",
               "controls", "ready", "trailing", "cpu_trip", "worst_cpu_ms", "worker_cpu_ms",
               "worker_thread", "loop_thread", "kind", "cpu_ms", "wall_ms", "start_wall",
               "end_wall", "thread", "manual", "measured", "i", "origin", "requests",
               "loop_ticks", "worker_function_completed", "task_settled", "errors", "summary",
               "worst_wall_ms", "noise_floor_ms", "wall_cpu_descheduling_ratio", "callbacks",
               "process_id", "loop_thread_id", "clock", "attribution", "worst_cpu_windows",
               "worst_wall_windows", "cpu_gap_ms", "wall_gap_ms", "count", "top_cpu", "top_wall",
               "gc", "name", "detail", "generation", "is_loop", "python", "textual", "sources",
               "loaded", "path", "sha256", "executable", "executable_sha256", "versions"}
    def visit(item, depth=0, mapping=False):
        if depth > 14:
            raise ValueError("privacy depth")
        if isinstance(item, dict):
            if any(not isinstance(k, str) or (not mapping and k not in allowed) for k in item):
                raise ValueError("privacy unknown fields")
            return {k: visit(v, depth+1, k in {"sources", "loaded", "versions"}) for k, v in item.items()}
        if isinstance(item, list):
            if len(item) > 250:
                raise ValueError("privacy list bound")
            return [visit(v, depth+1) for v in item]
        if item is None or type(item) is bool:
            return item
        if type(item) in (int, float) and math.isfinite(item):
            return item
        if isinstance(item, str) and len(item) <= 2048:
            # Timer arming paths are labels, not source text; keep basename/site.
            return re.sub(r"/(?:[^\s:'\"]+/)+([^/\s:'\"]+)", r"\1", item)
        raise ValueError("privacy scalar")
    clean = visit(value)
    if len(json.dumps(clean).encode()) > 200000:
        raise ValueError("privacy output bound")
    return clean


class RecordBudget:
    """Same 250 total slots, including two bounded control-plane trailers.

    Supplemental global GC is outside the measurement scope; original M15's
    scoped summary remains mandatory. Optional spans cannot consume reserved
    slots/bytes for summary, restoration proof and outcome. Overflow is DATA,
    never an exception from a collector, and makes the measurement invalid.
    """
    CRITICAL = frozenset({"recorder", "cleanup", "outcome", "error", "recorder_start",
                          "s2_idle", "s2_emit_burst", "s2_settle", "held_witness",
                          "controlled_async_delay"})
    OPTIONAL = frozenset({"harvest", "pause", "worker_entry", "held_loop_tick", "held_release"})

    def __init__(self):
        self.rows = []
        self.optional = self.critical = self.used_bytes = self.optional_bytes = 0
        self.dropped = self.critical_dropped = self.excluded_gc = 0

    def add(self, kind, **values):
        if kind == "gc":
            # Defensive exclusion for the obsolete producer seam, not silent
            # loss of an in-scope M15 event. No callback is registered anymore.
            self.excluded_gc += 1
            return False
        if kind not in self.CRITICAL | self.OPTIONAL:
            raise ValueError("unknown diagnostic record kind")
        row = dict(kind=kind, **values)
        size = len(json.dumps(row).encode()) + 2
        critical = kind in self.CRITICAL
        full = (self.critical >= 14 if critical else self.optional >= 234)
        # Bound an individual summary independently of optional occupancy.
        # One128KiB-ish summary +13small critical proofs +32k optional bytes
        # fits below172k, leaving the fixed trailers inside the old180k cap.
        if critical:
            full = full or size > (128000 if kind == "recorder" else 512)
        full = full or self.used_bytes + size > 172000
        if not critical:
            full = full or self.optional_bytes + size > 32000
        if full:
            self.dropped += 1
            self.critical_dropped += int(critical)
            return False
        self.rows.append(row)
        self.used_bytes += size
        if critical:
            self.critical += 1
        else:
            self.optional += 1
            self.optional_bytes += size
        return True

    def snapshot(self):
        rows = list(self.rows)
        rows.append(dict(kind="scope", supplemental_gc="NOT_COLLECTED", excluded_gc=self.excluded_gc))
        if self.dropped:
            rows.append(dict(kind="overflow", dropped_count=self.dropped,
                             critical_dropped=self.critical_dropped, record_limit=250))
        # At most234 optional +14 critical +scope +overflow =250. The byte
        # reserve covers trailers; no cap is raised and no proof is fabricated.
        return rows
