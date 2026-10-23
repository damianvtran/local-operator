"""Pure stdlib protocol, not app/native proof. Unexpected or missing evidence stops."""
import json

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
        return "qualified clock controls"
    if not data.get("setup_ok") or data.get("call") not in ("pass", "fail"):
        raise ValueError("setup/call missing STOP")
    records = data.get("records", [])
    clean = [r for r in records if r.get("kind") == "cleanup"]
    if len(clean) != 1 or not clean[0].get("worker_joined") or not clean[0].get("task_settled"):
        raise ValueError("unsettled worker/task STOP")
    summaries = [r for r in records if r.get("kind") == "recorder"]
    if len(summaries) != 1:
        raise ValueError("missing original recorder STOP")
    if mode == "s2-original":
        for phase in ("s2_idle", "s2_emit_burst", "s2_settle"):
            if len([r for r in records if r.get("kind") == phase]) != 1:
                raise ValueError("missing S2 phase STOP")
        if data["call"] == "fail" and not str(data.get("assertion", "")).startswith("event loop "):
            raise ValueError("wrong S2 failure oracle STOP")
    else:
        harvests = [r for r in records if r.get("kind") == "harvest"]
        pauses = [r for r in records if r.get("kind") == "pause"]
        if len(harvests) != 3 or len(pauses) != 3:
            raise ValueError("incomplete observed S3 window STOP")
        window = pauses[-1]["end_wall"] - harvests[0]["start_wall"]
        parts = sum(r["wall_ms"] for r in harvests+pauses)/1000
        if window < parts:
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
