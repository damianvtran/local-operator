"""Future standalone-CLI payload and read-only postlude; NEVER a supervisor.

No signals, retry or cleanup authority. Call only under the separately reviewed
native envelope. A zombie/unknown/residual means failure, not successful cleanup.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_operator import owned_process_group as owned  # noqa: E402

path = Path(sys.argv[2])
if sys.argv[1] == "record":
    scheme = owned.procstate.birth_scheme()
    pids = (os.getpid(), os.getppid())
    samples = owned.bounded_samples(pids, time.monotonic() + 1)
    rows = []
    for pid in pids:
        sample = samples.get(pid)
        if sample is None or not sample.birth or sample.zombie:
            raise SystemExit("CLI startup identity unavailable")
        rows.append({"pid": pid, "birth": sample.birth, "scheme": scheme,
                     "pgid": os.getpgid(pid), "sid": os.getsid(pid)})
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"source": str(ROOT), "identities": rows, "workload_pgid": rows[0]["pgid"]}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    print("ROOT_CLI_VALID", flush=True)
elif sys.argv[1] == "census":
    record = json.loads(path.read_text())
    facts = []
    for row in record["identities"]:
        samples = owned.bounded_samples([row["pid"]], time.monotonic() + 1)
        sample = samples.get(row["pid"])
        facts.append({**row, "exists": owned.resources._pid_exists(row["pid"]),
                      "same_birth": sample.is_birth(row["scheme"], row["birth"]) if sample else None,
                      "zombie": sample.zombie if sample else None})
    members = owned.strict_census(record["workload_pgid"], time.monotonic() + 1)
    print(json.dumps({"facts": facts, "workload_group": members}), flush=True)
    if members or any(f["exists"] is not False for f in facts):
        raise SystemExit("CLI residual/unknown; no automatic cleanup")
else:
    raise SystemExit("unknown read-only CLI receipt operation")
