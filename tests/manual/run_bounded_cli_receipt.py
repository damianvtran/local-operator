"""Future standalone-CLI payload and read-only postlude; NEVER a supervisor.

No signals, retry or cleanup authority. Call only under the separately reviewed
native envelope. A zombie/unknown/residual means failure, not successful cleanup.

Nothing happens at import time: this file is a payload with two subcommands, and
an import that set ``sys.path`` or parsed argv would be a side effect a reviewer
cannot see.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _owned():
    """Import the checkout-local authority only when a subcommand actually runs."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from local_operator import owned_process_group
    return owned_process_group


def _record(path: Path) -> None:
    owned = _owned()
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
        json.dump({"source": str(ROOT), "identities": rows,
                   "workload_pgid": rows[0]["pgid"]}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    print("ROOT_CLI_VALID", flush=True)


def _census(path: Path) -> None:
    owned = _owned()
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


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] not in {"record", "census"}:
        raise SystemExit("unknown read-only CLI receipt operation")
    path = Path(argv[1])
    if argv[0] == "record":
        _record(path)
    else:
        _census(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
