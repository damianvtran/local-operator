"""Harbor verifier for paid coding trials: prove isolation, open the uplink, then grade.

WHY: during the agent phase ``main`` sits only on the run's internal network
(``egress.py``), so the stock Terminal-Bench verifier -- which apt-installs curl
and fetches uv/pytest -- would fail there. Rather than change what the verifier
runs, this wrapper, in order:

1. runs ``gold_probe.py isolation`` again: no solution/test bytes appeared
   anywhere in the container during inference;
2. runs ``gold_probe.py quiescence`` and RECORDS every surviving
   agent-started process (pid, command-line hash, listening ports). It never
   signals them and never refuses on them (R6 F15): Terminal-Bench grades
   services the agent left running -- nginx-request-logging tests nginx on
   :8080, kv-store-grpc a gRPC server on :5328 -- so killing survivors turned
   those tasks into guaranteed failures for every harness, oracle included.
   A survivor is a disclosed flag in the trial record, not a void, and there
   is deliberately no per-task allowlist to maintain;
3. connects ``main`` to the run's uplink network, restoring the route Harbor
   would normally give the verifier;
4. delegates to Harbor's own ``Verifier.verify()`` unchanged (tests upload,
   ``test.sh``, reward parsing), i.e. grading is Harbor's stock behaviour.

Only step 1 can refuse grading: a gold-isolation failure means the agent saw
hidden inputs, so the score would not measure the agent. Results of 1-2 go to
``<trial>/verifier/bench-isolation.json`` on the host, written before tests
are uploaded, so the record cannot be shaped by them.

Because this path differs from the stock verifier only by the isolation
re-check and the late uplink, the earlier oracle acceptance (run under the
stock verifier) must be RE-QUALIFIED under ``BenchVerifier`` before any pilot.
That needs Docker and has not run.

CONSTRAINTS: imported by the Harbor process only (``--verifier-import-path
bench_verifier:BenchVerifier``); Harbor itself is not a dependency of this
repository, so it is imported lazily, exactly as ``adapters.py`` does.
"""

from __future__ import annotations

import importlib
import json
import os
import shlex

from egress import GATE_PYTHON, SUPPORT, UPLINK_NETWORK, run_docker
from gold_probe import provenance_args

PROBE = f"{GATE_PYTHON} {SUPPORT}/gold_probe.py"
SURVIVOR_FLAG = "agent-processes-survived-into-grading"


def gold_digests() -> list[str]:
    raw = os.environ.get("BENCH_GOLD_DIGESTS", "")
    digests = [d for d in raw.split(",") if d]
    if not digests or any(len(d) != 64 for d in digests):
        raise RuntimeError("BENCH_GOLD_DIGESTS must list the task's solution/test file hashes")
    return digests


def _parse(stdout: str | None) -> dict[str, object]:
    try:
        parsed = json.loads((stdout or "").strip().splitlines()[-1])
        return parsed if isinstance(parsed, dict) else {"ok": False}
    except (IndexError, ValueError):
        return {"ok": False, "unparsed": (stdout or "")[-500:]}


def compose_project(session_id: str) -> str:
    docker = importlib.import_module("harbor.environments.docker.docker")
    return docker._sanitize_docker_compose_project_name(session_id)


def connect_uplink(project: str) -> str:
    """Attach this trial's ``main`` container to its uplink network; return its id."""
    ids = run_docker(
        ["docker", "ps", "-q",
         "--filter", f"label=com.docker.compose.project={project}",
         "--filter", "label=com.docker.compose.service=main"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.split()
    if len(ids) != 1:
        raise RuntimeError(f"expected one main container for {project}, found {len(ids)}")
    run_docker(
        ["docker", "network", "connect", f"{project}_{UPLINK_NETWORK}", ids[0]],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return ids[0]


def _verifier_base() -> type:
    return importlib.import_module("harbor.verifier.verifier").Verifier


class BenchVerifier(_verifier_base()):  # type: ignore[misc]
    async def verify(self):  # noqa: D102 -- see module docstring
        record: dict[str, object] = {}
        run = self.environment.exec
        isolation = await run(f"{PROBE} isolation " + shlex.join(
            [*gold_digests(), *provenance_args()]), user="root")
        record["isolation_after_agent"] = _parse(isolation.stdout)
        quiet = _parse((await run(f"{PROBE} quiescence", user="root")).stdout)
        record["quiescence"] = quiet
        survivors = quiet.get("survivors")
        record["survivors"] = survivors if isinstance(survivors, list) else []
        record["flags"] = [SURVIVOR_FLAG] if record["survivors"] else []
        if "survivors" not in quiet:
            record["flags"].append("quiescence-probe-unparsed")
        record["graded"] = (
            isolation.return_code == 0
            and record["isolation_after_agent"].get("ok") is True
            and record["isolation_after_agent"].get("manifest_sha256")
            == os.environ.get("BENCH_PUBLIC_PROVENANCE_SHA256")
        )
        out = self.trial_paths.verifier_dir / "bench-isolation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")
        if not record["graded"]:
            raise RuntimeError("gold isolation failed; not grading")
        record["uplink_container"] = connect_uplink(compose_project(self.environment.session_id))
        out.write_text(json.dumps(record, indent=1) + "\n")
        return await super().verify()
