"""Harbor verifier for paid coding trials: prove isolation, open the uplink, then grade.

WHY: during the agent phase ``main`` sits only on the run's internal network
(``egress.py``), so the stock Terminal-Bench verifier -- which apt-installs curl
and fetches uv/pytest -- would fail there. Rather than change what the verifier
runs, this wrapper, in order:

1. runs ``gold_probe.py isolation`` again: no solution/test bytes appeared
   anywhere in the container during inference;
2. runs ``gold_probe.py quiescence``: nothing the agent started survives. Any
   survivor is recorded, then SIGKILLed, then the check is repeated; a second
   failure aborts verification (the trial is then incomplete, never scored);
3. connects ``main`` to the run's uplink network, restoring the route Harbor
   would normally give the verifier;
4. delegates to Harbor's own ``Verifier.verify()`` unchanged (tests upload,
   ``test.sh``, reward parsing).

Results of 1-2 go to ``<trial>/verifier/bench-isolation.json`` on the host,
written before tests are uploaded, so the record cannot be shaped by them.

CONSTRAINTS: imported by the Harbor process only (``--verifier-import-path
bench_verifier:BenchVerifier``); Harbor itself is not a dependency of this
repository, so it is imported lazily, exactly as ``adapters.py`` does.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess

from egress import GATE_PYTHON, SUPPORT, UPLINK_NETWORK

PROBE = f"{GATE_PYTHON} {SUPPORT}/gold_probe.py"
KILL_SURVIVORS = (
    f"{GATE_PYTHON} -c \"import json,os,signal,sys; "
    "[os.kill(s['pid'], signal.SIGKILL) for s in json.loads(sys.argv[1])['survivors']]\" "
)


def gold_digests() -> list[str]:
    raw = os.environ.get("BENCH_GOLD_DIGESTS", "")
    digests = [d for d in raw.split(",") if d]
    if not digests or any(len(d) != 64 for d in digests):
        raise RuntimeError("BENCH_GOLD_DIGESTS must list the task's solution/test file hashes")
    return digests


def _parse(stdout: str | None) -> dict[str, object]:
    try:
        return json.loads((stdout or "").strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {"ok": False, "unparsed": (stdout or "")[-500:]}


def compose_project(session_id: str) -> str:
    docker = importlib.import_module("harbor.environments.docker.docker")
    return docker._sanitize_docker_compose_project_name(session_id)


def connect_uplink(project: str) -> str:
    """Attach this trial's ``main`` container to its uplink network; return its id."""
    ids = subprocess.run(
        ["docker", "ps", "-q",
         "--filter", f"label=com.docker.compose.project={project}",
         "--filter", "label=com.docker.compose.service=main"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.split()
    if len(ids) != 1:
        raise RuntimeError(f"expected one main container for {project}, found {len(ids)}")
    subprocess.run(
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
        isolation = await run(f"{PROBE} isolation {' '.join(gold_digests())}", user="root")
        record["isolation_after_agent"] = _parse(isolation.stdout)
        quiet = await run(f"{PROBE} quiescence", user="root")
        first = _parse(quiet.stdout)
        record["quiescence_first"] = first
        if first.get("ok") is not True and first.get("survivors"):
            await run(KILL_SURVIVORS + _shell_quote(json.dumps(first)), user="root")
            record["quiescence_after_kill"] = _parse(
                (await run(f"{PROBE} quiescence", user="root")).stdout
            )
        quiescent = (record.get("quiescence_after_kill") or first).get("ok") is True
        record["ok"] = record["isolation_after_agent"].get("ok") is True and quiescent
        out = self.trial_paths.verifier_dir / "bench-isolation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")
        if not record["ok"]:
            raise RuntimeError("gold isolation or agent quiescence failed; not grading")
        record["uplink_container"] = connect_uplink(compose_project(self.environment.session_id))
        out.write_text(json.dumps(record, indent=1) + "\n")
        return await super().verify()


def _shell_quote(text: str) -> str:
    import shlex

    return shlex.quote(text)
