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
hidden inputs, so the score would not measure the agent.
S9b/S9c narrow that for the pre-grading scan only: a hidden digest first seen
after the agent phase is disclosed and graded only when the probe classes it
as a reviewed expected-output fixture, ``_preflight_receipt`` finds this
trial's clean host-side pre-inference receipt, and ``_fence_check`` finds the
fence intact. Solution, other hidden tests, unclassified digests and
structural failures still refuse. Results of 1-2 go to
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
from pathlib import Path

from egress import GATE_PYTHON, SUPPORT, UPLINK_NETWORK, run_docker
from gold_probe import PREFLIGHT_RECEIPT, preflight_binding, provenance_args

PROBE = f"{GATE_PYTHON} {SUPPORT}/gold_probe.py"
SURVIVOR_FLAG = "agent-processes-survived-into-grading"
DISCLOSED_FLAG = "hidden-digest-appeared-after-agent-phase"
ALLOWLIST = Path(__file__).resolve().parent / "package-egress-allowlist.json"


def _complete_row(row: dict) -> bool:
    return (isinstance(row.get("host"), str) and isinstance(row.get("allowed"), bool)
            and all(type(row.get(k)) is int for k in ("port", "bytes_up", "bytes_down"))
            and isinstance(row.get("end"), str) and bool(row["end"]))


def fence_anomalies(log_paths: list[str], allow: set[str]) -> list[dict[str, object]]:
    """Package-egress rows that void the S9b disclosure (not the trial).

    A denied model-host attempt, or an allowed connection to a host outside
    the frozen allowlist (a policy that let something unexpected through),
    means the fence cannot vouch that a late hidden match was the agent's own
    work. Ordinary denials (github.com, unlisted mirrors) are the fence
    working and are not anomalies.

    R11 F31: a MISSING log is an anomaly (fail closed). Every arm launches the
    gate's package proxy and the driver creates each listed log empty before
    the trial, so a missing file means it was removed, not "no connection".

    S9e: the proxy writes every row from ``finally`` with its bytes and an
    ``end`` reason, so a row the fence cannot read in full -- truncated JSON,
    a non-object line, or a missing/ill-typed field -- cannot vouch for its
    connection and is an anomaly (``unparsed`` / ``partial-row``). A complete
    row to an allowlisted host is the fence working however it ended.
    """
    anomalies = []
    for raw in log_paths:
        path = Path(raw)
        if not path.is_file():
            anomalies.append({"host": None, "port": None, "allowed": None,
                              "reason": "missing-log", "unparsed": None})
            continue
        for line in path.read_text().splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = {"unparsed": True}
            if not isinstance(row, dict):
                row = {"unparsed": True}
            elif not _complete_row(row):
                row = {**row, "reason": "partial-row"}
            if (row.get("unparsed") or row.get("reason") in ("model-host", "partial-row")
                    or (row.get("allowed") is True and row.get("host") not in allow)):
                anomalies.append({k: row.get(k) for k in ("host", "port", "allowed",
                                                          "reason", "unparsed")})
    return anomalies


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
    def _isolation_extra_args(self) -> list[str]:
        """Paid trials pass NOTHING extra: no authorized-source exception exists.

        Only ``oracle_qualification.QualificationVerifier`` (zero-model
        qualification, never selected by driver.py) overrides this, so the
        grading tail below stays one shared implementation (design step 6).
        """
        return []

    def _preflight_receipt(self) -> dict[str, object]:
        """The host-side receipt adapters._preinference_checks wrote, validated."""
        trial_dir = getattr(self.trial_paths, "trial_dir", None)
        if trial_dir is None:
            return {"ok": False, "reason": "no trial dir"}
        path = Path(trial_dir) / PREFLIGHT_RECEIPT
        try:
            receipt = json.loads(path.read_text())
        except (OSError, ValueError, TypeError):
            return {"ok": False, "reason": "missing or unreadable receipt"}
        expected = preflight_binding(self.environment.session_id)
        bad = [k for k, v in expected.items() if not v or receipt.get(k) != v]
        if receipt.get("ok") is not True:
            bad.append("ok")
        return {"ok": not bad, "mismatched": bad}

    async def _fence_check(self) -> dict[str, object]:
        """Fence evidence for a late hidden match: logs plus a fresh egress probe.

        Fail closed: missing driver inputs or an unparsed probe is an anomaly.
        """
        logs = [p for p in os.environ.get("BENCH_PACKAGE_EGRESS_LOGS", "").split(",") if p]
        ports = os.environ.get("BENCH_GATE_PORTS", "")
        canary = os.environ.get("BENCH_EGRESS_CANARY_PORT", "")
        # Only a driver that launched no package proxy for this arm may list no
        # log (none does today; documented in README "Gold isolation").
        no_proxy = os.environ.get("BENCH_PACKAGE_PROXY_LAUNCHED") == "none"
        if (not logs and not no_proxy) or not canary.isdigit() or not ports or not all(
                p.isdigit() for p in ports.split(",")):
            return {"ok": False, "reason": "missing fence inputs"}
        allow = set(json.loads(ALLOWLIST.read_text())["allow"])
        anomalies = fence_anomalies(logs, allow)
        probe = await self.environment.exec(
            f"{PROBE} egress --canary {canary} " + " ".join(ports.split(",")), user="root")
        egress = _parse(probe.stdout)
        return {"ok": not anomalies and probe.return_code == 0 and egress.get("ok") is True,
                "package_anomalies": anomalies, "egress_after_agent": egress}

    async def verify(self):  # noqa: D102 -- see module docstring
        record: dict[str, object] = {}
        run = self.environment.exec
        isolation = await run(f"{PROBE} isolation " + shlex.join(
            [*gold_digests(), *provenance_args(), *self._isolation_extra_args()]), user="root")
        record["isolation_after_agent"] = _parse(isolation.stdout)
        quiet = _parse((await run(f"{PROBE} quiescence", user="root")).stdout)
        record["quiescence"] = quiet
        survivors = quiet.get("survivors")
        record["survivors"] = survivors if isinstance(survivors, list) else []
        record["flags"] = [SURVIVOR_FLAG] if record["survivors"] else []
        if "survivors" not in quiet:
            record["flags"].append("quiescence-probe-unparsed")
        iso = record["isolation_after_agent"]
        bound = iso.get("manifest_sha256") == os.environ.get("BENCH_PUBLIC_PROVENANCE_SHA256")
        strict = isolation.return_code == 0 and iso.get("ok") is True
        disclosed = iso.get("disclosed_matches")
        record["post_agent_disclosed"] = disclosed if isinstance(disclosed, list) else []
        late_ok = False
        if bound and not strict and record["post_agent_disclosed"] and (
                isolation.return_code in (0, 1) and iso.get("post_agent_ok") is True):
            # S9b: grade and disclose, but only behind an intact fence AND a
            # clean, same-trial pre-inference receipt (R11 F28): the premise
            # "absent before inference" is checked, never assumed.
            record["preflight_receipt"] = self._preflight_receipt()
            record["fence"] = await self._fence_check()
            late_ok = record["preflight_receipt"]["ok"] and record["fence"]["ok"]
            if late_ok:
                record["flags"].append(DISCLOSED_FLAG)
        record["graded"] = bound and (strict or late_ok)
        out = self.trial_paths.verifier_dir / "bench-isolation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")
        if not record["graded"]:
            raise RuntimeError("gold isolation failed; not grading")
        record["uplink_container"] = connect_uplink(compose_project(self.environment.session_id))
        out.write_text(json.dumps(record, indent=1) + "\n")
        return await super().verify()
