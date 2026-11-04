"""Portable opt-in coding dispatcher; publication is NOT a runnable preregistration.

WHY: the source lane stopped before a paid coding pilot. An example must not
silently turn an unfinished subset or an installed release into frozen evidence.
Paid dispatch is unconditionally blocked until executable/runtime identity and
retained tool-wire capture admission are implemented; self-asserted hashes and
verification booleans are not evidence. The remaining preflight is source-only.
CONSTRAINTS: external task/toolchain/output paths, one image at a time, no host
profiles/auth DB, no payloads here. An external meter owns OPENROUTER_API_KEY.

Budget authority (R6 F19): the external ``meter_proxy`` ledger is the SINGLE
spend authority for this lane, exactly as for the desktop lane -- every
container route (meter port, LoP edge, Codex bridge -> translator -> meter)
ends at it, and it enforces trial and campaign caps on durable reservations.
The former lane-local reservation guard sat on no container route once the gate
existed, so it was removed rather than kept as a silent second layer; the
``--*-budget-usd`` and price arguments remain preregistration bindings that the
operator must also give the meter (BENCH_BUDGET_USD, BENCH_TRIAL_BUDGET_USD).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC_LOADOUT = (HERE / "../../public_loadout").resolve()
SOURCE_REVISION = "500883ef56ca504325ffd80c5b90bf2435d00e3f"  # released v0.94.22
BRIDGE_PORT = 8789  # lane-owned Codex Responses bridge, loopback only
HARNESS = {
    "claude-code": ("FrozenClaudeCode", "2.1.274"),
    "codex-best": ("FrozenCodex", "0.147.0"),
    "opencode": ("OpenCodeV2", "2.0.27"),
    "local-operator": ("LocalOperator", "500883ef-v0.94.22"),
    "omp": ("Omp", "18.11.2"),
    "hermes": ("FrozenHermes", "0.22.6"),
    "dsh": ("Dsh", "0.2.0-rc.2"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(path, expected):
    if len(expected) != 64 or digest(path) != expected:
        raise ValueError("frozen artifact SHA-256 mismatch")
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("frozen artifact must be an object")
    return value


def free_mb(volume):
    return shutil.disk_usage(volume).free // (1024 * 1024)


def check_space(volume, compressed):
    available = free_mb(volume)
    estimate = compressed * 3 / (1024 * 1024) + 512
    if available < 21 * 1024 or available - estimate < 17 * 1024:
        raise ValueError(
            "disk gate: need >=21 GiB free and estimated >=17 GiB after pull; hard floor 15 GiB"
        )


# The container metering closure: the shared desktop meter/edge/classifier
# bytes PLUS every coding-lane file on the paid container path. The zero-paid
# container S4 (egress denial, gold isolation, oracle=edge parity) is valid only
# for these exact bytes; any change invalidates its record.
CONTAINER_CLOSURE = (
    "../meter_proxy.py",
    "../lop_edge.py",
    "../radient_meter/classifier.py",
    "../radient_meter/json_types.py",
    "../radient_meter/model.py",
    "../radient_meter/sse_stream.py",
    "adapters.py",
    "bench_verifier.py",
    "driver.py",
    "egress.py",
    "gate_relay.py",
    "gold_probe.py",
    "lop_launch.py",
)


def container_closure_hashes():
    return {name: digest(HERE / name) for name in CONTAINER_CLOSURE}


def container_evidence(args):
    """Return the record digest iff a container S4 record matches these bytes.

    The record is an explicit CLI argument (``--coding-edge-evidence``), like
    every other frozen artifact: the hold is decided before ANY environment
    lookup, so no credential is touched on a refused run.

    Same shape as runner_v2.lop_edge_preflight (R4 F11): the record names an
    evidence artifact in its own directory whose bytes must still hash to the
    recorded digest, so the record is not self-attesting.
    """
    raw = getattr(args, "coding_edge_evidence", None)
    if not raw:
        return None
    path = Path(raw)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    rel = record.get("evidence_path")
    base = path.resolve().parent
    artifact = (base / rel).resolve() if isinstance(rel, str) and rel else None
    if (
        record.get("container_s4_passed") is not True
        or record.get("lop_head") != SOURCE_REVISION
        or record.get("files") != container_closure_hashes()
        or artifact is None
        or not artifact.is_relative_to(base)
        or not artifact.is_file()
        or digest(artifact) != record.get("evidence_sha256")
    ):
        return None
    return record["evidence_sha256"]


def block_paid_coding(args):
    # Hash-matching JSON proves byte identity, not executable or wire behavior.
    # Keep this hold before even auth-presence checks, including direct callers.
    # It lifts only for a container S4 record bound to the exact closure bytes.
    if container_evidence(args) is None:
        raise ValueError(
            "HARDBLOCK PAID CODING: no zero-paid container S4 record "
            "(--coding-edge-evidence) matches the current metering closure"
        )


def gold_digests(task_dir):
    """SHA-256 of every host-side solution/ and tests/ file (hashes only, never bytes)."""
    task_dir = Path(task_dir)
    digests = sorted(
        {digest(p) for sub in ("solution", "tests")
         for p in (task_dir / sub).rglob("*") if p.is_file()}
    )
    if not digests:
        raise ValueError("task has no solution/tests files to guard")
    return digests


@contextmanager
def egress_canary():
    """A run-owned, OPEN loopback listener the gate never forwards (R6 F16).

    The in-container egress probe dials it via the host-gateway name: because
    the port is known open, only a missing route can make that dial fail, so
    "refused" is not mistaken for "fenced". Never accepted or read; the kernel
    backlog completes a handshake, which is all a route test needs.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(8)
        yield sock.getsockname()[1]


def current_run_file(args):
    """The meter's ``BENCH_CURRENT_RUN_FILE``, named explicitly (R6 F19).

    Codex reaches the meter through a translator with a fixed base URL, so the
    meter tags its rows from this file. A mismatch is fail-closed, not silent:
    the meter would tag another id, ``wire_rows`` finds no rows for this run
    and the trial is recorded unverified and stops the lane.
    """
    path = getattr(args, "meter_current_run_file", None)
    if path is None or not Path(path).is_absolute():
        raise ValueError("--meter-current-run-file must be the meter's absolute path")
    return Path(path).resolve()


def container_plan(run_id, harness, image, *, meter_port, bridge_port, edge=None):
    """Harbor arguments that put one paid trial behind the gate (egress.py).

    ``edge`` (LoP only) is ``{"port": int, "env": dict, "ca_file": Path}`` from
    the host lop_edge; its loopback proxy URL is rewritten to the gate host on
    the SAME port, so the gate relays exactly the edge. Returns ``ports`` (the
    gate map), ``env`` (``--ae`` entries) and ``urls`` for the adapters.
    """
    import egress
    from urllib.parse import urlsplit

    ports = [meter_port, bridge_port] + ([edge["port"]] if edge else [])
    urls = {
        "anthropic": egress.gate_url(meter_port, f"/r/{run_id}"),
        "codex": egress.gate_url(bridge_port, "/v1"),
    }
    env = {}
    if harness == "local-operator":
        if not edge:
            raise ValueError("LoP runs only behind its host edge")
        child = dict(edge["env"])
        proxy = urlsplit(child["HTTPS_PROXY"])
        if proxy.hostname != "127.0.0.1" or proxy.port != edge["port"]:
            raise ValueError("edge announced an unexpected proxy address")
        gated = child["HTTPS_PROXY"].replace(
            f"@127.0.0.1:{edge['port']}", f"@{egress.GATE_SERVICE}:{edge['port']}"
        )
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            child[key] = gated
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            child[key] = egress.CA_TARGET
        env.update(child)
    elif edge:
        raise ValueError("only LoP is metered through the edge")
    return {"ports": ports, "env": env, "urls": urls, "image": image}


def preflight(args):
    """Fail closed before credentials, artifact reads or dispatch side effects."""
    block_paid_coding(args)
    for value in (
        args.total_budget_usd,
        args.trial_budget_usd,
        args.input_usd_per_m,
        args.output_usd_per_m,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("explicit positive finite caps and price bounds required")
    if args.trial_budget_usd > args.total_budget_usd:
        raise ValueError("trial cap exceeds total cap")
    if not args.billing_tier.strip():
        raise ValueError("explicit billing tier required")
    prereg = frozen(args.preregistration, args.preregistration_sha256)
    runtime = frozen(args.runtime_attestation, args.runtime_sha256)
    wire = frozen(args.toolwire_attestation, args.toolwire_sha256)
    oracle = frozen(args.oracle_attestation, args.oracle_sha256)
    if not args.tasks.strip() or not args.harnesses.strip() or args.rep < 0:
        raise ValueError("nonempty task/harness lists and nonnegative rep needed")
    current_run_file(args)
    for url in (args.meter_url, args.translator_url):
        from urllib.parse import urlsplit

        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("transport URL must be credential-free HTTP(S)")
    if args.output.resolve() == HERE or HERE in args.output.resolve().parents:
        raise ValueError("outputs must be outside published source")
    if (
        prereg.get("frozen") is not True
        or runtime.get("frozen") is not True
        or wire.get("verified") is not True
    ):
        raise ValueError("frozen prereg/runtime and verified toolwire attestations required")
    expected = {
        "billing_tier": args.billing_tier,
        "model": args.model,
        "harnesses": args.harnesses.split(","),
        "tasks": args.tasks.split(","),
        "rep": args.rep,
        "total_budget_usd": args.total_budget_usd,
        "trial_budget_usd": args.trial_budget_usd,
        "input_usd_per_m": args.input_usd_per_m,
        "output_usd_per_m": args.output_usd_per_m,
        "runtime_sha256": args.runtime_sha256,
        "toolwire_sha256": args.toolwire_sha256,
        "subset_sha256": digest(args.subset),
        "oracle_sha256": args.oracle_sha256,
        "source_sha256": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
        "catalog_sha256": digest(HERE / "codex-model-catalog.json"),
    }
    if any(prereg.get(k) != v for k, v in expected.items()):
        raise ValueError("requested run differs from frozen preregistration")
    schemas = PUBLIC_LOADOUT / "public_mcp_schemas.json"
    if (
        wire.get("schema_sha256") != digest(schemas)
        or wire.get("billing_tier") != args.billing_tier
    ):
        raise ValueError("toolwire schema or billing tier mismatch")
    if wire.get("runtime_sha256") != args.runtime_sha256 or wire.get("model") != args.model:
        raise ValueError("toolwire runtime or model mismatch")
    if (
        wire.get("preregistered_tools_observed") is not True
        or wire.get("output_limit_enforced") is not True
    ):
        raise ValueError("observed tool surface and enforced output limits required")
    if (
        wire.get("billing_tier_enforced") is not True
        or wire.get("guard_route_verified") is not True
    ):
        raise ValueError("explicit tier enforcement and guarded route evidence missing")
    if wire.get("text_only") is not True:
        raise ValueError("only text billing is conservatively priced")
    subset = json.loads(args.subset.read_text())
    if subset.get("complete") is not True or len(subset["tasks"]) != subset.get("target_count"):
        raise ValueError("subset incomplete: published snapshot is 21/30, not a frozen full run")
    if (
        oracle.get("frozen") is not True
        or oracle.get("verified") is not True
        or oracle.get("subset_sha256") != digest(args.subset)
    ):
        raise ValueError("external frozen oracle evidence missing or mismatched")
    tasks = {t["name"]: t for t in subset["tasks"]}
    if len(tasks) != 30:
        raise ValueError("frozen subset must have 30 unique task identifiers")
    names, harnesses = expected["tasks"], expected["harnesses"]
    if len(set(names)) != len(names) or len(set(harnesses)) != len(harnesses):
        raise ValueError("duplicate task/harness would reuse trial ID")
    for name in names:
        if Path(name).name != name or name in {".", ".."} or name not in tasks:
            raise ValueError("invalid selected task")
        proof = oracle.get("tasks", {}).get(name, {})
        if (
            proof.get("reward") != 1
            or proof.get("image_digest") != tasks[name].get("image_digest")
            or proof.get("task_checksum") != tasks[name].get("task_checksum")
        ):
            raise ValueError("selected task lacks matching external oracle evidence")
        if not (args.task_root / name / "task.toml").is_file():
            raise ValueError("external task payload missing")
        payload_hashes = prereg.get("task_files", {}).get(name, {})
        if not payload_hashes:
            raise ValueError("frozen external task file hashes missing")
        task_dir = (args.task_root / name).resolve()
        actual = {
            str(p.relative_to(task_dir)): digest(p) for p in task_dir.rglob("*") if p.is_file()
        }
        if actual != payload_hashes or any(
            not p.resolve().is_relative_to(task_dir) for p in task_dir.rglob("*")
        ):
            raise ValueError("external task payload hash mismatch or escaping symlink")
        image_digest = tasks[name].get("image_digest", "")
        if not image_digest.startswith("sha256:") or len(image_digest) != 71:
            raise ValueError("task image digest missing")
        if not isinstance(tasks[name].get("hub_tag_full_size_bytes"), int):
            raise ValueError("image compressed size missing")
    if len({tasks[n]["task8"] for n in names}) != len(names):
        raise ValueError("task8 collision would reuse run IDs")
    # The model guard cannot account for Radient classifier traffic. An
    # attestation cannot conjure an absent reservation/accounting path, and
    # disabling Radient would change the treatment. Hold paid LOP dispatch
    # until that separate gate has executable, independently validated code.
    # The edge meters classifier traffic on the shared ledger; that path is
    # admitted only by the container S4 record (block_paid_coding above).
    if "local-operator" in harnesses and container_evidence(args) is None:
        raise ValueError("paid Local Operator blocked: no container edge evidence")
    for harness in harnesses:
        if harness not in HARNESS:
            raise ValueError("unknown harness")
        record = runtime.get("runtimes", {}).get(harness, {})
        if record.get("version") != HARNESS[harness][1] or record.get("linux_verified") is not True:
            raise ValueError("runtime version or Linux verification missing")
        files = record.get("files", {})
        if not files:
            raise ValueError("frozen runtime file hashes missing")
        for relative, sha in files.items():
            path = (args.toolchain / relative).resolve()
            if not path.is_relative_to(args.toolchain.resolve()) or digest(path) != sha:
                raise ValueError("runtime file hash mismatch")
        if harness == "local-operator" and record.get("source_revision") != SOURCE_REVISION:
            raise ValueError(
                "Local Operator must be released v0.94.22 source 500883ef"
            )
    if not args.request_log.is_file():
        raise ValueError("existing external metering log needed")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY environment credential required")
    check_space(args.disk_volume, max(tasks[n]["hub_tag_full_size_bytes"] for n in names))
    return [tasks[n] for n in names], harnesses


STOP_GRACE_SECONDS = 15
_OWNED_GROUPS = {}


def owned_process(*args, **kwargs):
    # Only our fresh session grants group-wide cleanup authority. Never infer
    # ownership from an arbitrary Popen PID (especially an already-exited one).
    if kwargs.get("start_new_session") is not True:
        raise ValueError("owned processes require a fresh session")
    proc = subprocess.Popen(*args, **kwargs)
    _OWNED_GROUPS[proc] = (os.getpid(), proc.pid)
    return proc


def _signal_owned_group(group, sig):
    try:
        os.killpg(group, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Darwin reports EPERM, not ESRCH, for an orphan group containing only
        # zombies. Do not swallow real permission failures: prove no live member
        # remains, using a read-only process table (no ambient process signals).
        rows = subprocess.run(
            ["ps", "-axo", "pgid=,stat="], capture_output=True, text=True, check=True, timeout=5
        ).stdout.splitlines()
        members = [
            row.split()[1] for row in rows if len(row.split()) == 2 and row.split()[0] == str(group)
        ]
        if any(not status.startswith("Z") for status in members):
            raise
        return False


def stop(proc):
    ownership = _OWNED_GROUPS.get(proc)
    if ownership is None:
        return  # Includes completed cleanup: never signal a potentially reused ID.
    owner, group = ownership
    if owner != os.getpid() or group <= 1 or group == os.getpgrp() or group != proc.pid:
        raise ValueError("refusing cleanup outside this process's owned group")
    try:
        if not _signal_owned_group(group, signal.SIGTERM):
            return
        # Waiting only for the leader misses orphans. Allow the whole group a
        # bounded grace period, then sweep even when the leader exited first.
        deadline = time.monotonic() + STOP_GRACE_SECONDS
        while time.monotonic() < deadline:
            proc.poll()  # Reap our direct child, not someone else's descendants.
            if not _signal_owned_group(group, 0):
                return
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        _signal_owned_group(group, signal.SIGKILL)
        proc.wait(timeout=STOP_GRACE_SECONDS)
    finally:
        _OWNED_GROUPS.pop(proc, None)
        proc.poll()


@contextmanager
def lane_services(args):
    """Own only the lane's Codex bridge; external meter and translator are never killed.

    No lane budget process: the meter is the single budget authority (module
    docstring, R6 F19).
    """
    block_paid_coding(args)
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", BRIDGE_PORT))
    args.output.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "BRIDGE_UPSTREAM": args.translator_url,
        "BRIDGE_LOG": str(args.output / "bridge.jsonl"),
    }
    processes, streams = [], []
    try:
        for name, extra in (("codex_bridge", [str(BRIDGE_PORT)]),):
            stream = (args.output / (name + ".log")).open("a")
            streams.append(stream)
            processes.append(
                owned_process(
                    [sys.executable, str(HERE / (name + ".py")), *extra],
                    cwd=HERE,
                    env=env,
                    stdout=stream,
                    stderr=stream,
                    start_new_session=True,
                )
            )
        for _ in range(40):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("lane service failed setup; inspect local service logs")
            # The bridge has no health route; a listening socket is readiness.
            try:
                with socket.create_connection(("127.0.0.1", BRIDGE_PORT), timeout=1):
                    break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("bridge startup failed")
        yield
    finally:
        for proc in processes:
            stop(proc)
        for stream in streams:
            stream.close()


def wire_rows(log, run_id):
    rows = []
    for line in log.read_text().splitlines():
        row = json.loads(line)
        if row.get("run_id") == run_id:
            cost = (row.get("usage") or {}).get("cost")
            if cost is None or not math.isfinite(float(cost)) or float(cost) < 0:
                raise ValueError("missing/invalid exact-run billing evidence")
            rows.append(row)
    return rows


def run_trial(args, harness, task):
    block_paid_coding(args)
    run_id = f"coding__{harness}__{task['task8']}__r{args.rep}"
    job = args.output / "jobs" / run_id
    if job.exists() or (args.request_log.exists() and wire_rows(args.request_log, run_id)):
        raise ValueError("refusing reused run ID")
    cls, version = HARNESS[harness]
    from urllib.parse import urlsplit

    # Inside the container every host service is reached ONLY via the gate,
    # on the same port number it listens on at the host (egress.py).
    meter_port = urlsplit(args.meter_url).port
    edge = start_edge(args, run_id) if harness == "local-operator" else None
    try:
        plan = container_plan(
            run_id, harness, args.image, meter_port=meter_port, bridge_port=BRIDGE_PORT,
            edge=edge and {"port": edge["port"], "env": edge["env"], "ca_file": edge["ca_file"]},
        )
        # Held open for the whole Harbor run: the pre-inference probe dials it.
        with egress_canary() as canary:
            if canary in plan["ports"]:
                raise ValueError("egress canary collides with a gate port")
            return _run_gated_trial(
                args, harness, task, run_id, job, cls, version, plan, edge, canary
            )
    finally:
        stop_edge(edge)


def start_edge(args, run_id):
    """Host-side lop_edge for one LoP trial; the bearer goes in on stdin only.

    The real Radient bearer comes from the driver's own environment and is
    never forwarded: run_trial's Harbor env is an allowlist without it.
    """
    token = os.environ.get("BENCH_RADIENT_BEARER", "")
    if not token:
        raise ValueError("BENCH_RADIENT_BEARER required on the host for the LoP edge")
    run_dir = args.output / "edge" / run_id
    proc = owned_process(
        [sys.executable, str(HERE.parent / "lop_edge.py")],
        cwd=HERE.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, start_new_session=True,
    )
    config = {"run_id": run_id, "run_dir": str(run_dir), "meter_base": args.meter_url.rstrip("/")}
    if os.environ.get("BENCH_DECISIONS_URL"):  # zero-paid rig fixture only
        config["decisions_url"] = os.environ["BENCH_DECISIONS_URL"]
    proc.stdin.write(json.dumps({**config, "radient_token": token}) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        stop(proc)
        raise ValueError("lop_edge exited during startup")
    announced = json.loads(line)
    # The CA FILE only (R6 N6): run_dir also holds edge-requests.jsonl.
    return {"proc": proc, "port": announced["port"], "env": announced["env"],
            "ca_file": run_dir / "edge-ca.pem"}


def stop_edge(edge):
    if not edge:
        return
    proc = edge["proc"]
    try:
        proc.stdin.close()  # edge settles or latches in-flight work, then exits
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        pass
    stop(proc)


def _run_gated_trial(args, harness, task, run_id, job, cls, version, plan, edge, canary):
    import egress

    support = args.output / "support" / run_id
    support.mkdir(parents=True)
    for name in egress.SUPPORT_FILES:
        shutil.copy(HERE / name, support / name)
    override = egress.write_override(
        args.output / "support" / (run_id + ".compose.json"),
        egress.compose_override(run_id, plan["ports"], plan["image"], support, args.toolchain),
    )
    url = plan["urls"]["anthropic"]
    mounts = [
        {
            "type": "bind",
            "source": str(args.toolchain.resolve()),
            "target": "/opt/bench",
            "read_only": True,
        },
        *egress.support_mounts(support, edge and edge["ca_file"]),
    ]
    argv = [
        args.harbor,
        "run",
        "-p",
        str(args.task_root / task["name"]),
        "-a",
        "adapters:" + cls,
        "-m",
        "anthropic/" + args.model,
        "--ak",
        "version=" + version,
        "-o",
        str(args.output / "jobs"),
        "--job-name",
        run_id,
        "-n",
        "1",
        "-k",
        "1",
        "-r",
        "0",
        "-q",
        "-y",
        "--skip-credential-checks",
        "--mounts",
        json.dumps(mounts),
        "--extra-docker-compose",
        str(override),
        "--verifier-import-path",
        "bench_verifier:BenchVerifier",
        "--ae",
        "PATH=/opt/bench/node/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin",
        "--ae",
        "BENCH_ANTHROPIC_URL=" + url,
    ]
    if harness in ("claude-code", "opencode", "hermes"):
        argv += [
            "--base-url",
            url + ("/v1" if harness == "opencode" else ""),
            "--api-key-name",
            "ANTHROPIC_API_KEY",
            "--ae",
            "ANTHROPIC_API_KEY=benchmark-dummy",
        ]
    for key, value in plan["env"].items():  # LoP edge env (dummy key, gate proxy)
        argv += ["--ae", f"{key}={value}"]
    if harness == "claude-code":
        argv += ["--ae", "ENABLE_TOOL_SEARCH=true", "--effort", "high"]
    if harness == "codex-best":
        current_run_file(args).write_text(run_id)  # the meter tags Codex rows from it
        config = {
            "model_provider": "bench",
            "model_catalog_json": "/tmp/bench-support/codex-model-catalog.json",
            "model_providers": {
                "bench": {
                    "name": "bench",
                    "base_url": plan["urls"]["codex"],
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                }
            },
        }
        argv += [
            "--base-url",
            plan["urls"]["codex"],
            "--api-key-name",
            "OPENAI_API_KEY",
            "--api-format",
            "openai_responses",
            "--ae",
            "OPENAI_API_KEY=benchmark-dummy",
            "--ak",
            "config=" + json.dumps(config),
            "--effort",
            "high",
        ]
    env = {
        k: v
        for k, v in os.environ.items()
        if k in {"PATH", "HOME", "LANG", "TMPDIR", "DOCKER_HOST", "DOCKER_CONTEXT"}
    }
    # Harbor-process-only inputs for the pre-inference and verifier checks:
    # hashes and port numbers, never task bytes or credentials.
    env.update(
        PYTHONPATH=str(HERE),
        CODING_MODEL=args.model,
        BENCH_GOLD_DIGESTS=",".join(gold_digests(args.task_root / task["name"])),
        BENCH_GATE_PORTS=",".join(str(p) for p in plan["ports"]),
        BENCH_EGRESS_CANARY_PORT=str(canary),
    )
    started, disk_stop = time.time(), False
    with (args.output / (run_id + ".log")).open("x") as stream:
        proc = owned_process(
            argv, cwd=HERE, env=env, stdout=stream, stderr=stream, start_new_session=True
        )
        try:
            while proc.poll() is None:
                if free_mb(args.disk_volume) < 17 * 1024:
                    disk_stop = True
                    stop(proc)  # soft stop protects the 15 GiB emergency floor
                    break
                time.sleep(2)
            code = proc.wait()
        finally:
            stop(proc)
    rows = wire_rows(args.request_log, run_id)
    results = list(job.glob("*/result.json"))
    result = json.loads(results[0].read_text()) if len(results) == 1 else {}
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    record = {
        "run_id": run_id,
        "harness": harness,
        "version": version,
        "score": reward,
        "exit_code": code,
        "disk_stop": disk_stop,
        "wall_s": time.time() - started,
        "billing_tier": args.billing_tier,
        "cost_usd": sum(float(r["usage"]["cost"]) for r in rows),
        "requests": len(rows),
        "verified": bool(rows) and reward is not None and not disk_stop and code == 0,
    }
    with (args.output / "runs.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record))
    if code or disk_stop or not record["verified"]:
        raise RuntimeError(
            "trial incomplete; stopping rather than counting infrastructure as quality"
        )


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in (
        "toolchain",
        "task-root",
        "output",
        "request-log",
        "preregistration",
        "runtime-attestation",
        "toolwire-attestation",
        "oracle-attestation",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    for name in (
        "preregistration-sha256",
        "runtime-sha256",
        "toolwire-sha256",
        "oracle-sha256",
        "billing-tier",
        "model",
        "tasks",
        "harnesses",
        "meter-url",
        "translator-url",
    ):
        ap.add_argument("--" + name, required=True)
    for name in ("total-budget-usd", "trial-budget-usd", "input-usd-per-m", "output-usd-per-m"):
        ap.add_argument("--" + name, type=float, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument(
        "--meter-current-run-file", type=Path, required=True,
        help="absolute path of the meter's BENCH_CURRENT_RUN_FILE (Codex run tagging)",
    )
    ap.add_argument("--subset", type=Path, default=HERE / "subset.json")
    ap.add_argument("--disk-volume", type=Path, default=Path.cwd())
    ap.add_argument("--harbor", default="harbor")
    ap.add_argument("--coding-edge-evidence", type=Path)
    ap.add_argument("--execute", action="store_true")
    return ap


def main():
    import egress

    ap = parser()
    args = ap.parse_args()
    try:
        tasks, harnesses = preflight(args)
        if not args.execute:
            print("Preflight passed; dispatch requires --execute")
            return
        with lane_services(args):
            for task in tasks:
                check_space(args.disk_volume, task["hub_tag_full_size_bytes"])
                image = task["image"].split(":")[0] + "@" + task["image_digest"]
                config = args.task_root / task["name"] / "task.toml"
                original = config.read_text()
                source_image = tomllib.loads(original)["environment"]["docker_image"]
                # Task payloads are external scratch copies. Restore even when
                # a trial fails; never edit the upstream dataset checkout.
                replacement = original.replace(
                    'docker_image = "' + source_image + '"', 'docker_image = "' + image + '"'
                )
                if tomllib.loads(replacement)["environment"]["docker_image"] != image:
                    raise ValueError("task image cannot be pinned safely")
                args.image = image  # the gate reuses it: no second image pull
                # The ONLY pull on this path: explicit, digest-pinned, behind a
                # fresh disk gate. Compose services carry pull_policy: never.
                check_space(args.disk_volume, task["hub_tag_full_size_bytes"])
                egress.run_docker(["docker", "pull", "--platform", "linux/amd64", image],
                                  explicit_pull=True, check=True)
                try:
                    config.write_text(replacement)
                    for harness in harnesses:
                        check_space(args.disk_volume, 0)
                        run_trial(args, harness, task)
                finally:
                    config.write_text(original)
                    # Only this lane's exact image, never global prune.
                    egress.run_docker(["docker", "rmi", image], check=True, capture_output=True)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
