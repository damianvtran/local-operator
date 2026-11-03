"""Portable opt-in coding dispatcher; publication is NOT a runnable preregistration.

WHY: the source lane stopped before a paid coding pilot. An example must not
silently turn an unfinished subset or an installed release into frozen evidence.
Paid dispatch is unconditionally blocked until executable/runtime identity and
retained tool-wire capture admission are implemented; self-asserted hashes and
verification booleans are not evidence. The remaining preflight is source-only.
CONSTRAINTS: external task/toolchain/output paths, one image at a time, no host
profiles/auth DB, no payloads here. An external meter owns OPENROUTER_API_KEY.
Conservative durable reservations bound trial + total spend, including failures.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request

HERE = Path(__file__).resolve().parent
PUBLIC_LOADOUT = (HERE / "../../public_loadout").resolve()
SOURCE_REVISION = "07c324e9f613b378c437936ede1978c9fc0b49db"
HARNESS = {
    "claude-code": ("FrozenClaudeCode", "2.1.274"),
    "codex-best": ("FrozenCodex", "0.147.0"),
    "opencode": ("OpenCodeV2", "2.0.27"),
    "local-operator": ("LocalOperator", "07c324e9-source"),
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
        raise ValueError("disk gate: need >=21 GiB free and estimated >=17 GiB after pull; hard floor 15 GiB")


def block_paid_coding():
    # Hash-matching JSON proves byte identity, not executable or wire behavior.
    # Keep this hold before even auth-presence checks, including direct callers.
    raise ValueError("HARDBLOCK PAID CODING: executable/runtime and retained capture admission implementation pending; "
                     "paid Local Operator blocked: classifier reservation/accounting validation pending")


def preflight(args):
    """Fail closed before credentials, artifact reads or dispatch side effects."""
    block_paid_coding()
    for value in (args.total_budget_usd, args.trial_budget_usd,
                  args.input_usd_per_m, args.output_usd_per_m):
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
    for url in (args.meter_url, args.translator_url, args.container_guard_url):
        from urllib.parse import urlsplit
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("transport URL must be credential-free HTTP(S)")
    if args.output.resolve() == HERE or HERE in args.output.resolve().parents:
        raise ValueError("outputs must be outside published source")
    if prereg.get("frozen") is not True or runtime.get("frozen") is not True or wire.get("verified") is not True:
        raise ValueError("frozen prereg/runtime and verified toolwire attestations required")
    expected = {"billing_tier": args.billing_tier, "model": args.model,
                "harnesses": args.harnesses.split(","), "tasks": args.tasks.split(","), "rep": args.rep,
                "total_budget_usd": args.total_budget_usd, "trial_budget_usd": args.trial_budget_usd,
                "input_usd_per_m": args.input_usd_per_m, "output_usd_per_m": args.output_usd_per_m,
                "runtime_sha256": args.runtime_sha256, "toolwire_sha256": args.toolwire_sha256,
                "subset_sha256": digest(args.subset), "oracle_sha256": args.oracle_sha256,
                "source_sha256": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
                "catalog_sha256": digest(HERE / "codex-model-catalog.json")}
    if any(prereg.get(k) != v for k, v in expected.items()):
        raise ValueError("requested run differs from frozen preregistration")
    schemas = PUBLIC_LOADOUT / "public_mcp_schemas.json"
    if wire.get("schema_sha256") != digest(schemas) or wire.get("billing_tier") != args.billing_tier:
        raise ValueError("toolwire schema or billing tier mismatch")
    if wire.get("runtime_sha256") != args.runtime_sha256 or wire.get("model") != args.model:
        raise ValueError("toolwire runtime or model mismatch")
    if wire.get("preregistered_tools_observed") is not True or wire.get("output_limit_enforced") is not True:
        raise ValueError("observed tool surface and enforced output limits required")
    if wire.get("billing_tier_enforced") is not True or wire.get("guard_route_verified") is not True:
        raise ValueError("explicit tier enforcement and guarded route evidence missing")
    if wire.get("text_only") is not True:
        raise ValueError("only text billing is conservatively priced")
    subset = json.loads(args.subset.read_text())
    if subset.get("complete") is not True or len(subset["tasks"]) != subset.get("target_count"):
        raise ValueError("subset incomplete: published snapshot is 21/30, not a frozen full run")
    if oracle.get("frozen") is not True or oracle.get("verified") is not True or oracle.get("subset_sha256") != digest(args.subset):
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
        if proof.get("reward") != 1 or proof.get("image_digest") != tasks[name].get("image_digest") or proof.get("task_checksum") != tasks[name].get("task_checksum"):
            raise ValueError("selected task lacks matching external oracle evidence")
        if not (args.task_root / name / "task.toml").is_file():
            raise ValueError("external task payload missing")
        payload_hashes = prereg.get("task_files", {}).get(name, {})
        if not payload_hashes:
            raise ValueError("frozen external task file hashes missing")
        task_dir = (args.task_root / name).resolve()
        actual = {str(p.relative_to(task_dir)): digest(p) for p in task_dir.rglob("*") if p.is_file()}
        if actual != payload_hashes or any(not p.resolve().is_relative_to(task_dir) for p in task_dir.rglob("*")):
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
    if "local-operator" in harnesses:
        raise ValueError("paid Local Operator blocked: classifier reservation/accounting validation pending")
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
            raise ValueError("Local Operator must be reviewed source 07c324e9, not installed 0.94.22")
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
        rows = subprocess.run(["ps", "-axo", "pgid=,stat="], capture_output=True,
                              text=True, check=True, timeout=5).stdout.splitlines()
        members = [row.split()[1] for row in rows
                   if len(row.split()) == 2 and row.split()[0] == str(group)]
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
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
        _signal_owned_group(group, signal.SIGKILL)
        proc.wait(timeout=STOP_GRACE_SECONDS)
    finally:
        _OWNED_GROUPS.pop(proc, None)
        proc.poll()


@contextmanager
def lane_services(args):
    """Own only lane guard/bridge; external meter and translator are never killed."""
    block_paid_coding()
    import socket
    for port in (8788, 8789):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    args.output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CODING_RESERVATION_LEDGER": str(args.output / "reservations.json"),
           "CODING_CURRENT_RUN": str(args.output / "current_run"),
           "CODING_TOTAL_BUDGET_USD": str(args.total_budget_usd),
           "CODING_TRIAL_BUDGET_USD": str(args.trial_budget_usd),
           "CODING_INPUT_USD_PER_M": str(args.input_usd_per_m),
           "CODING_OUTPUT_USD_PER_M": str(args.output_usd_per_m),
           "CODING_PRICED_MODELS": args.model, "CODING_METER_URL": args.meter_url,
           "CODING_BRIDGE_URL": "http://127.0.0.1:8789", "BRIDGE_UPSTREAM": args.translator_url,
           "BRIDGE_LOG": str(args.output / "bridge.jsonl"), "CODING_GUARD_PORT": "8788",
           "CODING_BILLING_TIER": args.billing_tier}
    processes, streams = [], []
    try:
        for name, extra in (("codex_bridge", ["8789"]), ("budget_guard", [])):
            stream = (args.output / (name + ".log")).open("a")
            streams.append(stream)
            processes.append(owned_process([sys.executable, str(HERE / (name + ".py")), *extra],
                cwd=HERE, env=env, stdout=stream, stderr=stream, start_new_session=True))
        for _ in range(40):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("lane service failed setup; inspect local service logs")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8788/health", timeout=1) as response:
                    if json.load(response).get("ok"):
                        break
            except OSError:
                time.sleep(.25)
        else:
            raise RuntimeError("guard startup failed")
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
    block_paid_coding()
    run_id = f"coding__{harness}__{task['task8']}__r{args.rep}"
    job = args.output / "jobs" / run_id
    if job.exists() or (args.request_log.exists() and wire_rows(args.request_log, run_id)):
        raise ValueError("refusing reused run ID")
    cls, version = HARNESS[harness]
    base = args.container_guard_url.rstrip("/")
    url = f"{base}/r/{run_id}"
    mounts = [{"type": "bind", "source": str(args.toolchain.resolve()), "target": "/opt/bench", "read_only": True}]
    argv = [args.harbor, "run", "-p", str(args.task_root / task["name"]), "-a", "adapters:" + cls,
            "-m", "anthropic/" + args.model, "--ak", "version=" + version,
            "-o", str(args.output / "jobs"), "--job-name", run_id, "-n", "1", "-k", "1", "-r", "0", "-q", "-y",
            "--skip-credential-checks", "--mounts", json.dumps(mounts),
            "--ae", "PATH=/opt/bench/node/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--ae", "BENCH_ANTHROPIC_URL=" + url]
    if harness in ("claude-code", "opencode", "hermes"):
        argv += ["--base-url", url + ("/v1" if harness == "opencode" else ""),
                 "--api-key-name", "ANTHROPIC_API_KEY", "--ae", "ANTHROPIC_API_KEY=benchmark-dummy"]
    if harness == "claude-code":
        argv += ["--ae", "ENABLE_TOOL_SEARCH=true", "--effort", "high"]
    if harness == "codex-best":
        (args.output / "current_run").write_text(run_id)
        config = {"model_provider": "bench", "model_catalog_json": "/tmp/bench-support/codex-model-catalog.json",
                  "model_providers": {"bench": {"name": "bench", "base_url": base + "/codex/v1",
                                               "env_key": "OPENAI_API_KEY", "wire_api": "responses"}}}
        argv += ["--base-url", base + "/codex/v1", "--api-key-name", "OPENAI_API_KEY",
                 "--api-format", "openai_responses", "--ae", "OPENAI_API_KEY=benchmark-dummy",
                 "--ak", "config=" + json.dumps(config), "--effort", "high"]
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG", "TMPDIR", "DOCKER_HOST", "DOCKER_CONTEXT"}}
    env.update(PYTHONPATH=str(HERE), CODING_MODEL=args.model)
    started, disk_stop = time.time(), False
    with (args.output / (run_id + ".log")).open("x") as stream:
        proc = owned_process(argv, cwd=HERE, env=env, stdout=stream, stderr=stream, start_new_session=True)
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
    record = {"run_id": run_id, "harness": harness, "version": version, "score": reward,
              "exit_code": code, "disk_stop": disk_stop, "wall_s": time.time() - started,
              "billing_tier": args.billing_tier, "cost_usd": sum(float(r["usage"]["cost"]) for r in rows),
              "requests": len(rows), "verified": bool(rows) and reward is not None and not disk_stop and code == 0}
    with (args.output / "runs.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record))
    if code or disk_stop or not record["verified"]:
        raise RuntimeError("trial incomplete; stopping rather than counting infrastructure as quality")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("toolchain", "task-root", "output", "request-log", "preregistration", "runtime-attestation", "toolwire-attestation", "oracle-attestation"):
        ap.add_argument("--" + name, type=Path, required=True)
    for name in ("preregistration-sha256", "runtime-sha256", "toolwire-sha256", "oracle-sha256", "billing-tier", "model", "tasks", "harnesses", "meter-url", "translator-url", "container-guard-url"):
        ap.add_argument("--" + name, required=True)
    for name in ("total-budget-usd", "trial-budget-usd", "input-usd-per-m", "output-usd-per-m"):
        ap.add_argument("--" + name, type=float, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--subset", type=Path, default=HERE / "subset.json")
    ap.add_argument("--disk-volume", type=Path, default=Path.cwd())
    ap.add_argument("--harbor", default="harbor")
    ap.add_argument("--execute", action="store_true")
    return ap


def main():
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
                replacement = original.replace('docker_image = "' + source_image + '"', 'docker_image = "' + image + '"')
                if tomllib.loads(replacement)["environment"]["docker_image"] != image:
                    raise ValueError("task image cannot be pinned safely")
                subprocess.run(["docker", "pull", "--platform", "linux/amd64", image], check=True)
                try:
                    config.write_text(replacement)
                    for harness in harnesses:
                        check_space(args.disk_volume, 0)
                        run_trial(args, harness, task)
                finally:
                    config.write_text(original)
                    # Only this lane's exact image, never global prune.
                    subprocess.run(["docker", "rmi", image], check=True, capture_output=True)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
