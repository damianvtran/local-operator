#!/usr/bin/env python3
"""Slice-0 POC driver for remote cloud agents.

Design: docs/design/remote-cloud-agents.md §9.2 step 3 (the driver's job) and
§9.3 (what a recorded run has to prove). Spec: infra/remote-agents-poc/README.md.

WHAT THIS MUST NEVER DO
-----------------------
It never fetches, prints, logs, or stores the model key. §9.2 step 3 has the driver
read the key out of Secrets Manager and inject it as a task-level secret; the
authoritative resource list puts it in the TASK DEFINITION instead, which is
strictly better for the secret-handling rule — the key is resolved by the ECS
agent inside AWS and no driver process ever holds it. See README.md, "Divergences".

WHAT IT TALKS TO
----------------
Every AWS call goes through `lop-poc-controller`, so the driver's own blast radius
is the same one the POC claims for it (RunTask/StopTask/DescribeTasks on the
`lop-poc` cluster, two presigned PUTs, the task log group). The `status`
subcommand's tag inventory is the ONE exception, and it says why where it is used.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

REGION = "ca-central-1"
EXPECTED_ACCOUNT = "325492156725"
CLUSTER_NAME = "lop-poc"
CONTROLLER_ROLE_NAME = "lop-poc-controller"
#: 3 h: the driver holds one assumed session for the whole `--runs N` sweep, and a
#: run's own client deadline is 2 h + 10 min (below), so the session must outlive
#: at least one full run plus the polls around it.
CONTROLLER_SESSION_SECONDS = 10800
PRESIGN_EXPIRES_SECONDS = 10800
#: §9.2 step 3: "enforces a 2 h + 10 min client deadline (StopTask if exceeded)".
RUN_DEADLINE_SECONDS = 2 * 60 * 60 + 10 * 60
#: ECS rejects an overrides document above 8 KiB. Asserted rather than discovered
#: at RunTask time because the failure mode there is a 400 with no field named.
OVERRIDES_LIMIT_BYTES = 8192
DESCRIBE_POLL_SECONDS = 1.0

FIXTURE_URL_DEFAULT = "https://github.com/olafagbemi/lop-poc-fixture.git"
#: The fixture's recorded SHA: the single commit the POC clones, and the parent
#: every accepted run's `lop/<run-id>` branch must carry. Bump this only by
#: re-recording it in the POC report — a run verified against a different SHA than
#: the one the report names proves nothing.
FIXTURE_SHA_DEFAULT = "69db7e55fc14f918cccdf2fea62894fc37f1f642"
PROMPT_DEFAULT = "make the failing test `test_add` pass"

POC_TAGS = {"lop-poc": "true", "owner": "lopdev"}

#: The directory name the harness uses under HOME for its config root. verify's
#: acceptance-3 check transplants a session into a fresh root and must spell the
#: root EXACTLY as the runtime does, so the name lives in one place here.
LOCAL_OPERATOR_CONFIG_DIR_NAME = ".local-operator"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _log(message: str) -> None:
    """Progress goes to stderr; stdout is reserved for machine-readable output."""
    print(message, file=sys.stderr, flush=True)


def new_run_id() -> str:
    """`ct_<8 hex>`: short enough for a log stream name, random enough to collide
    never — the run id is also the S3 prefix, so a collision would mix two runs'
    artifacts."""
    return f"ct_{secrets.token_hex(4)}"


def assert_expected_account(session: Any) -> str:
    """Refuse to operate in any account but the POC's own.

    Runs FIRST, before a single mutating call, because every hard rule about this
    sandbox starts with "the account is 325492156725" and a stray AWS_PROFILE would
    otherwise retarget the whole sweep.
    """
    identity = session.client("sts").get_caller_identity()
    account = str(identity["Account"])
    if account != EXPECTED_ACCOUNT:
        raise SystemExit(
            f"refusing to run: caller account is {account}, expected {EXPECTED_ACCOUNT} "
            f"(check AWS_PROFILE; the POC lives in minerva_sandbox)"
        )
    return str(identity["Arn"])


def assume_controller(operator: Any, controller_role_arn: str) -> Any:
    """Assume the scoped driver role and return a session built on its credentials.

    The operator's own identity is used for exactly two things: this AssumeRole and
    the tag inventory in `status`. Everything that touches the POC's ECS resources
    runs as the role, so an over-broad operator identity cannot hide a missing
    controller permission until production.
    """
    sts = operator.client("sts")
    response = sts.assume_role(
        RoleArn=controller_role_arn,
        RoleSessionName="lop-poc-driver",
        DurationSeconds=CONTROLLER_SESSION_SECONDS,
    )
    credentials = response["Credentials"]
    _log(f"assumed {CONTROLLER_ROLE_NAME} for {CONTROLLER_SESSION_SECONDS}s")
    return boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=REGION,
    )


def read_stack_outputs(infra_dir: Path, backend_url: str) -> dict[str, Any]:
    """`pulumi stack output --json`, with the file backend and the passphrase.

    The passphrase is read from the secret store by `lop secret run` (which is how
    the whole deploy is driven) and reaches pulumi as PULUMI_CONFIG_PASSPHRASE. It
    is never an argument here and never printed.
    """
    env = dict(os.environ)
    env["PULUMI_BACKEND_URL"] = backend_url
    env["PULUMI_DISABLE_AUTOMATIC_PLUGIN_ACQUISITION"] = "true"
    completed = subprocess.run(
        ["pulumi", "stack", "output", "--json"],
        cwd=infra_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"pulumi stack output failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def load_outputs(args: argparse.Namespace) -> dict[str, Any]:
    """Stack outputs, from `--outputs` if given, otherwise from the stack itself."""
    if args.outputs:
        return json.loads(Path(args.outputs).read_text(encoding="utf-8"))
    return read_stack_outputs(Path(args.infra_dir), args.backend_url)


def presign_put(s3: Any, bucket: str, key: str) -> str:
    """A presigned PUT for exactly one object key.

    Signed with the CONTROLLER role's credentials, so the URL can write nothing but
    its own key and dies with the session — the pod never holds an S3 credential.
    """
    return str(
        s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=PRESIGN_EXPIRES_SECONDS,
            HttpMethod="PUT",
        )
    )


def build_environment(
    outputs: dict[str, Any],
    run_id: str,
    repo_url: str,
    sha: str,
    prompt: str,
    probes_url: str,
    results_url: str,
    hosting: str,
    model: str,
    mock: bool,
) -> list[dict[str, str]]:
    """The container's env overrides.

    `POC_MODEL_SECRET_ARN` is here for probe 4a, which must show that reading the
    model secret is DENIED: an ARN is not a secret, so naming it costs nothing.
    """
    environment = [
        {"name": "POC_RUN_ID", "value": run_id},
        {"name": "POC_REPO_URL", "value": repo_url},
        {"name": "POC_SHA", "value": sha},
        {"name": "POC_PROMPT", "value": prompt},
        {"name": "POC_PROBES_URL", "value": probes_url},
        {"name": "POC_RESULTS_URL", "value": results_url},
        {"name": "POC_MODEL_SECRET_ARN", "value": str(outputs.get("secretArn", ""))},
        {"name": "POC_MOCK", "value": "1" if mock else "0"},
    ]
    if hosting:
        environment.append({"name": "POC_HOSTING", "value": hosting})
    if model:
        environment.append({"name": "POC_MODEL", "value": model})
    return environment


def _timestamp_ms(value: Any) -> int | None:
    return None if value is None else int(value.timestamp() * 1000)


def cold_start_numbers(
    t_runtask_ms: int,
    first_running_wall_ms: int | None,
    described: dict[str, Any],
    timings: dict[str, int],
) -> dict[str, Any]:
    """The two cold-start measurements §9.3 item 5 asks for, plus the ECS phases.

    Two independent readings of the same span on purpose: ECS's own timestamps
    (`createdAt` → `startedAt`) and the driver's wall clock (RunTask call →
    first RUNNING observation). They are taken from different clocks and different
    observers, and a run where they disagree by more than a second is a measurement
    worth distrusting rather than averaging.
    """
    created = _timestamp_ms(described.get("createdAt"))
    pull_started = _timestamp_ms(described.get("pullStartedAt"))
    started = _timestamp_ms(described.get("startedAt"))
    first_model_event = timings.get("t_first_model_event")
    numbers: dict[str, Any] = {
        "runtask_to_running_wall_ms": (
            None if first_running_wall_ms is None else first_running_wall_ms - t_runtask_ms
        ),
        "ecs_created_to_pull_started_ms": (
            None if created is None or pull_started is None else pull_started - created
        ),
        "ecs_pull_started_to_started_ms": (
            None if pull_started is None or started is None else started - pull_started
        ),
        "ecs_created_to_started_ms": (
            None if created is None or started is None else started - created
        ),
        "container_start_to_first_model_event_ms": (
            None
            if first_model_event is None or "t_container_start" not in timings
            else first_model_event - timings["t_container_start"]
        ),
        "runtask_to_first_model_event_ms": (
            None if first_model_event is None else first_model_event - t_runtask_ms
        ),
        "ecs_started_to_first_model_event_ms": (
            None if first_model_event is None or started is None else first_model_event - started
        ),
    }
    return numbers


def _load_timings(results_dir: Path) -> dict[str, int]:
    path = results_dir / "timings.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(event["name"]): int(event["epoch_ms"]) for event in payload.get("events", [])}


def download_artifacts(s3: Any, bucket: str, run_id: str, run_dir: Path) -> dict[str, Any]:
    """Pull both presigned objects and unpack the results tarball.

    A missing object surfaces as HTTP 403, not 404: the controller policy grants
    `s3:GetObject` on `runs/*` and deliberately NOT `s3:ListBucket`, and S3 answers
    403 for an object the caller may not even list. So a 403 here means "the
    container never uploaded it" — which is a fact worth naming, because it reads
    like a permissions bug and sent the first diagnosis after the wrong thing.
    """
    import tarfile

    downloaded: dict[str, Any] = {}
    for name in ("probes.json", "results.tar.gz"):
        target = run_dir / name
        try:
            s3.download_file(bucket, f"runs/{run_id}/{name}", str(target))
            downloaded[name] = target.stat().st_size
        except (ClientError, OSError) as error:
            detail = f"FAILED: {type(error).__name__}: {error}"
            if "403" in str(error):
                detail += " (403 without s3:ListBucket means the object is absent)"
            downloaded[name] = detail
    tarball = run_dir / "results.tar.gz"
    results_dir = run_dir / "results"
    if tarball.exists():
        results_dir.mkdir(exist_ok=True)
        with tarfile.open(tarball) as archive:
            # The archive is our own container's output, and the filter refuses
            # absolute paths and traversal anyway — cheap, and it means a
            # mislabelled artifact cannot write outside the run directory.
            archive.extractall(results_dir, filter="data")
        downloaded["extracted"] = sorted(
            str(path.relative_to(results_dir)) for path in results_dir.rglob("*")
        )
    return downloaded


def run_once(
    args: argparse.Namespace,
    ecs: Any,
    s3: Any,
    outputs: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """One task, from RunTask to stopped-and-downloaded. Returns its run.json."""
    bucket = str(outputs["bucket"])
    probes_url = presign_put(s3, bucket, f"runs/{run_id}/probes.json")
    results_url = presign_put(s3, bucket, f"runs/{run_id}/results.tar.gz")
    environment = build_environment(
        outputs,
        run_id,
        args.fixture_url,
        args.fixture_sha,
        args.prompt,
        probes_url,
        results_url,
        args.hosting,
        args.model,
        args.mock,
    )
    overrides = {"containerOverrides": [{"name": "agent", "environment": environment}]}
    overrides_bytes = len(json.dumps(overrides).encode())
    if overrides_bytes >= OVERRIDES_LIMIT_BYTES:
        raise SystemExit(
            f"overrides document is {overrides_bytes} bytes, at or over the "
            f"{OVERRIDES_LIMIT_BYTES}-byte ECS limit; shorten the prompt"
        )
    _log(f"{run_id}: overrides {overrides_bytes} bytes, RunTask")
    t_runtask_ms = _now_ms()
    response = ecs.run_task(
        cluster=str(outputs["clusterName"]),
        taskDefinition=str(outputs["taskDefinitionArn"]),
        launchType="FARGATE",
        platformVersion="LATEST",
        count=1,
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": list(outputs["subnetIds"]),
                "securityGroups": [str(outputs["securityGroupId"])],
                "assignPublicIp": "ENABLED",
            }
        },
        overrides=overrides,
        tags=[
            {"key": "lop-poc", "value": "true"},
            {"key": "owner", "value": "lopdev"},
            {"key": "run-id", "value": run_id},
        ],
        # enableECSManagedTags + propagateTags TASK_DEFINITION: a run that fails to
        # start still carries lop-poc, so it is still found by the inventory and by
        # `stop-all` rather than being an untagged orphan.
        enableECSManagedTags=True,
        propagateTags="TASK_DEFINITION",
    )
    if response.get("failures"):
        raise SystemExit(f"{run_id}: RunTask failed: {response['failures']}")
    task_arn = str(response["tasks"][0]["taskArn"])
    _log(f"{run_id}: {task_arn}")
    described, first_running_wall_ms, over_deadline = poll_to_stopped(ecs, outputs, task_arn)
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "describe-tasks.json").write_text(
        json.dumps(described, indent=2, default=str) + "\n", encoding="utf-8"
    )
    downloads = download_artifacts(s3, bucket, run_id, run_dir)
    timings = _load_timings(run_dir / "results")
    containers = described.get("containers") or [{}]
    # DescribeTasks does NOT return `runtimePlatform` for Fargate — measured, the
    # first recorded run read None from here — so the architecture is read from the
    # TASK DEFINITION, which is where it is actually declared. The describe-tasks
    # reading is kept beside it so the divergence is visible rather than explained
    # away in prose.
    definition = ecs.describe_task_definition(taskDefinition=str(outputs["taskDefinitionArn"]))[
        "taskDefinition"
    ]
    record = {
        "run_id": run_id,
        "task_arn": task_arn,
        "cpu_architecture": (definition.get("runtimePlatform") or {}).get("cpuArchitecture"),
        "cpu_architecture_from_describe_tasks": (described.get("runtimePlatform") or {}).get(
            "cpuArchitecture"
        ),
        "task_definition": {
            "family": definition.get("family"),
            "revision": definition.get("revision"),
            "executionRoleArn": definition.get("executionRoleArn"),
            "taskRoleArn": definition.get("taskRoleArn"),
        },
        "last_status": described.get("lastStatus"),
        "stop_code": described.get("stopCode"),
        "stopped_reason": described.get("stoppedReason"),
        "exit_code": containers[0].get("exitCode"),
        "container_reason": containers[0].get("reason"),
        "over_deadline": over_deadline,
        "overrides_bytes": overrides_bytes,
        "t_runtask_ms": t_runtask_ms,
        "first_running_wall_ms": first_running_wall_ms,
        "ecs_timestamps": {
            key: described.get(key)
            for key in ("createdAt", "pullStartedAt", "pullStoppedAt", "startedAt", "stoppedAt")
        },
        "cold_start": cold_start_numbers(t_runtask_ms, first_running_wall_ms, described, timings),
        "container_timings_ms": timings,
        "downloads": downloads,
        "probes": read_probes(run_dir),
        "status": read_status(run_dir),
        "git": read_git(run_dir),
    }
    (run_dir / "run.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


def poll_to_stopped(
    ecs: Any, outputs: dict[str, Any], task_arn: str
) -> tuple[dict[str, Any], int | None, bool]:
    """Poll DescribeTasks on a 1 s grid until STOPPED. Returns (task, RUNNING wall, overdue)."""
    cluster = str(outputs["clusterName"])
    deadline = time.monotonic() + RUN_DEADLINE_SECONDS
    first_running_wall_ms: int | None = None
    over_deadline = False
    while True:
        described = ecs.describe_tasks(cluster=cluster, tasks=[task_arn], include=["TAGS"])["tasks"]
        if not described:
            # A task is not describable for the first fractions of a second after
            # RunTask. Polling rather than logging NotFound keeps the first-RUNNING
            # measurement honest instead of reporting the first successful read.
            time.sleep(DESCRIBE_POLL_SECONDS)
            continue
        task = described[0]
        status = task.get("lastStatus")
        if status == "RUNNING" and first_running_wall_ms is None:
            first_running_wall_ms = _now_ms()
            _log(f"{task_arn}: RUNNING")
        if status == "STOPPED":
            return task, first_running_wall_ms, over_deadline
        if time.monotonic() > deadline and not over_deadline:
            # The client deadline is a stop, not a surrender: the task is killed and
            # the loop continues so the STOPPED record is still captured.
            over_deadline = True
            _log(f"{task_arn}: past the {RUN_DEADLINE_SECONDS}s deadline, StopTask")
            ecs.stop_task(cluster=cluster, task=task_arn, reason="lop-poc driver deadline")
        time.sleep(DESCRIBE_POLL_SECONDS)


def read_probes(run_dir: Path) -> dict[str, Any]:
    """Summary of the container's probes.json — the 4a/4b/4c/4d verdicts."""
    path = run_dir / "results" / "probes.json"
    if not path.exists():
        path = run_dir / "probes.json"
    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "failed": payload.get("failed", []),
        "probes": {
            str(probe["name"]): {
                "pass": probe.get("pass"),
                "note": (probe.get("detail") or {}).get("note"),
            }
            for probe in payload.get("probes", [])
        },
    }


def read_status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "results" / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"available": False}


def read_git(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "results" / "git.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"available": False}


def _driver_session(args: argparse.Namespace) -> tuple[Any, dict[str, Any], Any]:
    """(operator session, stack outputs, controller-role session), in that order.

    The account assertion runs before the AssumeRole and before any outputs are
    read, because "which account am I in" is the question every other guard here
    depends on.
    """
    operator = boto3.Session(profile_name=args.profile, region_name=REGION)
    _log(f"operator identity: {assert_expected_account(operator)}")
    outputs = load_outputs(args)
    driver = assume_controller(operator, str(outputs["controllerRoleArn"]))
    return operator, outputs, driver


def cmd_run(args: argparse.Namespace) -> int:
    operator, outputs, driver = _driver_session(args)
    ecs = driver.client("ecs")
    s3 = driver.client("s3")
    records: list[dict[str, Any]] = []
    for _ in range(args.runs):
        # Re-asserted per run, not once per sweep: `--runs 5` is five RunTasks, and
        # the guard that matters is the one immediately before each of them.
        assert_expected_account(operator)
        record = run_once(args, ecs, s3, outputs, new_run_id())
        records.append(record)
        _log(
            "{run_id}: cpu={cpu} stop={stop} exit={exit_code} "
            "runtask->running={wall}ms runtask->first_model_event={first}ms".format(
                run_id=record["run_id"],
                cpu=record["cpu_architecture"],
                stop=record["stop_code"],
                exit_code=record["exit_code"],
                wall=record["cold_start"]["runtask_to_running_wall_ms"],
                first=record["cold_start"]["runtask_to_first_model_event_ms"],
            )
        )
    summary = Path(args.out_dir) / "runs-summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2, default=str))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    operator, outputs, driver = _driver_session(args)
    cluster = str(outputs["clusterName"])
    ecs = driver.client("ecs")
    active: list[str] = []
    for desired in ("RUNNING", "PENDING"):
        active.extend(ecs.list_tasks(cluster=cluster, desiredStatus=desired).get("taskArns", []))
    # The tag inventory is the ONE call the controller role deliberately cannot
    # make: resourcegroupstaggingapi has no resource-level scoping, so giving the
    # POC role access to it would be account-wide read for no POC benefit. The
    # operator's own identity runs it, and the tag filter is what makes it an
    # inventory of THIS stack rather than of the account.
    inventory: list[dict[str, Any]] = []
    paginator = operator.client("resourcegroupstaggingapi").get_paginator("get_resources")
    for page in paginator.paginate(TagFilters=[{"Key": "lop-poc", "Values": ["true"]}]):
        for resource in page.get("ResourceTagMappingList", []):
            inventory.append(
                {
                    "arn": resource.get("ResourceARN"),
                    "tags": {
                        str(tag["Key"]): str(tag["Value"]) for tag in resource.get("Tags", [])
                    },
                }
            )
    report = {
        "cluster": cluster,
        "active_tasks": active,
        "active_count": len(active),
        "tagged_resource_count": len(inventory),
        "tagged_resources": sorted(inventory, key=lambda item: str(item["arn"])),
    }
    print(json.dumps(report, indent=2, default=str))
    if active:
        _log(f"FAIL: {len(active)} active task(s) in {cluster}")
        return 1
    _log(f"OK: no RUNNING or PENDING tasks in {cluster}; {len(inventory)} tagged lop-poc resources")
    return 0


def cmd_stop_all(args: argparse.Namespace) -> int:
    _operator, outputs, driver = _driver_session(args)
    cluster = str(outputs["clusterName"])
    ecs = driver.client("ecs")
    stopped: list[str] = []
    for desired in ("RUNNING", "PENDING"):
        for arn in ecs.list_tasks(cluster=cluster, desiredStatus=desired).get("taskArns", []):
            ecs.stop_task(cluster=cluster, task=arn, reason="lop-poc stop-all")
            stopped.append(str(arn))
    print(json.dumps({"cluster": cluster, "stopped": sorted(stopped)}, indent=2))
    _log(f"stopped {len(stopped)} task(s) in {cluster}")
    return 0


def _run(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)


def _find(run_dir: Path, name: str) -> Path | None:
    """Artifacts live under results/ when the tarball was extracted, and directly
    under the run dir for the objects downloaded before extraction."""
    for candidate in (run_dir / "results" / name, run_dir / name):
        if candidate.exists():
            return candidate
    return None


def _fixture_test_command(clone: Path) -> list[str]:
    """The fixture ships a pytest-free runner so the container needs no pytest; use
    it when present so verify tests the same thing the agent was asked to fix."""
    if (clone / "run_tests.py").exists():
        return [sys.executable, "run_tests.py"]
    return [sys.executable, "-m", "pytest", "-q"]


class Verifier:
    """Collects (name, ok, detail) rows and prints them as a PASS/FAIL table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.rows.append((name, ok, detail))
        _log(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        return ok

    def report(self) -> int:
        failed = [name for name, ok, _ in self.rows if not ok]
        print(json.dumps({"checks": self.rows, "failed": failed}, indent=2, default=str))
        return 1 if failed else 0


def _verify_bundle(verifier: Verifier, clone: Path, bundle: Path, fixture_sha: str) -> str | None:
    """Acceptance 2: the bundle verifies, and lop/<id> is one commit on the fixture."""
    verify = _run(["git", "bundle", "verify", str(bundle)], cwd=clone)
    if not verifier.check(
        "acceptance2.bundle_verify",
        verify.returncode == 0,
        (verify.stdout + verify.stderr).strip()[:400],
    ):
        return None
    heads = _run(["git", "bundle", "list-heads", str(bundle)], cwd=clone).stdout.split()
    refs = [token for token in heads if token.startswith("refs/heads/lop/")]
    if not verifier.check("acceptance2.branch_present", bool(refs), f"heads: {heads}"):
        return None
    branch = refs[0]
    fetch = _run(["git", "fetch", str(bundle), f"{branch}:{branch}"], cwd=clone)
    if not verifier.check(
        "acceptance2.fetch",
        fetch.returncode == 0,
        (fetch.stdout + fetch.stderr).strip()[:400],
    ):
        return None
    count = _run(
        ["git", "rev-list", "--count", f"{fixture_sha}..{branch}"], cwd=clone
    ).stdout.strip()
    verifier.check("acceptance2.one_commit", count == "1", f"{count} commit(s) on {branch}")
    parent = _run(["git", "rev-parse", f"{branch}^"], cwd=clone).stdout.strip()
    verifier.check(
        "acceptance2.parent_is_fixture_sha", parent == fixture_sha, f"parent {parent or '<none>'}"
    )
    return branch


def _verify_fixture_tests(verifier: Verifier, clone: Path, fixture_sha: str, branch: str) -> None:
    """Acceptance 2: the test FAILS at the fixture SHA and PASSES on the branch."""
    command = _fixture_test_command(clone)
    checkout = _run(["git", "checkout", "--quiet", fixture_sha], cwd=clone)
    if not verifier.check(
        "acceptance2.checkout_fixture_sha", checkout.returncode == 0, fixture_sha
    ):
        return
    before = _run(command, cwd=clone)
    verifier.check(
        "acceptance2.test_fails_at_fixture_sha",
        before.returncode != 0,
        f"exit {before.returncode}: {(before.stdout + before.stderr).strip()[-300:]}",
    )
    checkout = _run(["git", "checkout", "--quiet", branch], cwd=clone)
    if not verifier.check("acceptance2.checkout_branch", checkout.returncode == 0, branch):
        return
    after = _run(command, cwd=clone)
    verifier.check(
        "acceptance2.test_passes_on_branch",
        after.returncode == 0,
        f"exit {after.returncode}: {(after.stdout + after.stderr).strip()[-300:]}",
    )


def _verify_session(verifier: Verifier, run_dir: Path, lop: str, session_id: str) -> None:
    """Acceptance 3: the session directory opens non-interactively in a fresh root."""
    import tarfile
    import tempfile

    archive = _find(run_dir, "session.tar.gz")
    if archive is None:
        verifier.check("acceptance3.session_archive", False, "session.tar.gz not found")
        return
    verifier.check("acceptance3.session_archive", True, str(archive))
    config_name = LOCAL_OPERATOR_CONFIG_DIR_NAME
    with tempfile.TemporaryDirectory(prefix="lop-poc-session-") as iso:
        root = Path(iso)
        sessions = root / config_name / "sessions"
        sessions.mkdir(parents=True)
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(sessions, filter="data")
        except (OSError, tarfile.TarError) as error:
            verifier.check("acceptance3.extract", False, f"{type(error).__name__}: {error}")
            return
        verifier.check("acceptance3.extract", True, f"into {sessions}")
        # env -i style, and constructed rather than inherited: an inherited
        # CMUX_WORKSPACE_ID or LOP_* variable is what AGENTS.md's "Isolating a run"
        # says must never reach a second runtime.
        env = {
            "HOME": iso,
            "LOCAL_OPERATOR_CONFIG_DIR": str(root / config_name),
            "PATH": os.environ.get("PATH", ""),
            "TERM": "dumb",
        }
        listed = _run([lop, "sessions", "--json"], env=env)
        listed_ok = listed.returncode == 0 and session_id in listed.stdout
        verifier.check(
            "acceptance3.lop_sessions_lists_id",
            listed_ok,
            f"exit {listed.returncode}: {(listed.stdout + listed.stderr).strip()[:300]}",
        )
        transcript = root / config_name / "sessions" / session_id / "transcript.jsonl"
        if transcript.exists():
            lines = [line for line in transcript.read_text(encoding="utf-8").splitlines() if line]
            verifier.check(
                "acceptance3.transcript_non_empty", bool(lines), f"{len(lines)} transcript line(s)"
            )
        else:
            verifier.check("acceptance3.transcript_non_empty", False, f"{transcript} is absent")


def _verify_key_scan(verifier: Verifier, run_dir: Path, probes: Path | None) -> None:
    """Acceptance 4: no copy of the model key exists in what was downloaded.

    The key is piped from the secret store into probes.py's stdin, never argv, so
    it is not in this process's command line and never in this transcript.
    """
    if probes is None:
        verifier.check("acceptance4.key_scan", False, "probes.py not found; cannot scan")
        return
    listing = _run(["lop", "secret", "list"])
    if "LOP_POC_MODEL_KEY" not in listing.stdout:
        verifier.check(
            "acceptance4.key_scan",
            True,
            "skipped: LOP_POC_MODEL_KEY is not in the secret store, so no key exists in this run",
        )
        return
    getter = subprocess.Popen(
        ["lop", "secret", "get", "LOP_POC_MODEL_KEY"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    scanner = subprocess.run(
        [sys.executable, str(probes), "--scan-dir", str(run_dir), "--key-fd", "0"],
        stdin=getter.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    if getter.stdout is not None:
        getter.stdout.close()
    getter.wait()
    result = (scanner.stdout + scanner.stderr).strip()[:300]
    verifier.check("acceptance4.key_scan", scanner.returncode == 0, result)


def cmd_verify(args: argparse.Namespace) -> int:
    import tempfile

    run_dir = Path(args.run_dir)
    fixture_sha = args.fixture_sha
    if not fixture_sha:
        _log("FAIL  --fixture-sha is required: acceptance 2 checks the parent commit")
        return 2
    verifier = Verifier()
    bundle = _find(run_dir, "repo.bundle")
    if bundle is None:
        verifier.check(
            "acceptance2.bundle_present", False, f"repo.bundle not found under {run_dir}"
        )
        return verifier.report()
    verifier.check("acceptance2.bundle_present", True, str(bundle))
    probes = Path(__file__).resolve().parents[1] / "infra/remote-agents-poc/image/probes.py"
    with tempfile.TemporaryDirectory(prefix="lop-poc-fixture-") as tmp:
        clone = Path(tmp) / "fixture"
        cloned = _run(["git", "clone", "--quiet", args.fixture_url, str(clone)])
        if not verifier.check(
            "acceptance2.clone_fixture",
            cloned.returncode == 0,
            (cloned.stdout + cloned.stderr).strip()[:300],
        ):
            return verifier.report()
        branch = _verify_bundle(verifier, clone, bundle, fixture_sha)
        if branch is not None:
            _verify_fixture_tests(verifier, clone, fixture_sha, branch)
    status = read_status(run_dir)
    session_id = str(status.get("session_id") or "")
    verifier.check("acceptance3.session_id_known", bool(session_id), session_id or "<none>")
    if session_id:
        _verify_session(verifier, run_dir, args.lop, session_id)
    # Acceptance 4: the probes' own verdict, reported as a summary (the detailed
    # per-probe evidence stays in probes.json and in run.json).
    probes_summary = read_probes(run_dir)
    failed = probes_summary.get("failed", [])
    verifier.check(
        "acceptance4.no_probe_failed",
        probes_summary.get("available") is True and not failed,
        f"{probes_summary.get('probes')}",
    )
    _verify_key_scan(verifier, run_dir, probes if probes.exists() else None)
    return verifier.report()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE", "minerva_sandbox"),
        help="operator AWS profile whose identity assumes the controller role",
    )
    parser.add_argument(
        "--infra-dir",
        default=str(Path(__file__).resolve().parents[1] / "infra/remote-agents-poc"),
        help="the Pulumi project directory (used with `pulumi stack output`)",
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("PULUMI_BACKEND_URL", ""),
        help="Pulumi backend URL; the POC uses file://$HOME/.lop-poc-pulumi-state",
    )
    parser.add_argument(
        "--outputs",
        default="",
        help="read stack outputs from this JSON file instead of running pulumi",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD", "") or "./poc-runs",
        help="where run artifacts are downloaded (gitignored)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the task once per --runs and record each run")
    _add_common(run)
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--mock", action="store_true", help="use lop's built-in mock provider")
    run.add_argument("--hosting", default="", help="provider id for a real run")
    run.add_argument("--model", default="", help="model id for a real run")
    run.add_argument("--fixture-url", default=FIXTURE_URL_DEFAULT)
    run.add_argument("--fixture-sha", default=FIXTURE_SHA_DEFAULT)
    run.add_argument("--prompt", default=PROMPT_DEFAULT)
    run.set_defaults(func=cmd_run)

    status = subparsers.add_parser("status", help="active tasks must be empty; print tag inventory")
    _add_common(status)
    status.set_defaults(func=cmd_status)

    stop_all = subparsers.add_parser("stop-all", help="StopTask everything in the cluster")
    _add_common(stop_all)
    stop_all.set_defaults(func=cmd_stop_all)

    verify = subparsers.add_parser("verify", help="local acceptance checks for one recorded run")
    _add_common(verify)
    verify.add_argument("run_dir")
    verify.add_argument("--fixture-url", default=FIXTURE_URL_DEFAULT)
    verify.add_argument("--fixture-sha", default=FIXTURE_SHA_DEFAULT, required=False)
    verify.add_argument(
        "--lop",
        default=str(Path(__file__).resolve().parents[1] / ".venv/bin/lop"),
        help="the lop binary used to prove the transplanted session loads",
    )
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.backend_url and not args.outputs:
        args.backend_url = f"file://{Path.home()}/.lop-poc-pulumi-state"
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
