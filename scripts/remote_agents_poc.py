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
from botocore.config import Config
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


def s3_client(session: Any) -> Any:
    """An S3 client whose PRESIGNED URLs address the regional endpoint.

    botocore signs `generate_presigned_url` for S3 against the LEGACY global host
    (`<bucket>.s3.amazonaws.com`) even when the client's own endpoint is regional —
    measured, botocore 1.43.109 — and S3 answers a request for a ca-central-1 bucket
    that way with `307 TemporaryRedirect`. Naming the signature version and the
    addressing style is what puts the bucket's own region back in the Host header,
    which is the only form the signature is good for.
    """
    return session.client(
        "s3",
        endpoint_url=f"https://s3.{REGION}.amazonaws.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


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


def _delta(later: int | None, earlier: int | None) -> int | None:
    return None if later is None or earlier is None else later - earlier


def cold_start_numbers(
    t_runtask_ms: int,
    first_running_wall_ms: int | None,
    described: dict[str, Any],
    timings: dict[str, int],
) -> dict[str, Any]:
    """Cold start, DECOMPOSED, because the headline number is not one cost.

    Two independent readings of the arrival on purpose: ECS's own timestamps
    (`createdAt` → `startedAt`) and the driver's wall clock (RunTask call → first
    RUNNING observation). They come from different clocks and different observers,
    and a run where they disagree by more than a second is a measurement worth
    distrusting rather than averaging.

    The RunTask→first-model-event span is then split, because most of it is not the
    agent at all: measured on the first five runs, the whole-filesystem key scan
    (probe 4c) was 21 s of a 45 s span, so a single number would report the
    instrument's cost as the platform's. `*_minus_probes_ms` is the honest answer to
    "how long until a model could start working", and `probes_duration_ms` says what
    was subtracted.
    """
    created = _timestamp_ms(described.get("createdAt"))
    pull_started = _timestamp_ms(described.get("pullStartedAt"))
    pull_stopped = _timestamp_ms(described.get("pullStoppedAt"))
    started = _timestamp_ms(described.get("startedAt"))
    container_start = timings.get("t_container_start")
    probes_start = timings.get("t_probes_start")
    probes_done = timings.get("t_probes_done")
    first_model_event = timings.get("t_first_model_event")
    probes_ms = _delta(probes_done, probes_start)
    runtask_to_model = _delta(first_model_event, t_runtask_ms)
    container_to_model = _delta(first_model_event, container_start)
    numbers: dict[str, Any] = {
        "runtask_to_running_wall_ms": (
            None if first_running_wall_ms is None else first_running_wall_ms - t_runtask_ms
        ),
        # The THREE phases, separated because the first one is not image pull. The
        # design doc called the 11-16 s "image pull (~57%)" and that was wrong: it is
        # scheduling plus ENI attachment, the pull itself is 4.5-5.9 s and the
        # container start 3.3-4.0 s (measured, six runs). SOCI would improve the pull
        # — the phase that is the reason Q1 asks about it — so mislabelling the
        # scheduling phase as pull overstated SOCI's upside by more than 2x.
        "ecs_created_to_pull_started_ms": _delta(pull_started, created),  # scheduling / ENI
        "ecs_pull_started_to_pull_stopped_ms": _delta(pull_stopped, pull_started),  # the pull
        "ecs_pull_stopped_to_started_ms": _delta(started, pull_stopped),  # container start
        "ecs_pull_started_to_started_ms": _delta(started, pull_started),  # pull + start, kept
        "ecs_created_to_started_ms": _delta(started, created),
        # RUNNING -> the entrypoint's first instruction, i.e. what the platform costs
        # AFTER the container is up (the payload is already resident at that point).
        "ecs_started_to_container_start_ms": _delta(container_start, started),
        "container_start_to_probes_start_ms": _delta(probes_start, container_start),
        "probes_duration_ms": probes_ms,
        "probes_done_to_first_model_event_ms": _delta(first_model_event, probes_done),
        "container_start_to_first_model_event_ms": container_to_model,
        "runtask_to_first_model_event_ms": runtask_to_model,
        "container_start_to_first_model_event_minus_probes_ms": (
            None
            if container_to_model is None or probes_ms is None
            else container_to_model - probes_ms
        ),
        "runtask_to_first_model_event_minus_probes_ms": (
            None if runtask_to_model is None or probes_ms is None else runtask_to_model - probes_ms
        ),
        "ecs_started_to_first_model_event_ms": _delta(first_model_event, started),
    }
    return numbers


def _load_timings(results_dir: Path) -> dict[str, int]:
    """The container's step timings, from `timings.json`, falling back to the jsonl.

    The jsonl is what the entrypoint appends to AS IT GOES; `timings.json` is the
    folded copy. Reading only the folded file left every cold-start number null on a
    run whose jsonl carried them all — the fold is written by the entrypoint's EXIT
    trap, which runs after the results tarball is already sealed. The fallback is the
    difference between a real measurement and an empty field.
    """
    folded = results_dir / "timings.json"
    if folded.exists():
        payload = json.loads(folded.read_text(encoding="utf-8"))
        return {str(event["name"]): int(event["epoch_ms"]) for event in payload.get("events", [])}
    raw = results_dir / "timings.jsonl"
    if not raw.exists():
        return {}
    stamps: dict[str, int] = {}
    for line in raw.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            event = json.loads(line)
            stamps[str(event["name"])] = int(event["epoch_ms"])
    return stamps


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
    definition: dict[str, Any],
) -> dict[str, Any]:
    """One task, from RunTask to stopped-and-downloaded. Returns its run.json.

    ``definition`` is the task definition, resolved by the caller with the OPERATOR
    session: the controller role deliberately does not hold
    `ecs:DescribeTaskDefinition` (the spec's policy ends at "Nothing else"), and the
    driver needs it for the CPU architecture, which DescribeTasks does not return
    for Fargate.
    """
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
            {"key": key, "value": value} for key, value in {**POC_TAGS, "run-id": run_id}.items()
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
    # TASK DEFINITION (resolved and required ACTIVE before RunTask, above), which
    # is where the architecture is actually declared. The describe-tasks reading is
    # kept beside it so the divergence is visible rather than explained away.
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
        "environ_watch": read_environ_watch(run_dir),
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


def read_environ_watch(run_dir: Path) -> dict[str, Any]:
    """Probe 4e's verdict: no process's INITIAL environment carried the key.

    Read from the run's artifact rather than re-derived: the watcher samples only
    while the agent is alive, and that window is gone by verify time. The residual it
    records (yama/ptrace_scope and whether the agent's memory was openable from a
    same-uid non-descendant) rides along, because it is the part of SEC-1 the environ
    claim cannot speak for.
    """
    path = _find(run_dir, "proc-env-watch.json")
    if path is None:
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "pass": payload.get("pass"),
        "samples": payload.get("samples"),
        "processes_scanned_max": payload.get("processes_scanned_max"),
        "matches_by_needle": payload.get("matches_by_needle"),
        "matching_processes": payload.get("matching_processes"),
        # What the watcher READ, matched or not: the coverage claim ("the child was in
        # the set") is only checkable if this rides along, and `verify` on a self-test
        # directory is where it is checked.
        "observed_processes": payload.get("observed_processes"),
        "residual": payload.get("residual"),
        "note": payload.get("note"),
    }


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


def active_task_definition(operator: Any, outputs: dict[str, Any]) -> dict[str, Any]:
    """The task definition the stack points at, required to be ACTIVE.

    A `pulumi up` that changes the task definition REPLACES it, retiring the
    previous revision; a stale outputs file then names an inactive ARN and RunTask
    answers the unhelpful "TaskDefinition is inactive". Naming the cause here is the
    difference between a five-second fix and an hour spent looking at IAM. Read with
    the OPERATOR session, because the controller role deliberately holds no
    `ecs:DescribeTaskDefinition`.
    """
    definition = operator.client("ecs").describe_task_definition(
        taskDefinition=str(outputs["taskDefinitionArn"])
    )["taskDefinition"]
    if str(definition.get("status", "")).upper() != "ACTIVE":
        raise SystemExit(
            f"task definition {outputs['taskDefinitionArn']} is {definition.get('status')!r}: "
            "its ARN is stale (a replaced task definition retires the old revision) — "
            "re-read the stack outputs before running"
        )
    return definition


def cmd_run(args: argparse.Namespace) -> int:
    operator, outputs, driver = _driver_session(args)
    ecs = driver.client("ecs")
    s3 = s3_client(driver)
    definition = active_task_definition(operator, outputs)
    _log(
        "task definition {family}:{revision} ({cpu}), image {image}".format(
            family=definition.get("family"),
            revision=definition.get("revision"),
            cpu=(definition.get("runtimePlatform") or {}).get("cpuArchitecture"),
            image=(definition.get("containerDefinitions") or [{}])[0].get("image"),
        )
    )
    records: list[dict[str, Any]] = []
    for _ in range(args.runs):
        # Re-asserted per run, not once per sweep: `--runs 5` is five RunTasks, and
        # the guard that matters is the one immediately before each of them.
        assert_expected_account(operator)
        record = run_once(args, ecs, s3, outputs, new_run_id(), definition)
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
    # The verdict line carries its own TIME AND COUNT. The inventory moves between
    # readings (STOPPED task records age out about an hour after each stop), so a quoted
    # line without a timestamp cannot be checked against a heading that counted later —
    # which is exactly how "45 tagged resources" ended up under a "51 ARNs" heading
    # (agent review round 3, finding 3).
    observed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = {
        "cluster": cluster,
        "active_tasks": active,
        "active_count": len(active),
        "tagged_resource_count": len(inventory),
        "observed_at": observed_at,
        "tagged_resources": sorted(inventory, key=lambda item: str(item["arn"])),
    }
    print(json.dumps(report, indent=2, default=str))
    if active:
        _log(f"FAIL: {len(active)} active task(s) in {cluster} at {observed_at}")
        return 1
    _log(
        f"OK: no RUNNING or PENDING tasks in {cluster}; {len(inventory)} tagged lop-poc "
        f"resources at {observed_at}"
    )
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


def harness_child_environment(overrides: dict[str, str]) -> dict[str, str]:
    """The environment a HARNESS gives the real CLI it drives.

    WHY NOT A BARE DICT: a child a rig drives is a session nobody is watching, and
    the ``test`` hosting's only reply is "Hello from the mock provider!" — a
    notification body being a snippet of the session's own last assistant line, so a
    drive-by rig puts that sentence on the operator's lock screen (17 recorded banner
    attempts across scratch stores in two days; see
    ``agent_shell.harness_child_env``). The gate and its value come from that
    helper's single definition rather than a literal here, and the nested-session
    allowance it also sets is what lets the driven CLI open a session at all when the
    rig itself is running inside an agent's shell. Pinned by
    ``tests/unit/test_notification_isolation.py``, which fails any module under
    ``scripts/`` that builds a child environment without one of the gate spellings.
    """
    from local_operator.agent_shell import harness_child_env

    return harness_child_env(overrides)


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a child. ``stdin`` exists for the resume check: a driven ``lop exec``
    must read stdin as ``/dev/null`` — an unattended run — not a terminal."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        stdin=stdin,
    )


def _find(run_dir: Path, name: str) -> Path | None:
    """Artifacts live under results/ when the tarball was extracted, and directly
    under the run dir for the objects downloaded before extraction."""
    for candidate in (run_dir / "results" / name, run_dir / name):
        if candidate.exists():
            return candidate
    return None


def _transcript_lines(path: Path) -> list[str]:
    """A session transcript as non-empty lines; ``[]`` when it is absent.

    Lines rather than a count, because acceptance 3 has to compare the PRIOR lines
    after the resume — a count alone cannot tell an appended turn from a rewritten
    store.
    """
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fixture_test_command(clone: Path) -> list[str]:
    """The fixture ships a pytest-free runner so the container needs no pytest; use
    it when present so verify tests the same thing the agent was asked to fix."""
    if (clone / "run_tests.py").exists():
        return [sys.executable, "run_tests.py"]
    return [sys.executable, "-m", "pytest", "-q"]


class Verifier:
    """Collects (name, ok, detail) rows and prints them as a PASS/FAIL table.

    BLOCKED is a third outcome and not a soft FAIL: acceptance 2 needs a commit, and
    a mock-provider run makes none by design, so reporting it as a failure would make
    the mock runs look broken when they are the ones that prove everything else. A
    blocked check names what would unblock it.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []
        self.blocked_rows: list[tuple[str, str]] = []

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.rows.append((name, ok, detail))
        _log(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        return ok

    def blocked(self, name: str, detail: str) -> None:
        self.blocked_rows.append((name, detail))
        _log(f"BLOCKED  {name}: {detail}")

    def report(self) -> int:
        failed = [name for name, ok, _ in self.rows if not ok]
        print(
            json.dumps(
                {
                    "checks": self.rows,
                    "failed": failed,
                    "blocked": self.blocked_rows,
                },
                indent=2,
                default=str,
            )
        )
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
        # says must never reach a second runtime. `harness_child_environment` adds the
        # notification gate the repo requires of every rig that drives the real CLI.
        env = harness_child_environment(
            {
                "HOME": iso,
                "LOCAL_OPERATOR_CONFIG_DIR": str(root / config_name),
                "PATH": os.environ.get("PATH", ""),
                "TERM": "dumb",
            }
        )
        transcript = root / config_name / "sessions" / session_id / "transcript.jsonl"
        before = _transcript_lines(transcript)
        verifier.check(
            "acceptance3.transcript_non_empty",
            bool(before),
            f"{len(before)} transcript line(s) before the resume",
        )
        # `--all` is load-bearing: a bare `lop sessions` lists only ACTIVE (running)
        # sessions, so a transplanted session — which is stored, never running — comes
        # back as an empty list and looks like a failed transplant. It is not: the
        # stored rows are what this check is for.
        listed = _run([lop, "sessions", "--all", "--json"], env=env)
        listed_ok = listed.returncode == 0 and session_id in listed.stdout
        verifier.check(
            "acceptance3.lop_sessions_lists_id",
            listed_ok,
            f"exit {listed.returncode}: {(listed.stdout + listed.stderr).strip()[:300]}",
        )
        # THE REAL RESUME PATH, driven headlessly. §9.3 item 3 says the transplanted
        # session must OPEN, and `lop --resume` (the TUI form) and `lop exec --resume`
        # load the same store through the same loader — `exec` is the non-TTY surface
        # of that load, so this exercises the load rather than a listing. It runs the
        # mock hosting, so it needs no key.
        resumed = _run(
            [
                lop,
                "exec",
                "--resume",
                session_id,
                "--hosting",
                "test",
                "--model",
                "test-model",
                "--json",
                "ping",
            ],
            env=env,
            stdin=subprocess.DEVNULL,
        )
        reused = session_id in resumed.stdout
        verifier.check(
            "acceptance3.resume_reuses_session",
            resumed.returncode == 0 and reused,
            f"exit {resumed.returncode}; same session id in the event stream: {reused}; "
            f"{(resumed.stdout + resumed.stderr).strip()[-220:]}",
        )
        after = _transcript_lines(transcript)
        verifier.check(
            "acceptance3.resume_grew_transcript",
            len(after) > len(before),
            f"{len(before)} -> {len(after)} transcript line(s)",
        )
        verifier.check(
            "acceptance3.resume_kept_prior_lines",
            bool(before) and after[: len(before)] == before,
            f"the first {len(before)} line(s) are byte-identical after the resume",
        )


def _verify_key_scan(verifier: Verifier, run_dir: Path, probes: Path | None) -> None:
    """Acceptance 4: no copy of the model key exists in what was downloaded.

    The key is piped from the secret store into probes.py's stdin, never argv, so it
    is not in this process's command line and never in this transcript — and this
    process never holds it either: the store's stdout IS the scanner's stdin.

    A SCAN THAT INSPECTED NOTHING IS NOT A PASS. Until this round the verdict was
    ``scanner.returncode == 0``, and ``probes.py:_main_scan`` returned 0 for an empty
    key — so a store hiccup reported PASS with ``{"scanned_files": 0}`` as its
    evidence, the "dead instrument returns a reading" shape. Now rc 2 (no key
    delivered), a nonzero exit from the getter, and a zero-file scan are all BLOCKED,
    and only a scan that actually read files can pass.
    """
    if probes is None:
        verifier.check("acceptance4.key_scan", False, "probes.py not found; cannot scan")
        return
    listing = _run(["lop", "secret", "list"])
    if "LOP_POC_MODEL_KEY" not in listing.stdout:
        verifier.blocked(
            "acceptance4.key_scan",
            "LOP_POC_MODEL_KEY is not in the secret store, so no real key exists to scan for; "
            "the container's own 4c/4c-env/4e probes still ran, against the injected placeholder",
        )
        return
    # stderr goes to DEVNULL rather than an undrained pipe: a chatty failure on a
    # pipe nobody reads can fill the buffer and deadlock the scan (review CODE-12).
    getter = subprocess.Popen(
        ["lop", "secret", "get", "LOP_POC_MODEL_KEY"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    # The same needles the container used: the exact value, and its first 8
    # characters, so "the local scan passed" means what "probe 4c passed" means.
    scanner = subprocess.run(
        [
            sys.executable,
            str(probes),
            "--scan-dir",
            str(run_dir),
            "--key-fd",
            "0",
            "--key-prefix-chars",
            "8",
        ],
        stdin=getter.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    if getter.stdout is not None:
        getter.stdout.close()
    getter_rc = getter.wait()
    if getter_rc != 0:
        verifier.blocked(
            "acceptance4.key_scan", f"`lop secret get` exited {getter_rc}: nothing was scanned"
        )
        return
    result = (scanner.stdout + scanner.stderr).strip()[:300]
    if scanner.returncode == 2:
        verifier.blocked("acceptance4.key_scan", f"no key reached the scanner: {result}")
        return
    scanned = 0
    try:
        scanned = int(json.loads(scanner.stdout)["scanned_files"])
    except (ValueError, KeyError, TypeError):
        scanned = 0
    if scanned == 0:
        verifier.blocked("acceptance4.key_scan", f"the scan inspected 0 files: {result}")
        return
    verifier.check("acceptance4.key_scan", scanner.returncode == 0, result)


#: Every probe, in one fixed order, so two runs' rows line up under each other.
_PROBE_ORDER = (
    "4a_creds_endpoint",
    "4b_egress",
    "4c_no_secret_on_disk",
    "4c_env_no_key_in_child_env",
    "4d_platform",
    "4f_no_ps",
)

#: The phases the results doc's table publishes, in the order it reads them. The span
#: row is here because the three ECS phases cover `createdAt`→`startedAt` and NOT
#: `RunTask`→`RUNNING`: those differ by the control plane's own ~1 s, and a table that
#: prints both without naming the span is how they get read as the same interval.
_PHASE_ORDER = (
    ("runtask_to_running_wall_ms", "RunTask call → first RUNNING (driver wall clock)"),
    (
        "ecs_created_to_pull_started_ms",
        "**scheduling + ENI attach** (`createdAt` → `pullStartedAt`)",
    ),
    ("ecs_pull_started_to_pull_stopped_ms", "**image pull** (`pullStartedAt` → `pullStoppedAt`)"),
    ("ecs_pull_stopped_to_started_ms", "**container start** (`pullStoppedAt` → `startedAt`)"),
    ("ecs_created_to_started_ms", "the three phases above as one span (`createdAt` → `startedAt`)"),
    ("container_start_to_probes_start_ms", "container's first stamp → probes start"),
    ("probes_duration_ms", "**probes (4a–4f, incl. the whole-filesystem key scan)**"),
    ("probes_done_to_first_model_event_ms", "probes done → first model event"),
    ("container_start_to_first_model_event_ms", "container start → first model event"),
    ("runtask_to_first_model_event_ms", "RunTask → first model event (raw)"),
    (
        "runtask_to_first_model_event_minus_probes_ms",
        "**RunTask → first model event, minus the probes**",
    ),
)


def _record_dirs(out_dir: Path) -> list[Path]:
    """Run directories under ``out_dir``, oldest first."""
    if not out_dir.is_dir():
        return []
    return sorted(
        (path for path in out_dir.iterdir() if path.is_dir() and (path / "run.json").exists()),
        key=lambda path: (path / "run.json").stat().st_mtime,
    )


def _record(run_dir: Path) -> dict[str, Any] | None:
    """One run's record, or None when it has no pair of files to derive from."""
    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        described = json.loads((run_dir / "describe-tasks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    containers = described.get("containers") or [{}]
    return {
        "dir": run_dir,
        "run_id": str(run.get("run_id", run_dir.name)),
        "task_arn": str(run.get("task_arn", "")),
        "digest": str(containers[0].get("imageDigest") or ""),
        "cpu": str(run.get("cpu_architecture", "")),
        "stop_code": str(run.get("stop_code", "")),
        "exit_code": run.get("exit_code"),
        "cold": run.get("cold_start", {}) or {},
        "probes": run.get("probes", {}).get("probes", {}) or {},
        "watch": run.get("environ_watch", {}) or {},
        # ECS's own creation time, for a CHRONOLOGICAL table: file mtimes reorder when
        # artifacts are re-downloaded, and a table whose rows move between readings is
        # harder to diff against the artifacts than one whose order is the run order.
        "created_at": str((run.get("ecs_timestamps") or {}).get("createdAt", "")),
    }


def _probe_flags(probes: dict[str, Any]) -> str:
    """P / F / n per probe in a fixed order.

    ``n`` is a probe that returned ``pass=None``: this run had nothing to look at (4c
    with no key injected), which is neither a pass nor a failure and must not be
    printed as either.
    """
    flags: list[str] = []
    for name in _PROBE_ORDER:
        entry = probes.get(name)
        if entry is None:
            flags.append("-")
        elif entry.get("pass") is True:
            flags.append("P")
        elif entry.get("pass") is False:
            flags.append("F")
        else:
            flags.append("n")
    return "/".join(flags)


def _watch_flag(watch: dict[str, Any]) -> str:
    """P / F / n for the watcher column, the same convention ``_probe_flags`` uses.

    A run with no ``environ_watch`` at all is ``n`` — "nothing to read" — not the string
    ``None`` printed into a column of P/F letters.
    """
    value = watch.get("pass")
    if value is True:
        return "P"
    if value is False:
        return "F"
    return "n"


def _secs(value: Any) -> str:
    return f"{value / 1000.0:.2f}" if isinstance(value, (int, float)) else "—"


def _min_median_max(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    return ordered[0], ordered[len(ordered) // 2], ordered[-1]


def cmd_report(args: argparse.Namespace) -> int:
    """Render the results doc's evidence tables FROM the recorded artifacts.

    WHY THIS EXISTS (agent review round 2, finding 1): the cold-start table's per-run
    decomposition was typed by hand, and three of its fifteen cells were values that
    occur nowhere in the records — while the identity columns and the summary row were
    right, which is exactly the shape that survives review. Every number below is read
    from ``run.json``/``describe-tasks.json``, so a reviewer's sweep either matches cell
    for cell or the tool is wrong, and the fix for the next drift is one command.
    """
    out_dir = Path(args.report_dir)
    records = [record for record in (_record(path) for path in _record_dirs(out_dir)) if record]
    if not records:
        print(f"no run records under {out_dir}", file=sys.stderr)
        return 2
    records.sort(key=lambda record: (str(record["created_at"]), str(record["run_id"])))
    digest = str(args.digest or "").strip()
    if digest in ("", "latest"):
        digest = str(records[-1]["digest"])
    chosen = [record for record in records if record["digest"] == digest]
    if not chosen:
        print(f"no runs on digest {digest} under {out_dir}", file=sys.stderr)
        return 2
    print(f"<!-- generated by: remote_agents_poc.py report {out_dir} --digest {digest} -->")
    print()
    print(f"#### Per-run cells, {len(chosen)} run(s) on `{digest}`")
    print()
    print(
        "| run id | task arn (suffix) | cpuArch | stopCode | exit | RunTask→RUNNING s "
        "| scheduling/ENI s | image pull s | container start s | probes s "
        "| RunTask→1st model event s | minus probes s | probes | 4e |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for record in chosen:
        cold = record["cold"]
        print(
            f"| {record['run_id']} | `…{record['task_arn'][-12:]}` | {record['cpu']} "
            f"| {record['stop_code']} | {record['exit_code']} "
            f"| {_secs(cold.get('runtask_to_running_wall_ms'))} "
            f"| {_secs(cold.get('ecs_created_to_pull_started_ms'))} "
            f"| {_secs(cold.get('ecs_pull_started_to_pull_stopped_ms'))} "
            f"| {_secs(cold.get('ecs_pull_stopped_to_started_ms'))} "
            f"| {_secs(cold.get('probes_duration_ms'))} "
            f"| {_secs(cold.get('runtask_to_first_model_event_ms'))} "
            f"| {_secs(cold.get('runtask_to_first_model_event_minus_probes_ms'))} "
            f"| {_probe_flags(record['probes'])} "
            f"| {_watch_flag(record['watch'])} |"
        )
    print()
    print(f"#### Summary over those {len(chosen)} run(s): min / median / max, seconds")
    print()
    print("| phase | min | median | max |")
    print("| --- | --- | --- | --- |")
    for key, label in _PHASE_ORDER:
        values = [
            record["cold"][key]
            for record in chosen
            if isinstance(record["cold"].get(key), (int, float))
        ]
        if not values:
            print(f"| {label} | — | — | — |")
            continue
        low, middle, high = _min_median_max([float(value) for value in values])
        print(f"| {label} | {low / 1000.0:.2f} | {middle / 1000.0:.2f} | {high / 1000.0:.2f} |")
    print()
    print("#### Population: every recorded run, grouped by the digest it RAN (not by what")
    print("the task definition says now)")
    print()
    print("| image digest | runs | run ids |")
    print("| --- | --- | --- |")
    by_digest: dict[str, list[str]] = {}
    for record in records:
        by_digest.setdefault(str(record["digest"]), []).append(str(record["run_id"]))
    for other, ids in sorted(by_digest.items(), key=lambda item: -len(item[1])):
        marker = " **← this table**" if other == digest else ""
        print(f"| `{other[:20]}…`{marker} | **{len(ids)}** | {', '.join(ids)} |")
    print()
    print(f"total recorded runs: {len(records)}")
    _print_selftests(out_dir)
    return 0


def _print_selftests(out_dir: Path) -> None:
    """The watcher self-tests' records, read from the same directories.

    A self-test directory carries no run.json, so it is not one of the runs above; it is
    where 4e's two non-mock properties are measured, and the driver's by-NAME check lives
    only here (see `selftest_verdict` for why it is not in the launcher).
    """
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    if out_dir.is_dir():
        for path in sorted(out_dir.iterdir()):
            if not path.is_dir():
                continue
            record = _read_selftest(path)
            if record is not None:
                rows.append((path.name, record, read_environ_watch(path)))
    if not rows:
        return
    print()
    print("#### Watcher self-tests: what each mode's child inherited, and what the watcher read")
    print()
    print(
        "| out-dir | mode | child pid | child argv | child env entries | variable name "
        "| key value | watcher | samples / scanned_max | child in observed set | driver verdict |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, record, watch in rows:
        ok, detail = selftest_verdict(record)
        expected = record.get("expected_key_in_child_env") is True
        observed = list(watch.get("observed_processes") or [])
        observed_flag = (
            "yes" if any(e.get("pid") == record.get("child_pid") for e in observed) else "no"
        )
        verdict = (
            "GREEN" if watch.get("pass") is True else "RED" if watch.get("pass") is False else "n/a"
        )
        print(
            f"| `{name}` | {record.get('mode')} | {record.get('child_pid')} "
            f"| `{' '.join(str(part) for part in record.get('child_argv') or [])}` "
            f"| {record.get('child_env_entries')} "
            f"| {record.get('provider_var_in_child_env')} (expected {expected}) "
            f"| {record.get('key_value_in_child_env')} (expected {expected}) "
            f"| {verdict} "
            f"| {watch.get('samples')} / {watch.get('processes_scanned_max')} "
            f"| {observed_flag} "
            f"| **{'PASS' if ok else 'FAIL'}** — {detail} |"
        )
    print()
    print(
        "The by-NAME column is the driver's check, not the launcher's `pass` predicate: "
        "the launcher asserts the VALUE only, and tightening it would mean changing the "
        "image the accepted runs were made with."
    )
    return None


def _read_selftest(run_dir: Path) -> dict[str, Any] | None:
    """The launcher's self-test record for a self-test directory, or None.

    The launcher's stdout is redirected into this file ahead of its JSON line
    (``lop-launch: …``), so the record is the LAST line that parses and carries the
    fields: anything before it is a log line.
    """
    path = _find(run_dir, "selftest_child.json")
    if path is None:
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            record = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(record, dict) and "expected_key_in_child_env" in record:
            return record
    return None


def selftest_verdict(record: dict[str, Any]) -> tuple[bool, str]:
    """Did a self-test child inherit what its mode predicts — by NAME and by VALUE?

    WHY THIS LIVES IN THE DRIVER AND NOT IN THE LAUNCHER. The launcher's own ``pass``
    predicate asserts the VALUE only (``key_value_in_child_env == expected``); the
    round-3 version also required the provider variable NAME to be absent, and that
    term was lost when the predicate was generalised over two modes. Tightening it
    again means editing ``image/lop_launch.py``, and the image is what the accepted runs
    were made with — the repository has to keep matching digest ``d70840cf…``. So the
    stricter check lives where it can be tightened without touching what ran: a filtered
    child that inherited the variable NAME is a regression the launcher reports as a
    pass and this fails.
    """
    expected = record.get("expected_key_in_child_env")
    if not isinstance(expected, bool):
        return False, "no expected_key_in_child_env in the record: cannot judge the mode"
    problems: list[str] = []
    for field, label in (
        ("provider_var_in_child_env", "provider variable NAME"),
        ("key_value_in_child_env", "key VALUE"),
    ):
        recorded = record.get(field)
        if recorded is not expected:
            problems.append(f"{label}: {field}={recorded!r}, expected {expected!r}")
    if record.get("child_exit") != 0:
        problems.append(f"child_exit={record.get('child_exit')}")
    if problems:
        return False, "; ".join(problems)
    return True, (
        f"mode={record.get('mode')}: the variable name and the value are both "
        f"{'present, as this mode expects' if expected else 'absent, as this mode expects'}"
    )


def _verify_selftest(verifier: Verifier, run_dir: Path, record: dict[str, Any]) -> None:
    """The self-test's own acceptance: inheritance, the watcher's agreement, coverage.

    Three checks rather than one, because they answer three different questions: did the
    child inherit what the mode predicts (the launcher's record), did the watcher
    independently agree (its verdict), and was the child in the set the watcher actually
    read (the coverage claim). The third is the one round 3 found was asserted rather
    than measured.
    """
    ok, detail = selftest_verdict(record)
    verifier.check("selftest.child_env_matches_mode", ok, detail)
    watch = read_environ_watch(run_dir)
    expects_red = record.get("expected_key_in_child_env") is True
    verifier.check(
        "selftest.watcher_agrees_with_mode",
        watch.get("pass") is (not expects_red),
        f"{watch.get('note')} (this mode expects {'RED' if expects_red else 'GREEN'})",
    )
    observed = list(watch.get("observed_processes") or [])
    child_pid = record.get("child_pid")
    verifier.check(
        "selftest.child_was_observed",
        any(entry.get("pid") == child_pid for entry in observed),
        f"child_pid={child_pid} in observed_processes={observed}",
    )


def cmd_verify(args: argparse.Namespace) -> int:
    import tempfile

    run_dir = Path(args.run_dir)
    verifier = Verifier()
    selftest = _read_selftest(run_dir)
    if selftest is not None:
        # A self-test directory is not a run: it has no session, no bundle and no
        # run.json, so the acceptance checks below would report FAILs about artifacts it
        # was never meant to produce. Its own three checks ARE the acceptance for it, and
        # this path needs no --fixture-sha.
        _verify_selftest(verifier, run_dir, selftest)
        return verifier.report()
    fixture_sha = args.fixture_sha
    if not fixture_sha:
        _log("FAIL  --fixture-sha is required: acceptance 2 checks the parent commit")
        return 2
    git_info = read_git(run_dir)
    bundle = _find(run_dir, "repo.bundle")
    if not git_info.get("commit"):
        # A mock-provider run makes no edits, so there is no branch and no bundle to
        # verify: acceptance 2 is BLOCKED, not failed, and it names what would unblock
        # it. Everything else still runs — a mock run produces a real session
        # directory and real probe verdicts, which is most of the acceptance.
        verifier.blocked(
            "acceptance2.bundle_and_branch",
            "no commit on this run (git.json has commit=null), so there is no lop/<id> "
            "branch to verify: acceptance 2 needs a run whose model actually edited the "
            "fixture, i.e. a real-key run",
        )
    elif bundle is None:
        verifier.check(
            "acceptance2.bundle_present", False, f"repo.bundle not found under {run_dir}"
        )
    else:
        verifier.check("acceptance2.bundle_present", True, str(bundle))
        with tempfile.TemporaryDirectory(prefix="lop-poc-fixture-") as tmp:
            clone = Path(tmp) / "fixture"
            cloned = _run(["git", "clone", "--quiet", args.fixture_url, str(clone)])
            if verifier.check(
                "acceptance2.clone_fixture",
                cloned.returncode == 0,
                (cloned.stdout + cloned.stderr).strip()[:300],
            ):
                branch = _verify_bundle(verifier, clone, bundle, fixture_sha)
                if branch is not None:
                    _verify_fixture_tests(verifier, clone, fixture_sha, branch)
    probes = Path(__file__).resolve().parents[1] / "infra/remote-agents-poc/image/probes.py"
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
    watch = read_environ_watch(run_dir)
    verifier.check(
        "acceptance4e.no_key_in_any_process_environ",
        watch.get("pass") is True,
        f"{watch.get('note')}; residual={watch.get('residual')}",
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

    report = subparsers.add_parser(
        "report",
        help="render the evidence tables from the recorded artifacts",
        description=(
            "Reads only the contents of a driver --out-dir: no AWS profile and no Pulumi "
            "backend are needed, so the directory is the one positional argument and this "
            "subcommand does NOT take the common options. It used to take both, and the "
            "optional silently won over the positional (agent review round 3, finding 5)."
        ),
    )
    report.add_argument("report_dir", metavar="out-dir", help="the driver's --out-dir")
    report.add_argument(
        "--digest",
        default="latest",
        help="the image digest whose runs to tabulate ('latest' = the newest record's)",
    )
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # `report` deliberately carries none of the common options, so this default is
    # applied only for the subcommands that have a backend to point at.
    if not getattr(args, "backend_url", None) and not getattr(args, "outputs", None):
        args.backend_url = f"file://{Path.home()}/.lop-poc-pulumi-state"
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
