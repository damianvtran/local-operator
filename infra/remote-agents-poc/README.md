# remote-agents-poc — Slice-0 POC infrastructure

One Fargate task that clones a public fixture repo, runs `lop exec` against it with
no AWS permissions of its own, commits a branch, and uploads its results. This is
**Slice 0** of `docs/design/remote-cloud-agents.md` §9.2: cloud run, no mesh. It
exists to answer "does the AWS lifecycle work, is the isolation what §7.1 claims,
and what does a cold start cost" — not to ship a product feature.

Nothing here is imported by the product. `scripts/remote_agents_poc.py` is a POC
driver, `infra/remote-agents-poc/` is a standalone Pulumi project, and the image is
built in CodeBuild, never on a laptop. **What it measured** — the §9.3 acceptance
table, the cold-start decomposition and the cost so far: `docs/design/
remote-cloud-agents-poc-results.md`.

## Layout

| Path | What it is |
| --- | --- |
| `Pulumi.yaml` | The whole stack, as a Pulumi YAML program (one stack, `poc`) |
| `image/Dockerfile` | `python:3.12-slim` (pinned by index digest) + git, curl, uv, and TWO venvs |
| `image/requirements.in` | The lop venv's input: `local-operator==0.68.3` |
| `image/requirements.txt` | Its hash-locked output, compiled for `aarch64-unknown-linux-gnu` |
| `image/requirements-probe.in` | The probe venv's input: `boto3` |
| `image/requirements-probe.txt` | Its hash-locked output |
| `image/entrypoint.sh` | Container steps 1-9 of §9.2 (timings, key handling, clone, probes, agent, commit, upload) |
| `image/probes.py` | The deterministic isolation probes (4a, 4b, 4c, 4c-env, 4d) and the pre-upload key rescan |
| `image/buildspec.yml` | CodeBuild buildspec (arm64 native, pushes by tag, prints the digest) |
| `../../scripts/remote_agents_poc.py` | The driver: `run`, `verify`, `status`, `stop-all` |

## Decisions

**Why Pulumi YAML.** The stack is a flat list of resources with no control flow, so a
typed program would add a second language plus a build step for nothing. The one
thing a typed language would buy — conditionals — is the one thing the two-phase
deploy below does not need: it uses `--exclude`, so the program never has to branch.

**Why CodeBuild and not a laptop `docker build`.** The image must be `linux/arm64`
and land in a private, IMMUTABLE-tag ECR repository. `ARM_CONTAINER` on a Graviton
host builds that natively (no qemu), and the service role's ECR permissions never
leave AWS, so no human holds push credentials. The build also smokes the image
(`docker run --entrypoint lop <img> --version`) BEFORE pushing.

**Why an AWS-managed `aws/s3` KMS key and not a CMK.** A CMK cannot be deleted
immediately, so `pulumi destroy` would leave a pending-deletion key that still
carries the `lop-poc` tag — and the teardown check ("zero tagged resources") could
then never pass. `aws/s3` gives the same SSE-KMS property with nothing to leak
after teardown. Note what "the aws/s3 key" means in the resource: `sseAlgorithm:
aws:kms` with **no `kmsMasterKeyId`**. Writing `kmsMasterKeyId: aws/s3` previews
clean and then fails every `PutObject` with `KMS.NotFoundException: Invalid keyId
'aws/s3'` — the managed key is what omitting the key id expresses. (That failure was
measured, on the first context upload.)

**Why its own VPC (10.77.0.0/24) instead of the existing `sbx-vpc`.** This account
is shared with Pergamon's sandbox, and the POC's whole point is a security group
with "no ingress, egress tcp/443 only". Adding subnets, routes or security groups
to `sbx-vpc` would mutate resources the POC does not own. A 30-address VPC costs
nothing (no NAT, no endpoints) and makes teardown a single `pulumi destroy`.

**Why DNS still resolves with an egress rule of tcp/443 only.** AWS exempts the VPC
resolver from security-group filtering ("You cannot filter traffic to or from the
Amazon DNS server using network ACLs or security groups"), so name resolution works
without a UDP/53 rule. Probe 4b's positive control (a 443 connect to github.com)
is what proves it at run time; `1.1.1.1:53` failing is what proves the exception is
scoped to the VPC resolver and not to port 53 in general.

**Why the image declares `VOLUME ["/workspace"]`.** A Fargate task volume arrives
mounted root-owned, and a container started as uid 10001 can neither write it nor
chown it — measured, the first real run died on `mkdir: cannot create directory
'/workspace/out': Permission denied` before the entrypoint's second step. The fix is
the documented pairing of a Dockerfile `VOLUME` with a matching task-definition
`containerPath`: when they are equal, the ECS agent copies the image's data **and its
ownership** into the mount, so the `/workspace` the image chowns to 10001 arrives
writable by 10001. The `chown` must run BEFORE the `VOLUME` declaration — changes to
a VOLUME path after it are discarded at run time. That is why the image ends
`chown … && VOLUME ["/workspace"]` and then `USER 10001:10001`, and why the
task definition's `user: "10001:10001"` is possible at all. An earlier revision ran
a root phase-0 that chowned the volume and re-exec'd the entrypoint as 10001; it
worked, and it is gone because it should not have been necessary. Probe 4d asserts
uid 10001 on every run, so a regression fails the run rather than the claim.

## State

Pulumi state is in a **file backend outside the repository**:

```sh
export PULUMI_BACKEND_URL="file://$HOME/.lop-poc-pulumi-state"
```

`pulumi login` is never run — it would repoint the operator's global backend at
whatever this project asked for. The passphrase lives in the secret store:

```sh
openssl rand -base64 32 | lop secret set LOP_POC_PULUMI_PASSPHRASE \
    --description "Pulumi file-backend passphrase for the lop remote-agents POC stack"
```

and every `pulumi` invocation is wrapped so it reaches the CLI as
`PULUMI_CONFIG_PASSPHRASE` rather than an argument:

```sh
lop secret run --secret LOP_POC_PULUMI_PASSPHRASE=PULUMI_CONFIG_PASSPHRASE -- pulumi preview
```

The AWS provider plugin is pinned by what is installed (`7.40.0`) with
`PULUMI_DISABLE_AUTOMATIC_PLUGIN_ACQUISITION=true`, so a run can never download a
different one.

## Deploy order (two phases, and why two)

The task definition needs an image that does not exist until the ECR repository
does, and CodeBuild needs the results bucket. So the stack goes up in two passes,
and the second one is just a plain `pulumi up`:

```sh
cd infra/remote-agents-poc
export AWS_PROFILE=minerva_sandbox AWS_REGION=ca-central-1
export PULUMI_BACKEND_URL="file://$HOME/.lop-poc-pulumi-state"
export PULUMI_DISABLE_AUTOMATIC_PLUGIN_ACQUISITION=true
aws sts get-caller-identity                    # MUST be 325492156725
PUL="lop secret run --secret LOP_POC_PULUMI_PASSPHRASE=PULUMI_CONFIG_PASSPHRASE -- pulumi"

# Phase 1 — everything except the task definition and the controller role.
# The URNs name the PULUMI RESOURCE KEYS (taskDefinition, controllerRole,
# controllerRolePolicy), not the AWS-side names; divergence 7 has the measurement
# behind that, and `--exclude '*::…'` globs matched nothing when it was tried.
$PUL up \
  --exclude 'urn:pulumi:poc::remote-agents-poc::aws:ecs/taskDefinition:TaskDefinition::taskDefinition' \
  --exclude 'urn:pulumi:poc::remote-agents-poc::aws:iam/role:Role::controllerRole' \
  --exclude 'urn:pulumi:poc::remote-agents-poc::aws:iam/rolePolicy:RolePolicy::controllerRolePolicy'

# Build the image (context zip -> S3, CodeBuild builds and pushes, prints the digest).
BUCKET=$($PUL stack output bucket)
cd image && zip -qr /tmp/context.zip . && cd ..
aws s3 cp /tmp/context.zip "s3://$BUCKET/build/context.zip"
BUILD_ID=$(aws codebuild start-build --project-name lop-poc-image-build \
  --query 'build.id' --output text)
aws codebuild batch-get-builds --ids "$BUILD_ID" --query 'builds[0].{s:buildStatus,d:phases}'
# the log's last line is `PUSHED_DIGEST=<repo>@sha256:...`

# Phase 2 — pin the digest and let the task definition come up.
$PUL config set imageDigest "sha256:<the digested hex>"
$PUL up
```

`--exclude` rather than `--target`: same intent, and it does not need every URN
appearing in the change to be enumerated. Neither excluded resource has dependents,
so no `--exclude-dependents` is needed.

## The model key

The secret exists after phase 1 with **no value**, and ECS cannot start a task whose
task-level secret resolves to nothing — so the five mock runs need a value in it
first. It is a placeholder, and the entrypoint never treats it as a credential:

```sh
# 1. The placeholder, written the same way as the real key: over stdin, never argv.
printf '%s' LOP-POC-PLACEHOLDER-NO-KEY | aws secretsmanager put-secret-value \
    --secret-id lop-poc/model-key --secret-string file:///dev/stdin

# 2. Later, the real key (from the local store, never printed, never an argument):
lop secret get LOP_POC_MODEL_KEY | aws secretsmanager put-secret-value \
    --secret-id lop-poc/model-key --secret-string file:///dev/stdin
```

**How it reaches the agent, and what that does and does not close.** The entrypoint
takes the value out of its own environment, carries it across a re-exec on a FIFO, and
hands it to every child over a file descriptor; `lop_launch.py` reads it there and sets
it with `os.environ[...]` — `setenv` writes the heap copy, while `/proc/PID/environ`
exposes the image the kernel copied at `exec`. So the key is in **no process's initial
environment**: the model's own bash child cannot read it out of its parent with `cat
/proc/$PPID/environ`, which is the path `local_operator/tools/shell_env.py` documents
and the one this POC originally left open. Probe 4e watches every readable
`/proc/<pid>/environ` while the agent runs and reports counts only.

**What it does not close, stated plainly.** The key is in the launcher process's
*memory*, every process in the task is uid 10001, and the watcher records whether a
same-uid non-descendant could open `/proc/<lop-pid>/mem` together with
`yama/ptrace_scope` — that residual is measured per run, not asserted away. The value
also exists in the ECS task definition's secret reference and in the container's
declared environment as the ECS agent sees it, outside the container's `/proc`.
Nothing in this repository, and no process the driver starts, ever reads or prints it.

**TWO CONDITIONS on the "no process's initial environment carries it" claim, both
enforced.** (1) It covers children started under the container's `allowlist`
shell-environment policy — the policy the bash and eval tools build children with, and
reproduced against this image's own `config.yml`. (2) It is conditional on there being
**no `ps` in the image**, because two product spawn sites hand the CALLER's environment
to `ps`: `local_operator/tools/group_reaper.py:229` runs `ps -o lstart= -p <pid>` with
`env={**os.environ, "LC_ALL": "C"}` (from the bash tool's group registration and the
teardown reaper) and `local_operator/memory_guard.py`'s `_default_runner` runs
`ps -axo pid=,pgid=,rss=` with **no `env=` at all**, on every tick of every guarded
command. Either child is a child of the process holding the key, so its own
`/proc/<pid>/environ` would carry it and the model's same-uid bash child could read
that file. Probe **4f** asserts no `ps` — including a `busybox`/`toybox` `ps` — is
reachable at runtime, and the Dockerfile **fails the build** if one ever appears.
4e cannot catch it: a `ps` lasts tens of milliseconds against 4e's one-second samples.
Closing it product-side (a filtered environment at those two sites, or a delivery that
never enters `os.environ`) is deferred product work, recorded in the PR thread.

## Running

```sh
.venv/bin/python scripts/remote_agents_poc.py run --mock --runs 5        # lifecycle only
.venv/bin/python scripts/remote_agents_poc.py status                     # must be empty
.venv/bin/python scripts/remote_agents_poc.py verify <run-dir> \
    --fixture-url https://github.com/olafagbemi/lop-poc-fixture.git --fixture-sha <sha>
.venv/bin/python scripts/remote_agents_poc.py stop-all
```

A real run (`--hosting openrouter --model <model>`) additionally needs the key in
the secret, and its `verify` adds the acceptance-3 session transplant. `--mock`
uses lop's own `test` provider (wire `mock`, `allows_missing_api_key`), so it proves
lifecycle, probes and cold start and never produces a fix — that is expected.

`verify` reports three outcomes, and only a FAIL makes it exit non-zero. **BLOCKED**
is for the checks a mock run cannot answer: acceptance 2 needs a branch, and a run
whose model made no edit has none, while the local key scan needs the real key in the
secret store. On a recorded mock run that leaves **10 PASS** and 2 BLOCKED, and the ten
name what they actually check: the transplanted session is listed by `lop sessions
--all --json` (`state: stored`) AND driven on the real resume path —
`lop exec --resume <id> --hosting test --model test-model --json ping` with stdin from
`/dev/null`, asserting the same session id comes back, the transcript grows 8 → 16
lines, and the first 8 are byte-identical — plus `transcript_non_empty`, all six
container probes (4a, 4b, 4c, 4c-env, 4d, **4f**) and probe 4e's no-key-in-any-process-
environment reading. It is not the TUI's `lop --resume`: that loads the same store
through the same loader, and `exec` is the form a non-TTY driver can run. Both BLOCKED
entries say they need the real key.

The pinned fixture is **https://github.com/olafagbemi/lop-poc-fixture.git at
`69db7e55fc14f918cccdf2fea62894fc37f1f642`** (public, so the container can clone it
with no credential): `calc.add` returns `a - b`, and the prompt is "make the failing
test `test_add` pass". `FIXTURE_SHA_DEFAULT` in the driver carries the same SHA
because a run verified against a different SHA than the report names proves nothing.

Artifacts land in `--out-dir` (default `$LOCAL_OPERATOR_SCRATCHPAD`, else
`./poc-runs`, both gitignored): `run.json`, `describe-tasks.json`, the downloaded
`probes.json`/`results.tar.gz`, and the extracted `results/` (timings, `git.json`,
`status.json`, `exec.jsonl`, `session.tar.gz`).

## Budgets, and a warning that needs reading

Two $25 monthly COST budgets, both notifying `ola.fagbemi@gominerva.com` at ACTUAL
80% and FORECASTED 100%:

* `lop-poc` filters on `user:lop-poc$true`. **This one tracks nothing yet.** This
  account is a LINKED account in organization `o-rxz2wexmo9`, and
  `ListCostAllocationTags` answers `AccessDenied` ("Linked account doesn't have
  access"), so `lop-poc` cannot be activated as a cost-allocation tag here; the
  filter stays inert until the management account activates it.
* `lop-poc-fargate` filters on `Service = Amazon Elastic Container Service` instead,
  which needs no activation and is the working backstop. It is sound only because
  no other Fargate task runs in this account (§9.1 records zero running tasks), and
  that is an assumption to re-check before relying on it.

## Destroy

```sh
$PUL destroy            # S3 forceDestroy + ECR forceDelete + recoveryWindowInDays 0
$PUL stack rm poc
aws resourcegroupstaggingapi get-resources --tag-filters Key=lop-poc,Values=true
```

The last command must return an empty list — that is §9.3 item 7's teardown check,
and it is the reason for the AWS-managed KMS key, the secret's zero recovery window
and `forceDestroy` on the bucket.

## Divergences from the spec and the design doc

Every one of these is a deliberate choice; each names what would have happened
otherwise.

1. **Mock runs need a placeholder secret value (the one that would have broken
   acceptance).** The spec creates `lop-poc/model-key` with no value and expects the
   five mock runs to work against it. ECS cannot start a task whose task-level secret
   resolves to nothing — the task dies with `ResourceInitializationError: unable to
   pull secrets` before the entrypoint runs — so the five mock runs would all have
   been BLOCKED on a key that does not exist yet. The deploy therefore writes a
   PLACEHOLDER value, `LOP-POC-PLACEHOLDER-NO-KEY`, and that value is never treated
   as a key: a mock run uses lop's own `mock` provider. Probe 4c is still a real
   verdict in a mock run — it searches for the EXACT injected value, placeholder
   included, plus the first 8 characters of a real key — and records only counts,
   the value's length and a truncated SHA-256 of it. `pass: null` survives only for
   the case that cannot happen in a deployed task: no value injected at all.
   `4c-env` then checks the other half of the same claim, that a child spawned after
   the unset sees the key in none of its environment entries. Replacing the
   placeholder with the real key is the documented `put-secret-value` command above.
2. **`image/requirements.in` and `image/requirements-probe.in` are extra files.** The
   spec lists only the hash-locked `.txt` outputs. Without the `.in` inputs the
   header uv writes ("autogenerated by uv via the following command") names a
   throwaway path and the lock cannot be regenerated from the repository.
3. **`ephemeralStorage` is OMITTED, not `sizeInGib: 20`.** The provider rejects 20
   (`expected ephemeral_storage.0.size_in_gib to be in the range (21 - 200)`), and
   declaring 21 would ADD a billed gigabyte. Omitting the block yields Fargate's
   20 GiB, which is what "default (20 GiB)" means.
4. **`--exclude` instead of `--target` for phase 1** (Pulumi's `--target` requires
   enumerating every URN in the change; `--exclude` names the two deferred resources).
5. **`imageDigest` is optional with a `""` default** rather than required. That is
   the spec's explicitly-sanctioned alternative ("or make imageDigest optional with a
   two-phase approach you document"); the two phases are above.
6. **The driver never reads the model key.** §9.2 step 3 has it fetch the key from
   Secrets Manager and inject it as a run override; the authoritative resource list
   puts the secret in the task definition, so the driver does neither. No driver
   process holds a model credential at any point.
7. **Probe 4d is a named probe.** The spec's "also record: uname -m, id -u, writes"
   is a probe like the others, so it reports `pass` the same way rather than being
   free text a reader has to interpret.
8. **A failed probe makes the run exit non-zero (5).** "Exit non-zero if any step
   failed" reads to us as including an isolation claim that did not hold. Probe 4c's
   `pass: null` (no key in a mock run) is NOT a failure and does not do this;
   `probes.json["failed"]` is the authoritative list either way.
9. **The provider key was first exported in a subshell — SUPERSEDED BY 21.** The spec's
   example puts the key in a process's argv, where `ps` in the task can read it, so this
   POC's first revision exported it in a subshell before `exec` instead. That is still the
   wrong channel: a subshell export leaves the value in the LAUNCHED process's initial
   environment, which the agent's own bash child reads with `cat /proc/$PPID/environ` —
   the read agent review round 1 raised as SEC-1 and divergence 21 replaced with the
   file-descriptor delivery.
10. **`lop exec` runs with stdin from `/dev/null`.** That is what makes the run
    unattended, which is the condition under which the `--tools read,write,edit,bash`
    declaration stands as the approval for those tools (a tty would re-prompt, and a
    headless run without it would deny every write).
11. **`timings.json` is folded from a `timings.jsonl` in an EXIT trap**, so a run that
    fails AFTER the results tarball is sealed still ships its timings (a probe rc, an
    agent rc). It does not cover the earlier failures: `$OUT` only reaches S3 at step 9,
    so the pre-upload key-scan refusal, a failed `git clone` and a failed probes PUT ship
    no timings at all — the entrypoint's own comment names that list, and this one used to
    over-claim them all.
12. **The driver's `--out-dir` defaults to `$LOCAL_OPERATOR_SCRATCHPAD`, else
    `./poc-runs`.** The spec's `$LOCAL_OPERATOR_SCRATCH` does not exist in this
    harness; `LOCAL_OPERATOR_SCRATCHPAD` is the variable that does.
13. **`git bundle` is created from the range `POC_SHA..lop/<id>`**, with `POC_SHA` as
    the recorded prerequisite, which is what makes `git bundle verify` succeed in a
    fresh clone of the fixture.
14. **`status.json` records the foreground-exec divergence explicitly** (a foreground
    `lop exec` writes no durable job row, so `lop exec --status <JOB_ID>` has nothing
    to read); §9.2 step 7 anticipates it and the file says so in its own `note`.
15. **No stack config file is committed at all**, and `Pulumi.*.yaml` is
    gitignored. Pulumi writes `Pulumi.poc.yaml` on the first config write, and what
    it contains BEFORE any value is the passphrase backend's secret-encryption
    `encryptionsalt` — not a credential, but not source either. The one value the
    stack does carry (`imageDigest`) is recorded in the POC report and in
    `pulumi stack output`, so nothing is lost by keeping the file local. This is
    stricter than the spec's "if a stack config file is needed it holds only
    non-secret values".
16. **A non-root container needed the image to declare `VOLUME ["/workspace"]`.**
    §9.2 asks for a non-root user, and the platform makes the naive version
    impossible: a Fargate task volume arrives mounted root-owned, so a container
    started as 10001 can neither create `/workspace` nor chown it, and the task dies
    on its first `mkdir` (measured). The image now chowns `/workspace` to 10001 and
    then declares `VOLUME ["/workspace"]`, and the task definition sets
    `user: "10001:10001"`: when the VOLUME path equals the volume's `containerPath`,
    ECS copies the image's data AND OWNERSHIP into the mount
    (docs.aws.amazon.com/AmazonECS/latest/developerguide/bind-mounts.html;
    aws/containers-roadmap#938). Verified in the account — probe 4d reads uid 10001
    and writes `/workspace` on all five recorded runs, with no privilege drop
    anywhere in the entrypoint. A first revision instead ran a root phase-0 that
    chowned the volume and re-exec'd itself as 10001; that also worked, and it was
    replaced because the VOLUME pairing is the mechanism the platform documents.
17. **`executionRoleArn` and `taskRoleArn` are set on the task definition; the
    spec's resource list (item 10) names neither field.** ECS refuses the
    registration without them: "When you are specifying container secrets, you must
    also specify a value for 'executionRoleArn'". Without `taskRoleArn` the task
    credentials endpoint has no role to serve, which is the subject of probe 4a — so
    the design's "an execution role and an empty task role" (step 1) only works if
    both are wired here.
18. **The account comes from `fn::invoke` of `getCallerIdentity`, not from a config
    value.** A string config default holding a 12-digit account id reaches an
    interpolation as a FLOAT: `325492156725` became `3.25492156725e+11`, which made
    an illegal S3 bucket name and an "Invalid principal in policy" on the IAM trust
    policy — and `preview` accepted both, because neither is validated until create
    time. The live call cannot be mis-typed, and both the deploy and the driver
    assert `sts get-caller-identity` == 325492156725 immediately before they act.
19. **The buildspec single-quotes every command that contains `": "`, a brace or a
    quote.** CodeBuild's YAML loader reads an unquoted `echo "a: b"` as a MAPPING,
    not a string, and fails the whole phase with `Expected Commands[N] to be of
    string type: found subkeys instead` — measured on the first build. This is a
    trap that a local `yaml.safe_load` gate does not catch unless it asserts every
    command is a `str`; that assertion is worth adding to the repo's buildspec
    checks (not done here: it is outside this slice).
20. **The bucket's SSE configuration is `aws:kms` with NO `kmsMasterKeyId`.** Writing
    `kmsMasterKeyId: aws/s3` previews clean and then fails every `PutObject` with
    `KMS.NotFoundException: Invalid keyId 'aws/s3'` — the AWS-managed key is what
    omitting the key id means. Two further hard-won details are in the commit that
    fixed it: the provider's READ of this resource normalised the bogus key id and
    reported "no diff", so Pulumi could not repair it; and `pulumi up --replace` on
    this singleton sub-resource creates the new config and then DELETES the old one,
    which deletes the bucket's encryption configuration entirely. The fix that
    converged stack and reality was `aws s3api put-bucket-encryption` with the
    intended rule, after which `pulumi preview` reports the resource unchanged.
21. **The key reaches the agent over a file descriptor, through a launcher
    (`image/lop_launch.py`), not through the environment** — the fix for SEC-1 of
    agent review round 1. The entrypoint carries it across a re-exec on a FIFO and
    hands it to each child on fd 3; the launcher reads it there and sets it in-process.
    The alternative that was rejected first (exporting it in a subshell, which this
    POC shipped and which is what a naive reading of "move it out of the environment"
    produces) leaves it in the `lop` process's LAUNCH environment, where the model's
    own bash child reads it with `cat /proc/$PPID/environ` — this repository's own
    `tools/shell_env.py` documents exactly that read. The residual (the key in memory,
    same uid) is stated in "The model key" above and measured per run by probe 4e.
22. **`linuxParameters.initProcessEnabled` is NOT set, though the spec asked for it.**
    With the ECS init shim as PID 1, PID 1 is a process the entrypoint cannot re-exec,
    so its `/proc/1/environ` keeps carrying the key for the life of the task and the
    agent can read it — which would make probe 4e RED, correctly. The entrypoint is
    PID 1 instead and re-execs itself once after taking the key out of its environment.
    The cost is the shim's zombie reaping; the agent reaps its own children and the
    task is 2 h bounded.
23. **The VPC's default security group is adopted and left rule-less**
    (`aws:ec2/defaultSecurityGroup`). AWS creates one per VPC with an allow-all
    self-ingress rule, and anything later added to this VPC without an explicit
    security group inherits it. It is the one pre-existing resource the stack manages,
    it lives inside the VPC the stack created, and AWS does not let it be deleted.
24. **The entrypoint carries a guarded watcher self-test** (`POC_ENVIRON_WATCH_SELFTEST=1`,
    never set by the driver). It launches a child the OLD way — key exported into its
    environment — and runs probe 4e against it, so the probe's RED case can be produced
    in the real container on demand. A probe that can only ever be green is not
    evidence, and the run's exit code (1) is the proof; the reading is recorded in the
    results doc.
25. **The controller role is broader than §9.2's list, and each extra action is
    named here with the driver function that needs it** (SEC-3 of the security review):
    `ecs:ListTasks` — `cmd_status` and `cmd_stop_all`; `ecs:TagResource` (gated on
    `ecs:CreateAction: RunTask`, scoped to `task/lop-poc/*`) — the run-id tags on
    `run_task`; `s3:GetObject` on `runs/*` — `download_artifacts`; and
    `logs:GetLogEvents`/`logs:FilterLogEvents` on the agent log group — the driver's
    log tail. §9.2 says "only RunTask/StopTask/DescribeTasks, PassRole, PutObject", so
    the four are a delta against the spec rather than against the code.
26. **The task has a public IPv4** (`mapPublicIpOnLaunch: true` on both subnets,
    `assignPublicIp: ENABLED` on `RunTask`), because there is no NAT gateway and no VPC
    endpoints — the POC's own cost decision. The address is not reachable: the task SG
    has `Ingress: []` (verified live). What 443-to-anywhere still permits is
    **exfiltration over HTTPS to any host**, and name resolution through the VPC
    resolver (probe 4b's `1.1.1.1:53` failure proves only that *that* resolver is
    unreachable). The empty task role bounds the VALUE of a leak, not the ability to
    make one; §7.2's allowlist proxy is the v1 control.
27. **The CodeBuild builder image is pinned by TAG (`…-standard:3.0`), the one build
    input that is not digest-pinned.** AWS publishes no per-region digest for its
    managed CodeBuild images, so a tag is the only pin available; every other base in
    this POC (the product image, both Dockerfile bases, the task definition's image)
    is digest-pinned. Recorded rather than hidden.
28. **Probe 4b dials two third-party hosts** (`portquiz.net:8080`, `1.1.1.1:53`). They
    are negative controls — the probe needs addresses that must be UNREACHABLE — and
    they cost one DNS lookup plus a refused connection each. Named here so the next
    reader does not have to wonder why the POC touches a public service.
29. **The watcher requires procfs and reports BLOCKED without it.** A Darwin fallback
    through `ps -Eww -ax` was written, measured, and removed: on this host `ps -E`
    printed only ARGV, so a key in some process's command line matched (false positive)
    while a child started with the key in its environment did not (false negative).
    The watcher's verdict is therefore Linux-only by construction, and the unit test
    stubs the environ source for the logic cases.
30. **Probe 4f, and the Dockerfile guard behind it.** The claim "no process's initial
    environment carries the key" is conditional on the image shipping **no `ps`**,
    because two product spawns hand the CALLER's environment to it
    (`tools/group_reaper.py:229`, `memory_guard._default_runner`). 4f asserts no `ps` is
    reachable (PATH, the six knowable paths, and a `busybox`/`toybox` `ps`), the
    Dockerfile fails the build if one ever appears, and both README and results doc
    state the condition instead of the unconditional claim. Closing it product-side is
    deferred work: this POC runs the released `local-operator==0.68.3` wheel and patches
    no product code.
31. **The watcher self-tests come in two modes, and both upload their artifacts.**
    `POC_ENVIRON_WATCH_SELFTEST=1` launches a child the OLD way (the red case);
    `POC_ENVIRON_WATCH_COEXIST=1` has the real launcher hold the key and spawn a bash
    child through `shell_env.child_environment` while 4e samples (the coexistence case,
    which the five mock runs cannot cover because a mock agent spawns no tool child).
    Neither is ever set by the driver, and the branch tars `$OUT` and PUTs it before
    exiting, so a self-test reading lives in the artifact set rather than only in
    CloudWatch.
32. **`report` renders the evidence tables from the artifacts.** Every per-run cell, the
    summary row and the digest population are read from `run.json` /
    `describe-tasks.json` by one subcommand, because the first version of the cold-start
    table was typed by hand and three of its fifteen cells were values that occur
    nowhere in the records (agent review round 2, finding 1).


