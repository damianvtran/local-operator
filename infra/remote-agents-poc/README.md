# remote-agents-poc — Slice-0 POC infrastructure

One Fargate task that clones a public fixture repo, runs `lop exec` against it with
no AWS permissions of its own, commits a branch, and uploads its results. This is
**Slice 0** of `docs/design/remote-cloud-agents.md` §9.2: cloud run, no mesh. It
exists to answer "does the AWS lifecycle work, is the isolation what §7.1 claims,
and what does a cold start cost" — not to ship a product feature.

Nothing here is imported by the product. `scripts/remote_agents_poc.py` is a POC
driver, `infra/remote-agents-poc/` is a standalone Pulumi project, and the image is
built in CodeBuild, never on a laptop.

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

**Why the entrypoint has a root phase.** A Fargate task volume is mounted
root-owned, and a container started as uid 10001 cannot write it or chown it. The
first run measured exactly that: `mkdir: cannot create directory '/workspace/out':
Permission denied`, before the entrypoint's second step. So the process starts as
root, chowns `/workspace`, and **re-execs itself as 10001** — every phase that
touches untrusted input (probes, agent, anything the agent spawns) is
unprivileged, and probe 4d's `uid_is_10001` proves it on every run. The image still
creates and chowns `/workspace` (so the CodeBuild smoke test, which runs the image
with no volume, behaves the same). Do not "simplify" this by putting
`USER 10001:10001` back: that is the configuration that cannot run.

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
$PUL up \
  --exclude '*::aws:ecs/taskDefinition:TaskDefinition::lop-poc-agent' \
  --exclude '*::aws:iam/role:Role::lop-poc-controller' \
  --exclude '*::aws:iam/rolePolicy:RolePolicy::lop-poc-controller'

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

The secret exists after phase 1 with **no value**. The task definition injects it as
an ECS secret, so the entrypoint never needs to fetch it. Put the real key with this
exact command (it reads the value over stdin, so it is never an argument, never in
the shell history):

```sh
lop secret get LOP_POC_MODEL_KEY | aws secretsmanager put-secret-value \
    --secret-id lop-poc/model-key --secret-string file:///dev/stdin
```

Until a value exists, only `--mock` runs will work — see "Divergences", the mock-run
note. Nothing in this repository, and no process the driver starts, ever reads or
prints that key.

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
secret store. On a recorded mock run that leaves 6 PASS — acceptance 3 in full (the
session transplants into a fresh config root and `lop sessions --all --json` lists it
as `state: stored` with its transcript) plus all five container probes — and 2
BLOCKED, both of which say they need the real key.

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
9. **The provider key is exported in a subshell, not via `env KEY=… lop exec`.** The
   spec's example puts the key in a process's argv, where `ps` in the task can read
   it; a subshell `export` followed by `exec` keeps it in the environment of exactly
   one process tree.
10. **`lop exec` runs with stdin from `/dev/null`.** That is what makes the run
    unattended, which is the condition under which the `--tools read,write,edit,bash`
    declaration stands as the approval for those tools (a tty would re-prompt, and a
    headless run without it would deny every write).
11. **`timings.json` is folded from a `timings.jsonl` in an EXIT trap**, so a run that
    fails still ships its timings. The timings of a failure are evidence too.
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
16. **The container runs as root for its first few milliseconds, then drops to
    uid 10001 — the task definition has no `user` key.** §9.2 asks for a non-root
    user, and that is the requirement the platform makes impossible: a Fargate task
    volume is mounted root-owned, so a container started as 10001 can neither create
    `/workspace` nor chown it, and the task dies on its first `mkdir` (measured on
    the first real run). The alternative was a writable root filesystem, which
    trades away the read-only-rootfs claim — a worse trade than a root phase that
    only chowns a volume the image ships owned by 10001. `phase 0` in the entrypoint
    is that block, it re-execs the entrypoint as 10001, and probe 4d asserts uid
    10001 afterwards, on every run.
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


