# Local Operator on Amazon Bedrock AgentCore Runtime

Run [Local Operator](../../README.md) (`lop`) headless on
[Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html):
one Firecracker microVM per conversation, models on Amazon Bedrock signed with
the runtime's IAM role (SigV4), no API keys anywhere, nothing to keep running
between requests.

This sample follows the shape of the AWS sample
[Host Hermes Agent on Amazon Bedrock AgentCore](https://github.com/aws-samples/sample-host-hermesagent-on-amazon-bedrock-agentcore)
and keeps its core idea: an unmodified agent harness inside an AgentCore
container, with Bedrock reached through role credentials. It leaves out that
sample's messaging channels (Telegram, Slack, WeChat/Feishu, the ECS gateway,
cron and the router); see [Next steps](#limitations-and-next-steps).

## Why AgentCore for lop

- **A microVM per session.** Each `runtimeSessionId` gets its own isolated
  microVM with its own filesystem and process tree. The agent can write files
  and run shell commands, and the next conversation never sees them.
- **SigV4, not secrets.** The container holds no model API key. The only
  credential is the execution role, and it can invoke just the model you deploy.
- **Serverless.** There are no instances to manage. A session's microVM is
  discarded after an idle timeout (15 minutes by default).
- **lop's whole harness.** The container runs lop through its embedding SDK
  ([`local_operator.sdk`](../../docs/SDK.md)), the same session machinery that
  `lop exec` uses: tools, compaction, approvals, skills. There is no second
  agent loop.

## Architecture

```
 caller (invoke.py / boto3 / aws cli)
   │  InvokeAgentRuntime(runtimeSessionId, {"prompt": ...})   SigV4 (caller's IAM)
   ▼
 ┌────────────────────── AgentCore Runtime ──────────────────────┐
 │  microVM for session A          microVM for session B   ...   │
 │  ┌──────────────────────────────────────────────┐             │
 │  │ main.py  BedrockAgentCoreApp :8080           │             │
 │  │   /ping, /invocations  ──▶ SSE event stream  │             │
 │  │        │                                     │             │
 │  │        ▼                                     │             │
 │  │ lop_runtime.py  one lop session per          │             │
 │  │   runtimeSessionId (local_operator.sdk)      │             │
 │  │        │ hosting "openai-compatible"         │             │
 │  │        ▼                                     │             │
 │  │ bedrock_bridge.py  127.0.0.1 only            │             │
 │  │   OpenAI chat.completions ⇄ Converse         │             │
 │  └────────┼─────────────────────────────────────┘             │
 └───────────┼───────────────────────────────────────────────────┘
             │ ConverseStream, SigV4 with the execution role
             ▼
        Amazon Bedrock (your model / inference profile)
```

### Why there is a bridge

lop has no native Bedrock provider. Its generic `openai-compatible` hosting
talks the OpenAI Chat Completions wire to any URL
([local providers](../../docs/LOCAL_PROVIDERS.md)). Bedrock's own
OpenAI-compatible endpoints accept SigV4, but when this sample was written they
served only open-weight models. A Claude inference profile returned
`404 model_not_found` on both `bedrock-runtime …/openai/v1` and
`bedrock-mantle`. So `app/bedrock_bridge.py` translates Chat Completions,
including streaming and tool calls, to Bedrock **ConverseStream**. The bridge is
about 500 lines of standard library and boto3, with no LiteLLM. Converse is
model-agnostic, so any Bedrock chat model that supports tool use works.

lop is configured without the interactive `/login`: at boot, the entrypoint
writes `config.yml` with `values.providers.openai-compatible.base_url` pointing
at the bridge. It also declares the model's context budget there, because the
bridge cannot report one.

## Layout

| Path | What it is |
| --- | --- |
| `app/Dockerfile` | arm64 Python 3.12 image: `local-operator==${LOP_VERSION}` + `bedrock-agentcore` |
| `app/main.py` | AgentCore entrypoint (`BedrockAgentCoreApp`) |
| `app/lop_runtime.py` | lop session per `runtimeSessionId`, payload parsing, event projection |
| `app/bedrock_bridge.py` | loopback OpenAI-compatible → Bedrock ConverseStream bridge (SigV4) |
| `infra/build.yaml` | CloudFormation: ECR repository, build-context bucket, arm64 CodeBuild project |
| `infra/runtime.yaml` | CloudFormation: least-privilege execution role + `AWS::BedrockAgentCore::Runtime` |
| `scripts/deploy.sh` | build stack → image build in CodeBuild → runtime stack |
| `scripts/invoke.py` | streaming client (`invoke_agent_runtime`) |
| `scripts/teardown.sh` | deletes both stacks |
| `tests/` | offline unit tests (no AWS needed) |

## Prerequisites

- An AWS account and an AWS CLI v2 **profile** that can create CloudFormation
  stacks, IAM roles, ECR, S3, CodeBuild and AgentCore resources.
- A region where AgentCore Runtime is available.
- Access in that region to the Bedrock model you choose. List the inference
  profiles with `aws bedrock list-inference-profiles --profile <profile> --region <region>`.
  Some Anthropic models first require the account's use-case form (see
  [Troubleshooting](#troubleshooting)).
- Python 3.10+ with `boto3`, for `invoke.py`.
- **No local Docker.** The arm64 image is built natively in CodeBuild.

## Quick start

```bash
cd examples/agentcore

# Deploy. --profile is required; --region defaults to the profile's region.
scripts/deploy.sh --profile <your-profile> --region <your-region> --stack-name lop-agentcore

# Talk to it. The session id is printed, so you can continue the conversation.
python3 scripts/invoke.py --profile <your-profile> --region <your-region> \
  --stack-name lop-agentcore "What can you do?"

# Clean up.
scripts/teardown.sh --profile <your-profile> --region <your-region> --stack-name lop-agentcore
```

`deploy.sh` is idempotent. The image tag hashes `app/` and the lop version, so a
re-run with no changes skips the build, and a changed `app/` builds and rolls
out a new runtime version. Measured builds took about one minute.

## Parameters

| `deploy.sh` flag | Default | Meaning |
| --- | --- | --- |
| `--profile` | **required** | AWS CLI profile. There is no fallback to ambient credentials. |
| `--region` | profile's region | Deployment region; required if the profile has none. |
| `--stack-name` | `lop-agentcore` | Base name: stacks `<name>-build` and `<name>-runtime`, ECR repo `<name>`, runtime `<name with _>`. |
| `--model-id` | `global.anthropic.claude-sonnet-5-5` | Bedrock model or system inference profile id. Only this model is granted to the role. |
| `--lop-version` | `0.92.1` | `local-operator` release installed from PyPI. The build asserts the installed version. |
| `--approvals` | `auto` | lop tool-approval policy in the microVM: `auto`, `refuse`, or a comma-separated tool list. See [Security](#security-notes). |
| `--tools` | lop default | Optional comma-separated lop tool allow-list. |

Runtime environment variables, set by `infra/runtime.yaml`: `LOP_MODEL_ID`,
`LOP_APPROVALS`, `LOP_TOOLS`, `LOP_BEDROCK_REGION`. The container also reads
`LOP_CONTEXT_WINDOW` (default `200000`) and `LOP_MAX_OUTPUT_TOKENS` (default
`16000`). Session lifetime is set by the template parameters
`IdleTimeoutSeconds` (900) and `MaxLifetimeSeconds` (28800).

## Invoking

### Payload and response

Request body: `{"prompt": "..."}`. `"message"` is accepted as an alias. An
optional `"model"` picks another Bedrock model for a **new** session, and the
execution role must also allow that model.

The response is `text/event-stream`, one JSON object per `data:` line:

```json
{"type": "session", "session_id": "...", "model": "..."}
{"type": "text", "text": "..."}
{"type": "tool_start", "tool": "bash", "args": {"command": "uname -m"}}
{"type": "tool_end", "tool": "bash", "is_error": false, "result": "..."}
{"type": "end", "error": null}
{"type": "error", "error": "..."}
```

`text` events are assistant deltas. `end` closes a turn, and its `error` is
non-null when the turn failed (for example, the model was denied). `error`
means the invocation itself was refused (for example, an empty prompt).

### `scripts/invoke.py`

```bash
python3 scripts/invoke.py --profile <p> --region <r> --stack-name lop-agentcore "hello"        # new session
python3 scripts/invoke.py --profile <p> --region <r> --session-id <id> "and then?"            # same session
python3 scripts/invoke.py --profile <p> --region <r> --json "hello"                            # raw events
python3 scripts/invoke.py --profile <p> --region <r> --runtime-arn <arn> "hello"               # skip stack lookup
```

AgentCore requires session ids of at least 33 characters. `invoke.py`
generates `lop-<uuid4 hex>` when you don't pass one.

### boto3

```python
import json, uuid, boto3

client = boto3.Session(profile_name="<profile>", region_name="<region>").client("bedrock-agentcore")
response = client.invoke_agent_runtime(
    agentRuntimeArn="<AgentRuntimeArn output of <stack>-runtime>",
    runtimeSessionId=f"lop-{uuid.uuid4().hex}",
    payload=json.dumps({"prompt": "What is 2**20? Use python to check."}).encode(),
)
for line in response["response"].iter_lines(chunk_size=1):  # chunk_size=1: don't buffer events
    if line.startswith(b"data: "):
        print(json.loads(line[6:]))
```

### AWS CLI

```bash
aws bedrock-agentcore invoke-agent-runtime --profile <p> --region <r> \
  --agent-runtime-arn "<AgentRuntimeArn>" \
  --runtime-session-id "lop-$(uuidgen | tr -d -)" \
  --payload "$(printf '%s' '{"prompt":"hello"}' | base64)" \
  response.txt && cat response.txt
```

(AWS CLI v2 expects `--payload` as base64 by default. Use
`--cli-binary-format raw-in-base64-out` to pass raw JSON instead.)

## Transcripts from a real deployment

These are captured from a deployment made with this sample's unmodified
scripts, with `--model-id` left at its default and lop `0.92.1`. Account ids,
ARNs and the profile name are replaced with placeholders. Each turn below took
2–14 s end to end. A new session's first turn includes the microVM cold start.

### Single turn

```text
$ python3 scripts/invoke.py --profile <profile> --region <region> --stack-name lop-agentcore-demo \
    --session-id lop-demo-session-a-89b7cae43a234c10908f8295d126c15e \
    "In one sentence: what are you, and which model are you running on?"
# session-id: lop-demo-session-a-89b7cae43a234c10908f8295d126c15e
[session lop-demo-session-a-89b7cae43a234c10908f8295d126c15e model=global.anthropic.claude-sonnet-5-5]
I'm Local Operator, a personal assistant that runs on your computer and uses tools to act directly
on it, and I'm running on the model `global.anthropic.claude-sonnet-5-5`, served through an
OpenAI-compatible provider.
```

### Multi-turn: the same session remembers

```text
$ python3 scripts/invoke.py ... --session-id lop-demo-session-a-89b7…c15e \
    "Remember this for later: my project codename is BLUE-HERON-42. Just acknowledge."
Noted: your project codename is BLUE-HERON-42.

$ python3 scripts/invoke.py ... --session-id lop-demo-session-a-89b7…c15e \
    "What is my project codename? Answer with just the codename."
BLUE-HERON-42
```

### Tool use inside the microVM

```text
$ python3 scripts/invoke.py ... --session-id lop-demo-session-a-89b7…c15e \
    "Write a Python script primes.py in your working directory that prints the sum of all primes
     below 10000, run it, then run uname -m and cat /etc/os-release | head -2. Also save the text
     'only session A knows this' to note-a.txt. Report the sum, the architecture and the OS."

[tool] write {"path": "/var/lib/lop/work/primes.py", "content": "def primes_below(n):\n    sieve = ..."}
[tool] write {"path": "/var/lib/lop/work/note-a.txt", "content": "only session A knows this"}
[tool ok] Created /var/lib/lop/work/note-a.txt (25 chars).
[tool ok] Created /var/lib/lop/work/primes.py (283 chars).

[tool] bash {"command": "cd /var/lib/lop/work && python3 primes.py && uname -m && cat /etc/os-release | head -2 && cat note-a.txt"}
[tool ok] exit code: 0
--- stdout ---
5736396
aarch64
PRETTY_NAME="Debian GNU/Linux 13 (trixie)"
NAME="Debian GNU/Linux"
only session A knows this
--- stderr ---
(empty)
I wrote and ran `primes.py`, and saved the note to `note-a.txt`. Both files are in `/var/lib/lop/work`.

- **Sum of primes below 10000:** 5736396
- **Architecture:** aarch64
- **OS:** Debian GNU/Linux 13 (trixie)
```

### Session isolation

A different session id gets a different microVM, and session A's files are not
there:

```text
$ python3 scripts/invoke.py ... --session-id lop-demo-session-b-<uuid> \
    "List the files in your working directory (ls -la) and check whether note-a.txt or primes.py
     exists anywhere under /var/lib/lop. Report exactly what you found."

[tool] bash {"command": "ls -la; echo ---; find /var/lib/lop \\( -name 'note-a.txt' -o -name 'primes.py' \\) 2>&1"}
[tool ok] exit code: 0
--- stdout ---
total 12
drwxr-xr-x. 2 lop lop 4096 Nov 13 04:55 .
drwxr-xr-x. 1 lop lop 4096 Nov 13 04:55 ..
---
--- stderr ---
(empty)
- `ls -la` in `/var/lib/lop/work`: The directory is empty. It lists only `.` and `..`, both owned by `lop`.
- Search under `/var/lib/lop`: `find` found no file named `note-a.txt` or `primes.py`.
```

Session A, invoked again afterwards, still has its file:

```text
$ python3 scripts/invoke.py ... --session-id lop-demo-session-a-89b7…c15e "Print the contents of note-a.txt with cat."
[tool] bash {"command": "cat /var/lib/lop/work/note-a.txt"}
[tool ok] exit code: 0
--- stdout ---
only session A knows this
```

### Streaming

The `--json` output below has each line prefixed with seconds since the command
started. The `session` event arrives before the model answers, and text deltas
arrive as the model produces them:

```text
  2.54s {"type": "session", "session_id": "lop-d2feed1f4edf4348b43a9fc0b0e213b2", "model": "global.anthropic.claude-sonnet-5-5"}
 12.71s {"type": "text", "text": "Coding agents execute"}
 12.71s {"type": "text", "text": " untrusted, unpred"}
 12.95s {"type": "text", "text": ", run shell commands, and"}
   …
 13.78s {"type": "end", "error": null}
```

### Errors reach the caller

A model the execution role is not granted. The error names the role and
proves the call was SigV4-signed by it:

```text
$ python3 scripts/invoke.py ... --model global.anthropic.claude-opus-5-5 "hello"
[session lop-53a59d1f96ee45639dfc830debef040d model=global.anthropic.claude-opus-5-5]

[turn error] authentication failed (HTTP 403): User: arn:aws:sts::<account-id>:assumed-role/lop-agentcore-demo-runtime-ExecutionRole-<suffix>/<session>
is not authorized to perform: bedrock:InvokeModelWithResponseStream on resource:
arn:aws:bedrock:<region>:<account-id>:inference-profile/global.anthropic.claude-opus-5-5 because no identity-based policy allows the bedrock:InvokeModelWithResponseStream action
```

A model the account has not enabled. This came from the first deployment,
which used a model that needed Anthropic's use-case form:

```text
[turn error] invalid request (HTTP 404): Model use case details have not been submitted for this account.
Fill out the Anthropic use case details form before using the model. ...
```

A missing stack or runtime, and deploying without a profile:

```text
$ python3 scripts/invoke.py --profile <profile> --region <region> --stack-name does-not-exist hello
error: cannot read stack 'does-not-exist-runtime' in <region>: Stack with id does-not-exist-runtime does not exist (deployed with scripts/deploy.sh?)

$ python3 scripts/invoke.py --profile <profile> --region <region> --runtime-arn arn:aws:bedrock-agentcore:<region>:<account-id>:runtime/lop_does_not_exist-AAAAAAAAAA hello
error: ResourceNotFoundException: No endpoint or agent found with qualifier 'DEFAULT' for agent 'arn:aws:bedrock-agentcore:<region>:<account-id>:runtime/lop_does_not_exist-AAAAAAAAAA'

$ scripts/deploy.sh
error: --profile is required (the AWS CLI profile to deploy with); refusing to use ambient credentials.
```

### Runtime logs: role credentials, SigV4

The log group is `/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT`. The
bridge logs one line per model call:

```text
INFO botocore.credentials Found credentials from IAM Role: execution_role
INFO lop_agentcore.bridge bridge credentials method=iam-role (SigV4; no API key)
INFO lop_agentcore lop runtime ready model=global.anthropic.claude-sonnet-5-5 approvals=auto bridge=http://127.0.0.1:37681/v1 home=/var/lib/lop
INFO lop_agentcore.bridge bedrock converse_stream model=global.anthropic.claude-sonnet-5-5 auth=sigv4 in=22345 out=80 2.1s
WARNING lop_agentcore.bridge bedrock converse_stream failed model=global.anthropic.claude-opus-5-5 code=AccessDeniedException
```

## Security notes

- **The execution role is the only credential in the microVM.** lop's shell
  tool can read it too, through the container credentials endpoint. The
  template therefore grants only `bedrock:InvokeModel` and
  `bedrock:InvokeModelWithResponseStream`, scoped to the configured inference
  profile and its foundation model. It adds pulling this image, writing this
  runtime's logs and metrics, and the runtime's own workload-identity token.
  Grant more (S3, Secrets Manager, …) only to deliberately give the agent that
  reach.
- **Approvals default to `auto`.** A headless microVM has no person to answer
  an approval prompt. Each session runs in a disposable, isolated microVM
  whose only credential is the narrow role above. Use `--approvals refuse` for
  a read-only agent, where writes and shell commands are refused, or
  `--approvals read,grep,glob` for a declared tool set where only the named
  tools exist.
- **Network.** The runtime uses `PUBLIC` network mode, so the agent can reach
  the internet with its tools. To confine it, switch `NetworkConfiguration` to
  VPC mode with your own subnets and security groups.
- **Who can invoke.** Any principal with `bedrock-agentcore:InvokeAgentRuntime`
  on the runtime ARN can drive the agent. Scope that permission, or add an
  `AuthorizerConfiguration` (JWT) to the runtime.
- The bridge binds `127.0.0.1` only, so it is not reachable from outside the
  microVM.
- The image runs as an unprivileged `lop` user. Build contexts sit in a
  private, TLS-only, encrypted bucket and expire after 7 days. ECR tags are
  immutable and scanned on push.

## Cost and cleanup

You pay for:

- **AgentCore Runtime:** per-second CPU and memory while sessions are active.
  A microVM idles out after `IdleTimeoutSeconds`.
- **Bedrock tokens:** lop's system prompt and tool definitions are about 22K
  input tokens per model call (see `in=` in the logs above), and a tool-using
  turn makes several calls.
- **CodeBuild minutes:** about one minute per image build.
- **ECR and S3 storage:** small.

`scripts/teardown.sh --profile <p> --region <r> --stack-name <name>` deletes the
runtime stack, then empties the build-context bucket and deletes the build
stack, ECR images included.

## Troubleshooting

- **`Model use case details have not been submitted for this account`.**
  Submit Anthropic's use-case form in the Bedrock console, or deploy with a
  `--model-id` your account can already use.
- **`AccessDeniedException ... is not authorized to perform: bedrock:InvokeModelWithResponseStream`.**
  The model in the request is not the one deployed. Redeploy with
  `--model-id`, or extend `InvokeConfiguredModel` in `infra/runtime.yaml`.
- **The image build failed.** `deploy.sh` prints the CodeBuild log link. The
  build also checks that the installed `local-operator` version matches
  `--lop-version`.
- **Logs.** `aws logs tail /aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT --follow --profile <p> --region <r>`.

## Limitations and next steps

- **Channels.** The Hermes sample adds Telegram, Slack, Feishu and WeChat
  through an ECS gateway and a router Lambda. The same pattern fits here,
  because a channel only needs to map a chat id to a `runtimeSessionId`.
  lop's own Team Bridge is another way to reach a chat surface.
- **Durable memory.** A session's files and history live only as long as its
  microVM. AgentCore Memory, or an S3/EFS-mounted `LOP_HOME`, would let a
  conversation survive idle timeouts.
- **Tools through AgentCore Gateway.** Expose company APIs as MCP tools through
  AgentCore Gateway and point lop's MCP config at them.
- **Native Bedrock provider.** A first-class Bedrock provider in lop would make
  the bridge unnecessary. Bedrock's OpenAI-compatible endpoints would also do,
  once they serve the model you need.
- **Not translated by the bridge:** extended-thinking blocks, prompt-cache
  points, images in tool results, and `n > 1`.

## Tests

The offline tests need no AWS credentials or network:

```bash
python -m venv .venv && .venv/bin/pip install "local-operator==0.92.1" boto3 pytest
.venv/bin/python -m pytest examples/agentcore/tests -q
```

`examples/agentcore/pytest.ini` makes this directory its own pytest root, so the
repository's suite options do not apply.
