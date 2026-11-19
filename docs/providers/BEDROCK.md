# Amazon Bedrock

Provider id: `bedrock`. Model ids: `bedrock/<inference-profile-or-model-id>`,
for example `bedrock/global.anthropic.claude-opus-5-5` or
`bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`.

Local Operator talks to Claude on Bedrock with the **Anthropic Messages API
over `InvokeModelWithResponseStream`** (`POST /model/{id}/invoke-with-response-stream`,
AWS event-stream framing). The request body is the same one the first-party
Anthropic client builds, so streaming, tool use, adaptive thinking/effort,
prompt caching (including the 1-hour TTL on models that support it) and
cache-aware usage and cost all work as they do on Anthropic.

Other text/chat models use **native Converse / ConverseStream** on the same
connection. No OpenAI loopback bridge is required. Streaming text and client
tool loops share the normal harness path. Llama 3.3 tool requests use JSON
`Converse` because its tool use and streaming cannot be combined. Unknown
streaming models may fall back once, before any output, only when AWS explicitly
rejects streaming; validation/auth errors are never retried as capability guesses.

Native reasoning (including opaque signatures/redacted data) is preserved, but
replayed only for the same model, endpoint, credential and unchanged preceding
conversation. Unsupported image placements fail explicitly; unsupported native
documents use the existing retained-byte extraction/reference policy. Nova
supports user images/documents and image tool results; document tool results are
not inferred from ordinary document input support. Audio/video/generation APIs
are not added by this chat transport.

Code: `local_operator/providers/bedrock/`. The design record lives with the
pull request that added it.

## Authentication

Every method reduces to one of two wire schemes: `Authorization: Bearer <Bedrock API key>`,
or AWS SigV4 (service `bedrock`) over credentials resolved by botocore.

| Method | How to use it | Wire |
| --- | --- | --- |
| Bedrock API key | `lop login bedrock` and paste the key, or `PUT /v1/auth/providers/bedrock/key` | Bearer |
| `AWS_BEARER_TOKEN_BEDROCK` | Set the variable | Bearer |
| AWS profile (static, SSO, `aws login`, assume-role, `credential_process`) | Stored connection `bedrock://profile?name=<profile>&region=<region>` (the interactive picker arrives in a follow-up) | SigV4 |
| IAM access keys | Stored connection; the key pair is sealed in the encrypted store | SigV4 |
| Default credential chain (IAM role, ECS/AgentCore, Lambda, EKS web identity, IMDS) | `LOP_BEDROCK_AUTH=ambient`, or `providers.bedrock.auth: ambient`, or automatic on AWS compute | SigV4 |

Secrets never live in `auth.db`. A stored connection's key is a non-secret
descriptor (`bedrock://api-key?region=…&ref=BEDROCK_SECRET_<hex>`); the secret
behind `ref` sits in the encrypted store. A raw API key saved through the
generic key route is moved into the store the same way when it is saved.

### When does Bedrock turn on by itself?

Only on an explicit signal, first match wins:

1. `AWS_BEARER_TOKEN_BEDROCK` is set.
2. `LOP_BEDROCK_AUTH=ambient`, or `providers.bedrock.auth: ambient` in `config.yml`.
3. An AWS-compute marker is present: `AWS_CONTAINER_CREDENTIALS_FULL_URI`,
   `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`, `AWS_WEB_IDENTITY_TOKEN_FILE`,
   `AWS_LAMBDA_FUNCTION_NAME` or `ECS_CONTAINER_METADATA_URI_V4`.

A plain `AWS_PROFILE` or `AWS_ACCESS_KEY_ID` in a laptop shell does **not**
enable Bedrock, since those usually belong to unrelated work. A stored
connection always outranks the environment.

An explicit profile connection ignores ambient `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`.

Local Operator never runs `aws` or prompts during a turn. An expired SSO or
`aws login` session fails before any request is sent, with the command to run
(`aws sso login --profile <name>` or `aws login --profile <name>`). Such a
failure does not block the stored connection: credentials are looked up again
on every attempt, so the next turn after you sign in works straight away. A
network failure while looking up credentials (SSO portal, STS or instance
metadata unreachable) is treated as temporary and retried; when connecting, the
connection is saved unverified instead of refused.

## Region

A stored connection keeps the Region chosen when it was saved. For the
environment and ambient tiers, the Region comes from, in order:
`providers.bedrock.region` → `AWS_REGION` → `AWS_DEFAULT_REGION` → the profile's
`region` → `us-east-1`. Inference, discovery and validation all use the same rule.

Two connections in two Regions are two accounts, so a throttled Region rotates
to the other one.

## Headless workers (`lop exec`, SDK, AgentCore)

All configuration comes from the environment:

```sh
export AWS_REGION=us-east-1
export LOP_BEDROCK_AUTH=ambient        # or rely on the container/web-identity markers
lop exec --hosting bedrock --model us.anthropic.claude-sonnet-5-5 "…"
```

SDK: `SessionSpec(hosting="bedrock", model="us.anthropic.claude-sonnet-5-5")`.
No SDK code changes are needed. Non-Claude workers use that same native path:

```python
from pathlib import Path
from local_operator.sdk import SessionRoots, SessionSpec, open_session

async def worker():
    # Dedicated durable worker roots, not the operator's ambient store.
    home = Path("/var/lib/lop-worker")
    roots = SessionRoots(config_dir=home / "config", agent_home=home / "agent", cwd=home / "work")
    spec = SessionSpec(hosting="bedrock", model="us.amazon.nova-2-lite-v1:0")
    async with open_session(spec, roots=roots) as session:
        return await session.prompt("Summarize this task.")
```

Choose an inference-profile ID actually listed for your source Region; this
example is a selector, not a claim of regional availability. Qwen3 Coder Next
(`qwen.qwen3-coder-next`) is an active text/tool seed. Nova Pro is another
multimodal seed. Legacy Llama 3.3 and DeepSeek R1 remain directly selectable;
R1 has reasoning output but no invented configurable effort/tool support.

Minimum IAM actions:

- `bedrock:InvokeModelWithResponseStream` (streaming inference)
- `bedrock:InvokeModel` (Converse JSON, including Llama tools and capability fallback)
- `bedrock:ListInferenceProfiles` and `bedrock:ListFoundationModels` (optional,
  for discovery; without them the built-in model list is used)
- Bedrock API keys also need `bedrock:CallWithBearerToken`

## Models and discovery

Discovery lists ACTIVE TEXT models from all vendors, on-demand foundation IDs
and paginated system/application inference profiles. Backing foundation metadata
supplies stated modalities/streaming and known model capabilities; an opaque
application profile does not become Claude by substring matching. Which profiles
exist depends on the source Region; routing prefixes are never invented from a
listing failure. Listings are cached per Region and connection. A complete live
inventory limits the picker; a denied/partial listing retains static fallback
rather than claiming absent model entitlement. Listing requires its own IAM
permissions, including for bearer API keys; list denial is not bad credentials.

Built-in rows cover Opus 5.5, Sonnet 5.5, Opus 5, Sonnet 5, Fable 5, Opus 4.8,
Opus 4.7, Opus 4.6, Sonnet 4.6, Opus 4.5 and Haiku 4.5 under every prefix,
with the first-party context windows. Models found by discovery but not built
in get a model-aware fallback: known Claude families retain their limits, while
unknown non-Claude models do not inherit Claude windows, images or caching.
Non-Claude static windows/capabilities are transcribed from the AWS model cards
cited in `providers/bedrock/models.py`; no synthetic regional profiles are added.

### Pricing

**Claude rows only** are built from the first-party prices, checked against the AWS Price List
(`AmazonBedrockFoundationModels`, us-east-1, publication 2026-11-09, read 2026-11-18):

- `global.` profiles are billed at the first-party Anthropic rate.
- Geographic profiles (`us.`, `eu.` …) and in-Region ids cost **1.10×**.
- Sonnet 5.5's cache read on Bedrock is $0.10/MTok global ($0.11 regional).
- The 1-hour cache write is 2× base input, only for the positively known Claude tariff.

Other-vendor Bedrock rates remain **unknown**, not zero/free and not a Claude
1.10× multiplier. Provider receipts do not contain a dollar cost. Converse input
accounting is disjoint: context = noncached input + cache read + cache write;
TTL detail subdivides writes and is not added again. Missing counts remain
incomplete rather than reported zero. Nova emits supported system/frozen-prefix
cache points (5m, without the TTL field), never tools points or checkpoints beyond
its documented 20K-token maximum. Nova 2 Lite exposes none/low/medium/high effort;
high excludes sampling parameters as required by Nova's API.

### Offline validation

`scripts/bedrock_fake_server.py` serves native Converse JSON and CRC-framed
ConverseStream as well as the unchanged Claude InvokeModel fixture.
`scripts/bedrock_converse_smoke.py` drives real SDK and `lop exec` tool loops for
Nova and Llama with isolated HOME/config, synthetic bearer/SigV4, a fake `aws`
negative control and a real harmless tool side effect. This is protocol evidence,
**not live AWS entitlement, cache-hit, signature-echo or modality validation**.

## Errors

| AWS error | What you see |
| --- | --- |
| `ThrottlingException`, `ServiceQuotaExceededException` | quota (retryable, rotates connections) |
| `ServiceUnavailable`, `InternalServer`, `ModelNotReady`, `modelStreamErrorException` | transient (retryable) |
| `ModelTimeoutException` | timeout (retryable) |
| `ValidationException` "input is too long" | classified as a context-length incident; compaction applies as for Anthropic (gated by the shared token estimate, not by this error) |
| `ResourceNotFoundException` | the model or profile is not in that Region; try `global.` or a geographic prefix |
| `AccessDeniedException` "access to the model" | enable model access in the Bedrock console. The credential stays usable for other models |
| `AccessDeniedException` (IAM) naming one `foundation-model/`, `inference-profile/` or `application-inference-profile/` resource ARN | per-model request error, not an auth failure: the message names the IAM action and resource your AWS admin has to grant (often a cross-Region profile routing to a Region the policy omits, or an identity with no Bedrock permission at all). The credential stays usable for other models |
| `AccessDeniedException` (IAM, no model resource), `UnrecognizedClient`, `InvalidSignature`, `ExpiredToken` | authentication error |

Per-model errors (`ResourceNotFoundException` and both per-model
`AccessDeniedException` rows) on the model you chose stop the turn with that
model's error; they do not move on to your fallback models. They depend on the
model, not on the request, so a fallback might work, but the error you need to
act on is the one for the model you picked. On a fallback model they move on to
the next fallback as usual.

## Testing without AWS

`scripts/bedrock_fake_server.py` is a local fake Bedrock endpoint. It speaks
event-stream framing and serves the discovery endpoints. Point the provider
at it with the **test-only** variable `LOP_BEDROCK_ENDPOINT_URL`:

```sh
.venv/bin/python scripts/bedrock_fake_server.py smoke              # runs lop exec twice and checks the result
.venv/bin/python scripts/bedrock_fake_server.py serve --port 8765 --log /tmp/br.jsonl
LOP_BEDROCK_ENDPOINT_URL=http://127.0.0.1:8765 AWS_BEARER_TOKEN_BEDROCK=fake \
  lop exec --hosting bedrock --model us.anthropic.claude-haiku-4-5-20251001-v1:0 "hi"
```

Model ids containing `denied`, `missing` or `throttle` return the matching AWS
error. A prompt containing `MIDSTREAM` sends an exception frame mid-stream.
`LOP_BEDROCK_ENDPOINT_URL` is not a supported production setting: VPC endpoint
URLs are a non-goal for this version.

## Not in this version

Non-Claude models (Converse), application inference profile ARNs, provisioned
throughput, guardrails, service tiers, VPC endpoint URLs, batch inference, and
usage-panel quotas (Bedrock has no quota endpoint).

The interactive AWS-profile/SSO/access-key sign-in surfaces (TUI, CLI and
desktop) arrive in a follow-up. The runtime already accepts every stored
connection method.
