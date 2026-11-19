# Amazon Bedrock (Claude)

Provider id: `bedrock`. Model ids: `bedrock/<inference-profile-or-model-id>`,
for example `bedrock/global.anthropic.claude-opus-5-5` or
`bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`.

Local Operator talks to Claude on Bedrock with the **Anthropic Messages API
over `InvokeModelWithResponseStream`** (`POST /model/{id}/invoke-with-response-stream`,
AWS event-stream framing). The request body is the same one the first-party
Anthropic client builds, so streaming, tool use, adaptive thinking/effort,
prompt caching (including the 1-hour TTL on models that support it) and
cache-aware usage and cost all work as they do on Anthropic.

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
(`aws sso login --profile <name>` or `aws login --profile <name>`).

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
No SDK code changes are needed.

Minimum IAM actions:

- `bedrock:InvokeModelWithResponseStream` (inference)
- `bedrock:ListInferenceProfiles` and `bedrock:ListFoundationModels` (optional,
  for discovery; without them the built-in model list is used)
- Bedrock API keys also need `bedrock:CallWithBearerToken`

## Models and discovery

Discovery lists the ACTIVE system-defined Claude inference profiles your source
Region can call (`global.`, `us.`, `eu.`, `apac.`, `jp.`, `au.` …), plus the
on-demand in-Region Claude models. Which prefixes exist depends on the source
Region; routing is never inferred from the prefix alone. Listings are cached
per Region and connection. If listing fails, the built-in rows are used.

Built-in rows cover Opus 5.5, Sonnet 5.5, Opus 5, Sonnet 5, Fable 5, Opus 4.8,
Opus 4.7, Opus 4.6, Sonnet 4.6, Opus 4.5 and Haiku 4.5 under every prefix,
with the first-party context windows. Models found by discovery but not built
in get the Claude family's limits, and their cost shows as unavailable.

### Pricing

Rows built from the first-party prices, checked against the AWS Price List
(`AmazonBedrockFoundationModels`, us-east-1, publication 2026-11-09, read 2026-11-18):

- `global.` profiles are billed at the first-party Anthropic rate.
- Geographic profiles (`us.`, `eu.` …) and in-Region ids cost **1.10×**.
- Sonnet 5.5's cache read on Bedrock is $0.10/MTok global ($0.11 regional).
- The 1-hour cache write is 2× base input, as on Anthropic.

## Errors

| AWS error | What you see |
| --- | --- |
| `ThrottlingException`, `ServiceQuotaExceededException` | quota (retryable, rotates connections) |
| `ServiceUnavailable`, `InternalServer`, `ModelNotReady`, `modelStreamErrorException` | transient (retryable) |
| `ModelTimeoutException` | timeout (retryable) |
| `ValidationException` "input is too long" | classified as a context-length incident; compaction applies as for Anthropic (gated by the shared token estimate, not by this error) |
| `ResourceNotFoundException` | the model or profile is not in that Region; try `global.` or a geographic prefix |
| `AccessDeniedException` "access to the model" | enable model access in the Bedrock console. The credential stays usable for other models |
| `AccessDeniedException` (IAM) naming one `foundation-model/` or `inference-profile/` resource ARN | per-model request error, not an auth failure: the policy denies that model (often a cross-Region profile routing to a Region the policy omits). The credential stays usable for other models |
| `AccessDeniedException` (IAM, no model resource), `UnrecognizedClient`, `InvalidSignature`, `ExpiredToken` | authentication error |

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
