# Financial17: original-corpus private-API feasibility — source-only decision

Source base: `003efac080484cc0ecc6d2b409303aa85d85ca7e`. This is a diagnostic,
not permission to expose any financial document, change routing/capture/policy,
run a task, or freeze an amendment. No private document body was reopened here.
See `FINANCIAL-DATAFLOW-AUDIT-v1.json` for source hashes, seven public-policy-text
receipt hashes/times and offline actual-source probes. Parent's four-input
approval is recorded separately in `PRIVATE-INPUT-USE-APPROVAL-v1.md`.

## Decision: prefer unchanged originals, conditional on a qualified private path

The public upstream benchmark contribution context and intended LLM evaluation
provide a practical basis for proposing restricted evaluation of the original
corpus. They do not establish that every third-party document is unrestricted,
that its contents are synthetic, or that an arbitrary API disclosure is allowed.
Do NOT substitute or redact documents silently. Treat all17 as restricted inputs
under one common data-handling policy across arms. The13 inventory leads remain
uncertain: number/checksum shapes and FPDF/ReportLab metadata do not prove PII,
non-PII or synthetic authorship. The one explicit distribution-restriction marker
needs a targeted private scope/context disposition; a marker alone is not a
finding of a prohibition on this evaluation. Do not require bespoke permission
for every opaque item merely because a notice/inventory audit is unfinished.

**The current code/account evidence does NOT qualify the desired path.** Prefer a
separately reviewed privacy-control/environment slice and artifact disposition,
then explicit parent approval for private processing of these17 originals. This
is feasible in principle under an applicable commercial processing agreement,
no-training controls and strictly private logs, not a legal opinion or current
clearance. No API exposure is authorized by this report.

If the pinned model/provider plus auxiliary classifier cannot meet the agreed
privacy policy, or a concrete document-use restriction cannot be resolved, take
the bounded fallback: propose a versioned synthetic17-document environment/task
amendment BEFORE outcomes, with deterministic generator, documented rights,
format/OCR/field-difficulty coverage and regenerated independently checked oracle.
Preserve the28-task selection unless a separate sample amendment is approved;
do not call new inputs exact recovery or compare them as unchanged historical
fixtures. Do not keep collecting speculative chains of title or burn the final
native attempt to answer a provider-policy question.

## Actual data flow and observed boundaries

| Boundary | Source / finding | Consequence |
|---|---|---|
| Documents → agent tools → conversation | Financial inputs remain private task files; reading/OCR/extraction can return their text/values as tool results. `coding/codex_bridge.py:120–166` preserves tool arguments/results; `meter_proxy.py:925–948,978–1020` accepts and forwards the conversation. Its text-only pricing rule does not remove sensitive text. | Local OCR is not a promise of local-only processing. Original bytes need not leave for document contents to leave. |
| Main/helper model routing | `meter_proxy.py:52–58,1001–1020`: default upstream OpenRouter; configurable base. Provider object is overwritten with Anthropic order, no fallback and price bounds. | Offline execution of the actual assignment confirmed caller `zdr`/`data_collection` preferences are discarded. Account guardrails MAY still constrain routing, but no account state was inspected or verified. Provider pinning is not privacy enforcement. |
| Raw meter body capture | `meter_proxy.py:58,979–987`; `ADAPTERS.md:77–79`: explicit opt-in stores first and last full request JSON, the latter a transcript. | Default opt-in flag is off in source, NOT proof of the launch environment. Explicitly set `BENCH_CAPTURE_BODIES=0` in a reviewed launch unless essential private evidence retention is separately approved. File modes currently depend on caller umask. |
| Meter metadata/error records | `meter_proxy.py:879–896,1120–1125`: no raw successful response bodies in normal usage rows, but upstream error type/code are arbitrary strings. | Private error logs must still be treated as possibly containing content. No raw error field is safe merely because it is named metadata. |
| LoP auxiliary classifier | `lop_edge.py:95,662–682,712–724`: raw decision JSON goes to `https://api.radienthq.com/v1/decisions`, outside the OpenRouter model route. `radient_meter/classifier.py:83–112` permits free-text criteria and does not sanitize arbitrary request contents before the unchanged body is forwarded. | No evidence that current financial contents actually reached this route; it is a genuine reachable content path. Radient/Typesafe processing, retention, training, subprocessors and server traces need their OWN policy evidence. OpenRouter/Anthropic settings do not cover it. Edge local records use body digests, not an external-service privacy guarantee. |
| Local run/harness logs | `coding/driver.py:1277–1280`: all Harbor-process stdout/stderr captured to `<opaque-run-id>.log`; Harbor job/agent artifacts also require a source-version-specific inventory. | No content-redaction guarantee on this stream. Keep logs, transcripts, tool outputs, generated JSON/CSV and verifier artifacts private. Native CLI telemetry/debug/crash uploads and bridge tracing are not proven disabled for all seven arms. |
| Retained wire evidence | `coding/driver.py:594–665` requires verified, matching tool-wire attestation and observed tools/routes/tier. | Do not quietly drop scientific admission evidence to claim no bodies. Use synthetic offline canaries for privacy/wire qualification; if real raw evidence is necessary, approve its private access/lifetime or a reviewed digest-only evidence amendment first. |
| Egress controls | `coding/egress.py`, `package_egress.py:149–159,230–250`, host gate/edge architecture. Package logs record connection metadata, not headers/paths; request bodies still traverse allowed routes. | Existing network fences are useful, not proof of provider retention or information-flow noninterference. Verify no telemetry, external observability, uploads or unintended proxy before dispatch. HTTP client's default environment/proxy behavior also needs a sanitized launch envelope. |
| Public metadata export | `publication.py:230–250` projects known fields, drops unknown usage/body fields and classifies upstream error strings; nested types fail closed. | Offline sentinel was removed from body/messages/unknown usage and raw error became a digest classifier; malformed token text was rejected. This demonstrates the helper, not every future export. |
| ZIP creation | `publication.py:253–307,315–341,375–399`: recursive root inventory, blocked bodies/auth classes, text heuristics, explicit checksum manifest, tracked-file requirement and ZIP packaging. PNG contents are not text-scanned. | Manifest/checksum binding does not itself select a privacy-safe file set. Private output copied to an innocuous filename or an image is not guaranteed detected. Keep quarantine/run roots outside the kit; use an explicit reviewed metadata-only export list, never recursive raw output copying. Do not generate/publish a ZIP in this phase. |

## Required provider/privacy configuration — proposed, NOT applied or verified

1. Dedicated benchmark commercial API identity; not a consumer chat product,
   shared session or feedback/bug-report channel. No key value in evidence.
2. OpenRouter **OpenRouter Use of Inputs/Outputs / data-discount sharing OFF**.
   **Input & Output Logging OFF** or an admin-locked exclusion of the benchmark
   key, with no include overriding that exclusion. Exclusion precedence is
   documented. **Broadcast/external observability body export OFF**; no public
   traces, dashboards, crash bodies or feedback uploads.
3. No-training filter for paid/free routes where applicable, plus a documented
   stricter retention policy: propose final-wire `provider.data_collection="deny"`
   AND `provider.zdr=true`, approved provider allowlist and no fallback. They are
   NOT interchangeable guarantees. Current code lacks both and replaces caller
   preferences; future implementation/account enforcement needs separate review.
   Dedicated-key/account guardrails must enforce relevant model groups, including
   `enforce_zdr_anthropic` for Sonnet/Haiku; exclude Safety Retention rather than
   accepting it silently. No settings or key state were accessed here.
4. **Concrete compatibility blocker:** the current official ZDR page says
   Anthropic-group ZDR removes first-party Anthropic endpoints while Bedrock and
   Vertex remain available. Therefore do not promise the existing pinned
   `order=["anthropic"]` will be runnable under strict ZDR. Verify exact endpoint
   eligibility/contract without sending documents; fail closed if none qualifies.
   A different endpoint or direct-provider/BYOK agreement is a prospective
   common environment/routing amendment needing identity, billing and wire
   qualification, NOT a silent replacement. Do not invent an endpoint slug.
5. Anthropic commercial API no-training is the default, except explicit feedback
   or opt-in. Its default API input/output retention is up to30days with longer
   legal/abuse exceptions; no-training is NOT ZDR. Obtain applicable endpoint
   terms/processing agreement if using that route; a direct Anthropic agreement
   does not automatically establish the OpenRouter shared endpoint's terms.
6. Separately establish Radient/Typesafe classifier no-training, request/log
   retention and approved processing chain, including free-text inputs and
   server-side traces. Do not assume it from a client-side digest or disable the
   classifier just to pass privacy: that would change the tested harness.
7. Record honest limits: router request/cost metadata persists; current router
   policy allows in-flight CSAM screening, flagged-content/legal preservation,
   and anonymous prompt categorization through a ZDR model even when ordinary
   data-sharing is off. In-memory prompt caching may count as ZDR under router
   policy. These are disclosed service behaviors, not literally zero processing
   or an unlimited no-retention guarantee. If unacceptable, require a different
   reviewed processing contract/path. Do not change caching arm-by-arm or label
   it disabled without proof.

## LoP recommendations and all-arm external MCP closure

The native Radient requirement is not satisfied merely by privacy-qualifying
OpenRouter. In this pinned source, `lop_edge.py:95–100,589–595` explicitly permits
model messages and Radient decisions, not a separate recommendation URL. The
classifier route carries free text; `coding/driver.py:712–718` expressly rejects
disabling Radient as a treatment change. Enumerate the chosen native version's
Radient recommendation/recs functionality as well as decisions: exact endpoints,
actual payload categories, processor/training/retention policy and metering.
No recommendation request was made or proven here. If a required recs route is
absent or refused, that is a qualification gap—not evidence the feature can be
silently turned off or bypass the edge. A prospective route change requires
separate review and all-arm fairness/identity proof before outcomes.

For ALL seven harnesses enumerate MCP servers, plugins, search/fetch services,
connector calls, tool child network requests, telemetry, crash reporting and
remote observability, including server-side downstream services. `coding/adapters.py`
documents host instructions/MCP/credentials isolation; the driver also binds
public MCP schemas at `coding/driver.py:647`. Neither schemas nor source intent
alone prove live destinations or absence of raw argument forwarding. Current
fences may refuse unlisted routes, which protects data but can invalidate a
required treatment; refusal is not a successful benchmark run. Use local
synthetic mock tools/canaries plus source-version destination/payload inventories
and deny-unapproved-route controls, with separately reviewed provider terms for
any required external tool. Do not attach real host email, storage, support or
other live MCP credentials to make a benchmark tool work. No external MCP/API
or classifier/recommendation request is authorized by this audit.

## Restricted local storage and admission proof

Use registered private HOLD roots (0700 directories,0600 files, reviewed
`umask077`) for working documents, extracted text, outputs and any approved
private transcript. Opaque campaign/run IDs only; no financial data in filenames,
URLs, request metadata, environment dumps, exception text or operator-visible
stdout. Restrict backups/sync/telemetry and define access plus eventual release
with the parent; no deletion is authorized here. Minimal public rows are
counts/status/usage/digests, not content, prompt samples or screenshots.

Before any real API exposure, require one coherent reviewed admission packet:

- Specific artifact/notice/privacy-lead disposition and explicit parent
  financial17-use approval (currently ONLY four other inputs have that approval).
- Exact approved model/helper/classifier processors, no-training/retention policy,
  account/key setting attestation without keys, and a resolved ZDR compatibility
  decision. Unknown policy is not evidence of passing.
- Approved source changes if needed; offline synthetic canaries through final
  request assignment, classifier route, logs/errors, all seven native harness
  output sinks and explicit metadata export. No financial content in canaries.
  Include negative controls for capture enabled, nested error content, fallback,
  absent privacy enforcement and accidental raw/image export. Account guardrails
  must not be assumed from a body fixture alone.
- Then normal runtime qualification, fresh resource admission and separate spend
  authorization; do not conflate a privacy decision with permission to execute.

The bounded alternative is a proposed synthetic corpus/oracle amendment if this
packet cannot be met—not automatic original-data exposure, not endless blanket
licensing HOLD. No task, sample, oracle, model, routing, logging or policy was
changed by this diagnostic. Source-only probes are NOT live privacy or runtime
proof; no inference, account API, credentials, app/VM/helper startup, install or
paid call was used. Operator ask `a-9509` / relay `a-1ca7` remains OPEN; native2/3.

## Public policy evidence

Seven exact public-text fetch receipts are in the companion JSON (time, byte
count, SHA256 and final URL); full web pages are not duplicated in the repository.
Read the current policy at qualification because this evidence describes the
fetch date, not an immutable service agreement:

- https://openrouter.ai/docs/guides/features/zdr
- https://openrouter.ai/docs/guides/privacy/data-collection
- https://openrouter.ai/docs/guides/features/input-output-logging
- https://openrouter.ai/docs/guides/routing/provider-selection
- https://openrouter.ai/docs/guides/privacy/provider-logging
- https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training
- https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data
