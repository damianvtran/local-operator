# Design: agent image generation (`generate_image`)

Status: implemented in `feat/image-generation-tool` (harness). Base:
`a8938ad8d7` (fresh `origin/main`). Author: architect report (session
`fd9761de35cf`), adapted by the implementing lane, 2026-10-08. **Depends on
lane D's output-attachment contract** (`docs/design-output-attachments.md`) —
the result rides `AttachmentContent` and registers bytes through
`session.attachments.cache_media`; that contract is frozen, this feature adds
no second transport.

**Superseded in part (media wave-2, 2026-10-09):** the kind-neutral interface
the image lane now rides — the walk, the rung seam, provider breadth — is
designed in `docs/design/artifact-generation.md`; read that for the current
shape of the cascade and rungs. This document remains the frozen image v1
design (D1–D12), and its behaviour is still the contract.

## 0. Problem

`local_operator/clients/fal.py` and `local_operator/clients/radient.py` carry
image-generation clients that are **called by nothing**: the old tool
(`generate_image_tool` / `generate_altered_image_tool`) lived in the classic
`ToolRegistry`, constructed from a `CredentialManager` in the old `cli.py` /
`server/utils/operator.py`, and was lost in the rewrite. The modern tree has
every seam the capability needs — `TOOL_BUILDERS` (createIf factories), the
`stt/` cascade as the availability/fail-forward precedent, the attachment
store — and no tool.

## 1. Decisions (the rest of this doc assumes these)

| # | Decision | Why |
|---|---|---|
| D1 | ONE tool `generate_image`; `source_image_path` present ⇒ image-to-image. | The footprint ladder: a second tool duplicates the whole schema; both old tools collapse into one params model. |
| D2 | No video surface in v1. | Divergent params/budgets, no requirement; the attachment pipeline is already video-ready (`kind:"video"`). |
| D3 | **Published**, not deferred; `deferral.py` untouched. Candidate purpose phrase for a future data-driven deferral: `"generate an image via configured providers"` (42 chars). | Measured adoption regressions on #2066 for deferred new capabilities; revisit with the adoption instrumentation, not here. |
| D4 | New `fal` provider row with `media_only=True`; capability vocabulary += `"image"`/`"video"`; radient row capabilities += `{"image","video"}`. | The key must be storable via the standard login path; without the flag the row leaks into chat-model surfaces (the defect `speech_only` exists for). |
| D5 | FAL rung is **key-based** (no third-party FAL OAuth exists). | Lane-verified; a real login can be added if that changes. |
| D6 | **httpx-async rungs; no threads/subprocess/`to_thread`.** | Native asyncio cancellation lands in the socket read and actually releases the connection (`loop.py:934-936`); `to_thread` cannot be cancelled. |
| D7 | **No `group_reaper` registration, no second reap concept.** | That module reaps detached process groups spawned by `bash`; this tool spawns none. A SIGKILLed runtime leaves a provider-side job that completes uncollected — identical to any client disconnect, documented, accepted. |
| D8 | Approval tier **`write`**, with a `describe_approval` naming provider/quantity/size. | A generation spends real money; default mode "ask" makes the prompt the ONE place cost is shown pre-spend. Tier records intent (not protection) — auto/yolo skip the prompt; a reviewable knob. |
| D9 | createIf gate = "a provider the harness can reach" (**sync, no network**); call time re-resolves. | Rung 3: zero schema where the capability cannot run. Radient lights from persisted rows only (the STT rule); FAL/OpenAI accept a stored row **or** an exported key (an exported key genuinely runs the call). |
| D10 | Surfacing = `_TOOL_ROSTER_NAMES` append + packaged guide + `TOOL_NOTES`. | Membership-gated roster rows; the guide enters discovery with zero code. |
| D11 | One PR, landed after lane D (rebase + scoped convergence). | A split yields an unreleased intermediate with cross-PR rebase risk. |
| D12 | Restart = a NEW call; never resume an orphaned provider job. | No cross-session job registry exists and a resumed orphan has no consumer for its bytes; the receipt carries prompt/model/seed for a re-issue. |

## 2. Surfaces

- **Tool**: `local_operator/tools/image_tool.py` — `GenerateImageParams` (7
  params: `prompt`, `source_image_path`, `strength` 0..1, `image_size` (6 FAL
  enum values, default `square_hd`), `num_images` 1..4, `seed`, `model`),
  `build_generate_image_tool` (gate §D9), `execute_generate_image`
  (`@_guard`), `_describe_generate_image_approval`. Appended at the END of
  `TOOL_BUILDERS` and `DEFAULT_TOOL_NAMES` (prompt-cache prefix stability).
- **Cascade**: `local_operator/imagegen/` — `availability.py` (sync probes,
  async key twins), `cascade.py` (`resolve_image_route`, `run_image_cascade`,
  `ImageGenerationCancelled`, `ImageGenerationUnavailable`), `rungs.py`
  (httpx-async executors + best-effort cancel), `media.py` (bounded
  downloads), `errors.py` (own failure mapping, own reason-class vocabulary).
- **Docs**: `tool://generate_image` (`TOOL_NOTES`), `guide://image-generation`
  (packaged), this file.

## 3. The cascade (frozen order, first credential wins)

1. **Radient** (hub media tools, `/v1/tools/media/*`, bearer): live model
   list (default marked `default: true`; **no ids pinned in code**) →
   affordability probe (`/v1/me/billing-sources/capacity` → `total_balance`
   ≥ `unit_price × num_images`; any probe failure → **proceed
   optimistically**; unaffordable → skipped attempt, next rung) → generate →
   poll `status` (2 s, 4 s after 30 s) → `result` → bounded downloads.
   Status/result/cancel take `request_id` ONLY; failures switch on
   `error_type` (`media_rejected|media_failed|media_rate_limited|
   media_unavailable`) + HTTP status, never prose.
2. **FAL** (queue API, `Authorization: Key`): `POST {base}/{app_path}`
   (`sync_mode: false`), response-carried `status_url`/`response_url`/
   `cancel_url` with app-path fallback derivation (the old client hardcoded
   its app root — fixed); poll → result → downloads. img2img rides the
   `/image-to-image` app route with `image_url`.
3. **OpenAI** (single synchronous request ≤120 s; **no provider-side
   cancel** — abort discards the wait, the server may still bill):
   `POST /v1/images/generations` with `size` mapped per model family; handles
   `b64_json` and `url` items. img2img is skipped with a recorded reason.

Fail-forward on every rung failure EXCEPT user cancellation (stop; no
failover) and local validation (raised before dispatch). All rungs failed →
`ImageGenerationUnavailable` carrying the full attempt list.

## 4. Cancel / steer / restart

All paths ride existing machinery; **no new cancel concept**. An abort
cancels every runner (loop `_execute_batch`); steering covers
`interruptible=True` tools — this tool declares it. The `CancelledError`
handler performs a best-effort provider cancel (FAL `PUT cancel_url`;
Radient `POST /v1/tools/media/cancel` `{request_id}`; OpenAI none), bounded
by 5 s (connect 2/read 3) under `asyncio.shield` + `asyncio.wait_for` inside
a suppress — a second Esc abandons the cleanup silently and never masks the
cancellation. The rare race where the abort lands between polls (no task
cancellation) raises `ImageGenerationCancelled`, which the tool turns into a
clean `is_error=True` receipt: "Generation cancelled before completion
(provider job …). Re-run to start a new generation."

## 5. Result shape (lane D's contract)

`ToolResult(content=[caption, *AttachmentContent], details={…})` — the
caption FIRST and self-sufficient (provider dispatch skips artifact blocks):
provider, model, dimensions, seed, digests, cost, and attachment failures.
Bytes are registered exactly ONCE through
`cache_media(raw, content_type, kind="image", name=…, source_url=…, width=…,
height=…, duration_s=…)`; a refusal degrades to a caption note, never a
raise; if every registration refuses, the result is an error naming the
sources. `details` carries provenance: `provider`, `model`, `prompt`, `seed`,
`generation_id`, `cancel_handle`, `cost_usd`, `attempts`, plus
`source_image_path`/`strength` for img2img. No `ImageContent` co-attach in v1.

## 6. Constants (named, test-pinned where useful)

`IMAGE_POLL_INTERVAL_S` 2.0 / `_SLOW_S` 4.0 (after 30 s) ·
`IMAGE_RUNG_TIMEOUT_S` 240 · `IMAGE_GENERATION_TIMEOUT_S` 300 ·
submit/poll/result timeouts 30/10/15 s · download 60 s + 32 MiB cap ·
`CANCEL_TIMEOUT_TOTAL_S` 5 · Radient models/capacity 5 s ·
`OPENAI_IMAGE_TIMEOUT_S` 120. Schema footprint: ~1.3k chars ≈ ~480 billed
tokens, paid only in sessions whose gate passes.

## 7. Verification

See the PR body: targeted unit suites (cascade order matrix, fail-forward per
class, cancel path, downloads, gate matrix, approval describe, result/attachment
shape), the scoped gate (`make check-changed`), CI, and the real-run evidence
(a pinned-seed Radient generation with its attachment artifact, a
cancel-mid-generation run with the provider cancel verified, a restart run,
and an honest failure path).
