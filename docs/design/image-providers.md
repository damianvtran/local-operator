# The image-generation provider matrix (media wave-2)

Status: deliverable, media wave-2 (`feat/media-provider-breadth`). Derived from
the wave-2 research matrix (scout lane, session `4dc01d842c67`, 2026-10-09 — no
live API calls, no secrets read), trimmed to the matrix the harness acts on and
**amended at implement time (2026-10-09)** where the implementer fetched the
providers' own documents, schemas and API listings; every amendment is marked.
`guide://image-generation` points here for provenance; the guide carries the
current user-facing posture.

Search date everywhere: **2026-10-09**. Prices move; re-check before quoting.

## How to read the rows

Every row and claim carries one of three tiers, stated where it matters:

- **Contract** — a first-party specification (docs page, OpenAPI schema)
  fetched and read on the date shown; the implementation follows it strictly.
- **Observed** — corroborated by shipped first-party artifacts or community
  (OSS) implementations, NOT a published wire contract; labelled per claim.
- **Open risk** — depends on something no static source settles (e.g. whether
  a harness-driven request is accepted); the live probe listed under
  *Live validation status* is what closes it.

Every implementation claim below was tiered at reviewer round 1 (F3): the
openai-sub wire facts (``stream``/``store`` flags, SSE event shapes, the
``result`` field) are **Observed** — the sources are OpenAI's shipped skill
asset plus two OSS implementations, not a published wire spec — and its
acceptance is an **open risk**; everything else in the matrix is **Contract**.

## The shipped rungs (7) — provider × auth × models × cost

| # | Rung | Auth shape(s) the harness holds | Image models (Oct 2026) | Wire | Cost source | Verdict | Sources |
|---|---|---|---|---|---|---|---|
| 1 | **Radient** | signed-in account; Pass key | hub media tools | hub route; model list read at call time | **reported** — `cost_usd` per generation | baseline, unchanged | `registry.py:768-794`; wave-1 merged evidence (PR #2089/#2091) |
| 2 | **FAL** | stored `fal` key; `FAL_API_KEY` env | full FAL catalog (`fal-ai/flux/dev` default) | `queue.fal.run` app endpoints; submit → poll → fetch | rate table (FAL dashboard) — documentation only | baseline, unchanged | `registry.py:897-916` |
| 3 | **OpenAI — platform key** | `openai-key` namespace; `OPENAI_API_KEY` env | `gpt-image-2.5-sunburst`/`-flare` (current), `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1-mini`; T2I + edits | `POST api.openai.com/v1/images/generations` (+`/edits`); `n`, `size`, `quality` | rate table — per-image ≈ $0.005–$0.211 by quality | shipped wave-1 | [pricing.md](https://developers.openai.com/api/docs/pricing.md); [guide](https://developers.openai.com/api/docs/guides/image-generation); `rungs.py` run_openai |
| 4 | **OpenAI — ChatGPT subscription** | the `openai` OAuth grant (Codex sign-in); `chatgpt-account-id` from the grant | built-in `image_generation` tool (Codex docs: `gpt-image-2`); refs/edits documented; **no seed**; 3–5× quota burn; **not on Free plan** | `POST https://chatgpt.com/backend-api/codex/responses` — SSE; `stream:true`+`store:false` mandatory (**OSS-observed**), `tools:[{"type":"image_generation"}]` + `tool_choice`; image b64 in `image_generation_call.result` | **subscription** — plan quota, NO cash figure ever | **implemented this wave — tiers: Observed wire / open-risk acceptance** | [learn.chatgpt.com pricing](https://learn.chatgpt.com/docs/pricing#image-generation-usage-limits); [SKILL.md](https://github.com/openai/codex/blob/main/codex-rs/skills/src/assets/samples/imagegen/SKILL.md); community: [codex-bridge](https://github.com/Sateezg/codex-bridge), [qwwiwi](https://github.com/qwwiwi/codex-gpt-image-2-subscription/blob/main/README.en.md); `providers/oauth/openai.py`, `clients.py:2374` |
| 5 | **Google — Gemini API key** | `google` row (`GOOGLE_AI_STUDIO_API_KEY`, paste login) | Nano Banana family: `gemini-nano-banana-2.1` (workhorse), `gemini-3.1-flash-lite-image` (cheapest), `gemini-3.1-flash-image`, `gemini-3-pro-image`, legacy `gemini-2.5-flash-image`; **Imagen shut down** | `POST generativelanguage.googleapis.com/v1beta/interactions` (`x-goog-api-key`); `response_format {type:"image", aspect_ratio}`; image b64 in `steps[]` model-output blocks (SDK convenience property `output_image`) | rate table — lite $0.0336; nb-2.1 $0.0336/$0.0504/$0.113 (1K/2K/4K); flash-image $0.067–$0.151; pro-image $0.134/$0.24; **no free tier** | **implemented this wave, live-unverified** | [image docs](https://ai.google.dev/gemini-api/docs/image-generation); [pricing](https://ai.google.dev/gemini-api/docs/pricing); `registry.py:649-665` |
| 6 | **xAI — key or OAuth** | `xai` key row; `xai-oauth` grant (device flow, `api:access` scope); `XAI_API_KEY` env — both classes store under `xai` | `grok-imagine-image-2.0`; 1K/2K; `n` 1–10; aspect ratio; b64 opt; editing ≤5 refs | `POST api.x.ai/v1/images/generations` (bearer); optional deferred poll `GET /v1/images/{request_id}` | **reported** — `usage.cost_in_usd_ticks` REQUIRED on the response (see amendment below); pricing page lists $0.04/image flat | **implemented this wave, live-unverified** (OAuth 403 class unprobed) | [Imagine docs](https://docs.x.ai/developers/model-capabilities/imagine) (updated 2026-10-08); [swagger](https://api.x.ai/docs/) + [openapi.json](https://api.x.ai/api-docs/openapi.json); [Hermes guide](https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth); `providers/oauth/xai.py:32-36`, `registry.py:566-583` |
| 7 | **OpenRouter** | `openrouter` row (`OPENROUTER_API_KEY`) | dedicated Images API; ~59 image models incl. `openai/gpt-image-2.5-*`, `google/gemini-nano-banana-2.1`, `x-ai/grok-imagine-image-2.0`, `black-forest-labs/flux.2-*`, `bytedance-seed/seedream-*` | `POST /api/v1/images`; discovery `GET /api/v1/images/models` + per-endpoint records; params incl. `n` (1–10), `aspect_ratio`, `seed` (where supported), `input_references`; response `data[].b64_json` + `usage.cost`; billing **all-or-nothing** (failed → 502, unbilled) | **reported** — `usage.cost` per request | **implemented this wave, live-unverified** | [docs](https://openrouter.ai/docs/guides/overview/multimodal/image-generation); live records: klein-4b $0.014/MP, seedream-4.5 $0.04/img; `registry.py:688-698` |

## Implement-time amendments (what changed against the research)

- **xAI cost: rate table → REPORTED.** The research pencilled xAI as
  rate-table ("$0.04/image flat" from the pricing page; nothing per request).
  The official OpenAPI schema fetched at implement time says otherwise:
  `usage` is `oneOf [null, MediaUsage]` and, **when present**, requires
  `cost_in_usd_ticks` ("the cost of this request", 1 USD = 10,000,000,000
  ticks) — the nullable-`usage` wording fixed at reviewer round 1 (F5); a
  null `usage` yields no figure and no label. Design D8 allows only
  reported figures on `cost_usd`, so the rung emits `ticks / 1e10` with
  `cost_source="reported"`. Source:
  [openapi.json](https://api.x.ai/api-docs/openapi.json) (fetched 2026-10-09).
- **Google `seed`/`n`: not documented — verified at implement time.** A full
  text search of the fetched image-generation page (2026-10-09) finds neither
  a seed parameter nor an image-count parameter; the page's own limitation
  note says the model "won't always follow the exact number of image outputs"
  (count is prompt-driven, not guaranteed). Implementation: seed dropped
  silently; `num_images > 1` is a recorded skip. Source:
  [image docs](https://ai.google.dev/gemini-api/docs/image-generation).
- **Google surface already on the new endpoint.** The research was right: the
  REST examples on the current docs use `/v1beta/interactions` with the
  content-block input (`[{"type":"text","text":...}]`) and the result rides
  `steps[].model_output` image blocks; `output_image` is the SDK-side
  convenience property, implemented as a fallback. Imagen: "shut down and no
  longer available through the Gemini API" (docs, 2026-10-09).
- **OpenAI-sub host model: read at call time, no pin.** The community example
  host model (`gpt-5.5`) retires from Codex 2026-10-14; the rung reads the
  harness's own suggestion table (`local_operator.model.defaults`) at call
  time, so a catalogue move updates it with no code change. The body's
  `input`/`tools`/`tool_choice` shape follows the community-verified call
  shape above (OpenAI does not publish this wire as public API).
- **OpenRouter request-body details.** `n` sent only when the caller asks for
  more than one (single-image providers reject `n>1`; `n=1` is the default);
  seed passed through when pinned (support varies per model/endpoint). The
  default model is `bytedance-seed/seedream-4.5` (multi-image workhorse;
  klein-4b is the cheap alternative — $0.014/MP, `n` max 1). Sources:
  [docs](https://openrouter.ai/docs/guides/overview/multimodal/image-generation)
  (fetched 2026-10-09).

## Cost posture in the harness

- **Reported figures only on `cost_usd`** (design D8): Radient (hub),
  xAI (`usage.cost_in_usd_ticks`), OpenRouter (`usage.cost`). Absent usage →
  no figure, no label.
- **Rate-table rungs are documentation-only**: Google, OpenAI-key, FAL —
  no in-code price tables; the tables above are the reference, with dates.
- **openai-sub is quota-funded**: `cost_usd` stays `None`,
  `cost_source="subscription"`; the guide states the 3–5× quota burn and the
  Free-plan exclusion.
- `num_images` multiplies cost on every rung; the ChatGPT-plan rung generates
  one image per call in v1 (a larger request is a recorded skip).

## Live validation status (honest)

**No live API call was made for the four new rungs in this wave** (operator
decision: the spent-credential rule — every new rung's spend was deferred for
an operator-approved probe). The acceptance probe for each is ONE call:

| Rung | What the probe settles | Notes |
|---|---|---|
| openai-sub | Codex tool acceptance for a harness-driven request; SSE parse against the real stream; the quota spend | spends subscription quota (not cash); 3–5× burn; **needs a non-Free plan** (the route is unavailable on Free) |
| Google | `/v1beta/interactions` call + `steps`/`output_image` parse against the live envelope | paid-tier only (no free tier) |
| xAI | **the OAuth 403 class** (xAI allowlists/tiers its OAuth surface; standard SuperGrok subscribers have been seen rejected with 403 — remedy: API key) and the `usage.cost_in_usd_ticks` figure | key path is the reliable one; the second probe needs an account holding ONLY the OAuth grant (any stored key wins the new key-first order) |
| OpenRouter | `/api/v1/images` + `usage.cost` against the live settlement shape | all-or-nothing billing |

Radient/FAL are the unchanged baseline rungs; their live evidence stands from
wave-1 (PRs #2089/#2091), and no new live call was made this wave.

## Document-only this wave (survey verdicts)

New login rows were explicitly out of the minimal set; every row below is
reachable later without touching the rungs already shipped.

| Provider | Auth shape | Models | Endpoint | Cost source | Why deferred |
|---|---|---|---|---|---|
| Replicate | new paste row (`r8_…` token) | flux-1.1-pro, flux-dev/schnell, ideogram-v3-quality, recraft-v3, … | `POST api.replicate.com/v1/predictions` (async poll) | [pricing](https://replicate.com/pricing): flux-1.1-pro $0.04; flux-dev $0.025; schnell $3/1k; ideogram-v3-quality $0.09; recraft-v3 $0.04 | new row; breadth better served by OpenRouter |
| Stability AI | new paste row (API key) | Stable Image Ultra / Core (SD3.5-based) | `POST api.stability.ai/v2beta/stable-image/generate/{core,ultra,sd3}` | 1cr=$0.01; Ultra $0.08; Core $0.03; SD3.5 L/M/T 6.5/3.5/4 cr — **pencil** ([pricing](https://platform.stability.ai/pricing) JS-gated; [secondary, Jun 2026](https://developer.puter.com/tutorials/stability-ai-api-pricing/)) | new row; JS-gated pricing |
| Black Forest Labs | new paste row (`x-key` header) | FLUX 3 Image; FLUX.2 [klein/pro/max/flex]; FLUX1.1 [pro] | `POST api.bfl.ai/v1/…` → `polling_url` poll (async) | [pricing](https://docs.bfl.ai/quick_start/pricing): FLUX3 $0.041→$0.607 by res; klein from $0.014; pro from $0.03; max from $0.07 | OpenRouter already fronts FLUX at parity |
| Ideogram | new paste row (`Api-Key` header; prepaid credits; subs ≠ API) | Ideogram 4.x (+3.0 legacy); Precise Edit 4.5 | `https://api.ideogram.ai/v2/…` | Turbo $0.03 / Default $0.06 / Quality $0.10; Precise Edit $0.06 — **pencil** ([api-setup](https://developer.ideogram.ai/ideogram-api/api-setup); [release notes Jun 2026](https://releasebot.io/updates/ideogram); pricing table JS) | new row; typography niche |
| Together AI | new paste row | GPT Image 2, Nano Banana Pro, FLUX.2 [pro/flex/dev], Ideogram 4.0, Wan 2.6 Image… | OpenAI-compatible `POST api.together.xyz/v1/images/generations` | [pricing](https://www.together.ai/pricing) (Image tab; via snippets — **pencil**): GPT Image 2 $0.053; Wan 2.6 $0.03; Nano Banana Pro $0.134; FLUX.2 [pro] $0.03 | aggregator; OpenRouter covers |
| Hugging Face Inference | new paste row (`hf_…` token) | text-to-image task via providers (fal, replicate, nscale, wavespeed…) | `router.huggingface.co` provider routing; Bearer; `inputs`/`parameters` | provider rates passed through, no markup; **PRO $2/mo credits** ([pricing](https://huggingface.co/docs/inference-providers/pricing); [text-to-image](https://huggingface.co/docs/inference-providers/tasks/text-to-image)) | new row |

Also seen, fronted rather than separate rows: Recraft v4.x, Krea 2, Microsoft
MAI, ByteDance Seedream 5.0, Sourceful Riverflow, Qwen-Image-3, Tencent
HY-Image — all reachable via OpenRouter (partly Together/Replicate).
Not surveyed: Cloudflare Workers AI, Nebius, Leonardo, Recraft direct,
Bedrock/Azure.

Google OAuth routes were also document-only: AI Studio web sessions are not an
API; the OAuth quickstart and Vertex are real but project-scoped
(operator GCP setup — [oauth quickstart](https://ai.google.dev/gemini-api/docs/oauth),
[Vertex auth](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gcp-auth)),
and the harness's stored Google credentials are Workspace-scoped — wrong
audience/scopes, not usable.

## Open items carried forward

1. **xAI OAuth 403 class** — unprobed; probe decides whether OAuth is a real
   rung for the operator's account or a documented best-effort (fallback: API
   key). Earlier sightings: Hermes issue note via
   [Hermes guide](https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth).
2. **Codex tool acceptance** for a harness-driven request — the one thing desk
   research cannot settle (openai-sub §unknown in the source matrix).
3. **Google `seed`/`n`** — revisit if the interactions docs add either.
4. **OpenRouter catalog churn** — per-model availability/pricing moves; the
   discovery endpoints above are the live source (`GET /api/v1/images/models`
   + per-model `/endpoints` records).
5. **Deferred shapes** (xAI deferred mode, BFL polling) — candidates for the
   video/async wave; the rung contract already carries the seams.
