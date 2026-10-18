# Design: audio generation (voiceover + sound effects) in the harness

Status: implemented — harness half, 2026-10-17 (`feat/audio-generation`).
Base: `722d90253b` (`origin/main`). Companion reading:
`docs/design/artifact-generation.md` (the kind-neutral walk this lane rides),
`docs/design/image-generation.md` (the sibling lane this one mirrors), the
frozen server contract (Radient Agent Server lane, session `6c9c8f09af70`),
and `local_operator/tts/` (the BYOK voicing map this lane reuses).

## 1. Why two tools, not one

A voiceover takes **text plus a voice**; a sound effect takes a **prompt plus
a mandatory duration**. The server contract itself exposes two routes
(`/tools/audio/voiceover`, `/tools/audio/sound-effects`), and one tool per
route keeps the model's reach aligned with the request shape: a merged tool
would push every field of both into one schema the context-budget ratchet
bills on every request, and the duration-required half would make the other
half look optional.

Both are `write` tier (real spend, billable product), `shared` concurrency and
`interruptible` (a generation must yield to steering), and **createIf-gated
per direction** — deliberately different gates:

- `generate_voiceover` exists where Radient (persisted login), ElevenLabs
  (stored/exported key) or OpenAI (`openai-key` rows / store / export) is
  reachable.
- `generate_sound_effect` exists where Radient or ElevenLabs is reachable:
  the frozen contract has **no OpenAI sound-effect path**, so an OpenAI-only
  machine must not advertise a tool whose every call could only end in "no
  provider for this request shape" (the D9 footprint rule).

All gates are sync and socket-free (`audiogen.availability`), the same
discipline the image/tone-video gates use; a probe failure reads "not
available" rather than raising.

## 2. The cascade (frozen order, fail-forward)

**Radient → ElevenLabs → OpenAI**, first match wins. Radient first because it
is the platform's own plane; the BYOK rungs follow with the TTS lane's
economics: the user's credits being out is not the platform broken, so a
Radient refusal falls forward to the user's own key.

The walk is the generic `artifacts.walk.run_job_walk` (one submit / progress /
cancel / steer / restart interface, unchanged): per-rung budget 180 s, overall
240 s, a failed rung records its attempt and fails FORWARD, user cancellation
is a STOP with no failover, and exhaustion raises
`AudioGenerationUnavailable` carrying the resolution and the full attempt
list. The direction filter pre-records `skipped`/`unsupported` attempts for a
rung that cannot serve the request shape (OpenAI on a sound effect), never
dispatches it, and when no rung can serve the request the refusal names the
remedy instead of failing later.

## 3. The wire (server contract v1)

- Submit: `POST /tools/audio/voiceover` | `POST /tools/audio/sound-effects`
  → `{request_id, status, model, provider, billing, usage, cost}` with
  `status ∈ IN_QUEUE | IN_PROGRESS | COMPLETED | CANCELLED`.
- Observe: `GET /tools/audio/status` / `GET /tools/audio/result`
  (`?request_id=`), bounded polls, **no progress fraction** — terminal words
  only; an unknown terminal word is an unreadable payload, not an infinite
  poll.
- **Failure is a pair**: `status:"COMPLETED" + error_type:"audio_failed"` is
  an error, classified through the ONE terminal classifier used for submit,
  status and result reads (so the branches cannot drift). The reason class
  `audio_failed` is carried verbatim into the attempt records.
- Result `audio` block: `{url, expires_at, content_type, format,
  sample_rate_hz, channels, bitrate_kbps, duration_seconds, size_bytes,
  sha256, provider_model_id}`. The signed URL (24 h) is downloaded
  immediately, redirects off, bounded at the shared 32 MiB asset cap;
  **sha256 is verified before the bytes are attached and a mismatch fails
  closed** (the payload is discarded and the rung fails, with the cascade's
  standard fail-forward).
- Cancellation (settled contract, security round 1): **a guaranteed stop only
  while QUEUED.** `POST /tools/audio/cancel` answers
  `CANCELLED` + `cancellation.vendor_stopped:true` before the provider was
  called → nothing is charged, and the receipt may say so. Once the job is
  RUNNING it answers `CANCELLED` + `vendor_stopped:false` +
  `reason:"provider_has_no_cancel_api"` → the generation continues and is
  charged if it produces audio. A repeat cancel of a queued-cancelled job is
  idempotent (200, `vendor_stopped:true`); a completed/failed job refuses with
  **409 `audio_already_completed`**. Nothing claims a guaranteed stop.
- Admission bound (HTTP 429, code `audio_rate_limited`, no `Retry-After`): a
  SUBMIT refusal happens before a job row exists, so it is retried with
  bounded backoff (2, 4, 8 s; four attempts) and surfaced as
  `reason_class="transient"` with `error_type="audio_rate_limited"` and the
  server's sentence. An ADMITTED job that hits a vendor rate limit settles
  `COMPLETED` + `error_type:"audio_rate_limited"` with zero cost — never
  retried automatically; a resubmit is the caller's decision.

## 4. Cost rules

Only `cost.amount_usd` is authoritative. `basis:"quote"` → `estimated`,
`basis:"billed"` → `billed`; missing labels on an amount degrade to
`estimated` (conservative, never a charge claim). The receipt prose names the
provenance (`cost_source` = `reported` for every Radient figure,
`cost_provenance` naming the basis/price_version).

**BYOK estimates.** ElevenLabs and OpenAI report no per-call figure at all
(`cost: null` by contract). The harness estimates from ONE central table,
`audiogen.pricing` (`price_version = "audio-rates@2026-10-17"`, rates read
from the vendors' published pages on that date), labelled
`estimated` at `rate_table` source and **never written as a charge** — the
same convention the web-search table established. Unknown provider/model →
`None` ("amount not stated"), never a fabricated zero. The approval prompt
carries the estimate before the spend; the ledger row keeps the estimate flag
end to end (`billing_basis` survives every hop).

**Spend ledger.** A new `audio` channel ("Generated audio",
`session/channel_spend.py`) receives one row per outcome — `ok`, `failed` or
`cancelled` — with unit `chars` (voiceover) or `seconds` (sound effect) and the
mapping done through `map_audio_cost_labels`, the audio twin of the image
mapper, so the live emission and the transcript backfill cannot disagree. An
unsettled cancel stays `amount=None` (may still be charged), like the image
lane's; a failed walk records `failed`/None with the documented
"presumed unbilled" presumption.

## 5. Peaks and the waveform route

The server contract v1 emits no `waveform` field, so the local default is
`audiogen.peaks`: decode with **miniaudio 1.71** to 8 kHz mono FLOAT32
(measured 63.7 ms for a 98 s MP3), 240 even buckets, max-abs per bucket
clamped 0..1. Peaks are computed **once, at artifact creation**, by
`cache_media(kind="audio")` and stored as the `<digest>.peaks.json` sidecar of
the same content-addressed store the bytes live in — never decoded in a
render loop, and never carried in `AttachmentContent`.

The dependency is an optional extra (`local-operator[audio]`): a host without
it writes no sidecar and every consumer degrades silently (no waveform state),
never an error. A successfully decoded silent clip is 240 zeros — silence is a
real waveform; `None` means "uncomputable here", and the two are distinct.

The waveform route `GET /v1/desktop/sessions/{session_id}/attachments/{digest}/waveform`
(+ its child twin) returns the raw `{version: 1, bars: 240, peaks: [...]}`
JSON, mirroring the byte route family's gates exactly: digest shape as the
traversal gate, the session existence check (and the peer-row 409), `nosniff`
+ `no-store`. A digest with no sidecar is an ordinary 404 — the surface paints
its no-waveform state.

## 6. Cancellation honesty (all paths)

- Radient: the server's answer, quoted (see §3). `vendor_stopped:true` is the
  one branch allowed to promise "you will not be charged"; `false` says the
  provider has no cancellation endpoint and the generation continues; absent
  says the server did not report whether the provider was called, which still
  says the work may bill — a receipt never invents an outcome.
- BYOK: there is no server-side job; the harness stopped waiting, and the
  receipt says the user's provider may still bill for work already delivered.
- `already_completed` (409): not a failure; the surfaces' frozen
  `media_already_completed` pair rides `details` so the card renders "already
  finished" rather than an error.

**The settle poll (a running stop is not an answer).** Because a refused
mid-flight cancel leaves the job running, the tool starts ONE bounded
observation when the server did not stop it: `audiogen.settle` polls
`GET /tools/audio/status` every 3 s for up to 120 s, on the SESSION's
background task set (`ToolContext.spawn_background`, cancelled by `dispose`).
No new job system, no wake schedule, no blocking of the turn. The status read
alone carries a late charge (`settled:true`, billed cost) and lands as a
ledger REVISION of the cancel row (`rev 1`, `billed`/`server_reported`); the
artifact is a bonus — when the result read and the sha256-verified download
succeed it is registered through the same `cache_media` path and its digest
rides an `audio_settle_v1` bookkeeping transcript row (via
`Session.journal_audio_settle`). If nothing terminal is observed inside the
bound the row says "settlement not observed" and no amount is invented. A
process death stops the observation only: the server's settlement is
authoritative and the audit gap is the same one a killed tool call leaves.

## 7. BYOK voicing (one map, two callers)

The BYOK voiceover legs reuse the vendored voicing map's executor
(`tts.adapters.resolve`) so `tone`/`expressiveness`/`pace`/`language`/`accent`/
`instructions` translate exactly as the TTS daemon's own BYO legs do — the
map's capability rows pin every provider's behaviour (pace clamps 0.7–1.2 on
ElevenLabs and 0.25–4 on OpenAI; `language` rides `language_code` only on
flash models and is ignored with a note elsewhere; OpenAI emulates tone,
expressiveness, language and accent as instructions sentences). The mapping's
notes travel into the result's voicing record (`{applied, degraded}` tokens),
so nothing is dropped silently.

ElevenLabs sound effects are a direct `/v1/sound-generation` call
(`duration_seconds` required, `prompt_influence`, `loop` when requested);
OpenAI is voiceover-only (`/v1/audio/speech` — the cascade never routes a
sound effect there, and a direct caller that arrives is refused honestly with
the `unsupported` class).

## 8. What is deliberately NOT here

- No `*_status`/`*_collect` tools: generations complete in-turn; there is no
  durable job record to resume. (If the server later grows long-running audio
  jobs, the video lane's `video_job_v1` pattern is the template.)
- No WAV/output-format parameter: the contract is MP3 (128/32 kbps) only.
- No auto-duration for sound effects, and no client-side duration derivation —
  the provider's `duration_seconds` is the fact.
- No pronunciation dictionaries or phoneme control; instructions text is the
  documented escape hatch where a provider takes it.
- No provider key ever rides a Radient route, and no provider-specific field
  rides the tool schema or UI (`provider`/`model` are opaque display strings).
- The harness does not delete the store's sidecar on any path: attachments are
  never evicted (the store's own contract), and reclaiming orphaned bytes is a
  user's explicit choice.
