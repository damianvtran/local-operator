# Session voice on Groq: cloud listening and speaking, no local models

Status: **architect proposal, 2026-10-19 (EDT). Not approved for building.**
Carrier: this file, uncommitted, on `feat/qwen-voice` in the backend worktree
(`lo-qwen-fix3be` @ `85cc13011`). UI worktree: `lo-ui-qwen-fix3` @
`origin/feat/qwen-voice` `89804f3b1` (includes "off by default; off means
nothing loads"). Harness speech code read at local-operator `origin/main`
`26723b8eb`. Line numbers are against those bases and drift with edits.

Operator (Ben), in his words: "ok let's try groq" — the answer to "Should voice
move its listening and speaking to a paid cloud service, to free the 6.6 GB of
memory it uses locally?". Earlier: "we want it to be usable for everyone, and
not everyone can afford or have that 36 gigs of memory", and "I want us to not
ship any of this duplicate code ... text-to-speech or speech-to-text that
already exists in the harness."

Frozen and not planned against: no merge and no PRs (work stays on
`feat/qwen-voice`); voice OFF by default, and off loads nothing; no demos and no
app, server, Electron, Gateway or s2s launches unless Ben asks; the 6.5 GB is
not re-investigated.

## The framing, corrected

The question was "how does Groq replace the local speech models". The change
that removes the memory is different: **voice stops using the s2s rig and uses
the harness's own speech cascades.** Groq is one new rung in those cascades.
Once voice calls the cascades, every rung already shipped (Radient, ElevenLabs,
OpenAI) also works for voice with no local model. Groq is the cheap rung, with a
free plan anyone can sign up for. So the plan has two parts that can be
reviewed separately:

1. Voice moves from the rig to the cascades. That part removes the RAM and the
   disk, and it is most of the work.
2. Groq joins the cascades. A small, additive change.

## Evidence the decisions rest on

Existing code (file:line):

- **The cascades are already the single STT/TTS implementation.** STT order and
  the persisted-rows rule: `local_operator/stt/cascade.py:1-48`; rung tuple
  `:104-108`; resolver `:251-333`; walk with fall-forward `:576`; the
  one-named-rung executor `transcribe_backend` `:753`. TTS: rung tuple
  `local_operator/tts/cascade.py:96-100`; walk `synthesize_speech` `:491`, which
  falls forward on EVERY failure (`:539-545`, docstring `:40-46`).
- **The OpenAI clients already speak Groq's wire.** `OpenAiSttClient` posts
  multipart to `{base_url}/audio/transcriptions` with `Bearer` (`stt/clients.py:257-329`);
  `OpenAiTtsClient` posts JSON `{model, input, voice, response_format}` to
  `{base_url}/audio/speech` (`tts/clients.py:173-227`). Both take `base_url`.
  Their failure label is hard-coded `"OpenAI"` (`stt/clients.py:306`, and the
  `label=` in `tts/clients.py`'s call to `_post_audio`). Commit `cac51c33c` on
  `origin/feat/session-voice` already added a `label=` parameter to
  `OpenAiSttClient` for exactly this reuse ("so a refusing local server is named
  as the local server, never as OpenAI").
- **Credential rows for speech-only keys have a precedent.** `openai-key`
  (`providers/registry.py:731-773`): own namespace, `env_keys` deliberately
  absent so an exported variable never lights the rung, `speech_only=True`,
  `capabilities`. The probe and the call-time key both use
  `AuthStore.get_persisted_api_key(namespace, session_id, kinds={"api_key"})`
  (`stt/cascade.py:188-210`, `tts/cascade.py:142-147`).
- **Opt-in rungs have a precedent, and the generic paths must not grow one.**
  The UI's speech settings parser rejects the WHOLE resolution when any rung
  path is unknown (`lo-ui …/features/settings/speech-settings-model.ts:90-95`
  `VOICE_PATHS`, `:122-150` `parseVoicePathResolution` returns `null`). A new
  rung on `GET /v1/tts/paths` would blank the shipped Speech settings group. The
  `say` rung (commit `20a012742`) and the loopback STT rung (`cac51c33c`) were
  made opt-in (`include_local`, `local_stt_url`) for that class of reason.
- **The voice routes used to call the cascades.** `origin/feat/session-voice`
  `local_operator/server/routes/desktop_voice.py`: `_save_upload` (`:475-500`),
  `_transcribe` via `transcribe_audio` + `emit_voice_spend` (`:503-555`), and
  `POST /v1/desktop/voice/speech` via `synthesize_speech` (`:786-870`). The Qwen
  design removed them (Q10, `docs/design/qwen-voice.md:86-89`; contract
  "Removed" list `:125-129`). Today's route "neither transcribes nor
  synthesises" (`desktop_voice.py:26`).
- **APIError carries no response headers.** Fields are `status_code`, `code`,
  `details`, `body` only (`clients/_http.py:225-266`). The one builder both
  speech clients use is `stt/errors.py:358-383`
  (`api_error_from_httpx_response`; TTS imports it at `tts/clients.py:27`). A
  `retry-after` is lost today.
- **429 is mapped to a generic sentence** (`server/routes/speech.py:277-280`).
- **The voicing map is vendored, never re-authored** (`tts/voicing_map/README.md`);
  `adapters.resolve` raises for any provider other than `elevenlabs`/`radient`/`openai`
  (`tts/adapters.py:644-665`).
- **`speakable` already cuts on sentences** (`voice/speakable.py:66`
  `_SENTENCE_END`, `cap` `:113`, `speakable` `:156`), and its cut-reply tail is
  spoken text (`:51`).
- **The rig cannot take a Groq key the harness's way.** s2s's OpenAI-compatible
  handlers read the key from the process environment `OPENAI_API_KEY`
  (`s2s-venv/…/speech_to_speech/TTS/openai_compatible_handler.py:398`,
  `STT/openai_compatible_handler.py:248`). The app deliberately gives s2s an
  allowlisted environment with no credentials (`qwen-rig.ts:203-218`).
- **The Gateway has no cloud STT/TTS stage.** Its provider registry lists
  realtime speech-to-speech models only — dashscope, stepfun, gpt-live,
  google-live, doubao-seeduplex, s2s, minicpm-o
  (`qwa/node_modules/qwen-audio-agent/server/src/voice/providers/registry.mjs:21-31`).
  A search of the whole package for `audio/transcriptions|audio/speech` returns
  no match.
- **What the hidden page already gives us.** `qwen-voice-page.ts:259-305`:
  memory-only partition, `sandbox`, `contextIsolation`, no preload,
  `backgroundThrottling: false`, autoplay without a gesture, crash recovery,
  navigation lock, mic preflight (`qwen-voice.ts:576`), `close()` releases the
  renderer and the microphone (commit `1ebfe7571`). The app's own documents
  already get audio-only `media` (`window-guards.ts:515`); the Gateway page has
  its own one-origin verdict (`:553`).
- **The main↔daemon bearer stays in main**, with the renderer's desktop
  transport; `voice.utterance` is a relay op today, text-only by approved
  deviation (`desktop-media.ts:206-207`; `qwen-voice.md:154-155`).
- **The renderer CSP** pins `connect-src` to loopback and named hosts, with no
  `wasm-unsafe-eval` (`src/renderer/index.html:42`).
- **Off-by-default** is a main-owned file read before any voice machinery
  (`src/main/voice/qwen-voice-setting.ts:1-38`).

Measured here (2026-10-18/19, EDT), read-only:

| What | Size |
|---|---|
| `~/.local/share/qwen-voice-rig/hf-cache/hub/blobs` | 7.0 GB |
| …Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit snapshot | 2.5 GB |
| …parakeet-tdt-0.6b-v3 snapshot | 2.3 GB |
| `s2s-venv` | 1.5 GB |
| `qwa` (Gateway) | 86 MB |
| Smart Turn v3.2 `smart-turn-v3.2-cpu.onnx` | 8.7 MB |
| Silero VAD `silero_vad.onnx` (torch hub cache) | 2.3 MB |
| npm `@ricky0123/vad-web` 0.0.31, ISC | 6.7 MB unpacked |
| npm `onnxruntime-web` 1.30.0, MIT | 144.6 MB unpacked, all variants |

Blobs + venv + Gateway ≈ 8.6 GB, matching the brief.

Groq, read from Groq's docs on 2026-10-18/19:

- `console.groq.com/docs/speech-to-text`: `/openai/v1/audio/transcriptions`,
  `whisper-large-v3-turbo` at $0.04/h; flac, mp3, mp4, mpeg, mpga, m4a, ogg,
  wav, webm; 25 MB max on the free plan; **minimum billed length 10 s per
  request**.
- `…/docs/text-to-speech/orpheus`: `input` max **200 characters**; `response_format`
  **wav only**; voices autumn, diana, hannah (female), austin, daniel, troy
  (male); $22 per 1M characters. **Bracketed text is a vocal direction**
  (`[cheerful]`, `[whisper]`). The model card adds that bare numbers are read as
  words ("203" → "two hundred and three").
- `…/docs/rate-limits`: orpheus-v1-english 10 RPM, 100 RPD, 1.2K TPM, 3.6K TPD;
  whisper-large-v3-turbo 20 RPM, 2K RPD, 7.2K ASH, 28.8K ASD. `retry-after`
  (seconds) is set only on a 429; `x-ratelimit-remaining-requests` always
  refers to the per-day count.
- `…/docs/your-data`: inference inputs are not retained by default, but may be
  logged for up to 30 days to troubleshoot reliability or investigate abuse,
  unless Zero Data Retention is turned on in Data Controls. Data is held in GCP
  in the United States.

## D1 — Where Groq plugs in

**Taken: option (a).** Groq becomes rung 4 of both harness cascades —
`provider_stt_groq` and `provider_tts_groq` — reached only by callers that pass
`include_groq=True`. In this work, those callers are the voice routes. Voice
calls the cascades again, through the route code `feat/session-voice` already
had.

- STT rung: `OpenAiSttClient(key, base_url="https://api.groq.com/openai/v1", label="Groq")`
  with `model="whisper-large-v3-turbo"` pinned. A pinned model skips the OpenAI
  fallback list (`stt/clients.py:289`). This adds no new HTTP code; it adds
  only `label=`, cherry-picked from `cac51c33c`.
- TTS rung: `OpenAiTtsClient(key, base_url=…, label="Groq")` with
  `model="canopylabs/orpheus-v1-english"`, `response_format="wav"`, and
  `Params(voice_id=<groq voice>)` with no instructions and no speed, so the
  body is exactly Groq's four fields (`tts/clients.py:203-215`). The only new
  parameter is `label=`.
- Voice selection does NOT go through the voicing map: the map is vendored from
  agent-server and refuses unknown providers (`adapters.py:665`). The rung reads
  the descriptor's `resolved_gender()` (`tts/descriptor.py:133-143`) and maps
  it with a two-row constant, `GROQ_TTS_VOICES = {"female": "hannah", "male": "troy"}`.
  Tone, pace and accent are not applied, and the rung says so in its attempt
  detail. Adding Groq to the map is a later agent-server change.
- The rung refuses text over 200 characters locally, raising an `APIError`
  before any network call. That is a guard, not the chunker (D3).
- The rung replaces `[` and `]` with `(` and `)`. Otherwise an agent's
  "[1]" or "[note]" would reach Orpheus as a vocal direction.
- `include_groq` is threaded through `resolve_audio_path`, `transcribe_audio`,
  `resolve_voice_path` and `synthesize_speech`, the same way `include_local`
  was (`20a012742`). `GET /v1/stt/paths`, `GET /v1/tts/paths`,
  `/v1/stt/transcriptions` and `/v1/tools/speech` are unchanged byte for byte.

**Rejected:**

- **(b) Groq handlers inside s2s.** This would be a second STT/TTS client set
  beside the cascades: the duplicate code Ben ruled out. The key would also
  have to enter a third-party Python process through `OPENAI_API_KEY`
  (`…/TTS/openai_compatible_handler.py:398`). That breaks the persisted-rows
  rule and the credential-free allowlist (`qwen-rig.ts:203-218`). And it keeps
  the s2s process, the Gateway, the relay and the 1.5 GB venv alive, with
  Silero and Smart Turn still in torch.
- **(c) The Gateway's own cloud path.** None exists. The registry
  (`registry.mjs:21-31`) offers only realtime voice MODELS, which talk on their
  own. That is the "separate voice model with its own personality" V1 rejected.
  It is also why `qwen-voice.md` Q2 removed the model from the loop.

**This replaces** `docs/design/qwen-voice.md` **Q10** ("No lop speech code"):
voice routes call the cascades again. It also retires **Q1** (topology), **Q2**
(relay LLM), **Q3** (the `⟦lop-say⟧` token), **Q4** (Gateway task scope) and
**Q12** (rig directory) on this path. They describe machinery that no longer
runs. Q5-Q8 stay: target and binding, approvals, the closed phrase set, full
duplex with an echo guard. So do V3, V5, V7, V8, V13, V14 and V15.

## D2 — What still runs locally

**Taken:** a hidden, app-owned voice page does the listening and the playback.
Python does not run, and neither does the Gateway.

- **The page.** Keep the shell `qwen-voice-page.ts` already built: the
  WebContentsView, partition, sandbox, `backgroundThrottling: false`, autoplay,
  crash recovery, navigation lock, mic preflight, and `close()` releasing the
  mic. Point it at an app-bundled `voice-page.html` instead of the Gateway's
  WebUI, with one narrow preload. That preload carries one message up (`take`:
  WAV bytes; `vad`: `speech_start | speech_end`) and three down (`play`: segment
  bytes and an id; `stop`; `mute`). The permission verdict becomes "audio, main
  frame, the app's own voice-page URL" — the same shape as `:553`, a different
  origin.
- **VAD in the page.** `@ricky0123/vad-web` (`MicVAD`, Silero via
  `onnxruntime-web`'s WASM build) runs on `getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}})`.
  On speech end, it encodes 16 kHz mono 16-bit WAV, which Groq downsamples to
  anyway, and sends it up. Ship only the one WASM variant the page loads, not
  the 144.6 MB package tree. The page gets its own CSP:
  `script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'`. The main
  renderer's CSP (`index.html:42`) is untouched.
- **Turn detection.** The first slice ships with Silero end-of-speech silence
  only, with a 700 ms target, tuned by Ben. Smart Turn v3.2 (8.7 MB ONNX, CPU)
  can run in the same `onnxruntime-web` later. It is deferred until Ben has
  lived with silence-only turns, because it adds a feature-extractor port
  (s2s uses `transformers`' `WhisperFeatureExtractor`, `VAD/smart_turn.py`) and
  its value has to be shown, not assumed.
- **Playback in the same page.** Chromium's echo cancellation removes the page's
  own output from its own capture. Barge-in: a VAD `speech_start` while a
  segment plays stops playback and cancels the queue (D3). The text echo guard
  (`qwen-voice.md:162-169`) stays as the backstop.
- **The take path:** page → main (`take`) → `voice.utterance` relay op with the
  WAV file (D7 B2) → `/v1/desktop/voice/utterance` → `transcribe_audio(include_groq=True)` → admission (V5).
  The key never leaves the daemon; the bearer never leaves main.
- **The speak path:** main → `/voice/text` (segments) → `/voice/speech` per
  segment → main → page `play`.

**Can s2s and its 8.6 GB disappear on the Groq path?** Yes. Nothing on this path
reads the rig directory. `rigMissing` (`qwen-rig.ts:74`), `s2sArgs` (`:178`),
the footprint watchdog (`:245`) and the memory budget all go unused. So does
the 16 GiB machine gate (`QWEN_MIN_MACHINE_BYTES`, `:116`;
`qwen-voice.ts:449-453`), which must not apply here: it is the gate that keeps
voice from small Macs today. Deleting the rig directory is Ben's own act; the
app never wrote it (`qwen-rig.ts` header).

**RAM with voice on: an estimate, not a measurement.** One sandboxed renderer
(the page) plus the WASM runtime and a 2.3 MB Silero model is likely about
100-200 MB. Today's healthy idle rig is ~5.5 GB (`qwen-voice.ts:449`). Measuring
it means launching the app, which is barred, so it **waits for Ben**: read the
page process's `footprint`, idle and while speaking.

**Off still loads nothing.** With the switch off (`qwen-voice-setting.ts`), the
page is never created and no voice route or path probe is called. With the
switch on but the slot empty, the page is closed and the mic released, as
`1ebfe7571` already does. No new poll: the cloud availability check
(`resolve_*_path`) runs on enable and after a failure, never on a timer. The
TTS resolver can cost a network round trip per probe (`tts/cascade.py:26-34`).

**Rejected:**

- **VAD in the main renderer.** No extra process, but a hidden or minimised
  main window is throttled outside `normal` mode (`window-mode.ts:143`). The
  mic would then live in the window the person is using, so a crash there takes
  both down. The existing page shell already solves both, and its close path is
  reviewed.
- **Energy-threshold VAD with no model.** Zero dependency, but it triggers on
  keyboard noise and fans. A false take is a paid STT request, billed at a
  10-second minimum, sent to the session as a steer.
- **Keeping the Gateway WebUI as the page.** The WebUI only speaks the Gateway
  protocol, so the Gateway would have to stay.

## D3 — The 200-character cap: segments, gap-free playback, interruption

**Taken:** the server cuts the spoken text into segments, and the app fetches
and plays them one by one.

- **Segmenter, server side** (V8: the server decides what is spoken). New pure
  module `local_operator/voice/segments.py`, run on `speakable()`'s output.
  Rules: whole sentences packed up to 200 characters; a sentence over 200 is
  cut at `, ; : —`, then at a space, never inside a word. **The first segment
  is the first sentence alone** (cut if over 200), so the first audio starts
  as soon as possible. The joined segments equal the speakable text up to
  whitespace. The cap is one constant, `VOICE_SEGMENT_MAX_CHARS = 200`, for
  every rung: the server does not know which rung will serve until it tries,
  and shorter segments start sooner on every rung.
- **Contract:** `POST /voice/text` gains `segments: [str]` (additive).
  `POST /voice/speech {session_id, source, segment: int}` returns one WAV for
  that segment, re-resolving the source and re-segmenting on the server, so no
  free text ever crosses (V8).
- **Gap-free playback** in the page: each WAV is decoded to an `AudioBuffer`
  and scheduled on one `AudioContext` at the previous segment's exact end time.
  This is sample-accurate, with no element switching. Main fetches segment
  *n+1* while *n* plays: one request ahead, never more, so an interrupted reply
  wastes at most one request of the daily quota. Groq's pricing page lists
  Orpheus at ~100 characters per second (unverified snippet). A 200-character
  segment lasts ~12-15 s spoken, so one-ahead keeps the queue full after the
  first segment.
- **Interruption:** a VAD `speech_start` during playback, a slot switch,
  pause, or voice off → the page stops its sources and drops the schedule. Main
  aborts the in-flight fetch (`AbortController`) and requests no more
  segments. The turn's text stays on screen. The spoken-prefix end marker
  ("Cut short · rest above ↑", voice-composer SPEC §4) is shown.
- **`max_spoken_chars`** keeps its meaning: it caps the whole spoken text before
  segmenting. At the default 800 (`speakable.py`), a full reply is 4-5 requests.
  The truncation tail (`speakable.py:51`) is a fixed phrase, so it is served
  from a phrase cache (below), not a fresh request.
- **Phrases** (Q7's closed set plus the truncation tail) go through
  `/voice/speech` with `source: {kind: "phrase", key}`, as `feat/session-voice`
  had (`:806-808`). Their audio is cached at `<config>/voice/phrase-cache/<rung>-<voice>-<key>.wav`.
  They are fixed strings, so each one is paid once per rung and voice, not per
  turn.
- **The spoken summary is now the lever on cost.** `voice-spoken-summary.md`
  option (i) — an agent-written `Spoken:` paragraph of at most two sentences —
  turns a typical reply into 1-2 requests instead of 4-5. It does not change
  this contract: segments are cut from whatever `speakable` selected. It stays
  a separate decision (open question 3).

**Rejected:** splitting inside the Groq rung and joining the WAVs. That keeps
"one text, one audio", but the first sound waits for every chunk, and an
interruption still pays for all of them. Client-side segmenting: it would send
free text back to the server, breaking V8.

## D4 — Free-plan limits: what the person hears and sees on a 429

Plumbing: the shared builder (`stt/errors.py:358`) records `retry-after` and
`x-ratelimit-remaining-requests` into `APIError.details` (`retry_after_s`,
`remaining_requests`) for any 429. Both clients get them, nothing else changes,
and no new field is added to `APIError`. The voice routes answer `429` with
`detail.code = "voice_rate_limited"`, `detail.scope ∈ {minute, day}` and
`detail.retry_after_s`. `scope` is `day` when `remaining_requests == "0"`,
otherwise `minute`. Fall-forward is free: a Groq 429 moves to the next rung
when one exists (`tts/cascade.py:539-545`). Groq is last (D8), so for a
Groq-only user the 429 reaches the app.

Behaviour, for the person:

| Case | What happens |
|---|---|
| Speaking, per-minute limit (`retry_after_s` ≤ 10) | Main waits once and retries that segment. The strip stays "Speaking". No sound is added. |
| Speaking, daily limit (or a second 429) | Speech stops after the last segment that played. The end earcon plays, so the reply's end is audible. The strip shows a danger state: "Groq's free daily speaking limit is used up. Replies stay on screen until <reset time, local>." with one action, "Open Settings". Listening continues; the STT quota is separate. Until reset, replies are not spoken and no TTS request is sent; the strip says so instead of failing per turn. |
| Listening, per-minute limit | The take's audio is still in memory: one retry after `retry_after_s` (≤ 10 s). The strip shows "Sending…". |
| Listening, daily limit | The take is not sent. The cached `failed` phrase plays (no TTS spend). The strip shows "Groq's free daily listening limit is used up — type below until <reset time>." The page stops sending takes but stays open, so no request is paid for audio that cannot be sent. |
| Any 429 on a non-Groq rung | The same table, naming that provider. |

Never: a silent failure, a retry loop, or a spoken error sentence that costs
another request.

Budget at today's defaults (Groq's listed free limits): 100 TTS requests a day
≈ 20-25 full 800-character replies, or ~50-100 short ones. **TPD 3.6K is the
open unit:** if Groq counts characters, it allows 3,600 spoken characters a day
≈ 4-5 full replies. That would make the spoken summary (D3) close to required
on the free plan. Settled by evidence, not by guessing: one live 200-character
request, then Groq's usage page (waits for Ben).

## D5 — The key and the disclosure

**Credential:** provider id and namespace **`groq`**, a new `ProviderDefinition`
in `providers/registry.py` beside `openai-key`:

- `name="Groq"` and `login=create_api_key_login("Groq", "https://console.groq.com/keys", "Paste an API key from the Groq console.")`.
- `base_url="https://api.groq.com/openai/v1"`.
- **No `env_keys`**, for the reason `openai-key` gives (`registry.py:737-746`):
  an exported `GROQ_API_KEY` must never light a rung.
- `speech_only=True` and `capabilities=frozenset({"stt", "tts"})`.
- One row in each `login_catalog.py` table (`:103` "Speech only (voice); not for
  chat", and its siblings).

Probe and call both use
`store.get_persisted_api_key("groq", session_id, kinds=frozenset({"api_key"}))`,
the `_openai_stt_key` shape (`stt/cascade.py:196-210`). These are persisted
rows only. Groq also serves chat with the same key, so a later Groq chat
provider flips `speech_only` on this row rather than adding a second one. The
key lives only in the daemon's encrypted store. Main, the page and the renderer
never see it.

**Disclosure, before any audio leaves the Mac.** Shown when voice is turned on
in Settings (`voice-switch-setting.tsx`), naming the rung `resolve_*_path` says
will serve, and acknowledged once per provider. The acknowledgement is stored
in main's `voice-setting.json` beside `enabled`. For Groq:

> Voice sends what you say, and the replies it reads aloud, to Groq in the
> United States to transcribe and speak them. Only speech is sent — silence is
> not. Groq does not keep it, except for up to 30 days when needed to fix
> errors or check for abuse. To turn that off too, enable Zero Data Retention
> in your Groq console's Data Controls.

While listening, the strip names the provider ("Listening · Groq"). This
replaces the R12 disclosure from `feat/session-voice`, which was Gemini's.

## D6 — The local path's future (Ben decides)

**Recommendation: remove the Qwen rig path from `feat/qwen-voice` after Ben's
live check of the cloud path passes, and do not keep it as an opt-in.**

- Keeping it means keeping ~3,300 lines of UI code that exist only for the rig:
  `qwen-rig.ts` 671, `qwen-gateway.ts` 214, `relay-core.mjs` 589,
  `gateway-launcher.mjs` 309, `s2s-launch.py` 52, plus the Gateway half of
  `qwen-voice.ts` (956 total). It also keeps a third-party page and process
  tree, the memory watchdog, and two engines to test against every coordinator
  change. One audience benefits: Macs with ≥16 GB and no wish to send audio
  anywhere.
- A key-free path does not need the rig. It fits the cascades as opt-in rungs
  that already exist on `feat/session-voice`: macOS `say` for speaking
  (`20a012742`, ~0 GB, no download) and a loopback OpenAI-compatible STT server
  the person runs themselves (`cac51c33c`). If Ben wants offline voice later,
  that is the shape — a rung, not a second stack.
- Removal stays reversible: the branch history keeps the rig at `89804f3b1`.

## D7 — Slices

All work on `feat/qwen-voice` in the two worktrees, no PRs (frozen). "Gates"
means the review rounds the team runs on a branch commit. QA here is tests
only, because launching the app is barred, and every live check is listed as
**waits for Ben**. Quality gates per AGENTS.md on `origin/main`: flake8,
pinned black, isort, pyright via `make type-check` (bounded by
`scripts/run_bounded.py`), and the whole unit tree for the backend; the
`test:desktop` lane and typecheck for the UI.

| Slice | Files | Tests (no Groq key needed) | Gates | Waits for Ben |
|---|---|---|---|---|
| **B1 Groq rungs** (backend) | `providers/registry.py` (row), `providers/login_catalog.py`; `stt/__init__.py` + `tts/__init__.py` (enum members); `stt/clients.py` + `tts/clients.py` (`label=`; Groq constants and voices); `stt/cascade.py` + `tts/cascade.py` (`include_groq`, `_run_groq_rung`, bracket escape, 200 guard); `stt/errors.py` (429 details; `_DIRECT_PROVIDER_LABELS`); `server/routes/speech.py:161` (vendor label) | `httpx.MockTransport` (the seam `tts/clients.py:_post_audio` documents): exact URL, headers and body for both calls; a pinned model never falls back; WAV passthrough; `[x]`→`(x)`; >200 refused before any network call; 401/402/429 classification with `retry_after_s`/`remaining_requests` from doc-shaped header fixtures; persisted rows only (an exported `GROQ_API_KEY` does not light the rung); **`/v1/stt/paths` and `/v1/tts/paths` responses unchanged** (snapshot) | reviewer, qa-tester (test matrix) | one real STT and TTS call to record real response and 429 fixtures in place of the doc-shaped ones; the TPD unit |
| **B2 Voice routes on the cascades** (backend) | `server/routes/desktop_voice.py`: restore `_save_upload`/`_transcribe` (file door, 16 MiB cap, wav/webm sniff) and `/voice/speech` from `origin/feat/session-voice`, with `include_groq=True`; `voice/segments.py` (new); `/voice/text` `segments`; phrase source and cache; `voice_rate_limited`; `server/models/desktop_voice.py`; `server/features.py` `features.voice: 2`; i18n `wire_voice` keys; `docs/design/qwen-voice.md` superseded-rows note | segmenter properties (every segment ≤200, the first is one sentence, the join equals the input, no word split) as a table test; routes with the cascade stubbed: file and text doors, `input_path` = rung token, spend emitted, segment index out of range → 404, 429 mapping, phrase cache hit makes no cascade call; V7/V13 pins still green | reviewer, qa-tester | — |
| **U1 App-owned voice page** (UI) | `src/voice-page/` (html, entry, page CSP), `src/preload/voice-page.ts`, `electron.vite.config` input; `qwen-voice-page.ts` (URL, verdict); `window-guards.ts` (voice-page verdict); `package.json` (`@ricky0123/vad-web`, `onnxruntime-web`, one WASM variant packaged) | node tests for the pure parts: WAV encoder (header, 16 kHz mono), segment scheduler (start times are contiguous; stop clears all), VAD wrapper over a fake VAD (take on end; barge-in on start while playing); verdict refuses video, subframes and other origins; **off → page never created** | reviewer, qa-tester | the real mic, echo cancellation, VAD feel, page `footprint` |
| **U2 Coordinator on the cascades** (UI) | `qwen-voice.ts` (Gateway and speech ports replaced by page port + transport; machine gate and budget not applied); `desktop-media.ts` (`voice.utterance` accepts file bytes; `voice.speech` op); `desktop-contract.ts`; `qwen-voice-controller.tsx`, `voice-turns.ts` (speak via segments) | the `qwen-voice-main.test.mjs` pattern with injected ports: take → upload; speak → one-ahead fetch; barge-in aborts; slot switch cancels; minute 429 retries once; day 429 stops and reports; speak guard still holds; **voice off → zero requests** (extends `89804f3b1`'s tests) | reviewer, qa-tester | end-to-end turn on Ben's machine |
| **U3 Copy and states** (UI) | `voice-switch-setting.tsx` (disclosure plus stored ack), `voice-composer.tsx` (provider label, rate-limited states), i18n `ui.chat.voice.*` / `ui.settings.voice.*` | composer tests for each state's text and action; the ack is required before the first enable | reviewer, qa-tester, **designer** (rendered frames; static HTML mocks like the SPEC's `gen.py` if Storybook stays barred), **ux-reviewer** (the enable → disclose → listen flow) | rendered app frames |
| **U4 Remove the rig** (UI, after D6) | delete `qwen-rig.ts`, `qwen-gateway.ts`, `resources/qwen-voice/{relay-core.mjs,gateway-launcher.mjs,s2s-launch.py}` (keep `phrases.json` if B2 reads keys from it), their tests, `docs/qwen-voice-local-run.md`, the mac resource entry `358e8d3df` restored | `test:desktop` lane green; a grep test shows no `LOP_QWEN_RIG_DIR` and no spawn of `speech-to-speech` | reviewer, qa-tester | — |

Order: B1 → B2 freezes the contract → U2. U1 can start alongside B1, because
it does not touch the backend. U3 comes after U2. Ben's live check comes after
U3. U4 comes only after D6.

## Risks and what would change the answer

- **TPD counts characters** (3.6K/day). Then the free plan speaks ~4 full
  replies a day, and "usable for everyone" on free Groq needs the spoken
  summary or a lower cap. If the summary is refused, the answer moves toward
  "Groq for listening, another rung (or `say`) for speaking".
- **Free-plan figures.** The rate-limit page's text calls its table the
  Developer plan's base limits, but its tabs read "Free Plan Limits / Developer
  Plan Limits". The figures above match the brief's free-plan numbers. Ben's
  console Limits page is the authority.
- **Silence-only turn detection cuts people off.** Watch for takes that end
  mid-sentence and turn into two steers. If it happens often, Smart Turn in the
  page moves forward.
- **Echo cancellation fails on speakers** (no headset). Then the text echo
  guard carries the load and barge-in false-fires. Evidence: Ben speaking over
  a reply on laptop speakers.
- **First-audio latency.** One Groq round trip for the first sentence plus the
  upload. If it reads as slow, shorten the first segment further. If Groq's TTS
  is slower than real time (the 100 chars/s figure is unverified), one-ahead
  prefetch leaves gaps and needs two-ahead, at quota cost.
- **Orpheus voices are generic.** No descriptor mapping until agent-server's map
  gains a Groq row.
- **`onnxruntime-web` packaging** — WASM loading under `sandbox`, the page CSP,
  and an asar path — fails only at runtime. It is a U1 risk that unit tests
  cannot close (waits for Ben).

## Open questions for Ben

1. **Groq first, or last, for voice?** Recommended: last. The cascade has one
   frozen order (`tts/cascade.py:96-100`), and paid rungs already signed in keep
   serving. For his trial, an isolated config dir holding only the Groq key
   exercises Groq without signing out of anything.
2. **The rig path:** remove after the live check (recommended, D6), or keep it as
   an opt-in?
3. **Spoken summary:** build `voice-spoken-summary.md` option (i) as the slice
   after U3? On the free plan it is the difference between 4-5 and 1-2 TTS
   requests per reply.

## Decisions

Prior rows this record relies on, or replaces: `qwen-voice.md` Q1-Q4, Q10, Q12
(replaced or retired on this path, D1/D2); Q5-Q8, V3, V5, V7, V8, V13-V15 (kept);
`voice-spoken-summary.md` (input to D3); `feat/session-voice` R8/R11/R12
(superseded by D5/D6).

- **D1 — Where does Groq plug into voice?**
  - **Taken:** as opt-in rung 4 of the harness STT and TTS cascades
    (`provider_stt_groq`/`provider_tts_groq`, `include_groq=True`), reusing
    `OpenAiSttClient`/`OpenAiTtsClient` with `base_url` and `label`. Voice
    routes call the cascades again.
  - **Rejected:** (b) Groq inside s2s — duplicate clients, and the key would
    pass through `OPENAI_API_KEY` in a third-party process
    (`…/TTS/openai_compatible_handler.py:398`; `qwen-rig.ts:203-218`); (c) the
    Gateway cloud path — none exists (`registry.mjs:21-31`; no
    `audio/transcriptions|audio/speech` in the package).
  - **Commits us to:** `stt/`/`tts/` cascade modules and the two clients
    (`label=`); generic `/v1/*/paths` frozen as they are; replacing
    `qwen-voice.md` Q10.
  - **Overturned by:** the UI's path parser (`speech-settings-model.ts:122-150`)
    learning to tolerate unknown rungs (then Groq could join the generic
    paths); or agent-server adding Groq to the voicing map.
  - **Owner:** coder (B1, B2).
- **D2 — What runs locally once speech is in the cloud?**
  - **Taken:** an app-owned hidden voice page (the existing `qwen-voice-page.ts`
    shell) running Silero VAD through `@ricky0123/vad-web` and gap-free Web
    Audio playback. No Python, no Gateway, no 16 GiB gate. Smart Turn is
    deferred.
  - **Rejected:** VAD in the main renderer (throttling outside `normal`,
    `window-mode.ts:143`; shared crash domain); energy-only VAD (false paid
    takes); keeping the Gateway WebUI (it keeps the Gateway).
  - **Commits us to:** two UI dependencies (ISC, MIT) with one WASM variant
    packaged; a page CSP with `wasm-unsafe-eval`; the page preload as a new
    contract.
  - **Overturned by:** measured page footprint far above ~200 MB; or VAD/echo
    cancellation failing Ben's live check.
  - **Owner:** coder (U1, U2).
- **D3 — How do replies fit Groq's 200-character cap?**
  - **Taken:** server-side segments (`voice/segments.py`, ≤200 characters, the
    first is one sentence) on `/voice/text`; `/voice/speech` per segment;
    one-ahead prefetch; sample-accurate scheduling; barge-in aborts the rest;
    cached phrase audio.
  - **Rejected:** split-and-join inside the rung (late first audio; pays for
    the unheard part); client-side segmenting (free text crosses, breaking V8).
  - **Commits us to:** additive `segments` and `segment` contract fields;
    `features.voice: 2`.
  - **Overturned by:** a Groq TTS model without the cap, or a streaming TTS
    endpoint.
  - **Owner:** coder (B2, U2).
- **D4 — What does voice do on a 429?**
  - **Taken:** a minute-scope 429 retries once (≤10 s); a day-scope 429 stops
    that direction with a strip message naming the reset time. No spoken
    error, no loop. `retry_after_s` and `remaining_requests` are carried in
    `APIError.details` by `stt/errors.py:358`.
  - **Rejected:** a new `APIError` field (one builder already serves both
    clients); silent skip (looks broken).
  - **Commits us to:** the `voice_rate_limited` code with `scope` and
    `retry_after_s`.
  - **Overturned by:** live 429s that lack `x-ratelimit-remaining-requests` (then
    `scope` falls back to the size of `retry-after`).
  - **Owner:** coder (B1, B2, U2, U3).
- **D5 — Key and disclosure.**
  - **Taken:** provider/namespace `groq`, API-key rows only, no `env_keys`,
    `speech_only`, `{stt, tts}`; a disclosure acknowledged per provider at
    enable, in the wording above; a provider label while listening.
  - **Rejected:** reusing `openai-key` (a different vendor's key); `env_keys`
    (ambient keys would light the rung, the defect `registry.py:737-746`
    names).
  - **Commits us to:** a new registry row and `login_catalog` copy; the
    acknowledgement in `voice-setting.json`.
  - **Overturned by:** Groq retention terms changing (re-read
    `docs/your-data`).
  - **Owner:** coder; copy reviewed by designer.
- **D6 — Keep the local rig?**
  - **Taken (recommended; Ben decides):** remove it (U4) after the cloud path
    passes Ben's live check. Key-free voice, if wanted, comes back as cascade
    rungs (`20a012742`, `cac51c33c`).
  - **Rejected:** keeping it as an opt-in (~3,300 lines and a second engine).
  - **Commits us to:** nothing until Ben answers. It is reversible from
    `89804f3b1`.
  - **Overturned by:** Ben requiring offline voice with natural STT quality.
  - **Owner:** manager (Ben's answer).
- **D7 — Slicing.**
  - **Taken:** B1 → B2 → U2 → U3 → (Ben's live check) → U4, with U1 alongside B1.
  - **Rejected:** one combined slice (the backend contract must freeze before
    the coordinator changes).
  - **Commits us to:** this order; live checks queued for Ben.
  - **Overturned by:** Ben allowing app launches (then QA adds live rounds per
    slice).
  - **Owner:** manager.
- **D8 — Groq's place in the order.**
  - **Taken (recommended; Ben decides, open question 1):** last, so the single
    frozen order is kept.
  - **Rejected:** first, for voice only (a second order beside
    `TTS_RUNG_PATHS`, "written HERE and nowhere else").
  - **Commits us to:** a Groq trial in an isolated config dir with only the
    Groq key.
  - **Overturned by:** Ben wanting Groq to serve even when paid rungs are
    signed in.
  - **Owner:** manager.
