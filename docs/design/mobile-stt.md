# Mobile speech-to-text: capability-aware mic, BYO providers, the silent annotation

Status: implemented, 2026-09-28 (branch `feat/mobile-speech-to-text`). This is the
durable half of the design: the contracts another slice codes against, the gates
a change here has to pass, and the boundaries this change deliberately leaves
open. The full exploration (rendered frames, QA sketch, research citations) lives
with the session that produced it; nothing here needs an image to be true.

## 1. Why

The phone is the surface where typing is worst, and the daemon is already the
only process that can hold a provider key (the browser must never see one). So
voice input is: **record on the phone, transcribe on the daemon, append to the
draft**. No streaming, no audio kept after the request, no new auth flow.

Two constraints shaped the whole design:

1. **Providers are rows, not code paths.** Radient is the incumbent; ElevenLabs
   and OpenAI are bring-your-own-key additions; SuperWhisper has **no
   audio-in/transcript-out API at all** (research, 2026-09-28) and is therefore
   permanently excluded with that honest reason rather than shipped as a
   "coming soon" that would never come.
2. **The annotation must be silent.** How a message was produced
   (`typed`/`dictated`/`mixed`) and which voice path produced it belong to the
   message, not the UI. Nothing renders differently; a reader who does not know
   the vocabulary sees exactly the frames it saw before.

## 2. Contracts

### 2.1 The annotation (input-mode-v1)

- `input_mode: typed | dictated | mixed` and `input_path: <resolver token>` are
  **optional strings** on the send envelope, beside `text`/`images`, and are
  part of its immutable identity: a retry replays the stored bytes, annotation
  included. Absence is the legacy reading for every producer that knows nothing
  of the vocabulary — absence, not an empty string, is what is sent, and a
  malformed present value is **refused (422), never coerced or dropped**,
  because the value lands on a durable row.
- Provenance is **history-based and sticky per draft window**: once the window
  since the draft's last accepted send/clear has seen typing or a dictation, it
  stays seen; `mixed` is both seen. The empty draft resets the window. Dictated
  *spans* exist only to name the `input_path` of the most recent dictation still
  present in the sent text.
- Wire gate: the attach-record capability token `input-mode-v1`, advertised by
  the runtime only when its handle can store the fields. Two layers apply it and
  they must agree: the relay strips the fields for an uncapable owner
  (`MobileDaemon.request`), and the attach client strips them where every
  control frame is written (`AttachClient._request_frame` →
  `_strip_unsupported_annotation`) — so no door can carry the pair to an owner
  that did not advertise it, including the wake path's
  `request_ack_with_duplicate`, which never passes the builders (agent review
  round 1, R1-1). The mobile side never gates on the HTTP `features` key.

### 2.2 The provider table

`clients/stt.py::STT_BACKENDS` is the single dispatch surface; a provider is a
row, not a branch, and the resolver owns the token vocabulary:

| token | execution | notes |
|---|---|---|
| `provider_stt_radient` | in-tree adapter over the shipped `RadientClient` | credentials via `resolve_radient_credential` |
| `provider_stt_elevenlabs` | the cascade session's BYO executor | `xi-api-key`, `/v1/speech-to-text`, `scribe_v2` |
| `provider_stt_openai` | the same executor | `/v1/audio/transcriptions`, `gpt-4o-transcribe` family |
| `provider_stt_superwhisper` | never (`servable=False`) | "SuperWhisper has no transcription API." |
| `model_audio_sidecar` | pending (`servable=False`) | the native model-audio rung |

Availability is the resolver's answer **filtered for executability** — the
baseline (first executable provider with a persisted credential) only runs on a
tree where the cascade module is absent; once it exists, its answer wins. The
result is TTL-cached (~30 s) and never raises: a broken resolver degrades to
`{available: false, reason: …}`.

### 2.3 `POST /api/transcribe`

- Multipart `audio` (+ optional `language`/`prompt`/`model` for
  forward-compat), the shared `gate()` (401 / cross-origin 403).
- **413** above 20 MB (declared `Content-Length` refuses before parse; the
  read bytes are the real bound), **422** for missing/empty part or a media type
  outside the allowlist (`audio/mp4`, `audio/webm`, `audio/ogg`, `audio/mpeg`,
  `audio/wav`, `audio/x-m4a`, `audio/aac` — the bare type; codec parameters are
  accepted).
- **200** `{text, provider, model, path}` — `path` is the token that ACTUALLY
  ran; the client stores it as the envelope's `input_path` and must not
  re-derive it from a cached capability.
- **402** for a quota refusal (Radient's own refused balance or a provider
  credit marker, same sentences as the desktop route), **502** for everything
  else upstream (transport failures pass the client's own text through),
  **503** `stt_unavailable` (the fixed sentence; the mic is hidden in that
  state, so this is the race-answer), **500** only for genuine internal faults.
  Upstream bodies are scrubbed at the client: the daemon never surfaces a raw
  upstream payload.

### 2.4 Capabilities

`capabilities: {features, stt}` rides `/api/sessions` and the `sessions` frame
(both transports, one builder). The mic is shown iff `stt.available` **and** the
browser can record one (secure context + `getUserMedia` + `MediaRecorder`); the
old plain-HTTP host deliberately shows nothing. Absence and `available: false`
mean the same thing: hide.

### 2.5 Web behaviour

- The composer's resting height is stable (D1): the placeholder is sized to fit
  the field's narrowest resting width beside the mic, so the empty and one-line
  states are equal and the first keystroke does not reflow the composer.
- Record → stop → **append** to the current draft (`joinDraft`; one separator,
  never a clobber); insertion never calls `focus()` — the appended span is
  revealed by scrolling the field, not by focusing it (U1).
- The 120 s cap **stops and transcribes**; explicit cancel discards and sends no
  request, and an in-flight transcription is aborted for real (U5); `send()`
  cancels an in-flight dictation, aborts its request, discards its result, and
  says so in the status row (U2).
- Outcome lines, in the same polite status row the live states use: a landed
  transcript announces `Transcript added` (U3; focus is deliberately not
  returned to the field — a programmatic focus pops the iOS keyboard), an empty
  transcript answers `Didn't catch that — try again.` (D2), and a cancelled
  dictation notes `Voice input discarded.` (U2).
- Failure copy: the daemon's own sentence for 402/413/422/503; a retry sentence
  for 502/transport. 401 keeps the shared reload rule.
- Voice controls are `size-11` with per-state `aria-label`s; the status row is
  `role="status"` (polite) and holds one height across recording / transcribing
  / outcome states (D3), so stopping a recording does not move the line; the
  pulse only animates under `prefers-reduced-motion: no-preference` — the word,
  dot and timer carry the state without it.

## 3. Boundaries and non-goals

- No audio capture outside the web view; no tray/desktop mic in this slice.
- No client-side transcoding: the recorded blob's own MIME type rides the POST,
  and the server allowlist is kept wide enough to accept all recorder defaults.
- No OAuth for any STT provider: all integrations are daemon-side API-key paste.
- The `local-operator-ui` (desktop) repo is untouched.
- The cascade resolver's symbol and call shape are consumed through **one
  adapter** (`mobile/stt.py`), sync-or-async tolerant; when the resolver lands,
  the adapter absorbs any drift and no other file learns the module's name.

## 4. Gates and evidence plan

- Python: flake8 + black (pinned) + isort on the changed set; pyright via the
  bounded runner; the whole-tree unit suite exactly as CI (isolated roots).
- Web: `pnpm build` (tsc) and `pnpm test`.
- End-to-end: an isolated daemon (`HOME` + `LOCAL_OPERATOR_CONFIG_DIR` in a
  fresh root, every `CMUX_*`/`LOP_*` unset) answering `POST /api/transcribe`
  over curl with a real generated clip, covering 401/403/413/422/503 and the
  success shape; the provider leg exercised against the table with a stubbed
  backend, since no BYO key resolves in an isolated root (the Radient leg's
  credential path is covered by unit tests and the desktop route's own suite).
- QA (owned by the QA pass, sketched here so the seams are known): the browser
  recorder path needs a real engine — a heavy pilot with
  `--use-fake-device-for-media-stream --use-fake-ui-for-media-stream
  --use-file-for-fake-audio-capture=<wav> --disable-features=AudioServiceOutOfProcess`
  for the success cell. WITHOUT the last flag the fake stream resolves but reads
  silence: measured RMS 0 over 5 s on Chrome 154, against RMS peak 128 / avg 60
  for the same wav with it — a silent mic that reads as an empty-transcript bug
  rather than a harness fault (QA round 1, Q1). The refusal cell uses
  `Browser.setPermission` (denied) with the descriptor `{"name": "microphone"}`
  — Chrome 154 rejects `audioCapture` with `Invalid PermissionDescriptor name`.
  The mime picker's iOS reality is a device check.

## 5. Known residuals

- The daemon reads the encrypted store for availability (cached); first read may
  start the broker. Watch during rollout.
- The `input_path` singular field records the most recent dictated span when a
  draft has several; multi-path drafts are out of scope for v1.
- `speech_only` providers are excluded from catalogues/ranking/failover exactly
  as `decision_only` ones are; if a future speech provider gains chat, it needs
  the flag split, not a new exception list.
