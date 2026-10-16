# Design: per-session voice mode (desktop app)

Status: architect proposal, 2026-10-15. Proposed carrier: commit this file as
`docs/design/session-voice.md` on `feat/session-voice` (lo-voice) as the branch's
first commit. Base: local-operator 6dcb9f2ee, local-operator-ui a3d1198d2.

Requirement (operator, as updated): voice is enabled PER SESSION; one session at a
time, switchable; the session's OWN agent answers, with its context and tool calls;
not a separate voice model with its own personality. "Aida only first" is dropped.

## Decisions

- **V1 — Topology.**
  **Taken:** a pipeline. Push-to-talk audio goes to the lop server, which runs STT and admits the transcript as an ordinary prompt to the enabled session. When the turn ends, the server synthesises that session's final assistant message, and the renderer plays it.
  **Rejected:**
  - (b) Gemini Live as a front end. It is a second model that decides what to say and when to delegate. That is the "own personality" the operator ruled out. The ACP POC also measured 4 of 6 hand-offs correct with a small front-end model. A native-audio model cannot be held to reading text verbatim.
  - (c) Hybrid. Same objection: any generative audio layer can speak words the session agent never wrote.
  **Commits us to:** answers arrive at agent-turn speed (seconds to minutes), not conversational speed.
  **Overturned by:** the operator hearing measured pipeline latency and accepting a front-end model that paraphrases.
  **Owner:** manager.
- **V2 — Which process owns what.**
  **Taken:**
  - The renderer owns the mic and playback. `getUserMedia` audio is already granted at window-guards.ts:531-535, and `speech-store.ts` already plays audio.
  - The lop server owns every provider call and every key.
  - Electron main only relays bytes through the `desktop-media.ts` allowlist.
  **Rejected:**
  - Provider calls from the renderer. The CSP pins `connect-src` to loopback (renderer/index.html:42), and keys live in lop's encrypted store, not the app (speech-settings-model.ts header).
  - Main owning the provider. That would put a second credential holder beside the cascade.
  **Owner:** coder.
- **V3 — What "enabled" is.**
  **Taken:** one machine-wide slot, `<config>/voice/active.json` `{session_id, enabled_at, instance_id}`.
  - Its single writer is `local_operator/voice/slot.py`, under a file lock in the pattern of aida/state.py:213.
  - Enabling B atomically replaces A. That swap is the switch, and it returns `replaced`.
  - Every voice route refuses a session that is not in the slot (409 `voice_not_enabled`).
  **Rejected:**
  - Renderer-only state. Several windows, a remote-paired backend and the agent itself all need the fact.
  - A per-session flag. One-at-a-time would then need a scan-and-clear across sessions.
  **Commits us to:** the slot file is a contract that the runtime also reads.
  **Owner:** coder.
- **V4 — Aida.**
  **Taken:** no special case. Aida is an ordinary session with a seed role (aida/__init__.py:3-6), so per-session voice covers her.
  **Rejected:** "Aida first". Dropped by the operator.
- **V5 — Session context and tool calls.**
  **Taken:** the utterance is admitted through the same admission path as `POST /v1/desktop/sessions/{id}/messages` (desktop_sessions.py:3688).
  - It is sent as `input_mode="dictated"` with `input_path=<AudioPath>`.
  - It is sent as `mode="steer"` when the session is streaming, else `mode="prompt"` (Prompt.mode, desktop_sessions.py:1151).
  - *Amendment (slice 1, 2026-10-16):* `/messages` itself takes `mode` from its client; the server-side rule this reuses is `admit_receipt_request`'s (desktop_sessions.py:554): steer while `is_streaming`, else prompt. A pending gate is not a separate term: a parked approval sits inside a streaming turn, so it already steers, and the steer is queued until the gate settles (`tests/unit/voice/test_steer_during_approval.py`). The utterance calls the `/messages` handler itself rather than a copy of it.
  - Tools run in the session as usual and render on screen as usual.
  **Rejected:** a separate voice context. It would fork the conversation the operator wants to keep.
  **Owner:** coder.
- **V6 — The agent knows it is being heard.**
  **Taken:** a `voice` section in `[session-state]`. Add it to `STATE_SECTION_HEADINGS` (prompts_api.py:1306), and have the runtime compute it per turn from the slot, the same way as the interactivity probe (session.py:6530, 16413).
  - The text is byte-stable.
  - An explicit empty section is sent on disable.
  - Tail-journalled, so the prefix cache is kept (prompts_api.py:929-933).
  **Rejected:**
  - `input_mode`. It is silent by design and never reaches a provider (harness/types.py:640-648).
  - A daemon-side journal append. None exists (voice_spend.py:4-8).
  **Owner:** coder. Ships in slice 2.
- **V7 — Approvals.**
  **Taken:** voice never answers a gate.
  - Approvals are ANNOUNCED aloud from host-authored gate text, then approved on screen through the existing card and presence path.
  - The utterance route has no code path to `answer_gate`. While a gate is pending, an utterance is admitted as a steer, never as an answer.
  **Rejected:** a spoken "yes". A transcript is produced by a provider pipeline, and the mic hears the app's own TTS output. A model-written reply could therefore become an approval, which breaks approval-authority.md §0.
  - Spoken deny and answers to non-secret asks are ordinary under §1 and allowed in principle. They are deferred to slice 3.
  **Overturned by:** never by voice transcription alone. It would need a presence gesture.
  **Owner:** reviewer must check that no route reaches `answer_gate`.
- **V8 — Who decides what is spoken.**
  **Taken:** the server. `POST /v1/desktop/voice/speech` takes a *source reference*: a session message id, a pending gate, or an ask. It never takes free text. The server resolves the text, makes it speakable (`voice/speakable.py`), caps it, and synthesises it.
  **Rejected:** the renderer posting text to `/v1/tools/speech`. That would let any renderer bug speak words the agent never wrote.
  **Commits us to:** the server must read a message by id. The coder verifies this against the `/history` reader (desktop_sessions.py:3242).
  **Owner:** coder.
- **V9 — Provider abstraction.**
  **Taken:** extend the two cascades that exist rather than add a third abstraction.
  - STT (`stt/cascade.py`, order Radient→ElevenLabs→OpenAI→model sidecar): add `provider_stt_gemini` after OpenAI.
  - TTS (`tts/cascade.py`, order Radient→ElevenLabs→OpenAI): add `provider_tts_gemini`, then `local_tts_system` (macOS `say`, key-free, zero download).
  - Gemini uses the existing `google` provider key `GOOGLE_AI_STUDIO_API_KEY` (registry.py:659-666) via `/login google`. ElevenLabs is already rung 2 in both cascades.
  - Keys stay only in the encrypted store.
  - *Amendment (slice 1, 2026-10-16):* `local_tts_system` is OPT-IN per caller (`include_local=True`), used only by the voice routes. As an always-on rung it would turn `/v1/tts/paths` servable on every Mac with a token the UI's closed list does not know, and hand the mp3 `/v1/tools/speech` WAV bytes, since `say` cannot write mp3. `GET /v1/desktop/voice` reports `tts.path = local_tts_system` when that is the rung the voice routes will use.
  **Rejected:**
  - Inserting Gemini first. That would re-route every existing dictation user.
  - A new VoiceProvider interface. It would sit beside cascades that already own credential probing, fall-forward and spend.
  **Owner:** coder. Gemini ships in slice 2.
- **V10 — Key-free path.**
  **Taken:**
  - TTS: `say`. Measured here: 0.82-1.38 s to render a 5.0 s clip, at load average 4-6.
  - STT for tests: the utterance route accepts a `text` field instead of audio. It has the same authority as `/messages`.
  - Agent for tests: the `test` mock provider (registry.py:896-901), whose `[tool]` / `[bash:N]` markers drive real tool calls (providers/clients.py:4933-4995).
  - Real key-free STT is a slice-3 local rung. The Parakeet GGUF is already cached (697 MB, `~/.cache/huggingface/hub/models--handy-computer--parakeet-unified-en-0.6b-gguf`); whisper.cpp base.en is about 150 MB.
  **Overturned by:** the operator's daemon having no STT path (`GET /v1/stt/paths` → `none`). If so, the local STT rung moves into slice 1.
  **Owner:** coder.
- **V11 — The ACP POC.**
  **Taken:** reuse no code. Keep two lessons:
  - "Unanswered is not a denial" (acp-agent.md D2).
  - The desktop approve route is presence-gated, so voice cannot approve (acp-agent.md D1).
  **Rejected:** merging `feat/acp-poc`. It solves an external-front-end topology that V1 rejects.
- **V12 — Input mode.**
  **Taken:** push-to-talk only in v1, reusing `keymap.push_to_talk`. Hands-free (VAD, echo cancellation, barge-in) is deferred.
  **Rejected:** always-listening by default. Echo from the app's own TTS, accidental prompts to an agent that holds tools, and per-utterance STT spend.

## Frozen contract (backend ↔ UI)

Feature key: `features.voice: 1` (features.py). Everything below is gated on it.
Every route uses `require_desktop`.

- `GET /v1/desktop/voice` → `{active: {session_id, enabled_at} | null, stt: {path, available, reason}, tts: {path, available, reason}, settings: {max_spoken_chars}}`
- `POST /v1/desktop/voice`
  - Body: `{op: "enable" | "disable", session_id}`.
  - 200: `{active, replaced: session_id | null}`.
  - 404: `unknown_session`.
  - 409: `voice_unavailable` (no TTS path; STT absence is allowed because of the text door) or `voice_store_busy`.
- `POST /v1/desktop/voice/utterance` (multipart)
  - Body: `session_id`, `request_id`, and either `file` (wav/webm/ogg/mp4, ≤16 MiB) or `text`.
  - 200: `{text, path, admission: MessageAdmission}`.
  - 409: `voice_not_enabled`, or `stt_unavailable` with the cascade's reason.
- `POST /v1/desktop/voice/speech`
  - Body: `{session_id, source: {kind: "message", message_id} | {kind: "gate", request_id} | {kind: "ask", ask_id}}`.
  - 200: audio bytes, with headers `X-Voice-Path` and `X-Voice-Spoken-Chars`.
  - 409: `voice_not_enabled`. 404: `unknown_source`.
- Machine feed `/v1/desktop/events`: frame `type: "voice"`, `{active}`, on every slot change.
- The turn trigger stays on the existing per-session stream. The renderer speaks the last assistant `message_end` before `agent_end`, for a turn it admitted, deduped by message id across `after_seq` replays. It speaks a gate when `pending_gate.kind == "approval"` appears.
- Both STT and TTS calls go through `emit_voice_spend` with the session id.

## File plan

local-operator (`feat/session-voice`):
- New:
  - `local_operator/voice/__init__.py`
  - `voice/slot.py`
  - `voice/speakable.py`
  - `voice/state_block.py`
  - `server/routes/desktop_voice.py`
  - `server/models/desktop_voice.py`
  - `tts/local_say.py`
  - `stt/gemini.py` and `tts/gemini.py` (slice 2)
- Edit:
  - `server/app.py` (include router)
  - `server/features.py`
  - `tts/__init__.py` and `tts/cascade.py` (rungs)
  - `stt/__init__.py` and `stt/cascade.py` (slice 2)
  - `settings_io.py` (`voice.max_spoken_chars`, default 800)
  - `prompts_api.py` and `session/session.py` (V6, slice 2)
- Tests: `tests/unit/voice/`, `tests/unit/server/test_desktop_voice.py`.

local-operator-ui:
- New, under `src/renderer/src/features/voice/`:
  - `voice-controller.tsx` (mounted at the app root, so it survives navigation)
  - `voice-store.ts` (phase: idle | listening | sending | working | speaking | error)
  - `voice-turns.ts` (the speak-trigger rule)
  - `use-voice-status.ts`
  - `voice-toggle.tsx`
  - `voice-indicator.tsx`
- New tests: `scripts/voice-turns.test.mjs`, `scripts/voice-media-relay.test.mjs`.
- Edit:
  - `src/shared/desktop-contract.ts`: `voice.status` and `voice.control`, mirroring `aida.*` at :3368, :7708.
  - `src/main/desktop-media.ts`: `voice.utterance` (upload) and `voice.speech` (download).
  - `use-speech-to-text-manager.ts`: a VOICE priority, so the PTT hold sends an utterance when the viewed session holds the slot.
  - `src/renderer/src/dev-driver/install.ts`: a `voiceUtterance(text)` verb. It never answers a gate.
  - i18n catalogues.

## Order

1. Backend PR: slot, routes, `say` rung, feature key.
2. UI PR: may be built in parallel once this contract is frozen; it gates on the key.
3. Slice 2: V6, Gemini rungs, speaking asks.
4. Slice 3: local STT rung, spoken deny and ask answers, hands-free.
