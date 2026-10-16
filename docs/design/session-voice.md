# Design: per-session voice mode (desktop app)

Status: architect proposal, 2026-10-15. Proposed carrier: commit this file as
`docs/design/session-voice.md` on `feat/session-voice` (lo-voice) as the branch's
first commit. Base: local-operator 6dcb9f2ee, local-operator-ui a3d1198d2.

Requirement (operator, as updated): voice is enabled PER SESSION; one session at a
time, switchable; the session's OWN agent answers, with its context and tool calls;
not a separate voice model with its own personality. "Aida only first" is dropped.

## Decisions

- **V1 — Topology.** *(SUPERSEDED BY R1 (pending): R1 is PROPOSED and awaits the operator; until it is accepted, V1 stands. Relay record 2026-10-16.)*
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
  - *Amendment (slice 1, 2026-10-16):* `/messages` itself takes `mode` from its client; the server-side rule this reuses is `admit_receipt_request`'s (desktop_sessions.py:554): steer while `is_streaming`, else prompt. A parked approval sits inside a streaming turn, so a steer is what `/messages` would send there. That steer never ANSWERS the gate, but it is not always queued behind it either: it is held until the card settles only for a non-interruptible tool (`write`, `tests/unit/voice/test_steer_during_approval.py`). For an interruptible tool (`bash`) the steer cancels the card and the call is skipped ("Tool call skipped: interrupted by steering.", QA round 1, Q3). The voice route therefore never steers into a pending approval: see V13. The utterance calls the `/messages` handler itself rather than a copy of it.
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
  - Approvals are ANNOUNCED aloud from the gate's text as the server resolves it (the tool name and the tool's approval summary), then approved on screen through the existing card and presence path. *Amendment (slice 1, round 1):* that summary is not host-authored prose: it can quote model-written arguments (a bash command, a path). It is the same text the card shows, so speaking it reveals nothing new.
  - The utterance route has no code path to `answer_gate`. While a card is pending, an utterance is refused (V13), never admitted as an answer.
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
- **V13 — Utterances while a card is pending.** *(Added slice 1, round 1, from QA Q3; widened and the race recorded in round 2.)*
  **Taken:** while the slot session has ANY pending card (an approval, or a blocking ask), `POST /v1/desktop/voice/utterance` refuses with 409 `approval_pending` ("Answer the question or approval on screen first.").
  - It covers the text door and audio alike.
  - Audio is refused BEFORE STT runs, so a refused recording spends nothing. The check repeats after STT, because a card can park while a transcription is running.
  - Asks: a QUEUED ask (the default) parks no card, so it never refuses. A BLOCKING ask (`LOP_ASK_NONBLOCKING=0`) parks one, and the `ask` tool is interruptible, so a steer withdraws it exactly as it skips `bash`. Refusing every pending card was the smaller correct choice, smaller than a per-kind rule that would have to know which tools are interruptible. Spoken answers to asks stay slice 2/3.
  **Known race, accepted (round 2, review N2):** the route reads the facade's pending card, and the steer is a separate owner op. A card that parks between the two (reproduced at 1-4 ms) is not seen, and the steer cancels it, exactly as a typed steer does today. The safety property still holds: nothing is approved and the tool does not run (the gate's future is cancelled, never resolved to approve). `tests/unit/voice/test_utterance_during_approval.py::test_the_known_race_admits_a_steer_but_never_approves` pins that property as one that must never change, and pins the card-skip as current behaviour that may improve.
  - Why it is not closed here: an owner-side "refuse if a card is pending" option would have to be carried through six layers (the `/messages` `Prompt` body, `AttachedSession.admit_prompt`, the attach client's frame, the runtime server's `steer` dispatch, `ServingSessionHandle.steer`, and the TUI handle's steer). It would also need a capability flag, because an older owner ignores an unknown frame key and would admit the steer anyway. That is more than the contained edit this feature should carry.
  **Rejected:**
  - Admitting the steer (V5 as first written). A steer cancels an interruptible tool's card, so a word the mic heard, including the app's own announcement of the card, would make the card vanish and skip the tool. Not an approval, but the operator loses the decision.
  - Queuing the utterance in the daemon until the gate settles. That is a second admission queue beside the owner's, with its own lifetime and replay rules.
  - Changing steer semantics for every surface. Typed steers behave the same way today (QA 13b); that is a harness decision outside this feature.
  **Overturned by:** the owner-side refusal landing, which closes the race; the pinned test's "current behaviour" half then flips to a 409.
  **Owner:** coder.
- **V14 — Binding a spoken take to its turn.** *(Added slice 1, round 3.)*
  **Taken:** a take is bound to the turn whose stream carries the user `message_start` with `message.id ==` the take's `request_id`; timing never binds. The id equality is pinned by `tests/e2e/test_voice_take_binding_e2e.py::test_a_take_binds_to_the_user_message_start_whose_id_is_its_request_id`.
  - It holds for every admission path: a take admitted as a prompt, a take steered mid-turn (delivered at a tool boundary), and a take steered into a turn's tail and carried by `_flush_parked_deliveries` after the turn ends. It holds live and on an `after_seq` replay.
  - `admission.command_id` in the utterance's 200 is the same UUID.
  - The test discriminates both id carriages, proven by mutation in review round 6: a steer that drops `message_id` fails at index 1, and a prompt that drops it fails at index 0.
  **Rejected:** binding by arrival order or by "the next turn after the POST". A steer can land at a tool boundary, at the turn's end, or in the next turn, and a replay re-delivers frames out of wall-clock time.
  **Owner:** coder (backend pin), UI (the binding).
- **V15 — Turn generation guards an absorbed bound turn (UI).** *(Added slice 1, round 6; backend half filled in round 7, R7-M1.)*
  **Taken:** the UI's generation guards (an idle or other-generation snapshot unbinds a take; an `agent_end` older than the live generation is dropped) rely on ONE counter. Snapshots, `frontend.update` frames and `agent_start`/`agent_end` all publish the generation of the last `agent_start` the session emitted (`Session._turn_generation`, read by `frontend_state.py`), never the raw per-run `_generation`.
  - The raw counter also counts post-compaction continuation runs (`_drain_continuation` re-enters `_run_turn`), and those stay the same turn. Before the fix, a snapshot taken during a continuation said N+1 while that turn's `agent_end` said N, so the guard dropped the take's own answer.
  - Between turns the published value stays at the last announced start. The next start is always above it, because `_run_turn` bumps the raw counter first.
  - Pinned by `tests/unit/session/test_published_generation.py::test_the_snapshot_publishes_the_turn_events_generation_through_continuations` (one and two continuations; both fail before the fix) and `::test_the_next_turn_after_a_continued_one_still_moves_the_generation_forward`.
  **Rejected:** publishing `_logical_generation`. It is cleared at `agent_end`, so the post-end refresh fell back to the raw counter and a live update still said N+1.
  **Owner:** coder (backend counter), UI (the guards).

## Relay front end (R1–R13)

*Added 2026-10-16 from the architect's relay record. The R rows replace V1 and change V2, V9, V10, V11 and V12 as each row says; every V row not named stays in force. R1 and R4 are PROPOSED and await the operator (R4: who speaks replies; and the order of slices A and B), so nothing built so far depends on them.*

The relay is a FRONT END bolted onto the slice-1 pipeline, not a replacement for it. Slice 1 already does the back half: the one-session slot (V3), the utterance door (V5), take-to-turn binding (V14/V15), exact reply speech by source reference (V8) and no approval by voice (V7/V13). The live model's only new job is to HEAR a take and decide RELAY or CONTROL.

- **R1 — Topology (replaces V1).** **PROPOSED — pending operator decision.**
  **Proposed:** a pluggable front end hears one push-to-talk take and classifies it. A RELAY take's words enter through `POST /v1/desktop/voice/utterance`. A CONTROL take is handled by the client. The session does the work, and replies are recited exactly through `/voice/speech`.
  **Why:** the operator's request, with V1's objection removed by R4.
  **Overturned by:** R13's measurement favouring the model's own voice for content (not expected).
  **Owner:** manager.
- **R2 — Who owns what (changes V2).**
  **Taken:**
  - Electron main owns the Live WebSocket (`src/main/voice-live.ts`).
  - The lop server holds the Google key and mints a single-use, config-locked ephemeral token (`POST /v1/desktop/voice/live-token`, `voice/live_token.py`, the body built in `voice/live_config.py`).
  - The renderer owns the mic and playback and streams PCM16 at 16 kHz to main over IPC (32 KB/s).
  - The token never crosses IPC, the rule desktop-stream.ts:12-13 already sets for the bearer. The route refuses a browser-shaped caller (`Sec-Fetch-Site` present) with 403 `live_main_only`.
  **Rejected:**
  - The renderer owning the socket: its CSP deliberately has no ws/wss (index.html:20-26,42), and the token would sit in the renderer.
  - The lop server owning it: no bidirectional audio transport exists (only SSE plus request/response media, desktop-media.ts:177-210), so it would add a transport and a hop.
  - A sidecar: one more lifecycle, no gain.
  **Commits us to:** a main-side socket client, and a token mint per Live session (one minute to open, thirty to use).
  **Overturned by:** a requirement for voice on mobile web. The server would then own the socket.
  **Owner:** coder.
- **R3 — Inbound words: the transcript is authoritative.**
  **Taken:** admitted text is the provider's own `inputTranscription` for the take (activityStart..activityEnd), not the model's tool argument. The tool's `text` is the fallback when the transcript is empty, and a drift metric otherwise. Provenance is recorded as the message's `input_path` through `/utterance`'s `text_origin`: `gemini_live_transcript` | `gemini_live_tool_arg`, or the STT rung for a recording. The local front end uses server STT.
  **Rejected:** admitting the tool argument. It is model output, so it is paraphrase-prone; a transcript is recognition output. The admitted text shows on screen as the user's message, so recognition errors are visible.
  **Commits us to:** a closed `text_origin` set (anything else is 422 `invalid_voice_request`), refused alongside a recording.
  **Overturned by:** `inputTranscription` arriving after the tool call, or missing, on 3.8 Live (unverified). R3 then degrades to the tool argument; if more than 5% of takes need the fallback, show the admitted text for a 2 s cancel window before sending.
  **Owner:** coder.
- **R4 — Outbound speech: exact recitation, model voice muted.** **PROPOSED — pending operator decision.**
  **Proposed:** session replies, gate announcements and asks are spoken ONLY by `/voice/speech {source}` (V8): server-resolved text → `speakable.clean/cap` → the TTS cascade. The Live model's audio output is discarded by main, and its tool responses use `scheduling: SILENT`. Session text is never sent to the Live model, so it has no content to paraphrase. Google's docs make the same split: TTS "is tailored for … exact text recitation", while Live is for "unstructured" audio.
  **Residual:** none for content. The remaining risks are a misheard take (R3) and a wrong RELAY/CONTROL call (R6).
  **Owner:** reviewer (checks that no path speaks free text or reaches `answer_gate`).
  *Built so far without depending on R4:* the system instruction and tool list live as data in `voice/live_config.py`, so an R4 change is a data edit there plus its pinned test.
- **R5 — Filler and phrases: a closed catalogue, triggered by the client.**
  **Taken:** a new `/speech` source, `{kind: "phrase", key}`, with keys from `wire.voice.phrase.*` only:
  - `passing_on` (on a 200);
  - `still_working` (every 20 s while a take is owed, at most 3);
  - `on_screen_first` (on 409 `approval_pending`);
  - `didnt_catch` (empty transcript);
  - `switched` (slot change);
  - `nothing_to_repeat`.
  The server renders the catalogue sentence; an unknown key is 404 `unknown_source`, so the source cannot carry free text. Audio is cached per TTS path on enable, so an ack costs no synthesis time.
  **Rejected:** model-authored filler, which would be the "own personality" channel again.
  **Commits us to:** the six keys as the contract; a new phrase is a catalogue key plus a contract change.
  **Overturned by:** R13 passing (slice C), which could let the model voice short acks.
  **Owner:** coder.
- **R6 — Controls never reach the session.**
  **Taken:** the closed set is `stop_speaking` (also any PTT press), `repeat_last` (replays cached bytes), `louder` and `softer`.
  - Gemini reports them through the `voice_control` tool.
  - The local front end matches a whole-utterance grammar in `voice/controls.py`, exact phrases only, so "stop the build" relays. `/utterance` with `controls=1` answers `control: action` and `admission: null`; nothing is admitted and no receipt is written.
  - Everything else is relayed. If no tool call arrives 3 s after activityEnd, the transcript is relayed.
  **Rejected:** keyword or prefix matching, which would turn a sentence for the agent into a client action; aborting a session turn by voice (a misheard "stop" would kill tool work), so aborting stays on screen.
  **Commits us to:** controls being said plainly.
  **Overturned by:** measured false-relay rates on real takes that the exact grammar cannot fix.
  **Owner:** coder.
- **R7 — Approvals and asks (keeps V7/V13).**
  **Taken:** the Live tool schema has no gate or answer tool, and the schema is locked into the token (R2). "Yes" is relayed like any take, gets 409 `approval_pending`, and the client speaks `on_screen_first`. Gates are announced through the existing `{kind: "gate"}`. Spoken deny and ask answers stay deferred (V7 §1). `tests/unit/voice/test_live_token.py` pins the tool set to exactly `relay_to_session` and `voice_control`.
  **Rejected:** any spoken answer path.
  **Commits us to:** approval on screen only.
  **Overturned by:** the V7 §1 deferral being lifted for spoken deny.
  **Owner:** reviewer.
- **R8 — Providers.**
  **Taken:**
  - `front_end` ∈ {`local`, `gemini_live`}, with `elevenlabs` later, behind one main-side interface: `start(take) / audio(chunk) / end() → {relay, transcript, toolText} | {control}`. It is not a cascade rung: V9's cascades stay STT/TTS only. Chosen on enable (`front_end`, 409 `front_end_unavailable` when unavailable) and persisted as `voice.front_end`.
  - The key-free STT rung is `local_stt_loopback`: the existing OpenAI client with `base_url` (stt/clients.py), pointed at a loopback OpenAI-compatible `/v1/audio/transcriptions` (`voice.local_stt_url`, empty = off; loopback hosts only, checked at write and at call time). It is opt-in like `include_local` for TTS: only the voice routes pass it, so `/v1/stt/paths` is unchanged. It is walked after the credentialed rungs, sends a placeholder bearer (never a stored key), and pins the model `mlx-community/parakeet-tdt-0.6b-v3`. lop gains no dependency.
  **Rejected:** the Qwen rig as a dev or QA path. It is dev-exploratory only: it relayed 4/6 (README:135), which breaks ⅓ of turns.
  **Commits us to:** the operator starting a local server (`mlx_audio.server`) for the key-free path.
  **Overturned by:** a bundled in-process STT model.
  **Owner:** coder.
- **R9 — The Live session is disposable.**
  **Taken:** it holds no session context (R4), so it is disposable: connect on the first PTT press, close after 5 min idle, use sessionResumption plus slidingWindow compression across the ~10 min connection resets, and mint a fresh token at the 30 min expiry. A slot switch retargets the relay without reconnecting. The token is `uses: 1`; resuming does not count as a use (API reference, `AuthToken.uses`).
  **Rejected:** a long-lived session carrying context.
  **Commits us to:** a reconnect path in main.
  **Overturned by:** R4 being decided so the model must carry session content.
  **Owner:** coder.
- **R10 — Input mode (keeps V12).**
  **Taken:** push-to-talk in every front end. Gemini runs with `automaticActivityDetection.disabled` and manual activityStart/End. The mic is closed while replies play, so there is no echo self-relay. Hands-free (VAD, echo cancellation, barge-in) is deferred.
  **Rejected:** server-side activity detection in v1.
  **Commits us to:** the client sending activity signals.
  **Overturned by:** hands-free (slice D).
  **Owner:** coder.
- **R11 — The slice-1 pipeline is kept** as the `local` front end: the key-free path and the fallback.
  **Taken:** nothing falls back silently from `gemini_live` to `local`; the indicator names the active front end. `local` is always listed available (its text door needs no STT rung); its reason is the STT state.
  **Rejected:** silent fallback, which would hide where the operator's speech went.
  **Commits us to:** an explicit front-end chip in the UI.
  **Overturned by:** nothing planned.
  **Owner:** coder.
- **R12 — Data disclosure.**
  **Taken:** on the free tier Google may use inputs and human reviewers may read them (Gemini API terms, "Unpaid Services"). Under R4 only the operator's own speech goes to Google, never session output. Enabling `gemini_live` shows that sentence once. A Gemini TTS rung (slice C) would send reply text too, so it is opt-in and disclosed.
  **Rejected:** enabling Live without the disclosure.
  **Commits us to:** a one-time disclosure in the UI.
  **Overturned by:** a paid-tier key with different terms.
  **Owner:** manager.
- **R13 — Unmuting model acks is a measured experiment, not a default.**
  **Taken:** slice C may let the model voice its own short ack, with `outputAudioTranscription` checked against the phrase set (≤6 words, no digits or paths) and logged as `voice.drift`. Keep it only if 0 of ≥30 scripted takes produce a self-answer.
  **Rejected:** unmuting by default.
  **Commits us to:** the output transcription being enabled in the locked setup (it is, in `voice/live_config.py`).
  **Overturned by:** the experiment failing (then acks stay R5 phrases).
  **Owner:** manager.

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
  - `request_id` is the take's BINDING KEY (V14): the user `message_start` that admits it on the session stream carries `message.id == request_id`, and the 200's `admission.command_id` equals it.
  - `request_id` is a dashed lowercase UUID, the `RequestID` type of `/messages` (`^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$`). It is the receipt journal key, so a retry reuses it and a new utterance mints a new one. *(Stated in slice 1, round 1; the type was always `/messages`' own.)*
  - 200: `{text, path, admission: MessageAdmission}`.
  - 409: `voice_not_enabled`, `approval_pending` (V13: any pending card), or `stt_unavailable` with the cascade's reason.
  - 422: `invalid_voice_request` (a malformed `request_id`, both or neither of `file`/`text`, an unsupported recording, or a transcript `/messages` would refuse, such as a whole-draft slash command), or `empty_transcript`.
- `POST /v1/desktop/voice/speech`
  - Body: `{session_id, source: {kind: "message", message_id} | {kind: "gate", request_id} | {kind: "ask", ask_id}}`.
  - 200: audio bytes, with headers `X-Voice-Path` and `X-Voice-Spoken-Chars`.
  - 409: `voice_not_enabled`. 404: `unknown_source`.
- Machine feed `/v1/desktop/events`: frame `type: "voice"`, `{active}`, on every slot change.
- The turn trigger stays on the existing per-session stream. The renderer speaks the last assistant `message_end` before `agent_end`, for a turn it admitted, deduped by message id across `after_seq` replays. It speaks a gate when `pending_gate.kind == "approval"` appears.
- Both STT and TTS calls go through `emit_voice_spend` with the session id.

### Relay additions (R2, R3, R5, R6, R8)

- `GET /v1/desktop/voice` adds `front_ends: [{id: "local" | "gemini_live", available, reason}]` and `settings.front_end`.
- `POST /v1/desktop/voice` `{op: "enable", session_id, front_end?}`. 409 `front_end_unavailable` (the slot and setting do not move). An enable without `front_end` keeps the configured one.
- `POST /v1/desktop/voice/utterance` adds optional `text_origin` ∈ {`gemini_live_transcript`, `gemini_live_tool_arg`} (with `text` only; recorded as `input_path`) and `controls=1` (the local grammar). The 200 adds `control: action | null`. A control take has `admission: null`, and nothing is admitted or journalled.
- `POST /v1/desktop/voice/speech` source adds `{kind: "phrase", key}` with `key` ∈ {`passing_on`, `still_working`, `on_screen_first`, `didnt_catch`, `switched`, `nothing_to_repeat`}. An unknown key returns 404 `unknown_source`, so free text stays impossible.
- `POST /v1/desktop/voice/live-token` (Electron main only; a browser-shaped caller gets 403 `live_main_only`): 200 `{token, expires_at, new_session_expires_at, model}`. 409 `live_unavailable` when no Google AI Studio key is stored (`lop login google` rows, the stored `GOOGLE_AI_STUDIO_API_KEY`, or the exported variable); 502 `live_failed` when Google refuses or cannot be reached. The locked setup (model, system instruction, tools, AUDIO, input and output transcription, resumption, sliding window, manual activity) is built only in `voice/live_config.py` and never logged.
- Live tools (behavior `NON_BLOCKING`; the client's response is always `{status}` with `SILENT` scheduling): `relay_to_session {text}` and `voice_control {action: "stop_speaking" | "repeat_last" | "louder" | "softer"}`.
- Settings: `voice.local_stt_url` (default `""` = off; loopback only) and `voice.front_end` (default `local`).

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

## Not addressed

- **The published user `message_start` carries `input_mode: null`** even though the journal row says `"dictated"` (found slice 1, round 3; also on origin/main `0301a5731`). The runtime emits the field: `Session._emit` and `ServingSessionHandle.subscribe_events` both carry `"dictated"`. The runtime then strips it per connection at `local_operator/session/runtime/server.py:8232` (`if not conn.input_metadata: frame = _strip_input_from_relay_frame(frame)`, the strip at `:479`), because `conn.input_metadata` (`:3894`) is set only when the viewer's auth frame says `input_mode: true`, and the desktop viewer's `AttachClient` auth (`local_operator/mobile/attach_client.py:1207-1255`) never sets it. The renderer cannot tell a dictated row from a typed one on the live stream. Voice does not depend on it (V14 binds by id). The fix is a one-line negotiation in the attach client plus a test, outside this feature.

## Order

1. Backend PR: slot, routes, `say` rung, feature key.
2. UI PR: may be built in parallel once this contract is frozen; it gates on the key.
3. Slice 2: V6, Gemini rungs, speaking asks.
4. Slice 3: local STT rung, spoken deny and ask answers, hands-free.
