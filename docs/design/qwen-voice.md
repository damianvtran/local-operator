# Qwen voice: one lop session as the voice of the Qwen Audio Agent

Status: slice 1 (backend). Source: the architect's record "Relay voice v2: the
Qwen rig is the voice, and the lop session gives the answers" (2026-10-16, EDT),
(`~/.local-operator/sessions/206635829b1d/scratchpad/relay-voice-design-v2.md`
on the machine it was written on), which holds the file:line evidence for every
decision below. This page is the
in-repo record of its decisions and its frozen contract; it does not restate
the evidence.

## What it is

The operator talks hands-free to the Qwen Audio Agent rig (Gateway WebUI,
speech-to-speech with Silero VAD, Smart Turn, Parakeet and Qwen3-TTS) that runs
on his machine. The rig passes his words, verbatim, to ONE voice-enabled lop
session, and Qwen3-TTS speaks that session's reply word for word. One voice
session at a time, switched from a button on each session.

Lop's own speech-to-text and text-to-speech (`local_operator/stt`,
`local_operator/tts`, `/v1/stt`, `/v1/tts`) are not part of this feature and
are not changed by it. Hearing and speaking belong to the rig; lop receives text
and returns text.

## Decisions (Q1–Q12)

- **Q1 Topology.** Gateway + s2s + a launcher in the desktop app + a relay LLM +
  a lop-slot backend adapter. Rejected: rebuilding VAD, duplex audio and
  barge-in in lop or the UI (a second voice stack); `lop acp` with an attach
  mode (drops updates outside a prompt, paraphrases results).
- **Q2 No language model in the rig's loop.** The Qwen3-4B is removed. A
  deterministic relay endpoint answers a user turn with
  `spawn_thinking({objective: <last user message, byte for byte>})`, a tool
  result with empty text, and anything unrecognised with empty text. The
  failure mode is silence. The admitted text is the transcript.
- **Q3 Verbatim speech goes through a token.** The rig's provider puts only
  `⟦lop-say:<uuid>⟧` into the instructions; the relay endpoint looks the token
  up and streams exactly the registered text. An unknown token produces empty
  output. The only edits to spoken text are s2s's own markdown/unspeakable
  stripping and lop's `speakable` cap.
- **Q4 Replies are spoken by the bridge, not by the Gateway's task result.** A
  Gateway task covers admission only (it resolves once `/utterance` answers),
  because Gateway tasks run one at a time per owner and a whole turn would hold
  every later take.
- **Q5 Target, auth and binding.** The target is always the V3 slot session,
  through the text door of `POST /v1/desktop/voice/utterance` (V5). The desktop
  bearer stays in Electron main and never reaches the Gateway process. Binding a
  take to its turn stays V14/V15, in the renderer.
- **Q6 Approvals.** Gates are announced from `/voice/text {kind: gate}`. A
  spoken "yes" gets 409 `approval_pending` (V13), and the rig says the
  "on screen first" phrase. Voice never answers a gate (V7).
- **Q7 Filler is a closed set of phrases in the launcher:** `sent`,
  `on_screen_first`, `voice_off`, `switched`, `failed` (and an optional
  `still_working`, slice 3). No model-written filler exists anywhere, and lop
  carries no phrase catalogue.
- **Q8 Hands-free full duplex**, as the rig already does it, with an echo guard
  in slice 2.
- **Q9 Where the operator talks.** Slice 2: the Gateway WebUI opened by the
  app. Slice 3: an in-app window. The per-session toggle sets the slot; the
  first enable starts the rig.
- **Q10 No lop speech code.** The server resolves what is spoken by source
  reference (V8), but RETURNS text from `/voice/text` instead of synthesising
  it. There is no recording upload, no transcription, no synthesis, no TTS
  precondition and no live token on the voice routes.
- **Q11 Fresh branches.** `feat/qwen-voice` cut from `origin/main`; only the
  carried pieces move across from the earlier `feat/session-voice` work.
- **Q12 The rig directory supplies the runtime only** (its virtualenv, node
  modules and model cache), located by the UI setting `voice.qwen_rig_dir`. The
  integration code lives in local-operator-ui.

## Frozen contract

Feature key `features.voice: 1` on `GET /v1/capabilities`. Every route requires
the desktop bearer (`require_desktop`). Refusals use the i18n envelope with a
machine `code` in `detail.code` and a sentence in `detail.text`.

| Route | Request | 200 | Refusals |
|---|---|---|---|
| `GET /v1/desktop/voice` | — | `{active: {session_id, enabled_at} \| null, settings: {max_spoken_chars}}` | — |
| `POST /v1/desktop/voice` | JSON `{op: enable \| disable, session_id}` | `{active, replaced}` | 404 `unknown_session`; 409 `voice_store_busy` |
| `POST /v1/desktop/voice/utterance` | form `session_id`, `request_id` (dashed UUID), `text` (required), `text_origin?` ∈ {`qwen_s2s`} | `{text, path, admission}` | 409 `voice_not_enabled` \| `approval_pending`; 422 `invalid_voice_request` \| `empty_transcript` |
| `POST /v1/desktop/voice/text` | JSON `{session_id, source}`; `source` is `{kind: "message", message_id}` or `{kind: "gate", request_id}` | `{text, spoken_chars}` | 409 `voice_not_enabled`; 404 `unknown_source` |

- `/voice/text` accepts no free text: an unknown `kind` or an extra field is a
  422. A message source resolves only an ASSISTANT message; a user row, a tool
  row, a non-message row or an unknown id is 404 `unknown_source`. A gate source
  resolves only the session's pending APPROVAL with that `request_id`. The text
  is then passed through `voice/speakable.py` (no code blocks, no URLs, capped at
  `voice.max_spoken_chars` on a sentence boundary).
- `path` in the utterance result is the `text_origin`, or `null`.
- The `type: "voice"` frame on `GET /v1/desktop/events` carries `{active}` on
  every slot change.
- Removed relative to the earlier session-voice contract: `stt`, `tts`,
  `front_ends` and `settings.front_end` on the status read; `front_end` and
  `voice_unavailable` on enable; `file`, `controls`, `control` and
  `stt_unavailable` on the utterance; the `/voice/speech` and
  `/voice/live-token` routes.

The desktop app's internal messages (main ↔ Gateway utility process, main ↔
renderer) are part of the same contract and are defined in the source record;
they do not touch this repository.

## Carried forward from session voice

These rows from the earlier session-voice design stay in force unchanged,
except where noted.

- **V3 — What "enabled" is.** One machine-wide slot,
  `<config>/voice/active.json` `{session_id, enabled_at, instance_id}`, whose
  single writer is `local_operator/voice/slot.py` under a file lock. Enabling B
  atomically replaces A and returns `replaced`. Every voice route but the status
  read refuses a session not in the slot (409 `voice_not_enabled`). A slot whose
  conversation was deleted reads as empty, and deleting the holder clears it.
- **V5 — Session context and tool calls.** The utterance is admitted through the
  `/messages` handler itself, as `input_mode="dictated"` with `input_path` set
  to the `text_origin`. It is a steer while the session streams, else a prompt
  (`admit_receipt_request`'s rule). Tools run and render as usual.
- **V7 — Approvals.** Voice never answers a gate. `desktop_voice.py` has no
  import of or call to `answer_gate` or the `/answers` handler, pinned
  statically in `tests/unit/server/test_desktop_voice.py`. Approvals are
  announced aloud and approved on screen. Never overturned by transcription
  alone.
- **V8 — Who decides what is spoken** (amended by Q10). The server, from a
  source reference, never free text; it now returns the text instead of
  synthesising it.
- **V13 — Utterances while a card is pending.** While the slot session has any
  pending card (an approval, or a blocking ask), `/utterance` refuses 409
  `approval_pending`. A steer would cancel an interruptible tool's card. The
  known race (a card that parks between the check and the steer) is accepted
  and pinned: nothing is approved and the tool does not run
  (`tests/unit/voice/test_utterance_during_approval.py`).
- **V14 — Binding a take to its turn.** A take binds to the turn whose stream
  carries the user `message_start` with `message.id ==` the take's
  `request_id`, for a prompt, a mid-turn steer and a tail steer, live and on
  replay; `admission.command_id` is the same id. Pinned by
  `tests/e2e/test_voice_take_binding_e2e.py`.
- **V15 — Turn generation.** Snapshots, `frontend.update` frames and
  `agent_start`/`agent_end` publish ONE counter, the generation of the last
  `agent_start` the session emitted (`Session._turn_generation`), never the raw
  per-run counter that also counts continuation runs. Pinned by
  `tests/unit/session/test_published_generation.py`.

## Settings

`voice.max_spoken_chars` (section `voice`, LIVE): the cap `/voice/text` applies,
default 800, 100–4000. Read from disk per request.
