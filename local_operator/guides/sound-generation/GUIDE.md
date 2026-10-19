---
name: sound-generation
description: Generate voiceover speech or sound-effect audio via the configured provider — routes, setup, cost and estimates, limits, cancellation, and the attached result.
---

# Generating audio (voiceover and sound effects)

Two tools, one per request shape: `generate_voiceover` turns text into spoken
audio (narrations, replies phrased aloud, voice memos), and
`generate_sound_effect` turns a prompt into a timed sound clip (a thunderclap,
a UI blip, ambience). Both attach the result to the session and return in the
same turn — the wait is seconds to tens of seconds, so there is no background
job to poll and no `*_status` tool beside them.

When NOT to use them:

- The user wants the app (or `lop`) to SPEAK something out loud right now —
  the harness's own Speak control / speech playback is the route for that;
  `generate_voiceover` produces an audio FILE as an artifact, it does not play
  anything on the user's machine.
- The user wants to transcribe audio — that is the STT path, not this one.

## Named sound-effect batches

Use ONE `generate_sound_effect` call with `sounds` for multiple clips (1–50):

```json
{"sounds":[{"name":"Door chime","prompt":"A warm two-note door chime","duration_s":1},{"prompt":"Soft mechanical click","duration_s":0.5}]}
```

Supply either `sounds` OR the single `prompt` + `duration_s`, never both.
One write-tier approval shows the item count and summed **estimated** spend
when every item is priceable, or a known subtotal and unpriced-item count;
Radient's per-item billed receipts remain authoritative. `loop`,
`prompt_influence`, and `model` apply to the whole batch. Names default from
trimmed, display-safe prompts (at most 100 characters, with an ellipsis when
shortened); explicit names are
bounded to 100 characters. Duplicate names remain distinct requests.
Each successful clip carries its name in the caption, attachment and session
file entry. A provider rejection belongs to that item, not its valid siblings.

The queue runs at most two items at once, with bounded jittered backoff for
admission rate limits. A 225-second batch budget leaves time for best-effort
stop receipts within the tool's 240-second bound: slow batches deliver partial
results, not a renewed budget per clip. Stopping prevents queued calls from
starting, requests stop for in-flight calls, and preserves completed clips;
**in-flight vendor cancellation is not guaranteed**. Inspect `details.artifacts`
for each item's status, started flag, error, request id, digest, duration, units
and cost/basis; totals report completed/failed/cancelled counts and known spend.
The tool retains every request, name and request id even for identical audio
bytes. Canvas/session files are content-addressed, so identical-byte outputs
can coalesce into one file entry; the batch result does not deduplicate them.
A partially successful batch is a success-with-failures; no attached clips is
an error. Do not retry a stopped in-flight item blindly: it may still be billed.

## Which provider runs

Fixed, append-only order, first credential wins; a refusal falls forward to
the next:

| Order | Provider | Needs | Directions |
|---|---|---|---|
| 1 | Radient (the platform) | a signed-in account (`lop login radient`) | voiceover + sound effects |
| 2 | ElevenLabs (BYOK) | a stored `elevenlabs` key | voiceover + sound effects |
| 3 | OpenAI (BYOK) | an `openai-key` platform key | **voiceover only** |

Sound effects have no OpenAI path (the vendor has no sound-effect endpoint):
an OpenAI-only machine gets the voiceover tool and not the sound-effect tool.
Store keys with `lop credential update ELEVENLABS_API_KEY` / `OPENAI_API_KEY`
(see `guide://credentials`), or `/login` for the Radient account.

The tools exist only where a route is reachable — a session built before a
login picks them up at the NEXT session start.

## Cost, honestly

- **Radient** quotes the price when the job runs and settles it at billing.
  Only the server's `cost.amount_usd` is authoritative; a quote reads
  `estimated` and a settled figure reads `billed`.
- **BYOK** (ElevenLabs / OpenAI direct) reports NO per-call figure at all.
  The harness estimates from its one central rate table (`audio-rates@…`,
  dated) and labels it `estimated` — never a charge. The user's provider
  account shows the real bill.
- The approval prompt (write tier) shows the route and, where the rate table
  can price the call, the estimated amount BEFORE the spend.
- Spend lands in the `audio` channel of the session's cost ledger (and the
  analytics store) as "Generated audio", with the estimate flag preserved.

## Limits worth knowing

- Voiceover text: 1–10,000 characters. Sound-effect prompt: 1–600 characters.
- `duration_s` on a sound effect is REQUIRED (0.5–30 s): there is no
  auto-duration, and the length is the main cost knob for effects.
- `language` is honoured only by the flash models
  (`radient/voiceover-flash`, `eleven_flash_v2_5`); the multilingual default
  (`radient/voiceover`, `eleven_multilingual_v2`) pronounces whatever it is
  given and IGNORES the field — the result's voicing notes say what was
  ignored or clamped rather than failing.
- There are NO pronunciation dictionaries or phoneme control. For tricky
  names, spell the pronunciation in the text or use `instructions` (honoured
  where the provider takes an instructions field).
- Output is **MP3 only** (no WAV). Radient serves a signed URL the harness
  downloads immediately (24 h life — nobody should rely on the URL later);
  the artifact is the attached copy, not the URL.
- The generated duration comes from the provider, never derived client-side.
- A busy platform answers the submit with "Audio generation is busy right
  now." (`audio_rate_limited`); the harness retries that a few times with
  backoff, then reports it as a busy—not a quota—failure. A job that hits a
  vendor rate limit after admission fails with the same word and zero cost.
- BYOK voice ids: `female` / `male` map to the vendors' canonical voices;
  a vendor id (an ElevenLabs voice id, an OpenAI voice name) passes through.
  A voice id for one vendor is not copied onto another during failover.

## Stopping a generation

The honest framing is **stop before it starts / request stop**, never
"cancelled for certain". Esc / steering stops the wait and sends a stop to the
server, and the receipt says what the server said:

- **Still queued** — a guaranteed stop: `cancellation.vendor_stopped: true`,
  the provider was never called, nothing is charged.
- **Already running** — `vendor_stopped: false` with
  `reason: provider_has_no_cancel_api`: the vendors behind the server have no
  cancel endpoint. The generation continues and is **charged if it produces
  audio**. The tool keeps observing for about two minutes and records the
  late charge (and the artifact, if one arrives). A switch-off or a closed
  session stops the observation — the server's own settlement is unaffected.
- **Already finished** — the stop is refused (`audio_already_completed`,
  HTTP 409); the result was discarded.
- **BYOK** — there is no server-side job at all: the harness stops waiting,
  but your provider may still bill for work already delivered.

Nothing is resumed after a stop: call again for a fresh generation. Retrying
while the same job may still be running can pay twice — the receipt tells you
which case you are in.

## The result

A single result is one caption plus the audio attached to the session
(`AttachmentContent(kind="audio", duration_s=…)`), registered once through
the content-addressed store — `details.artifacts` carries the digest for the
session-files index, and `details.audio` the provider's facts. A waveform is
available to surfaces straight off the stored sidecar (240 bars, computed
once at creation); the model itself does not need the bytes.

Do not re-save generated audio to disk unless the user asks for a file — the
attachment IS the delivery, and a copy on disk is a second copy the session
does not own.
