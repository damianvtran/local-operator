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
`prompt_influence`, and `model` apply to the whole batch. Each item's `name`
is that item's filename stem (see [Naming the result](#naming-the-result));
omitted, it derives from the item's prompt. Duplicate names remain distinct
requests.
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
- **Length is a spend lever on both sound-effect meters.** Radient bills
  $0.002 per started second of generated audio (the server quotes and
  settles it). A stored ElevenLabs key is billed by the vendor at 40 credits
  per second of requested duration — this tool always supplies `duration_s`,
  so a 30 s clip is about 1,200 credits. The rates differ; the meter does not.
  A BYOK voiceover is priced per character instead.

## Writing a sound-effect prompt

Describe the sound you want, not the ones you don't. This endpoint has no
negative-prompt parameter, and phrasing an exclusion — "no music, no
voices", "no background noise" — is read as a request for a quieter, weaker
result rather than as a filter (the vendor's prompting guidance and what
users report in practice agree on this). Prefer "a single dry click of a
wooden button" over "a click, no music": the unwanted element is least
likely to appear when the prompt describes the wanted one precisely and
never names it.

`duration_s` is required (0.5–30 s) and the clip is generated to fit it.
Length costs on both meters — Radient per started second, a BYOK key at the
vendor's 40 credits per second of requested duration — so budget the length
you actually need (see [Cost, honestly](#cost-honestly)).

## What reaches the vendor

A Radient generation runs on the platform under its retention terms. A BYOK
generation is a different hop: the text (voiceover) or prompt (sound
effect) goes to ElevenLabs or OpenAI under the user's own key. ElevenLabs'
`enable_logging` defaults to TRUE, and turning it off — zero-retention mode
— is an Enterprise feature; our measured refusal is a 403
`feature_not_available` on `param=enable_logging`. So a BYOK prompt or
voiceover text is logged by that vendor unless the user's own account is
Enterprise, and whatever retention Radient offers does not extend to the
vendor hop. The tools never send `enable_logging` themselves, and we cannot
switch it off on the user's behalf — say this plainly rather than implying
a privacy guarantee for the BYOK route.

## Limits worth knowing

- Voiceover text: 1–10,000 characters. Sound-effect prompt: 1–600 characters.
- `duration_s` on a sound effect is REQUIRED (0.5–30 s): there is no
  auto-duration, and length is the main cost lever on both sound-effect
  meters (see [Cost, honestly](#cost-honestly)).
- `language` is honoured only by the flash models
  (`radient/voiceover-flash`, `eleven_flash_v2_5`); the multilingual default
  (`radient/voiceover`, `eleven_multilingual_v2`) pronounces whatever it is
  given and IGNORES the field — the result's voicing notes say what was
  ignored or clamped rather than failing.
- There are NO pronunciation dictionaries or phoneme control. For tricky
  names, spell the pronunciation in the text or use `instructions` (honoured
  where the provider takes an instructions field).
- Output is **MP3 only** (no WAV). 192 kbps where the account's plan allows
  it, 128 kbps where the plan refuses it: the harness asks for the better
  format and retries once at 128 on a plan refusal, so a lower-tier key still
  gets audio rather than a failed generation. Radient serves a signed URL the
  harness downloads immediately (24 h life — nobody should rely on the URL
  later); the artifact is the attached copy, not the URL.
- **The duration is a property of the bytes, not of the provider's response.**
  It is derived at artifact creation from the same decode the waveform peaks
  use (`audiogen.peaks.compute_duration`). A provider that states one
  (Radient) is a cross-check: a disagreement is logged with the digest and the
  bytes win. A provider that states none (ElevenLabs, OpenAI) still yields a
  duration, because a sound record with no length renders with **no player at
  all** — a generated clip the user cannot play, and no error to explain it.
- A busy platform answers the submit with "Audio generation is busy right
  now." (`audio_rate_limited`). The harness retries that refusal four times
  with backoff — attempts at 0, 10, 30 and 65 s — then reports it as a busy
  failure (not a quota one), with the window it waited out in the receipt
  (`4 attempts over 65 s`; a batch adds jitter, so up to ~75 s). The 65 s is
  sized against the server's own
  45 s hold on a running vendor call (`ElevenLabsSpeechTimeout`), not guessed:
  a shorter ladder exhausted inside that window. Only the admission refusal is
  retried (it creates no job and charges nothing); timeouts, network errors and
  post-admission rate limits fail straight through. A job that hits a vendor
  rate limit after admission fails with the same word and zero cost.
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

### A stopped batch can leave the platform briefly full

The provider has no cancel API, and the audio server deliberately does NOT
abort a running vendor call when a client cancels. So a **batch** you stop does
not free its capacity instantly: the calls already in flight keep their
per-account slots until the provider's own 45 s timeout
(`ElevenLabsSpeechTimeout`) expires. That is invisible by design — the next
batch you call (for example, the remaining items of a 14-item batch stopped
at 11) waits out the hold and retries instead of failing:

- A submit refused while those slots are held is retried at 0, 10, 30 and
  65 s. Since 65 s clears the 45 s hold, the resumed items are admitted without
  the caller needing to know slots exist.
- An exhausted ladder reads `Audio platform at capacity: 4 attempts over 65 s.`
  — that sentence is the receipt on a single call and on each batch row, whose
  details carry the attempt list and `error_type: audio_rate_limited`. It is
  what distinguishes a genuinely sustained shortage from a one-off stop.
- Jitter between parallel workers only ever LENGTHENS a wait (1.0–1.15×), so a
  desynchronised retry cannot land back inside the hold.

The trade: a batch is still bounded by one 225 s tool call, and under constant
capacity a long batch could spend its budget waiting for admission. A 3-item
resume waits at least 65 s — up to about 75 s with jitter — plus one
generation.

## Naming the result

`filename` (voiceover, and the single sound-effect form) sets the audio's
display name — the Library row, the caption label and the session-file entry —
from a short lowercase hyphenated stem, no extension:

```json
{"text":"Welcome back — the build finished.","filename":"build-finished-announcement"}
```

Omitted, the stem derives from the request (the spoken text for a voiceover,
the prompt for a sound effect): `A kettle whistling` → `kettle-whistling`. The
extension always follows the returned bytes (`audio/wav` → `.wav`), never the
stem. A stem is a label, not a path: separators and `..` are sanitised away,
and the file materialises under the session's `scratchpad/generated/`; two
different clips with one name never overwrite — the second FILE gets a short
digest suffix, while its Library row and caption keep the requested label.

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
