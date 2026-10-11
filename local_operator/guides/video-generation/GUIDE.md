---
name: video-generation
description: Generate video asynchronously via the configured provider — modes, setup, duration and cost, cancel and restart, and collecting the result.
---

# Generating video

`generate_video` starts a video generation and returns a job receipt almost
immediately — it does **not** wait for the video. The generation runs as a
background job: the turn can end, other work can proceed, and when the job
settles you get a completion notice in the transcript with the video attached
to the session. Use `video_status` to check on a job at any time, including
after an app restart.

Why async: video generations take minutes to tens of minutes at the
provider. Waiting for one in-turn would hold the conversation for no reason;
the harness registers the job, keeps the runtime resident while it runs, and
delivers the result the same way every other background job is delivered.

## Modes

The mode comes from which source fields you pass — there is no mode knob:

| Mode | Source fields | Notes |
|---|---|---|
| **t2v** — text to video | none | prompt only |
| **i2v** — image to video | `source_image_path` (a local file) or `source_image_attachment` (a caption digest) | the image rides to the provider as a data URI; 16 MiB source cap |
| **v2v** — video to video / edit | `source_video_path` — an **https URL** of the source video | runs on FAL only; local video uploads are not supported yet |

Exactly one source per call. A v2v request never silently becomes a fresh
t2v: when the winning provider cannot do the requested mode, the harness
says so (and names who can).

## Which provider runs

Fixed, append-only order, first credential wins:

| Order | Provider | What it needs | Modes |
|---|---|---|---|
| 1 | Radient | a signed-in account (`/login radient`) | t2v |
| 2 | FAL | a stored key (`lop login fal`) or an exported `FAL_API_KEY` | t2v, i2v, v2v |

**Failover is automatic**: if a provider refuses (bad key, no credits,
model trouble), the next one runs. Radient's source path (i2v/v2v) is not
wired yet — until it ships, those requests go to FAL. Radient's video models
are read from its live model list at call time; nothing is pinned.

## Setup

- **Radient**: `/login radient` (or `lop login radient` from a shell).
- **FAL**: `lop login fal` — paste a key from your FAL dashboard. An exported
  `FAL_API_KEY` also works.

Keys are stored in the encrypted store and never printed. `guide://credentials`
covers the general rules for handling them.

## Duration, resolution, audio

`duration_s`, `resolution`, `aspect_ratio` and `audio` are neutral requests:
each provider family maps them onto its own wire, and a value the family
cannot represent is refused rather than quietly changed.

- **Duration** is in whole seconds. Wan accepts a range (roughly 2–30 s
  depending on model); Veo accepts 4, 6 or 8; MiniMax H3 takes a wider
  range; some edit models match the input video's length when you omit it.
  `duration_s` is also the spend multiplier — a 10 s clip costs 10× the
  per-second rate.
- **Resolution**: `480p`, `720p`, `1080p`, `4k` — mapped per family (H3
  calls its 720p tier `768P`; not every family offers every tier).
- **Audio**: `audio=true` requests a soundtrack where the model supports
  one. Some models default to generating audio (Veo does), which changes the
  per-second rate.
- Omitting a knob leaves the provider's default in place.

## Cost

Video is billed **per second of output** at the provider's rate. The
approval prompt shows a pre-spend estimate; the settled figure the provider
**reports** rides the completion:

- **Radient** reports a settled `cost_usd` on its media routes — the figure
  your account is charged, equal to the usage record.
- **FAL** bills your key at its published rates and the result carries no
  dollar figure — the run is genuinely unpriced by the provider's API. Your
  FAL dashboard is the ledger there; the model id and duration in the
  completion are what you price it from.
- Estimates shown at approval time use post-promo published rates and are
  labelled as estimates. Promotional rates are never banked.

The harness asks for approval before spending (write tier in the default
"ask" mode). In unattended modes there is no prompt — treat `generate_video`
as a paid call wherever it appears.

## Collecting the result — `video_status`

- The completion is delivered automatically: a notice row in the transcript
  with the model, duration, price (when reported) and the attachment digest.
  If the turn is busy or the session is cold, the harness holds the row and
  delivers it when it can; after a restart, the check-wake raises the
  session and the job is settled then.
- `video_status` answers on demand: pass the job id, or omit it for the
  newest pending job. States: **running** (with its stage), **completed**
  (with the digest), **failed** or **cancelled** (with the reason). A
  completed result is registered into the attachment store at most once —
  calling twice is safe.
- Providers expire generated files on their own schedule. A job whose
  result is no longer retrievable settles honestly as such rather than
  inventing an outcome.

## Cancelling

- Cancel a job with `jobs op='cancel', job_id=...` (the receipt names the
  exact call).
- The harness asks the provider to stop the job where that is possible.
  **Radient's cancel applies to jobs that have not started yet** (its
  queued-only cancel); once a job is generating there, it cannot be stopped
  and may complete upstream and bill — the cancelled record says this in so
  many words.
- **FAL** is signalled; a job already running may or may not stop before
  completion.
- Stopping the conversation (Esc, steering, leaving the session) does
  **not** cancel the provider job. The job survives a session stop by
  design; only an explicit cancel stops it, and disposing the runtime
  leaves the record so the outcome is still discoverable.
- There is no resume: a cancelled generation is gone. To steer, cancel and
  call again with the changed prompt.

## Model choice

`model` takes a provider-specific id and is optional:

- **Radient**: any video id from its live list (read at call time); its
  default row runs when you omit it.
- **FAL**: an app path like `alibaba/wan-3.0/text-to-video`,
  `minimax/h3-max/image-to-video` or `fal-ai/wan/v2.7/edit-video`. Defaults
  per mode run when omitted: Wan 3.0 for t2v, MiniMax H3-max for i2v, Wan
  v2.7 edit for v2v.

## Examples

Text to video, default model:

```json
{"prompt": "A paper boat drifts down a rain-soaked gutter, cinematic"}
```

Image to video, 8 seconds, 1080p:

```json
{"prompt": "The camera slowly pushes in as the crowd cheers",
 "source_image_path": "~/Pictures/stadium.png",
 "duration_s": 8, "resolution": "1080p"}
```

Video to video (edit), source as an https URL:

```json
{"prompt": "Transform the scene into watercolor style",
 "source_video_path": "https://example.com/clip.mp4"}
```

## When something refuses

- **"No available provider can …"** — sign in to Radient or store a FAL
  key; the message names who could run the request.
- **"Radient cannot take image or video sources yet"** — the hub's source
  mapping is not wired; i2v/v2v run on FAL today.
- **"requires an https URL"** — v2v takes a URL; upload the file somewhere
  reachable and pass that URL.
- **A duration or resolution refusal** — the named family cannot represent
  the value; pick a supported one (the message lists what it supports).
