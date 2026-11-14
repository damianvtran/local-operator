---
name: music-generation
description: Compose or modify music asynchronously, inspect capabilities, retain every track, and request honest cancellation.
---

# Music generation

Use `generate_music` for compositions, `modify_music` for a new composition that
keeps owned ranges, and `music_status` to inspect the durable local job or request
cancellation. These are music tools, not sound-effect synthesis.

## Availability, provider and price

Music is a paid Radient service: the public model is `radient/music`, backed by
ElevenLabs Music (`music_v2_5`) on the server, with attribution "Powered by
ElevenLabs Music" carried in the server's capability and completion details.
The estimate is $0.15 per requested minute, per variation ($0.0025 per second;
a 30 s track is about $0.075, two 30 s variations about $0.15). That figure is
an estimate for the user; the server's `cost` object is the only authority, and
`music_status(action="models")` reports the live tariff.

A persisted Radient login plus the session's background-job manager and durable
journal make the tools discoverable without network access. Core has no local
on/off switch: the SERVER owns activation (its commercial gate and attribution).
Before registering a purchase, the tool reads `/tools/music/models`; a disabled
capability returns its actual `disabled_reason` and no generation job is started.
`generate_music` and `modify_music` spend money, so they need the usual
write-tier approval; `music_status` is read-tier.

Uploaded references, stem separation and other unpriced operations are not
available. No upstream vendor key (ElevenLabs, Suno or other) belongs in a music
tool argument, and a stored BYOK ElevenLabs key does not route music.

## Compose and modify

- Set explicit `duration_ms` (3000–300000) and a prompt, or exact lyrics.
- `instrumental: true` is prompt-only. Do not combine it with lyrics/sections.
- Exact lyrics/styles use the server's typed composition chunks, not invented
  fields or lyrics embedded into a sound-effect prompt.
- `variations` is 1 or 2, reflecting the server's current account allowance.
  Each accepted child has a separate request identity, result and settlement;
  a failed sibling does not erase purchased audio.
- For `modify_music`, supply `sections` with at least one `kind: keep`, an owned
  `source.request_id` and a millisecond `range`. Generated sections have explicit
  durations. Their total must equal `duration_ms`. Arbitrary source URLs are not
  accepted. This is composition with retained ranges, not a promised DAW editor.

The submission returns promptly with a local job ID; work runs detached on the
ordinary job manager. `music_status(action="models")` reports live capability
reasons. `music_status(job_id=...)` follows an interrupted observation and does not
buy a replacement song. The durable transcript records the purchase's original
idempotency key before submission, so uncertain responses/restarts reuse that key.
No bearer token or signed playback grant is persisted in the recovery record.

## Cancellation, money and files

Turn Stop and session teardown stop neither an accepted provider composition nor
its purchased result. Explicit job cancellation requests the server's best-effort
stop. A queued job may stop before spend; a running composition can still finish
and be charged. The runtime retains an observer and registers any returned audio.
Never describe `requested: true` as a refund or a guaranteed upstream stop.

Only the server's versioned `cost` object is money authority. Its requested-ms
fixed quote is not an upstream invoice or a price derived from MP3 duration.
A quote stays estimated until the server reports a billed receipt; unknown does
not mean zero. Stable per-child spend revisions keep replay from adding a second
ledger line.

All returned audio is hash/size checked and registered through the existing
audio attachment store, which creates waveform sidecars. Completion details keep
track/variant/role identities, individual costs and approved attribution. Existing
chat, Files, Canvas and highlights consume the same audio artifacts. Partial
registration failures preserve successful siblings and allow a later status follow
to retrieve missing purchased output without resubmission. No autoplay is used.
