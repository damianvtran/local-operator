---
name: classification
description: Enable and troubleshoot smart agent hints — the decision model that suggests skills, guides and MCP servers per message, plus its cascade, logins, cost and logs.
---

# Smart agent hints: the classification layer

Once per user message a small decision model picks which of the installed skills,
guides and MCP servers are worth reading for that request, and Local Operator
appends a short advisory block to the prompt:

```
<resource_recommendations>
These may help with this request — read the ones that actually fit, ignore the rest:
- guide://tunnel
</resource_recommendations>
```

The layer is ADVISORY: it may not gate a capability, it only ever adds resources
(never removes one the embedder already chose), and it never blocks a turn. Off,
the prompt is byte-identical to the harness without it.

## Turn it on

Off by default, so an upgrade can never silently change behaviour or spend.

```bash
lop config edit classification.auto true
```

The same values are under `/settings → Resource recommendations` in the TUI. The
section is scoped to NEW sessions, so an edit lands on the next session you start —
which is also when the decision vendor is resolved. Every key lives under
`values.classification` (`lop config list` prints them all):

| key | default | what it changes |
| --- | --- | --- |
| `auto` | `false` | the master switch |
| `vendor` | `auto` | pin one leg: `radient`, `typesafe` or `openrouter` |
| `model` | `""` | the leg's model id; empty uses the leg's own default |
| `timeoutMs` | `1500` | the CALL's deadline, and the breaker's clock |
| `waitMs` | `50` | how long a TURN waits for an answer |
| `maxStateChars` | `6000` | cap on the serialized state |
| `maxCandidates` | `12` | candidates sent per kind |
| `maxRecommendations` | `3` | resources added per message |
| `notice` | `true` | the one-line "Suggestion added…" notice |

## The cascade, and a login for each leg

The legs are tried in order — Radient → TypeSafe → OpenRouter — and the session
uses the first that holds a usable credential:

| leg | login | also reads |
| --- | --- | --- |
| Radient | `lop login radient` (OAuth in the browser) | `RADIENT_API_KEY` |
| TypeSafe (Jev) | `lop login typesafe` — API key only, pasted | `TYPESAFE_API_KEY`, `JEV_API_KEY` |
| OpenRouter | `lop login openrouter` (paste a key) | `OPENROUTER_API_KEY`, `OPENROUTER_API_KEY_DEV` |

TypeSafe is decision-only — never offered as a chat model — so `lop login typesafe`
stores the key and leaves session routing alone.

A leg with no credential is skipped rather than fatal, and so is a leg that fails on
a call: the next leg answers, and the set is re-resolved on the following message,
so a key revoked or topped up mid-session heals without a restart. Pin one leg with
`lop config edit classification.vendor typesafe` while another misbehaves.

## What it costs, and what it costs a turn

Measured on the OpenRouter leg against a real skill library:

- **~$0.00014–$0.00018 per call** — input tokens are the whole bill, so a narrower
  roster costs proportionally less (2 / 4 / 12 candidates measured $0.00005 /
  $0.00008 / $0.00016);
- **~0.4–0.55 s** of the vendor's own model time;
- **$0 on a repeat**: answers are cached per session, and a hit reports no spend.

None of that is the turn's latency. A turn waits at most `waitMs` (50 ms) and then
stops waiting; the call keeps running and its answer rides the NEXT message, where
the notice says which message each suggestion was chosen for. That bound is the
point of the design: a slow or dead leg cannot delay a turn, and our own added
wall-clock is that wait — plus tens of milliseconds once, on a first message.

## Which leg served the call

Every call a leg answered is logged at INFO in the session's log file
(`~/Library/Logs/local-operator/` on macOS, `~/.local/state/local-operator/logs` on
Linux):

```
classification: vendor=openrouter model=- tokens=4238/380 cost=$0.000178 latency=0.744s resources=1
```

`vendor` names the leg that answered, `tokens` its input/output counts, `cost` and
`latency` that call's, and `resources` how many suggestions came back (`model` is
the leg's model id, and prints `-` where the answer carries none). A cache hit
keeps the vendor and reports `-` for the tokens and the cost, because it spent
nothing.

Two other lines answer the other questions. `classification: no recommendation
within 50 ms; the turn continues without one and the answer rides the next user
message` (INFO) is the ordinary result against a ~250 ms vendor. `classification:
no recommendation (skipped=…)` (debug) says why nothing was tried or nothing
survived: `no-vendor` (no leg held a credential), `circuit-open` (a leg failed
repeatedly), `timeout`, `error`, `empty-roster` (nothing is installed) or
`disabled` (the switch is off).

## Troubleshooting

1. **No suggestions.** Check `auto` is true and at least one leg holds a credential
   (`lop login-status`); and note the block only appears when the model actually
   picked something, so an empty answer is not a failure. It is the ordinary state
   on a machine with no decision key.
2. **One leg misbehaving is survivable.** Our Radient route answering 503, or a leg
   whose key expired, does not break the layer — that call moves to the next leg and
   the layer reports the answer it got.
3. **Every leg failing repeatedly** opens the session's circuit breaker after 3
   consecutive failures, which stops the calls for that session instead of paying a
   dead vendor on every message. Start a new session once the cause is fixed.
