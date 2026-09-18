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

The layer is ADVISORY: it may not gate a capability, it only adds resources (never
removing one the embedder chose), and it never blocks a turn. Off, the prompt is
byte-identical to the harness without it.

## Turn it on

Off by default, so an upgrade can never silently change behaviour or spend.

```bash
lop config edit classification.auto true
```

The same switch is `/settings → Smart hints` in the TUI — the tips, this guide and that
page all use that name for it, while the runtime's own message says *Suggestion added…*
and the config file calls it the classification layer. The section is scoped to NEW
sessions, so an edit lands on the next session you start — which is also when the
decision vendor is resolved. In the file every key is `values.classification.<key>`;
the CLI spells it `classification.<key>`, which is what `lop config list` prints:

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

A leg with no credential is skipped rather than fatal, and so is a leg that fails on a
call: the next leg answers THAT call, and the set is re-resolved on the next message, so
a key *revoked* mid-session heals without a restart. A session that began with no usable
credential is different: the empty result is remembered too, so it reports `no-vendor`
for its whole life and a login added mid-session needs a new session. The vendor pin and
the breaker are per session as well — `lop config edit classification.vendor typesafe`,
then start a new session.

## What it costs, and what it costs a turn

Measured against a real skill library, mostly on the OpenRouter leg:

- **~$0.00006–$0.00018 per call** — the bill is input tokens, so it tracks the roster: an
  isolated 14-candidate run measured $0.00006, and a full skill library ~$0.00018 (the
  example below is one of those calls);
- **~0.2–0.9 s** of the vendor's own model time: eight real calls on the OpenRouter leg
  measured 0.17 s to 0.87 s, median ~0.56 s (`scripts/classification_latency_probe.py`
  re-measures this arm; the figure moves with the roster and the route);
- **$0 on a repeat**: answers are cached per session, and a hit reports no spend.

None of that is the turn's latency: a turn waits at most `waitMs` (50 ms) and then stops
waiting, and the answer rides the NEXT message, where the notice says which message each
suggestion was chosen for. A slow or dead leg therefore cannot delay a turn — our own
added wall-clock is that wait, plus tens of milliseconds once on a first message.

## Which leg served the call

Every call a leg answered is logged at INFO in the session's log file
(`~/Library/Logs/local-operator/` on macOS, `~/.local/state/local-operator/logs` on
Linux):

```
classification: vendor=openrouter model=- tokens=4238/380 cost=$0.000178 latency=0.744s resources=1
```

`vendor` names the leg that answered, `resources` how many suggestions came back, and
`tokens`/`cost`/`latency` are that call's own figures. `model` is always `-`: a
recommendation carries no model id, so the field reports its absence rather than the
leg's default. A cache hit keeps the vendor and reports `-` for the tokens and the cost,
because it spent nothing.

Two other lines answer the rest. `classification: no recommendation within 50 ms; the
turn continues without one and the answer rides the next user message` (INFO) is the
ordinary result whenever the vendor outruns the budget. `classification: no
recommendation (skipped=…)` (DEBUG) names why: `no-vendor` (no leg held a credential),
`circuit-open`, `timeout`, `error`, `empty-roster` (nothing installed) or `disabled`
(the switch is off).

## Troubleshooting

1. **No suggestions.** Check `auto` is true and that a leg holds a credential
   (`lop login-status`). An empty answer is not a failure — the block appears only when
   the model actually picked something. And a session that started with no usable
   credential keeps reporting `no-vendor`: log in, then **start a new session**.
2. **One leg misbehaving is survivable.** A leg answering 5xx or 429, or one whose key
   expired, does not break the layer — that call moves to the next leg and the layer
   reports the answer it got.
3. **Every leg failing repeatedly** opens the session's circuit breaker after 3
   consecutive failures, which stops the calls for that session instead of paying a
   dead vendor on every message. Start a new session once the cause is fixed.
