# OpenRouter cache affinity

This change makes the harness **ask OpenRouter for the host that served the
previous turn**, so a long conversation's prompt cache stays on one upstream
provider. It adds one request field (`ChatRequest.provider_affinity`), one
stream field (`StreamEndEvent.served_provider`), a per-model pin held by
`SessionStreamFn`, and the `providers.openrouter.provider_affinity` setting
(default on). The existing `provider` routing preferences, `prompt_cache_key`,
reasoning replay and error handling are untouched.

`scripts/bench_openrouter_cache_rate.py` is the live A/B that produced the
numbers below. Raw per-turn JSONL lives on the PR, not in the tree (see
`AGENTS.md`, "Evidence goes on the PR, never into the repository").

## The mechanism

OpenRouter's default route for a model with many upstream endpoints is
**price-weighted load balancing**, not stickiness. `deepseek/deepseek-v4.1-flash`
currently has 11 endpoints (DeepSeek, DeepInfra, Fireworks, SiliconFlow, Modal,
Wafer, GMICloud, Morph, Io Net, Novita, Venice). Each of those keeps its own KV
cache, and DeepSeek-style caching needs a full prefix match **from token 0** —
so a turn routed to a different host than the last one re-bills the entire
conversation at the uncached rate.

`prompt_cache_key` asks for server-side sticky routing, and the harness already
sends it. **It does not hold under load**: a controlled 18-turn A/B measured 7
host switches with the key alone, and 9 with `session_id` added. Naming the host
explicitly is what holds — `provider.order` accepts the served display name and
is honoured ~96% of the time.

Two properties make the pin safe to ship on by default:

- **An unrecognised `order` entry is silently ignored** (verified live: HTTP
  200, default routing). A stale or renamed host degrades to today's behaviour
  rather than failing the call.
- **The served display name works verbatim** as an `order` entry — `AtlasCloud`,
  `Z.AI` and `Google AI Studio` are all accepted as spelled. Slug-normalising is
  wrong for 13 of 106 providers (`Z.AI` → `z-ai`, `AtlasCloud` → `atlas-cloud`),
  so the code deliberately keeps no slug table.

`order` is a **preference, not a constraint**: OpenRouter falls through to
another host when the pinned one is loaded or erroring, which is why the pin
reduces switching rather than eliminating it.

## What it measures

Live A/B, `deepseek/deepseek-v4.1-flash`, **6 concurrent conversations per arm,
15 turns each, 2 runs with alternating arm order** (24 conversations, 360 turns,
~32.5k-token prefixes, median inter-turn gap 3.8s, $1.96 total).

Concurrency is load-bearing in the method. A single sequential conversation is a
**no-pressure sample**: the load balancer has no reason to move it, so an
unpinned baseline parks on one host by itself and both arms measure the same
thing. An earlier sequential run (3×30 turns) recorded only 3 switches in 87
transitions and, correctly, showed no difference between arms.

| pooled | switches / transitions | cache share | avoidable loss | pin honoured |
| --- | ---: | ---: | ---: | ---: |
| affinity **off** | 23/168 (**13.7%**) | 49.2% | 44.1% | — |
| affinity **on** | 7/168 (**4.2%**) | 47.0% | 46.3% | 161/168 (95.8%) |

**Host switching drops 3.3×, which is the thing the change controls.** The
conditional split shows why that matters:

| transition | n | any cache hit | cache share |
| --- | ---: | ---: | ---: |
| host **same** | 306 | 56% | 55.5% |
| host **changed** | 30 | 10% | 9.9% |

A turn that stays on its host caches; a turn that moves does not.

### The cache-quality guard

The table above is the **plain pin**, and it exposed the failure mode the guard
exists for: the pin held conversations on SiliconFlow, whose same-host cache
share is 41% against 99% on its peers. A second pressure run with the guard
enabled shows it firing, and firing only where it should — **every retirement in
24 conversations named SiliconFlow, and no other host was ever retired.**

Splitting the ON lanes at the moment they retired:

| ON-arm lanes that retired a host | cache share |
| --- | ---: |
| before retirement | 38.0% |
| after retirement | **77.3%** |
| ON lanes that never needed to retire | 84.0% |

The guard's own retirements show up as extra "switches", which is why the raw
switch counts must be read by cause rather than by total. Separating a
deliberate move off a dead host from a forced re-route:

| arm | involuntary switches | retirement switches |
| --- | ---: | ---: |
| off | 16/168 (9.5%) | 0 |
| on | 16/168 (9.5%) | 9 |

**Involuntary switching is identical in both arms in that window** (a
low-pressure window — see the caveat below); the nine extra ON switches are all
the guard deliberately leaving a host that was billing full price for nothing.

### Why aggregate cache share did not move

It is the honest result, and it is a **host-composition confound rather than a
failure of the pin**. Upstream providers differ enormously in cache quality on
same-host turns:

| provider | same-host transitions | cache share |
| --- | ---: | ---: |
| Modal | 12 | 99.7% |
| Wafer | 27 | 99.6% |
| Novita | 22 | 90.4% |
| GMICloud | 24 | 82.8% |
| **SiliconFlow** | 221 | **41.4%** |

SiliconFlow is the cheapest healthy endpoint, so price-weighted routing sends
most traffic there — and it evicts aggressively. The plain pin then does exactly
what it is asked to do: it **stays** there. That raised SiliconFlow's share of
same-host turns from 61% (off) to 82% (on), and the loyalty to a weak-cache host
cancelled the benefit of switching less. (The cache-quality guard above is the
answer to this; these numbers are the pre-guard measurement that motivated it.)

Those zero-cache turns are **real misses, not a reporting gap** — checked
against OpenRouter's own generation records for a 10-turn sample: they carry no
`cache_discount` and bill at $0.30/Mtok, the full input price, against $0.009
for cached turns on the same host. A miss costs **33× a hit**.

Standardising both arms onto one host mix removes the confound:

| standard mix | off | on |
| --- | ---: | ---: |
| off-arm mix | 59.0% | **60.9%** |
| on-arm mix | 48.3% | **52.5%** |

**Mix-standardised, affinity is +1.9 to +4.2 points.** Per host it is neutral to
positive (SiliconFlow 38.0% → 43.7%; Modal, Wafer and GMICloud unchanged within
noise), which is what "keeps the cache warm without changing anything else"
should look like.

The remaining headroom is a **host-selection** question — preferring
cache-competent endpoints in the first place, rather than only leaving the worst
one after it proves itself. That is deliberately out of scope here: choosing
hosts on the user's behalf is a routing opinion this change does not take.

### What is and is not claimed

Stated plainly, because the aggregate is easy to oversell:

- **Proven.** The pin reaches the wire and is honoured ~90-96% of the time.
  Switch transitions cost ~3.8-5× a same-host transition, so switching is the
  loss. Under pressure the pin cut switching 3.3× (13.7% → 4.2%). The guard
  fires only on a genuinely cache-dead host and lifts those conversations from
  38.0% to 77.3%.
- **Not proven.** A pooled aggregate cache-share win. In both pressure windows
  the ON arm's raw share landed 2-3 points BELOW the OFF arm, because the pin
  concentrates traffic on whichever host it acquired and the acquisition is
  luck. Mix-standardised the pin is +1.9 to +4.2 points, but that is an
  adjustment, not a measured end-to-end improvement.
- **Window-dependent.** Switch pressure varies hour to hour. The second
  pressure window was calmer (9.5% baseline switching against 13.7% in the
  first), and in calm windows there is little for the pin to win — an unpinned
  conversation mostly stays put on its own.

The operator's real traffic is the regime this targets: live sessions measured
17.8-19.3% switching with hosts that cache well (86-100% hit-turns), which is
precisely where cutting switching pays and where the guard has nothing to do.

## What the harness does

1. **Capture.** `OpenAICompatClient.stream` reads the `provider` field that
   OpenRouter puts on every chunk and reports the last one as
   `StreamEndEvent.served_provider`. It is on the END event, not the start: a
   stream that dies mid-way must not move the pin. It is deliberately not part
   of `provider_payload`, which is persisted per message and is native-replay
   and compaction substrate.
2. **Record.** `SessionStreamFn._record_stream` writes it into a per-model dict
   on a successful end only. Re-pinning is immediate with no hysteresis — the
   host that just served is the one holding the prefix, and the previous host's
   entry is already decaying against a ~10-minute expiry.
3. **Stamp.** The next request carries the pin on `ChatRequest`, which is what
   failover clones for retries, so the pin follows a retry for free. It is held
   per model id and outside `FailoverRouteState` (that state is cleared on a
   model switch; a cache pin must survive a detour to another model and back).
4. **Render.** `_build_body` merges it into the `provider` object as
   `order: [name]`.

### When it deliberately does nothing

`SessionStreamFn._affinity_enabled` refuses to pin unless every condition holds.
Each is a place where the trade is not the harness's to make:

- **provider is `openrouter`.** Radient also aggregates, but its routing was
  never measured here.
- **the model advertises prompt caching.** Otherwise the pin narrows the host
  pool and buys nothing.
- **the request is not `isolated`.** A naming errand has no warm prefix, and
  letting one move the pin would drag the real conversation onto whatever host
  answered an unrelated question. Gated in **both** directions.
- **the model id has no `:` suffix.** `:nitro` and `:floor` sort by throughput
  or price by definition; a pin would quietly defeat the suffix the user typed.
- **`providers.openrouter.provider_affinity` is not false.**
- **no `order`/`only`/`ignore`/`sort` is configured.** The user's own routing
  opinion wins outright — `order` disables sticky routing on OpenRouter's side
  anyway, so a pin would either fight the setting or be silently overridden.
  `_build_body` repeats this check as defence in depth.

### Leaving a host whose cache is dead

A pin is only worth holding if the host actually caches, so affinity needs a way
to notice that it does not. A turn **strikes** when all of the following hold:
the served host is the one we *asked for* (a fallback elsewhere was never given
this prefix, so its cold turn is expected and never strikes); the gap since the
conversation's previous turn is under 300s (longer, and the miss is charged to
the ~10-minute expiry clock rather than to the host); the prefix is at least
8192 tokens; and reuse came back under `max(1024, 2%)` of it.

**Two consecutive strikes retire the host** for that conversation and model: the
pin is dropped and the host is sent as `provider.ignore` on later requests. Any
real reuse clears the counter, so isolated misses under load never accumulate.
Two rather than one because a single miss is ordinary — an eviction, a restart —
and retiring on it would churn the pin as badly as having none.

Retirement is capped at **3 hosts** per conversation and model. `ignore` is a
hard filter on OpenRouter's side, so an unbounded set walks a conversation
toward "no eligible endpoints"; a routing optimisation must never be able to
make a model unreachable. At the cap the harness stops retiring and says so once
at debug level.

Verified live that the two compose: with `order` naming one host and `ignore`
another, the ordered host served 8/8 calls and the ignored one was never
attempted.

A true transcript fork (one passed a `cache_lineage_id`) inherits the parent's
pin **and its retirements** as copies, for the same reason it inherits the cache
lineage: it replays a byte-identical prefix, so the parent's host really is warm
for it and the parent's findings about dead hosts apply. A fresh delegated
prompt inherits nothing.

The pin is memory-only. A resumed session re-acquires it after one cold call,
which is cheaper than persisting a hint a 10-minute expiry may already have
invalidated.

## Known limitation: a moving `cache_control` marker

`_message_cache_markers` marks the last message and the previous user turn, so
each turn **rewrites** the marker off the message that carried it last time.
That makes the request prefix not strictly append-only: message *N* is sent as a
`cache_control`-bearing content array on one turn and as a plain string on the
next.

Measured cost on this wire (over the 74k-prefix sequential runs, where the
effect is easiest to isolate): a median of **207 tokens per turn, ~0.28% of the
prefix** — the divergence lands near the tail, not the head, so the bulk of the
prefix still matches. It is not what causes the SiliconFlow misses above (a host
that stayed put for 10 turns still missed, while Wafer and Modal cached 99.6%
under the identical marker behaviour). Recorded here as a real but separate
finding; fixing it is not in this change's scope.
