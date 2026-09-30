# Pinned subagent fallback: probation, family ordering, and the visible refusal

Design note for the model-routing policy that protects a **pinned child** — a
subagent whose launch resolved an explicit model (a role's `effort` tier, or a
resumed child's recorded tier) rather than inheriting the parent's.

## The problem this closes

A role-pinned child (designer, ux-reviewer, …) is spawned correctly on its
pinned model, and then failover may move it to another **vendor's** model.
Two mechanisms made that both easy and silent:

1. **One refused call descended the waterfall.** A usage-limit refusal whose
   advertised reset exceeded the interactive cap (`MAX_USAGE_RETRY_AFTER_MS`,
   30 s) earned **zero** same-credential retries, so a single 429 was enough
   for the walk to pin a fallback — and a child's run is *one message*, so the
   fallback stayed sticky for the whole child with no later boundary at which
   it could self-heal.
2. **Nothing recorded why.** The persisted `active_model_route` row carried
   only the selector and effort, and the settle reason on the pin was the bare
   string `"provider failure"`, so a transcript could not even say *why* a
   route had moved.

Independence that can silently collapse into a different model (or, worst
case, the author's own) is not independence — the same principle the strict
tier-resolution path enforces at launch, now enforced on the routing layer
underneath it.

## Where the pin marker lives

`FailoverRouteState.launch_pin` — one string, `None` everywhere except a child
launched on a resolved model. It is set **once**, at child build, by
`SessionStreamFn.mark_launch_pin(selector)` called from
`local_operator/harness/subagent.py`, and only on the child's own forked
stream (never the parent's). Parent sessions and inherit-children never carry
it.

**Every policy below is gated on that marker.** An unpinned route — including
a parent session that merely *has* a `FailoverRouteState` — keeps its routing
byte-for-byte; that property is pinned by a negative-control test.

## The policy

### 1. Probation — one capped same-credential re-ask per credential

On a **pinned** route, every retryable usage-limit refusal earns at least one
same-credential re-ask, with

```
wait = max(min(advertised_retry_after, PINNED_USAGE_RETRY_AFTER_CAP_MS), backoff)
PINNED_USAGE_RETRY_AFTER_CAP_MS = 60_000
```

A child runs as one long background call, so it may afford a bounded wait an
interactive turn may not — but the wait is **capped, not refused**: a
multi-hour advertised reset is slept at most the cap, re-asked once, and then
treated as spent (`transport_retries` counts the re-ask). The practical bound
is therefore *at most one capped wait plus one request per credential* before
rotation or descent begins. The existing `min(max_retries, 2)` budget and the
30 s cap are untouched for **unpinned** routes.

### 2. Same-family ordering at both descent points

Candidates are ranked by **vendor**, not by `model_family()` — that function
is a *quota-scoping* notion ("which cap does this model draw on") with no
capability semantics, while the line this policy draws is "the pin's own
vendor" vs "another vendor". The ranks are implemented once, in
`failover.pinned_family_rank`, which normalizes BOTH selectors to the
`(vendor, model)` pair they ultimately serve — an aggregator's `vendor/model`
suffix is unwrapped no matter which side of the comparison the aggregator is
on (review round 1, F1: the one-directional form filtered out a pin's OWN
model on the direct route and miscalled it "cross-vendor"). `is_same_family`
is the boolean view the strict filter and the walk's availability flag call;
`order_pinned_targets` is the ordering both descent points share:

| rank | name | what it is | example for pin `anthropic/claude-sonnet-5-5` |
|---|---|---|---|
| 0 | PIN-PRESERVING | the same model through another route — and the round-trip: a pin resolved THROUGH an aggregator takes its own model on the direct route | `openrouter/anthropic/claude-sonnet-5-5`; for pin `openrouter/anthropic/claude-sonnet-5-5`: `anthropic/claude-sonnet-5-5`, `radient/anthropic/claude-sonnet-5-5` |
| 1 | SAME-VENDOR | a sibling model of the pin's vendor (directly, or via an aggregator whose model id's leading segment names that vendor; symmetric too) | `anthropic/claude-opus-5`, `openrouter/anthropic/claude-opus-5` |
| 2 | CROSS-VENDOR | anything else — including two routes of ONE aggregator whose underlying vendors differ (the old same-provider read had ranked that 1) | `deepseek/deepseek-flash`, `openrouter/deepseek/deepseek-chat` |

The same predicate orders and filters **both** entry points — the cascade walk
(`providers/failover.py`) and the message-boundary quota preflight
(`model/configure.py::_first_available_fallback`) — so they cannot drift into
two opinions about which target may serve a pinned child; the preflight's
`different_provider` preference is advisory beneath the policy.

### 3. `retry.pinnedFallback` — the disposition when no same-family target can serve

| value | behaviour |
|---|---|
| `same-family` (**default**) | a pinned child does **not** enter a cross-vendor target. After probation and every same-family candidate are spent it **fails**, with a refusal that LEADS with the actionable half — the failed child's dock row paints only its first ~58 cells — and names the pin, the cause, and both remedies in the `/settings` page's own vocabulary: "add a same-family hop to `retry.fallbackChains`", "set `retry.pinnedFallback` to allow cross-vendor". A pinned child with NO chain takes the same legible path rather than surfacing a bare provider error (review round 1, D4/D5/F3). At a quota boundary the same disposition is **announced** (a notice) instead of activated. |
| `cross-family` | the opt-in: cross-vendor targets remain available as the **last** resort, still after family-first ordering, and a descent states itself in the settle reason/report ("cross-vendor descent for pinned …"). |

Junk or absent values degrade to the default (`same-family`), matching the
fail-safe contract of the other `retry.*` readers. The key is read **per
call**, so the ~2 s live-reload rule applies with no `/reload`. It is
registered in the `/settings` registry (`retry.pinnedFallback`, ENUM) with its
default pinned to `DEFAULT_PINNED_FALLBACK` by
`tests/unit/test_settings_io.py::_consumer_defaults`.

**The trade-off, stated plainly:** the strict default trades *silent
downgrades* for *visible pinned-child failures* while a chain has no
same-family target. A machine whose `fallbackChains` resolve only to
cross-vendor targets will now see pinned children FAIL (with the remedies in
the error) instead of quietly running on another vendor's model. That is the
point — a failed review is recoverable; a review that silently ran on the
wrong model is not. `cross-family` restores the old reach for operators who
prefer it.

### 4. Observability — the reason travels

- The walk computes the settle reason from the error it holds
  (`settle_reason_for_target`: "provider failure: quota HTTP 429", plus the
  cross-vendor clause under the opt-in) and the preflight records its
  `condition`.
- `Session._on_route_settled` keeps the reason and
  `_persist_active_route` writes it as an **additive** `reason` key in the
  `active_model_route` row — older rows lack it, older readers ignore it.

## What is deliberately NOT changed

- **reviewer / qa-tester / coder carry no pins by design** — they inherit the
  parent's model, so their ladder (and every unpinned route's) is untouched.
- `subagents.models` and the `SubagentModelUnavailable` launch gate are
  untouched; this is the routing layer beneath them.
- `retry.fallbackChains` remains the candidate **pool**; the pin policy only
  orders/filters it for pinned children.

## Resolution hardening (the adjacent silent-substitution hole)

The same principle applies when the pin itself cannot be *read*:

- `Session._resolve_subagent_model` now forces a **fresh registry scan**
  (`AgentRegistry.refresh_now`) at spawn/resume, so a role pin written after
  the session started — by any process — reaches the next spawn rather than
  waiting out the registry's ≤5 s refresh interval.
- A registry that **cannot be read** is no longer treated as "no role of that
  name": `resolve_profile(strict_registry=True)` raises instead of falling
  through to the packaged seed (which carries no operator pin), and the strict
  launch path refuses with `SubagentModelUnavailable` naming the read failure.
  A genuinely absent row still resolves to the seed and inherits — the
  documented default keeps working.

## Tests that pin this

`tests/unit/providers/test_failover.py` (probation, ordering, strict refusal,
settle reason, negative control), `tests/unit/model/test_configure.py`
(preflight ordering + announce-don't-cross), `tests/unit/session/
test_active_route.py` (the persisted reason), and
`tests/unit/session/test_pinned_subagent_model.py` (fresh read; unreadable
registry refuses). Each was demonstrated red against the mechanism it names
before landing.
