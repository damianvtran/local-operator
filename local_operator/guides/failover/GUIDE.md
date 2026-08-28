---
name: failover
description: Model failover and load balancing — why a request fell back, which provider serves now, fallbackChains cascade order and effort routing, multi-account quota rotation, cooldown.
---

# Model failover: the cascade, quota rotation, and what is serving

Read this when a request went to a provider the user did not select, when the
route keeps flipping to a model they do not want, when they ask which provider
is answering right now, or when they want to change the order.

The fast answer for "what is my cascade and what is serving it" is the
`/failovers` command in the TUI. It prints the primary, which chain key matched,
the ordered targets with their configured effort, how many accounts each
provider has, and a `← serving` marker on the row actually answering. Everything
below explains what that listing means and how to change it.

## The shape of the thing

Routing is two nested loops, and almost every confusing report comes from
mistaking one for the other.

1. **Within a provider**, requests rotate across every account stored for it.
2. **Across providers**, the cascade descends the configured chain.

The inner loop is exhausted before the outer one moves. A user with four
Anthropic accounts does not leave Anthropic because one hit its limit — they
leave when all four are spent.

## Rotation within one provider

Every credential for a provider is a candidate for every request. Selection
order (`AuthStore._base_selection_order`) is: the session's sticky account
first, then a per-session hash of the session id so concurrent sessions start on
different accounts rather than stampeding one, then round-robin for callers with
no session. An account that just failed on a provider-side fault is *demoted*
(sorted last), never removed — a 500 is not the account's fault, so it stays
available as a last resort.

When a request fails on quota, `_recover_quota_blocked` (in
`providers/failover.py`) probes the *other* accounts' own usage before the chain
is allowed to leave the provider. This matters because **a model-tier cap is per
account**: Anthropic's `7 day (Fable)` limit against `claude-fable-5` blocks one
account for one model family while its shared windows still hold headroom, and
while other accounts are untouched. A block is therefore a verdict about one
account and one family, not about the provider, and siblings are tried first.

## Descending the cascade

Once no account on the current provider can serve, the chain's next target is
tried, in configured order. A fallback whose own provider is already at 0% is
skipped rather than pinned — hopping onto a maxed provider costs a full prompt
to learn what was already known.

## Quota reserve

`usageReservePercent` (default 10) holds back a slice of quota. Two rules that
are easy to get backwards:

- Reserve quota can select a **lower-effort route on the same provider** without
  blocking the account. Dropping from high to low effort on the model the user
  chose is cheaper than leaving for another vendor, so the reserve buys a
  cheaper route before it buys a different provider.
- **Fully exhausted** account-wide quota skips same-provider effort changes
  entirely: no effort level restores capacity that is gone, so trying one is a
  wasted request.

Usage endpoints that are missing or unreachable **fail open**. A provider with
no usage API, or one whose usage call times out, is treated as usable — the
alternative is refusing to route on a provider that was probably fine.

## Sticky routing, and the way back up

A successful fallback **stays pinned** for the rest of the user message: later
tool calls and other model calls in that turn go to the same target. A turn that
silently changed models halfway through would produce a transcript no one can
reason about.

Returning to the primary is three separate mechanisms:

- **`primary_retry_at_ms`** — a cooldown after a transport failure. It is set
  when the route **CHANGES**, and deliberately not bumped on every request that
  uses the pinned route. Bumping it turned a fixed cooldown into a sliding
  window: a user sending messages more often than the cooldown never reached the
  deadline and stayed on the fallback for the entire session, long after the
  primary recovered.
- **`quota_pinned`** — a pin placed by a *quota* verdict is re-probed at every
  message boundary **regardless of the cooldown**. Quota resets on a schedule
  the usage endpoint can state definitively, so a cooldown sized to a 24-hour
  advertised reset would glue the session to a fallback for a day after the
  primary's 5-hour window reopened.
- **`target_retry_at_ms`** — a per-target bench ("kimi 500'd 40s ago") so a
  settled session does not replay a dead target's failure at every message
  boundary. It is deliberately **not** cleared by `clear()`: it records facts
  about the fallback providers, and neither a `/model` switch nor the primary
  recovering changes those facts. Entries expire on their own, a target that
  serves sheds its mark, and the bench is **advisory** — the stream driver's
  loop-back sweep re-walks benched targets before declaring a turn exhausted, so
  it can delay a target but never remove the last route to a served turn.

## Retry and backoff

`maxRetries` is the fast budget against a *reachable* provider (5xx, timeout);
`baseDelayMs` seeds `backoff_delay_ms`, which is
`min(base * 2^(attempt-1), 8000)` with 25% downward jitter. Connectivity loss —
the machine itself offline, DNS or socket-connect failing before any HTTP — has
its own patient budget (`connectivityMaxRetries`, `connectivityBackoffCapMs`) so
an interactive session survives a laptop lid closing without burning the fast
budget on a network that is not there.

## Configuring it

Everything lives under `values.retry` in `config.yml`:

```yaml
values:
  retry:
    enabled: true
    maxRetries: 3
    baseDelayMs: 500
    modelFallback: true
    usageAwareFallback: true
    usageReservePercent: 10
    fallbackChains:
      anthropic/claude-opus-5:
        - provider: zai
          model: glm-5.3
        - provider: openai
          model: gpt-5.3-codex
          effort: high
```

Chain keys are matched by specificity: an exact `provider/model` key wins, then
the longest matching `provider/*` wildcard, then `default`. `default` applies to
any model without a more specific chain, and a `provider/*` target keeps the
failing model's id.

A target is either a legacy `provider/model` string or a mapping with
`provider`, `model`, and optional `effort`. The mapping form is what lets a
chain re-list the *current* model at a different effort as a real route.

`enabled: false` or `modelFallback: false` switches the cascade off entirely —
`/failovers` names whichever of the two did it.

**Never put tokens or API keys in this mapping.** Credentials live in the
credential store (`local-operator login <provider>`); a chain entry names a
route, never a secret.

## Inspecting it without the network

These run against the installed runtime and touch no provider API. The `LOP_PY`
line resolves the interpreter that `lop` itself runs on, so they work from any
directory:

```bash
LOP_PY="$(head -1 "$(command -v lop || command -v local-operator)" | sed 's|^#!||')"
```

**Which providers exist:**

```bash
"$LOP_PY" - <<'PY'
from local_operator.providers.registry import list_login_providers
for d in list_login_providers():
    print(f"{d.id:26} {d.name}")
PY
```

**Which models a provider serves.** Note the aggregator caveat: `static_models`
returns `{}` for `openrouter`, `radient`, and `ollama` **by design** — their
shipped entry describes the ROUTER, not a model, and for exactly those providers
the live listing is authoritative. `STATIC_MODEL_HOSTINGS` names the providers
this can answer for; for the others, use the live catalogue (`/model` in the
TUI), which is the only source.

```bash
"$LOP_PY" - <<'PY'
from local_operator.model.registry import STATIC_MODEL_HOSTINGS, static_models
print("answerable statically:", ", ".join(sorted(STATIC_MODEL_HOSTINGS)))
for model_id in sorted(static_models("anthropic")):
    print(" ", model_id)
print("openrouter ->", static_models("openrouter"))  # {} on purpose
PY
```

**The resolved cascade** — the same walk the engine performs, so this is what
would actually route:

```bash
"$LOP_PY" - <<'PY'
from local_operator.config import ConfigManager
from local_operator.paths import config_dir
from local_operator.providers.failover import (
    RetrySettings, expand_fallback_targets, resolve_chain, resolve_chain_key,
)

values = ConfigManager(config_dir()).get_config().values
retry = RetrySettings.from_settings(values)
primary = f"{values.get('hosting')}/{values.get('model_name')}"

print(f"primary      {primary}")
print(f"enabled      retry.enabled={retry.enabled} modelFallback={retry.model_fallback}")
print(f"reserve      usageReservePercent={retry.usage_reserve_percent}")
print(f"matched key  {resolve_chain_key(primary, retry.fallback_chains)}")
chain = resolve_chain(primary, retry.fallback_chains) or []
for i, target in enumerate(expand_fallback_targets(primary, chain), 1):
    print(f"  {i}. {target.selector:38} effort={target.effort or 'model default'}")
PY
```

**Account depth per provider** — how many accounts the inner loop can rotate
through before the cascade leaves a provider:

```bash
"$LOP_PY" - <<'PY'
# COUNTS AND PROVIDER IDS ONLY, never row.data: those rows carry live bearer
# tokens and account emails, so one added field here is a credential printed
# into a scrollback and a transcript.
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.registry import credential_provider_id

counts: dict[str, int] = {}
for row in AuthStore().list_credentials():
    pid = credential_provider_id(row.provider)  # xai-oauth and xai are ONE account
    counts[pid] = counts.get(pid, 0) + 1
for pid, total in sorted(counts.items()):
    print(f"{pid:24} {total} account(s)")
PY
```

## If this guide and the engine disagree

`local_operator/providers/failover.py` is authoritative. It carries the routing
loop, the cooldown rules, and the comments recording which regression each rule
exists to prevent; this guide is a reading of it and can fall behind.

For anything not answerable here, read the source on GitHub:
<https://github.com/damianvtran/local-operator>.
