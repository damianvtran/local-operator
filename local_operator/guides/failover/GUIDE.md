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

Account selection stays within the current provider and its existing credential
tier. It never activates a paid API-key tier, changes a model/effort pin, or buys
credits to obtain a better reset time. Hard blocks, revoked/disabled credentials,
cooldowns and per-attempt exclusions still apply; every measured relevant shared
and model-family quota window gates eligibility.

`retry.accountSelectionPolicy` chooses the preference after those checks:

- `reset-soonest` (new default): prefer the nearest included-plan weekly refill.
  Accounts within **60,000ms inclusive** of the fixed earliest deadline tie; use
  their five-hour deadlines with the same fixed-minimum tolerance, then stable
  session hashing. This is not transitive neighbour chaining. All relevant caps
  must still have capacity: one window reopening is only an opportunity to
  recheck another spent cap, not proof the account is healthy. If several
  applicable weekly windows exist, the earliest future refill is the next
  useful preference; every other overlapping allowance still gates eligibility.
- `headroom`: retain the cached-quota preference, spreading accounts within ten
  percentage points of the most remaining headroom.
- `fair`: use a stable per-session spread, without ranking reset or headroom
  forecasts. Eligibility and reserve last-resort ordering still apply.

No extra blocking usage probes are made for ranking. Reset forecasts use reports
at most 30 minutes old; missing, stale, malformed, past, unsupported or
future-fetched reset evidence does not invent a deadline. Weekly-unknown accounts
join a neutral tie pool rather than starving behind known accounts. Without
comparable weekly data use actual five-hour data, otherwise the stable hash.
No-session callers keep round-robin. Cached reports are observations, not quota
reservations; concurrent requests can still need normal reactive recovery.

**Every policy first honors a compatible, unexpired local account-affinity
lease derived from a successful request.** Resolving, inspecting or preflighting
an account is not success and never renews a lease. Failed/cancelled streams,
empty iterators and isolated errands do not renew it. A recognized successful
terminal event must finish cleanly; expiry is request-start plus its effective
wire lifetime, not response-completion plus a fresh TTL. The exact expiry
boundary is invalid. Own failed attempts and hard depletion override affinity;
another session's advisory demotion does not discard a valid warm lease.

A lease is a conservative routing hint, **not a guaranteed cache hit**. Built-in
Anthropic requests use the actual 5m/1h cache markers. Exact modern public OpenAI
Responses endpoints use the documented 30m default; legacy public retention and
Codex OAuth use explicitly bounded heuristics (30m/5m for legacy Responses, 5m
for Codex/chat). Unknown compatible gateways and unsupported adapters have a
zero effective TTL and create no affinity lease. Endpoint/model/cache-prefix or known cache-affecting setting
changes and compaction require compatible evidence again. Request content is
always built anew; an account lease never replays an old language prefix/body.

Receipts live in a machine-local sidecar, namespaced by session, auth-store path
and immutable account row creation identity, not in transported transcripts or
credential DB schema. Resume keeps the original expiry; corrupt/future/rollback
receipts are ignored. Persistence failure retains process memory only and loses
restart continuity. Policy saves hot-apply without invalidating a still-valid
lease or unrelated provider/model/effort route pins. The new preference applies
to the next unanchored request or expired lease, not by moving a live anchor.

For compatibility only, an explicitly stored legacy `usageAwareAccountPick`
strict boolean selects `headroom` when true and `fair` when false, **only when
the canonical policy is absent**. A valid canonical enum wins; malformed canonical
values fall back to `reset-soonest` with a diagnostic. Fresh configs emit only
the canonical enum. Resetting its settings row clears legacy overrides too.

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

## Pinned children (subagents launched on a resolved model)

A subagent launched on an explicitly resolved model — a role's `effort` tier,
or a resumed child's recorded tier — is **pinned** to it: the child's stream
carries a pin marker, and the pinned-child rules below are gated on that marker
alone. Parent sessions and inherit-children never carry it, and their routing
is unchanged.

- **Same-family first, always.** Candidates are ordered pin-preserving →
  same-vendor → cross-vendor, and that ordering is shared with the quota
  preflight (`_first_available_fallback`), so the cascade walk and the
  message boundary cannot disagree about which hop may serve a pin.
- **Cross-vendor is the announced last resort** (the shipped default; the
  `/settings` row is `Pinned child fallback`, showing **allow cross-vendor**
  for `retry.pinnedFallback: cross-family`). A pinned child whose own model
  cannot serve walks the configured chain and may end on another vendor's
  model — but only after every same-family hop, and the substitution is
  DISCLOSED, live: a parent-stream notice, the job badge and roster label, the
  durable completion row (`pinned X, ran on Y`), and the persisted route
  reason. The disclosure exists because a delegated review that silently
  collapses onto the author's model is not a review — a substituted run that
  SAYS it substituted stays auditable. One chain-hygiene caveat: a chain may
  include the parent's own model, in which case the disclosure will report a
  substitution that gained no independence.
- **`retry.pinnedFallback: same-family` is the strict opt-in** (the row's
  **same family only**). The child never enters a cross-vendor target; when no
  same-family hop can serve it, it fails with a legible error — the pin, the
  cause, and both remedies, never a silent substitution. A quota boundary
  under this policy **announces** instead of activating the hop.
- **An unserved pinned child fails legibly, never bare.** Whenever the walk
  ends without serving, the error names the pin, the cause, and the hops that
  were tried and could not serve it ("Every configured hop was tried…") — or
  says that no hop is configured, or that fallback is switched off — each with
  its remedy. Read it as "fix a hop or add another", not as "no fallback
  exists".

## Quota reserve

**Everything in this section requires `usageAwareFallback: true`, which is NOT
the shipped default.** With it false (the default), `preflight_usage` returns
before any of it runs, so no quota is probed at message boundaries and neither
rule below applies — the cascade still works, it just moves on provider errors
rather than on quota forecasts. Turn it on to spend one lightweight quota
request per user-message boundary in exchange for routing that leaves a
provider before it fails.

`usageReservePercent` (default 10) is the headroom below which an account
counts as **low**. Three rules that are easy to get backwards:

- **Low is a preference for unanchored picks, not an eviction from a valid
  success-derived lease.** A compatible warm account stays usable below the
  reserve threshold until depletion, a hard block/own failure, incompatibility
  or lease expiry. After expiry the selected policy can prefer a healthy
  sibling, even in the same old conversation. Low-but-usable rows remain a
  last resort when healthy/unknown siblings cannot serve. A tentative
  resolve/preflight pointer alone does not earn this warm-account exemption.
- Low quota can select a **lower-effort route on the same provider** without
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
    maxRetries: 10          # shipped default
    baseDelayMs: 500        # shipped default
    modelFallback: true     # shipped default
    usageAwareFallback: true   # NOT the default (ships false); see below
    usageReservePercent: 10 # shipped default
    accountSelectionPolicy: reset-soonest  # headroom | fair; see above
    # `false` restores the pure per-session hash spread for SESSIONS (the
    # setting is applied where a session's stream is built; store instances
    # the login/credential CLI or MCP auth build on their own never pass a
    # session id, so they never rank and never read this setting).
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
An omitted effort carries/clamps the primary's choice under the existing routing
rules. Explicit `effort: auto` instead clears that inherited override and uses
the **target's own default**: `radient/auto` sends its router's `auto`, while a
direct provider uses its default level or omits effort if it has none. This is a
chain reset, not another fixed depth on the effort ladder. For example:

```yaml
fallbackChains:
  default:
    - provider: radient
      model: auto
      effort: auto
```

To author this mapping, use `lop config edit retry.fallbackChains '<JSON>'` or
edit `config.yml`; the settings page preserves existing effort labels but its
selector input does not create them. The agent `setting` tool refuses cascade
writes because it cannot carry the editor's original snapshot.

An entry without an effort that names the primary is skipped, so a shared
`default` chain can safely contain models you also select as primary. When the
primary effort is known, quota preflight and the stream also skip an explicitly
identical effort; a genuinely different effort stays a separate route. Account
rotation still happens within each route before descending the chain.

`enabled: false` or `modelFallback: false` switches the cascade off entirely —
`/failovers` names whichever of the two did it.

**An edit reaches every running session.** Each process watches `config.yml`
and rebinds the stream's settings mapping when it changes — immediately in the
process that wrote it, within about two seconds in any other `lop` session on
the machine — so a cascade change needs no `/reload`. The other session prints
a `config.yml changed: retry.fallbackChains — applied` notice so the user knows
why routing moved. `/failovers` reads that same live mapping.

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
