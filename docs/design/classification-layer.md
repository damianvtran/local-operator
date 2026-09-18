# The classification layer: a swappable decision-model vendor for resource recommendations

Status: design + interface contract, 2026-09-18. This file is the contract three
implementation slices code against (core package, harness wiring, Radient server route).
Landing it in the same change as the code keeps the contract reviewable.

## 1. Why

Every semantic decision the harness makes today is a hand-tuned threshold, a hand-written
keyword table, or a sub-millisecond linear model:

- skill/guide selection — hashed char-ngram cosine + Jaccard with an absolute cut at 0.19
  (`skills/index.py:480`, `skills/embeddings.py:189`), a backend the module itself documents at
  ~71% recall on an independent query set with "selecting nothing is the expensive failure mode"
  (`skills/embeddings.py:144-176`).
- MCP server advertisement (`mcp/resources.py:272-338`) — ~200 lines of lexical weights plus an
  embedding fallback with two hand-tuned scalars (0.42 / 0.08).
- role suggestion (`tools/agent_tool.py:697-712`) — deliberately uncut at `threshold=0.0`.

A decision model ("Jev", TypeSafe's System One) changes the shape of the problem: you send a
state plus a set of *typed questions* and get typed answers back — a `choice` from named options
with a probability distribution, a `noul` yes/no probability, or a `score` over ordered levels.
It cannot generate prose, and it cannot invent an option. Measured on 31 labelled PEP-tier cases:
31/31 correct with the rubric carried in the option descriptions, 17/31 without it, p50 0.20 s,
~$0.00005 per call.

This layer does not replace those heuristics. It runs **beside** them, once per user message,
and adds a short, advisory list of resources the agent may want to read. Every existing
selection path keeps working, unchanged, when the layer is disabled or the vendor is down.

## 2. Scope of this change

In scope:

1. `local_operator/classification/` — a vendor-agnostic decision client (Jev today, another
   model tomorrow) with a cascade: **Radient → TypeSafe → OpenRouter**.
2. Skill, guide and MCP-server recommendations, rendered as a short advisory block, computed
   once per user message.
3. An API-key-only login for the TypeSafe (Jev) provider, kept out of the chat model list.
4. Configuration keys, notices, and metering.
5. The Radient server-side `POST /v1/decisions` route that makes the first cascade leg real.

Out of scope (deliberately, and recorded so nobody assumes otherwise): the embedder selection
paths, the approval gate, incident classification, compaction, subagent tier choice. Those are
separate decisions with their own gates; this change builds the substrate they would use.

## 3. Vendor contract

All three vendors accept the same request body — TypeSafe's native shape — and return the same
answer shape. That is why the cascade is a list, not a branch.

```jsonc
// POST <endpoint>,  Authorization: Bearer <key>,  Content-Type: application/json
{
  "model": "jev-1.13",              // vendor-specific id; see the table
  "state": "…" | { … } | [ … ],     // string, object or array of strings
  "questions": {
    "recommend_skill_flagship": {
      "type": "choice",
      "instructions": "…",
      "criteria": { "opt-a": "one-line description", "opt-b": "…", "none": "…" }
    },
    "needs_skill_flagship": {
      "type": "noul",
      "instructions": "…",
      "criteria": { "true": "…", "false": "…" }
    }
  }
}
```

```jsonc
// 200
{
  "model": "jev-1.13-20260917",
  "answers": {
    "recommend_skill_flagship": {
      "type": "choice",
      "choice": "opt-a",
      "probabilities": { "opt-a": 0.91, "opt-b": 0.05, "none": 0.04 },
      "confidence": 0.88
    },
    "needs_skill_flagship": { "type": "noul", "noul": 0.94 }
  },
  "usage": { "input_tokens": 812, "output_tokens": 41, "cost": 0.000034 }
}
```

Errors: `401` bad key, `404`/`400` wrong endpoint or body, `422` schema failure (the body names
the offending field), `429` rate limited, `529` upstream overloaded. All are treated as
"vendor unavailable, try the next leg" — never as a user-visible failure.

| leg | endpoint | model id | credential |
| --- | --- | --- | --- |
| `radient` | `https://api.radienthq.com/v1/decisions` | `jev-1.13` | Radient OAuth session (`AuthStore.get_api_key("radient")`), else `RADIENT_API_KEY` |
| `typesafe` | `https://api.typesafe.ai/v1/systemone` | `jev-1.13.0` | `TYPESAFE_API_KEY`, else `JEV_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/alpha/decisions` | `typesafe/jev-1.13` | `OPENROUTER_API_KEY`, else `OPENROUTER_API_KEY_DEV` |

Notes that matter:

- OpenRouter **rejects** this model on `/api/v1/chat/completions` with a 400 naming
  `/api/alpha/decisions`. Do not "simplify" the OpenRouter leg onto the chat path.
- OpenRouter's alpha route validates `choice.criteria` values as **strings**. The rich
  `{what, includes, not_for}` object form that TypeSafe's own docs show for a `choice` is
  rejected there with `invalid_type: expected string, received object`. Send strings.
- A `score` question's `criteria` is an **array** of level descriptions, not a map.
- TypeSafe reports `usage.cost`; OpenRouter does too. If a vendor does not, compute cost from
  the configured price row; never invent a number.
- Rate limit on our OpenRouter key is 0.5 req/s. One call per user message, plus the session
  cache and the circuit breaker, is what keeps that honest.

## 4. Module contract (`local_operator/classification/`)

```python
# local_operator/classification/__init__.py — the ONLY public surface

QuestionKind = Literal["choice", "noul", "score"]

@dataclass(frozen=True)
class Question:
    id: str
    kind: QuestionKind
    instructions: str
    criteria: dict[str, str] | tuple[str, ...]      # map for choice/noul, array for score

@dataclass(frozen=True)
class Answer:
    id: str
    kind: QuestionKind
    value: str | float                              # choice id, noul probability, score value
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None

@dataclass(frozen=True)
class DecisionRequest:
    state: str | dict[str, Any]                     # already bounded; see §5
    questions: tuple[Question, ...]

@dataclass(frozen=True)
class DecisionResponse:
    vendor: str                                     # "radient" | "typesafe" | "openrouter"
    model: str
    answers: dict[str, Answer]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0

class DecisionVendorError(RuntimeError):
    """Any leg failing to produce a usable answer; the cascade catches this and moves on."""

class DecisionVendor(Protocol):
    name: str
    #: Credential resolution is per-vendor because Radient's is an OAuth session
    #: held in AuthStore while the others are plain credential rows.
    async def credential(self, manager: CredentialManager) -> SecretStr | None: ...
    async def decide(self, request: DecisionRequest, *, timeout_s: float) -> DecisionResponse: ...

async def resolve_vendor(
    manager: CredentialManager,
    settings: Mapping[str, Any] | None = None,
) -> DecisionVendor | None:
    """First available leg honouring `values.classification.vendor`; None when none is usable."""

def vendor_status(manager: CredentialManager, settings: Mapping[str, Any] | None = None) -> list[tuple[str, bool]]:
    """[(vendor_name, available)] for notices, tests and diagnostics. Never performs I/O."""
```

Recommendation layer (same package, `recommend.py`):

```python
ResourceKind = Literal["skill", "guide", "mcp"]

@dataclass(frozen=True)
class Candidate:
    kind: ResourceKind
    name: str                       # skill/guide name, or MCP server id
    description: str                # harness-owned text only (see §6 security note)
    resource_url: str               # "skill://x" | "guide://x" | "mcp://x"

@dataclass(frozen=True)
class RecommendationRequest:
    user_message: str
    context: str | None
    candidates: Sequence[Candidate]
    max_recommendations: int = 3

@dataclass(frozen=True)
class Recommendation:
    resources: tuple[Candidate, ...] = ()
    block: str = ""                 # "" means "inject nothing"
    vendor: str | None = None
    cost_usd: float | None = None
    latency_s: float = 0.0
    skipped: str | None = None      # "disabled" | "no-vendor" | "empty-roster" | "timeout" | "error" | "circuit-open"

class ClassificationService:
    def __init__(self, *, manager: CredentialManager, settings: Mapping[str, Any] | None = None) -> None: ...
    @property
    def enabled(self) -> bool: ...
    @property
    def vendor_name(self) -> str | None: ...          # resolved lazily, cached for the session
    async def recommend_resources(self, request: RecommendationRequest) -> Recommendation: ...
    def notice(self, recommendation: Recommendation) -> str | None: ...   # one line, or None
```

`recommend_resources` contract, precisely:

- Returns `Recommendation()` (empty, `skipped` set) rather than raising, for every failure mode.
- Never calls a vendor when: the setting is off, the cascade has no available leg, the candidate
  list is empty, or the circuit breaker is open.
- Caches by `sha256(user_message + candidates_digest)` in a bounded per-session LRU (64 entries);
  a cache hit costs nothing and must be visible in the returned `latency_s` as ~0.
- Opens the circuit breaker after **3 consecutive failures** (transport error, 401/403, 429, 529,
  or a 5xx) and leaves it open for the rest of the session. A 422 (our request was malformed) is
  NOT a transport failure: it disables only that question shape for the session and is logged
  loudly, because it is a bug in this layer, not weather.
- Runs under `values.classification.timeoutMs` (default 1500). On timeout: empty recommendation,
  `skipped="timeout"`.
- Is safe to call concurrently; one in-flight call per (session, cache key) is enough.

## 5. Context budget — the 32k cap and the truncation policy

The model's per-request budget is 64k tokens (32k for `state` plus the longest question). Our
own ceiling is much lower because the entire point is to be cheap, and because a recommendation
that costs more than the resources it saves is a net loss.

Non-negotiable: **the transcript is never sent.** Not the history, not the compaction summary,
not tool results. What goes out is:

```jsonc
{
  "request": "<the newest user message, bounded>",
  "context": "<optional short representative context, bounded — see below>",
  "candidates": {
    "skills": ["name: description", …],
    "guides": ["name: description", …],
    "mcp_servers": ["name: description", …]
  }
}
```

Budgeting, applied in this order until the serialized state fits `values.classification.maxStateChars`
(default 6000 chars ≈ 1.5k tokens):

1. drop `context` entirely;
2. trim each candidate line to 120 chars, then to 60;
3. drop the lowest-priority candidate kind (mcp → guide → skill);
4. truncate `request` to the remaining budget, appending `" …[truncated]"`.

`context` is whatever short, representative, already-redacted string the caller can supply — in
practice the newest compaction summary line or the last assistant message's first line, capped at
600 chars by the caller before it reaches this module. It is optional: with nothing to add, the
state is the user message plus the roster.

A user message longer than the whole budget is classified on its head. That is acceptable
because every answer here is advisory, and the head of a request is where the intent is.

## 5a. Latency budget (operator requirement, 2026-09-18)

The layer runs once per user message on the critical path before the turn's first token, so
**our own overhead must be under 100 ms and ideally under 50 ms**, excluding the vendor's model
time. This is a design constraint, not an aspiration, and it rules several implementations out.

Rules, each of which has to be provable:

1. **One round trip, total.** The decision call is the only network call this layer makes: no
   credential preflight, no model listing, no usage/billing probe, no warm-up request.
2. **Credential resolution is memoized per service, never per call.** Resolving through
   `AuthStore.get_api_key(...)` walks a multi-step cascade over SQLite with possible OAuth
   refresh (`providers/auth_store.py:1652`); it is resolved once, lazily, kept in-process for
   the session, invalidated once on a 401/403 and re-resolved once — never in a loop. The
   Radient leg passes `read_only=True`, as `providers/radient_credentials.py` does, so a read
   cannot move account routing.
3. **One persistent HTTP client with keep-alive.** A fresh TCP + TLS handshake per call is
   50-150 ms on its own and would consume the entire budget before the request is sent.
4. **The roster half of the state is serialized once per roster digest**, so a warm turn
   serializes only the user message.
5. **The concurrent gather.** Wiring awaits this layer alongside the existing selection work, so
   the added wall-clock is the difference between the two, not their sum.
6. **The turn waits a bounded time and a late answer is not lost.** The wiring waits
   `values.classification.waitMs` (50 ms) and then stops waiting. The call keeps running in the
   background, and its recommendation is appended to the NEXT user message's knowledge block —
   which is the harness's existing late-host-state channel (the knowledge section is journaled as
   a `[session-state]` update, so the cached prefix is never rewritten). `timeoutMs` stays the
   CALL's deadline, enforced inside the service: the circuit breaker therefore counts the vendor's
   own failures once per call, never the turn's patience. This is what keeps our added wall-clock
   inside the budget while a real vendor answer is much slower than the budget: measured on a
   27-candidate roster, an answer takes ~250 ms median against OpenRouter, and the budget excludes
   that model time by its terms.
7. **Server side.** `POST /v1/decisions` must ride the agent-server's existing Redis `AuthCache`
   (`internal/cache/auth_cache.go`, 5-minute TTL for API-key→identity and billing balances),
   which is why the route sits on `jwtOrAPIKeyBillingMiddleware`. It must not introduce uncached
   per-request work — in particular, price resolution for the decision model must be a cache hit
   rather than triggering an upstream pricing refetch.

Evidence obligations:

- harness: a hermetic test asserting the local path (build state → cache lookup → credential memo
  hit → serialize) is far under budget, a hermetic test asserting the per-message wait is BOUNDED by
  `waitMs` (a seam that hangs costs the turn the budget and nothing more, and its answer is delivered
  by the following message), plus a measured median and worst-case added wall-clock per user message
  with the layer on versus off, on the real path, reported separately for a cache hit and a cache miss;
- server: a test proving a repeated call performs no second auth-repository lookup, a test proving
  price resolution performs no upstream fetch, and a timed warm-path measurement of the route's own
  handling (excluding upstream inference).

If a future vendor makes the budget unreachable, the correct answer is to keep the feature off by
default and say so — not to exceed the budget silently.

## 6. Security and privacy rails

- **Option text must be harness-owned.** Only the skill/guide description as discovered from the
  local filesystem, and the MCP server's own name plus harness-written capability hints
  (`mcp/resources.py:_CAPABILITY_HINTS`). Never config-authored or remote-authored prose as an
  *option description*: that reintroduces the prompt-injection surface `mcp/resources.py:10-13`
  deliberately excludes. Remote text may still be part of `state` as untrusted data — the model
  is judging a request, and the answer is only a suggestion.
- The state carries the user's own message, which may contain anything. It goes through the same
  redaction boundary as any other outbound text before it leaves the process; if a caller has
  nothing to redact, it passes the raw text and that is recorded here as a known limit.
- The classifier's answers are **advisory**. Nothing in this layer may gate a capability, change
  an approval tier, or alter a tool's availability.
- No secrets in `state`, ever — including no credential values that may appear inside a request.

## 7. Wiring

Hook: the knowledge block for a user row, built in `session_factory.py` (the same place
`SkillIndex.select` is consulted, lines ~1364-1420), which already freezes per `task_id` +
`compaction_id` — i.e. once per admitted user message. That is exactly the cadence we want, and
reusing it means one classification per user message with no new freeze machinery.

Sequence per user message:

1. build the candidate list — the same discovered skills/guides the router sees, plus the
   configured MCP server names with their capability hints;
2. `await asyncio.gather(...)` the existing selection and the classification, so the added
   latency is the *difference*, not the sum — and bound the WAIT for the classification by
   `values.classification.waitMs` rather than by its deadline (see §5a rule 6: a call that misses
   the wait keeps running, and its answer rides the next user message);
3. render the recommendation block, dropping anything already selected by the router, and append
   it to the knowledge/tail block. A LATE answer is rendered the same way by the next admitted
   user message, oldest first, inside the same per-message cap, and is delivered exactly once;
4. emit the notice once per user message, at the moment an answer actually reaches the prompt —
   never when a call times out (nothing was delivered then) and never a second time for an answer
   an earlier turn already rendered.

Rendered block (this is the whole token cost — target ≤ 6 lines):

```
<resource_recommendations>
These may help with this request — read the ones that actually fit, ignore the rest:
- skill://minerva-platform-deployments
- guide://tunnel
- mcp://hubspot
</resource_recommendations>
```

Phrasing rules: advisory ("may help", "ignore the rest"), never imperative, never exclusive, and
never a claim that a resource is authoritative for the turn. A wrong recommendation must cost a
line of context, not a wrong action.

Degradation: if the classifier is disabled, unavailable, timed out or returned nothing, the
block is empty and the prompt is byte-identical to today's. This is asserted by a test.

## 8. Configuration keys

All under `values.classification` (`Section` scoped to new sessions, since the service is built
per session), each with a module-level default constant next to the code that reads it and an
entry in `_consumer_defaults()` in `tests/unit/test_settings_io.py`:

| key | kind | default | meaning |
| --- | --- | --- | --- |
| `auto` | bool | `false` | master switch; off means the prompt is unchanged |
| `vendor` | choice | `auto` | `auto` \| `radient` \| `typesafe` \| `openrouter` (pins one leg) |
| `model` | text | `""` | override the vendor's model id |
| `timeoutMs` | int | `1500` | per-call deadline (the vendor's budget, and the breaker's clock) |
| `waitMs` | int | `50` | how long a TURN waits for an answer; a slower one rides the next message (§5a rule 6) |
| `maxStateChars` | int | `6000` | hard cap on the serialized state |
| `maxCandidates` | int | `12` | candidates sent per kind |
| `maxRecommendations` | int | `3` | recommendations injected per message |
| `notice` | bool | `true` | emit the one-line host notice |

`values.effort.auto` is the precedent for the shape and for defaulting off
(`model/effort_classifier.py:17-23`): an upgrade must never silently change behaviour or spend.

## 9. Provider login for Jev (API key only)

TypeSafe is API-key-only today — there is no OAuth flow to implement. The mechanism is the
existing paste-a-key path (`create_api_key_login`, as `radient-key` uses,
`providers/registry.py:499-509`).

Add one provider row:

```python
ProviderDefinition(
    id="typesafe",
    search_aliases=("jev", "typesafe-jev"),
    name="TypeSafe (Jev)",
    env_keys=("TYPESAFE_API_KEY", "JEV_API_KEY"),
    login=create_api_key_login("TypeSafe", "https://typesafe.ai/", "Paste the API key from the TypeSafe console."),
    base_url="https://api.typesafe.ai/v1",
    decision_only=True,
)
```

`decision_only=True` is a **new field** on `ProviderDefinition`, and the point of it is that Jev
must never be offered as a chat model: it rejects `chat/completions` outright on every host we
reach it through, so a user who picked it in `/model` would get a broken session, and the
fallback chain must never route a turn onto it. Honoured in: model discovery and listing, the
`/model` catalogue ranking, session model resolution, and the failover chain. Each of those
needs a test that names this provider and asserts it is absent.

## 10. Radient server route (`radient-ml/agent-server`)

New route beside the `tools/*` group, same middleware chain as the rest of `/v1`:

- `POST /v1/decisions` — request and response are a passthrough of TypeSafe's shape (§3), with
  `model` optional (defaults to the configured decision model) and an added `provider` field in
  the response.
- Upstream: `TYPESAFE_BASE_URL` (default `https://api.typesafe.ai/v1/systemone`) with
  `RADIENT_TYPESAFE_API_KEY` (falling back to `TYPESAFE_API_KEY`). Absent key ⇒ `503` with a
  clear message, not a 500.
- Billing: an input-token charge at the configured price for the model, recorded through the
  same usage path the other routes use, with **output tokens zero**. A call that fails upstream
  bills nothing.
- Errors: client errors from upstream (`422`) pass through with the body; `429`/`529` pass
  through with `Retry-After` when upstream sends one; any other upstream failure becomes `502`.
- It must **not** appear in `/v1/models` as a chat model (or must carry an explicit non-chat
  kind if the schema gains one) — otherwise a proxied Jev id is offered to clients that will
  400 on the chat path.

## 11. Evidence this change must produce

1. Unit tests for the package: cascade order and fallback, credential resolution per leg,
   context budgeting (each truncation rung, including a message larger than the cap), cache hit,
   circuit breaker open/close, and every `skipped` reason.
2. A wiring test proving the prompt is byte-identical to today's when the layer is off, and that
   the recommendation block is additive and deduped when on.
3. A test proving `typesafe`/`decision_only` is absent from the model list and cannot be
   selected as a session model.
4. An isolated end-to-end run against the real vendor (OpenRouter dev key is available) proving:
   the block is injected for a request that matches a known skill, the correct skill is in it,
   the recorded cost, and the added latency versus `values.classification.auto=false`.
5. The Radient route's own Go tests (success, upstream 429, missing key, input-only billing).
6. Cost arithmetic: measured per-message cost × a realistic session length, stated as a number
   in the PR.
7. Latency evidence per §5a: the hermetic overhead assertion, the measured added wall-clock per
   user message (median and worst case), and on the server the cache-hit proof plus the warm-path
   timing.

## 12. Open questions, recorded rather than guessed

- Whether the recommendation should be allowed to *remove* an embedder-selected resource when
  its confidence is high. Today: no — additive only. Revisit with data.
- Whether a second question ("is one of these actually needed at all?" as a `noul`) earns its
  tokens. The renderer already says "ignore the rest", so it is not required for safety.
- The right default for `values.classification.auto`. Off is the conservative choice that matches
  `values.effort.auto`; a proving run may justify on.
- §5's `context` half is UNBUILT on the caller side: the contract suggests "the newest compaction
  summary line or the last assistant message's first line", and the wiring always passes
  `context=None` (argued in `session_factory._RecommendationRequest`). Benign for an advisory
  layer — the model sees the message, the roster and no transcript — but it is a deviation, and
  the next slice should know the caller-side half is missing rather than assume it works.
