# Design — backend-originated strings: converge on the existing envelope

**Status: proposal for review, 2026-10-20. No implementation in this change.**
Author: architect agent, actual model **openai/gpt-6.1-sol**, temporary fallback
from anthropic/claude-sonnet-5-5 after quota exhaustion.

Source baseline: core `24dbccbb784bd1c9b690ce4601f489791ceacb09`.
Desktop citations below are **local `origin/main` at
`6f4398f451c7a5dfe2b51c20f63d062b06dc75f9`**, read with `git show`/`git grep`,
not the desktop working checkout: that checkout is September 29 revision
`a1e1c573ac8d7d8855e75b52e9ad3ac35174de9e` and has no `src/i18n/`.
References are pinned to those revisions; symbol names identify moved lines.
This document supplements [the accepted i18n design](i18n.md), especially
§§2.5–2.6, rather than replacing its catalogue, eligibility or locale contracts.
It is a source audit and proposed convergence contract, **not a claim about the
installed app, PyPI release, or completion of any client rollout**.

## 0. Decision in brief

**Fix the existing thing; do not add a second envelope or catalogue.**
`Msg`, generated constructors, English/translated catalogues, HTTP serving,
locale resolution and client formatters already exist. The prerequisite is
smaller than the original brief implied, but real:

1. Preserve the catalogue identity when `code` is already a machine category.
   Add optional `message_code` to the existing envelope; do not rename `code`.
2. Establish one render/fallback rule at each transport's existing adapter,
   retaining the envelope instead of collapsing it prematurely into prose.
3. Resolve locale at the CLI/SDK render boundary, not by assuming TUI process
   initialization happened. Keep existing machine tokens and exception contracts.
4. Carry existing message identities through notices/refusals where currently
   lost. Extract locally owned copy incrementally; do not catalogue arbitrary
   provider/model/user text merely because it travels through a tool.

Translations and unmigrated routes remain separate, incremental work. They do
not require another transport design. Backend language authority is already
settled; no connected client-local language override is proposed.

## 1. What exists, and what is actually used

### 1.1 The envelope is implemented, not hypothetical

`local_operator/i18n/messages.py:63–110` defines immutable `Msg(code, params)`,
`Msg.envelope()` and the sole general constructor returning `{code, params,
text}`. `text` is **required in the implemented shape**, not optional. Default
`locale=None` produces English; an explicitly supplied eligible locale renders
translated text. Thus the brief's “English fallback” is the normal default,
not an invariant that `text` can never be localized.

Lookup follows the defining English namespace, then the requested eligible
locale, then English **with English formatting/plural rules**. Unknown identity
or formatting failure returns the identity itself, not a friendly fallback:
`messages.py:38–60,132–168`. `RenderedMessage.source_locale` is internal
provenance, deliberately not wire metadata (`113–120`).

The important exception is `Msg.envelope(code=...)` (`70–86`): it replaces
`code` with an existing category such as `runtime_busy`, retaining params/text
but **discarding the original message identity**. This is compatible with
clients' category switches, but makes later catalogue rendering impossible for
those messages without a category-to-copy mapping in every client.

### 1.2 Measured extraction, not catalogue-presence claims

A Python AST census of direct `HTTPException(...)` construction at the pinned
core revision found:

| Source | Constructors | Direct `.envelope()` detail | Other detail forms |
|---|---:|---:|---|
| `server/desktop.py` | 15 | 9 | 6 literal strings |
| `server/routes/desktop_sessions.py` | 53 | 39 | 14 dict/helper/variable/merged-envelope expressions |
| `server/routes/agents.py` | 82 | 69 | 13 dict/helper/variable expressions |

These are **syntactic counts, not percentages of user-visible errors migrated**.
For example, `desktop_sessions.py:1799–1803` merges an envelope with retry
metadata and is in “other”, despite using the mechanism. Conversely,
`desktop_sessions.py:1780–1785` still emits `{code, message}`. The six literals
in `desktop.py` include pairing/origin checks whose comments say clients use
status, not body (`190–200`); do not make translating them a prerequisite.

Generated calls are real: AST counting imports from `i18n.keys` found 9 calls
in `desktop.py`, 47 in `desktop_sessions.py` (24 distinct functions), and 76
in `agents.py` (40 distinct functions). `agents.py:203–204,292–297,364` already
uses catalogued envelopes, while `460` still returns `{code, message}`.
Publication failures deliberately retain masked upstream prose and machine
metadata via `_publication_detail` (`777–814`), not a translated local sentence.

**Exec is partially extracted, not untouched and not fully localized.**
`exec_mode.py:34–43` imports `wire_exec`; it calls 12 distinct generated
functions at 12 sites. Receipt tails/pointers/wait notices are rendered at
`817–835`, and no-prompt/goal refusals at `1164–1183`. The namespace has 17
English keys; other uses include CLI help (`cli.py:138,1998–2001`). The receipt
skeleton `Background job <id>: <status>` and `exec failed: ` are intentionally
machine-stable, outside the catalogue (`exec_mode.py:817–824,1167–1172`).

`headless_print.py:49–66` uses `cli.exec_runtime` with 11 generated calls/11
distinct functions: thinking/compacting/aborted chrome and error/tool/model
status wrappers are extracted. It still inserts opaque error prose after
provider recovery (`535–553`). Extraction of a wrapper is not translation of
the inserted error. This census does not claim all exec-related modules or
all tool/approval producers have been audited.

SDK startup refusals resolve the **scoped** config and render an existing
message (`sdk.py:587–598`); four startup keys are used at `624–680`.
SDK/spec validation also has 21 source keys, but intentionally renders English
and raises a plain `SessionSpecError` (`sdk.py:908–919,987–1004`;
`session/spec.py:84,335–365,504–545`). These are real calls, not unused keys,
but callers receive only exception prose, not message identity.
`session/spec.py:84–92` explicitly explains the policy: preserve the historical
single-string exception and avoid config reads before SDK root isolation.
Changing that needs a caller-contract migration, not merely another locale argument.

### 1.3 Consumers do not all implement the same capability

- **Desktop knows the envelope's text slot, not a universal wire renderer.**
  At the desktop baseline, `publication-errors.ts:208–224` reads string detail,
  then `detail.text`, then legacy `detail.message`. `desktop-api.ts:360–407`
  likewise preserves category but chooses prose from the envelope. Neither
  translates these errors from `params`. Paths are under
  `src/renderer/src/shared/api/local-operator/`.
- Desktop `src/i18n/messages.ts:64–71,138–155` supports registering served maps
  and arbitrary-key translation. Its served-wire language in the module
  comment is **not integration proof**: `git grep 'registerCatalogue('` across
  `src` found only the definition and test callers. No production served-map
  registration call was established at this revision.
- **Relay has a real runtime and loader**, wired into
  `mobile/web/src/store.ts:85–95`. `i18n/runtime.ts:83–121` selects backend locale
  for `wire.*`, falls back to English, and preserves supplied text on unknown
  identity or invalid params. But a search for `renderEnvelope` in production
  web source found only its definition/return in that runtime, not a caller.
  `mobile/web/src/api.ts:28–43,84–94` still collapses responses to
  `{error, code}` / `HttpError(message, code)`, losing params and message ID.
  The daemon also emits that shape, e.g. `mobile/daemon.py:5039–5050,5646–5651`.
- **Notice schema is still prose-only**, despite the accepted design listing
  optional `code/params` as an adoption point: `harness/types.py:2470–2490`
  has `text`, `kind`, `headline`, not those fields. An extraction emitting a
  private notice format would create the duplication this proposal avoids.

## 2. Why English still reaches users

**Rendering integration is the first gate, not a task parallel to translation.**
The decisive UI cause is the text-only consuming path (§1.3): catalogue
registration in the inspected desktop source is used only in tests. Translating
144 source keys alone changes nothing in that path while routes still send
English `text` and the UI ignores identity/params. Land the identity-preserving
adapter and prove a fixture translation is rendered **before** starting content
translation for that path. Backend-local CLI/SDK display similarly needs its
locale/render boundary first; the desktop adapter cannot localize stderr.

For a healthy known identity, the adapter must render `message_code ?? code`
with `params` through the catalogue, never prefer supplied English `text` over
a successful lookup. Missing target-locale copy uses English **catalogue source**.
`text` is the compatibility fallback for an identity unknown to the active
runtime, not the normal path for known messages. Known identities with malformed
params are defects: show the client-owned generic error and report the identity,
rather than silently masking a broken integration with English wire text (§7).

There is no single fix called “turn on wire.errors”. These are distinct causes:

| Cause, established here | Evidence | Correct kind of fix |
|---|---|---|
| Source keys exist but translations do not | File census: `wire.errors.agents` 44 keys, `wire.errors.desktop` 68, `wire.errors.sdk_startup` 4, `wire.errors.sdk_spec` 21, `cli.exec_runtime` 11: each has **only an `en` file**. Parent `wire.errors.json` has zero keys in every carried locale. | Translate/audit those owned namespaces; adding parent keys does not translate nested files. |
| Backend renders the default English text | `.envelope()` defaults English (`messages.py:102–110`); the cited agent/desktop routes omit locale. | Pass backend-resolved locale at an owned emission boundary, or have a capable client render the retained identity; old readers need resolved text to benefit. |
| Locale resolves correctly but exec render locale stays English | `text.py:63–90` starts active locale at `en`; only production setter call found is TUI `app.py:42850`. `exec_mode.py`/headless use `tr`, not explicit per-call locale. | Initialize/explicitly pass scoped locale at the CLI/SDK display boundary; no dependency on TUI startup. |
| Client supports text compatibility, not envelope localization | Desktop and relay adapters in §1.3. | Retain identity/params through the transport and render once at display. |
| Machine category replaced translation identity | `_runtime_busy_refusal`, `desktop_sessions.py:1789–1803`; `Msg.envelope(code=...)`. | Add `message_code` without changing `code`. |
| Not extracted, or intentionally opaque | Legacy store dict in §1.2; origin checks; upstream publication text; headless provider errors. | Extract owned copy only; preserve status/category and opaque data. |

**A separate content lane, after that gate: 144 English-only keys.** Assign
translation/audit of `wire.errors.agents` **44**, `wire.errors.desktop` **68**,
`wire.errors.sdk_spec` **21**, and `cli.exec_runtime` **11** as a named lane;
44 + 68 + 21 + 11 = **144**. Do not bury this work inside TUI/exec surface
extraction, duplicate keys in clients, or report extraction as translation.
The four English-only `wire.errors.sdk_startup` keys are an additional explicit
scope item, **not included in 144** (148 with that family). Among these compared
families, `wire.exec` is the **only namespace with a second locale**. Each
translated namespace still needs normal independent audit/eligibility gates;
a file count does not authorize activation. Spec translations also require the
human-display adapter, retaining English programmer-facing exception strings.

`wire.exec` is a discriminating control: it has **17 English and 17 Hindi keys**,
so English on its translated messages cannot be explained solely by missing
translation files. The direct runtime check in §8 demonstrates the locale seam.
Arabic and Japanese are not in `i18n/data/shipped.json:1–11` (en/fr/es/ru/vi/hi),
so this design does **not** promise those locales activate today. Locale
eligibility/audit is a separate programme gate, not bypassed by this proposal.

## 3. Transport decision: retain the envelope, add the lost identity

### Options and trade-off

- **Do nothing to shape; render everything on the backend.** Cheapest for
  existing text-only clients. It cannot re-render retained notices on a language
  change and leaves category-overridden messages opaque to clients. Useful
  compatibility behavior, insufficient as the common extraction contract.
- **Rename `code` to a message key everywhere.** Superficially one field, but
  breaks retry/security/navigation/category switches in released clients.
  Rejected: `messages.py:73–81` documents this existing compatibility constraint.
- **Recommend: optional `message_code` on the existing shape.** For ordinary
  messages `code` is already the key. When overridden, carry both identities.
  No second envelope, global category map, or translated-key copy in a client.

Example proposed extension of the existing runtime-busy refusal:

```json
{
  "code": "runtime_busy",
  "message_code": "wire.errors.desktop.runtime_unreachable",
  "params": {},
  "text": "Session owner is unavailable. Reconnect and reconcile before retrying.",
  "retryable": true,
  "retry_after_ms": 2000
}
```

The delay matches `RUNTIME_BUSY_RETRY_AFTER_MS = 2000`
(`desktop_sessions.py:226`); preserve 503, `runtime_busy`, `retryable: true`,
`retry_after_ms: 2000` and `Retry-After: 2` exactly.
`message_code` is optional on the wire; the shared constructor should emit it
when a category override would otherwise lose identity. Readers choose
`message_code` when present, otherwise try `code` as a catalogue identity.
Do not guess that an arbitrary category is a catalogue key or translate by
English substring. A supplied unknown `message_code` falls back to `text`,
not to a guessed category mapping.

`params` must be named JSON scalar values suitable for ICU; no exceptions,
objects, markup, preformatted HTML, credentials or raw provider payloads. Keep
counts numeric for new messages. Do not silently change existing string-typed
parameters (e.g. `wire_exec.tools_unreachable`); version semantic/type changes
with a new key. Paths, names and opaque prose stay data, escaped by the display
adapter; bidi isolation belongs at paint, not in persisted raw values.

**Cost now:** one constructor addition, adapter schema/tests and one small error
family. **Cost later:** the same work plus redoing extracted notice/SDK types,
client-specific category maps and persisted data assumptions. Settle the additive
identity field before those extractions. A global envelope version is unnecessary
for this additive field; incompatible future semantics require a new contract,
not silently redefining existing codes or parameters.

## 4. Catalogue ownership and locale authority: reuse settled decisions

**One source per message, not one manually copied catalogue per client.**
Backend-originated `wire.*` source, context sidecars, translations and generated
Python keys remain in core `local_operator/i18n/`. Core serves them using
`GET /v1/i18n/catalogues/{locale}/{namespace}` with raw-file hash/ETag and
eligibility checks (`server/routes/i18n.py:41–73`). `i18n/http.py:1–25` already
shares validator mechanics with relay. Relay's existing loader has bounded,
origin/locale-scoped caching and withdrawal (`wire-cache.ts:50–81`); extend its
use, not its ownership. Generated client key types are derived artifacts, not a
second authored source (`mobile/web/src/i18n/keys.gen.ts:1–2`). Client-owned
`ui.*`/`relay.*` chrome may stay in their existing owned catalogues; **the same
backend message must never be re-authored in those namespaces**.

Backend `language` is already authoritative: explicit config → `LOP_LANG` → OS
→ English, then shipped eligibility (`resolve.py:304–349`); kill switch wins
(`293–301`). `LOP_LANG` does not override explicit language config. Capabilities
publish the resolved tag (`server/routes/capabilities.py:28–55`) and a shared
config frame carries `{changed_keys, language}` (`i18n/frames.py:12–26,37–45`).
Desktop `src/i18n/locale.ts:60–70` explicitly names the same backend setting.

**No persistent client-local override while connected.** This is already the
accepted design's §2.5, lines 328–330. A picker writes backend `language`; all
connected surfaces then follow it. Device locale is a preconnection fallback
only; tests may use local pins. Existing manual locale APIs are not permission
to create independent preferences. Wire text rendering uses backend locale,
not the preconnection UI locale (relay runtime already separates those).
Disconnected/old clients display provided fallback text; they do not invent a
new backend setting. No request-header locale negotiation is required.

`text` stays present, English by default for compatibility; an emitter that
already has backend-resolved locale should provide localized text for text-only
clients. Developer logs must record category/key plus developer-authored English
context, not repurpose a localized `text` field as a stable log template.

## 5. Boundaries that extraction must not erase

| Content | Treatment |
|---|---|
| Product-owned notices, approval chrome, tool-error wrappers, progress labels | Catalogue; preserve identity until the display boundary. |
| Model-facing system/tool prompts, developer diagnostics/log templates | English, not product catalogue. |
| Assistant conversational output, including streamed tokens | Model language instruction, not catalogue or post-hoc translation. |
| User text, file contents/paths, external/provider/model error prose, arbitrary tool payload | Opaque data; never use as a lookup key or fabricate a translated equivalent. |
| Machine categories, status enums, CLI flags, receipt prefixes, IDs, JSON field names | Stable protocol, not translated display labels. |

A `{error}` passthrough entry does not translate the error. Existing
`desktop_refusal` explicitly preserves it (`test_wire_errors_desktop.py:78–82`)
and `_publication_detail` masks upstream content before transport. Retain those
security boundaries. Map a known locally owned refusal to a local message where
possible; never classify arbitrary errors by English text merely to localize.

The existing English preference instruction deliberately uses
`resolve_preference`, which can differ from shipped UI locale
(`resolve.py:352–365,414–436`). Keep that distinction: conversational language is
not blocked because a UI namespace lacks an audited translation. Never pass
agent text through `Msg`, scrape it for keys, or catalogue generated answers.
An extraction inventory should classify each producer as owned copy, protocol,
model-facing or opaque, with narrow exclusions rather than blanket “tool text”.

### 5.1 Boundary invariant and a blocking enforcement story

**Invariant:** catalogue rendering may enter only an explicitly inventoried
human-display or human-wire sink. It must not flow into model system/tool
instructions or developer log templates. Conversational model output stays
opaque and follows `preferred_language_line`, not catalogue translation. A
message appearing in a tool does not make it product chrome.

**Current enforcement is insufficient.** Core's checker counts source literals,
checks catalogue/placeholder parity and validates sidecar shape; its sidecar
validation checks unknown keys, object shape and `maxCells`, not a semantic role
allow-list (`scripts/i18n/check.py:589–607`). Outside activated scopes, source
ratchet findings are advisory and exit zero (`check.py:714–730`), as CI documents
(`.github/workflows/ci.yml:388–421`). That does not stop someone translating a
system prompt. The reported UI advisory gate's 35 breaches are review input,
not a count re-derived here and not evidence of enforcement. Neither advisory
status nor a `# i18n: ignore` pragma may waive the invariant below.

**Recommend an explicit, default-deny eligibility manifest plus a blocking AST
boundary check in the existing i18n checker.** A namespace deny-list alone rots;
paths alone cannot separate display copy from prompts in mixed modules. The
manifest should name each owned namespace, owner, allowed source module/symbol,
and allowed human sink role (display, approval chrome, notice, human error,
progress). Existing carried namespaces need explicit inventory, not a wildcard
`wire.*` grant. Bootstrap requires a reviewed inventory of existing keys
and their actual sinks; automatically allowing every current entry would be a
baseline exemption, not boundary enforcement. It is a policy index referencing the existing catalogue, **not**
another copy store or another translation runtime.

The proposed check fails nonzero on:

1. A new/changed source key without manifest eligibility and a context sidecar
   declaring an allowed human role. `model-facing`, `system-prompt`, `tool-prompt`
   and `developer-log` roles cannot be catalogue-eligible. Reserved `log.*` and
   `wire.prompt.*` namespaces are rejected as defense in depth, not the main rule.
2. Translation/key-construction calls outside inventoried source symbols, even
   when the key itself is an allowed human key. Resolve direct imports/aliases
   of generated constructors, `Msg`/`msg`, `render`, `tr` and `.envelope`, plus
   client `t`/`translate` equivalents; unknown dynamic key construction requires
   a reviewed generic *display adapter* declaration, never a blanket path waiver.
3. Direct catalogue-derived expressions/local aliases supplied to known model
   prompt/message assembly sinks or developer logging calls. Scan these sinks
   even inside eligible mixed-purpose modules: importing a human key and then
   calling `logger.info(tr(...))` or appending it to a system prompt must fail.
   The sink inventory must include the actual prompt builders and logging APIs
   discovered in the implementation census, not just names containing `prompt`.

**What happens when it fires:** the existing CI i18n job fails; no baseline
increase, advisory switch or string-ignore pragma suppresses boundary findings.
The check runs on catalogue, manifest, generator/checker and relevant Python/TS
source changes; CI scope must fail closed on unclassified source changes. The
required-check/aggregate wiring must be verified with a deliberately failing PR
fixture before calling this a merge gate. No claim that this gate ships today.

**Build scope:** one focused guard implementation lane: add the policy manifest
and role schema, extend catalogue checks and alias-aware AST use-site checking,
inventory current call sites/model/log sinks, add equivalent checks to existing
client source tooling where these calls can occur, and wire CI classification.
Then make generators reject ineligible namespace/role declarations before
emitting artifacts. Until all languages are covered, report each uncovered
language as a hole, not an enforced surface. Do not grandfather forbidden prompt
or log usage; split mixed modules at symbol boundaries, or remove the misuse.
Do not block on a repository rewrite: inventory existing keyed use sites first,
then make new/changed uses default-deny from the first guard PR.

**Proof required before activation:** negative fixtures must show nonzero exit
for a keyed system prompt, a keyed tool instruction, `logger.info` with keyed
text, an alias of a renderer in each sink, a forbidden sidecar role, an
unregistered key/namespace, and a new unclassified source path. Passing controls
cover a human approval prompt, escaped opaque params, model output passthrough,
and English log text next to localized UI copy. Demonstrate the CI job/aggregate
fails too; green unit tests of the checker alone are not merge enforcement.

**Known holes:** static direct-call/local-alias analysis is not whole-program
information-flow proof. A helper returning an unannotated string, reflection,
dynamic imports, translation outside these runtimes, or falsely declaring a
model sink as a human sink can evade it. Runtime semantic intent cannot be
inferred from English wording or sidecar assertions. Review must verify manifest
changes against actual sink use, and prompt-boundary tests should assert that
controlled locale switches leave assembled system/tool instructions and log
bytes unchanged while human chrome changes. Such snapshots cover only exercised
branches and must not make model prompts catalogue-owned. A later dataflow or
typed display-message restriction can reduce those holes; it is not prerequisite
to the bounded guard above. This design explicitly does **not** claim airtight
protection against deliberate evasion or arbitrary cross-function string flow.

## 6. Gaps ranked by extraction dependency, not as a wish list

1. **Identity/adapter contract (blocking shared extraction and content translation).** Preserve message
   identity alongside machine category and agree on §7 fallback. Otherwise each
   client must invent a category map or different envelope. This is the smallest
   structural prerequisite. Prove the consumer renders a fixture translation
   before the 144-key content lane starts; text compatibility alone is not enough.
2. **Non-TUI locale boundary (blocking useful exec localization).** Exec and
   SDK display need scoped resolution independently of TUI's global setter.
   Prefer explicit locale through render helpers for SDK/concurrent roots;
   initialize active locale once only at a single-root CLI entry. Do not mutate
   process-global locale per SDK request. SDK already resolves after root
   isolation (`sdk.py:587–598`); preserve that order.
3. **Identity-preserving producer types (blocking notice/SDK extraction as those
   families are touched).** Add optional envelope fields to `NoticeEvent` rather
   than private event types; retain `text/kind/headline`. Expose optional
   `message_code/params` on owned SDK validation errors while keeping the existing
   exception class, `args` and English programmer-facing `str(error)` stable;
   localize only at a human display adapter. Unknown exceptions remain opaque.
   These are family-sized changes, not a mandatory all-events rewrite. A future
   separately rendered headline needs its own message, not blind truncation.
4. **Translations and legacy producers (incremental, not mechanism blockers).**
   Missing namespace translations, remaining `{code,message}`/string routes,
   tool receipts and progress families are owned slices with audit/QA. Do not
   claim global completion from the counts in §1.
5. **English-parsing consumers (block translation of their own paths).** Relay
   `pending-card.tsx:80–101` still uses English substrings; migrate those decisions
   to categories before changing their incoming text. This is not justification
   for delaying unrelated error-family extraction.

## 7. Stated failure behavior

| Condition | Current evidence | Convergence rule |
|---|---|---|
| New backend identity unknown to client | Relay runtime returns original text (`117–121`); desktop reads text directly. | Try lookup without unsafe casting to generated keys; on unknown identity show `text` unchanged. Do not display the unknown key when usable text exists. |
| Missing target-locale key/namespace | Python `render_record` falls back to English source and English formatter (`145–162`); relay runtime does likewise (`83–109`). | Keep that rule, including source-locale direction. Never format English fallback using requested-locale plural rules. |
| Missing/invalid params or malformed message | Python returns the code (`159–168`); relay returns supplied text on formatting failure. | A known but unformattable identity shows the client-owned generic error and emits a parameter/identity diagnostic; `text` is not the normal rendering bypass. No placeholder leakage/crash. Backend constructors remain typed/checked. If backend rendering fails and only the ID survives today, adapter shows its existing localized generic failure and preserves the ID for diagnostics. Improving backend renderer recovery is separate from silently changing this contract. |
| Older client, newer backend | Existing text slot is understood; fields are additive. | Keep all established text/legacy message/error slots and status/category semantics at each transport boundary. Older raw-string-only readers need their existing adapter compatibility tested; a dict was never automatically compatible with a string-only parser. |
| Newer client, old backend/string or `{code,message}` error | Desktop already accepts both. | Preserve legacy reader branches; no assumed params. Show existing text/message/error string. |
| Catalogue fetch unavailable, unsupported locale, stale cache or feature withdrawn | Core route returns 404 for unavailable namespace/locale; relay invalidates on capability withdrawal (`wire-cache.ts:68–81`). | Use available English source, otherwise supplied text; never retain another origin's or withdrawn locale's maps. No request retry solely to translate an error. |
| No useful text and no renderable identity | Current Python can expose just a key. | Display existing client-owned generic error plus machine status where appropriate; retain key for diagnosis, no invented translation. Missing fallback is a producer defect caught in fixtures. |

Do not stamp unknown/passthrough fallback with the requested language or direction;
relay's null provenance already models that. No unbounded “retry rendering” loop,
no raw stack trace substitution, no translation failure changing a command exit,
HTTP status, approval result, retry policy or job state.

## 8. Smallest convergence step and evidence

**First migration: the existing `_runtime_busy_refusal` family, not a new route.**
It is already centralized and emitted from two paths
(`desktop_sessions.py:1789–1803`), has a real machine-category collision and
retry semantics that must not regress. Extend `Msg.envelope(code=...)` to retain
`message_code`, consume it at one existing desktop error adapter, and retain all
current category/text/retry fields and legacy reader branches. Use isolated
catalogue fixtures to prove non-English rendering; missing production
translations must not be papered over by writing a duplicate client key.
No bulk conversion and no new locale release are part of that first step.

Acceptance for that future implementation: actual refusal response has unchanged
503/category/retry header/delay and added correct message identity; old reader
gets identical text; new reader renders a fixture translation; unknown identity,
missing params fail safely with a generic error, unknown identities/catalogue
unavailability use fallback text, and retry behavior does not change. Translation rollout for this family requires the normal translation and
rendered-review gates. Then separately wire exec's locale seam and notice/SDK
metadata before extracting those families. This document implements none of it.

### Retry-path proof still required, not supplied by the 26 tests

The 26 tests below cover the envelope, key rendering and byte samples; **they do
not exercise HTTP 503 or retry behavior**. Existing relevant source tests already
provide a useful foundation:

- `tests/unit/server/test_desktop_read_without_owner.py:399–441`,
  `test_a_control_route_answers_a_silent_owner_busy_and_fast`, drives an ASGI
  request against a synthetic socket owner in both welcome-then-silent and
  never-welcome states. It asserts **503**, `code == "runtime_busy"`,
  `retryable is True`, **2000 ms**, **`Retry-After: 2`**, a bounded response time
  and the existing sentence prefix. This is source inspection, **not a test run
  performed for this document**.
- `tests/unit/server/test_desktop_sessions.py:1820–1874` exercises the separate
  interrupt/`OwnerAckTimeout` route, asserting 503/category/retryable and one
  actual `abort` dispatch, rather than an invented idle result. It does not
  assert delay/header/byte-exact text in the inspected assertions.

The implementation owner must run both paths before/after on isolated synthetic
owners, extend **both** to assert the full unchanged status/category/retry
fields/header and byte-exact fallback text plus additive `message_code`, and
capture actual responses. Add client checks that known translated copy still
triggers the same retry disposition/delay, unknown identity preserves legacy
behavior, and old text-only parsing reads identical text. A response-shape test
alone cannot prove no extra retry or approval changed; count dispatched operations
and scheduled retries in the real adapter path. Preserve the existing timeout
bound; do not manufacture slow production owners or touch live sessions.

### Checks derived during this design

In isolated HOME/config, stripped inherited environment, source `.venv` interpreter:

- Runtime probe with `LOP_LANG=hi`: `resolve_language('auto') == 'hi'` while
  `active_locale() == 'en'`. Explicit
  `wire_exec.waiting_on_you().envelope(locale='hi')` produced Hindi; default
  `.envelope()` and fresh-process `tr('wire.exec')` both produced
  `Waiting on you? lop sessions shows what a run needs`.
- `desktop.runtime_unreachable().envelope(locale='hi')` produced English, as
  expected for a missing Hindi namespace. With `code='runtime_busy'`, output was
  exactly `{code: runtime_busy, params: {}, text: <English sentence>}`: the
  translation identity was absent. This verifies the structural gap independently
  of a missing-translation hypothesis.
- Unknown `wire.future.no_such_key` returned itself in `text`; missing params for
  `wire.exec.log_pointer` also returned the key. These are observed degradations,
  not claimed friendly fallbacks.
- Targeted existing tests: isolated `python -m pytest -n 0
  tests/unit/i18n/test_messages.py
  tests/unit/i18n/test_wire_errors_desktop.py
  tests/unit/i18n/test_wire_exec.py -q`: **26 passed in 1.54s**, 101 warnings
  (Pydantic deprecations and pytest pre-existing temporary-directory cleanup
  warnings). No full suite or CI run was warranted for the proposal.
- Actual isolated `.venv/bin/lop exec </dev/null` under `LOP_LANG=hi` exited 1
  at the earlier bootstrap guard with `Error: Hosting platform is not configured.`
  This did **not** reach the no-prompt receipt path, and is not evidence that
  foreground/background jobs or localized exec end-to-end work. An initial
  `python -m local_operator` attempt failed because no `__main__` exists;
  `pyproject.toml:381,389` identifies the supported entry point used on retry.

### Not established, and rollout risks

No installed desktop or browser was exercised; this role has no browser tool.
Desktop source integration findings are pinned-source evidence, not rendered
proof or a claim about installed binaries. No live service, model job, approval
flow, native mobile app or all-route census was run. Exact convergence coverage
outside the named files needs an owner-scoped inventory, not extrapolation.

Watch category-switch regressions (especially retry/security), relay flattening
loss, locale reads before SDK isolation, English-parsing consumers, catalogues
misattributing opaque prose, ICU parameter type drift, stale origin-scoped maps,
and RTL provenance/escaping. Independent code/QA and rendered design/UX review
belong to the later implementation PRs; this source-based proposal supplies no
waiver of those gates.
