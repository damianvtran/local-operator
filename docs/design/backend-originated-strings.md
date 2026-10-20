# Design — backend-originated strings: converge on the existing envelope

**Status: proposal for review; baseline-corrected after review round 1.**
**Single source baseline:** `fccedfa751d4ffcfd347d69fd991b451f2c99b01`.
Every source observation, count and file:line citation below was re-derived from
that revision. Symbols identify the intended anchors if line numbers move.
This supplements [the existing i18n design](i18n.md), not another catalogue or
translation runtime. No implementation is included.

Provenance: the original architect session disclosed temporary use of
**openai/gpt-6.1-sol** after Anthropic quota exhaustion. This corrected document
retains that disclosure, but does not retain its earlier branch's measurements
or execution results as evidence for this baseline.

## 0. Decision

**Fix the existing mechanism; do not add a parallel envelope.**

- Keep required `{code, params, text}`. Preserve a separate optional
  `message_code` when `code` is an established machine category.
- Prove the identity/params consuming path **before** content translation for
  that path. Existing fallback text alone is not proof of catalogue rendering.
- Keep backend language authority and core-owned shared catalogues.
- Reuse the **already implemented** notice fields and producer helpers; do not
  propose adding `NoticeEvent.code` or `NoticeEvent.params` again.
- Treat boundary enforcement as a bounded, blocking regression gate with an
  independently reviewed sink inventory, not an information-flow guarantee.

## 1. What is already built

### 1.1 Envelope and rendering

`local_operator/i18n/messages.py:63–110` defines `Msg(code, params)` and the
existing `{code, params, text}` constructor. `text` is required, not optional.
Omitting locale renders English; an explicitly supplied eligible locale may
render translated text. Thus English fallback is the default, not a rule that
`text` can never be localized.

`messages.py:132–168` resolves the requested eligible locale, falls back to
English source **with English formatting/plural rules**, and returns the key
itself if the identity is absent or formatting fails. Those last cases are
observed degradations, not friendly fallbacks.

`Msg.envelope(code=...)` (`70–86`) preserves an existing machine category by
replacing the original message ID in `code`. It retains text/params but loses
the render identity. That remains an actual structural gap on this baseline.

### 1.2 Source and translation census

The following counts were derived by parsing the JSON files under
`local_operator/i18n/catalogues/` at the single baseline. A source key count is
not a count of migrated routes or evidence of rendered translation.

| Namespace | English keys | Other catalogue present | Remaining content scope |
|---|---:|---|---|
| `wire.errors.agents` | 44 | None | English-only |
| `wire.errors.desktop` | 83 | None | English-only |
| `wire.errors.sdk_spec` | 21 | None | English-only |
| `cli.exec_runtime` | 11 | French: all 11 keys | Already fully translated to French by key coverage; not English-only debt |
| `wire.errors.sdk_startup` | 4 | None | Additional English-only family |
| `wire.exec` | 17 | Hindi: all 17 keys | Already has a second locale |

The named four families (agents, desktop, SDK spec, exec runtime) total
**159 source keys: 44 + 83 + 21 + 11**. Within them the English-only lane is
**148 keys in three families: 44 + 83 + 21**. Adding SDK startup makes
**152 English-only keys**. Do not count the 11 French exec-runtime translations
as absent, and do not claim `wire.exec` is the only second-locale namespace.
Catalogue presence/key coverage does not establish native-language quality,
audit sign-off, installation or live rendering.

**Recommendation:** give the 148-key translation/audit lane its own owner and
honest completion count after its consuming paths work. Track the extra four
SDK-startup keys explicitly if included. Do not bury content work inside TUI or
exec extraction, recreate shared keys in clients, or count extraction as
translation. Preserve the existing French and Hindi work rather than redo it.

### 1.3 Actual call-site adoption

A Python AST census of `HTTPException(...)` constructors on the baseline found:

| Source under `local_operator/` | Total constructors | Direct `.envelope()` detail | Other expressions |
|---|---:|---:|---:|
| `server/desktop.py` | 15 | 9 | 6 literal strings |
| `server/routes/desktop_sessions.py` | 56 | 39 | 17 helper/variable/dict/merged expressions |
| `server/routes/agents.py` | 82 | 69 | 13 helper/variable/dict expressions |

These are syntactic constructor counts, **not conversion percentages**. An
`envelope() | {...}` or helper result is in “other” and may already be migrated.
For example, the centralized runtime-busy helper returns a merged envelope
(`desktop_sessions.py:1812–1827`), while the store-error path still constructs
`{code, message}` (`1799–1808`). Publication errors retain masked upstream prose
in `{code, params: {error}, text, details}` (`agents.py:784–820`).

The same AST census found **12 generated-key calls, all distinct**, in
`exec_mode.py`, and **11, all distinct**, in `headless_print.py`. These are real
calls, not merely declared keys. They do not prove the entire exec surface is
extracted. Exec receipts include a protocol-owned skeleton composed outside the
catalogue (`exec_mode.py:817–829`); it must not become translated protocol.
Headless copy actually uses `cli_exec_runtime` and a `tr` resolver
(`headless_print.py:49–66`), so its French catalogue is relevant existing work.

**Notices are already structurally migrated.** `NoticeEvent` is defined at
`harness/types.py:2499`; `code` and `params` are at `2507–2508`. Existing helpers
in `harness/loop.py:130–148` and `session/session.py:384–401` populate those fields
from the existing envelope. Reuse them for remaining owned notices. No new
notice type or duplicate optional-field migration is needed.

SDK startup already resolves locale after scoped configuration and renders
messages (`sdk.py:587–598`). SDK/spec validation intentionally preserves English
single-string `SessionSpecError` diagnostics and avoids configuration reads
before isolation (`session/spec.py:84–92`; examples `sdk.py:908–919,987–1004`).
Do not call this a forgotten locale argument or silently change that contract.
A future human adapter can expose structured metadata while preserving exception
class, `args` and programmer-facing `str(error)`; that is a separate API decision.

## 2. Why English can persist, and the order of work

There is no single fix called “turn on wire.errors”. The baseline establishes
separate causes and therefore separate fixes:

| Cause | Evidence on this baseline | Required kind of work |
|---|---|---|
| Translation absent | The 148-key English-only lane in §1.2 | Translate/audit those namespaces, after rendering integration |
| Default English projection | `messages.py:102–110` | Supply resolved locale at an owned emission boundary or render retained identity in the client |
| Category replaces render identity | `messages.py:70–86`; runtime-busy helper | Add optional `message_code`, not a client category-to-copy map |
| Locale resolver and display locale are different seams | `text.py:63–90` initializes active locale to `en`; base-local probe in §7 resolves `fr` while active locale stays `en` | Resolve/activate locale at the owned non-TUI display boundary; do not assume resolution alone activates `tr` |
| Consuming adapter retains prose/category but not params | Relay `mobile/web/src/api.ts:34–49,90–100` builds `HttpError` from `error` and `code` | Preserve the envelope to the display boundary, then use the existing runtime |
| Opaque upstream prose or legacy producer | Publication/store paths in §1.3 | Key locally owned copy only; retain opaque data and machine behavior |

Relay already has an actual message runtime and loader
(`mobile/web/src/store.ts:97–103`). Its runtime looks up source and renders ICU,
and `renderEnvelope` preserves supplied text on failed rendering
(`mobile/web/src/i18n/runtime.ts:83–121`). The HTTP adapter's data loss is a
specific integration gap; adding another runtime does not fix it.

**Rendering integration gates content translation, not a parallel task.** For
each selected path, first prove the adapter reads `message_code ?? code` plus
`params` through the catalogue and displays a fixture translation. Only then
start that path's content lane. Translating files alone cannot help an adapter
that receives English prose and discards message params. The desktop client is
a separate repository: this baseline does not establish its implementation or
completion state, so neither a current desktop defect nor a completed fix is
asserted here. Its adapter must satisfy the same acceptance contract.

Known valid identities use catalogue rendering, not preferred wire text. Missing
requested-locale copy uses English catalogue source. Use `text` for an identity
unknown to the active runtime; do not silently hide known malformed params behind
English wire text. Handle that defect safely as specified in §6.

## 3. Transport: the smallest additive correction

### Options

- **Render everything on the backend, no shape change.** Cheapest for old readers,
  but category-overridden messages cannot be independently rendered by clients
  or re-rendered from retained identity on language change. Useful compatibility
  behavior, not a sufficient common extraction contract.
- **Make `code` always a catalogue key.** Breaks established machine categories;
  `messages.py:73–81` explicitly documents why overrides exist. Reject.
- **Recommend optional `message_code` within the existing envelope.** Keep
  `code` semantics, params and required fallback text. Emit the extra identity
  when an override would otherwise discard it. No parallel envelope/runtime.

Proposed extension of the existing runtime-busy response detail:

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

Current delay is 2000 ms (`desktop_sessions.py:226`); the helper returns HTTP
503 and `Retry-After: 2` (`1812–1827`). This proposal changes neither.
Readers use `message_code` when supplied, otherwise try `code` as a catalogue
identity. Unknown supplied identity uses text, not a guessed category mapping.
Do not translate machine categories by English substring.

Params should remain named JSON scalar values for ICU; preserve parameter types
and semantics for existing keys. Counts should be numeric for new messages.
Paths/names/opaque payloads remain data, escaped at paint rather than interpreted
as lookup keys or markup. Do not transport exceptions, credentials or arbitrary
objects as params. Keep existing masking at publication boundaries.

**Cost now:** one constructor addition, adapter/schema tests and one error family.
**Cost after extraction:** the same work plus redoing clients' invented category
maps and retained-message assumptions. Settle this before layering more shared
extraction on top. An additive field needs no second versioned envelope; future
incompatible semantics must not silently redefine an existing key.

## 4. Catalogue ownership and locale authority

Reuse core-owned `wire.*` catalogues, contexts and generated Python constructors.
The existing route serves namespace maps with hash/ETag and eligibility checks
(`server/routes/i18n.py:41–73`). Clients may cache derived data, not independently
author copies of the same backend messages. Relay already has a bounded loader
with context/capability invalidation (`mobile/web/src/i18n/wire-cache.ts:50–81`).
Client-owned chrome can keep its owned namespaces; do not re-author a backend
message there. There is no proposal to replace the existing generated keys.

Backend resolution already uses explicit config → `LOP_LANG` → OS → English,
filtered through shipped eligibility (`i18n/resolve.py:304–349`). Capabilities
publishes resolved language (`server/routes/capabilities.py:28,46–55`) and the
shared config-frame builder carries changed keys and language
(`i18n/frames.py:37–45`). Keep that authority. No connected client-local override
is proposed: a language-setting control should write the backend setting;
device-local behavior before connection must not override connected wire copy.
For SDK/concurrent roots prefer explicit scoped locale in human render adapters,
not mutation of process-global locale per request. Retain startup's existing
root-isolation-before-resolution ordering.

## 5. Do-not-key boundary

| Content | Treatment |
|---|---|
| Deterministic product notices, approval chrome, locally owned error/progress copy | Catalogue; retain identity to display |
| Model-facing system/tool instructions and developer log templates | English, outside product catalogue |
| Agent conversational output and streamed tokens | Language by instruction, never catalogue/post-hoc translation |
| User text, file/path contents, arbitrary tool/provider/model prose | Opaque data, not lookup keys |
| Categories, status enums, flags, receipt skeletons, IDs, JSON names | Stable protocol, not translated labels |

The existing `preferred_language_line` is deliberately English and uses the
supported preference rather than translation eligibility
(`i18n/resolve.py:411–436`). Preserve that separation. No lack of a UI translation
should force model conversation into English. A `{error}` passthrough envelope
does not translate its embedded prose; preserve masking and provenance rather
than pretending an entire tool result is product copy.

### 5.1 Invariant with a bounded enforcement story

**Invariant:** catalogue rendering may enter only inventoried human-display or
human-wire sinks; not model-facing system/tool instructions or developer log
templates. Agent output remains opaque and follows language instruction.

Current checking is insufficient for this invariant. Context validation checks
key/object shape and `maxCells`, not semantic role eligibility
(`scripts/i18n/check.py:592–610`). The source ratchet has advisory, exit-zero
findings outside activated scopes (`1071–1098`), and CI documents that distinction
(`.github/workflows/ci.yml:388–421`). Existing advisory gates are not the proposed
boundary enforcement; no unverified external breach count is used here.

**Recommend a default-deny eligibility manifest plus blocking AST boundary
checks in the existing i18n tooling.** A namespace deny-list alone rots; a path
allow-list cannot distinguish human copy and prompts in mixed modules. Record
owned namespaces, owners, allowed source modules/symbols and human sink roles.
Inventory existing uses explicitly; do not auto-authorize all existing keys or
allow arbitrary `wire.*`. This is a policy index, not another copy store.

The proposed check fails nonzero on:

1. New/changed keys without manifest eligibility and an allowed human context
   role. Model-facing/system-prompt/tool-prompt/developer-log roles cannot be
   catalogue-eligible. Reject reserved `log.*` and `wire.prompt.*` namespaces as
   defense in depth, not as the principal protection.
2. Translation/key-construction calls outside inventoried source symbols, even
   when the referenced key is eligible. Resolve direct imports/local aliases of
   generated constructors, `Msg`/`msg`, `render`, `tr`, `.envelope`, and client
   equivalents. Generic dynamic display adapters need explicit reviewed entries,
   not a blanket source-path waiver.
3. Direct/local-alias catalogue-derived expressions flowing into independently
   inventoried model prompt/message-assembly sinks or developer logging APIs,
   including inside otherwise eligible mixed modules.

**Known prohibited sinks are checked independently of eligibility metadata.**
Hand-maintenance is not fatal if those checks are independent and manifest edits
receive independent review against actual usage. **Lying about a role cannot
waive the known-sink check.** The gate becomes decorative if role declarations
themselves establish sink classification, or if nobody verifies actual use.

This is **a credible bounded regression gate, not airtight information-flow
protection; its reviewed sink inventory remains an explicit human trust
boundary.** That is the recommendation, not a claim this guard exists today.

**Failure consequence and build scope:** extend catalogue role validation and
alias-aware source checks; inventory real model/log sinks separately from key
metadata; make generators reject ineligible declarations; cover applicable
Python/TS tooling; wire the existing CI i18n check and fail-closed change scope.
Boundary violations cannot be suppressed by advisory mode, baseline increases
or string-ignore pragmas. Verify required-check/aggregate failure using a
negative PR fixture before calling it a merge gate. Add failing controls for a
keyed system prompt, keyed developer log and a prompt/log call falsely labelled
human; passing controls include localized human chrome beside unchanged English
instructions/logs. Do not claim uncovered languages are enforced.

**Holes and rollout risks:** static direct-call/local-alias analysis is not
whole-program flow analysis. Reflection, dynamic imports, unannotated helpers
returning strings, new unrecognized sink APIs and translation outside tracked
runtimes can evade it. Independent review must verify manifest/sink changes
against actual use. Controlled locale-switch tests should leave assembled
system/tool instructions and log bytes unchanged while human chrome changes;
such tests cover only exercised branches. Deliberate evasion and arbitrary
cross-function string flow remain outside this bounded gate. Later dataflow or
typed-display restrictions may reduce those holes; do not claim they ship here.

## 6. Failure and compatibility contract

| Condition | Required convergence behavior |
|---|---|
| Unknown identity/newer backend | Use usable supplied `text`; no unsafe cast to generated keys, guessed category map or displayed raw key when prose exists |
| Missing requested-locale key/namespace | Use English catalogue source and English formatting/plural rules |
| Known identity with malformed/missing params | Safe client-owned generic error plus identity diagnostic, not a silent normal-path bypass to English wire text; no crash or placeholder leakage |
| Catalogue unavailable and no source loaded | Treat identity as unknown to runtime, use text; no unbounded fetch/retry solely to translate an error |
| Older client/newer backend | Preserve established text/message/error slots, status/category/retry fields and legacy adapter handling; structured detail is not automatically compatible with a raw-string-only reader |
| New client/old string or `{code,message}` backend | Retain legacy string/message reader branches without assuming params |
| No renderable identity or useful text | Existing client-owned generic error/status; retain ID for diagnostics, no raw stack substitution |

The Python renderer currently returns the key on unknown identity or format
failure (`messages.py:142–168`); relay currently returns supplied text on failed
rendering (`runtime.ts:117–121`). The table is the proposed convergence rule,
not an assertion those implementations already match every row. Do not label
unknown/passthrough prose with the requested language/direction. No localization
failure may change HTTP status, approval outcome, retry policy, exit or job state.

## 7. Smallest convergence step and proof

**First family: existing `_runtime_busy_refusal`, not a new route or bulk
conversion.** Add render identity to the existing constructor override and one
consuming adapter, retaining current semantics. Use an isolated fixture
translation to prove rendering before real content translation. Notice fields
and producer helpers already exist; do not recreate them as prerequisite work.
Keep SDK programmer-facing validation policy unchanged unless separately agreed.

Required implementation proof: before/after actual synthetic-owner responses
have **503 / `runtime_busy` / retryable true / 2000 ms / `Retry-After: 2`** and
byte-identical fallback text, with only additive render identity. Exercise both
sync refusal and `OwnerAckTimeout`; count dispatched operations and client
scheduled retries to prove translated presentation changes neither behavior nor
retry count. Old readers must display identical fallback text; the new adapter
must render known identities and safely handle the §6 negatives.

Existing tests on this baseline provide a foundation, **not execution proof in
this document**:

- `tests/unit/server/test_desktop_read_without_owner.py:399–441`,
  `test_a_control_route_answers_a_silent_owner_busy_and_fast`, covers two silent
  owner states and asserts 503/category/retryable/2000 ms/header plus response
  bound and text prefix. Extend to byte-exact fallback and additive identity.
- `tests/unit/server/test_desktop_sessions.py:1828–1882`,
  `test_a_stale_mirror_press_whose_dial_times_out_is_not_idle`, asserts 503,
  category/retryable and one abort dispatch. Extend it to delay/header/text/
  identity parity and exercise the actual consumer retry disposition.

### Checks derived from this baseline

The census used `git show`/`git ls-tree` of the single baseline, JSON parsing and
Python AST inspection. In the isolated PR worktree, a diff against the baseline
showed only this document changed; imports were asserted to resolve inside that
worktree. An isolated-HOME/config runtime probe with `LOP_LANG=fr` observed:

- `NoticeEvent.model_fields` contains `code`, `params`, `text`.
- `resolve_language('auto')` returns `fr`, while fresh active display locale is
  `en`; resolution alone does not activate `tr`.
- `cli_exec_runtime.thinking().envelope(locale='fr')` renders `· réflexion…`.
- `runtime_unreachable().envelope(code='runtime_busy', locale='fr')` has only
  category/params/English text, without its catalogue identity.
- Unknown `wire.future.no_such_key` and missing params for
  `wire.exec.log_pointer` both degrade to their key in `text`.

These are targeted mechanism probes, not HTTP/retry, rendered UI or full exec
validation. No hosting/model environment was provisioned for exec end-to-end
work. No browser/rendered frame, installed desktop, native mobile, full-route
census or full test suite was exercised for this corrected baseline. Earlier
branch-local tests and CLI bootstrap failures are not carried forward as proof.
Future implementation needs independent review/QA and rendered design/UX
validation where applicable. This proposal grants no waiver of those gates.
