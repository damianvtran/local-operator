# zh-CN audit — `wire.errors.sdk_startup` — context-delta confirmation (reader 2)

- **audit id / mode:** `zh-r2-ctx-82f193` — context-delta confirmation + current-parameter integrity, SOURCE-ONLY
- **date:** 2026-11-03
- **locale / namespace:** `zh-CN` / `wire.errors.sdk_startup` (4 keys)
- **auditor:** `i18n-auditor`, session `0feda324e9e3` — independent of translator session `7fce46e29bee` and of the coder session
- **model:** requested route **Radient Auto** (`radient/auto`, the harness header label). Served underlying model **UNKNOWN** — no execution proof of a concrete upstream model was produced here; the header label is not served-model proof. Fallback: **UNKNOWN** (not exposed).
- **source binding:** unmerged core PR #3378 head `0066245ce4f90109933c59e74f1ad21b86d647bd`, read from worktree `~/local-operator/.worktrees/startup-owned-copy` (`.git` → `refs/heads/fix/startup-refusal-owned-copy` = that SHA). No shipped claim.

## Inputs (hashes recomputed from the frozen files in this session)

| role | path | sha256 |
|---|---|---|
| en | `…/sessions/3ef6295a5545/scratchpad/sdk-startup-followon/en.json` | `5390ecb2413e0978c34541a9a0db6c2a004ae86446d1c0fa0d2e697e4f9ca6bd` |
| context (new) | `…/sdk-startup-followon/context.json` | `82f193fd3c3e84526cbfaa0498fc47b776dd84df128cea3442df6d13369bb13e` |
| zh-CN | `…/sdk-startup-followon/zh-CN.json` | `fcff2fac99196a9e4dbd03957fb959c2af69013b1471532f819ddbc705293e02` |
| context (old, superseded) | `…/sdk-startup-inventory-20261103/candidates/zh-CN/context.json` | `ee0bbde5e99de22c04368e6bbfedf913d758ef6ba5075984ddf7d795b1ee7822` |

All five binding entries (`en`, `context`, `zh-CN`, `fr`, `es`) were re-hashed from **both** the binding's `original` and `frozen` paths: 10/10 byte-identical to the bound SHA-256 values. Target hash `fcff2fac…` is unchanged from the translator candidate and from the input of the two prior full zh-CN audits — **no re-authoring**.

## Context delta — what actually changed, and does it invalidate the approved Chinese?

The old (`ee0bbde5…`) and new (`82f193fd…`) sidecars are identical for three keys; the whole delta is the `attach_hosting_unconfigured.note`, which replaced two claims:

1. *“English-only child namespace falls back under RFC §6 per-CARRIED policy; future translated consumers still owe the RFC migration boundary.”* → **removed**. The new note states the frame renders in the resolved locale and only non-carrying locales fall back per-namespace.
2. *“error is opaque producer text … Never interpret either as ICU or a message code.”* → **refined**: the cause is opaque for *foreign* producers only; when it is the product's own not-configured hosting error (exact type + published sentence) the caller substitutes a **separately localized** cause sentence, so “this frame must never contain English cause wording of its own”.

Confirmed against the pinned code, not against the note (all read-only, no execution):

- `sdk.py:587–596` `startup_refusal_locale()` reads `language` from the **already scoped** `ConfigManager` (isolation preserved) and returns `resolve_language(...)` — no English pin.
- `sdk.py:598–601` renders the frame with exactly that locale.
- `sdk.py:603–632` `startup_cause_text()` substitutes a localized cause only when `type(error) is HostingNotConfiguredError` **and** `str(error)` equals the catalogued EN source of `wire.errors.hosting.not_configured`; every other producer's text is returned as `str(error)` verbatim.
- `sdk.py:660–665` passes `resume=repr(spec.resume)` — the “already-quoted identifier” the new note names; `sdk.py:692/705/725` likewise pass `repr(...)` / already-stringified tuples.

So the new context **matches the code**, and the old note's English-only claim was the false one (both prior zh-CN audits recorded that conflict: Z2-1 in `zh-CN-sdk-startup-second-6433b7a5b3e9`, `context_note_conflict` in the same record). Consequence for Chinese: `wire.errors.hosting.not_configured` = `托管平台未配置。`, so the frame now composes entirely in Chinese — the prior residual *English clause inside a zh-CN refusal* is closed by the source change, **with no change to any of the four Chinese strings**. Nothing in the four approved renderings depended on the removed claim, so the prior native judgments stand.

The changed note names three literals (`config.yml`, `spawn_session()`/`deliver()`) — all three are present in the current target (see per-key table). No new obligation is created by the note that the existing target fails.

## Findings

| id | severity | key | issue | suggested |
|---|---|---|---|---|
| NIT-1 | NIT (does not gate) | `attach_hosting_unconfigured` | Glue after `{error}` is bare: with a foreign cause that carries no terminal punctuation the remedy clause abuts it — `…无法启动一个：boom: socket closed为此根目录（其config.yml）配置托管…`. The owned-cause path (the only cause this product authors) renders correctly: `…无法启动一个：托管平台未配置。为此根目录（其config.yml）配置托管，然后用spawn_session()/deliver()启动该ID，待运行时存在后再附接。` | Keep as approved, or — if QA's rendered pass finds the opaque case jarring — carry the cause in a parenthetical instead: `…且此根目录无法启动一个（原因：{error}）为此根目录…`, which reads cleanly for both a `。`-terminated Chinese sentence and an unterminated ASCII message. No separator can be added *after* `{error}` that is correct in both cases, so this is a frame-shape choice, not a translation defect. |

Rendered check of the **owned** path (frame + `wire.errors.hosting.not_configured` = `托管平台未配置。`), string-substituted from the frozen bytes:

> 没有正在运行的运行时为'abc123'提供服务，且此根目录无法启动一个：托管平台未配置。为此根目录（其config.yml）配置托管，然后用spawn_session()/deliver()启动该ID，待运行时存在后再附接。

The frame is complete after the colon and the whole sentence is Chinese apart from the deliberately literal tokens — the prior residual gap (Z2-1: “a zh refusal still contains one English clause”) is closed **at the source** by this context/code pair, with no target edit.

### NIT-1 (non-gating) — `attach_hosting_unconfigured`, opaque-cause glue

Current: `…无法启动一个：{error}为此根目录（其config.yml）配置托管，…`
Problem: with the owned cause the splice is correct (`托运…未配置。为此…`). With a **foreign** cause that does not end in terminal punctuation — a case the new note newly emphasises as verbatim passthrough — the clause runs together, e.g. `：boom: socket closed为此根目录…`. Purely typographic; no meaning or safety change, and no wording in the frame is English.
Suggestion: leave as-is (recommended — the product's own cause path is the one users hit), or separate the remedy only if QA's rendered pass finds it objectionable, e.g. `…无法启动一个。原因：{error}` with the remedy kept as its own sentence. Do not add a separator after `{error}` that assumes a missing full stop.

Counts: BLOCKER 0 · MAJOR 0 · MINOR 0 · NIT 1. No line in the target contributes English wording of its own.

## Per-key confirmation (retained prior judgment + re-verified parameter integrity)

Mechanics (stdlib only: `json`, `hashlib`, `re`, `unicodedata`; see gaps): valid JSON, 4/4 keys present both sides, no key added/dropped; placeholder sets identical per key and all simple identifiers (`resume`,`error`,`profile`,`spec_tools`,`policy_tools`) with the generated signatures `attach_hosting_unconfigured(*, resume, error)`, `attach_runtime_missing(*, resume)`, `profile_attach_failed(*, profile)`, `tool_inventory_mismatch(*, spec_tools, policy_tools)` — names *and* types agree; brace balance holds; **no `plural`/`select` in source or target**, so no CLDR categories are owed for zh-CN here (plain argument substitution only); `''` doubling count equals source where an apostrophe occurs (`mode=''own''`, 2) and target has no lone `'`.

| key | verdict |
|---|---|
| `attach_hosting_unconfigured` | Retained PASS. Both facts (no live runtime serving the id; this root cannot start one) and the remedy order preserved; `config.yml`, `spawn_session()`, `deliver()` literal; no English cause wording of its own — the new context's core requirement is satisfied by the existing text. NIT-1 only. |
| `attach_runtime_missing` | Retained PASS. Disposal→refusal meaning, `mode='own'` remedy, engage-then-attach order preserved; `''` doubling byte-identical to source, which `runtime.py:19` documents as ICU `''→'` escaping (so it displays as `mode='own'`, my first naive substitution showed the doubled form and was corrected by that source read). |
| `profile_attach_failed` | Retained PASS. `无法附接配置档{profile}` keeps Could-not + quoted profile; `配置档` matches shipped zh-CN usage (`tui.authoring`, `wire.errors.desktop`, `wire.settings`), so no glossary divergence. |
| `tool_inventory_mismatch` | Retained PASS. Two named inventories + state-once remedy preserved; `spec.tools` and `ApprovalPolicy.declared(...)` literal; the space before `{spec_tools}` is Latin-to-Latin, not a Chinese–Latin gap (style guide forbids the latter only). |

Style/register retained: 你-voice n/a (no address), no `请`, no `！`, full sentences keep `。`, labels keep none (23-cell label `无法附接配置档{profile}` has no final `。`); `，。：；、` full-width, half-width punctuation only inside code fragments; no space inserted between Chinese and Latin runs; `ID` kept Latin as an identifier reference.

**Width (informational; no budget exists):** the sidecar carries no `maxCells`/`maxChars` — confirmed, none invented. Measured cells (CJK/full-width = 2): 170 / 143 / 23 / 110 vs EN 210 / 173 / 34 / 123. No fit verdict is claimed from this.

## Explicit gaps — NOT RUN

- **Repo validator (`scripts/i18n/check.py` / Node) — NOT RUN**, by the task's SOURCE-ONLY override (no process/bash/console/Node/scanner). Replaced by the stdlib verification above *and* both prior full audits' executed validator runs (mechanical PASS 4/4 with negative controls in `sdk-startup-zh-CN-r1`). A source pass is not runtime clearance.
- **Rendered fit / design, pseudo-localization, live SDK refusal execution under a real zh-CN config, ICU engine execution — NOT RUN.** ICU escaping and locale resolution were read from source (`runtime.py`, `sdk.py`), not executed.
- No claim of activation, release eligibility or shipped state: the catalogue and code live on an **unmerged** PR head.
- No ledger write, no repo edit, no PR comment; target not edited. Did not read the parallel new reader-1 report (independence preserved).
