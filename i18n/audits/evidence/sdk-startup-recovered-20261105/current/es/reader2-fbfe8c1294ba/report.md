# Audit report — es · wire.errors.sdk_startup (reader 2, first full pass)

- **Locale:** `es` (neutral Spanish, style guide `i18n/style-guides/es.md` + `i18n/glossary.md`)
- **Namespace / file group:** `wire.errors.sdk_startup` — 4 keys
- **Mode:** FULL first independent audit of all 4 keys against the current context sidecar.
  No prior audit of this namespace exists for `es` (`i18n/audits/evidence/es/wire.errors.sdk_startup*/` absent).
  No prior decision reused; no `wire.errors.hosting` verdict reused.
- **Date:** 2026-11-03
- **Dependency:** source/context depend on UNMERGED core PR #3378 head
  `0066245ce4f90109933c59e74f1ad21b86d647bd`. No shipped claim.

## Provenance

| Field | Value |
|---|---|
| agent | `i18n-auditor` (reader 2), independent translation auditor |
| session | `fbfe8c1294ba` |
| requested route | `radient/auto` |
| served model | **UNKNOWN** — not exposed as execution proof; the session header line is not served-model proof |
| actual_model_source | harness header line only (`Model: radient/auto`); no per-call proof captured |
| fallback | **UNKNOWN** (none observed, none provable) |
| independent_from_translator_session | true — translator session `804764f54d82` per binding |
| independence from reader 1 | true — reader 1's report was not read, per brief |

## Input hashes (computed from the FROZEN bytes here, not copied from any prose)

| Input | Path | SHA-256 |
|---|---|---|
| en | `…/sdk-startup-followon/en.json` (original `…/.worktrees/startup-owned-copy/local_operator/i18n/catalogues/en/wire.errors.sdk_startup.json`) | `5390ecb2413e0978c34541a9a0db6c2a004ae86446d1c0fa0d2e697e4f9ca6bd` |
| context | `…/sdk-startup-followon/context.json` (original `…/catalogues/en/wire.errors.sdk_startup.context.json`) | `82f193fd3c3e84526cbfaa0498fc47b776dd84df128cea3442df6d13369bb13e` |
| es | `…/sdk-startup-followon/es.json` (original `…/sessions/804764f54d82/scratchpad/es-sdk-startup-4-0e7cde3/wire.errors.sdk_startup.json`) | `2f60ae79a4350194cf1a22af6bcbbcc01452b0316835e8d7607f67a46790cfcd` |

en and es hashes agree with `binding.json`. The context hash is `82f193fd…`, **not** the
`82b…` value carried in the inventory chat.

## 1. Mechanical

Judged on frozen bytes with stdlib `json`/`re`/`hashlib` only (source-only mandate).

- JSON valid UTF-8; 4 keys in source, 4 in target, identical key set **and order**.
- Placeholders: exact name-set and order match per key —
  `{resume}`,`{error}` · `{resume}` · `{profile}` · `{spec_tools}`,`{policy_tools}`.
  No renamed, added or dropped arguments.
- ICU: no `plural`/`select`/`selectordinal` in any of the 4 messages in either locale — all four
  carry plain arguments only, so no CLDR category set applies (confirmed, not assumed).
- Apostrophe escaping: `mode=''own''` preserved byte-exactly (4 apostrophes = two literal pairs,
  renders `mode='own'`); no bare apostrophe in the target. Source `run''s` possessives correctly
  have no Spanish counterpart (Spanish has no possessive apostrophe) — 0 apostrophes there is right.
- Literals preserved byte-identically: `config.yml`, `spawn_session()`, `deliver()`,
  `mode='own'`, `spec.tools`, `ApprovalPolicy.declared`; `runtime` and `id` kept per style guide.
  No markup/Rich/backtick spans exist in this group.
- Width: the context sidecar carries only `role`/`tone`/`note` — **no `maxCells`/`maxChars`**.
  No declared budget to exceed; none invented.
- Repo validator (`scripts/i18n/check.py`, Node, generate `--check`): **NOT RUN** — the source-only
  exception in this brief forbids processes/Node/repo scripts. Mechanical equivalence was
  established directly on the bytes above; the ledger/generate command is likewise **NOT RUN**
  (no repo edits permitted here).

## 2. Accuracy and safety of meaning

All four preserve meaning; no omission, addition, polarity or scope error; no unsafe wording
(none of these strings is destructive or permission-granting, and the two refusal frames keep
the refusal direction: `ningún runtime activo atiende`, `no puede iniciar uno`).
`attach_hosting_unconfigured` contains **no English cause wording of its own**; its remedy names
hosting with the same term the separately localized cause uses
(`wire.errors.hosting.not_configured` = “La plataforma de alojamiento no está configurada.”).
`profile_attach_failed` keeps the diagnostic and its capitalization.

## 3. Findings

### es-sdk-r2-01 — MINOR — `wire.errors.sdk_startup.attach_runtime_missing`
Current: “ningún runtime activo atiende {resume}; **ábrelo** con mode=''own'' para **ejecutarlo** en este proceso, o **actívalo** antes…”
Problem: the object being opened/run is the session (`la sesión`, feminine) named by `{resume}`;
the masculine pronouns make the nearest masculine noun (`runtime`) a plausible antecedent —
a hasty read says “open the runtime in this process”, which is not the remedy. Internally
consistent, but the referent is not the one the English names.
Suggested: “ningún runtime activo atiende {resume}; **ábrela** con mode=''own'' para **ejecutarla** en este proceso, o **actívala** antes con spawn_session()/deliver() y conéctate cuando exista un runtime.”
(Equally acceptable: keep the masculine and name the referent explicitly, “abre el id …”.)

### es-sdk-r2-02 — MINOR — `wire.errors.sdk_startup.tool_inventory_mismatch`
Current: “… **nombran** inventarios distintos; declara el alcance de la ejecución una sola vez”
Problem: calque of EN “name different inventories”. Spanish `nombrar` is *to name/appoint/mention*;
applied to two values that each *list* tools it reads translated rather than authored.
Suggested: “… **enumeran** inventarios distintos; declara el alcance de la ejecución una sola vez”
(`especifican` also works; avoid `declaran` — the remedy already uses `declara`).

### NIT-1 — `attach_hosting_unconfigured`
“Configura el alojamiento para esta raíz” shortens the settled collocation **“plataforma de
alojamiento”** (`wire.errors.hosting.not_configured`, `cli.global_help`, `cli.exec_worker_help`,
`wire.settings.hosting.*`). Suggest “Configura la plataforma de alojamiento para esta raíz (su config.yml)”.
Not a term drift — the same word, elided; no cell budget forces the shortening.

### NIT-2 — `attach_runtime_missing`
“o actívalo **antes**” leaves `antes` as a bare adverb; `primero` states the ordering
(“engage it first”) unambiguously.

### NIT-3 — `attach_hosting_unconfigured`
“esta raíz” alone can read as a filesystem root; the corpus names the concept
`raíz de configuración` (`cli.sessions_help.reclaim_description`, `cli.browser.config_root_line`).
“para esta raíz de configuración” removes the ambiguity where width is free.

Accepted conventions (recorded, not findings): both `attach_*` frames start lowercase, exactly as
their English source does and as shipped `es` error copy does (`wire.errors.desktop.*`);
`tú` imperatives throughout (`Configura`, `activa`, `conéctate`, `ábrelo`, `declara`), no `usted`
mix, no Title Case, `:`/`;` without preceding space, no `!`.

## 4. Verdict

**`passed`** — 0 BLOCKER, 0 MAJOR, 2 MINOR (advisory, non-gating), 3 NIT.
The two MINORs are native-copy improvements, not meaning or mechanical defects; if the wave folds
them, es-sdk-r2-01 is the one worth taking.

## 5. Explicit gaps (do not read as clearance)

- Rendered fit / terminal cell measurement: **NOT RUN** — no `maxCells` exists and this auditor
  ran no renderer.
- RTL: not applicable to `es`; pseudo-locale: **NOT RUN**.
- Whole-frame/runtime assembly (`sdk.py` render path, locale resolution at runtime): **NOT
  RUN** — `sdk.py` 545–640 was read read-only for contract shape only.
- No English-in-frame scan of live output, no TUI/browser/CLI surface exercised.
- Ledger row and tracked evidence bundle: **NOT WRITTEN** by this auditor (source-only mandate).
  This record is the scratch of record; the coordinating session must append the shard row and
  commit `i18n/audits/evidence/es/wire.errors.sdk_startup/<audit-id>/`.
- This pass establishes source-level translation quality only — **not** the operator's
  “no English anywhere” acceptance for the es interface.
