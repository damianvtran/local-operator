# FR reader2 — context-delta confirmation, `wire.errors.sdk_startup` (4 keys)

Mode: **context-delta confirmation only** (not a re-literary audit of the four
already native-approved French strings, not a first audit).
Locale `fr` · namespace `wire.errors.sdk_startup` · date 2026-11-03.

## 1. Provenance (actual, this session)

| Field | Value |
|---|---|
| Agent | `i18n-auditor` (reader `fr-reader2-context-delta`) |
| Session | `3fb9e3fb5be6` |
| Requested route | Radient Auto (`radient/auto`) |
| Served underlying model | **UNKNOWN** — the session model header shows the requested route label only; no execution payload or provider response in this run exposes the served model. Header is the *primary* label, not served proof. |
| Fallback | **UNKNOWN** — not exposed by this run; nothing inferred, no `NONE` invented. |
| Independent from translator | session `7b2ba4ed501d` (binding `fr.original` producer) — not read as author, no authoring here |
| Independent from coder | coder session **not named in this run** (core PR #3378, unmerged head `0066245ce4f90109933c59e74f1ad21b86d647bd`); its contract was read read-only, nothing edited |
| Read constraint | STRICT SOURCE-ONLY: read/grep/glob/write + stdlib `json`/`hashlib`/`difflib`/`unicodedata` only. No process, bash, subprocess, project import, Node validator, scanner, generator, render, app, browser, network, repo edit. |

No translation was authored, edited, re-authored or patched here; no ledger row
and no PR comment were written. The four French strings are carried unchanged.

## 2. Fresh input hashes (computed here from the FROZEN files, `hashlib.sha256`)

| Input | Frozen path | Bytes | SHA-256 | Matches binding |
|---|---|---|---|---|
| en | `…/3ef6295a5545/scratchpad/sdk-startup-followon/en.json` | 772 | `5390ecb2413e0978c34541a9a0db6c2a004ae86446d1c0fa0d2e697e4f9ca6bd` | yes |
| context | `…/3ef6295a5545/scratchpad/sdk-startup-followon/context.json` | 2298 | `82f193fd3c3e84526cbfaa0498fc47b776dd84df128cea3442df6d13369bb13e` | yes |
| fr (target) | `…/3ef6295a5545/scratchpad/sdk-startup-followon/fr.json` | 933 | `82b61a755a8580e7cb4e4f86ae64a428fc83232fac92b22fa72f09d77b5f781d` | yes |

**Stale-value control (negative, executed):** the retained inventory copy
`…/sdk-startup-inventory-20261103/candidates/es/context.json` hashes to
`ee0bbde5e99de22c04368e6bbfedf913d758ef6ba5075984ddf7d795b1ee7822`. The current
context is **not** that blob — the `…ee0bbde5` prefix is stale and was not used
as a source. The inventory's `inputs/current-context.json` does hash to
`82f193fd…`, i.e. the inventory carries both values; only the frozen one is
authoritative here.

## 3. Delta: what actually changed

- EN source: byte-identical to both prior FR audits' `source_sha256`
  (`5390ecb2…`) → **source unchanged**.
- FR target: byte-identical to the prior approved `target_sha256`
  (`82b61a75…`) → **carriage unchanged, no re-author** — the strings I am
  confirming are exactly the ones that carried 0 unresolved BLOCKER/MAJOR.
- Context: `ee0bbde5…` → `82f193fd…`, and after parsing both sidecars the diff
  is confined to **one of four notes**:
  `wire.errors.sdk_startup.attach_hosting_unconfigured`. Notes for
  `attach_runtime_missing`, `profile_attach_failed`, `tool_inventory_mismatch`
  are **byte-identical** (dict equality, not just textual similarity).
- Substantive change in that one note: the old text asserted the `{error}` slot
  is *always* opaque producer text and framed the namespace as an
  English-only child falling back per RFC §6. The new text states the cause is
  **not always opaque**: for the product's own not-configured hosting error
  (exact type + published sentence) the caller substitutes the **separately
  localized cause sentence**, so the frame must never carry English cause
  wording of its own; every other producer's reason stays opaque and verbatim
  (whitespace, quotes, percent signs, brackets, non-ASCII preserved).

Contract corroboration, read-only source inspection of the unmerged PR #3378
worktree (`local_operator/sdk.py`): `startup_cause_text()` (≈603–632) renders
the catalogue key in the resolved locale only when `type(error) is
HostingNotConfiguredError` **and** `str(error)` equals the catalogued English
sentence, otherwise returns `str(error)` untouched; the frame is rendered at
≈601 with `locale=startup_refusal_locale()`. This is source inspection, **not
execution**. No shipped claim: the dependency is an unmerged PR head.

Localized cause that composites into the FR frame, FR catalogue
`wire.errors.hosting.not_configured` = « La plateforme d’hébergement n’est pas
configurée. » (ends with its own full stop, so the frame's following remedy
sentence starts cleanly after one space; verified by string composition on the
frozen/catalogue text, not by rendering).

## 4. Per-key confirmation (native French judgment, delta-scoped)

| Key | FR (carried) | Delta impact | Judgment |
|---|---|---|---|
| `attach_hosting_unconfigured` | « aucun runtime actif ne sert {resume}, et cette racine ne peut en démarrer un : {error} Configurez la plateforme d’hébergement de cette racine (son config.yml), puis démarrez l’identifiant avec spawn_session()/deliver() et rattachez-vous dès qu’un runtime existe. » | the only key whose note moved; frame contains **no English cause wording of its own** (French throughout), `{error}` is the sole cause slot, `config.yml` + both function names literal | **confirmed** under the new contract. Reads as native runbook French; `plateforme d’hébergement` also matches the owned cause term, so cause+remedy stay one vocabulary. |
| `attach_runtime_missing` | « aucun runtime actif ne sert {resume} ; ouvrez-le avec mode=''own'' pour l’exécuter dans ce processus, ou démarrez-le d’abord avec spawn_session()/deliver(), puis rattachez-vous dès qu’un runtime existe. » | note byte-identical | **confirmed**, prior judgment retained |
| `profile_attach_failed` | « Impossible de rattacher le profil {profile} » | note byte-identical | **confirmed**, prior judgment retained (capitalisation kept, no added period) |
| `tool_inventory_mismatch` | « spec.tools {spec_tools} et ApprovalPolicy.declared({policy_tools}) désignent des inventaires différents ; ne déclarez la portée de l’exécution qu’une seule fois » | note byte-identical | **confirmed**, prior judgment retained (identifiers literal, no added period) |

Terminology against `i18n/glossary.md` §1/§2 and `i18n/style-guides/fr.md`:
`runtime` stays English (recorded fr wave-1 addition, mirrors es) — consistent
in all four strings; identifiers/API tokens (`config.yml`, `mode='own'`,
`spawn_session()/deliver()`, `spec.tools`, `ApprovalPolicy.declared(…)`) kept
literal; `vous` address, lowercase mid-sentence, no « Veuillez »; typography
compliant (U+202F before `:`/`;` in all three prose cases, U+2019 apostrophes,
zero straight quotes outside the ICU-escaped `mode=''own''` pair).

## 5. Mechanics verified here (source-only, no validator)

- Valid JSON, 4 keys in en/context/fr, **no key added or dropped**.
- Placeholders identical in name, type and order per key: `{resume},{error}` ·
  `{resume}` · `{profile}` · `{spec_tools},{policy_tools}` — all plain string
  substitutions; braces balanced; **no ICU plural/select in either side**, so
  no CLDR category set is owed for these messages (CLDR: not applicable, 0
  plural/select nodes — confirmed from the actual strings).
- Literal-token presence is equal on both sides for every token listed above.
- Semantic deltas EN→FR carried over without addition/omission/negation flip;
  the refusal framing ("cannot …", "No/never") is preserved; no softening of
  destructive or permission text.
- Sidecar field set is exactly `role`, `tone`, `note` — **no `maxCells`/
  `maxChars` exists** for this namespace, so no cell budget is asserted.

## 6. Findings

Retained from prior FR audits (unchanged text, not re-scored, not escalated):

- **F1 — MINOR (retained, non-blocking).** `attach_hosting_unconfigured`,
  `attach_runtime_missing`: bare reflexive « (puis) rattachez-vous dès qu’un
  runtime existe » — `se rattacher` normally takes its complement in this
  corpus (« rattachés à un enregistrement »). Suggested, if ever re-authored:
  « … puis rattachez-vous à la session dès qu’un runtime existe. » Not a defect
  introduced or worsened by the context delta; meaning survives.

New, delta-specific, non-blocking:

- **N1 — NIT (optional).** `attach_hosting_unconfigured`: under the new
  contract the cause is now a French sentence, so the rendered frame reads
  « … ne peut en démarrer un : La plateforme d’hébergement n’est pas
  configurée. Configurez la plateforme d’hébergement de cette racine … » —
  « plateforme d’hébergement » twice in adjacent sentences. The repetition is a
  composition of two independently owned strings, not a mistranslation; a
  shorter remedy (« Configurez-la pour cette racine (son config.yml) ») would
  remove it but requires translator re-authoring this task forbids. No action
  requested.

Counts on this confirmation: **BLOCKER 0 · MAJOR 0 · MINOR 1 (retained F1) ·
NIT 1 (N1)**. Status **passed** — no unresolved BLOCKER/MAJOR; every finding
cites an actual key and its actual current text.

Prior-audit reconciliation: the prior FR record's out-of-scope observation O1
(“`{error}` carries the producer's English sentence, so English remains”) is
**superseded for the owned error type** by this context: the caller now
substitutes a localized cause. Foreign producer errors still pass through
verbatim by contract, so O1 becomes a narrowing (owned cause localized,
foreign causes intentionally untouched) rather than a defect.

## 7. Explicit gaps — nothing rendered, nothing executed beyond hashing

- Repo validator (`scripts/i18n/check.py`, `generate.py --check`): **NOT RUN** —
  forbidden by this task's strict source-only constraint. Mechanical claims
  above come from stdlib parsing of the frozen files, not the repo checker.
- Rendered fit / width measurement: **NOT RUN**, and **no cell budget
  invented** (sidecar carries no `maxCells`/`maxChars`). Lengths for the record
  only (chars incl. placeholders, en→fr): 210→262, 173→202, 34→43, 123→160.
- Pseudo-locale evidence: **NOT RUN**.
- Runtime/app/UI rendering, SDK guards, live requests: **NOT RUN** — this is
  not runtime clearance and not a whole-frame clearance; it is a source-level
  context-delta confirmation.
- RTL: not applicable to `fr`.
