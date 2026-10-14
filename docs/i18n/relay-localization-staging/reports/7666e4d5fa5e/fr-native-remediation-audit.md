# Independent French remediation native audit — PR2329

**CURRENT VERDICT: changes_requested for relay.shell; relay.session/login remain pass. F-FR-01 singular agreement is resolved; F-FR-02 (MAJOR) exact-numeral million grammar is unresolved. Rendered visual HOLD remains.**

## Authoritative correction — supersedes the pre-authority pass below

A second auditor raised an authoritative distinction, which I independently fetched and verified: [OQLF, Écriture de million et de milliard](https://vitrinelinguistique.oqlf.gouv.qc.ca/21647/la-typographie/nombres/ecriture-de-million-et-de-milliard), **Nombres exacts**: « La préposition de est absente. » Spelled-out *un million de crédits* takes de; exact digit-written numeric measurements **1 000 000 crédits** do not. The actual outer ICU `{value, number}` produces digits, not spelled-out *million*. My prior proposal incorrectly transferred the spelled-out grammar to digit-written quantities. The repository French style-guide example `# de sessions` is not authority to override this numeric distinction.

**F-FR-02 — MAJOR**, submitted shell lines **37,39–40**: the three `many {de …}` branches render `reste 1 000 000 de crédits` / `utilisation de 1 000 000 de crédits`. Absence of adjacent `de de` proves nothing about this separate numeric grammar issue. Correct: `reste 1 000 000 crédits` / `utilisation de 1 000 000 crédits`. Outer `utilisation de` remains correct; only the noun branch's de is removed.

**Required exact target delta:** same three noun entries only, `many {crédits}`, `many {jetons}`, `many {requêtes}`; alternatively omit many and use their existing plain-plural other branches. No English source/caller/outer-string change needed. All other 62 target entries remain retained. No target edits performed.

Independently executed `fr-million-numeral-counterfactual.mjs` using actual submitted targets and pinned relay runtime: positive/negative one-million samples reproduce the defect; scratch-only removal of unit de produces correct digit-written forms for all three nouns. Actual outputs: `fr-million-numeral-counterfactual.log`, command exit0. This is feasibility proof, **not approval of a translator-authored correction**.

The initial below pass, many-native claims and expected-output assertions are retained as **superseded pre-authority reasoning**, not current approval. Mechanical parity/NFC/typed-count and 0/1/2/1.5 agreement proofs still hold; exact-million native assessment does not. Failed initial finding and this new lead are preserved. Current machine-readable JSON records changes_requested/F-FR-02, while keeping F-FR-01 resolved. Await a narrow three-branch translator correction and rerun million composition before native pass.

---
## Historical pre-authority assessment (superseded where stated)


Auditor task/session `7666e4d5fa5e`; actual model **openai/gpt-6.1-sol**, no fallback used. Independent of translator session `c4131ebe3f64`; no self-audit, human-native-speaker identity claim, or target/source edits. Date 2026-10-14.

## Source binding

Source checkout `~/local-operator/.worktrees/coder-2329-convergence-20261014`; HEAD verified `de9b9a490c839d6de2c2e41ea03ec819f66be489`. **English counted-noun edits are UNCOMMITTED working-tree bytes, not committed de9b9a490 source.** This verdict binds the measured working bytes, not an invented committed revision.

| Namespace | English source SHA256 | French target SHA256 | Native verdict |
|---|---|---|---|
| relay.shell (44) | `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` | `b2ddca1751c5a650fa00f29035944701c17dc1039480f80abe74ae1a322bc84c` | pass |
| relay.session (17) | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` | `ff308557eaaac62556f20625039d9e70df97cf55f5b150580f7511c16ecd2018` | pass retained |
| relay.login (4) | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | `5f6cc95bffee5f11984b4a083af057ba96a5e332a623c5b5c3659a9e61530d24` | pass retained |

Shell English context SHA256: `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`.
Target root: `historical-scratch/c4131ebe3f64/relay-final65-counted/fr/`.
All source/target/context hashes independently derived and rechecked after execution.

## Narrow scope and preservation proof

Comparing translator's final65 to counted candidate, **only unit_credits, unit_tokens, unit_requests changed**. Exact full serialized entry lines for **62 retained keys** remain byte-identical: shell41/session17/login4. Session and login entire files are byte-identical; current English hashes remain unchanged for both. Prior native passes are retained by this source/target compatibility, not relabeled without evidence. Current high-visibility session/login text samples were additionally re-executed through the actual runtime.

Historical failed final65 `fr-native-audit.md` / `.json` remain intact. No shipping ledger, source catalogue, generated file or submitted target was changed by this auditor. No full65 retranslation or full native re-audit was needed.

## F-FR-01 — resolved

The three French noun messages now use numeric `count, plural` with native singular branches (crédit/jeton/requête), plural branches and **many** branches (de crédits/de jetons/de requêtes). They return the noun only; no duplicated number or `#`. USD stays invariant. Outer templates are byte-identical: `reste {value, number} {unit}` / `utilisation de {value, number} {unit}`. No awkward unit-caption workaround was used.

Real pinned relay `createMessageRuntime` importing existing `intl-messageformat` rendered **56** amount/family-row combinations: 0/1/2/1.5/-1/-1.5/1,000,000 × credits/USD/tokens/requests × remaining/used. Independently parsed every actual output and checked noun endings, owned U+202F row punctuation, preserved model-family identifier, and no `de de`.

| Count | Actual remaining request text | Actual used request text |
|---:|---|---|
| 0 | reste 0 requête | utilisation de 0 requête |
| 1 | reste 1 requête | utilisation de 1 requête |
| 2 | reste 2 requêtes | utilisation de 2 requêtes |
| 1.5 | reste 1,5 requête | utilisation de 1,5 requête |
| -1 | reste -1 requête | utilisation de -1 requête |
| -1.5 | reste -1,5 requête | utilisation de -1,5 requête |
| 1,000,000 | reste 1 000 000 de requêtes | utilisation de 1 000 000 de requêtes |

Same native agreement holds for crédit/jeton; USD remains unchanged. French one correctly governs zero, one, 1.5 and their tested negative counterparts. `reste` is a normal terse impersonal fragment with understood *il*, not a noun-agreeing plural verb. `utilisation de 1 000 000 de crédits` is grammatical: the first de introduces the quantity phrase; the second de belongs to *un million de crédits*. The numeric phrase separates them, so this is neither duplicated adjacent de nor a defect.

Read the actual working caller diff: measured amount is computed first, guarded finite/non-null, and the **same measured number** selects the localized noun and formats outer value. No floor/round, total amount, fraction or percent conversion substituted as count. This is caller source inspection, not mounted-component QA.

## Actual mechanics and negative controls

Commands executed with `env -i`, isolated scratch HOME/config, existing repo Python and Node, TZ=UTC for Node. This strips inherited CMUX/LOP/XPC variables. No installs, browser or heavyweight test runs.

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home" \
 LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home/.local-operator" \
 PATH="$PATH" ~/local-operator/.venv/bin/python \
 "$LOCAL_OPERATOR_SCRATCHPAD/validate-fr-remediation.py"
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home" \
 LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home/.local-operator" \
 PATH="$PATH" TZ=UTC node "$LOCAL_OPERATOR_SCRATCHPAD/audit-fr-remediation.mjs"
```

Both final commands exit zero. Setup initially selected the old final65 fixture, producing the expected counted-argument mismatch; corrected scratch input to the submitted counted candidate, then reran. This was a rig input error, not a remediation defect.

| Check | Actual outcome |
|---|---|
| Exact keyset/types | PASS shell44/session17/login4; all ordered argument names/kinds match current source |
| Real repository check_catalogues | PASS zero problems in own isolated current-source/target fixture |
| ICU styles/NFC | PASS explicit parsed name/kind/skeleton parity and full65 NFC check |
| Dropped count | PASS negative: invariant requêtes rejected by validator, `arguments () != en's (('count', 'plural'),)` |
| Number→simple | PASS negative: outer value kind mismatch rejected by validator |
| Restored fixture | PASS zero problems |
| Missing count runtime | PASS negative: all3noun lookups return key token, not falsely formatted text |
| NFC counterfactual | PASS audit-only check detects decomposed accent; do not claim repository validator enforces NFC |
| ICU style counterfactual | PASS audit-only check detects lost percent skeleton; do not claim repository validator enforces source/target skeleton parity |
| Composed amount/rows | PASS all56 actual outputs, typed count/native CLDR/negative/million cases |
| Direct retained formatting neighbors | PASS actual percent, dates, unknown-unit, full-row, English plural fallback and high-visibility text controls |

## Artifacts and remaining holds

This directory contains `fr-native-remediation-audit.json`, `fr-remediation-intl.log`, `fr-remediation-validator.log`, `fr-remediation-output-assertions.log`, `audit-fr-remediation.mjs`, `validate-fr-remediation.py` alongside this report. Full command outputs are durable; hashes and resolved finding are machine-readable.

**Rendered customer visual gate: HOLD/BLOCKED, customer/resource hold.** Text was actually formatted; no browser/pixel/frame, mounted UI, geometry, responsive layout or reflow proof claimed. RTL not applicable for French LTR. Pseudo-layout not run; clipping/width remains unproven. This native pass does not waive independent visual/design/UX/QA requirements or authorize shipping. Integration still needs committed source provenance; do not label working bytes as committed de9b9a490.
