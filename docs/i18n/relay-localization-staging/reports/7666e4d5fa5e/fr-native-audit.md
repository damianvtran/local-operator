# Independent French native-text audit — PR2329 final65

**Verdict: changes_requested for relay.shell; pass for relay.session and relay.login.** One MAJOR native-language finding, no other findings. This does not approve shipping or satisfy the separate rendered-visual gate.

Auditor: task/session `7666e4d5fa5e`, actual model **openai/gpt-6.1-sol**, no fallback used. Independent of translator `c4131ebe3f64`; no translation authored or submitted targets edited by this auditor. This is an LLM French-language assessment, not a claim of human native-speaker identity.
Date: 2026-10-14. Source: `~/local-operator/.worktrees/coder-2329-convergence-20261014`, independently verified HEAD **de9b9a490c839d6de2c2e41ea03ec819f66be489**.
Targets: `historical-scratch/c4131ebe3f64/relay-final65/fr/`.

## Exact provenance and delta

| Namespace | Keys | Source SHA256 | Target SHA256 | Native assessment |
|---|---:|---|---|---|
| relay.shell | 44 | `8b17439e9f31f159bc60a9e2940eb2d88400dacdcda9854ae1b3c8eb0ca00586` | `9c8b4ee29b5da192235ae6722d44e51e49e59e4d4829ebb03035e88ed88580b9` | changes_requested: F-FR-01 |
| relay.session | 17 | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` | `ff308557eaaac62556f20625039d9e70df97cf55f5b150580f7511c16ecd2018` | pass |
| relay.login | 4 | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | `5f6cc95bffee5f11984b4a083af057ba96a5e332a623c5b5c3659a9e61530d24` | pass |

Total **65**. Derived source delta by comparing pinned HEAD^ English shell catalogue to HEAD: **8 added / 5 modified**. Added: unit_credits, unit_usd, unit_tokens, unit_requests, remaining_percent, used_percent, remaining_unknown_unit, used_unknown_unit. Modified: remaining, used, fraction, reset, as_of. All 13 reviewed against current English/context and direct agreement/whole-row interactions.

Prior targets from `historical-scratch/5da630b34ee9/relay-final/fr/` are the supplied historical final57 baseline. Historical audit pass is parent-provided provenance, not a new current65 audit. Independently compared entire serialized retained JSON entry lines: all **52 byte-identical** (shell31/session17/login4). Session/login entire file bytes are identical. Shell prior SHA256 `92547e717820b43b02255197a9906ce7e4ede1749dc9e804ed6a0f9d79fcc0a2`; prior session/login hashes equal current target hashes. Exact retained/added/modified key lists and derived hashes are in `fr-native-provenance.json`. All source/target hashes rechecked after execution, unchanged.

## F-FR-01 — MAJOR — counted-noun agreement is still incorrect

Location: submitted **relay.shell.json lines 29–30**, interacting with invariant unit labels at **37, 39–40**. Keys `relay.shell.usage.remaining`, `relay.shell.usage.used`.

The translator's `utilisation de {value, number} {unit}` successfully avoids adjective-gender agreement with feminine *requêtes*, but **does not avoid numeric noun agreement**. Executing the actual pinned relay `createMessageRuntime` produces:

- 1 credit: `reste 1 crédits`; `utilisation de 1 crédits`
- 1 request: `reste 1 requêtes`; `utilisation de 1 requêtes`
- 1.5 tokens: `reste 1,5 jetons`; `utilisation de 1,5 jetons`
- complete visible row: `Quota par famille (opus) · 7 jours : utilisation de 1 requêtes`

These are prose counted-noun phrases, not a detached measurement-column heading, so plural labels are not acceptable invariant captions in this construction. `1 requêtes` is unambiguously malformed French. Actual `Intl.PluralRules('fr')` selects `one` for 0, 1 and 1.5, `other` for 2, `many` for 1,000,000. The finding covers changed/new interactions, not a reopened historical wording preference. It prevents an honest native-grammar pass for the shell namespace; structural validation remains green.

**Actionable translation-only remedy:** place the invariant label outside the counted-noun phrase, e.g. `Solde ({unit}) : {value, number}` / `Utilisation ({unit}) : {value, number}`. Same placeholder names/types, no wider source-contract change needed. Scratch-only counterfactual registered an alternate map with those two phrases in the same actual relay runtime; all 0/1/2/1.5 × crédits/USD/jetons/requêtes rendered as grammatically independent captions (`Utilisation (requêtes) : 1`). This is demonstrated feasibility, not an edit or sign-off of a future translator revision. Alternatively actual numeric unit inflection would require an agreed expanded source contract.

## Per-namespace native assessment

### relay.shell — changes_requested

Read all 44 current strings and source sidecars; all 13 changed-source entries received new assessment. All other messages retain correct compact French tone and meaning. *En attente des données…* correctly avoids false graphical/predictive meaning of daemon projection; *connexion à la session…* remains a distinct preconnection state. Cached/unavailable/loading/empty/last-known descriptions do not promise a provider refresh; included/paid/family allowances and continuation text stay neutral.

*Crédits*, *requêtes*, USD, and *jetons* are understandable measurement labels. General French developer glossary keeps “token”; this task's documented measurement-label exception explicitly requests localized API-unit labels, and *jetons* is consistent within this narrow measurement namespace. No global glossary rewrite endorsed. USD is kept as currency identifier without invented conversion. Percent *utilisé* is acceptable elliptical quota/utilization wording; one NBSP U+00A0 before percent comes from Intl, not manually appended punctuation. Whole rows own exactly one U+202F before colon; family `(opus)` remains machine metadata. No added punctuation outside templates or duplicated units. Fractions and dates use typed ICU, not preformatted English strings. Amount noun agreement is the sole native defect.

### relay.session — pass on exact current source/target hashes

Read all 17 unchanged current strings against current sidecars. Professional vous and action/status distinction preserved; conversations/session labels agree in gender; directory *répertoire de travail*, model *modèle*, search/loading/no-matches messages match their roles. Independently executed high-visibility current-hash samples: `Rechercher des conversations…`, `nouvelle session`, `démarrage…`, `envoyez un message ci-dessous pour commencer`, `modèle`, `ou saisissez un autre chemin…`. Historical semantics retained only after whole-file byte proof; this native assessment is current, not a repinned prior report.

### relay.login — pass on exact current source/target hashes

All four high-visibility entries independently sampled: `local operator — connexion`, `mot de passe`, `se connecter`, `Mot de passe incorrect.` Correct French action, field and error copy; current English context explicitly preserves lowercase brand. Whole-file bytes equal historical baseline. Runtime text samples are not evidence that the server-rendered login page was mounted or viewed.

## Execution evidence

Only existing dependencies and lightweight isolated probes used. No new dependency installs, browser, services, live sessions, config changes or repo edits. `env -i` strips inherited CMUX/LOP/XPC variables; HOME/config point to this auditor's scratch isolated-home; timezone UTC for reproducible dates.

Commands executed (paths resolve inside this audit scratch directory):

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home" \
 LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home/.local-operator" \
 PATH="$PATH" ~/local-operator/.venv/bin/python \
 "$LOCAL_OPERATOR_SCRATCHPAD/validate-fr-independent.py"
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home" \
 LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/isolated-home/.local-operator" \
 PATH="$PATH" TZ=UTC node "$LOCAL_OPERATOR_SCRATCHPAD/audit-fr.mjs"
```

Both exit 0. Full actual outputs are `fr-validator.log` and `fr-intl-probes.log`.

| Surface | Actual result | Assessment |
|---|---|---|
| Current hashes, exact keys, typed ICU all65 | shell44/session17/login4, argument-name/type parity; `check_catalogues` zero problems | PASS |
| Counterfactual validator | number→simple produced `arguments ... != en's ...`; restored fixture zero problems | PASS negative and restored control |
| Actual relay TS runtime / Intl | source `runtime.ts` imported directly by existing Node; no mock formatter | PASS real text execution |
| Amounts | 0/1/2/1.5/1234.5 × four units × remaining/used, full row_family composition | FAIL native grammar F-FR-01; formatting PASS |
| Numbers | `1 234,5`, comma decimal and U+202F grouping | PASS |
| Percent | 0/0.01/0.29/1/1.5 × fraction/remaining_percent/used_percent; 0.29→`29 % utilisé` / `reste 29 %`, exactly one % U+00A0 | PASS |
| Dates | reset `Réinitialisation le 14 oct. 2026 à 15:04`; as_of `Données du 14 oct. 2026 à 15:04.` | PASS |
| Unknown unit | `reste 1,5` / `utilisation de 1,5`; no invented unit | PASS |
| Row punctuation | `Quota inclus · Mensuel : reste 1,5 requêtes`, one owned U+202F colon, expected field order | punctuation PASS; noun grammar FAIL |
| Missing ICU param | missing unit rejected to key, no falsely formatted phrase | PASS negative control |
| English fallback | synthetic English-only plural at fr locale returns `0 requests`, not French `one` rules | PASS |
| Same-contract remediation feasibility | `Solde (requêtes) : 1` / `Utilisation (requêtes) : 1` and full matrix in scratch-only alternate map | PASS feasibility, not submitted-target approval |

## Limits / holds

**Rendered customer visual evidence: BLOCKED — customer/resource hold.** No screenshots, mounted customer UI, geometry, responsive width, clipping/reflow or layout observations claimed. Text was really formatted, not pixels rendered. This native audit cannot waive design/UX or visual evidence.

**RTL: not applicable** for French LTR; no mixed-bidi visual test claimed. **Pseudo: not run**; expansion/truncation and plural pseudo catalogue proof remain unverified. Actual French CLDR and typed numeric/date/percent checks did run; those are not substitutes for pseudo-layout proof. Source/native report is not a ship-eligibility ledger mutation. No ledger/generated/shipped files altered.

Terminal disposition for this submitted hash set: **changes_requested**, pending F-FR-01 remediation and narrow re-audit of corrected amount phrases plus direct row interactions. Session/login passes remain hash-bound.
