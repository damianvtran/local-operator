# French account-policy namespace — second independent audit

**Verdict: PASSED, terminal linguistic/mechanical audit.** Fourteen of fourteen assigned keys audited; no BLOCKER, MAJOR, MINOR or NIT findings. This is not activation, integration, native UI or design approval.

- Namespace: `wire.settings.account_policy`; locale: `fr`.
- English source commit: `0e7cde33ca8008d561e74f4aed774aa5f76ff7d5`.
- Auditor: `i18n-auditor`, session `35aac0a677bc`, actual served model `openai/gpt-6.1-sol`.
- Translator: separate session `7b2ba4ed501d`. No target/source/repository edits; no other independent reader's report inspected.
- Actual model is verified from this auditor's completed persisted usage for writing and executing the mechanical probes; see `provenance.json`. The requested/configured Anthropic model header is not treated as served-model evidence; no no-fallback claim is made.

## Input identity and context correction

| Input | SHA-256 |
|---|---|
| English catalogue | `1ad13f8198e28cb19fe0ecb2a6ef9b2a6990bc0d047566b71f5dd5322f594b25` |
| French target | `f998c7a608a0023275a39f4c75ff7db65d107924c78db3907c7d0a28b2d30744` |
| Original translator-input context, retained | `ccee657198d84a0bef6d20ae0343c9a864567e5a622a1b9736b1974ca296a254` |
| Corrected context used by this audit | `2c23cc38b8eb9fb354eda0e1dd476d53b1064ffe88c2e821b9e1b723b8508c02` |

Independently derived: all 14 context entries change **only** their `note` field. The roles, tones, five 29-cell limits and every other field are unchanged. English bytes equal the pinned commit and the packet's English input. Every one of the packet manifest's 13 file hashes was verified. Translation is **not** claimed to have been produced under the corrected context. Immutable copies of both contexts and source/target are retained in this bundle.

## Per-key independent linguistic assessment

All keys below have the prefix `wire.settings.account_policy.`. Full current strings are retained in `catalogues/fr/wire.settings.account_policy.json`.

| Key suffix | Assessment |
|---|---|
| `label` | `politique de choix du compte` is a natural compact account-selection-policy label; 28/29 Rich cells. |
| `help` | Eligible other accounts remain constrained by both the same provider and credential category. `bail compatible et non expiré issu d’une requête réussie` preserves all lease qualifications. `quelle que soit la politique` preserves priority for every policy. `il ne garantit pas de résultat en cache` does not promise a cache hit. Weekly refill remains primary, five-hour refill breaks deadlines within one minute. Explicit policy overrides the legacy usage-aware setting. |
| `reset_soonest.label` | `Prochaine réinitialisation` is a concise native option name, consistently referenced by help and legacy help; 26/29 cells. |
| `reset_soonest.description` | Included-plan quota reset, deterministic distribution for tied/unknown reports, and default status for new configurations are all preserved. |
| `headroom.label` | `Quota disponible maximal` correctly names the largest remaining quota rather than a larger account limit; 24/29 cells. |
| `headroom.description` | Cached remaining quota is explicit. `dix points de pourcentage ou moins` preserves **ten percentage points**, not ten percent; distributing requests among those accounts faithfully expresses spreading the picks. |
| `fair.label` | `Répartition équitable` is idiomatic and distinct from headroom/reset policies; 21/29 cells. |
| `fair.description` | Stable session-based spread is explicit and contrasted with ranking cached reset/remaining-quota data. No new guarantee of equal quota or request success is added. |
| `legacy.label` | `ancien choix selon l’usage` concisely conveys the legacy usage-aware selector; 26/29 cells. |
| `legacy.help` | Compatibility applies only when explicit account-selection policy is absent. Literal `true` selects the headroom label; literal `false` selects the fair label. Explicit policy precedence, resetting the policy row to restore reset-soonest, and clearing old overrides are retained. |
| `error.unavailable` | `Le routage des comptes est indisponible.` conveys unavailability without inventing an authentication failure. |
| `error.invalid_shape` | `Seul le champ value est attendu.` correctly restricts the payload to the literal `value` field. |
| `error.invalid_json` | `Un objet JSON est attendu.` requires an object, not merely any syntactically valid JSON. |
| `error.write_failed` | Cannot-save meaning and actionable local configuration-permission/retry guidance are intact. Professional `vous` imperatives are used; no credential paths or raw exception are introduced. |

French sentence case, U+2019 apostrophes and U+202F before French colon/semicolon are appropriate. `cache` and `JSON` follow the kept-English technical policy. No added English fallback text, unsupported claim, register error or binding glossary conflict was found. Compact labels may be lowercase under the French guide.

## Mechanical execution — actual, not notes

Executed the repository's `scripts.i18n.check.check_catalogues()` on an isolated fixture containing **only** this namespace in English/French plus the corrected English sidecar. The checker file was independently byte-compared with the pinned source commit and matches it. No scanner or full-suite execution occurred.

Command for the final run, from the core checkout:

```bash
for name in ${!CMUX_@}; do unset "$name"; done
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  "$LOCAL_OPERATOR_SCRATCHPAD/fr-account-policy-second-0e7cde33/verify.py"
```

Tool process-group ceiling: **512 MiB**; timeout: **60 seconds**. Exit **0**. Actual final stdout is `mechanical-clean-env.json`; stderr is empty.

- Valid duplicate-free JSON; exact 14-key parity and key order.
- All 14 English and target messages parse with the core ICU runtime and have zero arguments. Plural/select/CLDR-arm requirements are therefore not applicable, not untested missing arms.
- `true`, `false`, `value`, `JSON` remain literal. No Unicode `Cf` characters.
- All five labels fit their limits: **28, 26, 24, 21, 26 cells**, each at most 29.
- Five negative controls rejected: missing key and added ICU argument by the repository checker; translated `value`, 30-cell label and injected bidi control by independent extra checks. Clean fixture revalidated after restoration.

Pinned source inspection (`caller-source-proof.json`, not runtime execution) confirms enum values `reset-soonest`, `headroom`, `fair` remain separate from translated labels/help. The relay's machine `code` and HTTP statuses are separate from localized error text; `value` is the accepted payload field. This audit does not execute provider selection or settings writes.

## Explicit unverified gates and handoff

- **NOT RUN:** pseudo rendering, RTL rendering, native app, rendered fit, real requests, screenshots/design review, build/install/scanner/full suite.
- Rich-cell measurement is actual static width evidence, **not** a claim of native-widget or pixel fit.
- No ledger/activation/release/source changes and no PR/comment/commit/push. Per the task's read-only boundary, the coordinator must retain these original scratch artifacts in the canonical tracked evidence root and perform any eventual ledger operation. This bundle does not claim to be on main.
- Only this new namespace was audited. Earlier unrelated namespace findings neither disappear nor gate this verdict.
