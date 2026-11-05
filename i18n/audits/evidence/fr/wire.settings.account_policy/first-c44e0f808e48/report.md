# Independent French account-policy audit — first reader

Verdict: **passed**, terminal linguistic/mechanical review of all 14 keys in `wire.settings.account_policy`. Findings: BLOCKER 0, MAJOR 0, MINOR 0, NIT 0. No translation changes requested.

Auditor: i18n-auditor; session c44e0f808e48. Translator session: 7b2ba4ed501d, distinct from this reader. This reader did not translate or implement the contribution, inspect the other reader's report, or edit source/target/ledger files. The second independent settings reader remains a separate coordinator gate.

## Actual model provenance

Served model: **openai/gpt-6.1-sol**, proven by this session's completed assistant usage records captured in `served-model-evidence.json` (provider=openai, model_id=gpt-6.1-sol). This is not a claim derived from the requested/model header. The earlier run disclosed temporary quota fallback; this resumed audit binds its own completed usage, not a guessed no-fallback assertion.

## Inputs and context reconciliation

English source is pinned to core commit `0e7cde33ca8008d561e74f4aed774aa5f76ff7d5`.

- English: `1ad13f8198e28cb19fe0ecb2a6ef9b2a6990bc0d047566b71f5dd5322f594b25`.
- French original packet target: `f998c7a608a0023275a39f4c75ff7db65d107924c78db3907c7d0a28b2d30744`.
- Original translator-input context: `ccee657198d84a0bef6d20ae0343c9a864567e5a622a1b9736b1974ca296a254`.
- Proposed context actually audited: `2c23cc38b8eb9fb354eda0e1dd476d53b1064ffe88c2e821b9e1b723b8508c02`.

Both context snapshots are retained byte-for-byte here. Independently checked: exactly 14 note-only changes; roles, tones, maxCells and every non-note field are identical. This is an audit under the corrected context, **not** a claim that the translator originally used that context. The corrected notes preserve argument-free copy and separate localized prose from machine identifiers.

## All-key linguistic coverage (audit notes, not execution log)

1. `label`: « politique de choix du compte » is compact native settings terminology; account selection, not an authorization policy. Measured 28/29 cells.
2. `help`: eligibility stays conjunctive: same provider **and** same credential category. « bail compatible et non expiré issu d’une requête réussie » retains every lease qualifier. « quelle que soit la politique » makes priority universal. « il ne garantit pas de résultat en cache » does not promise a cache hit. Weekly refill remains primary, five-hour refill secondary when weekly timings differ by at most a minute. Explicit policy overrides the legacy usage-aware setting. « catégorie d’identifiants » is a reasonable developer-facing rendering of credential tier in this context.
3. `reset_soonest.label`: « Prochaine réinitialisation » naturally labels the earliest-reset policy, reused exactly in help. Measured 26/29 cells.
4. `reset_soonest.description`: nearest included-plan reset, deterministic distribution for ties/unknown data, and default for new configurations are all preserved.
5. `headroom.label`: « Quota disponible maximal » clearly denotes most available quota. Measured 24/29 cells.
6. `headroom.description`: cached known remaining quota and spreading across accounts within **ten percentage points** of the best are preserved; no erroneous ten-percent ratio.
7. `fair.label`: « Répartition équitable » is a natural concise policy label. Measured 21/29 cells.
8. `fair.description`: stable session-based distribution remains distinct from ranking cached resets or remaining quota.
9. `legacy.label`: « ancien choix selon l’usage » is clear compatibility terminology. Measured 26/29 cells.
10. `legacy.help`: compatibility applies only when account policy is absent. Literal `true` maps to the headroom display label; `false` maps to fair. Explicit policy precedence is retained. Reset targets the account-policy row, restores the default reset-soonest policy, and clears legacy overrides; it does not imply reset of an account or credential.
11. `error.unavailable`: account routing unavailability is preserved without introducing authorization, provider-outage or credential claims.
12. `error.invalid_shape`: only the literal `value` field is expected; no machine-field renaming.
13. `error.invalid_json`: requires an object, not merely arbitrary valid JSON; literal `JSON` retained.
14. `error.write_failed`: failed save plus actionable local configuration-permission remedy retained; formal vous imperatives, no credential path or raw exception disclosure.

French typography, register and glossary terminology conform. Stable IDs/codes/config names are not translated into this argument-free namespace; literal true/false/value/JSON occurrences were independently checked against English. No invented bidi controls, interpolation, code markup or plural branches.

## Executed mechanical evidence

Actual command:

`env -u XPC_FLAGS PYTHONDONTWRITEBYTECODE=1 /Users/damian/local-operator/.venv/bin/python /Users/damian/.local-operator/sessions/c44e0f808e48/scratchpad/account-policy-fr-first/audit.py`

Run with tool process limit **512 MiB**. Exit **0**. Actual stdout/stderr retained in `audit.log`; these report notes are not a substitute for that execution log.

- Existing core `scripts/i18n/check.py::check_catalogues()` executed against isolated scratch fixtures containing only this English/French namespace and the proposed sidecar: zero problems.
- Three discriminating negative controls rejected: unexpected argument, missing key, invalid ICU. Restored exact original target fixture and passing control afterward.
- Duplicate-aware JSON parsing; exact 14-key source/target/context parity; argument counts zero; no plural/select requirements apply; each French value renders byte-for-string unchanged through the existing core runtime.
- Every target has Unicode category Cf count zero. Literal true/false/value/JSON occurrence parity passes.
- Rich `cell_len` measured the five constrained labels: 28, 26, 24, 21, 26 cells, each <=29.

## Limits and disposition

Pseudo-localization: **NOT RUN**. RTL/native bidi rendering: **NOT RUN** (French is LTR; source Cf check is not a rendered bidi proof). Native/app fit and real-path behavior: **NOT RUN**. No native launch, app build, install, scanner, full suite, release/activation, PR/comment, or ledger write. Rendered integration/design/UX verification and the second independent linguistic reader remain coordinator-owned gates. This pass does not activate a language or establish interface-wide English-free completeness.

Original translation packet is untouched. Only this session's scratch was written. This immutable report/receipt must be retained by the coordinator through the core tracked evidence process; its present scratch location is not a claim of tracked-main retention.
