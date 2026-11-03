# French first-reader audit: wire.settings.hub_defaults

Date: 2026-11-03. Status: **passed** for the staged linguistic/mechanical catalogue contribution only. Findings: BLOCKER 0, MAJOR 0, MINOR 0, NIT 0. Terminal review of 9 keys; no translation changes requested.

Auditor: i18n-auditor / c44e0f808e48 / **openai/gpt-6.1-sol**, derived from own completed usage in served-model-evidence.json. Independent of translator 7b2ba4ed501d and coder eae0961e4e78; not job196c3a94b346. No other new reader report inspected. Known earlier fallback disclosed in record; no requested-header or guessed no-fallback provenance.

## Inputs

English and context read from immutable core 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5; target read from original translator packet /Users/damian/.local-operator/sessions/7b2ba4ed501d/scratchpad/fr-hub-defaults-0e7cde33. Each exact hash is in record.json and receipt.json. Original manifest verified completely after this audit; packet untouched.

The translator style snapshot differs from the pinned style guide only by omission of an unrelated relay.chrome width-exception paragraph. Independently compared the diff; all normative rules relevant to these namespaces are identical. Both style snapshots are retained and hashed. No exemption from those other namespaces was imported here.

## Native assessment — audit_notes_not_execution_log

- `wire.settings.hub_defaults.agents.help`: Non/Oui identifies the switch states. « gèle toutes les MAJ agents/compétences » covers ALL existing agent/skill-link updates, not only new subscriptions; Oui subscribes new ones. MAJ is an established, readily understood French abbreviation for mises à jour, justified at the measured 72/72-cell budget. It is French abbreviation, not English leakage. Compact omission of liens is resolved by the adjacent label and full off description; not a scope change.
- `wire.settings.hub_defaults.agents.label`: Mises à jour auto des liens d’agents et de compétences names both link kinds; compétence conforms to the glossary.
- `wire.settings.hub_defaults.agents.off.description`: suspendre ... de tous les liens retains the all-links pause, not just creation defaults.
- `wire.settings.hub_defaults.agents.on.description`: abonner les nouveaux liens retains subscription on new links only.
- `wire.settings.hub_defaults.model_default.note`: Les merges du Hub retains permitted French developer term merge, current default model, and ignored obsolete Hub override; no editable override invented.
- `wire.settings.hub_defaults.teams.help`: Par défaut pour les nouveaux liens; les anciens gardent leurs règles is native concise contextual help under the team label. It preserves creation-only defaults and existing links’ own policies, not a master switch. Measured 70/72 cells.
- `wire.settings.hub_defaults.teams.label`: Abonner les nouveaux liens d’équipes retains new-team-link scope.
- `wire.settings.hub_defaults.teams.off.description`: suspendre les mises à jour entrantes des nouveaux liens d’équipes retains incoming/new/team scope; no agent-style all-existing ceiling is introduced.
- `wire.settings.hub_defaults.teams.on.description`: abonner les nouveaux liens d’équipes matches the team label and subscription semantics.

## Executed mechanical evidence

Existing core check_catalogues() executed on isolated scratch en/fr fixtures containing only this namespace. Final command exited 0 within tool timeout 90 seconds and 256 MiB process limit. audit.log is actual stdout/stderr, not these notes.

- Duplicate-aware JSON; exact source/target/context key parity; ICU argument names/kinds/order preserved; zero plural/select forms in these inputs, hence CLDR branches not applicable.
- Negative controls for an unexpected argument, missing key and invalid ICU all rejected. Original target restored in fixture and passing control rechecked.
- Opaque SDK parameters containing apostrophe, braces and hash render literally rather than being parsed as translated ICU. Technical identifiers, mode literals and API snippets manually reviewed plus scoped exact token checks. Ordinary prose words are translated, not incorrectly frozen as technical tokens.
- Unicode Cf=0 for every target; no inserted bidi/ANSI/NUL controls. French thin nonbreaking spaces before prose colon/semicolon follow the guide; code syntax stays unchanged.
- Actual Rich cell measurements retained in widths.json. Hub help values are 72/72 and 70/72. Console and SDK sidecars declare no maxCells/maxChars constraints, so no numerical native-fit guarantee is inferred. Console bounds/defaults checked unchanged against English.

Preflight logs, when present, preserve actual initial rig failures: a style snapshot discrepancy (resolved by inspecting the narrow diff and retaining both), and SDK false-positive token checks that mistook ordinary prose/substrings for machine tokens. Those are audit-rig corrections, not defects or edits in the target. Successful final evidence is audit.log.

## Remaining proof and scope

Pseudo: NOT RUN. Native RTL rendering: NOT RUN; French LTR, source Cf check is not rendered proof. Native/app fit, real-path calls and design/UX: NOT RUN. Second independent reader: separate coordinator gate, not claimed by this report. No native/app launch, install, build, broad scanner/full suite, activation/eligibility, PR/comment, source edit, generation or ledger write. Catalogue linguistic approval does not prove interface-wide English-free operation or release eligibility.

Only own scratch written; prior policy and translation bundles untouched. This original report/record/receipt is immutable handoff evidence, not a claim of tracked-main retention. Coordinator must retain it in the core canonical evidence process.
