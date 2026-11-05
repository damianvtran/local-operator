# French wire.settings.hub_defaults staged translation

9/9 keys translated. Not audited, integrated or activated.

Source commit: 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5. Exact English/context snapshots, style/glossary snapshots and hashes are retained. caller-proof.json is source inspection, NOT runtime execution evidence. It includes complete sibling-locale key/non-English-value counts and caller excerpts with immutable file hashes.

## Native choices

- agent: agent
- skill: compétence
- team: équipe
- link: lien
- subscribe: abonner
- Hub: Hub, surface name retained
- merge: merge, kept-English technical word per French style guide
- MAJ: Compact French abbreviation for mises à jour, proposed only for 72-cell agents help; all-update pause and new-link subscription both retained. Full spelling in label and neighboring descriptions.
- scope: Agent/skill OFF is a ceiling on existing links plus creation default; team setting only affects new links. No master team-disable claim.
- model: Current default model used for Hub merges; old Hub-specific model override ignored.

## Caller/context caveat

At pinned settings_io.py only four _hd.agents_* functions have direct call sites. Team rows still use _ws.hub_auto_update_teams_* from wire.settings; no direct caller was found for the four hub_defaults.teams.* keys or model_default.note. All nine requested keys are translated, but this packet alone cannot claim changes to those five rendered surfaces.

## Actual validation

validation.log captures the actual existing ICU parser execution at 256 MB heap: 9 exact keys in English order, no added arguments, no bidi controls, numeric literal parity, two bad controls rejected. widths.log captures rich.cells.cell_len results; rendered fit was NOT RUN. No full validator/scanner/native/build/pseudo/RTL/semantic-audit claims.

## Provenance

Served model(s): openai/gpt-6.1-sol, derived from this packet's translation write/edit usage entries, not model header. Session 7b2ba4ed501d, i18n-translator. provenance.json names entries and quota-fallback routes. No MT service used.

## Handoff

Immutable French gap: 9; staged gap: zero; repository gap remains 9 until integration. Independent audit required. These are translator staging records, not audit records; do not mark passed or activate from them. No source-English/context edits, generated edits, ledger edits, commits, PRs or unrelated namespace changes.
