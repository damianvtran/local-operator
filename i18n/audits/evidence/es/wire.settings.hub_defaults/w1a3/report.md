# Spanish audit — wire.settings.hub_defaults — round 2 (w1a3)

- **Verdict: failed** — 0 blocker, 1 major, 3 minor, 0 nit
- Audit id `w1a3`, round 2, 2026-10-29; supersedes `w1a1` (round 1, which bound superseded bytes sha256 `87eec88988076fbd…`)
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; disclosed fallback — see Model)
- Revision `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch feat/i18n-es-wire-runtime
- **The es catalogue is untracked at this revision** (`git status --porcelain` → `??`). The audited bytes are pinned by sha256, not by HEAD: `5fa0cb23293c4697789c9ec2ee3b41aa30caea97e809d9f601d14f9ba3b6861f`
- Keys: 9 in `en`, 9 in `es`, 9 in the en context sidecar
- Independence: I authored none of this copy, audited none of it before, and edited nothing. My only writes are this directory.

- **Post-audit drift:** this catalogue changed after the audit bound it. Bound `5fa0cb23293c4697789c9ec2ee3b41aa30caea97e809d9f601d14f9ba3b6861f`; current `0816c139dda2caf2055e7b395a4bd0563c7c145a9508e9f0dc8bc757cfe01753`. This record binds the audited bytes, not the current file. The catalogue changed AFTER this audit bound it, so this record binds the AUDITED bytes and not the current file. Re-measured only (not re-audited) on the current bytes: agents.help 68 cells, teams.help 71 cells, both inside the declared 72, and agents.help now renders whole at 80 and 100 columns. es-w1-C03 and es-w1-C04 are therefore closed by measurement, pending a round that binds the new bytes. The fix also restored the ON behaviour the audit said was being cut ('activado suscribe los nuevos').

## Verdict reason

All nine keys read. Round-1's label major was applied (agents.label now fits 29 cells) but its HELP major was not: wire.settings.hub_defaults.agents.help is still 113 cells against a declared maxCells of 72, and this namespace's rows live in section 'hub', which is NOT the bridge section, so the detail band has no wrap rung for them - the renderer's floor is truncate_cells(., width) with a visible ellipsis. At 100 columns (98 cells) and at 80 columns (78) the sentence is cut, and the end that is cut is the ON behaviour. The coordinator's round-2 premise, that the wrap rung rescues this string, is refuted below with line-level code evidence. Separately, five of these nine keys have no live consumer at all: the teams row reads wire.settings.hub_auto_update_teams_*, not this namespace.

## What was verified

1. REFUTATION of the round-2 briefing's second code fact. The briefing says that when no rung fits, _wrap_cells(help_text, width, max_lines=2) renders the help across two rows, so 113 cells 'fits two lines of >=57 cells and only truncates below roughly 57 columns'. That is not what the code does. _wrap_cells is called only from the None rung (settings_view.py:4910), and None rungs are created only in the two bridge_row branches (settings_view.py:4881 and the clause-and-bridge branch above it). bridge_row is settings_view.py:4767 'row.setting.section == bridge_availability.BRIDGE_SECTION', and BRIDGE_SECTION = 'team_bridge' (team_bridge/availability.py:33). The hub_defaults agents row is section='hub' (settings_io.py:5120). Bridge rows are additionally the only rows whose section can be invisible (availability.py:86-87). _wrap_cells' own docstring scopes it: 'For the agent-messages explanation only (design round 1, D1/D4)'. So for a hub row the floor is truncate_cells(rendered.plain, width) at settings_view.py:4937, a marked cut, not a wrap - and the cut happens at 98 cells (100 columns) and 78 (80 columns), not below ~57.
2. CONFIRMATION of the round-2 briefing's first code fact: _LABEL_BUDGET is settings_view.py:434 = _VALUE_COLUMN 34 - _ROW_INDENT 2 - _MARKER_WIDTH 2 - 1 = 29, and settings_view.py:4020 applies truncate_cells(setting.label, _LABEL_BUDGET) to every row label. The truncation is a no-op for both labels here: agents.label is 29 cells (exactly at the limit) and teams.label is 28.
3. Round-1 MAJOR es-w1-A01 is APPLIED: the round-1 text 'Autoactualizar vínculos de agentes y habilidades' (48 cells) was replaced by a 29-cell label that no longer clips; the replacement's wording is raised separately as C05.
4. Round-1 MAJOR es-w1-A02 is NOT applied: the agents.help bytes are unchanged at 113 cells.
5. Meaning of the wired keys is faithful: 'pausa las actualizaciones de TODOS los vínculos' keeps the 'all' ceiling (not just new ones), 'activado suscribe los nuevos' keeps the creation default, 'pausar las actualizaciones entrantes' keeps 'incoming', and model_default.note keeps 'ignored' rather than 'removed' and keeps 'current default model'. 'merges' stays English per the glossary's kept-English git set; lowercase 'hub' matches the locale's established usage in wire.settings.hub.description, hub_auto_update_agents.help, hub_check_interval_min.help and speech.description, while the product name stays 'Agent Hub' in wire.settings.hub.title.
6. No placeholders, ICU arguments, code spans, digits, en dashes or machine literals exist in these nine keys, so there was nothing of that class to lose; the mechanical checks confirm zero such failures.

## Findings

### es-w1-C03 — major — carried-unresolved

Keys: `wire.settings.hub_defaults.agents.help`

Current 'Desactivado pausa las actualizaciones de todos los vínculos de agentes/habilidades; activado suscribe los nuevos.' is 113 cells against the sidecar's declared maxCells 72, which settings_io.py:5121 restates in code ('Help strings are budgeted to <=72 cells (the detail line at 100 columns)'). The row is section 'hub' (settings_io.py:5120-5133). settings_view.py:4767 sets bridge_row only when section == team_bridge.BRIDGE_SECTION, and BRIDGE_SECTION is the literal 'team_bridge' (team_bridge/availability.py:33); the wrap rung exists only inside the bridge_row branches (settings_view.py:4837 comment 'A ``None`` rung is the WRAP slot (bridge rows only)', created at 4881, consumed at 4910 by _wrap_cells). A hub row therefore falls to the ladder whose last rung is settings_view.py:4937, a marked truncation. Measured: the string is cut at 98 cells on a 100-column terminal and at 78 on an 80-column one, and fits whole only from 115 columns. What is cut is 'activado suscribe los nuevos' - the ON behaviour, i.e. the half that says what turning the switch on does.

**Proposed:** 'Desactivado pausa todas las actualizaciones; activado suscribe nuevas.' (70 cells, measured) - keeps the 'all' ceiling and both halves inside 72; the row's own on/off descriptions already spell out 'vínculos de agentes y habilidades'. This is round-1 es-w1-A02's proposal and it is still the right fix.

### es-w1-C04 — minor — carried-unresolved

Keys: `wire.settings.hub_defaults.teams.help`

Current 'Predeterminado para nuevos vínculos de equipo; los existentes conservan sus propias políticas.' is 94 cells against the same declared maxCells 72. It fits the 98-cell detail band at 100 columns once the key path sheds, and truncates at 80 columns (78 cells), losing 'propias políticas'. So unlike C03 this is a narrow-width limit rather than an always-cut string. NOTE: on the current wiring this key never renders at all - see the source_notes - so the width defect is latent rather than visible today.

**Proposed:** 'Predeterminado para vínculos nuevos; los existentes conservan la suya.' (70 cells, measured).

### es-w1-C05 — minor — new

Keys: `wire.settings.hub_defaults.agents.label`

Round 1's clip was fixed by inventing an abbreviation: 'Autoactualizar agentes/habil.' is 29 cells and no longer truncates, but 'habil.' is not a Spanish abbreviation of 'habilidad' and the label now reads unlike its sibling 'Suscribir vínculos de equipo', which keeps the full noun. The English keeps the same shape on both rows ('Auto-update agent/skill links', 'Subscribe new team links'), and 'links' is absent from the Spanish label although the help and both descriptions use 'vínculos'. Meaning stays recoverable (the switch does auto-update the agents' content).

**Proposed:** 'Vínculos de agente/habilidad' (28 cells, measured) - a noun label like its sibling, no invented abbreviation, and 'vínculos' comes back; or round-1's 'Autoactualizar vínculos' (23) if the verb must stay.

### es-w1-C07 — minor — carried-unresolved

Keys: `wire.settings.hub_defaults.agents.help`, `wire.settings.hub_defaults.agents.off.description`, `wire.settings.hub_defaults.agents.on.description`, `wire.settings.hub_defaults.teams.help`, `wire.settings.hub_defaults.teams.label`, `wire.settings.hub_defaults.teams.off.description`, `wire.settings.hub_defaults.teams.on.description`

One English term, three Spanish renderings across the locale: 'vínculo' for a hub subscription link (all eight link-bearing keys here), 'enlace' for a hyperlink (tui.notifications.section_links 'Enlaces', relay.notifications.detail.section.links 'Enlaces', wire.slash.links 'Abre un enlace de esta conversación en el navegador'), and wire.slash.hub left in English entirely ('Open Hub links — per-link subscribe and auto-push policy'). Neither i18n/glossary.md nor i18n/style-guides/es.md records a row for 'link'. The sense split (subscription relationship vs hyperlink) is defensible, which is why this is minor, but an unrecorded three-way split is drift rather than a decision. Round-1 es-w1-A04, unchanged.

**Proposed:** Record the row in i18n/style-guides/es.md: hub link -> 'vínculo', browser hyperlink -> 'enlace', and settle wire.slash.hub when that namespace is audited.

## Width note on the two over-budget help strings

**Status: finding.** Both help strings remain over the declared 72 cells and this audit records both as findings, not as accepted limits. agents.help (113 cells) is a REAL DEFECT at every terminal size below 115 columns, including the 100-column width the code comment budgets for: the hub section has no wrap rung, so the sentence is cut mid-clause with a visible ellipsis and loses 'activado suscribe los nuevos'. teams.help (94 cells) is a narrow-width limit only (fits at 100 columns after the key path sheds, truncates at 80), and it is latent because the key is unwired. The briefing's reasoning that the wrap rung saves them is refuted above.

## Source-side and display-side risk

- (wiring) FIVE OF NINE KEYS HAVE NO LIVE CONSUMER. wire_settings_hub_defaults is imported once (settings_io.py:72) and only agents_label, agents_on_description, agents_off_description and agents_help are called (_hd. sites: 5126, 5130, 5131, 5133). The teams row is registered with wire.settings keys instead: settings_io.py:5139 label=_t(_ws.hub_auto_update_teams_label()), 5142/5143 its choice descriptions, 5145 its help. So teams.label, teams.help, teams.on.description, teams.off.description and model_default.note are translated, hashed and shipped in the catalogue but render nowhere in this repository. Consequence for round 2: the round-1 nit that shortened teams.label from 35 to 28 cells (es-w1-A05) fixed copy that nothing displays, and C04's width defect is latent. Recorded as a source-side fact, not as a defect in the Spanish.
- (sidecar) The sidecar declares maxCells 72 on exactly the two help keys and on no label, although both labels are rendered through the 29-cell truncation. Combined with the wiring note, the budget the translator was given (72) is the one the code comment states, and the label budget they were not given (29) is the one that produced round 1's clip and this round's abbreviation (C05).
- (display) No value in this namespace displayed as a redaction marker, so no sanctioned read_public_copy re-read was needed to judge any of the copy. Verified source-side instead: the reader serves the HEAD blob and REFUSES a working-tree difference or an untracked file (local_operator/tools/public_copy.py:487-496), so it cannot read these untracked es bytes at all; and its catalogue reader rejects any value that is not a string (public_copy.py:343-345), so *.context.json sidecars are unreadable through it by construction. Both are recorded as source-side risks, never worked around.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Static copy audit only: no rendered-frame, runtime-service, eligibility or release claim. Width verdicts are derived from the renderer source at the audited revision and from the repository's own truncate_cells, not from screenshots.
- The eight es catalogues are UNTRACKED at efcfbe812609; every record binds on-disk sha256 values rather than a committed blob.
- The audit model is an auto-routing alias; the concrete backend is not identifiable from the session header.
- Evidence files were written but never committed: this role performs no git writes.

## Model

SELF-REPORTED from this session's own model-switch notices; not independently observable. The request opened on anthropic/claude-sonnet-5-5; three provider-failure switches fired before any audit work began and the entire audit ran under radient/auto, an auto-routing alias whose concrete backend the session header does not identify. No pinned model is claimed.

Switch chain: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition was a session model-switch notice reading "provider failure: quota".
