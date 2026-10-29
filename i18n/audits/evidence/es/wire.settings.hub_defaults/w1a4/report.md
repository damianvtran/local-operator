# Spanish audit — wire.settings.hub_defaults — round 3 (w1a4, delta only)

- **Verdict: passed** — 0 blocker, 0 major, 2 minor, 0 nit
- Audit id `w1a4`, round 3, 2026-10-29; supersedes `w1a3` for the **delta only**
- Delta keys: `wire.settings.hub_defaults.agents.help`, `wire.settings.hub_defaults.teams.help`
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; fallback disclosed)
- Bound bytes: `0816c139dda2caf2055e7b395a4bd0563c7c145a9508e9f0dc8bc757cfe01753` (untracked; verified by this round)
- Keys: 9 en / 9 es

## Prior rounds

- round 1 w1a1 — first independent audit of this namespace
- round 2 w1a3 — final-byte audit; raised the findings this delta judges
- round 3 w1a4 — this record; delta only, on bytes that moved after round 2

Wave context: wire.console w1a3 amendment A1 (NOT this namespace): that amendment established the same-code-control rule this delta applies — a width finding must first be run through the identical budget on the English source before it is called a translation defect.

## What this round did and did not do

It re-judged the four changed values and diffed everything else against what w1a3 read. It did not re-read or re-judge the unchanged keys.

## Delta judgement

### `agents.help` — PASS

PASS. The ON half is present - 'activado suscribe los nuevos' - which is exactly what es-w1-C03 said was being cut. The OFF half keeps the EN asymmetry (off covers the updates, on covers the new ones). The elided scope 'de todos los vínculos de agentes/habilidades' is recoverable rather than lost: the row label names agentes/habil., and the two unchanged choice descriptions spell out 'todos los vínculos de agentes y habilidades' and 'nuevos vínculos de agentes y habilidades'. Spanish definite plural 'las actualizaciones' carries the totality; the explicit 'todas' would cost 6 cells and put the string back over 72.

### `teams.help` — PASS

PASS. Both halves survive: the creation default ('por defecto para vínculos nuevos') and the ceiling on existing links ('los existentes mantienen su política'). The distributive singular 'su política' is correct Spanish for EN's per-link 'their own policies'. 'de equipo' is elided and recoverable from the row label 'Suscribir vínculos de equipo'. This key is unwired (see source note), so the width is latent either way.

## Findings closed by measurement

- **es-w1-C03** — agents.help 113 -> 68 cells: inside the declared 72, whole at 80 and 100 columns, ON behaviour restored.
- **es-w1-C04** — teams.help 94 -> 71 cells: inside the declared 72, whole at 80 and 100 columns.

## Findings still open

- **es-w1-C05** (minor) — open — not in the delta
  - Unchanged in this delta (still 29 cells, still a no-op at the budget): 'Autoactualizar agentes/habil.' abbreviates 'habilidades' to 'habil.' and drops 'vínculos', so the row does not read like its sibling 'Suscribir vínculos de equipo'.
  - Proposed: 'Vínculos de agente/habilidad' (28 cells) if the wave is ever reopened; not a gate for this round.
- **es-w1-C07** (minor) — open — not in the delta
  - Unchanged in this delta: 'vínculo' for a hub subscription link here, 'enlace' for a hyperlink in tui.notifications.section_links / relay.notifications.detail.section.links / wire.slash.links, and wire.slash.hub still in English. Defensible sense split, but unrecorded in i18n/glossary.md or the es guide.
  - Proposed: Record the row in i18n/style-guides/es.md: hub link -> vínculo, browser hyperlink -> enlace.

## Accepted trades (considered, not findings)

- agents.help dropped the explicit 'todas' (EN 'for all'); the definite plural plus the unchanged off description carry totality, and 'todas' would cost 6 cells against a 72-cell budget.
- teams.help dropped 'de equipo' (EN 'team links'); the row label supplies it.

## Source-side notes

- (wiring) FIVE OF NINE KEYS STILL HAVE NO LIVE CONSUMER: settings_io.py:72 imports the key module once and calls only agents_* (5126, 5130, 5131, 5133). The teams row reads wire.settings.hub_auto_update_teams_* (5137-5143). teams.* and model_default.note render nowhere; this delta's teams.help fix is therefore latent.
- (sidecar) The sidecar declares maxCells 72 on the two help keys and none on the labels, although both labels are rendered through the 29-cell truncation. Unchanged source-side gap.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Delta audit: only the four changed values were re-judged; the rest were diffed, not re-read.
- No rendered frame was captured; width verdicts come from the renderer's own truncate_cells.
- The catalogue is untracked, so this binding is to on-disk bytes and can be invalidated by a further edit. No copy of the superseded bytes is retained anywhere.
- The audit model is an auto-routing alias; the concrete backend is not identifiable.

## Model

SELF-REPORTED from this session's model-switch notices; not independently observable. The request opened on anthropic/claude-sonnet-5-5; three provider-failure switches fired before any audit work and the whole audit ran under radient/auto, an auto-routing alias whose concrete backend the header does not identify. No pinned model claimed.

Switch chain: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition a session model-switch notice reading "provider failure: quota".
