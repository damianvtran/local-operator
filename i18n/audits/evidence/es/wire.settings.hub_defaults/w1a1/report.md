# Audit record — es / wire.settings.hub_defaults / w1a1

- **Verdict: failed** — 0 blocker, 2 major, 2 minor, 2 nit
- Audit date: 2026-10-29
- Auditor: `translation-auditor (lopdev team)`, session `4e01ec3a1b46`, model **radient/auto** (self-reported; disclosed
  fallback — see "Model" below)
- Revision: `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch `feat/i18n-es-wire-runtime`
- **The es catalogue is untracked at this revision** (`git status --porcelain` → `?? …/es/wire.settings.hub_defaults.json`).
  The audited bytes are pinned by sha256, not by HEAD: `87eec88988076fbd22637d497f9117009d7028c64b6de00bd6172763573a90c0`
- Keys: 9 in `en`, 9 in `es`, 9 in the `en` context sidecar
- Independence: I authored none of this copy and repaired none of it. I edited no catalogue, sidecar, ledger, allowance
  or generated file. My only writes are this directory.

## What was verified

All **9 keys** were read side by side against `en/wire.settings.hub_defaults.json` and its context sidecar. Mechanical
checks (`parity.py.txt`, `labels.py.txt`, both re-runnable from `run-checks.sh`): key sets identical, **no placeholders**
in any key (so nothing to drop), no code spans, no ICU plural machinery (so the es `one`/`many`/`other` rule is not
exercised here), no illegal `few`/`zero`/`two` branch, no `!`, no `«»`, no ASCII ellipsis.

Meaning, register and product-name policy are **clean**:

| Key | Judgement |
|---|---|
| `agents.help` | clean — "pausa las actualizaciones de **todos** los vínculos" keeps the ceiling ("all", not just new ones); "activado suscribe los nuevos" keeps the creation default. |
| `agents.label` | meaning clean; **width fails** (A01). |
| `agents.off.description` | clean — infinitive fragment, matches the locale's action-label convention; long (A06). |
| `agents.on.description` | clean. |
| `model_default.note` | clean — `merges` stays English per the glossary's kept-English git set; "la antigua anulación … se ignora" keeps *ignored*, not *removed*; "tu modelo predeterminado actual" keeps *current default*. |
| `teams.help` | clean in meaning and it does **not** read as a master switch ("Predeterminado para nuevos vínculos de equipo; los existentes conservan sus propias políticas"); **width fails** (A03). |
| `teams.label` | clean; would clip on a 29-cell row (A05). |
| `teams.off.description` | clean — `entrantes` keeps *incoming*. |
| `teams.on.description` | clean. |

`vínculo` is used consistently for *link* across all eight link-bearing keys, `habilidad` for *skill* and `equipo` for
*team* per the glossary; `Desactivado … / activado …` mirrors the locale's existing On/Off register
(`wire.settings.web_fetch_allow_private.help` uses `Activado permite …`).

## Findings

### es-w1-A01 — major — `wire.settings.hub_defaults.agents.label` is cut to 29 cells

Current: `Autoactualizar vínculos de agentes y habilidades` — **48 cells** (en 29).

The TUI settings list truncates **every** row label unconditionally before painting
(`settings_view.py:4020` `truncate_cells(setting.label, _LABEL_BUDGET)`, `_LABEL_BUDGET = 34 − 2 − 2 − 1 = 29`), and the
agents row is fed by this exact key (`settings_io.py:5126` `label=_t(_hd.agents_label())`). It renders as
`Autoactualizar vínculos de ag…`. The sidecar declares `maxCells` only for the two `help` keys — the 29-cell label
budget that `account_policy.context.json` declares explicitly is absent here, so the translator had no budget to work to.

Proposed: `Autoactualizar vínculos` (23 cells) — keeps the action the switch performs, leaves the scope to the help
text and the on/off descriptions; or `Vínculos de agente/habilidad` (27) if the scope must stay on the label.

### es-w1-A02 — major — `wire.settings.hub_defaults.agents.help` cannot be read whole on the settings surface

Current: `Desactivado pausa las actualizaciones de todos los vínculos de agentes/habilidades; activado suscribe los nuevos.`
— **113 cells** against a declared `maxCells: 72`.

The declared 72 is not decorative: `settings_io.py:5121` states "Help strings are budgeted to <=72 cells (the detail line
at 100 columns)". The detail line is `_detail_width()` = terminal width − 2 (98 cells at 100 columns, 78 at 80), and the
**wrap slot is bridge-rows-only** — `settings_view.py:4931`: "A `None` rung is the WRAP slot (bridge rows only)". For a
`hub` row the shed ladder drops the key path and the clause, and when only the help remains and it still does not fit it
is **truncated with a visible `…`** (`truncate_cells(rendered.plain, width)`). At 113 cells the string exceeds even the
98-cell maximum, so it is cut on every measured terminal size — and what is cut is the end, exactly where the ON
behaviour ("activado suscribe los nuevos") lives.

The sidecar's instruction "Wrap explanatory copy rather than truncating it" is attached to all nine keys as a
namespace-level note; it tells the translator not to buy width by deleting meaning, but it does **not** make an
over-budget help string renderable, because nothing on this surface wraps a non-bridge help.

Proposed: `Desactivado pausa todas las actualizaciones; activado suscribe nuevas.` (70 cells) — both behaviours and the
"all" ceiling survive, and the label plus choice descriptions already carry the agent/skill scope.

### es-w1-A03 — minor — `wire.settings.hub_defaults.teams.help` is 94 cells against the 72-cell budget

Current: `Predeterminado para nuevos vínculos de equipo; los existentes conservan sus propias políticas.` (94 cells).

It fits the 98-cell detail line **only** when the key path and any state clause have been shed; with the key path it needs
120 and with an off-default clause 110. So at 100 columns the user reads the sentence but never learns which config key
the row is, and when the row is off-default the ladder can shed the help in favour of `default: on  hub.auto_update.teams`.
At 80 columns (78 cells) it truncates mid-clause and loses "sus propias políticas".

Proposed: `Solo define los vínculos nuevos; los existentes conservan su política.` (70 cells).

### es-w1-A04 — minor — glossary drift on "link": `vínculo` vs `enlace` vs untranslated `links`

This namespace maps *link* to **`vínculo`** (8 occurrences). Elsewhere in the same locale: `wire.slash.links`
("Open a link from this conversation in a browser") is **`enlace`**, and `wire.slash.hub` ("Open Hub links — per-link
subscribe and auto-push policy") is left **untranslated**. Three renderings of one English term, and neither
`i18n/glossary.md` nor `i18n/style-guides/es.md` records a row for *link*.

The sense split (a hub subscription link vs a browser hyperlink) is defensible and `vínculo` reads correctly for a
subscription relationship, which is why this is minor rather than a term violation — but an unrecorded three-way split
is drift, not a decision.

Recommendation: record the row (hub link → `vínculo`, browser hyperlink → `enlace`) and settle `wire.slash.hub` when that
namespace is audited.

### es-w1-A05 — nit — `wire.settings.hub_defaults.teams.label` (35 cells) would clip on a 29-cell row

`Suscribir nuevos vínculos de equipo` is 35 cells. The TUI teams row currently takes its label from the older
`wire.settings.hub_auto_update_teams_label` (`settings_io.py:5138`), so the clip is not proven on that surface; if this
key reaches any 29-cell row (desktop/mobile wiring is outside this audit), it clips. Shorter: `Suscribir vínculos de
equipo` (28).

### es-w1-A06 — nit — the two "off" choice descriptions run to 168–170 % of the English

`agents.off.description` 73 cells (en 43), `teams.off.description` 69 (en 41). No `maxCells` is declared for choice
descriptions, so no budget is broken; recorded because Spanish is running far past the guide's 15–25 % expansion norm
where the English is a terse fragment. If a chip budget ever applies: `pausar actualizaciones de todos los vínculos de
agentes y habilidades` (69) and `pausar las actualizaciones entrantes de los vínculos nuevos` (59).

## Disposition

**Send back to the translator for width, not for meaning.** The two majors are both width defects with measurable,
declared budgets behind them: the label truncates in the renderer that consumes it, and the help string is longer than
any detail line it can be painted on. Nothing here is a wrong-meaning, a broken placeholder or a behavioural lie, and
the register, glossary terms and product-name policy are already correct — so a remediation that preserves the current
wording and tightens the four long strings is a one-commit fix. Do not accept the "wrap rather than truncate" note as a
waiver: this surface has no wrap rung for non-bridge rows.

## Model

The session header names `anthropic/claude-sonnet-5-5` as the primary model. Three provider-failure switches fired on
`quota HTTP 429` during this session (`openai/gpt-6-astra` → `kimi/k3.1-pro` → `radient/auto`); everything in this
namespace was judged under **radient/auto**. This is a disclosed fallback, not the pinned primary model.

## Limitations

- No native-speaker review is claimed; the auditor is an LLM.
- No rendered frame was captured (host under load, heavy runs excluded). The width verdicts are derived from the
  renderer source at the audited revision, quoted in `audit.log` — not from screenshots. If a later QA pass captures the
  settings screen, `es-w1-A01` should show the `…` and `es-w1-A02` should show the cut sentence, which is the cheapest
  confirmation available.
