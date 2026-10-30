# Audit — es / tui.app / w6a2 (round 2)

- **Verdict: passed** — all six round-1 defects closed (APP-1/2/3/4/5/7), **no new defect** in the
  values the fixes touched or in their neighbours. The two remaining nits (APP-6, APP-8) were a
  preference and an extraction-scope question; both unchanged and accepted.
- Round 1: `i18n/audits/evidence/es/tui.app/w6a1/` (`changes_requested`, 0 blocker, 3 major,
  2 minor, 3 nit).
- Revision `b925c2e11be36653dfe88ac78f627bc2531427df`; es target **untracked**, so the on-disk
  sha256 binds it: es `6afd01d6d0aff0b2…`, en `7e5f39df72438663…`, context `a840ac61e3c1e117…`
  (full digests in `record.json`).
- Branch: `feat/i18n-es-tui` — the brief named `tui-wip`; `git rev-parse --abbrev-ref HEAD` says
  otherwise, and the log records what is actually checked out.
- Keys re-read: **46** en / 46 es / 46 context — every value, not only the six the implementer
  touched.
- Auditor: independent translation auditor (lopdev team), session `7fa3a39af5d9`; model
  **radient/auto** as actually run (fallback chain in `record.json`).

## Mechanical result — clean

`scripts/i18n/check.py` exits 0 (191 files, 2348 literals, 51 advisories). Probe section A on the
current bytes: key parity 46/46; ICU argument tuples identical on every key; **all six plurals**
(`led_by`, `config.effect_new`, `config.needs_relaunch`, `parked.wakes`, `parked.monitors`,
`stop_all.will_be_stopped`) carry `one`/`many`/`other` with no `few`/`zero`, no empty branch and
`other` defined, and each renders through the shipped runtime at 1 / 2 / 1000000 with `{shown}`
present in all three branches: `1 despertar`, `2 despertares`, `7 de despertares`,
`1 miembro`, `7 de miembros`; 17 machine literals intact (`config.yml`, the two
`LOCAL_OPERATOR_NO_*` env vars, `/agent <name> <message>`, `/agent clear`, `/notifications read`,
`/stop all`, `/new`, `Terminal.app`, `lo ›`/`lo ⣾`/`lo !`, `ctrl+c`, `ctrl+b`, `{seconds} s`, both
`lop config edit …` lines); every backticked span byte-identical to en; NBSP pair in
`aida.extras_used`, the two-space indent in `notifications.more`, the ` — ` and `…` conventions all
preserved.

## Fit — re-measured, not quoted

`name_width` was **derived from the live registry**, not assumed: `max(len(", ".join(names)))+2`
over `SLASH_COMMANDS` (62 commands) = **20**, longest `/settings, /config` and `/credential, /cred`.

| row | es description | composed (name_width 20) | ceiling 74 |
|---|---|---|---|
| `help.paste_note` | 48 | **68** | 6 spare |
| `help.skill_note` | 53 | **73** | **1 spare** |
| `help.cmd_paste_note` | 52 | **72** | 2 spare |
| `help.copy_note` (unchanged) | 45 | 65 | 9 spare |
| `help.copy_note_more` (unchanged) | 54 | **74** | 0 — at the ceiling already |
| `help.model_saved_note` (unchanged) | 44 | 64 | 10 spare |

All three previously-breaking rows now sit inside the pinned 74 at 80 columns, so none wraps and no
tail lands at the key gutter. `help.skill_note` has a single cell of headroom and `copy_note_more`
sits exactly on the ceiling: both are worth knowing before any later edit, and neither is a finding.

`stop_all.already_leaving` composes to **57** cells inside its parentheses at a two-digit count
(`(12 ya saliendo — se volvió a pedir y no se insistió más)`), against the 70-cell notice body at 80
columns (`app.py:45302-45318`) — 13 spare.

## Findings — every round-1 finding closed

### APP-1 · major · CLOSED · `config.toggle_on` / `config.toggle_off`

> `{label} activada — surte efecto desde tu próximo turno`
> `{label} desactivada — las llamadas se rechazarán hasta que se vuelva a activar`

Feminine agreement now matches both labels the receipt can carry. Re-verified at the source: the
probe re-reads `app.py` and finds exactly **two** `_x.config_label_*` builders
(`config_label_web_search`, `config_label_web_fetch`), whose es values are `búsqueda web` and
`consulta web` — both headed by a feminine noun. So `búsqueda web activada` / `consulta web
desactivada` are correct in every render. The off-state warning is still exactly as strong as en
(`las llamadas se rechazarán hasta que se vuelva a activar`).

### APP-2 · major · CLOSED · `goals.settled_heading`

> `objetivos finalizados`

Matches the panel heading that will ship beside it (`tui.small.goal_panel.settled_heading` =
`finalizados`, and `more_settled` = `… {hidden} más finalizados`), so the two files agree on screen.
The `asentados` reading is gone.

### APP-3 · major · CLOSED · the three `/help` rows

> `adjunta una imagen o un archivo del portapapeles` · `ejecuta la habilidad indicada en el resto de la línea` · `igual, si la terminal lo reenvía; no en Terminal.app`

Measured above: 68 / 73 / 72 against the 74 ceiling. The calque `con nombre` is gone
(`la habilidad indicada`), `del portapapeles` keeps the source's meaning without the redundant
`del sistema`, and the `cmd+v` note keeps the conditional `si` (which matches en's "where the
terminal forwards it") and names the same non-forwarding terminal, `Terminal.app`, byte-exact.

### APP-5 · minor · CLOSED with APP-3 · `help.cmd_paste_note` (mood)

The subjunctive `reenvíe` is gone; `lo reenvía` states the definite fact, as the finding asked.

### APP-4 · minor · CLOSED · `stop_all.already_leaving`

> `{shown} ya saliendo — se volvió a pedir y no se insistió más`

Count-neutral: the plural indirect object `se les` is gone, so one session and many both parse
(`1 ya saliendo — se volvió a pedir y no se insistió más`). The two facts en carries — asked again,
then left alone — both survive.

### APP-6 · nit · unchanged, accepted

`config.label_web_fetch` = `consulta web` was a preference, not a defect, and the label is now load
bearing for the feminine agreement above; changing it would need a fresh agreement check. Leave it.

### APP-7 · nit · CLOSED with APP-3

The `con nombre` calque is gone.

### APP-8 · nit · unchanged, recorded

`sidebar.why_retry_budget` / `why_retired` / `why_left_screen` are consumed only by a
`logger.warning(...)` line (`app.py:11503-11511`). The glossary keeps developer logs English, so
cataloguing them puts Spanish into logs nobody localises. Not the translator's defect and not this
round's: a catalogue-scope question for the extraction wave. Re-confirmed, still true.

## Neighbours checked because the fixes could have touched them

- `help.copy_note` / `copy_note_more` / `model_saved_note` sit in the same ladder as the three
  changed rows; all four measured and unchanged, and `copy_note_more` is still exactly at 74.
- The `{label}` receipt is shared by both switches, so one gendered participle had to serve both —
  it does (both heads feminine). A future masculine label would need `{label}: activado`; the probe
  now fails on purpose if a third builder appears.
- `skills.led_by`, `config.effect_new`, `config.needs_relaunch`, `parked.wakes`,
  `parked.monitors`, `stop_all.will_be_stopped`: re-rendered at 1 / 2 / 1000000 with no branch
  dropping its argument and no `few`/`zero` introduced.
- The `goals` heading's own neighbours (`tui.small.goal_panel.more_settled*`) re-read: all three
  settled-goal strings now say `finalizados`.

## Limitations

LLM judgement, no native-speaker attestation, no rendered frame captured: the /help widths are
computed against the pinned 74 with `rich.cells.cell_len` and the registry-derived `name_width`, not
photographed. No catalogue, sidecar, ledger or allowance was edited.

## Ledger row — prepared, NOT written

This round forbids ledger writes, so `i18n/ledger/es.json` was not touched. Row for the owner:

```json
{
  "locale": "es",
  "namespace": "tui.app",
  "source_sha256": "7e5f39df72438663a66afca0e5338e90b7b984bdca6b562195ce976945bd7875",
  "translator": "<from the wave's translation record — not restated here>",
  "auditor": {
    "agent": "i18n-auditor (independent; audited, did not translate)",
    "model": "radient/auto",
    "date": "2026-10-30",
    "result": "passed",
    "findings": 0
  },
  "status": "passed",
  "note": "r2 passed, all 6 w6a1 defects closed. Evidence i18n/audits/evidence/es/tui.app/w6a2/record.json; es sha256 6afd01d6…, en 7e5f39df…; independent, radient/auto, 2026-10-30. /help rows re-measured 68/73/72 vs 74; toggle agreement re-verified against both builders."
}
```

Note-length check: the `note` above is exactly 261 characters, inside the 280-character budget that binds a ledger row.
