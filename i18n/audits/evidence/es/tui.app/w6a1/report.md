# Audit — es / tui.app / w6a1

- **Verdict: changes_requested** — 0 blocker, **3 major**, 2 minor, 3 nit
- Revision `b925c2e11be36653dfe88ac78f627bc2531427df`; the es target is **untracked** at that
  revision: es `c735760d47d9985…`, en `7e5f39df72438663…`, context `a840ac61e3c1e117…`
  (full digests in `record.json`).
- Keys read: **46** en / 46 es / 46 context — every key has a context entry, all read.
- Auditor: independent translation auditor (lopdev team), session `6a2f6fa808a7`; model
  **radient/auto** as actually run (fallback chain in `record.json`).

## What I read

All 46 values against en and the full sidecar, then the code: the Help tab
(`app.py` 56170–57200, including the width contract in the comments), the config-change receipt
(`app.py:42825–42836`), the sidebar why-list (`app.py:11499–11511`), the parked/Aida rows, and
`goal_panel.py` for what "settled" means. Terminology witnesses: `wire.settings` (`contacto`,
`Ajustes`, choice words — ledger `passed` at the current en hash), `wire.settings.hub_defaults`,
`tui.viewer` and `tui.notifications` (committed TUI copy).

## Mechanical result — clean, with one addition

`scripts/i18n/check.py` exits 0. Section A of the probe passes everything, and adds two checks
the validator does not perform:

- **Full CLDR coverage** for all six plurals: `led_by`, `config.effect_new`,
  `config.needs_relaunch`, `parked.wakes`, `parked.monitors`, `stop_all.will_be_stopped` each
  carry `one`/`many`/`other`, no empty branch, `other` defined, no `few`/`zero`, and the
  generated CLDR table agrees (1 → `one`, 2 → `other`, 10⁶ → `many`). Rendered through the
  shipped runtime at 1 / 2 / 1000000 / 1500000: `1 despertar`, `2 despertares`,
  `1000000 de despertares`, `1500000 despertares` — correct for es, where `many` is exact
  multiples of 10⁶ only and the code passes the count as a raw string (`{shown}`), so digits
  stay unformatted by design in both locales.
- **Per-branch rendering**: every branch renders its source counterpart's arguments (the `many`
  branch is compared with en's `other`). A branch that silently drops `{shown}` — which still
  parses and still passes the repo validator — is caught; control 4 proves it.

Preserved and verified: both NBSPs in `contactos extra usados hoy: {used}\xa0de\xa0{maximum}`
(the sidecar calls them load-bearing); `config.yml`; the env vars and the `lop config edit …`
lines; the glyph code spans `lo ›`, `lo ⣾`, `lo !`; `ctrl+c`/`ctrl+b`; `/stop all`,
`/notifications read`, `/agent clear`, `/agent <name> <message>`, `/new`; `Terminal.app`;
`{seconds} s`; ` — ` with spaces; `…`; the two-space indents in `notifications.more` and
`providers.*`.

## Findings

### APP-1 · major · `tui.app.config.toggle_on` and `tui.app.config.toggle_off` (same defect)

> `{label} activado — surte efecto desde tu próximo turno`
> `{label} desactivado — las llamadas se rechazarán hasta que se vuelva a activar`

Agreement is wrong in **every** render. `{label}` is produced by exactly two builders
(`app.py:42825`: `config_label_web_search`, `config_label_web_fetch`), and both labels are
feminine — `búsqueda web`, `consulta web`. The receipt therefore reads
`búsqueda web activado`, `consulta web desactivado`.

Suggested: `{label} activada — surte efecto desde tu próximo turno` and
`{label} desactivada — las llamadas se rechazarán hasta que se vuelva a activar`.
(Both current labels are feminine, so one gendered form is enough; a future masculine label
would need a neutral shape such as `{label}: activado`.)

### APP-2 · major · `tui.app.goals.settled_heading`

> `objetivos asentados` (en `settled goals`)

The goals this heading introduces are the **finished** ones: `goal_panel.py` paints them with
`_DONE_STYLE = Style(strike=True, dim=True)`, whose own comment says the strike "reads as
'finished'". `asentado` is a calque of "settled" in its resting/established sense — a reader
can take the list as *goals that were set*, and the es catalogue already prints the same
heading inside the panel as `finalizados` (`tui.small.goal_panel.settled_heading`, in flight in
this branch and untracked, so not shipped precedent — but the two files would ship together and
would disagree on screen). The one committed use of `asentados` for "settled"
(`wire.settings.wakes_triggers_project_staleness_enabled.help`, about **projects** being
done/paused/archived) does not transfer to completed goals.

Suggested: `objetivos finalizados`.

### APP-3 · major · `tui.app.help.paste_note`, `tui.app.help.skill_note`, `tui.app.help.cmd_paste_note`

The `/help` block composes every key row as `name.ljust(name_width) + note` and the code pins
the row at **≤ 74 cells at 80 columns** (`app.py:56210–56216`, `56249–56265`);
`test_help_documents_the_composer_copy_key_and_its_release` pins the 74 and a row padded to 75
fails it, while `test_the_paste_key_rows_do_not_wrap_at_eighty_columns`
(`tests/unit/tui/test_app_pilot.py`) pins both paste rows against the painted frame. The
`cmd+v` row is documented as having **zero headroom** (`app.py:56352–56366`: "This row composes
to EXACTLY 74. One more character wraps it"), and a wrap puts the tail at the key gutter, where
it reads as another key row — design round 1 D2 verbatim. `name_width` is 20 on this registry
(confirmed twice from the source arithmetic: 51+20=71, 54+20=74).

| key | en | es | composed es | result |
|---|---|---|---|---|
| `help.paste_note` | 51 → 71 | 60 | **80** | 6 over → wraps |
| `help.skill_note` | 42 → 62 | 59 | **79** | 5 over → wraps |
| `help.cmd_paste_note` | 54 → **74 (pinned)** | 58 | **78** | 4 over → wraps |

Suggested wordings, measured with `rich.cells.cell_len` against the same 74:

- `help.paste_note` → `adjunta una imagen o un archivo del portapapeles` (48 → 68)
- `help.skill_note` → `ejecuta la habilidad indicada en el resto de la línea` (53 → 73)
- `help.cmd_paste_note` → `igual, si la terminal lo reenvía; no en Terminal.app` (52 → **72**,
two cells of headroom; the conditional `si` matches en's conditional claim and the indicative
`envía` fixes the mood)

The other three rows fit: `copy_note` 45→65, `model_saved_note` 44→64, and `copy_note_more`
54→**74 exactly** (at the ceiling, no spare cell — worth knowing before any later edit).

### APP-4 · minor · `tui.app.stop_all.already_leaving`

> `{shown} ya saliendo — se les volvió a pedir y no se insistió más`

The en phrase is deliberately count-neutral ("the phrase covers one and many, so only the digit
varies" — sidecar note); the es introduces a plural indirect object, so at one session it reads
`1 ya saliendo — se les volvió a pedir`. The impersonal fixes it without losing the meaning.

Suggested: `{shown} ya saliendo — se volvió a pedir y no se insistió más`

### APP-5 · minor · `tui.app.help.cmd_paste_note` (mood)

> `lo mismo, donde la terminal lo reenvíe; no en Terminal.app`

The fact is definite (this terminal does or does not forward the chord), so the indicative
reads naturally; the subjunctive makes it sound hypothetical.

Suggested: `igual, si la terminal lo reenvía; no en Terminal.app` — the same replacement APP-3
proposes, which fixes width and mood together.

### APP-6 · nit · `tui.app.config.label_web_fetch`

`consulta web` for "web fetch". `consulta` in Spanish dev usage means *query/request*, so the
pair reads `búsqueda web` / `consulta web` — two things a reader must disambiguate from context.
`descarga web` or `obtención web` names what the tool does. Preference only; the current pair is
distinguishable and matches the lower-case house style.

### APP-7 · nit · `tui.app.help.skill_note` (idiom)

`ejecuta una habilidad con nombre` is a calque of "a named skill" — `con nombre` reads as "with
a name". `la habilidad indicada` is what the gesture actually does. Folded into APP-3's
replacement.

### APP-8 · nit · `tui.app.sidebar.why_retry_budget` / `why_retired` / `why_left_screen`

All three are consumed only by a `logger.warning(...)` line (`app.py:11503–11511`), never by the
panel. The glossary keeps developer logs English (§1: dev logs are an allowlist category), so
cataloguing these three puts Spanish into logs nobody localises. Not the translator's defect —
a catalogue-scope question for the extraction wave; recorded so it is not rediscovered.

## Verified, not flagged

- `skills.led_by`: `Dirigido por {manager} · {shown} miembro(s)` with all three categories and
  `{shown}`/`{count}` kept exactly as en passes them (the count is raw by design).
- `stop_all.will_be_stopped`: the destructive warning survives intact — `se detendrá en vez de
  dejarla terminar` / `se detendrán en vez de dejarlas terminar`, with singular/plural agreement
  inside the branches and feminine pronouns matching *sesión*. Removing either half of the
  warning is caught by a control.
- `config.toggle_off`'s warning (`las llamadas se rechazarán hasta que se vuelva a activar`) is
  exactly as strong as en — a weaker rendering here would have been a blocker.
- `aida.next_checkin` / `aida.extras_used`: `contacto`/`contactos`, matching the committed
  `Hora de contacto diario` / `Contactos extra / día`.
- `signin.finished_hint`: `¿Terminaste en el navegador?` — tú, question marks correct on a whole
  question, `·` preserved.
- `notifications.sidebar_hint`: `muestra u oculta` — the `o→u` change is correct Spanish.
- `config.effect_new` / `needs_relaunch`: verb agreement moved inside the ICU branches, `/new`
  and plain `con /new` preserved.

## Limitations

LLM judgement, no native attestation. Widths are measured with the repo's own ruler
(`rich.cells.cell_len`) against the ceilings the source itself documents; **no rendered frame
was captured**, so a wrap is predicted from the pinned arithmetic, not photographed. No
catalogue, sidecar, ledger or allowance was edited.

## Ledger row — prepared, NOT written

Instruction for this audit was *no ledger writes*, so nothing was appended to
`i18n/ledger/es.json` (nor to the aggregate). The row below is what the ledger owner
should append, per the retention rule that the note stays ≤280 characters and points at
the tracked record:

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
    "result": "changes_requested",
    "findings": 8
  },
  "status": "changes_requested",
  "note": "changes_requested (0 blocker/3 major/2 minor/3 nit). Evidence i18n/audits/evidence/es/tui.app/w6a1/record.json; es sha256 c735760d…, en 7e5f39df…; independent auditor, model radient/auto, 2026-10-30. Majors: toggle agreement, settled heading, three /help rows over 74 cells."
}
```

Note length: 274 characters. `translator` is left as a placeholder on purpose:
that block belongs to the wave's translation record, and inventing it here would be a
false provenance claim.
