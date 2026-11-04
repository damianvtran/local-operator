# Independent audit — es `tui.small` (audit id w6a1)

Auditor: translation-auditor (lopdev team), independent of the translator.
I authored none of this copy, repaired none of it, and edited nothing outside
`i18n/audits/evidence/es/`. Model actually running: **radient/auto** (the
session header is `anthropic/claude-sonnet-5-5`; provider quota failures
dropped this run through `openai/gpt-6-astra` and `kimi/k3.1-pro` first).

- English source: `local_operator/i18n/catalogues/en/tui.small.json` (137 keys, sha256 `16d49c4ba2a514bbde7da494443abade26a3b28c0389e6bca124d00aa7d0ed75`)
- Context sidecar: `.../en/tui.small.context.json` (137 entries — every key has a role, tone and note)
- Target: `local_operator/i18n/catalogues/es/tui.small.json` (137 keys, sha256 `c48357ff20fcacfe996b687a2d035645639a8a6621d79ecf6b1436ca763ed145`, UNTRACKED at HEAD `b925c2e`)

## Verdict

**passed** — 0 blocker, 0 major, 2 minor, 3 nit. Nothing in this namespace
misstates a charge, weakens a destructive hint, or breaks a placeholder.
`checks.py` is clean on all eight mechanical checks and every negative control
fires (see `audit.log`).

## What I read

All 137 keys, twice.

1. **Against the source.** Each es value beside its en value and its sidecar
   entry (role + note), in file order. The sidecar is unusually explicit in this
   namespace — it names the load-bearing trailing space, the deliberate absence
   of plural branches in `copy.*`, the raw-digit contract of `{shown}`, and the
   design rounds behind each spend rung — so "did the translator respect the
   constraint" was answerable per key rather than by taste.
2. **Spanish only, no English in view.** The message values alone in file order,
   looking for what stumbles in Spanish. This pass produced two of the five
   findings below (`bajo profundidad`, and it confirmed the money ladder reads
   as four distinct states without the source in front of me).

## Mechanical floor

`checks.py` (this bundle) re-derives on the audited bytes: key parity and order,
ICU argument parity by name and kind, es plural categories exactly
`one`/`many`/`other`, machine tokens (backtick spans, `<angle>` tokens, arrows,
`ctrl`/`shift`/`enter`/`esc` combos, `$` amounts, `/slash` commands, `·`
separators), load-bearing edge whitespace and newlines, `maxCells`, settled
terminology and es typography, and English residue — plus a ninth check that
**executes** the Spanish plural selection through the shipped renderer
(`local_operator.i18n.messages.render` and the project's own CLDR data) at 0, 1,
2 and 1.000.000: `0 miembros`, `1 miembro`, `2 miembros`, `1.000.000 de
miembros`. All nine pass. The repo's
own catalogue validator (`python3 scripts/i18n/check.py --catalogues-only`)
reports `0 problem(s)` across the whole catalogue tree, which is the same parity
floor from the other direction.

Sixteen `maxCells` budgets are declared, **all sixteen on the spend ladder**.
**None is exceeded in es**, including the nine-cell `spend.no_charge` (9/9) and
the 30-cell `spend.unpriced_cancelled` (30/30, one cell longer than en's 29 —
measured on `{width}`-free text with `rich.cells.cell_len`, the renderer's own
measure, not `len()`).

## Money and spend (`tui.small.spend.*`) — the highest-stakes block

The four states the English distinguishes are still four distinct states in
Spanish, and the mapping is systematic rather than ad hoc: **charge → `cargo`**,
**billed → `cobro/cobrado`**, **price → `precio`**. Nothing collapses.

| State | English | Spanish | Read |
|---|---|---|---|
| failed (presumed unbilled) | `no charge` / `not billed` | `sin cargo` | correct — never reads as a measured zero |
| mixed (stated amount + failed attempt) | `failed attempt added no charge` / `failed attempt: +$0` | `intento fallido: sin cargo` / `intento fallido: +$0` | correct — the colon keeps it an increment, not a whole-row claim (sidecar D-1) |
| cancelled (may be billed) | `response lost · may be billed` / `may have been billed` / `bill unknown` | `sin respuesta · puede cobrarse` / `pudo haberse cobrado` / `cobro dudoso` | correct — never says "sin cargo", so the possible charge survives every rung |
| served but unpriced | `server stated no price` / `no price stated` | `sin precio del servidor` / `sin precio` | correct — distinct from both `cargo` and `cobro` |
| partial (figure is a lower bound) | `some requests unpriced` / `some unpriced` / `some unknown` | `peticiones sin precio` / `precio parcial` / `incompleto` | correct in substance; see F2 for the lost quantifier |

Calques of "billed" (`facturado`) are absent, which is the right call: on this
panel the user is looking at what they were charged, not at an invoice.

## Destructive and permission copy

- `approval.allow_verb` `permitir ` and `approval.decision_allowed/denied`
  `permitido `/`denegado ` keep the **load-bearing trailing space** (verified
  byte-for-byte, and the whitespace control fires when it is trimmed).
- `settings_view.reorder_hint` / `hop_row_hint` / `chain_row_hint` say `eliminar`
  and `la elimina` — the object is explicit in the chain hint, so `d` cannot be
  read as anything but deletion.
- `todo_panel.dropped` / `dropped_row` → `descartadas` / `descartada`, feminine
  and consistent with `Tareas`; `blocked` → `bloqueada`. The panel's genders
  agree throughout (`resueltas`, `hechas`, `descartadas`).
- `session_picker.all_archived` rewrites en's `all {total} conversation(s) are
  archived` as `no queda ninguna conversación sin archivar ({total})`. Same
  claim, and it dodges the `(s)` hedge without inventing a count form.

## Terminology against shipped precedent

| Term | This file | Shipped precedent | Verdict |
|---|---|---|---|
| `link` (hyperlink picker) | `enlace` | `tui.review`, `wire.slash`, `wire.settings` | correct — the guide reserves `vínculo` for a configured hub/agent/team link |
| `relay` | `relé` | glossary | correct |
| `wake` / `monitors` | `despertares` / `monitores` | glossary; `tui.app`, `tui.schedule` | correct, and `monitorización` is not used for the monitor object |
| `heartbeat` | `heartbeat` | `cli.browser_help`, `cli.browser` (already English, both shipped) | consistent, not a new split |
| `settings` | `ajustes` | style guide | correct |
| `team` (org chart) | `equipo` | `tui.team_authoring` | correct |
| `manager` (org chart legend) | `responsable` | `tui.team_authoring.manager`, `wire.slash.team` | correct |
| `org chart` | `organigrama` | `wire.slash.team` | correct |
| `off the record` (aside) | `sin registro` | `wire.slash.btw` (`… sin registro …`) | correct — matches shipped wording exactly |
| `Aside` (panel title) | `Aparte` | (no shipped panel title; `wire.slash.btw` says `al margen` adverbially) | accepted: `un aparte` is the Spanish noun, and `tui.library.ask_float` uses the same word |
| `todos` | `Tareas` | shipped `cli.global_help`, `relay.transcript` use `tareas` for work items | correct and internally consistent (`resueltas`, `hechas`, `descartadas` all agree) |
| `portapapeles` | `al portapapeles` | `tui.app`, `tui.welcome` | correct gender |

## Findings

**F1 — minor — `tui.small.info_panel.deeper_below_depth`**
Current `"  +{deeper} más abajo (bajo profundidad {depth})"`. `bajo` + a bare
noun without an article is not Spanish; it reads as a calque of *below depth N*.
Proposed `"  +{deeper} más abajo (bajo la profundidad {depth})"` (51 cells, no
declared budget) or the tighter `"… (por debajo de {depth})"` (45).

**F2 — minor — `tui.small.spend.unpriced_partial`**
Current `"peticiones sin precio"` for *some requests unpriced*. The quantifier is
the whole point of this rung: the sidecar says the shown figure is a **lower
bound**. Without `algunas` the note reads as "requests [are] unpriced", which a
reader can take as the whole row. Proposed `"algunas sin precio"` (18 cells,
within the 24 budget) — the row itself supplies `peticiones`. The narrower rungs
do keep the partiality (`precio parcial`, `incompleto`), so this is the odd rung,
not a systematic choice.

**F3 — nit — `tui.small.network_panel.suspect_tag`**
Current `"  [sospechoso: puede que la clave se haya copiado]"` (50 cells vs en
30). The network panel's own docstring measures frames at 56x20 and 60x20; en
wraps this marker to one continuation line there, es to two. Proposed
`"  [sospechoso: clave quizá copiada]"` (35) keeps every element.

**F4 — nit — `tui.small.copy.code_count`**
Current `"{shown} código"` renders `3 código`. This is a **constraint, not an
oversight**: the sidecar requires no plural branch, and the repo's argument
parity refuses a `simple → plural` upgrade (verified: `argument 'shown' changed
kind 'simple' -> 'plural'`), so the translator had no legal ICU option and
mirrored the English compromise (`3 code`). Recorded so that a future change to
either the key or the parity rule is a deliberate one.

**F5 — nit — `tui.small.goal_panel.more_settled` (+ `more_settled_indent`)**
Current `"… {hidden} finalizados más"`. Understandable, but the trailing `más`
is the less natural order; `"… {hidden} más finalizados"` is the same 26 cells.

## Checked and accepted (no finding)

- `spend.unpriced_cancelled_shortest` `cobro dudoso` (12/12) — the tightest
  wording available under a 12-cell cap that must still say the amount is
  unknown; `cobro incierto` (14) does not fit and a measured zero would be worse.
- `spend.unpriced_cancelled` `sin respuesta` for *response lost* — the wide rung
  says `respuesta perdida`; at 30 cells the shorter reading loses no money claim,
  and the rung still says `puede cobrarse`.
- `spend.channel_not_itemised` `sin desglose por canal` — adds `por canal`, which
  the sidecar states is exactly what "not itemised" means here.
- `aside.superseded` `(sustituida)` — feminine agrees with `respuesta`; a reader
  who takes `aparte` as the antecedent would want `sustituido`. Both defensible
  in a bare fragment; no change requested.
- `session.resume_hint` capitalises `Sesión finalizada` at line start — sentence
  case, not Title Case; the style guide allows a fragment to stay lowercase but
  a line-initial capital is standard.
- `network_panel` column alignment: the label column stays at cell 14 in all
  four `label: value` rows even though `relé` is one cell shorter than `relay`,
  and the `checking…` row keeps en's own single space. Alignment was measured,
  not eyeballed.
- `picker.showing_prefix`, `subagents.of_scope`, `copy_picker.from_line`,
  `todo_panel.*suffix`: leading/trailing spaces and `·` separators byte-identical
  to en (whitespace control green).

## Limitations

- LLM semantic judgement, no native-speaker attestation. The team's gate is this
  independent audit, not a native review.
- No rendered TUI frames were captured: this host is loaded and heavy runs are
  forbidden. Widths were measured with `rich.cells.cell_len`; the two budget
  rows at exactly 100% (`spend.no_charge` 9/9, `spend.unpriced_cancelled` 30/30)
  are at the cap but not over it.
- `tui.small` has no WIDGET-LEVEL render capture in this audit (no Textual app was
  driven), so what is proved by execution is the ICU layer, not the panel: C9
  renders every plural message through the shipped renderer and the operator's
  own locale data at 0/1/2/1.000.000, which is where a wrong branch would show.
  Layout is measured, not rendered.
- The target file is untracked at `b925c2e`; this verdict binds the sha256 in
  `record.json`, not a commit.
