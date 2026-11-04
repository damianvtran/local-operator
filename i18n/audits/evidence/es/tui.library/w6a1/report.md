# Independent audit — es `tui.library` (audit id w6a1)

Auditor: translation-auditor (lopdev team), independent of the translator.
I authored none of this copy, repaired none of it, and edited nothing outside
`i18n/audits/evidence/es/`. Model actually running: **radient/auto** (the
session header is `anthropic/claude-sonnet-5-5`; provider quota failures
dropped this run through `openai/gpt-6-astra` and `kimi/k3.1-pro` first).

- English source: `local_operator/i18n/catalogues/en/tui.library.json` (108 keys, sha256 `5a389b6a7d8de06621462f2685eff6dc95cd4cd6584d35dd6c26a0a19ad50b11`)
- Context sidecar: **none** — this namespace has no `en/tui.library.context.json`. Taken as given (the brief records the absence); the translator's own note says register was judged against shipped es TUI copy, and that is what I checked against, since there are no per-key budgets to consult.
- Target: `local_operator/i18n/catalogues/es/tui.library.json` (108 keys, sha256 `4b3e5982fdf26c71896579709f968dd25508f9fa48a1f942c0f616c0f4acbc62`, UNTRACKED at HEAD `b925c2e`)

## Verdict

**passed** — 0 blocker, 0 major, 1 minor, 1 nit. Every key carries the
English meaning (including the three prompt-shaped strings that go to the
agent, which are the ones that would do real damage if mistranslated), the
Librarian persona keeps its shipped capitalisation, and the asset vocabulary is
byte-identical to shipped copy where it overlaps.

## What I read

All 108 keys, twice: once beside the English source, once as Spanish-only text
in file order. With no sidecar, the second pass carried more weight here — it is
the only way to notice a line that is faithful to the source but stumbles on its
own.

## Mechanical floor

`checks.py` (this bundle): key parity and order, ICU arguments, es plural
categories (`one`/`many`/`other`), machine tokens, edge whitespace/newlines,
settled terminology and typography, English residue — **all eight pass**, plus a
ninth that **executes** the Spanish plural selection through the shipped
renderer at 0/1/2/1.000.000 (`1 recurso`, `2 recursos`, `1.000.000 de recursos`);
the repo validator reports `0 problem(s)`.

Two honest notes about this namespace:

- **No `maxCells` is declared anywhere here** (no sidecar), so the budget check
  has nothing to compare against and reports zero by construction. Fit was
  therefore judged against the render paths, not against a number: see "Fit"
  below.
- Two negative controls fall back to synthetic fixtures on this namespace
  because it carries no backtick span and no load-bearing edge space
  (`audit.log` says so per control). The controls still prove the checks can
  fail; they just cannot use a real value for those two classes here.

The plural ICU here is **byte-identical to shipped `relay.library.count`** —
`{count, plural, one {# recurso} many {# de recursos} other {# recursos}}` — which
is the strongest possible consistency evidence for `asset → recurso`.

## Fit (judged against the render path, since no budgets exist)

The library view was read to see what each string is drawn into:

- **Filter rail** (24 cells, `truncate_cells` on every label): `Hojas de cálculo`
  is the longest category at 16 cells and fits the label column (≥18 cells at
  realistic counts); `categorías` (10) matches en's `categories` (10) so the
  header does not grow; `etiquetas`/`estado` are well inside it.
- **Status line**: `.library-view-status` is `height: 2; padding: 1 0 0 0` — one
  content row, `1fr`. Anything longer than the content width is lost. This is
  where the one minor finding comes from (`rail_hidden`, 76 cells, painted only
  when the rail is hidden, i.e. below 90 columns).
- **Empty states**: `_paint_message` wraps its body with `textwrap.wrap` against
  the measured list width, so `empty_body` (81 cells) and `no_matches_body` (58)
  are not clipping risks.
- `search_placeholder` keeps en's **two spaces** before `/` (`buscar  /`); the
  whitespace control confirms it rather than complaining about it.

## The prompt-shaped strings

Three values are copied into the composer and sent to the agent, so a bad
translation changes what the model is asked, not just what the reader sees:

- `ask_prompt`: `Háblame de este recurso: {name} ({path}) — {kind}, de {session}. ¿Qué es y cómo podría usarlo?` — a faithful, natural rendering of *Tell me about this asset … What is it, and how might I use it?*; the path stays in its own parentheses and `{session}` keeps its place.
- `ask_prompt_digest`: `{name} (recurso {id})` — `asset` inside the identifier parenthesis became `recurso`, matching the noun; the id token is untouched.
- `ask_seeded`: `El compositor tiene lista una pregunta sobre {name}.` — `tener lista una pregunta` is idiomatic Spanish (`tener algo listo`), not a calque of "is ready with".

## Terminology against shipped precedent

| Term | This file | Shipped precedent | Verdict |
|---|---|---|---|
| `asset` | `recurso` | `relay.library`, `cli.sessions_help`, `wire.slash.library` | correct; the ICU is byte-identical to `relay.library.count` |
| `the Librarian` (prose) | `el Bibliotecario` | `wire.slash.library.librarian_help` (`Abre, pausa o ejecuta el Bibliotecario`), `wire.errors.desktop` | correct — the persona keeps its capital |
| `librarian: {state}` (row label) | `bibliotecario: {state}` | en is lower-case here too; `hint_librarian` is `bibliotecario` | correct — the case split mirrors the source's own |
| `preview` | `vista previa` | `tui.viewer`, `cli.agents_help`, `wire.remote_compute` | correct, shipped term |
| `tags` | `etiquetas` | `tui.tags`, `wire.settings` | correct |
| `Video` | `Vídeo` | `wire.settings` (`vídeo`) | correct — matches the only shipped spelling; `wire.errors.files` uses unaccented `video`, a pre-existing shipped split this file does not widen |
| `archive`/`archived` | `sin archivar` | — | correct reading of the state |
| `library` (title) | `biblioteca` (lowercase) | `wire.slash.library` capitalises it as a surface name inside a sentence | correct here: en is lower-case `library` in this row and es sentence case agrees |
| `file` | `archivo` | style guide (never `fichero`) | correct |
| `folder` vs `working directory` | no `carpeta` appears at all | — | correct: nothing here is a session cwd, so the `carpeta` / `directorio de trabajo` split is not engaged |

## Findings

**F1 — minor — `tui.library.rail_hidden`**
Current `"Los filtros aparecen cuando el terminal tiene al menos {width} columnas de ancho."`
→ **76 cells** at `{width}` = 90 (en: 61). This string is painted into
`.library-view-status`, a **one-content-row** Static, and it is painted
**exactly when the rail is hidden — i.e. below 90 columns**. So on a 61–75
column terminal the Spanish notice is cut (at 72 columns the reader loses
`columnas de ancho`), while the English still fits. Spanish is legitimately
~20% longer; this particular line has to survive the narrow regime that
triggers it. Proposed `"Los filtros aparecen a partir de {width} columnas."`
— **50 cells**, same claim, no actor, no `de ancho`.

**F2 — nit — `tui.library.form_name_placeholder`**
Current `"si no, el nombre del archivo"` for *falls back to the file name*. In a
field placeholder this reads as a fragment beginning mid-thought ("otherwise,
the file name"); a first-time reader has to infer that the *empty* field falls
back. Proposed `"si lo dejas vacío, se usa el nombre del archivo"` (47 cells,
no budget) or `"vacío = nombre del archivo"` if the field must stay short.

## Checked and accepted (no finding)

- `hint_preview` `vista previa` is a noun among infinitives (`aplicar`,
  `preguntar`, `volver`…). en's `preview` is equally ambiguous, `vista previa`
  is the shipped term in `tui.viewer`, and `previsualizar` would be longer and
  less familiar in a key hint. Accepted.
- `missing` → `falta` and `render`-side `[falta]`: consistent with the state
  vocabulary (`Sin organizar`), and it never reads as a zero.
- `sort_label` `ordenar: {sort}` — infinitive label beside a value; `ordenado
  por:` would be longer and the row is a control, not a readout.
- `librarian_no_launcher` is 86 cells (en 69), but it is delivered through
  `_system_notice`, which wraps itself when built — not a clipping surface.
- `form_saved` `Guardado {name}.` — terse past participle, per the style guide's
  "Guardado, not La sesión se ha guardado correctamente".
- `notice_opened` `Abierto {name}.` / `form_saving` `Guardando…` — same rule.

## Limitations

- No sidecar means no per-key role or budget: fit judgements rest on the render
  paths I read (`library_view.py`, `local_operator.tcss`), which is
  reconstruction from code rather than a rendered frame.
- No rendered TUI frames captured (loaded host, heavy runs forbidden); the
  rail/status measurements are `rich.cells.cell_len` on the catalogue strings,
  not a screenshot.
- LLM semantic judgement, no native-speaker attestation.
- The target file is untracked at `b925c2e`; this verdict binds the sha256 in
  `record.json`, not a commit.
