# Style guide — Spanish (`es`) — neutral Spanish

Read with `i18n/glossary.md`; everything here is es-specific and binding for
translators and the auditor. Wave 1 locale.

## Formality & tone

- Address: **tú** — recorded decision: modern dev/consumer convention (Apple
  es, Google es both address as tú): "tu sesión", "puedes". "usted" reads
  enterprise/customer-service and is avoided; "vosotros" is never used.
- Neutral Spanish: avoid region-marked words — use "archivo" (not "fichero");
  for a phone, "teléfono" reads everywhere ("celular" and "móvil" each mark a
  region) — prefer wording that reads the same in Madrid and Mexico City.
- Action labels: infinitive — "Guardar", "Reanudar", "Cancelar" (standard es
  UI convention).
- Terse: "Guardado", not "La sesión se ha guardado correctamente". No "!".
- Sentence case; fragments may stay lowercase ("guardado automático"); never
  Title Case.

## Product-name policy

- Keep exactly as in glossary §1: "Local Operator", `lop`, tool names,
  `--flags`, config keys, paths, identifiers.
- Kept-English technical list: commit, push, pull, merge, rebase, token,
  endpoint, script, prompt, deploy · API, CLI, TUI, JSON, YAML, HTTP,
  URL, GPU. Wave-1 recorded additions: runtime, harness, hook(s), handshake,
  backoff. Reason: settled dev usage across Spain and LatAm.
- Prefer Spanish where settled: servidor, archivo, carpeta, red, dispositivo,
  ajustes (settings), equipo (team), habilidad (skill).
- **`carpeta` is a plain folder; `directorio de trabajo` is a session's cwd.**
  The two are different things on these surfaces, and the shipped settings copy
  already says `directorio de trabajo` for the composer's working directory
  (`wire.settings.display_composer_cwd.*`). Use `carpeta` only for a folder the
  user browses. (Rule recorded 2026-10-30 after the mobile relay lane's audit
  flagged `carpeta de trabajo` for a session's directory — its own wording, not
  ours, and the point stands regardless of which copy carries it.)
- **Amendment (es W1, 2026-10-29): `cache` → `caché`.** The kept-English list
  above named `cache` before this amendment removed it. The shipped catalogue
  already used `caché` in 19 keys (20 occurrences: `wire.settings` 15,
  `wire.settings.account_policy` 3, `wire.host_speed` 2), including the required
  `no garantiza un acierto de caché` in `wire.settings.account_policy` — a
  string this wave ships. The guide now agrees with the shipped state; follow
  `caché` rather than reopening the split.

## Locale calls (es)

- session → sesión (feminine: "las sesiones")
- wake → despertar ("despertares programados")
- tool trace → traza · transcript → transcripción
- approvals → aprobaciones
- schedule → programación; one scheduled item reads "prompt programado".
- usage → uso · cost → coste (LatAm "costo" is accepted; wave 1 standardised
  on **coste** throughout — recorded 2026-10-11).
- job → trabajo (a backgrounded shell job; "control de trabajos" register —
  recorded 2026-10-11).
- link → **vínculo** for a configured hub/agent/team link; **enlace** stays for
  a hyperlink or a URL. W1 recorded both words in the catalogue before this
  row existed; the rule is that a *configured link* is `vínculo`, so
  `wire.slash.links`' `enlace` is a hyperlink and correct, not a drift.
- lease → **concesión** (a provider credential lease; a DHCP-style grant, not a
  rental contract).
- skill → **habilidad**; the tight-form abbreviation `habil.` is acceptable only
  where a hard cell budget forces it (`wire.settings.hub_defaults.agents.label`
  is 29 cells and the full phrase does not fit).
- **Amendment (es W2, 2026-10-29): three more recorded decisions.**
  - `daemon` → **`daemon` / `daemons`, kept English.** The shipped catalogue
    already used it (`relay.asks`, `wire.settings`, `cli.browser_help`) and the
    borrowed form is standard in Spanish dev usage; W2 had introduced
    `demonio`/`demonios` in eleven cli.* keys and the independent audit rejected
    the split. Do not reopen it.
  - `tenant` → **`tenant`, kept English**, lowercase mid-sentence
    (`tenant de la organización`). It is a multi-tenancy term, and
    `<tenant_id>` is a machine token that never translates.
  - `starter` → **`plantilla incluida`** (shipped in
    `tui.authoring.seed_not_installed`); the plural is `plantillas incluidas`.
    W2 briefly used `perfiles iniciales`, which the audit flagged as a second
    rendering of one product term.
- **Amendment (es TUI wave, 2026-10-30): `cuota` is not `asignación`.** Two
  different money objects share this panel and must keep two names. `cuota` is
  the *quota* — a ceiling on usage (`wire.settings`, `wire.slash.usage`).
  `asignación` is the *allowance* — the paid allocation (`Asignación de pago
  ilimitada`, `Asignación incluida agotada`, shipped in `relay.shell.usage`).
  `tui.usage.paid.continue` had said `cuota incluida agotada` for the sentence
  `relay.shell` already ships as `Asignación incluida agotada`; the audit called
  it drift between two surfaces naming one object, and the shard moved to the
  shipped term. Where a new string needs "quota" in the ceiling sense, it is
  `cuota`; where it means the paid allocation, it is `asignación`.
- **Amendment (es TUI wave, 2026-10-30): `pull` / `push` stay English, and the
  action label is the infinitive form `Hacer pull ahora`** (en: `Pull now`). The
  shipped catalogues already settled this, outside this wave: `Hacer pull de un
  agente desde Radient` (`cli.agents_help`), `Hacer pull de un equipo del hub
  por ID` (`cli.teams_help`) and `hace merge … de los que hiciste pull`
  (`wire.settings`). Do not render `pull` as `extraer` or `traer` — the user pairs the
  label with the subcommand they already type.

## Plurals

- Categories: `one` (1 only — 0 is `other`: "0 sesiones"), `many` (exact
  multiples of 10⁶ only — "1.000.000 de sesiones"; 1.500.000 → `other`),
  `other`. `few` is not a Spanish category — never use it.
- Example: `{count, plural, one {# sesión} many {# de sesiones} other {# sesiones}}`
- `#` renders the formatted number; never pre-format inside the branch.

## Numbers, dates, digits

- Never hand-written — the formatter renders "1.234,5" and dates "10 oct
  2026"; 24-hour times.
- Digits: Latin (v1, all locales).
- ¿…? / ¡…! when a whole string is a question or exclamation; not on
  fragments.

## Width rules

- es runs roughly 15-25% longer than en. Tighten ("de forma automática" →
  "automáticamente" → "auto" as budgets demand); shorter synonyms over
  truncation, always.
- Budgets: `maxCells` from the en sidecar; pinned: PERSIST_HINT 42/43 cells,
  tip pool 59, `/help` ≈ 55.
- Recorded width exception: `cli.services_status` in this wave:
  `serve_linger_required` measures 103 and `unsupervised_unknown` 87 cells with
  `rich.cells.cell_len` against the 80/78 status budget. Recorded rather than
  tightened to a shorter remedy, because the alternatives lose the logout-only
  qualification or the scope name.
- Recorded width exception: `tui.usage` paid annotations (es TUI wave,
  2026-10-30). `paid.family_caps` measures 53 cells against en's 29 and
  `paid.eligibility_unknown` 35 against en's 25; both are one-line annotations
  drawn as `TIER_INDENT + note` and truncated with an ellipsis, so at the
  40×24 floor (31-cell body) they clip where English does not. Kept long on
  purpose: both follow the shipped `relay.shell.usage` family, where the same
  allowance is described on the relay surface (wording adapted to this
  catalogue's sentence case and final-period rules), and shortening them here
  would name one money object two ways. `paid.continue` composes to 54 cells
  (en 44; the pre-alignment `cuota` wording was 49), so it needs a
  **63-column** terminal to show whole and fitted from 58 before.
  `family_caps` needs **64** columns and `eligibility_unknown` **46** (37 cells
  plus 9 of chrome), all measured with `rich.cells.cell_len` against
  `body = columns − 9`: in the 46–63 band `family_caps` and `paid.continue`
  clip, and all three fit only from 64 columns up.
- Recorded width exception: `tui.analytics.hint.expand.keys_bare` (es TUI wave,
  2026-10-30). es measures 14 cells against the declared `maxCells` of 12 — but
  **English measures 13 against the same 12**, so the cap is unattainable in the
  source itself. 14 is the floor once both keys stay named (`e` may not lose its
  action word); the footer crops with `overflow="crop"` rather than wrapping.

## Punctuation & typography

- Quotes: “ ” (angled « » only if an editorial register is ever needed);
  em dash " — " with spaces; ellipsis ….
- No space before : ; ? ! (unlike French).
- Code spans, flags, paths stay byte-identical; never apply es typography
  inside them.
