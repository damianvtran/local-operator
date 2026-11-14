# es · cli.permissions — independent audit r1-f8ab

- Audit date: 2026-11-09 · Worktree head `50badda034d755ebb049c4f542f3073439bf823e` (PR #3691)
- Auditor: i18n-auditor (team lopdev), session `f8ab0a45e07a`, model `anthropic/claude-opus-5-5`
  (from the session's own system-prompt `Model:` line; no fallback observed).
- Translator (not me): PR coder agent, job `ad99821ab9e1`, `anthropic/claude-opus-5-5`.
  Same model family; no cross-model pass.
- Scope: all 28 keys, read in full against en + context sidecar, es style guide, glossary.
- Execution evidence: `audit.log` (validator, hashes, parity script, precedent counts).

## Verdict: passed — 0 BLOCKER · 0 MAJOR · 4 MINOR · 0 NIT

Safety/permission accuracy checked line by line: the macOS path
`Ajustes del Sistema > Privacidad y seguridad > Grabación de pantalla` uses the
Apple es names (Ventura+, matching the en source's "Screen Recording"); the
keychain button `Permitir siempre` is the Apple es label for "Always Allow";
`¿Elegiste «Permitir siempre» (o no apareció ningún cuadro de diálogo)?` keeps
the either/or and polarity; `Cancelado. No se ha cambiado nada.`,
`La firma no terminó. Los runtimes conservan la fila …` and the `done` restart
caveat all say exactly what the en says. `[y/N]`, trailing spaces, all
placeholders, backtick commands, `codesign`, `guide://permissions`,
`Developer ID Application`, and both System Settings row names are byte-intact.
No plurals in the namespace. Tú register and infinitive help mood match the
style guide and sibling `cli.*_help` catalogues. No `maxCells` hints; CLI lines
wrap, no fit risk.

## Findings

**M1 — MINOR — quote style (7 keys).** `status.identity.team`,
`status.identity.adhoc`, `status.screen.denied`, `setup.no_identity`,
`setup.explain`, `setup.failed`, `setup.always_allow` use « » (10 pairs). The es
style guide (binding) says `“ ”`, angled only for an editorial register; the
shipped es tree has 32 `“` vs 1 `«`. RAE-valid and meaning-safe, hence MINOR.
Suggested: replace each «…» with “…”, e.g. `Activa “{row}” en Ajustes del Sistema > …`,
`Elige “Permitir siempre” para que …`.

**M2 — MINOR — "desktop app" term drift (8 keys).** This file says
`app de escritorio` / `App de escritorio`; every other es catalogue says
`aplicación de escritorio` (6 hits, 4 files; 0 hits of the short form outside
this file). Keys: `help.setup`, `status.identity.team`, `status.optin.off`,
`status.app.found`, `status.app.missing`, `setup.no_app`, `setup.explain`,
`setup.done`. Suggested: `aplicación de escritorio` throughout, e.g.
`Aplicación de escritorio: no instalada`,
`La aplicación de escritorio de Local Operator no está instalada, así que no hay ningún permiso de la aplicación que compartir.`

**M3 — MINOR — `status.identity.legacy` "build".** Current:
`… así que cada build de Python tiene su propia fila`. `build` is not on the es
kept-English list and the shipped tree renders it `compilación` (27 hits).
Suggested: `… así que cada compilación de Python tiene su propia fila`.

**M4 — MINOR — `setup.refused.isolated` referent.** Current:
`La firma de equipo no está permitida en este proceso ({reason}). Ejecútala desde tu propio shell de inicio de sesión.`
The feminine clitic binds to *la firma de equipo* ("run the signing"), an odd
collocation; en's "it" is the command (`permissions_cli.py:171` prints this
from `lop permissions setup`). Suggested:
`… Ejecuta el comando desde tu propio shell de inicio de sesión.`

## Not findings (checked)

- `permitida` for TCC "granted": matches macOS's own "permitir" framing.
- `runtime(s)`, `bundle` kept English: `runtime` is on the es kept list; `bundle`
  is the macOS term of art.
- `Firma de equipo` / `equipo {team}`: glossary team → equipo; here it is the
  Apple Developer team and the `{team}` id disambiguates.
- macOS 15 renamed the pane "Grabación de pantalla y audio del sistema"; the en
  source names "Screen Recording", so the es mirrors source — an en-side concern, not a translation defect.
