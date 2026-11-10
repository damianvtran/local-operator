# es · cli.permissions — independent re-audit r2-9c0f (round 2)

- Audit date: 2026-11-09 · Worktree head `788cc399410196b080cfc9f7e7bd558ccb713140` (PR #3691)
- Auditor: i18n-auditor (team lopdev), session `9c0ff651f525`, model `anthropic/claude-opus-5-5`
  (from this session's system-prompt `Model:` line; no fallback observed).
- Independent of the round-2 translator (wave i18n-translator, commit `788cc399`; its
  session id was not disclosed to the auditor) and of the round-1 auditor (`f8ab0a45e07a`).
- Scope: all 38 keys (28 carried + 10 new) read in full against en (`d5eed404…`),
  context sidecar (`f2d9ce73…`), es style guide and glossary.
- Execution evidence: `audit.log` (validator, hashes, parity script, precedent counts,
  Apple loctable lookups on macOS 27.2 build 26B123); lookup script `apple_labels.py.txt`.

## Verdict: changes_requested — 0 BLOCKER · 1 MAJOR · 1 MINOR · 0 NIT

## Round-1 findings

| ID | Round-1 finding | State on 788cc399 |
|----|-----------------|-------------------|
| M1 | « » vs guide “ ” | Resolved: 0 « », 16 “, no straight quotes in values |
| M2 | app → aplicación de escritorio | Resolved: 0 `app de escritorio`, 8 `aplicación de escritorio` |
| M3 | build → compilación | Resolved: 0 `build`; legacy/explain/already_signed/done_allow_once use compilación(es) |
| M4 | Ejecútala referent | Resolved: `Ejecuta el comando desde tu propio shell de inicio de sesión.` |

## macOS labels checked against Apple's installed es strings

| en label | Apple es (table · key) | es catalogue |
|---|---|---|
| System Settings | `Ajustes del Sistema` (System Settings.app InfoPlist.loctable · CFBundleDisplayName) | match |
| Privacy & Security | `Privacidad y seguridad` (System Settings.app Localizable.loctable · `Privacy & Security`) | match |
| Screen & System Audio Recording | `Grabación de pantalla y del audio del sistema` (SecurityPrivacyExtension.appex Localizable.loctable · SCREENANDAUDIOCAPTURE and SCREEN_CAPTURE_HEADER) | **mismatch — R2-1** |
| Screen Recording (service) | `Grabación de pantalla` (same table · SCREEN_CAPTURE) | match |
| Files & Folders | `Archivos y carpetas` (same table · FILE_ACCESS_COMBINED) | match |
| Accessibility | `Accesibilidad` (System Settings.app Localizable.loctable · `Accessibility`) | match |
| Allow / Always Allow | `Permitir` / `Permitir siempre` (Security.framework SecurityAgent.bundle ConfirmAccess.loctable · 100268.title / 100267.title) | match |
| Keychain Access | `Acceso a Llaveros` (CoreServices/Applications/Keychain Access.app InfoPlist.loctable · CFBundleName) | match |
| Access Control | `Control de acceso` (Keychain Access.app KeyEditor.loctable · 302.label) | match |

## Findings

**R2-1 — MAJOR — `status.screen.denied` — pane name not Apple's es label.**
Current: `… Activa “{row}” en Ajustes del Sistema > Privacidad y seguridad > Grabación de pantalla y audio del sistema.`
Apple es shows `Grabación de pantalla y del audio del sistema` both as the list row
(SCREENANDAUDIOCAPTURE) and the pane header (SCREEN_CAPTURE_HEADER). This line is the
one place the user is told where to click, and the sidecar requires the label exactly as
localized macOS shows it; the catalogue's form is Apple's support-doc prose title, not the
UI label. Suggested:
`Grabación de pantalla (comprobación del runtime): no permitida. Activa “{row}” en Ajustes del Sistema > Privacidad y seguridad > Grabación de pantalla y del audio del sistema.`

**R2-2 — MINOR — `setup.explain` — "la clave privada de esa clave".**
Current: `Para deshacerlo, abre Acceso a Llaveros, selecciona la clave privada de esa clave y quita codesign en Control de acceso.`
"clave … de esa clave" is a literal echo of "the key's private key" and makes the undo
step — the one instruction that revokes the Always Allow grant — read clumsily.
Suggested: `Para deshacerlo, abre Acceso a Llaveros, selecciona su clave privada y quita codesign en Control de acceso.`

## Safety/accuracy checks that pass (Always Allow disclosure)

- `setup.explain` keeps both "- " bullets and line breaks; “Permitir” = this time only;
  later terminal `lop update` asks again; terminal-less updates (agent-run, `/update`)
  skip team signing and new compilations use “Local Operator Runtime”. “Permitir siempre”
  = no more questions **and** any program running as the user can sign with this
  Developer ID key without asking — scope and consequence intact, no softening.
- `setup.always_allow` keeps the either/or and the "no dialog" polarity; `[y/N] ` and
  trailing space intact. `setup.already_signed`, `setup.done_allow_once`,
  `setup.failed` ("cualquier configuración anterior sigue sin cambios"),
  `setup.cancelled` say exactly what en says.
- New status keys (`optin.no_app`, `optin.unavailable`, `reason.*`, `screen.no_image`,
  `status.scope`) accurate; env-var names, `~/.local-operator`, `{reason}` intact;
  `está definida` agrees with the implied *variable*.

## Not findings (checked)

- es_419 macOS shows `Configuración del Sistema` and `Grabación de audio del sistema y
  pantalla`; the es style guide standardises on `ajustes`, i.e. Apple es (Spain) labels.
  A neutral-Spanish policy question for the locale owner, not a translation defect.
- `un terminal` (3×): masculine matches sibling CLI `cli.exec_help` and RAE accepts both
  genders; the wider tree leans `la terminal` (5) — not gated.
- `editor` for "publisher", `carpeta de inicio` for "login home", `llavero`: correct.
- `runtime`, `bundle`, `Developer ID (Application)`, `codesign` kept per guide/sidecar.
- No plurals, no `maxCells` hints; CLI output wraps — no fit risk.
