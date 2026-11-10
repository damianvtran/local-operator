# es · cli.permissions — independent re-audit r3-4186 (round 3)

- Audit date: 2026-11-09 · Worktree head `85fb35811f9d17fb6dfadfc2d5df3269db04bd12` (PR #3691)
- Auditor: i18n-auditor (team lopdev), session `41862228e657`, model `anthropic/claude-opus-5-5`
  (from this session's system-prompt `Model:` line; no fallback observed).
- Independent of the round-3 translator (commit `85fb3581` "fix(i18n): retranslate
  cli.permissions for review round 2"; session id not disclosed to the auditor) and of
  the prior auditors (r1 `f8ab0a45e07a`, r2 `9c0ff651f525`).
- Inputs: en `58e58c3b…`, context `6f8427bf…`, es `79015a98…` (full hashes in record.json).
- Scope (round 2+ rule): the two round-2 findings (r2-9c0f) and the keys changed in
  `git diff 788cc399..85fb3581` — `status.screen.denied`, `status.scope`,
  `setup.explain`, `setup.already_signed`, new `setup.always_allow_kept`. The other
  34 keys are unchanged and carried from r2-9c0f, not re-audited.
- Execution evidence: `audit.log` (validator, hashes, parity/placeholder script,
  Apple loctable lookups on macOS 27.2 build 26B123, including the two lookups that
  initially missed and how they were re-run).

## Verdict: passed — 0 BLOCKER · 0 MAJOR · 1 MINOR · 0 NIT

Mechanical: `check.py --catalogues-only` 0 problems; 39/39 keys, same order;
placeholders, backticks, line breaks and trailing spaces identical to en.

## Round-2 findings

| ID | Status | Evidence |
|----|--------|----------|
| R2-1 MAJOR `status.screen.denied` pane label | fixed | now `… > Grabación de pantalla y del audio del sistema.` = Apple es SCREENANDAUDIOCAPTURE / SCREEN_CAPTURE_HEADER (SecurityPrivacyExtension Localizable.loctable) |
| R2-2 MINOR `setup.explain` "clave privada de esa clave" | fixed | now `selecciona su clave privada`; antecedent is "esta clave de Developer ID" in the previous sentence; tú register means `su` cannot be read as "your" |

## Apple labels in changed keys (macOS 27.2, 26B123)

| en label | catalogue es | Apple es (table · key) | result |
|---|---|---|---|
| System Settings | Ajustes del Sistema | System Settings InfoPlist.loctable · CFBundleName | match |
| Privacy & Security | Privacidad y seguridad | (r2 lookup, unchanged) | match |
| Screen & System Audio Recording | Grabación de pantalla y del audio del sistema | SecurityPrivacyExtension Localizable · SCREENANDAUDIOCAPTURE | match |
| Files & Folders | Archivos y carpetas | SecurityPrivacyExtension Localizable · FILE_ACCESS_COMBINED | match |
| Always Allow / Allow | Permitir siempre / Permitir | SecurityAgent ConfirmAccess.loctable · 100267/100268.title | match |
| Keychain Access | Acceso a Llaveros | Keychain Access InfoPlist · CFBundleName | match |

Accessibility pane: the es catalogue no longer names it anywhere (Apple es on 27.2 is
`Control de dispositivos y acceso a datos`, key ACCESSIBILITY), matching the en delta.

## Changed/new keys — accuracy

- `setup.already_signed`: "…ya estaba firmado, así que no se ha firmado nada ahora y
  macOS no tenía nada que preguntar." — exact; the Always-Allow sentence is gone as the
  sidecar requires.
- `setup.always_allow_kept` (new): "Se conserva tu respuesta anterior de “Permitir
  siempre” para esta clave." — exact meaning, Apple label, “ ” quotes, tú; same wording
  as the r2-approved sentence it was split out of.
- `status.scope`: meaning exact; see R3-1.

## Findings

**R3-1 — MINOR — `status.scope` — doubled "y" blurs the list.**
Current: `Solo se comprueba la grabación de pantalla. Archivos y carpetas y los demás servicios de privacidad usan la misma fila.`
The en "Files & Folders and the other…" separates the label with `&`; in es the label
itself contains `y`, so "Archivos y carpetas y los demás" reads as a three-item run and
the pane name is hard to pick out. Not a meaning error. Suggested:
`Solo se comprueba la grabación de pantalla. Archivos y carpetas, así como los demás servicios de privacidad, usan la misma fila.`

## Not findings (checked)

- es_419 macOS shows `Configuración del Sistema` and `Grabación de audio del sistema y
  pantalla`; the style guide fixes es (Spain) labels — the r2 policy note stands.
- No plurals, no size hints; CLI output wraps, so it cannot overflow.
