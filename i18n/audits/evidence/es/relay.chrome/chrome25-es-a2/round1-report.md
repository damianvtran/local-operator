# Translation audit — `relay.chrome` · Spanish (`es`) — `chrome25-es-a1`

**Verdict: PASS** — 83/83 keys read; 0 blocker, 0 major, 2 minor, 5 nit.

## Auditor and provenance

| Field | Value |
|---|---|
| Audit ID | `chrome25-es-a1` |
| Role | `translation-auditor`, independent of the translator (no translator report or ledger entry read) |
| Session | `56f831d2290f` (subagent), parent `44f93fd45b09`, label `Audit chrome es` |
| Provider / model | `radient` / `auto`, from every assistant turn's `payload.native_replay` in `~/.local-operator/sessions/56f831d2290f/transcript.jsonl` (endpoint `https://api.radienthq.com/v1/chat/completions`) |
| Model caveat | `auto` is the provider's routing alias; the concrete upstream model is not exposed to this session and is not guessed |
| Run date | 2026-10-25 |
| Worktree | `~/local-operator/.worktrees/relay-chrome-i18n` |
| Base | `535cea359825411741069cbffcee42e2927df9d3` — `feat(i18n): freeze relay.chrome English copy for the borderless relay redesign` |
| Repo state | read-only audit; no file in the worktree was written or staged |

## Inputs and hashes (computed at audit time)

| Artifact | SHA-256 |
|---|---|
| `local_operator/i18n/catalogues/en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| `local_operator/i18n/catalogues/en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| `local_operator/i18n/catalogues/es/relay.chrome.json` (target) | `9e1b68d5e546317ff912e1aa9d463d9ae6d44d7b109e0e584ef5e82a1a667192` (5302 bytes) |

The two English hashes equal the values supplied with the task. The target hash was recomputed after the audit and is unchanged; `catalogues.catalogue_sha256("es", "relay.chrome")` returns the same value, so the loader reads the bytes verbatim.

## Scope and method

- All 83 keys read side by side with the context sidecar (role, tone, `maxCells`, and every must/keep/exactly directive), `i18n/glossary.md` (§1–§4) and `i18n/style-guides/es.md`.
- Locale corpus cross-check over `local_operator/i18n/catalogues/es/*.json`: `sesión`, `conversación`, `pregunta`, `aprobar`, `aprobaciones`, `ajustes`, `biblioteca`, `máquina`, `etiquetas`, `esfuerzo de razonamiento`, `sin conexión`, `inactiva`, `reanudar`, `envío`.
- Structural work used the repository's own ICU-subset parser/renderer (`local_operator.i18n.runtime`) and CLDR data (`local_operator.i18n.format`), not a hand-rolled parser.
- Repository validator `scripts/i18n/check.py`, plus its `check_catalogues()` in-process, filtered to `es` + `relay.chrome`.
- Every key rendered through the product entry point `local_operator.i18n.messages.render(code, params, locale="es")`.

## Verdict by key group

| Group | Keys | Verdict |
|---|---:|---|
| `relay.chrome.drawer.*` | 12 | pass with nit (`a1-04`) |
| `relay.chrome.link.*` | 3 | clean |
| `relay.chrome.list.*` | 21 | pass with minor + nit (`a1-01`, `a1-07`) |
| `relay.chrome.snackbar.*` | 1 | clean |
| `relay.chrome.row.*` | 10 | clean |
| `relay.chrome.appearance.*` | 5 | clean |
| `relay.chrome.session.*` | 25 | pass with minor + 3 nits (`a1-02`, `a1-03`, `a1-05`, `a1-06`) |
| `relay.chrome.composer.*` | 4 | clean |
| `relay.chrome.spend.*` | 1 | clean |
| `relay.chrome.model.*` | 1 | clean |
| **Total** | **83** | **PASS** |

## Findings

| ID | Severity | Key(s) | Current | Proposed |
|---|---|---|---|---|
| `chrome25-es-a1-01` | minor | `relay.chrome.list.empty_hint` | Empieza una con Nuevo, o desde Local Operator en tu máquina | Empieza una con el botón Nuevo, o desde Local Operator en tu máquina |
| `chrome25-es-a1-02` | minor | `relay.chrome.session.ended_hint` | Un envío o Reanudar la reabre en tu carpeta personal | Enviar un mensaje o Reanudar la reabre en tu carpeta personal |
| `chrome25-es-a1-03` | nit | `relay.chrome.session.degraded` | No responde — los envíos esperarán | No responde — los mensajes esperarán |
| `chrome25-es-a1-04` | nit | `relay.chrome.drawer.open`, `relay.chrome.drawer.close` | Abrir navegación / Cerrar navegación | Abrir el menú de navegación / Cerrar el menú de navegación |
| `chrome25-es-a1-05` | nit | `relay.chrome.session.back` | Volver a conversaciones | Volver a las conversaciones |
| `chrome25-es-a1-06` | nit | `relay.chrome.session.loading` | Cargando conversación… | Cargando la conversación… |
| `chrome25-es-a1-07` | nit | `relay.chrome.list.filter_empty` | No hay conversaciones con este filtro | No hay conversaciones que coincidan |

### `chrome25-es-a1-01` — minor — `relay.chrome.list.empty_hint`

- Current: `Empieza una con Nuevo, o desde Local Operator en tu máquina`
- Proposed: `Empieza una con el botón Nuevo, o desde Local Operator en tu máquina`
- Reason: `una` is feminine (conversación) while the articleless button name `Nuevo` is masculine in the same clause; naming the control removes the clash. The key has no `maxCells` budget and the sidecar says it already wraps to two lines, so `el botón ` costs no layout. `Nuevo` itself is the right button label (Spanish Drive uses it), and `en tu máquina` matches the locale's existing host wording.

### `chrome25-es-a1-02` — minor — `relay.chrome.session.ended_hint`

- Current: `Un envío o Reanudar la reabre en tu carpeta personal`
- Proposed: `Enviar un mensaje o Reanudar la reabre en tu carpeta personal`
- Reason: `Un envío` as a countable noun reads as a shipment; this locale's own `relay.asks.sheet.ended_wait` phrases the same behaviour with an infinitive (responder la reabrirá). Everything else is correct: `Reanudar` is the button's own word and `carpeta personal` is the settled rendering of the home folder, so the ended / history-kept / reopens-in-home guarantee survives.

### `chrome25-es-a1-03` — nit — `relay.chrome.session.degraded`

- Current: `No responde — los envíos esperarán`
- Proposed: `No responde — los mensajes esperarán`
- Reason: same noun choice as `a1-02`; the banner concerns the messages typed now, which this locale calls `mensajes` (`composer.label`). The proposal is 33 cells against a 36-cell budget.

### `chrome25-es-a1-04` — nit — `relay.chrome.drawer.open` and `relay.chrome.drawer.close`

- Current: `Abrir navegación` / `Cerrar navegación`
- Proposed: `Abrir el menú de navegación` / `Cerrar el menú de navegación`
- Reason: word-for-word calque that leaves `navegación` as a bare abstract object; a screen reader announces "Abrir navegación, botón" with no object named, where Spanish UI convention names the control ("el menú"). The landmark name (`drawer.label`) stays `Navegación`.

### `chrome25-es-a1-05` — nit — `relay.chrome.session.back`

- Current: `Volver a conversaciones`
- Proposed: `Volver a las conversaciones`
- Reason: the definite article is expected in a Spanish back label; the current form reads telegraphic, and this string is also the visible action on the load-failure screen. Proposal is 26 cells against a 28-cell budget.

### `chrome25-es-a1-06` — nit — `relay.chrome.session.loading`

- Current: `Cargando conversación…`
- Proposed: `Cargando la conversación…`
- Reason: same article omission; the line names one specific conversation. Proposal is 25 cells against a 28-cell budget.

### `chrome25-es-a1-07` — nit — `relay.chrome.list.filter_empty`

- Current: `No hay conversaciones con este filtro`
- Proposed: `No hay conversaciones que coincidan`
- Reason: `con este filtro` reads as a property the conversations have rather than the matching condition; `que coincidan` matches how the sibling search empty state (`list.no_matches`) phrases a match. Proposal is 35 cells against a 40-cell budget.

## Safety-related wording (explicit check)

| Key | Text | Verdict |
|---|---|---|
| `session.tray.approve` | Aprobar | PASS — one-word action pill for a run held until the operator approves or denies; `aprobar` is this locale's approvals verb (`relay.asks.pending.approve`) |
| `session.tray.approve_label` | Aprobar — hay una solicitud esperándote | PASS — starts with the pill's own word, satisfying the voice-control rule, and states that a request is waiting. As in English, the pill opens the approvals sheet rather than approving on the spot; the translation adds and removes nothing |
| `session.ended` / `ended_hint` / `resume` / `resuming` | Finalizada · historial guardado / Un envío o Reanudar la reabre en tu carpeta personal / Reanudar / Reanudando… | PASS on meaning: ended is stated, history is kept, the busy state replaces the label, `Reanudar` matches in both keys, and the session returns to the home folder, not its project folder. The only issue is wording (`chrome25-es-a1-02`) |
| `session.degraded` | No responde — los envíos esperarán | PASS on meaning: not answering, sends are kept for later, no action implied. Wording nit (`chrome25-es-a1-03`) |
| `list.mark_all` / `row.mark_read` | Marcar todo leído / Marcar como leído | PASS — both state what happens to unread work, same verb |

## Width

Character counts for Latin script are a proxy for `maxCells`; the targets below are the tightest in the file. No string exceeds its budget under any realistic value.

| Key | en chars | es chars | `maxCells` |
|---|---:|---:|---:|
| `model.effort` | 16 | 24 | 24 |
| `list.filter.needs_you` (count 12) | 14 | 16 | 16 |
| `link.since_seconds` (count 47 and 59) | 41 | 44 | 44 |
| `link.since_minutes` (count 59) | 41 | 43 | 44 |
| `appearance.group_by_tag` | 26 | 35 | 36 |
| `session.degraded` | 31 | 34 | 36 |
| `session.ended` | 28 | 31 | 32 |
| `list.filter.unread` (count 12) | 11 | 13 | 16 |

Three strings sit exactly on the budget (`model.effort`, `list.filter.needs_you` at two-digit counts, `link.since_seconds` from 47 s). This is a signal for a rendered confirmation, not a semantic defect. The `many` branch of the two link counters is unreachable in practice (seconds reset at 60, minutes at 60).

## Considered and cleared

- `list.filter.needs_you` — `Te necesita · 12`: a fixed singular label with the count separated by the middle dot; the alternative `Te necesitan` would be wrong at count 1, so singular is the better compromise. Kept.
- `session.ended` — `Finalizada · historial guardado`: `Sesión` is dropped to fit 32 cells (`Sesión finalizada · historial guardado` is 37); it agrees with the subtitle (`Finalizada`) and loses no meaning.
- `session.ended_hint` — `carpeta personal`: established Spanish rendering of the home folder (GNOME's Spanish UI uses `Carpeta personal`).
- `spend.title` / `composer.spend_label` — `Gasto y contexto`: the glossary's cost→coste row binds *cost*, not *spend*; `gasto` is correct and identical in both keys.
- `appearance.on` / `appearance.off` — `Sí` / `No`: `Activado`/`Desactivado` would break the 6-cell budget; Sí/No is a used Spanish toggle state value.
- `session.subtitle.idle` — `Inactiva`: agrees with `sesión` and matches the locale's `sesiones inactivas` in `wire.settings`.
- `list.section.previous` — `Anteriores` vs `drawer.past` — `Sesiones anteriores`: same root, different scopes; the drawer names the sessions explicitly.
- `model.effort` — `Esfuerzo de razonamiento`: matches `wire.settings.model_effort.label` and `wire.slash.effort`.
- `appearance.wide_view` — `Vista ancha`: literal but accurate for a full-width toggle; `Vista amplia` is no clearer.
- `drawer.status` / `link.*`: middle dot (U+00B7) and U+2026 ellipsis preserved in every branch; `Sin conexión · reintentando` matches the locale's offline wording.

## Checks run

Environment: `~/local-operator/.venv/bin/python` (the worktree has no venv), with `PYTHONPATH` set to the worktree so the worktree's package and catalogues are imported.

1. `shasum -a 256` on the three inputs — values as tabulated above.
2. `git rev-parse HEAD` — `535cea359825411741069cbffcee42e2927df9d3`.
3. `python scripts/i18n/check.py` — last line: `check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)`; exit 0.
4. `check_catalogues()` in-process — actual output: `total catalogue problems across all locales/namespaces: 0` and `problems under es/relay.chrome: 0`.
5. `audit_checks.py` — 16 PASS, 0 FAIL. `every key has identical (name,kind) argument tuples` (14 argument-bearing keys); each of the four plural keys: `selectors ['one', 'many', 'other'] are es categories/`=N` and define other` and `es categories covered ['many', 'one', 'other'] missing=[]`; negative controls: renamed argument detected as a tuple mismatch, a branch without `#` reported `renders_value=False`, missing `{count}` raised `MessageFormatError: message argument 'count' was not supplied (have: none)`.
6. `render_product.py` — actual: `rendered keys: 83 | keys with arguments: 14 | fell back to the code: 0`.
7. `checks-many.txt` — actual renders: `0 preguntas`, `1 pregunta`, `1.000.000 de preguntas`, `Reconectando · sincronizado hace 59 minutos`, `Agentes 2/4 · 1 fallido`, and the English control `1,000,000 questions`.
8. `unicode_checks.py` — 83 keys, key order identical to English, UTF-8 without BOM, trailing newline, all values NFC, no stray non-ASCII classes, no ASCII `...` where U+2026 belongs, no double spaces, no leading/trailing whitespace, no straight quotes, no usted/vosotros. All clean.
9. Catalogue discovery — `catalogues.namespaces("es")` includes `relay.chrome`; `load_catalogue("es", "relay.chrome")` returns 83 keys.

Logs in this directory: `checks.txt`, `checks-structural.txt`, `checks-catalogue.txt`, `checks-discovery.txt`, `checks-many.txt`, `checks-product-render.txt`, `checks-tokens.txt`, `checks-unicode.txt`.

One check was discarded as non-discriminating and is not used as evidence: a regex brace-head scan matched text inside ICU branch bodies and reported false differences (`{Connected` vs `{Conectado`). The parser-based `message_arguments` comparison replaced it and reports 0 mismatches.

## Not claimed here

- No rendered-frame or screenshot evidence: this pass judged text, sidecar roles and budgets only. The exact-at-budget strings need a rendered confirmation in the separate visual pass.
- No screen-reader or device test; the start-with rule was checked on the string, not on a device.
- Character counts approximate cells; `maxCells` comes from the sidecar, not from the app's font metrics.

## Verdict

**PASS** for `relay.chrome` in Spanish. The copy is accurate, idiomatic neutral Spanish in the `tú` register, glossary-consistent, structurally sound (arguments, plural categories, separators) and within every budget. The seven findings are copy polish (2 minor, 5 nit) and none changes meaning, so none blocks. `a1-02` is the one to fix first if only one is taken: it is the sentence that tells the operator how an ended session comes back.
