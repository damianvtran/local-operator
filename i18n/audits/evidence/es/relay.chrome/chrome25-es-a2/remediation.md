# `relay.chrome` — Spanish (`es`) — unified remediation round 1

**Locale:** `es` (neutral Spanish, `tú` register) · **Namespace:** `relay.chrome` (83 keys)
**Session:** `d69e1fa9cd3c` · **Role:** translator, team `lopdev` · **Model:** `radient/auto` (harness-reported; no fallback selected or used)
**Worktree:** `~/local-operator/.worktrees/relay-chrome-i18n` @ `535cea359825411741069cbffcee42e2927df9d3`
**File written:** `local_operator/i18n/catalogues/es/relay.chrome.json` (only file touched; no commit)

## Hashes

| Artifact | SHA-256 |
|---|---|
| `en/relay.chrome.json` (frozen, re-verified after the edit) | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| `en/relay.chrome.context.json` (frozen, re-verified) | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| `es/relay.chrome.json` **before** (snapshot kept in scratch) | `9e1b68d5e546317ff912e1aa9d463d9ae6d44d7b109e0e584ef5e82a1a667192` |
| `es/relay.chrome.json` **after** | `416f3d6c884403ea962f158b1718f3743352674b9e7ffc5a47aed4c7520df333` |

Snapshot of the pre-remediation candidate: `relay.chrome.es.before.json` in this directory (identical to the hash the two audits recorded), so the delta below is exact.

## Findings and dispositions

All nine reported items concern seven keys; every one was **fixed**. None rejected, none deferred.

| Finding | Source · severity | Key | Current → new |
|---|---|---|---|
| `chrome25-es-a1-01` | locale audit · minor | `list.empty_hint` | `Empieza una con Nuevo, …` → `Empieza una con el botón Nuevo, …` |
| `chrome25-es-a1-02` = `MIN-01` | locale · minor + safety · minor | `session.ended_hint` | `Un envío o Reanudar …` → `Enviar un mensaje o Reanudar …` |
| `chrome25-es-a1-03` = `MIN-02` | locale · nit + safety · minor | `session.degraded` | `No responde — los envíos esperarán` → `No responde — los mensajes esperarán` |
| `chrome25-es-a1-04` | locale · nit | `drawer.open`, `drawer.close` | `Abrir/Cerrar navegación` → `Abrir/Cerrar el menú de navegación` |
| `chrome25-es-a1-05` | locale · nit | `session.back` | `Volver a conversaciones` → `Volver a las conversaciones` |
| `chrome25-es-a1-06` | locale · nit | `session.loading` | `Cargando conversación…` → `Cargando la conversación…` |
| `chrome25-es-a1-07` | locale · nit | `list.filter_empty` | `No hay conversaciones con este filtro` → `No hay conversaciones que coincidan` |

**Changed keys (8, exact):** `drawer.open`, `drawer.close`, `list.empty_hint`, `list.filter_empty`, `session.back`, `session.loading`, `session.ended_hint`, `session.degraded`. Everything else is byte-identical to the audited candidate.

### Why each was accepted

- **a1-01** — `una` (conversación) beside the articleless button name `Nuevo` is a calque of "start one with New"; naming the control removes the clash while keeping the required button word byte-identical. No `maxCells`; length 60 → 68 cells, so it is flagged for rendered fit rather than called clipped.
- **a1-02 / MIN-01** — `Un envío` nominalises "a send" and in this corpus the noun means a dispatch (`wire.slash` "un envío anterior"; `wire.settings` "Registrar envíos sin confirmar"). The verb form matches `composer.label` `Mensaje`, keeps the button word `Reanudar` and `carpeta personal` untouched, and adds no new promise.
- **a1-03 / MIN-02** — same noun problem: the thing kept is the message the user wrote, which this catalogue calls "mensajes". `…esperarán` remains "will wait"; no queued/delivered guarantee is introduced. Width 36/36 (see residual).
- **a1-04** — accessible names only (no `maxCells`): "Abrir el menú de navegación" names the control, while the landmark (`drawer.label`) stays `Navegación`. Zero painted cost.
- **a1-05 / a1-06** — Spanish expects the definite article in a back label and in "loading the conversation…"; 27/28 and 25/28 cells.
- **a1-07** — the old line read as "conversations with this filter" (a property) instead of "conversations that match"; `que coincidan` matches how `list.no_matches` phrases a match, 35/40 cells.

## Verified against evidence, not just the reports

- Both frozen English hashes re-computed after the edit: unchanged.
- ICU argument tuples (`local_operator.i18n.runtime.message_arguments`) compared en↔es for all 83 keys: **0 mismatches** — no placeholder, kind or order changed by this round.
- Plural selectors re-read per plural key: `one, many, other` on all four (`link.since_seconds`, `link.since_minutes`, `tray.questions`, `tray.agents_failed`), every branch still carrying the number; no `few`.
- Key parity: 83/83, order identical to English, no additions or removals.
- Typography: U+00B7 and U+2026 preserved, spaced em dash kept, no double spaces, no ASCII `...`, no straight quotes, no trailing periods on labels, no `usted`/`vosotros`, UTF-8 NFC, trailing newline.
- Corpus cross-checks: `mensajes` (`composer.label`), `Nuevo` (FAB convention), `carpeta personal`, `Reanudar`, `Ajustes`, `Biblioteca`, `etiquetas`, `esfuerzo de razonamiento` all remain the locale's established renderings.

## Checks run (actual output)

```
$ cd ~/local-operator/.worktrees/relay-chrome-i18n && PYTHONPATH=. ~/local-operator/.venv/bin/python scripts/i18n/check.py
check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)
```
exit 0; no line names `es/relay.chrome`. The 34 advisories are the pre-existing literal-ratchet notes in `local_operator/*.py` source files (outside the enforced scope).

In-process `check_catalogues()`: `total catalogue problems: 0`, `es/relay.chrome problems: []`. This was taken with the other six locales' in-flight files present; the wave's own state was consistent at that moment.

No full test suite was run (not required for this round). The wave-level Hindi fixture failure reported earlier is unchanged and out of this locale's delta.

## Residual — actionable, for the rendered/QA pass

1. `session.degraded` is now **exactly 36/36** cells — confirm the banner neither wraps nor clips at 360 px. (Pure source-level width is not claimed as clipping.)
2. `list.empty_hint` grew **60 → 68** cells with no sidecar budget; confirm it still wraps to two lines in the empty list.
3. `link.since_seconds` sits at **44/44** from a two-digit count; confirm the one-line subtitle does not clip. Its `many` branches (and `since_minutes`') exceed 44 only at 1,000,000-scale counts — unreachable (seconds roll at 60; minutes would need ~2 years), and the safety audit recorded the same conclusion.
4. At or near budget: `list.filter.needs_you` 16/16 (two-digit count), `model.effort` 24/24, `session.ended` 31/32, `appearance.group_by_tag` 35/36.
5. `list.mark_all` `Marcar todo leído` is a budget-driven compression of `Marcar todo como leído` (23 cells vs 20); the safety report explicitly excluded it as a finding, but a rendered read is advisable if the header is tight.

## Left unchanged on purpose

`Te necesita · {count, number}` (singular fixed label, correct at count 1, action-required sense intact — the Hindi "for you" issue does not apply); `Finalizada · historial guardado` (states history kept, 31/32, both audits cleared the dropped subject); `Sí`/`No` toggles (6-cell budget); `Gasto y contexto` (spend ≠ glossary `coste`); `Inactiva`/`Esperándote` (agree with `sesión`, cleared by both audits); full-word `sincronizado hace # segundos/minutos` retained over abbreviation because it is the element that keeps the freshness meaning.
