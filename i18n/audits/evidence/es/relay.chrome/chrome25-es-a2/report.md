# `chrome25-es-a2` — terminal delta verification — `relay.chrome` / Spanish (`es`)

**Verdict: TERMINAL PASS** — no blocker, no major, no new findings. All 7 R1 findings fixed with the exact proposed text; the 75 untouched keys are byte-identical to the R1-audited snapshot, so R1 coverage carries forward.

## Identity and provenance

| Field | Value |
|---|---|
| Audit ID | `chrome25-es-a2` (R2 delta of `chrome25-es-a1`) |
| Role | `translation-auditor`, same owner as R1; independent of the translator |
| Session | `56f831d2290f` (subagent), parent `44f93fd45b09`, label `Audit chrome es` |
| Model (runtime header) | `radient/auto`, effort `auto` — read from this session's `selected_model` runtime events (`~/.local-operator/sessions/56f831d2290f/transcript.jsonl`, two events, both `{"selector":"radient/auto","effort":"auto","fast":false,"boot":"radient/auto"}`) |
| Model caveat | That is the harness-reported selector only. The underlying provider/model behind `radient/auto` is **not** inferred or claimed here. |
| Run date | 2026-10-25 |
| Worktree | `~/local-operator/.worktrees/relay-chrome-i18n` |
| Base | `535cea359825411741069cbffcee42e2927df9d3` |
| Repo state | read-only; no commit, no target or ledger edit; `git status` shows only the seven pre-existing untracked locale files |

## Hashes (all recomputed here, none copied from prose)

| Artifact | SHA-256 |
|---|---|
| `local_operator/i18n/catalogues/en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| `local_operator/i18n/catalogues/en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| es **before** = `relay.chrome.es.before.json` = R1 target | `9e1b68d5e546317ff912e1aa9d463d9ae6d44d7b109e0e584ef5e82a1a667192` |
| es **current** (final) | `416f3d6c884403ea962f158b1718f3743352674b9e7ffc5a47aed4c7520df333` |
| es current, as the loader reads it (`catalogue_sha256`) | `416f3d6c884403ea962f158b1718f3743352674b9e7ffc5a47aed4c7520df333` (identical — no normalisation) |

- English and context hashes are **unchanged** from the task-frozen values.
- The before snapshot the remediation record names hashes exactly to the R1-audited target (`9e1b68…`), so the delta measured below is between the audited bytes and the current bytes.
- The remediation record's claimed “after” hash equals the current file hash computed here.

## Exact delta

**8 changed keys** (of 83), JSON key order unchanged, key set identical:

| Key | Before | After |
|---|---|---|
| `relay.chrome.drawer.open` | Abrir navegación | Abrir el menú de navegación |
| `relay.chrome.drawer.close` | Cerrar navegación | Cerrar el menú de navegación |
| `relay.chrome.list.filter_empty` | No hay conversaciones con este filtro | No hay conversaciones que coincidan |
| `relay.chrome.list.empty_hint` | Empieza una con Nuevo, o desde Local Operator en tu máquina | Empieza una con el botón Nuevo, o desde Local Operator en tu máquina |
| `relay.chrome.session.back` | Volver a conversaciones | Volver a las conversaciones |
| `relay.chrome.session.loading` | Cargando conversación… | Cargando la conversación… |
| `relay.chrome.session.ended_hint` | Un envío o Reanudar la reabre en tu carpeta personal | Enviar un mensaje o Reanudar la reabre en tu carpeta personal |
| `relay.chrome.session.degraded` | No responde — los envíos esperarán | No responde — los mensajes esperarán |

**75 keys byte-identical** to the R1-audited snapshot. Canonical-JSON digests (sorted keys, no whitespace), recomputable from the two files:

- unchanged subset (75 keys, current bytes): `17d7150ce4cde41e3f18fd838506974d60e903460e56747048028e4a921e9ecb`
- delta map (8 keys, `{before, after}` pairs): `74dd8efb05cca745d20612495c11bc657dc4fe517343f329f01376754e0c617a`
- whole current catalogue: `8e5e4731a49cef1faec49ad4d99077838f3318a9f6a1388c3ac5d4214adcabc0` (canonicalisation digest; the file hash is the §Hashes value)

**R1 carry-forward:** the unchanged subset is byte-identical, so every R1 verdict on those 75 keys (clean / cleared) remains valid without re-reading them; scope-ownership is unchanged. This round re-read the 8 changed keys, the 7 prior findings, and the direct safety/ICU/terminology neighbours only.

## Finding dispositions (all 7 R1 findings)

| R1 finding | Severity | Keys | Disposition | Current text |
|---|---|---|---|---|
| `chrome25-es-a1-01` | minor | `list.empty_hint` | **fixed** — exact proposed wording | Empieza una con el botón Nuevo, o desde Local Operator en tu máquina |
| `chrome25-es-a1-02` | minor | `session.ended_hint` | **fixed** — exact proposed wording | Enviar un mensaje o Reanudar la reabre en tu carpeta personal |
| `chrome25-es-a1-03` | nit | `session.degraded` | **fixed** — exact proposed wording | No responde — los mensajes esperarán |
| `chrome25-es-a1-04` | nit | `drawer.open`, `drawer.close` | **fixed** — exact proposed wording (both keys) | Abrir el menú de navegación / Cerrar el menú de navegación |
| `chrome25-es-a1-05` | nit | `session.back` | **fixed** — exact proposed wording | Volver a las conversaciones |
| `chrome25-es-a1-06` | nit | `session.loading` | **fixed** — exact proposed wording | Cargando la conversación… |
| `chrome25-es-a1-07` | nit | `list.filter_empty` | **fixed** — exact proposed wording | No hay conversaciones que coincidan |

No finding was rejected and none deferred. The remediation record also lists safety-stream `MIN-01`/`MIN-02`; they land on `session.ended_hint` and `session.degraded`, i.e. the same two keys as `a1-02`/`a1-03`, and the same fixes cover them.

## Review of the changed text

- `drawer.open` / `drawer.close` — the calque is gone; the control is named (“el menú”), the pair stays parallel, and the landmark name `drawer.label` remains `Navegación` (27 and 28 cells; these are accessible names with no sidecar budget).
- `list.filter_empty` — “que coincidan” states the matching condition (subjunctive correct after an existential negative) and aligns with the sibling search empty state `list.no_matches` (“Ninguna conversación coincide”). 35/40 cells.
- `list.empty_hint` — the feminine `una` no longer collides with the masculine button name; the required button word `Nuevo` is byte-identical and the product name is untouched. 68 cells, no sidecar budget; sidecar says the hint wraps to two lines (rendered confirmation pending, see residual).
- `session.back` — definite article restored; this is also the visible action on the load-failure screen. 27/28.
- `session.loading` — names the one conversation being loaded. 25/28.
- `session.ended_hint` — the nominal “Un envío” is replaced by the action form this locale already uses for the same behaviour (`relay.asks.sheet.ended_wait`: “responder la reabrirá”); `Reanudar` still matches the button and `carpeta personal` still carries the home-folder guarantee. 61 cells, no budget.
- `session.degraded` — the kept thing is now “los mensajes”, matching `composer.label` (“Mensaje”); “esperarán” still says the send is kept for later, with no new delivery promise. 36/36 cells (see residual).

No changed string introduces a new accuracy, idiom, register, glossary or typography defect. Register remains consistent `tú`, sentence case, infinitive action labels, spaced em dash, U+2026 ellipsis.

## Safety, ICU and terminology neighbours (re-read, unchanged)

| Key | Rendered (es) | Result |
|---|---|---|
| `session.tray.approve` | Aprobar | PASS — pill word unchanged |
| `session.tray.approve_label` | Aprobar — hay una solicitud esperándote | PASS — still starts with the pill word; request-waiting stated |
| `session.resume` / `session.resuming` | Reanudar / Reanudando… | PASS — button pair unchanged; `ended_hint` still uses `Reanudar` |
| `session.ended` | Finalizada · historial guardado | PASS — ended + history kept unchanged |
| `composer.label` | Mensaje | consistent with the new “un mensaje” |
| `list.no_matches` | Ninguna conversación coincide | consistent with “que coincidan” |
| `drawer.label` | Navegación | landmark name unchanged |
| `list.mark_all` / `row.mark_read` | Marcar todo leído / Marcar como leído | PASS — unchanged |
| `session.tray.questions` (1, 3) | 1 pregunta / 3 preguntas | PASS — `one/many/other` intact |
| `session.tray.agents_failed` (1 failed) | Agentes 2/4 · 1 fallido | PASS — fraction and count intact |

ICU integrity across all 83 keys: argument `(name, kind)` tuples identical to English — **0 mismatches**; the four plural keys still carry `one, many, other`, define `other`, have no empty branch and render the value in every branch; no `few`.

## Width after the delta

| Key | en cells | es cells | `maxCells` | State |
|---|---:|---:|---:|---|
| `session.degraded` | 31 | 36 | 36 | exact — rendered confirmation pending |
| `session.back` | 21 | 27 | 28 | 1 spare |
| `session.loading` | 21 | 25 | 28 | 3 spare |
| `list.filter_empty` | 34 | 35 | 40 | 5 spare |
| `list.empty_hint` | 59 | 68 | — | no budget; wraps to two lines per sidecar — rendered confirmation pending |
| `model.effort` (unchanged) | 16 | 24 | 24 | exact |
| `link.since_seconds` at 59 s (unchanged) | 41 | 44 | 44 | exact |
| `list.filter.needs_you` at count 12 (unchanged) | 14 | 16 | 16 | exact |

Exact-at-budget strings are width **signals** for the rendered/visual pass, not semantic defects; none is claimed as clipping.

## Checks executed (actual output)

Environment: `~/local-operator/.venv/bin/python` with `PYTHONPATH` set to the worktree, so the worktree's package and catalogues are imported. All read-only.

1. `shasum -a 256` on en / context / before / current — the four values in §Hashes. `git rev-parse HEAD` → `535cea359825411741069cbffcee42e2927df9d3`.
2. `verify_a2.py` — delta: `changed keys: 8`, `byte-identical keys (R1 coverage carries forward): 75`, `key order identical to en: True`, `key sets equal: True`; `argument (name,kind) mismatches vs en: 0`; `plural losslessness problems: none`; unicode checks all `True`.
3. `scripts/i18n/check.py` — last line: `check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)`; exit 0 (advisories are pre-existing source-file ratchet notes).
4. `check_catalogues()` in-process — `total catalogue problems: 0`; `problems under es/relay.chrome: []`; `keys loaded: 83`.
5. `render_product.py` (all 83 keys via `messages.render(..., locale="es")`) — `rendered keys: 83 | keys with arguments: 14 | fell back to the code: 0`; changed keys render as quoted above.
6. Rendered-width pass at realistic values — table in §Width; no `OVER`.
7. `catalogues.catalogue_sha256("es","relay.chrome")` → `416f3d6c…` (equals the file hash).

Logs: `a2-verify.txt`, `a2-catalogue.txt`, `a2-digests.txt` in the R2 scratch directory.

## Not claimed here

- Rendered-frame fit for the exact-at-budget and no-budget strings (`session.degraded`, `list.empty_hint`, `model.effort`, `link.since_seconds`, `list.filter.needs_you`, `appearance.group_by_tag`, `session.ended`) — that belongs to the rendered/visual pass.
- The separate second-auditor safety delta signoff for the high-visibility/safety keys — a different auditor's artifact, not produced here.
- No translator summaries or agent final reports were used; the delta was measured from the bytes and the on-disk remediation record only.

## Verdict

**TERMINAL PASS** for `es/relay.chrome`. The remediation is exactly the 8 keys the record claims, all seven R1 findings are fixed with the proposed wording, the other 75 keys are byte-identical to the audited snapshot (so R1 coverage carries forward), English and context bytes are unchanged, ICU and plural structure are intact, and no new out-of-scope issue or nit loop was opened. Final target SHA-256: `416f3d6c884403ea962f158b1718f3743352674b9e7ffc5a47aed4c7520df333`.
