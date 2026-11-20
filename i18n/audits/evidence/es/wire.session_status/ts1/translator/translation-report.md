# Spanish terminal-status translation

Status: **translated, NOT audited**. Translator: `i18n-translator (terminal-status-es)`; session `b39246e0d986`; actual model `openai/gpt-6.1-sol`, from the session-state Model field. No fallback was reported by the session. No translation API/service was used.

## Scope and delivery

Frozen worktree: `/Users/damian/local-operator/.worktrees/i18n-terminal-status` at `2930492e05a6968a2e1adfe34a091b94802bacd0`.

- `wire.session_status.json`: all 17 keys, source order; 16 newly translated whole messages.
- Existing `wire.session_status.retired_label` value preserved exactly: `{base} — retirada por una actualización`.
- `provenance.json`: SHA-256 of English, context, previous Spanish, target, glossary/style/HOWTO/RFC and validator/runtime inputs, actual translator identity and scope limitations.
- `receipt.json`: status `translated`, null auditor identity (no independent audit performed), current append-compatible shape.
- `checks.log`: actual mechanical checker execution output.
- `format-samples.json`: 43 formatting samples, not a native-copy audit or integrated runtime/visual evidence.
- `check_translation.py`: reproducible narrow mechanical checker.
- `manifest.json`: deliverable and log hashes; does not hash itself.

All outputs are session scratch deliverables. No repo, generated file, audit-history, install, build, server or PR write was performed. Existing worktree status was clean on the initial and final tracked-file checks. No independent safety-copy audit or rendered/UI clearance is claimed. Two independent safety-copy audits remain mandatory.

## Wording decisions and ambiguities for independent auditors

- Unseen → **sin leer**, per context's explicit “not yet read,” rather than suggesting a new or undetected failure. `Interrupción sin leer`, `Finalización sin leer`, `Error sin leer` distinguish the outcomes.
- Session-based simple statuses are feminine: **Interrumpida**, **Completada**, **Detenida**. The noun “interrupción” is also feminine in unseen labels.
- Update retirement → **retirada para actualizar**. This explicitly attributes the retirement to an update, distinct from `detenida por el usuario` in `user_stop`. The old suffix remains unchanged even though it uses different wording, as requested.
- Signal stop → **detenida con una señal**; forced kill → **finalizada por la fuerza**. This distinguishes a signal-driven stop from a forced termination without introducing an invented specific signal such as SIGKILL.
- Command attribution uses **mediante {command}**. `none` has an empty body, so unknown commands are not blamed on a literal “none” or an invented actor. Commands remain interpolated verbatim.
- Dormant wakes → **despertar inactivo / despertares inactivos**, retaining the glossary's despertar and meaning inactive, not cancelled or scheduled to fire. Spanish categories are `one`, `many`, `other`, with **# de despertares** in `many` as prescribed.
- Runtime stays **runtime**, per Spanish kept-English policy. `runtime_stall_bound` → **se cerró al agotar su límite sin avances** preserves self-termination and a no-progress bound. `runtime_overdue` → **salió para actualizar al agotar su margen de progreso** preserves the update motive and a bounded opportunity for progress. English does not specify duration or an exact deadline; Spanish intentionally adds neither. These two technical bounds deserve the independent safety-copy readers' attention.
- `runtime_update_failed` says **conservó la compilación cargada**, not that the new version was loaded. `runtime_killed` retains both disappearance without clean exit and **no se solicitó detenerlo**, not a user-stop assertion.
- `continuation_limit` retains **mensaje pendiente descartado**; `output_limit` retains inability to continue. These are distinct, not combined under a generic model error.
- `disposed` → **sesión descartada** rather than deleted: avoids claiming durable transcript deletion that the English does not establish.
- There are no numeric maxCells/maxChars caps in this sidecar: one logical line is required. No newlines or manual bidi controls were added. No visual fit claim is made.

## Glossary proposals (not written to repository)

- terminal-status `dormant` → **inactivo**, including **despertar inactivo** for a wake disabled by stopping, not cancelled.
- terminal-status `forced kill` → **finalización por la fuerza** (verb phrase **finalizada por la fuerza**).
- terminal-status `unseen` → **sin leer** when its precise meaning is an unread outcome.
- runtime `stall bound` → **límite sin avances**; `progress bound` → **margen de progreso**, pending independent safety-copy review.
- session `disposed` → **descartada**, without implying durable data deletion.

## Mechanical validation actually executed

Command:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/damian/local-operator/.venv/bin/python \
  "$LOCAL_OPERATOR_SCRATCHPAD/terminal-status-es/check_translation.py" \
  > "$LOCAL_OPERATOR_SCRATCHPAD/terminal-status-es/checks.log" 2>&1
```

Exit 0. The script invokes **the repository's `scripts.i18n.check.check_catalogues()`** against a scratch fixture carrying only this frozen English namespace, context and Spanish target: **0 problems**. It separately checks exact key order, retained legacy value, placeholders/kinds, select arm codes/order and Spanish CLDR categories; executes 43 ICU-format samples with the repository formatter; verifies unknown command attribution is omitted and technical command bytes survive; and invokes `ledger.append(..., apply=False)` using freshly calculated current history SHA.

Actual output is in `checks.log`; no full-tree catalogue validator, full test suite, application runtime or UI screenshot was run. The targeted fixture route avoids unrelated incomplete locale files and obeys the no-repository-writes scope.

Receipt dry-run accepted against history SHA `133d29bbc8992b2b92758d4b6ff78407c6fb5ab3b732c863d35b07e0ffe3d27c`. This is optimistic concurrency evidence at execution time, not permission to append against a later changed history.

## Preserved programme gap

Unknown nonempty external diagnostic text intentionally retains its legacy English branch outside this namespace. This remains an **unresolved programme gap**, not an allowed localization exception and not localized by these strings.
