# Spanish audit — wire.errors.session_host — pass A

**PASS: 3/3 keys; no findings, no unresolved blocker/major.**

Auditor: i18n-auditor / session `1436c962e633` / actual `openai/gpt-6.1-sol` (quota fallback). Translator session `056852035b33` is distinct; same model does not imply same agent. Audit date: 2026-10-31. Source revision: `9c979a03d551020ce6ff1b16c8066aa69ce7bb2a`.

## `wire.errors.session_host.unsupported` — PASS

Este equipo refers naturally to the computer here. Cannot-run and not-started facts remain explicit. Usa uses required tú register; compatible correctly qualifies the background service manager. No suggestion of reconnection, automatic retry, or temporary platform support.

## `wire.errors.session_host.unavailable` — PASS

No ha respondido correctly reports host nonresponse without diagnosing a crash. No se ha iniciado preserves terminal startup failure. Espera un momento antes de volver a intentarlo is idiomatic tú and asks the person to retry later, not an automatic reconnection.

## `wire.errors.session_host.refused` — PASS

Ha rechazado iniciar esta sesión preserves refusal rather than unavailability or permission denial. Separate not-started sentence remains explicit. Revisa correctly directs log inspection before a manual retry. logs/session-host.log and Local Operator remain byte-exact.

## Mechanical execution

Independently ran committed `scripts/i18n/check.py::check_catalogues()` against the translator's existing read-only scoped en/es fixture after verifying validator/dependency/source bytes against the named commit and candidate bytes against the fixture. Actual result: `[]`, exit 0. Command: `PYTHONDONTWRITEBYTECODE=1 python3 "$LOCAL_OPERATOR_SCRATCHPAD/host-es-audit-a/validate_audit.py"`. See actual `audit.log`, not this note, for execution evidence. Independent literal/key/argument checks also passed. Source and context match requested SHA-256 values; target SHA-256 is `b27376785fe2c4bc4e3ca302e3f34097dad93bf483ccef5a921177ac6e8a00b8`.

Initial setup attempt incorrectly assumed fixture had `local_operator/__init__.py`; this namespace-package fixture does not. That check was removed without changing production/translator files; setup failure is retained in `validation-setup-attempt-1.log`. Final validator and dependency checks passed.

## Fit / policy / limits

No maxCells/maxChars hints are declared. Spanish text has 178/143/211 Unicode characters respectively; no rendered-fit claim is made. No ICU plurals/selects or dynamic arguments exist; path and product literals preserved. Native idiom, neutral Spanish and tú address are consistent with the glossary/style guide. Rendering background session host as servidor de sesiones en segundo plano correctly communicates a service/process, not a human anfitrión or physical machine. No glossary conflict found. Urdu/RTL not applicable.

This is the first independent pass only. High-visibility error copy requires a later separate second independent auditor; do not publish a passed ledger row solely from this proposal. No actual ledger write, shared-worktree write, translation edits, commits, installs, copytrees or full suite performed. RFC §6 specifies a ledger JSON schema, not a named audit-ledger command; no nonexistent CLI was invoked.

## Planned retention

`i18n/audits/evidence/es/wire.errors.session_host/es31a/record.json` plus companions. These files currently exist only in auditor session scratch, not tracked or on main. Coordinator must retain them at the canonical path before closing temporary references.
