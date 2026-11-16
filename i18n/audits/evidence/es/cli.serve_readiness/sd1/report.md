# Independent Spanish audit — cli.serve_readiness / sd1

- Date: 2026-11-16. Scope: all 11 keys in the new namespace, at base e4a6b6e1d01ef269604fe6d005e6bce1453b66ca plus the uncommitted catalogue bytes pinned in record.json.
- Auditor: Audit daemon Spanish; session e68b64992930; actual model openai/gpt-6.1-sol, as declared by the host session state. No fallback is recorded or observed.
- Translator: task acf0a84cd992, actual openai/gpt-6.1-sol (requester-provided provenance). Distinct sessions; auditor did not translate or modify these inputs.
- Verdict: **passed**, zero unresolved BLOCKER/MAJOR findings. One MINOR finding below is non-blocking.
- Method: source/context and Spanish copy read in full after the glossary and Spanish style guide. Reasoned independent reading, no translation/MT service. This is one independent pass, not evidence that the separately required second independent safety-copy audit has happened.

## Findings

### ES-SD1-01 — MINOR — cli.serve_readiness.not_reporting

**Current:** `está registrado pero no responde (su heartbeat se detuvo)`

**Problem:** The English noun and literal “se detuvo” make the otherwise Spanish diagnostic less idiomatic. “Heartbeat” is understandable developer jargon, so this is not an unsafe meaning error or a blocking glossary finding; it is not one of the guide’s explicitly retained technical terms.

**Suggested wording:** `está registrado pero no responde (dejó de enviar señales de actividad)`

## Per-key review (all keys)

| Key suffix | Mechanical | Semantic / idiom / terminology / safety result |
|---|---|---|
| unproven | PASS | Keeps the inability to verify process identity, including the unavailable start-time comparison. Does not assert that the recorded build is running. Natural factual wording. |
| not_reporting | PASS | Retains both missing response and stopped heartbeat; non-blocking Spanish clarity finding ES-SD1-01 above. |
| not_this_daemon | PASS | Correctly attributes the PID to another process now, not to the recorded daemon; PID remains the technical acronym. |
| unknown_state | PASS | Preserves recorded-but-unknown state, without upgrading uncertainty to failure. |
| install_unknown | PASS | Correct comparison direction and uncertainty: running build cannot be compared with installation. No stale-build claim. |
| cgroup_unknown | PASS | cgroup remains a technical identifier; “ubicar” is natural for inability to classify/place this group. |
| recorded_gap | PASS | Both build placeholders intact; “registrada” plus “identidad del proceso sin comprobar” explicitly avoid treating the recorded build as the current process. |
| unknown | PASS | one/many/other all present; singular and plural agreement correct; “coincide con la instalación” conveys “current” in this build-check context. Does not imply supervision is known. All device/reasons arguments survive every branch. |
| stale | PASS | Every branch distinguishes the running build(s) from the installed build; hosted remote sessions run the former, not the latter. Correct singular/plural daemon verbs; session plural is retained. |
| other_unit | PASS | Preserves another service versus a lop serve unit, with holders unchanged. “Está a cargo de” is acceptable for a process under another service’s charge. unidad/servicio agree with the glossary. |
| identity_restart | PASS | tú imperative “ejecuta”; both literal commands unchanged and in the right inspection-then-restart order. Preserves recording identities and checking each daemon before acting; does not imply permission to bypass the guarded restart. |

## Mechanical and fit evidence

`audit.log` is actual execution output, including the executed inline verification script, not audit notes. The lightweight repository command `.venv/bin/python scripts/i18n/check.py --catalogues-only` returned 0 problems. Separate strict JSON reading checked duplicate keys and exact 11-key/context parity. The repository ICU parser was used for name/kind parity and additional per-branch argument-type and literal checks; three plural messages all have Spanish one/many/other, exercised at counts 0, 1, 2 and 1,000,000. No select expressions, Rich/HTML tags, flags, arrows or additional code tokens occur. Backtick command spans remain byte-identical in corresponding branches; the plain `lop serve` unit label also remains unchanged by visual comparison.

No maxCells/maxChars hint occurs in these 11 sidecars. Sample rendered strings have their cell/character counts in the log; these long diagnostics are not button/tab labels and will wrap in narrow terminals. No hard-width violation is established. Spanish uses LTR Latin script; all target strings are NFC, with no bidi controls or combining-mark/wide-glyph concern. Punctuation follows Spanish rules. RTL/Urdu checks are not applicable to es. There is no live terminal/browser screenshot or full layout validation in this audit, and no claim of release eligibility.

## Retention and authority

Only this new evidence bundle is written. Source, translations, prior bundles, ledger/history and manifests are untouched. `receipt.json` is the append-ready ledger row; its note links record.json. `ledger-dry-run.log` records receipt validation only, not an applied append. The parent owns appending history and committing these presently new files. This report does not claim the bundle is already on main.
