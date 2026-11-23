# Spanish independent audit A — tui.provider_state

Verdict: **passed**; all five keys reviewed; zero unresolved BLOCKER/MAJOR findings; no findings requiring copy changes.

## Native product-copy assessment

- `logged_in` — `sesión iniciada`: idiomatic completed authentication status, not an instruction; does not introduce a network-health claim. `sesión` follows the glossary.
- `env_key` — `clave del entorno`: compact, natural developer-facing Spanish for a key supplied by the environment. In this authentication-state column, `entorno` denotes credential origin rather than a server name. Expanding to `clave de variable de entorno` would be explicit but unnecessary here and longer; the existing form is not an identifier and does not need to preserve `env`.
- `needs_login` — `requiere iniciar sesión`: natural requirement status with the provider as implicit subject; not a command or an assertion of successful login. Standard `iniciar sesión` terminology and tú-compatible impersonal register.
- `local_ready` — `servidor configurado`: completed configuration state, not connectivity/health/authentication. The provider context supplies locality, as in the English source; no need to add `local`.
- `local_unconfigured` — `configurar servidor`: standard infinitive instruction, clearly distinct from completed `configurado`; preserves the English fragment's brevity and context-supplied local scope.

All five are lowercase fragments without final punctuation, consistent with the context and Spanish style guide. No region-marked vocabulary, formal-address drift, product-name replacement, or destructive/permission polarity issue. No RTL requirement applies.

## Mechanical and fit scope

`audit.log` is actual execution output from `script.py.txt`, not these reasoning notes. The official checker is invoked with only catalogue enumeration narrowed to en/es and this namespace; the checker itself, context checks, ICU parser and render functions remain real. Missing-key and added-argument fixtures are in memory, never target edits. NFC, exact ordered five-key parity, zero arguments and zero plural/select branches, protected-token parity, English rendering controls, synthetic pseudo expansion, Rich rendering and cell widths are checked.

Spanish widths are 15/17/23/20/19 cells, English 9/7/11/17/16. A 17-cell example beside `openrouter` is diagnostic, not a declared global maximum. Longer status strings can truncate under a particular consumer budget; that does not establish a translation defect. Synthetic/plain Rich rendering is not a styled settings integration, native-font screenshot or application-flow proof. Current UI producer remains English until the separate typed resolver integration. No global eligibility/native clearance is claimed.

## Independence and provenance

Auditor session `a17a04a01711` is distinct from translator session `d37edc1334b7`. Translator identity is sourced from `es22a/record.json`; no translator notes, scripts or independent auditor conclusions were used. Native judgment is LLM reasoning only, with no translation service or back-translation. Host model-switch notice reports a quota fallback from `anthropic/claude-opus-5-5` to `openai/gpt-6.1-sol`; no independent per-response served-model telemetry is available. This is independent pass A, not the separately required pass B.

Notes classification: `audit_notes_not_execution_log`.
