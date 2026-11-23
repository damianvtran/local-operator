# Independent French A — tui.provider_state

Date: 2026-11-22. Source revision: a54ec02de48a94b68d6c3dc92e1307c97ea776ad.
Auditor: i18n-auditor, session 25fe1758fc50, actual model openai/gpt-6.1-sol. Host explicitly switched from anthropic/claude-opus-5-5 after provider quota failure before this assessment. Independent of translator e0c1b7382197 and English author 93b7b02899af. Translator model/source verified directly from fr22a/record.json: openai/gpt-6.1-sol, same quota fallback; model identity alone does not make sessions identical.

## Verdict

PASSED for this five-key linguistic/mechanical scope: zero unresolved BLOCKER/MAJOR. One non-gating MINOR integration risk. This is A only, not the required second independent pass; not a UI/native-font/overall French settings/release/eligibility certification. No consumer implementation was audited. No MT/API/translation service or other auditor opinions consulted. No target edits.

## Every key, independently assessed

| Key suffix | Current French | Assessment |
|---|---|---|
| logged_in | connecté | Idiomatic compact completed-login status, not the action « se connecter ». Does not claim storage mechanics beyond the source's shorthand. Accent correct; generic implicit user/account masculine is normal here. 8 cells. |
| env_key | clé d’environnement | Natural terse developer label identifying an environment-supplied key, not a literal variable name or a server configuration. Distinguishes the key source from a stored login. Curly apostrophe and accents correct. Expanding to « clé de variable d’environnement » is unnecessary in this compact provider context. 19 cells. |
| needs_login | connexion requise | Idiomatic factual requirement; not an imperative/button. Correct polarity and authentication meaning. 17 cells. |
| local_ready | serveur configuré | Correct completed configuration state, not « serveur opérationnel »/« disponible », which would assert health/reachability. Locality is supplied by the local-provider context, as in English, not omitted from a source word. 17 cells. |
| local_unconfigured | configurer le serveur | Natural French infinitive instruction, distinct from the completed state above. The article is idiomatic; matches French action-label convention. Does not turn missing configuration into a login failure. 21 cells. |

All five obey sentence-case/lower-case fragment conventions, consistent terse professional register, glossary serveur and technical-identifier policy. No pronoun/formality conflict, sentence punctuation, numeric formatting, RTL, markup, commands, flags, paths or embedded opaque identifiers is present. All five are zero-argument and have zero plural/select branches; therefore French one/many/other coverage is vacuous, not missing. Ordered key parity and valid JSON checked through the official loader and again by exact expected-key assertion. All English/French strings are NFC and contain no bidi format characters.

## F-A1 — MINOR: integration width risk, not an invented catalogue budget

Keys: tui.provider_state.env_key and tui.provider_state.local_unconfigured.
Current: « clé d’environnement » (19 cells), « configurer le serveur » (21 cells).
Problem: at the supplied illustrative 17-cell state allowance, plain Rich ellipsis produces « clé d’environnem… » and « configurer le se… ». The latter loses the complete noun. This is a real truncation risk for the future consumer integration, not proof that either string violates a declared maxCells: the source context explicitly has no fixed budget and derives it from pane and opaque ID widths. English controls fit 17 cells (7/16 respectively), so French adds risk. Narrower panes/longer IDs can also truncate English.
Suggested wording/disposition: retain the current idiomatic wording for this catalogue-only pass. Root must measure the real styled consumer geometry and preserve understandable access to the complete label where truncation occurs. Do not force a compressed calque or invent a universal 17-cell ceiling. No MAJOR overflow claim without the real geometry/rendered path. No patch requested at this stage.

## Actual execution and limitations

`audit.log` is actual successful output from `script.py.txt`, executed with the primary checkout interpreter, this worktree PYTHONPATH, isolated HOME/config and env -i stripping CMUX variables. Official `scripts/i18n/check.py::check_catalogues` runs with only namespace/locale enumeration narrowed to en/fr + tui.provider_state. Validator decisions are not overridden. Official context/ICU checks return []. In-memory missing-key and added-argument controls are rejected with specific diagnostics. Runtime rendering, NFC, plain Rich output and expanded pseudo width run across all five English and all five French labels. Frozen en/context/fr SHA-256 values match before and after.

First execution failed in the audit harness's importlib setup (module omitted from sys.modules, causing dataclass import error), before checks ran. Corrected the harness registration and reran successfully; original actual traceback is preserved in harness-failure.log. This is not a product failure. This report is audit_notes_not_execution_log; only the .log files prove execution. Pseudo/plain Rich checks are not styled TUI, native font, settings fit, or screenshot proof. No app startup, tests, build, installation, generation, ledger apply or history mutation performed. Ledger append dry-run is recorded separately in ledger-dry-run.log.
