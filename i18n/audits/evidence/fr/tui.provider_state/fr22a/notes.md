# French provider-state author receipt — translated, NOT audited

Date: 2026-11-22. Agent: i18n-translator. Session: e0c1b7382197.
Actual author/validator model: openai/gpt-6.1-sol. Host announced temporary fallback from anthropic/claude-opus-5-5 because of provider quota failure before authoring. No translation API, SDK, MT service or website used; the five French values were reasoned and authored directly by this LLM. No additional agents, model pins or runtime settings changed.

English source author: coder session 93b7b02899af, NOT this translator. Source receipt supplied by the manager: provider-state-source-receipt.md in that session's scratchpad; source private revision 1b77e201977c7f252a1fd8bac4baed36fc46c83a. Exact English/context input hashes are in record.json and checked by validator.py.txt.

## Native choices and ambiguities

- `logged_in` → **connecté**: conventional short authentication status, consistent with existing French tui.welcome `non connecté` / `se connecter`. Means stored login in this context, not a new network-health assertion. Lowercase status matches the source voice and French guide.
- `env_key` → **clé d’environnement**: explicitly a key sourced from the environment, not a literal environment-variable identifier, stored login or server configuration. Uses a typographic apostrophe as the French guide requires. Kept the full human term rather than an opaque `ENV` token or a less clear abbreviation.
- `needs_login` → **connexion requise**: factual requirement, not the action/button wording `se connecter`. French developer UI convention; consistent root terminology with welcome.
- `local_ready` → **serveur configuré**: past participle states completed configuration only. Deliberately avoids `prêt`, `disponible` or `opérationnel`, which could imply reachability/health.
- `local_unconfigured` → **configurer le serveur**: infinitive action per the French guide's action convention (its examples are Enregistrer/Reprendre/Ignorer); article retained for idiomatic French. Clearly distinct from `serveur configuré`.

No commands, identifiers, provider IDs, variable names, markup, emoji or placeholders occur in the source values. Five immutable ordered keys retained. Zero ICU arguments, plurals or selects in both languages. No manual bidi controls. NFC verified.

## Glossary proposals (not edits to the glossary)

The shared glossary has no authentication/environment-key entries. Propose: sign in / logged in → se connecter / connecté; login required → connexion requise; environment key → clé d’environnement. `serveur` is already prescribed by the French guide. These proposals do not change any current glossary policy.

## Actual mechanical validation

See validation.log for actual execution output and bounded-wrapper completion ID. The retained validator.py.txt calls the official scripts.i18n.check.check_catalogues with only enumeration narrowed to en/fr and this namespace. Catalogue loading, context schema checks, key checks, ICU/CLDR logic and argument parity remain the real repository APIs. A deliberate added-argument negative control was rejected. All five real messages.render_record calls in each locale returned the expected text with matching source provenance; no eligibility patch. Primary checkout's existing venv executed modules from this worktree (path asserted and printed).

French raw cell lengths in source order: **8, 19, 17, 17, 21**. Source context declares no maxCells/maxChars. Synthetic diagnostics apply the documented formula to a 31-cell pane with IDs openrouter (17-cell state budget) and ollama (21). The 19-cell environment label and 21-cell action ellipsize at 17; all fit 21. These are diagnostics, not a demonstrated settings layout defect: the actual settings filter only includes stored_credentials > 0, and openrouter is not a local-server example. Prefer the unambiguous full environment term and idiomatic action instead of dropping meaning to satisfy an invented universal cap. Pseudo samples are synthetic expanded copies, not translations or real UI coverage.

Isolated HOME/config and stripped CMUX environment are shown in the actual log. `env -i` is NOT proof of network isolation. No browser, native terminal, server, app startup, settings flow or rendered frame was exercised here. Plain Rich truncation is NOT real settings/native proof. Bounded wrapper reports cleanup=unknown under command-status; no stronger cleanup certification is claimed.

## Authority and remaining gates

This is an author bundle, NOT an independent audit. Status **translated** only. No auditor verdict, no passed ledger entry, no self-audit and no release/eligibility claim. Independent full-namespace A/B auditors must read committed source/context/target through the approved reader after the manager's private commit; this translator does not bypass uncommitted read refusal. No source/context/generated/history/ledger/baseline/cap/eligibility/consumer files changed. No staging, commit or push performed. Other locale files and original evidence bundles remain untouched.

This file is **audit_notes_not_execution_log**. Only validation.log records execution; record.json binds companions and inputs by SHA-256.
