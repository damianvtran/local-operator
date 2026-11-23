# Independent B French audit — tui.provider_state

Verdict: **passed**. Five of five keys reviewed independently; zero BLOCKER, MAJOR, MINOR or NIT findings. This is copy approval, not settings/native-font/whole-UI or release certification. No A report or A conclusion was read. The translation-author record fr22a/record.json was read solely for identity and independence.

Auditor: i18n-auditor, session 815a20325d52, actual model openai/gpt-6.1-sol. Host fallback from anthropic/claude-opus-5-5 due to provider quota happened before substantive audit. Translator: i18n-translator / e0c1b7382197 / openai/gpt-6.1-sol, as recorded by fr22a. Different sessions and independent authorship; shared model family is not shared authorship. Source author 93b7b02899af is also not this session.

## Native-fluent LLM judgment, every key

| Key suffix | Current French | Independent judgment |
|---|---|---|
| logged_in | connecté | Idiomatic terse completed authentication state. Matches stored successful login, not an instruction or fresh connectivity/health probe. No gratuitous success claim. |
| env_key | clé d’environnement | Compact French developer label for the environment-supplied credential. Here “environnement” expresses its source rather than an environment's name: authentication-state column and contrast with “connecté” disambiguate it. This is human copy, not an actual variable identifier; translating it is correct. “clé via une variable d’environnement” would be more explicit but unnecessarily long for this label; no rewrite requested. |
| needs_login | connexion requise | Standard factual requirement; avoids an imperative or a claim that authentication already succeeded. Correct contrast with “connecté”. |
| local_ready | serveur configuré | Correct completed configuration state; does not claim online/reachable/healthy, unlike “serveur disponible” or “opérationnel”. “serveur” agrees with configured technical vocabulary. |
| local_unconfigured | configurer le serveur | Natural infinitive instruction following French action-label convention. Distinguishes action from completed “serveur configuré”; the article is idiomatic, not a new semantic requirement. |

All five use compact lowercase fragments; accenting is correct; U+2019 in d’environnement follows the French style guide. No prose punctuation requiring U+202F occurs. No address pronouns, protected product names, commands or technical identifiers occur. No destructive or permission claim is introduced. No plural/select branch exists; French CLDR branch coverage is therefore not applicable rather than omitted.

## Mechanics and actual execution

The reproduction script and audit.log are actual execution artifacts, not reconstructed notes. The official `scripts.i18n.check.check_catalogues()` was exercised with enumeration restricted to en/fr and this namespace; no global checker or baseline/history/eligibility gate was run. It returned zero problems. In-memory missing-key and added-argument negatives each returned the expected real checker error; no catalogue bytes were edited. Catalogue loader, context schema and runtime ICU parsing/rendering ran on all five keys. Ordered keys match source and context, argument counts are zero, plural/select counts are zero. JSON loads, NFC, empty protected-token inventory, plain Rich rendering and English render controls passed. Both runtime.render_message and messages.render returned exact target/source text. Synthetic pseudo-expansion is a local stress transform, not MT or a claimed official pseudo-locale.

Invocation (from the worktree): `env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/b815a-home" LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/b815a-home/config" PATH="$PATH" PYTHONPATH="$PWD" TERM=xterm-256color /Users/damian/local-operator/.venv/bin/python i18n/audits/evidence/fr/tui.provider_state/b815a/script.py.txt > i18n/audits/evidence/fr/tui.provider_state/b815a/audit.log 2>&1`. Exit 0. Empty inherited environment strips CMUX; isolated HOME/config used only for the Python execution, not Git. Log proves primary interpreter and worktree module origins. An earlier source-discovery rg named nonexistent modules and exited 2; no validator was executed by that discovery command, and no validation harness failure was hidden.

## Fit: measured, not invented limits

English/French cell widths in ordered-key order: 9/8, 7/19, 11/17, 17/17, 16/21. All French strings are NFC and have identical character/cell counts. No fixed maxCells is present. The supplied example 31-cell pane beside `openrouter` leaves 17 state cells: env_key is 2 cells longer and local_unconfigured 4 longer. This identifies potential truncation, not a proven blocking overflow. Existing layout truncates the state segment; consumers and IDs differ. Retain these native labels and defer actual styled geometry/legibility validation to integration, rather than opening another copy cycle. The pseudo-expansions measure 12, 26, 27, 25, 31 cells; they are stress evidence only. No real settings frame, terminal font, Textual layout, app startup or browser was exercised.

## Scope and immutable inputs

Audited revision a54ec02de48a94b68d6c3dc92e1307c97ea776ad; English/context/French hashes match the task and remain identical before/after (see audit.log and record.json). Only this new canonical B bundle was written. No target/source edits, staging, commit, ledger/history apply, generation or eligibility changes. Ledger receipt is a direct row conforming to the actual append input contract, not an applied ledger entry.
