# Independent Urdu audit A — tui.provider_state

Date: 2026-11-22. Scope: all five ordered labels, worktree HEAD 2b924d84f0e663302fd93a7b36411ee6035576fe. Verdict: passed; zero BLOCKER/MAJOR, no findings. This report is audit_notes_not_execution_log; audit.log is actual command output.

## Native Urdu LLM judgment — every key

1. `logged_in` — **سائن ان ہے**: natural terse software Urdu for an existing signed-in state. The copula makes this a status, not an instruction; it does not add a new successful authentication action or reachability claim. Fits stored-login context.
2. `env_key` — **ماحولیاتی متغیر سے کلید**: technical Urdu noun phrase explicitly saying the key comes from an environment variable. ماحولیاتی متغیر is intelligible computing terminology in this context; سے identifies the source rather than translating an opaque variable name. No provider or variable identifier appears in the label. The English source's terse “env key” is not a literal identifier that must be preserved.
3. `needs_login` — **سائن ان درکار ہے**: idiomatic requirement status, neither falsely claiming success nor turning into an imperative. Reuses سائن ان consistently with logged_in.
4. `local_ready` — **سرور کی ترتیب مکمل**: compact completed-configuration status; مکمل modifies configuration, not network health, authentication or server reachability. Local-provider context supplies “local”, as in the English label; no extra qualifier is necessary.
5. `local_unconfigured` — **سرور ترتیب دیں**: natural compact respectful configure-server instruction. دیں supplies the required polite imperative and clearly differs from completed-state مکمل. Everyday Urdu technical vocabulary, no Hindi script/register or ornamental Arabic/Persian wording.

All five are intentionally unpunctuated short labels, with logical Urdu word order and no manually authored bidi controls. No Latin ID is embedded in these labels; synthetic /provider islands were tested separately at the existing opt-in display boundary, not inserted into catalogue text. Style guide آپ/register and Urdu punctuation policy are respected; an explicit آپ is unnecessary in these terse labels. Glossary contains no binding alternate rendering for these five terms; settled سرور/کلید and the consistent سائن ان are appropriate.

## Mechanical and fit evidence

Official scripts.i18n.check.check_catalogues executed with discovery scoped to en/ur and this namespace only; validation logic unchanged. Result []. Missing-key and extra-argument negative controls were actually detected. Catalogue/context ordered keys exactly match (5), zero arguments, no plural/select AST nodes; CLDR branch requirement is inapplicable, not waived. NFC/control-free, valid loader JSON, protected tokens absent, ICU rendering, plain Rich/markup interpretation, pseudo English control, unchanged English projection and logical RTL/LTR-island checks all passed. Actual primary-venv interpreter and worktree runtime module path, isolated HOME/config and stripped CMUX environment are in audit.log. Git/ledger commands below run with normal HOME.

Measured cells: 10, 23, 16, 18, 14. These are observations, NOT a global cap or native shaping/layout proof. Sidecar sets no fixed maxCells; actual integration must validate per-consumer truncation/width and styled frames. No copy rewrite justified by these measurements.

## Provenance and limits

Auditor: i18n-auditor/provider-state-ur-audit-A, session 3ba03317d047. Host model-switch notice reports openai/gpt-6.1-sol after anthropic/claude-opus-5-5 quota failure. No independent per-response served-model telemetry is available: unknown. Native LLM reasoning only; no MT/translation API, no self-translation, no other auditor conclusions read. Translator identity was read from ur-author/record.json: i18n-translator, ab6b56b73e1e, openai/gpt-6.1-sol; distinct session, no authored translation edits here.

No native glyph/shaping/terminal proof, no global Urdu clearance, no locale admission or release clearance. Existing provider producer remains English until a future coded resolver; this audit does not claim a UI repair. No application startup, global checker, full tests, native/browser/server/build, installs or other worktree. Source/target/context/history/generated files untouched. Ledger receipt and append dry-run are proposals only; no history applied. Manager must collect all independent passes before code/admission.
