# Urdu provider-state author notes — translated, not audited

These are `audit_notes_not_execution_log`. Actual validation output is in
`audit.log`; the exact reproduction source is retained as `validator.py.txt`.
No independent audit was performed by this translator.

## Native choices and scope

All five labels were composed by direct LLM reasoning, without a translation
API, SDK, MT service, website, or another agent. Urdu is the only authored locale.
The English source author is session `93b7`, not this translator. The approved
English catalogue/context readers supplied the source and role semantics.
The Urdu style guide and shared glossary were present and read. Existing
committed Urdu `relay.login.json` uses `سائن ان`; this is retained rather than
introducing a second login term. `cli.login_status_help.json` was also read
through the approved reader for existing credential terminology.

- `logged_in` → **سائن ان ہے**: a terse completed/current signed-in status,
  not an instruction. The context defines it as a stored login credential,
  not an assertion of current server reachability.
- `env_key` → **ماحولیاتی متغیر سے کلید**: explicitly identifies the
  environment variable as the credential source. `سے` means "from"; it is
  not an environment name or a literal variable identifier. The shorter
  "environment key" calque would obscure the source distinction.
- `needs_login` → **سائن ان درکار ہے**: "sign-in is required", factual status;
  not the imperative `سائن ان کریں`.
- `local_ready` → **سرور کی ترتیب مکمل**: "server configuration complete";
  reports configuration only, not health, connection, or authentication.
- `local_unconfigured` → **سرور ترتیب دیں**: respectful instruction to
  configure the server, visibly different from the completed-state wording.

### Glossary proposals (not glossary edits)

These namespace terms have no binding row in the glossary excerpts read:
`sign in` → `سائن ان` (already used by committed Urdu relay login),
`environment variable` → `ماحولیاتی متغیر`, credential `key` → `کلید`,
`server` → `سرور`, configuration noun → `ترتیب`, and configure instruction
→ `ترتیب دیں`. These are proposed consistent terminology, not a glossary change.

## Mechanical validation and limitations

The official `scripts.i18n.check.check_catalogues()` was invoked with discovery
scoped in memory to English/Urdu and `tui.provider_state`; no checker rule was
suppressed or changed. The real catalogue loader and context schema checker
were used. In-memory missing-key and added-argument mutations were both rejected.
The official ICU parser/renderer and `messages.render_record` returned all five
Urdu strings with Urdu provenance. No admission, generated-file, ledger, or
eligibility modifications were made. This does not certify those global gates.

All strings have zero arguments, zero plural/select branches, NFC normalization,
and no manual bidi controls. There are no protected command/URL/markup/backtick
or other token payloads in this English namespace; the zero-token check is
explicit rather than an invented preservation claim. IDs and source context
were not changed.

Plain/Rich cell counts in key order: **10, 23, 16, 18, 14**. The 31-cell pane
beside `openrouter` implies an illustrative 17-cell state budget, not a universal
cap. The explicit environment-source label (23) and configuration-complete status
(18) exceed that illustration; no context `maxCells` exists. Keeping the
credential-source and configuration/status distinctions explicit was preferred
to truncating authored text. Consumers still own fitting. English and expanded
synthetic pseudo controls were measured in the same rig.

All five targets contain zero nonzero-combining-class codepoints in the measured
strings. This does NOT mean Urdu has no joining, shaping, bidi, or font risks:
logical-order Arabic-script letters still need contextual shaping and bidi-aware
painting. Storage is logical RTL without manual controls per the style guide;
Latin provider IDs remain outside the labels and unchanged. Pure Rich cell
measurements do not certify native glyph shaping, Nastaliq font metrics, actual
mixed-direction layout, or style behavior. No app, browser, server, native font
capture, or independent language audit was run. The current UI producer remains
English; this authoring bundle is not a fixed-UI or released-product claim.

## Provenance

Translator: `i18n-translator`, session `ab6b56b73e1e`. The host-delivered fresh
model-switch notice in this transcript states the request is served by
`openai/gpt-6.1-sol`, temporarily falling back from
`anthropic/claude-opus-5-5` due to `provider failure: quota`. That notice, not a
configured model header, is the model provenance source. No additional served
request telemetry was available or independently fetched; provider request IDs
are unknown. Independent auditor identity/verdict remain unassigned.

No stage, commit, push, ledger/history edits, generated changes, source/context
edits, other-locale edits, consumer edits, or model configuration edits were made.
The manager must commit the complete target set before approved-reader audit
passes. An attempted non-existent/uncommitted `ur/tui.auth.json` read was refused;
it was not bypassed, and committed Urdu login copy was used instead.
