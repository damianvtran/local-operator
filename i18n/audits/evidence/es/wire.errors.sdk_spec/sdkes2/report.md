### i18n audit — es — round 2
Reviewer: i18n-auditor openai/gpt-6.1-sol
Scope: cf5c90bad0d1..f990c30424e

Verdict: **clean** — **0 blocker, 0 major, 0 minor, 0 nit**. Terminal Spanish language review on head `f990c30424edb03801b1ba2325f27092572c3a27`.

Independent second auditor instance, session `8b3d0ec1d217`; actual model `openai/gpt-6.1-sol`, no fallback. Translator: i18n-translator `anthropic/claude-sonnet-5-5`. I reviewed **all 21 keys** against the English source/context, Spanish guide/glossary and sibling error copy, formed my view, then read the round-1 Spanish comment. No translation service/MT or catalogue edits.

#### Round-1 confirmation
- **ES1 fixed:** `approval_callback_gate_required` now says “función de aprobación invocable”, correctly describing the callable requirement.
- **ES2 fixed:** `yolo_approval_unreachable` now says “comprobación de aprobación”. This translator choice is valid: it names the policy check bypassed by the Boolean switch, rather than inventing a physical gate. It need not copy round 1’s suggested “función”. `sdk.py:_install_approval_policy` explicitly documents that the switch skips the gate object entirely; both auto-policy/drop-switch remedies retain their meaning.
- **ES3 fixed:** `spawn_model_pair_required` preserves literal `NEW`; “un único conjunto de parámetros iniciales” accurately conveys the atomic hosting/model sample.
- **ES4 fixed:** `spawn_fields_unsupported` names `dispose()` and `spawn_session()` as APIs, with the resource-release/same-id/resume sequence intact.
- **ES5 fixed:** `spawn_birth_effort_pair_required` explicitly describes initial parameters, all three values and the in-process alternative, without the previous calque.

#### Mechanical and fit evidence
Tiny read-only Python checks against **git-show bytes at the head** found JSON validity, 21/21/21 source/target/context key parity, exact simple-placeholder parity (`{mode}`, `{fields}`, `{formats}`, `{approval_mode}`), doubled apostrophe code quoting without stray single apostrophes, preserved identifiers/quoted code, and no literal redaction marker or manual bidi controls. No plural/select structures exist, so CLDR branch requirements do not arise. No markup/backticks occur to preserve.

The first nine codepoints of `yolo_approval_unreachable` are **U+0079 U+006F U+006C U+006F U+003D U+0054 U+0072 U+0075 U+0065**, verified individually rather than trusting masked display text.

Read `check.py` catalogue/argument checks and `runtime.py` apostrophe/simple-argument parser logic. One exploratory generic quote-span regex falsely paired English `owner`’s doubled apostrophe with later code quotes; explicit known-code-token matching then passed all 21 values. This was a probe defect, not a catalogue finding.

Spanish tú imperative register, accents and punctuation conform. No untranslated prose remains outside allowed runtime/technical literals. Context supplies **no maxCells/maxChars cap** for these diagnostics; longest value is 393 characters. No terminal glyph-width hazard identified from source inspection, but rendered wrapping is unverified.

#### Terminology precedents / authority
All sibling paths below are under `local_operator/i18n/catalogues/es/`:
- aprobación → `wire.errors.desktop.json:wire.errors.desktop.question_no_longer_pending`.
- sesión → `wire.errors.session_host.json:wire.errors.session_host.unavailable`.
- equipo → `wire.errors.agents.json:wire.errors.agents.cannot_reconstruct_team`.
- perfil/vincular → `wire.errors.sdk_startup.json:wire.errors.sdk_startup.profile_attach_failed`.
- modelo/configuración → `wire.errors.desktop.json:wire.errors.desktop.agents_config_client_fields`.
- propietario → `wire.errors.desktop.json:wire.errors.desktop.answer_epoch_mismatch`.
- campo → `wire.errors.agents.json:wire.errors.agents.system_prompt_required`.
- runtime / en este proceso / activar / conectar → `wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_runtime_missing`. This also supports “actívala después” in `spawn_output_unsupported`.
- Callable tokens → `wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_hosting_unconfigured` preserves `spawn_session()/deliver()`.
- herramienta → glossary §2; no sibling error occurrence found.
- No sibling full-phrase precedent exists for **función/comprobación de aprobación**, **parámetros iniciales**, **formato de salida/aplicación forzosa**, **especificación**, **cadena vacía** or **no admitidos**. Accepted by Spanish meaning/context, not fabricated precedent. `sdk.py:_model_sample` confirms the initial-parameter concept. Prose **gate** remains rejected by the Spanish kept-English policy; NEW/dispose()/other API literals follow the explicit source/task token contract.

#### Limits and receipt
**Not run:** repository validator, test suites, builds, installs, runtime formatting or rendering, per the host-overload restriction. This is a full independent semantic audit plus bounded string checks, not a validator/QA/render sign-off. French is outside my assigned locale.

Candidate receipt: `~/local-operator-ui/.worktrees/_s9-audits/delta/core-es.json` (all 21 per-key assessments, hashes, provenance, checks and limitations). No repo audit records/ledger edited.
English SHA-256: `a92bc479321cd0ee938d338cff05c1fced88590df3d43ec39d14f4d7b3cf5855`
Spanish SHA-256: `d3c6426e7842ae43906e674e6fe65d0cd1dc4475460195620d8dde8ce219ad44`

