### i18n audit — es — round 1
Reviewer: i18n-auditor openai/gpt-6.1-sol
Scope: dfc7411cac4..cf5c90bad0d1

Verdict: **changes_requested** — 0 blocker, 4 major, 1 minor, 0 nit. Independently reviewed all 21 Spanish keys. Actual model: openai/gpt-6.1-sol; no fallback. Translator: i18n-translator anthropic/claude-sonnet-5-5.

**ES1 — MAJOR — `wire.errors.sdk_spec.approval_callback_gate_required`
- Current: ApprovalPolicy.callback() requiere un gate invocable
- Evidence/problem: Untranslated prose gate is not an allowed kept-English term and is not the API parameter name (callback takes handler). No wire.errors sibling uses gate. Translate the concept, preserving the API.
- Suggested value: ApprovalPolicy.callback() requiere una función de aprobación invocable
- Terminology authority/precedent: wire.errors.desktop.json:wire.errors.desktop.question_no_longer_pending uses aprobación; no sibling gate precedent; es style guide kept-English list excludes gate.

**ES2 — MAJOR — `wire.errors.sdk_spec.yolo_approval_unreachable`
- Current: yolo=True hace inalcanzable un gate ApprovalPolicy.{approval_mode}(); usa ApprovalPolicy.auto() (o quita yolo) en su lugar
- Evidence/problem: Untranslated prose gate repeats ES1; the restored technical Boolean-assignment token itself is exact and must stay untouched.
- Suggested value: yolo=True impide que se ejecute la función de aprobación de ApprovalPolicy.{approval_mode}(); usa ApprovalPolicy.auto() (o quita yolo) en su lugar
- Terminology authority/precedent: wire.errors.desktop.json:wire.errors.desktop.question_no_longer_pending uses aprobación; no sibling gate precedent.

**ES3 — MAJOR — `wire.errors.sdk_spec.spawn_model_pair_required`
- Current: spawn_session() envía hosting/model a un runtime NUEVO como una sola muestra de creación; indica ambos o ninguno (el hijo resolverá entonces su modelo a partir de la configuración de la raíz).
- Evidence/problem: NUEVO translates NEW despite this key’s explicit sidecar/user preserved-token contract. Existing other-locale translations are not authority to waive this requirement.
- Suggested value: spawn_session() envía hosting/model a un runtime NEW como un único conjunto de parámetros iniciales; indica ambos o ninguno (el hijo resolverá entonces su modelo a partir de la configuración de la raíz).
- Terminology authority/precedent: wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_runtime_missing endorses runtime; NEW requirement comes from en sidecar, not sibling wording.

**ES4 — MAJOR — `wire.errors.sdk_spec.spawn_fields_unsupported`
- Current: spawn_session() no puede entregar el estado posterior a la apertura a un nuevo runtime hijo; campos no admitidos en esta especificación: {fields}. Abre la sesión en el proceso actual (open_session), conecta lo que necesites, haz dispose y luego llama a spawn_session() con el mismo id: resume restaura lo conectado; o bien haz spawn sin estos campos.
- Evidence/problem: haz dispose / haz spawn are untranslated English verbs in Spanish prose, not exact callable identifiers. This is neither native diagnostic copy nor covered by the kept-English list. Name the actual API for the remedy.
- Suggested value: spawn_session() no puede transferir el estado configurado tras la apertura a un nuevo runtime hijo; campos no admitidos en esta especificación: {fields}. Abre la sesión en el proceso actual (open_session), conecta lo que necesites, libera sus recursos con dispose() y luego llama a spawn_session() con el mismo id: resume restaura lo conectado; o bien llama a spawn_session() sin estos campos.
- Terminology authority/precedent: wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_hosting_unconfigured preserves spawn_session()/deliver() and uses conéctate; no sibling dispose or prose spawn precedent. sdk.py:935-950 identifies dispose() as the exit operation.

**ES5 — MINOR — `wire.errors.sdk_spec.spawn_birth_effort_pair_required`
- Current: birth_effort de una sesión nueva creada con spawn viaja con la muestra de creación y necesita el par hosting/model a su lado; indica los tres, o abre la sesión en el proceso actual (open_session) para fijar un nivel inicial sin par de modelo.
- Evidence/problem: muestra de creación is decipherable but viaja con / a su lado is an English calque and obscures that the three fields form the initial configuration; use a concrete description.
- Suggested value: En una sesión nueva creada con spawn_session(), birth_effort forma parte de los parámetros iniciales y requiere también el par hosting/model; indica los tres valores, o abre la sesión en el proceso actual (open_session) para fijar un nivel inicial sin especificar ese par.
- Terminology authority/precedent: No wire.errors sibling birth-sample precedent. sdk.py:_model_sample documents provider/model_id/reasoning_effort as one initial sample; wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_runtime_missing uses en este proceso.

#### Checks and limitations
- All 21 keys read against English/context and Spanish guide/glossary. Scoped repository check.check_catalogues() for en/es sdk_spec executed: 0 problems. Full tree validator not run under host resource restriction.
- Actual shipped runtime.message_arguments parsed all English/Spanish values and matched argument names/kinds. No plurals/selects or markup in this group. No single unpaired ASCII apostrophes, manual bidi controls, or literal redaction artifacts in target. All listed technical tokens match except NEW (ES3).
- attach_fields_unsupported English/Spanish doubled-apostrophe pairs 5/4; spawn_model_pair_required 1/0. Actual runtime renders succeed: omitted owner/root possessives require no apostrophe in Spanish; quoted attach/own tokens remain literal.
- All five named existing non-English targets (zh-CN, ru, vi, ur, hi) checked only for argument/token presence, not wording. Brief says six but enumerates five; English is sixth if counted. Their argument signatures match; all five also lack literal NEW. That does not waive the explicit current target token requirement.
- Token equality was verified from git-show bytes, not tool-displayed text: Spanish begins exact yolo=True; neither English nor Spanish contains literal [redacted]. Tool display redaction masks the token, but Python equality tests return True.
- No maxCells/maxChars cap in sidecar; full SDK diagnostics may wrap, not fixed button/tab copy. Spanish tú imperatives and colon/semicolon spacing comply. No terminal-width/rendered UI claim made.
- Caller contract is not moot: session/spec.py:84-92 and all 21 SDK call sites explicitly pin envelope(locale="en"). This catalogue-only diff cannot localize those existing guard exceptions or alter approval policy. However messages.py:109-154 explicit es/fr envelope/render calls can now return translated text instead of English fallback. Existing translations do not establish a caller migration.
- Accepted terminology: runtime and id / conecta / sesión from wire.errors.sdk_startup.json:attach_runtime_missing and attach_hosting_unconfigured; ruta from wire.errors.desktop.json:claim_origin_not_plain; cadena from wire.errors.desktop.json:claim_origin_not_string; aprobación from wire.errors.desktop.json:question_no_longer_pending. herramienta and equipo follow glossary §2 (no wire.errors sibling herramienta precedent); modelo likewise follows glossary. No wire.errors sibling precedent for especificación, formato de salida, sesión vinculada, birth sample, gate or dispose; do not invent one.
- Translator doubts: Spanish muestra de creación warrants ES5 clarity improvement; NUEVO violates explicit NEW contract (ES3). runtime accepted; prose gate rejected (ES1/ES2); prose dispose/spawn rejected in favor of API identifiers (ES4). French wording judgments are outside assigned locale; no French audit/sign-off claimed.
- This is independent round 1, not the requested second independent high-visibility pass. A different auditor must supply that pass. No translation service or MT used; no catalogue edits.

Candidate receipt: `/Users/damian/local-operator-ui/.worktrees/_s9-audits/core-es.json`
English SHA-256: `a92bc479321cd0ee938d338cff05c1fced88590df3d43ec39d14f4d7b3cf5855`
Spanish SHA-256: `d95b92040e7b5aee30f8c10001f87208b279953909281df9c8f5975f0d6cecba`

No catalogue or ledger edits. This comment is round 1 only; do not treat it as the second independent high-visibility pass.

