### i18n audit — fr — round 1
Reviewer: i18n-auditor openai/gpt-6.1-sol
Scope: dfc7411cac4..cf5c90bad0d1
Verdict: changes_requested — 0 blocker, 2 major, 1 minor, 0 nit

Independent of translator i18n-translator (anthropic/claude-sonnet-5-5). Actual auditor session e94e8375895d; no model fallback. Audited all 21 French keys against English + sidecar, fr style/glossary and shipped fr sibling terminology. UTC audit date: 2026-11-10.

**FR1 — MAJOR — `wire.errors.sdk_spec.spawn_model_pair_required`**

Current: spawn_session() transmet hosting/model à un NOUVEAU runtime comme un seul échantillon de création ; indiquez les deux, ou aucun (l’enfant résout alors son modèle depuis la configuration de la racine).

The sidecar explicitly preserves the literal NEW; NOUVEAU does not preserve it. All five existing non-English peers also translate this marker, but precedent is not a waiver of the explicit preservation contract. No guard-side regression is claimed.

Evidence: en/wire.errors.sdk_spec.context.json: spawn_model_pair_required, note: Preserve technical tokens: spawn_session(), hosting/model, NEW. Tiny byte check: rendered French is missing NEW.

Suggested value (not applied): spawn_session() transmet le couple hosting/model à un runtime NEW dans un même ensemble de paramètres initiaux ; indiquez les deux, ou aucun (l’enfant résout alors son modèle depuis la configuration de la racine).

Terminology precedent: fr/wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_runtime_missing accepts runtime and spawn_session(); no sibling preserves NEW in this sense; NEW follows this key’s explicit sidecar.

**FR2 — MAJOR — `wire.errors.sdk_spec.spawn_birth_effort_pair_required`**

Current: birth_effort d’une nouvelle session lancée voyage avec l’échantillon de création et exige le couple hosting/model à ses côtés ; indiquez les trois, ou ouvrez la session dans le processus courant (open_session) pour fixer un niveau initial sans couple de modèle.

“voyage avec l’échantillon de création” is a literal calque, not natural French developer copy. An “échantillon” normally means a sample/subset, whereas sdk.py describes the initial provider/model/effort selection carried together to the child. The companion model-pair key repeats the same unsupported rendering. Translate the concept as initial parameters, not a creation sample.

Evidence: local_operator/sdk.py:1163–1170 and 1182–1204 describe WarmErrand carrying the birth sample/provider+model choice and enforcing the complete pair.

Suggested value (not applied): birth_effort d’une nouvelle session lancée fait partie des paramètres initiaux et exige que le couple hosting/model soit également fourni ; indiquez les trois, ou ouvrez la session dans le processus courant (open_session) pour fixer un niveau initial sans couple de modèle.

Related `wire.errors.sdk_spec.spawn_model_pair_required` suggestion: spawn_session() transmet le couple hosting/model à un runtime NEW dans un même ensemble de paramètres initiaux ; indiquez les deux, ou aucun (l’enfant résout alors son modèle depuis la configuration de la racine).

Terminology precedent: No échantillon/birth-sample precedent in any other fr wire.errors.* file. fr/wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_hosting_unconfigured establishes racine and runtime; fr/wire.errors.desktop.json:wire.errors.desktop.agents_config_client_fields establishes modèle/configuration. Suggested paramètres initiaux is a contextual rendering, not a claimed shipped glossary entry.

**FR3 — MINOR — `wire.errors.sdk_spec.spawn_output_unsupported`**

Current: spawn_session() ne peut pas porter de contrainte de sortie ({fields}) ; ouvrez la session avec open_session(..., mode=''own'') puis mettez-la ensuite au travail

“mettez-la ensuite au travail” is understandable and idiomatic as a metaphor, but overly anthropomorphic for a terse SDK diagnostic; puis/ensuite are also redundant. Meaning remains recoverable, so this is non-gating.

Evidence: English remedy is “open the session ... and engage it afterwards”, not a change to session ownership.

Suggested value (not applied): spawn_session() ne peut pas porter de contrainte de sortie ({fields}) ; ouvrez la session avec open_session(..., mode=''own'') puis confiez-lui une tâche

Terminology precedent: fr/wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_runtime_missing uses ouvrez-le ... pour l’exécuter dans ce processus. No sibling requires the anthropomorphic phrase.

**Mechanical execution:** scoped repository `scripts.i18n.check.check_catalogues()` (only en/fr + this namespace) returned `[]`; 21/21 exact keys and placeholder names/types; every target string parsed/rendered with the actual repository ICU runtime and sentinel parameters. No plural/select constructs, no manual bidi controls, no stray ASCII apostrophe, no bad French punctuation spacing. The two reduced doubled-apostrophe counts are valid removal of English possessives, not broken ICU. Sidecars impose no width caps. No builds, installs, test suite, or browser run under the resource hold.

**Redaction recheck:** raw `git show` bytes, not displayed output, contain zero literal redaction artifacts in en/fr/es; all three contain the exact ASCII `yolo=True` token. Output-display redaction is not a catalogue defect. Rendered code-token checks identify `NEW` as missing; other checked code examples retain their ASCII form.

**Terminology accepted:** runtime/racine/identifiant/rattacher and spawn_session()/mode='own' follow `fr/wire.errors.sdk_startup.json:attach_hosting_unconfigured, attach_runtime_missing`; approbation follows `fr/wire.errors.desktop.json:question_no_longer_pending`; modèle/configuration follow `fr/wire.errors.desktop.json:agents_config_client_fields`; range/enumeration/empty diagnostics follow `fr/wire.errors.files.json:sign_ttl, sign_route, sign_path`. Tool/équipe follow the glossary; profil follows sdk_startup:profile_attach_failed. No sibling precedent exists for birth sample, callable gate or dispose; I do not invent one. Callable-gate French is acceptable developer terminology; dispose is a preserved method identifier, not English prose. The receipt enumerates these decisions.

**Caller contract:** The old allowance rationale is not moot merely because five other non-English locales carry the namespace. Sidecar still requires English exception bytes. All 21 SDK guard sites in sdk.py/session/spec.py still explicitly use envelope(locale="en"), so adding fr/es text alone does not localize their existing raised exceptions. I independently rendered mode_invalid with explicit en and fr: the en result stays English and the fr result changes to French. Explicit-locale messages.render/envelope consumers and the eager wire catalogue can now expose localized text rather than English fallback; wire_namespaces(fr) includes this namespace and it is not deferred, so catalogue content/cache identity changes too. Publishing translations is not a caller-contract migration.

**Translator doubts:** birth sample and mettez-la au travail are addressed above. `NEW` is a shared explicit literal requirement, not a French/Spanish stylistic preference. Spanish gate/dispose/spawn prose needs the Spanish auditor: this audit does not certify Spanish wording. `runtime` is explicitly retained by the French style guide; code identifiers stay English.

**Peer comparison:** the brief names five existing non-English locales (zh-CN, ru, vi, ur, hi), not six. I compared their placeholders/token presence only; each has 21 keys and matching argument signatures. All five also lack literal NEW, but existing departures cannot override the requested sidecar contract.

Candidate receipt (not committed, not a ledger entry): `~/local-operator-ui/.worktrees/_s9-audits/core-fr.json`.
Target SHA-256: `e50078912a982dd2bb1ee9d77874cd28b8b2899dffa866e5db70e5711106ebe4`.
English SHA-256: `a92bc479321cd0ee938d338cff05c1fced88590df3d43ec39d14f4d7b3cf5855`.

No translations or repo audit records edited. This is one independent pass; a separately assigned high-visibility second pass is not supplied by this report. Major findings remain unresolved.

