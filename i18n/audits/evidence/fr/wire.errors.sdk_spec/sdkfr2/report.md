### i18n audit — fr — round 2
Reviewer: i18n-auditor openai/gpt-6.1-sol
Scope: cf5c90bad0d1..f990c30424e
Verdict: clean-with-findings — 0 blocker, 0 major, 3 minor, 0 nit. Terminal; typography follow-ups do not require another round.

Audited ALL 21 French keys at f990c30424edb03801b1ba2325f27092572c3a27 against English + context, fr style guide/glossary and sibling wire.errors terminology. Session 8b6d46ca9257; actual model openai/gpt-6.1-sol, no fallback. Independent of translator i18n-translator / anthropic/claude-sonnet-5-5 and round-1 auditor. Formed my own all-key view before reading round 1.

Round-1 confirmations: FR1 fixed (literal NEW preserved); FR2 fixed in both spawn_birth_effort_pair_required and spawn_model_pair_required (paramètres initiaux replaces the creation-sample calque); FR3 fixed (puis confiez-lui une tâche). Revised meanings and remedies are accurate; vous register is maintained. No untranslated English prose found beyond approved technical vocabulary/identifiers.

Non-gating findings, all French style-guide typography, not semantic defects:

**FR2-T1 — MINOR — `wire.errors.sdk_spec.spawn_birth_effort_pair_required`**
Current: birth_effort d’une nouvelle session lancée fait partie des paramètres initiaux et exige que le couple hosting/model soit également fourni ; indiquez les trois, ou ouvrez la session dans le processus courant (open_session) pour fixer un niveau initial sans couple de modèle.
Problem: ordinary U+0020 before prose semicolon (offset 138); fr guide requires U+202F.
Suggested full value (not applied): birth_effort d’une nouvelle session lancée fait partie des paramètres initiaux et exige que le couple hosting/model soit également fourni ; indiquez les trois, ou ouvrez la session dans le processus courant (open_session) pour fixer un niveau initial sans couple de modèle.

**FR2-T2 — MINOR — `wire.errors.sdk_spec.spawn_model_pair_required`**
Current: spawn_session() transmet le couple hosting/model à un runtime NEW dans un même ensemble de paramètres initiaux ; indiquez les deux, ou aucun (l’enfant résout alors son modèle depuis la configuration de la racine).
Problem: ordinary U+0020 before prose semicolon (offset 111); fr guide requires U+202F.
Suggested full value (not applied): spawn_session() transmet le couple hosting/model à un runtime NEW dans un même ensemble de paramètres initiaux ; indiquez les deux, ou aucun (l’enfant résout alors son modèle depuis la configuration de la racine).

**FR2-T3 — MINOR — `wire.errors.sdk_spec.spawn_output_unsupported`**
Current: spawn_session() ne peut pas porter de contrainte de sortie ({fields}) ; ouvrez la session avec open_session(..., mode=''own'') puis confiez-lui une tâche
Problem: ordinary U+0020 before prose semicolon (offset 70); fr guide requires U+202F.
Suggested full value (not applied): spawn_session() ne peut pas porter de contrainte de sortie ({fields}) ; ouvrez la session avec open_session(..., mode=''own'') puis confiez-lui une tâche

Mechanical checks: valid JSON; exactly 21 source/target keys; all per-key simple-string placeholders match ({mode}, {fields}, {formats}, {approval_mode}); every ASCII apostrophe occurs as a doubled escape. Read check.py argument/plural validation and runtime.py apostrophe parsing: no plural/select constructs to require French one/many/other branches; no stray apostrophes or unmatched placeholder braces. No markup/backticks introduced. No manual bidi controls. No sidecar maxCells/maxChars caps; long SDK prose is not assigned an invented UI budget. No rendering/overflow proof claimed.

The nine initial yolo-token codepoints are **U+0079 U+006F U+006C U+006F U+003D U+0054 U+0072 U+0075 U+0065** (decimal 121,111,108,111,61,84,114,117,101). Literal redaction-marker count in target values: **0**. Display masking is not a source defect.

Terminology decisions (all filenames below under local_operator/i18n/catalogues/fr/):
- approbation: wire.errors.desktop.json:wire.errors.desktop.question_no_longer_pending.
- runtime / racine / identifiant / rattacher: wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_hosting_unconfigured.
- processus / spawn_session / mode own: wire.errors.sdk_startup.json:wire.errors.sdk_startup.attach_runtime_missing.
- profil: wire.errors.sdk_startup.json:wire.errors.sdk_startup.profile_attach_failed.
- équipe: wire.errors.desktop.json:wire.errors.desktop.agents_config_local_only.
- modèle / configuration: wire.errors.desktop.json:wire.errors.desktop.agents_config_client_fields.
- tâche: wire.errors.agents.json:wire.errors.agents.job_cannot_cancel.
- session: wire.errors.session_host.json:wire.errors.session_host.unsupported.
- outil: No sibling wire.errors precedent found; glossary §2 tool → outil.
- paramètres initiaux / contrainte de sortie / contrôle d’approbation appelable / lecteur / objectif / libérer / restaurer: No sibling wire.errors precedent found; accepted contextual native French, not attributed to a shipped precedent.
- NEW / remaining API names and fields: English context sidecar + glossary §1 preserved identifiers; no sibling NEW precedent claimed.

No term rejected on meaning/register/glossary grounds. New contextual phrases above are accepted on French reasoning, not invented precedent. No Spanish audit is claimed.

Unverified: full repo validator, runtime render execution, rendered frames, caller/exception localization and full test suites. These were not run under the explicit overloaded-host restriction. Tiny read-only Python JSON/string checks were executed; an initial overly repetitive git-show sibling scan timed out, then the sibling JSON files were read once locally. No build/install/browser or translation service used. Source content unchanged; no ledger or repo audit records written.

Candidate receipt: `~/local-operator-ui/.worktrees/_s9-audits/delta/core-fr.json` (not committed; not a ledger pass).
Target git-show SHA-256: `d758be6d6f429af51854e2aacf702239f05eed54c5724565d4f43380cec9387e`.
English git-show SHA-256: `a92bc479321cd0ee938d338cff05c1fced88590df3d43ec39d14f4d7b3cf5855`.
Audit UTC: 2026-11-10T01:58:05.751396+00:00.

