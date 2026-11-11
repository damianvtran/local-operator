### i18n audit — fr — round 3 (final bytes)

Reviewer: i18n-auditor deepseek/deepseek-flash
Scope: dfc7411cac4..255b798d112

**Verdict: clean-with-findings — 0 blocker, 0 major, 2 minor, 2 nit. All 21 keys read against the English source and its context sidecar.**

Target `local_operator/i18n/catalogues/fr/wire.errors.sdk_spec.json` @ `255b798d112`, sha256 `e0b8217ba430c8d1b686f2bffc4581a3eef2eacc60071ae2e34c107bca76dd8e`; English source sha256 `a92bc479321cd0ee938d338cff05c1fced88590df3d43ec39d14f4d7b3cf5855`; sidecar `8735964058f901e1d97527801eb88f5a36585f0d7713c1e21f7f98ebaa55418e` (all over `git show` bytes).

**Mechanical — pass.** `python scripts/i18n/check.py --catalogues-only` → `check.py: catalogue checks: 0 problem(s)`. Independently: 21/21 keys both sides, no added/dropped key; `{mode}`, `{fields}`, `{formats}`, `{approval_mode}` byte-identical in name, type and count; no plural/select in this group; every value rendered through `local_operator.i18n.runtime.render_message(params, locale="fr")` — 0 errors, no surviving `''` (e.g. `mode_invalid` → `mode doit valoir 'own' ou 'attach', pas attach`). ASCII apostrophes occur only in doubled pairs; the two keys with fewer escapes are the dropped English possessives (`attach_fields_unsupported` 10→8: `owner''s` gone; `spawn_model_pair_required` 2→0: `root''s` gone). **Preserved tokens intact and in code form**: `NEW`, `ApprovalPolicy.*()`, `spawn_session()`, `dispose()`, `open_session`, `@latest`, `mode='own'`, `runtime`; the `yolo_approval_unreachable` value begins with codepoints `121,111,108,111,61,84,114,117,101` (`yolo=True`) — no `[` and no `redacted` substring in any value. **Typography**: all 16 occurrences of `; : ! ?` are preceded by U+202F; 10 U+2019 all elisions; `« U+202F@latest U+202F»`; 2 spaced em dashes; no U+00A0, no bidi controls, no leading/trailing/double spaces. **Register**: `vous` throughout, lowercase mid-sentence, no `Veuillez`.

**Findings**

| ID | Sev | Key | Issue | Suggested |
|---|---|---|---|---|
| F1 | minor | `spawn_fields_unsupported` | Two consecutive `à` phrases — `un état postérieur à l'ouverture à un nouveau processus enfant du runtime` — the dative has to close a group already opened by `postérieur à`; the reader backtracks to place `transmettre`'s destination. | `ne peut pas transmettre d'état post-ouverture à un nouveau processus enfant du runtime` |
| F2 | minor | `spawn_fields_unsupported` | The same attach operation is `attacher` here (`attachez ce dont vous avez besoin`, `ce qui a été attaché`) but `rattacher` in the three `attach_*` keys (`il ne rattache ni équipes…`, `un lecteur se rattache à un identifiant`). A reader cannot tell whether two different mechanisms are meant. | Align on the precedent verb: `rattachez ce dont vous avez besoin` … `resume restaure ce qui a été rattaché` (precedent: `fr wire.errors.sdk_startup.json:wire.errors.sdk_startup.profile_attach_failed`, `fr wire.exec.json:wire.exec.session_pointer`) |
| F3 | nit | `approval_callback_gate_required` | `appelable` (callable) qualifies a check rather than a function — mild category slip. The term `contrôle d'approbation` for approval gate is right. | optional: `exige une fonction de contrôle d'approbation appelable`; no meaning risk as written |
| F4 | nit | `attach_fields_unsupported` | `viewer` is `lecteur` here but `visionneuse` in the shipped TUI (`fr tui.viewer.json:tui.viewer.title` = `visionneuse de fichiers`). Different surfaces (SDK session handle vs file viewer), so this may be deliberate. | keep if deliberate; otherwise `visionneuse` |

No finding changes polarity, negation or safety advice: every remedy clause survives (`refuse()` for the undeclared bound session, `auto()`/drop `yolo`, the open → attach → dispose → spawn-same-id sequence in order, `laissez None`, `indiquez les trois`).

**Terminology accepted, with precedent**: `contrôle d'approbation` = approval gate (`fr cli.group_help.json:cli.group_help.operator`); `rattacher` = attach (`fr wire.errors.sdk_startup.json:profile_attach_failed`); `runtime` kept English (fr guide, wave-1 addition); `session`, `outil`, `modèle`, `identifiant`, `propriétaire` per `i18n/glossary.md` §2 and the fr siblings. `paramètres initiaux` / `niveau initial` (birth sample/level) and `contrainte de sortie` (output enforcement) are consistent within the group but glosses without corpus precedent — accepted. `spec` kept English alongside the `spec.resume` token (precedent `fr wire.errors.sdk_startup.json:tool_inventory_mismatch`). `session liée` (bound) is deliberately distinct from `session rattachée` (attached) — accepted.

**Not verified**: no rendered/terminal frame (these are SDK exception strings; sidecar declares no width cap, so no fit claim is made); `scripts/i18n/check.py --audit-base origin/main` not run — this audit writes no ledger row, at the requester's instruction; the SDK paths that actually raise these messages were not executed (`render_message` is the real formatting path, not the raise site); model id is taken from the session header.

**Independence**: I read `docs/I18N.md`, the fr style guide, the glossary, the English source, the sidecar and the fr siblings, and formed a view of all 21 values before opening any earlier receipt, PR comment or ledger row. No translation/MT service used; nothing edited.

Candidate receipt (outside the repo, no ledger/evidence entry created): `~/local-operator-ui/.worktrees/_s9-audits/final/core-fr.json`.

