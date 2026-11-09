# fr · cli.permissions — independent audit frpm1 (REPORT)

- PR: https://github.com/damianvtran/local-operator/pull/3691 · worktree `.worktrees/runtime-privacy-identity` @ 50badda034d755ebb049c4f542f3073439bf823e
- Auditor: i18n-auditor (lopdev team, task subagent), session 144ca81a304c, model anthropic/claude-opus-5-5 (the session's own model header; no fallback observed). Independent of the translator (coder, job ad99821ab9e1). No MT/translation service used; no catalogue edited.
- Read: docs/I18N.md "Add or extend a locale" 2–6, i18n/glossary.md, i18n/style-guides/fr.md, i18n/audits/evidence/README.md, en source + context sidecar, fr target, and local_operator/permissions_cli.py (plain `print`/`input`, answers parsed as y/yes only).
- Verdict: **changes_requested** — 1 MAJOR, 3 MINOR, 3 NIT; 0 BLOCKER.

## Mechanical (audit.log is the execution record)
28/28 keys, none added/dropped; placeholders and backtick spans byte-identical; no ICU plural/select in this namespace; `[y/N]` and the trailing space kept on both prompts; `check.py --catalogues-only`: 0 problems. No `maxCells` declared; lines are plain `print` output, so fit is not a gate (longest: setup.explain 300 cells, wraps).

## Accuracy / safety
What the user is told to click is correct: « Réglages Système > Confidentialité et sécurité > Enregistrement de l’écran » and « Toujours autoriser » match the macOS fr labels for the names the English source uses; « trousseau » is macOS’s term. Polarity is preserved in cancelled/failed/no_identity/done (nothing changed; runtimes keep the Runtime row; running sessions keep their image until restart). No meaning errors found. Observation (source-level, not a fr finding): the English names the pre-Sequoia pane « Screen Recording »; fr mirrors it faithfully.

## Findings
### FRPM-1 — MAJOR — 27 of 28 keys (all except cli.permissions.setup.refused.tty)
- Current: ASCII apostrophe U+0027 in 22 keys and U+0020 before : ; ? » / after « in 20 keys; zero U+2019 and zero U+202F in the namespace (e.g. "Signature d'équipe : activée", "« Local Operator »", "Continuer ? [y/N] ")
- Problem: Binding fr style guide (Punctuation & typography) requires ’ and U+202F before : ; ! ? and inside « »; the rest of the fr corpus complies (1 653 ’ / 1 007 U+202F). Systematic, not a stray site (contrast wire.settings le28r2 F3: 4 stray sites among 177, graded minor), so this whole CLI reads untypeset next to every other fr surface.
- Suggested: Replace U+0027→U+2019 outside code spans and the U+0020 before :;?!» and after « with U+202F; never inside backticks, guide://permissions or [y/N]. Full corrected values in report.md §Suggested values.

### FRPM-2 — MINOR — help.setup, status.identity.team, status.optin.off, status.app.found, status.app.missing, setup.no_app, setup.explain, setup.done
- Current: app de bureau / App de bureau / d'app
- Problem: Every other fr namespace (relay.agent_avatar, relay.session.tags, tui.team_authoring, wire.settings, wire.settings.keymap: 8 occurrences) says « application de bureau »; this namespace alone says « app de bureau ». No width budget forces the short form.
- Suggested: application de bureau (Application de bureau : {path} (équipe {team}); …aucune autorisation d’application à partager.)

### FRPM-3 — MINOR — cli.permissions.setup.explain
- Current: Cette étape signe le runtime … macOS peut demander une fois si codesign peut utiliser cette clé.
- Problem: ‘This’ is the command the user just ran, not a step; ‘peut demander … si codesign peut’ repeats pouvoir and reads as a calque of ‘may ask whether … may’.
- Suggested: Cette commande signe le runtime avec « {name} » pour que les runtimes partagent les autorisations de confidentialité de l’application de bureau. macOS vous demandera peut-être une fois d’autoriser codesign à utiliser cette clé. Choisissez « Toujours autoriser » pour que les prochains `lop update` puissent signer sans demander.

### FRPM-4 — MINOR — cli.permissions.setup.refused.isolated
- Current: … Lancez-la depuis votre propre shell de connexion.
- Problem: ‘-la’ grammatically binds to « la signature d’équipe » — one does not « lancer une signature »; the English ‘it’ is the command.
- Suggested: La signature d’équipe n’est pas autorisée dans ce processus ({reason}). Lancez la commande depuis votre propre shell de connexion.

### FRPM-N1 — NIT — cli.permissions.help.permissions
- Current: Inspecter ou configurer…
- Problem: ‘Inspecter’ is a calque of ‘Inspect’ for a read-only status view.
- Suggested: Consulter ou configurer l’identité de confidentialité macOS qu’utilisent les runtimes Local Operator

### FRPM-N2 — NIT — cli.permissions.setup.always_allow
- Current: (ou aucune boîte de dialogue n'est apparue) ?
- Problem: Statement inside a question; slightly clumsy, meaning and y-polarity are preserved.
- Suggested: Avez-vous choisi « Toujours autoriser » (ou aucune fenêtre ne s’est affichée) ? [y/N] 

### FRPM-N3 — NIT — status.screen.granted/denied/unknown
- Current: (vérification du runtime)
- Problem: ‘preflight’ is a check made without prompting; « vérification préalable » is closer if space allows (no budget declared).
- Suggested: (vérification préalable du runtime)

## Suggested values (all findings applied; placeholders/code spans verified identical to en)
| key | suggested fr |
|---|---|
| `help.permissions` | Inspecter ou configurer l’identité de confidentialité macOS qu’utilisent les runtimes Local Operator |
| `help.status` | Afficher l’identité utilisée par les runtimes et si l’enregistrement de l’écran est autorisé (lecture seule) |
| `help.setup` | Signer les runtimes avec votre Developer ID pour qu’ils partagent les autorisations de l’application de bureau (interactif) |
| `status.not_macos` | Les autorisations de confidentialité macOS ne s’appliquent pas sur cette plateforme. |
| `status.identity.team` | Identité des runtimes : {identifier}, qui partage la ligne « Local Operator » de l’application de bureau |
| `status.identity.adhoc` | Identité des runtimes : {identifier}, avec sa propre ligne « Local Operator Runtime » |
| `status.identity.legacy` | Identité des runtimes : une image identifiée par son chemin (aucun bundle n’a pu être créé ici) ; chaque build Python a donc sa propre ligne |
| `status.identity.none` | Identité des runtimes : pas encore créée ; elle le sera au prochain démarrage d’un runtime |
| `status.image` | Image : {path} |
| `status.optin.on` | Signature d’équipe : activée (identité {identity}, équipe {team}) |
| `status.optin.off` | Signature d’équipe : désactivée. Lancez `lop permissions setup` pour partager la ligne de l’application de bureau. |
| `status.optin.blocked` | Signature d’équipe : indisponible dans ce processus ({reason}) |
| `status.app.found` | Application de bureau : {path} (équipe {team}) |
| `status.app.missing` | Application de bureau : non installée |
| `status.screen.granted` | Enregistrement de l’écran (vérification du runtime) : autorisé |
| `status.screen.denied` | Enregistrement de l’écran (vérification du runtime) : non autorisé. Activez « {row} » dans Réglages Système > Confidentialité et sécurité > Enregistrement de l’écran. |
| `status.screen.unknown` | Enregistrement de l’écran (vérification du runtime) : impossible à vérifier |
| `status.guide` | Plus d’aide : guide://permissions |
| `setup.refused.isolated` | La signature d’équipe n’est pas autorisée dans ce processus ({reason}). Lancez la commande depuis votre propre shell de connexion. |
| `setup.no_app` | L’application de bureau Local Operator n’est pas installée : il n’y a donc aucune autorisation d’application à partager. |
| `setup.no_identity` | Aucune identité de signature « Developer ID Application » pour l’équipe {team} dans votre trousseau. Les runtimes gardent la ligne « Local Operator Runtime ». |
| `setup.explain` | Cette commande signe le runtime avec « {name} » pour que les runtimes partagent les autorisations de confidentialité de l’application de bureau. macOS vous demandera peut-être une fois d’autoriser codesign à utiliser cette clé. Choisissez « Toujours autoriser » pour que les prochains `lop update` puissent signer sans demander. |
| `setup.confirm` | Continuer ? [y/N]  |
| `setup.cancelled` | Annulé. Rien n’a été modifié. |
| `setup.failed` | La signature n’a pas abouti. Les runtimes gardent la ligne « Local Operator Runtime ». Lancez `lop permissions status` pour plus de détails. |
| `setup.always_allow` | Avez-vous choisi « Toujours autoriser » (ou aucune boîte de dialogue n’est apparue) ? [y/N]  |
| `setup.done` | Terminé. Les nouveaux runtimes utilisent {identifier} et partagent la ligne de l’application de bureau. Les sessions en cours gardent leur image actuelle jusqu’à leur redémarrage. |
