# fr · cli.permissions — independent audit frpm2 (round 2, REPORT)

- PR: https://github.com/damianvtran/local-operator/pull/3691 · worktree `.worktrees/runtime-privacy-identity` @ 788cc399410196b080cfc9f7e7bd558ccb713140
- Auditor: i18n-auditor (lopdev team task subagent), session c2890b9743dd, model anthropic/claude-opus-5-5, taken from this session's own system-prompt model header; no provider fallback was observed. I am independent of the round-1 translator (ad99821ab9e1), the round-1 auditor (144ca81a304c) and the round-2 fr translator (whose session I could not identify; commit 788cc399410). No MT or translation service was used, and no catalogue was edited.
- Inputs: en d5eed4044348f0fa…, context f2d9ce73f2791f20…, fr 5fb01ed775d2034e… (full hashes are in record.json). I also read i18n/style-guides/fr.md, i18n/glossary.md, frpm1/report.md and local_operator/permissions_cli.py (plain `print`/`input`, no Rich markup; answers are parsed as y/yes).
- Verdict: **passed**: 0 BLOCKER, 0 MAJOR, 2 MINOR, 4 NIT. I also recorded 2 source-level observations that are not fr findings.

## Mechanical (audit.log is the execution record)
- 38/38 keys; none were added or dropped. Placeholders, backtick spans, `[y/N]`, `/update`, `guide://permissions`, env-var names, the bullet structure and the trailing prompt space all match en. No ICU plural/select is used. Every value is NFC.
- Typography: zero ASCII `'` or `"` outside code. U+202F appears before every `:` `;` `?` and inside every « » (53 ’, 58 U+202F). Negative control: the same checker run on the round-1 fr bytes reports 32 failures.
- `scripts/i18n/check.py --catalogues-only`: 0 problems.
- Fit: these are plain `print` lines with no `maxCells` declared. The longest is setup.explain at 323 cells, which wraps. Fit does not gate this namespace.

## macOS labels: verified against Apple tables installed on this Mac (macOS 27.2), not from memory
| en label | fr in catalogue | Apple source (table :: key → fr) |
|---|---|---|
| System Settings | Réglages Système | System Settings.app InfoPlist.loctable :: CFBundleDisplayName → « Réglages Système » (Apple uses NBSP; a plain space is fine) |
| Privacy & Security | Confidentialité et sécurité | SecurityPrivacyExtension InfoPlist.loctable :: CFBundleDisplayName ✓ |
| Screen & System Audio Recording | Enregistrement de l’écran et des sons du système | SecurityPrivacyExtension Localizable.loctable :: SCREEN_CAPTURE_HEADER / SCREENANDAUDIOCAPTURE ✓ |
| Screen Recording (service, status lines) | Enregistrement de l’écran | Localizable.loctable :: SCREEN_CAPTURE ✓ |
| Files & Folders | Fichiers et dossiers | Localizable.loctable :: FILE_ACCESS_COMBINED ✓ |
| Allow / Always Allow | Autoriser / Toujours autoriser | SecurityAgent.bundle ConfirmAccess.loctable :: 100268.title / 100267.title ✓ |
| Keychain Access | Trousseaux d’accès | Keychain Access.app InfoPlist.loctable :: CFBundleDisplayName ✓ |
| Access Control | Contrôle d’accès | Keychain Access.app KeyEditor.loctable :: 302.label ✓ |
| private key | clé privée | Keychain Access.app Localizable.loctable :: private key ✓ |
| Accessibility | Accessibilité | Follows en. See FRPM2-S1: on macOS 27.2 the privacy service is labelled differently. |

## Round-1 findings: status
- FRPM-1 (MAJOR, typography): **resolved**. 0 defects remain (see above).
- FRPM-2 (app de bureau): **resolved**. 0 occurrences remain; « application de bureau » is used throughout.
- FRPM-3 (setup.explain wording): **resolved**. The text now reads « Cette commande signe… macOS vous demandera peut-être d’autoriser codesign à utiliser cette clé. » Dropping « une fois » is correct, because the new en no longer says "once".
- FRPM-4 (refused.isolated « Lancez-la »): **resolved**. It now reads « Lancez la commande… ».
- FRPM-N1/N3: **applied** (Consulter; vérification préalable du runtime).
- FRPM-N2: **superseded**. The en prompt changed; the new « (ou macOS n’a-t-il affiché aucune boîte de dialogue) ? » is grammatical and keeps the y-polarity.

## Accuracy / safety (strict pass on what users click and on what "Always Allow" grants)
- setup.explain says that « Autoriser » signs this time only. It also says that terminal `lop update` runs ask again, and that terminal-less updates skip team signing and use « Local Operator Runtime ». « Toujours autoriser » is described as stopping the prompts while letting *any program running under the user* use codesign with the Developer ID key *without asking*. Each of these matches the source, and none of the risk has been softened.
- Polarity holds in cancelled/failed/no_identity/already_signed/done/done_allow_once (nothing changed; earlier setup unchanged; the running sessions keep their image).
- status.screen.denied points to the exact macOS path, Réglages Système > Confidentialité et sécurité > Enregistrement de l’écran et des sons du système, and keeps the « {row} » placeholder.
- The reason phrases (status.reason.*) read correctly inside « indisponible dans ce processus (…) » and « n’est pas autorisée dans ce processus (…) ». « dossier de départ » is the standard macOS fr term for the home folder; I did not find it in an installed table, so that term comes from my own reasoning.

## Findings
### FRPM2-1 — MINOR — cli.permissions.setup.explain
- Current: tout programme exécuté sous votre utilisateur macOS peut alors se servir de codesign…
- Problem: « sous votre utilisateur macOS » is a calque of "as your macOS user". This is the key safety sentence, and native fr says « compte utilisateur ».
- Suggested: tout programme exécuté sous votre compte utilisateur macOS peut alors se servir de codesign pour signer avec cette clé Developer ID sans rien demander.

### FRPM2-2 — MINOR — cli.permissions.setup.explain
- Current: Pour annuler, ouvrez Trousseaux d’accès, sélectionnez la clé privée de cette clé, puis retirez codesign dans Contrôle d’accès.
- Problem: « la clé privée de cette clé » is a redundant calque inside a click instruction. « Pour annuler » reads as "to cancel" rather than "to undo that choice". The labels themselves are correct.
- Suggested: Pour revenir sur ce choix, ouvrez Trousseaux d’accès, sélectionnez la clé privée de cette identité Developer ID, puis retirez codesign dans l’onglet Contrôle d’accès.

### FRPM2-N1 — NIT — cli.permissions.setup.explain
- Current: …lancés dans un terminal redemanderont ; → Suggested: …lancés dans un terminal reposeront la question ;

### FRPM2-N2 — NIT — cli.permissions.setup.already_signed
- Current: rien n’a donc été signé maintenant → Suggested: rien n’a donc été signé cette fois-ci

### FRPM2-N3 — NIT — cli.permissions.status.optin.unavailable
- Current: …pour l’équipe {team}, qui n’est pas dans votre trousseau. Here « qui » binds to « l’équipe ».
- Suggested: Elle nécessite la clé « Developer ID Application » de l’éditeur de l’application de bureau (équipe {team}), absente de votre trousseau.

### FRPM2-N4 — NIT — cli.permissions.help.status
- Current: Afficher l’identité qu’utilisent les runtimes et si… → Suggested: Afficher l’identité utilisée par les runtimes et indiquer si l’enregistrement de l’écran est autorisé (lecture seule)

## Source-level observations (en/code; not fr findings and not counted in the verdict)
- **FRPM2-S1**: On this Mac (macOS 27.2, build 26B123), Privacy & Security labels the Accessibility service "Device Control and Data Access" (SecurityPrivacyExtension Localizable.loctable :: ACCESSIBILITY; fr « Contrôle de l’appareil et accès aux données »). status.scope and the context note say "Accessibility". fr follows en, which is right for macOS 15/26. If en changes, fr should take Apple's string.
- **FRPM2-S2**: `permissions_cli.py` prints setup.already_signed whenever `result.signed` is False, even when `kept` is False. In that case it tells the user "Your earlier "Always Allow" answer is kept" and then also prints setup.done_allow_once, so the two messages contradict each other. This is a code/en issue, and fr mirrors en faithfully.
