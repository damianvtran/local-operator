# fr · cli.permissions — independent audit frpm3 (round 3, REPORT)

- PR: https://github.com/damianvtran/local-operator/pull/3691 · worktree `.worktrees/runtime-privacy-identity` @ 85fb35811f9d17fb6dfadfc2d5df3269db04bd12
- Auditor: i18n-auditor (lopdev team task subagent), session cf1ab574baf9, model anthropic/claude-opus-5-5. The model comes from this session's own system-prompt model header, and no provider fallback was observed. This auditor is independent of the round-3 fr translator (commit 85fb35811f9, session not identifiable), the round-1 translator (ad99821ab9e1) and the prior auditors (144ca81a304c, c2890b9743dd). No MT or translation service was used, and no catalogue was edited.
- Scope (round 2+ rule): the frpm2 findings, plus the keys changed between 788cc399 and 85fb3581. The en keys that changed are status.scope, setup.already_signed and the new setup.always_allow_kept. The fr keys that changed are those three plus help.status, status.optin.unavailable and setup.explain (the frpm2 fixes). Unchanged keys were not re-audited.
- Inputs: en 58e58c3b19a07c02…, context 6f8427bf5dd20f8f…, fr 2f36fa071929503a… (full hashes are in record.json).
- Verdict: **passed**. There are 0 BLOCKER, 0 MAJOR, 0 MINOR and 0 NIT findings.

## Mechanical (audit.log is the execution record)
- Both en and fr have 39 keys; none were added or dropped. Placeholders, code spans, newlines, bullets and the trailing prompt space match en. No ICU is used. Every value is NFC. `check.py --catalogues-only` reports 0 problems.
- Typography: there is no ASCII ' or " outside code, and U+202F is placed correctly in the new keys. A negative control on a mutated scratch copy confirms the checker flags a renamed placeholder and missing U+202F.
- One checker line, `token 'Developer ID' 1 vs 2` in setup.explain, is expected and not a defect. It comes from the frpm2-2 wording « la clé privée de cette identité Developer ID », which names the identity explicitly for "the key's private key". It is a product term, not a placeholder or command.

## macOS labels in changed keys: Apple tables on this Mac (macOS 27.2), not from memory
- Fichiers et dossiers: SecurityPrivacyExtension Localizable.loctable :: FILE_ACCESS_COMBINED ✓
- Toujours autoriser: SecurityAgent ConfirmAccess.loctable :: 100267.title ✓
- Trousseaux d’accès: Keychain Access InfoPlist :: CFBundleDisplayName ✓. Contrôle d’accès: KeyEditor :: 302.label ✓. clé privée: Localizable :: private key ✓
- The Accessibility service (ACCESSIBILITY → fr « Contrôle de l’appareil et accès aux données » on 27.2) is no longer named in fr status.scope, which matches the en intent (checked in audit.log).

## Round-2 findings: status
- FRPM2-1 (MINOR): **fixed**. The text now reads « sous votre compte utilisateur macOS ».
- FRPM2-2 (MINOR): **fixed**. The text now reads « Pour revenir sur ce choix, ouvrez Trousseaux d’accès, sélectionnez la clé privée de cette identité Developer ID, puis retirez codesign dans l’onglet Contrôle d’accès. » Access Control is a tab of the Keychain Access item window, so « onglet » is accurate.
- FRPM2-N1 through N4: all **applied**. The texts now read « reposeront la question », « cette fois-ci », « (équipe {team}), absente de votre trousseau » and « Afficher l’identité utilisée par les runtimes et indiquer si… ».
- FRPM2-S1 (source observation): **resolved upstream**. en no longer names Accessibility.
- FRPM2-S2 (source observation): **resolved upstream**. The kept-answer sentence moved to setup.always_allow_kept, which `permissions_cli.py` prints only `if kept`.

## Changed/new keys: accuracy and idiom
- status.scope: « Les autres services de confidentialité, dont Fichiers et dossiers, utilisent la même ligne. » Restructuring the sentence around « dont » is idiomatic and keeps the meaning (all services share the row, with Files & Folders as the example). No version-specific pane name appears.
- setup.already_signed: it states that nothing was signed this time and that macOS had nothing to ask, and it no longer mentions the earlier answer, as the context requires. The polarity is correct.
- setup.always_allow_kept: « Votre précédente réponse « Toujours autoriser » pour cette clé est conservée. » The meaning is exact, « pour cette clé » maps to "for this key", and the button label is Apple's. Placing « précédente » before the noun is natural fr product register. It is a single line with no fit concern.
- The setup.explain safety sentence still says that *any program* under the user's account can use codesign with the Developer ID key *without asking*. It is not softened.

## Findings
None.
