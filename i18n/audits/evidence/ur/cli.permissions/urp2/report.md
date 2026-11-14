# Independent audit, round 2: ur / cli.permissions (urp2, 2026-11-09)

- PR: damianvtran/local-operator#3691, worktree HEAD `788cc399410196b080cfc9f7e7bd558ccb713140`
  (ur catalogue last changed in that commit, "fix(i18n): retranslate cli.permissions for review round 1").
- Auditor: i18n-auditor (lopdev team subagent), session `26eb57e200ee`, model `anthropic/claude-opus-5-5`
  (host system prompt; no fallback). Independent of the round-2 ur translator (separate i18n-translator
  child in this wave; its session id was not disclosed to me) and of the round-1 auditor
  (session `ffe80a68ac43`). The model family is the same, so the agent is independent but the vendor is not.
- Inputs: en `d5eed404…f186`, context `f2d9ce73…c099`, ur `d7c43d6f…0c50d` (full hashes are in record.json).
- Read: i18n/style-guides/ur.md, i18n/glossary.md, the round-1 report urp1, the en catalogue and context
  sidecar (all 38 keys), the ur catalogue, `local_operator/permissions_cli.py` and `local_operator/i18n/text.py`
  (`tr()` adds no bidi controls; the CLI uses plain `print`).
- No MT or translation service was used. Apple's labels were read from this Mac's own resources
  (macOS 27.2, 26B123). The commands and their output are in audit.log.

## Verdict: `passed`: 0 BLOCKER, 0 MAJOR, 1 MINOR, 3 NIT

## Mechanical layer
- `check.py --catalogues-only` reported 0 problems.
- My own check (`mech_check.py.txt`) found: 38/38 keys in the same order; placeholders identical; backtick spans
  identical; every verbatim token count matches the source (row names, Developer ID [Application],
  codesign, guide://permissions, /update, HOME, env vars, `[y/N] ` with its trailing space, and every
  macOS label); the explain key keeps its 3 lines and both `- ` bullets; no bidi controls; no Arabic-Indic
  digits; no danda.
- The sidecar has no `maxCells`/`maxChars` hint. These are multi-sentence CLI lines that wrap, so I found no fit issue.

## macOS labels checked against Apple's strings on this Mac
No `ur` table exists in any of these loctables, which confirms that macOS has no Urdu UI and English labels are correct for ur.
Base is the development language, and en_GB is the English table row:
- `Screen & System Audio Recording`: SecurityPrivacyExtension.appex Localizable.loctable [en] `SCREENANDAUDIOCAPTURE` / `SCREEN_CAPTURE_HEADER`
- `Privacy & Security`: SecurityPrivacyExtension.appex InfoPlist.loctable `CFBundleDisplayName`; System Settings.app Localizable `Privacy & Security`
- `System Settings`: System Settings.app InfoPlist.loctable [en_GB] `CFBundleDisplayName`
- `Files & Folders`: SecurityPrivacyExtension.appex Localizable [en] `FILE_ACCESS_COMBINED`; `Accessibility`: System Settings.app Localizable [en] `Accessibility`
- `Allow` / `Always Allow` / `Deny`: SecurityAgent.bundle ConfirmAccess.loctable [en_GB] `100268.title` / `100267.title` / `100266.title`, also present in Base.lproj/ConfirmAccess.nib
- `Keychain Access`: Keychain Access.app InfoPlist [en_GB] `CFBundleDisplayName`; `Access Control`: KeyEditor.loctable [en_GB] `302.label` (KeyEditor.nib);
  Kind `private key`: Keychain Access.app Localizable [en_GB] `private key`
Every label in the ur catalogue matches these strings byte for byte.

## Round-1 findings (urp1)
| ID | Sev | Status on this head |
|---|---|---|
| U1 | MAJOR | **Resolved.** screen.denied uses `System Settings > Privacy & Security > Screen & System Audio Recording` verbatim. The path forms one LTR run, because `>` between L runs resolves to L. |
| U2 | MAJOR | **Resolved.** explain uses `"Allow" (اجازت دیں)` and `"Always Allow" (ہمیشہ اجازت دیں)`: the label comes first and the gloss follows in brackets. always_allow and already_signed use `"Always Allow"`. |
| U3 | MINOR | **Resolved.** The context sidecar now says to keep the English label when macOS is not localized (e.g. Urdu). The ur style guide records the policy under "Product-name policy". |
| U4 | NIT | **Resolved.** legacy now uses `بنڈل`. |
| U5 | NIT | **Applied** to help.permissions and refused.tty, which now start in Urdu. no_app was left unchanged, as round 1 advised. |

## Accuracy of the safety copy (strict pass)
- explain line 1 matches the source: codesign asks whether it may use the key; "may" is kept (`پوچھ سکتا ہے`).
- "Allow": `صرف اس بار سائن کرتا ہے`. A later terminal `lop update` asks again. Updates with no terminal skip
  team signing and use the "Local Operator Runtime" row. Both the scope and the polarity are exact.
- "Always Allow": `آپ کے macOS صارف کے طور پر چلنے والا کوئی بھی پروگرام codesign کے ذریعے اس Developer ID کلید
  سے بغیر پوچھے سائن کر سکتا ہے`. This keeps the full grant: any program, running as your user, through codesign, with this
  key, without asking. Nothing is softened.
- Undo: open Keychain Access, select the private key, and remove codesign under Access Control. This matches the
  actual UI (KeyEditor `Access Control` tab; ACLEditor "Always allow access by these applications:"). The
  source says "the key's private key" and ur says `اس شناخت کی پرائیویٹ کلید` ("this identity's private key").
  Both point to the same Keychain item, so this is not a finding.
- always_allow asks whether the user chose "Always Allow" or saw no dialog, which matches the source. `[y/N]` is intact.
- cancelled, failed, no_identity, done and done_allow_once keep their polarity and scope.

## Findings

### P1: MINOR (non-gating, bidi): `cli.permissions.setup.explain`, lines 2–3
- Current: `- "Allow" (اجازت دیں) صرف اس بار سائن کرتا ہے۔ …` and `- "Always Allow" (ہمیشہ اجازت دیں) سوالات بند کر دیتا ہے، …`
- Problem: these are the only full-sentence ur lines whose first strong character is Latin (see mech_check in audit.log).
  `permissions_cli.py` prints through `tr()`, which adds no FSI wrap. On a terminal that sets paragraph direction
  from the first strong character, the two safety bullets therefore lay out LTR, and the runs of Urdu, `lop update`, `/update` and codesign appear in
  scrambled order. The bullets are the Always Allow disclosure. Under the guide's degraded-TUI stance this is not a
  defect, but a natural Urdu ordering exists that starts in Urdu anyway.
- Suggested:
  `- اگر آپ "Allow" (اجازت دیں) منتخب کریں تو صرف اس بار سائن ہو گا۔ بعد میں …` (rest unchanged)
  `- اگر آپ "Always Allow" (ہمیشہ اجازت دیں) منتخب کریں تو macOS دوبارہ نہیں پوچھے گا، لیکن پھر …` (rest unchanged)

### P2: NIT: `cli.permissions.setup.explain`, line 3
- Current: `اس شناخت کی پرائیویٹ کلید (private key) منتخب کریں`
- Problem: `private key` is the Kind text that Keychain Access shows (Localizable `private key`). The ur guide puts the on-screen
  English first and the gloss in brackets, which is how the line already treats "Allow"/"Always Allow". Here the order is reversed.
- Suggested: `اس شناخت کی private key (پرائیویٹ کلید) منتخب کریں`

### P3: NIT: `cli.permissions.setup.explain`, line 3
- Current: `سوالات بند کر دیتا ہے`
- Problem: this is a literal rendering of "stops the questions". In Urdu product copy the natural reading is that the prompt stops recurring.
- Suggested: `دوبارہ پوچھنا بند کر دیتا ہے` (or as worded in P1).

### P4: NIT: `cli.permissions.status.scope`
- Current: `صرف اسکرین ریکارڈنگ کی جانچ ہوتی ہے۔ Files & Folders اور Accessibility سمیت …`
- Problem: the sentence names three sibling privacy services, but writes one in Urdu and two as English pane labels. The status-line label
  "اسکرین ریکارڈنگ" is fine on its own. Here the inconsistency is visible.
- Suggested (optional): `صرف Screen & System Audio Recording کی جانچ ہوتی ہے۔ …`, or leave it as is.

## Not findings (checked)
- Register is consistent throughout: آپ with the respectful imperative. پرائیویسی is used for every "privacy" (no
  رازداری remains), قطار for every "row", کلید for "key", کی چین for the common noun "keychain". Glossary سیشن is used.
- `{detail}۔`: the English codesign line followed by an Urdu full stop is per the guide (Latin punctuation is kept only inside code and paths).
- reason.* lines start with an env-var name, but they are always inserted mid-sentence inside parentheses, so there is no paragraph-direction issue.
