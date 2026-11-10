# Independent audit, round 3: ur / cli.permissions (urp3, 2026-11-09)

- PR: damianvtran/local-operator#3691, worktree HEAD `85fb35811f9d17fb6dfadfc2d5df3269db04bd12`.
  The ur catalogue was last changed in that commit ("fix(i18n): retranslate cli.permissions for review round 2").
- Auditor: i18n-auditor (lopdev team subagent), session `df1f78befebc`, model `anthropic/claude-opus-5-5`
  (taken from the host system prompt, with no fallback). I am independent of the round-3 ur translator, whose session was not disclosed to me, and of the
  round-1 and round-2 auditors (sessions `ffe80a68ac43` and `26eb57e200ee`). The model family is the same, so the agent is independent but the vendor is not.
- Inputs: en `58e58c3b…7be41`, context `6f8427bf…fb1d2`, ur `83ea8d20…75132`. Full hashes are in record.json.
- Scope (round 2+ rule): this pass checks every urp2 finding and audits the keys changed or added in
  `788cc399..85fb3581`: `status.scope`, `setup.already_signed`, the new `setup.always_allow_kept`, and
  `setup.explain` (retranslated to apply P1–P3). I did not re-audit the unchanged keys.
- No MT or translation service was used. Apple labels were read from this Mac (macOS 27.2, build 26B123). audit.log
  records every command and its actual output.

## Verdict: `passed`. 0 BLOCKER, 0 MAJOR, 0 new findings

## Mechanical
- `check.py --catalogues-only` reported 0 problems.
- My own check (`mech_check.py.txt`) found 39/39 keys in the same order as en. The new key is present in both the catalogue and the context sidecar. For each changed key, the
  placeholders and backtick spans are identical, and the counts of the verbatim tokens match the source (`Files & Folders`, `"Allow"`,
  `"Always Allow"`, Keychain Access, Access Control, `private key`, codesign, Developer ID, `lop update`,
  /update, Local Operator Runtime). explain still has 3 lines. No string contains bidi or other Cf characters, and none uses Arabic-Indic digits. Every
  line of every changed key now starts with an Urdu strong character, so the paragraph direction is RTL.

## Apple labels in the changed keys (macOS 27.2)
No `ur` table exists in any of these loctables, so English labels are correct for ur.
- `Files & Folders` is SecurityPrivacyExtension.appex Localizable [en/en_GB] `FILE_ACCESS_COMBINED`.
- The same table maps `ACCESSIBILITY` to `Device Control and Data Access` on this build. That confirms the en change, and
  the ur line, which no longer names the pane, is now correct on every supported macOS version.
- `Allow` / `Always Allow` are SecurityAgent ConfirmAccess.loctable [en_GB] `100268.title` / `100267.title`.
- In Keychain Access.app, `Keychain Access` is Localizable [en_GB], `private key` is Localizable [en_GB] (the Kind column), and `Access Control` is
  KeyEditor.loctable [en_GB] `302.label`.

## Round-2 findings (urp2)
| ID | Sev | Status on this head |
|---|---|---|
| P1 | MINOR | **Fixed.** Both bullets now start in Urdu: `- اگر آپ "Allow" (اجازت دیں) منتخب کریں تو صرف اس بار سائن ہو گا۔` and `- اگر آپ "Always Allow" (ہمیشہ اجازت دیں) منتخب کریں تو macOS دوبارہ نہیں پوچھے گا، لیکن پھر …`. mech_check confirms that the first strong character is R on all 3 lines. |
| P2 | NIT | **Fixed.** `اس شناخت کی private key (پرائیویٹ کلید) منتخب کریں`: the on-screen Kind text comes first and the gloss follows in brackets. |
| P3 | NIT | **Fixed** through the P1 wording `macOS دوبارہ نہیں پوچھے گا` ("macOS will not ask again"), which is the natural Urdu for "stops the questions". |
| P4 | NIT | **Not applied, and accepted.** It was optional. With Accessibility gone, only one English pane label is left (`Files & Folders`), next to `اسکرین ریکارڈنگ`. That Urdu term matches the label `اسکرین ریکارڈنگ (رن ٹائم پری فلائٹ)` used on the 4 `status.screen.*` lines just above it. "Screen Recording" there is lop's own status term, not the pane name, so keeping it consistent is the better choice. Closed. |

## Changed and new keys
- `status.scope`: `صرف اسکرین ریکارڈنگ کی جانچ ہوتی ہے۔ Files & Folders اور دیگر پرائیویسی سروسز بھی یہی قطار استعمال کرتی ہیں۔`
  This renders the new source exactly. Accessibility was dropped as the source dropped it, and nothing else was added or lost. The glossary terms پرائیویسی and قطار
  are used. The second sentence starts with a Latin run inside an RTL paragraph, which is harmless: the `&` sits between two L runs and resolves to L.
- `setup.already_signed`: the tail about the earlier "Always Allow" answer was removed, as the context note requires.
  The remaining sentence is unchanged from the approved round-2 text and keeps its scope and polarity.
- `setup.always_allow_kept` (new): `اس کلید کے لیے "Always Allow" والا آپ کا پچھلا جواب برقرار ہے۔` This is accurate and includes "for this key"
  (`اس کلید کے لیے`). It uses the glossary term کلید and a natural `والا` construction, in the same register (آپ) as the rest. The ASCII quotes are
  symmetric, so they cannot be mirrored wrongly. The line starts in Urdu.
- `setup.explain`: in bullet 1, `صرف اس بار سائن ہو گا` keeps "this time only". In bullet 2, the full grant
  (any program running as your macOS user can sign through codesign with this Developer ID key without asking) is unchanged
  from the approved round-2 text, and so are the undo steps. No meaning has drifted. `ہو گا` uses the spaced spelling, which is standard and common in the ur catalogues.

## Not findings (checked)
- There is no `maxCells`/`maxChars` hint. These are wrapped CLI lines, so I found no fit risk.
- `۔` ends each sentence, and Latin punctuation stays only inside code, paths, labels and `[y/N]`, as the ur style guide requires.
