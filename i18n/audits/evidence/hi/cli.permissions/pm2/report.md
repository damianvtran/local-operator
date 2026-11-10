# Audit report — hi / cli.permissions — pm2 (round 2)

- Auditor: i18n-auditor, lopdev team subagent, session `d9395b58854d`, model `anthropic/claude-opus-5-5` (session system context; no fallback observed). Independent of the round-1 auditor (`d94d672abfb8`), the round-1 translator (`ad99821ab9e1`) and the round-2 i18n-translator (session not identified to me; I did not translate any of these strings).
- Scope: all 38 keys of `local_operator/i18n/catalogues/hi/cli.permissions.json` (sha256 `94cb61e796a3…`) at worktree head `788cc399410196b080cfc9f7e7bd558ccb713140`, against en `d5eed4044348…` and context `f2d9ce73f279…`.
- Method: native-fluent reading; no MT/translation service. macOS labels checked against Apple's own `.loctable` resources installed on this Mac (macOS 27.2, build 26B123) — tables and keys cited below and in `audit.log`.

## Verdict: changes_requested — BLOCKER 0 · MAJOR 2 · MINOR 1 · NIT 1

Mechanical: clean. 38/38 keys, no keys added/dropped; placeholders identical; backtick commands byte-identical; `[y/N] ` trailing space kept on both prompts; `setup.explain` keeps both `- ` bullets and line breaks; `Local Operator`, `Local Operator Runtime`, `Developer ID Application`, `Developer ID`, `codesign`, `guide://permissions`, `/update`, env-var names verbatim; no ZWJ/ZWNJ; no plurals. `check.py --catalogues-only`: 0 problems. No `maxCells` in the sidecar; every hi line is narrower in cells than its en line (`audit.log`).

Safety/accuracy: the new "Always Allow" disclosure is preserved fully — "हमेशा अनुमति दें" stops the questions, and afterwards *any* program running as the user can use codesign to sign with the Developer ID key without asking; "अनुमति दें" signs this time only; no-terminal updates skip team signing. No polarity errors; `[y/N]` default-no and "रद्द किया गया। कुछ भी नहीं बदला।" correct. The two MAJORs are both in *what the user is told to look for on screen*: the label in the instruction does not match the label Hindi macOS shows.

## Apple labels (installed hi strings)

| en | hi on this Mac | Source table [key] | Catalogue |
|---|---|---|---|
| System Settings | सिस्टम सेटिंग | System Settings.app `InfoPlist.loctable` [CFBundleDisplayName] | सिस्टम सेटिंग्ज़ ✗ |
| Privacy & Security | गोपनीयता और सुरक्षा | SecurityPrivacyExtension.appex `InfoPlist.loctable` [CFBundleDisplayName] | ✓ |
| Screen & System Audio Recording | स्क्रीन और सिस्टम ऑडियो रिकॉर्डिंग | SecurityPrivacyExtension.appex `Localizable.loctable` [SCREEN_CAPTURE_HEADER], [SCREENANDAUDIOCAPTURE] | ✓ |
| Screen Recording (service) | स्क्रीन रिकॉर्डिंग | same [SCREEN_CAPTURE] | ✓ |
| Files & Folders | फ़ाइल और फ़ोल्डर | same [FILE_ACCESS_COMBINED] | ✓ |
| Accessibility | ऐक्सेसिबिलिटी | AccessibilitySettingsExtension.appex `InfoPlist.loctable` [CFBundleDisplayName] | ✓ |
| Allow / Always Allow | अनुमति दें / हमेशा अनुमति दें | SecurityAgent.bundle `ConfirmAccess.loctable` [100268.title] / [100267.title] | ✓ |
| Keychain Access | कीचेन ऐक्सेस | Keychain Access.app `InfoPlist.loctable` [CFBundleDisplayName] | ✓ |
| Access Control (tab) | ऐक्सेस कंट्रोल | Keychain Access.app `KeyEditor.loctable` [302.label] | ऐक्सेस नियंत्रण (Access Control) ✗ |
| private key (item kind) | प्राइवेट-की | Keychain Access.app `Localizable.loctable` [private key] | निजी कुंजी |

(Keychain Access lives at `/System/Library/CoreServices/Applications/` on this macOS, not `Utilities/`; its tables carry `en_GB`, not `en`, as the English base.)

## Round-1 findings (pm1)

| R1 | Status in R2 |
|---|---|
| H1 MAJOR `setup.done` सेशन→सत्र | **Fixed** — "चल रहे सत्र … इस्तेमाल करेंगे।" |
| H2 MINOR `status.screen.denied` System Settings label | **Not applied; finding stands, raised to MAJOR (P1).** The translator's "सिस्टम सेटिंग्ज़" is not what macOS shows. The installed hi string settles it: `सिस्टम सेटिंग` (singular, no ज़). Round 1 rated it MINOR from a help-page reading; the sidecar requires the label "exactly as localized macOS shows it", and this is the first hop of the navigation path a user follows, so it is now a verified context-mandate violation. |
| H3 MINOR `status.identity.legacy` bundle→बंडल | **Fixed.** |
| H4 NIT पंक्ति | Kept; justified — consistent across all keys and 64 uses in hi. No further action. |
| H5 NIT `setup.explain` "runs" | **Resolved** by the rewritten source key ("बाद में टर्मिनल में `lop update` चलाने पर फिर पूछा जाएगा"). |

## Findings

| ID | Sev | Key | Current | Problem | Suggested |
|---|---|---|---|---|---|
| P1 | MAJOR | `status.screen.denied` | `सिस्टम सेटिंग्ज़ > गोपनीयता और सुरक्षा > स्क्रीन और सिस्टम ऑडियो रिकॉर्डिंग में "{row}" चालू करें।` | Apple hi is `सिस्टम सेटिंग` (System Settings.app InfoPlist [CFBundleDisplayName]). Context requires the exact localized label. The other two hops are correct. | `सिस्टम सेटिंग > गोपनीयता और सुरक्षा > स्क्रीन और सिस्टम ऑडियो रिकॉर्डिंग में "{row}" चालू करें।` |
| P2 | MAJOR | `setup.explain` (3rd line) | `… और ऐक्सेस नियंत्रण (Access Control) से codesign हटाएँ।` | This is how to undo the Always Allow grant. On Hindi macOS the Keychain Access tab reads `ऐक्सेस कंट्रोल` (KeyEditor [302.label]); neither "ऐक्सेस नियंत्रण" nor the English gloss appears on screen. The parenthetical shows the label was guessed. | `… और ऐक्सेस कंट्रोल में से codesign हटाएँ।` |
| P3 | MINOR | `setup.explain` (3rd line) | `उस कुंजी की निजी कुंजी चुनें` | Keychain Access lists the item kind as `प्राइवेट-की` (Localizable [private key]); "कुंजी की निजी कुंजी" also reads awkwardly (key's key). | `उसकी प्राइवेट-की चुनें` |
| P4 | NIT | 11 keys with quoted labels (`status.identity.team`, `.adhoc`, `optin.no_app`, `.unavailable`, `screen.denied`, `setup.no_identity`, `.explain`, `.failed`, `.always_allow`, `.already_signed`, `.done_allow_once`) | `"Local Operator Runtime"`, `"हमेशा अनुमति दें"` | Style guide prescribes “ ” for hi; 12 other hi files use them. Mirrors the en straight quotes, so harmless; optional. | `“Local Operator Runtime”`, `“हमेशा अनुमति दें”` |

Gate: P1 and P2 must be fixed for `passed`. P3 recommended in the same edit (same sentence as P2). P4 optional.
