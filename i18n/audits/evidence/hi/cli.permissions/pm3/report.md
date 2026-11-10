# Audit report — hi / cli.permissions — pm3 (round 3)

- Auditor: i18n-auditor, lopdev team subagent, session `c77ef00ae239`, model `anthropic/claude-opus-5-5` (session system context; no fallback observed). Independent of the translators (round 1 `ad99821ab9e1`; round 2/3 i18n-translator commits `788cc399410`, `85fb35811f9`) and of the prior auditors (`d94d672abfb8` pm1, `d9395b58854d` pm2). I translated none of these strings.
- Worktree head `85fb35811f9d17fb6dfadfc2d5df3269db04bd12`. Inputs: en `58e58c3b19a0…`, context `6f8427bf5dd2…`, hi `40aa4904e2a0…` (full hashes in `record.json`, `audit.log`).
- Scope (round 2+ rule): every pm2 finding, plus the 5 keys changed or added since `788cc399`: `status.screen.denied`, `status.scope`, `setup.explain`, `setup.already_signed`, `setup.always_allow_kept` (new). Unchanged keys approved earlier were not re-audited.
- Method: native-fluent reading; no MT/translation service. macOS labels checked against Apple's installed `.loctable` resources on macOS 27.2 (26B123); output in `audit.log`.

## Verdict: passed — BLOCKER 0 · MAJOR 0 · MINOR 0 · NIT 1 (carried P4, optional)

Mechanical: `check.py --catalogues-only` 0 problems; 39/39 keys, none missing or extra; placeholders (`{row}`, `{name}`), backtick spans, verbatim tokens, quote counts and newline count (explain: 2) identical to en on every changed key; no ZWJ/ZWNJ; every hi line narrower in cells than en (no `maxCells` in the sidecar).

## pm2 findings

| ID | pm2 sev | Status at 85fb358 |
|---|---|---|
| P1 | MAJOR | **Fixed.** `status.screen.denied` now reads `सिस्टम सेटिंग > गोपनीयता और सुरक्षा > स्क्रीन और सिस्टम ऑडियो रिकॉर्डिंग`; all three hops match Apple hi exactly. |
| P2 | MAJOR | **Fixed.** `setup.explain` now says `ऐक्सेस कंट्रोल में से codesign हटाएँ` = Keychain Access KeyEditor [302.label]; the guessed English gloss is gone. |
| P3 | MINOR | **Fixed.** `उसकी प्राइवेट-की चुनें` = Keychain Access Localizable [private key]; the "key's key" awkwardness is gone. |
| P4 | NIT | **Not applied; optional, carried.** Straight quotes still mirror en instead of the style guide's “ ” (now also in the new `setup.always_allow_kept`). Does not gate. |

## Changed / new keys

| Key | hi | Assessment |
|---|---|---|
| `status.scope` | केवल स्क्रीन रिकॉर्डिंग की जाँच होती है। फ़ाइल और फ़ोल्डर तथा अन्य गोपनीयता सेवाएँ भी इसी पंक्ति का उपयोग करती हैं। | Accessibility dropped as the en and context require; `फ़ाइल और फ़ोल्डर` = Apple [FILE_ACCESS_COMBINED]; `तथा` avoids a double `और` after a label that already has `और`. Good. |
| `setup.already_signed` | इस Python बिल्ड का रनटाइम पहले से साइन था, इसलिए अभी कुछ साइन नहीं किया गया और macOS के पास पूछने को कुछ नहीं था। | Matches the shortened en; no longer mentions Always Allow, as the context note says. Good. |
| `setup.always_allow_kept` (new) | इस कुंजी के लिए आपका पहले वाला "हमेशा अनुमति दें" जवाब बना रहता है। | Accurate ("for this key" kept, so it is scoped to the chosen key); `हमेशा अनुमति दें` = ConfirmAccess [100267.title]; reads naturally; `कुंजी` matches `Developer ID कुंजी` in explain. Good. |
| `setup.explain` | … इसे वापस लेने के लिए कीचेन ऐक्सेस खोलें, उसकी प्राइवेट-की चुनें, और ऐक्सेस कंट्रोल में से codesign हटाएँ। | Only the undo sentence changed (P2/P3). `कीचेन ऐक्सेस` = InfoPlist [CFBundleDisplayName]. Meaning matches en ("select the key's private key, and remove codesign under Access Control"). Safety disclosure from pm2 intact. Good. |
| `status.screen.denied` | see P1 | Good. |

## Apple labels (installed hi strings, macOS 27.2)

| en | hi | Source [key] | Catalogue |
|---|---|---|---|
| System Settings | सिस्टम सेटिंग | System Settings.app InfoPlist [CFBundleDisplayName] | ✓ |
| Privacy & Security | गोपनीयता और सुरक्षा | SecurityPrivacyExtension InfoPlist [CFBundleDisplayName] | ✓ |
| Screen & System Audio Recording | स्क्रीन और सिस्टम ऑडियो रिकॉर्डिंग | SecurityPrivacyExtension Localizable [SCREEN_CAPTURE_HEADER] | ✓ |
| Files & Folders | फ़ाइल और फ़ोल्डर | same [FILE_ACCESS_COMBINED] | ✓ |
| Allow / Always Allow | अनुमति दें / हमेशा अनुमति दें | SecurityAgent ConfirmAccess [100268.title] / [100267.title] | ✓ |
| Keychain Access | कीचेन ऐक्सेस | Keychain Access.app (CoreServices/Applications) InfoPlist [CFBundleDisplayName] | ✓ |
| Access Control | ऐक्सेस कंट्रोल | Keychain Access KeyEditor [302.label] | ✓ |
| private key | प्राइवेट-की | Keychain Access Localizable [private key] | ✓ |

## Findings

No new findings. Carried: P4 NIT (optional “ ” quotes). Not checked: a rendered screen of Hindi macOS; label checks are against the installed string tables only.
