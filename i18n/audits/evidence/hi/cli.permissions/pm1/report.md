# Audit report — hi / cli.permissions — pm1 (round 1)

- Auditor: i18n-auditor, lopdev team subagent, session `d94d672abfb8`, model `anthropic/claude-opus-5-5` (session system context; no fallback observed).
- Translator: PR #3691 coder agent, job `ad99821ab9e1`, `anthropic/claude-opus-5-5`. Independent session; same model family (disclosed).
- Scope: all 28 keys of `local_operator/i18n/catalogues/hi/cli.permissions.json` at worktree head `50badda034d7`, against en source sha256 `eba1e1dd19a6…` and its context sidecar.
- Method: native-fluent reading; no MT/translation service. Apple hi-IN help page fetched once to check localized macOS menu names (see audit.log).

## Verdict: changes_requested — BLOCKER 0 · MAJOR 1 · MINOR 2 · NIT 2

Mechanical: clean. 28/28 keys, placeholders `{identifier} {path} {identity} {team} {reason} {row} {name}` identical, backtick commands byte-identical, `[y/N] ` trailing space preserved on both prompts, `Local Operator`, `Local Operator Runtime`, `Developer ID Application`, `codesign`, `guide://permissions` verbatim; no ZWJ/ZWNJ; no plurals in this namespace. No `maxCells` in the sidecar; every hi value is narrower in cells than its en source. `check.py --catalogues-only`: 0 problems.

Safety/accuracy: what the user is told to click and what happens is preserved — "हमेशा अनुमति दें" choice, the one-time keychain prompt about codesign using the key, `[y/N]` default-no, "रद्द किया गया। कुछ भी नहीं बदला।", failure keeps the "Local Operator Runtime" row, read-only status. No polarity errors.

## Findings

| ID | Sev | Key | Current | Problem | Suggested |
|---|---|---|---|---|---|
| H1 | MAJOR | `setup.done` | `चल रहे सेशन फिर से शुरू होने तक अपनी मौजूदा इमेज रखेंगे।` | Glossary row `session` = सत्र for hi; style guide explicitly rejects "सेशन". This is the only "सेशन" in the whole hi catalogue (59 files use सत्र). | `चल रहे सत्र फिर से शुरू होने तक अपनी मौजूदा इमेज ही इस्तेमाल करेंगे।` |
| H2 | MINOR | `status.screen.denied` | `सिस्टम सेटिंग्स > गोपनीयता और सुरक्षा > स्क्रीन रिकॉर्डिंग` | Context asks for the localized macOS menu names; Apple's hi-IN Mac User Guide names the app "सिस्टम सेटिंग" (no plural). "गोपनीयता और सुरक्षा" matches Apple. Recognizable, but the instruction should match the on-screen label. | `सिस्टम सेटिंग > गोपनीयता और सुरक्षा > स्क्रीन रिकॉर्डिंग` |
| H3 | MINOR | `status.identity.legacy` | `(यहाँ bundle नहीं बन सका)` | Common noun left in Latin mid-sentence; the hi catalogue already renders it "बंडल" (cli.group_help, wire.settings). Not a product/tool name. | `(यहाँ बंडल नहीं बन सका)` |
| H4 | NIT | `status.identity.team`, `.adhoc`, `.legacy`, `optin.off`, `setup.no_identity`, `setup.failed`, `setup.done` | `पंक्ति` for "row" | Literal but acceptable; "प्रविष्टि" (entry) reads more naturally for an item in a System Settings list. Only if the team wants it; consistency across all 7 keys matters more than the choice. | optional: `प्रविष्टि` throughout |
| H5 | NIT | `setup.explain` | `ताकि आगे `lop update` बिना पूछे साइन कर सके।` | "runs" dropped; slightly compressed. | `ताकि बाद में `lop update` चलने पर बिना पूछे साइन हो सके।` |

Not verified: Apple's hi-IN label for the Screen Recording pane and the keychain dialog's "Always Allow" button (Apple page returned HTTP 429). "हमेशा अनुमति दें" is the standard Apple Hindi rendering of "Always Allow" as I know it, but this is an unverified reading, not a fetched fact. Also out of hi scope: on macOS 15+ the English pane is "Screen & System Audio Recording"; the en source says "Screen Recording", and hi follows the source.

Gate: H1 must be fixed for `passed`. H2/H3 are recommended in the same remediation; H4/H5 optional.
