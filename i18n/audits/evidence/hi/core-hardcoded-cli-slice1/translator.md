# Hindi core literals — translator note

- Translator session: 7b9be9a6099f. Actual model for translation: openai/gpt-6.1-sol, temporary fallback after provider quota failures (requested primary anthropic/claude-sonnet-5-5; intermediate fallback announcement anthropic/claude-opus-5-5). No translation service used.
- Completed 16 keys: cli.failures (4), cli.group_help (10), tui.setup_cues (2). Only these three Hindi catalogue files were written, plus this requested note. No ledger or generated assets changed.
- Followed i18n/glossary.md and i18n/style-guides/hi.md; existing hi CLI terminology was checked for consistency.
- Required-argument wording “इन आर्ग्युमेंट की ज़रूरत है: {names}” avoids singular/plural agreement depending on the interpolated list. The verb agrees with ज़रूरत, not list length.
- error → त्रुटि; invalid → अमान्य. Kept error_line order and all placeholders verbatim; quoted values/choices remain supplied by interpolation; None remains untouched.
- Agent Hub remains the product name; कौशल, सत्र, डिवाइस, मेश and रनटाइम follow the glossary/style guide. “pull” stays Latin per glossary wave-2 rules. “execute बिट” explicitly preserves the Unix permission-bit constraint, including non-owner execute bits; installing never executes code.
- Other term choices / glossary proposals: argument → आर्ग्युमेंट (familiar CLI loan); secret → सीक्रेट (stored secret object distinct from generic privacy); operator → ऑपरेटर (human authority, not a phone operator); encryption → एन्क्रिप्टेड; synchronization → सिंक में रखना. Headless fetching is explained as not showing a browser window; load-balanced search as spreading load across providers. No glossary edits made.
- Approval-gate copy retains that the key may loosen a running gate, without claiming to disable it. Phone pairing retains the human signature and one-time authorization.
- Actual width command used /Users/damian/local-operator/.venv/bin/python and rich.cells.cell_len. Interpolated cue: “पहले AI खाता जोड़ें: /login radient लिखें”. Returned 31 terminal cells (41 Unicode codepoints), within maxCells 74. The Devanagari control “क्षत्रिय” returned 5 cells; Rich counts combining marks at zero width. These are Rich terminal-cell measurements, not a claim about font shaping in a live terminal.
- Nothing exceeded the supplied width budget; no unresolved ambiguous strings.
- Lightweight readback verified exact English key order, placeholder sequence, valid canonical UTF-8 JSON with indent=2 and one trailing newline: all three files passed, 16 keys total.
- Repo catalogue validator, generate.py, tests and git commands intentionally NOT run under the explicit task restriction. Independent audit remains required; no self-audit or passed ledger status claimed.
