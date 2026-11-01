# hi / tui.small: core catalogue gap audit (cg1)

137 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **HI-001** (round 1, MINOR) `tui.small.network_panel.installed_row`
  - English: '  installed:  {value}\n'
  - Value at audit: '  इंस्टॉल:    {value}\n'
  - Problem: Value column misaligned: label+padding is 11 cells vs 14 in the sibling relay rows (and the English, where every row aligns at 14). Context note says the padding aligns the column.
  - Suggested: '"  इंस्टॉल:     {value}\\n" padded so {value} starts at cell 14 (e.g. 5 spaces after the colon; verify with cell_len)'
  - Disposition: fixed (remediation core-fix, session 76bcbfdf629d)
- **HI-002** (round 1, MINOR) `tui.small.network_panel.identity_row`
  - English: '  identity:   {identity}\n'
  - Value at audit: '  पहचान:      {identity}\n'
  - Problem: Value column at 13 cells, sibling rows at 14; one space short, so the identity value is visibly offset.
  - Suggested: '"  पहचान:       {identity}\\n" (one more space; value at cell 14)'
  - Disposition: fixed (remediation core-fix, session 76bcbfdf629d)
- **HI-003** (round 1, MINOR) `tui.small.aside.title`
  - English: 'Aside'
  - Value at audit: 'अलग से'
  - Problem: "अलग से" is an adverb ("separately"), not a name for the card; existing hi copy names the same aside surface "नोट" (tui.notices.unsaved_asides, wire.errors.desktop.aside_*), so one feature now has two names.
  - Suggested: 'Align with the shipped term: "नोट" — or, if "नोट" is to be replaced, a noun such as "अलग बातचीत" applied to all aside keys at once'
  - Disposition: fixed (remediation core-fix, session 76bcbfdf629d)
- **HI-004** (round 1, MINOR) `tui.small.session_picker.all_archived`
  - English: 'all {total} conversation(s) are archived'
  - Value at audit: 'सभी {total} वार्तालाप संग्रहीत हैं'
  - Problem: "वार्तालाप" diverges from the settled hi term for conversation, "बातचीत" (102 uses in shipped hi catalogues).
  - Suggested: '"सभी {total} बातचीत संग्रहीत हैं"'
  - Disposition: fixed (remediation core-fix, session 76bcbfdf629d)
- **HI-005** (round 1, MINOR) `tui.small.ask_queue.question_waiting`
  - English: '{count, plural, one {{shown} question waiting} other {{shown} questions waiting}}'
  - Value at audit: '{count, plural, one {{shown} प्रश्न प्रतीक्षा में} other {{shown} प्रश्न प्रतीक्षा में}}'
  - Problem: "प्रश्न" for an ask/question diverges from shipped hi copy, which uses "सवाल" (22 uses, e.g. relay.ask.*, relay.asks.*); same for ask_queue.question_timed_out.
  - Suggested: '"{count, plural, one {{shown} सवाल प्रतीक्षा में} other {{shown} सवाल प्रतीक्षा में}}" (and question_timed_out: "सवाल का/सवालों का समय समाप्त")'
  - Disposition: fixed (remediation core-fix, session 76bcbfdf629d)
- **HI-006** (round 1, NIT) `tui.small.todo_panel.header`
  - English: 'Todos'
  - Value at audit: 'टूडू'
  - Problem: "टूडू" is a bare transliteration that does not read as Hindi; a short native heading fits.
  - Suggested: '"कार्य सूची" (or "करने के काम")'
  - Disposition: fixed (remediation core-fix, session 76bcbfdf629d)
- **HI2-002** (round 2, NIT) `tui.small.network_panel.installed_row`
  - English: '  installed:  {value}\n'
  - Value at audit: '  इंस्टॉल:       {value}\n'
  - Problem: Alignment fix verified under rich.cell_len, which the style guide mandates: {value} lands at cell 14 for installed_row and identity_row, the same as the relay rows. A wcwidth-style count that gives spacing matras (Mc: ा U+093E, ॉ U+0949) width 1 puts both rows at 15, one cell to the right. The rule is met; whether a given terminal renders them aligned depends on how it handles Mc. No change is requested.
  - Suggested: 'none (keep); verify on a rendered frame when the panel is visually validated'
  - Disposition: open at PASS: minor/nit accepted, value unchanged
- **HI2-003** (round 2, NIT) `tui.small.todo_panel.header`
  - English: 'Todos'
  - Value at audit: 'कार्य सूची'
  - Problem: todo → कार्य/कार्य सूची is idiomatic. But bare कार्य also renders task/work elsewhere in this set (cli.exec_help.background 'कार्य को अलग करें', wire.console.*), so in todos_counts ' · कार्य {open}/{total}' and more_todos '… {count} और कार्य' a todo count can read as a task count.
  - Suggested: "optional: ' · कार्य सूची {open}/{total}' / '… {count} और कार्य-सूची आइटम' if space allows"
  - Disposition: open at PASS: minor/nit accepted, value unchanged

## Rephrased after the locale-wide audit, then audited by targeted session d8fb2395abc5 (PASS, keys_read=10 blocker=0 major=0 minor=1 nit=2)
- `tui.small.model_picker.window_position`
- `tui.small.wake_panel.monitors_need_attention`

## Auditor reports (locale-wide excerpts, verbatim)
### Round 1 (session 68a6549527ad)
# Hindi (hi) core catalogue patch audit — independent
- Auditor: i18n-auditor role, session 68a6549527ad, model anthropic/claude-opus-5-5 (no fallback). Independent of translator session 3ef6295a5545.
- Inputs: patches in 3ef6295a5545/scratchpad/core-i18n/hi/*.json (55 namespaces, 1294 values); en source + context sidecars from origin/main; i18n/style-guides/hi.md and i18n/glossary.md from origin/main; ambiguous.txt.
- Method: repo read-only through `git cat-file --batch`. A mechanical pass checked placeholder-name sets, ICU plural/select arms (hi one/other), brace balance, code spans/flags/paths/Rich tags/URLs, edge whitespace, NFC, control and bidi characters, and maxCells via rich.cells.cell_len. A semantic pass read EVERY value in every namespace against the English and its context. Terms were checked against all 44 shipped hi catalogues on origin/main. No MT service was used.
- **Verdict: CHANGES_REQUESTED** — BLOCKER 0 · MAJOR 2 · MINOR 17 · NIT 9.
## MAJOR

## Could not verify
- No rendered TUI frames. Width was checked only with cell_len on raw values, placeholders unexpanded. Values without maxCells (most tui.*) were not width-tested against real panes.
- The network_panel column alignment was measured statically (installed/identity rows land at 11/13 cells vs 14) and not seen on screen.
- Runtime fill-ins for {noun}/{verb}/{direction}/{suffix} word order (tui.projects.render.*, tui.hub_links.mark.retry) were judged from code context only.
- I ran no repo validator (`scripts/i18n/check.py`): the task forbade builds and tests, and the patches are not yet in the catalogue tree. Mechanical checks are my own script, not the repo gate.
- No ledger row was recorded: these are scratch patches, not catalogue commits.
### Round 2 (session e9c85b728add)
# Hindi (hi) core catalogue patches: round-2 independent audit
- **Auditor:** i18n-auditor role, session e9c85b728add, model `anthropic/claude-opus-5-5` (no fallback). This session did not translate these values, did not write the round-1 findings (68a6549527ad), and is not the translator session 3ef6295a5545. No MT service was used.
- **Repo:** read-only. I read only through `git show` / `git cat-file --batch` on `origin/main`. I ran no builds, tests or worktrees, and edited nothing outside this folder.
- **Verdict: PASS.** BLOCKER 0 · MAJOR 0 · MINOR 1 · NIT 3.
## (A) Changed keys: 96 of 96 judged against English
I judged each AFTER value fresh against the English, the glossary and the hi guide, not against the round-1 suggestion.

## Not verified
- I did not run the repo validator (`scripts/i18n/check.py`) or any test, because I was told no builds or tests. All mechanical checks above come from my own script.
- I rendered no frames.
- Seven namespaces have no context sidecar, so their intent comes from the English alone.
- I did not write a ledger or evidence record: this audit is read-only, and the requester owns ledger recording.
