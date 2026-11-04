# hi / cli.global_help: core catalogue gap audit (cg1)

17 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **HI-024** (round 1, MINOR) `cli.global_help.yolo`
  - English: 'Auto-approve all tool executions (read/write/exec tiers) without prompting'
  - Value at audit: 'बिना पूछे सभी टूल निष्पादन (read/write/exec स्तर) स्वतः अनुमोदित करें'
  - Problem: "Auto-approve" → "स्वतः अनुमोदित" here and in cli.exec_worker_help.yolo; approve/approval is fixed to स्वीकृति/स्वीकृत by the glossary and hi guide (this patch set itself uses "स्वीकृत करें" in cli.browser.no_changes and cli.browser_help.cleanup_yes). The --yolo safety flag should use the same verb as the approval surfaces.
  - Suggested: '"बिना पूछे सभी टूल निष्पादन (read/write/exec स्तर) अपने-आप स्वीकृत करें"; exec_worker_help.yolo: "सभी टूल स्तर अपने-आप स्वीकृत करें"'
  - Disposition: fixed (remediation core-fix, session 76bcbfdf629d)

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
