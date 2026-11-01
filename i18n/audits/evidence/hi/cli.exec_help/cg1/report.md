# hi / cli.exec_help: core catalogue gap audit (cg1)

20 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **HI-021** (round 1, MINOR) `cli.exec_help.control`
  - English: 'Route approvals and questions to an attached supervisor (may wait). Without this flag non-TTY approvals deny; discovery and live attachment are available either way. --yolo remains an explicit approval override.'
  - Value at audit: 'अनुमोदन और प्रश्न अटैच्ड सुपरवाइज़र को भेजें (प्रतीक्षा कर सकता है)। इस फ़्लैग के बिना non-TTY अनुमोदन अस्वीकृत होते हैं; डिस्कवरी और लाइव अटैचमेंट दोनों तरह उपलब्ध हैं। --yolo स्पष्ट अनुमोदन ओवरराइड बना रहता है।'
  - Problem: "approvals" rendered "अनुमोदन" here and in tools/workstream; glossary + hi guide fix approvals → स्वीकृतियाँ/स्वीकृति, and the shipped wire.exec.approval_advisory (same exec surface, same rule) says "स्वीकृति". Meaning is intact, but the same --control/--yolo rule would have two terms.
  - Suggested: '"स्वीकृतियाँ और सवाल अटैच्ड सुपरवाइज़र को भेजें … इस फ़्लैग के बिना non-TTY स्वीकृतियाँ अस्वीकार होती हैं; … --yolo स्पष्ट स्वीकृति ओवरराइड बना रहता है।" Likewise tools ("स्वीकृति मानी जाती है", "स्वीकृति गेट") and workstream ("स्वीकृतियाँ कैसे संभाली जाती हैं")'
  - Disposition: fixed (remediation core-fix, session 76bcbfdf629d)
- **HI-022** (round 1, MINOR) `cli.exec_help.tools`
  - English: 'Comma-separated tools this run may reach, and the ONLY ones. Excluded tools are unreachable, not merely unapproved. The declaration stands as the APPROVAL for its own members only where nobody can be asked — a non-tty run without --control; on a terminal, and under --control, every write/exec call is still decided by the approval gate. A name this build does not have is unreachable and reported'
  - Value at audit: 'अल्पविराम से अलग टूल जिन तक यह रन पहुँच सकता है, और केवल वही। बाहर रखे टूल पहुँच से बाहर हैं, केवल अस्वीकृत नहीं। घोषणा अपने सदस्यों के लिए तभी अनुमोदन मानी जाती है जहाँ किसी से पूछा नहीं जा सकता — --control के बिना non-tty रन; टर्मिनल पर, और --control के तहत, हर write/exec कॉल का निर्णय अब भी अनुमोदन गेट करता है। जो नाम इस बिल्ड में नहीं है वह पहुँच से बाहर है और रिपोर्ट किया जाता है'
  - Problem: "not merely unapproved" became "केवल अस्वीकृत नहीं" ("not merely rejected"). "Unapproved" means not yet approved (it would be asked about), not denied, so the contrast the sentence draws gets blurred.
  - Suggested: '"…बाहर रखे टूल पहुँच से बाहर हैं, सिर्फ़ बिना स्वीकृति के नहीं।"'
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
