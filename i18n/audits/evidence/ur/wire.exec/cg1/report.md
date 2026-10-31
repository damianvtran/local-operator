# ur / wire.exec: core catalogue gap audit (cg1)

17 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **UR-026** (round 1, MINOR) `wire.exec.exec_epilog`
  - English: "Examples:\n  lop exec 'Review the change' --team release --background\n  printf 'Inspect this report' | lop exec --profile reviewer\n  lop exec --goal 'Finish the checklist' --loop 3 --name 'Night audit'\n  lop exec --resume SESSION_ID --loop-goal 'Verify every acceptance criterion'\n  lop exec --status JOB_ID\n\nResume restores the transcript, team, profile, goal and name first; explicit\nstartup flags override their own slots. Team and profile may coexist. A loop\nruns after the optional initial prompt; --loop N counts continuations only.\nSaved loop progress is visible on resume, but iterations never replay automatically.\nA goal alone does not run a model. Prompts are literal, not slash commands.\n--goal only SETS the objective; unlike the TUI's /goal it does not also\nsend that text as a message, because the prompt is exec's message channel.\n--agent/--agent-id select legacy agent data and are mutually exclusive.\nDefault non-TTY approvals deny. --control may wait for a supervisor; --yolo\nis an explicit override, never implied by --background or --team.\n--tools bounds what the run can reach: excluded tools are unreachable.\nForeground events/text use stdout; receipts use stderr. Detached runs log\nboth streams and print distinct job/session IDs. Use lop --resume SESSION_ID\nto view a live run or resume a finished one; exec refuses a live owner."
  - Value at audit: "مثالیں:\n  lop exec 'Review the change' --team release --background\n  printf 'Inspect this report' | lop exec --profile reviewer\n  lop exec --goal 'Finish the checklist' --loop 3 --name 'Night audit'\n  lop exec --resume SESSION_ID --loop-goal 'Verify every acceptance criterion'\n  lop exec --status JOB_ID\n\nResume پہلے ٹرانسکرپٹ، ٹیم، پروفائل، ہدف اور نام بحال کرتا ہے؛ واضح\nstartup فلیگز اپنے اپنے سلاٹ کو اوور رائڈ کرتے ہیں۔ ٹیم اور پروفائل ساتھ رہ سکتے ہیں۔\nلوپ اختیاری ابتدائی پرامپٹ کے بعد چلتا ہے؛ --loop N صرف تسلسل گنتا ہے۔\nمحفوظ شدہ لوپ کی پیش رفت resume پر نظر آتی ہے، لیکن تکرارات خودکار طور پر دوبارہ نہیں چلتیں۔\nصرف ہدف ماڈل نہیں چلاتا۔ پرامپٹس لفظی ہیں، سلیش کمانڈز نہیں۔\n--goal صرف مقصد SET کرتا ہے؛ TUI کے /goal کے برعکس یہ متن کو\nپیغام کے طور پر نہیں بھیجتا، کیونکہ پرامپٹ exec کا پیغام چینل ہے۔\n--agent/--agent-id پرانے ایجنٹ ڈیٹا کا انتخاب کرتے ہیں اور باہم مانع ہیں۔\nڈیفالٹ غیر TTY منظوریاں مسترد ہوتی ہیں۔ --control سپروائزر کا انتظار کر سکتا ہے؛ --yolo\nواضح اوور رائڈ ہے، --background یا --team سے کبھی مفروضہ نہیں ہوتا۔\n--tools طے کرتا ہے کہ رن کیا حاصل کر سکتا ہے: خارج شدہ ٹولز ناقابلِ رسائی ہیں۔\nفورگراؤنڈ ایونٹس/متن stdout استعمال کرتے ہیں؛ رسیدیں stderr۔ الگ کیے گئے رنز دونوں\nاسٹریمز لاگ کرتے ہیں اور الگ الگ جاب/سیشن IDs چھاپتے ہیں۔ lop --resume SESSION_ID\nلائیو رن دیکھنے یا مکمل ہوئے رن کو دوبارہ شروع کرنے کے لیے استعمال کریں؛ exec لائیو مالک کو مسترد کرتا ہے۔"
  - Problem: Prose words left in Latin outside tokens: 'Resume', 'startup', and '--goal صرف مقصد SET کرتا ہے'. These are not flags or commands (English caps is emphasis). Reads as machine-mixed.
  - Suggested: 'Resume → «دوبارہ شروع کرنا»، startup فلیگز → «آغاز کے فلیگز»، «--goal صرف مقصد مقرر کرتا ہے»'
  - Disposition: fixed (remediation core-fix, session be04c022ebd4)
- **UR2-003** (round 2, NIT) `wire.exec.tools_unreachable`
  - English: 'Warning: --tools names no tool this run could reach: {names}. The session reached {count} tool(s).'
  - Value at audit: 'انتباہ: --tools کسی ایسے ٹول کا نام نہیں لیتا جس تک یہ رن پہنچ سکے: {names}۔ سیشن {count} ٹول(ز) تک پہنچا۔'
  - Problem: Sweep recurrence of the UR-006 class: the bracket plural hack «ٹول(ز)» copies English 'tool(s)'. It is not Urdu morphology.
  - Suggested: '…سیشن {count} ٹولز تک پہنچا۔ (or make {count} an ICU plural: one {# ٹول} other {# ٹولز}; argument name unchanged)'
  - Disposition: open at PASS: minor/nit accepted, value unchanged

## Auditor reports (locale-wide excerpts, verbatim)
### Round 1 (session dab7435f3cd0)
# ur core-catalogue patch audit — independent pass
Auditor: i18n-auditor subagent, session dab7435f3cd0, model anthropic/claude-opus-5-5 (as listed for this session; no fallback observed). Independent from translator session 3ef6295a5545. No MT service was used. Repo read-only (git show origin/main only); no builds, tests or worktrees.
**Verdict: CHANGES_REQUESTED.** BLOCKER 0 · MAJOR 1 · MINOR 18 · NIT 5.
The one MAJOR is `tui.hub_links.policy.auto_push.help`. «محفوظ ہونے پر» can be read as "when saved" as well as "when safe", and this locale uses محفوظ کریں for Save. So the string can read as consent to publish on every save, which is broader than what the English allows.

## What I could not verify
- Rendered width in a real terminal. Widths were measured with rich.cells.cell_len on the raw template (placeholders unexpanded), and only for the 30 keys with maxCells.
- RTL rendering in iTerm2 or other terminals.
- The repo validator and the ledger command. Running them was out of scope because the task is read-only, so no ledger row was recorded.
- Whether the context-less namespaces (tui.import, tui.library, wire.errors.agents/auth/files/media, wire.exec, wire.slash.library) have UI constraints beyond what their English shows.
- Most patched namespaces have no existing ur neighbour on origin/main. Terminology was cross-checked against the 44 ur catalogues that do exist.
### Round 2 (session 1a185836d3d9)
# ur core-catalogue patches — independent round-2 audit
Auditor: i18n-auditor subagent, model anthropic/claude-opus-5-5 (no fallback observed). I did not translate these values and did not write the round-1 findings. No MT service was used. The repo was read-only: I used `git show origin/main` only, with no builds, tests, worktrees, validator or ledger command.
**Verdict: PASS.** BLOCKER 0 · MAJOR 0 · MINOR 2 · NIT 3. Details are in findings.jsonl.
## (A) The 62 changed keys
I judged every after-value fresh against the English, the glossary and the ur style guide. All 62 match the current patch files. All are accepted, and the notes below record why.

## What I could not verify
- Rendered output in a real terminal, including RTL/bidi on iTerm2. Widths are template widths, measured with placeholders unexpanded.
- The repo validator and the ledger write, because the task is read-only. No ledger row was recorded.
- Gender and number agreement in plural/select arms is right for the values I read, but untouched values were only swept by pattern, not re-read.
