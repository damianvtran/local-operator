# vi / wire.exec: core catalogue gap audit (cg1)

17 keys, all judged by the round-1 auditor against the English source and context sidecar; 3 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **VI-054** (round 1, MINOR) `wire.exec.approval_advisory`
  - English: "Warning: this run cannot ask for approval (no terminal attached), so any tool call that needs approval will be denied.\n  Remedies: --control parks cards for a supervisor; --yolo auto-approves every tier; --tools NAME[,NAME] pre-approves the listed tools and bounds this run's reach to them; tool_approval_mode: auto in the config answers every call the same way for every run."
  - Value at audit: 'Cảnh báo: lần chạy này không thể xin phê duyệt (không có terminal gắn kèm), nên mọi lệnh gọi công cụ cần phê duyệt sẽ bị từ chối.\n  Cách khắc phục: --control tạm giữ các thẻ cho người giám sát; --yolo tự động chấp thuận mọi bậc; --tools NAME[,NAME] chấp thuận trước các công cụ được liệt kê và giới hạn phạm vi của lần chạy này trong đó; tool_approval_mode: auto trong cấu hình trả lời mọi lệnh gọi theo cùng một cách cho mọi lần chạy.'
  - Problem: Glossary: approvals → «phê duyệt» (used 9× in vi neighbours and in this wave's own cli.exec_help.control/tools); 'chấp thuận' is a second, unglossed term for the same safety act, so the user sees two words for one gate. Here 'phê duyệt' and 'chấp thuận' sit in the same message for the same act.
  - Suggested: '… --yolo tự động phê duyệt mọi bậc; --tools NAME[,NAME] phê duyệt trước các công cụ được liệt kê …'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)

## Auditor reports (locale-wide excerpts, verbatim)
### Round 1 (session 3ca692471511)
# vi core catalogue patch audit — independent
Verdict: **CHANGES_REQUESTED** — BLOCKER 0 · MAJOR 31 · MINOR 17 · NIT 5 · 53 findings over 1311 values in 56 namespaces
Auditor: i18n-auditor subagent, model anthropic/claude-opus-5-5 (no fallback). Independent of the translator session 3ef6295a5545. No MT/translation service used.
Inputs: patches in core-i18n/vi/*.json; English source, context sidecars, glossary, vi style guide and existing vi catalogues read from origin/main (extracted via one `git cat-file --batch`; copies under core-audit/vi/src/).
## What blocks PASS (31 MAJOR)

## Not verified
- No rendering in the real TUI and no repo validator run, because the brief makes the repo read-only with no builds or tests. Widths were checked with rich.cells.cell_len only where the sidecar declares maxCells. Widths of tui strings without a budget were not measured against live layouts.
- The audit-ledger row was **not** recorded and no tracked evidence bundle was written. The brief made the repo read-only, which overrides the auditor profile's ledger step. Outputs live only in this scratchpad: someone must copy them into i18n/audits/evidence/vi/… and add the ledger row once the remediation passes.
- NIT findings are capped at 5 per the auditor brief. Two lower-value NITs were dropped.
### Round 2 (session a3954428c96c)
# vi core catalogue patches — independent round-2 audit
**Verdict: CHANGES_REQUESTED.** BLOCKER 0 · MAJOR 2 · MINOR 5 · NIT 2 (9 findings, in `findings.jsonl`).
- **Auditor:** i18n-auditor, session a3954428c96c, model `anthropic/claude-opus-5-5`, no fallback.
- **Independence:** I did not write these values or the round-1 findings. The translator was session 3ef6295a5545, and round 1 was a separate auditor.
- **Method:** No MT or translation service was used. The repo was read-only: no builds, tests or worktrees.
- **Sources:** glossary, vi style guide, and en catalogues + context sidecars from `origin/main` @ `5a6683f6815a1c667a85fb8aa7c76a156bf01c32`.

## Not verified
- I ran no repo validator and did no live TUI render, because the brief made the repo read-only with no builds or tests. Widths are verified only where the sidecar declares `maxCells`.
- I wrote no ledger row or tracked evidence bundle (the repo is read-only). Someone must copy these outputs into `i18n/audits/evidence/vi/…` once a later round passes.
- I did not re-read the ~1,200 untouched values beyond the sweeps listed in (C).
### Round 3 (session 0cc22c4c8b4f)
# vi core catalogue patches — independent round-3 audit
**Verdict: PASS.** BLOCKER 0 · MAJOR 0 · MINOR 2 · NIT 1 (3 findings in `findings.jsonl`).
- **Auditor:** i18n-auditor, session 0cc22c4c8b4f, model `anthropic/claude-opus-5-5`, no fallback. I am independent: I did not write the translations, the round-1 findings or the round-2 findings (translator 3ef6295a5545; round 2 a3954428c96c).
- **Method:** reasoning only, with no MT or back-translation. Read-only on the repo: no builds, tests or worktrees. To work around slow `git show`, I extracted the en catalogues and sidecars from `origin/main` @ `1774a28df5a96cb0bffb9f4673c78e9b6d82e2b6` into my own scratch with `git archive`. I read the glossary (including the Wave-2 kept-English set) and `style-guides/vi.md` first.
## (A) The 8 changed values (changes.jsonl)

## Not verified
- I did not render anything in a real TUI. Width is computed, not seen in a frame.
- The 196 tui.import/library/hub_links and other values that lack sidecar context entries could only be checked mechanically and swept. They have no maxCells to measure.
- I did not re-read the untouched values beyond the sweep.
- I did not record an audit-ledger row or evidence bundle, because this task is read-only on the repo.
