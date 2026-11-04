# vi / cli.wake_help: core catalogue gap audit (cg1)

17 keys, all judged by the round-1 auditor against the English source and context sidecar; 3 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **VI-051** (round 1, MINOR) `cli.wake_help.create_notify`
  - English: "notify on completion of the wake's turn (default quiet: scripted arms match the contract; errored turns notify regardless)"
  - Value at audit: 'thông báo khi lượt của lần thức dậy hoàn tất (mặc định im lặng: các cánh tay theo kịch bản khớp với hợp đồng; lượt lỗi luôn thông báo)'
  - Problem: 'scripted arms' means wakes armed by scripts; 'các cánh tay theo kịch bản khớp với hợp đồng' renders it as literal (body) arms — the clause is meaningless.
  - Suggested: 'thông báo khi lượt của lần thức dậy hoàn tất (mặc định im lặng: các lần thức dậy do script đặt khớp với hợp đồng; lượt lỗi luôn thông báo)'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **R2-VI-002** (round 2, MAJOR) `cli.wake_help.create_message`
  - English: 'the self-prompt delivered when it fires (at most {max} characters)'
  - Value at audit: 'lời tự nhắc được gửi khi nó kích hoạt (tối đa {max} ký tự)'
  - Problem: Sweep (kept-English class): «self-prompt» rendered «lời tự nhắc». Glossary/style guide keep «prompt»; the same concept on main is «prompt mà lần thức dậy này…» (tui.schedule.form_message_placeholder), so this also splits the term.
  - Suggested: 'prompt tự gửi khi nó kích hoạt (tối đa {max} ký tự)'
  - Disposition: fixed (remediation core-fix2, session 053117b4ad36)
- **R2-VI-007** (round 2, MINOR) `cli.wake_help.create_notify`
  - English: "notify on completion of the wake's turn (default quiet: scripted arms match the contract; errored turns notify regardless)"
  - Value at audit: 'thông báo khi lượt của lần thức dậy hoàn tất (mặc định im lặng: các lần thức dậy do script đặt khớp với hợp đồng; lượt lỗi luôn thông báo)'
  - Problem: AFTER fixes «cánh tay», but «khớp với hợp đồng» still renders the API/behaviour contract as a legal contract (calque); in developer register «hợp đồng» means a signed agreement.
  - Suggested: '… mặc định im lặng: các lần thức dậy do script đặt khớp với quy ước này; lượt lỗi luôn thông báo)'
  - Disposition: fixed (remediation core-fix2, session 053117b4ad36)
- **R3-VI-002** (round 3, NIT) `cli.wake_help.create_notify`
  - English: "notify on completion of the wake's turn (default quiet: scripted arms match the contract; errored turns notify regardless)"
  - Value at audit: 'thông báo khi lượt của lần thức dậy hoàn tất (mặc định im lặng: các lần thức dậy do script đặt khớp với quy ước này; lượt lỗi luôn thông báo)'
  - Problem: «quy ước này» fixes the legal-contract calque, but «này» ('this') has no antecedent in the string; English says 'the contract' (sidecar: the contract reference stays).
  - Suggested: '… các lần thức dậy do script đặt khớp với quy ước (contract); lượt lỗi luôn thông báo)'
  - Disposition: open at PASS: minor/nit accepted, value unchanged

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
