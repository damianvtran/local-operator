# vi / cli.agents_help: core catalogue gap audit (cg1)

21 keys, all judged by the round-1 auditor against the English source and context sidecar; 3 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **VI-030** (round 1, MAJOR) `cli.agents_help.push`
  - English: 'Push (upload) an agent to Radient; `--org` and `--hub-id` push through the preview/confirm flow (exit codes: 0 published, 1 error, 2 preview-only with changes pending, 3 declined, 130 cancelled)'
  - Value at audit: 'Đẩy (tải lên) một tác nhân lên Radient; `--org` và `--hub-id` đẩy qua luồng xem trước/xác nhận (mã thoát: 0 đã xuất bản, 1 lỗi, 2 chỉ xem trước còn thay đổi đang chờ, 3 bị từ chối, 130 đã hủy)'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English and are inflected around, never translated; these help lines describe the `lop … push/pull` subcommands, so 'đẩy'/'kéo'/'hợp nhất' also diverge from the command names the user types.
  - Suggested: 'Push (tải lên) một tác nhân lên Radient; `--org` và `--hub-id` push qua luồng xem trước/xác nhận (mã thoát: …)'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-031** (round 1, MAJOR) `cli.agents_help.push_name`
  - English: 'Name of the agent to push to Radient'
  - Value at audit: 'Tên tác nhân cần đẩy lên Radient'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English and are inflected around, never translated; these help lines describe the `lop … push/pull` subcommands, so 'đẩy'/'kéo'/'hợp nhất' also diverge from the command names the user types.
  - Suggested: 'Tên tác nhân cần push lên Radient'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-032** (round 1, MAJOR) `cli.agents_help.push_id`
  - English: 'ID of the agent to push to Radient (explicit overwrite)'
  - Value at audit: 'ID tác nhân cần đẩy lên Radient (ghi đè tường minh)'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English and are inflected around, never translated; these help lines describe the `lop … push/pull` subcommands, so 'đẩy'/'kéo'/'hợp nhất' also diverge from the command names the user types.
  - Suggested: 'ID tác nhân cần push lên Radient (ghi đè tường minh)'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-033** (round 1, MAJOR) `cli.agents_help.pull`
  - English: 'Pull (download) an agent from Radient'
  - Value at audit: 'Kéo (tải xuống) một tác nhân từ Radient'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English and are inflected around, never translated; these help lines describe the `lop … push/pull` subcommands, so 'đẩy'/'kéo'/'hợp nhất' also diverge from the command names the user types.
  - Suggested: 'Pull (tải xuống) một tác nhân từ Radient'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-034** (round 1, MAJOR) `cli.agents_help.pull_id`
  - English: 'ID of the agent to pull from Radient'
  - Value at audit: 'ID tác nhân cần kéo từ Radient'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English and are inflected around, never translated; these help lines describe the `lop … push/pull` subcommands, so 'đẩy'/'kéo'/'hợp nhất' also diverge from the command names the user types.
  - Suggested: 'ID tác nhân cần pull từ Radient'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-035** (round 1, MAJOR) `cli.agents_help.pull_org`
  - English: "Pull from this organization's workspace instead of the public hub (Agent Hub org sharing; requires `lop login radient`)"
  - Value at audit: 'Kéo từ không gian làm việc của tổ chức này thay vì hub công khai (chia sẻ theo tổ chức của Agent Hub; cần `lop login radient`)'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English and are inflected around, never translated; these help lines describe the `lop … push/pull` subcommands, so 'đẩy'/'kéo'/'hợp nhất' also diverge from the command names the user types.
  - Suggested: 'Pull từ không gian làm việc của tổ chức này thay vì hub công khai (…)'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-036** (round 1, MAJOR) `cli.agents_help.sync`
  - English: 'Update installed starters and hub-pulled agents to their latest text'
  - Value at audit: 'Cập nhật các bản khởi đầu đã cài và tác nhân kéo từ hub lên văn bản mới nhất'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English and are inflected around, never translated; these help lines describe the `lop … push/pull` subcommands, so 'đẩy'/'kéo'/'hợp nhất' also diverge from the command names the user types.
  - Suggested: 'Cập nhật các bản khởi đầu đã cài và tác nhân đã pull từ hub lên văn bản mới nhất'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-037** (round 1, MAJOR) `cli.agents_help.sync_all`
  - English: 'Sync every installed starter and hub-pulled agent (default when --name is absent)'
  - Value at audit: 'Đồng bộ mọi bản khởi đầu đã cài và tác nhân kéo từ hub (mặc định khi không có --name)'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English and are inflected around, never translated; these help lines describe the `lop … push/pull` subcommands, so 'đẩy'/'kéo'/'hợp nhất' also diverge from the command names the user types.
  - Suggested: 'Đồng bộ mọi bản khởi đầu đã cài và tác nhân đã pull từ hub (mặc định khi không có --name)'
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
