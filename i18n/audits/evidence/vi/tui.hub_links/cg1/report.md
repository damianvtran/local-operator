# vi / tui.hub_links: core catalogue gap audit (cg1)

56 keys, all judged by the round-1 auditor against the English source and context sidecar; 3 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **VI-012** (round 1, MAJOR) `tui.hub_links.action.enable`
  - English: 'Enable auto-push'
  - Value at audit: 'Bật tự động đẩy'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'Bật auto-push'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-013** (round 1, MAJOR) `tui.hub_links.action.pull`
  - English: 'Pull now'
  - Value at audit: 'Kéo về ngay'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'Pull ngay'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-014** (round 1, MAJOR) `tui.hub_links.policy.auto_push.label`
  - English: 'Auto-push'
  - Value at audit: 'Tự động đẩy'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'Auto-push'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-015** (round 1, MAJOR) `tui.hub_links.hint.auto_push`
  - English: 'a auto-push'
  - Value at audit: 'a tự động đẩy'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'a auto-push'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-016** (round 1, MAJOR) `tui.hub_links.confirm.title`
  - English: 'Enable auto-push to {source}?'
  - Value at audit: 'Bật tự động đẩy tới {source}?'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'Bật auto-push tới {source}?'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-017** (round 1, MAJOR) `tui.hub_links.mark.healthy`
  - English: '{item}; {source}; Subscribe {subscribe}; Auto-push {auto_push}'
  - Value at audit: '{item}; {source}; Đăng ký {subscribe}; Tự động đẩy {auto_push}'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: '{item}; {source}; Đăng ký {subscribe}; Auto-push {auto_push}'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-018** (round 1, MAJOR) `tui.hub_links.reason.no_receipt`
  - English: "Auto-push isn't available yet: pull or publish this item once to set a starting point, then try again."
  - Value at audit: 'Chưa dùng được tự động đẩy: hãy kéo về hoặc xuất bản mục này một lần để tạo điểm xuất phát, rồi thử lại.'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'Chưa dùng được auto-push: hãy pull hoặc xuất bản mục này một lần để tạo điểm xuất phát, rồi thử lại.'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-019** (round 1, MAJOR) `tui.hub_links.refusal.ineligible`
  - English: 'Auto-push is not available for this link: {reason}.'
  - Value at audit: 'Không dùng được tự động đẩy cho liên kết này: {reason}.'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'Không dùng được auto-push cho liên kết này: {reason}.'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-020** (round 1, MAJOR) `tui.hub_links.policy.subscribe.help`
  - English: 'Receive updates from {source}. Manual pull stays available.'
  - Value at audit: 'Nhận cập nhật từ {source}. Vẫn có thể kéo về thủ công.'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'Nhận cập nhật từ {source}. Vẫn có thể pull thủ công.'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-021** (round 1, MAJOR) `tui.hub_links.reason.baseline_unknown`
  - English: "This link's history isn't known yet, so it can't publish unattended. Pull or resolve it first."
  - Value at audit: 'Chưa biết lịch sử của liên kết này nên không thể xuất bản khi không có người giám sát. Hãy kéo về hoặc giải quyết nó trước.'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'Chưa biết lịch sử của liên kết này nên không thể xuất bản khi không có người giám sát. Hãy pull hoặc giải quyết nó trước.'
  - Disposition: fixed (remediation core-fix, session 5e0b68c135be)
- **VI-022** (round 1, MAJOR) `tui.hub_links.confirm.concurrent`
  - English: 'Concurrent hub changes are merged or paused for review, never overwritten silently.'
  - Value at audit: 'Các thay đổi đồng thời trên hub sẽ được hợp nhất hoặc tạm dừng để xem xét, không bao giờ bị ghi đè âm thầm.'
  - Problem: Glossary §2 wave-2 kept-English set: `push`/`pull`/`merge` (git/hub operations) stay English, never translated; vi neighbours use push/pull (tui.review.activity_push, wire.settings.network_sync_debounce_s). 'đẩy'/'kéo về'/'hợp nhất' break the hub vocabulary shared with the `lop hub` CLI.
  - Suggested: 'Các thay đổi đồng thời trên hub sẽ được merge hoặc tạm dừng để xem xét, không bao giờ bị ghi đè âm thầm.'
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
