# PR #4221 independent safety audit: vi / wire.community
Reviewer: reviewer / openai/gpt-6.2-sol / 08c6dfffd961
Candidate: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + round-2 combined patch (tree `13010a25c61439070020148c7f01d0a6261a9f09`).
Approval-card keys only. The card describes audience, moderation, attribution and per-post consent, not the TUI keyboard gate. None of the five strings asserts approval or publication has already happened.

- `wire.community.post_card_audience` — PASS: Mọi thành viên names the full audience; kiểm tra trước khi puts community moderation before anyone sees it.
- `wire.community.post_card_each_time` — PASS: Mỗi bài đăng and chưa có quyền thường trực retain per-post prompts and no standing permission.
- `wire.community.post_card_label` — PASS: Chính bạn and người khác sẽ không biết retain self-attribution and hidden agent provenance.
- `wire.community.post_card_tainted` — PASS: Không quyền thường trực nào … cho đến phiên mới means no standing grant applies before a new session; chỉ … mới keeps posting conditional on your answer.
- `wire.community.post_card_title` — PASS: Với tư cách là bạn? asks whether to post as the reader, with channel and community.

Code cross-check: local_operator/tui/widgets/approval.py:1200–1228 (mutually exclusive cut cues), :2292–2295 (unread y opens viewer), :2309–2318 (unread label); local_operator/tui/widgets/community_post_view.py:85–100 (read-to-end hint switch).
Targeted Textual card tests: 4 passed (English flow; not native display or RTL evidence).
