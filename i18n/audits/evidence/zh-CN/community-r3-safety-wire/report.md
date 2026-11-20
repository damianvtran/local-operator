# PR #4221 independent safety audit: zh-CN / wire.community
Reviewer: reviewer / openai/gpt-6.2-sol / 08c6dfffd961
Candidate: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + round-2 combined patch (tree `13010a25c61439070020148c7f01d0a6261a9f09`).
Approval-card keys only. The card describes audience, moderation, attribution and per-post consent, not the TUI keyboard gate. None of the five strings asserts approval or publication has already happened.

- `wire.community.post_card_audience` — PASS: 所有成员 names the full audience, and 先由社区版主审核 puts moderator review before visibility.
- `wire.community.post_card_each_time` — PASS: 每条帖子 and 还没有长期权限 keep per-post confirmation and no standing permissions.
- `wire.community.post_card_label` — PASS: 你本人的帖子 and 看不出它来自你的智能体 preserve attribution and invisible agent origin.
- `wire.community.post_card_tainted` — PASS: 任何长期权限都不适用 before a new session; 只有你的回答才能 preserves the exclusive human decision.
- `wire.community.post_card_title` — PASS: 以你的身份 … ？ is a question, not a claim of approval, with both destinations.

Code cross-check: local_operator/tui/widgets/approval.py:1200–1228 (mutually exclusive cut cues), :2292–2295 (unread y opens viewer), :2309–2318 (unread label); local_operator/tui/widgets/community_post_view.py:85–100 (read-to-end hint switch).
Targeted Textual card tests: 4 passed (English flow; not native display or RTL evidence).
