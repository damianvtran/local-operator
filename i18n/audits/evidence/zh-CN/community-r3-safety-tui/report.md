# PR #4221 independent safety audit: zh-CN / tui.community
Reviewer: reviewer / openai/gpt-6.2-sol / 08c6dfffd961
Candidate: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + round-2 combined patch (tree `13010a25c61439070020148c7f01d0a6261a9f09`).
Approval-card keys only. Short cues say read first; the roomy cue and unread viewer explicitly say read the whole post/to the end. The two short cues do not literally say “full” in English or target; they do not suggest the preview counts as read.

- `tui.community.post_card_cut` — PASS: 按v阅读帖子: press literal v to read the post, not to allow.
- `tui.community.post_allow_unread` — PASS: 先读帖子，再允许 literally puts reading first, allowing afterward.
- `tui.community.post_card_more` — PASS: 按v阅读全文 / 需先读完才能 says read the complete text with v and finish before Allow becomes possible.
- `tui.community.post_view_keys` — PASS: 读到末尾后按esc返回 makes end-of-post reading precede returning.
- `tui.community.post_view_keys_read` — PASS: 已读完 marks only completion; esc returns to Allow or Deny.

Code cross-check: local_operator/tui/widgets/approval.py:1200–1228 (mutually exclusive cut cues), :2292–2295 (unread y opens viewer), :2309–2318 (unread label); local_operator/tui/widgets/community_post_view.py:85–100 (read-to-end hint switch).
Targeted Textual card tests: 4 passed (English flow; not native display or RTL evidence).

R3-M1 (MINOR, inherited from round 2, deferred): on a cut card showing post_card_more, the hint names “Allow” but the unread row label has already changed. Both signals still require reading first. If revisited, change the source hint to say read the whole post before approving, without naming the current button.
