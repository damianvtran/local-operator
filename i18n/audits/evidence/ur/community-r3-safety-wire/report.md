# PR #4221 independent safety audit: ur / wire.community
Reviewer: reviewer / openai/gpt-6.2-sol / 08c6dfffd961
Candidate: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + round-2 combined patch (tree `13010a25c61439070020148c7f01d0a6261a9f09`).
Approval-card keys only. The card describes audience, moderation, attribution and per-post consent, not the TUI keyboard gate. None of the five strings asserts approval or publication has already happened.

- `wire.community.post_card_audience` — PASS: تمام اراکین is all channel members; کسی کے دیکھنے سے پہلے puts moderator check before any visibility.
- `wire.community.post_card_each_time` — PASS: ہر پوسٹ and کوئی مستقل اجازت نہیں retain one prompt per post, no standing permission yet.
- `wire.community.post_card_label` — PASS: اپنی پوسٹ and دوسروں … نہیں keep own attribution and hidden agent origin.
- `wire.community.post_card_tainted` — PASS: نئے سیشن تک کوئی مستقل اجازت لاگو نہیں ہوتی suspends standing permission until a new session; صرف آپ کا جواب preserves human decision.
- `wire.community.post_card_title` — PASS: آپ کے نام سے … ؟ is a question about posting in the reader's name with both destination tokens.

Code cross-check: local_operator/tui/widgets/approval.py:1200–1228 (mutually exclusive cut cues), :2292–2295 (unread y opens viewer), :2309–2318 (unread label); local_operator/tui/widgets/community_post_view.py:85–100 (read-to-end hint switch).
Targeted Textual card tests: 4 passed (English flow; not native display or RTL evidence).
