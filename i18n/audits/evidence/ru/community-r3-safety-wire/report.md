# PR #4221 independent safety audit: ru / wire.community
Reviewer: reviewer / openai/gpt-6.2-sol / 08c6dfffd961
Candidate: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + round-2 combined patch (tree `13010a25c61439070020148c7f01d0a6261a9f09`).
Approval-card keys only. The card describes audience, moderation, attribution and per-post consent, not the TUI keyboard gate. None of the five strings asserts approval or publication has already happened.

- `wire.community.post_card_audience` — PASS: Все участники is the entire channel audience; moderation comes before anyone sees it.
- `wire.community.post_card_each_time` — PASS: Каждая публикация требует вашего ответа and пока нет retain per-post decision and no standing permission.
- `wire.community.post_card_label` — PASS: Ваша собственная публикация and не увидят preserve the user's attribution and hidden agent authorship.
- `wire.community.post_card_tainted` — PASS: До новой сессии … не действуют suspends existing grants; только ваш ответ retains the exclusive human decision.
- `wire.community.post_card_title` — PASS: От вашего имени? is an explicit posting question with channel and community.

Code cross-check: local_operator/tui/widgets/approval.py:1200–1228 (mutually exclusive cut cues), :2292–2295 (unread y opens viewer), :2309–2318 (unread label); local_operator/tui/widgets/community_post_view.py:85–100 (read-to-end hint switch).
Targeted Textual card tests: 4 passed (English flow; not native display or RTL evidence).
