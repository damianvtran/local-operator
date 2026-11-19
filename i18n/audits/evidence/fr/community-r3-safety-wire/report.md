# PR #4221 independent safety audit: fr / wire.community
Reviewer: reviewer / openai/gpt-6.2-sol / 08c6dfffd961
Candidate: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + round-2 combined patch (tree `13010a25c61439070020148c7f01d0a6261a9f09`).
Approval-card keys only. The card describes audience, moderation, attribution and per-post consent, not the TUI keyboard gate. None of the five strings asserts approval or publication has already happened.

- `wire.community.post_card_audience` — PASS: Tous les membres is the full channel audience; moderator checks before anyone sees it.
- `wire.community.post_card_each_time` — PASS: Chaque publication / pas encore: each post asks, without standing permission.
- `wire.community.post_card_label` — PASS: Votre propre publication and ne verront donc pas retain own attribution and lack of agent visibility.
- `wire.community.post_card_tainted` — PASS: Aucune autorisation permanente … avant une nouvelle session suspends standing permission; only the user's answer can post.
- `wire.community.post_card_title` — PASS: En votre nom is a question about posting in the user's name in the channel and community.

Code cross-check: local_operator/tui/widgets/approval.py:1200–1228 (mutually exclusive cut cues), :2292–2295 (unread y opens viewer), :2309–2318 (unread label); local_operator/tui/widgets/community_post_view.py:85–100 (read-to-end hint switch).
Targeted Textual card tests: 4 passed (English flow; not native display or RTL evidence).
