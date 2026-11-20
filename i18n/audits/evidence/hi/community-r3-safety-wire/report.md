# PR #4221 independent safety audit: hi / wire.community
Reviewer: reviewer / openai/gpt-6.2-sol / 08c6dfffd961
Candidate: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + round-2 combined patch (tree `13010a25c61439070020148c7f01d0a6261a9f09`).
Approval-card keys only. The card describes audience, moderation, attribution and per-post consent, not the TUI keyboard gate. None of the five strings asserts approval or publication has already happened.

- `wire.community.post_card_audience` — PASS: सभी सदस्य names all channel members; moderation happens किसी के देखने से पहले, before anybody sees it.
- `wire.community.post_card_each_time` — PASS: हर पोस्ट पर and कोई स्थायी अनुमति नहीं keep individual consent with no standing grant yet.
- `wire.community.post_card_label` — PASS: आपकी अपनी पोस्ट and पता नहीं चलेगा keep self-attribution and hidden agent provenance.
- `wire.community.post_card_tainted` — PASS: नए सत्र तक … लागू नहीं होती suspends standing grants until a new session; सिर्फ़ आपका जवाब leaves decision with the human.
- `wire.community.post_card_title` — PASS: आपके नाम से … ? asks about posting in the reader's name with channel and community.

Code cross-check: local_operator/tui/widgets/approval.py:1200–1228 (mutually exclusive cut cues), :2292–2295 (unread y opens viewer), :2309–2318 (unread label); local_operator/tui/widgets/community_post_view.py:85–100 (read-to-end hint switch).
Targeted Textual card tests: 4 passed (English flow; not native display or RTL evidence).
