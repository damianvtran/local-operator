# PR #4221 independent safety audit: es / wire.community
Reviewer: reviewer / openai/gpt-6.2-sol / 08c6dfffd961
Candidate: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + round-2 combined patch (tree `13010a25c61439070020148c7f01d0a6261a9f09`).
Approval-card keys only. The card describes audience, moderation, attribution and per-post consent, not the TUI keyboard gate. None of the five strings asserts approval or publication has already happened.

- `wire.community.post_card_audience` — PASS: Todos los miembros is the full audience; antes de que nadie lo vea means before anyone sees it.
- `wire.community.post_card_each_time` — PASS: Cada publicación / no hay permisos permanentes keeps individual prompting and no standing grant.
- `wire.community.post_card_label` — PASS: Tu propia publicación and nadie verá que vino de tu agente retain own attribution and lack of agent label.
- `wire.community.post_card_tainted` — PASS: Ningún permiso permanente applies until a new session; solo tu respuesta keeps the decision with the reader.
- `wire.community.post_card_title` — PASS: Como tú is interrogative, not a claim that posting has happened; channel and community retained.

Code cross-check: local_operator/tui/widgets/approval.py:1200–1228 (mutually exclusive cut cues), :2292–2295 (unread y opens viewer), :2309–2318 (unread label); local_operator/tui/widgets/community_post_view.py:85–100 (read-to-end hint switch).
Targeted Textual card tests: 4 passed (English flow; not native display or RTL evidence).
