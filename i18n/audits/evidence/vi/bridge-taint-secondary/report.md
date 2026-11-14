# Second independent safety audit — wire.bridge_taint (vi)

Auditor: reviewer (radientdev team), session dee255be2b29, model **deepseek/deepseek-flash**, read from the session header; no provider fallback.

Source: frozen English `330a111628350378ede70f1ec5d9a432239d59716b5ee1b0e668de39696ef4d9`; context `bdc6e370fc80…`; audited target sha256 `9850b72761959764fb531fe1759989f5cee60db4d19f0d47dc6232402dcf907a`.

Classification: high-visibility/safety copy (approval-gate egress latch). This is the SECOND of the two independent audits `docs/I18N.md` step 3 requires; the first audit is `bridge-taint-primary-r2` (anthropic/claude-opus-5-5). This audit runs on a different agent and a different model from both the translator and the first auditor.

Per-key verdicts (English meaning judged first, then compared with the first audit):

- **card_notice** — PASS. teammate's agent = 'tác nhân của một đồng đội' (a teammate, another person; glossary tác nhân); 'gửi gì và gửi đến đâu trước khi cho phép' keeps the check; bạn. Referent rendered as 'thao tác này' (this action) rather than bare 'this': same pending action, no softening.
- **card_options_off** — PASS. 'Allow all' quoted, Latin auto, 'trong terminal' kept, both commands verbatim.
- **full_request** — PASS. 'Yêu cầu đầy đủ: {request}', placeholder intact.
- **grant_suspended** — PASS. 'đã tạm dừng' = paused; 'giờ đây' = now; scope 'Các lệnh gọi công cụ có thể gửi dữ liệu ra ngoài hoặc tạo thay đổi' read restrictively (a non-restrictive reading would take commas); 'sẽ hỏi bạn trước' = ask you first.
- **ack_refused** — PASS. 'vẫn đang tắt' = still off; 'từ đây' = from here; 'phê duyệt' plain verb (glossary), nothing quoted.
- **card_notice_other_session** — PASS. 'một phiên hoặc thiết bị khác' = own session/device, no team word; 'có thể chứa' possibility.
- **grant_suspended_other_session** — PASS. own-session/device; possibility; restrictive scope clause unchanged.
- **ack_refused_other_session** — PASS. own-session/device, possibility, 'từ đây', plain phê duyệt, terminal kept.

Mechanical: key set and order = en (8/8); every string NFC; no bidi/zero-width controls; no ASCII apostrophe; {request}, /approvals auto and /new byte-identical with counts equal to en; 'auto' Latin everywhere; Agent-messages title byte-contained in the same three keys as en; only quoted span is the TUI label 'Allow all'; sentence-terminator counts equal to en per locale terminator; balanced quotes/parentheses; per-locale punctuation (fr U+202F before ':' and inside «»; hi danda; ur ۔/، with no danda; zh full-width with no CJK/Latin spacing; es/vi/ru no space before punctuation).

Agent messages title for this locale: `Tin nhắn tác nhân` — byte-identical inside the three keys that carry the product name, as en does.

Findings: none blocking. Non-gating nits carried from the first audit (no wording changed, deferred in the PR thread): zh-CN 请 where en has no 'please'; vi without 'mà' in the restrictive scope clause. This audit adds one non-gating nit of its own: vi and zh-CN render the English bare 'this' as 'thao tác này' / '这项操作' ('this action') — the referent is the pending action being allowed, so the meaning holds and nothing is softened.

Verdict: PASS at the audited bytes; 0 blocker/major. Linguistic/copy audit only — no native rendered flow or RTL paint proof is claimed here.
