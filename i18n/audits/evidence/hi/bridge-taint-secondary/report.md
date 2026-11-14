# Second independent safety audit — wire.bridge_taint (hi)

Auditor: reviewer (radientdev team), session dee255be2b29, model **deepseek/deepseek-flash**, read from the session header; no provider fallback.

Source: frozen English `330a111628350378ede70f1ec5d9a432239d59716b5ee1b0e668de39696ef4d9`; context `bdc6e370fc80…`; audited target sha256 `32702f7c55b9943fae16e8f5f83a04c39a4ba953347e32230f5b6275c7b9162e`.

Classification: high-visibility/safety copy (approval-gate egress latch). This is the SECOND of the two independent audits `docs/I18N.md` step 3 requires; the first audit is `bridge-taint-primary-r2` (anthropic/claude-opus-5-5). This audit runs on a different agent and a different model from both the translator and the first auditor.

Per-key verdicts (English meaning judged first, then compared with the first audit):

- **card_notice** — PASS. teammate's agent = 'टीम के किसी साथी के एजेंट' (a person on the team, another person's agent); 'देख लें कि यह क्या और कहाँ भेजता है' keeps check-what-and-where; आप.
- **card_options_off** — PASS. 'Allow all' quoted, Latin auto, 'टर्मिनल में' kept, both commands verbatim; danda full stops.
- **full_request** — PASS. 'पूरा अनुरोध: {request}' label, placeholder intact.
- **grant_suspended** — PASS. 'अपने-आप स्वीकृति रुकी हुई है' = paused; scope 'जो टूल कॉल … भेज सकती हैं या बदलाव कर सकती हैं, वे' restrictive correlative (जो…वे), not every call; 'पहले आपसे पूछेंगी' = ask you first.
- **ack_refused** — PASS. 'अभी भी बंद है' = still off; 'यहाँ से' = from here; 'स्वीकृत कर सकते हैं' plain verb (glossary), nothing quoted.
- **card_notice_other_session** — PASS. 'किसी दूसरे सत्र या डिवाइस' = own session/device (सत्र per glossary), no team word; 'हो सकता है' possibility.
- **grant_suspended_other_session** — PASS. own-session/device; possibility; restrictive जो…वे scope.
- **ack_refused_other_session** — PASS. own-session/device, possibility, 'यहाँ से', plain स्वीकृत, टर्मिनल में kept.

Mechanical: key set and order = en (8/8); every string NFC; no bidi/zero-width controls; no ASCII apostrophe; {request}, /approvals auto and /new byte-identical with counts equal to en; 'auto' Latin everywhere; Agent-messages title byte-contained in the same three keys as en; only quoted span is the TUI label 'Allow all'; sentence-terminator counts equal to en per locale terminator; balanced quotes/parentheses; per-locale punctuation (fr U+202F before ':' and inside «»; hi danda; ur ۔/، with no danda; zh full-width with no CJK/Latin spacing; es/vi/ru no space before punctuation).

Agent messages title for this locale: `एजेंट संदेश` — byte-identical inside the three keys that carry the product name, as en does.

Findings: none blocking. Non-gating nits carried from the first audit (no wording changed, deferred in the PR thread): zh-CN 请 where en has no 'please'; vi without 'mà' in the restrictive scope clause. This audit adds one non-gating nit of its own: vi and zh-CN render the English bare 'this' as 'thao tác này' / '这项操作' ('this action') — the referent is the pending action being allowed, so the meaning holds and nothing is softened.

Verdict: PASS at the audited bytes; 0 blocker/major. Linguistic/copy audit only — no native rendered flow or RTL paint proof is claimed here.
