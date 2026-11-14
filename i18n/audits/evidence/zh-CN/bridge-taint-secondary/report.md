# Second independent safety audit — wire.bridge_taint (zh-CN)

Auditor: reviewer (radientdev team), session dee255be2b29, model **deepseek/deepseek-flash**, read from the session header; no provider fallback.

Source: frozen English `330a111628350378ede70f1ec5d9a432239d59716b5ee1b0e668de39696ef4d9`; context `bdc6e370fc80…`; audited target sha256 `d07cd0b934bef7f335edb4e11e8dadf854234618cc3cd656b92303724cb3bbda`.

Classification: high-visibility/safety copy (approval-gate egress latch). This is the SECOND of the two independent audits `docs/I18N.md` step 3 requires; the first audit is `bridge-taint-primary-r2` (anthropic/claude-opus-5-5). This audit runs on a different agent and a different model from both the translator and the first auditor.

Per-key verdicts (English meaning judged first, then compared with the first audit):

- **card_notice** — PASS. teammate's agent = '一位队友的智能体' (a teammate, another person; glossary 智能体); '请先检查这项操作会发送什么内容、发往何处' keeps the check; 请 adds politeness the English does not have (carried NIT).
- **card_options_off** — PASS. '“Allow all”' quoted with mainland “”, Latin auto, '在终端中' kept, both commands verbatim, full-width punctuation.
- **full_request** — PASS. '完整请求：{request}' full-width colon, placeholder intact.
- **grant_suspended** — PASS. '已暂停' = paused; '现已包含' = now; scope '可能向外发送数据或进行更改的工具调用都会' restrictive prenominal modifier, not every call; '先征求你的同意' = ask you first.
- **ack_refused** — PASS. '仍处于关闭状态' = still off; '在这里' = from here; '批准' plain verb, nothing quoted.
- **card_notice_other_session** — PASS. '来自另一个会话或设备' = own session/device (会话), no team word; '可能有' possibility.
- **grant_suspended_other_session** — PASS. own-session/device; possibility; restrictive modifier, 都会 kept.
- **ack_refused_other_session** — PASS. own-session/device, possibility, '在这里', plain 批准, 终端 kept.

Mechanical: key set and order = en (8/8); every string NFC; no bidi/zero-width controls; no ASCII apostrophe; {request}, /approvals auto and /new byte-identical with counts equal to en; 'auto' Latin everywhere; Agent-messages title byte-contained in the same three keys as en; only quoted span is the TUI label 'Allow all'; sentence-terminator counts equal to en per locale terminator; balanced quotes/parentheses; per-locale punctuation (fr U+202F before ':' and inside «»; hi danda; ur ۔/، with no danda; zh full-width with no CJK/Latin spacing; es/vi/ru no space before punctuation).

Agent messages title for this locale: `智能体消息` — byte-identical inside the three keys that carry the product name, as en does.

Findings: none blocking. Non-gating nits carried from the first audit (no wording changed, deferred in the PR thread): zh-CN 请 where en has no 'please'; vi without 'mà' in the restrictive scope clause. This audit adds one non-gating nit of its own: vi and zh-CN render the English bare 'this' as 'thao tác này' / '这项操作' ('this action') — the referent is the pending action being allowed, so the meaning holds and nothing is softened.

Verdict: PASS at the audited bytes; 0 blocker/major. Linguistic/copy audit only — no native rendered flow or RTL paint proof is claimed here.
