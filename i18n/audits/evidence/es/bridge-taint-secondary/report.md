# Second independent safety audit — wire.bridge_taint (es)

Auditor: reviewer (radientdev team), session dee255be2b29, model **deepseek/deepseek-flash**, read from the session header; no provider fallback.

Source: frozen English `330a111628350378ede70f1ec5d9a432239d59716b5ee1b0e668de39696ef4d9`; context `bdc6e370fc80…`; audited target sha256 `41e7841f1f8bfb35473f535a122eb5f44ab34bc798ded114d7746c79d73ef79f`.

Classification: high-visibility/safety copy (approval-gate egress latch). This is the SECOND of the two independent audits `docs/I18N.md` step 3 requires; the first audit is `bridge-taint-primary-r2` (anthropic/claude-opus-5-5). This audit runs on a different agent and a different model from both the translator and the first auditor.

Per-key verdicts (English meaning judged first, then compared with the first audit):

- **card_notice** — PASS. teammate's agent = 'el agente de un compañero de equipo' (another person, not the user's own agents); 'Revisa qué envía esto y a dónde antes de permitirlo' keeps check-what-and-where-before-allowing; tú.
- **card_options_off** — PASS. 'Allow all' kept as the TUI label and quoted; 'el modo auto' + Latin auto; both sentences, /approvals auto and /new verbatim, 'en la terminal' kept.
- **full_request** — PASS. 'Solicitud completa: {request}' label; placeholder intact.
- **grant_suspended** — PASS. 'está en pausa' = paused; 'ahora incluye' = now; scope 'Las llamadas a herramientas que pueden enviar datos al exterior o hacer cambios' restrictive, not every call; 'te pedirán permiso primero' = will ask you first.
- **ack_refused** — PASS. 'sigue desactivada' = still off; 'no se puede activar desde aquí' = from here; 'puedes aprobar cada solicitud desde su tarjeta' plain verb aprobar, not a label; reassurance is the English reassurance only.
- **card_notice_other_session** — PASS. 'otra sesión o dispositivo' = the user's own other session/device, no team word; 'puede contener' keeps the possibility.
- **grant_suspended_other_session** — PASS. same own-session/device reading; 'puede contener' possibility; restrictive scope clause unchanged.
- **ack_refused_other_session** — PASS. own-session/device reading, possibility kept, 'desde aquí', plain aprobar, in-the-terminal kept.

Mechanical: key set and order = en (8/8); every string NFC; no bidi/zero-width controls; no ASCII apostrophe; {request}, /approvals auto and /new byte-identical with counts equal to en; 'auto' Latin everywhere; Agent-messages title byte-contained in the same three keys as en; only quoted span is the TUI label 'Allow all'; sentence-terminator counts equal to en per locale terminator; balanced quotes/parentheses; per-locale punctuation (fr U+202F before ':' and inside «»; hi danda; ur ۔/، with no danda; zh full-width with no CJK/Latin spacing; es/vi/ru no space before punctuation).

Agent messages title for this locale: `Mensajes de agentes` — byte-identical inside the three keys that carry the product name, as en does.

Findings: none blocking. Non-gating nits carried from the first audit (no wording changed, deferred in the PR thread): zh-CN 请 where en has no 'please'; vi without 'mà' in the restrictive scope clause. This audit adds one non-gating nit of its own: vi and zh-CN render the English bare 'this' as 'thao tác này' / '这项操作' ('this action') — the referent is the pending action being allowed, so the meaning holds and nothing is softened.

Verdict: PASS at the audited bytes; 0 blocker/major. Linguistic/copy audit only — no native rendered flow or RTL paint proof is claimed here.
