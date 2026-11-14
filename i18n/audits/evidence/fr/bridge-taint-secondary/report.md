# Second independent safety audit — wire.bridge_taint (fr)

Auditor: reviewer (radientdev team), session dee255be2b29, model **deepseek/deepseek-flash**, read from the session header; no provider fallback.

Source: frozen English `330a111628350378ede70f1ec5d9a432239d59716b5ee1b0e668de39696ef4d9`; context `bdc6e370fc80…`; audited target sha256 `6e00b366295c19a8128f094eef651c1b3ff97122d69fee6e6addaa593b28b1fa`.

Classification: high-visibility/safety copy (approval-gate egress latch). This is the SECOND of the two independent audits `docs/I18N.md` step 3 requires; the first audit is `bridge-taint-primary-r2` (anthropic/claude-opus-5-5). This audit runs on a different agent and a different model from both the translator and the first auditor.

Per-key verdicts (English meaning judged first, then compared with the first audit):

- **card_notice** — PASS. teammate's agent = 'l'agent d'un coéquipier' (another person); 'Vérifiez ce qui est envoyé, et vers où, avant de l'autoriser' keeps the check; vous.
- **card_options_off** — PASS. '« Allow all »' quoted with U+202F inside, Latin auto, 'dans le terminal' kept, both commands verbatim.
- **full_request** — PASS. 'Requête complète : {request}' with U+202F before the colon; placeholder intact.
- **grant_suspended** — PASS. 'est suspendue' = paused; 'contient désormais' = now; scope 'Les appels d'outils qui peuvent envoyer des données vers l'extérieur ou apporter des modifications' restrictive via 'qui'; 'vous demanderont d'abord votre accord' = ask you first.
- **ack_refused** — PASS. 'reste désactivée' = still off; 'd'ici' = from here; 'Vous pouvez toujours approuver chaque demande depuis sa carte' plain verb, no label; ':' carries U+202F.
- **card_notice_other_session** — PASS. 'd'une autre session ou d'un autre appareil' = own session/device (appareil), no team word; 'qui peut contenir' possibility.
- **grant_suspended_other_session** — PASS. own-session/device reading; restrictive 'qui'; possibility kept.
- **ack_refused_other_session** — PASS. own-session/device, possibility, 'd'ici', plain approuver, terminal kept; U+202F before ':'.

Mechanical: key set and order = en (8/8); every string NFC; no bidi/zero-width controls; no ASCII apostrophe; {request}, /approvals auto and /new byte-identical with counts equal to en; 'auto' Latin everywhere; Agent-messages title byte-contained in the same three keys as en; only quoted span is the TUI label 'Allow all'; sentence-terminator counts equal to en per locale terminator; balanced quotes/parentheses; per-locale punctuation (fr U+202F before ':' and inside «»; hi danda; ur ۔/، with no danda; zh full-width with no CJK/Latin spacing; es/vi/ru no space before punctuation).

Agent messages title for this locale: `Messages entre agents` — byte-identical inside the three keys that carry the product name, as en does.

Findings: none blocking. Non-gating nits carried from the first audit (no wording changed, deferred in the PR thread): zh-CN 请 where en has no 'please'; vi without 'mà' in the restrictive scope clause. This audit adds one non-gating nit of its own: vi and zh-CN render the English bare 'this' as 'thao tác này' / '这项操作' ('this action') — the referent is the pending action being allowed, so the meaning holds and nothing is softened.

Verdict: PASS at the audited bytes; 0 blocker/major. Linguistic/copy audit only — no native rendered flow or RTL paint proof is claimed here.
