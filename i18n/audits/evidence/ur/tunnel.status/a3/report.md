# Independent translation audit — ur / tunnel.status / a3

Commit: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (branch fix/tunnel-remedies-auto-up). Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4aaeb39ddb2f; neither the i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4aaeb39ddb2f: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (commit 1fce7ecb1f5f); English source drafted by coder session b0300e2c2c7f; model anthropic/claude-opus-5-5 (as stated by the delegating brief; not independently verified); 2026-11-14.
Prior audit: i18n/audits/evidence/ur/tunnel.status/a2/record.json.

**Verdict: passed** — 0 unresolved blocker/major; 1 minor/nit finding(s).

Scope: the 8 keys new in 1fce7ecb1f5f judged in full (meaning, safety, placeholders, glossary, tone, length); the receipt covers the whole namespace (21 keys) at the new English source hash. Previously audited keys were re-read only for terminology consistency, not re-litigated.

Read: docs/I18N.md L260-308, i18n/glossary.md (no tunnel/billing/credit row — corpus usage governs), i18n/style-guides/ur.md, en + context sidecar, every new value in full, corpus term counts across all ur catalogues.

## Keys judged

| key | EN | ur | result |
|---|---|---|---|
| `tunnel_inactive.label` | tunnel not active | ٹنل فعال نہیں | OK |
| `tunnel_inactive.detail` | Radient has paused this tunnel, but its billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review the tunnel at {url}. | Radient نے اس ٹنل کو روک دیا ہے، لیکن اس کی بلنگ درست ہے۔ جس کمپیوٹر پر یہ چل رہا ہے اس پر Local Operator میں فون رسائی دوبارہ منسلک کریں؛ یہ خود بھی دوبارہ کوشش کرتا رہتا ہے۔ اگر یہ رکا ہی رہے تو {url} پر ٹنل کا جائزہ لیں۔ | OK |
| `tunnel_disabled.label` | tunnel turned off in the console | کنسول میں ٹنل بند ہے | OK |
| `tunnel_disabled.detail` | This tunnel is turned off in the Radient console, so remote access stays off until it is turned back on there: {url}. It reconnects by itself once it is on. | یہ ٹنل Radient کنسول میں بند ہے، اس لیے جب تک اسے وہاں دوبارہ آن نہ کیا جائے، ریموٹ رسائی بند رہے گی: {url}۔ آن ہوتے ہی یہ خود دوبارہ منسلک ہو جاتا ہے۔ | OK |
| `billing_inactive.label` | billing not activated | بلنگ ایکٹیویٹ نہیں ہوئی | OK |
| `billing_inactive.detail` | Billing for this tunnel has not been activated, so Radient will not connect it. Open phone access setup in Local Operator on the computer running it and accept the monthly price. | اس ٹنل کی بلنگ ایکٹیویٹ نہیں کی گئی، اس لیے Radient اسے منسلک نہیں کرے گا۔ جس کمپیوٹر پر یہ چل رہا ہے اس پر Local Operator میں فون رسائی کا سیٹ اپ کھولیں اور ماہانہ قیمت قبول کریں۔ | OK |
| `billing_lapsed.label` | billing lapsed | بلنگ رک گئی | UR-02 |
| `billing_lapsed.detail` | This tunnel's billing lapsed because the Radient balance ran out, so Radient will not connect it. Add credit at {url}; it reconnects by itself once the balance is restored. | Radient بیلنس ختم ہونے کی وجہ سے اس ٹنل کی بلنگ رک گئی ہے، اس لیے Radient اسے منسلک نہیں کرے گا۔ {url} پر کریڈٹ شامل کریں؛ بیلنس بحال ہوتے ہی یہ خود دوبارہ منسلک ہو جاتا ہے۔ | OK |

## Judgement

Polite آپ register; Urdu punctuation ۔ ، ؛ only (no । or Latin ;/, in prose); no manual bidi controls; LTR islands (Radient, Local Operator, {url}, console URL, /mobile enable) byte-identical. Terms consistent with audited corpus: فون رسائی (26 prior uses), ٹنل, بلنگ, کریڈٹ, کنسول, ماہانہ قیمت, سیٹ اپ, بیک گراؤنڈ, سائن اِن, ایکٹیویٹ, 'reconnect' as دوبارہ منسلک (= audited mobile.setup.repoint). Gender agreement checked: رسائی f. («منسلک ہو رہی ہے», «ہو جاتی ہے»), ٹنل m. («رکا ہی رہے», «کرتا رہتا ہے»). Safety meanings verified per key: no terminal command (only `/mobile enable` in login_offer); «کوئی رقم نہیں لی جاتی» (reconnect, reconnecting); «سیٹ اپ کبھی کریڈٹ نہیں خریدتا» (billing_lapsed, identical to audited mobile.setup.balance); «بلنگ شروع ہونے سے پہلے کوئی بھی ماہانہ قیمت آپ کو دکھائی جاتی ہے» (login_offer); «کبھی بلنگ ایکٹیویٹ نہیں کرتا اور نہ کریڈٹ خرچ کرتا» (auto_setup.help); «آپ کے اس انتخاب کو تبدیل نہیں کرے گا» (disabled). network/radient_unavailable ask nothing of the reader.

## Findings

| ID | severity | key | finding | suggested fix |
|---|---|---|---|---|
| UR-02 | minor | `tunnel.status.billing_lapsed.label` | «بلنگ رک گئی» — same observation as HI-01: reads 'billing stopped', the out-of-balance cause only appears in the detail sentence. | Optional: «بلنگ رک گئی (بیلنس ختم)» or «بیلنس ختم، بلنگ رکی». |

## Translator open questions (this locale)

- billing-lapsed label: acceptable as «بلنگ رک گئی»; optional cause clarifier (UR-02).
- toggle label infinitive: grammatical, but inconsistent with the corpus polite-imperative toggle labels; recommend the imperative (UR-03).

## Mechanical

check.py --catalogues-only: 0 problems; key parity, placeholders, URLs, literal tokens, bidi controls, punctuation script and no-command checks OK (audit.log).

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame (RTL mirroring and the UR-05 slash placement are unverified in pixels). The Reconnect action label the copy points at lives outside this catalogue and was not compared. Judgement by reasoning only; no MT service used.

This bundle is written in the shared worktree, uncommitted; generate.py was not run (coordinator regenerates).
