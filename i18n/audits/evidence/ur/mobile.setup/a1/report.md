# Independent translation audit — ur / mobile.setup / a1

Commit: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (branch fix/tunnel-remedies-auto-up). Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4aaeb39ddb2f; neither the i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4aaeb39ddb2f: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (commit 1fce7ecb1f5f); English source drafted by coder session b0300e2c2c7f; model anthropic/claude-opus-5-5 (as stated by the delegating brief; not independently verified); 2026-11-14.
Prior audit: i18n/audits/evidence/ur/mobile.setup/r1/record.json.

**Verdict: passed** — 0 unresolved blocker/major; 1 minor/nit finding(s).

Scope: the 7 keys new in 1fce7ecb1f5f judged in full (meaning, safety, placeholders, glossary, tone, length); the receipt covers the whole namespace (86 keys) at the new English source hash. Previously audited keys were re-read only for terminology consistency, not re-litigated.

Read: docs/I18N.md L260-308, i18n/glossary.md (no tunnel/billing/credit row — corpus usage governs), i18n/style-guides/ur.md, en + context sidecar, every new value in full, corpus term counts across all ur catalogues.

## Keys judged

| key | EN | ur | result |
|---|---|---|---|
| `disabled` | This tunnel is turned off in the Radient console. Turn it back on at https://console.radienthq.com/dashboard/tunnels; it reconnects by itself, and setup will not override that choice. | یہ ٹنل Radient کنسول میں بند ہے۔ اسے https://console.radienthq.com/dashboard/tunnels پر دوبارہ آن کریں؛ یہ خود دوبارہ منسلک ہو جاتا ہے، اور سیٹ اپ آپ کے اس انتخاب کو تبدیل نہیں کرے گا۔ | OK |
| `billing_lapsed` | This tunnel's billing lapsed because the Radient balance ran out. Add credit at https://console.radienthq.com/dashboard/tunnels; setup never purchases credit, and the tunnel reconnects by itself once the balance is restored. | Radient بیلنس ختم ہونے کی وجہ سے اس ٹنل کی بلنگ رک گئی ہے۔ https://console.radienthq.com/dashboard/tunnels پر کریڈٹ شامل کریں؛ سیٹ اپ کبھی کریڈٹ نہیں خریدتا، اور بیلنس بحال ہوتے ہی ٹنل خود دوبارہ منسلک ہو جاتا ہے۔ | OK |
| `reconnect` | Radient has paused this tunnel, but its billing is in order. Reconnect to bring phone access back; nothing is charged. | Radient نے اس ٹنل کو روک دیا ہے، لیکن اس کی بلنگ درست ہے۔ فون رسائی واپس لانے کے لیے دوبارہ منسلک کریں؛ کوئی رقم نہیں لی جاتی۔ | OK |
| `network` | This computer could not reach Radient (its network may have changed or dropped). Phone access reconnects by itself once the connection is back; nothing needs doing. | یہ کمپیوٹر Radient تک نہیں پہنچ سکا (شاید اس کا نیٹ ورک بدل گیا ہے یا منقطع ہو گیا ہے)۔ کنکشن بحال ہوتے ہی فون رسائی خود دوبارہ منسلک ہو جاتی ہے؛ آپ کو کچھ کرنے کی ضرورت نہیں۔ | OK |
| `radient_unavailable` | Radient answered with a temporary server error. Nothing is wrong with this computer or its sign-in; phone access reconnects by itself once Radient recovers. | Radient نے ایک عارضی سرور خرابی کا جواب دیا۔ اس کمپیوٹر یا اس کے سائن اِن میں کوئی مسئلہ نہیں؛ Radient بحال ہوتے ہی فون رسائی خود دوبارہ منسلک ہو جاتی ہے۔ | UR-01 |
| `reconnecting` | Phone access is reconnecting in the background after your sign-in. Nothing is charged; this clears by itself in a minute or two. | آپ کے سائن اِن کے بعد فون رسائی بیک گراؤنڈ میں دوبارہ منسلک ہو رہی ہے۔ کوئی رقم نہیں لی جاتی؛ یہ ایک دو منٹ میں خود ٹھیک ہو جائے گا۔ | OK |
| `login_offer` | Reach this computer from the phone app: run /mobile enable. Any monthly price is shown to you before billing starts. | فون ایپ سے اس کمپیوٹر تک رسائی حاصل کریں: /mobile enable چلائیں۔ بلنگ شروع ہونے سے پہلے کوئی بھی ماہانہ قیمت آپ کو دکھائی جاتی ہے۔ | UR-05 |

## Judgement

Polite آپ register; Urdu punctuation ۔ ، ؛ only (no । or Latin ;/, in prose); no manual bidi controls; LTR islands (Radient, Local Operator, {url}, console URL, /mobile enable) byte-identical. Terms consistent with audited corpus: فون رسائی (26 prior uses), ٹنل, بلنگ, کریڈٹ, کنسول, ماہانہ قیمت, سیٹ اپ, بیک گراؤنڈ, سائن اِن, ایکٹیویٹ, 'reconnect' as دوبارہ منسلک (= audited mobile.setup.repoint). Gender agreement checked: رسائی f. («منسلک ہو رہی ہے», «ہو جاتی ہے»), ٹنل m. («رکا ہی رہے», «کرتا رہتا ہے»). Safety meanings verified per key: no terminal command (only `/mobile enable` in login_offer); «کوئی رقم نہیں لی جاتی» (reconnect, reconnecting); «سیٹ اپ کبھی کریڈٹ نہیں خریدتا» (billing_lapsed, identical to audited mobile.setup.balance); «بلنگ شروع ہونے سے پہلے کوئی بھی ماہانہ قیمت آپ کو دکھائی جاتی ہے» (login_offer); «کبھی بلنگ ایکٹیویٹ نہیں کرتا اور نہ کریڈٹ خرچ کرتا» (auto_setup.help); «آپ کے اس انتخاب کو تبدیل نہیں کرے گا» (disabled). network/radient_unavailable ask nothing of the reader.

## Findings

| ID | severity | key | finding | suggested fix |
|---|---|---|---|---|
| UR-01 | minor | `mobile.setup.radient_unavailable` | «Radient نے ایک عارضی سرور خرابی کا جواب دیا۔» — «X کا جواب دینا» means 'reply TO X', so the literal reading is 'Radient responded to a temporary server error'. Context (next clause) recovers the intended 'answered WITH an error'; no action or safety meaning changes. | «Radient نے ایک عارضی سرور خرابی کے ساتھ جواب دیا۔» or «Radient نے جواب میں ایک عارضی سرور خرابی دی۔» (cf. audited tunnel.status.server_unavailable.relay «…کا جواب ایک عارضی سرور خرابی سے دیا»). |
| UR-05 | note | `mobile.setup.login_offer` | Bidi: the leading '/' of the LTR island «/mobile enable» sits between Arabic «:» and Latin 'mobile'; under UBA it resolves with the RTL run, so bidi-capable terminals / dir=rtl surfaces may show it at the far end of the Latin run. Per the ur guide this is a renderer concern (no manual controls in strings); not counted as a translation finding. 0 bidi controls present (audit.log). | None in the catalogue; renderer-level isolation of slash commands if it shows in rendered QA. |

## Translator open questions (this locale)

- billing-lapsed label: acceptable as «بلنگ رک گئی»; optional cause clarifier (UR-02).
- toggle label infinitive: grammatical, but inconsistent with the corpus polite-imperative toggle labels; recommend the imperative (UR-03).

## Mechanical

check.py --catalogues-only: 0 problems; key parity, placeholders, URLs, literal tokens, bidi controls, punctuation script and no-command checks OK (audit.log).

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame (RTL mirroring and the UR-05 slash placement are unverified in pixels). The Reconnect action label the copy points at lives outside this catalogue and was not compared. Judgement by reasoning only; no MT service used.

This bundle is written in the shared worktree, uncommitted; generate.py was not run (coordinator regenerates).
