# Independent translation audit — ur / wire.phone_access / a1

Commit: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (branch fix/tunnel-remedies-auto-up). Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4aaeb39ddb2f; neither the i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4aaeb39ddb2f: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (commit 1fce7ecb1f5f); English source drafted by coder session b0300e2c2c7f; model anthropic/claude-opus-5-5 (as stated by the delegating brief; not independently verified); 2026-11-14.
Prior audit: none (new namespace).

**Verdict: passed** — 0 unresolved blocker/major; 2 minor/nit finding(s).

Scope: the 4 keys new in 1fce7ecb1f5f judged in full (meaning, safety, placeholders, glossary, tone, length); the receipt covers the whole namespace (4 keys) at the new English source hash. New namespace: all keys are new.

Read: docs/I18N.md L260-308, i18n/glossary.md (no tunnel/billing/credit row — corpus usage governs), i18n/style-guides/ur.md, en + context sidecar, every new value in full, corpus term counts across all ur catalogues.

## Keys judged

| key | EN | ur | result |
|---|---|---|---|
| `title` | Phone access | فون رسائی | OK |
| `description` | How this computer keeps its Radient tunnel for the phone app working. Changes apply to the next sign-in. | یہ کمپیوٹر فون ایپ کے لیے اپنے Radient ٹنل کو کیسے فعال رکھتا ہے۔ تبدیلیاں اگلے سائن اِن پر لاگو ہوتی ہیں۔ | OK |
| `auto_setup.label` | Reconnect phone access after signing in | سائن اِن کے بعد فون رسائی دوبارہ منسلک کرنا | UR-03 |
| `auto_setup.help` | When you sign in to Radient on a computer that already has phone access, reconnect or repair its tunnel in the background. It never activates billing or spends credit; any price or missing credit is shown to you instead. | جب آپ کسی ایسے کمپیوٹر پر Radient میں سائن اِن کرتے ہیں جس پر پہلے سے فون رسائی موجود ہے، تو اس کا ٹنل بیک گراؤنڈ میں دوبارہ منسلک یا ٹھیک کیا جاتا ہے۔ یہ کبھی بلنگ ایکٹیویٹ نہیں کرتا اور نہ کریڈٹ خرچ کرتا ہے؛ اس کے بجائے کوئی بھی قیمت یا کریڈٹ کی کمی آپ کو دکھائی جاتی ہے۔ | UR-04 |

## Judgement

Polite آپ register; Urdu punctuation ۔ ، ؛ only (no । or Latin ;/, in prose); no manual bidi controls; LTR islands (Radient, Local Operator, {url}, console URL, /mobile enable) byte-identical. Terms consistent with audited corpus: فون رسائی (26 prior uses), ٹنل, بلنگ, کریڈٹ, کنسول, ماہانہ قیمت, سیٹ اپ, بیک گراؤنڈ, سائن اِن, ایکٹیویٹ, 'reconnect' as دوبارہ منسلک (= audited mobile.setup.repoint). Gender agreement checked: رسائی f. («منسلک ہو رہی ہے», «ہو جاتی ہے»), ٹنل m. («رکا ہی رہے», «کرتا رہتا ہے»). Safety meanings verified per key: no terminal command (only `/mobile enable` in login_offer); «کوئی رقم نہیں لی جاتی» (reconnect, reconnecting); «سیٹ اپ کبھی کریڈٹ نہیں خریدتا» (billing_lapsed, identical to audited mobile.setup.balance); «بلنگ شروع ہونے سے پہلے کوئی بھی ماہانہ قیمت آپ کو دکھائی جاتی ہے» (login_offer); «کبھی بلنگ ایکٹیویٹ نہیں کرتا اور نہ کریڈٹ خرچ کرتا» (auto_setup.help); «آپ کے اس انتخاب کو تبدیل نہیں کرے گا» (disabled). network/radient_unavailable ask nothing of the reader.

## Findings

| ID | severity | key | finding | suggested fix |
|---|---|---|---|---|
| UR-03 | minor | `wire.phone_access.auto_setup.label` | Infinitive «…دوبارہ منسلک کرنا» is grammatical but breaks the corpus toggle-label convention (wire.settings labels: «گفتگو خودکار محفوظ کریں», «/resume کے بعد کام جاری رکھیں»). Answers the translator's open question. | «سائن اِن کے بعد فون رسائی دوبارہ منسلک کریں». |
| UR-04 | nit | `wire.phone_access.auto_setup.help` | «یہ کبھی بلنگ ایکٹیویٹ نہیں کرتا…» follows a passive sentence about the tunnel; antecedent of «یہ» ambiguous. Safety meaning survives. | «یہ عمل کبھی بلنگ ایکٹیویٹ نہیں کرتا اور نہ کریڈٹ خرچ کرتا ہے…» |

## Translator open questions (this locale)

- billing-lapsed label: acceptable as «بلنگ رک گئی»; optional cause clarifier (UR-02).
- toggle label infinitive: grammatical, but inconsistent with the corpus polite-imperative toggle labels; recommend the imperative (UR-03).

## Mechanical

check.py --catalogues-only: 0 problems; key parity, placeholders, URLs, literal tokens, bidi controls, punctuation script and no-command checks OK (audit.log).

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame (RTL mirroring and the UR-05 slash placement are unverified in pixels). The Reconnect action label the copy points at lives outside this catalogue and was not compared. Judgement by reasoning only; no MT service used.

This bundle is written in the shared worktree, uncommitted; generate.py was not run (coordinator regenerates).
