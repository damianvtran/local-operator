# ur / mobile.doctor — independent translation audit a2 (PR #3953 round-1 remediation)

- Verdict: **passed** — blocker 0 · major 0 · minor 1 · nit 3 (unresolved blocker/major: 0)
- Scope: 23 keys (14 changed + 9 new) at 1505e6d2f8dc51f219ddeba76ca0cc344c73806a; 8 unchanged keys byte-identical to 52c0e1c93; `tunnel.problem` removed from catalogue, keys module, code and tests
- Source sha256 (en): `0428c5c217967add291b75b8a493b08083cffe5270a6f8600c0fd0a82d89afe9`; target `726975edf49179a3d2ffb4d0a22f8be40011a9d29569a4f4f3e14df0fd699db9`
- Translator: coder subagent of host session 25fe8dfbf610 (round-1 remediation of PR #3953, commit 1505e6d2f; wrote the English source and all 7 translations) — model: not reported in the commit; host session 25fe8dfbf610 runs anthropic/claude-opus-5-5, the coder child's served model is unverified
- Auditor: translation-auditor, lopdev team member (independent `task` subagent), session 11bf5c282289; did not translate or edit any catalogue value, did not author the remediation — anthropic/claude-opus-5-5 (served model of this session; no provider fallback observed)

## Findings

| id | severity | key | current | proposed | reason |
|---|---|---|---|---|---|
| UR-R1 | minor | `mobile.doctor.daemon.slow` | فون رسائی سروس نے جواب دینے میں {ms} ms لیے، جسے فون ایپ جواب نہ دینا دکھا سکتی ہے۔ اسے lop mobile restart سے ری اسٹارٹ کریں؛ اگر دوبارہ ایسا ہو تو یہ رپورٹ سپورٹ کو بھیجیں۔ | فون رسائی سروس نے جواب دینے میں {ms} ملی سیکنڈ لیے، جسے فون ایپ جواب نہ دینا دکھا سکتی ہے۔ اسے lop mobile restart سے ری اسٹارٹ کریں؛ اگر دوبارہ ایسا ہو تو یہ رپورٹ سپورٹ کو بھیجیں۔ | Unit drift inside the namespace: daemon.ok renders “{ms} ملی سیکنڈ”, this key the Latin “ms”. |
| UR-R2 | nit | `mobile.doctor.tunnel.list_unchecked` | اکاؤنٹ کی ٹنل فہرست پڑھی نہیں جا سکی، اس لیے ڈپلیکیٹ ٹنلز کی جانچ نہیں ہوئی۔ اپنا انٹرنیٹ کنکشن چیک کریں اور جانچ دوبارہ چلائیں۔ | اکاؤنٹ کی ٹنل فہرست پڑھی نہیں جا سکی، اس لیے ڈپلیکیٹ ٹنل کی جانچ نہیں ہوئی۔ اپنا انٹرنیٹ کنکشن چیک کریں اور جانچ دوبارہ چلائیں۔ | Plural drift: “ٹنلز” here vs the invariant “ٹنل” in tunnel.duplicates (“{count} ٹنل”, “پرانے ٹنل”) and mobile.setup. |
| UR-R2 | nit | `mobile.doctor.tunnel.list_skipped` | ڈپلیکیٹ ٹنلز کی جانچ نہیں ہوئی کیونکہ اس کمپیوٹر پر Radient سائن اِن موجود نہیں یا اسے تازہ کرنے کی ضرورت ہے۔ Local Operator کھولیں، کہا جائے تو سائن اِن کریں، پھر جانچ دوبارہ چلائیں۔ | ڈپلیکیٹ ٹنل کی جانچ نہیں ہوئی کیونکہ اس کمپیوٹر پر Radient سائن اِن موجود نہیں یا اسے تازہ کرنے کی ضرورت ہے۔ Local Operator کھولیں، کہا جائے تو سائن اِن کریں، پھر جانچ دوبارہ چلائیں۔ | Plural drift: “ٹنلز” here vs the invariant “ٹنل” in tunnel.duplicates (“{count} ٹنل”, “پرانے ٹنل”) and mobile.setup. |
| UR-R3 | nit | `mobile.doctor.daemon.state_not_running` | نہیں چل رہا | نہیں چل رہی | Pre-existing (unchanged, not gating): gender drift with the new feminine sibling “جواب نہیں دے رہی”. |

## Method and limits

LLM audit, same model family as the coder (disclosed); no native-speaker pass and no rendered terminal/desktop frame claimed.
