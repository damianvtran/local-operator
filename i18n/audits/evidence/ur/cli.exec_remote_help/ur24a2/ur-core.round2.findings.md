# Urdu core — independent audit round 2

- Remediation: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`
- Delta: `2189b61739e08b5f219d317ead68b4a5f4cf709f..d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`
- Scope: 657 changed keys plus all 19 unchanged `relay.session.tags` neighbor keys; unrelated unchanged namespaces were not re-audited.
- Verdict: **FAILED / changes requested** — 0 blockers, 2 unresolved majors, 5 minors, 0 nits.
- Core scalar `auditor.findings`: **2** (unresolved BLOCKER/MAJOR only).
- All ten round-1 findings are resolved. `UR-R1-008` resolved through masculine ٹیگ standardization in `tui.tags`, making the unchanged relay neighbor consistent.
- Native rendered RTL remains a separate, unclaimed gate; four approximate /help risks are reserved for rendered review.

## UR-R2-001 — MAJOR — wire.settings

مکمل means “completed”, but settled explicitly includes done, paused and archived; the current wording can wrongly imply that paused or archived projects are still watched.

- `wire.settings.wakes_triggers_project_staleness_enabled.help`
  - Current: ان پروجیکٹس پر رابطے کے لیے بیدار ہوں جن کی پیش رفت پرانی ہو۔ مکمل (done/paused/archived) پروجیکٹس کی نگرانی کبھی نہیں۔
  - Proposed: ان پروجیکٹس پر رابطے کے لیے بیدار ہوں جن کی پیش رفت پرانی ہو۔ حتمی حالت والے (done/paused/archived) پروجیکٹس کی نگرانی کبھی نہیں۔

## UR-R2-002 — MAJOR — wire.settings

The coordinated negative is broken: فائلیں ... نہیں بھیجتیں makes the files the actor and does not clearly guarantee that they are neither listed nor sent to the decision model.

- `wire.settings.supplements_denyprefixes.help`
  - Current: نمایاں حصے آن ہوں۔ کاما سے الگ فولڈرز جن کی فائلیں کبھی فہرست میں شامل یا فیصلہ ماڈل کو نہیں بھیجتیں؛ گاہکوں کے ڈیٹا والی مشین پر استعمال کریں۔
  - Proposed: نمایاں حصے آن ہوں۔ کاما سے الگ فولڈرز جن کی فائلیں کبھی فہرست میں نہیں آتیں اور فیصلہ کرنے والے ماڈل کو نہیں بھیجی جاتیں؛ گاہکوں کے ڈیٹا والی مشین پر استعمال کریں۔

## UR-R2-003 — MINOR — wire.settings

The operator concept is translated آپریٹر on its choice label but left as English operator in help, and job is rendered کام while the settled background-job term is جاب.

- `wire.settings.subagents_model_choice.help`
  - Current: ذیلی ایجنٹ کا درجہ کون منتخب کرے؛ operator پروفائل میں مقرر کرتا ہے۔
  - Proposed: ذیلی ایجنٹ کا درجہ کون منتخب کرے؛ آپریٹر پروفائل میں مقرر کرتا ہے۔
- `wire.settings.subagents_max_running.label`
  - Current: زیادہ سے زیادہ پس منظر کام
  - Proposed: زیادہ سے زیادہ پس منظر جاب
- `wire.settings.supplements_maxcostusd.help`
  - Current: گرافکس آن ہوں۔ ایک کام کے خرچ کی نرم حد۔
  - Proposed: گرافکس آن ہوں۔ ایک جاب کے خرچ کی نرم حد۔
- `wire.settings.supplements_timeouts.help`
  - Current: نمایاں حصے آن ہوں۔ اتنی دیر بعد پورا کام چھوڑ دیں۔
  - Proposed: نمایاں حصے آن ہوں۔ اتنی دیر بعد پوری جاب چھوڑ دیں۔

## UR-R2-004 — MINOR — wire.slash

رکنیت means membership, not a per-link subscription policy; سبسکرپشن preserves the actual setting.

- `wire.slash.hub`
  - Current: Hub روابط کھولیں — فی ربط رکنیت اور خودکار push پالیسی
  - Proposed: Hub روابط کھولیں — فی ربط سبسکرپشن اور خودکار push پالیسی

## UR-R2-005 — MINOR — wire.settings

پیش نظارہ describes a visual preview elsewhere, whereas Preview marks feature availability here; پری ویو avoids that false friend.

- `wire.settings.remote_compute.description`
  - Current: پیش نظارہ: ایجنٹ کا کام Radient کے ہوسٹ کیے کمپیوٹ پر چلائیں، آپ کے ایک میش نیٹ ورک سے منسلک۔
  - Proposed: پری ویو: ایجنٹ کا کام Radient کے ہوسٹ کیے کمپیوٹ پر چلائیں، آپ کے ایک میش نیٹ ورک سے منسلک۔
- `wire.settings.remote_compute_enabled.help`
  - Current: پیش نظارہ: کام Radient کمپیوٹ کو بھیجے، ایک میش نیٹ ورک سے منسلک۔
  - Proposed: پری ویو: کام Radient کمپیوٹ کو بھیجے، ایک میش نیٹ ورک سے منسلک۔

## UR-R2-006 — MINOR — wire.remote_compute

خروجی کوڈ can read as “output code”; ایگزٹ کوڈ is the unambiguous technical term for the process status.

- `wire.remote_compute.finished_exit_code`
  - Current: خروجی کوڈ: {exit_code}
  - Proposed: ایگزٹ کوڈ: {exit_code}

## UR-R2-007 — MINOR — wire.settings

The source states what the rule does, while کرے turns it into a command or possibility; کرتی ہے restores the factual description.

- `wire.settings.display_rail.help`
  - Current: ایک لکیر ANSWER کو نشان زد کرے: ٹرن ختم کرنے والا پیغام۔ ٹرن کے دوران کے بیان پر ریل نہیں۔
  - Proposed: ایک لکیر ANSWER کو نشان زد کرتی ہے: ٹرن ختم کرنے والا پیغام۔ ٹرن کے دوران کے بیان پر ریل نہیں۔

## Namespace verdicts

- `cli.exec_remote_help` — **clean**; 3 scoped keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `relay.ask` — **clean**; 5 scoped keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.asks` — **clean**; 1 scoped keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.review` — **clean**; 2 scoped keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.tags` — **clean**; 6 scoped keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.viewer` — **clean**; 1 scoped keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.welcome` — **clean**; 2 scoped keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.errors.settings` — **clean**; 4 scoped keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.remote_compute` — **clean_with_followups**; 21 scoped keys read; findings=0 blocker/major, 1 minor, 0 nit.
- `wire.settings` — **failed**; 589 scoped keys read; findings=2 blocker/major, 3 minor, 0 nit.
- `wire.slash` — **clean_with_followups**; 23 scoped keys read; findings=0 blocker/major, 1 minor, 0 nit.
- `relay.session.tags` — **clean**; 19 scoped keys read; findings=0 blocker/major, 0 minor, 0 nit.
