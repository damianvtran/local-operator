# Urdu core — independent audit round 1

- Audited revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`
- Scope: 23 catalogues / 1,355 keys (including three empty namespaces).
- Verdict: **FAILED / changes requested** — 0 blockers, 5 unresolved majors, 5 minors, 0 nits.
- Core scalar `auditor.findings`: **5** (unresolved BLOCKER/MAJOR only).
- Rendered/native RTL and the required second independent high-visibility audit remain separate, unclaimed gates.

## UR-R1-001 — MAJOR — relay.ask

The sidecar explicitly requires the same dismiss verb as the terminal; ختم means “end/finish” and diverges from the terminal’s خارج for the same operation.

- `relay.ask.dismiss.open`
  - Current: ختم کریں…
  - Proposed: خارج کریں…
- `relay.ask.dismiss.title`
  - Current: یہ سوال ختم کریں
  - Proposed: یہ سوال خارج کریں
- `relay.ask.dismiss.receipt`
  - Current: ختم کر دیا گیا — کوئی جواب نہیں بھیجا
  - Proposed: خارج کیا گیا — کوئی جواب نہیں بھیجا گیا
- `relay.ask.dismiss.receipt_answered_in_chat`
  - Current: ختم کر دیا گیا — گفتگو میں جواب دے دیا گیا
  - Proposed: خارج کیا گیا — گفتگو میں جواب موجود ہے
- `relay.ask.dismiss.receipt_moot`
  - Current: ختم کر دیا گیا — اب ضرورت نہیں
  - Proposed: خارج کیا گیا — اب ضرورت نہیں

## UR-R1-002 — MAJOR — wire.settings

The visible label splices Urdu and English inside one word (ٹرانسکرپٹion) and leaves “cleanup” untranslated, producing garbled mixed-script copy.

- `wire.settings.speech_transcription_cleanup.label`
  - Current: ٹرانسکرپٹion cleanup
  - Proposed: ٹرانسکرپٹ کی صفائی

## UR-R1-003 — MAJOR — wire.settings

Settled Urdu terms from the wave provenance and other catalogues are replaced by English proactive, remote compute, run, turn and provider; these are prose concepts, not protected identifiers.

- `wire.settings.proactive.title`
  - Current: Proactive class
  - Proposed: پیش قدم قسم
- `wire.settings.remote_compute.title`
  - Current: Remote compute
  - Proposed: ریموٹ کمپیوٹ
- `wire.settings.remote_compute_enabled.label`
  - Current: Remote compute
  - Proposed: ریموٹ کمپیوٹ
- `wire.settings.monitor_runtimeoutms.label`
  - Current: Run timeout (ms)
  - Proposed: رن کی مہلت (ms)
- `wire.settings.runtime_background_on_resume.choice.false.description`
  - Current: سیشن چھوڑنے پر turn روک دیں
  - Proposed: سیشن چھوڑنے پر ٹرن روک دیں
- `wire.settings.runtime_background_on_resume.choice.true.description`
  - Current: Turn background میں چلتا رہے
  - Proposed: ٹرن پس منظر میں چلتا رہے
- `wire.settings.display_terminal_title.help`
  - Current: سیشن نام اور run state رکھنے والا OSC 0 window title۔
  - Proposed: سیشن نام اور رن کی حالت رکھنے والا OSC 0 window title۔
- `wire.settings.supplements_maxturns.label`
  - Current: ↳ زیادہ سے زیادہ turns
  - Proposed: ↳ زیادہ سے زیادہ ٹرن
- `wire.settings.hosting.label`
  - Current: Default provider
  - Proposed: ڈیفالٹ پرووائیڈر

Representative affected keys are named; remediation must sweep the same terms throughout wire.settings rather than fixing only these instances.

## UR-R1-004 — MAJOR — wire.settings

“Delta دیکھنے والے نگرانی” has gender/number disagreement and is a word-for-word calque; the proposed wording preserves all three monitor behaviours in natural Urdu.

- `wire.settings.monitor.description`
  - Current: Delta دیکھنے والے نگرانی: کتنی بار check کریں، کتنا رکھیں، اور کب ہار مانیں۔
  - Proposed: تبدیلیوں کی نگرانی: کتنی بار جانچ ہو، کتنا مواد رکھا جائے، اور کب کوشش ختم ہو۔

## UR-R1-005 — MAJOR — wire.slash

Common prose terms provider and manager are left in English here but rendered پرووائیڈر and مینیجر in tui.welcome, violating locale-wide glossary consistency.

- `wire.slash.login`
  - Current: provider میں لاگ اِن کریں
  - Proposed: پرووائیڈر میں لاگ اِن کریں
- `wire.slash.logout`
  - Current: محفوظ provider کریڈینشلز ہٹائیں
  - Proposed: محفوظ پرووائیڈر کریڈینشلز ہٹائیں
- `wire.slash.provider`
  - Current: providers اور ان کی لاگ اِن/استعمال کی حالت کی فہرست
  - Proposed: پرووائیڈرز اور ان کی لاگ اِن/استعمال کی حالت کی فہرست
- `wire.slash.usage`
  - Current: provider کا usage quota دکھائیں
  - Proposed: پرووائیڈر کا استعمالی کوٹہ دکھائیں
- `wire.slash.team`
  - Current: ٹیمیں اور تنظیمی خاکہ دیکھیں؛ manager کو درخواست بھیجیں
  - Proposed: ٹیمیں اور تنظیمی خاکہ دیکھیں؛ مینیجر کو درخواست بھیجیں

## UR-R1-006 — MINOR — wire.slash

آزمائیں means “test/try out”; دوبارہ کوشش کریں is the established imperative for retry.

- `wire.slash.highlights`
  - Current: جواب کے نمایاں حصے منسوخ، دوبارہ آزمائیں یا رخ بدلیں
  - Proposed: جواب کے نمایاں حصے منسوخ کریں، دوبارہ کوشش کریں یا رخ بدلیں

## UR-R1-007 — MINOR — wire.settings

Here child means a subagent and row is a UI row; the English generic nouns obscure both and drift from ذیلی ایجنٹ and قطار used elsewhere.

- `wire.settings.display_dock.choice.full.description`
  - Current: ہر child کی row، نیا پہلے
  - Proposed: ہر ذیلی ایجنٹ کی قطار، نیا پہلے

## UR-R1-008 — MINOR — relay.session.tags

tui.tags consistently treats ٹیگ as feminine (ٹیگ بن گئی، دوسری); masculine دوسرا/سکے here creates avoidable gender drift for the same term.

- `relay.session.tags.limit_hint`
  - Current: {count} ٹیگز ایک گفتگو کے لیے زیادہ سے زیادہ حد ہے — دوسرا شامل کرنے کے لیے ایک ہٹائیں۔
  - Proposed: {count} ٹیگز ایک گفتگو کے لیے زیادہ سے زیادہ حد ہے — دوسری شامل کرنے کے لیے ایک ہٹائیں۔
- `relay.session.tags.save_error`
  - Current: ٹیگز محفوظ نہیں ہو سکے
  - Proposed: ٹیگز محفوظ نہیں ہو سکیں

## UR-R1-009 — MINOR — tui.review

جائزہ {reviewed} reads as “review {hash}” rather than “reviewed at {hash}”; fronting the revision with پر restores the relation.

- `tui.review.lane_stale_head`
  - Current: پرانا — جائزہ {reviewed}، head {head}
  - Proposed: پرانا — {reviewed} پر جائزہ، head {head}
- `tui.review.lane_stale_reviewed`
  - Current: پرانا — جائزہ {reviewed}
  - Proposed: پرانا — {reviewed} پر جائزہ

## UR-R1-010 — MINOR — wire.settings

“منتخب ہونے کی حد” literally means “the limit of being selected”; the condition needs منطخب ہونے پر لاگو حد (“the limit that applies when selected”).

- `wire.settings.bash_memory_limit_mb.help`
  - Current: اوپر 'manual' منتخب ہونے کی حد۔ 0 کا مطلب auto حد استعمال کریں، اس لیے 0 رہنے سے guard بند نہیں ہوتا۔
  - Proposed: اوپر 'manual' منتخب ہونے پر لاگو حد۔ 0 کا مطلب auto حد استعمال کریں، اس لیے 0 رہنے سے guard بند نہیں ہوتا۔

## Namespace verdicts

- `cli.exec_remote_help` — **clean**; 11 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `relay.ask` — **failed**; 11 keys read; findings=1 blocker/major, 0 minor, 0 nit.
- `relay.session.tags` — **clean_with_followups**; 19 keys read; findings=0 blocker/major, 1 minor, 0 nit.
- `tui.asks` — **clean**; 11 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.authoring` — **clean**; 49 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.palette` — **clean**; 7 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.projects_edit` — **clean**; 35 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.review` — **clean_with_followups**; 73 keys read; findings=0 blocker/major, 1 minor, 0 nit.
- `tui.schedule` — **clean**; 61 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.status` — **clean**; 7 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.tags` — **clean**; 27 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.team_authoring` — **clean**; 36 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.transcript` — **clean**; 17 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.viewer` — **clean**; 19 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `tui.welcome` — **clean**; 30 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.errors` — **clean**; 0 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.errors.settings` — **clean**; 12 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.incidents` — **clean**; 0 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.memory` — **clean**; 17 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.notices` — **clean**; 0 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.remote_compute` — **clean**; 43 keys read; findings=0 blocker/major, 0 minor, 0 nit.
- `wire.settings` — **failed**; 818 keys read; findings=3 blocker/major, 2 minor, 0 nit.
- `wire.slash` — **failed**; 52 keys read; findings=1 blocker/major, 1 minor, 0 nit.
