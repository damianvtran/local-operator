# Second independent core Urdu audit — round 1

**Verdict: CHANGES_REQUESTED** — 0 BLOCKER, 7 MAJOR, 4 MINOR, 0 NIT.

Commit `2189b61739e08b5f219d317ead68b4a5f4cf709f`; 23 namespaces; **1355/1355 keys read**. Core scalar `auditor.findings` = **7** (unresolved BLOCKER/MAJOR only).

## Provenance and independence

- Auditor: translation-auditor (lopdev team); independent second core auditor
- Session: `3e0f499dc2ee`; served model: `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent of translator sessions: `3f68f20b2cad`, `52136848d1fc`, `7a648afb0f8d`, `7c4a4229b011`, `ffc94c8cb17d`
- No other new second-audit receipt was read. English, Urdu, context sidecars, glossary, and current core contract were read directly.
- SHA-256 was independently derived for every current en/ur/context file; all translator-provenance hashes match those bytes.

## Verification evidence

- `generate.py --check`: **91 artifacts byte-identical**.
- `check.py --catalogues-only`: **0 problems**.
- Independent key/ICU/literal-token checks: **0 mismatches**. One literal Urdu redaction artifact was found separately and is UR2-009.
- Plurals: **19 keys / 95 real renders**, 0 errors, 0 CLDR mismatches, 0 missing counts.
- Fit: **54 explicit budgets**, 0 actual rendered overflows, 0 worst-case-key overflows; 30% pseudo stress exceeds 26 budgets and remains fit-risk evidence for the separate rendered gate.
- Typography: 0 manual bidi controls, 0 Urdu-digit strings, 0 Hindi dandas, 0 تم address. The only `!` is the literal bang shell operator in `tui.welcome.tip.shell`.
- 26 materially longer candidates: one fits its declared budget; 25 have no declared `maxCells` and are listed for rendered review rather than called defects.

## Namespace verdicts

| Namespace | Keys read | Verdict | Findings |
|---|---:|---|---|
| `cli.exec_remote_help` | 11 | clean_with_followups | UR2-008 |
| `relay.ask` | 11 | changes_requested | UR2-006 |
| `relay.session.tags` | 19 | clean | — |
| `tui.asks` | 11 | changes_requested | UR2-005, UR2-006 |
| `tui.authoring` | 49 | clean | — |
| `tui.palette` | 7 | clean | — |
| `tui.projects_edit` | 35 | clean | — |
| `tui.review` | 73 | clean | — |
| `tui.schedule` | 61 | clean | — |
| `tui.status` | 7 | clean | — |
| `tui.tags` | 27 | clean_with_followups | UR2-010 |
| `tui.team_authoring` | 36 | clean | — |
| `tui.transcript` | 17 | clean | — |
| `tui.viewer` | 19 | clean_with_followups | UR2-011 |
| `tui.welcome` | 30 | changes_requested | UR2-004 |
| `wire.errors` | 0 | clean | — |
| `wire.errors.settings` | 12 | changes_requested | UR2-009 |
| `wire.incidents` | 0 | clean | — |
| `wire.memory` | 17 | clean | — |
| `wire.notices` | 0 | clean | — |
| `wire.remote_compute` | 43 | changes_requested | UR2-007 |
| `wire.settings` | 818 | changes_requested | UR2-001 |
| `wire.slash` | 52 | changes_requested | UR2-002, UR2-003 |

## Findings

### UR2-001 — MAJOR — wire.settings

Categories: English leakage, native idiom, glossary consistency.

Systematic across the 818-key namespace; the examples are representative, not a complete remediation list. Retranslate all ordinary English prose while preserving permitted machine tokens.

- `wire.settings.classification.title`
  - Current: Smart hints
  - Proposed: ذہین اشارے
- `wire.settings.bash_query_budget_enabled.label`
  - Current: Shell query budget
  - Proposed: شیل کوئری کا بجٹ
- `wire.settings.desktop.title`
  - Current: Desktop app
  - Proposed: ڈیسک ٹاپ ایپ
- `wire.settings.display_heading_markers.label`
  - Current: Heading markers
  - Proposed: سرخی کے نشانات
- `wire.settings.web_search_providers.label`
  - Current: Providers
  - Proposed: پرووائیڈرز
- `wire.settings.speech_transcription_cleanup.label`
  - Current: ٹرانسکرپٹion cleanup
  - Proposed: ٹرانسکرپشن صفائی

Reason: Whole user-visible labels and many ordinary help clauses remain English or hybrid rather than native Urdu, far beyond the style guide's narrow permission for product names, identifiers, commands, flags, paths, tool names, and acronyms.

### UR2-002 — MAJOR — wire.slash

Categories: English leakage, glossary consistency, native idiom.

Systematic Latin prose and drift from the wave's own پرووائیڈر decision across the 52 slash descriptions; inspect every value during remediation.

- `wire.slash.agent`
  - Current: ایجنٹس دیکھیں، class بدلیں، یا ایجنٹ بن کر بات کریں
  - Proposed: ایجنٹس دیکھیں، کسی ایجنٹ کی قسم بدلیں، یا اس سیشن سے بطور ایجنٹ بات کریں
- `wire.slash.provider`
  - Current: providers اور ان کی لاگ اِن/استعمال کی حالت کی فہرست
  - Proposed: پرووائیڈرز اور ان کے لاگ اِن/استعمال کی حالت کی فہرست
- `wire.slash.move`
  - Current: اس سیشن کی working directory بدلیں
  - Proposed: اس سیشن کی ورکنگ ڈائریکٹری بدلیں
- `wire.slash.usage`
  - Current: provider کا usage quota دکھائیں
  - Proposed: پرووائیڈر کے استعمال کا کوٹہ دکھائیں

Reason: High-visibility slash help repeatedly leaves ordinary nouns in Latin and alternates provider/providers with the translator-endorsed پرووائیڈر, producing copy that reads code-switched rather than written in Urdu.

### UR2-003 — MINOR — wire.slash

Categories: accuracy.

- `wire.slash.fork`
  - Current: شاخ؛ --switch یہیں، --window باہر؛ <message> آغاز کرے
  - Proposed: اس گفتگو کی شاخ بنائیں؛ --switch یہیں، --window کہیں اور؛ <message> سے کام شروع کریں

Reason: The terse Urdu drops both 'this chat' and the explicit 'starts work' relationship, making the command description less precise than the English.

### UR2-004 — MAJOR — tui.welcome

Categories: English leakage, high-visibility copy.

- `tui.welcome.tip.fork_placement`
  - Current: /settings → Fork → “فورک کہاں کھلے” سے مقام طے ہوتا ہے
  - Proposed: /settings → فورک → “فورک کہاں کھلے” سے مقام طے ہوتا ہے
- `tui.welcome.tip.hotkeys`
  - Current: /settings → Hotkeys سے یہ کلیدیں بدلیں
  - Proposed: /settings → ہاٹ کیز سے یہ کلیدیں بدلیں

Reason: Two opening-screen navigation tips leave the visible settings section names in English even though the surrounding path and the corresponding fork wording are Urdu.

### UR2-005 — MAJOR — tui.asks

Categories: accuracy, safety meaning.

- `tui.asks.dismiss.answered_in_chat`
  - Current: گفتگو میں جواب دیدہ
  - Proposed: گفتگو میں جواب دے دیا

Reason: دیدہ means 'seen' and does not express the operator's deliberate 'answered in chat' reason, so the dismiss chooser names the wrong action.

### UR2-006 — MAJOR — relay.ask, tui.asks

Categories: glossary consistency, explicit context contract.

- `relay.ask.dismiss.open`
  - Current: ختم کریں…
  - Proposed: خارج کریں…
- `relay.ask.dismiss.title`
  - Current: یہ سوال ختم کریں
  - Proposed: یہ سوال خارج کریں
- `relay.ask.dismiss.receipt`
  - Current: ختم کر دیا گیا — کوئی جواب نہیں بھیجا
  - Proposed: خارج کر دیا گیا — کوئی جواب نہیں بھیجا
- `relay.ask.dismiss.receipt_answered_in_chat`
  - Current: ختم کر دیا گیا — گفتگو میں جواب دے دیا گیا
  - Proposed: خارج کر دیا گیا — گفتگو میں جواب دے دیا گیا
- `relay.ask.dismiss.receipt_moot`
  - Current: ختم کر دیا گیا — اب ضرورت نہیں
  - Proposed: خارج کر دیا گیا — اب ضرورت نہیں

Reason: The phone uses ختم while the terminal uses خارج for the same dismiss act, directly violating the phone sidecar's instruction to use the terminal's verb and creating cross-surface terminology drift.

### UR2-007 — MAJOR — wire.remote_compute

Categories: English leakage, native idiom, glossary consistency.

Systematic Latin technical prose across the namespace; preserve commands, flags, URLs, identifiers, product names, and Git operations while translating ordinary labels and prose.

- `wire.remote_compute.finished_exit_code`
  - Current: exit code: {exit_code}
  - Proposed: خروجی کوڈ: {exit_code}
- `wire.remote_compute.detached`
  - Current: ریموٹ رن {run_id} جمع کر دیا گیا۔
network (Radient کی تصدیق شدہ): {network_id}
Idempotency key: {idempotency_key}
اس کی نگرانی: lop exec --remote-status {run_id}
اسے منسوخ کریں: lop exec --remote-cancel {run_id}
  - Proposed: ریموٹ رن {run_id} جمع کر دیا گیا۔
نیٹ ورک (Radient کی تصدیق شدہ): {network_id}
Idempotency key: {idempotency_key}
اس کی نگرانی: lop exec --remote-status {run_id}
اسے منسوخ کریں: lop exec --remote-cancel {run_id}
- `wire.remote_compute.key_not_printable`
  - Current: idempotency key صرف printable ASCII ہونی چاہیے: حروف، اعداد اور punctuation marks؛ accents، tabs یا newlines نہیں۔
  - Proposed: idempotency key صرف قابلِ طباعت ASCII حروف پر مشتمل ہونی چاہیے: حروف، اعداد اور اوقاف؛ ایکسنٹ والے حروف، ٹیبز یا نئی سطریں شامل نہیں۔

Reason: A complete output label and substantial ordinary prose remain in English, while نیٹ ورک is already the established rendering elsewhere in the same namespace.

### UR2-008 — MINOR — cli.exec_remote_help

Categories: native idiom, script policy, glossary consistency.

- `cli.exec_remote_help.idempotency_key`
  - Current: نئی key بنانے کے بجائے استعمال کرنے کے لیے idempotency key، تاکہ جمع شدہ رن کو retry کرنے پر وہ دوبارہ نہ چلے
  - Proposed: نئی کلید بنانے کے بجائے موجودہ idempotency key استعمال کریں، تاکہ جمع شدہ رن کی دوبارہ کوشش پر وہ دوبارہ نہ چلے
- `cli.exec_remote_help.repo`
  - Current: Git URL جسے ریموٹ رن checkout کرے گا۔ --remote کے ساتھ درکار: مقامی فائلیں کبھی upload نہیں ہوتیں
  - Proposed: Git URL جسے ریموٹ رن checkout کرے گا۔ --remote کے ساتھ درکار: مقامی فائلیں کبھی اپلوڈ نہیں ہوتیں
- `cli.exec_remote_help.size`
  - Current: ریموٹ رن کا compute size: s، m یا l (ڈیفالٹ: m)
  - Proposed: ریموٹ رن کا کمپیوٹ سائز: s، m یا l (ڈیفالٹ: m)

Reason: Several settled loanwords and ordinary operations remain Latin even though the same catalogue uses کلید and Urdu-script technical loanwords elsewhere.

### UR2-009 — MAJOR — wire.errors.settings

Categories: redaction artifact, accuracy, native idiom, English leakage.

- `wire.errors.settings.credential_key_empty`
  - Current: کریڈینشل key خالی نہیں ہو سکتی
  - Proposed: کریڈینشل کی کلید خالی نہیں ہو سکتی
- `wire.errors.settings.inline_credentials`
  - Current: ایسا [redacted] استعمال کریں جس میں inline credentials یا query parameters نہ ہوں۔
  - Proposed: ایسا اینڈ پوائنٹ استعمال کریں جس میں ان لائن کریڈینشلز یا کوئری پیرامیٹرز نہ ہوں۔
- `wire.errors.settings.not_registered`
  - Current: یہ ترتیب رجسٹر نہیں ہے۔
  - Proposed: یہ ترتیب رجسٹرڈ نہیں ہے۔

Reason: The Urdu error replaces the English endpoint noun with a literal [redacted] artifact and also leaves inline credentials/query parameters in English, corrupting a security-relevant instruction rather than localizing it.

### UR2-010 — MINOR — tui.tags

Categories: register, grammatical consistency.

- `tui.tags.added`
  - Current: “{name}” شامل کی گئی۔
  - Proposed: “{name}” شامل کیا گیا۔
- `tui.tags.create_help`
  - Current: مشترکہ ٹیگ تیار کر کے یہاں لاگو کی جاتی ہے۔
  - Proposed: مشترکہ ٹیگ تیار کر کے یہاں لاگو کیا جاتا ہے۔
- `tui.tags.created_applied`
  - Current: “{name}” تیار کر کے لاگو کی گئی۔
  - Proposed: “{name}” تیار کر کے لاگو کیا گیا۔
- `tui.tags.partial_create`
  - Current: ٹیگ بن گئی، مگر لاگو نہیں ہو سکی۔ دوبارہ لاگو کرنے کی کوشش کریں۔
  - Proposed: ٹیگ بن گیا، مگر لاگو نہیں ہو سکا۔ دوبارہ لاگو کرنے کی کوشش کریں۔
- `tui.tags.cap_applied`
  - Current: 8 ٹیگز لاگو ہیں۔ دوسری شامل کرنے سے پہلے ایک ہٹائیں۔
  - Proposed: 8 ٹیگز لاگو ہیں۔ دوسرا شامل کرنے سے پہلے ایک ہٹائیں۔

Reason: The terminal treats ٹیگ as feminine here but the relay and neighbouring plural constructions treat it as masculine, so the same noun changes gender across surfaces.

### UR2-011 — MINOR — tui.viewer

Categories: typography, native idiom.

- `tui.viewer.unreadable`
  - Current: یہ فائل پڑھی نہیں جا سکتی — موجود نہیں، ناقابلِ قرأت ہے، یا اجازت شدہ روٹ ڈائریکٹریوں سے باہر ہے
  - Proposed: یہ فائل موجود نہیں، پڑھنے کے قابل نہیں، یا اجازت شدہ روٹ ڈائریکٹریوں سے باہر ہے

Reason: ناقابلِ قرأت is misspelled and awkward; the proposed wording preserves all three refusal causes in idiomatic Urdu.

## Limits and remaining gates

No native/rendered RTL, bidi/shaping, fallback, long-text, or design sign-off is claimed. Those remain a separate gate. No source, catalogue, ledger, generated `shipped.json`, hold, or PR was edited.

## Evidence bundle

The portable sibling files include the complete `.pairs` review workbook, exact findings, hashes, mechanical checks, plural renders, fit/pseudo results, length candidates, this report, and the JSON receipt. `evidence-manifest.json` records their SHA-256 hashes.
