# Scoped independent core Urdu audit — remediation round 2

**Verdict: CHANGES_REQUESTED** — 0 BLOCKER, 1 MAJOR, 5 MINOR, 0 NIT.

Delta `2189b61739e08b5f219d317ead68b4a5f4cf709f..d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`; **657 changed keys + 19 unchanged direct-neighbor keys = 676 scoped values read** across 12 namespaces. Core `auditor.findings` = **1** (unresolved BLOCKER/MAJOR only).

## Provenance and scope

- Auditor: translation-auditor (lopdev team); independent scoped round-2 core auditor
- Session: `3e0f499dc2ee`; served model: `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent of every translator session; no other new audit receipt was read.
- Current en/ur/context SHA-256 values were independently derived for every scoped namespace; translator provenance matches all of them.
- English and context files are unchanged in this delta. Unrelated unchanged namespaces were not re-audited.

## Round-1 dispositions

**21 entries:** 19 verified fixed, 1 verified superseded, 1 unresolved/reopened (`UR2-009` → `UR-R2-001`).

## Verification evidence

- `generate.py --check`: **91 artifacts byte-identical**.
- `check.py --catalogues-only`: **0 problems**.
- Scoped key/ICU/technical-token/newline checks: **0 mismatches**.
- NFC: 0 failures. Manual bidi controls: 0. Native digits: 0. Hindi dandas: 0. Informal تم: 0.
- Scoped plural: **1 key / 5 real renders**, 0 errors, CLDR mismatches, or missing counts.
- Scoped fit: **8 explicit budgets**, 0 actual overflows; 4 pseudo-stretch risks retained for rendered review.
- Approximate `/help`: **3 risks** — `wire.slash.agent` 72/~55, `wire.slash.fork` 84/~55, `wire.slash.highlights` 59/~55. These are not silent passes or measured rendered failures.
- One literal redaction artifact remains in the scoped bytes and is UR-R2-001.

## Namespace verdicts

| Namespace | Scoped keys read | Verdict | Findings / rendered follow-ups |
|---|---:|---|---|
| `cli.exec_remote_help` | 3 | clean | — |
| `relay.ask` | 5 | clean_with_followups | pseudo_stretch_rendered_fit |
| `tui.asks` | 1 | clean | — |
| `tui.review` | 2 | clean | — |
| `tui.tags` | 6 | clean | — |
| `tui.viewer` | 1 | clean | — |
| `tui.welcome` | 2 | clean_with_followups | pseudo_stretch_rendered_fit |
| `wire.errors.settings` | 4 | changes_requested | UR-R2-001 |
| `wire.remote_compute` | 21 | clean | — |
| `wire.settings` | 589 | clean_with_followups | UR-R2-002, UR-R2-003, UR-R2-004, UR-R2-005, UR-R2-006 |
| `wire.slash` | 23 | clean_with_followups | approximate_help_rendered_fit |
| `relay.session.tags` | 19 | clean | — |

## Findings

### UR-R2-001 — MAJOR — wire.errors.settings

Categories: redaction artifact, accuracy, security-relevant copy.

Reopens: UR2-009.

- `wire.errors.settings.inline_credentials`
  - Current: ایسا اینڈ پوائنٹ استعمال کریں جس میں ان لائن [redacted]ز یا کوئری پیرامیٹرز نہ ہوں۔
  - Proposed: ایسا اینڈ پوائنٹ استعمال کریں جس میں ان لائن کریڈینشلز یا کوئری پیرامیٹرز نہ ہوں۔

Reason: The first safety predicate and query wording are fixed, but the credential noun is still stored as the 10-code-point literal `[redacted]` plus Urdu suffix, so UR2-009 is only partially remediated and the safety instruction still contains a redaction artifact.

### UR-R2-002 — MINOR — wire.settings

Categories: English leakage, glossary consistency.

- `wire.settings.compaction_wire_bytes_trigger.label`
  - Current: درخواست محرک (bytes)
  - Proposed: درخواست محرک (بائٹس)
- `wire.settings.network_audit_max_bytes.help`
  - Current: گردش کے بعد فی جنیریشن bytes۔ 8 MiB gzip شدہ JSONL میں تقریباً 55k ریکارڈز ہوتے ہیں؛ پانچ جنیریشنز کی حد اس لیے ~40 MiB ہے۔
  - Proposed: گردش کے بعد فی جنیریشن بائٹس۔ 8 MiB gzip شدہ JSONL میں تقریباً 55k ریکارڈز ہوتے ہیں؛ پانچ جنیریشنز کی حد اس لیے ~40 MiB ہے۔
- `wire.settings.session_cleanup_max_total_bytes.help`
  - Current: صفائی آن ہو۔ اتنے bytes سے بڑھنے پر پرانے ہٹائیں؛ 0 = کوئی حد نہیں۔
  - Proposed: صفائی آن ہو۔ اتنے بائٹس سے بڑھنے پر پرانے ہٹائیں؛ 0 = کوئی حد نہیں۔
- `wire.settings.web_fetch_max_bytes.label`
  - Current: ڈاؤن لوڈ حد (bytes)
  - Proposed: ڈاؤن لوڈ حد (بائٹس)
- `wire.settings.tui_theme.help`
  - Current: رنگوں کا تسلسل۔ /theme arrow-key پیش نظارہ کے ساتھ براہ راست بدلتی ہے۔
  - Proposed: رنگوں کا تسلسل۔ /theme ایرو کیز کے پیش نظارہ کے ساتھ براہ راست بدلتی ہے۔

Reason: These changed values leave ordinary `bytes` and `arrow-key` prose in Latin even though the remediation uses بائٹس and ایرو کیز for the same concepts elsewhere.

### UR-R2-003 — MINOR — wire.settings

Categories: accuracy, false friend.

- `wire.settings.remote_compute.description`
  - Current: پیش نظارہ: ایجنٹ کا کام Radient کے ہوسٹ کیے کمپیوٹ پر چلائیں، آپ کے ایک میش نیٹ ورک سے منسلک۔
  - Proposed: تجرباتی: ایجنٹ کا کام Radient کے ہوسٹ کیے کمپیوٹ پر چلائیں، آپ کے ایک میش نیٹ ورک سے منسلک۔
- `wire.settings.remote_compute_enabled.help`
  - Current: پیش نظارہ: کام Radient کمپیوٹ کو بھیجے، ایک میش نیٹ ورک سے منسلک۔
  - Proposed: تجرباتی: کام Radient کمپیوٹ کو بھیجے، ایک میش نیٹ ورک سے منسلک۔

Reason: Here Preview describes a pre-release feature, not a visual preview, so پیش نظارہ is the wrong sense; تجرباتی conveys the feature status naturally.

### UR-R2-004 — MINOR — wire.settings

Categories: accuracy, grammar.

- `wire.settings.proactive.description`
  - Current: پیش قدم ایجنٹ قسم کی حدود: پوشیدہ انتظار کی ڈیفالٹ مدت، دوبارہ کوششوں میں بیک آف، اور چکر کب ختم ہو۔ یہ انتظار مقرر یا متحرک ہونے پر پڑھی جاتی ہیں۔
  - Proposed: پیش قدم ایجنٹ قسم کی حدود: پوشیدہ انتظار کی ڈیفالٹ مدت، دوبارہ کوششوں میں بیک آف، اور چکر کب ختم ہو۔ یہ ترتیبات انتظار مقرر یا متحرک ہونے پر پڑھی جاتی ہیں۔

Reason: The English says the bounds/settings are read when a wait is armed or fires; یہ انتظار wrongly makes the singular wait the thing being read and disagrees with پڑھی جاتی ہیں.

### UR-R2-005 — MINOR — wire.settings

Categories: native idiom, grammar.

- `wire.settings.retry_connectivitybackoffcapms.help`
  - Current: کنیکٹیوٹی دوبارہ کوششیں کے درمیان سب سے طویل انتظار۔
  - Proposed: کنیکٹیوٹی کی دوبارہ کوششوں کے درمیان سب سے طویل انتظار۔

Reason: The genitive کی and plural کوششوں are required for the natural Urdu phrase 'between connectivity retries'.

### UR-R2-006 — MINOR — wire.settings

Categories: glossary consistency, English leakage.

- `wire.settings.subagents_model_choice.help`
  - Current: ذیلی ایجنٹ کا درجہ کون منتخب کرے؛ operator پروفائل میں مقرر کرتا ہے۔
  - Proposed: ذیلی ایجنٹ کا درجہ کون منتخب کرے؛ آپریٹر پروفائل میں مقرر کرتا ہے۔

Reason: The corresponding choice label now uses آپریٹر, so leaving the same person/role concept as Latin operator in the help is inconsistent.

## Rendered follow-ups and limits

Actual scoped Urdu fits every declared `maxCells`. Pseudo-stretch and approximate `/help` risks remain assigned to rendered review. Native RTL/bidi/shaping, layout, fallback, long-text, and design evidence are **not performed or claimed**.

No source, catalogue, ledger, generated `shipped.json`, eligibility, hold, or PR was edited.

## Evidence bundle

The portable sibling bundle contains the complete scoped `.pairs` workbook, current hashes, all 21 prior dispositions, findings, mechanical results, runtime plural/fit evidence, repository logs, this report, and the JSON receipt. `evidence-manifest.json` records payload SHA-256 hashes.
