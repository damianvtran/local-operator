# Canonical evidence assembly — ur / wire.settings / ur24b2

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur24b2; original receipt: `ur-core-round2-audit.receipt.json`
- Round: 2; source revision: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent scoped round-2 core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/wire.settings/ur24a2/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-round2-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.settings",
  "namespace_total_keys": 818,
  "changed_keys_reviewed": 589,
  "unchanged_neighbor_keys_reviewed": 0,
  "keys_reviewed": 589,
  "en_sha256": "c1e237c4dd3c2ea236f9e39d15cf4d0b34a231841357aee5a389f34201a73582",
  "ur_sha256": "24be0297765a8cf23a7c0cfb25f2cb8a0cb9d4e841259986cc87cc7a6df2d1f7",
  "context_sha256": "67662db2835ed0f6db3a65844efc318c79706735457eb974bb7c64cf34f861a9",
  "verdict": "clean_with_followups",
  "result": "passed",
  "unresolved_blocker_major_count": 0,
  "finding_ids": [
    "UR-R2-002",
    "UR-R2-003",
    "UR-R2-004",
    "UR-R2-005",
    "UR-R2-006"
  ],
  "accepted_minor_nit_ids": [
    "UR-R2-002",
    "UR-R2-003",
    "UR-R2-004",
    "UR-R2-005",
    "UR-R2-006"
  ],
  "rendered_followups": []
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR-R2-002",
    "severity": "MINOR",
    "namespaces": [
      "wire.settings"
    ],
    "category": [
      "English leakage",
      "glossary consistency"
    ],
    "examples": [
      {
        "key": "wire.settings.compaction_wire_bytes_trigger.label",
        "current": "درخواست محرک (bytes)",
        "proposed": "درخواست محرک (بائٹس)"
      },
      {
        "key": "wire.settings.network_audit_max_bytes.help",
        "current": "گردش کے بعد فی جنیریشن bytes۔ 8 MiB gzip شدہ JSONL میں تقریباً 55k ریکارڈز ہوتے ہیں؛ پانچ جنیریشنز کی حد اس لیے ~40 MiB ہے۔",
        "proposed": "گردش کے بعد فی جنیریشن بائٹس۔ 8 MiB gzip شدہ JSONL میں تقریباً 55k ریکارڈز ہوتے ہیں؛ پانچ جنیریشنز کی حد اس لیے ~40 MiB ہے۔"
      },
      {
        "key": "wire.settings.session_cleanup_max_total_bytes.help",
        "current": "صفائی آن ہو۔ اتنے bytes سے بڑھنے پر پرانے ہٹائیں؛ 0 = کوئی حد نہیں۔",
        "proposed": "صفائی آن ہو۔ اتنے بائٹس سے بڑھنے پر پرانے ہٹائیں؛ 0 = کوئی حد نہیں۔"
      },
      {
        "key": "wire.settings.web_fetch_max_bytes.label",
        "current": "ڈاؤن لوڈ حد (bytes)",
        "proposed": "ڈاؤن لوڈ حد (بائٹس)"
      },
      {
        "key": "wire.settings.tui_theme.help",
        "current": "رنگوں کا تسلسل۔ /theme arrow-key پیش نظارہ کے ساتھ براہ راست بدلتی ہے۔",
        "proposed": "رنگوں کا تسلسل۔ /theme ایرو کیز کے پیش نظارہ کے ساتھ براہ راست بدلتی ہے۔"
      }
    ],
    "reason": "These changed values leave ordinary `bytes` and `arrow-key` prose in Latin even though the remediation uses بائٹس and ایرو کیز for the same concepts elsewhere."
  },
  {
    "id": "UR-R2-003",
    "severity": "MINOR",
    "namespaces": [
      "wire.settings"
    ],
    "category": [
      "accuracy",
      "false friend"
    ],
    "examples": [
      {
        "key": "wire.settings.remote_compute.description",
        "current": "پیش نظارہ: ایجنٹ کا کام Radient کے ہوسٹ کیے کمپیوٹ پر چلائیں، آپ کے ایک میش نیٹ ورک سے منسلک۔",
        "proposed": "تجرباتی: ایجنٹ کا کام Radient کے ہوسٹ کیے کمپیوٹ پر چلائیں، آپ کے ایک میش نیٹ ورک سے منسلک۔"
      },
      {
        "key": "wire.settings.remote_compute_enabled.help",
        "current": "پیش نظارہ: کام Radient کمپیوٹ کو بھیجے، ایک میش نیٹ ورک سے منسلک۔",
        "proposed": "تجرباتی: کام Radient کمپیوٹ کو بھیجے، ایک میش نیٹ ورک سے منسلک۔"
      }
    ],
    "reason": "Here Preview describes a pre-release feature, not a visual preview, so پیش نظارہ is the wrong sense; تجرباتی conveys the feature status naturally."
  },
  {
    "id": "UR-R2-004",
    "severity": "MINOR",
    "namespaces": [
      "wire.settings"
    ],
    "category": [
      "accuracy",
      "grammar"
    ],
    "examples": [
      {
        "key": "wire.settings.proactive.description",
        "current": "پیش قدم ایجنٹ قسم کی حدود: پوشیدہ انتظار کی ڈیفالٹ مدت، دوبارہ کوششوں میں بیک آف، اور چکر کب ختم ہو۔ یہ انتظار مقرر یا متحرک ہونے پر پڑھی جاتی ہیں۔",
        "proposed": "پیش قدم ایجنٹ قسم کی حدود: پوشیدہ انتظار کی ڈیفالٹ مدت، دوبارہ کوششوں میں بیک آف، اور چکر کب ختم ہو۔ یہ ترتیبات انتظار مقرر یا متحرک ہونے پر پڑھی جاتی ہیں۔"
      }
    ],
    "reason": "The English says the bounds/settings are read when a wait is armed or fires; یہ انتظار wrongly makes the singular wait the thing being read and disagrees with پڑھی جاتی ہیں."
  },
  {
    "id": "UR-R2-005",
    "severity": "MINOR",
    "namespaces": [
      "wire.settings"
    ],
    "category": [
      "native idiom",
      "grammar"
    ],
    "examples": [
      {
        "key": "wire.settings.retry_connectivitybackoffcapms.help",
        "current": "کنیکٹیوٹی دوبارہ کوششیں کے درمیان سب سے طویل انتظار۔",
        "proposed": "کنیکٹیوٹی کی دوبارہ کوششوں کے درمیان سب سے طویل انتظار۔"
      }
    ],
    "reason": "The genitive کی and plural کوششوں are required for the natural Urdu phrase 'between connectivity retries'."
  },
  {
    "id": "UR-R2-006",
    "severity": "MINOR",
    "namespaces": [
      "wire.settings"
    ],
    "category": [
      "glossary consistency",
      "English leakage"
    ],
    "examples": [
      {
        "key": "wire.settings.subagents_model_choice.help",
        "current": "ذیلی ایجنٹ کا درجہ کون منتخب کرے؛ operator پروفائل میں مقرر کرتا ہے۔",
        "proposed": "ذیلی ایجنٹ کا درجہ کون منتخب کرے؛ آپریٹر پروفائل میں مقرر کرتا ہے۔"
      }
    ],
    "reason": "The corresponding choice label now uses آپریٹر, so leaving the same person/role concept as Latin operator in the help is inconsistent."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
