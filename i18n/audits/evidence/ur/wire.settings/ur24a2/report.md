# Canonical evidence assembly — ur / wire.settings / ur24a2

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur24a2; original receipt: `ur-core.round2.receipt.json`
- Round: 2; source revision: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); session `5ebbaba9537e`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none
- Independent peer record: `i18n/audits/evidence/ur/wire.settings/ur24b2/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `failed`; status `failed`; unresolved BLOCKER/MAJOR `2`
- Original source bundle: `/Users/damian/.local-operator/sessions/5ebbaba9537e/scratchpad/ur-core-round2` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.settings",
  "namespace_key_count": 818,
  "scoped_keys_read": 589,
  "changed_keys": 589,
  "direct_neighbor": false,
  "en_sha256": "c1e237c4dd3c2ea236f9e39d15cf4d0b34a231841357aee5a389f34201a73582",
  "ur_sha256": "24be0297765a8cf23a7c0cfb25f2cb8a0cb9d4e841259986cc87cc7a6df2d1f7",
  "context_sha256": "67662db2835ed0f6db3a65844efc318c79706735457eb974bb7c64cf34f861a9",
  "hashes_rederived_and_matched": true,
  "verdict": "failed",
  "finding_ids": [
    "UR-R2-001",
    "UR-R2-002",
    "UR-R2-003",
    "UR-R2-005",
    "UR-R2-007"
  ],
  "unresolved_blocker_major": 2,
  "unresolved_minor": 3,
  "unresolved_nit": 0
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR-R2-001",
    "severity": "major",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "مکمل means “completed”, but settled explicitly includes done, paused and archived; the current wording can wrongly imply that paused or archived projects are still watched.",
    "affected": [
      {
        "key": "wire.settings.wakes_triggers_project_staleness_enabled.help",
        "current": "ان پروجیکٹس پر رابطے کے لیے بیدار ہوں جن کی پیش رفت پرانی ہو۔ مکمل (done/paused/archived) پروجیکٹس کی نگرانی کبھی نہیں۔",
        "proposed": "ان پروجیکٹس پر رابطے کے لیے بیدار ہوں جن کی پیش رفت پرانی ہو۔ حتمی حالت والے (done/paused/archived) پروجیکٹس کی نگرانی کبھی نہیں۔"
      }
    ]
  },
  {
    "id": "UR-R2-002",
    "severity": "major",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "The coordinated negative is broken: فائلیں ... نہیں بھیجتیں makes the files the actor and does not clearly guarantee that they are neither listed nor sent to the decision model.",
    "affected": [
      {
        "key": "wire.settings.supplements_denyprefixes.help",
        "current": "نمایاں حصے آن ہوں۔ کاما سے الگ فولڈرز جن کی فائلیں کبھی فہرست میں شامل یا فیصلہ ماڈل کو نہیں بھیجتیں؛ گاہکوں کے ڈیٹا والی مشین پر استعمال کریں۔",
        "proposed": "نمایاں حصے آن ہوں۔ کاما سے الگ فولڈرز جن کی فائلیں کبھی فہرست میں نہیں آتیں اور فیصلہ کرنے والے ماڈل کو نہیں بھیجی جاتیں؛ گاہکوں کے ڈیٹا والی مشین پر استعمال کریں۔"
      }
    ]
  },
  {
    "id": "UR-R2-003",
    "severity": "minor",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "The operator concept is translated آپریٹر on its choice label but left as English operator in help, and job is rendered کام while the settled background-job term is جاب.",
    "affected": [
      {
        "key": "wire.settings.subagents_model_choice.help",
        "current": "ذیلی ایجنٹ کا درجہ کون منتخب کرے؛ operator پروفائل میں مقرر کرتا ہے۔",
        "proposed": "ذیلی ایجنٹ کا درجہ کون منتخب کرے؛ آپریٹر پروفائل میں مقرر کرتا ہے۔"
      },
      {
        "key": "wire.settings.subagents_max_running.label",
        "current": "زیادہ سے زیادہ پس منظر کام",
        "proposed": "زیادہ سے زیادہ پس منظر جاب"
      },
      {
        "key": "wire.settings.supplements_maxcostusd.help",
        "current": "گرافکس آن ہوں۔ ایک کام کے خرچ کی نرم حد۔",
        "proposed": "گرافکس آن ہوں۔ ایک جاب کے خرچ کی نرم حد۔"
      },
      {
        "key": "wire.settings.supplements_timeouts.help",
        "current": "نمایاں حصے آن ہوں۔ اتنی دیر بعد پورا کام چھوڑ دیں۔",
        "proposed": "نمایاں حصے آن ہوں۔ اتنی دیر بعد پوری جاب چھوڑ دیں۔"
      }
    ]
  },
  {
    "id": "UR-R2-005",
    "severity": "minor",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "پیش نظارہ describes a visual preview elsewhere, whereas Preview marks feature availability here; پری ویو avoids that false friend.",
    "affected": [
      {
        "key": "wire.settings.remote_compute.description",
        "current": "پیش نظارہ: ایجنٹ کا کام Radient کے ہوسٹ کیے کمپیوٹ پر چلائیں، آپ کے ایک میش نیٹ ورک سے منسلک۔",
        "proposed": "پری ویو: ایجنٹ کا کام Radient کے ہوسٹ کیے کمپیوٹ پر چلائیں، آپ کے ایک میش نیٹ ورک سے منسلک۔"
      },
      {
        "key": "wire.settings.remote_compute_enabled.help",
        "current": "پیش نظارہ: کام Radient کمپیوٹ کو بھیجے، ایک میش نیٹ ورک سے منسلک۔",
        "proposed": "پری ویو: کام Radient کمپیوٹ کو بھیجے، ایک میش نیٹ ورک سے منسلک۔"
      }
    ]
  },
  {
    "id": "UR-R2-007",
    "severity": "minor",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "The source states what the rule does, while کرے turns it into a command or possibility; کرتی ہے restores the factual description.",
    "affected": [
      {
        "key": "wire.settings.display_rail.help",
        "current": "ایک لکیر ANSWER کو نشان زد کرے: ٹرن ختم کرنے والا پیغام۔ ٹرن کے دوران کے بیان پر ریل نہیں۔",
        "proposed": "ایک لکیر ANSWER کو نشان زد کرتی ہے: ٹرن ختم کرنے والا پیغام۔ ٹرن کے دوران کے بیان پر ریل نہیں۔"
      }
    ]
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
