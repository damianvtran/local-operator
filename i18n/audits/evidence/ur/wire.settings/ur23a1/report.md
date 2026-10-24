# Canonical evidence assembly — ur / wire.settings / ur23a1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23a1; original receipt: `ur-core.round1.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); session `5ebbaba9537e`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none
- Independent peer record: `i18n/audits/evidence/ur/wire.settings/ur23b1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `failed`; status `failed`; unresolved BLOCKER/MAJOR `3`
- Original source bundle: `/Users/damian/.local-operator/sessions/5ebbaba9537e/scratchpad/ur-core-round1` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.settings",
  "key_count": 818,
  "keys_read": 818,
  "en_sha256": "c1e237c4dd3c2ea236f9e39d15cf4d0b34a231841357aee5a389f34201a73582",
  "ur_sha256": "a9df8aff733066dc9786c40098bd1f4c992f645143991e503e5c8d57b683a942",
  "context_sha256": "67662db2835ed0f6db3a65844efc318c79706735457eb974bb7c64cf34f861a9",
  "hashes_rederived_and_matched": true,
  "translator_session": "ffc94c8cb17d",
  "independent_auditor": true,
  "semantic_verdict": "failed",
  "mechanical_verdict": "passed",
  "finding_ids": [
    "UR-R1-002",
    "UR-R1-003",
    "UR-R1-004",
    "UR-R1-007",
    "UR-R1-010"
  ],
  "unresolved_blocker_major": 3,
  "unresolved_minor": 2,
  "unresolved_nit": 0
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR-R1-002",
    "severity": "major",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "The visible label splices Urdu and English inside one word (ٹرانسکرپٹion) and leaves “cleanup” untranslated, producing garbled mixed-script copy.",
    "affected": [
      {
        "key": "wire.settings.speech_transcription_cleanup.label",
        "current": "ٹرانسکرپٹion cleanup",
        "proposed": "ٹرانسکرپٹ کی صفائی"
      }
    ]
  },
  {
    "id": "UR-R1-003",
    "severity": "major",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "Settled Urdu terms from the wave provenance and other catalogues are replaced by English proactive, remote compute, run, turn and provider; these are prose concepts, not protected identifiers.",
    "affected": [
      {
        "key": "wire.settings.proactive.title",
        "current": "Proactive class",
        "proposed": "پیش قدم قسم"
      },
      {
        "key": "wire.settings.remote_compute.title",
        "current": "Remote compute",
        "proposed": "ریموٹ کمپیوٹ"
      },
      {
        "key": "wire.settings.remote_compute_enabled.label",
        "current": "Remote compute",
        "proposed": "ریموٹ کمپیوٹ"
      },
      {
        "key": "wire.settings.monitor_runtimeoutms.label",
        "current": "Run timeout (ms)",
        "proposed": "رن کی مہلت (ms)"
      },
      {
        "key": "wire.settings.runtime_background_on_resume.choice.false.description",
        "current": "سیشن چھوڑنے پر turn روک دیں",
        "proposed": "سیشن چھوڑنے پر ٹرن روک دیں"
      },
      {
        "key": "wire.settings.runtime_background_on_resume.choice.true.description",
        "current": "Turn background میں چلتا رہے",
        "proposed": "ٹرن پس منظر میں چلتا رہے"
      },
      {
        "key": "wire.settings.display_terminal_title.help",
        "current": "سیشن نام اور run state رکھنے والا OSC 0 window title۔",
        "proposed": "سیشن نام اور رن کی حالت رکھنے والا OSC 0 window title۔"
      },
      {
        "key": "wire.settings.supplements_maxturns.label",
        "current": "↳ زیادہ سے زیادہ turns",
        "proposed": "↳ زیادہ سے زیادہ ٹرن"
      },
      {
        "key": "wire.settings.hosting.label",
        "current": "Default provider",
        "proposed": "ڈیفالٹ پرووائیڈر"
      }
    ],
    "note": "Representative affected keys are named; remediation must sweep the same terms throughout wire.settings rather than fixing only these instances."
  },
  {
    "id": "UR-R1-004",
    "severity": "major",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "“Delta دیکھنے والے نگرانی” has gender/number disagreement and is a word-for-word calque; the proposed wording preserves all three monitor behaviours in natural Urdu.",
    "affected": [
      {
        "key": "wire.settings.monitor.description",
        "current": "Delta دیکھنے والے نگرانی: کتنی بار check کریں، کتنا رکھیں، اور کب ہار مانیں۔",
        "proposed": "تبدیلیوں کی نگرانی: کتنی بار جانچ ہو، کتنا مواد رکھا جائے، اور کب کوشش ختم ہو۔"
      }
    ]
  },
  {
    "id": "UR-R1-007",
    "severity": "minor",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "Here child means a subagent and row is a UI row; the English generic nouns obscure both and drift from ذیلی ایجنٹ and قطار used elsewhere.",
    "affected": [
      {
        "key": "wire.settings.display_dock.choice.full.description",
        "current": "ہر child کی row، نیا پہلے",
        "proposed": "ہر ذیلی ایجنٹ کی قطار، نیا پہلے"
      }
    ]
  },
  {
    "id": "UR-R1-010",
    "severity": "minor",
    "namespaces": [
      "wire.settings"
    ],
    "reason": "“منتخب ہونے کی حد” literally means “the limit of being selected”; the condition needs منطخب ہونے پر لاگو حد (“the limit that applies when selected”).",
    "affected": [
      {
        "key": "wire.settings.bash_memory_limit_mb.help",
        "current": "اوپر 'manual' منتخب ہونے کی حد۔ 0 کا مطلب auto حد استعمال کریں، اس لیے 0 رہنے سے guard بند نہیں ہوتا۔",
        "proposed": "اوپر 'manual' منتخب ہونے پر لاگو حد۔ 0 کا مطلب auto حد استعمال کریں، اس لیے 0 رہنے سے guard بند نہیں ہوتا۔"
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
