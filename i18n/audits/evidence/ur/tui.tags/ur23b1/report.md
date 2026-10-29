# Canonical evidence assembly — ur / tui.tags / ur23b1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23b1; original receipt: `ur-core-round1-second-audit.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent second core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/tui.tags/ur23a1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-second-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "tui.tags",
  "keys_read": 27,
  "en_sha256": "509d4e9ff9c7a90b3cf574a0ebc56432d37919aaddb74e6a6aa8ad945f8bd8bd",
  "ur_sha256": "7a2e08723312b513b5af38691134ec2fdc381f120e913b691588b777d4265821",
  "context_sha256": null,
  "verdict": "clean_with_followups",
  "result": "passed",
  "unresolved_blocker_major_count": 0,
  "finding_ids": [
    "UR2-010"
  ],
  "accepted_minor_nit_ids": [
    "UR2-010"
  ]
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR2-010",
    "severity": "MINOR",
    "namespaces": [
      "tui.tags"
    ],
    "category": [
      "register",
      "grammatical consistency"
    ],
    "examples": [
      {
        "key": "tui.tags.added",
        "current": "“{name}” شامل کی گئی۔",
        "proposed": "“{name}” شامل کیا گیا۔"
      },
      {
        "key": "tui.tags.create_help",
        "current": "مشترکہ ٹیگ تیار کر کے یہاں لاگو کی جاتی ہے۔",
        "proposed": "مشترکہ ٹیگ تیار کر کے یہاں لاگو کیا جاتا ہے۔"
      },
      {
        "key": "tui.tags.created_applied",
        "current": "“{name}” تیار کر کے لاگو کی گئی۔",
        "proposed": "“{name}” تیار کر کے لاگو کیا گیا۔"
      },
      {
        "key": "tui.tags.partial_create",
        "current": "ٹیگ بن گئی، مگر لاگو نہیں ہو سکی۔ دوبارہ لاگو کرنے کی کوشش کریں۔",
        "proposed": "ٹیگ بن گیا، مگر لاگو نہیں ہو سکا۔ دوبارہ لاگو کرنے کی کوشش کریں۔"
      },
      {
        "key": "tui.tags.cap_applied",
        "current": "8 ٹیگز لاگو ہیں۔ دوسری شامل کرنے سے پہلے ایک ہٹائیں۔",
        "proposed": "8 ٹیگز لاگو ہیں۔ دوسرا شامل کرنے سے پہلے ایک ہٹائیں۔"
      }
    ],
    "reason": "The terminal treats ٹیگ as feminine here but the relay and neighbouring plural constructions treat it as masculine, so the same noun changes gender across surfaces."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
