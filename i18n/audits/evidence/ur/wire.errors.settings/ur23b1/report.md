# Canonical evidence assembly — ur / wire.errors.settings / ur23b1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23b1; original receipt: `ur-core-round1-second-audit.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent second core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/wire.errors.settings/ur23a1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `changes_requested`; status `failed`; unresolved BLOCKER/MAJOR `1`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-second-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.errors.settings",
  "keys_read": 12,
  "en_sha256": "a3d05f4d250d92e8cda0f01a92c356957edbf67de5ddee19f8ec949b5d2f30bf",
  "ur_sha256": "7da3d38e00605b0ce286e1c1aa02dddd8766e5d483d517ed026966b02b0e4c0c",
  "context_sha256": null,
  "verdict": "changes_requested",
  "result": "failed",
  "unresolved_blocker_major_count": 1,
  "finding_ids": [
    "UR2-009"
  ],
  "accepted_minor_nit_ids": []
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR2-009",
    "severity": "MAJOR",
    "namespaces": [
      "wire.errors.settings"
    ],
    "category": [
      "redaction artifact",
      "accuracy",
      "native idiom",
      "English leakage"
    ],
    "examples": [
      {
        "key": "wire.errors.settings.credential_key_empty",
        "current": "کریڈینشل key خالی نہیں ہو سکتی",
        "proposed": "کریڈینشل کی کلید خالی نہیں ہو سکتی"
      },
      {
        "key": "wire.errors.settings.inline_credentials",
        "current": "ایسا [redacted] استعمال کریں جس میں inline credentials یا query parameters نہ ہوں۔",
        "proposed": "ایسا اینڈ پوائنٹ استعمال کریں جس میں ان لائن کریڈینشلز یا کوئری پیرامیٹرز نہ ہوں۔"
      },
      {
        "key": "wire.errors.settings.not_registered",
        "current": "یہ ترتیب رجسٹر نہیں ہے۔",
        "proposed": "یہ ترتیب رجسٹرڈ نہیں ہے۔"
      }
    ],
    "reason": "The Urdu error replaces the English endpoint noun with a literal [redacted] artifact and also leaves inline credentials/query parameters in English, corrupting a security-relevant instruction rather than localizing it."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
