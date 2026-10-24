# Canonical evidence assembly — ur / wire.errors.settings / ur24b2

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur24b2; original receipt: `ur-core-round2-audit.receipt.json`
- Round: 2; source revision: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent scoped round-2 core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/wire.errors.settings/ur24a2/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `changes_requested`; status `failed`; unresolved BLOCKER/MAJOR `1`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-round2-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.errors.settings",
  "namespace_total_keys": 12,
  "changed_keys_reviewed": 4,
  "unchanged_neighbor_keys_reviewed": 0,
  "keys_reviewed": 4,
  "en_sha256": "a3d05f4d250d92e8cda0f01a92c356957edbf67de5ddee19f8ec949b5d2f30bf",
  "ur_sha256": "b15d7f47abe473cfd019e1599e0d84569b9ceeec9159f1f78dc7391fefb1eda2",
  "context_sha256": null,
  "verdict": "changes_requested",
  "result": "failed",
  "unresolved_blocker_major_count": 1,
  "finding_ids": [
    "UR-R2-001"
  ],
  "accepted_minor_nit_ids": [],
  "rendered_followups": []
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR-R2-001",
    "severity": "MAJOR",
    "namespaces": [
      "wire.errors.settings"
    ],
    "category": [
      "redaction artifact",
      "accuracy",
      "security-relevant copy"
    ],
    "reopens": "UR2-009",
    "examples": [
      {
        "key": "wire.errors.settings.inline_credentials",
        "current": "ایسا اینڈ پوائنٹ استعمال کریں جس میں ان لائن [redacted]ز یا کوئری پیرامیٹرز نہ ہوں۔",
        "proposed": "ایسا اینڈ پوائنٹ استعمال کریں جس میں ان لائن کریڈینشلز یا کوئری پیرامیٹرز نہ ہوں۔"
      }
    ],
    "reason": "The first safety predicate and query wording are fixed, but the credential noun is still stored as the 10-code-point literal `[redacted]` plus Urdu suffix, so UR2-009 is only partially remediated and the safety instruction still contains a redaction artifact."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
