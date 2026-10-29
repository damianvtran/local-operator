# Canonical evidence assembly — ur / tui.viewer / ur23b1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23b1; original receipt: `ur-core-round1-second-audit.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent second core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/tui.viewer/ur23a1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-second-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "tui.viewer",
  "keys_read": 19,
  "en_sha256": "741c34bd0dbed2d1834e53a8546b39bb2d086e16df24f2ed786b6108c8a40a4b",
  "ur_sha256": "7a4f8d943e17fe8e706b1f1b567323bea822d51784b7778ec6b43032e849b743",
  "context_sha256": "ed4535ceb6e3119c8a3b35f0aa0570ed0398d61c212a43f35bc6d2aa9ac389ae",
  "verdict": "clean_with_followups",
  "result": "passed",
  "unresolved_blocker_major_count": 0,
  "finding_ids": [
    "UR2-011"
  ],
  "accepted_minor_nit_ids": [
    "UR2-011"
  ]
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR2-011",
    "severity": "MINOR",
    "namespaces": [
      "tui.viewer"
    ],
    "category": [
      "typography",
      "native idiom"
    ],
    "examples": [
      {
        "key": "tui.viewer.unreadable",
        "current": "یہ فائل پڑھی نہیں جا سکتی — موجود نہیں، ناقابلِ قرأت ہے، یا اجازت شدہ روٹ ڈائریکٹریوں سے باہر ہے",
        "proposed": "یہ فائل موجود نہیں، پڑھنے کے قابل نہیں، یا اجازت شدہ روٹ ڈائریکٹریوں سے باہر ہے"
      }
    ],
    "reason": "ناقابلِ قرأت is misspelled and awkward; the proposed wording preserves all three refusal causes in idiomatic Urdu."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
