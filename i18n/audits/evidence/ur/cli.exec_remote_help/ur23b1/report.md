# Canonical evidence assembly — ur / cli.exec_remote_help / ur23b1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23b1; original receipt: `ur-core-round1-second-audit.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent second core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/cli.exec_remote_help/ur23a1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-second-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "cli.exec_remote_help",
  "keys_read": 11,
  "en_sha256": "c555648c19dd25af507951dc828f3fffbedc44feb579192c9c3fa7c6952e365e",
  "ur_sha256": "6fe867390d79bd9c59172ffd17cfc0a3db6016df45b426b6a945fcbf74662cf6",
  "context_sha256": "fd6bbdfcac16ece10813da5ee3b7499cede367e26343b67415d8295112f13b8f",
  "verdict": "clean_with_followups",
  "result": "passed",
  "unresolved_blocker_major_count": 0,
  "finding_ids": [
    "UR2-008"
  ],
  "accepted_minor_nit_ids": [
    "UR2-008"
  ]
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR2-008",
    "severity": "MINOR",
    "namespaces": [
      "cli.exec_remote_help"
    ],
    "category": [
      "native idiom",
      "script policy",
      "glossary consistency"
    ],
    "examples": [
      {
        "key": "cli.exec_remote_help.idempotency_key",
        "current": "نئی key بنانے کے بجائے استعمال کرنے کے لیے idempotency key، تاکہ جمع شدہ رن کو retry کرنے پر وہ دوبارہ نہ چلے",
        "proposed": "نئی کلید بنانے کے بجائے موجودہ idempotency key استعمال کریں، تاکہ جمع شدہ رن کی دوبارہ کوشش پر وہ دوبارہ نہ چلے"
      },
      {
        "key": "cli.exec_remote_help.repo",
        "current": "Git URL جسے ریموٹ رن checkout کرے گا۔ --remote کے ساتھ درکار: مقامی فائلیں کبھی upload نہیں ہوتیں",
        "proposed": "Git URL جسے ریموٹ رن checkout کرے گا۔ --remote کے ساتھ درکار: مقامی فائلیں کبھی اپلوڈ نہیں ہوتیں"
      },
      {
        "key": "cli.exec_remote_help.size",
        "current": "ریموٹ رن کا compute size: s، m یا l (ڈیفالٹ: m)",
        "proposed": "ریموٹ رن کا کمپیوٹ سائز: s، m یا l (ڈیفالٹ: m)"
      }
    ],
    "reason": "Several settled loanwords and ordinary operations remain Latin even though the same catalogue uses کلید and Urdu-script technical loanwords elsewhere."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
