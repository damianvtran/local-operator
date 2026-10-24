# Canonical evidence assembly — ur / wire.remote_compute / ur23b1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23b1; original receipt: `ur-core-round1-second-audit.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent second core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/wire.remote_compute/ur23a1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `changes_requested`; status `failed`; unresolved BLOCKER/MAJOR `1`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-second-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.remote_compute",
  "keys_read": 43,
  "en_sha256": "d1d9119a235c1a8c09306f42438115f5211ede16addf80a8f88d6f78860367a8",
  "ur_sha256": "50991b04b030ff3851190fe1c7ef71d7b9306a712ca5af264b8edbfe7d4bb826",
  "context_sha256": "ef149146941e72c79a64446737a5d12be5016850b04df462f48e3a6eb0ae2e36",
  "verdict": "changes_requested",
  "result": "failed",
  "unresolved_blocker_major_count": 1,
  "finding_ids": [
    "UR2-007"
  ],
  "accepted_minor_nit_ids": []
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR2-007",
    "severity": "MAJOR",
    "namespaces": [
      "wire.remote_compute"
    ],
    "category": [
      "English leakage",
      "native idiom",
      "glossary consistency"
    ],
    "affected_scope": "Systematic Latin technical prose across the namespace; preserve commands, flags, URLs, identifiers, product names, and Git operations while translating ordinary labels and prose.",
    "examples": [
      {
        "key": "wire.remote_compute.finished_exit_code",
        "current": "exit code: {exit_code}",
        "proposed": "خروجی کوڈ: {exit_code}"
      },
      {
        "key": "wire.remote_compute.detached",
        "current": "ریموٹ رن {run_id} جمع کر دیا گیا۔\nnetwork (Radient کی تصدیق شدہ): {network_id}\nIdempotency key: {idempotency_key}\nاس کی نگرانی: lop exec --remote-status {run_id}\nاسے منسوخ کریں: lop exec --remote-cancel {run_id}",
        "proposed": "ریموٹ رن {run_id} جمع کر دیا گیا۔\nنیٹ ورک (Radient کی تصدیق شدہ): {network_id}\nIdempotency key: {idempotency_key}\nاس کی نگرانی: lop exec --remote-status {run_id}\nاسے منسوخ کریں: lop exec --remote-cancel {run_id}"
      },
      {
        "key": "wire.remote_compute.key_not_printable",
        "current": "idempotency key صرف printable ASCII ہونی چاہیے: حروف، اعداد اور punctuation marks؛ accents، tabs یا newlines نہیں۔",
        "proposed": "idempotency key صرف قابلِ طباعت ASCII حروف پر مشتمل ہونی چاہیے: حروف، اعداد اور اوقاف؛ ایکسنٹ والے حروف، ٹیبز یا نئی سطریں شامل نہیں۔"
      }
    ],
    "reason": "A complete output label and substantial ordinary prose remain in English, while نیٹ ورک is already the established rendering elsewhere in the same namespace."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
