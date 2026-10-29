# Canonical evidence assembly — ur / wire.remote_compute / ur24a2

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur24a2; original receipt: `ur-core.round2.receipt.json`
- Round: 2; source revision: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); session `5ebbaba9537e`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none
- Independent peer record: `i18n/audits/evidence/ur/wire.remote_compute/ur24b2/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/5ebbaba9537e/scratchpad/ur-core-round2` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.remote_compute",
  "namespace_key_count": 43,
  "scoped_keys_read": 21,
  "changed_keys": 21,
  "direct_neighbor": false,
  "en_sha256": "d1d9119a235c1a8c09306f42438115f5211ede16addf80a8f88d6f78860367a8",
  "ur_sha256": "155435eb060235e868761a235e22d55423a502daab51c61b62eb4d5f135245dc",
  "context_sha256": "ef149146941e72c79a64446737a5d12be5016850b04df462f48e3a6eb0ae2e36",
  "hashes_rederived_and_matched": true,
  "verdict": "clean_with_followups",
  "finding_ids": [
    "UR-R2-006"
  ],
  "unresolved_blocker_major": 0,
  "unresolved_minor": 1,
  "unresolved_nit": 0
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR-R2-006",
    "severity": "minor",
    "namespaces": [
      "wire.remote_compute"
    ],
    "reason": "خروجی کوڈ can read as “output code”; ایگزٹ کوڈ is the unambiguous technical term for the process status.",
    "affected": [
      {
        "key": "wire.remote_compute.finished_exit_code",
        "current": "خروجی کوڈ: {exit_code}",
        "proposed": "ایگزٹ کوڈ: {exit_code}"
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
