# Canonical evidence assembly — ur / wire.slash / ur24a2

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur24a2; original receipt: `ur-core.round2.receipt.json`
- Round: 2; source revision: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); session `5ebbaba9537e`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none
- Independent peer record: `i18n/audits/evidence/ur/wire.slash/ur24b2/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/5ebbaba9537e/scratchpad/ur-core-round2` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.slash",
  "namespace_key_count": 52,
  "scoped_keys_read": 23,
  "changed_keys": 23,
  "direct_neighbor": false,
  "en_sha256": "b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202",
  "ur_sha256": "1ce934002aa7bd474a64df26817ab7ebc8ac542d25e01ac908f54042ebb4838e",
  "context_sha256": "32b3475a5808048a6eeaae462fee0feb55405fca77026e41b9609a516d74a215",
  "hashes_rederived_and_matched": true,
  "verdict": "clean_with_followups",
  "finding_ids": [
    "UR-R2-004"
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
    "id": "UR-R2-004",
    "severity": "minor",
    "namespaces": [
      "wire.slash"
    ],
    "reason": "رکنیت means membership, not a per-link subscription policy; سبسکرپشن preserves the actual setting.",
    "affected": [
      {
        "key": "wire.slash.hub",
        "current": "Hub روابط کھولیں — فی ربط رکنیت اور خودکار push پالیسی",
        "proposed": "Hub روابط کھولیں — فی ربط سبسکرپشن اور خودکار push پالیسی"
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
