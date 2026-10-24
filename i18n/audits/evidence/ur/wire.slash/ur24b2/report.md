# Canonical evidence assembly — ur / wire.slash / ur24b2

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur24b2; original receipt: `ur-core-round2-audit.receipt.json`
- Round: 2; source revision: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent scoped round-2 core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/wire.slash/ur24a2/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-round2-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.slash",
  "namespace_total_keys": 52,
  "changed_keys_reviewed": 23,
  "unchanged_neighbor_keys_reviewed": 0,
  "keys_reviewed": 23,
  "en_sha256": "b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202",
  "ur_sha256": "1ce934002aa7bd474a64df26817ab7ebc8ac542d25e01ac908f54042ebb4838e",
  "context_sha256": "32b3475a5808048a6eeaae462fee0feb55405fca77026e41b9609a516d74a215",
  "verdict": "clean_with_followups",
  "result": "passed",
  "unresolved_blocker_major_count": 0,
  "finding_ids": [],
  "accepted_minor_nit_ids": [],
  "rendered_followups": [
    {
      "kind": "approximate_help_rendered_fit",
      "keys": [
        "wire.slash.agent",
        "wire.slash.fork",
        "wire.slash.highlights"
      ],
      "note": "Approximate text-cell counts exceed the ~55-cell /help guide; rendered review remains required and this is not a silent pass."
    }
  ]
}
```

## Namespace findings preserved from source findings JSON

```json
[]
```

## Rendered/fit follow-ups

```json
[
  {
    "kind": "approximate_help_rendered_fit",
    "keys": [
      "wire.slash.agent",
      "wire.slash.fork",
      "wire.slash.highlights"
    ],
    "note": "Approximate text-cell counts exceed the ~55-cell /help guide; rendered review remains required and this is not a silent pass."
  }
]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
