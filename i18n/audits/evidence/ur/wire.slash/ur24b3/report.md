# Canonical evidence assembly — ur / wire.slash / ur24b3

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur24b3; original receipt: `ur-core-round3-audit.receipt.json`
- Round: 3; source revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent scoped round-3 core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/wire.slash/ur24a3/record.json`
- Current-byte round used for ledger: yes
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-round3-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.slash",
  "namespace_total_keys": 52,
  "changed_keys_reviewed": 1,
  "direct_neighbors_reviewed": 0,
  "keys_reviewed": 1,
  "en_sha256": "b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202",
  "ur_sha256": "e1897b69d355d05d44dd6761affb866749fa431630234421dfb31c65141c8cb1",
  "context_sha256": "32b3475a5808048a6eeaae462fee0feb55405fca77026e41b9609a516d74a215",
  "verdict": "clean_with_followups",
  "result": "passed",
  "unresolved_blocker_major_count": 0,
  "finding_ids": [],
  "rendered_followups": [
    {
      "id": "UR-R3-FIT-001",
      "kind": "approximate_help_rendered_fit",
      "risks": [
        {
          "key": "wire.slash.hub",
          "cells": 57,
          "approx_guide_cells": 55
        }
      ],
      "note": "Changed subscription wording measures 57 cells against the approximate 55-cell /help guide. This is retained for rendered review, not a silent pass or a measured rendered failure."
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
    "id": "UR-R3-FIT-001",
    "kind": "approximate_help_rendered_fit",
    "risks": [
      {
        "key": "wire.slash.hub",
        "cells": 57,
        "approx_guide_cells": 55
      }
    ],
    "note": "Changed subscription wording measures 57 cells against the approximate 55-cell /help guide. This is retained for rendered review, not a silent pass or a measured rendered failure."
  }
]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
