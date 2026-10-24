# Canonical evidence assembly — ur / relay.ask / ur24b2

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur24b2; original receipt: `ur-core-round2-audit.receipt.json`
- Round: 2; source revision: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent scoped round-2 core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/relay.ask/ur24a2/record.json`
- Current-byte round used for ledger: yes
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-round2-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "relay.ask",
  "namespace_total_keys": 11,
  "changed_keys_reviewed": 5,
  "unchanged_neighbor_keys_reviewed": 0,
  "keys_reviewed": 5,
  "en_sha256": "2ead3ad32467d6e7c7a7a4aa96a4bde65f7451687fe0878f8dedb464340b24e7",
  "ur_sha256": "534aa6d0f7a34c153a088695fa052cc2342fe716de96a0fb0bfc9e5dffba863c",
  "context_sha256": "85cb7131423eab2649ff1b7600eeedde3308cdfb36491a04e32daae16ab68c42",
  "verdict": "clean_with_followups",
  "result": "passed",
  "unresolved_blocker_major_count": 0,
  "finding_ids": [],
  "accepted_minor_nit_ids": [],
  "rendered_followups": [
    {
      "kind": "pseudo_stretch_rendered_fit",
      "keys": [
        {
          "key": "relay.ask.dismiss.open",
          "actual_cells": 10,
          "maxCells": 12,
          "pseudo_cells": 13
        },
        {
          "key": "relay.ask.dismiss.receipt",
          "actual_cells": 39,
          "maxCells": 44,
          "pseudo_cells": 51
        },
        {
          "key": "relay.ask.dismiss.receipt_answered_in_chat",
          "actual_cells": 38,
          "maxCells": 44,
          "pseudo_cells": 50
        }
      ],
      "note": "Actual Urdu fits maxCells; 30% pseudo-stretch exceeds it and is retained only as rendered fit-risk evidence."
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
    "kind": "pseudo_stretch_rendered_fit",
    "keys": [
      {
        "key": "relay.ask.dismiss.open",
        "actual_cells": 10,
        "maxCells": 12,
        "pseudo_cells": 13
      },
      {
        "key": "relay.ask.dismiss.receipt",
        "actual_cells": 39,
        "maxCells": 44,
        "pseudo_cells": 51
      },
      {
        "key": "relay.ask.dismiss.receipt_answered_in_chat",
        "actual_cells": 38,
        "maxCells": 44,
        "pseudo_cells": 50
      }
    ],
    "note": "Actual Urdu fits maxCells; 30% pseudo-stretch exceeds it and is retained only as rendered fit-risk evidence."
  }
]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
