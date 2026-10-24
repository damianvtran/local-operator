# Canonical evidence assembly — ur / tui.welcome / ur24b2

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur24b2; original receipt: `ur-core-round2-audit.receipt.json`
- Round: 2; source revision: `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent scoped round-2 core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/tui.welcome/ur24a2/record.json`
- Current-byte round used for ledger: yes
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-round2-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "tui.welcome",
  "namespace_total_keys": 30,
  "changed_keys_reviewed": 2,
  "unchanged_neighbor_keys_reviewed": 0,
  "keys_reviewed": 2,
  "en_sha256": "cd5bb5b95c8e71d4bf89fdc7f26bede91dee8fadedb5ef786a7f3e7731c57a50",
  "ur_sha256": "db161318b2cf14288b928fc9aa9eaa9d597ea482e7a45dd88e7b989f8504718d",
  "context_sha256": "bfd46d9dbcc5c4dfbee6a3fd93e47264f17f123e01ddf84a7aae99869c41debd",
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
          "key": "tui.welcome.tip.fork_placement",
          "actual_cells": 54,
          "maxCells": 57,
          "pseudo_cells": 71
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
        "key": "tui.welcome.tip.fork_placement",
        "actual_cells": 54,
        "maxCells": 57,
        "pseudo_cells": 71
      }
    ],
    "note": "Actual Urdu fits maxCells; 30% pseudo-stretch exceeds it and is retained only as rendered fit-risk evidence."
  }
]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
