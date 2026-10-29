# Canonical evidence assembly — ur / tui.review / ur23a1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23a1; original receipt: `ur-core.round1.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); session `5ebbaba9537e`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none
- Independent peer record: `i18n/audits/evidence/ur/tui.review/ur23b1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/5ebbaba9537e/scratchpad/ur-core-round1` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "tui.review",
  "key_count": 73,
  "keys_read": 73,
  "en_sha256": "39c168340e3bac3f47e3a94358026e10b99806ce4147427e6fd7c70334e00a61",
  "ur_sha256": "683847761d616f3145caaaabd733f57af5c1e749c7c640403cd92f22e54bd868",
  "context_sha256": "b879498a3ba2debfac6ec65a640a75b8d53e71073a1d32676dc41f7325dbc202",
  "hashes_rederived_and_matched": true,
  "translator_session": "7c4a4229b011",
  "independent_auditor": true,
  "semantic_verdict": "clean_with_followups",
  "mechanical_verdict": "passed",
  "finding_ids": [
    "UR-R1-009"
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
    "id": "UR-R1-009",
    "severity": "minor",
    "namespaces": [
      "tui.review"
    ],
    "reason": "جائزہ {reviewed} reads as “review {hash}” rather than “reviewed at {hash}”; fronting the revision with پر restores the relation.",
    "affected": [
      {
        "key": "tui.review.lane_stale_head",
        "current": "پرانا — جائزہ {reviewed}، head {head}",
        "proposed": "پرانا — {reviewed} پر جائزہ، head {head}"
      },
      {
        "key": "tui.review.lane_stale_reviewed",
        "current": "پرانا — جائزہ {reviewed}",
        "proposed": "پرانا — {reviewed} پر جائزہ"
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
