# Canonical evidence assembly — ur / tui.asks / ur23b1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23b1; original receipt: `ur-core-round1-second-audit.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent second core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/tui.asks/ur23a1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `changes_requested`; status `failed`; unresolved BLOCKER/MAJOR `2`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-second-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "tui.asks",
  "keys_read": 11,
  "en_sha256": "e3ac8769116d2e2c38107b1918c200dd4ff846f3b1b9b092933e3bf91d787d7e",
  "ur_sha256": "a6b23ec177d56c65c3d7bb22c950f7e202a67ed2dd19467d847015a246a29a1f",
  "context_sha256": "e09cd8e796359ab8c4d8de2fd5a59bb990c76f8488c28846cfb1623610507362",
  "verdict": "changes_requested",
  "result": "failed",
  "unresolved_blocker_major_count": 2,
  "finding_ids": [
    "UR2-005",
    "UR2-006"
  ],
  "accepted_minor_nit_ids": []
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR2-005",
    "severity": "MAJOR",
    "namespaces": [
      "tui.asks"
    ],
    "category": [
      "accuracy",
      "safety meaning"
    ],
    "examples": [
      {
        "key": "tui.asks.dismiss.answered_in_chat",
        "current": "گفتگو میں جواب دیدہ",
        "proposed": "گفتگو میں جواب دے دیا"
      }
    ],
    "reason": "دیدہ means 'seen' and does not express the operator's deliberate 'answered in chat' reason, so the dismiss chooser names the wrong action."
  },
  {
    "id": "UR2-006",
    "severity": "MAJOR",
    "namespaces": [
      "relay.ask",
      "tui.asks"
    ],
    "category": [
      "glossary consistency",
      "explicit context contract"
    ],
    "examples": [
      {
        "key": "relay.ask.dismiss.open",
        "current": "ختم کریں…",
        "proposed": "خارج کریں…"
      },
      {
        "key": "relay.ask.dismiss.title",
        "current": "یہ سوال ختم کریں",
        "proposed": "یہ سوال خارج کریں"
      },
      {
        "key": "relay.ask.dismiss.receipt",
        "current": "ختم کر دیا گیا — کوئی جواب نہیں بھیجا",
        "proposed": "خارج کر دیا گیا — کوئی جواب نہیں بھیجا"
      },
      {
        "key": "relay.ask.dismiss.receipt_answered_in_chat",
        "current": "ختم کر دیا گیا — گفتگو میں جواب دے دیا گیا",
        "proposed": "خارج کر دیا گیا — گفتگو میں جواب دے دیا گیا"
      },
      {
        "key": "relay.ask.dismiss.receipt_moot",
        "current": "ختم کر دیا گیا — اب ضرورت نہیں",
        "proposed": "خارج کر دیا گیا — اب ضرورت نہیں"
      }
    ],
    "reason": "The phone uses ختم while the terminal uses خارج for the same dismiss act, directly violating the phone sidecar's instruction to use the terminal's verb and creating cross-surface terminology drift."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
