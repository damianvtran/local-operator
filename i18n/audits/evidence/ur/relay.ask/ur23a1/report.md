# Canonical evidence assembly — ur / relay.ask / ur23a1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23a1; original receipt: `ur-core.round1.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); session `5ebbaba9537e`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none
- Independent peer record: `i18n/audits/evidence/ur/relay.ask/ur23b1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `failed`; status `failed`; unresolved BLOCKER/MAJOR `1`
- Original source bundle: `/Users/damian/.local-operator/sessions/5ebbaba9537e/scratchpad/ur-core-round1` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "relay.ask",
  "key_count": 11,
  "keys_read": 11,
  "en_sha256": "2ead3ad32467d6e7c7a7a4aa96a4bde65f7451687fe0878f8dedb464340b24e7",
  "ur_sha256": "960c074ad43959e9fb01d09768d0ccdb4ced21f2b14db5a64713e998d0b78248",
  "context_sha256": "85cb7131423eab2649ff1b7600eeedde3308cdfb36491a04e32daae16ab68c42",
  "hashes_rederived_and_matched": true,
  "translator_session": "7a648afb0f8d",
  "independent_auditor": true,
  "semantic_verdict": "failed",
  "mechanical_verdict": "passed",
  "finding_ids": [
    "UR-R1-001"
  ],
  "unresolved_blocker_major": 1,
  "unresolved_minor": 0,
  "unresolved_nit": 0
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR-R1-001",
    "severity": "major",
    "namespaces": [
      "relay.ask"
    ],
    "reason": "The sidecar explicitly requires the same dismiss verb as the terminal; ختم means “end/finish” and diverges from the terminal’s خارج for the same operation.",
    "affected": [
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
        "proposed": "خارج کیا گیا — کوئی جواب نہیں بھیجا گیا"
      },
      {
        "key": "relay.ask.dismiss.receipt_answered_in_chat",
        "current": "ختم کر دیا گیا — گفتگو میں جواب دے دیا گیا",
        "proposed": "خارج کیا گیا — گفتگو میں جواب موجود ہے"
      },
      {
        "key": "relay.ask.dismiss.receipt_moot",
        "current": "ختم کر دیا گیا — اب ضرورت نہیں",
        "proposed": "خارج کیا گیا — اب ضرورت نہیں"
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
