# Canonical evidence assembly — ur / relay.session.tags / ur23a1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23a1; original receipt: `ur-core.round1.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); session `5ebbaba9537e`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none
- Independent peer record: `i18n/audits/evidence/ur/relay.session.tags/ur23b1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `clean_with_followups`; status `passed`; unresolved BLOCKER/MAJOR `0`
- Original source bundle: `/Users/damian/.local-operator/sessions/5ebbaba9537e/scratchpad/ur-core-round1` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "relay.session.tags",
  "key_count": 19,
  "keys_read": 19,
  "en_sha256": "7254ba273a4148ddef8279161682ada6405745a9b1634c59a2c5b6ec9e677af0",
  "ur_sha256": "4f0fab7f40bacc152b1cabba0244372b8d8dec40213c778ef8c417ac6be242aa",
  "context_sha256": "8051bf6e05e0fd4720c8dfbf03094b81eb85b3239b7603c6c817f47cb40714c7",
  "hashes_rederived_and_matched": true,
  "translator_session": "7a648afb0f8d",
  "independent_auditor": true,
  "semantic_verdict": "clean_with_followups",
  "mechanical_verdict": "passed",
  "finding_ids": [
    "UR-R1-008"
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
    "id": "UR-R1-008",
    "severity": "minor",
    "namespaces": [
      "relay.session.tags"
    ],
    "reason": "tui.tags consistently treats ٹیگ as feminine (ٹیگ بن گئی، دوسری); masculine دوسرا/سکے here creates avoidable gender drift for the same term.",
    "affected": [
      {
        "key": "relay.session.tags.limit_hint",
        "current": "{count} ٹیگز ایک گفتگو کے لیے زیادہ سے زیادہ حد ہے — دوسرا شامل کرنے کے لیے ایک ہٹائیں۔",
        "proposed": "{count} ٹیگز ایک گفتگو کے لیے زیادہ سے زیادہ حد ہے — دوسری شامل کرنے کے لیے ایک ہٹائیں۔"
      },
      {
        "key": "relay.session.tags.save_error",
        "current": "ٹیگز محفوظ نہیں ہو سکے",
        "proposed": "ٹیگز محفوظ نہیں ہو سکیں"
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
