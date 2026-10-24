# Canonical evidence assembly — ur / tui.welcome / ur23b1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23b1; original receipt: `ur-core-round1-second-audit.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent second core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/tui.welcome/ur23a1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `changes_requested`; status `failed`; unresolved BLOCKER/MAJOR `1`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-second-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "tui.welcome",
  "keys_read": 30,
  "en_sha256": "cd5bb5b95c8e71d4bf89fdc7f26bede91dee8fadedb5ef786a7f3e7731c57a50",
  "ur_sha256": "8c931ffd4d9284216d113daa043b99bda068acc1b34b3600e87011496333dacc",
  "context_sha256": "bfd46d9dbcc5c4dfbee6a3fd93e47264f17f123e01ddf84a7aae99869c41debd",
  "verdict": "changes_requested",
  "result": "failed",
  "unresolved_blocker_major_count": 1,
  "finding_ids": [
    "UR2-004"
  ],
  "accepted_minor_nit_ids": []
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR2-004",
    "severity": "MAJOR",
    "namespaces": [
      "tui.welcome"
    ],
    "category": [
      "English leakage",
      "high-visibility copy"
    ],
    "examples": [
      {
        "key": "tui.welcome.tip.fork_placement",
        "current": "/settings → Fork → “فورک کہاں کھلے” سے مقام طے ہوتا ہے",
        "proposed": "/settings → فورک → “فورک کہاں کھلے” سے مقام طے ہوتا ہے"
      },
      {
        "key": "tui.welcome.tip.hotkeys",
        "current": "/settings → Hotkeys سے یہ کلیدیں بدلیں",
        "proposed": "/settings → ہاٹ کیز سے یہ کلیدیں بدلیں"
      }
    ],
    "reason": "Two opening-screen navigation tips leave the visible settings section names in English even though the surrounding path and the corresponding fork wording are Urdu."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
