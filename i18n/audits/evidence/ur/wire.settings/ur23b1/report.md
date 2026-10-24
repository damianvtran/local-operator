# Canonical evidence assembly — ur / wire.settings / ur23b1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23b1; original receipt: `ur-core-round1-second-audit.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent second core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/wire.settings/ur23a1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `changes_requested`; status `failed`; unresolved BLOCKER/MAJOR `1`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-second-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.settings",
  "keys_read": 818,
  "en_sha256": "c1e237c4dd3c2ea236f9e39d15cf4d0b34a231841357aee5a389f34201a73582",
  "ur_sha256": "a9df8aff733066dc9786c40098bd1f4c992f645143991e503e5c8d57b683a942",
  "context_sha256": "67662db2835ed0f6db3a65844efc318c79706735457eb974bb7c64cf34f861a9",
  "verdict": "changes_requested",
  "result": "failed",
  "unresolved_blocker_major_count": 1,
  "finding_ids": [
    "UR2-001"
  ],
  "accepted_minor_nit_ids": []
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR2-001",
    "severity": "MAJOR",
    "namespaces": [
      "wire.settings"
    ],
    "category": [
      "English leakage",
      "native idiom",
      "glossary consistency"
    ],
    "affected_scope": "Systematic across the 818-key namespace; the examples are representative, not a complete remediation list. Retranslate all ordinary English prose while preserving permitted machine tokens.",
    "examples": [
      {
        "key": "wire.settings.classification.title",
        "current": "Smart hints",
        "proposed": "ذہین اشارے"
      },
      {
        "key": "wire.settings.bash_query_budget_enabled.label",
        "current": "Shell query budget",
        "proposed": "شیل کوئری کا بجٹ"
      },
      {
        "key": "wire.settings.desktop.title",
        "current": "Desktop app",
        "proposed": "ڈیسک ٹاپ ایپ"
      },
      {
        "key": "wire.settings.display_heading_markers.label",
        "current": "Heading markers",
        "proposed": "سرخی کے نشانات"
      },
      {
        "key": "wire.settings.web_search_providers.label",
        "current": "Providers",
        "proposed": "پرووائیڈرز"
      },
      {
        "key": "wire.settings.speech_transcription_cleanup.label",
        "current": "ٹرانسکرپٹion cleanup",
        "proposed": "ٹرانسکرپشن صفائی"
      }
    ],
    "reason": "Whole user-visible labels and many ordinary help clauses remain English or hybrid rather than native Urdu, far beyond the style guide's narrow permission for product names, identifiers, commands, flags, paths, tool names, and acronyms."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
