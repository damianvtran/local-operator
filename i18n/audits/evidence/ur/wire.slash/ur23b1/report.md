# Canonical evidence assembly — ur / wire.slash / ur23b1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23b1; original receipt: `ur-core-round1-second-audit.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); independent second core auditor; session `3e0f499dc2ee`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent peer record: `i18n/audits/evidence/ur/wire.slash/ur23a1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `changes_requested`; status `failed`; unresolved BLOCKER/MAJOR `1`
- Original source bundle: `/Users/damian/.local-operator/sessions/3e0f499dc2ee/scratchpad/ur-core-second-audit` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.slash",
  "keys_read": 52,
  "en_sha256": "b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202",
  "ur_sha256": "35ebd056ec897ded1cba799498ab2c67ba9fdff4636dcf1ca6a1cc1a2bec421a",
  "context_sha256": "32b3475a5808048a6eeaae462fee0feb55405fca77026e41b9609a516d74a215",
  "verdict": "changes_requested",
  "result": "failed",
  "unresolved_blocker_major_count": 1,
  "finding_ids": [
    "UR2-002",
    "UR2-003"
  ],
  "accepted_minor_nit_ids": []
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR2-002",
    "severity": "MAJOR",
    "namespaces": [
      "wire.slash"
    ],
    "category": [
      "English leakage",
      "glossary consistency",
      "native idiom"
    ],
    "affected_scope": "Systematic Latin prose and drift from the wave's own پرووائیڈر decision across the 52 slash descriptions; inspect every value during remediation.",
    "examples": [
      {
        "key": "wire.slash.agent",
        "current": "ایجنٹس دیکھیں، class بدلیں، یا ایجنٹ بن کر بات کریں",
        "proposed": "ایجنٹس دیکھیں، کسی ایجنٹ کی قسم بدلیں، یا اس سیشن سے بطور ایجنٹ بات کریں"
      },
      {
        "key": "wire.slash.provider",
        "current": "providers اور ان کی لاگ اِن/استعمال کی حالت کی فہرست",
        "proposed": "پرووائیڈرز اور ان کے لاگ اِن/استعمال کی حالت کی فہرست"
      },
      {
        "key": "wire.slash.move",
        "current": "اس سیشن کی working directory بدلیں",
        "proposed": "اس سیشن کی ورکنگ ڈائریکٹری بدلیں"
      },
      {
        "key": "wire.slash.usage",
        "current": "provider کا usage quota دکھائیں",
        "proposed": "پرووائیڈر کے استعمال کا کوٹہ دکھائیں"
      }
    ],
    "reason": "High-visibility slash help repeatedly leaves ordinary nouns in Latin and alternates provider/providers with the translator-endorsed پرووائیڈر, producing copy that reads code-switched rather than written in Urdu."
  },
  {
    "id": "UR2-003",
    "severity": "MINOR",
    "namespaces": [
      "wire.slash"
    ],
    "category": [
      "accuracy"
    ],
    "examples": [
      {
        "key": "wire.slash.fork",
        "current": "شاخ؛ --switch یہیں، --window باہر؛ <message> آغاز کرے",
        "proposed": "اس گفتگو کی شاخ بنائیں؛ --switch یہیں، --window کہیں اور؛ <message> سے کام شروع کریں"
      }
    ],
    "reason": "The terse Urdu drops both 'this chat' and the explicit 'starts work' relationship, making the command description less precise than the English."
  }
]
```

## Rendered/fit follow-ups

```json
[]
```

No native TUI, relay, desktop or rendered RTL/design completion is claimed.
