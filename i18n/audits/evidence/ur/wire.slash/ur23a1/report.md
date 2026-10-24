# Canonical evidence assembly — ur / wire.slash / ur23a1

This is an assembly index, not a new audit and not a replacement for the preserved source report/receipt.

- Source audit ID: ur23a1; original receipt: `ur-core.round1.receipt.json`
- Round: 1; source revision: `2189b61739e08b5f219d317ead68b4a5f4cf709f`; current assembly revision: `5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`
- Auditor: translation-auditor (lopdev team); session `5ebbaba9537e`; actual model `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none
- Independent peer record: `i18n/audits/evidence/ur/wire.slash/ur23b1/record.json`
- Current-byte round used for ledger: no — historical evidence retained
- Source namespace verdict: `failed`; status `failed`; unresolved BLOCKER/MAJOR `1`
- Original source bundle: `/Users/damian/.local-operator/sessions/5ebbaba9537e/scratchpad/ur-core-round1` (provenance only; canonical tracked companions are in this directory).

## Source namespace row

```json
{
  "namespace": "wire.slash",
  "key_count": 52,
  "keys_read": 52,
  "en_sha256": "b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202",
  "ur_sha256": "35ebd056ec897ded1cba799498ab2c67ba9fdff4636dcf1ca6a1cc1a2bec421a",
  "context_sha256": "32b3475a5808048a6eeaae462fee0feb55405fca77026e41b9609a516d74a215",
  "hashes_rederived_and_matched": true,
  "translator_session": "7a648afb0f8d",
  "independent_auditor": true,
  "semantic_verdict": "failed",
  "mechanical_verdict": "passed",
  "finding_ids": [
    "UR-R1-005",
    "UR-R1-006"
  ],
  "unresolved_blocker_major": 1,
  "unresolved_minor": 1,
  "unresolved_nit": 0
}
```

## Namespace findings preserved from source findings JSON

```json
[
  {
    "id": "UR-R1-005",
    "severity": "major",
    "namespaces": [
      "wire.slash"
    ],
    "reason": "Common prose terms provider and manager are left in English here but rendered پرووائیڈر and مینیجر in tui.welcome, violating locale-wide glossary consistency.",
    "affected": [
      {
        "key": "wire.slash.login",
        "current": "provider میں لاگ اِن کریں",
        "proposed": "پرووائیڈر میں لاگ اِن کریں"
      },
      {
        "key": "wire.slash.logout",
        "current": "محفوظ provider کریڈینشلز ہٹائیں",
        "proposed": "محفوظ پرووائیڈر کریڈینشلز ہٹائیں"
      },
      {
        "key": "wire.slash.provider",
        "current": "providers اور ان کی لاگ اِن/استعمال کی حالت کی فہرست",
        "proposed": "پرووائیڈرز اور ان کی لاگ اِن/استعمال کی حالت کی فہرست"
      },
      {
        "key": "wire.slash.usage",
        "current": "provider کا usage quota دکھائیں",
        "proposed": "پرووائیڈر کا استعمالی کوٹہ دکھائیں"
      },
      {
        "key": "wire.slash.team",
        "current": "ٹیمیں اور تنظیمی خاکہ دیکھیں؛ manager کو درخواست بھیجیں",
        "proposed": "ٹیمیں اور تنظیمی خاکہ دیکھیں؛ مینیجر کو درخواست بھیجیں"
      }
    ]
  },
  {
    "id": "UR-R1-006",
    "severity": "minor",
    "namespaces": [
      "wire.slash"
    ],
    "reason": "آزمائیں means “test/try out”; دوبارہ کوشش کریں is the established imperative for retry.",
    "affected": [
      {
        "key": "wire.slash.highlights",
        "current": "جواب کے نمایاں حصے منسوخ، دوبارہ آزمائیں یا رخ بدلیں",
        "proposed": "جواب کے نمایاں حصے منسوخ کریں، دوبارہ کوشش کریں یا رخ بدلیں"
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
