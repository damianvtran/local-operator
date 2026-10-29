# Scoped independent core Urdu audit — remediation round 3

**Verdict: CLEAN WITH FOLLOW-UPS — terminal** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT.

Audit `UR-R3-AUDIT-001`; delta `d3f5e21b6b01e82ff20faa4dc49f4f9420a5c40a..5eaddeff1d4cb9bfee9c2a831d1e48938c9cce9f`. Read **19 changed keys + 8 named direct neighbors = 27 scoped values** across four namespaces. Core `auditor.findings` = **0**.

## Provenance and independence

- Auditor: translation-auditor (lopdev team); independent scoped round-3 core auditor
- Session: `3e0f499dc2ee`; served model: `openrouter/moonshotai/kimi-k3.1-pro`; fallback: none observed; the served model recorded above performed this audit
- Independent of every translator session; no other new audit receipt was read.
- Current en/ur/context SHA-256 values were independently derived and match translator provenance; English and context are unchanged in this delta.

## Round-2 dispositions

**13/13 verified fixed; 0 unresolved.** IDs are session-qualified because the two round-2 auditors reused the same local finding numbers.

- Settled now covers done/paused/archived as حتمی حالت والے without implying only completed.
- Deny-prefix copy guarantees files are neither listed nor sent to the decision model.
- Operator/background-job, Preview, exit-code, display-rail, subscription, proactive, and connectivity wording all pass in their direct context.
- The credential redaction artifact is gone; the inline instruction reuses the exact established native plural and preserves both safety prohibitions.

## Verification evidence

- `generate.py --check`: **91 artifacts byte-identical**.
- `check.py --catalogues-only`: **0 problems**.
- Scoped key/ICU/technical-token/newline mismatches: **0**.
- NFC failures: 0. Manual bidi controls: 0. Native digits: 0. Hindi dandas: 0. Informal تم: 0. Literal redaction artifacts: 0.
- No scoped value declares `maxCells`, so no explicit-budget pass is claimed.
- `wire.slash.hub` measures **57 cells against the approximate 55-cell `/help` guide** and remains `UR-R3-FIT-001` for rendered review—not a silent pass or a measured rendered failure.

## Namespace verdicts

| Namespace | Changed keys | Direct neighbors | Verdict |
|---|---:|---:|---|
| `wire.errors.settings` | 1 | 1 | clean |
| `wire.remote_compute` | 1 | 0 | clean |
| `wire.settings` | 16 | 7 | clean |
| `wire.slash` | 1 | 0 | clean_with_followups |

## Remaining findings and follow-ups

No BLOCKER, MAJOR, MINOR, or NIT findings remain in scope. The only follow-up is **UR-R3-FIT-001** for approximate `/help` fit on `wire.slash.hub`.

Native RTL/bidi/shaping, layout, fallback, long-text/design, and approximate `/help` rendered evidence remain separate gates and are not claimed.

No source, catalogue, ledger, generated `shipped.json`, eligibility, hold, or PR was edited. Unrelated values and unrelated plural/fit suites were not re-audited.

## Evidence bundle

The portable sibling bundle includes the complete 27-value `.pairs` workbook, current hashes, all 13 disposition verifications, scoped byte/fit checks, repository logs, this report, and the JSON receipt. `evidence-manifest.json` records payload SHA-256 hashes.
