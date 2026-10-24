# i18n audit evidence

Canonical record root for independent translation audits. One bundle per
(locale, namespace, audit id):

```
evidence/<locale>/<namespace>/<audit-id>/
  record.json   required — inputs, hashes, auditor provenance, verdict
  audit.log     actual command output (not notes)
  report.md     findings and reasoning
  receipt.json  hashes of the bundle companions
```

`record.json` carries `locale`, `namespace`, `audit_date`, `key_count`,
`auditor` (agent, session, model, independence, actual model source),
`input_hashes` (en, context, target locale), `artifacts`, and
`current_report_verdict`. Bundle files are evidence only: not shipped, and
the ledger row references the record path. Keep audit ids short so the
record path fits the ledger note's 280-character budget.

Urdu core bundles retain the source auditor's byte-identical receipt under
its original filename; `report.md` identifies the original report and all
additional preserved companions. Reproduction Python is retained as `.py.txt`
so immutable evidence is not rediscovered as a second executable source tree.
`source_audit_revision` records the audited revision, while `input_hashes` and
`current_bytes_match_audited_hashes` bind the current-byte ledger assignment.
Historical failed audits remain beside the later passing rounds; neither set
is a native/rendered RTL or release-eligibility claim.
