# Guardrails settings extraction compatibility

The settings registry renders its guardrails section from the independently
translated `wire.guardrails` child namespace. The older extraction contract in
`test_wire_settings_slice.py`, however, derives a `wire.settings` key from
every registry slot: for example `wire.settings.guardrails_enabled.help`.
These guardrails compatibility slots are new in this PR, not historical keys
used by external clients. Adding them satisfies the registry-derived extraction
contract without exempting the new section or changing its canonical messages.

The CI remediation copies the ten existing, audited guardrails values into
`wire.settings` in **all eight locales**, and adds the English values to the
extraction golden. These new registry-derived compatibility slots reuse exact
audited values; they are not new translations. Runtime registry messages
continue to use the child namespace.
The mapping is:

- `wire.settings.guardrails.title` → `wire.guardrails.section.title`
- `wire.settings.guardrails.description` → `wire.guardrails.section.description`
- `wire.settings.guardrails_<row>.<field>` → `wire.guardrails.<row>.<field>`,
  where rows are `enabled`, `critical_override`, `judge_timeout_s`, `rules_dir`,
  and fields are `label`, `help`.

`test_guardrails_extraction_slots_reuse_audited_bytes_in_every_locale` requires
identical UTF-8 string-value bytes for each pair in each locale. The extraction
script also asserted that every pre-existing parent value stayed identical.
English context metadata is copied from the same audited child keys. No new
wording, translation, locale pin, exception or coverage allowance is introduced.
Future copy edits must update both spellings and pass the normal audit gate;
the equality test prevents these compatibility slots from silently diverging.

## Audit lineage

Before extraction, `wire.settings` English source SHA-256 was
`4cb113ca6fc28e2cb2a6b24d3aa1a7ece1a6dca7292bcf19cb359998af44080d`.
The unchanged `wire.guardrails` English source SHA-256 is
`2ab2689fe217cdee757173027e0ecb93c3ca75c9a9806047fe6a055b118ed664`.
The resulting parent English source hash is
`9b75f7801d6de7b895d021dfc5261cc1c4951cf756269335a464fa4e2581ee36`.

The append-only locale ledger rows carry the existing parent translator/auditor
identities and embed the copied child row, including its second independent
audit, under `extraction_copy_audits`. They explicitly describe a mechanical
provenance carry, **not a newly performed linguistic audit**. Previous source
hashes, target file hashes and the exact key mapping are recorded in each row.
The independent reviewers can verify that all old parent values are unchanged
and all new values equal the already-audited child values. No auditor is
represented as having reviewed new wording or newly translated text.

The guardrails translator and both original independent auditors used the same
Claude model family, as disclosed in PR #3440's i18n audit comment. The prior
parent ledger retains its own provenance separately. `i18n/ledger.json` and the
bundles are regenerated from the authoritative shards; they are not edited by
hand. Normal catalogue, ledger, extraction and language-manifest checks still
apply in full.
