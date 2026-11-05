# Integrating the independently audited `tui.sound` artifacts

Artifact root: `/Users/damian/.local-operator/sessions/796c1db4d0d0/scratchpad/c4-i18n-audit-r1/`
Pinned source: PR #3433 head `27695a49855f`; source SHA-256 `0d23d23b34918c3bdb003afbf6ba6f2694316a7c55214142428185956e4fa578`.

1. Confirm the target branch's English `tui.sound.json` and `.context.json` still match `inputs/en.tui.sound.source.json` and `inputs/en.tui.sound.context.json` byte for byte. A changed source needs a scoped follow-up audit, not a relabeled hash.
2. Copy the seven `overlay/local_operator/i18n/catalogues/<locale>/tui.sound.json` files into the matching repo paths. These corrected bytes, not the original drafts, are approved. SHA256SUMS.json binds their content.
3. Copy `overlay/i18n/audits/evidence/<locale>/tui.sound/c4r1/` and `overlay/i18n/audits/tui-sound-c4r1/` into the matching paths. Their paths are recorded by the ledger rows. Keep receipts, reports and evidence together. The shared report discloses limitations.
4. Append the matching row in `ledger-rows.proposed.json` to `entries` of each locale's `i18n/ledger/<locale>.json`, preserving every prior row. The overlay's full shard files are reference snapshots only: do not overwrite newer unrelated audit rows. No row should be marked passed for an uncorrected target.
5. Remove only the seven `(locale, namespace=tui.sound)` coverage allowance pairs listed in `coverage-removals.json` (27 key IDs each, 189 total). Do not copy the whole allowance snapshot over concurrent work or add allowances. The independent snapshot's exact language-manifest test passed after this removal.
6. Run `.venv/bin/python scripts/i18n/generate.py` to regenerate the aggregate ledger. `generated/ledger.json` is the scratch-generated reference only, never hand-edit or overwrite the aggregate with it. This audit changed no other generated output.
7. On the actual PR branch run `.venv/bin/python scripts/i18n/generate.py --check`, `.venv/bin/python scripts/i18n/check.py`, and the targeted language-manifest coverage test with `-n 1`, under the repository's bounded resource contract. The scratch checks passed, but a Git archive cannot check allowance growth against the actual PR base.
8. Reply to the PR audit comment with the integration commit SHA and evidence paths. Keep code/QA/design/UX review gates separate: this audit does not waive them or claim real speaker playback, live terminal shaping, full RTL, or native-speaker certification.

Reproducers: `catalogue_probe.py`, `render_probe.py`, `tree/` (sparse pinned snapshot; dependencies not vendored). Final validator logs: `generate-check-final.log`, `check-final.log`, `coverage-test.log`, `catalogue-probe.log`. Real app frames: `frames/`. Only audit receipts, copy and selected evidence live in the overlay; never commit the full `tree/`, `homes/` or executable scratch rigs.
