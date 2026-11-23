# PR 4420 — private candidate preflight; no native verdict

Integration lane: bounded fix/preflight <=25 minutes. Actual integration model: openai/gpt-6.1-sol after host quota fallback from anthropic/claude-opus-5-5. This is mechanical integration, not translation authoring, independent audit or release clearance.

## Immutable inputs and scope

Candidate parent: `8f1e45a3a98ca9f8b7312b6707c36e21b758966d`. Exact candidate SHA and target manifest are in `pr4420-candidate-pin.json` (written after the private commit; owner can commit that receipt in the final audit/generation phase).

Source SHA-256: `e53e4c156c9c079e38672440742a6e973fba0dc469fc95d37542118b56a94434`.
Context SHA-256: `aee21715fca769227b3bda5b500a5e81e51763cf86817f741baff8b1a5ef0ae7`.
Both byte-identical to parent. All seven target files have 277 unique keys, exact source/before order. Before hashes match original R1 target evidence AND each authoring receipt. Changed-key sets and after hashes match the authoring receipts. Hindi four-key intermediate is reconstructed byte-exactly by restoring only state.stale; its immutable receipt hash matches the final fifth-key receipt. Details and per-key actors: `pr4420-integration-preflight.json`.

All seven original R1 evidence files contain 277 per-key results. Their actual verdicts remain unchanged: zh-CN/ru/hi/ur failed; fr/es/vi passed with the findings retained. Do not relabel a minor-bearing passed receipt as failed, nor a failed receipt as passed. New authors own only changed keys; original R1 translator identities remain authoritative for retained keys. Hindi's first four changes were authored by GPT fallback, fifth state.stale by actual Opus according to that author's host report; do not attribute all five to one model. zh-CN authoring receipt has a session/model but no agent-label field; retain that missing field rather than fabricate one.

R2 reviews the WHOLE CURRENT 277-key namespace: carry only prior semantic results with proven byte/source compatibility; review prior FAIL findings, changed target keys, all 12 changed context entries, both changed source ICU fields and direct semantic neighbors. Explicitly distinguish retained R1 rows from freshly reviewed delta rows; never claim 277 newly read translations. For any changed context, a previously unchanged target still requires semantic reconsideration. Authors' reports of reading updated contexts are authoring evidence, not independent R2 approval.

## Auditor count: supported rule, not an invented visibility waiver

`docs/I18N.md:271-277` requires one different independent LLM auditor per locale/namespace and a second independent audit for high-visibility OR safety copy. `docs/design/i18n.md:571-573` describes the default one-translator/one-auditor lane. tui.info is diagnostic copy, not a primary toast; that fact alone does not waive safety review. This namespace includes held-vs-hung uncertainty and kill-ladder authorization/remedies (`tui.info.status.bound_held` context explicitly says held is NOT proof of a hung runtime and does NOT authorize the kill ladder). Therefore plan TWO independent auditors per locale for this safety-bearing namespace. No claim that the schema itself enforces this count: policy does. Each actual auditor must preserve its own identity/verdict/evidence; do not synthesize two approvals from one review or count the translator/integrator as independent auditor. The two-auditor requirement does not mean two fresh full reads of previously approved unchanged prose: both attest current namespace coverage with retained-result provenance and delta scope.

## Current supported receipt and append API

Authoritative executable schema: `scripts/i18n/ledger.py:101-138,440-454`; contributor example `docs/design/i18n.md:588-600`; HOWTO `docs/I18N.md:288-329`.

CLI accepts ONE receipt object, not an `entries` wrapper. Required structural members:

- `locale`, `namespace`: nonempty path-safe components.
- `source_sha256`: actual 64-lowercase-hex English source hash.
- `status`: nonempty string; real passed status only after actual independent verdicts.
- `translator`, `auditor`: objects containing `agent` and `model` (strings or explicit unknown nulls). Dates, actual fallback disclosure, audit result and unresolved blocker/major count are policy provenance. `auditor.findings` means unresolved blocker/major findings, not total minor observations.
- Extensions are supported: record context/target hashes, reviewed commit, per-key authors, retained and reviewed rows, evidence references and second-auditor identity/verdict. Structural acceptance alone never establishes truthful provenance/native eligibility.

`pr4420-r2-receipt-CONTROL.json` is an EXACT CLI-accepted example, dry-run tested. It deliberately says `control-not-a-native-verdict`, has no invented native result or findings count and leaves missing identity/date fields null. It is evidence ONLY: NEVER apply it. Future auditors create their OWN NEW evidence receipt through the evidence tool, replace control identity/status with their ACTUAL verdict, bind to the sealed candidate/source/context/target, and carry per-key original/remediation authors from the manifest. Do not copy integrator identity into an auditor verdict or collapse mixed authors into one new actor. Attach both independent verdicts in explicit provenance extensions/notes; no special new top-level schema is required.

From this worktree, with primary interpreter and worktree PYTHONPATH, strip CMUX_* (env -i is sufficient) and use scratch HOME/config:

```sh
PYTHON="$HOME/local-operator/.venv/bin/python"
$PYTHON scripts/i18n/ledger.py history-sha --locale LOCALE --namespace tui.info
# Save the emitted digest, then use the exact same expected digest:
$PYTHON scripts/i18n/ledger.py append --receipt ACTUAL_RECEIPT.json --expected-history-sha DIGEST --dry-run
# ONLY after actual independent R2 verdicts and owner gate:
$PYTHON scripts/i18n/ledger.py append --receipt ACTUAL_RECEIPT.json --expected-history-sha DIGEST --apply
$PYTHON scripts/i18n/ledger.py export --output SCRATCH_CURRENT_LEDGER.json
```

Re-read history-sha before every append: last FILE row per key wins, not timestamp. No real append happened in this preflight. Never rewrite historical ledger.json, locale shards or immutable marker; never use accept-stranded or legacy overrides to fabricate eligibility.

## Preserving R1 history

All seven R1 canonical rows were extracted without changing original source hash `909a00c7f98b291863ae6508648ede30b3edddf868e5c0d84a23999666e3cdaf`, identity, status, verdict or findings. Hindi/Russian canonical row is nested in `receipt`; other five use top-level canonical fields. ACTUAL append --dry-run returned 0 for each against current empty namespace history. Thus the supported append API CAN preserve their failed/stale-old-source records without pretending they audited the current source. Future owner can append those original canonical rows BEFORE R2 rows, preserving target/context/commit and full evidence references as extensions; each append uses the newly read history digest. This preflight appended nothing. Original full immutable evidence is committed regardless. No provenance must be invented: Spanish translator date is explicitly unknown; schema permits such extension/null date. Four failed rows stay failed; three original passed rows remain old-source passed and are stale against current English, never current clearance.

## Mechanical evidence and limits

`pr4420-integration-preflight.json` contains actual results: every target parsed, ordered argument contracts and select-arm sets matched English, both whole-field missing-duration branches in every locale omit English unknown/unmeasured and differ from the zero-present branch, and missing-argument negative control raised MessageFormatError. Actual formatter source uses `value is not None`, routing only None to unmeasured; zero remains present. These are ICU mechanical controls, NOT native semantics, measured terminal fit, or real App/clipboard proof. Earlier source-remediation receipt provides prior execution claims, not new independent R2 approval. Initial overly strict mechanical comparison counted repeated simple args across locale-specific plural expansions and was corrected to compare ordered unique argument signature plus branching-select/skeleton structure, retaining CLDR freedom; no target was changed.

No pytest/App/browser/native/build/install/new worktree/full-check/generation, push/rebase/reset or CI wait. The manager owns the single old-E9 CI read/classification and the later sealed pushed-head formal gates. Pending audits and generation are NOT passed merely because these controls passed.
