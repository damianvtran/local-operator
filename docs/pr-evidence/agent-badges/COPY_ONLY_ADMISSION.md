# COPY ONLY — agent badge native-copy admission

This records native-copy semantic admission for the complete one-key
`wire.agent_badges` namespace in `zh-CN`, `fr`, `es`, `ru`, `vi`, `hi`, `ur`.
It is not runtime eligibility, rendered-fit, design/QA, pseudo-localization or
native Urdu/Hindi shaping clearance. ONE shared 27-cell facts budget remains
conditional on whole proactive-marker/scope spill in the private implementation.

## Frozen inputs and independent actors

- Copy freeze: `7d03eb5874ddbf8d3d16aeed8a12d7d204d68603`.
- English source/context commit: `741a45d8465c5f4608e374a1bffeb9d2190065cf`.
- Author actor `8a63c771f926`: actual `openai/gpt-6.1-sol` quota fallback.
- Independent A actor `e1e6b5514389`: actual `openai/gpt-6.1-sol` quota fallback.
- Independent B actor `aa681ae7e68e`: actual `openai/gpt-6.1-sol` quota fallback.
- Same model, distinct actors; no model-independent or native-speaker claim.
- Each auditor freshly reviewed one key in each of seven locales; no retained
  keys, no findings at any severity. Raw receipts preserve their original scope
  limitations and private-not-admission qualifications. Root separately authorized
  this supported ledger admission after reading both audit reports.

## Byte-exact evidence

- A: `agent-badges-audit-A.json`, SHA-256
  `2e3bf1d02ac8b2956142c78534d1f744594671f2696dcee9a2c8be733ccdc141`.
- B: `agent-badges-audit-B.json`, SHA-256
  `cbc6ae3ae59b49bd19e0042990a59a85e57d378ed1fda9ebff44cc7c6b6886b6`.
- `inputs/manifest.md` and all nine `*.reader.json` are archived unchanged.
  Returned Reader text hashes are distinct from raw file and canonical JSON
  hashes; payloads were checked against the frozen committed blobs.
- English raw/source SHA-256:
  `5caa688fa6d6a78f315e32f7bb1db3b8cae08c77f727dd36af22e45a0206a23b`.
- English canonical-JSON SHA-256:
  `db915e9d40dd3e036e2c7401e6af74c7bf3cdfc97e40e1305ec6bb039b1308f9`.
- Context raw SHA-256:
  `8ab96370595b180339f4a6392daaef10a6f8324f202f3738dfc97574df5591e7`.
- Context canonical-JSON SHA-256:
  `3cd7f264b7726aff1d0cc09f2fff7343314ce7447b74c10ae0406a14f85c049c`.
- All seven target raw/canonical hashes match the author and B receipts.

## Supported write path and actual results

`docs/I18N.md` lines 288–308 prescribe receipt normalization followed by the
optimistic `scripts/i18n/ledger.py append` helper. `admission/<locale>.json`
contains its documented receipt-object shape, using B as primary auditor and
A as the separately identified second auditor with immutable evidence paths.
No original receipt was reauthored and no suffix was hand-written.

For each of the seven exact locales:

```sh
python scripts/i18n/ledger.py history-sha --locale <locale> --namespace wire.agent_badges
python scripts/i18n/ledger.py append --receipt docs/pr-evidence/agent-badges/admission/<locale>.json --expected-history-sha <returned-sha> --dry-run
python scripts/i18n/ledger.py append --receipt docs/pr-evidence/agent-badges/admission/<locale>.json --expected-history-sha <returned-sha> --apply
python scripts/i18n/ledger.py validate --base 7d03eb5874ddbf8d3d16aeed8a12d7d204d68603
```

Actual output is preserved in `admission-cli.log`: seven dry runs, seven applied
new namespace suffix paths, then `audit history: valid and append-only`.
A namespace-scoped check using supported `load_entries(checkout_root)` found
exactly seven passed/fresh records, each matching the current raw English and
target hashes, one freshly reviewed key and zero retained keys. Both actual
reviewer models/actors remain recorded. The source qualification printed the
`i18n-agent-facts/local_operator/__init__.py` worktree path from the primary
checkout's Python interpreter, with `PYTHONPATH` selecting this worktree.

## Deliberately stopped before generation

No generator was run in this metadata-only allocation: it writes generated
paths already carrying paused private implementation changes, including
`local_operator/i18n/keys/__init__.py`,
`local_operator/i18n/keys/wire_agent_badges.py`, and
`local_operator/mobile/web/src/i18n/keys.gen.ts`. All eight paused product,
test and generated files stayed byte-identical. The normal product owner must
run `generate.py`, `generate.py --check` and the normal base-aware i18n check
once that source slot resumes. No all-seven runtime eligibility claim is made.
No tests, typechecker, build, browser/native dispatch, installs, PR, push,
rebase, fetch, baseline, global holds or immutable ledger snapshots changed.

Integrator: coder `05ab22c0ed05`, actual `anthropic/claude-opus-5-5` with no
fallback in this allocation. This normalization is not a self-audit.
