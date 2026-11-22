# PR #4439 — one terminal conflict convergence

Pinned upstream: `a8ecfef75d0245cb20d925c0bde0f94fad4b5973`.
Previous head: `6561e6241461d0c7f4f49111a1ab6dc8456542e6`.
Original source base: `d8d81b6ea98095d982a213cb21513b1fa4742502`.

GitHub initially returned `CONFLICTING` / `DIRTY`. A single ordinary merge of
that fetched main into the existing feature branch preserves all four authored
commits. No rebase, force-push, base chasing, version edit or shared-primary
checkout change is involved. Review/QA/design have not yet run on this head.

## Actual overlap and resolution

The sole overlapping/conflicted path is
`local_operator/tui/widgets/settings_view.py`. The sole conflict is adjacent
imports: retain the feature's `AgentFact`, `AgentFacts`, `agent_fact_record`
import AND main's `catalogues` import. `conflict-resolution.diff` preserves
the actual combined working-tree resolution, not a range-diff inference.
Main's settings-choice localization is independent of the feature's agent
facts producer/painter. No source wording or runtime feature behavior changes.

`source-proof.json` is produced by the accompanying read-only comparison script
(the script writes only its own report). It establishes:

- All four authored commits remain equal in `range-diff.txt`.
- The added/removed line multiset of the original 80-path feature patch is
  identical against the pinned main: **zero authored feature-line delta**.
  Convergence evidence in this new directory is explicitly outside those
  original paths and is not misrepresented as byte-identical source.
- 79 original authored paths remain byte-identical; the remaining settings
  module contains only the independent upstream changes plus the same feature.
- All feature-changed settings function ASTs remain identical to the old head;
  all four upstream-added/changed functions match main's ASTs exactly.
- Main's 605 existing audit-history paths are untouched. The only history
  delta against main is the seven new `wire.agent_badges` suffixes, unchanged
  from the admitted head. Nine raw English/context/target files remain exactly
  at the seven-target freeze. Producer, app adapter, tests and all prior
  receipts/frame artifacts remain unchanged. `pyproject.toml` equals main.

The first per-file subprocess proof exceeded 30 seconds; a partially batched
version exceeded 60 seconds. Neither was reported as success. They were stopped;
the final proof uses bounded batched Git diffs instead of materializing every
historical blob. The successful report is the actual final script output.

## Targeted execution and qualifications

`targeted.log`: **55 passed, 5 existing Pydantic deprecation warnings in 7.05s**.
This is the prior 50-case agent facts/class/listing/shadowing/routed matrix plus
the five upstream settings-choice parameter cases. The run includes the real
styled `OperatorApp.run_test` live-language repaint and settings choice
browse/cancel/commit flows; it is not only source inspection.

The run uses the existing primary Python 3.12 interpreter with `PYTHONPATH`
bound to this worktree (module source path verified), `env -i`, fresh scratch
HOME/config, `LOP_LANG=en`, no inherited CMUX/LOP ambient state, and notification,
desktop, terminal-title and shimmer suppression. No dependencies installed.
The exact selectors are preserved in `targeted-command.txt`.

The first runner invocation refused **before spawning** because Darwin does
not support its `cleanup-required` certification; exact output is preserved
in `runner-refused-before-spawn.log`. This was a runner contract refusal, not
a guardrail denial. The supported rerun explicitly used
`--contract=command-status --timeout 540`; actual child/wrapper status is zero,
and runner cleanup certification remains **UNKNOWN**, not certified.
No browser/dev server/native process was launched by this convergence task.

`generate-check.log`: **201 artifacts byte-identical** on the merged source.
`git diff --check` passes. No strict base-aware checker retry: its two prior
local timeouts remain an explicit qualification gap. Actual hosted strict
CI must pass; neither this document nor the earlier catalogue-only check
substitutes for it. No full suite, native shaping audit, browser/capture,
typechecker, dependency install or baseline/eligibility edit was run.

## Retained visual evidence, not fresh frames

The old product frame archive and receipts remain byte-identical. The producer,
app snapshot adapter, catalogue/context bytes and all feature painter ASTs
remain identical; only the independent upstream settings-choice functions and
imports changed. Therefore the prior agent-badge frame evidence remains
source-compatible. This does not claim a fresh capture or independent design
approval. Native Urdu/Hindi shaping stays **UNVERIFIED**. Existing descriptions,
footer/hosting prose and other queued i18n work remain outside this slice.

Coder: session `cf41cc5a1ab1`, actual served model `openai/gpt-6.1-sol` after
provider-quota fallback from `anthropic/claude-opus-5-5`.
