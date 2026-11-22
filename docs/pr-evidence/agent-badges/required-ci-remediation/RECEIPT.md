# PR4439 required-CI remediation — private coder receipt

2026-11-22. Fix lane, bounded 20-minute allocation; finished within allocation. Actual served model switched by host from anthropic/claude-opus-5-5 to **openai/gpt-6.1-sol** for provider quota failure. No model/default/tool changes made.

Worktree: `/Users/damian/local-operator/.worktrees/i18n-agent-facts`
Base / still-unmodified upstream ref: `e12cb57872faf05f74fcef66b926d286983ce32f`
Private commit: `3cb0e8c1f91cc93c9db18b07f423a9318c6e7401`
Conventional subject: `fix(i18n): repair agent facts required CI contracts`
Status after commit: clean. Not pushed; no fetch/rebase/new worktree/install/PR comments/merge/release.

## Source and scope

Primary failure source personally read: `/Users/damian/.local-operator/sessions/4109c3d80dc1/scratchpad/localization-inventory/pr4439-ci-qualification-ledger.md`, structured companion `.json`, and `pr4439-ci-source.json`.
Hosted failures (source attribution, not newly executed local typing): run45006983925, type job137160744133: ten errors; guards137160744178: thirteen namespace assertions plus two list.remove call sites in one guard assertion. Hosted strict job137160744123 already passed on **old** e12 head only, per primary ledger. New-head CI/type/strict qualification remains pending manager publication.

B1: test_agent_facts.py retains runtime FrozenInstanceError assertion using setattr; narrows effort result with `assert isinstance(..., Msg)` before unchanged params check; asserts optional settings view non-None before all eight dereferences. No Any/cast/ignore/suppression.
B2: test_catalogues.py exact expected set adds only wire.agent_badges, with admission rationale. Existing equality and all twelve mutation controls unchanged.
B3: settings_view.py projects out exactly the first matching scope/marker by first-index slicing. No guard changes; no filesystem effects. Added duplicate-pair regression explicitly checks projection inputs/order as well as painted output and immutable source tuple. A remove-all mutant fails the new projection assertion even though truncation can hide the duplicate difference in painted output.

`source-proof.json` binds exact head/hashes: only these three files differ. All other tracked paths (copy, history, data, machine IDs, adapters, version, locks, metadata, fixtures and guard source) unchanged. Settings module AST outside `_paint_facts_lines` identical; namespace test diff is exactly one entry plus comment; all namespace mutation source unchanged.

## Preflight and execution

Initial HEAD clean and matched upstream e12; df available19 GiB (postcommit20). Used primary existing `.venv/bin/python` with PYTHONPATH bound to worktree, `env -i`, isolated HOME/config under this session scratch; TERM=xterm-256color, -n0. No inherited CMUX variables. No npm/npx/pnpm/venv/formatter installs or pyright wrapper invocation. No native/browser/window/server dispatch. Local typing intentionally UNRUN: known implicit npm bootstrap prohibited, actual CI diagnostics directly repaired; fresh hosted type job required.

All pytest paths run through existing `scripts/run_bounded.py --contract=command-status` (300/540/120-second bounds), and actual wrapper rc read. Standard matrix uses exact twelve selectors in existing `docs/pr-evidence/agent-badges/convergence/targeted-command.txt` (full agent-facts file plus eleven caller selectors), adding exact namespace test, all twelve mutation cases, and destructive-call assertion.

- `before.log`: two exact pre-fix nodes (`test_shipped_wire_namespaces_exist`, `test_no_session_directory_removal_outside_cleanup`) **2 failed in13.87s**, rc1. Failures name extra wire.agent_badges and two head.remove sites536/540.
- `after.log`: prior55-target matrix + new duplicate test + thirteen namespace nodes + main safety guard: **70 passed,5 warnings in19.98s**, rc0. Warnings are Pydantic deprecations. Original55 tests/assertions retained.
- Strengthened duplicate assertion to inspect actual projection arguments after recognizing truncation can hide duplicate changes; `duplicate-final.log`: exact final regression **1 passed,2 warnings in1.30s**, rc0. Rest of final delta unchanged from70-test pass.
- `duplicate-negative.log`: in-memory remove-all mutant via `duplicate_negative.py` (no worktree modification), exact final regression **1 failed,2 warnings in1.12s**, expected rc1 at `painted_facts` assertion. Confirms control discriminates first-only removal from remove-all.
- `safety-neighbors.log`: stale allowlist assertion, every-remover-shape controls, string/container exclusions: **18 passed in12.37s**, rc0. No whole directory/suite run.
- `/Users/damian/local-operator/.venv/bin/python -m flake8 local_operator/tui/widgets/settings_view.py tests/unit/tui/test_agent_facts.py tests/unit/i18n/test_catalogues.py`: rc0, no output, repeated after final assertion change.
- `git diff --check`: rc0 on this repair delta, including after final assertion change. No modifications to historic SVG/log whitespace.
- black/isort UNRUN: import availability probe found both absent; no install/bootstrap. Flake8 installed and executed.

## Presentation retention proof

`projection_parity.py` executes the actual prior function AST extracted by git show e12 and current source against identical production paint helpers; imports resolved to worktree. `projection-parity.json`: **3696 matching projections**, eight admitted locales, both kinds, actual catalogues + synthetic expanded kind + synthetic expansion of all facts, 11 widths including27, reordered optional technical tails, duplicate scope/marker pairs, opaque model/scope collisions and legacy strings. Compares complete rendered fragments, style strings and cell geometry; exact equality implies byte-identical serialized fragments. Shared result SHA256 `f7c24b47cee51e3f4e5055e3d260156e0c5a594337706bd3b28b34ba811762ab`. Current settings source hash matches committed source-proof.

This is real projection parity, NOT a new native/font/shaping certificate or global-English clearance. No changed output observed; R1 design retention can be assessed by manager using this proof. Real executable delta still needs independent R2 code/QA. No new rendered frames or translation admission changes.

## Cleanup qualification

Every wrapper reports command-status **cleanup=unknown** (contract never certifies group/descendant cleanup), preserved as UNKNOWN, not upgraded to proven-empty. `process-census.json` independently records zero live processes matching this scratch/invocation identity (excluding current census/ancestors); not a global descendant proof. No server/browser/dev-process groups started or left pending.

Manager next: publish ONE head; run independent delta code/QA and fresh actual required CI including normal strict/type job. No claim of hosted new-head pass, native coverage, or merge readiness here.
