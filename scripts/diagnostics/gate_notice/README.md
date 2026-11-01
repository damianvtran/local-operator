# Gate lifecycle diagnostic — NEVER MERGE

This is an inert, manual-only carrier, not a product fix or release candidate.
It replaces **only this branch's** existing `ci.yml` entrypoint so opening the
PR cannot launch normal full-suite CI. Do not merge, tag, publish or dispatch
before independent code review and the manager's explicit green light.

## Preflight and identities

The workflow uses Ubuntu/Python **3.12**, one non-editable dependency environment,
and detached source checkouts pinned to:

- current: `318f23af220592c16fcdb66d8faa096b04c7c521`
- base: `b42471d3935cb7e95d64428af27c32626665f4c0`

Setup installs `current[all,dev]` once. Neither source borrows an editable venv.
`calibrate` must complete both observer controls and both source probes before
`run` is eligible. Source probes verify actual imported module `__file__` paths
belong to the selected checkout, record file SHA256s, and reject missing or
changed synchronous hook signatures. The inherited Textual scheduler contract
comes from the single installed dependency environment. Pip freeze and the
observer/controller hashes accompany the artifacts.

```sh
# LIGHT LOCAL CHECK ONLY: existing interpreter, no app/module dispatch/install.
python scripts/diagnostics/gate_notice/run.py calibrate \
  --work "$LOCAL_OPERATOR_SCRATCHPAD/gate-observer-check" \
  --current-root <existing-checkout-at-318f23> \
  --base-root <existing-checkout-at-b42471d> --light-only

# CI preparation, performed by ci.yml after its dependency step:
"$WORK/venv/bin/python" scripts/diagnostics/gate_notice/run.py calibrate --work "$WORK"
# Only a full, unchanged preflight permits this entrypoint:
"$WORK/venv/bin/python" scripts/diagnostics/gate_notice/run.py run --work "$WORK"
```

A light preflight cannot authorize `run`. No Linux-root-cause claim follows from
local syntax/calibration. Setup failure stops; do not broaden or retry blindly.

## Controls and safety

Calibration uses synthetic objects and stubbed underlying OS/network functions:
return/exception identity and task transitions must survive synchronous wrappers;
missing, asynchronous and signature-mismatched hooks are rejected; deleting a
required event must fail validation. Positive loopback/liveness operations reach
the stubs, while prohibited DNS/connect/signals cannot reach them. No actual
foreign PID, network destination or product runtime is used for calibration.

Each product process gets a fresh HOME/config/CWD/TMPDIR, explicit source
PYTHONPATH, and an allowlisted environment without CMUX/LOP/XPC inheritance or
credentials. Network policy is active before conftest/product imports: only
loopback DNS/TCP and sockets beneath the owned isolation root are allowed.
Nonzero `os.kill`/`os.killpg` and real runtime spawning are prohibited; the
existing fixture's spawn-refusal guard remains active. The outer bounded runner
alone may terminate its own test process group on a bound. No real reaper,
browser, native-app window or operator session is exercised.

## Passive trace, not a scheduling experiment

No coroutine is wrapped and no await, sleep, gate answer, callback replacement
or recovery-state mutation is inserted. Synchronous before/after observers record:

- `_maybe_start_gate` trigger caller stacks, old/new bridge task identities,
  pending/answered keys and the actual property/predicate results;
- app/card/source identity, binding tuple and generation, cold/recovery state;
- undelivered report, real `call_later(show)` scheduling and closure source;
- actual `show` current-source predicate and notice creation;
- retirement before/after state and caller/predicate provenance, including source
  token matching. A wrong-source rejection is distinct from later retirement.

Identifiers/text are hashed; bodies, arbitrary locals and exception messages
are excluded. Per-process NDJSON has PID/worker/test/task/sequence/monotonic time,
8KiB-record,12k-event and16MiB bounds. Observer-budget/I/O failure invalidates the
diagnostic without being converted into a product failure. Trace SHA256 and
terminal summaries are mandatory.

## Budget, results and review

Run exactly the real-entry gate **module once per ref**, sequentially, with two
xdist workers and no worker restart/retry. The exact residual test must appear
once in terminal JUnit. The child wall limit is400s, outer per-ref limit420s,
and aggregate job cap20min. Missing summaries, required instruments, source
identity, JUnit or artifacts fail closed. A product failure is retained for the
comparison, not relabeled as setup failure or rerun until green.

Artifacts contain only bounded traces/logs, source/contract/calibration records,
JUnit and dependency metadata—not checkouts or the venv. Original red product CI
and the independent shared UI composition blocker remain unwaived.

After independent review and manager authorization, use the **existing** workflow
ID (a brand-new workflow absent from default main is not dispatchable):

```sh
gh workflow run ci.yml --ref diag/gate-notice-lifecycle -f authorization='TRACE ONLY'
```

This document is an operational diagnostic runbook, not recorded PR evidence.
The historical CI symptom and root remain unclassified until an actual failing
trace establishes the boundary. No fix or merge follows automatically.
