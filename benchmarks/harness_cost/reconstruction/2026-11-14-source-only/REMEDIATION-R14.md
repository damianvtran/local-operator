# R14 — conservative engine history before dispatch and before journal

Base: 0be5e33c4ca55ee45deefae9f93fb79f05aeddd6. Round12 reports R13 FIXED
and normal R12 foreign-work fence fixed; R14 remains the named MAJOR. One
history-ordering commit fixes R14. Review scope: base..new head, retest R14 and
direct R12 fence regressions. The final handoff gives the exact remediation SHA.
No new R13 ownership/dispatch/journal policy is introduced.

## Invariant and change

An unavailable reply cannot erase evidence that the engine was available OR that
a previous dispatch/response outcome was unknown. In-flight is marked BEFORE the
real command adapter is called. query_engine_info is shared by readiness and
cleanup so both entry paths populate actual history rather than relying on
callers to preset booleans.

A zero-exit response latches available immediately and validates its JSON/server
fields before ANY readiness journal I/O. Malformed responses, non-transient
errors, dispatch exceptions, deadline errors and cancellation latch unknown.
BaseException is intentional: a signal-style unwind is not evidence that no
request ran. Neither sticky flag is cleared by a later recognized unavailable
response. In-flight is cleared only after response classification or latching
unknown; a previously pending entry is also conservatively latched unknown.

Only the existing exact recognized never-ready response can leave both history
flags false; it does not reset flags that were previously set. Existing R12
available-engine foreign-work probe and unknown-history termination refusal are
unchanged. Existing R13 in-memory causal ownership, cleanup dispatch and failed
journal treatment are retained. Resource caps, timers and launch configuration
are unchanged. No settings/root/path/cap workaround is introduced.

## Actual-adapter RED / GREEN, no native execution

R14-RED.txt preserves BOTH original reproductions: real NativeOps.info and command
produce successful info then readiness-journal I/O failure with available false;
and a command deadline failure with unknown false. Home prefix is redacted.

The tests DO NOT preset history flags. Only subprocess.Popen is inert; NativeOps
info, command, startup guards, history population and quit paths execute. The
transport asserts in-flight already true when Popen dispatch is reached. Cases:
- successful info then journal failure then disconnect: no termination;
- actual command deadline failure then disconnect: no termination;
- signal-style cancellation, malformed JSON/object/server fields, permission error:
  unknown is sticky through repeated recognized-unavailable replies;
- successful readiness followed by disconnect: available stays sticky;
- cleanup's own query dispatch exception: unknown remains sticky;
- recognized never-ready response followed by journal I/O failure: proven-owned
  cleanup exception still works (not mislabeled unknown/available).

R14-GREEN.txt records 32 targeted tests across test_engine_history,
test_cleanup_fences.CleanupFenceTests and test_causal_startup. Earlier unrelated
regressions retain their last-head evidence rather than rerunning the whole tree.
Static/privacy verification and the proposal-only default command are recorded
in the same GREEN artifact. No app/helper/VM/Docker/network runtime, native binding
probe, install/build or model call occurred. No real process is signaled.

Native budget remains 1/3 consumed. Optional pull remains NO. Original attempt-1
receipts are unchanged. Independent delta review and NEW GO are required before
any attempt 2; no runtime or all-arm qualification is claimed.
