# Agent review remediation — R12/R13, one coordinated source-only commit

Base c375f2dc5e6b66d4a07a6951a8256774832aeb17. Round 11 reviewer
 d7efa8698727 found two MAJOR findings. BOTH are fixed in this one commit;
exact SHA is supplied in the parent handoff. Scope: base..new head; retest R12/R13.
No finding is rejected/deferred. Optional pull remains NO; no native attempt 2,
probe, source-live patch or old GO reuse occurred.

## R12 — responding engine foreign-work fence, regardless of our work phase

quit_broker_owned now classifies the engine with bounded info. A valid available
response ALWAYS requires a successful EMPTY ps -q result before any termination,
even if this attempt never pulled or created a resolver. Over-cap is not authority
to stop foreign auto-restarted containers. Failed ps, malformed/unknown info,
permission errors and lost/unknown availability history fail closed.

The daemon-unavailable exception applies only to the exact observed never-ready
Cannot-connect-to-local-socket error, no previous available/unknown engine history,
and no admitted pull/container. Available history is latched BEFORE verify_caps,
so an over-cap response followed by lost connectivity cannot bypass the fence.
Evidence of cleanup info/ps/state is retained in the final native receipt. This
adds no identity/resource/deadline relaxation or retry/alternative launch path.

RED actual adapter fixture: prework over-cap/available engine terminated without
ANY ps probe. GREEN actual lifecycle now makes info -> cleanup info -> ps and
preserves the fixture foreign container/engine. Matrix covers available empty,
auto-restarted foreign work, failed ps, unknown/malformed engine, never-ready
unavailable, available-then-lost and unknown history. No Docker client executes.

## R13 — in-memory authority before I/O and teardown despite journal failure

Validated signed root map AND owner are established before the causal-root event
write. Descendant proof is likewise retained before event I/O. The dispatcher can
recover authority from the validated causal map if the owner attribute is absent;
this is NOT adoption from a new singleton/time window.

Ownership-event errors are retained in memory. They abort new work, but during
cleanup they do not interrupt collection/retirement of already proven identities.
A termination/journal error is recorded without another fallible journal call
before the exact-PID fallback. The native final receipt includes those errors,
marks cleanup-unverified and raises after writing; no failure becomes success.
Unknown/foreign process identity still blocks all unsafe termination.

RED actual adapter fixture retained the ticket/root map but left owner None when
root event I/O failed. GREEN tests cover callback-ticket journaling, root event,
descendant edge, startup-owner file, lost owner attribute, termination event,
refused graceful termination + broken journal, and the actual final receipt.
The tests execute actual NativeOps and lifecycle dispatch with inert OS/daemon
transports, not merely event-only fakes. No real PID/app is signaled.

## Evidence and retained boundaries

R12-R13-RED.txt preserves original failures (home prefix redacted; terminal
trailing whitespace normalized). R12-R13-GREEN.txt contains final targeted tests
and source/privacy proof. The existing no-launch broker binding proof is retained,
not rerun or misrepresented as native launch/cleanup qualification.
Original attempt-1 HOLD receipts and failure status remain unchanged. Current
native budget remains 1/3 consumed. Readiness <=120 s, resource/slot fences,
causal app identity, no-focus/no-prompt configuration and independent resolver
deadlines are unchanged. There was no app/VM/Docker/helper/network runtime,
install/build/model call. NEW independent review and NEW GO are mandatory.
