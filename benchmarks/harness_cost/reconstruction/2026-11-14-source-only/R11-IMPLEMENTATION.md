# R11 implemented candidate — no-launch binding proof, NOT native qualification

Base: afcef5f2a756e2b59b7e36f68d72de3a186faf0b (design checkpoint). The
executable baseline is fca0d547c4de. One unified candidate follows the parent's
explicit causal-broker technical choice and authorized NO-LAUNCH binding probe.
Native budget is still 1/3 consumed. No app/VM/Docker launch or retry occurred.
The old design note remains historical, not the completion claim for this patch.

## Thin system binding, with causal authority

launch_services_broker.py uses only ctypes and existing macOS AppKit/Foundation/
Objective-C/Blocks runtime. No installation, compilation, privileged service,
AppleEvents or SystemEvents automation. The default CLI ONLY accepts --probe.
Native execution is reachable only through the reviewed lifecycle and fresh GO.

The broker invokes NSWorkspace openApplicationAtURL:configuration:completionHandler:
for the signed OUTER Docker.app. New instance is explicitly requested; activation,
prompts and running-app substitution are disabled. The exact returned retained
NSRunningApplication supplies PID, bundle/executable identity and launch date.
The API object is compared using isEqual against the current running-app object,
not PID alone. Old/foreign/invalid tickets and duplicate launch requests fail.
No singleton/time-window search or unrelated VM adoption is used.

Validated authority is retained even if journaling fails, preventing a known app
from being lost to cleanup. The callback journal is durable before successful
startup returns. A signal/parent unwind drains late completion during the existing
cleanup reserve. If the OS NEVER returns authority, the code explicitly records
UNRESOLVED and fails: it cannot safely kill a guessed process. This failure branch
requires escalation/manual recovery and blocks any automatic next attempt; it is
NOT represented as proof that no app launched or that cleanup succeeded.

## Identity, handoff and cleanup

Snapshot PID/PPID/PGID/start/executable identities are retained. Only signed,
canonical bundle-path descendants ACTUALLY observed under a known live identity
become owned. A reparented successor needs that recorded edge and a unique proven
live backend group leader; timing/newness alone is never sufficient. Unknown or
reused PIDs, foreign races and unowned new VM helpers fail closed. Existing helper
basenames are unchanged; new VM qualification additionally requires causal owned
lineage, not merely an approved raw-file descriptor.

No-pull/no-container startup cleanup does NOT require daemon ps availability.
It revalidates the causal root/observed descendants, requests termination through
the retained application, then signals only exact known identities as necessary.
Foreign/unobserved Docker processes block unsafe shutdown. Once work was admitted,
the prior daemon/workload fence remains. Refused graceful termination remains
unverified even if exact-PID recovery later proves absence. System vmnetd and
unowned processes are never targeted. All operations share the existing cleanup
budget; no force/global quit, retry or settings workaround is introduced.

## Readiness and fresh attempts

FIRST Docker API remains info. Only the actual Cannot-connect-to-local-daemon
response is polled, inside one <=120 s launch/readiness envelope bounded also by
whole-work and slot cleanup reserves. Every command and sleep fits remaining time.
Swap/raw/free/process ownership and emerging attributable VM RSS are checked
throughout; the 10 s RSS cadence remains. Successful over-cap info aborts before
any pull/container; malformed/refused/permission responses do not retry.
Independent resolver deadlines remain 840/900 s. Existing resource caps are not
relaxed or claimed historically verified.

Fresh admissions/<attempt_id>.json, lifecycle/<attempt_id> and resolver/<attempt_id>
paths are canonical/owned, validated and exclusive. Native entry and supervisor
carry the SAME explicit admission path; no fixed original desk-admission.json is
read or overwritten. The old linux-amd64-01 records and admission remain immutable.
New review file hashes MUST include launch_services_broker.py in addition to the
three original files. New review and NEW GO are required before attempt 2.

## Actual authorized NO-LAUNCH probe and targeted tests

R11-BINDING-PROBE.json records real class/selector availability for the open API
and all required configuration setters. It validates pid_t extraction and copied
native block callback ABI against an NSObject fixture containing this probe's own
PID, using NSSet enumeration. Launch/activation/quit counts are ZERO. Unbundled
Python's NSRunningApplication currentApplication returns -1, which Apple documents
as valid for an application without a PID; the fixture is explicitly identified,
not mislabeled an actual launched app. No UI/app registration was forced.

Public contracts consulted:
https://developer.apple.com/documentation/appkit/nsworkspace/openapplication(at:configuration:completionhandler:)
https://developer.apple.com/documentation/appkit/nsworkspace/openconfiguration/createsnewapplicationinstance
https://developer.apple.com/documentation/appkit/nsrunningapplication/processidentifier.md

The NO-LAUNCH check is binding evidence, not proof of a real Workspace launch
completion, Docker readiness, caps, native helper attribution or resource cleanup.
R11-IMPLEMENTATION-TESTS.txt records actual adapter + inert driver tests for causal
handoff/foreign races, slow/refused/over-cap info, 120 s/resource/slot/work bounds,
callback timeout/late capture, never-returned authority, signal unwind, journal
failure, daemon-absent cleanup and fresh namespaces, with retained direct safety
regressions. No native attempt was consumed by those checks.

Original attempt-1 receipts and failure status remain unchanged. The safe public
incident copy from the design checkpoint is retained. No host environment install,
build, model call, paid trial, all-arm qualification or freeze is claimed.
