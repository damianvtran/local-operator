# R15 proof-monitoring + prospective created-session candidate

Base 6de60b316a35dda02a8970b9990c0b2cd1d3ff77. Native attempt2 failed
before FIRST Docker API, caps, pull/container or resolver. Budget 2/3 consumed.
Original receipts remain unchanged. This is NOT native qualification or a new GO.
The parent-accepted technical choice is implemented below, pending independent
review before the LAST instrumented native attempt.

## Implemented bounded proof phase

Ownership scans and signature proof have explicit depth scopes, restored in
finally. command's real half-second wait performs resource/time guarding during
initial proof, never recursively observe_owned or use a provisional owner. A
completed signature is cached only after successful checked command completion.
Timeout/error retains the creation authority (or the historical OS application
ticket), but never fakes executable qualification. The direct-session path
separates kernel ownership from signature qualification, as amended below. Cleanup may re-acquire proof under
its existing independent budget; unprovable identity stays unverified.

The resource guard checks startup/work/slot/admission deadlines, swap and raw/free
disk even during long signature waits. While proof is pending it reads new named
VM candidates and applies the existing per-helper RSS ceiling as an ABORT-ONLY
safety check, not ownership, an attributed peak, or permission to kill. No sum of
unrelated VMs is taken. Once proof is complete, full ownership/helper monitoring
resumes on command timeouts; monitoring is NOT globally disabled. An already
in-flight sysctl/ps resource query keeps its own command deadline and pure time
checks rather than recursively spawning another copy. Errors are not swallowed.

## Exact regression evidence — real elapsed clocks, inert transports

R15-RED.txt records the original missing resource check during a >0.5s signature
wait. It also replays exact committed baseline NativeOps in memory against the
same actual-adapter slow-proof test: two nested codesign dispatches, not one.
That replay refuses any non-mock broker factory; it never launches an app.
No source tree, runtime state or test expectation was edited to fake the RED.

GREEN uses real time.monotonic and sleep/communicate intervals (>=0.60s for a
0.65s inert signature), actual NativeOps.command, verify_owned_path, observe_owned
and startup_guard. It covers proof both inside/outside startup monitoring,
checks swap/resource guarding while owner is still None, and rejects signature
failure, deadline, swap, slot and emerging-VM RSS failures. Depth flags unwind;
failed signatures are not cached; the causal ticket is retained. All child/PID
teardown in tests is inert. R12/R13/R14 direct regression files are included.

## Actual topology is a SECOND blocker, not repaired by monitoring

OS ticket PID75147 is a CHILD of leader75032; sibling75149 launches the UI/build
subtree. VM75501 is a child of the returned ticket backend. These are the actual
native2 relationships. The exact-topology fixture proves the ticket+VM can be
recorded, while leader/sibling remain unowned and teardown is refused. It is a
failure/blocker test, NOT a topology-success test.

Apple's createsNewApplicationInstance promises a new application instance. It
does NOT promise authority over that instance's ancestors/process group:
https://developer.apple.com/documentation/appkit/nsworkspace/openconfiguration/createsnewapplicationinstance.md
Docker's historical issue documents a backend watchdog parent, but is not a
supported launcher/ownership contract for installed 4.43.1:
https://github.com/docker/for-mac/issues/6451
No public immutable Docker launch implementation establishing ancestor ownership
was identified in the bounded lookup. Newness, signatures, timing and exclusivity
alone cannot change the direction of a process-parent edge.

## Parent-accepted technical choice — implemented, not executed

Independent technical reviewer cf86bb90bf93 recommended directly launching the
signed OUTER CFBundleExecutable com.docker.backend in a NEW POSIX session; parent
accepted. owned_session_launch.py implements that choice. It never adopts the
NSWorkspace-returned process's parent/group. The old exact-topology regression
remains a negative control for the historical NSWorkspace adapter, not the product
launch path. The direct-session positive fixture uses creator75032 with the same
backend/UI/build/VM topology, now legitimately inherited from OUR setsid.

The token is published under protected signals before journal I/O. Creator remains
unreaped (NO poll/wait/WNOHANG) until all live members retire, preventing numeric
SID reuse even after a fast fork/root exit. Kernel microsecond birth/path/SID
capture is checked against public ABI and an actual NO-LAUNCH own-process probe.
Creator birth is explicitly pending until the first snapshot, never fabricated.

See SESSION-AUTHORITY-AMENDMENT.md for authority, escape, error and cleanup
contracts, required new admission fields/hashes and remaining native gates.
Prospective topology readiness is now TRUE ONLY for this implemented private-
session mode. That does NOT provide review approval, runtime GO or qualification.
The historical NSWorkspace mode is refused by real preflight and retained only
for direct regression fixtures. No arbitrary executable/launcher fallback exists.

R15-GREEN.txt records final targeted/static/privacy results. No Docker/app/VM/
helper startup, native probe, install/build/model call or paid work occurred.
Optional pull remains NO. No scope expansion to settings, resource caps, sample,
artifact versions, global process stops, or old receipt rewriting.
