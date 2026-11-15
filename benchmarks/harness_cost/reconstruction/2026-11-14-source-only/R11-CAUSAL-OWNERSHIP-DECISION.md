# R11 design checkpoint — no implementation or runtime approval

Base: fca0d547c4de3295bbf10907c46f7b9fb133ae02. Native budget consumed: 1/3;
GO is closed. Original attempt-1 files remain immutable in registered HOLD.
This proposal deliberately does NOT label a readiness-only patch a safe fix.

## What the actual evidence can and cannot prove

Our launcher was PID 37838, created at 05:47:49 UTC, after a clean pre-start
Docker absence check. The recovered backend was PID/PGID 37865, with PPID 1;
its group members started 05:47:49–54 UTC. Both original launcher and controller
had exited before recovery. The receipt does NOT contain an observed parent-child
edge from 37838 to 37865, an inherited launcher session/group, or a launch API
response attesting that 37865 was the result of our request.

Before/after absence, matching signatures, exact bundle paths, a narrow start
interval and one candidate group constrain identity; they do NOT establish causal
ownership. Another launch racing inside that interval could satisfy all of them.
Manual recovery was valid under the independent parent/desk exact-set clearance;
that clearance must not be silently generalized into an automatic adoption rule.
Increasing ps polling frequency cannot guarantee capture of a short fork/reparent
edge. We must fail closed if no causal edge/launch ticket was actually observed.

Authoritative receipts, preserved rather than rewritten:
* controller-start.json SHA256
  cdf55419d5ad082e4c043639910330ccd200bc7482d2f3da37fce37a7a4aba4d
* manual-owned-recovery.json SHA256
  da7d024826ca13f5a441b9cf64d7cb3a8de0e9bc3430454df339d80e5ed7d622
* final-outcome.json SHA256
  f77693892e903ea04a40eb54c113d7b3db8108b86c78088cb9e219e472776caf

## Decision: require a causal launch authority, not time-window adoption

Preferred next implementation is a reviewed system LaunchServices broker for the
SIGNED OUTER Docker.app (whose CFBundleExecutable is com.docker.backend), with
explicit new-instance and non-activation configuration. Obtain the returned
NSRunningApplication PID/bundle/executable identity as the launch ticket. Reject
an old PID, wrong signature/bundle, dead ticket, ambiguous process/group or an
unobserved successor; do not adopt a merely new singleton backend.

The broker response must itself be bound to the one exact owned request and
recorded before Docker API work. Descendants and a reparented group may be adopted
only through actually observed lineage from that ticket or an equally strong
platform attestation. Signed new group membership + time alone is not enough.
If the returned root is itself transient and no successor edge is recorded,
STOP and retain uncertainty. Do not guess private Docker flags, launch another
instance, read blocked settings, add helper basenames or consume attempt 2.

Apple public API source checked read-only (no broker/helper executed):
* https://developer.apple.com/documentation/appkit/nsworkspace/openapplication(at:configuration:completionhandler:)
  Documents returning the launched NSRunningApplication in the completion handler.
* https://developer.apple.com/documentation/appkit/nsworkspace/openconfiguration/createsnewapplicationinstance
  Documents explicit new instance versus returning an already running instance.

These docs establish a candidate causal API, NOT a tested Python/JXA bridge or a
guarantee of Docker's backend lifetime. This changes the launch authority beyond
the current direct-Popen contract; its broker binding/receipt need independent
review. Current scope prohibits native helper execution, so bridge/runtime
compatibility is NOT claimed or substituted with mocks.

## Readiness/resource/cleanup contract for that implementation

FIRST Docker API remains info. Poll only the observed transient daemon-not-ready
condition for at most the user-authorized 120 s; never retry bad JSON, auth,
permission, ownership, platform or over-cap responses. The actual stop deadline is
min(readiness start +120 s, whole-work deadline, slot end minus cleanup reserve).
Every sleep and probe must fit remaining time, including command execution.

Before/after probes: enforce swap/raw growth/free space, slot/work deadlines and
causal process identity. When a VM helper appears, apply the unchanged new-helper
lineage/raw-holder rule and 10 s RSS guard; an ambiguous or denied observation is
not proof that no VM exists. A successful info reporting >4.3 GB or >2 CPUs aborts
BEFORE any pull/container. No model or paid calls are involved.

Daemon-unavailable cleanup may signal ONLY the causally proven new owned group
and exact revalidated descendants; it must not depend on a successful Docker ps
when no pull/container was ever admitted. Unknown members/foreign activity mean
unverified cleanup, not a broader name-based quit. Preserve vmnetd/system services.
Existing image/container cleanup and 840/900 s resolver bounds remain unchanged.

Attempt outputs MUST use a new exclusive namespace. The current reviewed code
has linux-amd64-01 literals in the supervisor/cleanup paths; attempt 1 must never
be overwritten or renamed. Namespace support is a direct prerequisite, not a
reason to reuse the original admission or forge a clean trial continuation.

Required source tests: actual adapter + fake launch ticket/transport for delayed
info, over-cap, refused info, ticket exit, observed versus missing handoff edges,
foreign race/unknown PID, timeout/signal cleanup without daemon, slot/resource
failure and namespace exclusivity. Mocks cannot certify the native bridge.

## Current result

No executable source was changed and no native proof is claimed. No second
attempt, app/VM/helper launch, Docker probe, network runtime, install/build,
settings access, secret access or model call occurred. A source-only design
handoff is safer than committing a timestamp-based ownership relaxation or a
readiness-only patch that would leave the same orphan-ownership failure.
