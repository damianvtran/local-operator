# NEW FULL-SCOPE proposal — operator a-7072; NO FRESH GO / NO EXECUTION

This supersedes only the admission/lifecycle policy of reviewed source baseline
`e040a0a8fcfa06a02c509c593e9e6bb74f6dce02`. Round 5 was clean for that baseline;
it does NOT approve this expanded Desktop lifecycle or optional pull. The next
independent review MUST cover the WHOLE command/lifecycle/ownership/resource/pull
scope, not merely a delta, and explicitly sign off any optional digest pull.

## Round 6 disposition — NO optional-pull signoff

Full review 401077fb590a at 1163494896d6 found MAJOR R6/R7. The unified fixes,
actual-adapter RED/GREEN tests and immutable Docker CLI authentication-source proof
are in REMEDIATION-R6-R7.md and docker-anonymous-source.json. NOPULL stays the
current disposition; no review or desk execution permission is inferred here.
Cleanup checks now use their own bounded budget, never an expired WORK lease;
monitor/client failures cannot skip known-CID removal and absence verification.
Failures are reported after recovery attempts, not represented as successful
cleanup. CLI config is explicitly anonymous (a non-empty empty-auth record), not
an empty config that can autodetect osxkeychain; CLI auth environment overrides
are excluded. Source/sentinel proof is NOT native credential-helper proof.

## Round 7 root integration repair — pending review, still NOPULL

Review f2df8b75a5f5 marked R6/R7 FIXED and identified MAJOR R8. See
REMEDIATION-R8.md. The supervisor now receives BENCH_REGISTERED_ROOT explicitly,
validated against the native account's canonical registered campaign directory,
not Docker CLI HOME. CLI HOME/auth isolation is unchanged. The actual adapter
launch environment is tested against the emitted supervisor's admission/artifact
read fragment, with invalid/escaping roots rejected. This source-only integration
proof grants no runtime GO or optional-pull approval.

## Operator acknowledgement, not fabricated verification

Operator answer **a-7072**, relayed via desk 5170, verbatim:

> Find a way around this that doesn't require my manual involvement, validate in other ways or skip that validation with acknowledgement but make sure the code analysis and review is thorough

The inaccessible <=4000 MB/2 CPU PRE-start settings verification is explicitly
**SKIPPED WITH OPERATOR ACKNOWLEDGEMENT**, not passed, inferred or populated with a
`vm_caps_verified_before_start` flag. No settings writes or alternative blocked
settings read route is introduced. New mandatory post-start evidence is actual
`docker info` MemTotal <= **4,300,000,000 decimal bytes** and NCPU <= **2**, before
ANY container. Neither a successful probe nor this operator answer authorizes the
resolver without a fresh full-scope review and desk scope/slot GO.

## Present state and artifact receipt

NO FRESH SLOT is granted. The old slot was released; neither 18:15 nor a tentative
18:45 extension is assumed. Parent relayed desk fleet **27,154 MB: HOLD**, not a
fresh eligible reading. Before any future start desk must refresh fleet <=26,000
MB and free >=30 GiB, memory GO, an exclusive Desktop lease and exact slot times.
Daemon remains absent as last relayed; cached carrier availability is UNKNOWN.
No carrier prefetch/pull has occurred.

The separately authorized wheel-only staging HAS occurred: **17,389,450 bytes**,
SHA256 `e18a57b8b352eb7c8feb317b9f6f533c8f0ad9673fd6cf34bbe3cf1a9f6a8984`,
stream and durable read-back both matched. It was not extracted, installed or
executed. HOLD receipt:
`~/benchmark-runs/reconstruction-20261114-source-v1/artifact-receipts/uv-wheel-20261114T170521953178Z.json`;
receipt SHA256 `1bbd6cf7f2611a0875b96219773268935935a132c71ac43abc9a13b252626d60`.
A byte-identical safe public copy is committed as `wheel-staging-receipt.json`.
Staging authorizes NO runtime use. No private credentials were accessed; binary
secret scanning was not performed, and is not represented as passed.

## Only proposed public entry

From this source artifact directory, after a NEW full-scope clean review and
fresh desk admission (not now):

```sh
python3 -B lifecycle_proposal.py --execute \
  --admission ~/benchmark-runs/reconstruction-20261114-source-v1/desk-admission.json
```

Without --execute the real entry point prints PROPOSAL ONLY and performs no
native operations. Imports have no native effects; optimization is rejected.
The reviewed inner resolver stage remains in RESOLVER-COMMAND.md and may run only
under this owner wrapper, never directly to skip lifecycle/RSS qualification.

The future desk record must name: `go=true`, `scope=resolver-only`, explicit new
slot_start_utc/slot_end_utc/valid_until_utc, fresh_memory_go, fleet_mb,
exclusive_desktop_lease, attempt_id, approved docker_raw_path, desktop_executable,
operator_ack_id=a-7072, exact operator_ack_quote above,
prestart_cap_validation=skipped-with-operator-ack. It must NOT contain a fabricated
vm_caps_verified_before_start flag. Full_scope_review must contain kind=full-scope,
clean=true, an independent reviewer, source_commit matching HEAD, and SHA256 of
lifecycle_proposal.py, RESOLVER-COMMAND.md and resolver-pins.json. No real GO record
is generated by this source change; no default slot or self-approval exists.

Default pull permission is FALSE. To permit the ONE carrier pull, the NEW full
review must explicitly set allow_digest_pull=true AND desk must separately set
authorize_digest_pull=true for that exact scope. Old/delta-only review cannot
permit it. No tag/latest/base build or task-image pull is allowed. The only
optional pull is the resolver-pins.json content digest with --platform=linux/amd64;
afterward OS/architecture AND RepoDigests must match. The resolver stage itself
retains --pull=never. A cache miss, absent wheel or wrong SHA is not an install,
network download, arbitrary dependency fallback or implicit pull request.

## Whole lifecycle and ownership, in order

1. Validate scope/current independent full review and file hashes, exact operator
   acknowledgement, fresh slot/lease/fleet/free-space conditions. Preserve a
   before-start Docker.raw **allocated blocks*512** baseline (not sparse st_size)
   and before-start process identities. Reject any pre-existing Desktop/backend
   or a VM already holding the approved raw file; never take over another instance.
2. Launch ONLY the approved exact Desktop executable as an owned child/group.
   Track Popen handle, PID, observed start time and executable identity. Pending
   SIGTERM/SIGINT cannot race the ownership assignment. If launcher identity
   cannot be established, stop the exact owned root child and record cleanup
   uncertainty; no container may follow an unproved start.
3. FIRST readiness API is docker info. Over-cap/invalid MemTotal/NCPU aborts before
   any container and enters owned cleanup. No Desktop/global configuration write or settings read. The explicitly anonymous
   CLI config is written only inside the new owned attempt directory.
4. Attribute exactly ONE actual VM runner: recognized Apple virtualization helper,
   Docker virtualization helper or bundled qemu runner. A familiar name is NOT
   enough. It must be new since pre-start and descend from the still-live exact
   owner, OR be a new helper with an open descriptor for the exact approved
   Docker.raw. Old/unrelated VMs are excluded, not summed. Denied metadata,
   ambiguous/missing association or lost owner identity aborts before a container.
5. Sample that attributed helper's RSS every 10 seconds after qualification,
   including polling during the optional long carrier pull. Abort above the same
   4.3 GB decimal cap. RSS is explicitly NOT footprint, guest allocation or total
   host memory: compressed/shared/kernel/driver/VM overhead may not be represented.
   Sampling and process metadata are not atomic physical-memory guarantees.
6. Continuously monitor allocated raw growth and filesystem free disk; >5 GiB
   growth or free <25 GiB stops owned work. A changed raw-file identity invalidates
   the comparison and aborts. Keep baseline/during/after samples, including cleanup.
7. Check already-cached carrier, optionally perform ONLY the explicitly reviewed
   digest pull, verify digest/platform, verify the staged wheel hash. No installation.
   Refuse foreign auto-restarted/running containers. Execute the existing bounded
   resolver only after all gates; preserve 2 GiB/1 CPU container limits, read-only
   root, tmpfs/file/output fences and independent 840 s PID1 deadline. Whole working
   lease is 840 s with 60 s reserved cleanup inside the 900 s lifecycle envelope.
8. On success, abort, SIGTERM/SIGINT or deadline: stop/reap the exact owned resolver
   group; verify/remove only its recorded owned container through the existing
   supervisor. Quit only processes proven to belong to this startup instance,
   rechecking PID/start/executable identity before each signal. NO bundle-wide
   quit, pkill/name kill, global prune, cache/image deletion or shared daemon quit.
   If ownership changed, unknown Desktop processes or foreign workloads appear,
   automatic shutdown is NOT proven safe: refuse to kill other work and record
   cleanup-unverified, never successful cleanup. Known helper/owned process
   survival or inability to prove association/absence also records failure.

Important source-only limitation: process metadata and XPC parentage differ by
Desktop/macOS version. This source has NOT been exercised natively and does not
claim that attribution or quit will succeed. The fail-closed path may make this
machine inadmissible without an independently approved alternative. Foreign work
joining a session-owned engine is an explicit cleanup conflict, not permission to
stop it. The exclusive desk lease is essential, not a fabricated universal lock. Preflight
also refuses an already-present daemon socket target, rather than accidentally
using another/stale engine. Unknown shutdown association is recorded unverified,
not treated as successful automatic recovery. SIGKILL/power loss of the whole
host wrapper cannot prove Desktop teardown; the independent container deadline
still bounds the inner workload. No native failure-recovery guarantee is claimed.

## Receipts, tests and remaining gates

New attempts use exclusive directories under the registered HOLD root; nothing
unknown is overwritten/deleted. Full lifecycle receipt records the SKIPPED state,
operator quote, source head, selected RSS identity, samples, cleanup failures and
unqualified status. Root artifacts are retained until explicit parent/disk-owner
disposition. An ignore/HOLD marker is not deletion protection.

`LIFECYCLE-RED.txt` is honest new-contract TDD absence at the old baseline, NOT a
native runtime reproduction. Stub tests exercise over-cap memory/CPU, unknown or
ambiguous/unrelated VM attribution, missing/wrong wheel, wrong carrier digest,
default no-pull, delta-only review denial, owned abort/termination/timeout and
cleanup-ownership failure. Existing R4/R5 stub signal/independent-deadline and
optimization tests remain applicable. No Desktop/Docker/container is used in tests.
The first combined run exposed a TEST clock setup race: scaling interpreter
startup could expire the fake lease before the stub PID was ready. The corrected
clock advances beyond the unchanged 900 s lease only after owned PID readiness;
LIFECYCLE-TEST-ATTEMPT1.txt preserves the failure. No production timeout was
increased, bypassed or suppressed. LIFECYCLE-GREEN.txt contains the subsequent
checks and the actual default CLI's no-execution response.

All 26 media inputs remain unacquired/uncleared. Original seven outcomes remain
lost; no paid/model call, freeze, actual resolver execution or runtime qualification
is claimed. Before future execution parent must obtain NEW FULL-SCOPE independent
clean review and an actual fresh desk GO for exactly the reviewed operation.
