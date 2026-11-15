# Prospective launch-authority amendment — direct signed outer backend

Parent accepted independent reviewer cf86bb90bf93's recommendation BEFORE any
new native outcome. Native2's original failure remains unchanged; no attempt3
has run. This amendment is part of the one R15 candidate, not a retrospective
claim that the LaunchServices ancestor/siblings were previously owned.

## Exact launch and original settings

Only /Applications/Docker.app/Contents/MacOS/com.docker.backend, the locally signed
OUTER Info.plist CFBundleExecutable (bundle com.docker.docker, version4.43.1,
Apple anchor + Team9BNSXJN65R), is invoked. No arguments, shell, guessed CLI flags,
privileged service, compiler/install or extra broker process. Original bundle
validation/provenance remains desktop-layout-provenance.json. Launch cwd is /,
not the reconstruction checkout. HOME/settings are the operator's ORIGINAL HOME,
not a fresh onboarding profile and not CLI's anonymous Docker HOME. Environment
is a narrow settings/locale/PATH allowlist; agent API keys/harness routing are not
passed or printed. No explicit settings write is performed.

Direct execution DOES NOT inherit NSWorkspace activation/prompt options. UI,
focus and onboarding/prompt behavior are UNQUALIFIED and must be recorded in the
last instrumented native attempt; no headless/no-focus/no-prompt claim or flags
are invented. Raw backend stdout is private HOLD/backend-launch.log, mode0600,
never automatically published as safe source evidence.

## Kernel authority, not ancestor/signature/time inference

subprocess.Popen([exact_backend],start_new_session=True) invokes setsid before
exec. POSIX creates a session containing ONLY that child with SID=initial PGID=PID.
Other sessions cannot join it; new process groups may be created WITHIN it.
The actual returned Popen child/SID token is published under blocked SIGTERM,
SIGINT and SIGALRM before journal I/O or signal restoration. Original child mask
is restored in preexec; Python-threaded or SIGCHLD-auto-reaping controllers are
refused BEFORE spawn. No callback discovery, singleton/group guessing or public
process signature alone grants ownership.

The creator is kept UNREAPED for the entire live-member lifetime. No poll(),
waitpid(WNOHANG), wait or implicit Popen-client polling touches it before teardown.
Its reserved PID prevents SID numeric reuse after a fast fork/root exit. The token
records executable, createdSID/PGID and later observed birth (pending until the
first snapshot). That reservation is only valid while THIS controller retains it;
it is never authority for a later process/manual retry after controller exit.

A single snapshot closure captures process membership, SID, path and microsecond
kernel birth. Relevant row identity is checked around path/SID reads using the
published proc_bsdinfo layout. PID/birth changes fail closed. A held creator's
zombie transition does not destroy authority; it is not confused with a newly
observed LS ancestor. Foreign signed races in OTHER sessions remain unowned.

Same-session children remain owned across different child PGIDs and reparenting.
An escape to another session has no blanket authority: it needs an actually
observed signed parent edge to a still-owned session member; an unobserved or
subsequently disconnected escape stays unverified. No group-wide signal is used.

## Executable qualification versus ownership/cleanup

Private-session membership is unconditional OWNERSHIP of the process we created
and its inherited members, not qualification. Work still requires bounded
canonical signature checks: Docker bundle members use the existing Apple+Docker
Team requirement. Transient system helpers are admitted ONLY inside that already
owned session, under explicit SIP/system paths and Apple signature anchor; a
foreign-session Apple binary is NOT adopted. Unexpected/failed signature aborts
work and is retained as an identity-proof error. It never silently passes.

A failed signature or journal must not leak our created child. Cleanup may retire
only the kernel-proven private session after the unchanged foreign-engine fence,
without pretending failed executable qualification passed. Escapes still need
signed observed-edge proof. Final receipt/exit retains failure/unverified status.
Kernel member identities are included in the receipt even if an earlier journal
write failed. No cleanup error becomes success.

## Bounds and retirement

R15 resource-only guards run during >0.5s signature waits without recursively
observing ownership. Time/work/slot/swap/raw/free/RSS fences stay active. Pending
resource reads keep their own command limits rather than recursively spawning
more reads. Qualified post-proof ownership monitoring resumes normally.

The existing 120s startup envelope includes process creation and proof; no extra
120s is granted after launch. Python documents initial OS process creation itself
may be noninterruptible; no false hard-preemption claim is made. The external
execution owner must retain its bounded controller/resource supervisor and
truthful unresolved/manual-recovery outcome if a creation syscall stalls.
https://docs.python.org/3.14/library/subprocess.html
https://pubs.opengroup.org/onlinepubs/9799919799/functions/setsid.html

The 840s work/900s resolver limits and 60s cleanup reserve remain unchanged.
Available engines ALWAYS pass successful empty ps -q before retirement (R12).
Unknown/previously available info history is sticky (R14). Container recovery,
exact new image cleanup and then own backend retirement ordering is retained.
Each exact PID/birth/path/session is rechecked before SIGTERM; no SIGKILL/global
quit/prune or other-work stop is introduced. Creator wait/reap happens ONLY after
no live owned members remain. Exact known PID absence is then checked without
any further SID adoption/signals; surviving external raw-holding VMs are left
untouched and make shutdown unverified. Cleanup deadline/failure remains explicitly
unverified and requires separately proven recovery; it never permits late SID
adoption after the controller/PID reservation ends.

## Evidence, fresh review and last native gate

R15-KERNEL-BINDING-PROBE.json is a TINY read-only own-process ABI check: PID/SID/
PGID match, 136-byte struct/start offset120, path and microsecond birth. ZERO
launches/reaps; not native attempt3 or app qualification. Immutable header source
metadata is R15-SOURCE-RECEIPTS.jsonl. Tests cover the real half-second command
wait plus private-session topology, fast reparent, child groups, escapes, foreign
races, microsecond PID reuse, protected signal/journal windows, exec failure,
root exit, deadline, R12/R14 fences and final creator reap. All app/PID transports
are inert. No app/helper/VM startup occurred.

NEW independent review must bind six files: lifecycle_proposal.py,
RESOLVER-COMMAND.md, resolver-pins.json, launch_services_broker.py (historical
fixture dependency), owned_session_launch.py and darwin_process_identity.py.
NEW GO must set direct_backend_ui_effects=acknowledged-unqualified-default-backend,
truthfully acknowledging unqualified direct-backend UI effects, and
fresh resource/absence checks; one NEW linux-amd64-03 namespace only. Existing
operator skip acknowledgement remains truthful, no prestart cap flag is invented.
Optional pull is still NO until new explicit signoff. Only ONE instrumented
attempt3of3 may follow review+GO; if it fails, escalate rather than change the
mechanism/retry. No paid trial/all-arm qualification/freeze is authorized.
