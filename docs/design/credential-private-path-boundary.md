# Credential private-directory and operational-lock boundary

## Why this is a shared repair

Credential permission application already uses consent, journal, lifecycle,
placement and sync document locks. Directory preparation previously walked every
owned ancestor and requested mode 0700, including the configured profile root and
potentially HOME. A GUI-only lock replacement would not fix the same call during
successful existing inheritance application. This repair changes the shared
preparation boundary, not signing, consent scope or a new GUI engine.

This is a foundational shared change requiring independent code/security/QA and
human merge approval. Source design or authored tests are not runtime approval.

## Chosen root and compatible paths

Each public operation/object selects its configured root from the supplied path
or the current `config_dir()`. Relative paths use that call's working directory;
trusted configured aliases are canonicalized. Concrete roots travel through
private preparation and locks; stateful objects retain their selected root for
path/load/save. Separate default-root calls observe a changed configuration, not
a global cached root. Network IDs retain their existing validation. An OS path,
including Windows drive/UNC syntax, is not validated as a network ID.

Existing configured root and ancestor permissions are preserved, even when the
selected root is HOME or a filesystem root. Readers and path constructors create
nothing. A missing selected root leaf can be created privately under an existing
parent; a missing custom parent produces `profile_parent_missing` with a profile
initialization prerequisite. Preparation never recursively creates ancestors
above the selected root.

Only intended private descendants below that root receive existing requested
0700 directory protection. Descendant symlinks, Windows reparse points/junctions,
wrong types, observed ownership violations or identity changes refuse. A trusted
alias selecting the root itself is not blanket-refused. No installation or
relocation of operator authority is performed.

## Platform-qualified directory/lock protection

On POSIX, available no-follow/directory-open flags, descriptor identity checks and
`fchmod` protect the observed private directory object. Missing optional APIs use
lstat/identity-checked pathname permission changes. This compatibility fallback
has a check/use race; it is not descriptor-equivalent confinement.

Windows retains healthy non-reparse roots, drive/UNC path behavior, binary lock
open and the existing `msvcrt` primitive. lstat reparse/type/identity checks and
legacy requested 0700/0600 modes are retained. Windows `chmod` is **not an
owner-only ACL confidentiality guarantee**. Absence of POSIX-only flags alone is
not grounds to reject healthy Windows credential writes. OS access errors and
observed invalid targets refuse; there is no ignore-and-continue or unlocked
fallback.

The lock leaf is validated before the existing lock primitive, opened as the
same stable regular file, and requested mode 0600 where meaningful. It is never
unlinked to release or repair a lock. Existing thread-before-process order,
filenames, wait/deadline behavior and busy refusal remain. The private helper root
keyword is mandatory so an ambient-root change cannot silently choose the lock's
preparation boundary.

## Explicit exclusions

This promise covers **directory preparation and operational-lock permissions in
a stable namespace**. It does not defeat arbitrary malicious same-UID renames or
confine all later payload publication. In particular, existing `network.store`
path-based private staging/reopen/immediate-parent preparation, serialization,
fsync, temporary files and replace remain unchanged. Descriptor preparation does
not turn those later writers into descriptor-relative transactions. Stronger
publication confinement needs a separate allocation, not an inflated claim here.

No signing/key agent, signature domain, permission/grant/lifetime, network role,
consent schema/revision, lifecycle counter/fence, sync cadence or withdrawal
semantics change. The credential GUI consumes this repair separately; this slice
does not complete GUI, native presence, finite-share UI or remote availability.

## Evidence contract

Targeted tests are authored for whole-envelope write guards, public/root/default/
relative/alias/no-create compatibility, retained object roots, missing parent,
observed links/types/identities, actual full inheritance through consent, and
actual two-process stable inode/busy/reacquire behavior. The test envelope permits
unsafe baseline effects inside disposable parents so the original defect can be
observed, but refuses tested OS writes outside the entire fixture before effect.
No original reproduction may target an actual HOME. Signing in these filesystem
controls is the existing software fixture seam, not genuine native presence.

A causal before/after run must use unchanged public entrypoints on the preserved
baseline and repaired source. The root-only fixture seeds identical valid inputs
using the unchanged placement serializer/private JSON input writer, rather than
calling the defective `PlacementDocument.save` during setup. The tested actual
public inheritance apply and consent locks are never replaced. Request, signature
verification and approval run admission remain real; no broker is constructed by
this permission-recording fixture. Root/parent sentinels are set after input
preconditions, not pre-chmodded to 0700 to conceal the defect.

New private-keyword TypeError, import or fixture-setup failure is NOT the required
APPLY reproduction. Its structural `public-apply-phase.json` receipt must show one
public call attempt and an actual persisted connecting/failed/connected run state.
The baseline exception still propagates; a missing receipt or unentered run is a
setup failure, not causal evidence. Fixed source must actually return permission
recorded and verify executed consent. Record original/fixed outputs, affected
fixture modes/bytes and source identity separately.

Windows branch simulation exercises dispatch/checks only. Native Windows
filesystem/file-lock compatibility remains a separate platform gate; POSIX
results cannot prove Windows ACL privacy. Test-local environment settings and
write guards are NOT retroactive parent startup/site/import/bytecode isolation,
do not reset a previously cached tempfile directory, and do not cover spawned
interpreter startup before the target function installs its guard. They are
trusted tested-call-boundary support, never a whole-process sandbox.

Future execution requires a separately accepted existing controller/profile with
`-B`/`PYTHONDONTWRITEBYTECODE`, HOME/config/TMPDIR isolation BEFORE interpreter
startup, verified source winners and actual tempfile-cache paths, and ownership
of the whole child process group including multiprocessing/resource-tracker
auxiliaries and any remaining threads. The two explicit Process cleanup paths do
not prove auxiliary quiescence. No new startup controller is supplied by these
tests. Those admission/setup and native Windows gates remain BLOCKED until
actually qualified. All source-authored controls remain unverified; no source-only
check is reported as QA PASS.
