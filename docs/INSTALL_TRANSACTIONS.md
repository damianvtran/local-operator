# Install-root transactions

All cooperating managed-install writers use `install_transaction.transaction`
against the canonical stable install root, **not the session/config profile**.
The permanent `install.lock` inode is never removed or replaced. Opening is
no-follow and directory-descriptor anchored; nonregular, foreign-owned,
multiply-linked, writable-by-others or changed lock/root identities refuse
before metadata writes. Unsafe observer names are not followed either. A bounded
nonblocking POSIX flock and a process-local mutex protect pointer changes,
launcher/shim changes, candidate registration/discard, migration/rollback and
pruning. Nested primitives reuse and validate the actual held descriptor; they
do not open a second lock. Unsupported locking platforms refuse before writes.
This does not exclude older installers or other noncooperating processes.

## Preparation and promotion

1. Check admission under the short transaction; reserve a generation and publish
   an owned attempt record before preparation can be pruned. The record includes
   random attempt identity, exact candidate/root, target/mode/full source identity,
   baseline current, process PID/birth token, phase and preparation deadline.
2. Build outside the root transaction. Real installer children are bounded and
   their exact spawned process groups are reaped on success, failure and
   cancellation, including the production mesh output-capture adapter (capture
   uses a file and bounded diagnostic read, not raw subprocess execution).
   An installer-worker PID/birth identity protects a live child
   even if the updater is hard-killed. Ambiguous spawn/identity state is retained.
3. Outside the lock, verify the staged artifact and applicable signed macOS helper.
   PyPI routes require positive per-target publication metadata, a compatible
   non-yanked wheel, and its SHA256. Download into the owned candidate with a
   time/size bound; verify SHA256 before uv consumes that local wheel and rehash
   before promotion. PEP 610 must name that exact owned artifact. An actual uv
   probe accepted an all-zero URL hash fragment and emitted empty archive_info,
   so neither is treated as integrity proof. Refresh the publication proof after
   preparation, without using the background cache.
4. Acquire the short transaction, check proof freshness, reread current and owned
   candidate identity, and check admission again. An expired proof is refreshed
   outside the lock with a bounded retry count. Latest means positively latest;
   an exact published version is not silently changed when latest advances.
   Older, equal-version-different-build, and unorderable moved-current candidates
   refuse. Explicit snapshot downgrade/migration can act only against the exact
   baseline it was authorized to replace, never a later changed current.
5. Publish marker/pointer/launchers/shim together and retire the owned attempt.
   Release before service convergence and re-engagement. Mesh receipts sample
   current again after those tails and distinguish the promoted version, fresh
   current and executing relay's serving image; a superseded pass never reverts
   the newer pointer. Identical exact targets no-op only with fresh matching
   artifact/origin/signature proof and a completed marked current generation.
   Prune retains registered
   live or unknown candidates even when a finished marker is already present.

## Admission is separate from mutual exclusion

Both preparation and promotion of a MUTABLE (pip/pipx, in-place) install require
no busy session in the configured and locally discoverable session namespaces.
Missing/unreadable/incompletely parsed records fail closed; a wedged busy record
is not idle. There is no force bypass.
This census is **not** an exclusion protocol for a session becoming busy after
it is sampled, nor a promise to discover an unregistered arbitrary profile.
Immutable generations do NOT take this gate at all: a promotion lands in its own
tree and moves a pointer, never rewriting a running runtime's imports, so the
fleet may be busy. Central components update promptly; a running session keeps
running on its own root and rolls onto the new build at its own next idle.

## Observation and recovery

`lop install status` never acquires, creates, clears or refreshes the lock or
attempt records. It distinguishes last-writer metadata/birth identity from OS
lock ownership. Linux `/proc/locks` can identify the actual flock owner. macOS
`lsof` can report an open descriptor with a blank lock field while flock is held;
that is **unknown**, not proof that the inode is unlocked or that metadata owns
it. A dead PID/reused birth identity may allow owned candidate reclaim; failed
identity observation does not. Lock metadata is never authority for entry.

The desktop application's app-managed Python root remains a separate authority
with its existing whole-preparation shlock transaction. It is not the stable
core install root and does not gain core guarantees merely by sharing a UI.
