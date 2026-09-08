"""Where the master key lives at rest, and how the mode bits are enforced.

Design reference: ``docs/design/secret-store.md`` §2.3.

Two tiers were specified. This module ships the DEFAULT one — ``keyfile``: the
master key sits in ``<config>/secrets/master.key``, mode 0600, inside a 0700
directory, and any lop invocation reads it with no prompt. The honest claim for
that mode, repeated here so it cannot drift out of the code: it is *equivalent
to today's plaintext ``.env`` against an attacker who specifically reads the key
file*, and materially better against the opportunistic "grep the disk for
credentials" malware that a bad link actually drops. It is not a vault.

The opt-in ``passphrase`` tier (``lop secret harden``) wraps this key with
scrypt and keeps the unwrapped copy only in the broker's memory. It needs the
broker daemon to hold that memory, so it arrives with the broker; see
:func:`key_mode` for the seam.

Permissions are applied with an explicit ``chmod`` after creation rather than
through ``mkdir(mode=...)``/``os.open(mode=...)`` alone, because both of those
are masked by the process umask. A developer running with ``umask 000`` would
otherwise create a world-readable key and nothing would say so.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from local_operator.paths import config_dir
from local_operator.secrets.crypto import KEY_BYTES, generate_master_key
from local_operator.secrets.errors import InsecurePermissions, SecretStoreError

#: Subdirectory of the config dir holding everything this feature owns.
SECRETS_DIRNAME = "secrets"

DIR_MODE = 0o700
FILE_MODE = 0o600

#: Bits that must be clear on the key file and the database: any permission
#: for group or other. Checked on every open, not only at creation — a store
#: whose mode was loosened after the fact is exactly the case worth catching.
_FORBIDDEN_MODE_BITS = 0o077


def secrets_dir(base: Path | None = None) -> Path:
    """The directory holding the store, the key and the audit log.

    ``base`` overrides the config dir for tests. Resolved on every call rather
    than cached, for the reason :func:`local_operator.paths.config_dir`
    documents: a module constant freezes whatever the first importer saw.
    """
    return (base if base is not None else config_dir()) / SECRETS_DIRNAME


def store_path(base: Path | None = None) -> Path:
    """Path to the SQLite database."""
    return secrets_dir(base) / "store.db"


def key_path(base: Path | None = None) -> Path:
    """Path to the master key file."""
    return secrets_dir(base) / "master.key"


def audit_log_path(base: Path | None = None) -> Path:
    """Path to the mirrored append-only audit log."""
    return secrets_dir(base) / "audit.log"


def key_mode(base: Path | None = None) -> str:
    """Which at-rest tier the store is in: ``keyfile`` or ``passphrase``.

    Always ``keyfile`` today. The seam the broker PR fills: ``harden`` writes a
    scrypt-wrapped ``master.key.wrapped`` beside the plain key and removes the
    plain one, at which point this returns ``passphrase`` and callers route the
    unwrap through the broker instead of reading the file. Reported by
    ``lop secret status`` so the operator can see which tier is live.
    """
    if (secrets_dir(base) / "master.key.wrapped").exists():
        return "passphrase"
    return "keyfile"


def ensure_secrets_dir(base: Path | None = None) -> Path:
    """Create the secrets directory 0700 and return it.

    The ``chmod`` runs even when the directory already existed: an operator who
    once created it by hand, or a umask that widened it at creation, should not
    leave the key sitting in a traversable directory forever.
    """
    directory = secrets_dir(base)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, DIR_MODE)
    return directory


def check_mode(path: Path) -> None:
    """Raise :class:`InsecurePermissions` if ``path`` is group/other accessible.

    This does NOT repair the mode. By the time a key file is world-readable the
    exposure has already happened; silently tightening it would hide from the
    operator that it was ever open. Windows reports POSIX bits that do not mean
    what they do on Unix, so the check is skipped there rather than producing a
    failure nobody can act on.
    """
    if os.name == "nt":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & _FORBIDDEN_MODE_BITS:
        raise InsecurePermissions(
            f"{path} has mode {mode:04o}; it must not be readable by group or others. "
            f"Fix it with: chmod {FILE_MODE:04o} {path}"
        )


def write_private_file(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` such that it is never briefly world-readable.

    ``os.open`` with mode 0600 creates the file already private, and the
    explicit ``chmod`` covers the umask masking that mode and the case where
    the caller replaced an existing file. Writing then chmod-ing would leave a
    window in which another process can open it — small, but this is the one
    file where that window costs everything.

    **``O_TRUNC``, not ``O_EXCL``, and the caller owns the exclusivity.** This
    function is for a path the caller alone names: the per-rotation temporaries
    in :func:`stage_master_key` and :func:`replace_master_key` carry a pid and
    random suffix, so nobody else can be writing them and truncation is the
    correct handling of a leftover from a dead process reusing that pid.

    Do NOT call it on a path several processes may create at once. Its docstring
    used to CLAIM ``O_EXCL`` while the code passed ``O_TRUNC``, and that lie is
    the whole reason a last-writer-wins clobber on ``master.key`` survived a
    sweep that was specifically looking for that shape (review round 3): the
    site was checked against the docstring and read as guarded. For a shared
    final path use :func:`create_private_file`, which really is exclusive.
    """
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, FILE_MODE)


def create_private_file(path: Path, data: bytes) -> bool:
    """Create ``path`` holding ``data``, or return ``False`` if it already exists.

    The exclusive counterpart to :func:`write_private_file`, for the case where
    several processes may try to create the SAME path at once and exactly one
    must win. ``True`` means this caller created it; ``False`` means somebody
    else did and the caller must read theirs rather than proceeding with its
    own content.

    **Why a temporary plus ``os.link`` rather than plain ``O_EXCL``.**
    ``O_EXCL`` alone picks a single winner, but it publishes the path at
    CREATION — the empty file is visible to every other process for as long as
    it takes to write and fsync the payload. Measured on this machine with 12
    concurrent first-use callers, that window is wide enough to hit: losers
    re-read and get ``master.key is 0 bytes; the key file is damaged``, turning
    a lost race into a hard failure. Worse, a crash inside that window leaves a
    PERMANENT 0-byte key file that no later run can get past, because the
    winner-picking is done and nothing will ever create it again.

    Writing a uniquely-named temporary first and linking it into place closes
    both: ``os.link`` is atomic and refuses an existing target with
    ``FileExistsError``, so the path appears only when it is already complete
    and only one caller can make it appear. It is ``os.replace`` with the
    clobber removed, which is precisely the property wanted here — the same
    distinction round 3 established for the staged keys, where atomicity was
    never the missing property and coexistence was.

    ``EXDEV`` cannot arise: the temporary is created in the destination's own
    directory, so both names are always on one filesystem.
    """
    temporary = path.with_name(f"{path.name}.new.{os.getpid()}.{os.urandom(6).hex()}")
    try:
        write_private_file(temporary, data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
    finally:
        # The temporary is this call's alone; removing it is correct on every
        # exit, including the lost race (the link left the target untouched)
        # and an exception mid-write.
        temporary.unlink(missing_ok=True)
    os.chmod(path, FILE_MODE)
    return True


def load_master_key(base: Path | None = None, *, create: bool = False) -> bytes:
    """Read the master key, optionally creating one on first use.

    ``create`` is passed by the write verbs only. A ``get`` against a store
    that does not exist must report that, not quietly initialise an empty one
    and then report the secret missing — those are different problems and the
    operator needs to be able to tell them apart.

    **Creation is a race between equals, and the loser must ADOPT the winner's
    key rather than keep its own.** Concurrent first use is not a corner case
    here: the operator runs ~11 sessions at once and the store is created by
    whichever of them writes a secret first. This used to be an ``exists()``
    check followed by an unguarded write to a fixed path — the same
    last-writer-wins shape as the staged keys, on the master key itself. Every
    caller generated its own key and every caller returned the key it had
    generated, so all but one went on to seal a database under a key that was
    no longer the one on disk. Measured at 12 concurrent callers: 10 of 12 held
    a key that did not match the file, against 0 of 6 serialised.

    :func:`create_private_file` makes exactly one caller the creator; everybody
    else falls through to the ordinary read below and ends up on the winner's
    key, which is what makes concurrent first use converge instead of brick.
    """
    path = key_path(base)
    if not path.exists():
        if key_mode(base) == "passphrase":
            # Unreachable until the broker PR writes the wrapped key; stated
            # here so the failure is a clear message rather than "no such file".
            raise SecretStoreError(
                "This store is hardened with a passphrase, which needs the secret broker. "
                "Run `lop secret unlock` once the broker is available."
            )
        if not create:
            raise SecretStoreError(
                f"No secret store found at {secrets_dir(base)}. "
                "Create one by storing a secret: lop secret set NAME"
            )
        ensure_secrets_dir(base)
        # The return value is deliberately not branched on: winner and loser
        # take the SAME path from here, re-reading the file so both return a
        # key that provably came off disk and passed the checks below. A winner
        # that returned its in-memory copy would be the one caller whose result
        # was never validated.
        create_private_file(path, generate_master_key())

    check_mode(path)
    key = path.read_bytes()
    if len(key) != KEY_BYTES:
        raise SecretStoreError(
            f"{path} is {len(key)} bytes; a master key is {KEY_BYTES}. "
            "The key file is damaged and the store cannot be decrypted with it."
        )
    return key


#: Basename prefix of a staged key. Every staged file is this plus a suffix
#: unique to the rotation that wrote it. Matching by prefix is what lets
#: :func:`resolve_master_key` consider ALL of them, and a legacy file written
#: under the bare prefix is still found.
STAGED_KEY_PREFIX = "master.key.incoming"

#: Prefix of the temporary a staging write lands on before it is renamed into
#: place. Deliberately NOT under :data:`STAGED_KEY_PREFIX`: a reader globbing
#: for staged keys must never observe a half-written file, so the temporary
#: has to be invisible to that glob until the rename makes it complete.
_STAGING_TEMP_PREFIX = "master.stage"


def staged_key_paths(base: Path | None = None) -> list[Path]:
    """Every key staged by a rotation that has not yet installed it.

    A LIST rather than one path, because rotation is concurrent: several
    sessions can each be between their COMMIT and their install at the same
    instant, and each one's staged file is the only on-disk copy of the key
    its committed database needs. See :func:`stage_master_key`.

    **This is a SUPERSET of the rotations in flight, not exactly that set.**
    A rotation that dies between staging and installing leaves its file behind
    forever: :func:`discard_staged_master_key` only removes the caller's OWN
    key, which is load-bearing (removing another's is the loss it exists to
    prevent) and means nothing reaps an abandoned one. The residue is inert for
    correctness — :func:`resolve_master_key` adopts by fingerprint, so a
    leftover is never mistaken for the store's key — but each file is 32 bytes
    of real key material that stays on disk, and the set grows without bound in
    a store whose rotations keep crashing. A sweep is deliberately NOT done
    here: it would need to distinguish "abandoned" from "staged by a rotation
    that has not committed yet", which is exactly the judgement the ownership
    check refuses to make from outside the owning process.

    Sorted for a deterministic scan order. A missing directory yields an empty
    list rather than raising: "nothing staged" is the ordinary answer for a
    store that has never been rotated.
    """
    try:
        return sorted(secrets_dir(base).glob(f"{STAGED_KEY_PREFIX}*"))
    except OSError:
        return []


def stage_master_key(base: Path | None, key: bytes) -> Path:
    """Persist a rotation's new key BEFORE the re-seal transaction commits.

    This is the ordering that makes rotation crash-safe, and getting it wrong
    loses every secret in the store. ``rotate`` re-seals every record under the
    new key; if the process dies after that COMMIT but before the key reaches
    disk, the database is sealed under a key that existed only in the dead
    process's memory and nothing can ever open it again. An ordinary power cut
    during a routine rotate is total, unrecoverable loss.

    So the new key is written here first, alongside the old one. At every
    instant of a rotation, a key that opens the store exists on disk:

    * before staging — database sealed under the old key, ``master.key`` is it;
    * staged, not committed — database still under the old key, ``master.key``
      is still it, and the staged file is inert;
    * committed, not installed — database under the new key, which is the
      staged file; :func:`resolve_master_key` finds it by fingerprint and
      completes the install;
    * installed — ``master.key`` is the new key and the staged file is removed.

    **The staged file's name is unique PER ROTATION, and that is what makes the
    invariant survive concurrency.** ``rotate`` is reachable from every session
    at once, so several rotations are in flight together. With one shared
    filename the invariant above is false: rotator B's staging replaced
    rotator A's, and if A had already COMMITTED, the only on-disk copy of the
    key A's database needs was gone — a power cut there lost every secret in
    the store, which is the exact loss this staging exists to prevent. Making
    the write atomic is NOT sufficient on its own; an atomic clobber is still a
    clobber. Only a name nobody else writes keeps every committed rotation's
    key on disk simultaneously, which is why the recovery path scans all of
    them (:func:`staged_key_paths`) instead of looking at one place.

    The write itself still goes through a temporary and ``os.replace`` so that
    a concurrent reader globbing for staged keys never observes a partially
    written one, and so a crash mid-write leaves no truncated candidate behind.

    Returns the path written, which is the handle the caller passes back to
    :func:`discard_staged_master_key`.
    """
    ensure_secrets_dir(base)
    directory = secrets_dir(base)
    unique = f"{os.getpid()}.{os.urandom(6).hex()}"
    path = directory / f"{STAGED_KEY_PREFIX}.{unique}"
    temporary = directory / f"{_STAGING_TEMP_PREFIX}.{unique}.tmp"
    try:
        write_private_file(temporary, key)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def discard_staged_master_key(base: Path | None, key: bytes) -> None:
    """Remove staged copies of THIS key, and never another rotation's.

    Called when a rotation failed before committing, and again once a key has
    been installed and its staged copy is redundant. Leaving a file would be
    harmless for correctness — :func:`resolve_master_key` only adopts a staged
    key matching the database's recorded fingerprint — but a stray copy of key
    material on disk is worth not keeping around.

    **Ownership is checked by content, and it is load-bearing.** An earlier
    version unlinked a single fixed path unconditionally, which meant a
    rotation that LOST the epoch race deleted the staged key of the rotation
    that WON — and if the winner had committed but not yet installed, that
    delete removed the only on-disk copy of the key the database needed. So a
    staged file is removed only when it still holds the key this caller is
    entitled to remove. Master keys are 32 random bytes, so "holds this key"
    identifies the rotation exactly.

    Comparison is not constant-time on purpose: both sides are key material
    this process already holds in full, so there is no secret for a timing
    side channel to leak.
    """
    for path in staged_key_paths(base):
        try:
            if path.read_bytes() != key:
                # Another rotation's key. Removing it is the loss this check
                # exists to prevent.
                continue
            path.unlink(missing_ok=True)
        except OSError:
            # Raced with the owning rotation removing it, or with a reader.
            # Either way the file is not ours to insist on.
            continue


def replace_master_key(base: Path | None, key: bytes) -> None:
    """Install a new master key, used by ``rotate`` once every record has moved.

    Written through a temporary file and ``os.replace`` so a crash mid-write
    leaves either the old key or the new one, never a truncated file that
    decrypts nothing. The temporary lives in the same 0700 directory, so it is
    never more exposed than the key it replaces.

    The staged copy is removed last, and only the copy holding THIS key: until
    ``os.replace`` lands, the staged file is the only on-disk copy of the key
    the committed database needs, and a concurrent rotation's staged file is
    the same thing for ITS database (see :func:`discard_staged_master_key`).

    The temporary name is unique PER CALL, not fixed and not merely per-pid,
    because this is no longer a single-caller path: ``resolve_master_key``
    completes an interrupted rotation, so several sessions — and several
    threads inside one session — can land here at once. With a shared name one
    caller's ``os.replace`` consumes the file another has just written, which
    surfaced as a spurious ``FileNotFoundError`` from an install that had in
    fact succeeded. Each caller renaming its OWN file makes the concurrent case
    a harmless last-writer-wins between identical keys.
    """
    path = key_path(base)
    temporary = path.with_name(f"{path.name}.new.{os.getpid()}.{os.urandom(6).hex()}")
    try:
        write_private_file(temporary, key)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.chmod(path, FILE_MODE)
    discard_staged_master_key(base, key)
