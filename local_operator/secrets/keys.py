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

#: File holding the registration ticket: the secret a process must present to
#: claim it is a lop session (design §2.1, PR-2 review R1). See
#: :func:`registration_ticket` for why this is a file rather than a property of
#: the connecting process.
TICKET_FILENAME = "register.ticket"

#: Ticket length. 32 bytes from the OS CSPRNG — the same budget as the master
#: key, because a guessable ticket is a self-registration bypass.
TICKET_BYTES = 32


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


def wrapped_key_path(base: Path | None = None) -> Path:
    """Path to the scrypt-wrapped master key used by ``passphrase`` mode."""
    return secrets_dir(base) / "master.key.wrapped"


def key_mode(base: Path | None = None) -> str:
    """Which at-rest tier the store is in: ``keyfile`` or ``passphrase``.

    ``harden`` writes a scrypt-wrapped ``master.key.wrapped`` and removes the
    plain key, at which point this returns ``passphrase`` and callers route the
    unwrap through the broker — which is the only process that ever holds the
    unwrapped copy. Reported by ``lop secret status`` so the operator can see
    which tier is live.

    **A plaintext ``master.key`` means ``keyfile``, whatever else is on disk,
    and that precedence is the whole point of this function (QA Q10).** It used
    to answer purely on the presence of ``master.key.wrapped``, so a store
    carrying BOTH files reported ``passphrase`` while the unwrapped key sat in
    the clear beside the database — the store advertised the one guarantee the
    hardened tier is sold on (§2.3: nothing on disk decrypts the store on its
    own) while not providing it. A tier is defined by what is absent from disk,
    so it has to be decided by looking for that file rather than for the other
    one.

    Answering ``keyfile`` there is also what makes the damage REPAIRABLE from
    the CLI. ``harden`` refuses a store already in ``passphrase`` mode, so the
    old answer left an operator whose rotation had written a plaintext key with
    no way to re-wrap it; with the honest answer, ``harden`` sees the tier it
    can act on and re-wraps the live key. :func:`key_of_record_inconsistency`
    names the state so ``status`` can point at it rather than leaving the
    operator to notice a mode line that silently changed.
    """
    if wrapped_key_path(base).exists() and not key_path(base).exists():
        return "passphrase"
    return "keyfile"


def key_of_record_inconsistency(base: Path | None = None) -> str | None:
    """Describe a store whose key files contradict each other, or ``None``.

    There is exactly one such state, and it is the one QA Q10 produced: both a
    plaintext ``master.key`` and a ``master.key.wrapped`` on disk. It means a
    hardening was undone — by a rotation that was not tier-aware, or by a crash
    inside :func:`wrap_master_key` between writing the wrapped file and
    removing the plain one. Either way the operator believes their master key
    is only on disk scrypt-wrapped and it is not.

    Reported rather than repaired here. Which of the two files holds the key
    the DATABASE is sealed under cannot be decided without opening the database
    (and, for the wrapped one, without the passphrase), so the repair belongs
    to ``harden``, which has both. This function exists so ``status`` can say
    the words out loud instead of printing a tier line that is now false.
    """
    if key_path(base).exists() and wrapped_key_path(base).exists():
        return (
            f"a plaintext master key is present at {key_path(base)} alongside the "
            f"scrypt-wrapped one, so this store is NOT protected at rest the way "
            f"passphrase mode claims. Re-run `lop secret harden` to re-wrap the live "
            f"key and remove the plaintext copy."
        )
    return None


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


def ticket_path(base: Path | None = None) -> Path:
    """Path to the registration ticket."""
    return secrets_dir(base) / TICKET_FILENAME


def registration_ticket(base: Path | None = None) -> bytes:
    """The secret a caller must present to register itself as a lop session.

    **Why a file secret and not a property of the process.** Review finding R1
    was that ``register`` was unauthenticated, and the obvious repairs are all
    unsound against the actual threat model. Measured on this machine rather
    than assumed: the attacker shape §8 cares about is a double-forked
    ``setsid`` script, and a same-uid process can forge *every* signal the
    kernel exposes about itself — it reparents to launchd (``ppid=1``, which
    is also what a real detached session has), becomes a session leader, and
    can even allocate its own pty via ``pty.fork()`` so a controlling-tty check
    passes. There is therefore NO process-shape predicate that separates a
    genuine session from the malware, and code-signature verification is
    unavailable here (``csops`` denied, §2.1). Requiring the registrant to *be*
    an ancestor of itself is what created the bypass in the first place.

    What remains is a secret the attacker cannot read. This file is 0600 inside
    the 0700 secrets directory, so it is exactly as reachable as the master key
    — which is the point, and why this is honest rather than circular:

    - In ``keyfile`` mode the master key sits beside it, so an attacker who can
      read the ticket could have read the key directly and skipped the broker
      entirely. The ticket costs that attacker nothing they did not already
      have, and §8 now says so plainly instead of claiming the socket stops
      them.
    - In ``passphrase`` mode there is no unwrapped key on disk. The ticket is
      the ONLY thing on disk, and holding it grants standing to *ask* — it does
      not unwrap anything. An attacker with the ticket still needs the broker
      to be unlocked, and unlocking needs the passphrase, which is never on
      disk in any form. That is the tier the design nominates as load-bearing,
      and it is the tier where this check does real work.

    Created lazily and never rotated in place: a rotation would deregister
    every live session on a machine the operator runs ~10 of them on, and the
    ticket protects standing to ask rather than the secrets themselves.
    """
    path = ticket_path(base)
    if path.exists():
        check_mode(path)
        ticket = path.read_bytes()
        if len(ticket) == TICKET_BYTES:
            return ticket
        # A truncated or padded ticket is damage, not an attack signal: rewrite
        # it rather than bricking registration for every session on the host.
    ensure_secrets_dir(base)
    ticket = os.urandom(TICKET_BYTES)
    write_private_file(path, ticket)
    return ticket


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
            # In this tier the unwrapped key exists ONLY in broker memory, so
            # there is deliberately nothing on disk this function could return.
            raise SecretStoreError(
                "This store is hardened with a passphrase, which needs the secret broker. "
                "Run `lop secret unlock` once to unlock it for this boot."
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

#: Prefix of a staged WRAPPED key, written by a rotation of a hardened store.
#: Deliberately not under :data:`STAGED_KEY_PREFIX`, and checked as disjoint by
#: `test_staged_key_globs_do_not_overlap`: the two staging sets hold different
#: FORMATS (32 raw key bytes against a scrypt-wrapped blob), so a glob that
#: collected both would hand `resolve_master_key` a wrapped blob to fingerprint
#: as though it were a key.
STAGED_WRAPPED_PREFIX = "master.wrapped.incoming"


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


# --- the opt-in passphrase tier (design §2.3) --------------------------------

#: scrypt parameters. n=2^15 measured at 182 ms / 32 MiB on this machine
#: (design §2.3, spike 8) — unnoticeable once per boot, and a meaningful
#: brute-force cost against a stolen wrapped key. n=2^14 (92 ms) is too cheap
#: and n=2^17 (841 ms) is a noticeable stall for no proportionate gain.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_SALT_BYTES = 16

#: Magic prefix and version of the wrapped-key file, so a file from a future
#: format is refused with a message rather than fed to the KDF as if it were
#: this one.
_WRAP_MAGIC = b"lopsecwrap\x00"
_WRAP_VERSION = 1


def _derive_wrapping_key(passphrase: str, salt: bytes) -> bytes:
    """scrypt over the operator's passphrase. Imported lazily, as everywhere."""
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    return Scrypt(salt=salt, length=KEY_BYTES, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P).derive(
        passphrase.encode("utf-8")
    )


def wrap_master_key(base: Path | None, key: bytes, passphrase: str) -> None:
    """Wrap ``key`` under ``passphrase`` and remove the plaintext key file.

    The order is load-bearing: the wrapped file is written and fsync'd FIRST,
    and only then is the plain key removed. The reverse order loses the store
    outright if the process dies in between — there would be no key on disk and
    none in any broker's memory. A crash in this order leaves both files, and
    :func:`key_mode` then reports ``passphrase`` while the stale plain key is
    removed on the next successful call.
    """
    write_private_file(wrapped_key_path(base), _wrap_blob(key, passphrase))
    key_path(base).unlink(missing_ok=True)


def _wrap_blob(key: bytes, passphrase: str) -> bytes:
    """The on-disk representation of ``key`` wrapped under ``passphrase``.

    Split out of :func:`wrap_master_key` because ``rotate`` on a hardened store
    needs the same bytes written to a STAGING path rather than to the key of
    record. Two spellings of this format would be two things to keep in step,
    and the header is bound as AAD, so a drift between them is a store that
    cannot be unwrapped.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = os.urandom(SCRYPT_SALT_BYTES)
    nonce = os.urandom(12)
    # The header is bound as AAD so the parameters cannot be edited down to a
    # cheaper KDF cost by an attacker holding the file.
    header = _WRAP_MAGIC + bytes([_WRAP_VERSION]) + salt + nonce
    return header + AESGCM(_derive_wrapping_key(passphrase, salt)).encrypt(nonce, key, header)


def staged_wrapped_key_paths(base: Path | None = None) -> list[Path]:
    """Every WRAPPED key staged by a rotation of a hardened store.

    The hardened-tier counterpart of :func:`staged_key_paths`, and separate from
    it for a reason that is the whole point of QA Q10: the keyfile tier stages
    the raw 32 key bytes, and doing that on a hardened store would put a
    plaintext master key on disk for the duration of every rotation — defeating
    the one property §2.3 sells the tier on, just for a narrower window than the
    bug QA found. So the staged copy is wrapped under the same passphrase as the
    key of record, and the two staging sets never mix.
    """
    try:
        return sorted(secrets_dir(base).glob(f"{STAGED_WRAPPED_PREFIX}*"))
    except OSError:
        return []


def stage_wrapped_master_key(base: Path | None, key: bytes, passphrase: str) -> Path:
    """Persist a hardened rotation's new key, wrapped, BEFORE the re-seal commits.

    Same crash-safety contract as :func:`stage_master_key` — at every instant a
    key that opens the store exists on disk — with the difference that what is
    written is the wrapped blob, so the invariant holds without ever putting an
    unwrapped key beside the database.

    Recovery differs accordingly and is worth stating: a plaintext staged key
    can be matched against the database's fingerprint by anyone, so
    :func:`~local_operator.secrets.access.resolve_master_key` repairs an
    interrupted keyfile rotation with no operator involvement. A wrapped one
    cannot be opened without the passphrase, so the equivalent repair happens
    at ``unlock``, which is the moment the passphrase is available — see
    :func:`unwrap_master_key_matching`.
    """
    ensure_secrets_dir(base)
    directory = secrets_dir(base)
    unique = f"{os.getpid()}.{os.urandom(6).hex()}"
    path = directory / f"{STAGED_WRAPPED_PREFIX}.{unique}"
    temporary = directory / f"{_STAGING_TEMP_PREFIX}.{unique}.wrapped.tmp"
    try:
        write_private_file(temporary, _wrap_blob(key, passphrase))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def install_staged_wrapped_key(base: Path | None, staged: Path) -> None:
    """Promote a staged wrapped key to the key of record, atomically.

    ``os.replace`` within the same directory, so a reader sees the old wrapping
    or the new one and never a truncated file — a crash mid-write here would
    leave the only on-disk copy of the store's key a partial ciphertext, which
    is unrecoverable loss of every secret.

    The plaintext unlink afterwards is not defensive clutter: it is the tier's
    defining property (§2.3), enforced at the one place a hardened store's key
    of record is replaced. A store that arrives here carrying a plaintext key
    was damaged by the pre-fix ``rotate``, and finishing the install without
    removing it would preserve exactly the silent downgrade Q10 reported.
    """
    os.replace(staged, wrapped_key_path(base))
    os.chmod(wrapped_key_path(base), FILE_MODE)
    key_path(base).unlink(missing_ok=True)


def discard_staged_wrapped_key(base: Path | None, staged: Path) -> None:
    """Remove a staged wrapped key this caller owns.

    Identified by PATH, not by content as :func:`discard_staged_master_key`
    does: a wrapped blob carries a fresh random salt and nonce per write, so
    two wrappings of the same key are different bytes and content comparison
    cannot recognise ownership. The path is unique per rotation and the caller
    holds the one it created, which identifies the owner exactly as well.
    """
    try:
        staged.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - raced with a reader; not ours to insist on
        pass


def assert_key_of_record_invariant(base: Path | None, mode: str) -> None:
    """Fail loudly if the on-disk key files contradict the tier just written.

    The one invariant this feature cannot afford to get wrong, stated once and
    checked wherever a key is installed: **in the hardened tier no plaintext
    ``master.key`` exists, and in the keyfile tier no wrapped key pretends the
    store is hardened.** QA Q10 was a single verb (``rotate``) writing the wrong
    file for the tier, which silently downgraded the hardened store to plaintext
    at rest while ``status`` kept reporting ``passphrase``.

    A silent downgrade is the worst available failure because nothing surfaces
    it — the operator keeps trusting a guarantee that has stopped holding. So a
    violation raises here, at the moment the offending write happens, rather
    than being discovered later by an audit. It is a post-condition on the
    writer, not a validation of operator input: reaching it means this code
    installed the wrong file, which is a bug in the caller.
    """
    plain = key_path(base).exists()
    wrapped = wrapped_key_path(base).exists()
    if mode == "passphrase" and plain:
        raise SecretStoreError(
            f"Internal error: a plaintext master key was left at {key_path(base)} on a "
            "store hardened with a passphrase. Refusing to leave the store's key readable "
            "beside it; please report this."
        )
    if mode == "keyfile" and not plain:
        raise SecretStoreError(
            f"Internal error: no master key was installed at {key_path(base)} for a store "
            "in keyfile mode."
        )
    if mode == "keyfile" and wrapped:
        raise SecretStoreError(
            f"Internal error: {wrapped_key_path(base)} still claims this store is hardened "
            "while a plaintext master key sits beside it."
        )


def unwrap_master_key(base: Path | None, passphrase: str) -> bytes:
    """Recover the master key from the wrapped file, or raise.

    A wrong passphrase surfaces as ``InvalidTag`` from AES-GCM and is
    translated into a plain sentence: this is the one prompt the operator sees
    per boot, and a traceback would be a poor way to say "that was the wrong
    passphrase".
    """
    path = wrapped_key_path(base)
    if not path.exists():
        raise SecretStoreError(f"No hardened key at {path}; this store is not in passphrase mode.")
    check_mode(path)
    return _unwrap_blob(path.read_bytes(), passphrase, path)


def _unwrap_blob(blob: bytes, passphrase: str, path: Path) -> bytes:
    """Decode and decrypt a wrapped-key blob. ``path`` only names it in errors.

    Shared by the key of record and by the staged copies a hardened rotation
    leaves behind, so the format is parsed in exactly one place.
    """
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    prefix = len(_WRAP_MAGIC) + 1 + SCRYPT_SALT_BYTES + 12
    if len(blob) <= prefix or not blob.startswith(_WRAP_MAGIC):
        raise SecretStoreError(f"{path} is not a valid wrapped master key.")
    if blob[len(_WRAP_MAGIC)] != _WRAP_VERSION:
        raise SecretStoreError(
            f"{path} was written in wrapped-key format {blob[len(_WRAP_MAGIC)]}; "
            f"this runtime understands {_WRAP_VERSION}. Upgrade local-operator."
        )
    salt = blob[len(_WRAP_MAGIC) + 1 : len(_WRAP_MAGIC) + 1 + SCRYPT_SALT_BYTES]
    nonce = blob[len(_WRAP_MAGIC) + 1 + SCRYPT_SALT_BYTES : prefix]
    header, ciphertext = blob[:prefix], blob[prefix:]
    try:
        key = AESGCM(_derive_wrapping_key(passphrase, salt)).decrypt(nonce, ciphertext, header)
    except InvalidTag as exc:
        raise SecretStoreError("Wrong passphrase for this secret store.") from exc
    if len(key) != KEY_BYTES:
        raise SecretStoreError("The wrapped key is damaged; it did not contain a master key.")
    return key


def unwrap_master_key_matching(base: Path | None, passphrase: str) -> bytes:
    """Unwrap the key that actually opens this store, finishing a broken rotation.

    The hardened-tier counterpart of
    :func:`~local_operator.secrets.access.resolve_master_key`, and it has to be
    a separate function for one reason: that one repairs an interrupted
    rotation by fingerprinting STAGED PLAINTEXT keys, which a hardened store
    deliberately does not have. Here the staged copies are wrapped, so they can
    only be tested at the single moment the passphrase exists — ``unlock``.

    Without this, a power cut in ``rotate``'s commit-to-install window left a
    hardened store permanently unreadable: the wrapped key of record still
    wraps the superseded key while the database has moved on. The keyfile tier
    has recovered from that window since round 1; this gives the hardened tier
    the same guarantee rather than a weaker one hidden behind the same verb.

    Ordinary case first and unconditionally: when the key of record opens the
    store — or when there is no store or no fingerprint row to compare against
    — nothing else is read and no staged file is touched.
    """
    from local_operator.secrets.crypto import key_fingerprint
    from local_operator.secrets.store import recorded_key_fingerprint

    key = unwrap_master_key(base, passphrase)
    expected = recorded_key_fingerprint(base)
    if expected is None or key_fingerprint(key) == expected:
        return key

    for staged in staged_wrapped_key_paths(base):
        try:
            candidate = _unwrap_blob(staged.read_bytes(), passphrase, staged)
        except (SecretStoreError, OSError):
            # A blob wrapped under a DIFFERENT passphrase, or a file that
            # vanished under a concurrent rotation's install. Neither is this
            # caller's to repair, and neither is a reason to fail the unlock.
            continue
        if key_fingerprint(candidate) != expected:
            continue
        # The re-seal committed under this key and only the install is
        # outstanding, exactly as `resolve_master_key` establishes before it
        # completes a keyfile rotation. Finishing it here means the next crash
        # does not find the same half-done state.
        install_staged_wrapped_key(base, staged)
        return candidate

    raise SecretStoreError(
        "This secret store is sealed under a master key that the passphrase-wrapped key "
        "on disk does not contain. A key rotation appears to have been interrupted, and "
        "no staged wrapped key matching this store was found beside it."
    )
