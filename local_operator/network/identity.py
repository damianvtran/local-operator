"""Device identity: one Ed25519 keypair per install, and what a copy of it means.

WHAT THE KEY IS FOR. It signs handshake transcripts and membership statements
and it never encrypts: forward secrecy comes from a fresh X25519 ephemeral per
link, which is why a long-term Ed25519-only key is sufficient. The device id is
DERIVED from the public key, so a member list can be verified without a CA and
a row's authority is its public key — the id is a name, never authority.

WHY A 0600 FILE AND NOT THE KEYCHAIN. Three reasons, in order of weight:

1. **The relay must boot unattended.** A install that has logged in keeps
   working across sessions and restarts with no re-authentication, and a launchd
   job that runs at load — or on a machine where nobody is logged into the GUI —
   cannot rely on an unlocked login keychain: ``security find-generic-password``
   either fails or raises a UI prompt into a session that may not exist. This
   repo has already measured that class of pain; the mobile portal password is
   the only keychain caller in the tree and ``LOP_MOBILE_PASSWORD`` exists
   precisely as its escape hatch.
2. **The keychain answers a different question.** It protects a value from other
   user ACCOUNTS and from filesystem reads; this threat model includes a hostile
   process in the SAME account, and the keychain is unlocked for the session so
   it does not help with that.
3. **Portability.** Linux and CI installs — the on-demand pod of R20 — have no
   macOS keychain at all.

What this honestly is: a 0600 file is the same boundary as a session's
``control_key`` and the secret store's ``master.key``. A process running as this
account can read it. The device key protects the NETWORK from other accounts and
from hosts that are not members; it does not protect the device from itself.

WHY A COPY OF THE KEY IS DETECTED AND NOT PREVENTED. At the crypto layer a
copied key IS the device: same id, same signature, indistinguishable. So the
design makes the copy *visible* and *remediable* instead — one active link per
device id, evicting the older claim and auditing ``duplicate_identity``, with the
restart grace window below distinguishing an ordinary restart from a copy. The
remedy is ``lop network member rm`` (which takes effect without visiting the
other device); rotating the device key is the WRONG remedy, because the attacker
holds it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from secrets import token_bytes, token_hex
from typing import Any, Literal

from local_operator.network.wire import b64u, crockford, unb64u
from local_operator.paths import config_dir

#: The identity store's modes. Named rather than inlined because they ARE the
#: authorization model, and because a test asserts them (a mode that drifted to
#: 0644 would still pass every functional test in this package).
DIR_MODE = 0o700
FILE_MODE = 0o600

IDENTITY_SCHEMA = 1
IDENTITY_ALGORITHM = "ed25519"

#: Domain separation for the derived id. A prefix that never appears in any other
#: hash in this package, so a digest computed for one purpose can never be
#: presented as another's.
DEVICE_ID_DOMAIN = b"lop-device-id-v1\x00"
DEVICE_FINGERPRINT_DOMAIN = b"lop-device-fingerprint-v1\x00"

#: How recently the previous link must have carried a frame for a same-id,
#: different-instance reconnect to be read as a RESTART rather than a copy.
#: Sized from the relay's SIGTERM path: a clean restart closes links, exits and
#: is respawned by launchd inside a second or two, so five seconds covers the
#: honest case with margin while still being far shorter than any plausible
#: hand-off of a stolen key.
LINK_RESTART_GRACE_S = 5.0

#: Distinct instance ids within this window before the member is flagged
#: ``suspect``. Three is chosen so a crash-restart loop (launchd's throttled
#: restarts are 10 s apart) shows up quickly, while the ordinary
#: upgrade-and-restart of a working machine does not.
DUPLICATE_FLAG_WINDOW_S = 3600.0
DUPLICATE_FLAG_COUNT = 3


def network_root(root: Path | None = None) -> Path:
    """``<config>/network``, created 0700 on first use.

    ``root`` exists so a test — or the QA harness running two "devices" on one
    host — can point at its own directory without monkeypatching ``HOME``. The
    default is ``paths.config_dir()``, which honours ``LOCAL_OPERATOR_CONFIG_DIR``.
    """
    path = (root or config_dir()) / "network"
    _ensure_dir(path)
    return path


def identity_dir(root: Path | None = None) -> Path:
    path = network_root(root) / "identity"
    _ensure_dir(path)
    return path


def identity_path(root: Path | None = None) -> Path:
    return identity_dir(root) / "device.json"


def _ensure_dir(path: Path) -> None:
    """Create a private directory, and REPAIR the mode of an existing one.

    The repair matters: an upgrade path or a restored backup can leave the
    directory at 0755, and a directory that is merely created correctly on the
    first-ever run is a permission bug waiting for the second run's circumstances.
    """
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, DIR_MODE)


@dataclass
class DeviceIdentity:
    """This install's long-term keypair, as it is stored.

    The private key is the 32-byte Ed25519 SEED in unpadded base64url, not the
    64-byte expanded form: the seed is what ``Ed25519PrivateKey.from_private_bytes``
    takes, and storing the pair would let the two disagree.
    """

    device_id: str
    public_key: str
    private_key: str = ""
    schema: int = IDENTITY_SCHEMA
    algorithm: str = IDENTITY_ALGORITHM
    generation: int = 1
    created_at: float = field(default_factory=time.time)
    name: str = ""
    #: The previous ``device_id`` when this identity came from a rotation. An id
    #: is a key fingerprint, so it cannot survive a key change; this is what lets
    #: a rotation statement prove the link between the two.
    rotated_from: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: dict[str, Any]) -> DeviceIdentity:
        known = set(DeviceIdentity.__dataclass_fields__)
        return DeviceIdentity(**{k: v for k, v in data.items() if k in known})

    @property
    def public_key_bytes(self) -> bytes:
        return unb64u(self.public_key)

    @property
    def private_key_bytes(self) -> bytes:
        return unb64u(self.private_key)

    def sign(self, message: bytes) -> bytes:
        """Sign ``message`` with the device key. The signing key never leaves here.

        Function-local crypto import: this module is imported by ``lop network
        status`` on the CLI startup path, and paying for ``cryptography`` on
        every ``lop`` invocation is exactly what the repo's import-graph guard
        exists to prevent.
        """
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.from_private_bytes(self.private_key_bytes)
        return key.sign(message)

    def fingerprint(self) -> str:
        """A short Crockford base32 label for THIS device, for a human to compare.

        Cosmetic and derived — never authority, and never a substitute for
        comparing the public key. It exists so ``lop network show`` can print an
        id beside a name and a person can check two screens agree.
        """
        digest = hashlib.sha256(DEVICE_FINGERPRINT_DOMAIN + self.public_key_bytes).digest()
        return crockford(digest[:10])


def device_id_for(public_key: bytes) -> str:
    """``device_id`` from the public key, exactly as §3.1 defines it.

    The truncation is 128 bits: an accidental collision is bounded by 2⁻⁶⁴, and
    an ADVERSARIAL one buys only a name clash, because authorisation compares the
    full public key and admission refuses an existing id presented with a
    different key (``device_id_conflict``).
    """
    digest = hashlib.sha256(DEVICE_ID_DOMAIN + public_key).hexdigest()
    return f"d_{digest[:32]}"


def mint_instance_id() -> str:
    """A per-relay-PROCESS id, minted at startup and bound into the handshake.

    Twelve random bytes in Crockford base32. It is what distinguishes a restart
    (new process, new id, same device key) from a COPY of the key (a second
    process claiming one device id at the same time), and it is inside the signed
    transcript so an intermediary cannot swap it.
    """
    return f"i_{crockford(token_bytes(12))}"


def load(root: Path | None = None) -> DeviceIdentity | None:
    """Read this install's identity, or ``None`` when it has never had one."""
    path = identity_path(root)
    if not path.exists():
        return None
    # A malformed identity is NOT quarantined the way a network record is: a
    # network record is a membership list we must not silently reinterpret, while
    # this is the device's own proof of identity, and a corrupt one has no
    # salvageable meaning. It raises with the path so the operator sees which
    # file to remove and re-pair from.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"the device identity at {path} cannot be read ({exc}). Move it aside and "
            "re-pair this device with a new invite."
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"the device identity at {path} is not an object; remove it and re-pair")
    identity = DeviceIdentity.from_json(data)
    if not identity.device_id or not identity.public_key:
        raise RuntimeError(f"the device identity at {path} is incomplete; remove it and re-pair")
    # Verify the stored id against the stored key, because the id is what every
    # peer compares and a file edited to claim someone else's id must be refused
    # HERE, on the device that would otherwise present the claim.
    expected = device_id_for(identity.public_key_bytes)
    if expected != identity.device_id:
        raise RuntimeError(
            f"the device identity at {path} claims device id {identity.device_id} for a key "
            f"that derives {expected}; the file has been edited. Remove it and re-pair."
        )
    return identity


def mint(root: Path | None = None, *, name: str = "") -> DeviceIdentity:
    """Generate a new identity and write it. Never overwrites an existing one."""
    if identity_path(root).exists():
        raise RuntimeError(
            f"{identity_path(root)} already exists; use `lop network identity rotate` to "
            "replace a device key deliberately"
        )
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    identity = DeviceIdentity(
        device_id=device_id_for(public),
        public_key=b64u(public),
        # The private half of the pair is written where only this account can read
        # it, and nowhere else: no log, no record, no frame, no --json payload.
        private_key=b64u(seed),
        name=name or _default_name(),
    )
    save(identity, root)
    return identity


def save(identity: DeviceIdentity, root: Path | None = None) -> Path:
    """Write the identity file atomically: 0600, staged, then renamed."""
    path = identity_path(root)
    _staged_write(path, identity.to_json())
    return path


def load_or_mint(root: Path | None = None, *, name: str = "") -> DeviceIdentity:
    """The relay's startup call: the existing identity, or a fresh one."""
    existing = load(root)
    if existing is not None:
        return existing
    return mint(root, name=name)


def rotate(
    root: Path | None = None, *, name: str | None = None
) -> tuple[DeviceIdentity, DeviceIdentity]:
    """Mint a replacement key and adopt it.

    Returns ``(new, old)`` — the caller signs rotation STATEMENTS with the old
    key and must do so before the old identity is dropped from memory, so
    returning it rather than discarding it is the whole point of this signature.
    A rotation is refused if no identity exists: rotating nothing would silently
    create an identity with ``generation: 1`` and no history.
    """
    old = load(root)
    if old is None:
        raise RuntimeError(
            "this install has no device identity to rotate; run `lop network init` first"
        )
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    new = DeviceIdentity(
        device_id=device_id_for(public),
        public_key=b64u(public),
        private_key=b64u(seed),
        generation=old.generation + 1,
        name=name if name is not None else old.name,
        rotated_from=old.device_id,
    )
    save(new, root)
    return new, old


# ---------------------------------------------------------------------------
# Rotation statements (§3.3)
# ---------------------------------------------------------------------------

ROTATION_STATEMENT_KIND = "lop-device-rotate"


def rotation_statement(
    old: DeviceIdentity,
    new: DeviceIdentity,
    network_id: str,
    *,
    signed_at: float | None = None,
) -> dict[str, Any]:
    """The continuity proof, signed by the OLD key (``sig_old``).

    Without it a rotation is indistinguishable from an impostor claiming a
    member's name with a new key, so the statement carries both ids and the new
    public key, and the receiver checks all three plus the signature.
    """
    statement: dict[str, Any] = {
        "kind": ROTATION_STATEMENT_KIND,
        "network_id": network_id,
        "old_device_id": old.device_id,
        "new_device_id": new.device_id,
        "new_public_key": new.public_key,
        "rotated_at": time.time() if signed_at is None else signed_at,
    }
    canonical = json.dumps(statement, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    statement["sig_old"] = b64u(old.sign(canonical.encode("utf-8")))
    return statement


def verify_rotation_statement(statement: dict[str, Any], old_public_key: str) -> None:
    """Verify ``sig_old`` over the canonical statement. Raises on any failure.

    ``old_public_key`` is the caller's, read from the member row the receiver
    already trusts, and it is a PARAMETER rather than a field of the statement on
    purpose: a statement that carried its own old key would verify against itself
    and prove nothing. A member row is not a licence to re-identify as anybody,
    so the old id must be the fingerprint of that trusted key.

    Checks, in this order and each one fatal: the statement has the fields it
    needs; the OLD id is the fingerprint of the OLD key we hold; the NEW id is
    the fingerprint of the carried public key (so the row we write cannot be made
    to claim an id nobody can prove); and the signature verifies over the
    statement with ``sig_old`` removed.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from local_operator.network.types import HandshakeRefusal

    required = ("old_device_id", "new_device_id", "new_public_key", "sig_old", "rotated_at")
    missing = [field for field in required if not statement.get(field)]
    if missing:
        raise HandshakeRefusal(
            "bad_rotation_statement",
            f"a device rotation statement is missing {', '.join(missing)}",
        )
    public = unb64u(str(statement["new_public_key"]))
    if device_id_for(public) != statement["new_device_id"]:
        raise HandshakeRefusal(
            "bad_rotation_statement",
            "a device rotation statement names an id that is not the fingerprint of the "
            "public key it carries",
        )
    if device_id_for(unb64u(old_public_key)) != statement["old_device_id"]:
        raise HandshakeRefusal(
            "bad_rotation_statement",
            "a device rotation statement names an old id that is not the fingerprint of the "
            "public key on that member's row",
        )
    body = {k: v for k, v in statement.items() if k != "sig_old"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    try:
        Ed25519PublicKey.from_public_bytes(unb64u(old_public_key)).verify(
            unb64u(str(statement["sig_old"])), canonical.encode("utf-8")
        )
    except InvalidSignature as exc:
        raise HandshakeRefusal(
            "bad_rotation_statement",
            "the signature on a device rotation statement does not verify against the old key",
        ) from exc


def _default_name() -> str:
    import socket

    return socket.gethostname()


def _staged_write(target: Path, payload: Any) -> None:
    """Write ``payload`` as JSON, 0600, staged and renamed into place.

    The SAME shape ``session/runtime/registry._staged_write`` implements, and
    deliberately a second copy of five lines rather than an import of that
    module's private helper: this one must not depend on the run-directory's
    module (the CLI startup path pays for whatever the network package imports,
    and the registry module belongs to the session plane). The two are pinned to
    the same mode by ``tests/unit/network/test_identity.py``, which asserts the
    file and its directory modes directly.

    THE STAGING NAME IS NOT JUST THE PID: a process has many threads, and two of
    them writing one identity file shared a staging file — one ``os.replace``
    takes it away under the other's ``os.chmod``, exactly the window QA round 15
    measured in ``store._write_private_json``. The consequence here is worse than
    a lost update: ``load`` treats an unreadable identity as an ABSENT one and
    mints a fresh key, i.e. a new device id the mesh has never seen. Thread id
    and four random bytes make the name unique per call; the store's lock is not
    copied here because the writer is a single payload — two savers of one
    identity write the same bytes, so ordering does not matter.
    """
    directory = target.parent
    _ensure_dir(directory)
    temporary = directory / (
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.{token_hex(4)}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, FILE_MODE)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The duplicate-identity fence (§3.4)
# ---------------------------------------------------------------------------

UseVerdictKind = Literal["new", "restart", "duplicate"]


@dataclass
class LinkClaim:
    """One live link's claim on a device id."""

    link_id: str
    instance_id: str
    opened_at: float
    last_frame_at: float


@dataclass
class UseVerdict:
    """What an incoming claim means, and what the relay must do about it."""

    kind: UseVerdictKind
    evicted: LinkClaim | None = None
    #: True when this claim incremented the member's ``duplicate_count``.
    flagged: bool = False


class IdentityUseTracker:
    """One live link per device id, and whether a second one is a copy.

    The eviction policy is NEWEST WINS in both branches, because refusing the new
    link would let a stale copy pin a device's slot and deny service. The grace
    window is what separates the two honest cases from the dishonest one:

    * a different ``instance_id`` inside :data:`LINK_RESTART_GRACE_S` of the
      previous link's last frame is a RESTART (audit ``link_replaced``);
    * a different instance id outside it is a copy — or an ungraceful restart,
      which the flag accepts as its false-positive cost;
    * the SAME ``instance_id`` on a second link is impossible from a correct peer
      (one process, one id), so it is a fork or a copy either way.
    """

    def __init__(self) -> None:
        self._claims: dict[str, LinkClaim] = {}
        self._instances: dict[str, list[tuple[float, str]]] = {}

    def observe(
        self,
        device_id: str,
        *,
        instance_id: str,
        link_id: str,
        now: float | None = None,
    ) -> UseVerdict:
        moment = time.time() if now is None else now
        previous = self._claims.get(device_id)
        verdict = UseVerdict(kind="new")
        if previous is not None:
            within_grace = (moment - previous.last_frame_at) <= LINK_RESTART_GRACE_S
            if previous.instance_id != instance_id and within_grace:
                verdict = UseVerdict(kind="restart", evicted=previous)
            else:
                verdict = UseVerdict(kind="duplicate", evicted=previous, flagged=True)
        self._claims[device_id] = LinkClaim(
            link_id=link_id,
            instance_id=instance_id,
            opened_at=moment,
            last_frame_at=moment,
        )
        self._record_instance(device_id, instance_id, moment)
        return verdict

    def note_frame(self, device_id: str, *, now: float | None = None) -> None:
        claim = self._claims.get(device_id)
        if claim is not None:
            claim.last_frame_at = time.time() if now is None else now

    def released(self, device_id: str, link_id: str) -> None:
        """Forget a claim when ITS link closed, never when another's did."""
        claim = self._claims.get(device_id)
        if claim is not None and claim.link_id == link_id:
            del self._claims[device_id]

    def claim_for(self, device_id: str) -> LinkClaim | None:
        return self._claims.get(device_id)

    def recent_instance_count(self, device_id: str, *, now: float | None = None) -> int:
        moment = time.time() if now is None else now
        window = [
            (seen, instance)
            for seen, instance in self._instances.get(device_id, [])
            if moment - seen <= DUPLICATE_FLAG_WINDOW_S
        ]
        return len({instance for _, instance in window})

    def _record_instance(self, device_id: str, instance_id: str, moment: float) -> None:
        seen = self._instances.setdefault(device_id, [])
        seen.append((moment, instance_id))
        # Bounded: the window is an hour and the relay's restart rate is launchd's,
        # so a hundred entries is already far past any honest machine and stops a
        # hostile peer from growing this list without limit.
        cutoff = moment - DUPLICATE_FLAG_WINDOW_S
        self._instances[device_id] = [entry for entry in seen if entry[0] >= cutoff][-100:]
