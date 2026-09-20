"""The trust anchor: the operator's public key, pinned where a subject cannot move it.

THE ONE FACT THE WHOLE MECHANISM RESTS ON. Everything else in
:mod:`local_operator.operator` assumes the runtime verifies signatures against a
key it read from a place the party being gated cannot write. If the anchor were
readable-or-writable under the same uid, a model's tool subprocess would point
the runtime at its own key and every claim here would be false — so the anchor is
a ROOT/ADMIN-OWNED FILE with public data only:

* macOS  — ``/Library/Application Support/local-operator/operators/<uid>.json``
* Linux  — ``/etc/local-operator/operators/<uid>.json``
* Windows — ``%PROGRAMDATA%\\local-operator\\operators\\<sid>.json``

MEASURED, IN A THROWAWAY KEYCHAIN (macOS 26.6.2, uid 501, SIP on), and the reason
this is a file and not a keychain item:

* a ``login`` keychain item is SILENTLY SUBSTITUTABLE by a same-uid process —
  ``delete-generic-password`` then ``add-generic-password`` with a different
  value both return 0, the readback shows the substituted value, and no prompt or
  privilege is involved;
* two items with the same service+account cannot coexist inside ONE keychain
  (``errSecDuplicateItem``), so the attacker must replace rather than add — but
  SHADOWING ACROSS KEYCHAINS WORKS, and the search order is rewritable with no
  authentication (``security list-keychains -s``, rc=0);
* ``/Library/*`` is not writable by that admin user at all.

TWO FURTHER CONSEQUENCES, STATED RATHER THAN GLOSSED:

* the enforcement point matters as much as the anchor. A verifier inside a
  user-writable tree can be replaced or bypassed by the same subject. The harness
  install (``~/.local/share/uv/tools/local-operator``) is user-writable today;
  that is the pre-existing Stage-2 confining epic, not something this change
  fixes, and the level report keeps it visible instead of implying uniformity;
* the path must not be redirectable by anything the uid can write. It is built
  from constants only — no environment variable, no config value, no ``PATH``
  lookup — and every component is ``lstat``-checked for symlinks, with the final
  open using ``O_NOFOLLOW`` and validating owner and mode from the OPEN FILE
  DESCRIPTOR rather than from a second ``stat`` of the path.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.operator.keychain import FILE_ONLY, SECURE_ENCLAVE
from local_operator.operator.verify import key_id_for

#: The anchor's own format version. Read rather than assumed so a future shape
#: can be refused by an old runtime instead of misread.
ANCHOR_VERSION = 1

#: Where the anchor lives. A CONSTANT per platform, deliberately not a function
#: of anything the gated subject can influence.
_ANCHOR_ROOTS = {
    "darwin": Path("/Library/Application Support/local-operator/operators"),
    "linux": Path("/etc/local-operator/operators"),
}

#: Overridable ONLY in-process, for tests. Nothing reads an environment variable
#: for this: an env-redirectable anchor path is the substitution attack this
#: module exists to prevent, so the seam is a module global a test patches rather
#: than a lookup the runtime performs (see ``test_the_anchor_path_cannot_be_redirected``).
_ANCHOR_ROOT_OVERRIDE: Path | None = None


def _owner_key() -> str:
    """The per-human component of the anchor's filename.

    POSIX answers with the uid, which is the identity the runtime and the
    operator's own processes agree on. Windows has no ``os.getuid`` at all —
    reading the attribute there raises ``AttributeError``, which the probe
    battery catches as a fatal unguarded POSIX attribute — so it answers with the
    account NAME.

    What that name is worth is deliberately small: the anchor's authority comes
    from WHERE the file lives and WHO owns it (``%PROGRAMDATA%`` is
    administrator-writable), never from this string. Its only job is that two
    humans on one host do not share a key.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is not None:
        return str(getuid())
    account = os.environ.get("USERNAME") or os.environ.get("USER") or "default"
    return re.sub(r"[^A-Za-z0-9._-]", "_", account)[:64]


def anchor_dir() -> Path:
    """The directory the anchor is read from."""
    if _ANCHOR_ROOT_OVERRIDE is not None:
        return _ANCHOR_ROOT_OVERRIDE
    if os.name == "nt":  # pragma: no cover — Windows only; CI runs POSIX
        program_data = os.environ.get("PROGRAMDATA") or r"C:\ProgramData"
        return Path(program_data) / "local-operator" / "operators"
    root = _ANCHOR_ROOTS.get(sys.platform)
    if root is None:  # pragma: no cover — an unknown POSIX platform
        # Deliberately NOT a home-relative fallback: a fallback directory under
        # the operator's own home is writable by the very subject this file is
        # meant to be out of reach of, and silently reading an anchor from there
        # would report a boundary that does not exist. An unknown platform
        # reports "no anchor" instead.
        return Path("/nonexistent-local-operator-anchor")
    return root


def anchor_path(uid: int | str | None = None) -> Path:
    """One file per human, so two admins on one host do not share a key."""
    who = uid if uid is not None else _owner_key()
    return anchor_dir() / f"{who}.json"


@dataclass(frozen=True)
class OperatorAnchor:
    """The pinned public key, plus the facts needed to report the level honestly.

    ``devices`` is the revocation list the design asks for and is part of the
    pinned file rather than a separate store: an operator-signed certificate that
    is not listed here is refused, so revoking a lost phone is an edit to a
    root-owned file rather than an unauthenticated write to a cache.
    """

    key_id: str
    spki: bytes
    backend: str
    presence: bool
    label: str = ""
    created_at: int = 0
    devices: tuple[dict[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "v": ANCHOR_VERSION,
            "key_id": self.key_id,
            "alg": "ES256",
            "spki": self.spki.hex(),
            "backend": self.backend,
            "presence": self.presence,
            "label": self.label,
            "created_at": self.created_at,
            "devices": list(self.devices),
        }

    @classmethod
    def from_json(cls, body: Any) -> OperatorAnchor | None:
        """Parse an anchor, or ``None`` for anything that is not one.

        Weighted toward refusal: this is the root of trust, and a partially
        understood anchor is worse than a missing one, because a missing one
        reports a lower level while a misread one reports a boundary that is not
        there.
        """
        if not isinstance(body, dict) or body.get("v") != ANCHOR_VERSION:
            return None
        if body.get("alg") != "ES256":
            return None
        raw = body.get("spki")
        if not isinstance(raw, str):
            return None
        try:
            spki = bytes.fromhex(raw)
        except ValueError:
            return None
        if len(spki) != 65 or spki[0] != 0x04:
            return None
        key_id = body.get("key_id")
        if not isinstance(key_id, str) or key_id != key_id_for(spki):
            # The id is DERIVED, so a mismatch means the file was hand-edited or
            # assembled by something that does not know the rule. Refusing is
            # what stops a frame's ``operator_key_id`` from being matched against
            # a value unrelated to the key that will actually verify.
            return None
        devices = body.get("devices")
        if devices is None:
            devices = []
        if not isinstance(devices, list) or not all(isinstance(d, dict) for d in devices):
            return None
        backend = body.get("backend")
        presence = body.get("presence")
        return cls(
            key_id=key_id,
            spki=spki,
            backend=str(backend) if isinstance(backend, str) else FILE_ONLY,
            presence=bool(presence),
            label=str(body.get("label") or ""),
            created_at=int(body["created_at"]) if isinstance(body.get("created_at"), int) else 0,
            devices=tuple(devices),
        )


@dataclass(frozen=True)
class AnchorLoad:
    """What ``load_anchor`` found, and why — the report is the product.

    ``anchor is None`` with ``path`` present means the file is there but not
    usable; the level function distinguishes "no anchor at all" (the spawn
    capability is the only source) from "an anchor is installed but not
    root-owned" (someone put a file there, and the runtime will not trust it),
    because those two are a fresh host and a substituted anchor respectively.
    """

    anchor: OperatorAnchor | None
    path: Path
    root_owned: bool
    reason: str
    exists: bool = False

    @property
    def usable(self) -> bool:
        return self.anchor is not None and self.root_owned


def _path_is_symlink_free(path: Path) -> bool:
    """Whether every component of ``path`` is a real directory, not a link.

    Checked component by component with ``lstat``: a symlink anywhere above the
    file lets a same-uid subject point the runtime at a file it owns while the
    final path still reads as the constant one.
    """
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                return False
        except OSError:
            # A missing component is not a link: the load below reports it as
            # absent, and refusing here would report "redirected" for a host
            # that simply has not installed an anchor yet.
            return True
    return True


def _open_no_follow(path: Path) -> tuple[int, os.stat_result] | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        return descriptor, os.fstat(descriptor)
    except OSError:  # pragma: no cover — fstat on a fresh descriptor
        os.close(descriptor)
        return None


def load_anchor(uid: int | str | None = None) -> AnchorLoad:
    """Read and validate the pinned anchor. Never raises, never follows a link.

    FAIL-CLOSED AT EVERY STEP. A missing file, an unreadable file, a symlinked
    path, a file owned by anyone but root, a group- or world-writable file, and
    malformed content all produce an :class:`AnchorLoad` with no anchor — which
    the seam reads as "no operator source", leaving the spawn capability as the
    only way to loosen. The runtime must work in that state (:func:`operator_
    authority_level` reports it) rather than refusing to start.
    """
    path = anchor_path(uid)
    if not _path_is_symlink_free(path):
        # ``exists=True``, because something IS in the anchor's place and the
        # runtime will not trust it: that is a pinned-but-unusable anchor
        # (``anchor-unpinned``), not a host that never installed one
        # (``spawn-capability-only``). Reporting the second for the first state
        # told a reader with a redirect in the path the wrong reason (agent
        # review round 6, R6-6) in a report whose whole job is to distinguish
        # them.
        return AnchorLoad(
            anchor=None,
            path=path,
            root_owned=False,
            reason="a component of the anchor path is a symbolic link",
            exists=True,
        )
    opened = _open_no_follow(path)
    if opened is None:
        return AnchorLoad(anchor=None, path=path, root_owned=False, reason="no anchor installed")
    descriptor, info = opened
    try:
        if os.name != "nt":  # pragma: no branch — the POSIX check is the measured one
            if info.st_uid != 0:
                return AnchorLoad(
                    anchor=None,
                    path=path,
                    root_owned=False,
                    reason=f"the anchor is owned by uid {info.st_uid}, not root",
                    exists=True,
                )
            mode = stat.S_IMODE(info.st_mode)
            if mode & 0o022:
                return AnchorLoad(
                    anchor=None,
                    path=path,
                    root_owned=False,
                    reason=f"the anchor is writable by others (mode {mode:o})",
                    exists=True,
                )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read()
    finally:
        if descriptor >= 0:  # pragma: no cover — the with-block owns it on the happy path
            os.close(descriptor)
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return AnchorLoad(
            anchor=None,
            path=path,
            root_owned=info.st_uid == 0,
            reason="the anchor is not readable JSON",
            exists=True,
        )
    anchor = OperatorAnchor.from_json(body)
    if anchor is None:
        return AnchorLoad(
            anchor=None,
            path=path,
            root_owned=info.st_uid == 0,
            reason="the anchor does not describe an ES256 operator key",
            exists=True,
        )
    return AnchorLoad(
        anchor=anchor, path=path, root_owned=info.st_uid == 0, reason="ok", exists=True
    )


def anchor_bytes(anchor: OperatorAnchor) -> bytes:
    """The exact bytes ``lop operator install`` writes."""
    return json.dumps(anchor.to_json(), indent=2, sort_keys=True).encode("utf-8") + b"\n"


def staging_path(config_root: Path) -> Path:
    """Where the anchor waits for the one privileged step.

    In the operator's own tree, mode 0600, because it is not authoritative: the
    runtime reads the ROOT-OWNED path and nothing else. Its only job is to carry
    the bytes to ``sudo install`` without an operator retyping a public key.
    """
    return config_root / "operator" / "anchor.json"


def load_staged_anchor(config_root: Path) -> OperatorAnchor | None:
    """The staged anchor, as a HINT about which backend holds the private half.

    Deliberately not authoritative and deliberately not trust-checked: nothing
    about a loosening decision reads this file. It exists for one narrow job —
    ``lop operator sign`` run after ``lop operator init`` but BEFORE the
    privileged ``install`` step must sign with the backend ``init`` chose rather
    than with whichever backend the presence ladder would pick on this host
    (which, on a host where ``init --backend file-only`` was used because the
    Secure Enclave refused, would be the wrong one). A hint that is wrong costs
    one failed load and a precise error; it cannot authorise anything, because
    the runtime checks the signature against the ROOT-OWNED anchor.
    """
    try:
        raw = staging_path(config_root).read_bytes()
    except OSError:
        return None
    try:
        return OperatorAnchor.from_json(json.loads(raw.decode("utf-8")))
    except (UnicodeDecodeError, ValueError):
        return None


def install_commands(staging: Path, target: Path) -> list[list[str]]:
    """The privileged commands onboarding needs, spelled in ONE place.

    A list of argv lists rather than one joined string, so the caller runs them
    without a shell: there is no quoted value here, and going through a shell
    would add a second place for the path to be reinterpreted. Root ownership is
    the property that makes the anchor an anchor, so owner and group are set
    explicitly rather than left to ``install``'s defaults, which inherit the
    invoking user's umask and group.

    Windows takes one command because ``%PROGRAMDATA%`` is already
    admin-writable and has no POSIX ownership to set.
    """
    if os.name == "nt":  # pragma: no cover — Windows only
        return [
            [
                "powershell",
                "-Command",
                f"Copy-Item -Force '{staging}' '{target}'",
            ]
        ]
    group = "wheel" if sys.platform == "darwin" else "root"
    return [
        ["sudo", "install", "-d", "-m", "0755", "-o", "root", "-g", group, str(target.parent)],
        ["sudo", "install", "-m", "0644", "-o", "root", "-g", group, str(staging), str(target)],
    ]


def device_is_revoked(anchor: OperatorAnchor, device_id: str) -> bool:
    """Whether the anchor's revocation list names this device.

    An empty list is "no opinion" (an anchor written before device support), and
    that is deliberately different from a list that names the device. Stage D
    fills this in; the predicate lives here so the runtime does not grow its own
    copy of the rule.
    """
    if not anchor.devices:
        return False
    for entry in anchor.devices:
        if entry.get("device_id") == device_id and entry.get("revoked"):
            return True
    return False


def is_presence_backend(backend: str) -> bool:
    """Whether this backend raises a human gesture per signature.

    The only two that do; ``file-only`` is the one the level report calls
    not-a-boundary, and it is spelled as a positive list so a future backend has
    to be added here deliberately rather than inheriting a claim.
    """
    return backend in (SECURE_ENCLAVE, "cng-presence")


#: Entry point used by the runtime's cached read (see ``server.py``).
AnchorProvider = Any


#: How long a runtime may keep trusting the REVOCATION LIST it read at first need.
#:
#: A window is needed at all because caching the anchor forever meant an
#: installed revocation never reached a runtime already running: the operator
#: edited the root-owned anchor, and every session started before that edit went
#: on honouring the revoked phone for as long as it lived — which the design's own
#: words make days (agent review round 6, R6-1, measured on a file-backed anchor).
#:
#: Thirty seconds rather than the certificate TTL's five minutes: a revocation is
#: a security action taken about a device that may already be in someone else's
#: hands, and this is the number the revoke receipt can be honest about. The cost
#: is one read of a small root-owned file per runtime per window, which is nothing.
ANCHOR_REFRESH_S = 30.0


@dataclass
class AnchorCache:
    """A read-once-in-memory view of the anchor, refreshed under a bound.

    READ ONCE, at first need, rather than per frame: the anchor is a file on
    disk, and a runtime that re-read it on every increasing frame would let a
    same-uid subject race the read. The design's phrase is "read once and pinned
    in memory", and this is where that pin lives. A successful load is cached, and
    a FAILED load is cached too — an attacker who can make the anchor unreadable
    could otherwise force a re-read per frame.

    IT IS REFRESHED, though, and that is the correction agent review round 6
    (R6-1) measured the need for. Caching forever made the pin absolute, and the
    pin is only supposed to be about WHO made the file and how often it is read —
    not about a revocation list frozen for the lifetime of a session:

    * the re-read happens at most once per :data:`ANCHOR_REFRESH_S`, never per
      frame, so the property the pin exists for is intact: a same-uid subject
      still cannot make a runtime read the root-owned file on demand;
    * it is adopted only when it names the SAME operator key id. A key
      ROTATION is a privileged step the operator takes deliberately, and a
      running session must not silently swap its root of trust underneath
      itself; a rotation therefore still takes effect at each runtime's next
      start, which is what the design says and what the level report reads;
    * a re-read that is no longer USABLE is adopted, because that direction is
      stricter: an anchor deleted, chmod-ed or replaced mid-session drops this
      runtime to the spawn capability rather than keeping device authority it
      can no longer justify.

    A cache whose ``load`` was INJECTED (the ``operator_anchor`` constructor
    seam a test or a caller uses) is never refreshed: a caller who hands one in
    owns its lifetime, and re-reading the real path from under it would replace
    the value it explicitly passed with a file it deliberately did not read.
    """

    load: AnchorLoad = field(default=None)  # type: ignore[assignment]
    #: The refresh window, a field rather than the constant so the runtime can
    #: pass its own (and a test can shrink it to zero) without a private poke at
    #: the loaded value.
    refresh_s: float = ANCHOR_REFRESH_S
    #: When the next re-read is due, on ``time.monotonic``. Also the flag that
    #: says whether this cache may refresh at all: it is set only by a read THIS
    #: object performed, so an injected load leaves it at zero and never
    #: refreshes.
    _refreshable: bool = False
    _next_refresh: float = 0.0

    def get(self) -> AnchorLoad:
        now = time.monotonic()
        if self.load is None or (self._refreshable and now >= self._next_refresh):
            fresh = load_anchor()
            self._next_refresh = now + self.refresh_s
            if self.load is None or self._is_adoptable(fresh):
                self.load = fresh
                self._refreshable = True
            return self.load
        return self.load

    def _is_adoptable(self, fresh: AnchorLoad) -> bool:
        """Whether a re-read may replace the pinned anchor.

        Two ways, and the asymmetry is the point: a load that is still USABLE
        has to name the same operator key (a rotation must not take effect
        mid-session), while a load that is no longer usable is adopted
        unconditionally because refusing a downgrade would keep authority the
        runtime can no longer justify — the one direction a cache must never
        fail in.
        """
        pinned = self.load
        if pinned is None:  # pragma: no cover - the caller reads None first
            return True
        # A load that is no longer USABLE is adopted unconditionally (the stricter
        # direction), and the ``is None`` half is written out rather than left to the
        # ``usable`` property: `usable` is a property, so a reader of this line — a
        # type checker included — cannot narrow ``fresh.anchor`` from it.
        if not fresh.usable or fresh.anchor is None:
            return True
        return pinned.anchor is not None and pinned.anchor.key_id == fresh.anchor.key_id
