"""The trust anchor: the operator's public key, pinned where a subject cannot move it.

THE ONE FACT THE WHOLE MECHANISM RESTS ON. Everything else in
:mod:`local_operator.operator` assumes the runtime verifies signatures against a
key it read from a place the party being gated cannot write. If the anchor were
readable-or-writable under the same uid, a model's tool subprocess would point
the runtime at its own key and every claim here would be false — so the anchor is
a ROOT/ADMIN-OWNED FILE with public data only:

* macOS  — ``/Library/Application Support/local-operator/operators/<uid>.json``
* Linux  — ``/etc/local-operator/operators/<uid>.json``
* Windows — ``<ProgramData known folder>\\local-operator\\operators\\<account>.json``
  (the folder is asked of the shell's Known Folder API, never of ``%PROGRAMDATA%``)

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

import ctypes
import functools
import hashlib
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
from local_operator.operator.verify import key_id_for, spki_fp
from local_operator.paths import config_dir

#: The anchor's own format version. Read rather than assumed so a future shape
#: can be refused by an old runtime instead of misread.
ANCHOR_VERSION = 1

#: Where the anchor lives. A CONSTANT per platform, deliberately not a function
#: of anything the gated subject can influence.
_ANCHOR_ROOTS = {
    "darwin": Path("/Library/Application Support/local-operator/operators"),
    "linux": Path("/etc/local-operator/operators"),
}

#: TEST-ONLY seam: overridable ONLY in-process, by a test. Nothing reads an
#: environment variable for this: an env-redirectable anchor path is the
#: substitution attack this module exists to prevent, so the seam is a module
#: global a test patches rather than a lookup the runtime performs (see
#: ``test_the_anchor_path_cannot_be_redirected``). Production code never assigns
#: it; the Windows ACL verifier still runs against an overridden root, which is
#: what lets the windows-latest job exercise the real verifier in a temp dir.
_ANCHOR_ROOT_OVERRIDE: Path | None = None

#: Where the anchor reads from when the Windows root cannot be resolved. A path
#: that never exists, so an unresolvable root reports "no anchor" (fail closed)
#: rather than falling back to anything the user's environment names.
_UNRESOLVED_ROOT = Path("/nonexistent-local-operator-anchor")

#: ``FOLDERID_ProgramData`` — {62AB5D82-FDC1-4DC3-A9DD-070D1D495D97}.
_FOLDERID_PROGRAMDATA = (
    0x62AB5D82,
    0xFDC1,
    0x4DC3,
    (0xA9, 0xDD, 0x07, 0x0D, 0x1D, 0x49, 0x5D, 0x97),
)


@functools.lru_cache(maxsize=1)
def _windows_program_data() -> Path | None:  # pragma: no cover — Windows only
    """``C:\\ProgramData`` as the SHELL knows it, never as the environment says.

    WHY THE KNOWN FOLDER API AND NOT A CONSTANT OR ``%PROGRAMDATA%`` (F4):
    ``%PROGRAMDATA%`` is a value the gated subject writes for its own tool
    subprocesses, so reading it let a same-user process point the runtime at a
    directory it owns. A hard-coded ``C:\\ProgramData`` would be env-proof but
    wrong on hosts whose system drive is not ``C:``. ``SHGetKnownFolderPath``
    answers from the machine's registry (``HKLM``, administrator-writable), which
    is the same trust level as the anchor file itself and moves with the drive.

    Cached for the process: the folder does not move under a running process,
    and ``anchor_dir`` is read on every anchor refresh. Any failure answers
    ``None`` and the caller reads :data:`_UNRESOLVED_ROOT` — never a fallback.
    """

    class _Guid(ctypes.Structure):
        _fields_ = [
            ("data1", ctypes.c_ulong),
            ("data2", ctypes.c_ushort),
            ("data3", ctypes.c_ushort),
            ("data4", ctypes.c_ubyte * 8),
        ]

    d1, d2, d3, d4 = _FOLDERID_PROGRAMDATA
    folder_id = _Guid(d1, d2, d3, (ctypes.c_ubyte * 8)(*d4))
    out = ctypes.c_wchar_p()
    try:
        shell32 = getattr(ctypes, "WinDLL")("shell32", use_last_error=True)
        ole32 = getattr(ctypes, "WinDLL")("ole32", use_last_error=True)
        hresult = shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(out))
    except (OSError, AttributeError):
        return None
    try:
        if hresult != 0 or not out.value:
            return None
        return Path(out.value)
    finally:
        ole32.CoTaskMemFree(out)


def _windows_account_name() -> str | None:  # pragma: no cover — Windows only
    """The account name from the process TOKEN (``GetUserNameW``), not ``%USERNAME%``.

    Same reason as the root (F4): ``%USERNAME%`` is writable by the subject, and
    naming another account's file would make this runtime read a different
    human's anchor. ``GetUserNameW`` answers the same string ``%USERNAME%``
    normally holds, so an anchor installed under the old spelling keeps its name.
    """
    size = ctypes.c_ulong(257)
    buffer = ctypes.create_unicode_buffer(size.value)
    try:
        advapi32 = getattr(ctypes, "WinDLL")("advapi32", use_last_error=True)
        if not advapi32.GetUserNameW(buffer, ctypes.byref(size)):
            return None
    except (OSError, AttributeError):
        return None
    return buffer.value or None


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
    # No environment fallback (F4): an unreadable token names a file that is
    # never installed, so the load reports "no anchor" rather than reading
    # whichever account the environment claims.
    account = _windows_account_name() or "unresolved-account"
    return re.sub(r"[^A-Za-z0-9._-]", "_", account)[:64]


def _on_windows() -> bool:
    """Whether the Windows rules apply — ONE seam a POSIX test can flip.

    Patching ``os.name`` itself would turn every ``Path`` in the test process
    into a ``WindowsPath``; a function the Windows branches all read lets the
    Ubuntu suite drive the Windows root, verifier and install command while the
    rest of the process stays POSIX.
    """
    return os.name == "nt"


def anchor_dir() -> Path:
    """The directory the anchor is read from."""
    if _ANCHOR_ROOT_OVERRIDE is not None:
        return _ANCHOR_ROOT_OVERRIDE
    if _on_windows():
        program_data = _windows_program_data()
        if program_data is None:  # pragma: no cover — the shell API failed
            return _UNRESOLVED_ROOT
        return program_data / "local-operator" / "operators"
    root = _ANCHOR_ROOTS.get(sys.platform)
    if root is None:  # pragma: no cover — an unknown POSIX platform
        # Deliberately NOT a home-relative fallback: a fallback directory under
        # the operator's own home is writable by the very subject this file is
        # meant to be out of reach of, and silently reading an anchor from there
        # would report a boundary that does not exist. An unknown platform
        # reports "no anchor" instead.
        return _UNRESOLVED_ROOT
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
        anchor, _ = cls._parse(body)
        return anchor

    @classmethod
    def _parse(cls, body: Any) -> tuple["OperatorAnchor | None", str]:
        """ONE parse, and the refusal's own sentence when there is one to give.

        ``from_json`` is the public spelling; this pair-valued form exists
        because ONE refusal is actionable and the reader should be told it (F5
        slice B). When everything about the body holds except
        ``key_id != key_id_for(spki)`` — the shape a hand edit or a foreign
        writer produces — both values are named, because the generic sentence
        ("does not describe an ES256 operator key") sent a reader whose
        statement looks coherent hunting for the problem in the wrong place.
        Every other refusal returns ``""``; those bodies are not operator
        statements at all, and the caller's generic sentence is the whole story.
        The checks, their order and their strictness are exactly ``from_json``'s;
        only the explanation is richer.
        """
        if not isinstance(body, dict) or body.get("v") != ANCHOR_VERSION:
            return None, ""
        if body.get("alg") != "ES256":
            return None, ""
        raw = body.get("spki")
        if not isinstance(raw, str):
            return None, ""
        try:
            spki = bytes.fromhex(raw)
        except ValueError:
            return None, ""
        if len(spki) != 65 or spki[0] != 0x04:
            return None, ""
        key_id = body.get("key_id")
        if not isinstance(key_id, str):
            return None, ""
        if key_id != key_id_for(spki):
            # The id is DERIVED, so a mismatch means the file was hand-edited or
            # assembled by something that does not know the rule. Refusing is
            # what stops a frame's ``operator_key_id`` from being matched against
            # a value unrelated to the key that will actually verify. The reason
            # names both values — and only the product action that replaces the
            # statement from this machine's own key (§2.9: no terminal
            # commands in refusal copy).
            return None, (
                f"the statement declares key id {key_id} but its key's id is "
                f"{key_id_for(spki)} — it was hand-edited, or written by a different "
                "key; ask Local Operator to set up operator authority again for this "
                "machine"
            )
        devices = body.get("devices")
        if devices is None:
            devices = []
        if not isinstance(devices, list) or not all(isinstance(d, dict) for d in devices):
            return None, ""
        backend = body.get("backend")
        presence = body.get("presence")
        return (
            cls(
                key_id=key_id,
                spki=spki,
                backend=str(backend) if isinstance(backend, str) else FILE_ONLY,
                presence=bool(presence),
                label=str(body.get("label") or ""),
                created_at=(
                    int(body["created_at"]) if isinstance(body.get("created_at"), int) else 0
                ),
                devices=tuple(devices),
            ),
            "",
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


#: SIDs allowed to own, and to hold write-class rights over, the anchor and the
#: two directories lop creates for it: LocalSystem and BUILTIN\\Administrators.
#: The same two principals an elevated install leaves in the DACL; TrustedInstaller
#: and every other principal are deliberately absent, so a widened ACL refuses.
_WINDOWS_TRUSTED_SIDS = frozenset({"S-1-5-18", "S-1-5-32-544"})

#: Access-mask bits that let a principal change what the anchor SAYS or WHO may
#: change it: write/append data (add file / add subdirectory on a directory),
#: write EA and attributes, delete and delete-child (replace by rename), WRITE_DAC,
#: WRITE_OWNER, and the generic/maximum bits that map onto them.
_WINDOWS_WRITE_MASK = (
    0x0000_0002  # FILE_WRITE_DATA / FILE_ADD_FILE
    | 0x0000_0004  # FILE_APPEND_DATA / FILE_ADD_SUBDIRECTORY
    | 0x0000_0010  # FILE_WRITE_EA
    | 0x0000_0040  # FILE_DELETE_CHILD
    | 0x0000_0100  # FILE_WRITE_ATTRIBUTES
    | 0x0001_0000  # DELETE
    | 0x0004_0000  # WRITE_DAC
    | 0x0008_0000  # WRITE_OWNER
    | 0x0200_0000  # MAXIMUM_ALLOWED
    | 0x1000_0000  # GENERIC_ALL
    | 0x4000_0000  # GENERIC_WRITE
)

#: ACE types that only DENY. Every other type in a DACL is read as granting
#: (fail closed: an ACE layout this module does not know is refused below).
_WINDOWS_DENY_ACE_TYPES = frozenset({1, 6, 10, 12})
#: The allow/deny types whose layout is header + mask + SID, the only ones a
#: file-system DACL carries. Object ACEs (5/6/11/12) never appear on files.
_WINDOWS_SIMPLE_ACE_TYPES = frozenset({0, 1, 9, 10})
_INHERIT_ONLY_ACE = 0x08


@dataclass(frozen=True)
class WindowsAce:
    """One DACL entry as the verifier needs it: type, flags, mask and trustee SID."""

    ace_type: int
    flags: int
    mask: int
    sid: str


@dataclass(frozen=True)
class WindowsSecurity:
    """Owner and DACL of one object. ``aces is None`` is a NULL DACL (everyone: full)."""

    owner: str
    aces: tuple[WindowsAce, ...] | None


def windows_acl_refusal(what: str, security: WindowsSecurity) -> str | None:
    """Why this owner+DACL may not hold the anchor, or ``None`` when it may.

    THE WINDOWS COUNTERPART OF ``st_uid == 0 and not mode & 0o022`` (F4). Before
    this, ``load_anchor`` skipped every ownership check on Windows, and the
    default ``C:\\ProgramData`` ACL (``BUILTIN\\Users:(CI)(WD,AD)`` plus
    ``CREATOR OWNER:(OI)(CI)(IO)(F)``) lets a standard user create
    ``ProgramData\\local-operator`` and own it outright — so a same-user agent
    could plant its own key. Pure over a value, so the Ubuntu suite drives it
    with fake descriptors and the windows-latest job drives it with real ones.

    Inherit-only ACEs are skipped: they grant nothing on THIS object (the
    ``CREATOR OWNER`` template is exactly that), and the children they would
    reach are checked in their own right.
    """
    if security.owner not in _WINDOWS_TRUSTED_SIDS:
        return f"the {what} is owned by {security.owner}, not SYSTEM or Administrators"
    if security.aces is None:
        return f"the {what} has no access list, so everyone may write it"
    for ace in security.aces:
        if ace.flags & _INHERIT_ONLY_ACE:
            continue
        if ace.ace_type not in _WINDOWS_SIMPLE_ACE_TYPES:
            return f"the {what} carries an access entry of a kind the runtime does not read"
        if ace.ace_type in _WINDOWS_DENY_ACE_TYPES:
            continue
        if ace.sid not in _WINDOWS_TRUSTED_SIDS and ace.mask & _WINDOWS_WRITE_MASK:
            return f"the {what} is writable by {ace.sid} (access mask {ace.mask:#x})"
    return None


def _windows_security(  # pragma: no cover — Windows only; the windows-latest job runs it
    target: Path | int,
) -> WindowsSecurity | None:
    """Owner SID and DACL of a path, or of an OPEN descriptor, via advapi32.

    ctypes rather than pywin32: pywin32 reaches this project only through the
    optional ``mcp`` extra, and the root of trust must not depend on an extra.
    A descriptor is read with ``GetSecurityInfo`` on its OS handle, so the file
    that is checked is the file that is then read — the Windows half of the
    POSIX ``fstat``-from-the-descriptor rule. ``None`` on any failure; the caller
    refuses it.
    """
    from ctypes import wintypes

    advapi32 = getattr(ctypes, "WinDLL")("advapi32", use_last_error=True)
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    se_file_object, owner_and_dacl = 1, 0x1 | 0x4
    if isinstance(target, int):
        import msvcrt

        handle = getattr(msvcrt, "get_osfhandle")(target)
        rc = advapi32.GetSecurityInfo(
            wintypes.HANDLE(handle),
            se_file_object,
            owner_and_dacl,
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
    else:
        rc = advapi32.GetNamedSecurityInfoW(
            wintypes.LPCWSTR(str(target)),
            se_file_object,
            owner_and_dacl,
            ctypes.byref(owner),
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
    if rc != 0:
        return None
    try:

        def sid_string(sid: Any) -> str | None:
            text = wintypes.LPWSTR()
            if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
                return None
            try:
                return text.value
            finally:
                kernel32.LocalFree(text)

        owner_sid = sid_string(owner)
        if owner_sid is None:
            return None
        if not dacl.value:
            return WindowsSecurity(owner=owner_sid, aces=None)

        class _AclSize(ctypes.Structure):
            _fields_ = [
                ("count", wintypes.DWORD),
                ("in_use", wintypes.DWORD),
                ("free", wintypes.DWORD),
            ]

        class _Ace(ctypes.Structure):
            # ACE_HEADER + Mask; the SID starts at ``sid_start``.
            _fields_ = [
                ("ace_type", ctypes.c_ubyte),
                ("flags", ctypes.c_ubyte),
                ("size", wintypes.WORD),
                ("mask", wintypes.DWORD),
                ("sid_start", wintypes.DWORD),
            ]

        size = _AclSize()
        if not advapi32.GetAclInformation(dacl, ctypes.byref(size), ctypes.sizeof(size), 2):
            return None
        aces: list[WindowsAce] = []
        for index in range(size.count):
            pointer = ctypes.c_void_p()
            if not advapi32.GetAce(dacl, index, ctypes.byref(pointer)):
                return None
            ace = ctypes.cast(pointer, ctypes.POINTER(_Ace)).contents
            sid = ""
            if ace.ace_type in _WINDOWS_SIMPLE_ACE_TYPES:
                address = (pointer.value or 0) + _Ace.sid_start.offset
                sid = sid_string(ctypes.c_void_p(address)) or ""
            aces.append(WindowsAce(ace.ace_type, ace.flags, ace.mask, sid))
        return WindowsSecurity(owner=owner_sid, aces=tuple(aces))
    finally:
        kernel32.LocalFree(descriptor)


def _windows_anchor_refusal(path: Path, descriptor: int) -> str | None:
    """Run :func:`windows_acl_refusal` over the anchor and the directories lop owns.

    WHICH DIRECTORIES, and why it stops there: the file, ``operators`` and
    ``local-operator`` are what the elevated install creates and hardens, and a
    principal that can write, delete or re-ACL any of them can replace the anchor.
    The ProgramData root above them is the system's: its default ACL lets Users
    add entries but not delete or rename an existing child (no DELETE_CHILD), so
    a hardened ``local-operator`` cannot be swapped from there — and checking the
    root would refuse every stock install for an ACE that cannot reach the file.
    """
    targets: list[tuple[str, Path | int]] = [
        ("anchor", descriptor),
        ("anchor directory", path.parent),
        ("anchor's parent directory", path.parent.parent),
    ]
    for what, target in targets:
        security = _windows_security(target)
        if security is None:
            return f"the {what}'s owner and access list could not be read"
        refusal = windows_acl_refusal(what, security)
        if refusal is not None:
            return refusal
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
        # WHAT THE READER IS TOLD CHANGED HERE (design round 1, D1). "no anchor installed"
        # named the FILE; this names the CONSEQUENCE, so `lop operator status` can put it
        # on a `reason` line beside the key agent's own fault rather than in place of it —
        # the two are independent facts and one used to overwrite the other.
        return AnchorLoad(
            anchor=None,
            path=path,
            root_owned=False,
            reason="no anchor is installed, so the runtime trusts no key yet",
        )
    descriptor, info = opened
    # Whether an administrator, not the gated user, controls the file. POSIX reads
    # it from ``st_uid``; on Windows ``st_uid`` is always 0 and means nothing, so
    # it is True only once the DACL verifier below has accepted the file.
    owner_ok = info.st_uid == 0
    try:
        if _on_windows():
            # ``st_uid`` is always 0 on Windows, so the POSIX checks below would
            # pass ANY file there; the DACL verifier is the Windows check (F4).
            refusal = _windows_anchor_refusal(path, descriptor)
            if refusal is not None:
                return AnchorLoad(
                    anchor=None, path=path, root_owned=False, reason=refusal, exists=True
                )
            owner_ok = True
        else:
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
            root_owned=owner_ok,
            reason="the anchor is not readable JSON",
            exists=True,
        )
    anchor, refusal = OperatorAnchor._parse(body)
    if anchor is None:
        return AnchorLoad(
            anchor=None,
            path=path,
            root_owned=owner_ok,
            # ``refusal`` carries the ONE specific sentence ``_parse`` gives (the
            # declared-vs-derived key id — F5 slice B); every other refusal keeps
            # the corpus's generic sentence, because none of them names a
            # different next move.
            reason=refusal or "the anchor does not describe an ES256 operator key",
            exists=True,
        )
    return AnchorLoad(anchor=anchor, path=path, root_owned=owner_ok, reason="ok", exists=True)


def anchor_bytes(anchor: OperatorAnchor) -> bytes:
    """The exact bytes ``lop operator install`` writes."""
    return json.dumps(anchor.to_json(), indent=2, sort_keys=True).encode("utf-8") + b"\n"


def statement_digest(anchor: OperatorAnchor) -> str:
    """``sha256:<hex>`` of :func:`anchor_bytes` — the F4b provenance value.

    ONE builder, so the digest an approval mints, the digest the runner
    re-derives before planting, and the digest a receipt records are the same
    computation over the same bytes rather than three agreements in prose.
    """
    digest = hashlib.sha256(anchor_bytes(anchor)).hexdigest()
    return f"sha256:{digest}"


def anchor_trio(anchor: OperatorAnchor) -> dict[str, Any]:
    """The provenance trio ``{key_id, spki_fp, statement_digest}`` for one statement.

    ONE builder (F5): the approval mint records these values, approve re-derives
    them, and the onboarding runner re-derives them once more before planting.
    Three derivations over one function is the design; three AGREEMENTS in prose
    is what let two truncations of the same ``sha256`` ship side by side — the
    card recorded ``94A6-A6C1-110D`` (12 hex, 3 groups) while the runner derived
    ``94A6-A6C1-110D-773B`` (16 hex, 4 groups), so every card that build filed
    was refused ``{"mismatch": ["spki_fp"]}`` before anything was planted.
    ``spki_fp`` here is ``local_operator.operator.verify.spki_fp`` — the value
    ``lop operator anchor export`` prints as "the value to compare" — and the
    digest is :func:`statement_digest` over the canonical bytes, so a card can
    never disagree with the statement it authorises.
    """
    return {
        "key_id": key_id_for(anchor.spki),
        "spki_fp": spki_fp(anchor.spki),
        "statement_digest": statement_digest(anchor),
    }


#: Product words for the trio's machine field names (F5 review, D3): the human
#: clauses say "key fingerprint" where ``data.mismatch`` says ``spki_fp`` — the
#: ledger token is for machines, and the labels match what ``lop operator anchor
#: export`` prints ("key id", "fingerprint"). ONE map so the approve refusal
#: (``network/approvals.py``) and the runner's refusal (``network/onboard.py``)
#: cannot drift apart.
TRIO_LABELS: dict[str, str] = {
    "key_id": "key id",
    "spki_fp": "key fingerprint",
    "statement_digest": "statement digest",
}


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


def load_local_anchor(config_root: Path | None = None) -> OperatorAnchor | None:
    """The local statement a provenance derivation reads, in ONE trust order.

    Installed-and-usable first — it is what the runtime honours — then the
    STAGED statement: the bootstrap artifact between ``init``/``setup`` and the
    privileged install, and the carrier ``devices --revoke``/``--authorise``
    stage before their install consumes it. The staged read resolves against
    ``config_root`` (the ambient config dir when omitted).

    ONE reader, deliberately (F5): the approval mint (``local_anchor_trio``),
    the approve-time comparison, ``verify_for_run`` and the onboarding runner's
    ``step_anchor`` all derive from this function, so "the local store" cannot
    mean two different files on the two sides of one handshake. The precedence
    only matters where the two files differ (a re-init, or a staged device-list
    change before its install) — and there, installed-first is the coherent
    choice: a card records, and a runner ships, what this machine currently
    trusts; a staged candidate becomes readable the moment its own privileged
    install lands it.
    """
    loaded = load_anchor()
    if loaded.usable:
        return loaded.anchor
    return load_staged_anchor(config_root if config_root is not None else config_dir())


#: The privileged half of a POSIX install, run as ``sh -c <this> lop-anchor-install
#: <staged> <target> <sha256-hex> <group>``. Every value arrives as a POSITIONAL
#: argument, never interpolated into the script, so no path or digest is ever
#: re-parsed as shell. WHY IT COPIES, HASHES, THEN RENAMES (the staging-swap
#: race): the staged file is user-writable, so a same-user process could replace
#: it between the moment lop read and showed it and the moment root installs it.
#: Root first copies it into the ROOT-OWNED target directory (where the user can
#: no longer touch the copy), hashes THAT copy, refuses unless it is the exact
#: digest lop computed from the bytes it read, and only then renames it into
#: place — so what lands is byte-for-byte what was validated, atomically.
_POSIX_INSTALL_SCRIPT = (
    "set -eu; "
    'staged="$1"; target="$2"; expected="$3"; group="$4"; '
    'tmp="$(dirname "$target")/.anchor-install.$$"; '
    "trap 'rm -f \"$tmp\"' EXIT; "
    'install -m 0600 -o root -g "$group" "$staged" "$tmp"; '
    'actual="$({hash} "$tmp" | cut -d " " -f 1)"; '
    'if [ "$actual" != "$expected" ]; then '
    "echo 'the staged anchor changed after it was read; nothing was installed' >&2; "
    "exit 3; fi; "
    'chmod 0644 "$tmp"; mv -f "$tmp" "$target"; trap - EXIT'
)

#: Well-known SIDs, spelled as SIDs so the Windows command is locale-independent
#: (``Administrators`` is ``Administratoren`` on a German install).
_SID_SYSTEM, _SID_ADMINS, _SID_USERS = "S-1-5-18", "S-1-5-32-544", "S-1-5-32-545"


def _ps_quote(value: str) -> str:
    """A PowerShell single-quoted literal: no expansion, ``'`` doubled."""
    return "'" + value.replace("'", "''") + "'"


def _windows_install_script(staging: Path, target: Path, digest_hex: str) -> str:
    """The script the ELEVATED PowerShell runs (F4, problem 3).

    The old step was a non-elevated ``Copy-Item`` into ``%PROGRAMDATA%`` that set
    no ACL, so the file inherited ``CREATOR OWNER: F`` and stayed writable by the
    very user it gates. This one, as an administrator: creates both directories,
    strips inheritance and grants Full to SYSTEM and Administrators and read to
    Users, makes Administrators the owner, reads the staged bytes ONCE, refuses
    unless they hash to the digest lop read (the staging-swap race), writes those
    bytes to a hardened temp file inside the hardened directory and renames it
    into place. The result is what :func:`windows_acl_refusal` accepts.
    """
    tmp = target.parent / ".anchor-install.tmp"
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"$root = {_ps_quote(str(target.parent.parent))}",
        f"$dir = {_ps_quote(str(target.parent))}",
        f"$tmp = {_ps_quote(str(tmp))}",
        "function Harden($p, $inherit) {",
        f'  icacls $p /inheritance:r /grant:r "*{_SID_SYSTEM}:$($inherit)F" '
        f'"*{_SID_ADMINS}:$($inherit)F" "*{_SID_USERS}:$($inherit)R" | Out-Null',
        '  if ($LASTEXITCODE -ne 0) { throw "icacls could not restrict $p" }',
        f'  icacls $p /setowner "*{_SID_ADMINS}" | Out-Null',
        '  if ($LASTEXITCODE -ne 0) { throw "icacls could not take ownership of $p" }',
        "}",
        "New-Item -ItemType Directory -Force -Path $dir | Out-Null",
        "Harden $root '(OI)(CI)'",
        "Harden $dir '(OI)(CI)'",
        f"$bytes = [IO.File]::ReadAllBytes({_ps_quote(str(staging))})",
        "$sha = [Security.Cryptography.SHA256]::Create()",
        "$actual = -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') })",
        f"if ($actual -ne {_ps_quote(digest_hex)}) {{ throw 'the staged anchor changed "
        "after it was read; nothing was installed' }",
        "[IO.File]::WriteAllBytes($tmp, $bytes)",
        "Harden $tmp ''",
        f"Move-Item -Force -LiteralPath $tmp -Destination {_ps_quote(str(target))}",
    ]
    return "\n".join(lines)


def install_commands(staging: Path, target: Path, digest_hex: str) -> list[list[str]]:
    """The privileged commands onboarding needs, spelled in ONE place.

    A list of argv lists rather than one joined string, so the caller runs them
    without a shell: there is no quoted value here, and going through a shell
    would add a second place for the path to be reinterpreted. Root ownership is
    the property that makes the anchor an anchor, so owner and group are set
    explicitly rather than left to ``install``'s defaults, which inherit the
    invoking user's umask and group.

    ``digest_hex`` is the sha256 of the staged bytes AS THE CALLER READ THEM; the
    privileged step installs only content with that digest (see
    :data:`_POSIX_INSTALL_SCRIPT` for the race it closes).

    Windows takes one command, which the operator runs and approves at the UAC
    prompt — lop never runs it itself. The elevated script travels as
    ``-EncodedCommand`` (UTF-16LE base64) inside the argv rather than as a script
    file, because a script file would have to live somewhere the user can write,
    which reopens the same swap race for the script itself.
    """
    if _on_windows():
        import base64

        script = _windows_install_script(staging, target, digest_hex)
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return [
            [
                "powershell",
                "-NoProfile",
                "-Command",
                # ``-PassThru`` + ``exit``: ``Start-Process -Wait`` alone drops the
                # elevated child's exit code, so a refused digest would print as
                # success.
                "$p = Start-Process -FilePath powershell -Verb RunAs -Wait -PassThru "
                f"-ArgumentList '-NoProfile -NonInteractive -EncodedCommand {encoded}'; "
                "exit $p.ExitCode",
            ]
        ]
    group = "wheel" if sys.platform == "darwin" else "root"
    hasher = "shasum -a 256" if sys.platform == "darwin" else "sha256sum"
    script = _POSIX_INSTALL_SCRIPT.replace("{hash}", hasher)
    return [
        ["sudo", "install", "-d", "-m", "0755", "-o", "root", "-g", group, str(target.parent)],
        [
            "sudo",
            "sh",
            "-c",
            script,
            "lop-anchor-install",
            str(staging),
            str(target),
            digest_hex,
            group,
        ],
    ]


def format_install_commands(command: list[list[str]]) -> str:
    """The commands as ONE line an operator (or ``sh -c``) can run as printed.

    Quoted per argument, because the macOS target contains a space
    (``Application Support``) and the install step now carries a script; the
    remote-onboarding runner executes this exact line under ``sh -c``.
    Windows prints its single PowerShell command with Windows quoting.
    """
    if _on_windows():
        import subprocess

        return " && ".join(subprocess.list2cmdline(argv) for argv in command)
    import shlex

    return " && ".join(shlex.join(argv) for argv in command)


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
