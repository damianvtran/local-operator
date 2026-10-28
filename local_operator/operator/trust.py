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
import ntpath
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

#: The ``admission`` value that means "a device's certificate counts only if the
#: anchor LISTS it". Spelled as a positive string rather than a boolean because it
#: is a POLICY NAME: the file records which rule this host was set up under, and a
#: future rule (a per-approval gesture, a Windows Hello key) is a new value rather
#: than a second boolean whose interaction with the first has to be reasoned about.
#:
#: ONLY THE ROOT WRITE SETS IT. The confirm flow is the one writer, and it writes
#: it in the same privileged step that lists the phones the human kept — so an
#: agent that can write its own anchor (file-only hosts) still cannot turn
#: enforcement ON for itself, and one that can read the file key cannot turn it
#: OFF, because turning it off needs the same root write.
ADMISSION_LISTED = "listed"

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


def _system_windows_directory() -> str | None:  # pragma: no cover — Windows only
    """``C:\\Windows`` from ``GetSystemWindowsDirectoryW`` — the ENV-FREE drive source.

    It answers from the kernel's shared user data (``NtSystemRoot``), which no
    process environment variable feeds, so it is the reference the Known Folder
    answer is checked against (R1-1).
    """
    buffer = ctypes.create_unicode_buffer(261)
    try:
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        length = kernel32.GetSystemWindowsDirectoryW(buffer, len(buffer))
    except (OSError, AttributeError):
        return None
    if not 0 < length < len(buffer):
        return None
    return buffer.value or None


def _known_folder_program_data() -> str | None:  # pragma: no cover — Windows only
    """``SHGetKnownFolderPath(FOLDERID_ProgramData)``, verbatim, or ``None``."""

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
        return str(out.value)
    finally:
        ole32.CoTaskMemFree(out)


def resolve_program_data(
    system_windows_dir: str | None, known_folder: str | None
) -> tuple[Path | None, str]:
    """``<system drive>\\ProgramData`` when both sources agree, else ``None`` + why.

    WHY TWO SOURCES (R1-1). ``SHGetKnownFolderPath`` is not answered purely from
    ``HKLM``: the registry value it reads is ``%SystemDrive%\\ProgramData``
    (``REG_EXPAND_SZ``), expanded against the CALLING PROCESS's environment — and
    ``SystemDrive`` is a variable a same-user process can set for lop. The drive
    from ``GetSystemWindowsDirectoryW`` is env-free, so it pins the expected
    root; the Known Folder answer only CONFIRMS it. Any disagreement, or either
    source missing, refuses (fail closed: the anchor loads as "not installed"
    with this reason) rather than picking one — a mismatch means something is
    steering the lookup, and a host that relocated ProgramData off the system
    drive is rare enough to be told so plainly.

    Pure over its two strings, so the Ubuntu suite drives every branch.
    """
    if not system_windows_dir:
        return None, "the Windows system directory could not be read"
    drive = ntpath.splitdrive(system_windows_dir)[0]
    if not re.fullmatch(r"[A-Za-z]:", drive):
        return None, f"the Windows system directory {system_windows_dir!r} names no drive letter"
    expected = drive.upper() + "\\ProgramData"
    if not known_folder:
        return None, "the ProgramData folder could not be resolved"
    if ntpath.normcase(ntpath.normpath(known_folder)) != ntpath.normcase(expected):
        return None, (
            f"the ProgramData folder resolves to {known_folder!r}, not {expected!r} on the "
            "system drive, so the runtime will not read an anchor from it (something in "
            "this process's environment may be steering the lookup)"
        )
    return Path(expected), ""


@functools.lru_cache(maxsize=1)
def _windows_program_data_resolution() -> tuple[Path | None, str]:  # pragma: no cover
    """The resolved root and its refusal reason, cached for the process.

    Cached because the folder does not move under a running process and
    ``anchor_dir`` is read on every anchor refresh.
    """
    return resolve_program_data(_system_windows_directory(), _known_folder_program_data())


def _windows_program_data() -> Path | None:
    """``C:\\ProgramData`` as the OS knows it, never as the environment says.

    WHY NOT A CONSTANT OR ``%PROGRAMDATA%`` (F4): ``%PROGRAMDATA%`` is a value the
    gated subject writes for its own tool subprocesses, so reading it let a
    same-user process point the runtime at a directory it owns. A hard-coded
    ``C:\\ProgramData`` would be env-proof but wrong on hosts whose system drive
    is not ``C:``. :func:`resolve_program_data` derives it from the env-free
    system drive and cross-checks the Known Folder. ``None`` on any failure or
    disagreement; the caller reads :data:`_UNRESOLVED_ROOT` — never a fallback.
    """
    return _windows_program_data_resolution()[0]


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
    #: ``""`` (the default) means this host has NOT adopted the admission rule and
    #: every device the certificate check accepts stays accepted — exactly the
    #: behaviour of every anchor written before this field existed. ``"listed"``
    #: means the runtime additionally requires the anchor to name the device (see
    #: :func:`device_is_admitted`).
    #:
    #: DORMANT BY DEFAULT, ON PURPOSE. Turning the rule on for every host in the
    #: build that introduced it would de-admit every phone already paired on a
    #: file-only host — they are not listed, because nothing ever listed them — and
    #: send those humans through a re-pair to restore access they never lost. The
    #: switch is therefore written by the one flow that also lists the phones the
    #: human keeps (the confirm card), and until a host runs it nothing here moves.
    admission: str = ""

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {
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
        # EMITTED ONLY WHEN SET, and that is a compatibility requirement rather
        # than tidiness: ``anchor_bytes`` is what ``statement_digest`` hashes, and
        # every live approval card pins that digest in its provenance trio. Always
        # emitting ``"admission": ""`` would change the bytes of every anchor in
        # existence and make every card filed before this build refuse with
        # ``approval_anchor_mismatch``. An anchor with no marker must serialise to
        # the SAME bytes it did before this field existed.
        if self.admission:
            body["admission"] = self.admission
        return body

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
        # AN ABSENT ``admission`` KEY IS THE NORMAL CASE, not a malformation: every
        # anchor written before this field existed lacks it, and those hosts must
        # keep parsing (and keep behaving as they do today). A present but
        # non-string value is likewise read as "no marker" rather than refused: the
        # field is a policy NAME, and a runtime that refuses the whole anchor over
        # one it does not recognise would take operator authority away from a host
        # whose key is perfectly good. Under-enforcement is reported instead
        # (``lop operator status`` says admission is not enforced), so an
        # unrecognised value is visible rather than silently claimed as active.
        admission = body.get("admission")
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
                admission=admission if isinstance(admission, str) else "",
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
            info = os.lstat(current)
            if stat.S_ISLNK(info.st_mode):
                return False
            # R1-5: a Windows directory JUNCTION is a mount-point reparse point,
            # which ``S_ISLNK`` does not report, yet the kernel follows it like a
            # link. Refuse every reparse point: the anchor's path never needs one.
            if getattr(info, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
                return False
        except OSError:
            # A missing component is not a link: the load below reports it as
            # absent, and refusing here would report "redirected" for a host
            # that simply has not installed an anchor yet.
            return True
    return True


#: ``FILE_ATTRIBUTE_REPARSE_POINT``: set on symlinks AND junctions (``os.lstat``
#: reports it in ``st_file_attributes`` on Windows only).
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


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


def _windows_security(target: Path | int) -> WindowsSecurity | None:
    """Owner SID and DACL of ``target``, or ``None`` — this NEVER raises (R1-6).

    ``load_anchor`` promises never to raise, and the ctypes reader below can:
    ``WinDLL`` loading, ``msvcrt.get_osfhandle`` on a bad descriptor and the
    ctypes conversions all throw rather than return an error code. Every such
    failure becomes "could not be read", which the caller refuses — a reader
    that crashes the anchor refresh would take the runtime down with it.
    """
    try:
        return _read_windows_security(target)
    except Exception:  # noqa: BLE001 — fail closed on ANY reader fault, see above
        return None


def _read_windows_security(  # pragma: no cover — Windows only; the windows-latest job runs it
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
    if _on_windows() and _ANCHOR_ROOT_OVERRIDE is None and _windows_program_data() is None:
        # R1-1: the root could not be pinned (or the Known Folder disagreed with
        # the env-free system drive). "No anchor", with the reason named, so the
        # status line says WHY rather than "not installed" on a host that has one.
        return AnchorLoad(
            anchor=None,
            path=path,
            root_owned=False,
            reason=_unresolved_root_reason(),
        )
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
            reason="a component of the anchor path is a symbolic link or junction",
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
    return parse_staged_anchor(raw)


def parse_staged_anchor(raw: bytes) -> OperatorAnchor | None:
    """Parse staged-anchor BYTES the caller already read, or ``None``.

    Split from :func:`load_staged_anchor` so the install can judge, hash and
    install ONE read (QA Q3): reading the file again for each would let a swap
    between the reads pass a benign copy to the guard and install another.
    """
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


#: Every character PowerShell's tokenizer reads as a single quote: ASCII ``'``
#: AND the typographic U+2018..U+201B (R2-3). Doubling only ``'`` left
#: ``o\u2019brien`` able to close the literal, and these values (the staged path,
#: the target) reach the ELEVATED script.
_PS_SINGLE_QUOTES = "'\u2018\u2019\u201a\u201b"


def _ps_quote(value: str) -> str:
    """A PowerShell single-quoted literal: no expansion, every single quote doubled.

    Each quote character is doubled AS ITSELF, so the literal's value is the
    input unchanged — PowerShell un-doubles any pair of quote-class characters.
    """
    return "'" + re.sub(f"([{_PS_SINGLE_QUOTES}])", r"\1\1", value) + "'"


#: The only directory shape the elevated script will touch (R2-1, defence in
#: depth): an absolute drive-letter path ending ``\ProgramData\local-operator\
#: operators``, with no ``.``/``..`` segment. The Python side already refuses an
#: unresolved root (:func:`install_refusal`); this is what still holds if a
#: future caller forgets. Without it the unresolved root
#: ``\nonexistent-local-operator-anchor`` made ``$root`` the DRIVE ROOT, and the
#: administrator's approval would have replaced ``C:\``'s DACL. A SUFFIX, not
#: the exact system path, because the native suite installs into scratch trees
#: and must keep exercising this same script. Valid for Python's ``re`` and
#: .NET's regex alike (``-match`` is case-insensitive, as the tests use it).
_WINDOWS_ROOT_PATTERN = (
    r"[A-Za-z]:\\(?:(?!\.\.?\\)[^\\/:*?\"<>|]+\\)*ProgramData\\local-operator\\operators"
)

#: Runs before any native call: ``$dir`` must match the pattern, ``$root`` must be
#: exactly its parent and ``$target`` exactly one name inside it, so the three
#: values cannot disagree about where the install lands.
_WINDOWS_ROOT_GUARD = (
    f"if(($dir -notmatch '^{_WINDOWS_ROOT_PATTERN}$') -or "
    "($root -ne $dir.Substring(0,$dir.Length-10)) -or "
    "($target.Substring(0,[Math]::Max(0,$target.LastIndexOf('\\'))) -ne $dir)){"
    'throw "refusing to install into $dir, which is not a ProgramData\\local-operator'
    '\\operators folder; nothing was installed"}'
)

#: The elevated script's native calls, declared through ``Reflection.Emit`` rather
#: than ``Add-Type``: ``Add-Type`` compiles C# through files in the ELEVATED
#: user's ``%TEMP%``, which that same (gated) user can write, so the code the
#: administrator runs could be swapped mid-compile. Emit builds the P/Invoke
#: stubs in memory; nothing the user can touch is ever loaded. The security
#: descriptor is written with ``SetKernelObjectSecurity`` on the held handle, NOT
#: ``SetSecurityInfo``: the latter re-propagates inheritable ACEs into existing
#: children and would walk through a planted child junction (the R1-2 deputy).
_WINDOWS_PINVOKE = """$tb=[Reflection.Emit.AssemblyBuilder]::DefineDynamicAssembly([Reflection.AssemblyName]::new('LopAnchor'),'Run').DefineDynamicModule('LopAnchor').DefineType('LopAnchorNative','Public,Class')
$di=[Runtime.InteropServices.DllImportAttribute]
function LopDef($dll,$name,$ret,$params){$m=$tb.DefineMethod($name,'Public,Static,PinvokeImpl',$ret,[Type[]]$params)
$m.SetCustomAttribute([Reflection.Emit.CustomAttributeBuilder]::new($di.GetConstructor([Type[]]@([string])),[object[]]@($dll),[Reflection.FieldInfo[]]@($di.GetField('SetLastError'),$di.GetField('CharSet')),[object[]]@($true,[Runtime.InteropServices.CharSet]::Unicode)))}
$I=[IntPtr];$U=[uint32]
LopDef kernel32 CreateFileW $I @([string],$U,$U,$I,$U,$U,$I)
LopDef kernel32 CreateDirectoryW ([bool]) @([string],$I)
LopDef kernel32 GetFileInformationByHandle ([bool]) @($I,[byte[]])
LopDef kernel32 CloseHandle ([bool]) @($I)
LopDef kernel32 MoveFileExW ([bool]) @([string],[string],$U)
LopDef kernel32 LocalFree $I @($I)
LopDef advapi32 GetSecurityInfo $U @($I,$U,$U,$I,$I,$I,$I,$I.MakeByRefType())
LopDef advapi32 GetSecurityDescriptorLength $U @($I)
LopDef advapi32 SetKernelObjectSecurity ([bool]) @($I,$U,[byte[]])
$LopNative=$tb.CreateType()
function LopErr{[Runtime.InteropServices.Marshal]::GetLastWin32Error()}"""  # noqa: E501

#: The elevated install's body. Every object is OPENED FIRST without following a
#: reparse point (``FILE_FLAG_OPEN_REPARSE_POINT``), refused if the HANDLE says it
#: is a junction/symlink, and then checked and re-ACL'd THROUGH THAT HANDLE, so
#: no path is re-resolved between the check and the write (R1-2). Directories
#: are held open without ``FILE_SHARE_DELETE`` until the install ends, so they
#: cannot be renamed or swapped underneath it, and closed in ``finally`` on every
#: path, refusals included (R2-N2). Every helper is ``Lop``-prefixed
#: because PowerShell resolves an ALIAS before a function: a helper named ``Dir``
#: would silently run ``Get-ChildItem``. Variables are case-INSENSITIVE too
#: (a local ``$n`` once shadowed the native type ``$N``), hence ``$LopNative``.
#: A PREVIOUS anchor file is refused only if it is a reparse point, not for its
#: owner: it is replaced by rename inside the now-hardened directory, never
#: written through, and an anchor left by the old non-elevated ``Copy-Item``
#: install (user-owned) must stay re-installable.
_WINDOWS_INSTALL_BODY = """$Trusted='S-1-5-18','S-1-5-32-544'
function LopBytes($sddl){$r=[Security.AccessControl.RawSecurityDescriptor]::new($sddl);$b=[byte[]]::new($r.BinaryLength);$r.GetBinaryForm($b,0);,$b}
function LopSA($b){$Mar=[Runtime.InteropServices.Marshal];$p=$Mar::AllocHGlobal($b.Length);$Mar::Copy($b,0,$p,$b.Length);$z=[IntPtr]::Size;$a=$Mar::AllocHGlobal(3*$z);$Mar::WriteInt32($a,0,3*$z);$Mar::WriteIntPtr($a,$z,$p);$Mar::WriteInt32($a,2*$z,0);$a}
function LopOpen($path,$access,$disp,$share,$sa,$dir,$missingOk){
if($missingOk -and -not [IO.Directory]::GetFileSystemEntries([IO.Path]::GetDirectoryName($path),[IO.Path]::GetFileName($path)).Count){return $null}
$h=$LopNative::CreateFileW($path,$access,$share,$sa,$disp,0x02200000,[IntPtr]::Zero)
if($h.ToInt64() -eq -1){throw "could not open $path without following links (error $(LopErr)); nothing was installed"}
$info=[byte[]]::new(52)
if(-not $LopNative::GetFileInformationByHandle($h,$info)){[void]$LopNative::CloseHandle($h);throw "could not read the attributes of $path; nothing was installed"}
$at=[BitConverter]::ToUInt32($info,0)
if($at -band 0x400){[void]$LopNative::CloseHandle($h);throw "$path is a junction or symbolic link, so it was refused; nothing was installed"}
if($dir -and -not ($at -band 0x10)){[void]$LopNative::CloseHandle($h);throw "$path is not a directory; nothing was installed"}
$h}
function LopSd($h,$what){$p=[IntPtr]::Zero
$rc=$LopNative::GetSecurityInfo($h,1,5,[IntPtr]::Zero,[IntPtr]::Zero,[IntPtr]::Zero,[IntPtr]::Zero,[ref]$p)
if($rc -ne 0){throw "could not read the owner and access list of $what (error $rc); nothing was installed"}
try{$len=$LopNative::GetSecurityDescriptorLength($p);$b=[byte[]]::new($len);[Runtime.InteropServices.Marshal]::Copy($p,$b,0,$len);[Security.AccessControl.RawSecurityDescriptor]::new($b,0)}finally{[void]$LopNative::LocalFree($p)}}
function LopOwned($sd,$what){if($Trusted -notcontains $sd.Owner.Value){throw "$what is owned by $($sd.Owner.Value), not SYSTEM or Administrators, so it was refused; an administrator should remove it, then run this again"}}
function LopVerify($sd,$what){LopOwned $sd $what
if($null -eq $sd.DiscretionaryAcl){throw "$what has no access list"}
foreach($a in $sd.DiscretionaryAcl){
if($a -isnot [Security.AccessControl.CommonAce]){throw "$what carries an access entry of an unknown kind"}
if($a.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly){continue}
if($a.AceQualifier -ne [Security.AccessControl.AceQualifier]::AccessAllowed){continue}
if(($Trusted -notcontains $a.SecurityIdentifier.Value) -and ($a.AccessMask -band 0x520d0156)){throw "$what is still writable by $($a.SecurityIdentifier.Value)"}}}
$DirSd=LopBytes 'O:BAD:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;0x1200a9;;;BU)'
$FileSd=LopBytes 'O:BAD:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;0x1200a9;;;BU)'
function LopDir($path){
[void]$LopNative::CreateDirectoryW($path,(LopSA $DirSd))
$h=LopOpen $path 0xE0080 3 3 ([IntPtr]::Zero) $true $false
try{LopOwned (LopSd $h $path) $path
if(-not $LopNative::SetKernelObjectSecurity($h,[uint32]5,$DirSd)){throw "could not restrict $path (error $(LopErr)); nothing was installed"}
LopVerify (LopSd $h $path) $path}catch{[void]$LopNative::CloseHandle($h);throw}
$h}
$hr=$null;$hd=$null
try{
$hr=LopDir $root
$hd=LopDir $dir
$old=LopOpen $target 0x80 3 7 ([IntPtr]::Zero) $false $true
if($old){[void]$LopNative::CloseHandle($old)}
$stale=LopOpen $tmp 0x10080 3 7 ([IntPtr]::Zero) $false $true
if($stale){[void]$LopNative::CloseHandle($stale);[IO.File]::Delete($tmp)}
$bytes=[IO.File]::ReadAllBytes($staged)
$actual=-join([Security.Cryptography.SHA256]::Create().ComputeHash($bytes)|ForEach-Object{$_.ToString('x2')})
if($actual -ne $expected){throw 'the staged anchor changed after it was read; nothing was installed'}
$hf=LopOpen $tmp 0x40020080 1 0 (LopSA $FileSd) $false $false
$fs=[IO.FileStream]::new([Microsoft.Win32.SafeHandles.SafeFileHandle]::new($hf,$true),[IO.FileAccess]::Write)
try{$fs.Write($bytes,0,$bytes.Length);$fs.Flush($true);LopVerify (LopSd $hf $tmp) $tmp}finally{$fs.Dispose()}
if(-not $LopNative::MoveFileExW($tmp,$target,9)){throw "could not move the anchor into place (error $(LopErr))"}
$hv=LopOpen $target 0x20080 3 7 ([IntPtr]::Zero) $false $false
try{LopVerify (LopSd $hv $target) $target}finally{[void]$LopNative::CloseHandle($hv)}
}finally{foreach($x in $hd,$hr){if($x){[void]$LopNative::CloseHandle($x)}}}
Write-Output "installed the operator anchor at $target\""""  # noqa: E501


def _windows_install_script(staging: Path, target: Path, digest_hex: str) -> str:
    """The script the ELEVATED PowerShell runs (F4 problem 3; R1-2).

    The old step was a non-elevated ``Copy-Item`` that set no ACL. The round-1
    step ran ``icacls`` on whatever already sat at the target paths, so a user
    who pre-created ``local-operator\\`` (ProgramData lets Users add children)
    could make ``operators`` a JUNCTION and have the administrator's approval
    re-ACL an arbitrary directory, or leave an explicit grant ``/grant:r`` never
    removes. This one, as an administrator:

    - creates each directory WITH its hardened descriptor (``CreateDirectoryW``
      + ``SECURITY_ATTRIBUTES``), so a fresh one is never briefly open;
    - opens each directory, any pre-existing anchor and the temp file without
      following a reparse point, and REFUSES a junction/symlink or a directory
      owned by anyone but SYSTEM/Administrators (the handle is what is checked);
    - REPLACES the whole DACL through that handle (protected; SYSTEM and
      Administrators full, Users read) rather than editing it, so no foreign
      explicit ACE survives — and re-reads it through the handle to prove it;
    - hashes the staged bytes once, refuses unless they match ``digest_hex``
      (the staging-swap race), writes them to a ``CREATE_NEW`` temp file born
      with the hardened descriptor, and renames it into place.

    No ``icacls`` at all any more, so there is no link-following path tool to
    pass ``/L`` to. The result is what :func:`windows_acl_refusal` accepts.
    """
    tmp = target.parent / ".anchor-install.tmp"
    head = "\n".join(
        [
            "$ErrorActionPreference='Stop'",
            # R2-N1: module auto-load's progress record otherwise reaches the
            # redirected output as CLIXML after the success line.
            "$ProgressPreference='SilentlyContinue'",
            f"$root={_ps_quote(str(target.parent.parent))}",
            f"$dir={_ps_quote(str(target.parent))}",
            f"$target={_ps_quote(str(target))}",
            f"$tmp={_ps_quote(str(tmp))}",
            f"$staged={_ps_quote(str(staging))}",
            f"$expected={_ps_quote(digest_hex)}",
        ]
    )
    # ``try``/``catch`` around everything: ``throw`` from a redirected
    # ``-EncodedCommand`` session reaches stderr as CLIXML, wrapped at console
    # width, so a refusal would read as XML noise. The plain message on stderr
    # and exit code 1 are what the operator (and the launcher's exit code) see.
    return "\n".join(
        [
            head,
            "try{",
            _WINDOWS_ROOT_GUARD,
            _WINDOWS_PINVOKE,
            _WINDOWS_INSTALL_BODY,
            "}catch{[Console]::Error.WriteLine($_.Exception.Message);exit 1}",
        ]
    )


#: Characters the outer launcher must never contain, because one of the two
#: shells an operator pastes into would rewrite them: PowerShell expands ``$``
#: and the backtick escape inside a double-quoted argument; cmd.exe expands
#: ``%VAR%`` (and ``!VAR!`` under delayed expansion); a ``"`` would end the quoted
#: argument in both. Asserted on every build of the line (R1-3).
_PASTE_UNSAFE = frozenset('$`"%!')


def _windows_outer_command(script: str, powershell: str) -> str:
    """The non-elevated launcher: unpack ``script`` in memory, run it elevated.

    WHY IT IS SPELLED THIS WAY (R1-3). Round 1 printed
    ``-Command "$p = Start-Process ...; exit $p.ExitCode"``: correct from cmd.exe,
    but pasted into PowerShell the OUTER shell expanded ``$p`` to nothing before
    launching, so the install silently never ran. The launcher is now ONE
    expression with no variable at all and no character either shell rewrites
    (:data:`_PASTE_UNSAFE`), so the same printed line runs from both.

    WHY NOT ``-EncodedCommand`` FOR THE OUTER STEP TOO, as the review suggested:
    that is UTF-16 base64 of a script that already carries the inner script, and
    the result (~9.3k characters) is past cmd.exe's 8191-character line limit.
    The inner script rides as a gzip+base64 blob (no unsafe character) and the
    launcher builds the elevated ``-EncodedCommand`` from it at run time, in
    memory — never a file the user could swap. ``.ExitCode`` of the
    ``-PassThru`` process is the launcher's own exit code, so a refusal is
    not reported as success (``Start-Process -Wait`` alone drops it).
    """
    import base64
    import gzip

    blob = base64.b64encode(gzip.compress(script.encode("utf-8"), 9, mtime=0)).decode("ascii")
    outer = (
        f"exit (Start-Process -FilePath {_ps_quote(powershell)} -Verb RunAs -Wait -PassThru "
        "-ArgumentList "
        "('-NoProfile -NonInteractive -EncodedCommand '+[Convert]::ToBase64String("
        "[Text.Encoding]::Unicode.GetBytes([IO.StreamReader]::new("
        "[IO.Compression.GZipStream]::new([IO.MemoryStream]::new("
        f"[Convert]::FromBase64String('{blob}')),"
        "[IO.Compression.CompressionMode]::Decompress)).ReadToEnd())))).ExitCode"
    )
    unsafe = _PASTE_UNSAFE.intersection(outer)
    if unsafe:  # pragma: no cover — a constant-shape guard; the test drives quoting
        raise ValueError(f"the Windows install line would carry {sorted(unsafe)}")
    return outer


def windows_powershell_path() -> str | None:
    """``<windir>\\System32\\WindowsPowerShell\\v1.0\\powershell.exe``, never by PATH.

    WHY A FULL PATH (R2-4): the operator pastes the line into a NON-elevated
    shell, which resolves a bare ``powershell`` through the user's own (HKCU-
    writable) PATH — so the binary the UAC prompt elevates could be the gated
    user's. The directory comes from ``GetSystemWindowsDirectoryW`` (env-free,
    the same source R1-1 pins the root with), never ``%SystemRoot%``.

    ``None`` unless every segment is plain ``[A-Za-z0-9._-]``: the path is
    embedded LITERALLY in the printed line, and there is no spelling of a path
    with a space (or ``%``/``$``/quote) that both cmd.exe and PowerShell run as a
    bare first token — PowerShell needs ``& "..."``, which cmd.exe rejects.
    Refusing is the honest answer on such a host; no default install has one.
    """
    windows_dir = _system_windows_directory()
    if not windows_dir or not re.fullmatch(r"[A-Za-z]:(?:\\[A-Za-z0-9._-]+)+", windows_dir):
        return None
    return windows_dir + "\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"


def _unresolved_root_reason() -> str:
    """Why :func:`anchor_dir` answered :data:`_UNRESOLVED_ROOT`, in one sentence."""
    if _on_windows():
        return (
            _windows_program_data_resolution()[1] or "the ProgramData folder could not be resolved"
        )
    return "this platform has no location for the operator anchor"


def install_refusal() -> str | None:
    """Why no install command can be produced on this host, or ``None``.

    R2-1: with the root unresolved, the target is under
    ``\\nonexistent-local-operator-anchor``, whose parent is the DRIVE ROOT —
    the printed elevated command would have re-ACL'd ``C:\\``. The load side
    already reads that state as "no anchor"; the install side must not print a
    privileged command for it. On Windows an unlocatable PowerShell (R2-4) is
    the same kind of refusal. Every install path asks this FIRST.
    """
    if anchor_dir() == _UNRESOLVED_ROOT:
        return _unresolved_root_reason()
    if _on_windows() and windows_powershell_path() is None:
        return (
            "Windows PowerShell could not be located from the system directory "
            f"({_system_windows_directory()!r}), so there is no safe command to print"
        )
    return None


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
        powershell = windows_powershell_path()
        if powershell is None:
            raise ValueError(install_refusal() or "Windows PowerShell could not be located")
        script = _windows_install_script(staging, target, digest_hex)
        outer = _windows_outer_command(script, powershell)
        return [[powershell, "-NoProfile", "-NonInteractive", "-Command", outer]]
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
    return device_id in revoked_device_ids(anchor)


def revoked_device_ids(anchor: OperatorAnchor) -> list[Any]:
    """The anchor's revoked device ids, RAW — exactly as the runtime compares them.

    ONE definition for the runtime (:func:`device_is_revoked`) and the install
    guard that judges whether a write admits a device (R2-2): the guard used to
    ``str()`` the ids while the runtime compared raw values, so a revoked
    ``"1234"`` re-staged as the JSON number ``1234`` looked unchanged to the
    guard yet revoked nothing at runtime. A list, not a set: JSON values need not
    be hashable, and ``in`` on a list is the same ``==`` the runtime used.
    """
    return [e.get("device_id") for e in anchor.devices if e.get("revoked")]


def open_device_ids(anchor: OperatorAnchor) -> list[Any]:
    """The anchor's listed, NOT revoked device ids, raw (see :func:`revoked_device_ids`)."""
    return [e.get("device_id") for e in anchor.devices if not e.get("revoked")]


def admission_enforced(anchor: OperatorAnchor) -> bool:
    """Whether this host's anchor adopted the ADMISSION rule (see ``OperatorAnchor.admission``).

    ONE reading of the field, so a status line and the runtime cannot disagree
    about whether listing is being required — a report that says "enforced" beside
    a runtime that is not looking at the list is worse than the dormant state it
    claims to have left.
    """
    return anchor.admission == ADMISSION_LISTED


def device_is_admitted(anchor: OperatorAnchor, device_id: str, spki: bytes) -> bool:
    """Whether the anchor ADMITS this device — not merely fails to revoke it.

    THE ONE PREDICATE THE THREE CALLERS SHARE (the runtime's certificate point,
    both arms of it, and the relay's command path). Revocation was the only
    question a certificate had to answer, so an operator-signed certificate naming
    a phone this host had never been told about was accepted — which is the whole
    gap the two-lock pairing design closes: the six-digit code proves WHICH phone
    is asking, and the computer's own admin gesture is what proves a PERSON
    admitted it.

    THREE CONDITIONS, each of which a caller cannot add or drop without this
    function changing:

    * **not revoked** — today's rule, unchanged at every level. A revoked device is
      refused whether or not the host lists devices.
    * **the id IS the key's id** (``device_id == key_id_for(spki)``).
      ``verify_device_cert`` checks the domain, the operator signature and the
      expiry, and ``read_device_cert`` takes ``device_id`` from the body AS GIVEN,
      so a statement naming an admitted id while carrying a different key verifies
      — measured in the S4a spike: signed with the file key, it verified, and the
      id did not derive from the ``spki`` it carried. A list keyed on the id alone
      would therefore be satisfied by a forged statement; the binding is what makes
      the id mean the key that will actually sign the frames. It refuses nothing
      legitimate: ``devices.new_device_id`` IS this same derivation and it is the
      only way a certificate's id is ever chosen (``lop pair`` mints through it),
      so only a hand-forged statement can fail the check.
    * **the anchor lists it**, unless this host does not need a list. TWO exemptions,
      both facts written by a root-owned step rather than by anything the gated uid
      can reach: ``presence`` (a backend that costs a human gesture per signature —
      the signature IS the confirmation, so asking for a second touch buys no
      security) and a MISSING marker (the dormant switch, below).

    THE DORMANT SWITCH IS THE MIGRATION, and it is honest only because it is
    reported: hosts whose anchor carries no ``admission`` marker behave EXACTLY as
    they do today, including anchors written before this field existed (which
    serialise to identical bytes, so no pinned ``statement_digest`` moves). A phone
    paired before the upgrade therefore stays admitted after it with no re-pair.
    What that costs is stated where the operator reads it — ``lop operator status``
    says admission is not enforced yet on such a host — rather than being implied to
    be a boundary this slice does not yet deliver.

    ``presence: true`` with an UNLISTED device is admitted BY DESIGN (see above):
    the presence backends are the ones whose signature already costs a gesture.
    """
    if device_is_revoked(anchor, device_id):
        return False
    if not spki or device_id != key_id_for(spki):
        return False
    if anchor.presence:
        return True
    if not admission_enforced(anchor):
        return True
    return device_id in open_device_ids(anchor)


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
