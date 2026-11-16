"""Verified desktop answers: the runtime checks the APP'S CODE SIGNATURE, not a key.

WHY THIS EXISTS (F2; design note "card answers verified by the app's code
signature", 2026-11-16). ``asks/surfaces`` stamps ``desktop`` on an ask answer
only for a frame that proved the runtime's in-memory operator capability. That
capability lives in whichever process STARTED the runtime, so a runtime started
by the wake supervisor (or a peer, or a model's ``bash``) gives the desktop app
nothing to prove with, and the operator's real click on a card is recorded as
``unverified`` (incident a-48e3). Handing the capability to ``lop serve`` would
not help: serve is unsigned Python in a user-writable venv, so same-uid code can
edit it and inherit whatever it holds.

The one process whose code the KERNEL binds to a signature is the Electron
main process of the signed app, and it already sees every card answer
(renderer -> ``desktop-request`` IPC -> main -> serve). So, before it forwards
an answer, main dials THIS module's per-runtime AF_UNIX listener and
ATTESTS the exact answer content. The runtime identifies the dialler by its
kernel audit token and Security.framework, never by anything the dialler says,
and records the attested content in a short-lived single-use table. When
serve's ordinary ``ask_respond`` frame then arrives with the SAME content, the
stamp is ``desktop``. There is no bearer token anywhere: a same-uid process has
nothing to steal or replay, and a tampered serve that swaps "No" for "Yes" sends
content nobody attested.

WHAT "VERIFIED" MEANS (all must hold, see :meth:`AppAttestVerifier.verify`):

1. The peer's audit token (``LOCAL_PEERTOKEN``) names a live process; the token
   carries the pid VERSION, so a recycled pid fails Security.framework's lookup.
2. ``SecCodeCheckValidity(kSecCSStrictValidate)`` against
   :data:`DESIGNATED_REQUIREMENT`: Apple-anchored Developer ID, team
   ``SHA2U6KT7V`` AND identifier ``com.local-operator`` (the team pin alone
   admits the bundled, team-signed ``python3`` seed - measured in the spike),
   AND ``info[LOAttestationEpoch] = "1"``, the downgrade lock: only hardened
   builds (fuses flipped, entitlements tightened) carry that Info.plist key, so
   an old genuine release that still honours ``ELECTRON_RUN_AS_NODE`` fails.
   That lock relies on Info.plist being bound at exec, which amfid does for this
   app because it carries a profile-backed restricted entitlement
   (``keychain-access-groups``). If that entitlement is ever removed,
   re-measure before trusting the epoch.
3. Dynamic code state: ``CS_VALID`` and ``CS_RUNTIME`` set; ``CS_GET_TASK_ALLOW``
   and ``CS_DEBUGGED`` clear; no entitlement that lets same-uid code run inside
   the process (:data:`FORBIDDEN_ENTITLEMENTS`).
4. The BUNDLE the process runs from (``kSecCodeInfoMainExecutable``, a kernel
   vnode path) passes a nested, strict static check, and its Electron fuse wire
   shows node mode, ``NODE_OPTIONS`` and ``--inspect`` off and asar integrity
   on (:data:`REQUIRED_FUSES`). Without this a NEW main executable next to an
   OLD same-team framework would pass step 2.
5. TOCTOU guard for step 4, run on EVERY verify and never cached: every entry
   under ``Contents`` (``os.walk``, symlinks not followed, any unreadable
   directory a refusal) has a ``ctime`` older than the peer's start time.
   Unprivileged code cannot set ``ctime``, an in-place write bumps the file's
   own, and renaming a staged directory into place bumps both the moved
   directory's and its parent's. It covers the WHOLE tree, not just the files
   step 4's fuse read touches, because the cached nested check vouches for all
   of it: helper apps, framework resources, ``app.asar.unpacked`` (agent review
   round 1, R1-M1: a helper overwritten in place after the first check stayed
   ``ok`` while a guard over four files looked on).

Step 4 costs 1.6-3.3 s (one measurement each, warm cache, design note), so its
result is cached per ``(pid, pidversion, bundle)``. That cache is sound only
because step 5 runs after it on every call: the cached check saw the tree as it
was at some moment after launch, and a ctime older than launch on every entry
proves nothing has changed since. The walk measured 2355 entries in 0.04-0.23 s
on the installed 0.57.8 app (5 runs, this Mac, 2026-11-16) and runs in the
listener's worker thread with the rest of ``verify`` (``asyncio.to_thread``),
never on the event loop.

WHAT THIS DOES NOT DECIDE. Whether the app was in a state where a PERSON
clicked (visible window, no CDP, no dev driver) is decided by the app's main
process before it attests (UI PR #3); this module proves only that the dialler
is that main process, unmodified and un-instrumented.

CONSTRAINTS.

* stdlib + ``ctypes`` only, macOS only (:func:`supported`). On Linux the
  AppImage/deb is unsigned and ``SO_PEERCRED`` carries no code identity, so no
  listener is bound and the desktop keeps the capability path (subject to the
  minted-here rule in ``asks/surfaces``).
* Dev and unsigned builds NEVER attest, with no override (operator decision
  D4): any override an env var or file could flip is forgeable by the same uid.
* Fail closed everywhere: any error, missing symbol or unexpected shape is a
  refusal, and a refused attestation only means the answer settles
  ``unverified`` as it would have without this module.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import ctypes
import ctypes.util
import hashlib
import json
import logging
import mmap
import os
import socket
import stat
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

logger = logging.getLogger(__name__)

TEAM_ID = "SHA2U6KT7V"
APP_IDENTIFIER = "com.local-operator"
#: Info.plist key only hardened builds write (UI PR #1, ``extendInfo``). Bump the
#: VALUE (and this constant) when a later hardening must lock out the builds
#: before it; the requirement compares strings exactly.
EPOCH_KEY = "LOAttestationEpoch"
EPOCH_VALUE = "1"

#: The installed app's own DR (``codesign -d -r-`` on 0.57.8) plus the epoch.
DESIGNATED_REQUIREMENT = (
    f'identifier "{APP_IDENTIFIER}" and anchor apple generic'
    " and certificate 1[field.1.2.840.113635.100.6.2.6]"
    " and certificate leaf[field.1.2.840.113635.100.6.1.13]"
    f" and certificate leaf[subject.OU] = {TEAM_ID}"
    f' and info[{EPOCH_KEY}] = "{EPOCH_VALUE}"'
)

#: Entitlements that let same-uid code execute inside a signed process (dyld
#: injection, unsigned dylibs, a debugger attach). The spike ran a dylib
#: constructor inside the genuine 0.57.8 app through exactly these.
FORBIDDEN_ENTITLEMENTS = (
    "com.apple.security.cs.allow-dyld-environment-variables",
    "com.apple.security.cs.disable-library-validation",
    "com.apple.security.get-task-allow",
)

CS_VALID = 0x1
CS_GET_TASK_ALLOW = 0x4
CS_RUNTIME = 0x10000
CS_DEBUGGED = 0x10000000

#: Electron's fuse sentinel; the wire follows as ``<version byte><length byte>``
#: then one ASCII byte per fuse (``'0'``/``'1'``, ``'r'`` = removed).
FUSE_SENTINEL = b"dL7pKGdnNz796PbbjQWNKmHXBZaB9tsX"
#: Wire position -> required byte, for wire version 1 (``@electron/fuses``
#: ``FuseV1Options``). Indexed by POSITION and only these five are required, so a
#: later Electron with a longer wire still passes (design risk 9).
REQUIRED_FUSES: Mapping[int, int] = {
    0: ord("0"),  # RunAsNode
    2: ord("0"),  # EnableNodeOptionsEnvironmentVariable
    3: ord("0"),  # EnableNodeCliInspectArguments
    4: ord("1"),  # EnableEmbeddedAsarIntegrityValidation
    5: ord("1"),  # OnlyLoadAppFromAsar
}

_FRAMEWORK_BINARY = "Contents/Frameworks/Electron Framework.framework/Versions/A/Electron Framework"

#: Where every runtime's attestation socket lives, under the config root:
#: ``<config>/run/attest/<session_id>.sock``. This is the CONTRACT the desktop
#: app dials (UI PR #3, design note "socket path"); pinned by
#: ``test_the_runtime_binds_the_contract_path``. Its own namespace, not
#: ``run/mobile`` (QA round 1, Q1: the default ``run_dir`` namespace put the
#: socket where no app would look).
ATTEST_RUN_DIRNAME = "run/attest"

#: Unix socket paths are capped at 104 bytes on macOS (``sun_path``); keep a
#: margin so a long config dir degrades to "no listener" rather than a bind error.
MAX_SOCKET_PATH = 100

#: How long an attestation waits for serve's frame. Main attests and then
#: forwards in the same request, bounded by the desktop request deadline, so the
#: gap is one local hop; 30 s is the design note's figure, generous on purpose
#: because an expired attestation fails closed (``unverified``) silently.
ATTESTATION_TTL_S = 30.0
#: Pending attestations kept per (session, ask, act): a person re-clicking after
#: an error re-attests; newest wins, the rest age out.
MAX_PENDING_PER_ASK = 4
#: Concurrent attest connections. A verified main keeps one open per card it is
#: warming; an unverified dialler is closed right after its verdict.
MAX_CONNECTIONS = 16
_FRAME_LIMIT = 64 * 1024

ATTEST_ACTS = frozenset({"respond", "revise", "decline", "dismiss"})
#: The control op each attestation act may stamp.
OP_TO_ACT = {
    "ask_respond": "respond",
    "ask_revise": "revise",
    "ask_decline": "decline",
    "ask_dismiss": "dismiss",
}


def supported() -> bool:
    """Whether this host can attest at all: macOS with Security.framework."""
    return sys.platform == "darwin"


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    pid: int = 0


@dataclass(frozen=True)
class GuestFacts:
    """What Security.framework reported about the peer's running code.

    ``status`` is ``SecCodeCopyGuestWithAttributes``'s OSStatus and
    ``requirement_status`` is ``SecCodeCheckValidity``'s against
    :data:`DESIGNATED_REQUIREMENT`; 0 is success for both.
    """

    status: int
    requirement_status: int = -1
    cs_flags: int = 0
    entitlements: frozenset[str] = frozenset()
    main_executable: str = ""


class SecurityBackend(Protocol):
    """The OS calls the verifier makes; a test injects results here."""

    def guest_facts(self, token: bytes) -> GuestFacts: ...

    def static_bundle_status(self, bundle: str) -> int: ...

    def process_start(self, pid: int) -> float | None: ...


class PeerVerifier(Protocol):
    """What the listener needs from a verifier (``AppAttestVerifier``; tests fake it)."""

    def verify(self, token: bytes) -> Verdict: ...


def parse_token(token: bytes) -> tuple[int, int]:
    """``(pid, pidversion)`` from a 32-byte ``audit_token_t`` (val[5], val[7])."""
    if len(token) != 32:
        raise ValueError("audit token must be 32 bytes")
    vals = struct.unpack("8I", token)
    return int(vals[5]), int(vals[7])


def peer_audit_token(sock: socket.socket) -> bytes:
    """The connecting process's audit token, captured by the kernel at connect."""
    sol_local, local_peertoken = 0, 6
    return sock.getsockopt(sol_local, local_peertoken, 32)


def fuse_wires(path: str) -> list[bytes]:
    """Every fuse wire in a (possibly universal) framework binary."""
    wires: list[bytes] = []
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        at = mm.find(FUSE_SENTINEL)
        while at != -1:
            head = at + len(FUSE_SENTINEL)
            version, length = mm[head], mm[head + 1]
            if version != 1:
                # An unknown wire version could reorder fuses; refuse rather
                # than read positions under the wrong layout.
                wires.append(b"")
            else:
                wires.append(bytes(mm[head + 2 : head + 2 + length]))
            at = mm.find(FUSE_SENTINEL, head)
    return wires


def fuses_hardened(wires: list[bytes]) -> bool:
    """Every architecture slice's wire carries :data:`REQUIRED_FUSES`."""
    if not wires:
        return False
    for wire in wires:
        for index, want in REQUIRED_FUSES.items():
            if index >= len(wire) or wire[index] != want:
                return False
    return True


def _raise(exc: OSError) -> None:
    raise exc


def newest_ctime(bundle: Path) -> float:
    """The newest ``ctime`` of ``Contents`` and every entry beneath it.

    ``Contents`` downward only: the bundle ROOT gains xattrs after launch
    (``com.apple.macl`` from TCC, LaunchServices' last-used date), which bump
    its ctime with no code change, and the root itself cannot be swapped under
    a running process (the main executable is resolved by vnode, so a renamed
    bundle moves with it). ``lstat`` throughout and ``followlinks=False``: a
    symlink's own ctime is what a swap changes. ``os.walk`` swallows listing
    errors by default; ``_raise`` makes an unreadable directory a refusal.
    """
    root = bundle / "Contents"
    newest = os.lstat(root).st_ctime
    for parent, dirs, files in os.walk(root, onerror=_raise, followlinks=False):
        for name in (*dirs, *files):
            ctime = os.lstat(os.path.join(parent, name)).st_ctime
            if ctime > newest:
                newest = ctime
    return newest


def bundle_of(main_executable: str) -> Path | None:
    """``X.app`` from ``X.app/Contents/MacOS/<exe>``; ``None`` for any other layout."""
    exe = Path(main_executable)
    if exe.parent.name != "MacOS" or exe.parent.parent.name != "Contents":
        return None
    bundle = exe.parent.parent.parent
    return bundle if bundle.suffix == ".app" else None


class AppAttestVerifier:
    """Decides whether an audit token names the genuine, hardened app's main process."""

    def __init__(
        self,
        backend: SecurityBackend | None = None,
        *,
        read_fuses: Callable[[str], list[bytes]] = fuse_wires,
        newest_ctime: Callable[[Path], float] = newest_ctime,
    ) -> None:
        self._backend = backend
        self._read_fuses = read_fuses
        self._newest_ctime = newest_ctime
        self._lock = threading.Lock()
        #: (pid, pidversion, bundle) -> static verdict reason ("" = ok). Sound
        #: only with the whole-tree ctime walk after it on every call (step 5).
        self._static_cache: dict[tuple[Any, ...], str] = {}

    def _get_backend(self) -> SecurityBackend:
        if self._backend is None:
            self._backend = _CtypesSecurity()
        return self._backend

    def verify(self, token: bytes) -> Verdict:
        try:
            return self._verify(token)
        except Exception as exc:  # noqa: BLE001 — any failure is a refusal
            logger.debug("app attest verification raised", exc_info=True)
            return Verdict(False, f"error:{type(exc).__name__}")

    def _verify(self, token: bytes) -> Verdict:
        backend = self._get_backend()
        pid, pidversion = parse_token(token)
        # START TIME FIRST, token checks after: if the pid were recycled
        # between the two reads, the token lookup below fails on the pid
        # version, so a start time that passes always belongs to the peer.
        started = backend.process_start(pid)
        if started is None:
            return Verdict(False, "no-process", pid)
        facts = backend.guest_facts(token)
        if facts.status != 0:
            return Verdict(False, f"no-code:{facts.status}", pid)
        if facts.requirement_status != 0:
            return Verdict(False, f"requirement:{facts.requirement_status}", pid)
        flags = facts.cs_flags
        if not flags & CS_VALID or not flags & CS_RUNTIME:
            return Verdict(False, f"flags:{flags:#x}", pid)
        if flags & (CS_GET_TASK_ALLOW | CS_DEBUGGED):
            return Verdict(False, f"debuggable:{flags:#x}", pid)
        bad = sorted(facts.entitlements & frozenset(FORBIDDEN_ENTITLEMENTS))
        if bad:
            return Verdict(False, "entitlement:" + bad[0].rsplit(".", 1)[-1], pid)
        bundle = bundle_of(facts.main_executable)
        if bundle is None:
            return Verdict(False, "bundle-layout", pid)
        key = (pid, pidversion, str(bundle))
        with self._lock:
            cached = self._static_cache.get(key)
        if cached is None:
            cached = self._static_reason(backend, bundle)
            with self._lock:
                # Bounded by the number of distinct app processes a runtime
                # ever sees; drop stale entries for this pid on a new version.
                for old in [k for k in self._static_cache if k[0] == pid]:
                    del self._static_cache[old]
                self._static_cache[key] = cached
        if cached:
            return Verdict(False, cached, pid)
        # AFTER the static result, cached or fresh, on every call (R1-M1): a
        # change made before or during the static check, or at any time since
        # a cached one, carries a ctime at or after launch and is caught here.
        if self._newest_ctime(bundle) >= started:
            return Verdict(False, "modified-after-launch", pid)
        return Verdict(True, "ok", pid)

    def _static_reason(self, backend: SecurityBackend, bundle: Path) -> str:
        status = backend.static_bundle_status(str(bundle))
        if status != 0:
            return f"static:{status}"
        if not fuses_hardened(self._read_fuses(str(bundle / _FRAMEWORK_BINARY))):
            return "fuses"
        return ""


class _CtypesSecurity:
    """Security.framework through ``ctypes`` (macOS). Every CF object is released."""

    _kSecCSSigningInformation = 1 << 1
    _kSecCSDynamicInformation = 1 << 3
    _kSecCSCheckNestedCode = 1 << 3
    _kSecCSStrictValidate = 1 << 4
    _UTF8 = 0x08000100

    def __init__(self) -> None:
        if not supported():
            raise OSError("app attestation needs macOS")
        vp = ctypes.c_void_p
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        sec = ctypes.CDLL(ctypes.util.find_library("Security"))
        self._libc = ctypes.CDLL(None)
        cf.CFDataCreate.restype = vp
        cf.CFDataCreate.argtypes = [vp, ctypes.c_char_p, ctypes.c_long]
        cf.CFDictionaryCreate.restype = vp
        cf.CFDictionaryCreate.argtypes = [
            vp,
            ctypes.POINTER(vp),
            ctypes.POINTER(vp),
            ctypes.c_long,
            vp,
            vp,
        ]
        cf.CFStringCreateWithCString.restype = vp
        cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFDictionaryGetValue.restype = vp
        cf.CFDictionaryGetValue.argtypes = [vp, vp]
        cf.CFDictionaryContainsKey.restype = ctypes.c_bool
        cf.CFDictionaryContainsKey.argtypes = [vp, vp]
        cf.CFNumberGetValue.restype = ctypes.c_bool
        cf.CFNumberGetValue.argtypes = [vp, ctypes.c_int, vp]
        cf.CFRelease.argtypes = [vp]
        cf.CFURLGetFileSystemRepresentation.restype = ctypes.c_bool
        cf.CFURLGetFileSystemRepresentation.argtypes = [
            vp,
            ctypes.c_bool,
            ctypes.c_char_p,
            ctypes.c_long,
        ]
        cf.CFURLCreateFromFileSystemRepresentation.restype = vp
        cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
            vp,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_bool,
        ]
        sec.SecCodeCopyGuestWithAttributes.argtypes = [vp, vp, ctypes.c_uint32, ctypes.POINTER(vp)]
        sec.SecCodeCopySigningInformation.argtypes = [vp, ctypes.c_uint32, ctypes.POINTER(vp)]
        sec.SecRequirementCreateWithString.argtypes = [vp, ctypes.c_uint32, ctypes.POINTER(vp)]
        sec.SecCodeCheckValidity.argtypes = [vp, ctypes.c_uint32, vp]
        sec.SecStaticCodeCreateWithPath.argtypes = [vp, ctypes.c_uint32, ctypes.POINTER(vp)]
        sec.SecStaticCodeCheckValidity.argtypes = [vp, ctypes.c_uint32, vp]
        self._libc.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            vp,
            ctypes.c_int,
        ]
        self._cf, self._sec = cf, sec
        self._dict_key_cb = vp.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")
        self._dict_val_cb = vp.in_dll(cf, "kCFTypeDictionaryValueCallBacks")
        self._k_audit = vp.in_dll(sec, "kSecGuestAttributeAudit")
        self._k_status = vp.in_dll(sec, "kSecCodeInfoStatus")
        self._k_ents = vp.in_dll(sec, "kSecCodeInfoEntitlementsDict")
        self._k_exe = vp.in_dll(sec, "kSecCodeInfoMainExecutable")

    def _release(self, *refs: Any) -> None:
        for ref in refs:
            if ref:
                self._cf.CFRelease(ref)

    def _requirement(self) -> ctypes.c_void_p:
        text = self._cf.CFStringCreateWithCString(None, DESIGNATED_REQUIREMENT.encode(), self._UTF8)
        req = ctypes.c_void_p()
        try:
            status = self._sec.SecRequirementCreateWithString(text, 0, ctypes.byref(req))
        finally:
            self._release(text)
        if status != 0 or not req.value:
            raise OSError(f"requirement did not compile: {status}")
        return req

    def guest_facts(self, token: bytes) -> GuestFacts:
        cf, sec = self._cf, self._sec
        data = cf.CFDataCreate(None, token, len(token))
        keys = (ctypes.c_void_p * 1)(self._k_audit)
        values = (ctypes.c_void_p * 1)(data)
        attrs = cf.CFDictionaryCreate(
            None,
            keys,
            values,
            1,
            ctypes.addressof(self._dict_key_cb),
            ctypes.addressof(self._dict_val_cb),
        )
        code, info, req = ctypes.c_void_p(), ctypes.c_void_p(), None
        try:
            status = sec.SecCodeCopyGuestWithAttributes(None, attrs, 0, ctypes.byref(code))
            if status != 0 or not code.value:
                return GuestFacts(status=status or -1)
            req = self._requirement()
            req_status = sec.SecCodeCheckValidity(code, self._kSecCSStrictValidate, req)
            info_status = sec.SecCodeCopySigningInformation(
                code,
                self._kSecCSDynamicInformation | self._kSecCSSigningInformation,
                ctypes.byref(info),
            )
            if info_status != 0 or not info.value:
                return GuestFacts(status=0, requirement_status=req_status or -1)
            flags = ctypes.c_uint32(0)
            sref = cf.CFDictionaryGetValue(info, self._k_status)
            if not sref or not cf.CFNumberGetValue(sref, 3, ctypes.byref(flags)):
                flags.value = 0
            ents: set[str] = set()
            eref = cf.CFDictionaryGetValue(info, self._k_ents)
            if eref:
                for name in FORBIDDEN_ENTITLEMENTS:
                    key = cf.CFStringCreateWithCString(None, name.encode(), self._UTF8)
                    try:
                        if cf.CFDictionaryContainsKey(eref, key):
                            ents.add(name)
                    finally:
                        self._release(key)
            exe = ""
            xref = cf.CFDictionaryGetValue(info, self._k_exe)
            if xref:
                buf = ctypes.create_string_buffer(4096)
                if cf.CFURLGetFileSystemRepresentation(xref, True, buf, len(buf)):
                    exe = buf.value.decode("utf-8", "replace")
            return GuestFacts(
                status=0,
                requirement_status=req_status,
                cs_flags=int(flags.value),
                entitlements=frozenset(ents),
                main_executable=exe,
            )
        finally:
            self._release(info.value, code.value, req.value if req else None, attrs, data)

    def static_bundle_status(self, bundle: str) -> int:
        cf, sec = self._cf, self._sec
        raw = bundle.encode()
        url = cf.CFURLCreateFromFileSystemRepresentation(None, raw, len(raw), True)
        static, req = ctypes.c_void_p(), None
        try:
            status = sec.SecStaticCodeCreateWithPath(url, 0, ctypes.byref(static))
            if status != 0 or not static.value:
                return status or -1
            req = self._requirement()
            return int(
                sec.SecStaticCodeCheckValidity(
                    static, self._kSecCSCheckNestedCode | self._kSecCSStrictValidate, req
                )
            )
        finally:
            self._release(static.value, req.value if req else None, url)

    def process_start(self, pid: int) -> float | None:
        # struct proc_bsdinfo (136 bytes): pbi_start_tvsec at 120, tvusec at 128.
        buf = ctypes.create_string_buffer(136)
        proc_pidtbsdinfo = 3
        got = self._libc.proc_pidinfo(pid, proc_pidtbsdinfo, 0, buf, 136)
        if got != 136:
            return None
        sec_, usec = struct.unpack_from("QQ", buf.raw, 120)
        return sec_ + usec / 1_000_000


# -- the attestation table ---------------------------------------------------


def normalize_answers(raw: Any) -> dict[str, list[str]]:
    """The exact normalization the ``ask_respond`` arm applies before answering."""
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): [str(item) for item in (value or [])] for key, value in raw.items()}


#: One attested image: ``(question_id, mime_type, sha256 hex)``.
ImageKey = tuple[str, str, str]


def image_digests(images: Any) -> list[ImageKey]:
    """Sorted ``(question_id, mime_type, sha256)`` of each wire ``images`` entry.

    The contract with the app: main hashes the base64-DECODED bytes it forwards
    and attests each image with the question it answers and its MIME type, the
    same three fields ``_answer_attachment_kwargs`` stores. Binding the digest
    alone let a tampered serve move an attested picture to another question or
    relabel its type (agent review round 1, R1-m1). Order is not part of the
    answer, so the multiset is compared sorted. ``str()`` mirrors the dispatch
    arm's own coercion, so the two sides normalise identically.
    """
    out: list[ImageKey] = []
    for item in images or []:
        if not isinstance(item, Mapping):
            out.append(("", "", "\x00invalid"))
            continue
        try:
            blob = base64.b64decode(str(item.get("data_b64") or ""), validate=False)
        except (binascii.Error, ValueError):
            blob = b"\x00invalid"
        out.append(
            (
                str(item.get("question_id") or ""),
                str(item.get("mime_type") or ""),
                hashlib.sha256(blob).hexdigest(),
            )
        )
    return sorted(out)


def attested_images(raw: Any) -> list[ImageKey] | None:
    """The app frame's ``images`` as sorted keys; ``None`` for any bad shape.

    Wire shape: ``[{"question_id": str, "mime_type": str, "sha256": hex}]``.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        return None
    out: list[ImageKey] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        question_id, mime_type, digest = (
            item.get("question_id"),
            item.get("mime_type"),
            item.get("sha256"),
        )
        if not (
            isinstance(question_id, str) and isinstance(mime_type, str) and isinstance(digest, str)
        ):
            return None
        out.append((question_id, mime_type, digest.lower()))
    return sorted(out)


@dataclass
class _Pending:
    answers: dict[str, list[str]]
    images: list[ImageKey]
    #: The dismissal's free-text reason (R1-m1); ``""`` for every other act.
    reason: str
    expires_at: float


@dataclass
class AttestationTable:
    """Attested answer CONTENT per ``(session, ask, act)``: single use, short-lived.

    Holding content rather than a token is the design: a matching frame can only
    record what the verified app attested, so the table is useless to anyone
    who cannot also make serve send that same content (and a race that does is
    benign, design row N14: the operator's own answer lands, labelled as theirs).
    """

    clock: Callable[[], float] = time.monotonic
    ttl_s: float = ATTESTATION_TTL_S
    _rows: dict[tuple[str, str, str], deque[_Pending]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        session_id: str,
        ask_id: str,
        act: str,
        answers: Any,
        images: list[ImageKey],
        reason: str = "",
    ) -> None:
        key = (session_id, ask_id, act)
        now = self.clock()
        row = _Pending(normalize_answers(answers), sorted(images), reason, now + self.ttl_s)
        with self._lock:
            # Prune EVERY key's expired rows here (R1-n2): ``consume`` only
            # touches its own key, so an attested-but-never-answered ask would
            # otherwise keep its deque for the life of the process. Linear in
            # the live rows, which only a verified app can create.
            for stale_key in list(self._rows):
                live = [p for p in self._rows[stale_key] if p.expires_at > now]
                if live:
                    self._rows[stale_key] = deque(live, maxlen=MAX_PENDING_PER_ASK)
                else:
                    del self._rows[stale_key]
            pending = self._rows.setdefault(key, deque(maxlen=MAX_PENDING_PER_ASK))
            pending.append(row)

    def consume(
        self,
        session_id: str,
        ask_id: str,
        act: str,
        answers: Any,
        images: list[ImageKey],
        reason: str = "",
    ) -> bool:
        """Spend the newest unexpired attestation matching this content, if any.

        A non-matching frame consumes NOTHING (row N13): a tampered serve that
        sends B cannot burn the operator's attestation for A.
        """
        key = (session_id, ask_id, act)
        want_answers, want_images = normalize_answers(answers), sorted(images)
        now = self.clock()
        with self._lock:
            pending = self._rows.get(key)
            if not pending:
                return False
            live = deque((p for p in pending if p.expires_at > now), maxlen=MAX_PENDING_PER_ASK)
            hit = None
            for candidate in reversed(live):
                if (
                    candidate.answers == want_answers
                    and candidate.images == want_images
                    and candidate.reason == reason
                ):
                    hit = candidate
                    break
            if hit is not None:
                live.remove(hit)
            if live:
                self._rows[key] = live
            else:
                self._rows.pop(key, None)
            return hit is not None


# -- the listener --------------------------------------------------------------


def socket_path(attest_dir: Path, session_id: str) -> Path:
    """``<attest_dir>/<session_id>.sock``; the runtime passes
    ``run_dir(config_root, ATTEST_RUN_DIRNAME)``, so the bound path is
    ``<config>/run/attest/<session_id>.sock``, where the app dials."""
    return attest_dir / f"{session_id}.sock"


class AttestService:
    """One runtime's AF_UNIX attestation listener and its table.

    Protocol (newline-JSON, each line capped at 64 KiB): on accept the runtime
    verifies the peer and writes ``{"ok":true}`` or ``{"ok":false,"reason":…}``
    (then closes). A verified peer then sends any number of
    ``{"op":"ask_attest","session_id","ask_id","act","answers","images","reason"}``
    frames; each is re-verified (cheap after the first: the static result is
    cached) and answered ``{"ok":true}`` or ``{"ok":false,"reason":…}``.
    ``images`` is ``[{"question_id","mime_type","sha256"}]`` (respond only);
    ``reason`` is the dismissal's free text exactly as serve will forward it
    (dismiss only; absent or ``null`` for none). Both are bound, R1-m1.

    Squatting the path is denial of service only and fails closed: the frame
    carries no secret, so a fake listener learns an answer the operator chose.
    """

    def __init__(
        self,
        session_id: str,
        *,
        verifier: PeerVerifier | None = None,
        table: AttestationTable | None = None,
        token_of: Callable[[socket.socket], bytes] = peer_audit_token,
    ) -> None:
        self.session_id = session_id
        self.table = table or AttestationTable()
        self._verifier: PeerVerifier = verifier or AppAttestVerifier()
        self._token_of = token_of
        self._server: asyncio.AbstractServer | None = None
        self._connections = 0
        self._writers: set[asyncio.StreamWriter] = set()
        self._ident: tuple[int, int] = (0, 0)
        self.path: Path | None = None

    async def start(self, attest_dir: Path) -> Path | None:
        """Bind ``<attest_dir>/<sid>.sock`` 0600 in a 0700 leaf; ``None`` (logged)
        when this host cannot. The runtime passes ``<config>/run/attest``."""
        path = socket_path(attest_dir, self.session_id)
        if len(os.fsencode(str(path))) > MAX_SOCKET_PATH:
            logger.info("attest: socket path too long (%d bytes); no listener", len(str(path)))
            return None
        # The 0700 DIRECTORY is the access control (no umask dance: umask is
        # process-wide and this process has other threads creating files). The
        # socket's own 0600 is belt and braces for a reader of the path alone.
        # ``lstat`` + ``S_ISDIR`` before the chmod (R1-n1): a pre-created
        # symlink at the leaf would otherwise redirect both the chmod and the
        # bind. Same-uid only, so denial of service at worst; refused anyway.
        attest_dir.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):
            os.mkdir(attest_dir, 0o700)
        if not stat.S_ISDIR(os.lstat(attest_dir).st_mode):
            logger.warning("attest: %s is not a directory; no listener", attest_dir)
            return None
        os.chmod(attest_dir, 0o700)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        self._server = await asyncio.start_unix_server(
            self._on_connection, path=str(path), limit=_FRAME_LIMIT
        )
        os.chmod(path, 0o600)
        self.path = path
        bound = os.lstat(path)
        self._ident = (bound.st_dev, bound.st_ino)
        return path

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            # ``wait_closed`` (3.12+) waits for every open connection, and main
            # keeps a warm one per card: close them or retirement hangs.
            for writer in list(self._writers):
                writer.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()
        if self.path is not None:
            # Unlink only OUR socket: a successor runtime for the same session
            # (a cwd move, an update retire) may already have bound this path.
            with contextlib.suppress(FileNotFoundError):
                now = os.lstat(self.path)
                if (now.st_dev, now.st_ino) == self._ident:
                    self.path.unlink()

    async def _verdict(self, token: bytes) -> Verdict:
        return await asyncio.to_thread(self._verifier.verify, token)

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        writer.write(json.dumps(payload).encode() + b"\n")
        await writer.drain()

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._connections >= MAX_CONNECTIONS:
            writer.close()
            return
        self._connections += 1
        self._writers.add(writer)
        try:
            sock = writer.get_extra_info("socket")
            token = self._token_of(sock)
            verdict = await self._verdict(token)
            logger.info(
                "attest: peer=%s verdict=%s reason=%s",
                verdict.pid,
                "ok" if verdict.ok else "refused",
                verdict.reason,
            )
            if not verdict.ok:
                await self._reply(writer, {"ok": False, "reason": verdict.reason})
                return
            await self._reply(writer, {"ok": True})
            while True:
                line = await reader.readline()
                if not line:
                    return
                reply = await self._frame(token, line)
                await self._reply(writer, reply)
                if not reply.get("ok") and reply.get("reason", "").startswith("peer:"):
                    return
        except (ConnectionError, asyncio.IncompleteReadError, ValueError, OSError):
            logger.debug("attest connection ended", exc_info=True)
        finally:
            self._connections -= 1
            self._writers.discard(writer)
            writer.close()

    async def _frame(self, token: bytes, line: bytes) -> dict[str, Any]:
        """One frame's reply; ANY exception is ``bad-frame`` (R1-m2, fail closed).

        Without this a malformed body that slipped past the shape checks (a
        ``TypeError`` in the table's normalisation) escaped as an unhandled
        ``client_connected_cb`` exception instead of a refusal.
        """
        try:
            return await self._frame_inner(token, line)
        except Exception:  # noqa: BLE001 — a refusal, never an escape
            logger.debug("attest: frame raised", exc_info=True)
            return {"ok": False, "reason": "bad-frame"}

    async def _frame_inner(self, token: bytes, line: bytes) -> dict[str, Any]:
        try:
            frame = json.loads(line)
        except ValueError:
            return {"ok": False, "reason": "bad-json"}
        if not isinstance(frame, dict) or frame.get("op") != "ask_attest":
            return {"ok": False, "reason": "bad-op"}
        if str(frame.get("session_id") or "") != self.session_id:
            return {"ok": False, "reason": "wrong-session"}
        ask_id, act = str(frame.get("ask_id") or ""), str(frame.get("act") or "")
        if not ask_id or act not in ATTEST_ACTS:
            return {"ok": False, "reason": "bad-frame"}
        images = attested_images(frame.get("images"))
        reason = frame.get("reason")
        if images is None or not (reason is None or isinstance(reason, str)):
            return {"ok": False, "reason": "bad-frame"}
        if not isinstance(frame.get("answers", {}), (dict, type(None))):
            return {"ok": False, "reason": "bad-frame"}
        # RE-VERIFIED PER FRAME, not trusted from the accept: the process could
        # have changed state since (it is cheap; the static half is cached).
        verdict = await self._verdict(token)
        if not verdict.ok:
            logger.info("attest: peer=%s verdict=refused reason=%s", verdict.pid, verdict.reason)
            return {"ok": False, "reason": f"peer:{verdict.reason}"}
        self.table.record(
            self.session_id,
            ask_id,
            act,
            frame.get("answers"),
            images,
            # Only a dismissal carries one; the server forwards it as
            # ``str(reason)`` when truthy, so ``""`` and ``None`` both mean none.
            str(reason or "") if act == "dismiss" else "",
        )
        return {"ok": True}
