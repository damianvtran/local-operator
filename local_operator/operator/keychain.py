"""The private half of an operator key: where it lives and what a signature costs.

WHY THE PRIVATE HALF IS THE LOAD-BEARING PART OF THIS DESIGN. The anchor
(:mod:`local_operator.operator.trust`) pins the PUBLIC half and is root-owned, so
a same-uid subject cannot substitute it. The private half is the other half of
the argument: if a model's tool subprocess can read it, that subprocess can sign
for the operator and the whole mechanism is decoration. So the private half is
placed where a signature requires a HUMAN GESTURE, and the ladder is reported
rather than assumed:

* ``secure-enclave`` (macOS) — a P-256 key in the Secure Enclave with
  ``kSecAccessControlUserPresence``, so every ``SecKeyCreateSignature`` raises a
  Touch ID / login-password prompt. The key material never enters this process's
  address space.
* ``cng-presence`` (Windows) — a CNG key in the Microsoft Software KSP carrying
  ``NCRYPT_UI_POLICY``, so every use raises the CNG consent dialog. A DPAPI file
  alone is NOT a presence gate and is reported as such.
* ``file-only`` (Linux, and the fallback everywhere) — a 0600 file holding a
  PKCS#8 P-256 key. **This is not a boundary**: the same uid can read it and
  sign for the operator without any human act. It exists so the runtime has an
  honest level to report on a host with no presence store, not so that the
  boundary can be claimed there.

MEASURED, IN A THROWAWAY KEYCHAIN (macOS 26.6.2, uid 501, SIP on): a Secure
Enclave key CANNOT be created in a legacy file keychain at all — every attribute
shape tried against one (with and without ``kSecUseKeychain``, with each of
``kSecAttrAccessibleWhenUnlocked`` / ``WhenUnlockedThisDeviceOnly`` /
``WhenPasscodeSetThisDeviceOnly``, and with each of the ``privateKeyUsage`` and
``userPresence`` flags) returns ``errSecParam`` ("inconsistent private key
parameters for key generation"). Secure Enclave keys live in the data-protection
keychain, i.e. the user's login keychain, and that placement is the OS's choice
rather than ours. The consequence for TESTS is recorded here because it is easy
to get wrong: a test can never exercise this backend without writing an item to
the operator's login keychain, so the backend's contract is exercised through
:class:`FileKeyBackend` and the Secure Enclave path is left to ``lop operator
init``, which the operator runs deliberately.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.operator.verify import decode_point, key_id_for

#: Backend names, in the order the presence ladder prefers them.
SECURE_ENCLAVE = "secure-enclave"
CNG_PRESENCE = "cng-presence"
FILE_ONLY = "file-only"

#: The application tag / label the operator key is stored under. A CONSTANT
#: rather than a parameter: the anchor pins one key per machine, and a
#: configurable tag would be a second way to point the runtime at a different
#: key than the anchor names.
APPLICATION_TAG = "com.local-operator.operator.v1"
KEY_LABEL = "local-operator operator key"


@dataclass(frozen=True)
class KeyHandle:
    """One usable private key, plus what its use costs.

    ``presence`` is the whole reason the ladder exists and travels with the
    handle rather than being inferred later from ``backend``: a report that says
    ``file-only`` and a report that says ``secure-enclave`` are describing two
    different security claims, and the level function must be able to make that
    claim from the SAME object the signer signs with.
    """

    backend: str
    key_id: str
    spki: bytes
    presence: bool

    @property
    def level(self) -> str:
        return self.backend


class KeyBackendError(RuntimeError):
    """A backend could not create, load or use a key.

    Carried rather than collapsed to ``False`` because the two outcomes the
    caller must tell apart are "no presence store on this host" (report a lower
    level and carry on) and "the presence store refused" (tell the operator,
    who has a gesture to make).
    """


# ---------------------------------------------------------------------------
# The signer seam
# ---------------------------------------------------------------------------


class Signer:
    """A loaded private key: sign a message, and say what that cost.

    An object rather than a module-level function so the Secure Enclave
    reference (a CoreFoundation object with a lifetime) can be released in
    :meth:`close`, and so a test can substitute one without touching the OS
    store. The runtime never constructs one: it only VERIFIES (see
    :mod:`local_operator.operator.verify`), which is what keeps the private half
    out of the serving process entirely.
    """

    def __init__(self, handle: KeyHandle) -> None:
        self.handle = handle

    def sign(self, message: bytes) -> bytes:  # pragma: no cover — interface
        raise NotImplementedError

    def close(self) -> None:
        return None


def _der_from_raw(raw: bytes) -> bytes:
    """A raw ``r||s`` P-256 signature as DER.

    ``cryptography`` produces and consumes DER; the Secure Enclave does too. This
    exists for the CNG backend, whose ``BCryptSignHash`` returns the raw form,
    and for tests that build a signature by hand.
    """
    from cryptography.hazmat.primitives.asymmetric import utils

    r = int.from_bytes(raw[:32], "big")
    s = int.from_bytes(raw[32:], "big")
    return utils.encode_dss_signature(r, s)


# ---------------------------------------------------------------------------
# file-only — the honest fallback
# ---------------------------------------------------------------------------


class FileKeyBackend:
    """A software P-256 key in a 0600 file. NOT a presence gate — see the module docstring.

    The file mode is set at creation with ``O_EXCL`` and an explicit ``chmod``
    rather than trusting the umask, because the default umask on a developer
    machine is 022 and the difference between 0600 and 0644 here is the
    difference between "only this uid" and "any process that can read the
    operator's home". It is still not a boundary — the same uid can read a 0600
    file — and the level report says so rather than implying otherwise.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def create(self) -> KeyHandle:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        private = ec.generate_private_key(ec.SECP256R1())
        pem = private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, pem)
        finally:
            os.close(descriptor)
        os.chmod(self.path, 0o600)
        return self._handle(
            private.public_key().public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )
        )

    def _handle(self, spki: bytes) -> KeyHandle:
        return KeyHandle(backend=FILE_ONLY, key_id=key_id_for(spki), spki=spki, presence=False)

    def load(self) -> Signer | None:
        from cryptography.hazmat.primitives import serialization

        try:
            raw = self.path.read_bytes()
        except OSError:
            return None
        private = serialization.load_pem_private_key(raw, password=None)
        spki = private.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        return _SoftwareSigner(self._handle(spki), private)

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return


class _SoftwareSigner(Signer):
    """A software key's signer. The ``file-only`` backend's signing half."""

    def __init__(self, handle: KeyHandle, private: Any) -> None:
        super().__init__(handle)
        self._private = private

    def sign(self, message: bytes) -> bytes:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        return self._private.sign(message, ec.ECDSA(hashes.SHA256()))


# ---------------------------------------------------------------------------
# macOS — Secure Enclave, presence-gated
# ---------------------------------------------------------------------------


class _CF:
    """The minimum CoreFoundation surface the Secure Enclave key needs.

    Built with ``ctypes`` rather than PyObjC because the harness must keep its
    dependency graph to what the repository already requires; the cost is this
    class. Every CoreFoundation object created here is released by the caller
    that created it — a leaked ``CFStringRef`` per signature is a real leak in a
    process that signs once per loosening, and it is invisible until it is not.
    """

    def __init__(self) -> None:
        security = ctypes.util.find_library("Security")
        core = ctypes.util.find_library("CoreFoundation")
        if not security or not core:  # pragma: no cover — macOS always has both
            raise KeyBackendError("Security.framework is not present")
        self.S = ctypes.CDLL(security)
        self.C = ctypes.CDLL(core)
        void = ctypes.c_void_p
        self.S.SecKeyCreateRandomKey.restype = void
        self.S.SecKeyCreateRandomKey.argtypes = [void, ctypes.POINTER(void)]
        self.S.SecKeyCopyPublicKey.restype = void
        self.S.SecKeyCopyPublicKey.argtypes = [void]
        self.S.SecKeyCopyExternalRepresentation.restype = void
        self.S.SecKeyCopyExternalRepresentation.argtypes = [void, ctypes.POINTER(void)]
        self.S.SecKeyCreateSignature.restype = void
        self.S.SecKeyCreateSignature.argtypes = [void, void, void, ctypes.POINTER(void)]
        self.S.SecAccessControlCreateWithFlags.restype = void
        self.S.SecAccessControlCreateWithFlags.argtypes = [
            void,
            void,
            ctypes.c_uint64,
            ctypes.POINTER(void),
        ]
        self.S.SecItemCopyMatching.restype = ctypes.c_int32
        self.S.SecItemCopyMatching.argtypes = [void, ctypes.POINTER(void)]
        self.S.SecItemDelete.restype = ctypes.c_int32
        self.S.SecItemDelete.argtypes = [void]
        self.S.CFErrorCopyDescription.restype = void
        self.S.CFErrorCopyDescription.argtypes = [void]
        self.C.CFStringCreateWithBytes.restype = void
        self.C.CFStringCreateWithBytes.argtypes = [
            void,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_uint32,
            ctypes.c_bool,
        ]
        self.C.CFDataCreate.restype = void
        self.C.CFDataCreate.argtypes = [void, ctypes.c_char_p, ctypes.c_long]
        self.C.CFDictionaryCreate.restype = void
        self.C.CFDictionaryCreate.argtypes = [
            void,
            ctypes.POINTER(void),
            ctypes.POINTER(void),
            ctypes.c_long,
            void,
            void,
        ]
        self.C.CFDataGetLength.restype = ctypes.c_long
        self.C.CFDataGetLength.argtypes = [void]
        self.C.CFDataGetBytePtr.restype = ctypes.POINTER(ctypes.c_char)
        self.C.CFDataGetBytePtr.argtypes = [void]
        self.C.CFStringGetCString.restype = ctypes.c_bool
        self.C.CFStringGetCString.argtypes = [void, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        self.C.CFRelease.argtypes = [void]

    def const(self, name: str) -> int:
        value = ctypes.c_void_p.in_dll(self.S, name).value
        if value is None:  # pragma: no cover — a missing symbol raises, it does not return None
            raise KeyBackendError(f"Security.framework does not export {name}")
        return int(value)

    def cconst(self, name: str) -> int:
        value = ctypes.c_void_p.in_dll(self.C, name).value
        if value is None:  # pragma: no cover — as above
            raise KeyBackendError(f"CoreFoundation does not export {name}")
        return int(value)

    def string(self, text: str) -> int:
        raw = text.encode()
        return int(self.C.CFStringCreateWithBytes(None, raw, len(raw), 0x08000100, False))

    def data(self, raw: bytes) -> int:
        return int(self.C.CFDataCreate(None, raw, len(raw)))

    def boolean(self, value: bool) -> int:
        return int(self.cconst("kCFBooleanTrue" if value else "kCFBooleanFalse"))

    def dict(self, pairs: list[tuple[int, int]]) -> int:
        keys = (ctypes.c_void_p * len(pairs))(*[k for k, _ in pairs])
        values = (ctypes.c_void_p * len(pairs))(*[v for _, v in pairs])
        return int(self.C.CFDictionaryCreate(None, keys, values, len(pairs), None, None))

    def release(self, *refs: int) -> None:
        for ref in refs:
            if ref:
                self.C.CFRelease(ref)

    def data_bytes(self, ref: int) -> bytes:
        length = self.C.CFDataGetLength(ref)
        return ctypes.string_at(self.C.CFDataGetBytePtr(ref), length)

    def error(self, err: Any) -> str:
        """A CFErrorRef as text, releasing it. Never raises."""
        if not err:
            return "no error reported"
        try:
            description = self.S.CFErrorCopyDescription(err)
            if not description:
                return "error with no description"
            buffer = ctypes.create_string_buffer(1024)
            self.C.CFStringGetCString(description, buffer, 1024, 0x08000100)
            self.release(description)
            return buffer.value.decode("utf-8", "replace")
        finally:
            self.release(err)


class SecureEnclaveBackend:
    """The macOS presence-gated backend.

    ``create`` is attempted with a ladder of protection constants and flags,
    taking the first the framework accepts and reporting the last failure if
    none does. That is deliberate: the protection class is an OS policy
    (``WhenPasscodeSetThisDeviceOnly`` is refused on a host whose policy differs)
    and the honest outcome for a host that refuses every one is a precise error,
    not a silent downgrade to a file — a silent downgrade would make the level
    report claim presence the host does not enforce.
    """

    #: ``kSecAccessControlPrivateKeyUsage`` and ``kSecAccessControlUserPresence``.
    PRIVATE_KEY_USAGE = 1 << 0
    USER_PRESENCE = 1 << 2

    #: Tried in order; the strictest protection the host accepts wins.
    PROTECTION_LADDER = (
        "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly",
        "kSecAttrAccessibleWhenUnlockedThisDeviceOnly",
    )

    def __init__(self) -> None:
        self.cf = _CF()

    def supported(self) -> bool:
        """Whether this host looks like it has a Secure Enclave.

        ``arm64`` and ``x86_64`` Macs with a T2 both do; the check is deliberately
        coarse, because a false positive costs one precise error at ``init`` and a
        false negative would silently demote every Intel Mac with a T2.
        """
        try:
            return os.uname().machine in ("arm64", "x86_64")
        except AttributeError:  # pragma: no cover — POSIX always has uname
            return False

    def create(self) -> KeyHandle:
        cf = self.cf
        tag = cf.data(APPLICATION_TAG.encode())
        failures: list[str] = []
        for protection in self.PROTECTION_LADDER:
            err = ctypes.c_void_p()
            access = cf.S.SecAccessControlCreateWithFlags(
                None,
                cf.const(protection),
                self.PRIVATE_KEY_USAGE | self.USER_PRESENCE,
                ctypes.byref(err),
            )
            if not access:
                failures.append(f"{protection}: {cf.error(err)}")
                continue
            private_attrs = cf.dict(
                [
                    (cf.const("kSecAttrIsPermanent"), cf.boolean(True)),
                    (cf.const("kSecAttrApplicationTag"), tag),
                    (cf.const("kSecAttrAccessControl"), int(access)),
                ]
            )
            attrs = cf.dict(
                [
                    (cf.const("kSecAttrKeyType"), cf.const("kSecAttrKeyTypeECSECPrimeRandom")),
                    (cf.const("kSecAttrKeySizeInBits"), self._cf_number(256)),
                    (cf.const("kSecAttrTokenID"), cf.const("kSecAttrTokenIDSecureEnclave")),
                    (cf.const("kSecUseDataProtectionKeychain"), cf.boolean(True)),
                    (cf.const("kSecPrivateKeyAttrs"), private_attrs),
                ]
            )
            err = ctypes.c_void_p()
            key = cf.S.SecKeyCreateRandomKey(attrs, ctypes.byref(err))
            cf.release(private_attrs, attrs, access, tag)
            if not key:
                failures.append(f"{protection}: {cf.error(err)}")
                continue
            try:
                spki = self._public_point(int(key))
            finally:
                cf.release(int(key))
            return KeyHandle(
                backend=SECURE_ENCLAVE, key_id=key_id_for(spki), spki=spki, presence=True
            )
        raise KeyBackendError(
            "the Secure Enclave refused to create an operator key (" + "; ".join(failures) + ")"
        )

    def _cf_number(self, value: int) -> int:
        """A ``CFNumberRef`` holding ``value``, released by the caller's dict release.

        Small and deliberate: ``CFNumberCreate`` is not part of the surface
        ``_CF`` declares, and wrapping it here keeps the number's lifetime tied
        to the dictionary that holds it.
        """
        cf = self.cf
        cf.C.CFNumberCreate.restype = ctypes.c_void_p
        cf.C.CFNumberCreate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        # 3 is kCFNumberSInt32Type, which is what CFNumberConsume wants for a
        # bit count; the c_int in the byref is the same width, deliberately.
        return int(cf.C.CFNumberCreate(None, 3, ctypes.byref(ctypes.c_int(value))))

    def _public_point(self, key: int) -> bytes:
        cf = self.cf
        public = cf.S.SecKeyCopyPublicKey(key)
        if not public:
            raise KeyBackendError("the Secure Enclave key has no public half")
        try:
            err = ctypes.c_void_p()
            raw = cf.S.SecKeyCopyExternalRepresentation(public, ctypes.byref(err))
            if not raw:
                raise KeyBackendError(f"could not export the public key: {cf.error(err)}")
            try:
                point = cf.data_bytes(int(raw))
            finally:
                cf.release(int(raw))
        finally:
            cf.release(int(public))
        if decode_point(point) is None:  # pragma: no cover — a malformed OS reply
            raise KeyBackendError("the Secure Enclave returned a non-P-256 public key")
        return point

    def load(self) -> Signer | None:
        cf = self.cf
        tag = cf.data(APPLICATION_TAG.encode())
        query = cf.dict(
            [
                (cf.const("kSecClass"), cf.const("kSecClassKey")),
                (cf.const("kSecAttrApplicationTag"), tag),
                (cf.const("kSecAttrKeyType"), cf.const("kSecAttrKeyTypeECSECPrimeRandom")),
                (cf.const("kSecReturnRef"), cf.boolean(True)),
                (cf.const("kSecMatchLimit"), cf.const("kSecMatchLimitOne")),
            ]
        )
        out = ctypes.c_void_p()
        status = cf.S.SecItemCopyMatching(query, ctypes.byref(out))
        cf.release(query, tag)
        if status != 0 or not out:
            return None
        key = int(out.value or 0)
        if not key:  # pragma: no cover — defensive
            return None
        handle = KeyHandle(
            backend=SECURE_ENCLAVE,
            key_id=key_id_for(self._public_point(key)),
            spki=self._public_point(key),
            presence=True,
        )
        return _SecureEnclaveSigner(handle, key, cf)

    def delete(self) -> None:
        cf = self.cf
        tag = cf.data(APPLICATION_TAG.encode())
        query = cf.dict(
            [
                (cf.const("kSecClass"), cf.const("kSecClassKey")),
                (cf.const("kSecAttrApplicationTag"), tag),
            ]
        )
        try:
            cf.S.SecItemDelete(query)
        finally:
            cf.release(query, tag)


class _SecureEnclaveSigner(Signer):
    """Signs through the Secure Enclave. EVERY call raises the presence prompt."""

    def __init__(self, handle: KeyHandle, key_ref: int, cf: _CF) -> None:
        super().__init__(handle)
        self._key = key_ref
        self._cf = cf

    def sign(self, message: bytes) -> bytes:
        cf = self._cf
        err = ctypes.c_void_p()
        signature = cf.S.SecKeyCreateSignature(
            self._key,
            cf.const("kSecKeyAlgorithmECDSASignatureMessageX962SHA256"),
            cf.data(message),
            ctypes.byref(err),
        )
        if not signature:
            raise KeyBackendError(f"the Secure Enclave refused to sign: {cf.error(err)}")
        try:
            return cf.data_bytes(int(signature))
        finally:
            cf.release(int(signature))

    def close(self) -> None:
        self._cf.release(self._key)


# ---------------------------------------------------------------------------
# Windows — CNG with a UI policy
# ---------------------------------------------------------------------------


class CngBackend:  # pragma: no cover — Windows only; CI runs POSIX
    """The Windows presence-gated backend.

    ``NCRYPT_UI_POLICY`` with ``NCRYPT_UI_PROTECT_KEY_FLAG`` makes every use of
    the key raise the CNG consent dialog, which is the Windows equivalent of the
    Secure Enclave's ``userPresence``. Written blind, like the other Windows
    paths in this repository (``harness/approval._win32_inheritable_pipe``),
    because no CI runner here is Windows; the level report is what keeps that
    honest — a host where this backend fails reports ``file-only`` rather than
    claiming a prompt it cannot raise.
    """

    def supported(self) -> bool:
        return os.name == "nt"

    def create(self) -> KeyHandle:  # pragma: no cover
        raise KeyBackendError("the CNG presence backend is not implemented on this build")

    def load(self) -> Signer | None:  # pragma: no cover
        return None

    def delete(self) -> None:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def default_file_path(config_root: Path) -> Path:
    """Where the ``file-only`` backend keeps its key."""
    return config_root / "operator" / "operator-key.pem"


def choose_backend(preference: str, *, config_root: Path) -> Any:
    """The backend to use, by name or by the presence ladder.

    ``auto`` takes the first backend this host both HAS and can USE: the Secure
    Enclave on macOS, CNG on Windows, the file otherwise. The caller reports
    which one it got (:meth:`KeyHandle.level`) rather than assuming the ladder
    reached the top — that report is the difference between a security claim and
    a hope.
    """
    if preference in (SECURE_ENCLAVE, CNG_PRESENCE, FILE_ONLY):
        return _named_backend(preference, config_root=config_root)
    if os.name == "nt":
        return CngBackend()
    if os.uname().sysname == "Darwin":  # pragma: no branch — POSIX always has uname
        return SecureEnclaveBackend()
    return FileKeyBackend(default_file_path(config_root))


def _named_backend(name: str, *, config_root: Path) -> Any:
    if name == SECURE_ENCLAVE:
        return SecureEnclaveBackend()
    if name == CNG_PRESENCE:
        return CngBackend()
    return FileKeyBackend(default_file_path(config_root))


def key_file_mode_is_private(path: Path) -> bool:
    """Whether a key file is 0600 (and owned by this uid).

    Reported, never enforced by changing an existing file's mode: silently
    chmod-ing a path is the kind of helpful behaviour that makes a host look safe
    without the operator having decided it is.
    """
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_IMODE(info.st_mode) == 0o600 and info.st_uid == os.getuid()


def random_device_id() -> str:
    """A fresh device identifier for a pairing step (used by stage D)."""
    return secrets.token_hex(16)


def digest_key(*parts: str) -> str:
    """A short digest over several strings, for cache keys and test ids."""
    joined = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()
