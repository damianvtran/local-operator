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

MEASURED (macOS 26.6.2, uid 501, SIP on, arm64), and recorded here because both
findings are easy to re-derive wrongly:

1. The ``errSecParam`` (-50) an earlier revision of this paragraph attributed to
   keychain PLACEMENT is the ACCESS-CONTROL FLAG PAIR, not the keychain. A native
   probe that creates nothing and touches no keychain reproduces it: with the pair
   this file shipped, ``SecAccessControlCreateWithFlags`` returns NULL and OSStatus
   -50 for every protection class, and the framework's own message names the
   remedy — "kSecAccessControlUserPresence can be combined only with
   kSecAccessControlApplicationPassword and kSecAccessControlPrivateKeyUsage".
   With Apple's values (``userPresence`` 1<<0, ``privateKeyUsage`` 1<<30) it
   returns an access object for every class. The old measurement could not tell
   those apart, because access-control creation failed one step BEFORE anything
   reached a keychain. See :attr:`SecureEnclaveBackend.PRIVATE_KEY_USAGE`.

2. A Secure Enclave key can only live in the DATA-PROTECTION keychain, and that
   keychain refuses a process with no code signature: ``SecKeyCreateRandomKey``
   then fails with ``errSecMissingEntitlement`` (-34018, "failed to add key to
   keychain: <SecKeyRef:('com.apple.setoken')>"). Measured on this host with a
   native C probe, in BOTH attribute shapes, and reproduced through this package's
   own ``create`` — while a generic-password add to the same keychain succeeds from
   a SIGNED process (Apple's own ``python3``, ``Identifier=com.apple.python3``), and
   under it a real Enclave key is created, exported as a 65-byte uncompressed P-256
   point, found again by tag and deleted. Ad-hoc signing is not a workaround: with
   ``keychain-access-groups`` / ``application-identifier`` in an ad-hoc signature
   the kernel kills the process (SIGKILL, 137). A runtime distributed as an
   unsigned Python build therefore cannot create this key on any host; ``init``
   reports that failure rather than a level it did not reach.

The consequence for TESTS is recorded here because it is easy to get wrong: a
test can never exercise this backend without writing an item to the operator's
login keychain, and on an unsigned host it cannot exercise it at all, so the
backend's contract is exercised through :class:`FileKeyBackend` and through the
CF-level tests in ``tests/unit/operator`` that assert structure and ownership
without creating a key; the end-to-end path is left to ``lop operator init``,
which the operator runs deliberately.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.operator.verify import decode_point, key_id_for

#: Backend names, in the order the presence ladder prefers them.
#:
#: NO BACKEND OFFERS A ``delete``. Removing a key is not in the verb set this
#: change ships (``lop operator init|trust|install|sign|status``), and a method
#: that calls ``Path.unlink`` / ``SecItemDelete`` from here would trip the
#: repository's session-directory guard (``test_no_session_deletion``), which
#: exists precisely so no module outside ``session/cleanup.py`` can remove a
#: directory. Rotation is the anchored operation anyway: write a new key, install
#: a new anchor, and the old public half is simply no longer the pinned one.
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
        self.S.CFErrorGetCode.restype = ctypes.c_long
        self.S.CFErrorGetCode.argtypes = [void]
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
        # kCFTypeDictionaryKeyCallBacks / kCFTypeDictionaryValueCallBacks, by
        # ADDRESS. Each symbol is a struct whose first field is its version (0), so
        # reading the symbol as a pointer would read that 0; the address is what
        # CFDictionaryCreate takes. Declared here once because they never change.
        self.TYPED_KEY_CALLBACKS = ctypes.addressof(
            (ctypes.c_char * 0).in_dll(self.C, "kCFTypeDictionaryKeyCallBacks")
        )
        self.TYPED_VALUE_CALLBACKS = ctypes.addressof(
            (ctypes.c_char * 0).in_dll(self.C, "kCFTypeDictionaryValueCallBacks")
        )

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
        """A ``CFDictionaryRef`` that RETAINS its keys and values.

        Built with ``kCFTypeDictionaryKeyCallBacks`` / ``ValueCallBacks`` rather
        than NULL callbacks. With NULL callbacks a dictionary holds BORROWED
        pointers — it retains nothing and releases nothing — so every object put
        into one must outlive it by hand, and that discipline is exactly what one
        caller got wrong (a released ``CFDataRef`` was passed into a later
        dictionary). With the typed callbacks a dictionary is self-sufficient and
        the caller may release its own reference as soon as the dictionary is
        built, which is the ownership rule :meth:`SecureEnclaveBackend.create`
        follows.

        Measured, and the reason this is not merely hygiene: on macOS 26.6.2 an
        Enclave key-generation dictionary carrying the corrected flags SEGFAULTS
        inside ``SecKeyCreateRandomKey`` when built with NULL callbacks (6 of 6
        runs, from both a ctypes probe and a native C probe), and the identical
        dictionary with these callbacks returns a clean OSStatus (6 of 6). NULL is
        legal but means "these objects are not managed"; passing it asked the
        framework to keep a promise nothing was keeping.
        """
        keys = (ctypes.c_void_p * len(pairs))(*[k for k, _ in pairs])
        values = (ctypes.c_void_p * len(pairs))(*[v for _, v in pairs])
        return int(
            self.C.CFDictionaryCreate(
                None,
                keys,
                values,
                len(pairs),
                self.TYPED_KEY_CALLBACKS,
                self.TYPED_VALUE_CALLBACKS,
            )
        )

    def release(self, *refs: int) -> None:
        for ref in refs:
            if ref:
                self.C.CFRelease(ref)

    def data_bytes(self, ref: int) -> bytes:
        length = self.C.CFDataGetLength(ref)
        return ctypes.string_at(self.C.CFDataGetBytePtr(ref), length)

    def status(self, err: Any) -> int:
        """The ``OSStatus`` inside a ``CFErrorRef``, WITHOUT releasing it.

        Separate from :meth:`error`, which releases: the CODE picks the remedy the
        operator is offered, while the DESCRIPTION is the precise detail, and
        ``create`` needs both from the same error object.
        """
        if not err:
            return 0
        return int(self.S.CFErrorGetCode(err))

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


#: ``OSStatus`` values from the presence store that carry a REMEDY, with the
#: sentence the operator needs. The framework's own text is precise but does not
#: say what to do; these two do, and they are the two this repository has measured
#: on a real host (see the module docstring).
#:
#: ``-50`` is ``errSecParam``: the access-control flags were refused. It means a
#: build is carrying a flag pair Apple does not accept — ``userPresence`` combines
#: only with ``applicationPassword`` and ``privateKeyUsage`` — and NO host will
#: accept it, so the sentence says that rather than implying a host problem.
_ERR_SEC_PARAM = -50

#: ``-34018`` is ``errSecMissingEntitlement``. It is NOT a host defect and not a
#: flag defect: the Secure Enclave keeps its keys in the data-protection keychain,
#: and that keychain refuses a caller carrying no keychain entitlement — i.e. a
#: process that is not code-signed (measured: an ad-hoc/linker-signed interpreter
#: gets this, Apple's signed ``python3`` does not). The remedy names the level such
#: a runtime can honestly create instead, so the failure routes the operator to a
#: working path rather than leaving them with a negative number.
_ERR_SEC_MISSING_ENTITLEMENT = -34018

_SECURE_ENCLAVE_REMEDIES: dict[int, str] = {
    _ERR_SEC_PARAM: (
        "the access-control flags were refused (errSecParam): kSecAccessControlUserPresence "
        "may be combined only with kSecAccessControlApplicationPassword and "
        "kSecAccessControlPrivateKeyUsage, so a build carrying a wrong flag pair cannot "
        "create a key on any host"
    ),
    _ERR_SEC_MISSING_ENTITLEMENT: (
        "this process carries no keychain entitlement (errSecMissingEntitlement): the "
        "Secure Enclave keeps its keys in the data-protection keychain, which refuses a "
        "caller that is not code-signed, so this runtime cannot use the presence tier here "
        "— `lop operator init --backend file-only` creates the level this runtime can "
        "enforce, and `lop operator status` reports it as that"
    ),
}


def secure_enclave_remedy(status: int) -> str:
    """The actionable sentence for an ``OSStatus`` from the presence store.

    An empty string for a status with no known remedy, deliberately: the caller
    appends what it has and contributes nothing on a code we cannot explain,
    rather than guessing a cause. Classification is a seam of its own so it can be
    tested without a Secure Enclave, which matters because the two codes above
    cannot both be produced on one host.
    """
    return _SECURE_ENCLAVE_REMEDIES.get(int(status), "")


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

    #: ``kSecAccessControlPrivateKeyUsage`` and ``kSecAccessControlUserPresence``,
    #: as Apple's header defines them
    #: (``Security.framework/Headers/SecAccessControl.h``:
    #: ``kSecAccessControlUserPresence = 1u << 0``,
    #: ``kSecAccessControlPrivateKeyUsage = 1u << 30``).
    #:
    #: DO NOT SIMPLIFY THESE BACK into a small shift pair. The two are not
    #: interchangeable and the failure is not a subtle degradation:
    #: ``SecAccessControlCreateWithFlags`` REFUSES a wrong pair with ``errSecParam``
    #: (-50) — "kSecAccessControlUserPresence can be combined only with
    #: kSecAccessControlApplicationPassword and kSecAccessControlPrivateKeyUsage" —
    #: so every protection class in the ladder fails and ``create`` never reaches key
    #: generation. This file shipped ``PRIVATE_KEY_USAGE = 1 << 0`` and
    #: ``USER_PRESENCE = 1 << 2``: transposed, and two bits off Apple's
    #: ``PrivateKeyUsage``. Measured with a native probe against this machine's SDK,
    #: creating nothing and touching no keychain: the shipped pair returns NULL/-50
    #: for both protection classes, Apple's pair returns a non-NULL access object for
    #: both. ``tests/unit/operator/test_operator_authority.py`` asserts both constants
    #: against the SDK header itself for exactly that reason.
    PRIVATE_KEY_USAGE = 1 << 30
    USER_PRESENCE = 1 << 0

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
        """Create the Enclave key, taking the first protection class the host accepts.

        OWNERSHIP RULE — every CoreFoundation object created here is released
        exactly once, in the scope that created it. The dictionaries RETAIN what
        they hold (see :meth:`_CF.dict`), so an object handed to one is released as
        soon as the dictionary holding it exists. ``tag`` is the deliberate
        exception: it is created here, used by every iteration, and released once in
        the ``finally`` below.

        The previous shape released ``tag`` INSIDE the loop, on the iteration that
        failed to make a key, and then handed it to the next iteration's dictionary:
        a released ``CFDataRef``, i.e. freed memory walked by ``CFDictionaryCreate``
        / ``objc_retain``. That path is reachable exactly when the access control
        SUCCEEDS and key creation then FAILS — the documented case is a host whose
        policy refuses the strictest protection class — which is why it went
        unnoticed while the flag bug made every iteration fail one step earlier. The
        two defects are coupled: fixing only the flags would have turned a precise
        error into a crash on such a host, so both are fixed together.
        """
        cf = self.cf
        tag = cf.data(APPLICATION_TAG.encode())
        failures: list[str] = []
        statuses: list[int] = []
        try:
            for protection in self.PROTECTION_LADDER:
                err = ctypes.c_void_p()
                access = cf.S.SecAccessControlCreateWithFlags(
                    None,
                    cf.const(protection),
                    self.PRIVATE_KEY_USAGE | self.USER_PRESENCE,
                    ctypes.byref(err),
                )
                if not access:
                    statuses.append(cf.status(err))
                    failures.append(f"{protection}: {cf.error(err)}")
                    continue
                private_attrs = cf.dict(
                    [
                        (cf.const("kSecAttrIsPermanent"), cf.boolean(True)),
                        (cf.const("kSecAttrApplicationTag"), tag),
                        (cf.const("kSecAttrAccessControl"), int(access)),
                    ]
                )
                cf.release(access)
                number = self._cf_number(256)
                # ``kSecUseDataProtectionKeychain`` is not in Apple's documented
                # Enclave generation dictionary, and on macOS 26.6.2 it is not
                # LOAD-BEARING either: measured under a signed interpreter, both
                # shapes create and re-find a key identically for both protection
                # classes, and with it the round trip through this module's own
                # ``load`` query (which does not set it) finds the same key. It is
                # kept because it changes nothing measurable and removing it would
                # be an unvalidated change; what puts the key in the data-protection
                # keychain is ``kSecAttrTokenIDSecureEnclave``, not this attribute.
                attrs = cf.dict(
                    [
                        (cf.const("kSecAttrKeyType"), cf.const("kSecAttrKeyTypeECSECPrimeRandom")),
                        (cf.const("kSecAttrKeySizeInBits"), number),
                        (cf.const("kSecAttrTokenID"), cf.const("kSecAttrTokenIDSecureEnclave")),
                        (cf.const("kSecUseDataProtectionKeychain"), cf.boolean(True)),
                        (cf.const("kSecPrivateKeyAttrs"), private_attrs),
                    ]
                )
                cf.release(number, private_attrs)
                err = ctypes.c_void_p()
                key = cf.S.SecKeyCreateRandomKey(attrs, ctypes.byref(err))
                cf.release(attrs)
                if not key:
                    statuses.append(cf.status(err))
                    failures.append(f"{protection}: {cf.error(err)}")
                    continue
                try:
                    spki = self._public_point(int(key))
                finally:
                    cf.release(int(key))
                return KeyHandle(
                    backend=SECURE_ENCLAVE, key_id=key_id_for(spki), spki=spki, presence=True
                )
            # The framework's own text stays first and stays precise; the remedy is
            # appended once per distinct status so the operator is told what to do
            # WITHOUT the failure being softened into a generic sentence.
            detail = "; ".join(failures)
            message = f"the Secure Enclave refused to create an operator key ({detail})"
            remedies = [
                remedy
                for remedy in dict.fromkeys(secure_enclave_remedy(s) for s in statuses)
                if remedy
            ]
            if remedies:
                message += " — " + " ; ".join(remedies)
            raise KeyBackendError(message)
        finally:
            cf.release(tag)

    def _cf_number(self, value: int) -> int:
        """A ``CFNumberRef`` holding ``value``. THE CALLER OWNS the +1 reference.

        Small and deliberate: ``CFNumberCreate`` is not part of the surface
        ``_CF`` declares, so it is wrapped here. The ownership is the ordinary
        CoreFoundation rule and NOT what this docstring claimed before: a dictionary
        built with ``kCFTypeDictionaryKeyCallBacks``/``ValueCallBacks`` (see
        :meth:`_CF.dict`) RETAINS what it holds, so putting this number into one
        does not consume the reference the caller holds. ``create`` releases it, and
        its predecessor leaked one CFNumberRef per successful call.
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
        point = self._public_point(key)
        handle = KeyHandle(
            backend=SECURE_ENCLAVE,
            key_id=key_id_for(point),
            spki=point,
            presence=True,
        )
        return _SecureEnclaveSigner(handle, key, cf)


class _SecureEnclaveSigner(Signer):
    """Signs through the Secure Enclave. EVERY call raises the presence prompt."""

    def __init__(self, handle: KeyHandle, key_ref: int, cf: _CF) -> None:
        super().__init__(handle)
        self._key = key_ref
        self._cf = cf

    def sign(self, message: bytes) -> bytes:
        """Sign through the Secure Enclave. EVERY call raises the presence prompt.

        THE PROMPT CANNOT CARRY OUR COPY, and that is a measured property of the
        API rather than an omission (agent review round 6, D3/U3).
        ``SecKeyCreateSignature`` takes no parameters dictionary, so there is
        nowhere to pass a reason; ``kSecUseOperationPrompt`` was the key that
        would have carried one, and Apple deprecated it in macOS 11
        (availability 10.10-11.0). The human's information therefore comes from
        the surface that can speak: ``effect_copy`` on stderr for the CLI, the
        pane notice for an attached viewer, the log line otherwise — each wired
        at its own call site, and ``docs/design/approval-authority.md`` records
        the bound rather than claiming the sheet itself says it.
        """
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

    IT IS NOT IMPLEMENTED ON THIS BUILD, and both halves of that are enforced
    rather than described (agent review round 6, R6-2/R6-8 = UX round 6, U7):
    ``create`` raises, so ``supported`` must not answer "this host has one", and
    the ladder must not return this object, or ``lop operator init`` dies with an
    internal backend name on a host whose honest level is ``file-only``. Windows
    is therefore a ``file-only`` host HERE, and ``lop operator status`` says so.
    """

    def supported(self) -> bool:
        """Whether this BUILD can create the key — not whether the host is Windows.

        The conflation was the defect: ``os.name == "nt"`` is a fact about the
        host, and the caller of this predicate (``choose_backend``) promises "the
        first backend this host both HAS and can USE". A build that cannot make
        the key answers ``False`` for both, which is what sends ``auto`` down the
        ladder to ``file-only`` — the level the operator is actually getting.
        """
        return False

    def create(self) -> KeyHandle:  # pragma: no cover
        raise KeyBackendError(
            "the Windows presence backend (CNG) is not implemented on this build; "
            "run `lop operator init --backend file-only` — and note that a file-backed "
            "operator key raises no consent prompt, which `lop operator status` reports "
            "as the level `operator-file-only`"
        )

    def load(self) -> Signer | None:  # pragma: no cover
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
    if not CngBackend().supported() and os.name == "nt":
        # WINDOWS IS NOT A PRESENCE TIER ON THIS BUILD (agent review round 6,
        # R6-2 = UX round 6, U7). `auto` promises "the first backend this host
        # both HAS and can USE"; returning `CngBackend` on `os.name == "nt"`
        # ignored the second half, so `lop operator init` exited 1 with an
        # internal backend name instead of creating the key it could have
        # created — and instead of reporting the `file-only` level that host
        # actually has. Asking `supported()` rather than hardcoding a second
        # opinion about Windows is what makes this self-correcting: when a build
        # implements CNG, `supported()` becomes True and this branch stops
        # applying without a second edit here.
        return FileKeyBackend(default_file_path(config_root))
    if os.name == "nt":  # pragma: no cover — Windows only; CI runs POSIX
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
