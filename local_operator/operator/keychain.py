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
   keychain admits a caller only when the caller's CODE SIGNATURE carries the
   keychain entitlement AND a PROVISIONING PROFILE authorizes it. Measured on this
   host, one shape per line, identity ``Developer ID Application``:

   * bare tool, signed, no entitlement -> ``errSecMissingEntitlement`` (-34018);
   * bare tool, signed, entitlement carried -> -34018 as well;
   * app-like bundle, entitlement, NO embedded profile -> -34018 with only the
     application identifier, and **SIGKILL (137)** once ``keychain-access-groups``
     is added;
   * app-like bundle, entitlement, profile embedded -> **PASS** (key created,
     exported as a 65-byte uncompressed P-256 point, re-found by tag, deleted).

   So -34018 does NOT mean "this process has no code signature" — the signed bare
   tool gets it too. It means "no authorized keychain entitlement", and the profile
   is what authorizes one; the one shape the kernel cannot even report is a bundle
   that claims a keychain access group its profile does not grant, which is why
   this build signs with ``com.apple.application-identifier`` only (see
   ``packaging/macos/keyagent.entitlements``). Ad-hoc signing is not a workaround:
   with ``keychain-access-groups`` / ``application-identifier`` in an ad-hoc
   signature the kernel kills the process (SIGKILL, 137).

   WHICH IS WHY THIS FILE NO LONGER MAKES THE NATIVE CALLS. A Python runtime cannot
   carry that signature: the harness is installed by ``uv tool install`` / ``pip``,
   at paths and on hosts that never see this repository. The Enclave verbs are made
   by a signed one-shot helper that ships in the macOS wheel
   (``local_operator/operator/macos/``, built from
   ``packaging/macos/lop-keyagent/se-keyagent.c``), and the in-process ctypes
   sequence that used to live here — which could never work, and no test on any host
   could exercise — is gone rather than kept as a second implementation.

   THE ENTITLEMENT GATES READING TOO, which is the finding that shapes this module:
   a Secure Enclave key created by an entitled process is INVISIBLE to an unsigned
   one, and the answer it gets is ``errSecItemNotFound`` (-25300, measured) — "not
   found", not "refused". So every verb, including "is there a key?", is answered
   by the helper (see :attr:`SecureEnclaveBackend.load`), and no code path here may
   conclude "no operator key" from a query of its own.

The consequence for TESTS is recorded here because it is easy to get wrong: a
test can never exercise this backend without writing an item to the operator's
login keychain, and an unsigned runtime cannot exercise it at all, so the
backend's contract is exercised through :class:`FileKeyBackend`, through a FAKE
helper speaking the JSON protocol (``tests/unit/operator/test_operator_keyagent.py``,
no Secure Enclave and no keychain writes), and through the framework-level probes
in ``tests/unit/operator/test_operator_authority.py`` that assert structure
without creating a key; the end-to-end path is the signed helper itself, which the
release job builds and does notarize-gate on, and which ``lop operator init``
uses.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.operator.macos import keyagent
from local_operator.operator.verify import key_id_for

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

    def __init__(self, message: str, *, status: int | None = None) -> None:
        """``status`` is the ``OSStatus`` the OS returned, when there was ONE such code.

        An attribute rather than only prose because a caller sometimes has to BRANCH on
        it rather than read it: the opt-in hardware test must SKIP (not fail) when the OS
        itself refuses this caller's entitlement, and matching that on the message text
        would be the string-sniffing this removes (agent review round 1, R1-6).

        UNANIMOUS OR ``None``, never the first refusal's code: a caller branching on this
        must fail closed, and a ladder whose classes refused with DIFFERENT codes has no
        single code to report — handing it one class's would silently mis-attribute the
        remedy (agent review round 2, R2-1). ``None`` also covers "the framework refused
        without publishing an error", so ``0`` — a SUCCESS code — is never carried here.
        """
        super().__init__(message)
        self.status = status


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

    def sign(self, message: bytes, *, timeout: float | None = None) -> bytes:
        """Sign ``message``. ``timeout`` bounds the WAIT for a human, where one is needed.

        Optional and keyword-only so the one call site every surface shares
        (``sign.sign_message``) can bound a presence-gated signature without the file
        backend, which returns immediately, having to care.
        """
        raise NotImplementedError  # pragma: no cover — interface

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

    def sign(self, message: bytes, *, timeout: float | None = None) -> bytes:
        # ``timeout`` is accepted and ignored: a software key costs no gesture, so there
        # is nothing to wait for. The parameter exists so one call site serves both
        # backends without asking which kind it holds.
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        return self._private.sign(message, ec.ECDSA(hashes.SHA256()))


# ---------------------------------------------------------------------------
# macOS — Secure Enclave, presence-gated (through the signed key agent)
# ---------------------------------------------------------------------------
#
# WHAT USED TO BE HERE, and why it is not. This section formerly held a ctypes
# binding of the CoreFoundation + Security.framework call sequence that creates an
# Enclave key: the access-control object, the typed-callback dictionaries, the ladder
# over two protection classes. It was correct — its flag pair is asserted against the
# SDK header below — and it could never work, because the entitlement that admits a
# process to the data-protection keychain comes from the CODE SIGNATURE and an
# embedded PROVISIONING PROFILE, and a Python runtime has neither. Two
# implementations of one native sequence, one of which no test on any host can
# exercise, is a second way of doing things rather than a fallback, so the sequence
# lives only in the C helper now (packaging/macos/lop-keyagent/se-keyagent.c) and this
# section is the client of it.


#: ``OSStatus`` values from the presence store, named so the code below reads as prose
#: rather than as bare integers.
#:
#: ``-50`` is ``errSecParam``, the framework's GENERIC parameter refusal, and it does not
#: mean one thing: at the access-control call it is a flag pair Apple will not accept,
#: while at key generation it is "inconsistent private key parameters". A diagnosis is
#: therefore keyed by the call site as well — see :func:`secure_enclave_diagnosis`
#: (design round 1, D2/R1-1). ``-34018`` is ``errSecMissingEntitlement``.
_ERR_SEC_PARAM = -50
_ERR_SEC_MISSING_ENTITLEMENT = -34018
#: ``errSecItemNotFound``. READ IT AS "the entitled process found nothing", never as
#: "there is no key": an unsigned query returns this for a key that exists, which is the
#: trap :meth:`SecureEnclaveBackend.load` is built to close.
_ERR_SEC_ITEM_NOT_FOUND = -25300
#: ``errSecUserCanceled``. The human dismissed the presence prompt: a normal outcome
#: rather than a defect, and the one refusal that must not be reported as a failure of
#: the machine.
_ERR_SEC_USER_CANCELED = -128

#: The two call sites a refusal can come from, as the vocabulary the diagnosis is keyed
#: on. Named constants rather than a boolean because ``errSecParam`` is read differently
#: at each (see above).
ACCESS_CONTROL_REFUSED = "access control"
KEY_GENERATION_REFUSED = "key generation"
#: The signing call, which is where the presence prompt lives and therefore the only
#: site that can answer ``errSecUserCanceled``.
SIGNATURE_REFUSED = "signature"
#: The lookup, which is where "there is no key stored under this tag" is answered. A
#: site of its own because its ``errSecItemNotFound`` is a statement about the STORE
#: (and, per the trap above, only believable from the entitled process), while the same
#: code at key generation would be a create that could not get as far as the item.
KEY_LOOKUP_REFUSED = "key lookup"

#: ``(site, OSStatus) -> the lines that explain it``. The FIRST line is the diagnosis and
#: LEADS the message; the rest say what it costs and what to do with it, one per line, in
#: the shape :class:`CngBackend` established for the same situation: name the fallback,
#: name what giving up the gesture costs, and name the level ``lop operator status`` will
#: report. Naming the fallback is NOT naming it as the sanctioned path — a file-backed
#: key is a downgrade, and the copy has to say which protection is being given up
#: (design round 1, D1).
#: The downgrade sentence, shared by the two KEY-GENERATION refusals.
#:
#: At both sites a presence-gated key cannot be made for a reason the host will not
#: change, and ``file-only`` is the level this host can actually enforce — so the honest
#: next action is the fallback, and it is the same sentence twice. Shared rather than
#: written out twice so the two sites cannot drift into describing the same fallback
#: differently, and written as the shape this table's docstring requires: name the
#: fallback, name what giving up the gesture costs, name the level ``lop operator
#: status`` reports (design round 1, D1).
#:
#: It also replaces a steer that led nowhere: the keygen ``-50`` entry used to end by
#: pointing at ``lop operator status`` alone, and on the state a keygen refusal leaves —
#: nothing created, no anchor — that report says ``private-half backend : (none)`` and its
#: own way out names `lop operator init`, the command that just refused (design round 1,
#: D1-1).
_FILE_ONLY_FALLBACK = (
    "run `lop operator init --backend file-only` for a file-backed operator key — and "
    "note that a file-backed key raises no presence prompt, and any process running as "
    "you can read it, which `lop operator status` reports as the level "
    "`operator-file-only`"
)

_SECURE_ENCLAVE_DIAGNOSES: dict[tuple[str, int], tuple[str, ...]] = {
    (ACCESS_CONTROL_REFUSED, _ERR_SEC_PARAM): (
        "the access-control flags were refused (errSecParam), which no host accepts: "
        "kSecAccessControlUserPresence may be combined only with "
        "kSecAccessControlApplicationPassword and kSecAccessControlPrivateKeyUsage",
        "that is a defect in the build carrying the flag pair, not a state this host can "
        "work around — no protection class can succeed with it, so the fix is a build "
        "carrying the corrected pair: a build from `main` has it now, `lop update` "
        "takes the next release, and `lop-update` rebuilds a checkout",
    ),
    (KEY_GENERATION_REFUSED, _ERR_SEC_MISSING_ENTITLEMENT): (
        "the key agent could not reach the data-protection keychain: the Secure Enclave "
        "keeps its keys there, and that keychain admits the caller only when its code "
        "signature carries the keychain entitlement AND its embedded provisioning "
        "profile authorizes the application identifier it claims "
        "(errSecMissingEntitlement) — so this is a signature/profile pair to fix, not a "
        "flag to change (`lop operator status` reports whether the key agent itself "
        "verifies)",
        _FILE_ONLY_FALLBACK,
    ),
    (SIGNATURE_REFUSED, _ERR_SEC_USER_CANCELED): (
        "the presence prompt was cancelled (errSecUserCanceled), so nothing was signed "
        "and the operator key is unchanged",
    ),
    (KEY_LOOKUP_REFUSED, _ERR_SEC_ITEM_NOT_FOUND): (
        "there is no operator key under this tag on this host: run `lop operator init` "
        "(this is the KEY AGENT's answer, which is the only process that can see the "
        "item)",
    ),
    (KEY_GENERATION_REFUSED, _ERR_SEC_PARAM): (
        "key generation was refused (errSecParam): the parameters are inconsistent for a "
        "Secure Enclave key, and a protection class this host will not accept is the "
        "usual cause — the classes tried are listed below",
        _FILE_ONLY_FALLBACK,
    ),
}

#: A CoreFoundation object address as the framework PRINTS it: a separator, then ``0x`` and
#: at least 9 hex digits (`> 0x75929d8380`, `> 0x10137ede0`).
#:
#: NINE, NOT EIGHT, and the threshold is the measured one: every address the framework has
#: printed on this host is 9-10 digits (`0x10137ede0`, `0x10375c430`, `0x75929d8380`,
#: `0x79b58040c0`, `0x77d52fc540`), while a real 8-digit value in the same text is a
#: PARAMETER — `0x00000008` for `kSecAttrKeyType`, `0xdeadbeef` — which the strip has no
#: business deleting. Eight would eat those; nine keeps every address ever measured here
#: and spares the parameters (agent review round 1, R1-1).
#:
#: DELIBERATELY NARROW, because the first version was not: ``\s*0x[0-9a-fA-F]+`` also
#: deleted a small hex literal (`id=0x7f8 ref=0x1` became `id= ref=`), truncated a path
#: segment (`/tmp/0x9f/probe.pem` became `/tmp//probe.pem`) and — because ``\s*`` matches a
#: newline — could join two framework lines. The separator is required, so an address at
#: the very start of a description is left alone; that is not a shape this framework
#: produces, and the docstring below says so rather than pretending otherwise (agent
#: review round 2, R2-2 / QA round 2, Q2-2).
_RUN_ADDRESS = re.compile(r"[ \t]0x[0-9a-fA-F]{9,}\b")


def without_run_addresses(text: str) -> str:
    """The framework's words with per-run object ADDRESSES removed, nothing else.

    ``0x75929d8380`` differs on every run, which is why it is stripped: this text is
    printed to be pasted into a bug report, two identical failures must not look
    different, and the same strip is what lets two protection classes' identical refusals
    collapse into one detail line. What is removed is only a separator followed by ``0x``
    and 9 or more hex digits — a short hex literal, a length-8 parameter such as
    ``0x00000008``, a hex path segment, and a line break all survive, as
    :data:`_RUN_ADDRESS` explains. The framework's own wording, including the object's
    description, is kept.
    """
    return _RUN_ADDRESS.sub("", text)


def secure_enclave_diagnosis(site: str, status: int) -> tuple[str, ...]:
    """The lines that explain ONE refusal, or ``()`` for a status we cannot explain.

    Keyed by the call SITE as well as the code, because ``errSecParam`` is the
    framework's generic parameter refusal and does not mean the same thing at both sites
    (design round 1, D2/R1-1). An unknown pair contributes NOTHING rather than a guessed
    cause; the framework's own words are appended either way. Classification lives in its
    own seam so it is testable without a Secure Enclave — the codes above cannot all be
    produced on one host.
    """
    return _SECURE_ENCLAVE_DIAGNOSES.get((site, int(status)), ())


#: What the message says when the framework's refusal has no diagnosis this file can
#: stand behind. Keyed by CALL SITE because the verb matters to the reader: a key that
#: could not be created and a signature that was refused are different failures with
#: different next steps, and one headline for both would have the message name an
#: operation that did not happen (found while the key agent learned to report a
#: signature site).
_REFUSAL_HEADLINES: dict[str, str] = {
    ACCESS_CONTROL_REFUSED: "the Secure Enclave refused to create an operator key",
    KEY_GENERATION_REFUSED: "the Secure Enclave refused to create an operator key",
    SIGNATURE_REFUSED: "the Secure Enclave refused the operator key's signature",
    KEY_LOOKUP_REFUSED: "the operator key could not be read from the Secure Enclave",
}


def secure_enclave_refusal_message(refusals: dict[tuple[str, int, str], list[str]]) -> str:
    """The message for a refused key creation: DIAGNOSIS first, framework detail last.

    ``refusals`` maps ``(site, status, framework detail) -> the protection classes that
    produced it``, so a refusal that is identical for every class is said ONCE with the
    classes listed beside it instead of repeating one sentence per class, and the operator
    reaches the next command in the first two lines rather than at the end of a long
    single line (design round 1, D3).

    The framework's own text is never replaced by ours — it is the last line, kept
    verbatim (minus per-run addresses), because it is what a report should quote.
    """
    lines: list[str] = []
    written: set[tuple[str, int]] = set()
    for site, status, _detail in refusals:
        if (site, status) in written:
            continue
        written.add((site, status))
        lines.extend(secure_enclave_diagnosis(site, status))
    if not lines:
        # No explanation we can stand behind: say the plain fact, IN THE VERB OF THE
        # SITE that refused, and let the framework's detail line below carry
        # everything we actually know.
        headline = next(
            (
                _REFUSAL_HEADLINES[site]
                for site, _status, _detail in refusals
                if site in _REFUSAL_HEADLINES
            ),
            "the Secure Enclave refused the operator key operation",
        )
        lines.append(headline)
    for (site, status, detail), classes in refusals.items():
        # The code is prefixed only when the framework's own text does not already state
        # it: ``CFErrorCopyDescription`` normally renders "… (OSStatus error -34018 - …)",
        # and saying the same number twice on one line is the restatement this message
        # exists to avoid (design round 1, D3). The framework's words are never edited.
        #
        # A status of 0 means NO error object was published, so there is no code to state:
        # printing "OSStatus 0 -" would put a SUCCESS code in a failure line (agent review
        # round 2, R2-1 / QA round 2, Q2-1).
        stated = not status or re.search(rf"error\s+{re.escape(str(status))}\b", detail)
        lines.append(
            f"framework detail: {site} — {', '.join(classes)}: "
            f"{'' if stated else f'OSStatus {status} - '}{detail}"
        )
    # ONE LINE PER SENTENCE, STARTING AT COLUMN 0, with no hand-set indent: the terminal
    # is what wraps, so a wrapped continuation must not be stranded mid-indent (design
    # round 2, D2-2 — the same convention `test_the_spawn_only_status_lines_are_wrapped_
    # by_the_terminal_not_by_hand` pins for the status block).
    return "\n".join(lines)


#: The remedy sentence the key-agent states share, and the cost of the alternative it
#: names: the SAME shape the table above uses for a `file-only` fallback, because it is
#: the same offer with the same price.
_KEYAGENT_REINSTALL = (
    "reinstall the macOS wheel with `uv tool install local-operator --force` "
    "(`lop-update` rebuilds a checkout) to get the key agent; to keep a file-backed key "
    "instead run `lop operator init --backend file-only` — it raises no presence prompt, "
    "and any process running as you can read it, which `lop operator status` reports as "
    "the level `operator-file-only`"
)

#: ``key-agent state -> (what `init` says, what `status` reports)``.
#:
#: TWO REGISTERS, ONE TABLE: `init` prints the state, the remedy and the cost of the
#: alternative, while `status` has a one-line field to fill. Describing one broken
#: install differently in the two commands is how a reader learns to trust neither, so
#: both come from here. The keys are the ``kind`` values
#: :class:`local_operator.operator.macos.keyagent.KeyagentError` raises.
#:
#: Note what is NOT here: any state that would let the runtime conclude "there is no
#: key" on its own. ``no-key`` is the key agent's answer, and it is the only entry that
#: reports absence.
_KEYAGENT_STATES: dict[str, tuple[str, str]] = {
    "absent": (
        "the macOS key agent is not installed: this installation has no "
        "`lop-keyagent.app` (an sdist install, or a wheel for another platform), so "
        "nothing here can reach the operator key. " + _KEYAGENT_REINSTALL,
        "the macOS key agent is not installed (broken install)",
    ),
    "unverified": (
        "the macOS key agent is present but is not ours: its signature does not verify, "
        "or it carries no embedded provisioning profile, so the kernel would refuse it "
        "before it could run — the bundle is verified BEFORE it is executed for exactly "
        "that reason. " + _KEYAGENT_REINSTALL,
        "the key agent failed verification",
    ),
    "killed": (
        "the kernel refused the key agent's keychain entitlement: its embedded "
        "provisioning profile is missing, stale, or does not authorize its application "
        "identifier, so the process was killed before it could run (measured shape: "
        "SIGKILL, exit 137). " + _KEYAGENT_REINSTALL,
        "the key agent was killed: bad or missing embedded profile",
    ),
    "no-key": (
        "no operator key on this host: run `lop operator init`",
        "no key on this host",
    ),
    "cancelled": (
        "the presence prompt was cancelled: nothing was signed and the key is unchanged",
        "the presence prompt was cancelled",
    ),
    "timeout": (
        "the presence prompt was not answered in time: nothing was signed and the key is "
        "unchanged",
        "the presence prompt was not answered",
    ),
    "protocol": (
        "the key agent does not match this runtime (its reply is not the protocol this "
        "build speaks) — reinstall so both come from one wheel",
        "the key agent does not match this runtime",
    ),
}


def keyagent_state_copy(state: str, *, long: bool = True) -> str:
    """The copy for one key-agent state, in the register ``init`` or ``status`` uses."""
    entry = _KEYAGENT_STATES.get(state)
    if entry is None:  # pragma: no cover — every kind is in the table, pinned by a test
        return f"the key agent failed ({state}); reinstall the macOS wheel"
    return entry[0] if long else entry[1]


def keyagent_refusal_message(exc: "keyagent.KeyagentError") -> str:
    """One ``KeyagentError`` as the message an operator reads.

    A refusal that came from the framework carries the same ``site``/``status``/``detail``
    triples the in-process refusal used to, so it goes through the SAME builder —
    diagnosis first, framework detail last, one line per protection class — rather than
    gaining a second, thinner explanation of the same OSStatus. Everything else is a
    state of the installation, and gets :data:`_KEYAGENT_STATES`.
    """
    if exc.kind != "refused":
        return keyagent_state_copy(exc.kind)
    entries = list(exc.refusals) or [
        {"site": exc.site or KEY_GENERATION_REFUSED, "status": exc.status, "detail": exc.detail}
    ]
    grouped: dict[tuple[str, int, str], list[str]] = {}
    for entry in entries:
        raw_status = entry.get("status")
        status = int(raw_status) if isinstance(raw_status, (int, float)) else 0
        site = str(entry.get("site") or KEY_GENERATION_REFUSED)
        detail = without_run_addresses(str(entry.get("detail") or ""))
        grouped.setdefault((site, status, detail), []).append(
            str(entry.get("protection") or "(no protection class)")
        )
    return secure_enclave_refusal_message(grouped)


class SecureEnclaveBackend:
    """The macOS presence-gated backend: a CLIENT of the signed key agent.

    Every verb is one short-lived subprocess of
    ``local_operator/operator/macos/lop-keyagent.app/Contents/MacOS/lop-keyagent``,
    invoked directly rather than through LaunchServices, because the entitlement that
    admits a process to the keychain is a property of the CODE SIGNATURE — and this
    process has none. Nothing in this class touches CoreFoundation, the keychain or the
    Secure Enclave.

    THE CONSTANTS BELOW STAY, and they are not decoration: the C source that now makes
    these calls is asserted against the same SDK header, and against these values, by
    ``tests/unit/operator/test_operator_authority.py``. Drift between the two copies of
    one native sequence is the hazard this arrangement creates, so it is pinned.
    """

    #: The flag pair, from ``Security.framework/Headers/SecAccessControl.h``:
    #: ``kSecAccessControlUserPresence = 1u << 0`` and
    #: ``kSecAccessControlPrivateKeyUsage = 1u << 30``.
    PRIVATE_KEY_USAGE = 1 << 30
    USER_PRESENCE = 1 << 0

    #: The protection classes, STRICTEST FIRST. The key agent walks them in this order
    #: and reports the class it achieved, so a host that accepts passcode-set never
    #: settles for unlocked. Both rungs were measured accepted under an entitled
    #: process; the second exists for a host whose keychain has no passcode to bind to.
    PROTECTION_LADDER = (
        "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly",
        "kSecAttrAccessibleWhenUnlockedThisDeviceOnly",
    )

    def __init__(self) -> None:
        """No native state: every verb is a process, and nothing is loaded here."""

    def _client(self) -> "keyagent.KeyagentClient":
        """A client for the installed helper, under the tag this module names.

        The tag is read HERE rather than captured at construction: ``APPLICATION_TAG``
        is a module constant, and a test replaces it to point the whole path at a test
        item rather than at the operator's own key.
        """
        return keyagent.KeyagentClient(tag=APPLICATION_TAG)

    def supported(self) -> bool:
        """Whether THIS INSTALLATION can reach an entitled process.

        NOT "is there a Secure Enclave on this host". On macOS the build can do this and
        only the INSTALL can be wrong, so the honest predicate is about the key agent in
        this installation: present, verifying as ours, and able to run with its
        entitlement — the :class:`CngBackend` precedent read in the other direction
        (there the BUILD cannot do it; here the build can and a broken install would
        otherwise be indistinguishable from a host limitation).

        A ``False`` here is NOT a licence to fall back. ``init`` and ``sign`` hard-fail
        with the copy that names the remedy, because ``file-only`` is a different and
        weaker level and has to be asked for by name.
        """
        return self.health()[0]

    def health(self) -> tuple[bool, str]:
        """:meth:`supported` with the reason attached, for the report that must give it."""
        state = keyagent.helper_health()
        if state.ok:
            return True, state.detail
        return False, keyagent_state_copy(state.kind, long=False)

    def create(self) -> KeyHandle:
        """Create the operator key through the key agent, or return the one already here.

        Idempotent by construction: the key agent answers ``reused`` when an item already
        exists under the tag, which is what makes a second ``lop operator init`` a report
        rather than a second key — and a second key would invalidate every device
        certificate signed under the first anchor.
        """
        try:
            created = self._client().create()
        except keyagent.KeyagentError as exc:
            raise KeyBackendError(keyagent_refusal_message(exc), status=exc.status) from exc
        return KeyHandle(
            backend=SECURE_ENCLAVE,
            key_id=key_id_for(created.point),
            spki=created.point,
            presence=True,
        )

    def load(self) -> Signer | None:
        """A signer for the key the KEY AGENT finds, or ``None`` when it finds none.

        THE -25300 TRAP, closed here. An unsigned process asking for this item gets
        ``errSecItemNotFound`` for a key that exists, so this method may never answer
        from a query of its own: ``None`` means the ENTITLED process said the item is not
        there, and that is the only source of an absence claim the runtime may believe.
        A key agent that is missing, unverifiable or killed RAISES instead, because
        "broken install" and "no key yet" have different remedies and converting the
        first into the second is the silent failure this whole design exists to prevent.
        """
        try:
            point = self._client().public()
        except keyagent.KeyagentError as exc:
            if exc.kind == "no-key":
                return None
            raise KeyBackendError(keyagent_refusal_message(exc), status=exc.status) from exc
        return _KeyagentSigner(
            KeyHandle(
                backend=SECURE_ENCLAVE,
                key_id=key_id_for(point),
                spki=point,
                presence=True,
            ),
            tag=APPLICATION_TAG,
        )


class _KeyagentSigner(Signer):
    """Signs through the key agent. EVERY call raises the presence prompt.

    The prompt is the tier: it is raised by the OS inside the entitled process, so no
    caller of this — in-process, in a subprocess, or a model's tool child — can obtain a
    signature without a human gesture. The sheet cannot carry this project's copy
    (``SecKeyCreateSignature`` takes no parameters dictionary and ``kSecUseOperationPrompt``
    was deprecated in macOS 11), so what names the session and the effect is the caller's
    own line on stderr (``sign.py``); what this adds is the bound — past its ``timeout``
    the helper is killed and nothing is signed.
    """

    def __init__(self, handle: KeyHandle, *, tag: str, timeout: float | None = None) -> None:
        super().__init__(handle)
        self._tag = tag
        self._timeout = timeout if timeout is not None else keyagent.SIGN_TIMEOUT_SECONDS

    def sign(self, message: bytes, *, timeout: float | None = None) -> bytes:
        """Sign exactly these bytes through the entitled process. THIS PROMPTS."""
        try:
            return keyagent.KeyagentClient(tag=self._tag).sign(
                message, timeout=timeout if timeout is not None else self._timeout
            )
        except keyagent.KeyagentError as exc:
            raise KeyBackendError(keyagent_refusal_message(exc), status=exc.status) from exc

    def close(self) -> None:
        """Nothing to release: the one-shot helper that held the key has exited.

        Kept because :class:`Signer` requires it and every surface ends a signing session
        with it — the file backend's signer does hold an object, and a caller must not
        have to know which kind it is holding to know whether to close it.
        """


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

    THE ONE PLACE THE PROMISE IS DELIBERATELY NOT KEPT, because keeping it would
    be worse: on macOS the presence backend is returned whether or not its key agent
    is usable, and it RAISES when it is not (design §6). "The first backend this host
    has and can use" presumes the alternatives are equivalent ways to reach the same
    level; a file-backed key is not — it is a key any process running as you can read
    — so falling back to it silently would turn a broken install into a weaker key
    that reports success. `--backend file-only` is the only route to that level.
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
        # NOT gated on `SecureEnclaveBackend().supported()`, and that is the whole
        # point (design §6): on macOS the key agent is part of the INSTALL, so a
        # missing or unverifiable one is a BROKEN INSTALLATION rather than a host
        # limitation. Returning the file backend here would be the silent downgrade
        # this design forbids — the operator would get a key any process running as
        # them can read while `init` reported success. `create`/`load` raise with the
        # remedy instead, and the only way to a file-backed key is to say
        # `--backend file-only`.
        return SecureEnclaveBackend()
    return FileKeyBackend(default_file_path(config_root))


def _named_backend(name: str, *, config_root: Path) -> Any:
    if name == SECURE_ENCLAVE:
        return SecureEnclaveBackend()
    if name == CNG_PRESENCE:
        return CngBackend()
    return FileKeyBackend(default_file_path(config_root))
