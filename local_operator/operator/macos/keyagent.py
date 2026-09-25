"""The Python side of the operator key's presence-bearing process.

WHY THIS PACKAGE EXISTS. A Secure Enclave key can only live in the
data-protection keychain, and that keychain admits a caller only when the caller's
code signature carries the keychain entitlement *and* an embedded provisioning
profile authorizes it. A Python interpreter cannot be signed that way — the
runtime is installed by ``uv tool install`` / ``pip``, at many paths, on hosts
that never see this repository — so **no Python process, however correct its
CoreFoundation calls, can create, find or use this key on any Mac**. That is the
measurement the whole design turns on: a key created by an entitled process is
invisible (``errSecItemNotFound``, -25300) to an unsigned one, and the answer the
unsigned process gets is "not found", not "refused".

So the native work lives in a small Developer-ID-signed app bundle that ships
inside the macOS wheel (``lop-keyagent.app``, built by
``packaging/macos/assemble_keyagent_bundle.sh``), and every verb — create, load,
sign — is a one-shot subprocess call to it. :mod:`local_operator.operator.keychain`
owns the failure copy and the diagnosis vocabulary; this module owns the protocol,
the pre-flight verification and the bounded execution, and it raises
:class:`KeyagentError` carrying machine-readable facts rather than prose.

THE RULE THIS MODULE EXISTS TO ENFORCE, stated once because it is easy to
reintroduce: **nothing here may ever let a caller conclude "there is no operator
key" from a query this process made itself.** ``exists()``, ``public()`` and
``load()`` all answer from the entitled process, and ``load()`` returning ``None``
means the *helper* said ``errSecItemNotFound`` — never a ``SecItemCopyMatching``
made here.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The protocol version the helper echoes in every reply. Helper and client ship
#: in the same wheel, so a mismatch means a broken or mixed install, and the
#: honest answer is to refuse rather than to interpret.
PROTOCOL = 1

#: The identity the helper must have been signed under. Pinned because "the
#: artefact is ours" is the same test the anchor applies to a public half: a
#: helper signed by someone else is not this team's helper, and the fact that a
#: reader can determine it from a released wheel is exactly why pinning it costs
#: nothing (it is in the signature, and ``codesign -d --entitlements`` prints it).
#:
#: A FORK WITH ITS OWN APP ID changes these two constants, ``Info.plist``,
#: ``packaging/macos/keyagent.entitlements`` and its profile together. Until then
#: a wheel signed for another identity is REFUSED here rather than run.
BUNDLE_IDENTIFIER = "com.local-operator.lopkeyagent"
TEAM_IDENTIFIER = "SHA2U6KT7V"

#: The bundle and the executable inside it, as the build writes them.
BUNDLE_NAME = "lop-keyagent.app"
EXECUTABLE_NAME = "lop-keyagent"

#: Bounds. The quick verbs must finish in a moment or something is wedged; `sign`
#: waits for a human, so its bound is a caller-side one (``--timeout`` on
#: ``lop operator sign``) and the helper has no internal timeout at all.
QUICK_TIMEOUT_SECONDS = 10.0
SIGN_TIMEOUT_SECONDS = 180.0

#: How long a terminated helper has to exit before it is killed outright.
_KILL_GRACE_SECONDS = 2.0

#: The tag ``helper_health`` probes under. A tag NOTHING writes to, on purpose: the
#: question "can this installation reach an entitled process?" must not depend on
#: whether the operator's own key exists, and a tag with no item is the cheapest run
#: that still proves the process was entitled — it answers errSecItemNotFound, not
#: errSecMissingEntitlement, and is not killed.
HEALTH_TAG = "com.local-operator.lopkeyagent.health"


class KeyagentError(RuntimeError):
    """One way the key agent failed, as facts rather than as prose.

    ``kind`` is the closed set below, and it is what a caller branches on; the
    user-facing sentence is built in
    :mod:`local_operator.operator.keychain`, so the copy for these states lives
    beside the copy for every other Secure Enclave refusal instead of in a second
    place that can drift from it.

    * ``absent`` — the bundle or its executable is not there, or is not runnable.
    * ``unverified`` — present, but the pre-flight refused it.
    * ``killed`` — exited by signal (the kernel refused the entitlement).
    * ``timeout`` — did not answer inside the bound.
    * ``cancelled`` — the human dismissed the presence prompt (-128).
    * ``no-key`` — the helper says there is no item under the tag (-25300).
    * ``refused`` — the helper ran and the framework refused a call.
    * ``protocol`` — it answered something this client does not understand.
    """

    def __init__(
        self,
        kind: str,
        detail: str = "",
        *,
        status: int | None = None,
        site: str = "",
        exit_code: int | None = None,
        refusals: Sequence[dict[str, Any]] = (),
    ) -> None:
        super().__init__(detail or kind)
        self.kind = kind
        self.detail = detail
        self.status = status
        #: The call SITE the helper reported (``access control`` / ``key generation`` /
        #: ``signature`` / ``key lookup``), because the framework's generic refusals mean
        #: different things at different sites — see ``keychain._SECURE_ENCLAVE_DIAGNOSES``.
        self.site = site
        self.exit_code = exit_code
        #: The ladder's refusals as the helper reported them, so the message
        #: builder in ``keychain.py`` can group them exactly as the in-process
        #: ladder used to (``(site, status, detail) -> classes``).
        self.refusals = tuple(refusals)


@dataclass(frozen=True)
class KeyPublic:
    """A public half, and whether this call also created it."""

    point: bytes
    reused: bool = False


@dataclass(frozen=True)
class DoctorReport:
    """What ``doctor`` measured: no writes, no prompts."""

    rung: str
    keychain: str
    keychain_status: int
    profile: str


@dataclass(frozen=True)
class KeyagentHealth:
    """Whether this installation can reach an entitled process, and why not.

    ``kind`` is a :class:`KeyagentError` kind when ``ok`` is False, so the reporter can
    choose its own register for the same state rather than parsing ``detail``.
    """

    ok: bool
    detail: str
    kind: str = ""


def helper_bundle_path() -> Path:
    """Where the helper bundle lives in this installation.

    A sibling of this package inside ``site-packages/local_operator/operator/``,
    resolved from ``__file__`` so it follows an editable install, a wheel, a
    ``uv tool`` generation and a plain checkout identically. There is deliberately
    no environment override: an env-var path would be settable by a tool child
    that cannot write ``site-packages``, and a swapped helper must not be
    reachable by anything weaker than a filesystem write (see the design document's
    residual section — a swap cannot forge a signature anyway, because the anchor
    pins the public half, but it should still not be a one-variable attack).
    """
    return Path(__file__).resolve().parent / BUNDLE_NAME


def _run(argv: Sequence[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[bytes]:
    """``subprocess.run`` for a probing command, never raising on a non-zero exit."""
    return subprocess.run(
        list(argv),
        capture_output=True,
        check=False,
        timeout=timeout,
        env={**os.environ, "LC_ALL": "C"},
    )


def _codesign_fields(app: Path) -> dict[str, str]:
    """``codesign -d --verbose=4``'s key lines, as a mapping.

    Parsed rather than assumed because all three of the checks below are the
    difference between running our helper and running someone else's: the
    identifier, the team, and whether the signature verifies at all.
    """
    try:
        result = _run(["codesign", "--display", "--verbose=4", str(app)])
    except OSError as exc:  # pragma: no cover — codesign ships with macOS
        raise KeyagentError("unverified", f"codesign could not be run: {exc}") from exc
    # `codesign -d` writes its report to STDERR, on success as well as failure.
    text = (result.stderr or b"").decode("utf-8", "replace")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key and " " not in key.strip():
            fields[key.strip()] = value.strip()
    fields["__exit"] = str(result.returncode)
    fields["__text"] = text
    return fields


def _profile_decodes(profile: Path) -> bool:
    """Whether ``security cms -D -i`` can DECODE the profile in this environment.

    It builds the certificate chain that verifies the profile's own signature, so it
    needs a keychain context. Measured on macOS 27.0 with a HOME that has no login
    keychain (an isolated test root, a container image): it fails with
    ``security: cert import failed: A default keychain could not be found``, silently —
    no dialog, no prompt — while the same file decodes normally under the operator's own
    HOME. So this answers "does this blob decode HERE", which is a weaker question than
    "does this blob decode", and the caller says which one it got.
    """
    try:
        decoded = _run(["security", "cms", "-D", "-i", str(profile)], timeout=30.0)
    except OSError:  # pragma: no cover — security(1) ships with macOS
        return False
    return decoded.returncode == 0 and bool(decoded.stdout)


def _profile_authorizes(profile: Path) -> tuple[bool, str]:
    """Whether the profile authorizes THIS helper's identity, and how that was decided.

    THE FIRST CHECK IS THE LOAD-BEARING ONE, and it deliberately needs no external tool.
    A provisioning profile is a DER blob and the application identifier is a UTF8 string
    inside it, so the authorization the kernel will enforce can be read straight from the
    file: the pinned ``<team>.<bundle id>`` must appear in its bytes. That string is what
    ``packaging/macos/keyagent.entitlements`` claims and what
    ``assemble_keyagent_bundle.sh`` fails the build without — and the profile's absence
    is measured to be a kernel SIGKILL, which is why it is checked before exec at all.

    The SECOND check, when the environment can decode, proves the blob is not truncated
    or mangled in a way that leaves the string but loses the structure. Where it cannot
    be run the description SAYS SO rather than reporting a pass it did not observe —
    ``uncached in this environment`` is a fact a report can carry; a silent pass is not.
    """
    claimed = f"{TEAM_IDENTIFIER}.{BUNDLE_IDENTIFIER}"
    raw = b""
    try:
        raw = profile.read_bytes()
    except OSError as exc:
        return False, f"the embedded provisioning profile is unreadable: {exc}"
    if claimed.encode("utf-8") not in raw:
        return False, f"the embedded provisioning profile does not authorize {claimed}"
    if _profile_decodes(profile):
        return True, f"{claimed} (profile decodes)"
    return True, f"{claimed} (profile uncached in this environment: no keychain to decode it with)"


def verify_bundle(app: Path) -> str:
    """Pre-flight the helper BEFORE exec'ing it. Returns a one-line description.

    WHY BEFORE, AND NOT BY TRUSTING THE RUN. A missing or stale profile does not
    produce an error: the kernel SIGKILLs the process, so a run that discovers it
    has nothing to report and no way to report it. Every check here therefore has
    to happen while a failure is still a return value.

    The checks, in the order the design fixes them:

    1. the bundle and its executable are present and the executable is runnable;
    2. ``codesign --verify --strict`` passes;
    3. ``Contents/embedded.provisionprofile`` exists and parses, and the app id it
       authorizes matches the bundle identifier's team-prefixed form;
    4. the signature's identifier is ours and its TeamIdentifier is
       :data:`TEAM_IDENTIFIER`.

    Raises :class:`KeyagentError` (``absent`` / ``unverified``) with the failing
    check in ``detail``; the operator-facing sentence is built by the caller.
    """
    executable = app / "Contents" / "MacOS" / EXECUTABLE_NAME
    if not app.is_dir():
        raise KeyagentError("absent", f"no key agent bundle at {app}")
    if not executable.is_file():
        raise KeyagentError("absent", f"the key agent bundle has no {EXECUTABLE_NAME}")
    if not os.access(executable, os.X_OK):
        raise KeyagentError("absent", f"the key agent at {executable} is not executable")

    try:
        verified = _run(["codesign", "--verify", "--strict", str(app)], timeout=60.0)
    except OSError as exc:  # pragma: no cover — codesign ships with macOS
        # Not "unverified": a host with no codesign cannot have built this bundle
        # either, so the artifact being checked is not the one in question.
        raise KeyagentError("absent", f"codesign could not be run: {exc}") from exc
    if verified.returncode != 0:
        tail = (verified.stderr or verified.stdout or b"").decode("utf-8", "replace").strip()
        raise KeyagentError("unverified", f"codesign --verify --strict refused it: {tail}")

    fields = _codesign_fields(app)
    identifier = fields.get("Identifier", "")
    if identifier != BUNDLE_IDENTIFIER:
        raise KeyagentError(
            "unverified",
            f"its signature names {identifier or 'no identifier'}, not {BUNDLE_IDENTIFIER}",
        )
    team = fields.get("TeamIdentifier", "")
    if team != TEAM_IDENTIFIER:
        raise KeyagentError(
            "unverified",
            f"its signature is not from team {TEAM_IDENTIFIER} (it reports "
            f"{team or 'no team identifier'})",
        )

    profile = app / "Contents" / "embedded.provisionprofile"
    if not profile.is_file():
        raise KeyagentError(
            "unverified",
            # Named as load-bearing because the failure it prevents is a kernel
            # kill, which no amount of error handling can turn into a message.
            "it carries no embedded provisioning profile (an entitled bundle "
            "without one is killed by the kernel, not refused)",
        )
    app_id_ok, app_id_detail = _profile_authorizes(profile)
    if not app_id_ok:
        raise KeyagentError("unverified", app_id_detail)
    return f"{identifier} (team {team}); {app_id_detail}"


class KeyagentClient:
    """One-shot calls to the signed helper, bounded and pre-flighted.

    ``tag`` is passed explicitly rather than read from a constant here: the
    product's tag is ``keychain.APPLICATION_TAG`` and the production backend always
    passes that, while a test passes its own so it can exercise the whole path —
    including create and delete — without ever touching the operator's real key.
    """

    def __init__(
        self,
        *,
        tag: str,
        bundle: Path | None = None,
        preflight: Callable[[Path], str] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
        quick_timeout: float = QUICK_TIMEOUT_SECONDS,
        sign_timeout: float = SIGN_TIMEOUT_SECONDS,
    ) -> None:
        self.tag = tag
        self.bundle = Path(bundle) if bundle is not None else helper_bundle_path()
        # Injected rather than a module global so a test can hold the protocol and
        # the failure mapping to account without a signed bundle on the host — and
        # so "the pre-flight runs before EVERY sign" is assertable, which is the
        # one place it must not be cached. Resolved HERE rather than as a default
        # argument, so `None` means the real check at construction time and a test
        # replacing `keyagent.verify_bundle` is not silently ignored.
        self._preflight = preflight or verify_bundle
        self._runner = runner or self._spawn
        self.quick_timeout = quick_timeout
        self.sign_timeout = sign_timeout

    # -- process ------------------------------------------------------------

    def executable(self) -> Path:
        return self.bundle / "Contents" / "MacOS" / EXECUTABLE_NAME

    def _spawn(
        self, argv: Sequence[str], *, payload: bytes = b"", timeout: float
    ) -> subprocess.CompletedProcess[bytes]:
        """Run the helper in ITS OWN PROCESS GROUP and reap that group by pid.

        Not ``subprocess.run(timeout=...)``: that kills the child only, and the
        fleet rule here is that a reaped process group is what a bound means —
        ``killpg`` on the pid we started, never a search by program name, because
        a bare name match has already killed other sessions' trees on this host.
        """
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "LC_ALL": "C"},
            start_new_session=True,
        )
        try:
            out, err = process.communicate(input=payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            self._reap(process)
            raise KeyagentError(
                "timeout",
                f"the key agent did not answer within {timeout:g}s",
                exit_code=None,
            ) from None
        return subprocess.CompletedProcess(argv, process.returncode, out, err)

    @staticmethod
    def _reap(process: subprocess.Popen[bytes]) -> None:
        """SIGTERM the group, then SIGKILL it, then wait it out — by pid."""
        pid = process.pid
        try:
            group = os.getpgid(pid)
        except OSError:  # pragma: no cover — it exited between the two calls
            group = pid
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(group, sig)
            except OSError:  # pragma: no cover — already gone
                break
            try:
                process.wait(timeout=_KILL_GRACE_SECONDS)
                return
            except subprocess.TimeoutExpired:
                continue
        # A group that ignores both signals: wait it out rather than leak it, and
        # let the caller's bound be the thing that reports the failure.
        try:  # pragma: no cover — SIGKILL cannot be ignored
            process.wait(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass

    # -- protocol -----------------------------------------------------------

    def _invoke(
        self, verb: str, *, payload: bytes = b"", timeout: float | None = None
    ) -> dict[str, Any]:
        """One verb, one process, one JSON object — or a :class:`KeyagentError`.

        The pre-flight runs before every call, and it is NOT cached: a bundle can
        be replaced between two calls in the same process, and the check is cheap
        next to what it protects against (an unreportable kernel kill).
        """
        executable = self.executable()
        self._preflight(self.bundle)
        argv = [str(executable), verb, "--tag", self.tag]
        bound = self.sign_timeout if timeout is None and verb == "sign" else timeout
        if bound is None:
            bound = self.quick_timeout
        try:
            result = self._runner(argv, payload=payload, timeout=bound)
        except KeyagentError:
            raise
        except FileNotFoundError as exc:
            raise KeyagentError("absent", str(exc)) from exc
        except PermissionError as exc:
            raise KeyagentError("absent", f"{executable} is not executable: {exc}") from exc
        except OSError as exc:
            raise KeyagentError("absent", f"the key agent could not be started: {exc}") from exc

        rc = result.returncode
        stdout = result.stdout.decode("utf-8", "replace").strip()
        stderr = result.stderr.decode("utf-8", "replace").strip()
        if rc < 0:
            raise KeyagentError("killed", stderr or f"signal {-rc}", exit_code=rc)
        if rc == 3:
            raise KeyagentError("no-key", stdout or stderr, exit_code=rc)
        if rc == 2:
            raise KeyagentError("cancelled", stdout or stderr, exit_code=rc)
        if rc == 5:
            raise KeyagentError("protocol", stdout or stderr, exit_code=rc)
        reply: dict[str, Any] = {}
        if stdout:
            try:
                parsed = json.loads(stdout)
            except ValueError as exc:
                raise KeyagentError(
                    "protocol", f"the key agent's reply is not JSON: {stdout[:200]}", exit_code=rc
                ) from exc
            if not isinstance(parsed, dict):
                raise KeyagentError("protocol", "the key agent's reply is not an object")
            reply = parsed
        if rc != 0:
            # The helper's own reply carries the site and the framework's detail; stderr
            # is only the fallback, because a process that died before printing JSON
            # (a signal, a missing library) is exactly the case with nothing on stdout.
            raise KeyagentError(
                "refused",
                str(reply.get("detail") or stderr),
                status=_int_or_none(reply.get("status")),
                site=str(reply.get("site") or ""),
                exit_code=rc,
                refusals=reply.get("refusals") or (),
            )
        if reply.get("protocol") != PROTOCOL or reply.get("ok") is not True:
            raise KeyagentError(
                "protocol",
                f"the key agent answered protocol {reply.get('protocol')!r}, "
                f"this runtime speaks {PROTOCOL}",
                exit_code=rc,
            )
        return reply

    # -- verbs --------------------------------------------------------------

    def create(self) -> KeyPublic:
        """Create the operator key, or return the one already there."""
        reply = self._invoke("create")
        return KeyPublic(point=_point(reply), reused=bool(reply.get("reused")))

    def public(self) -> bytes:
        """The public half of the stored key, or ``no-key`` if there is none."""
        return _point(self._invoke("public"))

    def exists(self) -> bool:
        """Whether the entitled process can see a key under the tag."""
        return bool(self._invoke("exists").get("present"))

    def sign(self, message: bytes, *, timeout: float | None = None) -> bytes:
        """Sign exactly these bytes. THIS RAISES THE PRESENCE PROMPT."""
        reply = self._invoke("sign", payload=message, timeout=timeout)
        encoded = reply.get("signature")
        if not isinstance(encoded, str):
            raise KeyagentError("protocol", "the key agent returned no signature")
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))

    def doctor(self) -> DoctorReport:
        """Measure the bundle's health without writing to any keychain."""
        reply = self._invoke("doctor")
        return DoctorReport(
            rung=str(reply.get("rung", "")),
            keychain=str(reply.get("keychain", "")),
            keychain_status=int(reply.get("keychain_status", 0)),
            profile=str(reply.get("profile", "")),
        )

    def purge(self) -> int:
        """Delete the item under THIS tag and return ``SecItemDelete``'s status.

        Used by tests and by a QA teardown, never by the product: the helper
        refuses the operator's own application tag (see ``se-keyagent.c``), because
        the product has no delete verb by design — rotation writes a new key.
        """
        return int(self._invoke("purge").get("deleted", 0))


def helper_health(*, bundle: Path | None = None, tag: str = HEALTH_TAG) -> KeyagentHealth:
    """Whether THIS INSTALLATION can reach an entitled process, and why not.

    The question ``SecureEnclaveBackend.supported()`` answers — deliberately not
    "is there a Secure Enclave on this host". On macOS the *build* can do this and
    only the install can be wrong, so the honest predicate is about the helper in
    this installation: present, verifying as ours, and able to run with its
    entitlement. A pre-flight alone would not prove the last part (a profile can
    parse and still not be authorized, which is a kernel kill), so a ``doctor`` run
    under a tag nothing writes to finishes the answer.

    Never raises: callers report the reason rather than dying on it.
    """
    app = Path(bundle) if bundle is not None else helper_bundle_path()
    try:
        described = verify_bundle(app)
        KeyagentClient(tag=tag, bundle=app).doctor()
        return KeyagentHealth(True, described)
    except KeyagentError as exc:
        return KeyagentHealth(False, f"{exc.kind}: {exc.detail}", exc.kind)
    except OSError as exc:  # pragma: no cover — a filesystem in a bad state
        return KeyagentHealth(False, f"unreadable: {exc}", "absent")


def _point(reply: dict[str, Any]) -> bytes:
    """The 65-byte uncompressed P-256 point in a reply, or a protocol error."""
    encoded = reply.get("spki")
    if not isinstance(encoded, str):
        raise KeyagentError("protocol", "the key agent returned no public key")
    point = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    if len(point) != 65 or point[0] != 0x04:
        raise KeyagentError(
            "protocol",
            f"the key agent returned a {len(point)}-byte public value, not a P-256 point",
        )
    return point


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
