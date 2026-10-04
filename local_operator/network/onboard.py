"""The onboarding runner: connect → install → anchor → join, driven by an approval.

WHAT THIS IS (design §3; slice (b)). One runner with an EXPLICIT transport
interface in two phases, so the same step machine runs against real SSH in the
field and against a scripted fake in CI:

* **credential-free handshake** — reachability, the host-key fingerprint and the
  SSH banner (:func:`probe`). No ``-i``, no auth, no secret of any kind. This is
  the WHOLE of pre-approval contact; it is what makes an approval card factual
  (the card's "where" is built from it) and it must never learn anything a
  credential could teach.
* **credentialed phase** — everything after the signed approval. The FIRST
  credentialed action is the receipted pre-read ("step zero", §3.3 step 4); a
  fact that contradicts the card HALTS the run before any state-changing step
  (``OnboardContradiction``) instead of proceeding on corrected facts nobody
  approved. Every credentialed step re-checks the approval record (state +
  expiry) through :mod:`local_operator.network.onboard_approvals` — the ONE
  adapter onto slice (a)'s store — and the record's signature is re-verified at
  that same point (§2.4's second verification point).

THE STEPS (named receipts, §3.3 steps 3-10; steps 1-2 are the request path):

    invite    mint the one invite + pre-answer the relay's parked confirm (§2.6)
    pre_read  the credentialed pre-read; halt-on-contradiction
    install   ``uv tool install [--force] --refresh local-operator==<tag>`` | ``lop-update <tag>``
    join      identity + ``lop network join @<token> --automated``
    anchor    ``lop operator anchor export`` → node → ``install --from`` (F4b)
    grants    ``lop network member grant <net> <mac> approve unattended``
    relay     install/start the relay service; linger check (OQ11)
    verify    ``doctor``/``ready``/``peers`` → the record folds to ``connected``

CREDENTIAL HANDLING (§3.2). The record stores a REFERENCE, never material. The
reference is resolved IN PLACE at the pre-read: ``lop secret get NAME`` bytes
land in a 0600 temp file which is unlinked in a ``finally``; a ``path:`` ref
uses the key file where it already lives; an ``agent:`` ref passes an
``SSH_AUTH_SOCK`` through ``lop secret run`` for the child only. Nothing here
prints, receipts or audits a secret value — receipts carry the reference's
NAME, and the audit log's ``FORBIDDEN_DETAIL_KEYS`` is the backstop.

COPY RULES (§2.9). No sentence this module emits — refusal, receipt or remedy —
names a terminal command for its reader to run. Remedies name PRODUCT actions
("file a fresh request…", "set up operator authority…"). The verbs themselves
are documented in guides, which is where they belong.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from local_operator.network import onboard_approvals as approvals_adapter
from local_operator.network.types import MeshRefusal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Step names, in order. These ARE the interface (§6 row (b): "runner step
#: names" are frozen): the record's receipts and the run payload both carry
#: them, and the drill's matrix asserts them.
STEP_NAMES: tuple[str, ...] = (
    "invite",
    "pre_read",
    "install",
    "join",
    "anchor",
    "grants",
    "relay",
    "verify",
)

#: The per-step timeouts, as one table so a caller can see every bound at once.
#: Generous where a package download or a pairing window is involved; the point
#: is that NO step may hang forever, not that any is tight.
STEP_TIMEOUTS: dict[str, float] = {
    "invite": 60.0,
    "pre_read": 60.0,
    "install": 900.0,
    "join": 300.0,
    "anchor": 180.0,
    "grants": 60.0,
    "relay": 180.0,
    "verify": 180.0,
}

#: Prepended to every remote script: the node's ``lop`` may live in
#: ``~/.local/bin`` (the uv tool shim) and a non-interactive SSH command does
#: not source a profile, so the runner ships the PATH fact rather than assuming
#: the login shell would.
_SCRIPT_PROLOGUE = 'PATH="$HOME/.local/bin:$PATH"; export PATH; '

#: The credentialed pre-read (§3.3 step 4). POSIX sh only, ``key=value`` lines
#: so the parse is trivial and a missing tool degrades to a value rather than an
#: aborted script. It WRITES NOTHING: the identical discipline
#: ``readiness.py`` states for its probes ("READ-ONLY, AND PROVEN SO").
_PRE_READ_SCRIPT = _SCRIPT_PROLOGUE + r"""
printf 'os=%s\n' "$(uname -s 2>/dev/null || echo unknown)"
printf 'arch=%s\n' "$(uname -m 2>/dev/null || echo unknown)"
printf 'user=%s\n' "$(id -un 2>/dev/null || echo unknown)"
if command -v lop >/dev/null 2>&1; then
  printf 'lop=yes\n'
  printf 'lop_path=%s\n' "$(command -v lop)"
  printf 'lop_version=%s\n' "$(lop --version 2>/dev/null | head -n 1)"
else
  printf 'lop=no\nlop_path=\nlop_version=\n'
fi
if command -v lop-update >/dev/null 2>&1; then
  printf 'lop_update=yes\n'
else
  printf 'lop_update=no\n'
fi
if command -v uv >/dev/null 2>&1; then
  printf 'uv=yes\n'
else
  printf 'uv=no\n'
fi
if command -v systemctl >/dev/null 2>&1; then
  printf 'systemctl=yes\n'
else
  printf 'systemctl=no\n'
fi
user="$(id -un)"
printf 'linger=%s\n' "$(loginctl show-user "$user" -p Linger --value 2>/dev/null || echo unknown)"
if command -v sudo >/dev/null 2>&1; then
  printf 'sudo=yes\n'
else
  printf 'sudo=no\n'
fi
if sudo -n true >/dev/null 2>&1; then printf 'sudo_nopass=yes\n'; else printf 'sudo_nopass=no\n'; fi
anchor=""
for cand in "/etc/local-operator/operators/$(id -u).json" \
            "/Library/Application Support/local-operator/operators/$(id -u).json"; do
  if [ -e "$cand" ]; then anchor="$cand"; fi
done
printf 'anchor=%s\n' "$anchor"
printf 'pre_read=done\n'
"""


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class OnboardContradiction(MeshRefusal):
    """The pre-read contradicted the card: halt before ANY state-changing step.

    Carries the ``finding`` (what was compared and how it disagreed) so the
    caller can file the fresh request §3.3 step 4 asks for, and ``refiled``
    (a new request id, when the store could file one).
    """

    def __init__(self, sentence: str, *, finding: dict[str, Any], refiled: str | None = None):
        super().__init__("pre_read_contradiction", sentence)
        self.finding = finding
        self.refiled = refiled


# ---------------------------------------------------------------------------
# Probe (credential-free handshake, §3.1 phase (a))
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """What the credential-free handshake learned — and nothing more."""

    ok: bool
    host: str
    port: int
    user: str = ""
    banner: str = ""
    host_key_fp: str = ""
    host_keys: tuple[str, ...] = ()
    at: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    rc: int
    stdout: str = ""
    stderr: str = ""
    at: float = 0.0
    timed_out: bool = False


def ssh_fingerprint(blob_b64: str) -> str:
    """``SHA256:…`` over a base64 key blob — the spelling ``ssh-keygen -lf`` prints.

    Written here rather than shelling to ``ssh-keygen``: the value is one hash
    over bytes, and a second process for it would be one more thing to keep in
    step with the tool whose output operators compare against.
    """
    try:
        blob = base64.b64decode(blob_b64, validate=True)
    except (ValueError, binascii.Error):
        return ""
    digest = hashlib.sha256(blob).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def parse_keyscan(output: str) -> list[tuple[str, str, str]]:
    """``(hostspec, keytype, fingerprint)`` per scanned key, comments skipped.

    Keeps every key type: an SSH client may negotiate any of them, so the
    known_hosts seed needs them all even though the card shows one fingerprint.
    """
    rows: list[tuple[str, str, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append((parts[0], parts[1], ssh_fingerprint(parts[2])))
    return rows


def pick_fingerprint(rows: Sequence[tuple[str, str, str]]) -> str:
    """The ONE fingerprint the card carries: ed25519 if the host offers it.

    Preference, not exclusivity — the seed keeps all types, and the compare in
    :meth:`SshTransport.connect` is exact-string so a host that rotates its key
    TYPE still fails loudly rather than being silently re-pinned.
    """
    for _host, keytype, fingerprint in rows:
        if keytype == "ssh-ed25519":
            return fingerprint
    return rows[0][2] if rows else ""


def _read_banner(host: str, port: int, *, timeout: float) -> str:
    """One TCP connect, one bounded read: the SSH identification line.

    Credential-free by construction — the banner precedes auth, which is the
    whole reason this shape is allowed before the approval. A host that accepts
    the connection and says nothing is a detail, not an error: the fingerprint
    scan below is what the card needs most.
    """
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        buf = b""
        while b"\n" not in buf and len(buf) < 512:
            chunk = sock.recv(256)
            if not chunk:
                break
            buf += chunk
    return buf.split(b"\r\n", 1)[0].decode("latin-1", errors="replace").strip()


def probe(
    host: str,
    *,
    port: int = 22,
    user: str = "",
    timeout: float = 6.0,
    keyscan_timeout: float = 8.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ssh_keyscan: str = "ssh-keyscan",
) -> Probe:
    """The credential-free handshake: reachability, host key, banner (§3.1).

    Never touches a credential: no ``-i``, no agent, no auth attempt. The
    ``ssh-keyscan`` invocation is deliberately the bare tool — it is an
    unauthenticated read of the host's public keys, the same fact the banner
    is, and it is what the approval card's fingerprint is made from.
    """
    run = runner or subprocess.run
    moment = _now()
    try:
        banner = _read_banner(host, port, timeout=timeout)
    except OSError as exc:
        return Probe(
            ok=False,
            host=host,
            port=port,
            user=user,
            at=moment,
            detail=f"nothing accepted a connection at {host}:{port} ({exc.__class__.__name__})",
        )
    try:
        scanned = run(
            [ssh_keyscan, "-T", str(int(keyscan_timeout)), "-p", str(port), host],
            capture_output=True,
            text=True,
            timeout=keyscan_timeout + 2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Probe(
            ok=False,
            host=host,
            port=port,
            user=user,
            banner=banner,
            at=moment,
            detail=f"the host key could not be read from {host}:{port} ({exc.__class__.__name__})",
        )
    lines = [
        line.strip()
        for line in (scanned.stdout or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    rows = parse_keyscan(scanned.stdout or "")
    if not rows:
        return Probe(
            ok=False,
            host=host,
            port=port,
            user=user,
            banner=banner,
            at=moment,
            detail=f"{host}:{port} did not offer a host key (ssh-keyscan returned nothing)",
        )
    return Probe(
        ok=True,
        host=host,
        port=port,
        user=user,
        banner=banner,
        host_key_fp=pick_fingerprint(rows),
        host_keys=tuple(lines),
        at=moment,
    )


# ---------------------------------------------------------------------------
# Credentials (§3.2: references resolved in place, never logged)
# ---------------------------------------------------------------------------


@dataclass
class ResolvedCredential:
    """A credential as the transport needs it — argv fragment plus cleanup."""

    kind: str  # "file" | "agent"
    label: str  # the reference's NAME; safe to put in a receipt
    argv: list[str] = field(default_factory=list)  # ssh argv fragment (-i …)
    prefix: list[str] = field(default_factory=list)  # wrapper argv (secret run)
    env: dict[str, str] = field(default_factory=dict)
    cleanup_dir: Path | None = None  # temp dir to remove in the runner's finally


def _local_cli_argv() -> list[str]:
    """How this process runs its OWN CLI as a child: the `lop` binary when on
    PATH, else this interpreter with the CLI's entry point.

    ``python -m local_operator`` is NOT a fallback (there is no ``__main__``),
    and the console script may be called ``lop`` on one host and something else
    on another — so the interpreter form is spelled explicitly.
    """
    found = shutil.which("lop")
    if found:
        return [found]
    return [
        sys.executable,
        "-c",
        "import sys; from local_operator.cli import main; sys.exit(main())",
    ]


def resolve_credential(
    reference: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
    cli: Sequence[str] | None = None,
) -> ResolvedCredential:
    """Resolve the record's ``credential_ref`` into something ssh can use.

    Three shapes, one per §3.2:

    * ``{"kind": "ssh", "ref": "path:~/.ssh/lop-mesh-nprod.pem"}`` — a key file
      that stays where it lives;
    * ``{"kind": "ssh", "ref": "agent:<secret name>"}`` — an ssh-agent socket
      passed through ``lop secret run``, for the child's environment only;
    * ``{"kind": "ssh", "ref": "<secret name>"}`` — ``lop secret get`` bytes
      written to a 0600 temp file (the plaintext IS on disk for the run's
      lifetime; that cost is the store's documented one, and the unlink is the
      runner's ``finally``).

    A value never appears in :attr:`ResolvedCredential.label`, in an exception,
    or in any receipt: the label is the reference itself.
    """
    run = runner or subprocess.run
    cli_argv = list(cli) if cli is not None else _local_cli_argv()
    ref = str((reference or {}).get("ref") or "")
    if ref.startswith("path:"):
        path = Path(os.path.expanduser(ref[len("path:") :]))
        if not path.exists():
            raise MeshRefusal(
                "credential_missing",
                "the key file this request names isn't on this machine; supply the key "
                "again, or choose a key file that exists",
            )
        return ResolvedCredential(
            kind="file",
            label=ref,
            argv=["-i", str(path), "-o", "IdentitiesOnly=yes"],
        )
    if ref.startswith("agent:"):
        name = ref[len("agent:") :].strip()
        if not name:
            raise MeshRefusal(
                "credential_missing",
                "the request names an agent passthrough with no secret behind it",
            )
        # The wrapper runs the ssh child with SSH_AUTH_SOCK set from the value
        # of the named secret; the socket path never reaches this process.
        return ResolvedCredential(
            kind="agent",
            label=ref,
            prefix=[*cli_argv, "secret", "run", "--secret", f"{name}=SSH_AUTH_SOCK", "--"],
        )
    if ref:
        directory = Path(tempfile.mkdtemp(prefix="lop-onboard-key-"))
        os.chmod(directory, 0o700)
        target = directory / "key"
        try:
            result = run(
                [*cli_argv, "secret", "get", ref],
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise MeshRefusal(
                "credential_unresolved",
                f"the secret store could not be asked for this request's key "
                f"({exc.__class__.__name__}); nothing was read",
            ) from None
        if result.returncode != 0 or not result.stdout:
            shutil.rmtree(directory, ignore_errors=True)
            # The NAME is safe; the store's own stderr is deliberately NOT
            # echoed here (it can carry a decryption diagnostic, never the
            # value, but the discipline is that refusals quote nothing).
            raise MeshRefusal(
                "credential_unresolved",
                "no usable key is stored under the name this request carries; store the "
                "key again, or use a key file instead",
            )
        payload = result.stdout
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
        return ResolvedCredential(
            kind="file",
            label=ref,
            argv=["-i", str(target), "-o", "IdentitiesOnly=yes"],
            cleanup_dir=directory,
        )
    raise MeshRefusal(
        "credential_missing",
        "this request carries no credential reference, so the runner has nothing to "
        "connect with",
    )


def release_credential(credential: ResolvedCredential | None) -> None:
    """The ``finally`` half of §3.2: remove the temp key material, best effort."""
    if credential is None or credential.cleanup_dir is None:
        return
    shutil.rmtree(credential.cleanup_dir, ignore_errors=True)
    credential.cleanup_dir = None


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class Transport(Protocol):
    """The two-phase transport interface (§3.1), five operations.

    The CONTRACT the runner relies on:

    * :meth:`probe` is credential-free and is legal before approval;
    * :meth:`connect` begins the credentialed phase and must PIN the host key:
      it refuses ``host_key_changed`` when the live key no longer matches
      ``expected_fingerprint``, which is the fingerprint the handshake observed
      and the card was approved against;
    * :meth:`run` and :meth:`copy` are credentialed actions;
    * :meth:`close` is the teardown the runner always calls.
    """

    def probe(self) -> Probe: ...

    def connect(
        self, credential: ResolvedCredential | None, *, expected_fingerprint: str
    ) -> Probe: ...

    def run(
        self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None
    ) -> CommandResult: ...

    def copy(self, local_path: Path, remote_path: str) -> CommandResult: ...

    def close(self) -> None: ...


class SshTransport:
    """``ssh(1)`` with ``BatchMode=yes``, pinned ``accept-new``, per-step timeouts.

    The pinning mechanism, stated once: ``connect`` re-reads the host key
    (credential-free), refuses on any difference from the fingerprint the card
    carries, and writes the scanned key into a private ``known_hosts`` seed.
    Every ``ssh`` call then carries ``UserKnownHostsFile=<seed>`` +
    ``StrictHostKeyChecking=accept-new``: with the seed present, "accept-new"
    can only ever accept the key that was pinned, and a swapped key fails
    exactly as ``StrictHostKeyChecking=yes`` would — while a host absent from
    the seed (impossible by construction, kept for explicitness) still cannot
    be TOFU'd into an unrelated key because the runner's compare runs first.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 22,
        user: str = "",
        ssh_binary: str = "ssh",
        ssh_keyscan: str = "ssh-keyscan",
        connect_timeout: float = 10.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.host = host
        self.port = int(port or 22)
        self.user = user
        self._ssh = ssh_binary
        self._keyscan = ssh_keyscan
        self._connect_timeout = connect_timeout
        self._runner = runner or subprocess.run
        self._credential: ResolvedCredential | None = None
        self._seed_dir: Path | None = None
        self._known_hosts: Path | None = None
        self.last_probe: Probe | None = None

    # -- phase (a) ---------------------------------------------------------

    def probe(self) -> Probe:
        found = probe(
            self.host,
            port=self.port,
            user=self.user,
            runner=self._runner,
            ssh_keyscan=self._keyscan,
        )
        self.last_probe = found
        return found

    # -- phase (b) ---------------------------------------------------------

    def connect(self, credential: ResolvedCredential | None, *, expected_fingerprint: str) -> Probe:
        found = self.probe()
        if not found.ok:
            raise MeshRefusal("unreachable", found.detail or f"{self.host} is not reachable")
        if not expected_fingerprint:
            raise MeshRefusal(
                "host_key_unpinned",
                "the request doesn't say which host key to expect, so this connection "
                "can't be checked against what was approved",
            )
        if found.host_key_fp != expected_fingerprint:
            raise MeshRefusal(
                "host_key_changed",
                f"the host key at {self.host}:{self.port} is {found.host_key_fp or 'unreadable'}, "
                "not the one this request was approved against; nothing was contacted with "
                "a credential. That machine may have been rebuilt, or something else may "
                "be answering at that address — it is your call whether to approve a "
                "fresh request for it",
            )
        seed = Path(tempfile.mkdtemp(prefix="lop-onboard-hosts-"))
        os.chmod(seed, 0o700)
        known_hosts = seed / "known_hosts"
        known_hosts.write_text("\n".join(found.host_keys) + "\n", encoding="utf-8")
        os.chmod(known_hosts, 0o600)
        self._seed_dir = seed
        self._known_hosts = known_hosts
        self._credential = credential
        self.last_probe = found
        return found

    def _ssh_argv(self, argv: Sequence[str]) -> list[str]:
        if self._credential is None or self._known_hosts is None:
            raise MeshRefusal(
                "transport_not_connected",
                "the credentialed phase has not been opened for this connection",
            )
        # THE REMOTE COMMAND IS ONE QUOTED STRING. ssh joins its argv with
        # spaces and hands the result to the remote shell, so every element is
        # shell-quoted here — the same round trip as a local `sh -c`, applied at
        # the boundary where ssh would otherwise reinterpret the pieces.
        command = " ".join(shlex.quote(part) for part in argv)
        return [
            *self._credential.prefix,
            self._ssh,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self._known_hosts}",
            "-o",
            f"ConnectTimeout={int(max(1.0, self._connect_timeout))}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=8",
            *self._credential.argv,
            "-p",
            str(self.port),
            f"{self.user}@{self.host}" if self.user else self.host,
            "--",
            command,
        ]

    def run(
        self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None
    ) -> CommandResult:
        full = self._ssh_argv(argv)
        env = os.environ.copy()
        env.update(self._credential.env if self._credential else {})
        moment = _now()
        # ``input=`` and ``stdin=`` are mutually exclusive in ``subprocess.run``;
        # without either, a child inherits THIS process's stdin — which for ssh
        # must never happen (a credential prompt or a stray read would consume
        # the runner's stream). DEVNULL is the no-input shape.
        kwargs: dict[str, Any] = {"capture_output": True, "timeout": timeout, "env": env}
        if stdin is not None:
            kwargs["input"] = stdin
        else:
            kwargs["stdin"] = subprocess.DEVNULL
        try:
            result = self._runner(full, **kwargs)
        except subprocess.TimeoutExpired:
            return CommandResult(
                argv=tuple(argv),
                rc=-1,
                stderr=f"the command did not finish within {int(timeout)}s",
                at=moment,
                timed_out=True,
            )
        except OSError as exc:
            return CommandResult(
                argv=tuple(argv),
                rc=-1,
                stderr=f"the command could not be started ({exc.__class__.__name__})",
                at=moment,
            )
        stdout = result.stdout
        stderr = result.stderr
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return CommandResult(
            argv=tuple(argv),
            rc=int(result.returncode),
            stdout=stdout or "",
            stderr=stderr or "",
            at=moment,
        )

    def copy(self, local_path: Path, remote_path: str) -> CommandResult:
        """Push a file over the SAME ssh channel the steps use.

        ``cat >`` under ``umask 077`` rather than ``scp``: the trust options,
        the credential and the timeout are then one code path, and the file
        lands 0600 because the umask says so — which matters for the one file
        this carries that is itself a bearer credential (the invite token).
        """
        payload = Path(local_path).read_bytes()
        return self.run(
            ["sh", "-c", f"umask 077; cat > {shlex.quote(remote_path)}"],
            timeout=60.0,
            stdin=payload,
        )

    def close(self) -> None:
        if self._seed_dir is not None:
            shutil.rmtree(self._seed_dir, ignore_errors=True)
            self._seed_dir = None
            self._known_hosts = None
        self._credential = None


# ---------------------------------------------------------------------------
# Local helpers (the runner's own machine)
# ---------------------------------------------------------------------------


def _run_local(
    argv: Sequence[str], *, timeout: float = 60.0, stdin: bytes | None = None
) -> CommandResult:
    moment = _now()
    kwargs: dict[str, Any] = {"capture_output": True, "timeout": timeout}
    if stdin is not None:
        kwargs["input"] = stdin
    else:
        kwargs["stdin"] = subprocess.DEVNULL
    try:
        result = subprocess.run(list(argv), **kwargs)
    except subprocess.TimeoutExpired:
        return CommandResult(
            tuple(argv), -1, stderr=f"did not finish in {int(timeout)}s", at=moment
        )
    except OSError as exc:
        return CommandResult(
            tuple(argv), -1, stderr=f"could not start ({exc.__class__.__name__})", at=moment
        )
    return CommandResult(
        tuple(argv),
        int(result.returncode),
        (
            (result.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(result.stdout, bytes)
            else (result.stdout or "")
        ),
        (
            (result.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(result.stderr, bytes)
            else (result.stderr or "")
        ),
        at=moment,
    )


def _json_from(text: str) -> dict[str, Any] | None:
    """The first JSON object in ``text`` — CLI payloads arrive amid prose.

    A parse failure is ``None``, never an exception: every caller has a receipt
    to write and a sentence to say, and neither needs the payload.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        loaded = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _invite_network_label(
    token_path: Any, *, network_id: str = "", network_name: str = ""
) -> tuple[str, str] | None:
    """``(network_id, name)`` of the network an invite token joins, or ``None``.

    THE MINT'S OWN PAYLOAD is the first source (``step_invite`` records it the
    moment the token is minted), because the copy must still be network-aware
    when the token file has served its purpose and is gone. The token FILE is
    the fallback — the authority for what is being joined, so the copy can
    compare the network the join was FOR against what a device's relay already
    serves (drill decision, 2026-10-04: "join ADOPTS a relay already serving the
    SAME network; a DIFFERENT one refuses with cause"). Best-effort by
    contract: neither source available — a hand-made fixture, a file whose bytes
    are already consumed — answers ``None``, and the caller falls back to the
    generic sentence, because "cannot tell" must never be dressed up as
    "serves nothing".
    """
    if network_id or network_name:
        return str(network_id), str(network_name)
    try:
        from local_operator.network import invite as invite_mod

        token = Path(str(token_path)).read_text(encoding="utf-8").strip()
        envelope = invite_mod.decode(token)
        return str(envelope.network_id or ""), str(envelope.network_name or "")
    except Exception:  # noqa: BLE001 — a copy probe must never fail a step
        return None


def _relay_kind_note(status: dict[str, Any]) -> str:
    """What a restart would and would not do to the relay a status reported.

    Round-1 D1: the old sentence stated the hand-started replacement mechanism
    for ANY relay, but a supervised one restarts back onto the same served set —
    the relay serves ``store.list_networks()``, and a failed handshake never
    writes the join target's membership — so nothing about a restart clears the
    join failure in EITHER case. This note says what is true per kind and never
    promises the restart fixes the join.

    Round-2 D6: the kind reads ``status['relay_served_by']`` — the probe's
    VERIFIED reading of the serving process (:func:`relay._serving_relay_kind`:
    the supervisor's own pid, or the foreground ``serve`` shape that no unit of
    this product runs) — and NEVER unit-file presence: a unit file can exist
    while a hand-started relay serves the port (the drill node's exact state —
    the arm's failed install left the file, ``stop`` leaves it), and calling
    that process "the service's own" was the round-1 copy's one unverified
    claim. The second clause — a restart cannot change what the relay serves —
    is true in every case and is always printed; an unproven kind gets only it.
    """
    kind = status.get("relay_served_by")
    if kind == "service":
        return (
            "That relay is the service's own, and restarting it would not change what " "it serves."
        )
    if kind == "manual":
        return (
            "That relay was started by hand, not by the service — `lop network restart` "
            "replaces it with the service's own relay — and neither changes what the "
            "relay serves."
        )
    return "Restarting the relay would not change what it serves."


def _served_networks(status: dict[str, Any]) -> list[dict[str, str]]:
    """The networks a node's status says its relay SERVES, ``[]`` when unknown.

    ONLY A LIVE REPLY COUNTS. ``status["networks"]`` is the answering relay's
    own list when it answered and the LOCAL RECORDS when it did not — and a
    record is not a serving relay, so a relay that is down must not be reported
    as serving what its store remembers. Rows without both a ``network_id`` and
    a ``name`` are kept with empty strings (the sentence handles them) rather
    than dropped silently.
    """
    if not status.get("relay_answering"):
        return []
    rows = status.get("networks")
    if not isinstance(rows, list):
        return []
    served: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        served.append(
            {
                "network_id": str(row.get("network_id") or ""),
                "name": str(row.get("name") or ""),
            }
        )
    return served


def _facts_from(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        facts[key.strip()] = value.strip()
    return facts


def _version_tuple(text: str) -> tuple[int, ...]:
    """``v0.63.2`` → ``(0, 63, 2)``; unparseable sorts as an EMPTY tuple."""
    import re

    match = re.search(r"(\d+(?:\.\d+)+)", text or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _version_display(tag: str) -> str:
    return tag.lstrip("v") if tag else ""


def _compiled_tag() -> str:
    """This build's version — the fallback pin when the record names none.

    The record SHOULD name the tag (§3.3 step 5: "the tag is pinned in the
    record"); this fallback keeps a request that predates that field usable,
    and the install receipt records which tag was actually used so the
    ambiguity is visible rather than silent.
    """
    try:
        from local_operator.update import installed_version

        value = installed_version()
        if value:
            return str(value)
    except Exception:  # noqa: BLE001 — a version probe must not fail the run
        pass
    try:
        from importlib.metadata import version

        return version("local-operator")
    except Exception:  # noqa: BLE001
        return ""


#: The resolver-class install failure, however uv wraps it: "there is no version
#: of local-operator==X … unsatisfiable", "No compatible version found". This is
#: what uv emits whether the index response is stale OR the version is genuinely
#: absent — knowledge of neither — see ``_install_failure_detail``.
_INSTALL_RESOLVER_MARKERS: tuple[str, ...] = (
    "unsatisfiable",
    "no version of",
    "no compatible version",
)


def _install_failure_detail(tag: str, result: CommandResult) -> str:
    """The install step's failure sentence, in the family's cause+remedy shape.

    WHY (drill finding F3, 2026-10-04): two drill runs ended at ``install`` with
    uv's raw text — "there is no version of local-operator==0.67.4 and you
    require local-operator==0.67.4, we can conclude that your requirements are
    unsatisfiable" — six minutes and again ~1 h after the release was published,
    because the node's uv served a CACHED simple-index response; the same node's
    curl showed the version present, and ``--refresh`` cured it by hand. Raw,
    the text told the reader something false with no remedy.

    Shape (design round 1, D1-D5 + N1-N3): the ACTION leads — a clipping surface
    keeps cause and remedy, not the machine fragment; the cache is named as the
    likely cause and the escape names genuine absence, because the same uv text
    covers a version that is not on the index and a retry must not be a loop;
    one name for the artefact ("the approved build <tag>"); the window matches
    the drill's own clock ("shortly before the run"); uv's words ride last,
    flattened and glyph-stripped (terminal box-drawing renders as tofu in a UI
    sheet), head-kept when long (the head names the resolver) and never with a
    doubled stop (§2.9: remedies name product actions, never terminal commands).
    Any other failure keeps the previous surface.
    """
    output = (result.stderr or result.stdout or "").strip()
    flat = " ".join(output.split())
    if any(marker in flat.lower() for marker in _INSTALL_RESOLVER_MARKERS):
        excerpt = flat.replace("×", "").replace("╰─▶", "")
        excerpt = " ".join(excerpt.split()).rstrip(". ")
        if len(excerpt) > 300:
            excerpt = excerpt[:300].rstrip() + "…"
        return (
            "the approved build could not be installed. Retry the install: it "
            "re-resolves against a refreshed index, so a release published "
            "shortly before the run can no longer stay hidden by a cached "
            "answer — a cached index is the likely cause. If the approved build "
            "is still missing after the retry, it is not on the index — ask "
            "Local Operator to file a fresh request with the corrected tag. "
            f"(The machine's uv could not see the approved build {tag} — "
            f"{excerpt}.)"
        )
    tail = output.splitlines()
    return "the approved build could not be installed: " + (tail[-1][:200] if tail else "no output")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class _StepOutcome:
    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


class OnboardRun:
    """One execution attempt at one approval record (one ``run_id``).

    Holds the step results and the machine the steps talk through. Deliberately
    NOT a context manager: the orchestration lives in :func:`execute_approval`
    so the failure paths (receipt-then-terminal-state) exist exactly once.
    """

    def __init__(
        self,
        approval_id: str,
        *,
        view: Any,
        transport: Transport,
        resolve: Callable[[dict[str, Any]], ResolvedCredential],
        local_cli: Sequence[str] | None = None,
        run_local: Callable[..., CommandResult] = _run_local,
        clock: Callable[[], float] = _now,
        refile: bool = True,
        root: Any = None,
    ) -> None:
        self.approval_id = approval_id
        self.view = view
        self.transport = transport
        self.resolve = resolve
        self.local_cli = list(local_cli) if local_cli is not None else _local_cli_argv()
        self.run_local = run_local
        self.clock = clock
        self.refile = refile
        self.root = root
        self.facts: dict[str, str] = {}
        self.invite_id = ""
        self.invite_path: Path | None = None
        #: The network the invite joins, from the mint's own payload — carried
        #: so the join-failure copy can compare it against what a serving relay
        #: reports, even when the token file is already gone (drill finding,
        #: 2026-10-04: "join ADOPTS a relay already serving the SAME network").
        self.invite_network_id = ""
        self.invite_network_name = ""
        self.credential: ResolvedCredential | None = None
        self._connected = False

    # -- shared plumbing ---------------------------------------------------

    def _gate(self) -> None:
        """Re-check state, expiry and signature before one credentialed step."""
        self.view = approvals_adapter.require_step_allowed(
            self.approval_id, run_id=self.view.run_id, root=self.root
        )

    def _step_timeout(self, step: str) -> float:
        return float(STEP_TIMEOUTS.get(step, 120.0))

    def _remote_lop(self, command: str, *, timeout: float) -> CommandResult:
        return self.transport.run(["sh", "-c", _SCRIPT_PROLOGUE + command], timeout=timeout)

    def _node_json(self, command: str, *, timeout: float) -> dict[str, Any] | None:
        result = self._remote_lop(f"{command} --json", timeout=timeout)
        if result.rc != 0:
            return None
        return _json_from(result.stdout)

    def _pinned_tag(self) -> str:
        what = self.view.what
        build = what.get("build") or what.get("tag")
        if not build and isinstance(what.get("install"), str):
            build = what.get("install")
        return str(build or "").strip() or _compiled_tag()

    # -- steps -------------------------------------------------------------

    def step_invite(self) -> _StepOutcome:
        """§3.3 step 3: mint the ONE invite and pre-answer the parked confirm.

        The invite is minted through this device's own `lop network invite` so
        the token's file discipline (0600, path-not-value) is the CLI's, and it
        is bound to the node's device id when the card knew one. The pre-answer
        is a one-shot ``PairDecision`` keyed to the invite: the relay's parked
        question consumes it exactly once — and it can never bypass the code
        compare, which runs BEFORE any decision is read
        (``relay.py``'s ``sas_matches``) — so an automated join still refuses a
        mismatch, burns the invite's attempt budget and audits ``sas_mismatch``.
        """
        role = str(self.view.what.get("role") or "drive")
        device_id = str(self.view.device.get("device_id") or "")
        argv = [*self.local_cli, "network", "invite", "--role", role, "--json"]
        if device_id:
            argv += ["--device", device_id]
        result = self.run_local(argv, timeout=self._step_timeout("invite"))
        payload = _json_from(result.stdout or "")
        if result.rc != 0 or not payload or not payload.get("invite_id"):
            detail = (result.stderr or result.stdout or "the invite was not minted").strip()
            return _StepOutcome(False, f"the invite was not minted: {detail[:200]}")
        self.invite_id = str(payload["invite_id"])
        self.invite_network_id = str(payload.get("network_id") or "")
        self.invite_network_name = str(payload.get("network_name") or "")
        path = str(payload.get("path") or "")
        if not path or not Path(path).exists():
            return _StepOutcome(
                False, "the invite was minted but its token file was not written; nothing ran"
            )
        self.invite_path = Path(path)
        from local_operator.network import store as network_store
        from local_operator.network.types import PairDecision

        network_store.save_pair_decision(
            PairDecision(
                invite_id=self.invite_id,
                decision="admit",
                matched=True,
                reason="",
                answered_by=f"approval:{self.approval_id}",
                shares=[],
            )
        )
        return _StepOutcome(
            True,
            f"invite {self.invite_id} minted for role {role}"
            + (f" bound to {device_id}" if device_id else "")
            + "; the approval pre-answered the pairing confirm",
            {
                "invite_id": self.invite_id,
                "role": role,
                "device_bound": device_id,
                "expires_in_s": payload.get("expires_in_s"),
                "pre_answered": True,
            },
        )

    def step_pre_read(self) -> _StepOutcome:
        """§3.3 step 4 — the FIRST credentialed action, and the halt it can cause.

        On any contradiction: a failing receipt is written, the record is marked
        ``failed``, a fresh request is filed when the store can file one, and
        ``OnboardContradiction`` carries the finding. Nothing state-changing has
        run at this point, so the fresh card is a re-approval of corrected
        facts, never a rollback.
        """
        expected_fp = str(self.view.device.get("host_key_fp") or "")
        if not expected_fp:
            # OBSERVE BEFORE HALTING (drill finding, 2026-10-03): a request filed
            # without the host key halts here, and the auto-refiled replacement
            # used to be minted with the same hole — the remedy reproduced the
            # failure. The host key is a credential-free read (the same handshake
            # step zero runs), so the halt can carry the value the fresh request
            # needs instead of filing another card that cannot run.
            found = self.transport.probe()
            raise self._contradiction(
                {
                    "check": "host_key_fp",
                    "approved": "",
                    "observed": found.host_key_fp if found.ok else "",
                    "why": "the request does not say which host key to expect",
                },
                "the request does not carry the host-key fingerprint it was approved "
                "against, so the connection cannot be pinned",
            )
        try:
            self.credential = self.resolve(self.view.credential_ref)
        except MeshRefusal as refusal:
            # §4: "failure to resolve ⇒ step fails, record keeps credential_ref
            # only" — a failing step, not a halt, so the operator can supply the
            # missing reference and retry the same record.
            return _StepOutcome(
                False,
                f"the connection credential could not be resolved: {refusal.sentence}",
                {"credential": str((self.view.credential_ref or {}).get("ref") or "")},
            )
        try:
            found = self.transport.connect(self.credential, expected_fingerprint=expected_fp)
        except MeshRefusal as refusal:
            if refusal.code == "host_key_changed":
                # A CHANGED KEY IS A HALT, not a retry: the facts moved after the
                # operator approved them, exactly the class step zero exists for.
                # The probe the refusal came from IS the observed value (the real
                # transport records it before the fingerprint check), so the fresh
                # card can name the key that actually answered.
                probe = getattr(self.transport, "last_probe", None)
                observed = str(getattr(probe, "host_key_fp", "") or "")
                raise self._contradiction(
                    {
                        "check": "host_key_fp",
                        "approved": expected_fp,
                        "observed": observed,
                        "why": "the host key changed between the handshake and the run",
                    },
                    refusal.sentence,
                ) from None
            return _StepOutcome(
                False,
                f"the credentialed connection could not be established: {refusal.sentence}",
            )
        if found.host_key_fp != expected_fp:
            raise self._contradiction(
                {
                    "check": "host_key_fp",
                    "approved": expected_fp,
                    "observed": found.host_key_fp,
                    "why": "the host key changed between the handshake and the run",
                },
                f"the host key at {found.host}:{found.port} is no longer the one this "
                "request was approved against",
            )
        result = self.transport.run(
            ["sh", "-c", _PRE_READ_SCRIPT], timeout=self._step_timeout("pre_read")
        )
        if result.rc != 0 or "pre_read=done" not in result.stdout:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            return _StepOutcome(
                False,
                "the setup checks on the other machine did not complete: "
                + (tail[-1][:200] if tail else "no output"),
            )
        self.facts = _facts_from(result.stdout)
        findings = self._contradictions()
        if findings:
            finding = findings[0]
            raise self._contradiction(
                finding,
                "the other machine does not match what was approved "
                f"({finding['why']}); nothing state-changing ran",
            )
        return _StepOutcome(
            True,
            "pre-read: "
            f"{self.facts.get('os', '?')}/{self.facts.get('arch', '?')}, "
            f"lop={self.facts.get('lop_version') or 'absent'}, "
            + "sudo="
            + (
                "nopass"
                if self.facts.get("sudo_nopass") == "yes"
                else str(self.facts.get("sudo") or "?")
            ),
            {
                "os": self.facts.get("os", ""),
                "arch": self.facts.get("arch", ""),
                "lop": self.facts.get("lop", ""),
                "lop_version": self.facts.get("lop_version", ""),
                "lop_update": self.facts.get("lop_update", ""),
                "uv": self.facts.get("uv", ""),
                "systemctl": self.facts.get("systemctl", ""),
                "linger": self.facts.get("linger", ""),
                "sudo": self.facts.get("sudo", ""),
                "sudo_nopass": self.facts.get("sudo_nopass", ""),
                "anchor": self.facts.get("anchor", ""),
            },
        )

    def _contradictions(self) -> list[dict[str, str]]:
        """The pre-read vs the card (§3.3 step 4). Empty list = proceed.

        Sources for the expectation, in order: the record's own fields (a
        ``what.os``/``what.arch`` the request surface filled), the LAST run's
        pre-read receipt (a retry must land on the same machine it was approved
        for), and nothing (record-only). Version checks are one-directional:
        a node NEWER than the approved build is a contradiction — the run would
        be a downgrade nobody approved — while older/absent is the ordinary
        upgrade path.
        """
        findings: list[dict[str, str]] = []
        what = self.view.what
        facts = self.facts
        want_os = str(what.get("os") or "")
        want_arch = str(what.get("arch") or "")
        if not want_os and not want_arch:
            prior = None
            for receipt in reversed(self.view.receipts):
                if str(receipt.get("step")) == "pre_read" and receipt.get("ok"):
                    prior = receipt
                    break
            if prior is not None:
                # The prior run's structured facts ride the digest's source; a
                # receipt that predates this field degrades to no check rather
                # than a fabricated one.
                want_os = str((prior.get("data") or {}).get("os") or "")
                want_arch = str((prior.get("data") or {}).get("arch") or "")
        observed_os = facts.get("os", "")
        observed_arch = facts.get("arch", "")
        if want_os and observed_os and want_os != observed_os:
            findings.append(
                {
                    "check": "os",
                    "approved": want_os,
                    "observed": observed_os,
                    "why": "the machine's OS is not the one this run is for",
                }
            )
        if want_arch and observed_arch and want_arch != observed_arch:
            findings.append(
                {
                    "check": "arch",
                    "approved": want_arch,
                    "observed": observed_arch,
                    "why": "the machine's architecture is not the one this run is for",
                }
            )
        tag = self._pinned_tag()
        observed_version = _version_tuple(facts.get("lop_version", ""))
        approved_version = _version_tuple(tag)
        if observed_version and approved_version and observed_version > approved_version:
            findings.append(
                {
                    "check": "build",
                    "approved": tag,
                    "observed": facts.get("lop_version", ""),
                    "why": "the machine already runs a build newer than the approved one",
                }
            )
        if facts.get("lop") != "yes" and not bool(what.get("install")):
            findings.append(
                {
                    "check": "install_scope",
                    "approved": "no install was approved",
                    "observed": "no build installed",
                    "why": "there is nothing to connect with and installing is not in scope",
                }
            )
        if what.get("anchor") and facts.get("sudo") != "yes":
            findings.append(
                {
                    "check": "sudo",
                    "approved": "the anchor install was approved",
                    "observed": "no sudo on the machine",
                    "why": "the admin approval cannot be raised on this machine",
                }
            )
        return findings

    def _contradiction(self, finding: dict[str, str], sentence: str) -> OnboardContradiction:
        refiled: str | None = None
        if self.refile:
            try:
                refiled = approvals_adapter.refile_after_contradiction(
                    self.approval_id, finding, root=self.root
                )
            except MeshRefusal:
                refiled = None
        why = str(finding.get("why") or "").strip()
        suffix = (
            f" A new request {refiled} now carries the corrected facts — ask Local "
            "Operator to take it to approval."
            if refiled
            else " Nothing can run until this changes"
            + (f": {why}" if why else "")
            + ". A new request is needed then — ask Local Operator to file one."
        )
        return OnboardContradiction(sentence + "." + suffix, finding=finding, refiled=refiled)

    def step_install(self) -> _StepOutcome:
        """§3.3 step 5: install the pinned build, or upgrade it (§3.4)."""
        tag = self._pinned_tag()
        if not tag:
            return _StepOutcome(
                False,
                "the approved build tag could not be determined; nothing was installed",
            )
        facts = self.facts
        node_version = _version_display(facts.get("lop_version", ""))
        if facts.get("lop") == "yes" and node_version == _version_display(tag):
            return _StepOutcome(
                True,
                f"build {tag} is already installed",
                {"tag": tag, "method": "current", "from": node_version, "to": node_version},
            )
        if facts.get("lop") != "yes":
            if facts.get("uv") != "yes":
                return _StepOutcome(
                    False,
                    "the machine has no build and no `uv` to install one with; nothing "
                    "was installed",
                )
            # `--refresh` revalidates uv's cached simple-index response: the
            # drill (F3, 2026-10-04) measured a release published shortly
            # before the run resolving as "no version … unsatisfiable" off
            # that cache.
            command = f"uv tool install --refresh local-operator=={shlex.quote(tag)}"
            method = "uv-tool-install"
        elif facts.get("lop_update") == "yes":
            command = f"lop-update {shlex.quote(tag)}"
            method = "lop-update"
        elif facts.get("uv") == "yes":
            # §3.4 names `lop-update` for an existing build; a node with a build
            # but no updater script still has uv, and a pinned reinstall is the
            # remaining documented spelling. The running relay picks the new
            # build up at step 9's restart. `--refresh` as at the fresh-install
            # branch: a cached index can hide the release this run is for.
            command = f"uv tool install --force --refresh local-operator=={shlex.quote(tag)}"
            method = "uv-tool-reinstall"
        else:
            return _StepOutcome(
                False,
                "the machine runs an older build but has neither `lop-update` nor `uv`, "
                "so the approved build cannot be installed",
            )
        result = self._remote_lop(command, timeout=self._step_timeout("install"))
        if result.rc != 0:
            return _StepOutcome(
                False,
                _install_failure_detail(tag, result),
                {"tag": tag, "method": method},
            )
        check = self._remote_lop("lop --version", timeout=min(60.0, self._step_timeout("install")))
        installed = _version_display((check.stdout or "").strip())
        if check.rc != 0 or installed != _version_display(tag):
            return _StepOutcome(
                False,
                f"the build install finished but `lop --version` reports "
                f"{installed or 'nothing'} rather than {tag}",
                {"tag": tag, "method": method},
            )
        return _StepOutcome(
            True,
            f"installed the approved build {tag} on the other machine "
            f"(was {node_version or 'absent'})",
            {"tag": tag, "method": method, "from": node_version, "to": installed},
        )

    def step_join(self) -> _StepOutcome:
        """§3.3 step 6: identity + ``join --automated`` — compare-then-admit.

        ``--automated`` automates only the JOINING end's transcription: the node
        sends its OWN derived code (``cli._join_one``'s ``typed = result.sas``,
        wire-identical to a typed one), and the inviter still compares it with
        its own derivation — a mismatch refuses, spends the attempt and audits
        ``sas_mismatch``. The compare lives in the relay; this step must never
        turn the send into a bless, and does not.
        """
        if self.invite_path is None:
            return _StepOutcome(False, "no invite token is held; the invite step must run first")
        show = self._node_json("lop network identity show", timeout=60.0)
        identity_before = ""
        if show and show.get("ok"):
            identity_before = str(show.get("device_id") or "")
        remote_token = f"/tmp/lop-invite-{self.invite_id}.invite"
        pushed = self.transport.copy(self.invite_path, remote_token)
        if pushed.rc != 0:
            return _StepOutcome(
                False,
                "the invite token could not be delivered to the machine: "
                + (
                    pushed.stderr.strip().splitlines()[-1][:200]
                    if pushed.stderr.strip()
                    else "no output"
                ),
            )
        try:
            joined = self._remote_lop(
                f"lop network join @{shlex.quote(remote_token)} --automated --json",
                timeout=self._step_timeout("join"),
            )
            payload = _json_from(joined.stdout or "")
            if joined.rc != 0 or not payload or not payload.get("ok"):
                refusal = ""
                message = ""
                if isinstance(payload, dict):
                    refusal = str(payload.get("code") or "")
                    message = str(payload.get("message") or payload.get("error") or "")
                tail = (joined.stderr or "").strip().splitlines()
                # THE SENTENCE IS THE MESSAGE; THE CODE IS A MACHINE FIELD (drill
                # finding, 2026-10-03; design round 1, D6): preferring the code
                # alone rendered every failed re-onboard as a bare ``join_failed``
                # and dropped the payload's ``message`` — where the per-host detail
                # lives ("nothing was listening at …"). The message now rides the
                # sentence; the code rides ``data.code`` for the machine register.
                embedded = message or refusal
                embedded = embedded or (tail[-1][:200] if tail else "the join was refused")
                data: dict[str, Any] = {
                    "invite_id": self.invite_id,
                    "code": refusal,
                    "node_refused": bool(refusal or message),
                }
                # WHAT THE NODE PROBE VERIFIED, IN ITS OWN WORDS (drill finding,
                # 2026-10-03; design round 1, D4/D5; extended 2026-10-04). Four
                # failed re-onboards ran while the node's own systemd relay held
                # :4097 (its log: ``OSError: [Errno 98] Address already in use``).
                # The probe is the one ``step_relay`` reads; the failure names what
                # it SAW — never a mechanism this build cannot produce, never a port
                # it did not verify. AND THE SENTENCE IS NETWORK-AWARE (drill
                # decision, 2026-10-04): a relay that already serves the network
                # this join was FOR is ADOPTED — named as correct, left in place,
                # and the remedy never sends the reader to restart it; a relay
                # serving other networks gets the restart remedy, which is true
                # because ``lop network restart`` now replaces a hand-started relay
                # with the supervised one (``relay._supervised_action``). The
                # generic sentence stays for a status payload that cannot tell the
                # two apart (no decoded token, no served list).
                # FAILURE-PATH ONLY: a join that completes never probes.
                status = self._node_json("lop network status", timeout=60.0)
                where = str(self.view.device.get("name") or "").strip() or "that machine"
                status = status if isinstance(status, dict) else {}
                listening = status.get("listening")
                live_port = None
                if isinstance(listening, dict):
                    try:
                        live_port = int(listening.get("port") or 0) or None
                    except (TypeError, ValueError):
                        live_port = None
                served = _served_networks(status)
                target = _invite_network_label(
                    self.invite_path,
                    network_id=self.invite_network_id,
                    network_name=self.invite_network_name,
                )
                if status.get("relay_answering") and live_port is not None:
                    # VERIFIED serving: the relay answered THIS probe AND named
                    # its port — the only state that may say "serving :<port>".
                    # The parenthetical prevents the one misread this sentence
                    # invites: the serving listener is the node's OWN port, not
                    # one of the addresses the join's refusal names.
                    #
                    # WHAT THE SERVING RELAY SERVES DECIDES THE SENTENCE (drill
                    # decision, 2026-10-04 — "join ADOPTS a relay already serving
                    # the SAME network; a DIFFERENT one refuses with cause").
                    # Round 1 sharpened every branch to say only what is true
                    # (D1/D3/D4/Q-1): a restart never clears a join failure in
                    # ANY branch, no branch prescribes it as the fix, and with no
                    # usable target the probe claims no comparison. Round 2
                    # closed the three claims left in these same sentences: the
                    # EMPTY served list is its own branch (D7 — an answering
                    # relay that lists nothing TOLD the probe that, it is the
                    # ordinary fresh-device shape), the "once a join completes"
                    # claim is scoped to the join target (D8 — `lop network init`
                    # puts a network in the store with no join at all), and the
                    # kind note reads the probe's verified `relay_served_by`,
                    # never unit-file presence (D6). Each branch still ends on
                    # the next move: retry once the cause the node reported is
                    # cleared.
                    same_network = bool(target) and any(
                        (target[0] and row["network_id"] == target[0])
                        or (target[1] and row["name"] == target[1])
                        for row in served
                    )
                    label = ""
                    if target is not None:
                        label = target[1] or target[0] or "the network this join is for"
                    if same_network:
                        detail = (
                            f"a relay is already serving :{live_port} on {where}, and it "
                            f"serves {label} — that relay was left as it is and is not the "
                            f"cause of this failure ({embedded}). Retry the join once the "
                            "cause in the node's message is cleared; nothing about the relay "
                            "needs restarting"
                        )
                        data["relay_serving"] = True
                        data["relay_port"] = live_port
                        data["relay_serves_target"] = True
                    elif target is not None and served:
                        # Empty display values are filtered BEFORE the join
                        # (round-1 R-NIT-1): two all-empty rows used to render
                        # "a different network (, )".
                        names = ", ".join(
                            value
                            for value in (row["name"] or row["network_id"] for row in served)
                            if value
                        )
                        names = names or "a different network"
                        detail = (
                            f"a relay is already serving :{live_port} on {where}, but it does "
                            f"not serve {label} — it serves {names} instead; the network this "
                            f"join is for can appear there only after a completed join, so the "
                            f"relay is not the cause of this failure ({embedded}). "
                            f"{_relay_kind_note(status)} Retry the join once the cause in the "
                            "node's message is cleared"
                        )
                        data["relay_serving"] = True
                        data["relay_port"] = live_port
                        data["relay_serves_target"] = False
                    elif target is not None and status.get("networks") == []:
                        # D7: the probe DID tell here — the relay answered and
                        # lists no network at all (the ordinary fresh-device
                        # shape). It certainly does not serve the target.
                        detail = (
                            f"a relay is already serving :{live_port} on {where}, and it serves "
                            f"no network yet; the network this join is for can appear there "
                            f"only after a completed join, so the relay is not the cause of "
                            f"this failure ({embedded}). {_relay_kind_note(status)} Retry the "
                            "join once the cause in the node's message is cleared"
                        )
                        data["relay_serving"] = True
                        data["relay_port"] = live_port
                        data["relay_serves_target"] = False
                    else:
                        # Q-1: with NO usable target (no payload ids, undecodable
                        # token) the probe cannot compare anything — it must not
                        # render the different-network sentence or claim a
                        # ``relay_serves_target`` it never determined. It says it
                        # could not MATCH the relay to the join's network (D7's
                        # wording: the honest failure of the comparison, not an
                        # overclaim in either direction), and writes no target
                        # field.
                        detail = (
                            f"a relay is already serving :{live_port} on {where} (its own "
                            "listener — a different address from any endpoint the join's "
                            "refusal names), and this probe could not match it to the network "
                            f"this join is for, so it cannot be cleared as the cause "
                            f"({embedded}). {_relay_kind_note(status)} Retry the join once the "
                            "cause in the node's message is cleared"
                        )
                        data["relay_serving"] = True
                        data["relay_port"] = live_port
                elif status.get("relay_running"):
                    # RUNNING BUT UNCONFIRMED (review round 1, R-MINOR): the earlier
                    # shape fell back to the port it was ASKED about and asserted
                    # it serving — a claim the probe did not verify. It says what
                    # is known instead, with the same one action.
                    detail = (
                        f"a relay process is registered on {where} but this probe "
                        f"could not confirm it serving (state: "
                        f"{status.get('relay_state') or 'unknown'}), and the join did "
                        f"not complete ({embedded}). Restart the relay on {where} with "
                        "`lop network restart`, then retry the join"
                    )
                    data["relay_state"] = str(status.get("relay_state") or "")
                else:
                    detail = f"the join did not complete ({embedded})"
                # The token file is removed by the ``finally`` below — one place.
                return _StepOutcome(False, detail, data)
            after = self._node_json("lop network identity show", timeout=60.0)
            device_id = str((after or {}).get("device_id") or payload.get("device_id") or "")
            fingerprint = str((after or {}).get("fingerprint") or payload.get("fingerprint") or "")
            return _StepOutcome(
                True,
                f"joined {payload.get('name') or 'the network'} as {device_id}"
                + (" (identity minted)" if not identity_before else "")
                + "; the code compare passed on the inviting side",
                {
                    "invite_id": self.invite_id,
                    "device_id": device_id,
                    "fingerprint": fingerprint,
                    "network_id": payload.get("network_id"),
                    "network_name": payload.get("name"),
                    "members": payload.get("members"),
                    # The node's own derived code, recorded for the receipts
                    # (§2.6: "both SAS codes are recorded"). The inviter's value
                    # lives in the relay's parked record and its audit; a
                    # mismatch would have refused here, so this value IS the one
                    # the compare saw.
                    "sas": payload.get("sas"),
                    "identity_minted": not identity_before,
                },
            )
        finally:
            self._cleanup_remote(remote_token)

    def _cleanup_remote(self, remote_path: str) -> None:
        try:
            self._remote_lop(f"rm -f {shlex.quote(remote_path)}", timeout=30.0)
        except MeshRefusal:
            pass

    def step_anchor(self) -> _StepOutcome:
        """§3.3 step 7 (+ F4b): copy EXACTLY the digested statement and install it.

        Three derivations of the trio — at mint (slice (a)), at approve (slice
        (a)), and HERE, before planting — are the design, not decoration: the
        runner re-derives ``{key_id, spki_fp, statement_digest}`` from the local
        store, refuses on any difference from the record, and ships the exact
        bytes whose digest it checked.
        """
        from local_operator.operator import anchor_bytes, load_anchor
        from local_operator.operator.trust import load_staged_anchor, statement_digest
        from local_operator.operator.verify import key_id_for, spki_fp
        from local_operator.paths import config_dir

        anchor = load_staged_anchor(config_dir())
        if anchor is None:
            anchor = load_anchor().anchor
        if anchor is None:
            return _StepOutcome(
                False,
                "operator authority is not set up on this machine, so there is no "
                "anchor to install on the machine being onboarded",
            )
        payload = anchor_bytes(anchor)
        trio = {
            "key_id": key_id_for(anchor.spki),
            "spki_fp": spki_fp(anchor.spki),
            "statement_digest": statement_digest(anchor),
        }
        want = self.view.what.get("anchor") or {}
        mismatched = [
            field_name
            for field_name in ("key_id", "spki_fp", "statement_digest")
            if str(want.get(field_name) or "") != trio[field_name]
        ]
        if mismatched:
            return _StepOutcome(
                False,
                "the anchor this machine holds does not match the one the request "
                f"approved ({', '.join(mismatched)}); nothing was planted",
                {"mismatch": mismatched},
            )
        staging = Path(tempfile.mkdtemp(prefix="lop-onboard-anchor-"))
        try:
            local_statement = staging / "anchor.json"
            local_statement.write_bytes(payload)
            os.chmod(local_statement, 0o600)
            remote_statement = f"/tmp/lop-anchor-{trio['key_id'][:8]}.json"
            pushed = self.transport.copy(local_statement, remote_statement)
            if pushed.rc != 0:
                return _StepOutcome(False, "the anchor statement could not be delivered")
            try:
                result = self._remote_lop(
                    f"lop operator install --from {shlex.quote(remote_statement)}",
                    timeout=self._step_timeout("anchor"),
                )
                if result.rc != 0 and self.facts.get("sudo_nopass") != "yes":
                    result = self._install_with_sudo_secret(remote_statement, result)
                if result.rc != 0:
                    tail = (result.stderr or result.stdout or "").strip().splitlines()
                    return _StepOutcome(
                        False,
                        "the anchor install did not complete: "
                        + (tail[-1][:200] if tail else "no output"),
                        {"key_id": trio["key_id"], "spki_fp": trio["spki_fp"]},
                    )
                trust = self._remote_lop("lop operator trust", timeout=60.0)
                if trust.rc != 0:
                    return _StepOutcome(
                        False,
                        "the anchor landed but the machine does not trust it yet; "
                        "the runtime will keep refusing allows there",
                        {"key_id": trio["key_id"], "spki_fp": trio["spki_fp"]},
                    )
                return _StepOutcome(
                    True,
                    f"operator anchor {trio['key_id']} installed and trusted",
                    {
                        "key_id": trio["key_id"],
                        "spki_fp": trio["spki_fp"],
                        "statement_digest": trio["statement_digest"],
                        "trusted": True,
                    },
                )
            finally:
                self._cleanup_remote(remote_statement)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _install_with_sudo_secret(
        self, remote_statement: str, first: CommandResult
    ) -> CommandResult:
        """OQ16's ask-once path: one ``sudo -S``, the password on the child's stdin.

        The password is resolved from the record's OWN credential block
        (``credential_ref.sudo`` — a secret-store name) and written ONLY to the
        child's stdin for this single command. It is never an argv element, an
        environment value, a log line or a receipt. The printed install command
        is run under ``sudo -S … sh -c`` so the one credential entry authorises
        the whole privileged step; nothing else on the node gains it.
        """
        sudo_ref = str((self.view.credential_ref or {}).get("sudo") or "")
        if not sudo_ref:
            return first
        secret = self.run_local([*self.local_cli, "secret", "get", sudo_ref], timeout=120.0)
        if secret.rc != 0 or not secret.stdout:
            return first
        printed = self._remote_lop(
            f"lop operator install --from {shlex.quote(remote_statement)} --print-only",
            timeout=60.0,
        )
        if printed.rc != 0 or not printed.stdout.strip():
            return first
        password = (
            secret.stdout.encode("utf-8")
            if isinstance(secret.stdout, str)
            else bytes(secret.stdout)
        )
        script = f"sudo -S -p '' sh -c {shlex.quote(printed.stdout.strip())}"
        return self._remote_lop_stdin(script, password, timeout=self._step_timeout("anchor"))

    def _remote_lop_stdin(self, command: str, stdin: bytes, *, timeout: float) -> CommandResult:
        return self.transport.run(
            ["sh", "-c", _SCRIPT_PROLOGUE + command], timeout=timeout, stdin=stdin
        )

    def step_grants(self) -> _StepOutcome:
        """§3.3 step 8: what the node lets THIS device do, granted per scope.

        Each device decides what a peer may do on IT, so both the ``approve``
        scope ("answer approval prompts for sessions here") and the
        ``unattended`` scope ("start sessions here without approval prompts")
        are grants the NODE holds about the operator's device id.
        """
        caps = [str(cap) for cap in (self.view.what.get("grant") or []) if cap]
        if bool(self.view.what.get("unattended")):
            caps.append("unattended")
        caps = sorted(set(caps))
        if not caps:
            return _StepOutcome(True, "no grants were requested", {"capabilities": []})
        from local_operator.network import identity as identity_mod
        from local_operator.network import store as network_store

        mac = identity_mod.load()
        if mac is None:
            return _StepOutcome(False, "this machine has no mesh identity to grant capabilities to")
        network_id = str(self.view.what.get("network_id") or "")
        network_name = ""
        for record in network_store.list_networks():
            if record.network_id == network_id:
                network_name = record.name
                break
        if not network_name:
            return _StepOutcome(
                False,
                "this machine does not know the network this request names, so the "
                "grant cannot be scoped",
            )
        command = " ".join(
            [
                "lop network member grant",
                shlex.quote(network_name),
                shlex.quote(mac.device_id),
                *(shlex.quote(cap) for cap in caps),
                "--json",
            ]
        )
        result = self._remote_lop(command, timeout=self._step_timeout("grants"))
        payload = _json_from(result.stdout or "")
        if result.rc != 0 or not payload or not payload.get("ok"):
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            return _StepOutcome(
                False,
                "the capability grant did not complete: "
                + (tail[-1][:200] if tail else "no output"),
                {"device_id": mac.device_id, "capabilities": caps},
            )
        # D7: the receipt is where a person confirms what they handed over, so
        # it says it in the sentences the card showed — not in capability tokens.
        human = {
            "approve": "answer approval prompts for sessions",
            "unattended": "start sessions without approval prompts",
        }
        granted = " and ".join(human.get(cap, cap) for cap in caps)
        return _StepOutcome(
            True,
            f"granted to this device on {network_name}: {granted}",
            {
                "device_id": mac.device_id,
                "network": network_name,
                "capabilities": list(payload.get("capabilities") or caps),
                "applied": payload.get("applied"),
            },
        )

    def step_relay(self) -> _StepOutcome:
        """§3.3 step 9 (+ OQ11): supervision via the service arm; linger caveat.

        ``lop network restart`` is the one job: after the build step it also
        moves a running relay onto the new build, and on Linux it installs the
        systemd ``--user`` unit when the host has none (both arms live in
        ``network/relay.py``'s supervision block). Linger absence does NOT fail
        the onboard — it is recorded, because OQ11's default is the caveat and
        an onboarded device that drops its relay at logout is still onboarded.
        """
        restart = self._remote_lop(
            "lop network restart --json", timeout=self._step_timeout("relay")
        )
        # A refusal is JSON TOO (rc 1 + {ok: false, code, message}): parsing only
        # on rc 0 dropped the one sentence that says why the service call failed.
        result = _json_from(restart.stdout or "")
        ok = bool(result and result.get("ok"))
        if not ok:
            status = self._node_json("lop network status", timeout=60.0)
            running = bool(
                status and (status.get("relay_running") or status.get("relay_answering"))
            )
            detail = ""
            if isinstance(result, dict):
                detail = str(result.get("message") or result.get("error") or "")
            if not detail and restart.stderr.strip():
                detail = restart.stderr.strip().splitlines()[-1]
            if running:
                detail = detail or "a relay answers here but the service action failed"
                return _StepOutcome(
                    False,
                    f"the relay restart did not complete ({detail}); the relay itself "
                    "still answers. If an old process is holding the connection, ask "
                    "Local Operator to stop it and retry this step",
                    {"action": "restart", "relay_answer": True},
                )
            return _StepOutcome(
                False,
                "the relay service could not be started" + (f": {detail}" if detail else ""),
                {"action": "restart", "relay_answer": False},
            )
        linger = self.facts.get("linger", "")
        caveat = ""
        user = self.facts.get("user") or ""
        if self.facts.get("systemctl") == "yes" and user:
            # Best-effort (OQ11): a self-linger usually needs no elevation, and a
            # refusal must not fail an otherwise-complete onboard.
            self._remote_lop(f"loginctl enable-linger {shlex.quote(user)}", timeout=60.0)
            recheck = self._remote_lop(
                f"loginctl show-user {shlex.quote(user)} -p Linger --value", timeout=60.0
            )
            linger = (recheck.stdout or "").strip() or linger
        if linger != "yes" and self.facts.get("systemctl") == "yes":
            caveat = (
                "the relay will stop when that machine's login session ends; keeping "
                "it running after logout is not enabled there yet"
            )
        return _StepOutcome(
            True,
            "relay service restarted and answering" + (f"; caveat: {caveat}" if caveat else ""),
            {
                "action": "restart",
                "linger": linger,
                "caveat": caveat,
                "service": self._service_name(),
            },
        )

    def _service_name(self) -> str:
        return "systemd --user" if self.facts.get("systemctl") == "yes" else "launchd"

    def step_verify(self) -> _StepOutcome:
        """§3.3 step 10: the acceptance surface, then the fold to ``connected``.

        The Mac-side ``lop network ready --peer <device>`` is the acceptance
        surface (its rows are what the operator sees); the node-side ``doctor``
        and ``peers`` are recorded alongside. ``verify`` fails on a ready
        payload that says it is not ok, and on a member row that never appeared.
        """
        name = str(self.view.device.get("name") or "")
        device_id = ""
        for receipt in reversed(self.view.receipts):
            if str(receipt.get("step")) == "join" and receipt.get("ok"):
                device_id = str((receipt.get("data") or {}).get("device_id") or "")
                break
        doctor = self._node_json("lop network doctor", timeout=self._step_timeout("verify"))
        peers = self._remote_lop("lop network peers --json", timeout=60.0)
        ready: dict[str, Any] | None = None
        if name or device_id:
            peer = name or device_id
            ready_result = self.run_local(
                [*self.local_cli, "network", "ready", "--peer", peer, "--json"],
                timeout=self._step_timeout("verify"),
            )
            ready = _json_from(ready_result.stdout or "")
            if ready is None:
                tail = (ready_result.stderr or ready_result.stdout or "").strip().splitlines()
                return _StepOutcome(
                    False,
                    "the readiness check did not produce a report: "
                    + (tail[-1][:200] if tail else "no output"),
                    {"peer": peer},
                )
            if not ready.get("ok"):
                failing = [
                    str(row.get("check") or row.get("name") or "?")
                    for row in (ready.get("rows") or ready.get("checks") or [])
                    if not (row.get("ok") if isinstance(row, dict) else True)
                ]
                return _StepOutcome(
                    False,
                    "the machine is not ready yet"
                    + (f" — these checks still fail: {', '.join(failing)}" if failing else ""),
                    {"peer": peer, "ready": {"ok": ready.get("ok")}},
                )
        else:
            return _StepOutcome(
                False,
                "the request names no device to verify against; the join step must "
                "have recorded one",
            )
        return _StepOutcome(
            True,
            "every readiness check passed" + (" for " + name if name else ""),
            {
                "peer": name or device_id,
                "ready_ok": bool(ready and ready.get("ok")),
                "doctor_ok": bool(doctor and doctor.get("ok")),
                "peers_rc": peers.rc,
            },
        )

    # -- dispatch ----------------------------------------------------------

    def run_step(self, step: str) -> _StepOutcome:
        return {
            "invite": self.step_invite,
            "pre_read": self.step_pre_read,
            "install": self.step_install,
            "join": self.step_join,
            "anchor": self.step_anchor,
            "grants": self.step_grants,
            "relay": self.step_relay,
            "verify": self.step_verify,
        }[step]()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _receipt(
    run_id: str, step: str, outcome: _StepOutcome, *, at: float | None = None
) -> dict[str, Any]:
    """The §2.2 receipt row, plus one additive key: ``data``.

    §2.2 freezes ``{run_id, step, at, ok, detail, digest}``; ``data`` rides
    beside them because the runner READS ITS OWN RECEIPTS BACK (the retry's
    contradiction check, the verify step's device id) and because a receipt the
    UI can open is worth more than a sentence it cannot. The ``digest`` binds
    the data — sha256 of its canonical JSON — so a receipt cannot be re-worded
    without the digest disagreeing, and slice (a)'s writer appends rows verbatim.
    """
    digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(outcome.data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    return {
        "run_id": run_id,
        "step": step,
        "at": float(at if at is not None else _now()),
        "ok": bool(outcome.ok),
        "detail": outcome.detail[:4000],
        "digest": digest,
        "data": dict(outcome.data),
    }


def execute_approval(
    approval_id: str,
    *,
    transport: Transport | None = None,
    transport_factory: Callable[[Any], Transport] | None = None,
    resolve: Callable[[dict[str, Any]], ResolvedCredential] | None = None,
    local_cli: Sequence[str] | None = None,
    run_local: Callable[..., CommandResult] = _run_local,
    clock: Callable[[], float] = _now,
    root: Any = None,
) -> dict[str, Any]:
    """Execute an approved onboarding record. THE runner entry point.

    This is what the approvals CLI's ``run`` verb calls (slice (a)); the return
    value is the §3.5 run payload: ``{approval_id, state, steps[], next}``, with
    ``ok`` and ``error`` added on the failure paths. It RETURNS on step
    failures — a failed run is an outcome with receipts, not an exception — and
    raises only for caller errors (unknown id, a record that cannot start).

    ``transport``/``transport_factory`` and ``resolve`` are the test seams: the
    unit cells drive the whole machine against a scripted fake with no SSH
    anywhere in CI.
    """
    view = approvals_adapter.begin_run(approval_id, root=root)
    if transport is None:
        if transport_factory is None:
            device = view.device
            host = str(device.get("host") or "")
            port = int(device.get("port") or 0) or _port_from(host) or 22
            host = host.partition(":")[0] if ":" in host else host
            transport = SshTransport(
                host=host,
                port=port,
                user=str(device.get("user") or ""),
            )
        else:
            transport = transport_factory(view)
    resolver = resolve or (lambda ref: resolve_credential(ref, cli=local_cli))
    run = OnboardRun(
        approval_id,
        view=view,
        transport=transport,
        resolve=resolver,
        local_cli=local_cli,
        run_local=run_local,
        clock=clock,
        root=root,
    )
    steps: list[dict[str, Any]] = []
    state = "connecting"
    error: dict[str, str] | None = None
    next_step = "done"
    try:
        for step in STEP_NAMES:
            if step != "invite":
                # The per-step gate BEFORE every credentialed step (§2.4). The
                # invite step runs on the approval's own edge (begin_run), and
                # the gate's refusals are terminal for this run.
                run._gate()
            try:
                outcome = run.run_step(step)
            except OnboardContradiction as contradiction:
                receipt = _receipt(
                    run.view.run_id,
                    step,
                    _StepOutcome(False, contradiction.sentence, contradiction.finding),
                )
                steps.append(receipt)
                # The FAILING receipt is written by the store's own terminal
                # write in the same locked mutation (mark_failed); appending it
                # separately would leave the step on the record twice.
                approvals_adapter.finish(
                    approval_id,
                    "failed",
                    run.view.run_id,
                    step=step,
                    detail=contradiction.sentence,
                    root=root,
                )
                state = "failed"
                next_step = step
                error = {"code": contradiction.code, "message": contradiction.sentence}
                break
            receipt = _receipt(run.view.run_id, step, outcome)
            steps.append(receipt)
            if not outcome.ok:
                approvals_adapter.finish(
                    approval_id,
                    "failed",
                    run.view.run_id,
                    step=step,
                    detail=outcome.detail,
                    root=root,
                )
                state = "failed"
                next_step = step
                error = {"code": f"{step}_failed", "message": outcome.detail}
                break
            if step == STEP_NAMES[-1]:
                # The LAST step's receipt is the terminal write's own
                # (mark_connected lands it in the same mutation); every earlier
                # ok step appends here.
                approvals_adapter.finish(
                    approval_id,
                    "connected",
                    run.view.run_id,
                    step=step,
                    detail=outcome.detail,
                    root=root,
                )
                state = "connected"
                break
            approvals_adapter.append_receipt(approval_id, run.view.run_id, receipt, root=root)
    except MeshRefusal as refusal:
        # A gate refusal (denied / expired / not approved / already connected)
        # lands as its own failing receipt, then the record's state. Expiry is
        # the ONE case that folds to `expired` rather than `failed` (§2.4).
        receipt = _receipt(run.view.run_id, "gate", _StepOutcome(False, refusal.sentence))
        steps.append(receipt)
        terminal = "expired" if refusal.code == "approval_expired" else "failed"
        try:
            # ``failed`` writes its own receipt (mark_failed); an expiry is an
            # observation the store's fold already reflects, so finish() is a
            # no-op there. A deny is already on the record — its write is
            # refused, and that is the honest outcome for this run.
            approvals_adapter.finish(
                approval_id,
                terminal,
                run.view.run_id,
                step="gate",
                detail=refusal.sentence,
                root=root,
            )
        except MeshRefusal:
            pass
        state = terminal
        next_step = "request"
        error = {"code": refusal.code, "message": refusal.sentence}
    finally:
        release_credential(run.credential)
        try:
            transport.close()
        except Exception:  # noqa: BLE001 — teardown must never mask a result
            pass
        if state != "connected":
            _release_invite_decision(run)
    latest = approvals_adapter.load(approval_id, root=root)
    return {
        "ok": state == "connected",
        "approval_id": approval_id,
        "state": state,
        "run_id": run.view.run_id,
        "steps": steps,
        "next": next_step,
        "error": error,
        "record_state": latest.state if latest else state,
    }


def _release_invite_decision(run: OnboardRun) -> None:
    """Clear the pre-answered admit for the invite a run ends WITHOUT consuming.

    The decision IS the operator's gesture for ONE ceremony — slice (b)
    pre-answers the relay's parked confirm. The relay clears it in its own
    ``finally`` when a ceremony consumes it; a run that ends earlier (install
    failed, a deny observed, expiry, a crash mid-run) has no ceremony to do that,
    and a stale admit decision would otherwise sit waiting for any later parking
    of the same invite until the TTL sweep (agent review round 1, Finding 1's
    secondary note). ONE WRITER, ONE OWNER: the run that wrote it clears it —
    and only for outcomes where nothing consumed it (a successful run's
    ceremony already did, and the cleanup is skipped so the success path stays
    byte-for-byte what the relay left).
    """
    if not run.invite_id:
        return
    from local_operator.network import store as network_store

    try:
        network_store.clear_pair_decision(run.invite_id)
    except MeshRefusal:
        pass


def step_runner(record: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """The seam ``lop network approvals run`` looks up (``APPROVAL_RUNNER_MODULE``).

    Slice (a) documents the contract, verbatim: this module exposes
    ``step_runner`` → a callable ``runner(record, *, root) -> dict`` that drives
    the record-side phase machine and answers the frozen run shape
    ``{approval_id, state, steps, next}``. The binding is one call: the runner
    is :func:`execute_approval`, which already drives the store through the
    ``onboard_approvals`` adapter; ``root`` is threaded to every store call so
    a caller that relocated the store reaches the same one.
    """
    approval_id = str(record.get("approval_id") or "")

    def runner(record: dict[str, Any], *, root: Any = None) -> dict[str, Any]:
        # ``record`` is the already-loaded dict the seam hands back; the id it
        # carries is what the adapter re-reads (and re-verifies) per step — one
        # source of truth, not a second copy of the record state.
        return execute_approval(approval_id, root=root)

    return runner


def _port_from(host: str) -> int:
    _, _, port_text = host.rpartition(":")
    try:
        return int(port_text)
    except ValueError:
        return 0
