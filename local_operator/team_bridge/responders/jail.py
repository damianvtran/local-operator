"""``ResponderJail``: the kernel boundary one alert-responder run lives inside (S-41).

UNWIRED BY DESIGN. Nothing in the shipping tree constructs this yet: the
production spawner that applies it lands in slice C2 together with GATE
R3-HASH, and ``test_nothing_wires_the_responder_runner_before_gate_r3_hash``
keeps failing any PR that reaches the runner sooner. This module only builds
and proves the boundary; it never starts a run.

WHY A NEW PROFILE AND NOT AN EXISTING ONE. A responder run reads input an
outsider wrote (an alert body) and holds ``bash`` plus egress, so it is the
first confined shell in the tree that both reads host files AND talks to the
internet. Neither existing shape fits:

* :class:`~local_operator.tools.confinement.ShellWriteConfinement` (the scout
  shell) allows EVERY read minus a credential denylist (``(allow file-read*)``).
  A denylist names credential stores, not data: ``~/workspace`` customer
  exports, private source trees, ``~/.zsh_history`` and other sessions'
  transcripts all stay readable, and with egress that is exfiltration
  (security design round 1, S-3). So reads here are DENY-DEFAULT with an
  allowlist, the shape :meth:`ToolConfinement.profile` already uses.
* :class:`~local_operator.tools.confinement.ToolConfinement` has that read
  shape but NO network, and the run's job (reading AWS/EKS state for triage)
  needs HTTPS.

READS. ``(literal "/")``, metadata on the whole tree (path resolution stats
ancestors; see ToolConfinement.profile's second point), and content reads of
exactly: the system read roots MINUS two changes, the interpreter tree, and
the run home. The two changes are genuine edits to the tuple, not reuse
(security design round 2, S-3 note):

* ``/Library`` is granted as its CHILDREN with ``Keychains`` OMITTED, never as
  ``/Library`` plus a trailing deny. Seatbelt's filtered-rule precedence is
  measured, not documented (confinement.py ``kernel_exposed_read_subtrees``),
  and an omission does not depend on it. A ``/Library`` child created after
  the profile is built is simply not granted: that fails closed.
* ``/private/var/folders`` (every user's temp and cache area) is replaced by
  the run's own ``TMPDIR`` inside the run home (S-15 / design §4).

Nothing else under ``$HOME`` is granted: ``~/workspace``, ``~/minervaai``,
``~/pergamon-labs``, a repository not materialized for this run, shell history,
other sessions and ``auth.db`` all answer EPERM. The credential denials
(:func:`~local_operator.tools.confinement._secret_read_denials`) are appended
last as belt-and-braces; with deny-default they should never be the rule that
decides.

WRITES. The run home and the output devices. The materialized repositories
under ``<home>/repos`` are then re-denied for writes: a LATER filtered deny
beats an earlier filtered allow on this host (measured, confinement.py
``kernel_exposed_read_subtrees`` correction of 2026-10-26), and the chmod the
materializer applies is not a boundary against a same-uid process.

NETWORK (architect finding A-1, security S-15b). There is no bare
``(allow network-outbound)``: a network allow defeats a network deny in either
order (ToolConfinement.profile, fourth point). Instead a FILTERED allow for TCP
to port 443, followed by a filtered deny of every loopback address. ``*`` in
``(remote tcp "*:443")`` includes localhost, so without the trailing deny any
local listener on 443 (a dev proxy, a tunnel client) is reachable; the deny is
what the C1 probes measure. AF_UNIX connects match no ``remote tcp`` allow and
are refused, except the one DNS socket.

RESIDUALS, stated rather than hidden (security round 1, S-22). The port
filter is NOT an exfiltration bound; the confidentiality control is the read
allowlist plus a read-only, at-most-one-hour STS triple. Specifically:

* any public host on :443 is reachable, and so are other machines on the LAN,
  VPN or tailnet on :443 -- seatbelt's network filter names only ``*`` and
  ``localhost``. This host's own LAN address was measured REFUSED (EPERM) on
  macOS 27.2.1, so lop's listeners are not reachable that way either;
* in-process ``web_fetch`` runs outside seatbelt and reaches any public host
  on any port (only its SSRF guard applies, forced on below);
* DNS through mDNSResponder is a covert channel to any authoritative server;
* everything readable is exfiltrable by design: materialized radient-ml
  source, the STS triple itself, and whatever pod logs/specs the role reads.

A host-allowlisting egress proxy (allow only ``localhost:<proxy>``; the proxy
admits ``*.amazonaws.com`` and the EKS endpoint) is the real fix and is a
follow-up slice, not this one.

PROCESS ARGUMENTS (security rounds 1-2, S-17) -- A RECORDED C1 RESIDUAL,
GATED. A jailed child can read another same-uid process's argv AND
environment through the numeric ``sysctl({CTL_KERN, KERN_PROCARGS2, pid})``.
Six profile variants were measured on macOS 27.2.1 and none blocks it:
removing ``sysctl-read`` entirely, a ``sysctl-name`` allowlist, a trailing
``(deny sysctl-read (sysctl-name "kern.procargs2"))``, and three
``process-info*`` target filters (a bare ``(deny process-info*)`` kills the
child itself); security round 2 corroborated it independently. The kernel's
procargs check is uid-based, so the boundary is a DISTINCT UID, not this
profile. ``(allow sysctl-read)`` therefore stays (narrowing it buys nothing
measurable). Security accepted this as a C1 residual only because GATE
S17-UID (``test_nothing_uses_the_responder_jail_before_gate_s17_uid``, a
permanent PR guard) fails any production import of this module or
``isolation`` until the PR that runs responder children under a separate uid
and proves, from the real spawn path, that a jailed child reads no canary from
an operator-uid non-platform process. Removing GATE R3-HASH does not lift it.

MACH. The two opendirectory names every confined child needs plus the DNS
resolver. No securityd, so the keychain CLIs fail; no pasteboard.

T0 IN-PROCESS TOOLS (security design round 2, S-14). seatbelt wraps CHILD
processes only. Tier T0 is ``read, grep, glob, bash, web_fetch,
bridge_thread, ask, todo``; of those only ``bash`` is a child. The others run
in lop's own process and need the same boundary stated in code:

* ``read``/``grep``/``glob`` take :meth:`ResponderJail.tool_confinement` --
  a ``ToolConfinement`` rooted at the run home. Its ``path_denial`` judges the
  RESOLVED path the tool's own resolver computed, so a symlink the run plants
  in its home pointing at ``~/workspace`` is refused, not followed. The
  in-process path tools are therefore NARROWER than the shell (run home only,
  no system trees); nothing a responder needs to read with them lives outside
  its home.
* ``web_fetch`` takes :func:`responder_web_fetch_settings`, which forces
  ``allow_private=False`` whatever the owner's settings say: the setting is
  owner-writable, so a planter could otherwise flip it and reach the serve
  port over loopback in-process, where the kernel filter above never applies.
* ``bridge_thread``, ``ask`` and ``todo`` read no host path and open no socket
  of their own; ``bridge_thread`` is scoped by ``bridge_scope``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from local_operator.tools import confinement as _confinement

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.tools.confinement import ToolConfinement
    from local_operator.web_fetch.models import WebFetchSettings

#: ``/Library`` children never granted. Omitted from the allowlist, never
#: denied after it (module docstring, READS).
_LIBRARY_OMITTED: frozenset[str] = frozenset({"Keychains", "Logs"})

#: Package-manager prefixes granted CHILD BY CHILD, with their mutable state
#: omitted (security round 1, S-16). ``var`` holds the owner's local data
#: stores -- measured on the build host: ``/opt/homebrew/var/mongodb`` (live
#: WiredTiger files), ``postgresql@16`` (mode 0700, owner = the run's uid) and
#: ``var/log`` -- and ``etc`` holds service configs (``mongod.conf``,
#: ``redis.conf``, ``gnupg``). The toolchain needs neither: git, python3,
#: aws, kubectl and curl were measured working without them; TLS uses the
#: run home's own bundle (``CA_BUNDLE_NAME``), not ``etc/openssl@3``.
#:
#: ``/usr`` is in this map too (review R2-1 / security r2 S-16): seatbelt's
#: ``subpath`` covers every descendant, so a whole ``/usr`` grant silently
#: re-granted ``/usr/local/{var,etc}`` (the Intel/migrated Homebrew layout)
#: no matter what the ``/usr/local`` row omitted. ``/usr`` is therefore granted
#: child by child WITHOUT ``local``, and ``/usr/local`` by its own row.
_PREFIX_OMITTED: Mapping[str, frozenset[str]] = {
    "/usr": frozenset({"local"}),
    "/opt/homebrew": frozenset({"var", "etc"}),
    "/usr/local": frozenset({"var", "etc"}),
}

#: System read roots the jail replaces: ``/Library`` by its children minus
#: Keychains, ``/private/var/folders`` by the run's own TMPDIR.
_REPLACED_SYSTEM_ROOTS: frozenset[str] = frozenset(
    {"/Library", "/private/var/folders", *_PREFIX_OMITTED}
)

#: The one tree under ``$HOME`` an interpreter root may sit in (S-21): uv's
#: managed Pythons, which is where both the ``lop`` tool install and the dev
#: venvs resolve ``sys.executable`` on this fleet. Anything else under HOME
#: (``~/workspace/x/.venv``) would grant that tree to the run.
_UV_PYTHON_SUBDIR = Path(".local") / "share" / "uv" / "python"

#: The resolver's mach service. ``getaddrinfo`` on macOS asks mDNSResponder;
#: without it a jailed ``curl https://…`` fails at name resolution. Measured
#: on the build host while writing this (see the PR's probe table).
_DNS_MACH_NAMES: tuple[str, ...] = ("com.apple.mDNSResponder",)

#: The one AF_UNIX socket a jailed child may connect: the DNS resolver's.
_DNS_UNIX_SOCKET = "/private/var/run/mDNSResponder"

#: The only outbound TCP port. HTTPS is the whole need of a read-only triage
#: run (AWS APIs, the EKS API server, public docs).
EGRESS_TCP_PORT = 443

#: Output devices a jailed child may open for writing. Not ``/dev/tty``: a
#: responder run has no terminal, and the operator's tty is not its to write.
_WRITE_DEVICES: tuple[str, ...] = ("/dev/null", "/dev/stdout", "/dev/stderr")

#: Environment keys a responder child may carry. Rebuilt, never inherited:
#: anything not here is dropped (no ``LOCAL_OPERATOR_*``, no hub bearer, no
#: ``AWS_PROFILE``, no ``GH_TOKEN``). The STS triple is added by the spawner
#: from the pinned credential resolver (``credentials.py``).
ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "KUBECONFIG",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "GIT_CONFIG_NOSYSTEM",
)

#: The system CA bundle the home copy is made from (public trust anchors).
_SYSTEM_CA_BUNDLE = Path("/private/etc/ssl/cert.pem")

#: The CA bundle's name INSIDE the run home. Deliberately not ``*.pem``: the
#: belt-and-braces credential denials include ``(regex #"/[^/]*\.pem$")``
#: (supplements/denylist.py, derived), which also blocks every CA bundle on the
#: host -- measured: jailed ``curl https://sts…`` failed with ``(77) error
#: setting certificate verify locations: CAfile: /etc/ssl/cert.pem``, and the
#: bisect over the 79 denial filters named that one rule. Copying the public
#: bundle in under a ``.crt`` name keeps the denial intact and TLS working.
CA_BUNDLE_NAME = "ca-bundle.crt"

#: The subset of :data:`ENV_ALLOWLIST` a caller may supply to ``child_env``:
#: exactly what ``PinnedCredentials.env()`` emits, plus ``LC_ALL`` (S-23).
_EXTRA_KEYS: frozenset[str] = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "LC_ALL",
    }
)

#: ``PATH`` for the child: system and Homebrew binaries only. Never the
#: owner's ``PATH``, which can name ``~/.local/bin`` launchers (``lop`` itself).
CHILD_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


class JailUnavailable(RuntimeError):
    """The jail cannot be built safely here; the caller must not spawn (§4)."""


def forbidden_read_roots() -> tuple[str, ...]:
    """Every path the allowlist deliberately omits; no grant may cover one."""

    omitted = [f"/Library/{name}" for name in sorted(_LIBRARY_OMITTED)]
    omitted += [f"{p}/{name}" for p, names in _PREFIX_OMITTED.items() for name in sorted(names)]
    omitted.append("/private/var/folders")
    return tuple(omitted)


def _library_children() -> tuple[str, ...]:
    """``/Library``'s children minus :data:`_LIBRARY_OMITTED`."""

    return _children("/Library", _LIBRARY_OMITTED)


def _children(root: str, omitted: frozenset[str]) -> tuple[str, ...]:
    """``root``'s children minus ``omitted``, sorted; ``()`` if ``root`` is absent.

    Enumerated at build time, so a child created later is simply not granted
    (fails closed). A child whose name the profile cannot carry is skipped
    (fail narrow, the same choice ``ToolConfinement.at`` makes for a
    scratchpad), never escaped.
    """

    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return ()
    children: list[str] = []
    for name in entries:
        # Dot-entries (Homebrew's own ``.git``, ``.github``…) are skipped
        # (review R2-3): nothing a run executes lives there, and they made the
        # profile -- and its attested hash -- longer and less stable.
        if name in omitted or name.startswith("."):
            continue
        path = f"{root}/{name}"
        if _confinement.profile_path_refusal(path) is not None:
            continue
        children.append(path)
    return tuple(children)


def _protected_roots() -> tuple[Path, Path]:
    """``(HOME, config dir)``, resolved: the trees no read root may cover."""

    from local_operator.paths import config_dir  # local: paths imports widely

    return Path.home().expanduser().resolve(), config_dir().expanduser().resolve()


def read_root_refusal(path: Path, *, home: Path, config: Path) -> str | None:
    """Why ``path`` cannot be a jail read root, or ``None`` (S-15a).

    A read root that is ``/``, ``$HOME``, an ancestor of either, or a tree
    containing the config dir silently re-grants everything the allowlist
    removed. The interpreter subtree is the realistic way in: a Python whose
    ``sys.executable`` resolves to ``~/bin/python`` makes its "subtree"
    ``$HOME``. Refuse to build instead of granting it.
    """

    root = Path("/")
    if path == root:
        return f"read root {path} is the filesystem root"
    if path == home or home.is_relative_to(path):
        return f"read root {path} covers HOME ({home})"
    if path == config or config.is_relative_to(path):
        return f"read root {path} contains the config dir ({config})"
    return None


def machinery_root_refusal(path: Path, *, home: Path, config: Path) -> str | None:
    """:func:`read_root_refusal` plus S-21 for an INTERPRETER root.

    An interpreter tree under ``$HOME`` that is not an ancestor still widens
    the jail: a copied-interpreter venv at ``~/workspace/x/.venv`` would grant
    ``~/workspace/x``. Only uv's managed-Python tree is accepted under HOME.
    (The run home is under HOME by design, so this rule is for machinery only.)
    """

    refusal = read_root_refusal(path, home=home, config=config)
    if refusal is not None:
        return refusal
    if path.is_relative_to(home) and not path.is_relative_to(home / _UV_PYTHON_SUBDIR):
        return f"interpreter root {path} is under HOME outside uv's managed Pythons"
    return None


@dataclass(frozen=True)
class ResponderJail:
    """The seatbelt boundary for one responder run, rooted at its own home.

    ``home`` is ``<session_dir>/responder-home``: the child's ``HOME``, its
    only writable tree, and (under ``repos/``) the read-only materialized
    repositories. It must never be an ancestor of HOME or the config dir.
    """

    home: Path
    #: Extra read roots the host decided at build time (the interpreter tree).
    #: Normally the default; a parameter so S-15a is testable with a fake path.
    machinery: tuple[Path, ...] = field(default=())

    @classmethod
    def at(
        cls, home: str | Path, *, machinery: Sequence[str | Path] | None = None
    ) -> ResponderJail:
        """Build the jail for ``home``, refusing any root that would widen it.

        Raises :class:`JailUnavailable` instead of returning a jail whose
        meaning a path could change; the spawner turns that into "no spawn".
        """

        raw = Path(home).expanduser()
        # The LEAF must not be a symlink (review R1-3): resolving one would
        # make whatever it points at (``~/workspace``, ``~/.ssh``) the run's
        # read+write root. Only the parent is resolved; the leaf is kept.
        if raw.is_symlink():
            raise JailUnavailable(f"run home {raw} is a symlink")
        if raw.exists() and not raw.is_dir():
            raise JailUnavailable(f"run home {raw} is not a directory")
        resolved = raw.parent.resolve() / raw.name
        if machinery is None:
            subtree = _confinement._interpreter_subtree()
            machinery = () if subtree is None else (subtree,)
        roots = tuple(Path(p).expanduser().resolve() for p in machinery)
        jail = cls(home=resolved, machinery=roots)
        jail.refusal_or_raise()
        return jail

    @property
    def tmpdir(self) -> Path:
        """The run's ``TMPDIR``: inside the home, so the narrowed temp grant holds."""

        return self.home / "tmp"

    @property
    def repos_dir(self) -> Path:
        """Where the materializer writes read-only repository trees."""

        return self.home / "repos"

    def _path_refusal(self) -> str | None:
        """Why a root this jail interpolates would widen it, or ``None``.

        Only PATH checks: building a jail (and its profile text) works on any
        platform so the shape is testable on Linux CI; enforcement is
        :meth:`unavailable_reason`.
        """

        home, config = _protected_roots()
        checks = [(self.home, read_root_refusal)]
        checks += [(root, machinery_root_refusal) for root in self.machinery]
        for root, check in checks:
            refusal = check(root, home=home, config=config)
            refusal = refusal or _confinement.profile_path_refusal(root)
            if refusal is not None:
                return refusal
        return _confinement._profile_roots_refusal(self.home)

    def refusal_or_raise(self) -> None:
        """Raise :class:`JailUnavailable` for a root that would widen the jail."""

        refusal = self._path_refusal()
        if refusal is not None:
            raise JailUnavailable(refusal)

    def unavailable_reason(self) -> str | None:
        """Why this jail cannot be enforced here, or ``None`` (fail closed)."""

        return self._path_refusal() or _confinement.shell_sandbox_unavailable_reason()

    def read_subpaths(self) -> tuple[str, ...]:
        """Every content-read root, in profile order (deny-default allowlist)."""

        system = tuple(
            path
            for path in _confinement._SYSTEM_READ_SUBPATHS
            if path not in _REPLACED_SYSTEM_ROOTS
        )
        reads = (
            str(self.home),
            *(str(path) for path in self.machinery),
            *system,
            *_library_children(),
            *(
                child
                for prefix, omit in _PREFIX_OMITTED.items()
                for child in _children(prefix, omit)
            ),
        )
        # COVERAGE, not membership (review R2-1): ``subpath`` grants every
        # descendant, so an omission only holds if no grant is the omitted path
        # or an ANCESTOR of it. A future edit to the system list that re-adds
        # an ancestor fails closed here instead of silently re-granting.
        for omitted in forbidden_read_roots():
            covering = [r for r in reads if omitted == r or omitted.startswith(r.rstrip("/") + "/")]
            if covering:
                raise JailUnavailable(f"read grant {covering[0]} covers omitted {omitted}")
        return reads

    def profile(self) -> str:
        """The seatbelt profile text (``sandbox-exec -p``). See the module docstring."""

        return self._render(egress_port=EGRESS_TCP_PORT)

    def _render(self, *, egress_port: int) -> str:
        """The profile with the egress port as a parameter -- FOR PROBES ONLY.

        Exists because an unprivileged process cannot bind 127.0.0.1:443 on
        the build host (measured: ``[Errno 13] Permission denied`` on macOS
        27.2.1), so the loopback-deny-beats-filtered-allow property is
        measured on an ephemeral port with the IDENTICAL rule shape (security
        S-15b's "highest-precedence substitute"). Every shipping caller goes
        through :meth:`profile`, which pins :data:`EGRESS_TCP_PORT`.
        """

        self.refusal_or_raise()
        reads = " ".join(f'(subpath "{path}")' for path in self.read_subpaths())
        writes = " ".join(f'(literal "{path}")' for path in _WRITE_DEVICES)
        mach = " ".join(
            f'(global-name "{name}")'
            for name in (*_confinement._MACH_LOOKUP_GLOBAL_NAMES, *_DNS_MACH_NAMES)
        )
        denials = " ".join(_confinement._secret_read_denials())
        return (
            "(version 1)\n"
            "(deny default)\n"
            "(allow process-fork)\n"
            "(allow process-exec)\n"
            "(allow signal (target same-sandbox))\n"
            "(allow sysctl-read)\n"
            f"(allow mach-lookup {mach})\n"
            '(allow file-read-metadata (subpath "/"))\n'
            f'(allow file-read* (literal "/") {reads})\n'
            f'(allow file-write* (subpath "{self.home}") {writes})\n'
            '(allow file-ioctl (literal "/dev/null"))\n'
            # Materialized repositories are read-only even to the run that
            # owns the files: a LATER filtered deny beats the filtered allow.
            f'(deny file-write* (subpath "{self.repos_dir}"))\n'
            # Egress: filtered allow, then the loopback carve-out AFTER it
            # (module docstring, NETWORK). Never a bare network allow.
            f'(allow network-outbound (remote tcp "*:{int(egress_port)}"))\n'
            f'(allow network-outbound (remote unix-socket (path-literal "{_DNS_UNIX_SOCKET}")))\n'
            '(deny network-outbound (remote ip "localhost:*"))\n'
            f"(deny file-read* {denials})\n"
        )

    def profile_sha256(self) -> str:
        """The digest the isolation attestation records for this exact profile."""

        return hashlib.sha256(self.profile().encode("utf-8")).hexdigest()

    def wrap(self, argv: Sequence[str]) -> list[str]:
        """``argv`` under the jail. Raises :class:`JailUnavailable`: no jail, no spawn."""

        reason = self.unavailable_reason()
        if reason is not None:
            raise JailUnavailable(reason)
        return [str(_confinement.SANDBOX_EXEC), "-p", self.profile(), *argv]

    def tool_confinement(self) -> ToolConfinement:
        """The in-process path tools' boundary: the run home, resolved (S-14a)."""

        from local_operator.tools.confinement import ToolConfinement

        return ToolConfinement.at(self.home)

    @property
    def ca_bundle(self) -> Path:
        """The run's CA bundle (see :data:`CA_BUNDLE_NAME`)."""

        return self.home / CA_BUNDLE_NAME

    def prepare_home(self) -> None:
        """Create the home's fixed layout: ``tmp/``, ``repos/`` and the CA bundle.

        Run by the (unjailed) host before the first spawn. Everything written
        here is public or empty; credentials are added by the spawner from the
        pinned resolver, never copied from the owner's ``~/.aws``.

        INVARIANT: ONE FRESH HOME PER RUN (review R1-2). The jailed child may
        write anywhere under the home, so a reused home could hold a planted
        ``ca-bundle.crt -> ~/.zshrc`` that this unjailed host would then
        overwrite. Every create here therefore refuses an existing entry
        (``exist_ok=False``, ``O_EXCL|O_NOFOLLOW``); C2 must allocate a new
        home per alert.
        """

        if not _SYSTEM_CA_BUNDLE.is_file():
            # No trust anchors means every HTTPS call fails inside the run;
            # refuse here rather than spawn a run that can only error.
            raise JailUnavailable(f"no system CA bundle at {_SYSTEM_CA_BUNDLE}")
        try:
            self.home.mkdir(exist_ok=False)
            self.tmpdir.mkdir(exist_ok=False)
            self.repos_dir.mkdir(exist_ok=False)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self.ca_bundle, flags, 0o444)
        except FileExistsError as exc:
            raise JailUnavailable(f"run home is not fresh ({exc.filename} exists)") from None
        with os.fdopen(fd, "wb") as handle:
            handle.write(_SYSTEM_CA_BUNDLE.read_bytes())

    def child_env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """The child's environment, rebuilt from :data:`ENV_ALLOWLIST` only.

        ``extra`` is the spawner's contribution: ONLY the keys in
        :data:`_EXTRA_KEYS` (the STS triple, the region, ``LC_ALL``). Anything
        else raises -- including the keys this method sets itself (security
        round 1, S-23), so a caller can neither hand a run ``AWS_PROFILE`` or
        a bearer nor replace ``HOME``, ``PATH``, ``TMPDIR`` or a CA bundle.
        """

        env = {
            "PATH": CHILD_PATH,
            "HOME": str(self.home),
            "TMPDIR": str(self.tmpdir),
            "LANG": "en_US.UTF-8",
            # Homebrew git reads ``/opt/homebrew/etc/gitconfig``, which S-16
            # no longer grants -- measured: jailed ``git --version`` died with
            # "unable to access '/opt/homebrew/etc/gitconfig': Operation not
            # permitted". Skipping system config is also what keeps the host's
            # ``credential.helper=osxkeychain`` out of the run.
            "GIT_CONFIG_NOSYSTEM": "1",
            "KUBECONFIG": str(self.home / ".kube" / "config"),
            "AWS_CONFIG_FILE": str(self.home / ".aws" / "config"),
            "AWS_SHARED_CREDENTIALS_FILE": str(self.home / ".aws" / "credentials"),
            # One bundle for every TLS stack a triage run uses: LibreSSL curl,
            # OpenSSL (SSL_CERT_FILE), botocore/aws CLI, Python requests.
            **{
                key: str(self.ca_bundle)
                for key in (
                    "AWS_CA_BUNDLE",
                    "SSL_CERT_FILE",
                    "CURL_CA_BUNDLE",
                    "REQUESTS_CA_BUNDLE",
                )
            },
        }
        for key, value in (extra or {}).items():
            if key not in _EXTRA_KEYS:
                raise ValueError(f"{key} is not a responder environment key a caller may set")
            env[key] = value
        return env


def responder_web_fetch_settings(owner: WebFetchSettings) -> WebFetchSettings:
    """The owner's web_fetch settings with the SSRF guard forced ON (S-14b).

    ``allow_private`` is owner-writable configuration; a responder run must
    never inherit a ``True`` a planter wrote, because ``web_fetch`` runs
    in-process where the jail's loopback deny does not apply. Every other
    setting (size ceilings, timeouts) is the owner's to keep.
    """

    return owner.model_copy(update={"allow_private": False})


__all__ = [
    "CHILD_PATH",
    "EGRESS_TCP_PORT",
    "ENV_ALLOWLIST",
    "JailUnavailable",
    "ResponderJail",
    "forbidden_read_roots",
    "machinery_root_refusal",
    "read_root_refusal",
    "responder_web_fetch_settings",
]
