"""`lop browser setup --node`: the node-managed browser on a headless mesh node.

WHY A NODE IS DIFFERENT FROM A USER'S MACHINE
---------------------------------------------
On a user's machine the browser tool drives the USER'S OWN Chrome through the
Local Operator extension: their logins, their screen, their decisions. The rule
"never install or script a browser engine" protects exactly that — a throwaway
browser cannot carry their logins and they cannot reach into it.

A headless mesh node has none of those things: no desktop, no user browser, no
screen, no logins to protect. Sessions offloaded there therefore had NO
`browser` tool at all, and visual/E2E work could not run there. This module
installs the ONE sanctioned browser for such a node: Google's signed Chrome
package, pinned, supervised by a user unit under a memory cap, loopback-only,
hosting the same extension and bridge daemon a user machine uses. Sessions get
the ordinary `browser` tool through the bridge — nothing in the tool layer is
special-cased for nodes.

SHAPE
-----
``plan`` → ``apply`` → ``status``. Every step has an idempotent ``check`` and
an ``apply``; re-running on a provisioned node reports each step "already in
place" and changes nothing. Privileged steps run through ``sudo -n`` and are
refused, with the exact remedy, when sudo would prompt (a prompt on a node is a
hang nobody can answer). External effects go through an injectable
:class:`NodeEnv`, so the unit tests drive every branch with fakes and never
touch dnf, rpm, systemd or the network.

REFUSALS (typed, each with a remedy): not Linux/x86_64; not a dnf/rpm distro;
this machine is a USER machine (the desktop app's discovery record exists, or a
graphical session is logged in) — that is guide://browser's extension path,
never this one; not a mesh member; ``--repair`` with nothing recorded to repair;
apply without ``--authorised-by``.

PAIRING WITHOUT A POPUP
-----------------------
Nobody can type a code into a headless Chrome, so setup mints one through the
daemon's own ``/pairing/codes`` route (what ``lop browser pair`` calls) and
delivers it via Chrome managed policy, ``3rdparty.extensions.<id>.pairingCode``,
which the extension reads from ``chrome.storage.managed`` and submits through
the same ``pair`` event the popup uses (extension/src/managed-pairing.ts).
This does not impersonate a user and adds no trust:

* the daemon still checks the code; no extension id is ever auto-trusted;
* the policy file is root-written and Chrome scopes the key to this extension
  id — no page and no other extension can read it;
* the code was minted by a same-uid ``lop`` process, which could already read
  it; nothing new is exposed;
* the key is removed from the policy as soon as the daemon reports the install
  paired (and in a ``finally`` on every failure path); the code also expires on
  the daemon's own TTL.

SECURITY CAVEAT (stated, not solved here): on a node whose account has
passwordless sudo, every node-side boundary — the policy file's ownership, the
unit's limits — is advisory and audited, not enforced against that account.
Per-site approvals on a node stay default-deny and are the operator's decision.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import functools
import getpass
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- what gets installed ------------------------------------------------------

CHROME_PACKAGE = "google-chrome-stable"
CHROME_BINARY = "/opt/google/chrome/chrome"
#: Google's Linux package signing key. The fingerprint is cross-checked against
#: https://www.google.com/linuxrepositories/ and is the trust anchor: a key file
#: whose primary fingerprint differs is refused before ``rpm --import``.
SIGNING_KEY_URL = "https://dl.google.com/linux/linux_signing_key.pub"
SIGNING_KEY_FINGERPRINT = "EB4C1BFD4F042F6DDDCCEC917721F63BD38B4796"
#: rpm names an imported key ``gpg-pubkey-<last 8 hex of the key id>``.
SIGNING_KEY_RPM = "gpg-pubkey-d38b4796"
REPO_PATH = Path("/etc/yum.repos.d/google-chrome.repo")
REPO_BASEURL = "https://dl.google.com/linux/chrome/rpm/stable/x86_64"
POLICY_PATH = Path("/etc/opt/chrome/policies/managed/local-operator.json")
VERSIONLOCK_PLUGIN = "python3-dnf-plugin-versionlock"
#: Enough coverage that ordinary pages do not render tofu: Latin/Cyrillic/Greek
#: (DejaVu), broad script coverage (Noto) and colour emoji. Fidelity is still
#: not a designer's Mac — see guide://browser "Headless nodes".
FONT_PACKAGES: tuple[str, ...] = (
    "dejavu-sans-fonts",
    "dejavu-serif-fonts",
    "dejavu-sans-mono-fonts",
    "google-noto-sans-fonts",
    "google-noto-serif-fonts",
    "google-noto-emoji-color-fonts",
)
CHROME_UNIT = "local-operator-node-browser.service"
#: Mirrors ``MANAGED_PAIRING_KEY`` in extension/src/managed-pairing.ts.
MANAGED_PAIRING_KEY = "pairingCode"
EXTENSION_RELEASE_REPO = "damianvtran/local-operator"
EXTENSION_ASSET_PREFIX = "local-operator-extension-v"

#: How long setup waits for the extension to CONNECT after Chrome (re)starts,
#: and then for it to report paired after the code lands in policy. Chrome's
#: Linux policy loader watches the policy directory and reloads after a short
#: settle delay (measured on this node's Chrome 157: the pair landed within
#: seconds of the write); the worker also retries on its reconnect alarm, whose
#: period is one minute. 120 s covers one missed watcher event plus one alarm
#: tick with margin, and stays well inside the daemon's pairing-code TTL so the
#: code cannot expire mid-wait.
CONNECT_WAIT_S = 120.0
PAIR_WAIT_S = 120.0
#: Bounds for the forcelist cycle that makes Chrome load a NEW CRX version
#: (``extension_loaded``). Measured on this node's Chrome 157 (n=1 each, the
#: 2026-11-13 run): the policy watcher uninstalled the extension 5 s after the
#: forcelist entry was removed, and installed 0.1.27 from the update manifest
#: 6 s after it was restored. Restarting Chrome does NOT do it — measured: a
#: restart kept 0.1.26 and pairing timed out — because a force-installed
#: extension is only re-checked on Chrome's own update timer, hours apart. 60 s
#: each is ~10x the observed latency and covers one missed watcher event.
UNINSTALL_WAIT_S = 60.0
INSTALL_WAIT_S = 60.0
POLL_INTERVAL_S = 1.0

RECORD_SCHEMA = 1

# --- step states and refusal codes -------------------------------------------

STATE_IN_PLACE = "in_place"
STATE_WOULD_APPLY = "would_apply"
STATE_APPLIED = "applied"
STATE_FAILED = "failed"
STATE_SKIPPED = "skipped"
#: A step a refusal stopped before it ran (sudo would prompt, or another step
#: was certain to fail). Distinct from ``skipped``, which follows a REAL failure:
#: "skipped, because an earlier step failed" was wrong when nothing had run.
STATE_NOT_RUN = "not_run"
#: Dry-run only: apply is certain to refuse or fail this step as things stand
#: (sudo would prompt, an extension that predates managed pairing). The dry-run
#: is what an operator reads BEFORE authorising, so it must predict that.
STATE_WOULD_FAIL = "would_fail"

REFUSE_PLATFORM = "unsupported_platform"
REFUSE_DISTRO = "unsupported_distro"
REFUSE_USER_MACHINE = "user_machine"
REFUSE_NOT_MEMBER = "not_mesh_member"
REFUSE_SUDO = "sudo_would_prompt"
REFUSE_NO_RECORD = "nothing_to_repair"
#: uninstall with no record of a lop install: what is there (a Chrome, Google's
#: repo file) is someone else's, and uninstall removes only what setup put there.
REFUSE_NOT_INSTALLED = "nothing_to_uninstall"
REFUSE_AUTHORISATION = "authorisation_required"


class StepError(RuntimeError):
    """A step's apply failed; the message is the technical reason shown."""


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _default_runner(
    argv: Sequence[str], *, input_text: str | None = None, timeout: float = 900.0
) -> subprocess.CompletedProcess[str]:
    # 900 s bounds the slowest real step (a ~110 MB Chrome rpm download plus a
    # dnf transaction on a small node); every other command returns in seconds.
    return subprocess.run(
        list(argv), input=input_text, capture_output=True, text=True, timeout=timeout, check=False
    )


def _fetch(url: str, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "local-operator-node-setup"})
    with urllib.request.urlopen(
        request, timeout=timeout
    ) as response:  # noqa: S310 - https constants
        return response.read()


def _user_machine_reason() -> str:
    """Why this looks like a USER machine, or "" when it does not.

    Two signals, either sufficient: the desktop app's discovery record (the app
    only runs where a person sits) and a logged-in graphical session (X11 or
    Wayland). A node with neither is headless by construction.
    """
    try:
        from local_operator.ui_browser import state as ui_state

        if ui_state.state_path().exists():
            return f"the desktop app's discovery record exists ({ui_state.state_path()})"
    except Exception:  # noqa: BLE001 - a probe that cannot answer is not a signal
        pass
    for var in ("DISPLAY", "WAYLAND_DISPLAY"):
        if os.environ.get(var):
            return f"a graphical session is attached (${var} is set)"
    if shutil.which("loginctl"):
        try:
            listing = subprocess.run(
                ["loginctl", "list-sessions", "--no-legend"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout
            for line in listing.splitlines():
                session_id = line.split()[0] if line.split() else ""
                if not session_id:
                    continue
                kind = subprocess.run(
                    ["loginctl", "show-session", session_id, "-p", "Type", "--value"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                ).stdout.strip()
                if kind in ("x11", "wayland", "mir"):
                    return f"a graphical login session is open (session {session_id}, {kind})"
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def node_name() -> str:
    """This device's mesh member name (what `lop network ready` calls it), else the hostname."""
    try:
        from local_operator.network import store as network_store

        for record in network_store.list_networks():
            member = record.self_member()
            if member is not None and getattr(member, "name", ""):
                return str(member.name)
    except Exception:  # noqa: BLE001 - a label falls back, never fails a command
        pass
    import socket

    return socket.gethostname()


def _mesh_member() -> bool:
    try:
        from local_operator.network import store as network_store

        return bool(network_store.list_networks())
    except Exception:  # noqa: BLE001 - unreadable mesh state is "not a member"
        return False


@dataclass
class NodeEnv:
    """Every external effect setup has, injectable for tests."""

    config_root: Path
    home: Path
    user: str
    runner: Runner = _default_runner
    fetch: Callable[[str], bytes] = _fetch
    system: Callable[[], str] = lambda: sys.platform
    machine: Callable[[], str] = platform.machine
    which: Callable[[str], str | None] = shutil.which
    user_machine_reason: Callable[[], str] = _user_machine_reason
    mesh_member: Callable[[], bool] = _mesh_member
    daemon_status: Callable[[], Mapping[str, Any]] = field(default=lambda: _daemon_status())
    install_daemon: Callable[[], Mapping[str, Any]] = field(default=lambda: _install_daemon())
    enable_linger: Callable[[], bool] = field(default=lambda: _enable_linger())
    request_codes: Callable[[], Mapping[str, Any]] = field(default=lambda: _request_codes())
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    now: Callable[[], str] = field(
        default=lambda: _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    @property
    def browser_dir(self) -> Path:
        return self.config_root / "browser"

    @property
    def extension_dir(self) -> Path:
        return self.browser_dir / "extension"

    @property
    def signing_key(self) -> Path:
        return self.extension_dir / "signing-key.pem"

    @property
    def update_manifest(self) -> Path:
        return self.extension_dir / "update.xml"

    @property
    def profile_dir(self) -> Path:
        return self.browser_dir / "node-profile"

    @property
    def record_path(self) -> Path:
        return self.browser_dir / "node-host.json"

    @property
    def unit_path(self) -> Path:
        return self.home / ".config" / "systemd" / "user" / CHROME_UNIT

    def run(self, *argv: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return self.runner(list(argv), input_text=input_text)

    def sudo(self, *argv: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        # ``-n``: never prompt. A prompt on a headless node is a hang nobody can
        # answer; the preflight refuses with a remedy before any step gets here.
        return self.run("sudo", "-n", *argv, input_text=input_text)


def _daemon_status() -> Mapping[str, Any]:
    from local_operator.browser_bridge import install

    return install.status()


def _install_daemon() -> Mapping[str, Any]:
    from local_operator.browser_bridge import install

    return install.install(install.DEFAULT_PORT)


def _enable_linger() -> bool:
    from local_operator.browser_bridge import install

    return install.enable_linger()


def _request_codes() -> Mapping[str, Any]:
    from local_operator.browser_bridge import install

    return install.request_pairing_codes()


def default_env() -> NodeEnv:
    from local_operator.paths import config_dir

    home = Path.home()
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - fall back to the environment's word
        user = os.environ.get("USER", "")
    return NodeEnv(config_root=config_dir().expanduser(), home=home, user=user)


# --- options, steps, results --------------------------------------------------


@dataclass
class Options:
    dry_run: bool = False
    repair: bool = False
    upgrade: bool = False
    extension_dir: Path | None = None
    authorised_by: str = ""


@dataclass
class Check:
    in_place: bool
    detail: str = ""
    #: Apply is CERTAIN to fail this step as things stand. Setup then stops
    #: before changing anything (rather than failing after Chrome and the
    #: other root steps landed), and the dry-run reports it as ``would_fail``.
    fails: bool = False


@dataclass
class Step:
    id: str
    privileged: bool
    check: Callable[[], Check]
    apply: Callable[[], str]


@dataclass
class StepResult:
    id: str
    privileged: bool
    state: str
    detail: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "privileged": self.privileged,
            "state": self.state,
            "detail": self.detail,
        }


@dataclass
class Outcome:
    mode: str
    ok: bool = True
    refusal: dict[str, Any] | None = None
    steps: list[StepResult] = field(default_factory=list)
    record_path: str = ""
    paired: bool | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "refusal": self.refusal,
            "steps": [step.as_json() for step in self.steps],
            "record": self.record_path,
            "paired": self.paired,
        }


def _refusal(code: str, message: str, remedy: str, **args: str) -> dict[str, Any]:
    # ``message``/``remedy`` are the English, machine-facing (--json) form;
    # ``args`` are the parameters the CLI's keyed copy renders from.
    return {"code": code, "message": message, "remedy": remedy, "args": dict(args)}


# --- the record ----------------------------------------------------------------


def read_record(env: NodeEnv) -> dict[str, Any] | None:
    try:
        value = json.loads(env.record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_private(path: Path, text: str, mode: int = 0o600) -> None:
    """Atomic write at ``mode``: the file never exists with wider permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _home_form(env: NodeEnv, path: Path) -> str:
    """``~/...`` when under home: the record is read by people and pasted into PRs."""
    try:
        return "~/" + str(path.relative_to(env.home))
    except ValueError:
        return str(path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _primary_fingerprints(listing: str) -> list[str]:
    """Each primary key's fingerprint in ``gpg --with-colons --show-keys`` output.

    A primary key is a ``pub`` record; ITS fingerprint is the ``fpr`` record
    directly after it. Every ``fpr`` after a ``sub`` belongs to a subkey and is
    deliberately not counted.
    """
    found: list[str] = []
    previous = ""
    for line in listing.splitlines():
        fields = line.split(":")
        if fields[0] == "fpr" and previous == "pub" and len(fields) > 9:
            found.append(fields[9])
        previous = fields[0]
    return found


# --- renderers (pure; their output IS the idempotency check) -------------------


def render_repo() -> str:
    return (
        "[google-chrome]\n"
        "name=google-chrome\n"
        f"baseurl={REPO_BASEURL}\n"
        "enabled=1\n"
        "gpgcheck=1\n"
        f"gpgkey={SIGNING_KEY_URL}\n"
    )


def render_update_manifest(extension_id: str, crx: Path, version: str) -> str:
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<gupdate xmlns='http://www.google.com/update2/response' protocol='2.0'>\n"
        f"  <app appid='{extension_id}'>\n"
        f"    <updatecheck codebase='{crx.as_uri()}' version='{version}' />\n"
        "  </app>\n"
        "</gupdate>\n"
    )


def render_policy(
    extension_id: str, update_manifest: Path, pairing_code: str = ""
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "ExtensionInstallForcelist": [f"{extension_id};{update_manifest.as_uri()}"]
    }
    if pairing_code:
        policy["3rdparty"] = {"extensions": {extension_id: {MANAGED_PAIRING_KEY: pairing_code}}}
    return policy


def render_unit(env: NodeEnv) -> str:
    """The Chrome user unit. Each line is a constraint, not a preference:

    * ``--headless=new``: no screen exists; measured, the extension service
      worker runs and connects to the daemon in this mode (no Xvfb needed);
    * NO ``--remote-debugging-port`` and NO ``--no-sandbox``: the extension's
      ``chrome.debugger`` over the loopback bridge is the only control channel;
    * ``--password-store=basic``: a headless node has no keyring;
    * ``MemoryMax=2G`` with ``OOMScoreAdjust=500``: the node OOM'd running
      sessions; under pressure the kernel takes Chrome before the relay or
      ``lop serve``, and the tool reports the dead tab as ``tab_closed``.
    """
    profile = env.profile_dir
    try:
        profile_arg = "%h/" + str(profile.relative_to(env.home))
    except ValueError:
        profile_arg = str(profile)
    return (
        "[Unit]\n"
        "Description=Local Operator node browser (headless Google Chrome, extension host)\n"
        "After=local-operator-browser.service\n"
        "Wants=local-operator-browser.service\n"
        "\n"
        "[Service]\n"
        "# Headless branded Chrome hosting the force-installed Local Operator extension,\n"
        "# which connects to the loopback bridge daemon (127.0.0.1:4099). No remote\n"
        "# debugging port and the sandbox stays ON: the extension's chrome.debugger is\n"
        "# the only control channel.\n"
        f"ExecStart={CHROME_BINARY} --headless=new --user-data-dir={profile_arg} "
        "--no-first-run --no-default-browser-check --password-store=basic about:blank\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "MemoryHigh=1536M\n"
        "MemoryMax=2G\n"
        "OOMScoreAdjust=500\n"
        "TasksMax=512\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


# --- extension identity ---------------------------------------------------------


def extension_id_for_key(pem: bytes) -> str:
    """Chrome's id for a signing key: sha256 of the DER public key, first 16
    bytes, each nibble mapped 0-f → a-p. Computed rather than read back from
    Chrome so setup knows the id before the CRX is ever installed."""
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(pem, password=None)
    der = key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    digest = hashlib.sha256(der).hexdigest()[:32]
    return "".join(chr(ord("a") + int(ch, 16)) for ch in digest)


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return ()


# --- the plan -------------------------------------------------------------------


class _Plan:
    """Builds the ordered steps; holds the facts later steps need."""

    def __init__(self, env: NodeEnv, options: Options, record: dict[str, Any] | None) -> None:
        self.env = env
        self.options = options
        self.record = record or {}
        self.updates: dict[str, Any] = {}
        self._extension_source: tuple[str, Path | None, str] | None = None
        #: Temporary directories this run created (an unpacked release zip);
        #: removed by :func:`setup` when the run ends.
        self.scratch: list[Path] = []

    # -- helpers --
    def _rpm_installed(self, name: str) -> str:
        result = self.env.run("rpm", "-q", name)
        return result.stdout.strip() if result.returncode == 0 else ""

    def _require(self, result: subprocess.CompletedProcess[str], what: str) -> str:
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:] or [""]
            raise StepError(f"{what} exited {result.returncode}: {tail[0][:300]}")
        return result.stdout

    def _sudo_install_file(self, text: str, target: Path, mode: str, group: str = "root") -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, prefix="lop-node-") as handle:
            handle.write(text)
            staged = handle.name
        try:
            self._require(
                self.env.sudo("mkdir", "-p", str(target.parent)), f"mkdir {target.parent}"
            )
            self._require(
                self.env.sudo(
                    "install", "-m", mode, "-o", "root", "-g", group, staged, str(target)
                ),
                f"install {target}",
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(staged)

    # -- steps --
    def steps(self) -> list[Step]:
        return [
            Step("signing_key", True, self.check_key, self.apply_key),
            Step("repo", True, self.check_repo, self.apply_repo),
            Step("versionlock_plugin", True, self.check_plugin, self.apply_plugin),
            Step("chrome", True, self.check_chrome, self.apply_chrome),
            Step("versionlock", True, self.check_lock, self.apply_lock),
            Step("fonts", True, self.check_fonts, self.apply_fonts),
            Step("bridge_daemon", False, self.check_daemon, self.apply_daemon),
            Step("linger", False, self.check_linger, self.apply_linger),
            Step("extension", False, self.check_extension, self.apply_extension),
            Step("update_manifest", False, self.check_update_manifest, self.apply_update_manifest),
            Step("policy", True, self.check_policy, self.apply_policy),
            Step("profile_dir", False, self.check_profile, self.apply_profile),
            Step("chrome_unit", False, self.check_unit, self.apply_unit),
            Step("extension_loaded", True, self.check_loaded, self.apply_loaded),
            Step("pairing", True, self.check_pairing, self.apply_pairing),
        ]

    def check_key(self) -> Check:
        found = self._rpm_installed(SIGNING_KEY_RPM)
        return Check(bool(found), found or f"rpm -q {SIGNING_KEY_RPM}: not imported")

    def apply_key(self) -> str:
        data = self.env.fetch(SIGNING_KEY_URL)
        with tempfile.TemporaryDirectory(prefix="lop-node-gpg-") as gnupg:
            key_file = Path(gnupg) / "linux_signing_key.pub"
            key_file.write_bytes(data)
            listing = self._require(
                self.env.run(
                    "gpg",
                    "--homedir",
                    gnupg,
                    "--batch",
                    "--with-colons",
                    "--show-keys",
                    str(key_file),
                ),
                "gpg --show-keys",
            )
            primaries = _primary_fingerprints(listing)
            # EXACTLY one primary key, and it is the pin. `rpm --import` imports
            # every key in the file, so "the pin appears somewhere" (a subkey
            # row, or Google's key with another appended) is not enough
            # (round 1, R1-5).
            if primaries != [SIGNING_KEY_FINGERPRINT]:
                raise StepError(
                    f"{SIGNING_KEY_URL} does not carry the expected key "
                    f"{SIGNING_KEY_FINGERPRINT} as its only primary key "
                    f"(primary keys found: {', '.join(primaries) or 'none'}); refused"
                )
            self._require(self.env.sudo("rpm", "--import", str(key_file)), "rpm --import")
        self.updates.setdefault("chrome", {})["signing_key"] = {
            "fingerprint": SIGNING_KEY_FINGERPRINT,
            "rpm_pubkey": self._rpm_installed(SIGNING_KEY_RPM),
            "key_file_sha256": _sha256(data),
        }
        return f"imported {SIGNING_KEY_FINGERPRINT}"

    def check_repo(self) -> Check:
        try:
            current = REPO_PATH.read_text(encoding="utf-8")
        except OSError:
            return Check(False, f"{REPO_PATH} absent")
        return Check(current == render_repo(), f"{REPO_PATH} (gpgcheck=1)")

    def apply_repo(self) -> str:
        self._sudo_install_file(render_repo(), REPO_PATH, "0644")
        return f"wrote {REPO_PATH}"

    def check_plugin(self) -> Check:
        found = self._rpm_installed(VERSIONLOCK_PLUGIN)
        return Check(bool(found), found or f"{VERSIONLOCK_PLUGIN} not installed")

    def apply_plugin(self) -> str:
        self._require(self.env.sudo("dnf", "-y", "install", VERSIONLOCK_PLUGIN), "dnf install")
        return f"installed {VERSIONLOCK_PLUGIN}"

    def check_chrome(self) -> Check:
        found = self._rpm_installed(CHROME_PACKAGE)
        if found and self.options.upgrade:
            return Check(False, f"{found} installed; --upgrade moves it to the newest signed build")
        return Check(bool(found), found or f"{CHROME_PACKAGE} not installed")

    def apply_chrome(self) -> str:
        """Download, verify, THEN install the exact file that was verified.

        ``dnf install <name>`` would verify too, but recording what was
        installed needs the file in hand: its sha256 and ``rpm -K`` verdict are
        the audit trail a reviewer reads in node-host.json.
        """
        if self.options.upgrade and self._rpm_installed(CHROME_PACKAGE):
            # The ONLY path that moves the pin: an unattended dnf upgrade must
            # never change the browser under running sessions.
            self.env.sudo("dnf", "versionlock", "delete", CHROME_PACKAGE)
        with tempfile.TemporaryDirectory(prefix="lop-node-rpm-") as staging:
            os.chmod(staging, 0o755)
            self._require(
                self.env.sudo("dnf", "-y", "download", "--destdir", staging, CHROME_PACKAGE),
                "dnf download",
            )
            rpms = sorted(Path(staging).glob(f"{CHROME_PACKAGE}-*.rpm"))
            if not rpms:
                raise StepError("dnf download produced no rpm")
            rpm = rpms[-1]
            verdict = self._require(self.env.run("rpm", "-K", str(rpm)), "rpm -K")
            if "digests signatures OK" not in verdict:
                raise StepError(f"rpm -K did not verify {rpm.name}: {verdict.strip()[:200]}")
            digest = _file_sha256(rpm)
            # `--setopt=localpkg_gpgcheck=1`: dnf does NOT verify a local file's
            # signature by default (measured on this node's dnf 4.14:
            # localpkg_gpgcheck=0), and the staging dir is user-owned, so a
            # same-uid process could swap the file between `rpm -K` and this
            # install. Re-verifying inside the root transaction closes that gap;
            # `rpm -K` stays for the recorded verdict (round 1, R1-4).
            self._require(
                self.env.sudo("dnf", "-y", "--setopt=localpkg_gpgcheck=1", "install", str(rpm)),
                "dnf install chrome",
            )
        version = self.env.run(CHROME_BINARY, "--version").stdout.strip()
        chrome = self.updates.setdefault("chrome", {})
        chrome.update(
            {
                "version": version,
                "package": self._rpm_installed(CHROME_PACKAGE),
                "rpm_filename": rpm.name,
                "rpm_sha256": digest,
                "rpm_K": "digests signatures OK",
                "repo": {"file": str(REPO_PATH), "baseurl": REPO_BASEURL, "gpgcheck": 1},
            }
        )
        return f"installed {rpm.name} (sha256 {digest[:12]}…, rpm -K OK)"

    def check_lock(self) -> Check:
        listing = self.env.run("dnf", "versionlock", "list").stdout
        locks = [
            line.strip() for line in listing.splitlines() if line.strip().startswith(CHROME_PACKAGE)
        ]
        installed = self._rpm_installed(CHROME_PACKAGE)
        if not locks:
            return Check(False, f"{CHROME_PACKAGE} not version-locked")
        # A lock naming another version than the installed one is a stale pin.
        version = (
            installed.removeprefix(CHROME_PACKAGE + "-").rsplit(".", 1)[0] if installed else ""
        )
        if version and not any(version.split("-")[0] in lock for lock in locks):
            return Check(False, f"lock {locks[0]} does not match {installed}")
        return Check(True, locks[0])

    def apply_lock(self) -> str:
        self.env.sudo("dnf", "versionlock", "delete", CHROME_PACKAGE)
        self._require(
            self.env.sudo("dnf", "versionlock", "add", CHROME_PACKAGE), "dnf versionlock add"
        )
        lock = self.check_lock().detail
        self.updates.setdefault("chrome", {})["versionlock"] = lock
        return f"locked {lock}"

    def check_fonts(self) -> Check:
        missing = [name for name in FONT_PACKAGES if not self._rpm_installed(name)]
        if missing:
            return Check(False, "missing " + ", ".join(missing))
        return Check(True, f"{len(FONT_PACKAGES)} font packages")

    def apply_fonts(self) -> str:
        missing = [name for name in FONT_PACKAGES if not self._rpm_installed(name)]
        self._require(self.env.sudo("dnf", "-y", "install", *missing), "dnf install fonts")
        self.updates["fonts_installed"] = [self._rpm_installed(name) for name in FONT_PACKAGES]
        return "installed " + ", ".join(missing)

    def check_daemon(self) -> Check:
        status = self.env.daemon_status()
        if status.get("installed") and status.get("healthy"):
            return Check(True, f"bridge daemon healthy on 127.0.0.1:{status.get('port')}")
        return Check(
            False,
            (
                "bridge daemon not installed"
                if not status.get("installed")
                else "bridge daemon not healthy"
            ),
        )

    def apply_daemon(self) -> str:
        result = self.env.install_daemon()
        if result.get("ok") is False:
            raise StepError(
                str(result.get("error") or result.get("message") or "lop browser install failed")
            )
        return "lop browser install"

    def check_linger(self) -> Check:
        out = self.env.run("loginctl", "show-user", self.env.user, "-p", "Linger").stdout.strip()
        return Check(out == "Linger=yes", out or "loginctl: no answer")

    def apply_linger(self) -> str:
        if self.env.enable_linger():
            return "loginctl enable-linger"
        self._require(
            self.env.sudo("loginctl", "enable-linger", self.env.user), "loginctl enable-linger"
        )
        return "sudo loginctl enable-linger"

    # -- extension --
    def _recorded_extension(self) -> Mapping[str, Any]:
        value = self.record.get("extension")
        return value if isinstance(value, Mapping) else {}

    def _extension_id(self) -> str:
        try:
            return extension_id_for_key(self.env.signing_key.read_bytes())
        except (OSError, ValueError):
            return str(self._recorded_extension().get("id") or "")

    def _crx_path(self, version: str) -> Path:
        return self.env.extension_dir / f"local-operator-{version}.crx"

    def _desired_extension(self) -> tuple[str, Path | None, str]:
        """(version, staged source dir or None for "keep recorded", provenance).

        Order: an explicit ``--extension-dir``; else, unless ``--upgrade``, the
        CRX already recorded (a re-run must not hit the network to stay a
        no-op); else the newest GitHub Release carrying the extension zip,
        verified against the asset's published sha256 digest.
        """
        if self._extension_source is not None:
            return self._extension_source
        if self.options.extension_dir is not None:
            source = self.options.extension_dir.expanduser()
            manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
            self._extension_source = (str(manifest["version"]), source, f"--extension-dir {source}")
            return self._extension_source
        recorded = self._recorded_extension()
        version = str(recorded.get("version") or "")
        kept = str(recorded.get("source") or "recorded")
        if version and not self.options.upgrade and self._crx_path(version).is_file():
            self._extension_source = (version, None, kept)
            return self._extension_source
        release = self._release_extension()
        if (
            version
            and self._crx_path(version).is_file()
            and _version_tuple(release[0]) <= _version_tuple(version)
        ):
            # `--upgrade` moves FORWARD only. The newest GitHub Release can lag
            # an install packed from `--extension-dir` (measured: release 0.1.26
            # vs installed 0.1.27), and offering it was a downgrade the dry-run
            # presented as an upgrade (round 1, U5c).
            self._extension_source = (
                version,
                None,
                f"{kept}; the newest release ({release[0]}, {release[2]}) is not newer, so kept",
            )
            return self._extension_source
        self._extension_source = release
        return self._extension_source

    def _release_extension(self) -> tuple[str, Path | None, str]:
        listing = json.loads(
            self.env.fetch(
                f"https://api.github.com/repos/{EXTENSION_RELEASE_REPO}/releases?per_page=100"
            )
        )
        for release in listing:
            if release.get("draft") or release.get("prerelease"):
                continue
            for asset in release.get("assets") or []:
                name = str(asset.get("name") or "")
                if not (name.startswith(EXTENSION_ASSET_PREFIX) and name.endswith(".zip")):
                    continue
                digest = str(asset.get("digest") or "")
                if not digest.startswith("sha256:"):
                    raise StepError(
                        f"{name} on {release.get('tag_name')} carries no sha256 digest; refused"
                    )
                data = self.env.fetch(str(asset["browser_download_url"]))
                if _sha256(data) != digest.removeprefix("sha256:"):
                    raise StepError(f"{name} does not match its published digest; refused")
                staged = Path(tempfile.mkdtemp(prefix="lop-node-ext-"))
                self.scratch.append(staged)
                archive = staged / name
                archive.write_bytes(data)
                unpacked = staged / "extension"
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(unpacked)
                version = name.removeprefix(EXTENSION_ASSET_PREFIX).removesuffix(".zip")
                return (
                    version,
                    unpacked,
                    f"GitHub Release {release.get('tag_name')} {name} ({digest})",
                )
        raise StepError(f"no release of {EXTENSION_RELEASE_REPO} carries an extension zip")

    def check_extension(self) -> Check:
        from local_operator.browser_bridge.protocol import (
            MANAGED_PAIRING_MIN_EXTENSION_VERSION,
        )

        try:
            version, _source, provenance = self._desired_extension()
        except (OSError, ValueError, KeyError, StepError) as error:
            return Check(False, f"extension source unavailable: {error}")
        if _version_tuple(version) < _version_tuple(MANAGED_PAIRING_MIN_EXTENSION_VERSION):
            # Certain: apply_extension raises for exactly this, so setup stops
            # before Chrome or any root step changes (round 1, U5c).
            return Check(
                False,
                f"extension {version} ({provenance}) predates managed pairing "
                f"({MANAGED_PAIRING_MIN_EXTENSION_VERSION}); "
                "pass --extension-dir with a newer build",
                fails=True,
            )
        crx = self._crx_path(version)
        recorded = self._recorded_extension()
        in_place = (
            self.env.signing_key.is_file()
            and crx.is_file()
            and str(recorded.get("version") or "") == version
            and str(recorded.get("crx_sha256") or "") == _file_sha256(crx)
            and (self.options.extension_dir is None or self._source_matches(recorded))
        )
        return Check(
            in_place, f"{self._extension_id() or '(no key yet)'} {version} from {provenance}"
        )

    def _source_matches(self, recorded: Mapping[str, Any]) -> bool:
        """An explicit dir counts as in place only if it is the tree recorded."""
        return str(recorded.get("source_tree_sha256") or "") == self._tree_digest(
            self.options.extension_dir.expanduser()  # type: ignore[union-attr]
        )

    @staticmethod
    def _tree_digest(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def apply_extension(self) -> str:
        from local_operator.browser_bridge.protocol import (
            MANAGED_PAIRING_MIN_EXTENSION_VERSION,
        )

        version, source, provenance = self._desired_extension()
        if _version_tuple(version) < _version_tuple(MANAGED_PAIRING_MIN_EXTENSION_VERSION):
            raise StepError(
                f"extension {version} predates managed pairing "
                f"({MANAGED_PAIRING_MIN_EXTENSION_VERSION}): "
                "it would install and stay unpaired; pass --extension-dir with a newer build"
            )
        if source is None:
            raise StepError("the recorded CRX is missing; pass --extension-dir or --upgrade")
        self.env.extension_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.env.extension_dir, 0o700)
        with tempfile.TemporaryDirectory(prefix="lop-node-pack-") as work:
            stage = Path(work) / "extension"
            shutil.copytree(source, stage)
            argv = [
                CHROME_BINARY,
                f"--user-data-dir={Path(work) / 'pack-profile'}",
                f"--pack-extension={stage}",
            ]
            new_key = not self.env.signing_key.is_file()
            if not new_key:
                # The SAME key every time: the extension id is derived from it,
                # and the policy, the daemon's pairing record and every CRX
                # after this one depend on that id staying put.
                argv.append(f"--pack-extension-key={self.env.signing_key}")
            self._require(self.env.run(*argv), "chrome --pack-extension")
            packed = Path(work) / "extension.crx"
            if not packed.is_file():
                raise StepError("chrome --pack-extension produced no CRX")
            if new_key:
                generated = Path(work) / "extension.pem"
                _write_private(self.env.signing_key, generated.read_text(encoding="utf-8"))
            crx = self._crx_path(version)
            shutil.copyfile(packed, crx)
            os.chmod(crx, 0o644)
        extension = self.updates.setdefault("extension", {})
        extension.update(
            {
                "delivery": (
                    "off-store CRX3, ExtensionInstallForcelist with file:// update manifest"
                ),
                "id": self._extension_id(),
                "version": version,
                "source": provenance,
                "crx_path": _home_form(self.env, crx),
                "crx_sha256": _file_sha256(crx),
                "signing_key": _home_form(self.env, self.env.signing_key)
                + " (0600; required to keep this id on every future CRX)",
            }
        )
        if self.options.extension_dir is not None:
            extension["source_tree_sha256"] = self._tree_digest(
                self.options.extension_dir.expanduser()
            )
        # Later steps read the record; keep their view current.
        self.record.setdefault("extension", {}).update(extension)
        return f"packed {crx.name} ({provenance})"

    def _current_version(self) -> str:
        try:
            return self._desired_extension()[0]
        except (OSError, ValueError, KeyError, StepError):
            return str(self._recorded_extension().get("version") or "")

    def check_update_manifest(self) -> Check:
        version = self._current_version()
        extension_id = self._extension_id()
        if not extension_id or not version:
            return Check(False, "no extension yet")
        wanted = render_update_manifest(extension_id, self._crx_path(version), version)
        try:
            current = self.env.update_manifest.read_text(encoding="utf-8")
        except OSError:
            return Check(False, f"{_home_form(self.env, self.env.update_manifest)} absent")
        return Check(
            current == wanted, f"{_home_form(self.env, self.env.update_manifest)} → {version}"
        )

    def apply_update_manifest(self) -> str:
        version = self._current_version()
        text = render_update_manifest(self._extension_id(), self._crx_path(version), version)
        _write_private(self.env.update_manifest, text, mode=0o644)
        self.updates.setdefault("extension", {}).update(
            {
                "update_manifest": _home_form(self.env, self.env.update_manifest),
                "update_manifest_sha256": _sha256(text.encode()),
            }
        )
        return f"update manifest → {version}"

    def _policy_now(self) -> dict[str, Any] | None:
        try:
            value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def check_policy(self) -> Check:
        extension_id = self._extension_id()
        if not extension_id:
            return Check(False, "no extension id yet")
        wanted = render_policy(extension_id, self.env.update_manifest)
        current = self._policy_now()
        if current is None:
            return Check(False, f"{POLICY_PATH} absent or unreadable")
        # A pairing code left behind by an interrupted run is NOT in place:
        # the re-apply rewrites the file without it.
        return Check(current == wanted, f"{POLICY_PATH} force-installs {extension_id}")

    def apply_policy(self) -> str:
        policy = render_policy(self._extension_id(), self.env.update_manifest)
        self._write_policy(policy)
        return f"wrote {POLICY_PATH}"

    def _write_policy(self, policy: Mapping[str, Any], *, secret: bool = False) -> None:
        text = json.dumps(policy, indent=2) + "\n"
        # 0640 root:<user's group> while a code is in it: Chrome (running as
        # this user) can read it, other local accounts cannot — PROVIDED the
        # group is private. A shared primary group (`users`) would hand the
        # code to its other members for the pairing window, so that case is
        # refused rather than written (0600 root would keep Chrome out too).
        group = self.env.run("id", "-gn").stdout.strip() if secret else "root"
        if secret:
            sharers = self._group_sharers(group)
            if sharers:
                raise StepError(
                    f"the pairing code would be readable by group {group!r}, which "
                    f"{', '.join(sharers)} also belong to; refused. Run setup from an "
                    "account whose primary group is private"
                )
        self._sudo_install_file(
            text, POLICY_PATH, "0640" if secret else "0644", group=group or "root"
        )
        if not secret:
            self.updates["policy"] = {
                "path": str(POLICY_PATH),
                "sha256": _sha256(text.encode()),
                "content": dict(policy),
            }

    def _group_sharers(self, group: str) -> list[str]:
        """Accounts other than this one in ``group`` (listed members or primary gid).

        Fails CLOSED: a group that cannot be read is treated as shared, because
        the alternative is writing a secret for an audience nobody checked
        (round 1, R1-7).
        """
        entry = self.env.run("getent", "group", group)
        fields = entry.stdout.strip().split(":")
        if entry.returncode != 0 or len(fields) < 4:
            return [f"(unreadable group entry for {group!r})"]
        gid = fields[2]
        others = {name for name in fields[3].split(",") if name and name != self.env.user}
        accounts = self.env.run("getent", "passwd")
        for line in accounts.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 4 and parts[3] == gid and parts[0] != self.env.user:
                others.add(parts[0])
        return sorted(others)

    def check_profile(self) -> Check:
        path = self.env.profile_dir
        if not path.is_dir():
            return Check(False, f"{_home_form(self.env, path)} absent")
        mode = path.stat().st_mode & 0o777
        return Check(mode == 0o700, f"{_home_form(self.env, path)} ({mode:04o})")

    def apply_profile(self) -> str:
        self.env.profile_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.env.profile_dir, 0o700)
        return f"{_home_form(self.env, self.env.profile_dir)} 0700"

    def _systemctl(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.env.run("systemctl", "--user", *args)

    def check_unit(self) -> Check:
        try:
            current = self.env.unit_path.read_text(encoding="utf-8")
        except OSError:
            return Check(False, f"{CHROME_UNIT} not installed")
        if current != render_unit(self.env):
            return Check(False, f"{CHROME_UNIT} differs from this build's unit")
        enabled = self._systemctl("is-enabled", CHROME_UNIT).stdout.strip()
        active = self._systemctl("is-active", CHROME_UNIT).stdout.strip()
        return Check(
            enabled == "enabled" and active == "active", f"{CHROME_UNIT} {enabled}, {active}"
        )

    def apply_unit(self) -> str:
        rendered = render_unit(self.env)
        changed = True
        with contextlib.suppress(OSError):
            changed = self.env.unit_path.read_text(encoding="utf-8") != rendered
        if changed:
            _write_private(self.env.unit_path, rendered, mode=0o644)
            self._require(self._systemctl("daemon-reload"), "systemctl --user daemon-reload")
        self._require(self._systemctl("enable", CHROME_UNIT), "systemctl --user enable")
        # restart, not start: a changed unit or a new CRX takes effect only in
        # a fresh Chrome. Restart of a stopped unit starts it.
        self._require(self._systemctl("restart", CHROME_UNIT), "systemctl --user restart")
        self.updates["chrome_unit"] = {
            "path": _home_form(self.env, self.env.unit_path),
            "sha256": _sha256(rendered.encode()),
            "profile": _home_form(self.env, self.env.profile_dir) + " (0700)",
            "limits": {
                "MemoryHigh": "1536M",
                "MemoryMax": "2G",
                "OOMScoreAdjust": 500,
                "TasksMax": 512,
            },
        }
        return f"{CHROME_UNIT} enabled and (re)started"

    def _installed_versions(self) -> list[str]:
        """Versions Chrome has unpacked for our id in the node profile."""
        root = self.env.profile_dir / "Default" / "Extensions" / self._extension_id()
        try:
            return sorted(
                entry.name.rsplit("_", 1)[0] for entry in root.iterdir() if entry.is_dir()
            )
        except OSError:
            return []

    def check_loaded(self) -> Check:
        version = self._current_version()
        installed = self._installed_versions()
        return Check(
            bool(version) and installed == [version],
            f"Chrome has {', '.join(installed) or 'nothing'} installed; the update manifest "
            f"offers {version or '?'}",
        )

    def _wait_for(self, predicate: Callable[[], bool], bound: float, what: str) -> None:
        deadline = self.env.monotonic() + bound
        while not predicate():
            if self.env.monotonic() >= deadline:
                raise StepError(f"{what} did not happen within {bound:.0f}s")
            self.env.sleep(POLL_INTERVAL_S)

    def apply_loaded(self) -> str:
        """Cycle the forcelist so Chrome drops the old CRX and installs the new one.

        Chrome installs a force-listed extension it does not have IMMEDIATELY,
        but re-checks one it already has only on its update timer (hours). So
        the entry is removed (Chrome uninstalls), then restored (Chrome
        installs the version update.xml offers). The extension's local storage
        goes with the uninstall — its pairing token included — which is why
        ``pairing`` runs after this step and reads the LIVE link, not only the
        daemon's saved identities.
        """
        version = self._current_version()
        extension_id = self._extension_id()
        clean = render_policy(extension_id, self.env.update_manifest)
        try:
            self._write_policy({**clean, "ExtensionInstallForcelist": []})
            self._wait_for(
                lambda: not self._installed_versions(),
                UNINSTALL_WAIT_S,
                f"Chrome uninstalling {extension_id}",
            )
        finally:
            # Never leave the node with the extension delisted.
            self._write_policy(clean)
        self._wait_for(
            lambda: self._installed_versions() == [version],
            INSTALL_WAIT_S,
            f"Chrome installing {extension_id} {version}",
        )
        # Wait for the NEW install's link before ``pairing`` reads it: until it
        # connects, the only answer is the daemon's saved identity, which the
        # uninstall just made stale (a false "paired").
        self._wait_for(
            lambda: _link_version(self.env.daemon_status()) == version,
            CONNECT_WAIT_S,
            f"{extension_id} {version} connecting to the bridge daemon",
        )
        return f"Chrome reloaded {extension_id} at {version}"

    def _paired(self) -> bool:
        """Paired = the daemon has our identity AND, if our link is up, authorises it.

        The saved identity alone is not enough: the ``extension_loaded`` cycle
        wipes the extension's token while the daemon still lists the id, and
        that install connects unpaired. When the link is connected the daemon's
        ``/health`` names the ids it authorised; that is the live answer.
        """
        return _live_paired(self.env.daemon_status(), self._extension_id())

    def check_pairing(self) -> Check:
        if self._paired():
            return Check(True, f"{self._extension_id()} paired")
        return Check(False, f"{self._extension_id() or 'extension'} not paired")

    def apply_pairing(self) -> str:
        """Mint → deliver through policy → bounded wait → remove. See module doc."""
        extension_id = self._extension_id()
        code = self._wait_for_code(extension_id)
        clean = render_policy(extension_id, self.env.update_manifest)
        try:
            self._write_policy(
                render_policy(extension_id, self.env.update_manifest, code), secret=True
            )
            deadline = self.env.monotonic() + PAIR_WAIT_S
            while self.env.monotonic() < deadline:
                if self._paired():
                    break
                self.env.sleep(POLL_INTERVAL_S)
            else:
                raise StepError(
                    f"the extension did not report paired within {PAIR_WAIT_S:.0f}s "
                    "of the policy write; "
                    "re-run with --repair (a fresh code is minted)"
                )
        finally:
            # Never leave a code in policy, on success or failure.
            self._write_policy(clean)
        self.updates["paired"] = True
        return f"{extension_id} paired through managed policy; code removed from policy"

    def _wait_for_code(self, extension_id: str) -> str:
        deadline = self.env.monotonic() + CONNECT_WAIT_S
        last = ""
        while True:
            answer = self.env.request_codes()
            if answer.get("ok"):
                for entry in answer.get("codes") or []:
                    if isinstance(entry, Mapping) and entry.get("extension_id") == extension_id:
                        code = str(entry.get("code") or "")
                        if code:
                            return code
                last = f"{answer.get('connected_count', 0)} connected, none is {extension_id}"
            else:
                last = str(answer.get("reason") or "no answer")
            if self.env.monotonic() >= deadline:
                raise StepError(
                    f"no pairing code for {extension_id} within {CONNECT_WAIT_S:.0f}s ({last}); "
                    f"is {CHROME_UNIT} running?"
                )
            self.env.sleep(POLL_INTERVAL_S)


def _link_version(status: Mapping[str, Any]) -> str:
    """The extension version on the daemon's live link, or "" when none is up."""
    health = status.get("health")
    if isinstance(health, Mapping) and health.get("extension_connected"):
        return str(health.get("extension_version") or "")
    return ""


def _live_paired(status: Mapping[str, Any], extension_id: str) -> bool:
    """See :meth:`_Plan._paired`; shared with :func:`browser_fact`."""
    if not extension_id:
        return False
    identities = status.get("identities") or []
    saved = any(
        isinstance(entry, Mapping) and entry.get("extension_id") == extension_id
        for entry in identities
    )
    health = status.get("health")
    if (
        saved
        and isinstance(health, Mapping)
        and health.get("extension_connected")
        and health.get("driver_extension_id") in (extension_id, None, "")
    ):
        # ``paired`` on /health is the LINK's authentication. Measured after a
        # reinstall: ``authorized_extension_ids`` still listed the id (it is
        # the saved list) while ``paired`` was false — the token was gone.
        return bool(health.get("paired"))
    return saved


# --- entry points ---------------------------------------------------------------


#: The removal remedy users read on a refused uninstall/authorisation, and the
#: placeholder every command example uses: ONE spelling (round 1, U11).
AUTHORISED_BY_HINT = "'<who, when>'"


def _host_refusal(env: NodeEnv, *, user_check: bool = True) -> dict[str, Any] | None:
    """Platform, distribution and user-machine refusals: shared by setup,
    uninstall and :func:`browser_fact`, so all three agree on which hosts the
    node-managed browser exists for (round 1, R1-2 / R1-6)."""
    remedy_user = "use the browser extension or the desktop app instead (guide://browser)"
    if env.system() != "linux" or env.machine() not in ("x86_64", "amd64"):
        return _refusal(
            REFUSE_PLATFORM,
            "the node-managed browser supports Linux x86_64 only "
            f"(this is {env.system()} {env.machine()})",
            remedy_user,
            system=env.system(),
            machine=env.machine(),
        )
    if not (env.which("dnf") and env.which("rpm")):
        return _refusal(
            REFUSE_DISTRO,
            "the node-managed browser needs a dnf/rpm distribution "
            "(Amazon Linux 2023, Fedora, RHEL family)",
            "run it on a supported node, or install a browser there by hand outside Local Operator",
        )
    reason = env.user_machine_reason() if user_check else ""
    if reason:
        return _refusal(
            REFUSE_USER_MACHINE,
            f"this is a user machine: {reason}",
            "on a user machine the browser tool drives the user's own Chrome through the extension "
            "or the desktop app (guide://browser); never install the node-managed one there",
            reason=reason,
        )
    return None


def _preflight(
    env: NodeEnv, options: Options, record: dict[str, Any] | None
) -> dict[str, Any] | None:
    refusal = _host_refusal(env)
    if refusal:
        return refusal
    if not env.mesh_member():
        return _refusal(
            REFUSE_NOT_MEMBER,
            "this device is not a member of a lop mesh network",
            "join the network first (guide://network), then re-run",
        )
    if options.repair and record is None:
        return _refusal(
            REFUSE_NO_RECORD,
            f"nothing to repair: {env.record_path} does not exist",
            f"run `lop browser setup --node --authorised-by {AUTHORISED_BY_HINT}` to install it",
            path=str(env.record_path),
        )
    if not options.dry_run and not options.authorised_by.strip():
        return _refusal(
            REFUSE_AUTHORISATION,
            "installing a browser on a node needs the operator's authorisation, recorded verbatim",
            f"re-run with --authorised-by {AUTHORISED_BY_HINT}",
        )
    return None


#: Called with each step's result the moment it is known, and the outcome so
#: far (whose ``refusal`` is already set when a refusal produced the rows).
OnStep = Callable[["StepResult", "Outcome"], None]


def _emit(outcome: Outcome, result: StepResult, on_step: OnStep | None) -> None:
    """Record a step and report it NOW.

    A fresh install runs for minutes (a ~110 MB rpm, two bounded 120 s waits),
    and its caller is usually an agent's shell call with a timeout: printing
    only at the end meant a killed call left no trace of which steps landed
    (round 1, U2). The CLI streams through this.
    """
    outcome.steps.append(result)
    if on_step is not None:
        on_step(result, outcome)


@contextlib.contextmanager
def _unwind_on_signals() -> Iterator[None]:
    """Turn SIGTERM/SIGINT/SIGHUP into an exception for the duration of a run.

    WHY: Python's default SIGTERM/SIGHUP action kills the process WITHOUT
    unwinding, so ``apply_pairing``'s ``finally`` — the one that takes the
    pairing code back out of Chrome's policy — never ran when an agent's
    command timeout fired mid-pairing (round 1, U2). Raising ``SystemExit(128 +
    signum)`` unwinds through every ``finally`` and still exits with the
    conventional killed-by-signal code. A second signal during that cleanup is
    ignored so the cleanup itself is not cut short (SIGKILL still works). A
    signal the caller IGNORES (a background job's SIGINT) stays ignored, and
    off the main thread, where handlers cannot be installed, nothing changes.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    fired: list[int] = []

    def _handler(signum: int, _frame: object) -> None:
        if fired:
            return
        fired.append(signum)
        raise SystemExit(128 + signum)

    previous: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        current = signal.getsignal(sig)
        if current is signal.SIG_IGN:
            continue
        previous[sig] = signal.signal(sig, _handler)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def setup(
    options: Options, env: NodeEnv | None = None, *, on_step: OnStep | None = None
) -> Outcome:
    env = env or default_env()
    record = read_record(env)
    outcome = Outcome(
        mode="dry_run" if options.dry_run else ("repair" if options.repair else "apply")
    )
    refusal = _preflight(env, options, record)
    if refusal:
        outcome.ok = False
        outcome.refusal = refusal
        return outcome
    plan = _Plan(env, options, record)
    try:
        with _unwind_on_signals():
            return _run_plan(env, options, record, plan, outcome, on_step)
    finally:
        for path in plan.scratch:
            shutil.rmtree(path, ignore_errors=True)


def _sudo_refusal(pending_privileged: list[str]) -> dict[str, Any]:
    return _refusal(
        REFUSE_SUDO,
        f"{len(pending_privileged)} step(s) need root "
        f"({', '.join(pending_privileged)}) and `sudo -n` would prompt",
        "run it from an account with non-interactive sudo; at an interactive terminal, run "
        "`sudo -v` first and then this command again in the same terminal "
        "(guide://browser 'Headless nodes')",
        steps=", ".join(pending_privileged),
    )


def _run_plan(
    env: NodeEnv,
    options: Options,
    record: dict[str, Any] | None,
    plan: _Plan,
    outcome: Outcome,
    on_step: OnStep | None = None,
) -> Outcome:
    steps = plan.steps()
    checks = [(step, step.check()) for step in steps]
    pending_privileged = [
        step.id for step, check in checks if not check.in_place and step.privileged
    ]
    # Probed in the dry-run too (read-only: `sudo -n true`): the dry-run is
    # what the operator reads before authorising, and it used to exit 0 with
    # every row "would apply" where apply would refuse (round 1, U5a).
    sudo_blocked = bool(pending_privileged) and env.sudo("true").returncode != 0
    certain = any(not check.in_place and check.fails for _step, check in checks)
    if options.dry_run:
        for step, check in checks:
            if check.in_place:
                state, detail = STATE_IN_PLACE, check.detail
            elif check.fails:
                state, detail = STATE_WOULD_FAIL, check.detail
            elif sudo_blocked and step.privileged:
                state = STATE_WOULD_FAIL
                detail = f"needs root and `sudo -n` would prompt ({check.detail})"
            else:
                state, detail = STATE_WOULD_APPLY, check.detail
            _emit(outcome, StepResult(step.id, step.privileged, state, detail), on_step)
        outcome.ok = not (sudo_blocked or certain)
        outcome.paired = plan._paired()  # noqa: SLF001 - same module
        return outcome
    if sudo_blocked or certain:
        # Stop BEFORE any change: a refusal (sudo) or a step certain to fail
        # (an extension without managed pairing) used to surface only after
        # Chrome and the other root steps had already been installed (U5c).
        # Nothing ran, so nothing is recorded either (round 1, U4b).
        outcome.ok = False
        if sudo_blocked:
            outcome.refusal = _sudo_refusal(pending_privileged)
        for step, check in checks:
            if check.in_place:
                result = StepResult(step.id, step.privileged, STATE_IN_PLACE, check.detail)
            elif check.fails:
                result = StepResult(step.id, step.privileged, STATE_FAILED, check.detail)
            else:
                result = StepResult(step.id, step.privileged, STATE_NOT_RUN, check.detail)
            _emit(outcome, result, on_step)
        return outcome
    failed = False
    applied: list[str] = []
    for step, _initial in checks:
        if failed:
            _emit(outcome, StepResult(step.id, step.privileged, STATE_SKIPPED), on_step)
            continue
        # Re-check at apply time: an earlier step (a new CRX, a restarted
        # unit) can change a later step's answer.
        check = step.check()
        if check.in_place:
            _emit(
                outcome,
                StepResult(step.id, step.privileged, STATE_IN_PLACE, check.detail),
                on_step,
            )
            continue
        try:
            detail = step.apply()
        except (StepError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            failed = True
            outcome.ok = False
            _emit(outcome, StepResult(step.id, step.privileged, STATE_FAILED, str(error)), on_step)
            continue
        applied.append(step.id)
        _emit(outcome, StepResult(step.id, step.privileged, STATE_APPLIED, detail), on_step)
    outcome.paired = plan._paired()  # noqa: SLF001 - same module
    _write_record(env, record, plan.updates, options, applied, outcome)
    # Set only once written: the CLI prints `Record:` from it, and a path for a
    # file that was never written read as an audit trail that did not exist.
    outcome.record_path = str(env.record_path)
    return outcome


def _write_record(
    env: NodeEnv,
    record: dict[str, Any] | None,
    updates: Mapping[str, Any],
    options: Options,
    applied: list[str],
    outcome: Outcome,
) -> None:
    """Merge this run into node-host.json, keeping the spike's schema-1 fields.

    ``authorised_by`` names who authorised the CURRENT install state (the run
    that last changed something); every run, including no-ops, is appended to
    ``history`` verbatim so the audit trail survives repairs.
    """
    merged: dict[str, Any] = dict(record or {})
    merged["schema"] = RECORD_SCHEMA
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    if applied or not record:
        merged["recorded_at"] = env.now()
        merged["authorised_by"] = options.authorised_by
    if applied:
        # A setup that changed something is an install again: an earlier
        # uninstall's marker would otherwise read as "not installed" forever,
        # and the readiness remedy (run setup) could not clear it (round 1, R1-3).
        merged.pop("uninstalled_at", None)
    merged["paired"] = bool(outcome.paired)
    merged.setdefault("mode", "headless (--headless=new); xvfb not needed")
    merged.setdefault("sandbox", "on; no --no-sandbox, no remote-debugging port")
    prior = merged.get("history")
    history: list[Any] = list(prior) if isinstance(prior, list) else []
    history.append(
        {
            "at": env.now(),
            "mode": outcome.mode,
            "authorised_by": options.authorised_by,
            "applied": applied,
            "ok": outcome.ok,
        }
    )
    merged["history"] = history[-50:]
    _write_private(env.record_path, json.dumps(merged, indent=2) + "\n")


def _owns_install(record: Mapping[str, Any] | None) -> bool:
    """Whether the record proves lop installed a Chrome here that is still there.

    ``chrome`` is written by ``apply_chrome`` only; ``uninstalled_at`` by a
    SUCCESSFUL uninstall only (a partial one can be re-run to finish).
    """
    if not isinstance(record, Mapping) or record.get("uninstalled_at"):
        return False
    chrome = record.get("chrome")
    return isinstance(chrome, Mapping) and bool(chrome)


def uninstall(
    options: Options, env: NodeEnv | None = None, *, on_step: OnStep | None = None
) -> Outcome:
    """Remove what setup installed that would keep running or updating.

    Removes the Chrome unit, the policy, the version lock, the package and the
    repo file. KEEPS the extension signing key (a reinstall must get the same
    extension id), the imported Google key and the fonts (inert, shared), the
    bridge daemon, and the record (marked uninstalled, for the audit trail).

    OWNERSHIP FIRST (round 1, R1-2): the presence probes below match ANY
    ``google-chrome-stable`` and ANY ``google-chrome.repo``, so without the
    same host checks as setup and a record proving lop installed it, this
    removed a person's own Chrome on a Fedora desktop. It now refuses there.
    """
    env = env or default_env()
    record = read_record(env)
    outcome = Outcome(mode="uninstall_dry_run" if options.dry_run else "uninstall")
    refusal = _host_refusal(env)
    if refusal is None and not _owns_install(record):
        refusal = _refusal(
            REFUSE_NOT_INSTALLED,
            f"{env.record_path} records no node-managed browser installed by lop, so there is "
            "nothing of lop's to remove; a browser installed some other way is not lop's to remove",
            "nothing to do",
            path=str(env.record_path),
        )
    if refusal is None and not options.dry_run and not options.authorised_by.strip():
        refusal = _refusal(
            REFUSE_AUTHORISATION,
            "removing the node-managed browser needs the operator's authorisation, "
            "recorded verbatim",
            f"re-run with --authorised-by {AUTHORISED_BY_HINT}",
        )
    if refusal is not None:
        outcome.ok = False
        outcome.refusal = refusal
        return outcome
    plan = _Plan(env, options, record)

    def _remove_unit() -> None:
        plan._systemctl("disable", "--now", CHROME_UNIT)  # noqa: SLF001 - same module
        env.unit_path.unlink(missing_ok=True)
        plan._systemctl("daemon-reload")  # noqa: SLF001 - same module

    def _sudo_ok(*argv: str) -> None:
        plan._require(env.sudo(*argv), " ".join(argv))  # noqa: SLF001 - same module

    def _locked() -> bool:
        try:
            listed = env.run("dnf", "versionlock", "list")
        except (OSError, subprocess.SubprocessError):
            return False
        return listed.returncode == 0 and f"{CHROME_PACKAGE}-" in listed.stdout

    # Order matters: stop Chrome before its package goes, drop the policy that
    # names the CRX, and unlock before `dnf remove` (a lock does not block a
    # removal, but a stale lock entry would pin a later reinstall).
    actions: list[tuple[str, bool, Callable[[], bool], Callable[[], None]]] = [
        ("chrome_unit", False, lambda: env.unit_path.exists(), _remove_unit),
        (
            "policy",
            True,
            lambda: POLICY_PATH.exists(),
            lambda: _sudo_ok("rm", "-f", str(POLICY_PATH)),
        ),
        (
            "versionlock",
            True,
            _locked,
            lambda: _sudo_ok("dnf", "versionlock", "delete", CHROME_PACKAGE),
        ),
        (
            "chrome",
            True,
            lambda: env.run("rpm", "-q", CHROME_PACKAGE).returncode == 0,
            lambda: _sudo_ok("dnf", "-y", "remove", CHROME_PACKAGE),
        ),
        ("repo", True, lambda: REPO_PATH.exists(), lambda: _sudo_ok("rm", "-f", str(REPO_PATH))),
    ]
    pending = list(actions)
    needs_root = [name for name, privileged, present, _r in pending if privileged and present()]
    sudo_blocked = bool(needs_root) and env.sudo("true").returncode != 0
    if sudo_blocked and not options.dry_run:
        outcome.ok = False
        outcome.refusal = _refusal(
            REFUSE_SUDO,
            f"{', '.join(needs_root)} need root to remove and `sudo -n` would prompt",
            "run it from an account with non-interactive sudo; at an interactive terminal, run "
            "`sudo -v` first and then this command again in the same terminal",
            steps=", ".join(needs_root),
        )
        return outcome
    for name, privileged, present, remove in pending:
        if not present():
            _emit(outcome, StepResult(name, privileged, STATE_IN_PLACE, "already absent"), on_step)
            continue
        if options.dry_run:
            # The dry-run predicts the sudo refusal, as setup's does (U5a).
            if sudo_blocked and privileged:
                outcome.ok = False
                result = StepResult(
                    name, privileged, STATE_WOULD_FAIL, "needs root and `sudo -n` would prompt"
                )
            else:
                result = StepResult(name, privileged, STATE_WOULD_APPLY, "would remove")
            _emit(outcome, result, on_step)
            continue
        try:
            remove()
        except (StepError, OSError) as error:
            outcome.ok = False
            _emit(outcome, StepResult(name, privileged, STATE_FAILED, str(error)), on_step)
            continue
        _emit(outcome, StepResult(name, privileged, STATE_APPLIED, "removed"), on_step)
    if not options.dry_run and record is not None:
        record = dict(record)
        if outcome.ok:
            # Only a COMPLETE removal ends lop's ownership: a partial one keeps
            # the record live so the same command can finish the job.
            record["uninstalled_at"] = env.now()
        record["paired"] = False
        prior = record.get("history")
        history: list[Any] = list(prior) if isinstance(prior, list) else []
        history.append(
            {
                "at": env.now(),
                "mode": "uninstall",
                "authorised_by": options.authorised_by,
                "ok": outcome.ok,
            }
        )
        record["history"] = history[-50:]
        _write_private(env.record_path, json.dumps(record, indent=2) + "\n")
        outcome.record_path = str(env.record_path)
    return outcome


# --- the readiness/prompt facts ----------------------------------------------------


def is_headless_node() -> bool:
    """Whether THIS host is a headless mesh node (the prompt's node branch).

    Cheap and file-first: Linux, a mesh member, no desktop discovery record and
    no graphical session. Never raises — prompt rendering must not break.
    """
    try:
        if sys.platform != "linux":
            return False
        if _user_machine_reason():
            return False
        return _mesh_member()
    except Exception:  # noqa: BLE001
        return False


@functools.lru_cache(maxsize=1)
def headless_node_once() -> bool:
    """:func:`is_headless_node`, read ONCE per process.

    For the browser tool's approval copy (round 1, U1): its probe runs
    ``loginctl`` and reads mesh state, too slow to repeat per tool call, and
    node-ness does not change under a running process. Tests monkeypatch this.
    """
    return is_headless_node()


def browser_fact(env: NodeEnv | None = None) -> dict[str, Any]:
    """This device's browser hosting, as facts for ``lop network ready``.

    ``host``: ``user`` (a desktop app or a user's extension serves this device —
    the node-managed browser does not apply), ``node`` (the node-managed browser
    is recorded here) or ``none``. For ``node``: whether the Chrome unit is
    active, the daemon healthy, and the extension paired. Local reads only (a
    systemctl query and the daemon's /health), well inside the readiness op's
    5 s budget.
    """
    env = env or default_env()
    try:
        if env.system() != "linux":
            # The node-managed browser is Linux-only (setup refuses elsewhere),
            # and a macOS/Windows device is a user's machine in practice: its
            # remedy is the extension or the desktop app, never this command.
            return {"host": "unsupported_platform", "platform": env.system()}
        if env.user_machine_reason() and not env.record_path.exists():
            return {"host": "user"}
        record = read_record(env)
        status = dict(env.daemon_status())
        if record is None or record.get("uninstalled_at"):
            # A paired extension without a node record is a user's browser
            # (an SSH'd-into workstation, say): not this feature's concern.
            if status.get("paired"):
                return {"host": "user"}
            # A node setup would refuse (aarch64, Debian) is n/a, not "not
            # installed": that row's remedy is the very command that refuses
            # (round 1, R1-6). Same checks as setup, minus the user-machine
            # one already answered above.
            refusal = _host_refusal(env, user_check=False)
            if refusal is not None and refusal["code"] == REFUSE_PLATFORM:
                return {"host": "unsupported_platform", "platform": f"linux {env.machine()}"}
            if refusal is not None:
                return {"host": "unsupported_distro"}
            return {"host": "none"}
        chrome_value = record.get("chrome")
        extension_value = record.get("extension")
        chrome: Mapping[str, Any] = chrome_value if isinstance(chrome_value, Mapping) else {}
        extension: Mapping[str, Any] = (
            extension_value if isinstance(extension_value, Mapping) else {}
        )
        active = env.run("systemctl", "--user", "is-active", CHROME_UNIT).stdout.strip()
        extension_id = str(extension.get("id") or "")
        paired = _live_paired(status, extension_id)
        connected = False
        health = status.get("health")
        if isinstance(health, Mapping):
            connected = bool(health.get("extension_connected") or health.get("connected"))
        return {
            "host": "node",
            "chrome_version": str(chrome.get("version") or ""),
            "extension_version": str(extension.get("version") or ""),
            "unit_active": active == "active",
            "unit_state": active,
            "daemon_healthy": bool(status.get("healthy")),
            "paired": paired,
            "connected": connected,
        }
    except Exception as error:  # noqa: BLE001 - a fact that cannot be read says so
        return {"host": "unknown", "error": str(error)[:200]}
