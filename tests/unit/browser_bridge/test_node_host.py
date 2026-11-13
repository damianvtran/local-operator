"""`lop browser setup --node`: plan → apply → status against a FAKE node.

No real dnf, rpm, sudo, systemd, Chrome or network: :class:`FakeNode` emulates
the handful of commands node_host issues (an rpm database, ``install``/``rm``
for root-owned files, the version lock, ``systemctl --user``, Chrome's
``--pack-extension``) and the daemon's pairing answers, so every branch —
idempotency, each refusal, the pairing delivery and its cleanup, the
extension-reload cycle — runs in milliseconds and is deterministic. The real
path on a node is the PR's testing evidence, not this file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from local_operator.browser_bridge import node_host

EXTENSION_VERSION = "0.1.27"


def _pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


PEM = _pem()


def _done(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class FakeNode:
    """A small, stateful stand-in for an AL2023 node."""

    def __init__(self, tmp: Path, *, provisioned: bool) -> None:
        self.tmp = tmp
        self.calls: list[list[str]] = []
        self.sudo_ok = True
        self.rpms: set[str] = set()
        self.locked = False
        self.unit_enabled = False
        self.unit_active = False
        self.paired_ids: set[str] = set()
        self.link_up = False
        self.link_version = ""
        self.pairs_on_code = True
        self.codes_minted = 0
        self.policy_writes: list[dict[str, Any]] = []
        self.policy_path = tmp / "etc" / "policies" / "local-operator.json"
        self.repo_path = tmp / "etc" / "google-chrome.repo"
        self.env = node_host.NodeEnv(
            config_root=tmp / "home" / ".local-operator",
            home=tmp / "home",
            user="ec2-user",
            runner=self.run,
            fetch=self.fetch,
            system=lambda: "linux",
            machine=lambda: "x86_64",
            which=lambda name: f"/usr/bin/{name}",
            user_machine_reason=lambda: "",
            mesh_member=lambda: True,
            daemon_status=self.daemon_status,
            install_daemon=self._install_daemon,
            enable_linger=self._enable_linger,
            request_codes=self.request_codes,
            sleep=lambda _s: self._tick(),
            monotonic=lambda: self.clock,
            now=lambda: "2026-11-13T00:00:00Z",
        )
        self.clock = 0.0
        self.linger = False
        self.daemon_installed = False
        self.extension_dist = tmp / "dist"
        self.extension_dist.mkdir()
        (self.extension_dist / "manifest.json").write_text(
            json.dumps({"version": EXTENSION_VERSION}), encoding="utf-8"
        )
        (self.extension_dist / "worker.js").write_text("// worker", encoding="utf-8")
        if provisioned:
            self._provision()

    def _install_daemon(self) -> dict[str, Any]:
        self.daemon_installed = True
        return {"ok": True}

    def _enable_linger(self) -> bool:
        self.linger = True
        return True

    # -- the clock: each poll advances one second and lets Chrome react --
    def _tick(self) -> None:
        self.clock += 1.0
        self._chrome_reacts()

    def _installed_dir(self) -> Path:
        extension_id = node_host.extension_id_for_key(self.env.signing_key.read_bytes())
        return self.env.profile_dir / "Default" / "Extensions" / extension_id

    def _chrome_reacts(self) -> None:
        """Chrome's policy watcher: delisted → uninstall; listed → install update.xml's offer."""
        if not self.env.signing_key.is_file() or not self.policy_path.is_file():
            return
        policy = json.loads(self.policy_path.read_text(encoding="utf-8"))
        installed = self._installed_dir()
        if not policy.get("ExtensionInstallForcelist"):
            shutil.rmtree(installed, ignore_errors=True)
            self.link_up = False
            return
        if not installed.exists() and self.env.update_manifest.is_file():
            offered = self.env.update_manifest.read_text(encoding="utf-8").split("version='")[-1]
            version = offered.split("'")[0]
            (installed / f"{version}_0").mkdir(parents=True)
            self.link_up = True
            self.link_version = version
            # A fresh install has no token: the daemon still lists the id.
            self.fresh_install = True
        code = (policy.get("3rdparty") or {}).get("extensions", {})
        for extension_id, values in code.items():
            if values.get("pairingCode") and self.pairs_on_code:
                self.paired_ids.add(extension_id)
                self.fresh_install = False

    def _provision(self) -> None:
        self.rpms = {
            node_host.SIGNING_KEY_RPM + "-570c8cd3",
            node_host.VERSIONLOCK_PLUGIN + "-4.3.0",
            node_host.CHROME_PACKAGE + "-157.0.8103.78-1.x86_64",
            *(name + "-1.0" for name in node_host.FONT_PACKAGES),
        }
        self.locked = True
        self.linger = True
        self.daemon_installed = True
        self.repo_path.parent.mkdir(parents=True, exist_ok=True)
        self.repo_path.write_text(node_host.render_repo(), encoding="utf-8")
        node_host._write_private(self.env.signing_key, PEM)  # noqa: SLF001
        options = node_host.Options(
            extension_dir=self.extension_dist, authorised_by="test", repair=False
        )
        plan = node_host._Plan(self.env, options, None)  # noqa: SLF001
        # Pack + manifest + policy + profile + unit through the real code.
        plan.apply_extension()
        plan.apply_update_manifest()
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        plan.apply_policy()
        plan.apply_profile()
        plan.apply_unit()
        self._chrome_reacts()
        extension_id = node_host.extension_id_for_key(PEM.encode())
        self.paired_ids.add(extension_id)
        self.fresh_install = False
        record = {"schema": 1, "authorised_by": "test", **plan.updates}
        node_host._write_private(self.env.record_path, json.dumps(record))  # noqa: SLF001
        self.calls.clear()
        self.policy_writes.clear()

    # -- the runner --
    def run(
        self, argv: Sequence[str], *, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] == ["sudo", "-n"]:
            if not self.sudo_ok:
                return _done(returncode=1, stderr="sudo: a password is required")
            argv = argv[2:]
            if argv == ["true"]:
                return _done()
        head = argv[0]
        if head == "rpm" and argv[1] == "-q":
            name = argv[2]
            found = sorted(r for r in self.rpms if r.startswith(name))
            return _done(found[0] + "\n") if found else _done(returncode=1)
        if head == "rpm" and argv[1] == "--import":
            self.rpms.add(node_host.SIGNING_KEY_RPM + "-570c8cd3")
            return _done()
        if head == "rpm" and argv[1] == "-K":
            return _done(f"{argv[2]}: digests signatures OK\n")
        if head == "gpg":
            return _done(
                "pub:-:4096:1:X:1:::-:::scESC::::::23::0:\n"
                f"fpr:::::::::{node_host.SIGNING_KEY_FINGERPRINT}:\n"
            )
        if head == "dnf":
            return self._dnf(argv[1:])
        if head == "mkdir":
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            return _done()
        if head == "install":
            source, target = Path(argv[-2]), Path(argv[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if target == self.policy_path:
                self.policy_writes.append(
                    {
                        "mode": argv[argv.index("-m") + 1],
                        "group": argv[argv.index("-g") + 1],
                        "content": json.loads(target.read_text(encoding="utf-8")),
                    }
                )
            return _done()
        if head == "rm":
            Path(argv[-1]).unlink(missing_ok=True)
            return _done()
        if head == "id":
            return _done("ec2-user\n")
        if head == "loginctl":
            return _done("Linger=yes\n" if self.linger else "Linger=no\n")
        if head == "systemctl":
            return self._systemctl(argv[2:])
        if head == node_host.CHROME_BINARY and any(a.startswith("--pack-extension=") for a in argv):
            stage = Path(
                next(a for a in argv if a.startswith("--pack-extension=")).split("=", 1)[1]
            )
            stage.with_suffix(".crx").write_bytes(
                b"Cr24" + stage.joinpath("manifest.json").read_bytes()
            )
            if not any(a.startswith("--pack-extension-key=") for a in argv):
                stage.with_suffix(".pem").write_text(PEM, encoding="utf-8")
            return _done()
        if head == node_host.CHROME_BINARY and argv[1:] == ["--version"]:
            return _done("Google Chrome 157.0.8103.78\n")
        raise AssertionError(f"unexpected command {argv}")

    def _dnf(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["versionlock", "list"]:
            return _done("google-chrome-stable-0:157.0.8103.78-1.*\n" if self.locked else "")
        if args[:2] == ["versionlock", "add"]:
            self.locked = True
            return _done()
        if args[:2] == ["versionlock", "delete"]:
            self.locked = False
            return _done()
        if args[:2] == ["-y", "download"]:
            destdir = Path(args[args.index("--destdir") + 1])
            (destdir / "google-chrome-stable-157.0.8103.78-1.x86_64.rpm").write_bytes(b"rpm")
            return _done()
        if args[:2] == ["-y", "install"]:
            for name in args[2:]:
                if name.endswith(".rpm"):
                    self.rpms.add(node_host.CHROME_PACKAGE + "-157.0.8103.78-1.x86_64")
                else:
                    self.rpms.add(name + "-1.0")
            return _done()
        if args[:2] == ["-y", "remove"]:
            self.rpms = {r for r in self.rpms if not r.startswith(args[2])}
            return _done()
        raise AssertionError(f"unexpected dnf {args}")

    def _systemctl(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        verb = args[0]
        if verb == "is-enabled":
            return _done("enabled\n" if self.unit_enabled else "disabled\n")
        if verb == "is-active":
            return _done("active\n" if self.unit_active else "inactive\n")
        if verb == "enable":
            self.unit_enabled = True
        elif verb in ("restart", "start"):
            self.unit_active = True
        elif verb == "disable":
            self.unit_enabled = False
            self.unit_active = False
        return _done()

    def fetch(self, url: str) -> bytes:
        if url == node_host.SIGNING_KEY_URL:
            return b"-----BEGIN PGP PUBLIC KEY BLOCK-----"
        raise AssertionError(f"unexpected fetch {url}")

    # -- the daemon --
    def daemon_status(self) -> Mapping[str, Any]:
        extension_id = (
            node_host.extension_id_for_key(self.env.signing_key.read_bytes())
            if self.env.signing_key.is_file()
            else ""
        )
        live = self.link_up and self.unit_active
        authorised = extension_id in self.paired_ids and not getattr(self, "fresh_install", False)
        return {
            "installed": self.daemon_installed,
            "healthy": self.daemon_installed,
            "port": 4099,
            "identities": [{"extension_id": i} for i in sorted(self.paired_ids)],
            "health": {
                "extension_connected": live,
                "extension_version": self.link_version if live else "",
                "driver_extension_id": extension_id if live else "",
                "paired": live and authorised,
            },
        }

    def request_codes(self) -> Mapping[str, Any]:
        self.codes_minted += 1
        extension_id = node_host.extension_id_for_key(self.env.signing_key.read_bytes())
        if not (self.link_up and self.unit_active):
            return {"ok": True, "codes": [], "connected_count": 0}
        return {
            "ok": True,
            "codes": [{"extension_id": extension_id, "code": "424242"}],
            "connected_count": 1,
        }


MUTATING = ("install", "remove", "download", "--import", "add", "delete", "restart", "enable", "rm")


def _mutations(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if any(word in call for word in MUTATING)]


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        node_host, "POLICY_PATH", tmp_path / "etc" / "policies" / "local-operator.json"
    )
    monkeypatch.setattr(node_host, "REPO_PATH", tmp_path / "etc" / "google-chrome.repo")
    return tmp_path


def _states(outcome: node_host.Outcome) -> dict[str, str]:
    return {step.id: step.state for step in outcome.steps}


def test_provisioned_node_is_a_no_op_in_every_mode(paths: Path) -> None:
    node = FakeNode(paths, provisioned=True)
    dry = node_host.setup(node_host.Options(dry_run=True), node.env)
    assert dry.ok and dry.refusal is None
    assert set(_states(dry).values()) == {node_host.STATE_IN_PLACE}, _states(dry)
    assert dry.paired is True
    applied = node_host.setup(
        node_host.Options(repair=True, authorised_by="Operator (test)"), node.env
    )
    assert applied.ok
    assert set(_states(applied).values()) == {node_host.STATE_IN_PLACE}
    # Idempotent means NOTHING mutating ran: no sudo write, no restart, no dnf.
    assert _mutations(node.calls) == []
    record = json.loads(node.env.record_path.read_text(encoding="utf-8"))
    assert record["history"][-1]["applied"] == []
    assert record["history"][-1]["authorised_by"] == "Operator (test)"
    assert (node.env.record_path.stat().st_mode & 0o777) == 0o600


def test_every_step_id_is_reported_in_order(paths: Path) -> None:
    node = FakeNode(paths, provisioned=True)
    ids = [step.id for step in node_host.setup(node_host.Options(dry_run=True), node.env).steps]
    assert ids == [
        "signing_key",
        "repo",
        "versionlock_plugin",
        "chrome",
        "versionlock",
        "fonts",
        "bridge_daemon",
        "linger",
        "extension",
        "update_manifest",
        "policy",
        "profile_dir",
        "chrome_unit",
        "extension_loaded",
        "pairing",
    ]


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"system": lambda: "darwin"}, node_host.REFUSE_PLATFORM),
        ({"machine": lambda: "aarch64"}, node_host.REFUSE_PLATFORM),
        (
            {"which": lambda name: None if name == "dnf" else f"/usr/bin/{name}"},
            node_host.REFUSE_DISTRO,
        ),
        (
            {"user_machine_reason": lambda: "a graphical session is attached ($DISPLAY is set)"},
            node_host.REFUSE_USER_MACHINE,
        ),
        ({"mesh_member": lambda: False}, node_host.REFUSE_NOT_MEMBER),
    ],
)
def test_refusals_change_nothing(paths: Path, override: dict[str, Any], code: str) -> None:
    node = FakeNode(paths, provisioned=False)
    for name, value in override.items():
        setattr(node.env, name, value)
    outcome = node_host.setup(node_host.Options(authorised_by="Operator (test)"), node.env)
    assert not outcome.ok
    assert outcome.refusal is not None and outcome.refusal["code"] == code
    assert outcome.refusal["remedy"]
    assert outcome.steps == []
    assert node.calls == []


def test_apply_needs_a_recorded_authorisation(paths: Path) -> None:
    node = FakeNode(paths, provisioned=False)
    outcome = node_host.setup(node_host.Options(), node.env)
    assert outcome.refusal is not None and outcome.refusal["code"] == node_host.REFUSE_AUTHORISATION
    assert _mutations(node.calls) == []
    # A dry run needs none: it changes nothing.
    dry = node_host.setup(
        node_host.Options(dry_run=True, extension_dir=node.extension_dist), node.env
    )
    assert dry.refusal is None


def test_repair_with_nothing_recorded_is_refused(paths: Path) -> None:
    node = FakeNode(paths, provisioned=False)
    outcome = node_host.setup(node_host.Options(repair=True, authorised_by="x"), node.env)
    assert outcome.refusal is not None and outcome.refusal["code"] == node_host.REFUSE_NO_RECORD
    assert outcome.refusal["args"]["path"] == str(node.env.record_path)


def test_sudo_that_would_prompt_is_refused_before_any_privileged_step(paths: Path) -> None:
    node = FakeNode(paths, provisioned=False)
    node.sudo_ok = False
    outcome = node_host.setup(
        node_host.Options(extension_dir=node.extension_dist, authorised_by="x"), node.env
    )
    assert outcome.refusal is not None and outcome.refusal["code"] == node_host.REFUSE_SUDO
    assert "signing_key" in outcome.refusal["args"]["steps"]
    # Only the probe ran under sudo; it never prompts (-n).
    assert [c for c in node.calls if c[:1] == ["sudo"]] == [["sudo", "-n", "true"]]


def test_fresh_node_installs_pairs_and_records(paths: Path) -> None:
    node = FakeNode(paths, provisioned=False)
    node_host._write_private(node.env.signing_key, PEM)  # noqa: SLF001 - a persistent key
    outcome = node_host.setup(
        node_host.Options(extension_dir=node.extension_dist, authorised_by="Operator (test)"),
        node.env,
    )
    assert outcome.ok, [s.as_json() for s in outcome.steps]
    states = _states(outcome)
    assert states["chrome"] == node_host.STATE_APPLIED
    assert states["pairing"] == node_host.STATE_APPLIED
    assert outcome.paired is True
    # The package was verified before it was installed, from the downloaded file.
    order = [" ".join(c) for c in node.calls]
    k = next(i for i, c in enumerate(order) if c.startswith("rpm -K"))
    i = next(i for i, c in enumerate(order) if "dnf -y install" in c and c.endswith(".rpm"))
    assert k < i
    # The code went in as 0640 root:<user's group>, and came OUT again.
    secret = [w for w in node.policy_writes if "3rdparty" in w["content"]]
    assert secret and secret[0]["mode"] == "0640" and secret[0]["group"] == "ec2-user"
    assert secret[0]["content"]["3rdparty"]["extensions"]
    final = json.loads(node.policy_path.read_text(encoding="utf-8"))
    assert "3rdparty" not in final and final["ExtensionInstallForcelist"]
    record = json.loads(node.env.record_path.read_text(encoding="utf-8"))
    assert record["authorised_by"] == "Operator (test)"
    assert record["paired"] is True
    assert record["extension"]["version"] == EXTENSION_VERSION
    assert record["chrome"]["rpm_K"] == "digests signatures OK"
    # A second run is a no-op.
    node.calls.clear()
    again = node_host.setup(node_host.Options(authorised_by="Operator (test)"), node.env)
    assert set(_states(again).values()) == {node_host.STATE_IN_PLACE}, [
        s.as_json() for s in again.steps if s.state != node_host.STATE_IN_PLACE
    ]
    assert _mutations(node.calls) == []


def test_pairing_code_is_removed_from_policy_even_when_pairing_times_out(paths: Path) -> None:
    node = FakeNode(paths, provisioned=True)
    node.paired_ids.clear()
    node.pairs_on_code = False
    outcome = node_host.setup(node_host.Options(repair=True, authorised_by="x"), node.env)
    assert _states(outcome)["pairing"] == node_host.STATE_FAILED
    assert f"{node_host.PAIR_WAIT_S:.0f}s" in outcome.steps[-1].detail
    final = json.loads(node.policy_path.read_text(encoding="utf-8"))
    assert "3rdparty" not in final
    assert any("3rdparty" in w["content"] for w in node.policy_writes)


def test_a_new_extension_version_cycles_the_forcelist_and_re_pairs(paths: Path) -> None:
    node = FakeNode(paths, provisioned=True)
    (node.extension_dist / "manifest.json").write_text(
        json.dumps({"version": "0.1.28"}), encoding="utf-8"
    )
    outcome = node_host.setup(
        node_host.Options(repair=True, extension_dir=node.extension_dist, authorised_by="x"),
        node.env,
    )
    states = _states(outcome)
    assert outcome.ok, [s.as_json() for s in outcome.steps]
    assert states["extension"] == node_host.STATE_APPLIED
    assert states["extension_loaded"] == node_host.STATE_APPLIED
    # The reinstall wiped the token; the LIVE link said unpaired, so the code
    # was delivered again rather than trusting the saved identity.
    assert states["pairing"] == node_host.STATE_APPLIED
    delisted = [
        w for w in node.policy_writes if w["content"].get("ExtensionInstallForcelist") == []
    ]
    assert delisted, "the forcelist entry was removed so Chrome uninstalled the old CRX"
    assert json.loads(node.policy_path.read_text(encoding="utf-8"))["ExtensionInstallForcelist"]


def test_an_extension_without_managed_pairing_is_refused(paths: Path) -> None:
    node = FakeNode(paths, provisioned=True)
    (node.extension_dist / "manifest.json").write_text(
        json.dumps({"version": "0.1.26"}), encoding="utf-8"
    )
    outcome = node_host.setup(
        node_host.Options(repair=True, extension_dir=node.extension_dist, authorised_by="x"),
        node.env,
    )
    assert _states(outcome)["extension"] == node_host.STATE_FAILED
    assert "predates managed pairing" in outcome.steps[8].detail


def test_release_asset_is_refused_when_its_digest_does_not_match(paths: Path) -> None:
    node = FakeNode(paths, provisioned=False)
    listing = [
        {
            "tag_name": "v9",
            "draft": False,
            "prerelease": False,
            "assets": [
                {
                    "name": "local-operator-extension-v0.1.27.zip",
                    "digest": "sha256:" + hashlib.sha256(b"published").hexdigest(),
                    "browser_download_url": "https://example.invalid/ext.zip",
                }
            ],
        },
    ]

    def fetch(url: str) -> bytes:
        return json.dumps(listing).encode() if "api.github.com" in url else b"tampered"

    node.env.fetch = fetch
    plan = node_host._Plan(node.env, node_host.Options(), None)  # noqa: SLF001
    with pytest.raises(node_host.StepError, match="does not match its published digest"):
        plan._desired_extension()  # noqa: SLF001


def test_signing_key_with_the_wrong_fingerprint_is_not_imported(paths: Path) -> None:
    node = FakeNode(paths, provisioned=False)
    original = node.run

    def run(
        argv: Sequence[str], *, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        if list(argv)[:1] == ["gpg"]:
            return _done("fpr:::::::::0000000000000000000000000000000000000000:\n")
        return original(argv, input_text=input_text)

    node.env.runner = run
    plan = node_host._Plan(node.env, node_host.Options(), None)  # noqa: SLF001
    with pytest.raises(node_host.StepError, match="does not carry the expected key"):
        plan.apply_key()
    assert not any("--import" in c for c in node.calls)


def test_uninstall_removes_what_keeps_running_and_keeps_the_key(paths: Path) -> None:
    node = FakeNode(paths, provisioned=True)
    dry = node_host.uninstall(node_host.Options(dry_run=True), node.env)
    assert {s.state for s in dry.steps} == {node_host.STATE_WOULD_APPLY}
    assert _mutations(node.calls) == []
    refused = node_host.uninstall(node_host.Options(), node.env)
    assert refused.refusal is not None and refused.refusal["code"] == node_host.REFUSE_AUTHORISATION
    done = node_host.uninstall(node_host.Options(authorised_by="Operator (test)"), node.env)
    assert done.ok and {s.state for s in done.steps} == {node_host.STATE_APPLIED}
    assert not node.policy_path.exists() and not node.locked
    assert not any(r.startswith(node_host.CHROME_PACKAGE) for r in node.rpms)
    assert node.env.signing_key.is_file()
    record = json.loads(node.env.record_path.read_text(encoding="utf-8"))
    assert record["uninstalled_at"] and record["paired"] is False


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"identities": [{"extension_id": "abc"}], "health": None}, True),
        (
            {
                "identities": [{"extension_id": "abc"}],
                "health": {
                    "extension_connected": True,
                    "driver_extension_id": "abc",
                    "paired": False,
                    "authorized_extension_ids": ["abc"],
                },
            },
            False,
        ),
        (
            {
                "identities": [{"extension_id": "abc"}],
                "health": {
                    "extension_connected": True,
                    "driver_extension_id": "abc",
                    "paired": True,
                },
            },
            True,
        ),
        ({"identities": [], "health": {"extension_connected": True, "paired": True}}, False),
    ],
)
def test_live_paired_trusts_the_link_over_the_saved_identity(
    status: dict[str, Any], expected: bool
) -> None:
    """Measured on the node: after a reinstall /health listed the id in
    ``authorized_extension_ids`` while ``paired`` was false."""
    assert node_host._live_paired(status, "abc") is expected  # noqa: SLF001


def test_browser_fact_states(paths: Path) -> None:
    node = FakeNode(paths, provisioned=True)
    fact = node_host.browser_fact(node.env)
    assert fact["host"] == "node" and fact["paired"] and fact["unit_active"]
    node.unit_active = False
    assert node_host.browser_fact(node.env)["unit_active"] is False
    bare = FakeNode(paths / "bare", provisioned=False) if (paths / "bare").mkdir() is None else None
    assert bare is not None
    assert node_host.browser_fact(bare.env) == {"host": "none"}
    bare.env.user_machine_reason = lambda: "desktop record"
    assert node_host.browser_fact(bare.env) == {"host": "user"}
    bare.env.system = lambda: "darwin"
    assert node_host.browser_fact(bare.env)["host"] == "unsupported_platform"


def test_extension_id_matches_chromes_derivation() -> None:
    """Chrome's id: sha256(DER SubjectPublicKeyInfo)[:16 bytes], nibbles → a-p."""
    key = serialization.load_pem_private_key(PEM.encode(), password=None)
    der = key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    expected = "".join("abcdefghijklmnop"[int(c, 16)] for c in hashlib.sha256(der).hexdigest()[:32])
    assert node_host.extension_id_for_key(PEM.encode()) == expected
    assert len(expected) == 32


# --- the CLI surface -------------------------------------------------------------


def _cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]):
    from local_operator import cli

    monkeypatch.setattr("sys.argv", ["lop", "browser", *argv])
    code = cli.main()
    return code, capsys.readouterr().out


def test_cli_renders_a_refusal_through_keyed_copy_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], paths: Path
) -> None:
    node = FakeNode(paths, provisioned=False)
    monkeypatch.setattr(node_host, "default_env", lambda: node.env)
    code, out = _cli(monkeypatch, capsys, ["setup", "--node", "--repair"])
    assert code == 1
    assert out.startswith("Refused: there is nothing to repair, because ")
    assert str(node.env.record_path) in out


def test_cli_json_carries_every_step(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], paths: Path
) -> None:
    node = FakeNode(paths, provisioned=True)
    monkeypatch.setattr(node_host, "default_env", lambda: node.env)
    code, out = _cli(monkeypatch, capsys, ["setup", "--node", "--dry-run", "--json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["mode"] == "dry_run" and payload["paired"] is True
    assert {step["state"] for step in payload["steps"]} == {node_host.STATE_IN_PLACE}


def test_cli_human_dry_run_lists_steps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], paths: Path
) -> None:
    node = FakeNode(paths, provisioned=True)
    monkeypatch.setattr(node_host, "default_env", lambda: node.env)
    monkeypatch.setattr(node_host, "node_name", lambda: "node-x")
    code, out = _cli(monkeypatch, capsys, ["setup", "--node", "--dry-run"])
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "Node browser on node-x — dry run, nothing was changed:"
    assert "  chrome: already in place — " in out
    assert lines[-1].startswith("Paired: yes.")


def test_cli_bare_setup_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out = _cli(monkeypatch, capsys, ["setup"])
    assert code == 2
