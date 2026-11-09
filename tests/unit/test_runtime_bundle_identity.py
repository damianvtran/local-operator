"""The runtime's macOS privacy identity: bundle-wrapped images and their two flavors.

WHAT IS AT RISK, and why each test exists:

1. **The ad-hoc flavor must NEVER carry the app's bundle id.** An ad-hoc image
   claiming ``com.local-operator`` would collide with the desktop app's TCC row,
   and a csreq mismatch can re-prompt and overwrite the APP's grant.
2. **No keychain on the session path.** Developer ID signing can raise the
   keychain dialog — and under a scratch ``HOME`` macOS instead asks to CREATE a
   login keychain (the 2026-09-18 rig incident). The team flavor is built only
   behind every guard; these tests record ``codesign``'s argv through a fake and
   assert that every ``--sign`` the session path issues is ``-``.
3. **Selection and migration.** A healthy team image wins; a venv link to the
   legacy ``<key>/bin`` image repoints on its next start; nothing published is
   deleted or rewritten (a running process may still be executing it).
4. **A failed bundle never fails a launch.** It falls back to today's shape and
   does not retry on every start.

macOS only: the real path clones the running interpreter (APFS clone, never a
write to it) and, where a test does not install the fake, runs the real ad-hoc
``codesign`` — which never reads a keychain.
"""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator import procname

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="bundle-keyed privacy identity is macOS-specific"
)

_FAKE_SHA1 = "0123456789ABCDEF0123456789ABCDEF01234567"
_FAKE_TEAM = "ABCDE12345"


@pytest.fixture(autouse=True)
def stable_runtime(tmp_path, monkeypatch):
    """Every plant lands under ``tmp_path``; never the operator's real runtime root."""
    root = tmp_path / "runtime"
    monkeypatch.setenv("LOCAL_OPERATOR_RUNTIME_ROOT", str(root))
    return root


def _copied_interpreter(tmp_path: Path) -> Path:
    """A throwaway interpreter tree (real bytes, real ``lib``) safe to re-key."""
    real = Path(os.path.realpath(sys.executable))
    base = tmp_path / "base"
    (base / "bin").mkdir(parents=True)
    (base / "lib").mkdir(parents=True)
    copied = base / "bin" / real.name
    shutil.copy2(real, copied)
    for entry in (real.parent.parent / "lib").iterdir():
        if entry.name.startswith("python") or entry.name.startswith("libpython"):
            os.symlink(entry, base / "lib" / entry.name)
    return copied


@pytest.fixture()
def fake_codesign(tmp_path, monkeypatch):
    """Replace ``codesign`` with a script that records each argv as one JSON line.

    It "signs" nothing and exits 0, unless ``fail-verify`` exists beside the log
    (then ``--verify`` exits 1) or ``fail-all`` exists (every call exits 1).
    Tests that use it never EXECUTE the produced image.
    """
    log = tmp_path / "codesign" / "argv.jsonl"
    log.parent.mkdir()
    script = tmp_path / "codesign" / "codesign"
    script.write_text(
        "#!/bin/sh\n"
        f'here="{log.parent}"\n'
        f'"{sys.executable}" -I -c \'import json,sys; '
        f'open(sys.argv[1], "a").write(json.dumps(sys.argv[2:]) + "\\n")\' "{log}" "$@"\n'
        '[ -e "$here/fail-all" ] && exit 1\n'
        'case " $* " in *" --verify "*) [ -e "$here/fail-verify" ] && exit 1;; esac\n'
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setattr(procname, "_CODESIGN", str(script))

    def calls() -> list[list[str]]:
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    calls.dir = log.parent  # type: ignore[attr-defined]
    return calls


def _signed_with(calls: list[list[str]]) -> list[str]:
    """Every value passed to ``--sign`` across the recorded calls."""
    return [argv[i + 1] for argv in calls for i, arg in enumerate(argv[:-1]) if arg == "--sign"]


@pytest.fixture()
def opted_in_host(team_host_but_isolated, monkeypatch):
    """Every team guard satisfied, without touching the real home, app or keychain."""
    monkeypatch.setattr(procname, "team_signing_refusal", lambda: None)
    return team_host_but_isolated


@pytest.fixture()
def team_host_but_isolated(tmp_path, monkeypatch):
    """Opt-in, app, team and requirement all valid; ONLY the isolation guard stands.

    So a test using it fails if that guard is removed (proved by mutation),
    rather than passing because some other precondition happened to be missing.
    """
    optin = tmp_path / "home" / ".local-operator" / procname.SIGNING_OPTIN_FILENAME
    monkeypatch.setattr(procname, "_optin_path", lambda: optin)
    app = tmp_path / "Local Operator.app"
    app.mkdir()
    monkeypatch.setattr(procname, "installed_app", lambda: app)
    monkeypatch.setattr(procname, "code_identity", lambda path: {"TeamIdentifier": _FAKE_TEAM})
    monkeypatch.setattr(
        procname,
        "app_requirement",
        lambda path: f'identifier "{procname.APP_BUNDLE_ID}" and anchor apple generic',
    )
    procname.write_signing_optin(_FAKE_SHA1, _FAKE_TEAM, always_allow_confirmed=True)
    return optin


class TestBundleShape:
    def test_the_adhoc_plant_is_a_strict_clean_bundle_with_the_runtime_id(self, tmp_path):
        real = _copied_interpreter(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")

        assert image is not None and procname._is_bundle_image(image)
        home = procname._image_home(image)
        assert home.name.endswith(f"-{procname.FLAVOR_ADHOC}")
        bundle = image.parents[2]
        assert bundle.name == procname.RUNTIME_BUNDLE_DIRNAME
        info = plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())
        assert info["CFBundleIdentifier"] == procname.RUNTIME_BUNDLE_ID
        assert info["CFBundleExecutable"] == procname.BRAND
        assert info["CFBundleShortVersionString"] == "0"
        assert info["LSBackgroundOnly"] is True and info["LSUIElement"] is True
        # ``lib`` lives beside the bundle; nothing out-of-bundle inside it.
        assert (home / "lib").is_symlink() and not (bundle / "Contents" / "lib").exists()
        shown = subprocess.run(
            ["/usr/bin/codesign", "-dv", str(bundle)], capture_output=True, text=True
        )
        assert f"Identifier={procname.RUNTIME_BUNDLE_ID}" in shown.stderr
        assert "Signature=adhoc" in shown.stderr
        strict = subprocess.run(
            ["/usr/bin/codesign", "--verify", "--strict", str(bundle)],
            capture_output=True,
            text=True,
        )
        assert strict.returncode == 0, strict.stderr
        sidecar = (home / procname._IMAGE_SIDECAR).read_text().split()
        assert sidecar[2:] == [procname.FLAVOR_ADHOC, procname.RUNTIME_BUNDLE_ID]

    def test_the_bundle_image_starts_and_keeps_the_venv_prefix(self, tmp_path):
        """20 launches (the warm-dyld trap): stdlib, ssl and sqlite3 through the new rpath."""
        real = _copied_interpreter(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")
        assert image is not None
        for launch in range(20):
            ran = subprocess.run(
                [str(image), "-I", "-c", "import encodings, ssl, sqlite3; print('ok')"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert ran.returncode == 0 and ran.stdout.strip() == "ok", (launch, ran.stderr)

    def test_the_adhoc_identifier_is_never_the_apps(self):
        assert procname._bundle_identifier(procname.FLAVOR_ADHOC) == procname.RUNTIME_BUNDLE_ID
        assert procname._bundle_identifier(procname.FLAVOR_ADHOC) != procname.APP_BUNDLE_ID
        assert procname._bundle_identifier("anything-else") != procname.APP_BUNDLE_ID
        assert procname._bundle_identifier(procname.FLAVOR_TEAM) == procname.APP_BUNDLE_ID

    def test_the_session_plant_signs_ad_hoc_only(self, tmp_path, fake_codesign):
        real = _copied_interpreter(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")

        assert image is not None
        calls = fake_codesign()
        assert calls == [
            [
                "--force",
                "--sign",
                "-",
                "--identifier",
                procname.RUNTIME_BUNDLE_ID,
                (
                    str(
                        procname._flavor_dir(
                            tmp_path / "runtime", procname._build_key(real), "adhoc"
                        ).resolve()
                        / procname.RUNTIME_BUNDLE_DIRNAME
                    )
                    if False
                    else calls[0][-1]
                ),
            ]
        ]
        assert calls[0][-1].endswith(
            f".tmp.{os.getpid()}.{procname._temp_thread_token()}/{procname.RUNTIME_BUNDLE_DIRNAME}"
        ), "signed in STAGING, before the publish rename"

    def test_a_missing_sidecar_is_unhealthy(self, tmp_path):
        real = _copied_interpreter(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")
        assert image is not None and procname._healthy_image(image, real)
        (procname._image_home(image) / procname._IMAGE_SIDECAR).unlink()
        assert not procname._healthy_image(image, real)

    def test_source_for_image_answers_at_bundle_depth(self, tmp_path):
        real = _copied_interpreter(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")
        assert image is not None
        assert procname._source_for_image(image) == Path(os.path.realpath(real))


class TestRpath:
    def test_the_rewrite_reaches_lib_beside_the_bundle(self, tmp_path):
        real = _copied_interpreter(tmp_path)
        clone = tmp_path / "clone"
        procname._clone_or_copy(real, clone)
        assert procname._rewrite_rpath(clone) is True
        shown = subprocess.run(["/usr/bin/otool", "-l", str(clone)], capture_output=True, text=True)
        assert "path @loader_path/../../../lib" in shown.stdout
        assert "@executable_path/../lib" not in shown.stdout
        assert Path(real).read_bytes() != clone.read_bytes(), "the clone changed..."
        assert (
            subprocess.run(
                ["/usr/bin/otool", "-l", str(real)], capture_output=True, text=True
            ).stdout.count("@executable_path/../lib")
            == 1
        ), "...and the interpreter it was cloned from did not"

    def test_an_unexpected_rpath_writes_nothing(self, tmp_path):
        real = _copied_interpreter(tmp_path)
        clone = tmp_path / "clone"
        procname._clone_or_copy(real, clone)
        assert procname._rewrite_rpath(clone) is True
        before = clone.read_bytes()
        assert (
            procname._rewrite_rpath(clone) is False
        ), "an rpath we did not expect is not guessed at"
        assert clone.read_bytes() == before

    def test_a_shared_inode_is_never_written(self, tmp_path):
        real = _copied_interpreter(tmp_path)
        second = tmp_path / "second-name"
        os.link(real, second)
        before = real.read_bytes()
        assert procname._rewrite_rpath(second) is False
        assert real.read_bytes() == before

    def test_an_rpath_that_cannot_be_rewritten_falls_back_to_an_inbundle_lib(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(procname, "_rewrite_rpath", lambda exe: False)
        real = _copied_interpreter(tmp_path)
        image = procname._plant_runtime(real, tmp_path / "runtime")
        assert image is not None and procname._is_bundle_image(image)
        assert (image.parents[1] / "lib").is_symlink(), "Contents/lib backs @executable_path/../lib"
        ran = subprocess.run(
            [str(image), "-I", "-c", "import encodings; print('ok')"],
            capture_output=True,
            text=True,
        )
        assert ran.stdout.strip() == "ok", ran.stderr


class TestFallbackAndMigration:
    def test_a_bundle_that_cannot_be_signed_falls_back_to_the_legacy_image_once(
        self, tmp_path, fake_codesign
    ):
        (fake_codesign.dir / "fail-all").touch()
        real = _copied_interpreter(tmp_path)
        root = tmp_path / "runtime"
        image = procname._plant_runtime(real, root)

        assert image == procname._legacy_image(root, procname._build_key(real))
        assert procname._healthy_image(image, real)
        assert not procname._flavor_dir(root, procname._build_key(real), "adhoc").exists()
        assert procname.runtime_image(real) == image, "the marker stops a retry on every start"
        assert len(fake_codesign()) == 1

    def test_a_legacy_link_repoints_and_the_legacy_tree_is_left_alone(self, tmp_path):
        real = _copied_interpreter(tmp_path)
        root = tmp_path / "runtime"
        key = procname._build_key(real)
        legacy = procname._publish_tree(
            real,
            root,
            key,
            procname._legacy_image(root, key),
            lambda staging, base: procname._populate_legacy(staging, real, base),
        )
        assert legacy is not None
        link = tmp_path / "venv" / "bin" / procname.BRAND
        link.parent.mkdir(parents=True)
        os.symlink(legacy, link)
        running = legacy.stat()

        assert procname._needs_replant(
            link, real, procname.runtime_image(real)
        ), "a link to the legacy, path-keyed image is stale"
        assert procname._plant_branded_image(link, real, None) == link
        target = Path(os.readlink(link))
        assert procname._is_bundle_image(target)
        assert legacy.is_file(), "nothing published is deleted"
        after = legacy.stat()
        assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
            running.st_ino,
            running.st_size,
            running.st_mtime_ns,
        ), "an image a process may be executing is never rewritten"


class TestTeamFlavor:
    def test_no_optin_means_no_keychain_signature(self, tmp_path, fake_codesign, monkeypatch):
        monkeypatch.setattr(procname, "_optin_path", lambda: tmp_path / "absent.json")
        monkeypatch.setattr(procname, "team_signing_refusal", lambda: None)
        real = _copied_interpreter(tmp_path)
        assert procname.publish_team_runtime(real, interactive=True) is None
        assert fake_codesign() == []

    @pytest.mark.parametrize(
        "env",
        [
            {"HOME": "/nonexistent-isolated-home"},
            {"LOCAL_OPERATOR_CONFIG_DIR": "/nonexistent-config"},
            {},  # the autouse LOCAL_OPERATOR_RUNTIME_ROOT override alone
        ],
    )
    def test_isolated_runs_never_team_sign_even_when_opted_in(
        self, tmp_path, fake_codesign, team_host_but_isolated, monkeypatch, env
    ):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        real = _copied_interpreter(tmp_path)

        assert procname.team_signing_refusal() is not None
        assert procname.publish_team_runtime(real, interactive=True) is None
        procname._plant_runtime(real, tmp_path / "runtime")
        assert set(_signed_with(fake_codesign())) <= {"-"}

    def test_a_non_tty_update_without_always_allow_does_not_sign(
        self, tmp_path, fake_codesign, opted_in_host, monkeypatch
    ):
        procname.write_signing_optin(_FAKE_SHA1, _FAKE_TEAM, always_allow_confirmed=False)
        monkeypatch.setattr(sys, "stdin", open(os.devnull))
        real = _copied_interpreter(tmp_path)
        assert procname.publish_team_runtime(real) is None
        assert fake_codesign() == []

    def test_an_opted_in_host_publishes_a_verified_team_image_that_wins(
        self, tmp_path, fake_codesign, opted_in_host
    ):
        real = _copied_interpreter(tmp_path)
        root = tmp_path / "runtime"
        adhoc = procname._plant_runtime(real, root)
        assert adhoc is not None

        team = procname.publish_team_runtime(real)

        assert team is not None
        assert procname._image_home(team).name.endswith(f"-{procname.FLAVOR_TEAM}")
        info = plistlib.loads((team.parents[1] / "Info.plist").read_bytes())
        assert info["CFBundleIdentifier"] == procname.APP_BUNDLE_ID
        sign, verify = fake_codesign()[-2:]
        assert sign[:5] == ["--force", "--sign", _FAKE_SHA1, "--identifier", procname.APP_BUNDLE_ID]
        assert verify[:3] == [
            "--verify",
            "--strict",
            f'-R=identifier "{procname.APP_BUNDLE_ID}" and anchor apple generic',
        ]
        assert procname.runtime_image(real) == team, "a healthy team image is preferred"
        assert procname._plant_runtime(real, root) == team
        assert adhoc.is_file(), "the ad-hoc image stays for anything still running it"

    def test_a_team_image_that_fails_the_apps_requirement_is_never_published(
        self, tmp_path, fake_codesign, opted_in_host
    ):
        (fake_codesign.dir / "fail-verify").touch()
        real = _copied_interpreter(tmp_path)
        assert procname.publish_team_runtime(real) is None
        root = tmp_path / "runtime"
        assert not procname._flavor_dir(root, procname._build_key(real), "team").exists()

    def test_a_different_app_team_is_refused(
        self, tmp_path, fake_codesign, opted_in_host, monkeypatch
    ):
        monkeypatch.setattr(
            procname, "code_identity", lambda path: {"TeamIdentifier": "ZZZZZ99999"}
        )
        real = _copied_interpreter(tmp_path)
        assert procname.publish_team_runtime(real) is None
        assert fake_codesign() == []

    def test_a_malformed_optin_is_ignored(self, tmp_path, monkeypatch):
        optin = tmp_path / "optin.json"
        monkeypatch.setattr(procname, "_optin_path", lambda: optin)
        optin.write_text(
            json.dumps({"identity": "Developer ID Application: X", "team": _FAKE_TEAM})
        )
        assert procname.read_signing_optin() is None


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({"HOME": "/nonexistent-isolated-home"}, "isolated-home"),
        ({"LOCAL_OPERATOR_RUNTIME_ROOT": "/nonexistent-root"}, "runtime-root-override"),
        ({"LOCAL_OPERATOR_CONFIG_DIR": "/nonexistent-config"}, "config-dir-override"),
    ],
)
def test_each_isolation_guard_refuses_on_its_own(monkeypatch, env, reason):
    """Each guard is discriminating by itself (pure check; nothing is written)."""
    home = procname._passwd_home()
    assert home is not None
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOCAL_OPERATOR_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    assert procname.team_signing_refusal() is None, "control: the operator's own shell"
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert procname.team_signing_refusal() == reason


def test_the_default_config_dir_spelled_out_is_not_an_isolated_run(monkeypatch):
    home = procname._passwd_home()
    assert home is not None
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOCAL_OPERATOR_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / ".local-operator"))
    assert procname.team_signing_refusal() is None
