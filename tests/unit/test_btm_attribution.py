"""Background Task Management attribution: the signed launcher, and what names it.

WHY THESE TESTS EXIST. macOS names a LaunchAgent's Login Items row — and its
"… can run in the background" banner — after the code at the plist's ``Program``
(measured on macOS 27.2, 2026-11-14; see AGENTS.md "BTM attribution"). An
unsigned ``Program`` read "python3"/"python3.12" with one banner per item; the
CI-signed launcher reads "Local Operator" under "Radient Inc." Every property
below is one whose failure is SILENT on a developer machine:

* ``daemon_program`` choosing an unverified launcher would name code nobody
  signed (or the retired individual certificate) in every plist;
* the plist losing ``Program``/``AssociatedBundleIdentifiers`` brings "python3" back;
* the episode label carrying a personal prefix ships it into every user's launchd;
* the legacy-episode sweep deleting a RUNNING job or a plist we did not write;
* the signing opt-in accepting an individual Developer ID.

Nothing here runs ``codesign`` or ``launchctl``: signatures and launchd answers
are injected, and every filesystem path is under ``tmp_path``.
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator import launchd, owned_process, procname, update

RADIENT = "Developer ID Application: Radient Inc. (SHA2U6KT7V)"
INDIVIDUAL = "Developer ID Application: Jane Example (SHA2U6KT7V)"


# ---------------------------------------------------------------------------
# daemon_program: the launcher only when verified, else exactly the shim
# ---------------------------------------------------------------------------


@pytest.fixture
def stable_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A generation-layout ``bin`` with a shim, and a planted launcher bundle."""
    bin_dir = tmp_path / "lop" / "bin"
    bin_dir.mkdir(parents=True)
    shim = bin_dir / "python3"
    shim.write_text("#!/bin/sh\n")
    shim.chmod(0o755)
    exe = bin_dir / update.LAUNCHER_BUNDLE_DIRNAME / "Contents" / "MacOS" / "Local Operator"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"\xcf\xfa\xed\xfe")
    exe.chmod(0o755)
    monkeypatch.setattr(update, "daemon_image_path", lambda: shim)
    monkeypatch.setattr(update, "daemon_image", lambda: shim)
    monkeypatch.setattr(update, "_HOST_IS_MACOS", True)
    monkeypatch.setattr(update, "_LAUNCHER_VERDICTS", {})
    return bin_dir


def _signature(monkeypatch: pytest.MonkeyPatch, fields: dict[str, str] | None) -> None:
    monkeypatch.setattr(update, "_launcher_signature", lambda _bundle: fields)


def _good() -> dict[str, str]:
    return {
        "Identifier": update.LAUNCHER_BUNDLE_ID,
        "TeamIdentifier": "SHA2U6KT7V",
        "Authority": RADIENT,
    }


def test_a_verified_launcher_is_the_program(
    stable_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _signature(monkeypatch, _good())
    program = update.daemon_program()
    assert program == update.launcher_executable(stable_bin / update.LAUNCHER_BUNDLE_DIRNAME)


@pytest.mark.parametrize(
    "fields",
    [
        None,  # codesign --verify --strict failed (ad-hoc tamper, unsigned, truncated)
        {**_good(), "Authority": INDIVIDUAL},  # same team, retired individual leaf
        {**_good(), "Authority": "Developer ID Application: Radient Inc. Evil (SHA2U6KT7V)"},
        {**_good(), "TeamIdentifier": "ABCDE12345"},
        {**_good(), "Identifier": "com.local-operator"},  # some other signed bundle here
        {"Identifier": update.LAUNCHER_BUNDLE_ID, "Signature": "adhoc"},
    ],
    ids=["unverified", "individual", "lookalike", "other-team", "other-id", "adhoc"],
)
def test_an_unverifiable_launcher_falls_back_to_the_shim(
    stable_bin: Path, monkeypatch: pytest.MonkeyPatch, fields: dict[str, str] | None
) -> None:
    _signature(monkeypatch, fields)
    assert update.daemon_program() == stable_bin / "python3"


def test_no_launcher_and_non_macos_both_answer_the_shim(
    stable_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _signature(monkeypatch, _good())
    monkeypatch.setattr(update, "_HOST_IS_MACOS", False)
    assert update.daemon_program() == stable_bin / "python3"
    monkeypatch.setattr(update, "_HOST_IS_MACOS", True)
    monkeypatch.setattr(update, "_LAUNCHER_VERDICTS", {})
    (stable_bin / update.LAUNCHER_BUNDLE_DIRNAME / "Contents" / "MacOS" / "Local Operator").unlink()
    assert update.daemon_program() == stable_bin / "python3"


def test_no_layout_is_still_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """``None`` keeps its meaning — every installer's pre-layout fallback."""
    monkeypatch.setattr(update, "daemon_image", lambda: None)
    assert update.daemon_program() is None


def test_ensure_daemon_launcher_copies_only_a_verified_source(
    stable_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = stable_bin / update.LAUNCHER_BUNDLE_DIRNAME
    import shutil

    shutil.rmtree(dest)
    generation = tmp_path / "gen"
    site = generation / "tools" / "local-operator" / "lib" / "python3.12" / "site-packages"
    source = site / update.LAUNCHER_PACKAGE_PATH
    (source / "Contents" / "MacOS").mkdir(parents=True)
    (source / "Contents" / "MacOS" / "Local Operator").write_bytes(b"launcher")
    (source / "Contents" / "MacOS" / "Local Operator").chmod(0o755)

    verified: set[Path] = set()
    monkeypatch.setattr(update, "launcher_verified", lambda bundle: bundle in verified)
    assert update.ensure_daemon_launcher(generation) is None
    assert not dest.exists(), "an unverified wheel launcher must never be planted"

    verified.update({source, dest})
    assert update.ensure_daemon_launcher(generation) == dest
    assert (dest / "Contents" / "MacOS" / "Local Operator").read_bytes() == b"launcher"
    # Idempotent: an unchanged tree is not rewritten (same inode).
    inode = (dest / "Contents" / "MacOS" / "Local Operator").stat().st_ino
    assert update.ensure_daemon_launcher(generation) == dest
    assert (dest / "Contents" / "MacOS" / "Local Operator").stat().st_ino == inode


# ---------------------------------------------------------------------------
# First upgrade: an old installer plants only the shim, then NEW code repairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("child", ["services", "mobile"])
@pytest.mark.parametrize("wheel_launcher", ["signed", "unsigned", "missing"])
def test_refresh_child_plants_launcher_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child: str,
    wheel_launcher: str,
) -> None:
    """New-wheel children must not depend on the old installer knowing this hook.

    The pre-feature parent leaves a valid pointer and shim, but no launcher.
    Exercise both independently spawned repair entrypoints, including the real
    plant and plist Program selection; only signature answers and service I/O
    are replaced. Native signed-bundle handoff evidence is recorded on the PR.
    """
    generation = tmp_path / "lop" / "generations" / "new"
    package = generation / "tools" / "local-operator" / "lib" / "python3.12"
    package = package / "site-packages" / "local_operator"
    package.mkdir(parents=True)
    pointer = tmp_path / "lop" / "current"
    pointer.symlink_to(generation, target_is_directory=True)
    bin_dir = tmp_path / "lop" / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "python3"
    shim.write_text(update._DAEMON_SHIM)
    shim.chmod(0o755)
    source = package.parent / update.LAUNCHER_PACKAGE_PATH
    if wheel_launcher != "missing":
        executable = update.launcher_executable(source)
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"launcher")
        executable.chmod(0o755)
    monkeypatch.setattr(update, "_STABLE_ROOT", str(tmp_path / "lop"))
    monkeypatch.setattr(update, "process_install_root", lambda: package.parents[3])
    monkeypatch.setattr(update, "__file__", str(package / "update.py"))
    monkeypatch.setattr(update, "_HOST_IS_MACOS", True)
    monkeypatch.setattr(update, "generation_layout_supported", lambda: True)
    monkeypatch.setattr(update, "_repair_refusal", lambda: None)
    monkeypatch.setattr(update, "_sweep_legacy_episode_plists", lambda: ())
    monkeypatch.setattr(update, "_LAUNCHER_VERDICTS", {})
    _signature(monkeypatch, _good() if wheel_launcher == "signed" else None)
    assert update.daemon_program() == shim

    rendered: list[dict[str, object]] = []

    def render() -> launchd.PlistRefresh:
        job = procname.launchd_job("local_operator.wakes.supervisor", label="Local Operator")
        # Round-trip a real plist rather than inspecting an unrendered mock.
        rendered.append(plistlib.loads(plistlib.dumps(job)))
        return launchd.PlistRefresh(name=child, kind="current")

    if child == "services":
        monkeypatch.setattr(update, "_platform_refresh_steps", lambda: ((child, "", render),))
        assert update.daemons_refresh_command() == 0
    else:
        from local_operator.mobile import install as mobile_install

        monkeypatch.setattr(mobile_install, "refresh_plist_if_stale", render)
        monkeypatch.setattr(mobile_install, "_supervised_pid", lambda: 12345)
        monkeypatch.setattr(update, "generation_staleness_of_process", lambda _pid: False)
        assert update.mobile_refresh_command() == 0
    expected = (
        update.launcher_executable(bin_dir / update.LAUNCHER_BUNDLE_DIRNAME)
        if wheel_launcher == "signed"
        else shim
    )
    assert len(rendered) == 1
    assert rendered[0]["Program"] == str(expected)
    assert rendered[0]["AssociatedBundleIdentifiers"] == ["com.local-operator"]
    assert (bin_dir / update.LAUNCHER_BUNDLE_DIRNAME).exists() == (wheel_launcher == "signed")


@pytest.mark.parametrize("child", [update.daemons_refresh_command, update.mobile_refresh_command])
def test_refused_refresh_does_not_plant_launcher(
    monkeypatch: pytest.MonkeyPatch, child: Any
) -> None:
    monkeypatch.setattr(update, "_repair_refusal", lambda: "source checkout")

    def forbidden() -> None:
        pytest.fail("a refused refresh must not mutate the stable installation")

    monkeypatch.setattr(update, "ensure_daemon_image", forbidden)
    assert child() == 0


# ---------------------------------------------------------------------------
# The plist shape every installer gets through launchd_job
# ---------------------------------------------------------------------------


def test_launchd_job_names_the_program_and_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = Path("/stable/bin/Local Operator.lopruntime/Contents/MacOS/Local Operator")
    monkeypatch.setattr(procname, "supervised_image", lambda: launcher)
    job = procname.launchd_job("local_operator.wakes.supervisor", label="Local Operator [wakes]")
    assert job == {
        "Program": str(launcher),
        "ProgramArguments": ["Local Operator [wakes]", "-m", "local_operator.wakes.supervisor"],
        "AssociatedBundleIdentifiers": ["com.local-operator"],
    }
    # The rendered plist is valid launchd input.
    assert plistlib.loads(plistlib.dumps({"Label": "x", **job}))["Program"] == str(launcher)


def test_supervised_image_is_daemon_program(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = Path("/x/Local Operator")
    monkeypatch.setattr(update, "daemon_program", lambda: sentinel)
    assert procname.supervised_image() == sentinel


def test_the_relay_counts_the_launcher_as_the_shim(
    stable_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network import relay

    monkeypatch.setattr(
        update, "launcher_bundle_path", lambda: stable_bin / "Local Operator.lopruntime"
    )
    exe = update.launcher_executable(stable_bin / update.LAUNCHER_BUNDLE_DIRNAME)
    assert relay._names_the_shim(exe)
    assert relay._names_the_shim(stable_bin / "python3")
    assert not relay._names_the_shim(Path("/usr/bin/python3"))


# ---------------------------------------------------------------------------
# Episode jobs: product label, attributed Program, unchanged interpreter
# ---------------------------------------------------------------------------


def test_the_episode_label_is_in_the_product_namespace() -> None:
    assert owned_process.job_label_for("abc123") == "com.local-operator.episode.abc123"
    assert "damian" not in owned_process.JOB_LABEL_PREFIX


def test_an_episode_job_runs_its_own_interpreter_through_the_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = Path("/stable/bin/Local Operator.lopruntime/Contents/MacOS/Local Operator")
    monkeypatch.setattr(update, "signed_launcher", lambda: launcher)
    argv = ("/work/.venv/bin/python3.12", "scripts/run_episode.py", "--owned-host")
    keys, env = owned_process._attributed_program(argv)
    assert keys == {
        "Program": str(launcher),
        "AssociatedBundleIdentifiers": ["com.local-operator"],
    }
    # TARGET mode: the launcher execs exactly argv[0], never the installed shim.
    assert env == {update.LAUNCHER_TARGET_ENV: argv[0]}


@pytest.mark.parametrize("launcher", [None, Path("/l")])
def test_an_episode_job_keeps_the_old_shape_without_a_launcher_or_absolute_argv0(
    monkeypatch: pytest.MonkeyPatch, launcher: Path | None
) -> None:
    monkeypatch.setattr(update, "signed_launcher", lambda: launcher)
    argv = ("python3.12", "x") if launcher else ("/venv/bin/python3.12", "x")
    assert owned_process._attributed_program(argv) == ({}, {})


# ---------------------------------------------------------------------------
# The legacy-episode sweep: only ours, only idle
# ---------------------------------------------------------------------------


class _Launchd:
    """``launchctl`` answers keyed by label; a bootout makes the label absent."""

    def __init__(self, answers: dict[str, tuple[int, str]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        label = args[-1].rsplit("/", 1)[-1]
        if args[0] == "bootout":
            self.answers[label] = (113, "")
            return subprocess.CompletedProcess(["launchctl", *args], 0, "", "")
        rc, out = self.answers.get(label, (113, ""))
        return subprocess.CompletedProcess(["launchctl", *args], rc, out, "")


def _plist(directory: Path, label: str, argv: list[str], *, file_label: str | None = None) -> Path:
    path = directory / f"{file_label or label}.plist"
    path.write_bytes(plistlib.dumps({"Label": label, "ProgramArguments": argv}))
    return path


@pytest.fixture
def agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    directory = home / "Library" / "LaunchAgents"
    directory.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(launchd, "real_home", lambda: home)
    monkeypatch.setattr(launchd, "_sleep", lambda _s: None)
    return directory


def test_the_sweep_removes_only_idle_legacy_episode_jobs(agents: Path) -> None:
    host = ["/venv/bin/python3.12", "run_episode.py", "--owned-host"]
    gone = _plist(agents, "com.damian.lop-episode.gone", host)
    stopped = _plist(agents, "com.damian.lop-episode.stopped", host)
    running = _plist(agents, "com.damian.lop-episode.running", host)
    unknown = _plist(agents, "com.damian.lop-episode.unknown", host)
    not_owned = _plist(agents, "com.damian.lop-episode.foreign", ["/bin/sh", "-c", "true"])
    relabelled = _plist(
        agents, "com.damian.other", host, file_label="com.damian.lop-episode.relabelled"
    )
    personal = _plist(agents, "com.damian.backup", host)
    runner = _Launchd(
        {
            "com.damian.lop-episode.stopped": (0, "state = not running\n"),
            "com.damian.lop-episode.running": (0, "state = running\npid = 4242\n"),
            "com.damian.lop-episode.unknown": (5, ""),
        }
    )

    removed = update._sweep_legacy_episode_plists(runner=runner)

    assert sorted(removed) == sorted([gone, stopped])
    for kept in (running, unknown, not_owned, relabelled, personal):
        assert kept.exists(), f"{kept.name} must be left alone"
    # Only the stopped job was booted out; nothing addressed a kept job's state.
    bootouts = [call for call in runner.calls if call[0] == "bootout"]
    assert bootouts == [("bootout", f"{launchd.job_domain()}/com.damian.lop-episode.stopped")]
    # A plist we did not write is never even asked about.
    asked = {call[-1].rsplit("/", 1)[-1] for call in runner.calls}
    assert "com.damian.lop-episode.foreign" not in asked
    assert "com.damian.backup" not in asked


def test_the_sweep_under_a_redirected_home_addresses_nothing(
    agents: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``observe_job``'s real-home check: a rig's HOME is never the operator's launchd."""
    host = ["/venv/bin/python3.12", "--owned-host"]
    path = _plist(agents, "com.damian.lop-episode.rig", host)
    monkeypatch.setattr(launchd, "real_home", lambda: tmp_path / "operator")
    runner = _Launchd({})
    assert update._sweep_legacy_episode_plists(runner=runner) == []
    assert path.exists()
    assert runner.calls == []


# ---------------------------------------------------------------------------
# Local signing: the organization identity only
# ---------------------------------------------------------------------------


@pytest.fixture
def optin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "runtime-signing.json"
    monkeypatch.setattr(procname, "_optin_path", lambda: path)
    return path


_SHA = "A" * 40


def test_an_individual_identity_is_refused_before_anything_is_written(optin: Path) -> None:
    with pytest.raises(procname.SigningIdentityRefused):
        procname.write_signing_optin(
            _SHA, "SHA2U6KT7V", always_allow_confirmed=True, name=INDIVIDUAL
        )
    assert not optin.exists()
    procname.write_signing_optin(_SHA, "SHA2U6KT7V", always_allow_confirmed=True, name=RADIENT)
    assert procname.read_signing_optin() is not None


def test_a_recorded_individual_optin_falls_back_to_adhoc_with_a_reason(
    optin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's pre-rule record (no ``name``, individual SHA) is no opt-in at all."""
    import json

    optin.write_text(
        json.dumps({"identity": _SHA, "team": "SHA2U6KT7V", "always_allow_confirmed": True})
    )
    assert procname.read_signing_optin() is None
    assert procname.signing_optin_refusal() == "identity-not-allowed"
    monkeypatch.setattr(procname, "team_signing_refusal", lambda: None)
    result = procname.publish_team_runtime_result("/usr/bin/python3")
    assert result.image is None and result.reason == "identity-not-allowed"
    assert result.signed is False, "no codesign may run for a refused identity"


def test_setup_lists_only_the_organization_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator import permissions_cli

    listing = (
        f'  1) {"A" * 40} "{INDIVIDUAL}"\n'
        f'  2) {"B" * 40} "{RADIENT}"\n'
        "     2 valid identities found\n"
    )
    monkeypatch.setattr(
        permissions_cli.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, listing, ""),
    )
    assert permissions_cli._developer_identities("SHA2U6KT7V") == [("B" * 40, RADIENT)]


def test_an_unmarked_team_tree_is_never_selected(tmp_path: Path, monkeypatch: Any) -> None:
    """A team tree signed before the rule carries no signer marker."""
    monkeypatch.setattr(procname, "_healthy_image", lambda image, real=None: True)
    tree = procname._flavor_dir(tmp_path, "key", procname.FLAVOR_TEAM)
    tree.mkdir()
    real = Path("/usr/bin/python3")
    assert procname._team_image(tmp_path, "key", real) is None
    (tree / procname._TEAM_SIGNER_MARKER).write_text(INDIVIDUAL + "\n")
    assert procname._team_image(tmp_path, "key", real) is None
    (tree / procname._TEAM_SIGNER_MARKER).write_text(RADIENT + "\n")
    assert procname._team_image(tmp_path, "key", real) == procname._bundle_image(
        tmp_path, "key", procname.FLAVOR_TEAM
    )


def test_no_product_code_carries_the_personal_label_prefix() -> None:
    """``git grep com.damian -- local_operator scripts``: only the commented migration matcher."""
    root = Path(__file__).resolve().parents[2]
    hits = []
    for base in ("local_operator", "scripts"):
        for path in (root / base).rglob("*.py"):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "com.damian" in line:
                    hits.append((path.relative_to(root).as_posix(), number, line.strip()))
    assert [hit[0] for hit in hits] == ["local_operator/update.py"] * len(hits), hits
    assert any("_LEGACY_EPISODE_LABEL_PREFIX = " in hit[2] for hit in hits), hits
