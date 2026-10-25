"""Non-native proof for the relay native probe's generation import guard.

These cells use real, minimal venv interpreters but no systemd, service, network,
or generation installation. CI can therefore prove the provenance mechanism even
when the hosted private-manager prerequisite is blocked.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import native_relay_update_probe as probe

pytestmark = pytest.mark.skipif(os.name == "nt", reason="a generation venv uses a POSIX symlink")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """This file is outside tests/unit/network, so supply its own isolated root."""
    return tmp_path


def _package(root: Path, *, poison: bool = False) -> None:
    package = root / "local_operator"
    network = package / "network"
    network.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise RuntimeError("driver checkout imported")\n' if poison else "",
        encoding="utf-8",
    )
    (package / "update.py").write_text("", encoding="utf-8")
    (network / "__init__.py").write_text("", encoding="utf-8")
    (network / "relay.py").write_text("", encoding="utf-8")


def _minimal_venv(root: Path, *, with_package: bool = True) -> None:
    """Build only the venv identity pieces; no copied interpreter or pip install."""
    root.mkdir(parents=True)
    executable = Path(getattr(sys, "base_executable", sys.executable)).resolve()
    (root / "pyvenv.cfg").write_text(
        f"home = {executable.parent}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python3").symlink_to(executable)
    site = (
        root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    site.mkdir(parents=True)
    if with_package:
        _package(site)


def test_generation_child_ignores_driver_python_paths_and_imports_its_own_tree(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = root / "generations" / "base"
    install_root = root / "install-base"
    _minimal_venv(install_root)
    monkeypatch.setattr(probe.update, "_generation_install_root", lambda _generation: install_root)

    driver = root / "driver"
    _package(driver, poison=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(driver)
    env["PYTHONHOME"] = str(driver)
    work = root / "work"
    work.mkdir()
    sanitized = probe._generation_environment(env)  # noqa: SLF001
    assert "PYTHONPATH" not in sanitized and "PYTHONHOME" not in sanitized
    assert sanitized["LOCAL_OPERATOR_NO_NOTIFICATIONS"] == "1"

    receipt = probe._assert_generation_imports(generation, env, work, "base")  # noqa: SLF001

    assert receipt["interpreter"] == str(install_root / "bin" / "python3")
    assert set(receipt["modules"]) == {"local_operator", "update", "relay"}
    for value in receipt["modules"].values():
        assert Path(value).is_relative_to(install_root), receipt


def test_generation_child_import_outside_its_tree_is_refused(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = root / "generations" / "base"
    install_root = root / "install-base"
    _minimal_venv(install_root, with_package=False)
    monkeypatch.setattr(probe.update, "_generation_install_root", lambda _generation: install_root)

    # A -c child sees its cwd before site-packages. This decoy proves the guard
    # rejects an import outside the expected generation without PYTHONPATH too.
    _package(root)
    env = probe._generation_environment()  # noqa: SLF001

    with pytest.raises(probe.ProbeFailure, match="outside intended generation"):
        probe._assert_generation_imports(generation, env, root, "base")  # noqa: SLF001


def test_one_shot_label_preserves_full_head_within_github_limit() -> None:
    """The shortened prefix fits GitHub without weakening exact-head parsing."""
    repo = Path(__file__).resolve().parents[2]
    wrapper = (repo / "scripts/native-delegated-once-wrapper.sh").read_text(encoding="utf-8")
    workflow = (repo / ".github/workflows/native-delegated-once.yml").read_text(encoding="utf-8")
    prefix = "native-"
    head = "0123456789abcdef0123456789abcdef01234567"
    label = f"{prefix}{head}"
    assert len(label) == 47 <= 50

    bash_pattern = r"^native-([0-9a-f]{40})$"
    assert f'[[ "$EXPECTED_HEAD" =~ {bash_pattern} ]]' in wrapper
    assert f"startsWith(github.event.label.name, '{prefix}')" in workflow
    match = re.fullmatch(bash_pattern.removeprefix("^").removesuffix("$"), label)
    assert match is not None and match.group(1) == head
    for invalid in (
        f"native-delegated-once-{head}",
        f"{prefix}{head[:-1]}",
        f"{prefix}{head}0",
        f"{prefix}{head.upper()}",
    ):
        assert re.fullmatch(bash_pattern.removeprefix("^").removesuffix("$"), invalid) is None


def test_historical_unit_uses_the_renderers_source_checkout_fallback(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent generation shim must not make the fixture reject its own unit."""
    generation = root / "generation"
    install_root = root / "install-root"
    _minimal_venv(install_root, with_package=False)
    monkeypatch.setattr(probe.update, "_generation_install_root", lambda _generation: install_root)
    monkeypatch.setattr("local_operator.procname.supervised_image", lambda: None)
    monkeypatch.setattr(probe, "_require", lambda _argv: None)
    unit = root / "relay.service"

    assert probe.relay._systemd_unit_image() == Path(sys.executable)  # noqa: SLF001
    probe._write_pinned_unit(unit, generation, 50295)  # noqa: SLF001

    # Old fixture code searched for an empty image and raised before writing the
    # unit; the shared renderer choice instead reaches the generation interpreter.
    assert str(install_root / "bin" / "python3") in unit.read_text(encoding="utf-8")


def test_historical_unit_creates_the_append_log_parent(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct unit writes need the same fresh-log prerequisite as the installer."""
    generation = root / "generation"
    install_root = root / "install-root"
    _minimal_venv(install_root, with_package=False)
    config = root / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setattr(probe.update, "_generation_install_root", lambda _generation: install_root)
    monkeypatch.setattr("local_operator.procname.supervised_image", lambda: None)
    monkeypatch.setattr(probe.relay.supervisors, "systemd_writes_log_file", lambda: True)
    monkeypatch.setattr(probe, "_require", lambda _argv: None)
    unit = root / "relay.service"
    log = probe.relay.log_path()
    assert not log.parent.exists()

    probe._write_pinned_unit(unit, generation, 50295)  # noqa: SLF001

    # This assertion fails without the mkdir even though the unit itself writes.
    assert log.parent.is_dir()
    rendered = unit.read_text(encoding="utf-8")
    assert f"StandardOutput=append:{log}" in rendered
    assert f"StandardError=append:{log}" in rendered


def test_readiness_failure_preserves_observation_and_log_tails(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A readiness timeout must upload evidence, not just a generic sentence."""
    monkeypatch.setenv("RUNNER_TEMP", str(root))
    config = root / "config"
    log = config / "logs" / "network.log"
    log.parent.mkdir(parents=True)
    log.write_text("network-tail\n", encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    observed = {"pid": 4321, "advertised_source_ref": "wrong", "answering": False}
    monkeypatch.setattr(probe, "_require", lambda _argv: None)
    monkeypatch.setattr(probe, "_observation", lambda: observed)

    class _Clock:
        def __init__(self) -> None:
            self._ticks = iter([0.0, 0.0, 31.0])

        def monotonic(self) -> float:
            return next(self._ticks)

        def sleep(self, _seconds: float) -> None:
            return None

    # probe.time is the shared time module; replace only the probe's reference
    # so pytest's own watchdog keeps its real clock.
    monkeypatch.setattr(probe, "time", _Clock())
    monkeypatch.setattr(probe.shutil, "which", lambda _name: "/usr/bin/journalctl")
    monkeypatch.setattr(
        probe,
        "_run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 0, stdout="journal-tail\n", stderr=""
        ),
    )

    with pytest.raises(probe.ProbeFailure, match="did not become ready"):
        probe._start_and_wait(Path("/generation"), "base")  # noqa: SLF001

    receipt = json.loads((root / "native-relay-readiness.json").read_text(encoding="utf-8"))
    assert receipt["last_observation"] == observed
    assert "journal-tail" in receipt["journal_tail"]
    assert "network-tail" in receipt["network_log_tail"]
    assert receipt["network_log_path"] == str(probe.relay.log_path())
