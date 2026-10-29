"""Bootstrap input negatives, not ordinary UI/signed-release execution evidence."""

import argparse
import hashlib
import os
import sys
import types
from pathlib import Path

import pytest

import local_operator
from local_operator import update_bootstrap as bootstrap


def _prepared_base_interpreter(
    destination, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub: types.ModuleType
) -> str:
    """Run ``bootstrap`` to the controller-preparation step and return its base.

    ``scripts/lop-update.sh`` runs this module from the TARGET ref under the
    INSTALLED interpreter, so the ``local_operator.procname`` it imports is the
    OLD generation's module — the stub here. ``_run`` and the download are
    stubbed so the cell is about the ask, not about uv.
    """
    root, generation = destination
    wheel = b"fixture"
    archive = hashlib.sha256(wheel).hexdigest()
    monkeypatch.setattr(
        bootstrap,
        "_publication",
        lambda _target: ("0.70.99", "x.whl", "https://example.invalid/x.whl", archive),
    )
    monkeypatch.setattr(bootstrap, "_read_url", lambda *_args, **_kwargs: wheel)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _name: "/usr/bin/uv")
    commands: list[list[str]] = []
    monkeypatch.setattr(bootstrap, "_run", lambda argv, **_kwargs: commands.append(argv))
    # The PACKAGE attribute, not ``sys.modules``: ``from local_operator import
    # procname`` answers from the package first, which is exactly the path the
    # seed takes against the installed generation.
    monkeypatch.setattr(local_operator, "procname", stub, raising=False)
    python = tmp_path / "gen" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(os.path.realpath(sys.executable)))
    monkeypatch.setattr(bootstrap.sys, "executable", str(python))
    args = argparse.Namespace(
        root=str(root),
        baseline=str(generation),
        target=None,
        caller=bootstrap.CALLER_CONTRACT,
        snapshot=None,
        no_services=False,
    )

    assert bootstrap.bootstrap(args) == 0

    assert commands[0][:3] == ["/usr/bin/uv", "venv", "--python"], commands[0]
    return commands[0][3]


def test_preparation_survives_a_procname_without_real_interpreter(
    destination, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B1: the seed cannot import a name the INSTALLED generation does not have.

    The reproduction is the one the reviewer ran: origin/main's ``procname`` on
    the path, and ``real_interpreter`` does not exist there.
    """
    stub = types.ModuleType("local_operator.procname")
    assert not hasattr(stub, "real_interpreter")

    base = _prepared_base_interpreter(destination, monkeypatch, tmp_path, stub)

    assert base == os.path.realpath(
        sys.executable
    ), "the venv is built from the resolved executable"


def test_preparation_uses_the_private_seam_when_that_is_all_the_old_module_has(
    destination, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The seam's other spelling: ``_real_interpreter`` is what v0.74.0 carries."""
    stub = types.ModuleType("local_operator.procname")
    stub._real_interpreter = lambda: Path("/base/python3.12")  # type: ignore[attr-defined]

    base = _prepared_base_interpreter(destination, monkeypatch, tmp_path, stub)

    assert base == "/base/python3.12"


@pytest.fixture
def destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".local" / "share" / "lop"
    generation = root / "generations" / "owned-baseline"
    image = generation / "tools" / "local-operator"
    image.mkdir(parents=True)
    (image / "uv-receipt.toml").write_text("fixture")
    (root / "current").symlink_to(generation)
    return root, generation


def test_destination_does_not_promote_unknown_or_app_root(destination, tmp_path):
    root, generation = destination
    proof = bootstrap.destination(str(root), str(generation))
    assert proof.root == root and proof.baseline == generation
    with pytest.raises(bootstrap.BootstrapError, match="canonical"):
        bootstrap.destination(str(tmp_path / "app-managed"), str(generation))
    (generation / "tools" / "local-operator" / "uv-receipt.toml").unlink()
    with pytest.raises(bootstrap.BootstrapError, match="kind override"):
        bootstrap.destination(str(root), str(generation))


def test_current_change_refuses_before_preparation(destination):
    root, generation = destination
    other = root / "generations" / "owned-newer"
    other.mkdir()
    (root / "current").unlink()
    (root / "current").symlink_to(other)
    with pytest.raises(bootstrap.BootstrapError, match="current changed"):
        bootstrap.destination(str(root), str(generation))


def test_bad_download_hash_never_loads_controller_or_creates_environment(
    destination, monkeypatch: pytest.MonkeyPatch
):
    root, generation = destination
    data = b"not-a-controller"
    monkeypatch.setattr(
        bootstrap,
        "_publication",
        lambda _target: (
            "0.70.99",
            "local_operator-0.70.99-py3-none-any.whl",
            "https://files.pythonhosted.org/fixture.whl",
            hashlib.sha256(b"different").hexdigest(),
        ),
    )
    monkeypatch.setattr(bootstrap, "_read_url", lambda *_args, **_kwargs: data)
    calls = []
    monkeypatch.setattr(bootstrap, "_run", lambda *_args, **_kwargs: calls.append("worker"))
    args = argparse.Namespace(
        root=str(root),
        baseline=str(generation),
        target=None,
        caller=bootstrap.CALLER_CONTRACT,
        snapshot=None,
        no_services=False,
    )
    with pytest.raises(bootstrap.BootstrapError, match="hash disagrees"):
        bootstrap.bootstrap(args)
    assert not calls


def test_legacy_caller_has_no_implicit_rolling_authority(destination):
    root, generation = destination
    args = argparse.Namespace(
        root=str(root),
        baseline=str(generation),
        target=None,
        caller="old-ui",
        snapshot=None,
        no_services=False,
    )
    with pytest.raises(bootstrap.BootstrapError, match="caller"):
        bootstrap.bootstrap(args)
