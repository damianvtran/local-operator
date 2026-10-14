"""Bootstrap input negatives, not ordinary UI/signed-release execution evidence."""

import argparse
import hashlib
from pathlib import Path

import pytest

from local_operator import update_bootstrap as bootstrap


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
    monkeypatch.setattr(bootstrap, "_publication", lambda _target: (
        "0.70.99", "local_operator-0.70.99-py3-none-any.whl",
        "https://files.pythonhosted.org/fixture.whl", hashlib.sha256(b"different").hexdigest(),
    ))
    monkeypatch.setattr(bootstrap, "_read_url", lambda *_args, **_kwargs: data)
    calls = []
    monkeypatch.setattr(bootstrap, "_run", lambda *_args, **_kwargs: calls.append("worker"))
    args = argparse.Namespace(root=str(root), baseline=str(generation), target=None,
                              caller=bootstrap.CALLER_CONTRACT, snapshot=None, no_services=False)
    with pytest.raises(bootstrap.BootstrapError, match="hash disagrees"):
        bootstrap.bootstrap(args)
    assert not calls


def test_legacy_caller_has_no_implicit_rolling_authority(destination):
    root, generation = destination
    args = argparse.Namespace(root=str(root), baseline=str(generation), target=None,
                              caller="old-ui", snapshot=None, no_services=False)
    with pytest.raises(bootstrap.BootstrapError, match="caller"):
        bootstrap.bootstrap(args)
