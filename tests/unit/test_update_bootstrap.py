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


# --- The pre-flip build rides to the tail (PR #4135, review round 2, R2-1) ---
#
# The desktop/shim rolling path: `controller` installs and flips `current`, then
# runs the NEW current's `lop update --controller-tail`. Under rolling-no-kill
# the replaced build keeps running, and only `--controller-previous` names it
# to the tail's prune, so keep 0 must not delete it. These drive the real
# `controller`/`controller_tail` and the real CLI parser; every collaborator
# that would verify a wheel, take the install lock or touch services is a fake.

TARGET = "0.70.99"


def _fake_controller_world(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    generation: Path,
    *,
    current_after_flip: str,
) -> tuple[list[str], list[list[str]], Path, argparse.Namespace]:
    """Stub ``controller``'s collaborators; returns (order, tail commands, promoted, args).

    ``current_after_flip`` names the generation ``current`` points at once the
    install has run: ``"promoted"`` is the normal flip, anything else is a
    superseding update that won the pointer in between.
    """
    import contextlib
    import time
    from types import SimpleNamespace

    from local_operator import install_transaction, update

    order: list[str] = []
    commands: list[list[str]] = []
    generations = root / "generations"
    promoted = generations / "promoted"
    artifact = root.parent / f"local_operator-{TARGET}-py3-none-any.whl"
    state = {"current": generation}

    real_destination = bootstrap.destination

    def destination(root_arg: str, baseline: str) -> object:
        order.append("baseline")
        return real_destination(root_arg, baseline)

    def install(_source: object, **_kwargs: object) -> Path:
        order.append("install")
        promoted.mkdir()
        other = generations / current_after_flip
        other.mkdir(exist_ok=True)
        state["current"] = other  # the flip
        return promoted

    @contextlib.contextmanager
    def transaction(_root: Path, *, operation: str):
        yield SimpleNamespace(validate=lambda _root: None)

    # Only reached on darwin; the signed helper is not part of a unit fixture.
    keyagent = types.ModuleType("local_operator.operator.macos.keyagent")
    keyagent.KeyagentError = RuntimeError  # type: ignore[attr-defined]
    keyagent.verify_bundle = lambda _path: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "local_operator.operator.macos.keyagent", keyagent)
    monkeypatch.setattr(bootstrap, "destination", destination)
    monkeypatch.setattr(bootstrap, "verify_controller", lambda *_a: (TARGET, "src"))
    monkeypatch.setattr(bootstrap, "_run", lambda argv, **_kw: commands.append(argv))
    monkeypatch.setattr(update, "stable_root", lambda: root)
    monkeypatch.setattr(update, "install_kind", lambda **_kw: update.InstallKind.UV_TOOL)
    monkeypatch.setattr(
        update,
        "_fresh_publication",
        lambda *_a: SimpleNamespace(
            digest="d" * 64, artifact=artifact.name, fresh_until=time.monotonic() + 600
        ),
    )
    monkeypatch.setattr(install_transaction, "transaction", transaction)
    monkeypatch.setattr(update, "current_generation", lambda: state["current"])
    monkeypatch.setattr(update, "install_into_generation", install)
    args = argparse.Namespace(
        controller_root=str(root),
        controller_baseline=str(generation),
        controller_artifact=str(artifact),
        controller_digest="d" * 64,
        controller_caller=bootstrap.CALLER_CONTRACT,
        controller_target=TARGET,
        from_snapshot=None,
        no_services=False,
    )
    return order, commands, promoted, args


def _flag(command: list[str], name: str) -> str | None:
    return command[command.index(name) + 1] if name in command else None


def test_the_controller_hands_the_pre_flip_baseline_to_the_tail(
    destination, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, generation = destination
    order, commands, promoted, args = _fake_controller_world(
        monkeypatch, root, generation, current_after_flip="promoted"
    )

    assert bootstrap.controller(args) == 0

    # Read BEFORE the flip: the baseline proof precedes the install, and the
    # value sent is the pre-flip build, not what `current` names afterwards.
    assert order == ["baseline", "install"], order
    (command,) = commands
    assert _flag(command, "--controller-tail") == str(promoted)
    assert _flag(command, "--controller-previous") == str(generation.resolve())
    assert _flag(command, "--controller-previous") != str(promoted.resolve())


def test_a_superseded_tail_is_not_sent_a_flag_its_release_may_not_know(
    destination, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A superseding current may be an OLDER release; an unknown flag exits 2."""
    root, generation = destination
    _order, commands, promoted, args = _fake_controller_world(
        monkeypatch, root, generation, current_after_flip="superseding"
    )

    assert bootstrap.controller(args) == 0

    (command,) = commands
    assert _flag(command, "--controller-tail") == str(promoted)
    assert "--controller-previous" not in command


def _run_tail(
    monkeypatch: pytest.MonkeyPatch, root: Path, argv: list[str], promoted: Path
) -> list[tuple[Path | None, Path | None]]:
    """Parse ``argv`` with the REAL CLI parser and run the real tail; return reclaims."""
    from local_operator import update
    from local_operator.cli import build_cli_parser

    reclaims: list[tuple[Path | None, Path | None]] = []
    monkeypatch.setattr(update, "current_generation", lambda: promoted)
    monkeypatch.setattr(update, "_is_generation_install", lambda: True)
    monkeypatch.setattr(update, "generations_dir", lambda: root / "generations")
    monkeypatch.setattr(update, "generation_version", lambda _gen: TARGET)
    monkeypatch.setattr(bootstrap.sys, "prefix", str(update._generation_install_root(promoted)))
    monkeypatch.setattr(update, "_print_current_generation", lambda: None)
    monkeypatch.setattr(
        update,
        "reclaim_after_promotion",
        lambda generation, *, previous=None, on_prune=None: reclaims.append((generation, previous)),
    )
    monkeypatch.setattr(update, "_run_services_tail", lambda services: update.StageReport())
    monkeypatch.setattr(update, "_roll_after_update", lambda **_kw: None)
    monkeypatch.setattr(update, "_emit_update_report", lambda **_kw: None)
    args = build_cli_parser().parse_args(argv)
    assert bootstrap.controller_tail(args) == 0
    return reclaims


def test_the_tail_forwards_controller_previous_to_the_reclaim(
    destination, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, generation = destination
    promoted = root / "generations" / "promoted"
    promoted.mkdir()
    argv = [
        "update",
        "--controller-tail",
        str(promoted),
        "--controller-target",
        TARGET,
        "--controller-previous",
        str(generation),
    ]

    reclaims = _run_tail(monkeypatch, root, argv, promoted)

    assert reclaims == [(promoted.resolve(), generation)], reclaims


def test_an_older_controller_without_previous_still_runs_the_tail(
    destination, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Backward compatibility: no flag means previous=None plus a debug line."""
    root, _generation = destination
    promoted = root / "generations" / "promoted"
    promoted.mkdir()
    argv = ["update", "--controller-tail", str(promoted), "--controller-target", TARGET]

    with caplog.at_level("DEBUG", logger=bootstrap.__name__):
        reclaims = _run_tail(monkeypatch, root, argv, promoted)

    assert reclaims == [(promoted.resolve(), None)], reclaims
    assert "no --controller-previous" in caplog.text
