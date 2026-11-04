"""A late census is a deletion veto, never merely an attribution opportunity.

Only synthetic generation directories are created; removal/attestation are
spies and cannot delete a tree or write a runtime marker.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from local_operator import update
from local_operator.network import meshupdate


@pytest.fixture
def candidates(tmp_path, monkeypatch):
    root = tmp_path / "generations"
    paths = [root / name for name in ("old-a", "old-b")]
    for path in paths:
        marker = path / "tools/local-operator/.lop-source"
        marker.parent.mkdir(parents=True)
        marker.write_text("synthetic main\n")
    monkeypatch.setattr(update, "generations_dir", lambda: root)
    monkeypatch.setattr(update, "current_generation", lambda: None)
    monkeypatch.setattr(update, "pointer_path", lambda: tmp_path / "current")
    monkeypatch.setattr(update, "_running_generation", lambda: None)
    monkeypatch.setattr(update, "_sweep_staging_links", lambda moment: None)
    monkeypatch.setattr(update, "_config_roots", lambda: [tmp_path])
    remove = Mock(return_value=True)
    attest = Mock(return_value=[])
    monkeypatch.setattr(update, "_remove_tree", remove)
    monkeypatch.setattr(update, "note_doomed_runtimes", attest)
    return paths, remove, attest


def test_new_reference_vetoes_removal_and_attestation(candidates, monkeypatch):
    paths, remove, attest = candidates
    record = SimpleNamespace(install_root=str(paths[0] / "tools/local-operator"))
    monkeypatch.setattr(
        update, "_records_under", lambda root: ([(record, record.install_root)], ())
    )
    plan = update.prune_generations(keep=0, referenced=[])
    assert paths[0] not in plan.removed
    assert all(call.args[0] != paths[0] for call in remove.call_args_list)
    assert all(call.args[0] != paths[0] for call in attest.call_args_list)


def test_new_census_gap_vetoes_removal(candidates, monkeypatch):
    paths, remove, attest = candidates
    monkeypatch.setattr(
        update, "_records_under", lambda root: ([], ("synthetic unreadable record",))
    )
    plan = update.prune_generations(keep=0, referenced=[])
    assert not plan.removed
    assert not plan.references_complete
    remove.assert_not_called()
    attest.assert_not_called()


def test_each_candidate_gets_fresh_reference_read(candidates, monkeypatch):
    paths, remove, attest = candidates
    calls = 0
    record = SimpleNamespace(prefix=str(paths[1] / "tools/local-operator"))

    def records(root):
        nonlocal calls
        calls += 1
        return ([], ()) if calls == 1 else ([(record, record.prefix)], ())

    monkeypatch.setattr(update, "_records_under", records)
    plan = update.prune_generations(keep=0, referenced=[])
    assert plan.removed == (paths[0],)
    assert calls == 2
    remove.assert_called_once_with(paths[0])


def test_mesh_promotion_does_not_collect_generations():
    module = SimpleNamespace(
        install_into_generation=Mock(),
        prune_generations=Mock(),
        referenced_install_roots=Mock(),
        prune_notice_lines=Mock(return_value=[]),
        ACTOR_UPGRADE="synthetic",
        UpdateError=RuntimeError,
    )
    meshupdate._install_generation("0.0.0", runner=Mock(), update_mod=module)
    module.install_into_generation.assert_called_once()
    module.prune_generations.assert_not_called()
    module.referenced_install_roots.assert_not_called()
