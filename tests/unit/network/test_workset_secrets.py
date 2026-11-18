"""Secret-shaped material in a move's working set (design §8.1; security ruling 2026-11-18).

The classify half (PR A): dirty/untracked files are judged by the harness's own path
denylist (``supplements/denylist.is_sensitive``) and credential-shape table
(``team_bridge/secret_guard.credential_classes``); a hit is listed as WITHHELD with its
rule or class — never its value. Outgoing history cannot be path-filtered, so a
secret-shaped path or added line in the unpushed commits leaves the WHOLE worktree
behind (``history_secret``), with no opt-in in v1, and history too large to scan is
treated the same (``history_unscannable``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.network import workset
from tests.unit.network.workset_rig import Rig, git, make_rig, point_session_at

SID = "a1b2c3d4e5f6"
#: AWS's documented example access key id — credential-SHAPED, not a credential.
AWS_SHAPED = "AKIA" + "IOSFODNN7EXAMPLE"


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    built = make_rig(tmp_path / "ws")
    monkeypatch.setenv("HOME", str(built.home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    point_session_at(tmp_path / "cfg" / "sessions" / SID, built.feat)
    return built


def _feat(rig: Rig) -> workset.WorktreeFacts:
    root = rig.base.parent / "cfg"
    manifest = workset.compute_manifest(root, SID, to_device="d_dest", home=rig.home)
    return {row.path: row for row in manifest.worktrees}["~/proj/.worktrees/feat"]


def test_secret_shaped_files_are_withheld_by_rule_and_class(rig: Rig) -> None:
    (rig.feat / "id_rsa").write_text("not a real key\n", encoding="utf-8")
    (rig.feat / "settings.py").write_text(f"KEY = '{AWS_SHAPED}'\n", encoding="utf-8")

    feat = _feat(rig)

    withheld = {item.path: (item.why, item.rule) for item in feat.excluded}
    assert withheld["~/proj/.worktrees/feat/.env.local"][0] == "secret_path"
    assert withheld["~/proj/.worktrees/feat/id_rsa"][0] == "secret_path"
    assert withheld["~/proj/.worktrees/feat/settings.py"][0] == "secret_content"
    assert withheld["~/proj/.worktrees/feat/settings.py"][1]  # a class label
    # Withheld files do not make the worktree stay behind: only they are excluded.
    assert feat.state == "planned"
    # Never the value: the block carries the class label, not the key.
    assert AWS_SHAPED not in json.dumps([item.to_dict() for item in feat.excluded])
    assert "~/proj/.worktrees/feat/notes.txt" not in withheld


def test_a_secret_in_unpushed_history_leaves_the_worktree_behind(rig: Rig) -> None:
    (rig.feat / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
    git(rig.feat, "add", "-f", ".env")
    git(rig.feat, "commit", "-q", "-m", "oops")

    feat = _feat(rig)

    assert (feat.state, feat.reason) == ("left_behind", "history_secret")
    assert ("~/proj/.worktrees/feat/.env", "history_secret_path") in {
        (item.path, item.why) for item in feat.excluded
    }


def test_secret_shaped_content_in_unpushed_history_leaves_it_behind(rig: Rig) -> None:
    (rig.feat / "config.py").write_text(f"KEY = '{AWS_SHAPED}'\n", encoding="utf-8")
    git(rig.feat, "add", "config.py")
    git(rig.feat, "commit", "-q", "-m", "config")
    (rig.feat / "config.py").write_text("KEY = None\n", encoding="utf-8")
    git(rig.feat, "commit", "-q", "-am", "scrub (the blob is still in the bundle)")

    feat = _feat(rig)

    assert (feat.state, feat.reason) == ("left_behind", "history_secret")
    assert "history_secret_content" in {item.why for item in feat.excluded}


@pytest.mark.parametrize("attribute", ["*.txt -diff", "*.txt binary", "*.txt diff=hide"])
def test_git_attributes_cannot_hide_history_content_from_the_scan(
    rig: Rig, attribute: str
) -> None:
    """Security S4 (round 1): ``.gitattributes`` turned the patch into "Binary files
    differ" (or a textconv rewrite) and the secret-shaped line was never scanned."""
    (rig.feat / ".gitattributes").write_text(attribute + "\n", encoding="utf-8")
    git(rig.feat, "config", "diff.hide.textconv", "true")
    (rig.feat / "notes.txt").unlink()  # the rig's untracked notes.txt: commit our own
    (rig.feat / "notes.txt").write_text(f"key {AWS_SHAPED}\n", encoding="utf-8")
    git(rig.feat, "add", ".gitattributes", "notes.txt")
    git(rig.feat, "commit", "-q", "-m", "hidden")

    feat = _feat(rig)

    assert "history_secret_content" in {item.why for item in feat.excluded}, feat.excluded
    assert (feat.state, feat.reason) == ("left_behind", "history_secret")


def test_history_too_large_to_scan_is_never_read_as_clean(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workset, "_HISTORY_DIFF_MAX_BYTES", 8)

    feat = _feat(rig)

    assert (feat.state, feat.reason) == ("left_behind", "history_unscannable")


def test_v1_has_no_history_secret_opt_in() -> None:
    """Security ruling 2026-11-18: no flag, kwarg, body field or checkbox."""
    import inspect

    from local_operator.network import mobility
    from local_operator.server.models.desktop_mesh import TransferSession

    assert "include_history_secrets" not in inspect.signature(mobility.request_move).parameters
    assert "include_history_secrets" not in TransferSession.model_fields
    assert "include_history_secrets" not in inspect.signature(workset.compute_manifest).parameters
