from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.hub_sync import provenance as prov


def test_fingerprints_match_the_legacy_agent_hash_and_ignore_line_endings() -> None:
    from local_operator.agents import hub_fingerprint

    # The legacy marker form is kept byte-identical so existing tags stay valid.
    assert prov.legacy_agent_fingerprint("a", "b") == hub_fingerprint("a", "b")
    assert prov.fingerprint_agent("x.\r\n\r\n", "d") == prov.fingerprint_agent("x.", "d")


def test_team_fingerprint_ignores_name_version_and_roster_order() -> None:
    a = prov.team_fields(
        description="d",
        manager="m",
        members=[{"role": "b"}, {"role": "A"}],
        instructions="i",
        project="p",
    )
    b = prov.team_fields(
        description="d",
        manager="m",
        members=[{"role": "A"}, {"role": "b"}],
        instructions="i",
        project="p",
    )
    assert prov.fingerprint_team(a) == prov.fingerprint_team(b)
    assert prov.normalize_roster([{"role": "x", "kind": "weird"}])[0]["kind"] == "agent"


def test_write_read_round_trip_and_atomic_no_temp_left(tmp_path: Path) -> None:
    rec = prov.make_record(
        "agent", "abc", "hub1", "org-1", {"instructions": "i", "description": "d"}, "pull"
    )
    prov.write_baseline(tmp_path, rec)
    got = prov.read_baseline(tmp_path, "agent", "abc")
    assert (
        got
        and got.hub_id == "hub1"
        and got.tenant_id == "org-1"
        and got.fields["instructions"] == "i"
    )
    assert [p.name for p in prov.baselines_dir(tmp_path).iterdir()] == ["agent-abc.json"]


def test_an_unusable_record_reads_as_absent_not_trusted(tmp_path: Path) -> None:
    path = prov.baseline_path(tmp_path, "team", "t1")
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert prov.read_baseline(tmp_path, "team", "t1") is None
    path.write_text(json.dumps({"schema": 99}))
    assert prov.read_baseline(tmp_path, "team", "t1") is None


def test_a_symlinked_record_or_directory_is_refused(tmp_path: Path) -> None:
    rec = prov.make_record(
        "agent", "abc", "h", None, {"instructions": "", "description": ""}, "pull"
    )
    prov.baselines_dir(tmp_path).mkdir(parents=True)
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    os.symlink(target, prov.baseline_path(tmp_path, "agent", "abc"))
    with pytest.raises(prov.BaselineError):
        prov.write_baseline(tmp_path, rec)
    assert target.read_text() == "{}"  # the link target was not written through


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a" * 200])
def test_ids_that_are_not_one_safe_path_segment_are_refused(tmp_path: Path, bad: str) -> None:
    with pytest.raises(prov.BaselineError):
        prov.baseline_path(tmp_path, "agent", bad)


def test_prune_drops_records_of_deleted_items_only(tmp_path: Path) -> None:
    for i in ("keep", "gone"):
        prov.write_baseline(
            tmp_path,
            prov.make_record(
                "team",
                i,
                "h",
                None,
                prov.team_fields(
                    description="", manager="m", members=[], instructions="i", project=""
                ),
                "pull",
            ),
        )
    assert prov.prune(tmp_path, "team", {"keep"}) == ["gone"]
    assert prov.read_baseline(tmp_path, "team", "keep") is not None


def test_backups_keep_only_the_newest_five_and_return_a_relative_path(tmp_path: Path) -> None:
    paths = [
        prov.write_backup(tmp_path, "agent", "abc", {"instructions": str(i)}, "t") for i in range(8)
    ]
    assert not Path(paths[0]).is_absolute() and paths[0].startswith("hub/backups/")
    assert len(list(prov.backups_dir(tmp_path).glob("agent-abc-*.json"))) == 5


def test_recording_never_raises_so_a_pull_cannot_fail_over_bookkeeping(tmp_path: Path) -> None:
    assert (
        prov.record_agent_baseline(
            tmp_path, local_id="../bad", hub_id="h", instructions="i", description="d"
        )
        is None
    )

    class T:  # noqa: D401 - a team with no hub id in its document
        id, name = "t1", "n"

    assert prov.record_team_pull(tmp_path, T(), {}, tenant_id="o") is None


def test_the_baseline_is_absent_from_an_agent_export_archive(tmp_path: Path) -> None:
    """A published archive can never plant a baseline: the record lives outside the agent dir."""

    import zipfile

    from local_operator.agents import AgentRegistry
    from tests.unit.tools.test_agent_tool import _edit_fields

    reg = AgentRegistry(tmp_path)
    row = reg.create_agent(_edit_fields(name="a", description="d", tags=["hub:abc"]))
    prov.record_agent_baseline(
        tmp_path, local_id=row.id, hub_id="abc", instructions="i", description="d"
    )
    zip_path, _name = reg.export_agent(row.id)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    assert names, "the archive must not be empty for this to prove anything"
    assert not any("baseline" in n or n.endswith(f"agent-{row.id}.json") for n in names)


def test_prune_keeps_a_record_the_caller_cannot_confirm_gone(tmp_path: Path) -> None:
    prov.write_baseline(
        tmp_path,
        prov.make_record(
            "team",
            "racing",
            "h",
            None,
            prov.team_fields(description="", manager="m", members=[], instructions="i", project=""),
            "pull",
        ),
    )
    assert prov.prune(tmp_path, "team", set(), confirmed_absent=lambda _id: False) == []
    assert prov.read_baseline(tmp_path, "team", "racing") is not None
    assert prov.prune(tmp_path, "team", set(), confirmed_absent=lambda _id: True) == ["racing"]
