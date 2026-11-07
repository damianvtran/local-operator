"""Two hub-sync defects measured against the prod hub on 2026-11-07.

1. ``lop skills sync --name <skill> --json`` with an edit/edit text conflict in
   ONE file said "an edit against a removal" (there was none) and exited 0, and
   the CLI had no way to settle the conflict by choosing a side -- while agents
   cannot hand-edit the managed skills directory.
2. ``lop skills push`` sent the merge engine's NORMALIZED spelling of each file
   (trailing newline dropped, blank lines collapsed) and then verified the hub
   head with a whitespace-tolerant comparator, so it certified that the hub held
   "exactly" a document it did not.

Both are driven through the real ``lop skills`` CLI arms against the stateful
in-process hub (which, like agent-server, stores exactly the body it is sent).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli
from local_operator.hub_sync.merge import MergeOptions
from local_operator.hub_sync.outbound import (
    merge_skill_document,
    merge_skill_file,
    project_outbound,
    skill_merge_refusal_message,
)
from local_operator.hub_sync.skill_document import SkillDocument, SkillFile

from .test_push_stale_baseline import STALE, _edit_skill_md, _push, _rig

#: A same-sentence rewrite on both sides that the additive fallback cannot
#: combine (each side's rewrite drops words the other keeps): a genuine
#: edit/edit text conflict, with no file removed anywhere.
LOCAL_RULE = "Slack is a fallback only when the operator approves it in writing.\n"
HUB_RULE = "Bridge replaced Slack entirely; never message there again.\n"


def _sync(root: Path, name: str | None, capsys: pytest.CaptureFixture[str], **flags: Any):
    args = argparse.Namespace(
        skills_command="sync",
        name=name,
        all=False,
        check=flags.get("check", False),
        yes=False,
        json=flags.get("json", True),
        prefer=flags.get("prefer"),
    )
    code = cli.skills_command(args, root)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _conflicted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, storage, identity, document, hub = _rig(tmp_path, monkeypatch)
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, LOCAL_RULE))
    hub.document["files"] = [
        (
            {**row, "content": row["content"].replace(STALE, HUB_RULE)}
            if row["path"] == "SKILL.md"
            else row
        )
        for row in hub.document["files"]
    ]
    hub.revision = '"revision-hub-edit"'
    return root, document, hub


def _local_skill_md(root: Path, name: str) -> str:
    return (root / "skills" / name / "SKILL.md").read_text(encoding="utf-8")


# -- bug 1: the refusal names the real conflict, exits non-zero, and resolves -------


def test_an_edit_edit_conflict_is_named_as_one_and_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, document, _hub = _conflicted(tmp_path, monkeypatch)

    code, out, _err = _sync(root, document.name, capsys)

    payload = json.loads(out)
    (item,) = payload["reports"]
    assert item["outcome"] == "needs-review" and item["error_class"] == "merge-refused"
    assert item["message"] == (
        "the merge could not combine SKILL.md "
        "(edit/edit text conflict: both sides changed the same text)"
    )
    assert "removal" not in item["message"]
    # Two runnable commands: a pasted ``local|remote`` would be a shell pipe.
    assert payload["next_steps"] == [
        f"lop skills sync --name {document.name} --prefer local",
        f"lop skills sync --name {document.name} --prefer remote",
    ]
    assert code == 1
    assert LOCAL_RULE in _local_skill_md(root, document.name)  # nothing written


def test_the_human_output_names_the_file_and_the_way_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, document, _hub = _conflicted(tmp_path, monkeypatch)

    code, out, _err = _sync(root, document.name, capsys, json=False)

    assert code == 1
    assert "SKILL.md (edit/edit text conflict" in out
    assert f"lop skills sync --name {document.name} --prefer local\n" in out
    assert f"lop skills sync --name {document.name} --prefer remote\n" in out
    assert "local|remote" not in out


@pytest.mark.parametrize("side", ["remote", "local"])
def test_prefer_takes_the_chosen_sides_whole_file_and_settles_the_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    side: str,
) -> None:
    """Q1 + Q2 (PR #3574 round 1): a whole-file choice that stays settled.

    The engine reads the hub's rewrite as "remove the old sentence + add a new
    one"; passage-level precedence kept the hub's added sentence under
    ``--prefer local`` (two contradictory rules). And when the result equalled
    the local copy, nothing recorded the decision: the next sync re-raised the
    conflict and a push was refused as hub-moved.
    """

    root, document, hub = _conflicted(tmp_path, monkeypatch)
    # Non-conflicting files on each side must survive either choice.
    (root / "skills" / document.name / "references").mkdir(exist_ok=True)
    (root / "skills" / document.name / "references" / "local.md").write_text("mine\n")
    hub.document["files"].append({"path": "references/hub.md", "content": "theirs\n"})
    local_md = _local_skill_md(root, document.name)
    hub_md = hub.skill_md()
    assert _sync(root, document.name, capsys)[0] == 1

    code, out, _err = _sync(root, document.name, capsys, prefer=side)

    (item,) = json.loads(out)["reports"]
    assert code == 0, item
    assert item["outcome"] == "merged" and item["applied"] is True, item
    on_disk = _local_skill_md(root, document.name)
    assert on_disk == (local_md if side == "local" else hub_md)  # exact bytes, whole file
    assert (HUB_RULE in on_disk) is (side == "remote")
    assert (LOCAL_RULE in on_disk) is (side == "local")
    references = root / "skills" / document.name / "references"
    assert (references / "local.md").read_text() == "mine\n"
    assert (references / "hub.md").read_text() == "theirs\n"
    # Settled: the next plain sync has nothing left to decide ...
    assert _sync(root, document.name, capsys)[0] == 0
    # ... and a push is no longer refused as hub-moved.
    payload = _push(root, document.name, capsys)
    # Either way the local-only ``references/local.md`` is still unpublished, so
    # the push publishes; SKILL.md lands as the chosen side's exact bytes.
    assert payload["exit_code"] == 0, payload
    assert payload["state"] == "pushed" and payload["verified"] is True
    assert hub.skill_md() == (local_md if side == "local" else hub_md)
    assert {"references/local.md", "references/hub.md"} <= {
        r["path"] for r in hub.document["files"]
    }


@pytest.mark.parametrize("removed_on", ["hub", "local"])
@pytest.mark.parametrize("side", ["remote", "local"])
def test_prefer_settles_an_edit_against_a_removal_both_ways(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    removed_on: str,
    side: str,
) -> None:
    root, _storage, _identity, document, hub = _rig(tmp_path, monkeypatch)
    inert = root / "skills" / document.name / "scripts" / "inert.py"
    original = inert.read_text()
    if removed_on == "hub":
        inert.write_text(original.replace("never run", "never load"))
        hub.document["files"] = [
            r for r in hub.document["files"] if r["path"] != "scripts/inert.py"
        ]
    else:
        inert.unlink()
        for row in hub.document["files"]:
            if row["path"] == "scripts/inert.py":
                row["content"] = original.replace("never run", "never load")
    hub.revision = '"revision-hub-edit"'
    code, out, _err = _sync(root, document.name, capsys)
    assert code == 1 and "edit/removal conflict" in json.loads(out)["reports"][0]["message"]

    code, out, _err = _sync(root, document.name, capsys, prefer=side)

    assert code == 0, out
    # The chosen side's state of the file wins: present-and-edited or absent.
    keeps_file = (side == "local") == (removed_on == "hub")
    assert inert.exists() is keeps_file
    if keeps_file:
        assert "never load" in inert.read_text()
    assert _sync(root, document.name, capsys)[0] == 0


def test_an_unreachable_hub_under_prefer_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R3: ``unavailable`` settled nothing, so it is not exit 0."""

    from local_operator.hub_sync import service as svc

    root, document, _hub = _conflicted(tmp_path, monkeypatch)
    monkeypatch.setattr(
        svc,
        "apply_items",
        lambda *_a, **_k: svc.ApplyReport(
            (
                svc.ItemMergeReport(
                    "skill",
                    "immutable-local",
                    document.name,
                    "remote",
                    "unavailable",
                    error_class="network",
                ),
            )
        ),
    )

    assert _sync(root, document.name, capsys, prefer="local")[0] == 1


@pytest.mark.parametrize("flags", [{"prefer": "remote"}, {"prefer": "local", "check": True}])
def test_prefer_without_a_named_apply_is_refused_before_any_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: dict[str, Any],
) -> None:
    root, document, _hub = _conflicted(tmp_path, monkeypatch)
    name = document.name if flags.get("check") else None

    code, out, err = _sync(root, name, capsys, **flags)

    assert code == 1 and out == ""
    assert "--prefer needs --name" in err
    assert LOCAL_RULE in _local_skill_md(root, document.name)


def test_prefer_parses_on_skills_sync() -> None:
    parser = cli.build_cli_parser()
    args = parser.parse_args(["skills", "sync", "--name", "x", "--prefer", "remote", "--json"])
    assert args.prefer == "remote" and args.name == "x"
    with pytest.raises(SystemExit):
        parser.parse_args(["skills", "sync", "--name", "x", "--prefer", "hub"])


def _doc(**files: str | None) -> dict[str, Any]:
    skill = "---\nname: demo\ndescription: d\n---\n\nbody\n"
    rows = [SkillFile("SKILL.md", skill)] + [
        SkillFile(path.replace("__", "/").replace("_md", ".md"), content)
        for path, content in files.items()
        if content is not None
    ]
    return SkillDocument("demo", "d", tuple(rows)).to_json()


def test_the_refusal_message_names_each_conflict_kind() -> None:
    base = _doc(references__a_md="# A\n\nold text here\n", references__b_md="# B\n\nkeep me\n")
    local = _doc(
        references__a_md="# A\n\nThe token lives in LD_TOKEN and expires daily.\n",
        references__b_md="# B\n\nkeep me, edited locally\n",
        references__c_md="LOCAL\n",
    )
    remote = _doc(
        references__a_md="# A\n\nThe secret is ADMIN_KEY and never expires at all.\n",
        references__b_md=None,
        references__c_md="REMOTE\n",
    )

    merged, results = merge_skill_document(base, local, remote, opts=MergeOptions(allow_llm=False))

    assert merged is None
    message = skill_merge_refusal_message(base, local, remote, results)
    assert message == (
        "the merge could not combine "
        "references/a.md (edit/edit text conflict: both sides changed the same text); "
        "references/b.md (edit/removal conflict: removed on the hub, edited in this copy); "
        "references/c.md (add/add conflict: both sides added this file with different content)"
    )


def test_a_clean_merge_that_is_not_a_valid_document_is_not_called_a_conflict() -> None:
    assert (
        skill_merge_refusal_message(_doc(), _doc(), _doc(), [])
        == "the merged skill document is not valid"
    )


@pytest.mark.parametrize(("side", "kept"), [("local", "LOCAL\n"), ("remote", "REMOTE\n")])
def test_prefer_also_settles_an_add_add_file_conflict(side: str, kept: str) -> None:
    content, result = merge_skill_file(
        "references/c.md",
        base=None,
        local="LOCAL\n",
        remote="REMOTE\n",
        opts=MergeOptions(allow_llm=False, prefer=side),  # type: ignore[arg-type]
        base_present=False,
    )
    assert content == kept and result.outcome in ("merged", "unchanged")


# -- bug 2: a push publishes the exact bytes and verifies them exactly ---------------


def test_a_push_publishes_the_exact_bytes_and_verifies_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _storage, _identity, document, hub = _rig(tmp_path, monkeypatch)
    # A trailing newline, trailing spaces and a blank-line run inside a fence:
    # every one of them is the operator's content, not a merge artifact.
    fence = "\n```text\nline one   \n\n\nline two\n```\n"
    _edit_skill_md(root, document.name, lambda s: s + fence)
    local = _local_skill_md(root, document.name)
    assert local.endswith("```\n")

    payload = _push(root, document.name, capsys)

    assert payload["state"] == "pushed" and payload["exit_code"] == 0
    assert hub.skill_md() == local  # byte-for-byte, trailing newline included
    assert payload["verified"] is True


def test_verification_is_byte_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hub head differing only by a dropped trailing newline is NOT "exactly" it."""

    root, _storage, _identity, document, hub = _rig(tmp_path, monkeypatch)
    _edit_skill_md(root, document.name, lambda s: s + "\nMore.\n")
    original = hub.commit

    async def stripping_commit(**kw: Any) -> dict[str, Any]:
        result = await original(**kw)
        for row in hub.document["files"]:
            row["content"] = row["content"].rstrip("\n")
        return result

    monkeypatch.setattr(hub, "commit", stripping_commit)

    payload = _push(root, document.name, capsys)

    assert payload["state"] == "pushed"
    assert payload["verified"] is False and payload["verification_reason"] == "head-differs"
    assert payload["exit_code"] == 1


def test_a_whitespace_only_difference_from_the_hub_is_still_a_no_op() -> None:
    """Exact bytes must not turn every LF-less hub copy into a billed republish."""

    with_lf = _doc(references__a_md="# A\n\ntext\n")
    without_lf = {
        **with_lf,
        "files": [{**row, "content": row["content"].rstrip("\n")} for row in with_lf["files"]],
    }

    result = project_outbound(
        "skill",
        captured_local=with_lf,
        accepted=without_lf,
        current_local=with_lf,
        remote_latest=without_lf,
    )

    assert result.no_op is True


def _hub_rejoin_name(content: str, name: str) -> str:
    """The agent-server SKILL.md rewrite (SplitSkillName + JoinSkillName), test-side.

    Written independently of the product's ``_without_hub_name_line`` so the test
    cannot share its bug: the top-level ``name:`` line is lifted out of the
    frontmatter and ``name: <name>\\n`` (bare LF) is inserted right after the
    opening ``---`` line.
    """

    first = content.index("\n") + 1
    head, rest = content[:first], content[first:]
    lines = rest.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("---"):
            break
        if line.startswith("name:"):
            del lines[i]
            break
    return head + f"name: {name}\n" + "\n".join(lines)


@pytest.mark.parametrize("variant", ["crlf", "name-trailing-spaces", "name-not-first"])
def test_verification_tolerates_only_the_hubs_name_line_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variant: str,
) -> None:
    """R1 (PR #3574 round 1): the hub re-renders SKILL.md's name line; nothing else."""

    root, _storage, _identity, document, hub = _rig(tmp_path, monkeypatch)
    path = root / "skills" / document.name / "SKILL.md"
    text = path.read_text(encoding="utf-8") + "\nMore.\n"
    if variant == "crlf":
        text = text.replace("\n", "\r\n")
    elif variant == "name-trailing-spaces":
        text = text.replace(f"name: {document.name}\n", f"name: {document.name}   \n", 1)
    else:  # the name line moved to the end of the frontmatter block
        text = text.replace(f"name: {document.name}\n", "", 1)
        lines = text.split("\n")
        close = lines.index("---", 1)
        lines.insert(close, f"name: {document.name}")
        text = "\n".join(lines)
    path.write_bytes(text.encode("utf-8"))
    original = hub.commit

    async def rejoining_commit(**kw: Any) -> dict[str, Any]:
        result = await original(**kw)
        for row in hub.document["files"]:
            if row["path"] == "SKILL.md":
                row["content"] = _hub_rejoin_name(row["content"], document.name)
        return result

    monkeypatch.setattr(hub, "commit", rejoining_commit)

    payload = _push(root, document.name, capsys)

    assert payload["state"] == "pushed"
    assert hub.skill_md() != text  # the hub really rewrote the name line
    assert payload["verified"] is True and payload["exit_code"] == 0, payload


def test_verification_still_reports_real_drift_beside_the_name_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _storage, _identity, document, hub = _rig(tmp_path, monkeypatch)
    _edit_skill_md(root, document.name, lambda s: s + "\nMore.\n")
    original = hub.commit

    async def drifting_commit(**kw: Any) -> dict[str, Any]:
        result = await original(**kw)
        for row in hub.document["files"]:
            if row["path"] == "SKILL.md":
                row["content"] = _hub_rejoin_name(row["content"], document.name).replace(
                    "More.", "Less."
                )
        return result

    monkeypatch.setattr(hub, "commit", drifting_commit)

    payload = _push(root, document.name, capsys)

    assert payload["verified"] is False and payload["verification_reason"] == "head-differs"
    assert payload["exit_code"] == 1
