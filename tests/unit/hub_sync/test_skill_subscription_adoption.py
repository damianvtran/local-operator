"""Direct Subscribe must not pretend an occupying authored folder is absent.

The operator's common path is Subscribe, not Download followed by a policy edit.
An unadopted copy must reach the same reviewed adoption transition, and a failed
subscribe must leave ALL baseline/policy bytes intact. These drive real services
and storage in isolated profiles; only the Hub fetch/model boundaries are fakes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.identity import LinkError, LinkIdentity

from .test_skill_adoption import FaithfulModel, _authored, _bytes, _install, _municipal
from .test_skill_state import ALIAS, _record, _rig, _subscribe


def _link_policy_bytes(root: Path) -> dict[str, bytes]:
    paths = [root / "config.yml", *(root / "hub" / "baselines").glob("*.json")]
    return {str(path.relative_to(root)): path.read_bytes() for path in paths if path.is_file()}


def test_direct_subscribe_occupied_equal_requires_review_without_writes(tmp_path: Path) -> None:
    ctx, manager, _remote, base, document = _rig(tmp_path)
    folder = _authored(manager.config_dir, document)
    before = _link_policy_bytes(manager.config_dir)
    original = _bytes(folder)
    with pytest.raises(LinkError) as refused:
        _subscribe(ctx, base)
    assert refused.value.code == "alias-occupied"
    assert _link_policy_bytes(manager.config_dir) == before
    assert _bytes(folder) == original


@pytest.mark.parametrize("different", [False, True])
def test_subscribe_intent_survives_review_and_confirmation(tmp_path: Path, different: bool) -> None:
    ctx, manager, remote, base, original_document = _rig(tmp_path)
    if different:
        folder, document, _local, _hub = _municipal(manager.config_dir)
        remote.document = document.to_json()
        ctx.resolver = FaithfulModel()
    else:
        folder = _authored(manager.config_dir, original_document)
    before = _link_policy_bytes(manager.config_dir)
    original = _bytes(folder)
    with pytest.raises(LinkError) as refused:
        _subscribe(ctx, base)
    assert refused.value.code == "alias-occupied"
    preview = _install(ctx, base, dry_run=True, subscribe=True, auto_push=False)
    assert preview["adoption"]["state"] in ("equal", "ready")
    # Cancel is simply leaving this preview: no initial subscribe, preview or
    # cache projection may fabricate a link before the reviewed confirmation.
    assert _link_policy_bytes(manager.config_dir) == before
    assert _bytes(folder) == original
    result = _install(
        ctx,
        base,
        confirm=True,
        adoption_receipt=preview["adoption_receipt"],
        subscribe=True,
        auto_push=False,
    )
    assert result["link"]["policy"]["subscribe"] is True
    assert result["link"]["policy"]["auto_push"] is False
    assert _record(manager.config_dir).content_state == "present"
    if not different:
        assert _bytes(folder) == original


def test_subscribe_other_owner_refuses_before_any_link_or_policy_write(tmp_path: Path) -> None:
    ctx, manager, _remote, base, document = _rig(tmp_path)
    folder = _authored(manager.config_dir, document)
    foreign = LinkIdentity("skill", "other-local", base, "public", None, "other-hub")
    baseline = replace(
        prov.make_record("skill", foreign.local_id, foreign.hub_id, None, document, "pull"),
        identity=foreign,
        local_alias=ALIAS,
        lineage="clean",
    )
    prov.write_baseline(manager.config_dir, baseline)
    before = _link_policy_bytes(manager.config_dir)
    original = _bytes(folder)
    with pytest.raises(LinkError) as refused:
        _subscribe(ctx, base)
    assert refused.value.code == "origin-conflict"
    assert _link_policy_bytes(manager.config_dir) == before
    assert _bytes(folder) == original


def test_prior_absent_subscription_does_not_claim_newly_authored_copy(tmp_path: Path) -> None:
    ctx, manager, _remote, base, document = _rig(tmp_path)
    _subscribe(ctx, base)
    folder = _authored(manager.config_dir, document)
    before = _link_policy_bytes(manager.config_dir)
    original = _bytes(folder)
    with pytest.raises(LinkError) as refused:
        _subscribe(ctx, base)
    assert refused.value.code == "alias-occupied"
    assert _link_policy_bytes(manager.config_dir) == before
    assert _bytes(folder) == original


@pytest.mark.parametrize("change", ["local", "remote"])
def test_direct_subscribe_preview_invalidates_on_either_side_change(
    tmp_path: Path, change: str
) -> None:
    ctx, manager, remote, base, document = _rig(tmp_path)
    folder = _authored(manager.config_dir, document)
    with pytest.raises(LinkError):
        _subscribe(ctx, base)
    preview = _install(ctx, base, dry_run=True, subscribe=True, auto_push=False)
    if change == "local":
        path = folder / "SKILL.md"
        path.write_text(path.read_text() + "\nA concurrent clerk safety instruction.\n")
    else:
        remote.document = {
            **remote.document,
            "files": [dict(row) for row in remote.document["files"]],
        }
        remote.document["files"][0]["content"] += "\nNew Hub instruction.\n"
    before = _link_policy_bytes(manager.config_dir)
    original = _bytes(folder)
    with pytest.raises(LinkError) as refused:
        _install(
            ctx,
            base,
            confirm=True,
            adoption_receipt=preview["adoption_receipt"],
            subscribe=True,
            auto_push=False,
        )
    assert refused.value.code == "adoption-preview-changed"
    assert _link_policy_bytes(manager.config_dir) == before
    assert _bytes(folder) == original
