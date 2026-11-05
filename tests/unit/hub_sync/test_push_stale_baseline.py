"""A push must never publish content older than the hub's current head.

THE INCIDENT (2026-11-04, minerva-support-workspace revisions 12f5b146 and
a485be77): an operator pushed a fix from this machine, ran ``lop skills sync``
(which reported ``up-to-date``), edited the skill and pushed again -- and the hub
copy was reverted to the stale pre-fix text it held two revisions earlier.

The chain below is reproduced through the REAL ``lop skills push`` CLI arm
and the runner's inbound ``apply_one(auto=True)`` call against a stateful
in-process hub. The incident transcript establishes the on-disk stale pair;
this fixture derives one concrete path that creates it (not proof of which
background tick ran in the historic session):

1. A push advances the OUTBOUND receipt (A, P, revision) but deliberately leaves
   the inbound baseline B on the last INTEGRATED document (the old text).
2. The runner's genuine L == R agreement advanced B's ``remote_revision`` to the
   new head WITHOUT advancing B's fields -- B now claimed "revision X holds the
   old text".
3. The next check revalidated with ``If-None-Match: X``; the hub answered 304,
   and the check substituted B's (old) fields as the remote R.
4. The projection merged the local edit onto that stale R, and the CAS on X
   passed -- the hub accepted the stale document as its new head.

These tests drive the CLI the operator drove and assert on the HUB'S stored
document, not on the projection: the defect was only ever visible there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli
from local_operator.clients._http import NotModified
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync.push import RemoteRevisionChanged
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.hub_sync.skill_storage import ManagedSkillStorage

from .test_skill_storage import _input

STALE = "Slack is a fallback when authorized Bridge access is missing.\n"
FRESH = "Slack is **not** a fallback for agent coordination; hold instead.\n"


def _body(rule: str, extra: str = "") -> str:
    return f"# Coordination\n\nIntro paragraph.\n\n{rule}\nKeep links.\n{extra}"


class StatefulHub:
    """One skill row: a conditional GET (304 on a matching validator) plus the
    preview/CAS-commit publish surface, sharing ONE document and ETag.

    Mirrors the server contract the scout confirmed at agent-server 72923e3:
    the commit stores exactly the body it is sent, the CAS is on ``If-Match``,
    and ``GET /v1/skills/:id`` serves the head with its ETag.
    """

    #: The link's recorded ``hub_source``: a response is credited to the origin
    #: only when the serving client's base IS that source (``_served_by``).
    base_url = "https://example.invalid/v1"

    def __init__(self, document: SkillDocument, revision: str) -> None:
        self.document: dict[str, Any] = dict(document.to_json())
        self.revision = revision
        self.counter = 1
        self.commits: list[dict[str, Any]] = []
        self.gets: list[str | None] = []

    # -- the check/pull seam (a RadientClient-shaped client) -----------------

    def get_skill(
        self, skill_id: str, *, timeout: float = 0, if_none_match: str | None = None, **_kw: Any
    ) -> dict[str, Any]:
        self.gets.append(if_none_match)
        if if_none_match is not None and if_none_match == self.revision:
            raise NotModified(etag=self.revision)
        payload = json.loads(json.dumps(self.document))
        payload.update(
            {"revision": self.revision, "sync_contract": {"revision_cas": 1, "skills": 1}}
        )
        return payload

    # -- the push transport ----------------------------------------------------

    async def preview(self, *, document, expected_revision, budget):
        if expected_revision != self.revision:
            raise RemoteRevisionChanged()
        return {
            "revision": self.revision,
            "status": "ready",
            "document": dict(document),
            "unresolved": [],
            "pin": {"token": f"pin-{self.counter}"},
        }

    async def commit(
        self, *, document, preview_token, accept_unresolved, expected_revision, budget
    ):
        if expected_revision != self.revision:
            raise RemoteRevisionChanged()
        self.counter += 1
        self.revision = f'"revision-{self.counter}"'
        self.document = json.loads(json.dumps(dict(document)))
        self.commits.append(self.document)
        return {
            "revision": self.revision,
            "content_revision": self.counter,
            "accepted_document": json.loads(json.dumps(self.document)),
            "operation_id": f"srv-op-{self.counter}",
        }

    # -- assertions ---------------------------------------------------------------

    def skill_md(self) -> str:
        return next(f["content"] for f in self.document["files"] if f["path"] == "SKILL.md")


def _rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    identity, document, baseline = _input(_body(STALE))
    root = tmp_path / "profile"
    storage = ManagedSkillStorage(root)
    storage.apply(identity, document.name, document, baseline, expected_fingerprint=None)
    hub = StatefulHub(document, baseline.remote_revision or '"revision-1"')
    monkeypatch.setattr(
        "local_operator.hub_sync.service.build_clients_sync", lambda _cm: (lambda _t: hub, "ok")
    )
    monkeypatch.setattr(
        "local_operator.hub_sync.service.push_transport", lambda _ctx, _rec: (hub, None)
    )
    return root, storage, identity, document, hub


def _edit_skill_md(root: Path, name: str, transform) -> None:
    path = root / "skills" / name / "SKILL.md"
    path.write_text(transform(path.read_text(encoding="utf-8")), encoding="utf-8")


def _set_local_to_hub(root: Path, name: str, hub: StatefulHub) -> None:
    """The operator's "rebase onto the hub": local files := the hub's raw bytes."""

    for row in hub.document["files"]:
        (root / "skills" / name / row["path"]).write_text(row["content"], encoding="utf-8")


def _push(root: Path, name: str, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    code = cli.skills_command(
        argparse.Namespace(skills_command="push", name=name, yes=True, json=True), root
    )
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    payload["exit_code"] = code
    return payload


def _runner_pull(root: Path, name: str) -> str:
    """The runner tick's INBOUND pass for one link (``runner.py``: ``apply_one``
    with ``auto=True`` on every subscribed check, up-to-date ones included).

    This, not ``lop skills sync`` (which applies only ``available`` rows), is
    the path that met a genuine L == R agreement in the incident: the operator's
    running ``lop`` ticks it in the background.
    """

    from local_operator.config import ConfigManager

    ctx = svc.sync_context(ConfigManager(root))
    check = svc.find_check(svc.check_items(ctx, kinds=("skill",), names=[name]), "skill", name)
    assert check is not None
    return svc.apply_one(ctx, check, auto=True, allow_llm=False).outcome


def test_a_push_after_an_agreeing_sync_never_republishes_the_old_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE INCIDENT, end to end: fix -> push -> rebase onto hub -> runner tick
    (up-to-date) -> edit -> push."""

    root, _storage, identity, document, hub = _rig(tmp_path, monkeypatch)

    # 1. Fix the stale rule locally and publish it: the hub now holds FRESH.
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, FRESH))
    first = _push(root, document.name, capsys)
    assert first["state"] == "pushed", first
    assert FRESH in hub.skill_md()

    # 2. The operator rebases local onto the hub (byte-exact), and the runner's
    #    inbound tick then agrees (L == R): up-to-date.
    _set_local_to_hub(root, document.name, hub)
    assert _runner_pull(root, document.name) == "up-to-date"

    # 3. A small unrelated edit, pushed: the hub must keep FRESH.
    _edit_skill_md(root, document.name, lambda s: s + "\nAppendix line.\n")
    second = _push(root, document.name, capsys)

    assert second["state"] == "pushed", second
    assert STALE not in hub.skill_md(), "the push republished the stale pre-fix text"
    assert FRESH in hub.skill_md() and "Appendix line." in hub.skill_md()
    assert second["verified"] is True


def test_an_agreement_advances_the_baseline_fields_with_its_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The root cause in isolation: B's revision and B's fields move TOGETHER.

    A baseline whose ``remote_revision`` names a revision its ``fields`` do not
    hold turns every later 304 into a stale remote.
    """

    root, _storage, identity, document, hub = _rig(tmp_path, monkeypatch)
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, FRESH))
    assert _push(root, document.name, capsys)["state"] == "pushed"
    _set_local_to_hub(root, document.name, hub)

    assert _runner_pull(root, document.name) == "up-to-date"

    record = prov.read_baseline(root, "skill", identity.local_id)
    assert record is not None
    assert record.remote_revision == hub.revision
    stored = next(f["content"] for f in record.fields["files"] if f["path"] == "SKILL.md")
    assert FRESH in stored and STALE not in stored


def _poison(root: Path, identity: Any) -> None:
    """Recreate the PRE-FIX on-disk state: B's revision advanced, its fields not.

    Records written by a runtime without the fix still look like this on
    operators' machines, so the check must heal them rather than trust them.
    """

    from dataclasses import replace

    record = prov.read_baseline(root, "skill", identity.local_id)
    assert record is not None and record.outbound is not None
    prov.write_baseline(root, replace(record, remote_revision=record.outbound.remote_revision))


def test_a_poisoned_baseline_is_never_used_as_a_304_stand_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record whose receipt names the same revision with different content is
    proof B is wrong: the check reads the body instead of sending a validator."""

    root, _storage, identity, document, hub = _rig(tmp_path, monkeypatch)
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, FRESH))
    assert _push(root, document.name, capsys)["state"] == "pushed"
    _poison(root, identity)
    _edit_skill_md(root, document.name, lambda s: s + "\nAppendix line.\n")
    hub.gets.clear()

    second = _push(root, document.name, capsys)

    assert hub.gets[0] is None, "the check trusted a baseline that cannot be the hub's document"
    assert second["state"] == "pushed", second
    assert STALE not in hub.skill_md() and "Appendix line." in hub.skill_md()


def test_a_hub_that_moved_past_the_push_baseline_refuses_then_rebases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE GUARD (negative control): another writer moves the hub; the push
    refuses ``stale-base`` with the safe next step and publishes nothing; the
    baseline is rebased onto the head; the next push lands the local edit ON TOP
    of the other writer's change instead of reverting it."""

    root, _storage, identity, document, hub = _rig(tmp_path, monkeypatch)
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, FRESH))
    assert _push(root, document.name, capsys)["state"] == "pushed"
    base = hub.revision

    # Another machine publishes a change to a DIFFERENT paragraph.
    other = json.loads(json.dumps(hub.document))
    for row in other["files"]:
        if row["path"] == "SKILL.md":
            row["content"] = row["content"].replace("Intro paragraph.", "Intro, revised elsewhere.")
    hub.document = other
    hub.counter += 1
    hub.revision = f'"revision-{hub.counter}"'
    head = hub.revision
    commits_before = len(hub.commits)

    _edit_skill_md(root, document.name, lambda s: s + "\nAppendix line.\n")
    refused = _push(root, document.name, capsys)

    assert refused["state"] == "refused" and refused["reason"] == "stale-base", refused
    assert refused["exit_code"] == 1
    assert (refused["base_revision"], refused["head_revision"]) == (base, head)
    assert len(hub.commits) == commits_before, "a stale-base push must publish nothing"
    record = prov.read_baseline(root, "skill", identity.local_id)
    assert record is not None and record.outbound is not None
    assert record.outbound.remote_revision == head  # rebased onto the head

    # Human wording names both revisions and the safe next step.
    code = cli.skills_command(
        argparse.Namespace(skills_command="push", name=document.name, yes=True, json=False), root
    )
    human = capsys.readouterr().out
    assert code == 0, human  # the rebased baseline lets the push through now
    assert "Verified:" in human

    md = hub.skill_md()
    assert "Intro, revised elsewhere." in md, "the other writer's change was reverted"
    assert "Appendix line." in md and FRESH in md and STALE not in md


def test_the_stale_base_refusal_wording_names_the_next_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _storage, _identity, document, hub = _rig(tmp_path, monkeypatch)
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, FRESH))
    assert _push(root, document.name, capsys)["state"] == "pushed"
    hub.counter += 1
    hub.revision = f'"revision-{hub.counter}"'  # a revision-only move is still a move
    _edit_skill_md(root, document.name, lambda s: s + "\nAppendix line.\n")

    code = cli.skills_command(
        argparse.Namespace(skills_command="push", name=document.name, yes=True, json=False), root
    )
    human = capsys.readouterr().out

    assert code == 1
    assert "refused" in human and hub.revision in human
    assert f"lop skills sync --name {document.name} --check" in human
    assert f"lop skills push {document.name}" in human


def test_a_refused_merge_rebases_the_receipt_onto_the_head(tmp_path: Path) -> None:
    """After ``merge-refused`` the receipt's P and revision are the hub head the
    push read, and A (the captured local input) is kept so no local edit is lost."""

    from types import SimpleNamespace

    from local_operator.config import ConfigManager
    from local_operator.hub_sync.provenance import OutboundReceipt

    identity, document, baseline = _input(_body(STALE))
    root = tmp_path / "profile"
    ManagedSkillStorage(root).apply(
        identity, document.name, document, baseline, expected_fingerprint=None
    )
    captured = _input(_body(STALE, "local\n"))[1].to_json()
    receipt = OutboundReceipt(
        local_fields=captured,
        accepted_fields=document.to_json(),
        input_fingerprint=prov.fingerprint_snapshot("skill", captured),
        remote_revision='"revision-1"',
        operation_id=None,
    )
    from dataclasses import replace

    prov.write_baseline(root, replace(baseline, outbound=receipt))
    head_doc = _input(_body(FRESH))[1].to_json()
    check = SimpleNamespace(
        kind="skill", local_id=identity.local_id, remote=head_doc, remote_revision='"revision-9"'
    )
    ctx = svc.HubSyncContext(
        config_dir=root, config_manager=ConfigManager(root), client_for_tenant=lambda _t: None
    )

    moved = svc.rebase_outbound_after_refusal(
        ctx, check, SimpleNamespace(state="needs-review", reason="merge-refused")
    )

    stored = prov.read_baseline(root, "skill", identity.local_id)
    assert moved and stored is not None and stored.outbound is not None
    assert stored.outbound.remote_revision == '"revision-9"'
    assert stored.outbound.accepted_fields == head_doc
    assert stored.outbound.local_fields == captured
    # An unrelated refusal leaves the receipt alone.
    assert not svc.rebase_outbound_after_refusal(
        ctx, check, SimpleNamespace(state="needs-review", reason="generalization-needs-ack")
    )


class _RevertingHub(StatefulHub):
    """A hub whose stored head is NOT what the commit sent (the shape the
    incident had from the operator's point of view; also an idempotent replay
    answering with an older pin snapshot)."""

    async def commit(
        self, *, document, preview_token, accept_unresolved, expected_revision, budget
    ):
        result = await super().commit(
            document=document,
            preview_token=preview_token,
            accept_unresolved=accept_unresolved,
            expected_revision=expected_revision,
            budget=budget,
        )
        for row in self.document["files"]:
            if row["path"] == "SKILL.md":
                row["content"] = row["content"].replace(FRESH, STALE)
        return result


def _reverting_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, _storage, _identity, document, hub = _rig(tmp_path, monkeypatch)
    reverting = _RevertingHub(SkillDocument.from_json(hub.document), hub.revision)
    monkeypatch.setattr(
        "local_operator.hub_sync.service.build_clients_sync",
        lambda _cm: (lambda _t: reverting, "ok"),
    )
    monkeypatch.setattr(
        "local_operator.hub_sync.service.push_transport", lambda _ctx, _rec: (reverting, None)
    )
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, FRESH))
    return root, document


def test_the_push_reports_a_hub_head_that_differs_from_what_it_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, document = _reverting_rig(tmp_path / "json", monkeypatch)

    payload = _push(root, document.name, capsys)

    assert payload["state"] == "pushed"
    assert payload["verified"] is False and payload["verification_reason"] == "head-differs"
    assert payload["exit_code"] == 1

    root, document = _reverting_rig(tmp_path / "human", monkeypatch)
    code = cli.skills_command(
        argparse.Namespace(skills_command="push", name=document.name, yes=True, json=False), root
    )
    assert "does NOT match" in capsys.readouterr().out and code == 1


def test_a_replayed_acceptance_records_the_verified_head_not_the_old_pin_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An idempotent receipt may name an older revision than the current GET
    even when their documents agree. Ancestry records the GET's revision."""

    root, _storage, identity, document, hub = _rig(tmp_path, monkeypatch)
    original = hub.commit

    async def replayed_receipt(**kw: Any) -> dict[str, Any]:
        result = await original(**kw)
        result["revision"] = '"historic-pin-revision"'
        return result

    monkeypatch.setattr(hub, "commit", replayed_receipt)
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, FRESH))

    payload = _push(root, document.name, capsys)

    assert payload["verified"] is True
    assert payload["verified_revision"] == hub.revision
    stored = prov.read_baseline(root, "skill", identity.local_id)
    assert stored is not None and stored.outbound is not None
    assert stored.outbound.remote_revision == hub.revision


def test_a_mismatching_head_does_not_advance_the_outbound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, document = _reverting_rig(tmp_path, monkeypatch)
    payload = _push(root, document.name, capsys)
    assert payload["verified"] is False
    stored = prov.read_baseline(root, "skill", "immutable-local")
    assert stored is not None and stored.outbound is not None
    assert stored.outbound.remote_revision == '"revision-1"'


def test_unavailable_verification_is_not_claimed_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _storage, _identity, document, hub = _rig(tmp_path, monkeypatch)
    original = hub.get_skill

    def unavailable_after_commit(*args: Any, **kw: Any) -> dict[str, Any]:
        if hub.commits:
            raise ConnectionError("fixture outage")
        return original(*args, **kw)

    monkeypatch.setattr(hub, "get_skill", unavailable_after_commit)
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, FRESH))
    payload = _push(root, document.name, capsys)
    assert payload["verified"] is None and payload["exit_code"] == 1
    assert payload["verification_reason"] == "ConnectionError"


@pytest.mark.parametrize("base_revision", ['"older-head"', ""])
def test_the_shared_push_engine_refuses_stale_or_missing_bases_without_dispatch(
    tmp_path: Path, base_revision: str
) -> None:
    import asyncio

    from local_operator.hub_sync.budget import OperationBudget
    from local_operator.hub_sync.outbox import Outbox
    from local_operator.hub_sync.push import push_item

    document = _input(_body(FRESH))[1]
    hub = StatefulHub(document, '"current-head"')
    outcome = asyncio.run(
        push_item(
            kind="skill",
            local_id="throwaway",
            captured_local=document.to_json(),
            accepted=document.to_json(),
            current_local=document.to_json(),
            remote_latest=document.to_json(),
            expected_revision=hub.revision,
            base_revision=base_revision,
            transport=hub,
            outbox=Outbox(tmp_path / "profile"),
            operation_id="push-skill-throwaway",
            budget=OperationBudget.start("manual", "push-skill-throwaway", 1),
        )
    )
    assert outcome.state == "refused" and outcome.reason == "stale-base"
    assert hub.commits == []


def test_a_receipt_with_the_heads_revision_but_old_fields_is_refused_and_rebased(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace

    root, _storage, identity, document, hub = _rig(tmp_path, monkeypatch)
    _edit_skill_md(root, document.name, lambda s: s.replace(STALE, FRESH))
    assert _push(root, document.name, capsys)["verified"] is True
    record = prov.read_baseline(root, "skill", identity.local_id)
    assert record is not None and record.outbound is not None
    damaged = replace(record.outbound, accepted_fields=document.to_json())
    prov.write_baseline(root, replace(record, outbound=damaged))
    _edit_skill_md(root, document.name, lambda s: s + "\nAppendix line.\n")
    count = len(hub.commits)
    refused = _push(root, document.name, capsys)
    assert refused["reason"] == "stale-base" and len(hub.commits) == count
    stored = prov.read_baseline(root, "skill", identity.local_id)
    assert stored is not None and stored.outbound is not None
    assert stored.outbound.accepted_fields == hub.document


def test_an_inbound_refused_merge_refreshes_only_the_outbound_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, storage, identity, document, hub = _rig(tmp_path, monkeypatch)
    assert _runner_pull(root, document.name) == "up-to-date"  # seeds a receipt
    old = prov.read_baseline(root, "skill", identity.local_id)
    assert old is not None and old.outbound is not None
    path = root / "skills" / document.name / "scripts" / "inert.py"
    path.write_text("raise RuntimeError('local unpublished edit')\n", encoding="utf-8")
    hub.document["files"] = [f for f in hub.document["files"] if f["path"] != "scripts/inert.py"]
    hub.revision = '"revision-2"'
    assert _runner_pull(root, document.name) == "needs-review"
    current = prov.read_baseline(root, "skill", identity.local_id)
    assert current is not None and current.outbound is not None
    assert current.fields == old.fields and current.remote_revision == old.remote_revision
    assert current.outbound.remote_revision == hub.revision
    assert current.outbound.accepted_fields == hub.document
    assert "local unpublished edit" in path.read_text(encoding="utf-8")
