"""A push must never publish content older than the hub's current head.

THE INCIDENT (2026-11-04, minerva-support-workspace revisions 12f5b146 and
a485be77): an operator pushed a fix from this machine, ran ``lop skills sync``
(which reported ``up-to-date``), edited the skill and pushed again -- and the hub
copy was reverted to the stale pre-fix text it held two revisions earlier.

The chain, reproduced here through the REAL ``lop skills sync`` / ``lop skills
push`` CLI arms against a stateful in-process hub:

1. A push advances the OUTBOUND receipt (A, P, revision) but deliberately leaves
   the inbound baseline B on the last INTEGRATED document (the old text).
2. The sync's genuine L == R agreement advanced B's ``remote_revision`` to the
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
from typing import Any, Mapping

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
