"""CLI hub publish preview/confirm (design p2p3 §7.3; public-teams design §9 PR 2).

What this pins: the parser surface (`teams push --id`/`--public`, `agents push
--hub-id`, the five shared flags), the preview -> diff -> confirm -> commit
flow and its exit codes (0 published, 1 refusal, 2 preview-only with changes
pending, 3 declined), the consent gate (`--allow-internal-ops`), the
republish-side allowance warning, the old-server compat path, the
`preview_expired`/`preview_stale` one-retry rule, the public `teams pull`/
`teams search` arms, and that reference VALUES never reach a log record.

The hub is faked at the two seams the CLI owns -- ``RadientClient`` and the
``radient_credentials`` resolvers; the wire behavior is the client suite's.
Scripted ``input()`` answers drive the interactive cells (``isatty`` is forced
True); the non-TTY cells force it False.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from local_operator.cli import _hub_cause, build_cli_parser, main
from local_operator.clients._http import APIError  # noqa: F401  # used below


@pytest.fixture
def parser():
    return build_cli_parser()


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() so no test touches the real ~/.local-operator."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def quiet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_operator.cli.setup_cross_platform_environment", lambda: None)


def _preview(
    document: Dict[str, Any],
    *,
    mode: str = "create",
    status: str = "unchanged",
    changes: Optional[List[Dict[str, Any]]] = None,
    unresolved: Optional[List[Dict[str, Any]]] = None,
    resolution: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The hub's preview object at the shape the client parses (p2p3 §4.1)."""
    unresolved = unresolved or []
    return {
        "status": status,
        "document": document,
        "changes": changes or [],
        "unresolved": unresolved,
        "advisories": [],
        "resolution": resolution or {"mode": mode, "removed": [], "added": []},
        "review": "not_run",
        "pin": {
            "token": "pin-fixture",
            "expires_at": "2030-01-01T00:00:00Z",
            "commit_header": "X-Radient-Preview-Token",
            "accept_header": "X-Radient-Accept-Unresolved",
            "accept_ids": [str(item.get("id")) for item in unresolved],
        },
        "transform": {"version": "gen-v1", "model": "stub"},
    }


_CHANGES = [
    {
        "field": "instructions",
        "class": "person",
        "original": "Damian Tran",
        "placeholder": "[PERSON_1]",
        "occurrences": 3,
    }
]
_UNRESOLVED = [
    {
        "id": "u1",
        "field": "project",
        "value": "Q4 pilot",
        "reason": "may be a project code name",
    }
]


class _Hub:
    """A scriptable hub: queues for previews/commits, rows for the republish reads."""

    def __init__(self) -> None:
        self.team_rows: Dict[str, Dict[str, Any]] = {}
        self.agent_rows: Dict[str, Dict[str, Any]] = {}
        #: Each item: a preview dict, or an Exception to raise.
        self.preview_queue: List[Any] = []
        self.commit_queue: List[Any] = []
        #: (kind, id-or-None, request-kwargs) for every preview call.
        self.preview_calls: List[Tuple[str, Any, Dict[str, Any]]] = []
        #: (kind, id-or-None, request-kwargs) for every commit call.
        self.commit_calls: List[Tuple[str, Any, Dict[str, Any]]] = []
        self.memberships: List[Dict[str, Any]] = [
            {
                "tenant_id": "org-a",
                "tenant_name": "Org A",
                "role": "admin",
                "status": "active",
                "is_home": False,
                "plan": {"status": "active", "seats": 2},
            }
        ]
        self.team_documents: Dict[str, Dict[str, Any]] = {}
        self.public_teams: List[Dict[str, Any]] = []
        self.get_team_calls: List[Tuple[str, bool]] = []

    # -- reads -----------------------------------------------------------------
    def list_memberships(self) -> List[Dict[str, Any]]:
        return self.memberships

    def get_team(self, team_id: str, *, with_credential: bool = True, **_kw: Any) -> Dict[str, Any]:
        self.get_team_calls.append((team_id, with_credential))
        if team_id not in self.team_rows:
            raise APIError("Team not found.", status_code=404, code="team_not_found")
        return self.team_rows[team_id]

    def get_agent(self, agent_id: str, **_kw: Any) -> Optional[Dict[str, Any]]:
        row = self.agent_rows.get(agent_id)
        if row is None:
            return None
        return {"msg": "Agent retrieved successfully", "result": row}

    def list_public_teams(self, *, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        return {
            "page": page,
            "per_page": per_page,
            "total_pages": 1,
            "total_records": len(self.public_teams),
            "records": list(self.public_teams),
        }

    # -- previews --------------------------------------------------------------
    def _preview(self, kind: str, ident: Any, document: Any, visibility: Any, tenant_id: Any):
        self.preview_calls.append(
            (
                kind,
                ident,
                {"document": document, "visibility": visibility, "tenant_id": tenant_id},
            )
        )
        if self.preview_queue:
            item = self.preview_queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return _preview(document, mode="overwrite" if ident else "create")

    def preview_publish_team_document(self, document, *, visibility=None, tenant_id=None):
        return self._preview("team-create", None, document, visibility, tenant_id)

    def preview_republish_team_document(
        self, team_id, document, *, visibility=None, tenant_id=None
    ):
        return self._preview("team-republish", team_id, document, visibility, tenant_id)

    def preview_publish_agent_instruction_set(self, document, *, visibility=None, tenant_id=None):
        return self._preview("agent-create", None, document, visibility, tenant_id)

    def preview_republish_agent_instruction_set(
        self, agent_id, document, *, visibility=None, tenant_id=None
    ):
        return self._preview("agent-republish", agent_id, document, visibility, tenant_id)

    # -- commits ---------------------------------------------------------------
    def _commit(self, kind: str, ident: Any, document: Any, **kwargs: Any):
        self.commit_calls.append((kind, ident, {"document": document, **kwargs}))
        if self.commit_queue:
            item = self.commit_queue.pop(0)
            if isinstance(item, Exception):
                raise item
        return (
            {"team": {"id": "hub-team-1", "name": "x", "version": "1.0.0"}}
            if kind.startswith("team")
            else {"agent_id": ident or "hub-agent-1", "name": "x", "version": "1.0.0"}
        )

    def publish_team_document(self, document, tenant_id=None, **kwargs):
        return self._commit("team-create", None, document, tenant_id=tenant_id, **kwargs)

    def republish_team_document(self, team_id, document, **kwargs):
        return self._commit("team-republish", team_id, document, **kwargs)

    def publish_agent_instruction_set(self, document, **kwargs):
        return self._commit("agent-create", None, document, **kwargs)

    def republish_agent_instruction_set(self, agent_id, document, **kwargs):
        return self._commit("agent-republish", agent_id, document, **kwargs)


@pytest.fixture
def hub(tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch) -> _Hub:
    from local_operator.providers import radient_credentials

    instance = _Hub()
    monkeypatch.setattr("local_operator.clients.radient.RadientClient", lambda **kwargs: instance)
    monkeypatch.setattr(
        radient_credentials,
        "resolve_radient_oauth_access_sync",
        lambda *args, **kwargs: types.SimpleNamespace(access_token="jwt-fixture", kind="oauth"),
    )
    monkeypatch.setattr(
        radient_credentials,
        "resolve_radient_credential_sync",
        lambda *args, **kwargs: "key-fixture",
    )
    return instance


def _tty(monkeypatch: pytest.MonkeyPatch, answers: List[str]) -> None:
    """Force a TTY and feed the scripted answers to input(), EOF when exhausted.

    The stand-in ECHOES the prompt to stdout, because the real ``input()`` does:
    without it, every cell asserting on prompt text would fail falsely (and one
    that asserts a question was asked could pass vacuously).
    """
    scripted = list(answers)

    def fake_input(prompt: str = "") -> str:
        if prompt:
            sys.stdout.write(prompt)
        if not scripted:
            raise EOFError
        return scripted.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))


def _no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))


def _make_team(name: str = "release-crew", project: str = "rad-1", instructions: str = "You ship."):
    from local_operator.paths import config_dir
    from local_operator.teams import TeamEditFields, TeamRegistry, parse_members

    registry = TeamRegistry(config_dir())
    return registry.create_team(
        TeamEditFields(
            name=name,
            description="Ships it.",
            manager="manager",
            members=parse_members(["coder:2"]),
            instructions=instructions,
            project=project,
        )
    )


# --- parser surface -------------------------------------------------------------


def test_parse_surface_is_additive(parser) -> None:
    push = parser.parse_args(
        ["teams", "push", "--org", "org-a", "--id", "hub-1", "--yes", "--preview-only", "crew"]
    )
    assert (push.teams_command, push.name, push.org, push.hub_id) == (
        "push",
        "crew",
        "org-a",
        "hub-1",
    )
    assert push.yes and push.preview_only
    assert (push.json, push.accept_unresolved, push.allow_internal_ops) == (False, False, False)

    public = parser.parse_args(
        ["teams", "push", "--public", "--accept-unresolved", "--json", "crew"]
    )
    assert public.public is True and public.accept_unresolved is True and public.json is True

    agent = parser.parse_args(["agents", "push", "--name", "X", "--hub-id", "h1", "-y"])
    assert agent.hub_id == "h1" and agent.yes is True

    search = parser.parse_args(["teams", "search", "pep", "--page", "2", "--perpage", "5"])
    assert (search.teams_command, search.query, search.page, search.perpage) == (
        "search",
        "pep",
        2,
        5,
    )

    with pytest.raises(SystemExit):
        parser.parse_args(["teams", "push", "--org", "a", "--public", "crew"])


def test_teams_push_id_without_a_target_demands_one(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--id", "hub-1", "release-crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert "pass --org <tenant_id>, or --public" in out
    assert hub.preview_calls == []


def test_teams_push_empty_id_refuses(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passed-but-empty `--id` must never fall through to a create (R1-1)."""
    _make_team()
    monkeypatch.setattr(
        "sys.argv", ["program", "teams", "push", "--org", "org-a", "--id", "", "release-crew"]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "the value was empty" in out
    assert hub.preview_calls == [] and hub.commit_calls == []


# --- teams push: republish by hub id ---------------------------------------------


def test_teams_push_id_reads_shows_and_republishes_with_the_token(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.team_rows["hub-1"] = {
        "id": "hub-1",
        "tenant_id": "org-a",
        "name": "release-crew",
        "version": "1.0.0",
    }
    _tty(monkeypatch, ["y"])
    monkeypatch.setattr(
        "sys.argv",
        ["program", "teams", "push", "--org", "org-a", "--id", "hub-1", "release-crew"],
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert "Overwriting hub team hub-1 \"release-crew\" v1.0.0 in organization 'org-a'" in out
    assert (
        "Successfully republished team 'release-crew' (ID: hub-team-1) in organization 'org-a'"
        in out
    )
    assert len(hub.commit_calls) == 1
    kind, ident, kwargs = hub.commit_calls[0]
    assert (kind, ident) == ("team-republish", "hub-1")
    assert kwargs["visibility"] == "org" and kwargs["tenant_id"] == "org-a"
    assert kwargs["preview_token"]
    assert kwargs["accept_unresolved"] is None


def test_teams_push_id_wrong_org_refused_before_preview(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.team_rows["hub-1"] = {
        "id": "hub-1",
        "tenant_id": "org-beta",
        "name": "crew",
        "version": "1.0.0",
    }
    monkeypatch.setattr(
        "sys.argv",
        ["program", "teams", "push", "--org", "org-a", "--id", "hub-1", "release-crew", "--yes"],
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "belongs to organization 'org-beta', not 'org-a'" in out
    assert hub.preview_calls == [] and hub.commit_calls == []


def test_teams_push_id_missing_row_names_both_causes(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    monkeypatch.setattr(
        "sys.argv",
        ["program", "teams", "push", "--org", "org-a", "--id", "hub-9", "release-crew", "--yes"],
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "no hub team 'hub-9' answers for this account" in out
    assert "the hub answers both the same way" in out
    assert hub.preview_calls == []


# --- the confirm flow ------------------------------------------------------------


def test_confirm_decline_writes_nothing(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.preview_queue.append(_preview({}, status="ready", changes=_CHANGES))
    _tty(monkeypatch, ["n"])
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 3

    assert hub.commit_calls == []
    out = capsys.readouterr().out
    assert "1 reference would be generalized" in out
    assert '"Damian Tran"' in out and "→ [PERSON_1]" in out and "×3" in out


def test_full_diff_is_printed_on_d(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team(instructions="You ship. Damian Tran reviews.")
    submitted = {
        "name": "release-crew",
        "description": "Ships it.",
        "manager": "manager",
        "members": [],
        "instructions": "You ship. Damian Tran reviews.",
        "version": "1.0.0",
    }
    pinned = dict(submitted, instructions="You ship. [PERSON_1] reviews.")
    hub.preview_queue.append(_preview(pinned, status="ready", changes=_CHANGES))
    _tty(monkeypatch, ["d", "y"])
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 0

    out = capsys.readouterr().out
    assert "--- instructions: submitted" in out and "+++ instructions: generalized" in out
    assert "-You ship. Damian Tran reviews." in out


def test_confirm_commits_the_pinned_document_not_the_local_one(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_team(instructions="You ship. Damian Tran reviews.")
    submitted = {
        "name": "release-crew",
        "description": "Ships it.",
        "manager": "manager",
        "members": [],
        "instructions": "You ship. Damian Tran reviews.",
        "version": "1.0.0",
    }
    pinned = dict(submitted, instructions="You ship. [PERSON_1] reviews.")
    hub.preview_queue.append(_preview(pinned, status="ready", changes=_CHANGES))
    _tty(monkeypatch, ["y"])
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 0

    _kind, _ident, kwargs = hub.commit_calls[0]
    assert kwargs["document"] == pinned  # the previewed bytes, not the local copy
    assert "[PERSON_1]" in kwargs["document"]["instructions"]


def test_non_tty_without_yes_refuses_before_the_preview(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    _no_tty(monkeypatch)
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert (
        "stdin is not a terminal; re-run with --yes to confirm or --preview-only to inspect" in out
    )
    assert hub.preview_calls == [] and hub.commit_calls == []


def test_preview_only_exits_two_when_changes_are_pending(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.preview_queue.append(_preview({}, status="ready", changes=_CHANGES))
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["program", "teams", "push", "--org", "org-a", "--preview-only", "release-crew"]
    )

    assert main() == 2

    assert hub.commit_calls == []
    assert "would be generalized" in capsys.readouterr().out


def test_preview_only_exits_zero_when_unchanged(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["program", "teams", "push", "--org", "org-a", "--preview-only", "release-crew"]
    )

    assert main() == 0

    assert "No personal references found." in capsys.readouterr().out
    assert hub.commit_calls == []


def test_preview_only_json_emits_the_preview_object(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.preview_queue.append(_preview({}, status="ready", changes=_CHANGES))
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        ["program", "teams", "push", "--org", "org-a", "--preview-only", "--json", "release-crew"],
    )

    assert main() == 2

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["changes"][0]["original"] == "Damian Tran"


# --- unresolved values -----------------------------------------------------------


def test_yes_refuses_unresolved_unless_accept(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.preview_queue.append(_preview({}, status="needs_ack", unresolved=_UNRESOLVED))
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["program", "teams", "push", "--org", "org-a", "--yes", "release-crew"]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "--yes does not accept unresolved values" in out
    assert hub.commit_calls == []


def test_accept_unresolved_acknowledges_the_ids(hub: _Hub, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_team()
    hub.preview_queue.append(_preview({}, status="needs_ack", unresolved=_UNRESOLVED))
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "teams",
            "push",
            "--org",
            "org-a",
            "--yes",
            "--accept-unresolved",
            "release-crew",
        ],
    )

    assert main() == 0

    _kind, _ident, kwargs = hub.commit_calls[0]
    assert kwargs["accept_unresolved"] == ["u1"]


def test_unresolved_prompts_one_at_a_time_and_n_aborts(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.preview_queue.append(_preview({}, status="needs_ack", unresolved=_UNRESOLVED))
    _tty(monkeypatch, ["n"])
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert 'Publish with "[u1] Q4 pilot" left as-is? [y/N]' in out
    assert "Edit the value in your local copy" in out
    assert hub.commit_calls == []

    hub.preview_queue.append(_preview({}, status="needs_ack", unresolved=_UNRESOLVED))
    _tty(monkeypatch, ["y", "y"])
    assert main() == 0
    _kind, _ident, kwargs = hub.commit_calls[0]
    assert kwargs["accept_unresolved"] == ["u1"]


def test_unresolved_values_never_reach_a_log_record(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], caplog
) -> None:
    _make_team()
    hub.preview_queue.append(_preview({}, status="needs_ack", unresolved=_UNRESOLVED))
    _tty(monkeypatch, ["y", "y"])
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 0

    out = capsys.readouterr().out
    assert "Q4 pilot" in out  # the publisher's own terminal is where it renders
    assert "Q4 pilot" not in caplog.text  # and a log is never a second surface


# --- consent (--allow-internal-ops) ----------------------------------------------


def test_allow_internal_ops_without_org_is_refused_locally(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    monkeypatch.setattr(
        "sys.argv",
        ["program", "teams", "push", "--public", "--allow-internal-ops", "release-crew"],
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "moderation_allowance is only valid together with visibility=org" in out
    assert hub.preview_calls == [] and hub.commit_calls == []


def test_allow_internal_ops_requires_typed_yes(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    _tty(monkeypatch, ["yes", "y"])
    monkeypatch.setattr(
        "sys.argv",
        ["program", "teams", "push", "--org", "org-a", "--allow-internal-ops", "release-crew"],
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert "org_internal_ops_v1" in out and 'Type "yes" to confirm' in out
    _kind, _ident, kwargs = hub.commit_calls[0]
    assert kwargs["moderation_allowance"] == "org_internal_ops_v1"

    # Declining the consent publishes nothing: exit 3, zero commits.
    hub.preview_queue.append(_preview({}, status="unchanged"))
    _tty(monkeypatch, ["no"])
    assert main() == 3
    assert len(hub.commit_calls) == 1


def test_allow_internal_ops_non_interactive_needs_yes(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-interactive consent is spelled `--allow-internal-ops --yes`, both flags."""
    _make_team()
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "teams",
            "push",
            "--org",
            "org-a",
            "--allow-internal-ops",
            "--yes",
            "release-crew",
        ],
    )

    assert main() == 0

    _kind, _ident, kwargs = hub.commit_calls[0]
    assert kwargs["moderation_allowance"] == "org_internal_ops_v1"


def test_republish_resend_consent_prompts_before_dropping(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.team_rows["hub-1"] = {
        "id": "hub-1",
        "tenant_id": "org-a",
        "name": "release-crew",
        "version": "1.0.0",
        "moderation": {"scope": "org_allowance"},
    }
    _tty(monkeypatch, ["y", "y"])
    monkeypatch.setattr(
        "sys.argv",
        ["program", "teams", "push", "--org", "org-a", "--id", "hub-1", "release-crew"],
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert "published under the internal-ops allowance" in out
    assert "Continue without? [y/N]" in out
    _kind, _ident, kwargs = hub.commit_calls[0]
    assert kwargs["moderation_allowance"] is None

    # N on the resend prompt stops before anything is spent.
    hub.preview_queue.append(_preview({}, status="unchanged"))
    _tty(monkeypatch, ["n"])
    assert main() == 3
    assert len(hub.preview_calls) == 1  # unchanged from the successful run above


def test_republish_resend_consent_with_yes_keeps_running_with_a_notice(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.team_rows["hub-1"] = {
        "id": "hub-1",
        "tenant_id": "org-a",
        "name": "release-crew",
        "version": "1.0.0",
        "moderation": {"scope": "org_allowance"},
    }
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "teams",
            "push",
            "--org",
            "org-a",
            "--id",
            "hub-1",
            "--yes",
            "release-crew",
        ],
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert "published under the internal-ops allowance" in out
    assert "Continuing without it (--yes)." in out


def test_moderation_rejected_hints_at_the_allowance_and_never_retries(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.commit_queue.append(
        APIError("Moderation rejected this agent.", status_code=422, code="moderation_rejected")
    )
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["program", "teams", "push", "--org", "org-a", "--yes", "release-crew"]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "--allow-internal-ops" in out
    assert len(hub.commit_calls) == 1  # never auto-retried with consent


# --- old-server compat and the one-retry rule ------------------------------------


def test_preview_404_falls_back_to_the_plain_publish(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.preview_queue.append(APIError("No such route", status_code=404))
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["program", "teams", "push", "--org", "org-a", "--yes", "release-crew"]
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert "does not support publication previews yet" in out
    _kind, _ident, kwargs = hub.commit_calls[0]
    assert kwargs["preview_token"] is None


def test_preview_404_with_preview_only_refuses(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.preview_queue.append(APIError("No such route", status_code=404))
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        ["program", "teams", "push", "--org", "org-a", "--preview-only", "release-crew"],
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "cannot run against a hub without preview support" in out
    assert hub.commit_calls == []


def test_preview_expired_repreviews_and_reconfirms(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.preview_queue.append(_preview({}, status="ready", changes=_CHANGES))
    hub.preview_queue.append(_preview({}, status="ready", changes=_CHANGES))
    hub.commit_queue.append(
        APIError("This preview is no longer valid.", status_code=409, code="preview_expired")
    )
    _tty(monkeypatch, ["y", "y"])
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 0

    out = capsys.readouterr().out
    assert "previewing again" in out
    assert len(hub.preview_calls) == 2 and len(hub.commit_calls) == 2


def test_yes_never_auto_commits_a_changed_diff(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_team()
    hub.preview_queue.append(_preview({}, status="ready", changes=_CHANGES))
    hub.preview_queue.append(
        _preview({}, status="ready", changes=[dict(_CHANGES[0], original="Someone Else")])
    )
    hub.commit_queue.append(
        APIError("This team changed after the preview.", status_code=409, code="preview_stale")
    )
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["program", "teams", "push", "--org", "org-a", "--yes", "release-crew"]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "the preview changed after the commit was refused" in out
    assert len(hub.commit_calls) == 1  # the new diff is never auto-committed


# --- the public arm: push --public, pull, search ---------------------------------


def test_teams_push_public_strips_project_visibly(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D3: a public publish carries no project; the CLI says so and keeps the local copy."""
    team = _make_team(project="rad-1-internal")
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["program", "teams", "push", "--public", "--yes", "release-crew"]
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert "'project' is internal context and is not published on the public hub" in out
    kind, _ident, preview_kwargs = hub.preview_calls[0]
    assert kind == "team-create" and preview_kwargs["visibility"] == "public"
    assert "project" not in preview_kwargs["document"]
    _kind, _ident, kwargs = hub.commit_calls[0]
    assert "project" not in kwargs["document"]
    assert "Successfully pushed team 'release-crew' to the public hub. Team ID: hub-team-1" in out
    # The local copy is untouched.
    from local_operator.paths import config_dir
    from local_operator.teams import TeamRegistry

    assert TeamRegistry(config_dir()).get_team_by_name("release-crew") is not None
    assert team.project == "rad-1-internal"


def test_teams_pull_public_by_name_resolves_then_imports(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hub.public_teams.append(
        {
            "id": "a" * 24,
            "name": "open-crew",
            "description": "Open.",
            "manager": "m",
            "version": "1.0.0",
        }
    )
    hub.team_rows["a" * 24] = {
        "id": "a" * 24,
        "tenant_id": "home-someone",
        "name": "open-crew",
        "description": "Open.",
        "manager": "manager",
        "members": [],
        "instructions": "You ship.",
        "project": "",
        "version": "1.0.0",
    }
    monkeypatch.setattr("sys.argv", ["program", "teams", "pull", "open-crew"])

    assert main() == 0

    out = capsys.readouterr().out
    assert "Successfully pulled team 'open-crew'" in out
    assert "from the public hub" in out
    # The name resolved to the listing's id, and the pull was ANONYMOUS
    # (with_credential=False -- the public arm's contract).
    assert hub.get_team_calls == [("a" * 24, False)]


# --- agents push --hub-id (org and public republish) -----------------------------


def _make_agent(name: str = "OrgCoder"):
    from local_operator.agents import AgentEditFields, AgentRegistry
    from local_operator.paths import config_dir

    registry = AgentRegistry(config_dir())
    agent = registry.create_agent(
        AgentEditFields.model_validate({"name": name, "description": "Writes code."})
    )
    registry.set_agent_system_prompt(agent.id, "You write code.")
    return agent


def test_agents_push_hub_id_org_republishes_with_the_token(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_agent()
    hub.agent_rows["hub-agent-1"] = {
        "tenant_id": "org-a",
        "visibility": "org",
        "name": "OrgCoder",
        "version": "1.0.0",
    }
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "agents",
            "push",
            "--name",
            "OrgCoder",
            "--org",
            "org-a",
            "--hub-id",
            "hub-agent-1",
            "--yes",
        ],
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert "Overwriting hub agent hub-agent-1 \"OrgCoder\" v1.0.0 in organization 'org-a'" in out
    kind, ident, preview_kwargs = hub.preview_calls[0]
    assert (kind, ident) == ("agent-republish", "hub-agent-1")
    assert preview_kwargs["visibility"] == "org" and preview_kwargs["tenant_id"] == "org-a"
    kind, ident, kwargs = hub.commit_calls[0]
    assert (kind, ident) == ("agent-republish", "hub-agent-1")
    assert kwargs["preview_token"] == "pin-fixture"
    assert (
        "Successfully republished agent 'OrgCoder' (ID: hub-agent-1) in organization 'org-a'" in out
    )


def test_agents_push_hub_id_public_republishes(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_agent()
    hub.agent_rows["hub-pub-1"] = {
        "tenant_id": "home-someone",
        "visibility": "public",
        "name": "OrgCoder",
        "version": "2.0.0",
    }
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        ["program", "agents", "push", "--name", "OrgCoder", "--hub-id", "hub-pub-1", "--yes"],
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert "on the public hub" in out
    kind, _ident, kwargs = hub.commit_calls[0]
    assert kind == "agent-republish"
    assert kwargs["visibility"] == "public" and kwargs["tenant_id"] is None
    assert "Successfully republished agent 'OrgCoder' (ID: hub-pub-1) on the public hub" in out


def test_agents_push_hub_id_wrong_scope_or_tenant_refuses_before_preview(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_agent()
    hub.agent_rows["hub-agent-1"] = {"tenant_id": "org-b", "visibility": "org"}
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "agents",
            "push",
            "--name",
            "OrgCoder",
            "--org",
            "org-a",
            "--hub-id",
            "hub-agent-1",
            "--yes",
        ],
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "belongs to organization 'org-b', not 'org-a'. Check --org." in out
    assert hub.preview_calls == [] and hub.commit_calls == []

    # A public listing addressed with --org is refused too.
    hub.agent_rows["hub-pub-9"] = {"tenant_id": "org-a", "visibility": "public"}
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "agents",
            "push",
            "--name",
            "OrgCoder",
            "--org",
            "org-a",
            "--hub-id",
            "hub-pub-9",
            "--yes",
        ],
    )
    assert main() == 1
    assert "is published publicly, not into an organization" in capsys.readouterr().out
    assert hub.preview_calls == []

    # And an org listing addressed WITHOUT --org (the public arm).
    hub.agent_rows["hub-agent-1"] = {"tenant_id": "org-a", "visibility": "org"}
    monkeypatch.setattr(
        "sys.argv",
        ["program", "agents", "push", "--name", "OrgCoder", "--hub-id", "hub-agent-1", "--yes"],
    )
    assert main() == 1
    assert "is an organization listing, not a public one" in capsys.readouterr().out
    assert hub.preview_calls == []


def test_agents_push_hub_id_requires_name_not_a_local_id(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _make_agent()
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "program",
            "agents",
            "push",
            "--id",
            agent.id,
            "--org",
            "org-a",
            "--hub-id",
            "hub-agent-1",
            "--yes",
        ],
    )

    assert main() == 1

    assert "--id is a local agent id" in capsys.readouterr().out
    assert hub.preview_calls == []


def test_agents_push_zip_path_refuses_preview_flags(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The public archive path cannot preview: a preview flag is refused by name."""
    _make_agent()
    _no_tty(monkeypatch)
    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "push", "--name", "OrgCoder", "--preview-only"]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "--preview-only need a preview-publishing target" in out
    assert hub.preview_calls == [] and hub.commit_calls == []


# --- _hub_cause: the new codes ----------------------------------------------------


def test_hub_cause_renders_the_new_codes() -> None:
    cases = [
        ("generalization_required", {}, "generalization_required"),
        ("preview_mismatch", {}, "preview_mismatch"),
        ("preview_expired", {}, "preview_expired"),
        ("preview_stale", {}, "preview_stale"),
        (
            "generalization_unresolved",
            {"ids": ["u1", "u2"]},
            "outstanding: u1, u2",
        ),
        (
            "moderation_unavailable",
            {"stage": "generalization"},
            "the reference check could not run",
        ),
        (
            "moderation_unavailable",
            {"stage": "generalization", "windows_needed": 41, "windows_max": 24},
            "needs 41 windows; the limit is 24",
        ),
        ("resolution_conflict", {}, "resolution_conflict"),
    ]
    for code, details, expected in cases:
        exc = APIError("Refused.", status_code=422, code=code, details=details)
        assert expected in _hub_cause(exc), (code, _hub_cause(exc))


# --- teams search ------------------------------------------------------------------


def test_teams_search_filters_name_and_description(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hub.public_teams = [
        {
            "id": "1" * 24,
            "name": "pep-screener",
            "description": "Screens PEPs.",
            "manager": "m",
            "version": "1.0.0",
        },
        {
            "id": "2" * 24,
            "name": "enrichment",
            "description": "PEP enrichment.",
            "manager": "m",
            "version": "1.0.0",
        },
        {
            "id": "3" * 24,
            "name": "unrelated",
            "description": "Nothing here.",
            "manager": "m",
            "version": "1.0.0",
        },
    ]
    monkeypatch.setattr("sys.argv", ["program", "teams", "search", "pep"])

    assert main() == 0

    out = capsys.readouterr().out
    assert "pep-screener" in out and "enrichment" in out
    assert "unrelated" not in out

    monkeypatch.setattr("sys.argv", ["program", "teams", "search", "nothing-matches"])
    assert main() == 0
    assert "No public team matches 'nothing-matches'" in capsys.readouterr().out


def test_teams_search_json_emits_the_rows(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hub.public_teams = [
        {
            "id": "1" * 24,
            "name": "pep-screener",
            "description": "Screens PEPs.",
            "manager": "m",
            "version": "1.0.0",
        }
    ]
    monkeypatch.setattr("sys.argv", ["program", "teams", "search", "pep", "--json"])

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "pep-screener"


def test_teams_pull_public_by_hex_id_skips_the_listing(
    hub: _Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hub.team_rows["a" * 24] = {
        "id": "a" * 24,
        "tenant_id": "home-someone",
        "name": "open-crew",
        "description": "Open.",
        "manager": "manager",
        "members": [],
        "instructions": "You ship.",
        "project": "",
        "version": "1.0.0",
    }
    monkeypatch.setattr("sys.argv", ["program", "teams", "pull", "a" * 24])

    assert main() == 0

    out = capsys.readouterr().out
    assert "Successfully pulled team 'open-crew'" in out
    assert hub.get_team_calls == [("a" * 24, False)]
