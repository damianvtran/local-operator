"""``lop team-bridge holds``: the read-only held-messages list (operator report 2026-11-18).

Pins the command's contract against a REAL held message produced by the real
screening spine (``test_hold_release``'s rig, a ``model_hold``):

* the rows are the desktop ``/holds`` reader's rows (one source of truth);
* under an agent's ``bash`` the owner-audit join is refused and no audit client
  is even built (the human-only rule, ``test_owner_audit_isolation``);
* for a person, sender comes from the audit page at the hold's seq, and the
  excerpt only with ``--excerpt``, redacted BEFORE the 160-character cut;
* a dead sign-in still prints the local rows;
* nothing is written: no store is created where none exists, and an existing
  store's bytes are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.agent_shell import AGENT_SHELL_ENV
from local_operator.redaction_shapes import REDACTION_MARKER
from local_operator.server.utils import desktop_team_bridge_holds as holds
from local_operator.team_bridge import holds_cli
from local_operator.team_bridge.client import BridgeRefusal, BridgeSignInRequired
from local_operator.team_bridge.store import state_db_path
from tests.unit.team_bridge.test_hold_release import (  # noqa: F401 — ``rig`` is a fixture
    CONVERSATION_ID,
    DOUBT,
    MESSAGE_ID,
    Runner,
    _spine,
    rig,
)

#: A GitHub-token shape the shared redaction table masks; built from halves so
#: this file is not itself a secret-scanner hit.
TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


class FakeAuditClient:
    """Records each audit call and answers with the held message's page."""

    def __init__(self, *, body: str = "hello", raise_with: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.body = body
        self.raise_with = raise_with

    async def owner_audit_messages(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"conversation_id": conversation_id, **kwargs})
        if self.raise_with is not None:
            raise self.raise_with
        return {
            "messages": [
                {
                    "id": MESSAGE_ID,
                    "conversation_id": conversation_id,
                    "seq": 1,
                    "sender": {
                        "account_id": "acc_sender",
                        "kind": "human",
                        "display_label": "Victor",
                    },
                    "agent_label": "Victor's Aida",
                    "body": self.body,
                    "created_at": "2026-11-18T00:00:00Z",
                }
            ]
        }


@pytest.fixture
def held(rig, monkeypatch):  # noqa: F811 — imported fixture
    config_dir, _store, owner = rig
    _server, _wakes, run = _spine(monkeypatch, config_dir, owner, Runner(DOUBT))
    assert run()["code"] == "model_hold"
    monkeypatch.setattr(holds_cli, "_config_dir", lambda: config_dir)
    return config_dir, owner


def _as_person(monkeypatch, client: Any) -> None:
    monkeypatch.delenv(AGENT_SHELL_ENV, raising=False)
    monkeypatch.setattr(holds_cli, "_client_for", lambda root: client)


def test_rows_are_the_desktop_readers_rows(held, monkeypatch) -> None:
    config_dir, owner = held
    _as_person(monkeypatch, FakeAuditClient())
    answer = holds_cli.collect(config_dir)
    route = holds.list_holds(config_dir, owner)["items"]
    assert [r["id"] for r in answer["items"]] == [i["message_id"] for i in route]
    row, item = answer["items"][0], route[0]
    for ours, theirs in (
        ("reason_code", "hold_code"),
        ("kind", "kind"),
        ("can_deliver", "can_deliver"),
        ("can_discard", "can_discard"),
        ("conversation_id", "conversation_id"),
        ("seq", "seq"),
        ("held_at", "held_at"),
    ):
        assert row[ours] == item[theirs], ours
    assert row["reason_code"] == "model_hold" and row["can_discard"] is True
    assert row["first_held_at"] is not None and row["age_seconds"] is not None
    assert "model_hold" in holds_cli.render(answer)


def test_agent_shell_gets_metadata_and_no_audit_request(held, monkeypatch) -> None:
    config_dir, _owner = held
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")

    def no_client(root: Path) -> Any:
        raise AssertionError("an agent shell must not even build the audit client")

    monkeypatch.setattr(holds_cli, "_client_for", no_client)
    answer = holds_cli.collect(config_dir, excerpt=True)
    assert answer["audit"] == "withheld_agent_shell"
    row = answer["items"][0]
    assert row["id"] == MESSAGE_ID and row["sender_label"] is None and row["excerpt"] is None
    assert "from:          withheld" in holds_cli.render(answer)


def test_person_sees_sender_and_redacted_excerpt_only_on_request(held, monkeypatch) -> None:
    config_dir, owner = held
    client = FakeAuditClient(body=f"please use {TOKEN} to deploy " + "x " * 200)
    _as_person(monkeypatch, client)

    plain = holds_cli.collect(config_dir)
    assert plain["audit"] == "ok" and "excerpt" not in plain["items"][0]
    assert plain["items"][0]["sender_label"] == "Victor"
    assert "Victor (human) via Victor's Aida" in holds_cli.render(plain)
    # The captured-owner receipt is the COMMITTED owner. This rig's hold row is
    # stored under the bare id (no seq), so the newest bounded page is searched.
    call = client.calls[0]
    assert call["conversation_id"] == CONVERSATION_ID
    assert call["expected_account_id"] == owner.account_id
    assert call["tenant_id"] == owner.tenant_id
    assert call["before_seq"] is None and call["limit"] == holds_cli._NO_SEQ_PAGE

    shown = holds_cli.collect(config_dir, excerpt=True)["items"][0]["excerpt"]
    assert len(shown) <= holds_cli.EXCERPT_MAX
    assert TOKEN not in shown and TOKEN[:12] not in shown and REDACTION_MARKER in shown


def test_a_hold_with_a_seq_reads_exactly_its_own_message(monkeypatch) -> None:
    import asyncio

    from local_operator.team_bridge.store import OwnerSnapshot

    client = FakeAuditClient()
    _as_person(monkeypatch, client)
    owner = OwnerSnapshot(7, "acc_recv", "tnt_1", "bep_recv", "https://hub.example/v1", 1)
    item = {"message_id": MESSAGE_ID, "conversation_id": CONVERSATION_ID, "seq": 41}
    state, found = asyncio.run(holds_cli._join_audit(Path("."), owner, [item], want_body=False))
    assert state == "ok" and found[MESSAGE_ID]["body"] is None
    assert client.calls[0]["before_seq"] == 42 and client.calls[0]["limit"] == 1


def test_sender_text_cannot_drive_the_terminal(monkeypatch) -> None:
    import asyncio

    from local_operator.team_bridge.store import OwnerSnapshot

    client = FakeAuditClient(body="hi\x1b[2J\x1b]0;pwned\x07 there")
    _as_person(monkeypatch, client)
    owner = OwnerSnapshot(7, "acc_recv", "tnt_1", "bep_recv", "https://hub.example/v1", 1)
    item = {"message_id": MESSAGE_ID, "conversation_id": CONVERSATION_ID, "seq": 1}
    _state, found = asyncio.run(holds_cli._join_audit(Path("."), owner, [item], want_body=True))
    assert "\x1b" not in (holds_cli.excerpt_of(found[MESSAGE_ID]["body"]) or "")
    assert "\x07" not in (holds_cli.excerpt_of(found[MESSAGE_ID]["body"]) or "")


def test_redaction_happens_before_the_cut() -> None:
    # The token straddles character 160: cut first and the surviving prefix
    # would be too short to match the shape.
    shown = holds_cli.excerpt_of("a" * 150 + " " + TOKEN)
    assert shown is not None and len(shown) <= holds_cli.EXCERPT_MAX
    assert TOKEN[:8] not in shown


@pytest.mark.parametrize(
    ("failure", "state"),
    [
        (BridgeSignInRequired("sign in"), "sign_in_required"),
        (BridgeRefusal("account_mismatch", "x", http_status=409), "refused:account_mismatch"),
        (BridgeRefusal("forbidden", "x", http_status=403), "refused:forbidden"),
        (BridgeRefusal("not_found", "x", http_status=404), "partial:1_not_in_audit"),
    ],
)
def test_a_failed_join_still_prints_the_local_rows(held, monkeypatch, failure, state) -> None:
    config_dir, _owner = held
    _as_person(monkeypatch, FakeAuditClient(raise_with=failure))
    answer = holds_cli.collect(config_dir)
    assert answer["audit"] == state
    assert [r["id"] for r in answer["items"]] == [MESSAGE_ID]
    assert "from:          unavailable" in holds_cli.render(answer)


def test_it_writes_nothing(held, monkeypatch, tmp_path_factory) -> None:
    config_dir, _owner = held
    _as_person(monkeypatch, FakeAuditClient())
    db = state_db_path(config_dir)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    holds_cli.collect(config_dir, excerpt=True)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before

    empty = tmp_path_factory.mktemp("no-store")
    answer = holds_cli.collect(empty)
    assert answer == {"enrolled": False, "audit": "no_owner", "items": []}
    assert not state_db_path(empty).exists()


def test_cli_parses_both_spellings_and_prints_json(held, monkeypatch, capsys) -> None:
    from local_operator.cli import build_cli_parser

    _as_person(monkeypatch, FakeAuditClient())
    parser = build_cli_parser()
    for argv in (
        ["team-bridge", "holds", "--json"],
        ["team-bridge", "holds", "list", "--json", "--excerpt"],
    ):
        args = parser.parse_args(argv)
        assert args.team_bridge_command == "holds"
        assert holds_cli.main(args) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["count"] == 1
        assert ("excerpt" in printed["items"][0]) is ("--excerpt" in argv)


def test_an_unreadable_store_exits_one(tmp_path: Path, monkeypatch, capsys) -> None:
    from local_operator.cli import build_cli_parser

    db = state_db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"not a database at all, just bytes" * 64)
    monkeypatch.setattr(holds_cli, "_config_dir", lambda: tmp_path)
    args = build_cli_parser().parse_args(["team-bridge", "holds"])
    assert holds_cli.main(args) == 1
    assert "could not be read" in capsys.readouterr().out
