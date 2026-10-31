"""``lop team-bridge avatar``: the agent-callable path over the same rules.

The wrappers hold no rules of their own — every decision belongs to
``avatar_publish`` — so what is pinned here is the CONTRACT of the command: which
action reaches which publisher, what a caller can read in the answer, and that a
request that did not take effect exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import pytest

from local_operator.team_bridge import avatar_publish, cli

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 8
PNG_SHA = hashlib.sha256(PNG).hexdigest()


class StubClient:
    """The bridge client, without an auth store or a transport behind it."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "tenants": [{"tenant_id": "tnt_1", "eligible": True, "endpoint": {"id": "bep_1"}}]
        }

    async def status(self, *, access: Any = None) -> dict[str, Any]:
        return self.payload

    async def current_access(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(account_id="acc_1")

    async def put_endpoint_avatar(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    async def delete_endpoint_avatar(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubClient:
    client = StubClient()
    monkeypatch.setattr(cli, "_client_for", lambda root: client)
    return client


def _args(action: str, **extra: Any) -> argparse.Namespace:
    fields: dict[str, Any] = {
        "team_bridge_command": "avatar",
        "avatar_command": action,
        "json": True,
        "tenant": "tnt_1",
    }
    fields.update(extra)
    return argparse.Namespace(**fields)


def test_the_parser_accepts_the_three_documented_actions():
    parser = argparse.ArgumentParser(prog="team-bridge")
    cli.configure_parser(parser)

    for action in cli.ACTIONS:
        args = parser.parse_args(["avatar", action])
        assert (args.team_bridge_command, args.avatar_command) == ("avatar", action)
        assert args.json is False and args.tenant is None


def test_publish_reports_the_endpoint_agent_and_digest(
    stub: StubClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    plan = avatar_publish.AvatarPlan(
        action=avatar_publish.PUBLISH,
        endpoint_id="bep_1",
        agent_id="aida",
        name="Aida",
        data=PNG,
        digest=PNG_SHA,
        hub_sha256="0" * 64,
    )
    calls: list[tuple[str, str]] = []

    async def publish(client: Any, tenant_id: str, root: Any) -> avatar_publish.AvatarPlan:
        calls.append(("publish", tenant_id))
        return plan

    monkeypatch.setattr(avatar_publish, "publish_agent_avatar", publish)
    assert cli.main(_args("publish")) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert (printed["endpoint_id"], printed["agent_id"]) == ("bep_1", "aida")
    assert printed["local_sha256"] == PNG_SHA and printed["local_bytes"] == len(PNG)
    assert calls == [("publish", "tnt_1")]


def test_publish_with_nothing_to_send_exits_non_zero(
    stub: StubClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    async def publish(client: Any, tenant_id: str, root: Any) -> avatar_publish.AvatarPlan:
        return avatar_publish.AvatarPlan(action=avatar_publish.NONE, endpoint_id="bep_1")

    monkeypatch.setattr(avatar_publish, "publish_agent_avatar", publish)

    assert cli.main(_args("publish")) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_clear_reports_the_endpoint_it_removed(
    stub: StubClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    async def clear(client: Any, tenant_id: str) -> avatar_publish.AvatarPlan:
        return avatar_publish.AvatarPlan(
            action=avatar_publish.DELETE, endpoint_id="bep_1", hub_sha256=PNG_SHA
        )

    monkeypatch.setattr(avatar_publish, "clear_agent_avatar", clear)

    assert cli.main(_args("clear")) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["planned"] == "delete"


def test_status_is_read_only_and_reports_both_sides(
    stub: StubClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    stub.payload = {
        "tenants": [
            {
                "tenant_id": "tnt_1",
                "eligible": True,
                "endpoint": {
                    "id": "bep_1",
                    "avatar": {"hash": "0123456789abcdef", "source_sha256": "0" * 64},
                },
            }
        ]
    }

    assert cli.main(_args("status")) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["planned"] == "none"
    assert printed["hub_sha256"] == "0" * 64


def test_a_refusal_is_reported_with_its_code_and_never_a_traceback(
    stub: StubClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from local_operator.team_bridge.client import BridgeRefusal

    async def publish(client: Any, tenant_id: str, root: Any) -> avatar_publish.AvatarPlan:
        raise BridgeRefusal("avatar_too_large", "refused", http_status=413)

    monkeypatch.setattr(avatar_publish, "publish_agent_avatar", publish)

    assert cli.main(_args("publish")) == 1
    out = capsys.readouterr().out
    assert "BridgeRefusal" in out and "avatar_too_large" in out
    assert "Traceback" not in out


def test_no_enrolled_organization_is_a_refusal_not_a_crash(
    stub: StubClient, capsys: pytest.CaptureFixture[str]
):
    stub.payload = {"tenants": []}

    assert cli.main(_args("publish", tenant=None)) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
