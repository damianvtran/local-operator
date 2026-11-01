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
import os
import subprocess
import sys
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


def test_the_verbs_ride_the_one_team_bridge_group():
    """R1-F1/F2: one `team-bridge` group, and the pre-existing verbs survive.

    Registering a SECOND top-level ``team-bridge`` parser made
    ``build_cli_parser()`` raise for every command, and an early return in the
    dispatcher swallowed ``setup``/``receiver`. Both are asserted here against
    the real parser build, which is what a user's `lop` actually constructs.
    """

    from local_operator.cli import build_cli_parser

    parser = build_cli_parser()
    group = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ).choices["team-bridge"]
    children = list(group._subparsers._group_actions[0].choices)
    assert children[:2] == ["setup", "receiver"]
    assert "avatar" in children

    for action in cli.ACTIONS:
        args = parser.parse_args(["team-bridge", "avatar", action])
        assert (args.team_bridge_command, args.avatar_command) == ("avatar", action)
        assert args.json is False and args.tenant is None
    assert parser.parse_args(["team-bridge", "setup", "--json"]).team_bridge_command == "setup"


def _lop(env: dict[str, str], *argv: str) -> subprocess.CompletedProcess[str]:
    """One REAL `lop` invocation: the parser build and dispatch a user gets."""

    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from local_operator.cli import main; sys.exit(main())",
            *argv,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def test_both_team_bridge_surfaces_are_reachable(tmp_path):
    """R1-F1/F2 end to end: `lop` starts, and neither verb shadows the other.

    Before the fix the parser build raised for EVERY command, and the avatar
    branch returned before the pre-existing handler, so `receiver status` came
    back as an AttributeError line.
    """

    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / ".local-operator"),
        "NO_COLOR": "1",
    }

    avatar = _lop(env, "team-bridge", "avatar", "status", "--json")
    receiver = _lop(env, "team-bridge", "receiver", "status", "--json")
    version = _lop(env, "--version")

    assert version.returncode == 0, version.stderr
    # The avatar command has no organization on this machine; what matters is
    # that it answered AS ITSELF rather than crashing the binary.
    assert "conflicting subparser" not in (avatar.stdout + avatar.stderr)
    assert "AttributeError" not in (avatar.stdout + avatar.stderr)
    assert receiver.returncode in (0, 1), receiver.stderr
    assert "conflicting subparser" not in (receiver.stdout + receiver.stderr)


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

    async def publish(
        client: Any, tenant_id: str, root: Any, **kwargs: Any
    ) -> avatar_publish.AvatarPlan:
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
    async def publish(
        client: Any, tenant_id: str, root: Any, **kwargs: Any
    ) -> avatar_publish.AvatarPlan:
        return avatar_publish.AvatarPlan(action=avatar_publish.NONE, endpoint_id="bep_1")

    monkeypatch.setattr(avatar_publish, "publish_agent_avatar", publish)

    assert cli.main(_args("publish")) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_clear_reports_the_endpoint_it_removed(
    stub: StubClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    async def clear(client: Any, tenant_id: str, **kwargs: Any) -> avatar_publish.AvatarPlan:
        return avatar_publish.AvatarPlan(
            action=avatar_publish.DELETE, endpoint_id="bep_1", hub_sha256=PNG_SHA
        )

    monkeypatch.setattr(avatar_publish, "clear_agent_avatar", clear)

    assert cli.main(_args("clear")) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True and printed["planned"] == "delete"


def test_status_reports_both_sides_and_plans_the_clear_arm(
    stub: StubClient,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
):
    """R1-F5: a HERMETIC config root, and the expectation the contract names.

    The test used to read the developer's real ``~/.local-operator`` (so it
    failed on any machine with a designated chief of staff) and asserted
    ``none`` where the fixture — hub holds a picture, local definition has none —
    is exactly the DELETE arm.
    """

    from local_operator.team_bridge import cos_name

    monkeypatch.setattr(cli, "_config_dir", lambda: tmp_path)
    monkeypatch.setattr(cos_name, "_AVATAR_FIELD", True)
    root = tmp_path
    agent_dir = root / "agents" / "aida"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.yml").write_text("id: aida\nname: aida\nversion: '1'\n", encoding="utf-8")
    from local_operator import settings_io
    from local_operator.aida import state as aida_state
    from local_operator.config import ConfigManager

    aida_state.write_state(root, {"session_id": "sess_aida"})
    settings_io.write_setting(
        ConfigManager(root), settings_io.BY_KEY["team_bridge.session"], "sess_aida"
    )
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
    assert printed["ok"] is True
    assert printed["planned"] == avatar_publish.DELETE
    assert printed["hub_sha256"] == "0" * 64 and printed["local_sha256"] is None


def test_status_reads_the_hub_once(
    stub: StubClient,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
):
    """R1-F10: one status answer serves the tenant selection and the report."""

    monkeypatch.setattr(cli, "_config_dir", lambda: tmp_path)
    calls: list[Any] = []
    original = StubClient.status

    async def counting(self: Any, *, access: Any = None) -> Any:
        calls.append(access)
        return await original(self, access=access)

    monkeypatch.setattr(StubClient, "status", counting)

    assert cli.main(_args("status", tenant=None)) == 0

    assert len(calls) == 1


def test_a_refusal_is_reported_with_its_code_and_never_a_traceback(
    stub: StubClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from local_operator.team_bridge.client import BridgeRefusal

    async def publish(
        client: Any, tenant_id: str, root: Any, **kwargs: Any
    ) -> avatar_publish.AvatarPlan:
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


def test_the_enrolment_carries_her_own_name_when_one_is_designated(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
):
    """R1-F8: the other half of the naming fix, asserted on the wire body."""

    import httpx

    from local_operator import settings_io
    from local_operator.aida import state as aida_state
    from local_operator.config import ConfigManager
    from local_operator.team_bridge.client import TeamBridgeClient
    from tests.unit.team_bridge.test_client import FakeRow, FakeStore

    aida_state.write_state(tmp_path, {"session_id": "sess_aida"})
    settings_io.write_setting(
        ConfigManager(tmp_path), settings_io.BY_KEY["team_bridge.session"], "sess_aida"
    )
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "msg": "ok",
                    "result": {
                        "endpoint": {
                            "id": "bep_1",
                            "state": "active",
                            "consent": {"consented_via": "default_on", "version": "v2"},
                        }
                    },
                },
            )
        return httpx.Response(200, json={"msg": "ok", "result": {"head": 0}})

    client = TeamBridgeClient(
        tmp_path,
        transport=httpx.MockTransport(handler),
        store=FakeStore(rows=[FakeRow(1, data={"account_id": "acc_1", "access": "token-1"})]),
    )

    import asyncio

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        client.enroll_default_on(
            tenant_id="tnt_1", expected_account_id="acc_1", consent_version="bridge_consent_v3"
        )
    )

    assert bodies and bodies[0]["agent_name"] == "Aida"
    assert bodies[0]["consent_version"] == "bridge_consent_v3"


def test_the_enrolment_omits_the_name_when_nothing_is_designated(
    tmp_path: Any,
):
    """R1-F8: absent, never ``""`` — an enrolment has no name to clear."""

    import asyncio

    import httpx

    from local_operator.team_bridge.client import TeamBridgeClient
    from tests.unit.team_bridge.test_client import FakeRow, FakeStore

    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"msg": "ok", "result": {"endpoint": None}})
        return httpx.Response(200, json={"msg": "ok", "result": {"head": 0}})

    client = TeamBridgeClient(
        tmp_path,
        transport=httpx.MockTransport(handler),
        store=FakeStore(rows=[FakeRow(1, data={"account_id": "acc_1", "access": "token-1"})]),
    )

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        client.enroll_default_on(
            tenant_id="tnt_1", expected_account_id="acc_1", consent_version="bridge_consent_v3"
        )
    )

    assert bodies and "agent_name" not in bodies[0]
    assert bodies[0]["consent_version"] == "bridge_consent_v3"
