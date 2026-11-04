"""Team Bridge refusals that tell the agent the truth (2026-11-04).

Three failures reported from production, each pinned here at the layer that
owned it:

* a ``channel_post`` carrying AWS credentials was SENT — only the edge WAF
  stopped it — so a credential-shaped body must now be refused locally, before
  any client call, and the refusal must never echo the value;
* that WAF block came back as ``bridge_error`` "Radient refused the request
  (HTTP 403)", which reads as the hub's judgement; a non-JSON 403 is the edge's;
* ``dm`` to a person failed as a generic ``dm_failed`` with a hub footer, when
  the hub refuses agent→person DMs by rule and the next step is a group.

Every credential fixture is ASSEMBLED from parts at runtime so no file in the
repository spells a credential: secret scanners on push and in CI must not see
one, and a reader grepping for a leaked key must not find a test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import BridgeRefusal, _refusal
from local_operator.team_bridge.secret_guard import (
    SECRET_DETECTED_CODE,
    VAULT_HANDOFF_SENTENCE,
    credential_classes,
)
from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge
from tests.unit.team_bridge.test_channel_tool import RecordingClient, _runtime

# --- synthetic credentials, built from parts (see the module docstring) -----
# Lengths are the real issuers' (asserted below): a fixture of the wrong length
# tests a shape no credential has.
AWS_KEY_ID = "AK" + "IA" + "Z7QW3RTY5UIOP2AS"
AWS_STS_KEY_ID = "AS" + "IA" + "Q4WE8RTY2UIOP9ZX"
AWS_SECRET = "q8Zr" + "Tn3vLw" + "9Yb2Kd/Mf5Xp+Hs7Jc4Ng1Ve6Ua0Rx"  # AWS: exactly 40
PEM = (
    "-----BEGIN " + "RSA PRIVATE" + " KEY-----\n"
    "MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn" + "ygWyF8PbnGy0AHB7MaFcGf9yqXRVp2yS\n"
    "-----END " + "RSA PRIVATE" + " KEY-----"
)
GITHUB_TOKEN = "gh" + "p_" + "R4nd0mT0k3nV4lu3F0rT3st1ng0nlyAbCd12"
GITHUB_PAT = (
    "github"
    + "_pat_"
    + "11ABCDEFG0"
    + "Zq9Wx8Vu7Ts6Rq5Po4Nm3Lk2Jh1Gf0EdCbA9z8Y7x6W5v4U3t2S1r0Q9p8O7n6"
)
SLACK_TOKEN = "xo" + "xb-" + "123456789012-1234567890123-" + "AbCdEfGhIjKlMnOpQrStUvWx"
OPENAI_KEY = "s" + "k-" + "proj-" + "Ab3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3fG5"
GOOGLE_KEY = "AI" + "za" + "SyA1234567890abcdefghijklmnopqrstu"
PASSWORD = "Tr0ub4dor" + "&3xQ9"

#: (case id, message body, the one secret value that must never be echoed,
#:  a class label the refusal must name)
SECRET_CASES: list[tuple[str, str, str, str]] = [
    (
        "aws-credentials-file",
        "creds from my ~/.aws/credentials:\n"
        f"aws_access_key_id = {AWS_KEY_ID}\naws_secret_access_key = {AWS_SECRET}",
        AWS_SECRET,
        "aws-access-key-id",
    ),
    ("aws-key-id-bare", f"use key {AWS_KEY_ID} for the deploy", AWS_KEY_ID, "aws-access-key-id"),
    ("aws-sts-key-id", f"session key {AWS_STS_KEY_ID}", AWS_STS_KEY_ID, "aws-access-key-id"),
    (
        "aws-secret-console-label",
        f"Secret access key: {AWS_SECRET}",
        AWS_SECRET,
        "aws-secret-access-key",
    ),
    (
        "aws-secret-sdk-json",
        json.dumps({"Credentials": {"SecretAccessKey": AWS_SECRET}}),
        AWS_SECRET,
        "aws-secret-access-key",
    ),
    ("pem-private-key", f"here is the deploy key:\n{PEM}", PEM.splitlines()[1], "pem-private-key"),
    ("labelled-password", f"the db password: {PASSWORD}", PASSWORD, "credential-assignment"),
    ("labelled-token", f"token={GITHUB_TOKEN[4:]}", GITHUB_TOKEN[4:], "credential-assignment"),
    ("github-token", f"try {GITHUB_TOKEN}", GITHUB_TOKEN, "github-token"),
    ("github-pat", f"PAT {GITHUB_PAT}", GITHUB_PAT, "github-pat"),
    ("slack-token", f"bot token {SLACK_TOKEN}", SLACK_TOKEN, "slack-token"),
    ("openai-key", f"my key is {OPENAI_KEY}", OPENAI_KEY, "vendor-prefixed-token"),
    ("google-api-key", f"maps key {GOOGLE_KEY}", GOOGLE_KEY, "google-api-key"),
]

#: Bodies an agent legitimately sends that sit near a credential shape.
BENIGN = [
    "my password manager is 1Password",
    "please reset your password and tell me when done",
    "AWS long-term key ids start with AKIA, session ones with ASIA",
    "the ASIAN markets team owns that report",
    "sha1 of the release is 3f786850e387550fdab836ed7e6dc881de23001b",
    "fixed in 47e19a6ada9acf203e12630f301556ccd82e3735 (#3335)",
    "the secret: radient/prod/agent-server is the Secrets Manager entry",
    "token: expired yesterday, please re-auth",
    "set `password: ${DB_PASSWORD}` in the config, never the literal",
    "curl -H 'Authorization: Bearer $TOKEN' https://api.radienthq.com/v1/models",
    "use the sk-learn package and --password-stdin for docker login",
]

#: Every op that sends a message body, with the field it reads it from and the
#: rest of a request that would otherwise be valid.
MESSAGE_OPS: list[tuple[str, dict[str, Any]]] = [
    ("channel_post", {"channel_id": "bcv_abcd1234"}),
    ("dm", {"account_id": "acc_2"}),
    ("send", {"to": {"account_id": "acc_2", "endpoint_id": "bep_2"}}),
    ("reply", {"to": {"account_id": "acc_2", "endpoint_id": "bep_2"}, "thread_id": "thr_1"}),
]


class StrictClient:
    """A client that fails the test on ANY use: a refused message reaches none."""

    def __init__(self) -> None:
        self.touched: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.touched.append(name)
        raise AssertionError(f"a refused message touched the client: {name}")


@pytest.fixture(autouse=True)
def _fences_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fences have their own coverage (test_outbound_session_scope.py,
    # test_h5d_tool.py); open them so the guard is what decides here. admit_soon
    # is stubbed so no case can schedule a real status read.
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admit_soon", lambda runtime: None)


def _admitted(client: Any, tmp_path: Path) -> BridgeRuntime:
    runtime = _runtime(client)
    runtime.config_dir = tmp_path
    runtime.admission = "admitted"
    return runtime


# --- the credential guard ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("op", "extra"), MESSAGE_OPS, ids=[op for op, _ in MESSAGE_OPS])
@pytest.mark.parametrize(
    ("case", "body", "secret", "label"), SECRET_CASES, ids=[c[0] for c in SECRET_CASES]
)
async def test_a_credential_is_refused_before_any_client_call(
    tmp_path: Path, op: str, extra: dict[str, Any], case: str, body: str, secret: str, label: str
) -> None:
    del case
    client = StrictClient()
    field = "text" if op in ("send", "reply") else "body"
    outcome = await execute_bridge(_admitted(client, tmp_path), {"op": op, field: body, **extra})

    assert outcome["ok"] is False
    assert outcome["code"] == SECRET_DETECTED_CODE
    assert label in outcome["classes"]
    assert client.touched == []
    # No local trace either: no outbox intent, no reply-guard slot or key.
    assert list(tmp_path.iterdir()) == []
    # The value never comes back, in any field.
    assert secret not in json.dumps(outcome)
    assert VAULT_HANDOFF_SENTENCE in outcome["error"]


@pytest.mark.asyncio
async def test_the_alias_field_is_scanned_too(tmp_path: Path) -> None:
    """``text`` on channel_post is folded onto ``body`` first, so it cannot dodge."""

    client = StrictClient()
    outcome = await execute_bridge(
        _admitted(client, tmp_path),
        {"op": "channel_post", "channel_id": "bcv_abcd1234", "text": f"key {AWS_KEY_ID}"},
    )
    assert outcome["code"] == SECRET_DETECTED_CODE and client.touched == []


def test_the_refusal_never_logs_the_value(caplog: pytest.LogCaptureFixture) -> None:
    from local_operator.team_bridge.secret_guard import refuse_secret_in

    with caplog.at_level("DEBUG"):
        refused = refuse_secret_in("dm", f"Secret access key: {AWS_SECRET}")
    assert refused is not None
    assert AWS_SECRET not in caplog.text
    assert "aws-secret-access-key" in caplog.text


@pytest.mark.parametrize("body", BENIGN)
def test_near_misses_are_not_credentials(body: str) -> None:
    assert credential_classes(body) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("body", BENIGN[:4])
async def test_a_benign_post_still_goes_out(tmp_path: Path, body: str) -> None:
    client = RecordingClient()
    outcome = await execute_bridge(
        _admitted(client, tmp_path),
        {"op": "channel_post", "channel_id": "bcv_abcd1234", "body": body},
    )
    assert outcome["ok"] is True
    assert [c["method"] for c in client.calls] == ["conversation_send"]
    assert client.calls[0]["text"] == body


def test_a_40_character_value_needs_its_aws_label() -> None:
    """A bare 40-char base64-alphabet run is a hash as often as a secret."""

    assert len(AWS_SECRET) == 40 and len(AWS_KEY_ID) == len(AWS_STS_KEY_ID) == 20
    assert credential_classes(f"artifact digest {AWS_SECRET}") == []
    assert credential_classes(f"SecretAccessKey: {AWS_SECRET}") == ["aws-secret-access-key"]


# --- the edge block ---------------------------------------------------------

WAF_PAGE = "<html><head><title>403 Forbidden</title></head><body>403 Forbidden</body></html>"


def test_an_html_403_is_the_edge_with_its_headers_and_never_its_body() -> None:
    refusal = _refusal(
        httpx.Response(
            403,
            text=WAF_PAGE,
            headers={"content-type": "text/html; charset=utf-8", "server": "awselb/2.0"},
        )
    )
    assert refusal.code == "edge_blocked"
    assert refusal.http_status == 403
    assert refusal.details == {"content_type": "text/html; charset=utf-8", "server": "awselb/2.0"}
    assert "before it reached Radient" in refusal.message
    assert "nothing was posted" in refusal.message
    assert "Forbidden" not in refusal.message


def test_a_bodyless_403_is_the_edge_and_hostile_headers_are_dropped() -> None:
    refusal = _refusal(httpx.Response(403, headers={"server": "evil\x1b[31m" + "x" * 200}))
    assert refusal.code == "edge_blocked"
    assert refusal.details == {}


def test_the_hubs_own_json_403_keeps_its_code() -> None:
    refusal = _refusal(
        httpx.Response(
            403,
            json={"code": "agent_not_permitted", "error": "That agent is not allowed here."},
        )
    )
    assert refusal.code == "agent_not_permitted"
    assert refusal.message == "That agent is not allowed here."


@pytest.mark.parametrize(
    "response",
    [
        # A gateway page is a failure, not a block.
        httpx.Response(502, text=WAF_PAGE, headers={"content-type": "text/html"}),
        # The uncoded 404 is the outdated-hub signal and must stay generic.
        httpx.Response(404, text="Not Found"),
        # An uncoded JSON 403 is the hub answering, just without a code.
        httpx.Response(403, json={"error": "forbidden"}),
    ],
)
def test_only_a_non_json_403_is_re_coded(response: httpx.Response) -> None:
    assert _refusal(response).code != "edge_blocked"


@pytest.mark.asyncio
async def test_channel_post_names_the_edge_block_and_its_evidence(tmp_path: Path) -> None:
    edge = _refusal(
        httpx.Response(
            403, text=WAF_PAGE, headers={"content-type": "text/html", "server": "awselb/2.0"}
        )
    )

    class EdgeClient(RecordingClient):
        async def conversation_send(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise edge

    outcome = await execute_bridge(
        _admitted(EdgeClient(), tmp_path),
        {"op": "channel_post", "channel_id": "bcv_abcd1234", "body": "see ../notes for context"},
    )
    assert outcome["ok"] is False and outcome["code"] == "post_failed"
    assert "edge firewall" in outcome["error"]
    assert outcome["error"].endswith(
        "[refusal code=edge_blocked status=403 server=awselb/2.0 content_type=text/html]"
    )
    assert "Forbidden" not in outcome["error"]


# --- agent -> person DM -----------------------------------------------------


def _hub_refusal(field: str) -> BridgeRefusal:
    return BridgeRefusal(
        "invalid_request",
        "This messaging request is not valid. See details for the field.",
        http_status=400,
        details={"field": field, "rule": "agent_required"},
    )


class HumanDmClient(RecordingClient):
    def __init__(self, failure: BaseException) -> None:
        super().__init__()
        self.failure = failure

    async def directory(self, tenant_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "directory"})
        return {
            "members": [
                {
                    "account_id": "acc_1",
                    "display_label": "Owner",
                    "bridge_enabled": True,
                    "bridge_state": "active",
                    "agent": {"endpoint_id": "bep_1", "name": "Mine", "state": "active"},
                },
                {
                    "account_id": "acc_2",
                    "display_label": "Dana",
                    "bridge_enabled": True,
                    "bridge_state": "active",
                },
            ]
        }

    async def create_conversation(self, body: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "create_conversation", "body": dict(body)})
        raise self.failure


@pytest.mark.asyncio
async def test_the_hubs_agent_only_dm_rule_is_answered_honestly(tmp_path: Path) -> None:
    client = HumanDmClient(_hub_refusal("members"))
    outcome = await execute_bridge(
        _admitted(client, tmp_path),
        {"op": "dm", "account_id": "acc_2", "target": "human", "body": "hi Dana"},
    )
    assert outcome == {
        "ok": False,
        "code": "dm_human_unsupported",
        "error": (
            "Nothing was sent: this Radient hub can't open an agent→person DM yet. Post in "
            "a private group with them instead: channel_create kind=group with them as a "
            "member, then channel_post to it."
        ),
    }
    assert [c["method"] for c in client.calls] == ["directory", "create_conversation"]


@pytest.mark.asyncio
async def test_the_same_rule_on_another_field_stays_dm_failed(tmp_path: Path) -> None:
    """``rule=agent_required`` also guards ``origin_session_id``; that is not this."""

    client = HumanDmClient(_hub_refusal("origin_session_id"))
    outcome = await execute_bridge(
        _admitted(client, tmp_path),
        {"op": "dm", "account_id": "acc_2", "target": "human", "body": "hi Dana"},
    )
    assert outcome["code"] == "dm_failed"
    assert "field=origin_session_id rule=agent_required" in outcome["error"]
