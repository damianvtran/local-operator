"""The agent picture: what the local definition says, and what reaches the hub.

Runs the REAL resolver over real ``agents/<id>/agent.yml`` files and a real Aida
state file, and the real client/discipline over ``httpx.MockTransport``, so what
is pinned is what ships: which file is read, which digest is sent, when a write
happens at all, and — the part that costs other people their picture if it is
wrong — when NOTHING must happen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import avatar_publish
from local_operator.team_bridge.client import Boundary, BridgeRefusal
from local_operator.team_bridge.cos_name import (
    AVATAR_MAX_BYTES,
    cos_display_name,
    degenerate_agent_name,
    published_agent,
)
from tests.unit.team_bridge.test_client import FakeRow, FakeStore, _make_client, _ok
from tests.unit.team_bridge.test_cos_name import (  # noqa: F401 — autouse fixture
    _destination_open,
    _aida_lives_in,
    _designate,
    _enrolled_aida,
)
from tests.unit.team_bridge.test_discipline import (
    _committed_owner,
    _Counter,
    _discipline,
    _endpoint_payload,
)

pytestmark = pytest.mark.asyncio

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 8
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"pixels" * 8
PNG_SHA = hashlib.sha256(PNG).hexdigest()
WEBP_SHA = hashlib.sha256(WEBP).hexdigest()


def _agent(
    root: Path,
    *,
    name: str = "aida",
    agent_id: str = "aida",
    block: str | None = None,
    files: dict[str, bytes] | None = None,
) -> Path:
    """Write one ``agents/<id>`` definition with an optional ``avatar`` block."""

    agent_dir = root / "agents" / agent_dir_id(agent_id)
    agent_dir.mkdir(parents=True, exist_ok=True)
    body = f"id: {agent_dir.name}\nname: {name}\nversion: '1.0'\n"
    if block is not None:
        body += block
    (agent_dir / "agent.yml").write_text(body, encoding="utf-8")
    for filename, data in (files or {}).items():
        (agent_dir / filename).write_bytes(data)
    return agent_dir


def agent_dir_id(value: str) -> str:
    """The directory name an agent id is stored under (no path semantics)."""

    return value


def _png_block(*, filename: str | None = "avatar.png", sha: str = PNG_SHA, size: int | None = None):
    lines = ["avatar:", f"  sha256: {sha}", "  mime: image/png", f"  bytes: {size or len(PNG)}"]
    if filename is not None:
        lines.append(f"  filename: {filename}")
    return "\n".join(lines) + "\n"


def _enrolled(root: Path, name: str = "Aida") -> None:
    _enrolled_aida(root, name)


# -- what the local definition says -----------------------------------------


async def test_a_stored_picture_resolves_with_its_recorded_digest(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})

    agent = published_agent(tmp_path)

    assert agent is not None
    assert (agent.local_agent_id, agent.name) == ("aida", "Aida")
    assert agent.avatar_path is not None and agent.avatar_path.name == "avatar.png"
    assert agent.avatar_sha256 == PNG_SHA
    assert agent.avatar_broken is False


async def test_the_filename_field_wins_and_a_webp_store_reads_back(tmp_path: Path):
    """lopdev's store may end up ``avatar.webp``; the block names it, not us."""

    _enrolled(tmp_path)
    block = "avatar:\n  sha256: {sha}\n  mime: image/webp\n  bytes: {n}\n  filename: avatar.webp\n"
    _agent(tmp_path, block=block.format(sha=WEBP_SHA, n=len(WEBP)), files={"avatar.webp": WEBP})

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_path is not None
    assert agent.avatar_path.name == "avatar.webp"


async def test_a_legacy_block_without_a_filename_falls_back_to_its_mime(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(filename=None), files={"avatar.png": PNG})

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_path is not None
    assert agent.avatar_path.name == "avatar.png"


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("../../etc/passwd", id="traversal"),
        pytest.param("/etc/passwd", id="absolute"),
        pytest.param("nested/avatar.png", id="separator"),
        pytest.param("..", id="dotdot"),
    ],
)
async def test_a_name_outside_the_agents_own_directory_is_a_doubt(tmp_path: Path, filename: str):
    """A hand-edited block is untrusted input: it may never name a file to publish."""

    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(filename=filename), files={"avatar.png": PNG})

    agent = published_agent(tmp_path)

    assert agent is not None
    assert agent.avatar_path is None
    assert agent.avatar_broken is True


async def test_a_symlink_out_of_the_directory_is_a_doubt(tmp_path: Path):
    _enrolled(tmp_path)
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(PNG)
    agent_dir = _agent(tmp_path, block=_png_block(filename="avatar.png"))
    (agent_dir / "avatar.png").symlink_to(outside)

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_path is None and agent.avatar_broken is True


async def test_a_block_whose_file_is_gone_is_a_doubt_not_a_clear(tmp_path: Path):
    """A transient filesystem fault must never wipe the hub's copy."""

    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={})

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_path is None and agent.avatar_broken is True


async def test_an_oversized_file_is_a_doubt(tmp_path: Path):
    _enrolled(tmp_path)
    huge = b"x" * (AVATAR_MAX_BYTES + 1)
    _agent(tmp_path, block=_png_block(size=len(huge)), files={"avatar.png": huge})

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_broken is True


async def test_no_avatar_block_is_a_deliberate_clear(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=None)

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_path is None and agent.avatar_broken is False


async def test_no_agent_definition_behind_her_publishes_nothing(tmp_path: Path):
    _enrolled(tmp_path)

    assert published_agent(tmp_path) is None
    assert published_agent(None) is None


async def test_a_designated_session_that_is_not_her_publishes_nothing(tmp_path: Path):
    _aida_lives_in(tmp_path)
    _designate(tmp_path, "sess_other")
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})

    assert published_agent(tmp_path) is None
    assert cos_display_name(tmp_path) is None


async def test_the_name_still_resolves_without_any_agent_definition(tmp_path: Path):
    """The avatar lookup must never turn the name into ``None`` (its old shape)."""

    _enrolled(tmp_path)

    assert cos_display_name(tmp_path) == "Aida"


class AvatarStore(FakeStore):
    """The credential store avatar reads need: a stable login incarnation."""

    incarnation: str | None = "login-a"

    def native_login_incarnation(self, credential_id: Any) -> str | None:
        return self.incarnation


def _avatar_client(handler: Any) -> Any:
    from local_operator.team_bridge.client import TeamBridgeClient

    return TeamBridgeClient(
        transport=httpx.MockTransport(handler),
        store=AvatarStore(rows=[FakeRow(1, data={"account_id": "acc_1", "access": "token-1"})]),
    )


# -- one reconcile -----------------------------------------------------------


class FakeClient:
    """The three calls the publisher needs, recorded."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.puts: list[tuple[str, bytes, str]] = []
        self.deletes: list[str] = []
        self.failure: Exception | None = None

    async def status(self, *, access: Any = None) -> dict[str, Any]:
        return self.payload

    async def current_access(self) -> Any:
        return SimpleNamespace(account_id="acc_1")

    async def put_endpoint_avatar(
        self,
        endpoint_id: str,
        data: bytes,
        source_sha256: str,
        *,
        expected_account_id: str | None = None,
    ) -> dict[str, Any]:
        if self.failure is not None:
            raise self.failure
        self.puts.append((endpoint_id, data, source_sha256))
        return {"endpoint_id": endpoint_id}

    async def delete_endpoint_avatar(
        self, endpoint_id: str, *, expected_account_id: str | None = None
    ) -> dict[str, Any]:
        if self.failure is not None:
            raise self.failure
        self.deletes.append(endpoint_id)
        return {}


def _status(hub_sha: str | None = None, *, holds: bool | None = None, label: str = "Org agent"):
    """One status answer for one tenant, with the hub's ``avatar`` block."""

    payload = _endpoint_payload("active")
    endpoint = payload["tenants"][0]["endpoint"]
    endpoint["label"] = label
    if holds is not False:
        block: dict[str, Any] = {"hash": "0123456789abcdef", "updated_at": "2026-10-30T00:00:00Z"}
        if hub_sha is not None:
            block["source_sha256"] = hub_sha
        endpoint["avatar"] = block
    return SimpleNamespace(payload=payload, account_id="acc_1")


async def test_a_different_digest_publishes_the_local_file(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    client = FakeClient(_status(hub_sha="0" * 64).payload)

    await avatar_publish.reconcile(client, _status(hub_sha="0" * 64), "tnt_1", tmp_path)

    assert client.puts == [("bep_1", PNG, PNG_SHA)]
    assert client.deletes == []


async def test_the_same_digest_writes_nothing(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    client = FakeClient(_status(hub_sha=PNG_SHA).payload)

    await avatar_publish.reconcile(client, _status(hub_sha=PNG_SHA), "tnt_1", tmp_path)

    assert client.puts == [] and client.deletes == []


async def test_no_local_picture_clears_a_hub_copy(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=None)
    client = FakeClient(_status(hub_sha=PNG_SHA).payload)

    await avatar_publish.reconcile(client, _status(hub_sha=PNG_SHA), "tnt_1", tmp_path)

    assert client.deletes == ["bep_1"] and client.puts == []


async def test_no_local_picture_and_nothing_at_the_hub_writes_nothing(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=None)
    client = FakeClient({})

    await avatar_publish.reconcile(client, _status(holds=False), "tnt_1", tmp_path)

    assert client.deletes == [] and client.puts == []


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param("broken", id="block-present-file-unresolvable"),
        pytest.param("no-agent", id="no-agent-definition"),
        pytest.param("no-cos", id="no-designated-chief-of-staff"),
    ],
)
async def test_a_doubt_never_deletes_the_hub_copy(tmp_path: Path, setup: str):
    """The single most expensive mistake here: a doubt read as a clear."""

    if setup == "broken":
        _enrolled(tmp_path)
        _agent(tmp_path, block=_png_block(), files={})
    elif setup == "no-agent":
        _enrolled(tmp_path)
    else:
        # No Aida state at all: nothing designates a chief of staff, so this
        # machine has no opinion about the hub's copy.
        _agent(tmp_path, block=None)
    client = FakeClient(_status(hub_sha=PNG_SHA).payload)

    await avatar_publish.reconcile(client, _status(hub_sha=PNG_SHA), "tnt_1", tmp_path)

    assert client.deletes == [] and client.puts == []


async def test_a_hub_holding_nothing_is_the_ordinary_publish(tmp_path: Path):
    """Absence of the ``avatar`` block is "nothing published", never "old hub"."""

    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    fresh = _endpoint_payload("active")
    client = FakeClient(fresh)

    await avatar_publish.reconcile(
        client, SimpleNamespace(payload=fresh, account_id="acc_1"), "tnt_1", tmp_path
    )

    assert client.puts == [("bep_1", PNG, PNG_SHA)] and client.deletes == []


async def test_a_failure_never_raises_and_leaves_the_status_answer_untouched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    status = _status(hub_sha="0" * 64)
    before = json.dumps(status.payload, sort_keys=True)
    client = FakeClient(status.payload)
    client.failure = BridgeRefusal("avatar_too_large", "refused", http_status=413)

    with caplog.at_level("WARNING", logger="local_operator.team_bridge.avatar_publish"):
        await avatar_publish.reconcile(client, status, "tnt_1", tmp_path)

    assert json.dumps(status.payload, sort_keys=True) == before
    assert status.account_id == "acc_1"
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "BridgeRefusal" in logged and "avatar_too_large" in logged
    # Never the bytes, a path, the agent's id or the digest.
    assert tmp_path.name not in logged and PNG_SHA not in logged and "aida" not in logged


async def test_a_hand_edited_file_is_published_under_its_true_digest(tmp_path: Path):
    """The recorded digest is stale for a hand edit; the SIZE is what catches it."""

    _enrolled(tmp_path)
    edited = PNG + b"tampered"
    _agent(tmp_path, block=_png_block(), files={"avatar.png": edited})
    client = FakeClient(_status(hub_sha=PNG_SHA).payload)

    await avatar_publish.reconcile(client, _status(hub_sha=PNG_SHA), "tnt_1", tmp_path)

    assert client.puts == [("bep_1", edited, hashlib.sha256(edited).hexdigest())]


async def test_the_explicit_publish_forces_the_write_whatever_the_hub_holds(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    client = FakeClient(_status(hub_sha=PNG_SHA).payload)

    plan = await avatar_publish.publish_agent_avatar(client, "tnt_1", tmp_path)

    assert plan.action == "publish" and client.puts == [("bep_1", PNG, PNG_SHA)]


async def test_the_explicit_publish_has_nothing_to_send_without_a_picture(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=None)
    client = FakeClient(_status(hub_sha=PNG_SHA).payload)

    plan = await avatar_publish.publish_agent_avatar(client, "tnt_1", tmp_path)

    assert plan.action == "none" and client.puts == [] and client.deletes == []


async def test_the_explicit_clear_deletes_the_hub_copy_only(tmp_path: Path):
    _enrolled(tmp_path)
    agent_dir = _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    client = FakeClient(_status(hub_sha=PNG_SHA).payload)

    plan = await avatar_publish.clear_agent_avatar(client, "tnt_1")

    assert plan.action == "delete" and client.deletes == ["bep_1"]
    # The local store is the local writer's: the file it owns is still there.
    assert (agent_dir / "avatar.png").read_bytes() == PNG


# -- the name rule -----------------------------------------------------------


@pytest.mark.parametrize(
    ("agent_name", "label", "degenerate"),
    [
        ("Sherman Tsui", "Sherman Tsui", True),
        ("sherman   tsui", "Sherman Tsui", True),
        ("Aida", "Sherman Tsui", False),
        ("Sherman Tsui", "My agent", False),  # a label the user renamed past
        ("", "Sherman Tsui", False),  # already cleared: never re-PUT
        (None, "Sherman Tsui", False),
    ],
)
async def test_the_degenerate_name_rule(agent_name: Any, label: Any, degenerate: bool):
    assert degenerate_agent_name(agent_name, label) is degenerate


async def test_a_degenerate_hub_name_is_cleared_at_the_boundary(tmp_path: Path):
    """The default-on rows nobody typed a name into: send ``agent_name: ""``."""

    # The committed owner with NO designated CoS: this machine has no name to
    # publish, which is exactly the state the clearing arm exists for.
    _committed_owner(tmp_path)
    payload = _endpoint_payload("active")
    payload["tenants"][0]["endpoint"].update(
        {"agent_name": "Sherman Tsui", "label": "Sherman Tsui"}
    )
    counter = _Counter(lambda request: _ok(payload))

    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)

    puts = [json.loads(r.content) for r in counter.requests if r.method == "PUT"]
    assert check.state == "ready"
    assert puts and puts[-1]["agent_name"] == ""
    assert puts[-1]["state"] == "active" and puts[-1]["tenant_id"] == "tnt_1"


async def test_a_chosen_hub_name_is_never_cleared_locally(tmp_path: Path):
    _enrolled_aida(tmp_path)
    payload = _endpoint_payload("active")
    payload["tenants"][0]["endpoint"].update({"agent_name": "Friday", "label": "Sherman Tsui"})
    counter = _Counter(lambda request: _ok(payload))

    await _discipline(counter, tmp_path).check(Boundary.APP_START)

    # The local side resolves "Aida", which differs from the echo: the ordinary
    # rename arm re-sends it. It must never degrade into a clearing write.
    puts = [json.loads(r.content) for r in counter.requests if r.method == "PUT"]
    assert [body["agent_name"] for body in puts] == ["Aida"]


# -- the hub's avatar routes -------------------------------------------------


def _multipart_echo(request: httpx.Request) -> httpx.Response:
    assert request.headers["x-source-sha256"] == PNG_SHA
    assert b'name="file"' in request.content
    assert PNG in request.content
    return httpx.Response(
        200,
        json={
            "msg": "ok",
            "result": {
                "endpoint_id": "bep_1",
                "hash": "0123456789abcdef",
                "source_sha256": PNG_SHA,
                "variants": {"64": {"content_type": "image/png", "bytes": 10}},
                "updated_at": "2026-10-30T00:00:00Z",
            },
        },
    )


async def test_the_publish_is_multipart_with_the_source_digest_header():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _multipart_echo(request)

    client = _make_client(handler)
    result = await client.put_endpoint_avatar("bep_1", PNG, PNG_SHA, expected_account_id="acc_1")

    assert seen[0].method == "PUT"
    assert seen[0].url.path == "/v1/messaging/endpoints/bep_1/avatar"
    assert result["hash"] == "0123456789abcdef"


async def test_a_204_from_the_publish_is_a_success_with_no_body():
    client = _make_client(lambda request: httpx.Response(204))

    assert (
        await client.put_endpoint_avatar("bep_1", PNG, PNG_SHA, expected_account_id="acc_1") == {}
    )


async def test_the_clear_is_one_idempotent_delete():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    client = _make_client(handler)

    assert await client.delete_endpoint_avatar("bep_1", expected_account_id="acc_1") == {}
    assert [(r.method, r.url.path) for r in seen] == [
        ("DELETE", "/v1/messaging/endpoints/bep_1/avatar")
    ]


@pytest.mark.parametrize(
    "endpoint_id",
    ["", "bep_1/../../x", "bep 1", "a" * 129],
)
async def test_an_endpoint_id_that_could_reshape_the_path_is_refused(endpoint_id: str):
    client = _make_client(lambda request: httpx.Response(204))

    with pytest.raises(ValueError):
        await client.put_endpoint_avatar(endpoint_id, PNG, PNG_SHA)
    with pytest.raises(ValueError):
        await client.delete_endpoint_avatar(endpoint_id)


async def test_an_oversized_picture_is_refused_before_any_request():
    seen: list[httpx.Request] = []
    client = _make_client(lambda request: seen.append(request) or httpx.Response(204))

    with pytest.raises(BridgeRefusal) as info:
        await client.put_endpoint_avatar(
            "bep_1", b"x" * (AVATAR_MAX_BYTES + 1), PNG_SHA, expected_account_id="acc_1"
        )

    assert info.value.code == "avatar_too_large"
    assert info.value.details == {"limit_bytes": AVATAR_MAX_BYTES}
    assert seen == []


async def test_a_malformed_digest_is_refused_locally():
    client = _make_client(lambda request: httpx.Response(204))

    with pytest.raises(ValueError):
        await client.put_endpoint_avatar("bep_1", PNG, PNG_SHA.upper(), expected_account_id="acc_1")


# -- the boundary wiring ------------------------------------------------------


async def test_a_fresh_status_read_publishes_through_the_shared_client(tmp_path: Path):
    """The whole point: the ONE existing boundary does the write, unprompted."""

    # `_enrolled` commits the owner AND designates the CoS: the real state.
    _enrolled(tmp_path)
    agent = _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    status = _endpoint_payload("active")
    endpoint = status["tenants"][0]["endpoint"]
    endpoint.update({"agent_name": None})
    endpoint["avatar"] = {"hash": "0123456789abcdef", "source_sha256": "0" * 64}
    seen: list[httpx.Request] = []

    def responder(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "PUT":
            return _multipart_echo(request)
        return _ok(status)

    await _discipline(_Counter(responder), tmp_path).check(Boundary.APP_START)

    # The name re-sync shares this boundary and may also PUT; the picture is the
    # ONE request this test is about, and it carries the local file's digest.
    puts = [r for r in seen if r.method == "PUT"]
    avatar = [r for r in puts if r.url.path == "/v1/messaging/endpoints/bep_1/avatar"]
    assert len(avatar) == 1 and avatar[0].headers["x-source-sha256"] == PNG_SHA
    assert PNG in avatar[0].content
    assert agent.exists()


async def test_a_status_answer_is_never_changed_by_a_failed_publish(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    status = _endpoint_payload("active")
    status["tenants"][0]["endpoint"]["avatar"] = {
        "hash": "0123456789abcdef",
        "source_sha256": "0" * 64,
    }
    before = json.dumps(status, sort_keys=True)

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return httpx.Response(404, json={"error": "no such endpoint", "code": "not_found"})
        return _ok(status)

    check = await _discipline(_Counter(responder), tmp_path).check(Boundary.APP_START)

    assert check.state == "ready"
    # The served payload is untouched: a refusal on the picture is not a status error.
    assert json.dumps(status, sort_keys=True) == before


# -- the read route ----------------------------------------------------------


async def test_the_agent_picture_read_follows_the_account_read_shape():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, content=WEBP, headers={"content-type": "image/webp", "etag": '"abc-64"'}
        )

    client = _avatar_client(handler)
    body, mime = await client.fetch_endpoint_avatar(
        "bep_1", v="0123456789abcdef", size=64, expected_account_id="acc_1"
    )

    assert (body, mime) == (WEBP, "image/webp")
    assert seen[0].url.path == "/v1/messaging/endpoints/bep_1/avatar"
    assert dict(seen[0].url.params) == {"v": "0123456789abcdef", "size": "64"}


@pytest.mark.parametrize("size", [0, 32, 512, True])
async def test_the_agent_picture_read_refuses_a_size_outside_the_closed_set(size: Any):
    client = _avatar_client(lambda request: httpx.Response(200))

    with pytest.raises(ValueError):
        await client.fetch_endpoint_avatar(
            "bep_1", v="0123456789abcdef", size=size, expected_account_id="acc_1"
        )


async def test_a_stale_picture_version_is_a_refusal_not_a_fresh_image():
    client = _avatar_client(
        lambda request: httpx.Response(404, json={"error": "stale", "code": "not_found"})
    )

    with pytest.raises(BridgeRefusal) as info:
        await client.fetch_endpoint_avatar(
            "bep_1", v="ffffffffffffffff", size=64, expected_account_id="acc_1"
        )

    assert info.value.code == "not_found"
