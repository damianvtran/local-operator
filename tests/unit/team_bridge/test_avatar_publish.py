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
    _aida_lives_in,
    _designate,
    _destination_open,
    _enrolled_aida,
)
from tests.unit.team_bridge.test_discipline import (
    _committed_owner,
    _Counter,
    _discipline,
    _endpoint_payload,
)

pytestmark = pytest.mark.asyncio


def _canonical_png() -> bytes:
    from io import BytesIO
    from PIL import Image
    from local_operator import agent_avatar

    buffer = BytesIO()
    Image.new("RGBA", (512, 512), (100, 20, 80, 255)).save(buffer, format="PNG")
    return agent_avatar.normalise(buffer.getvalue()).data


PNG = _canonical_png()
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
    # ``updated_at`` is required by the store's own model (``AgentAvatarMeta``), and
    # the publisher resolves through that model, so a block without it is a doubt.
    lines = [
        "avatar:",
        f"  sha256: {sha}",
        "  mime: image/png",
        f"  bytes: {size or len(PNG)}",
        "  updated_at: '2026-10-30T00:00:00+00:00'",
    ]
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
    assert agent.avatar_data == PNG
    assert agent.avatar_sha256 == PNG_SHA
    assert agent.avatar_broken is False


async def test_a_webp_block_is_a_doubt_because_the_store_never_writes_one(tmp_path: Path):
    """R3-F3: the store holds ``avatar.png``/``avatar.jpg`` only (it re-encodes).

    A ``webp`` block therefore names nothing the store would serve, and the
    publisher follows the store: a doubt, never a publish and never a delete.
    """

    _enrolled(tmp_path)
    block = (
        "avatar:\n  sha256: {sha}\n  mime: image/webp\n  bytes: {n}\n"
        "  updated_at: '2026-10-30T00:00:00+00:00'\n  filename: avatar.webp\n"
    )
    _agent(tmp_path, block=block.format(sha=WEBP_SHA, n=len(WEBP)), files={"avatar.webp": WEBP})

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True


async def test_a_legacy_block_without_a_filename_falls_back_to_its_mime(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(filename=None), files={"avatar.png": PNG})

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_data == PNG


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("../../etc/passwd", id="traversal"),
        pytest.param("/etc/passwd", id="absolute"),
        pytest.param("nested/avatar.png", id="separator"),
        pytest.param("..", id="dotdot"),
        pytest.param(".", id="dot"),
        pytest.param("./../avatar.png", id="dot-slash-traversal"),
    ],
)
async def test_a_name_outside_the_agents_own_directory_is_a_doubt(tmp_path: Path, filename: str):
    """A hand-edited block is untrusted input: it may never name a file to publish."""

    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(filename=filename), files={"avatar.png": PNG})

    agent = published_agent(tmp_path)

    assert agent is not None
    assert agent.avatar_data is None
    assert agent.avatar_broken is True


async def test_a_symlink_out_of_the_directory_is_a_doubt(tmp_path: Path):
    _enrolled(tmp_path)
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(PNG)
    agent_dir = _agent(tmp_path, block=_png_block(filename="avatar.png"))
    (agent_dir / "avatar.png").symlink_to(outside)

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True


# -- the publisher resolves THROUGH the store (R3-F1/F2) -----------------------
#
# ``agent.yml`` is writable by the agent itself and the hub copy is visible to
# every teammate in the tenant, so "which file is the avatar" must be the
# STORE's answer. Each cell below runs the real boundary against a recording
# transport and asserts the file's CONTENT never appears in any request — an
# exit code or an empty ``puts`` list alone would not notice a body that went out
# under a different request.

SECRET = "security_prompt: NEVER-REVEAL-THIS-PRIVATE-PROMPT"


def _jpeg_block(*, mime: str = "image/png", filename: str = "avatar.jpg") -> str:
    return (
        "avatar:\n"
        f"  sha256: {PNG_SHA}\n"
        f"  mime: {mime}\n"
        f"  bytes: {len(PNG)}\n"
        "  updated_at: '2026-10-30T00:00:00+00:00'\n"
        f"  filename: {filename}\n"
    )


async def _boundary_requests(root: Path) -> list[httpx.Request]:
    """Run ONE real status boundary against a hub that holds a different picture."""

    status = _endpoint_payload("active")
    status["tenants"][0]["endpoint"].update({"agent_name": None})
    status["tenants"][0]["endpoint"]["avatar"] = {
        "hash": "0123456789abcdef",
        "source_sha256": "0" * 64,
    }
    seen: list[httpx.Request] = []

    def responder(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/avatar") and request.method == "PUT":
            return _multipart_echo(request)
        return _ok(status)

    await _discipline(_Counter(responder), root).check(Boundary.APP_START)
    return seen


def _avatar_traffic(seen: list[httpx.Request]) -> list[httpx.Request]:
    return [r for r in seen if r.url.path.endswith("/avatar")]


def _doubt_agent(root: Path, *, block: str, files: dict[str, bytes] | None = None) -> Path:
    agent_dir = _agent(root, block=block, files=files)
    # The private text lives in agent.yml itself, exactly where the real one does.
    with (agent_dir / "agent.yml").open("a", encoding="utf-8") as handle:
        handle.write(SECRET + "\n")
    return agent_dir


def _never_reached(seen: list[httpx.Request]) -> None:
    assert _avatar_traffic(seen) == [], "a doubt must send neither a PUT nor a DELETE"
    for request in seen:
        assert b"NEVER-REVEAL" not in request.content
        assert b"security_prompt" not in request.content


async def test_a_filename_naming_agent_yml_is_a_doubt_and_discloses_nothing(tmp_path: Path):
    """R3-F1(a): ``filename: agent.yml`` once uploaded the agent's private prompt."""

    _enrolled(tmp_path)
    _doubt_agent(tmp_path, block=_png_block(filename="agent.yml"), files={"avatar.png": PNG})

    agent = published_agent(tmp_path)
    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True

    _never_reached(await _boundary_requests(tmp_path))


async def test_a_symlink_to_agent_yml_is_a_doubt_and_discloses_nothing(tmp_path: Path):
    """R3-F1(b): ``avatar.png -> agent.yml`` is a same-directory link, so the old
    containment check (parent == directory) accepted it."""

    _enrolled(tmp_path)
    agent_dir = _doubt_agent(tmp_path, block=_png_block())
    (agent_dir / "avatar.png").symlink_to(agent_dir / "agent.yml")

    agent = published_agent(tmp_path)
    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True

    _never_reached(await _boundary_requests(tmp_path))


async def test_a_symlink_to_a_file_outside_the_agent_is_a_doubt_and_discloses_nothing(
    tmp_path: Path,
):
    """R3-F1(c): the target is outside the agent directory altogether."""

    _enrolled(tmp_path)
    outside = tmp_path / "private-notes.txt"
    outside.write_text(SECRET + "\n", encoding="utf-8")
    agent_dir = _doubt_agent(tmp_path, block=_png_block())
    (agent_dir / "avatar.png").symlink_to(outside)

    agent = published_agent(tmp_path)
    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True

    _never_reached(await _boundary_requests(tmp_path))


async def test_a_filename_that_disagrees_with_the_mime_is_a_doubt(tmp_path: Path):
    """R3-F1(d): ``avatar.jpg`` recorded as ``image/png`` is no file the store serves."""

    _enrolled(tmp_path)
    _doubt_agent(tmp_path, block=_jpeg_block(), files={"avatar.jpg": PNG})

    agent = published_agent(tmp_path)
    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True

    _never_reached(await _boundary_requests(tmp_path))


async def test_a_digest_mismatch_is_a_doubt_and_the_bytes_never_leave(tmp_path: Path):
    """R3-F2(e): the store refuses a file that does not hash to its block."""

    _enrolled(tmp_path)
    other = b"\x89PNG\r\n\x1a\n" + b"NEVER-REVEAL-OTHER-BYTES"
    _doubt_agent(tmp_path, block=_png_block(), files={"avatar.png": other})

    agent = published_agent(tmp_path)
    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True

    _never_reached(await _boundary_requests(tmp_path))


async def test_a_normal_block_still_publishes_exactly_its_picture(tmp_path: Path):
    """R3-F1(f): the control. The same rig, a valid block: the PNG goes, nothing else."""

    _enrolled(tmp_path)
    _doubt_agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})

    seen = await _boundary_requests(tmp_path)

    puts = [r for r in _avatar_traffic(seen) if r.method == "PUT"]
    assert len(puts) == 1 and puts[0].headers["x-source-sha256"] == PNG_SHA
    assert PNG in puts[0].content
    for request in seen:
        assert b"NEVER-REVEAL" not in request.content


async def test_a_block_whose_file_is_gone_is_a_doubt_not_a_clear(tmp_path: Path):
    """A transient filesystem fault must never wipe the hub's copy."""

    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={})

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True


async def test_an_oversized_file_is_a_doubt(tmp_path: Path):
    _enrolled(tmp_path)
    huge = b"x" * (AVATAR_MAX_BYTES + 1)
    _agent(tmp_path, block=_png_block(size=len(huge)), files={"avatar.png": huge})

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_broken is True


async def test_no_avatar_block_is_a_deliberate_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Only once the store PRESERVES the block: absence is then the user's choice."""

    _preserves_avatar(monkeypatch)
    _enrolled(tmp_path)
    _agent(tmp_path, block=None)

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is False


async def test_an_absent_block_is_a_doubt_while_the_store_would_erase_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """R1-F4: the DELETE arm must not depend on an invariant this tree lacks.

    Both ``agents.py`` writers serialise ``AgentData.model_dump()`` and pydantic
    drops an undeclared key, so while ``avatar`` is not a field an ordinary save
    (a rename, a settings edit) removes the block. Reading that absence as "the
    user cleared it" would delete the teammate-visible copy at the next boundary
    and the resolver could never re-publish it.
    """

    from local_operator.agents import AgentData

    if "avatar" in AgentData.model_fields:  # pragma: no cover - the paired PR
        pytest.skip("avatar is already a first-class AgentData field")

    _preserves_avatar(monkeypatch, yes=False)
    _enrolled(tmp_path)
    _agent(tmp_path, block=None)

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True


async def test_the_agent_store_can_preserve_the_avatar_block():
    """CROSS-PR GATE (R1-F4a): the DELETE arm's premise, asserted, not assumed.

    Deliberately RED until the local-storage lane lands ``AgentData.avatar``.
    Both writers round-trip through ``model_dump()``, so without the field the
    block cannot survive an ordinary save; the arm that clears the hub copy on
    absence is only sound once this passes. Do not merge this PR before it.
    """

    from local_operator.agents import AgentData

    assert "avatar" in AgentData.model_fields, (
        "AgentData has no `avatar` field: an ordinary registry save DROPS the "
        "block, so 'no block' cannot mean 'the user cleared it'"
    )


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


def _preserves_avatar(monkeypatch: pytest.MonkeyPatch, *, yes: bool = True) -> None:
    """Declare whether ``AgentData`` carries the avatar field.

    The DELETE arm needs the store to preserve the block: until ``avatar`` is a
    first-class ``AgentData`` field, an absent block means "the registry erased
    it", not "the user cleared it" (review R1-F4). A delete-arm test therefore
    states the world it is testing, and the protection has its own test below.
    """

    from local_operator.team_bridge import cos_name

    monkeypatch.setattr(cos_name, "_AVATAR_FIELD", yes)


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


async def test_no_local_picture_clears_a_hub_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _preserves_avatar(monkeypatch)
    _enrolled(tmp_path)
    _agent(tmp_path, block=None)
    client = FakeClient(_status(hub_sha=PNG_SHA).payload)

    await avatar_publish.reconcile(client, _status(hub_sha=PNG_SHA), "tnt_1", tmp_path)

    assert client.deletes == ["bep_1"] and client.puts == []


async def test_no_local_picture_and_nothing_at_the_hub_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _preserves_avatar(monkeypatch)
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
async def test_a_doubt_never_deletes_the_hub_copy(
    tmp_path: Path, setup: str, monkeypatch: pytest.MonkeyPatch
):
    """The single most expensive mistake here: a doubt read as a clear."""

    if setup == "broken":
        _enrolled(tmp_path)
        _agent(tmp_path, block=_png_block(), files={})
    elif setup == "no-agent":
        _enrolled(tmp_path)
    else:
        # No Aida state at all: nothing designates a chief of staff, so this
        # machine has no opinion about the hub's copy.
        _preserves_avatar(monkeypatch)
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


async def test_a_hand_edited_file_is_a_doubt_not_an_upload(tmp_path: Path):
    """R3-F2: bytes that do not hash to the block's digest are never published.

    The store refuses to serve such a file, so the publisher does too; the hub
    keeps what it has (no PUT) and nothing is deleted.
    """

    _enrolled(tmp_path)
    edited = PNG + b"tampered"
    _agent(tmp_path, block=_png_block(), files={"avatar.png": edited})
    client = FakeClient(_status(hub_sha=PNG_SHA).payload)

    await avatar_publish.reconcile(client, _status(hub_sha=PNG_SHA), "tnt_1", tmp_path)

    assert client.puts == [] and client.deletes == []
    assert edited not in b"".join(p[1] for p in client.puts)


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


# -- round 1 remediation: the cases the review measured ----------------------


async def test_a_same_size_hand_edit_is_a_doubt_not_an_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The size-matching edit is the one a length check cannot see.

    The block records the OLD digest and the file now holds same-length bytes.
    The store's digest check refuses it (R3-F2), so nothing is sent — neither the
    stale digest (R1-F3's 204 trap) nor the new bytes.
    """

    _preserves_avatar(monkeypatch)
    _enrolled(tmp_path)
    edited = bytes([PNG[0] ^ 0xFF]) + PNG[1:]  # same length, different bytes
    assert len(edited) == len(PNG) and edited != PNG
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    (tmp_path / "agents" / "aida" / "avatar.png").write_bytes(edited)
    client = FakeClient(_endpoint_payload("active"))

    await avatar_publish.reconcile(
        client,
        SimpleNamespace(payload=_endpoint_payload("active"), account_id="acc_1"),
        "tnt_1",
        tmp_path,
    )

    assert client.puts == [] and client.deletes == []


async def test_a_paused_endpoint_is_never_published_or_cleared(tmp_path: Path):
    """R1-F6: the state guard must stop the picture path on its own.

    The name path is stopped by its echo comparison before the state is
    consulted; the picture has no echo to short-circuit, so only the guard in
    ``_owned_endpoint`` stands between a paused endpoint and a write that would
    resume it.
    """

    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    status = _endpoint_payload("paused")
    status["tenants"][0]["endpoint"]["avatar"] = {
        "hash": "0123456789abcdef",
        "source_sha256": "0" * 64,
    }
    seen: list[httpx.Request] = []

    def responder(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(status)

    await _discipline(_Counter(responder), tmp_path).check(Boundary.APP_START)

    assert [r.method for r in seen] == ["GET"]


async def test_a_duplicated_tenant_row_is_a_doubt_not_a_first_match(tmp_path: Path):
    """R1-S3: ambiguity resolves to NO write, on both addressing paths."""

    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    payload = _endpoint_payload("active")
    first = dict(payload["tenants"][0])
    first["endpoint"] = {"id": "ep_NOT_MINE", "label": "Org", "state": "active"}
    payload["tenants"] = [first, payload["tenants"][0]]
    client = FakeClient(payload)

    await avatar_publish.reconcile(
        client, SimpleNamespace(payload=payload, account_id="acc_1"), "tnt_1", tmp_path
    )

    assert client.puts == [] and client.deletes == []


async def test_the_plan_only_addresses_the_endpoint_the_guard_validated(tmp_path: Path):
    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(), files={"avatar.png": PNG})
    payload = _endpoint_payload("active")
    # The guard approved `bep_1`; a plan asked for anything else refuses.

    plan = avatar_publish.plan_avatar(
        SimpleNamespace(payload=payload, account_id="acc_1"), "tnt_1", tmp_path, "ep_OTHER"
    )

    assert plan.action == avatar_publish.NONE and plan.endpoint_id is None


async def test_a_bounded_multipart_publish_still_carries_its_body():
    """R1-S1: the measured trap — `files` + `max_response_bytes` sent no body."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"endpoint_id": "bep_1", "hash": "0123456789abcdef"})

    client = _make_client(handler)
    await client.put_endpoint_avatar("bep_1", PNG, PNG_SHA, expected_account_id="acc_1")

    assert len(seen) == 1
    assert b'name="file"' in seen[0].content and PNG in seen[0].content
    assert seen[0].headers["content-length"] == str(len(seen[0].content))


async def test_an_oversized_publish_answer_is_refused_while_it_streams():
    """R1-S1/F7, R2-F1: the bound is applied at DISPATCH, not after parsing.

    The answer must be a VALID house envelope that is merely too big, or the
    cell would pass for the wrong reason: a ``result`` that is not an object is
    refused by ``_envelope_result`` at any size, so it proves nothing about the
    bound (round-2 review measured exactly that). Here the envelope parses and
    only ``max_response_bytes`` can stop it — drop that argument and this cell
    fails with no refusal raised at all.
    """

    oversized = json.dumps({"msg": "ok", "result": {"blob": "x" * (70 * 1024)}}).encode()
    assert len(oversized) > 64 * 1024
    client = _make_client(lambda request: httpx.Response(200, content=oversized))

    with pytest.raises(BridgeRefusal) as info:
        await client.put_endpoint_avatar("bep_1", PNG, PNG_SHA, expected_account_id="acc_1")

    assert info.value.code == "bridge_invalid_response"
    # The same envelope under the cap is accepted: the bound, not the shape, is
    # what this cell is measuring.
    small = json.dumps({"msg": "ok", "result": {"blob": "x" * 32}}).encode()
    ok_client = _make_client(lambda request: httpx.Response(200, content=small))
    result = await ok_client.put_endpoint_avatar("bep_1", PNG, PNG_SHA, expected_account_id="acc_1")
    assert result == {"blob": "x" * 32}


async def test_an_oversized_clear_answer_carries_the_avatar_code():
    """R2-F2: the clear is bounded too, and with this route's own refusal class.

    Inheriting ``attachment_too_large`` named a class the caller cannot act on
    and pointed at the wrong feature.
    """

    oversized = json.dumps(
        {"error": "no", "code": "not_found", "details": {"x": "y" * (40 * 1024)}}
    )
    client = _make_client(lambda request: httpx.Response(404, content=oversized.encode()))

    with pytest.raises(BridgeRefusal) as info:
        await client.delete_endpoint_avatar("bep_1", expected_account_id="acc_1")

    assert info.value.code == "bridge_invalid_response"


async def test_the_explicit_paths_read_the_status_with_the_access_they_pin(tmp_path: Path):
    """R1-S4: one identity for the read and the write, or the pin proves nothing."""

    plan_payload = _endpoint_payload("active")
    plan_payload["tenants"][0]["endpoint"]["avatar"] = {
        "hash": "0123456789abcdef",
        "source_sha256": "0" * 64,
    }

    class Pinning(FakeClient):
        def __init__(self) -> None:
            super().__init__(plan_payload)
            self.status_access: list[Any] = []
            self.pinned: list[Any] = []

        async def status(self, *, access: Any = None) -> dict[str, Any]:
            self.status_access.append(access)
            return self.payload

        async def put_endpoint_avatar(
            self,
            endpoint_id: str,
            data: bytes,
            source_sha256: str,
            *,
            expected_account_id: Any = None,
        ) -> dict[str, Any]:
            self.pinned.append(expected_account_id)
            return {}

    client = Pinning()
    access = await client.current_access()
    root = tmp_path / "preset"
    _enrolled(root)
    _agent(root, block=_png_block(), files={"avatar.png": PNG})

    await avatar_publish.publish_agent_avatar(
        client, "tnt_1", root, access=access, status=plan_payload
    )

    assert client.status_access == [] and client.pinned == ["acc_1"]


async def test_the_plan_repr_never_carries_the_picture_bytes_or_a_path(tmp_path: Path):
    """R1-S5: a repr is one exception away from a log.

    The block MUST carry the matching digest: with a wrong one the store refuses
    the file, ``avatar_data`` and ``plan.data`` are ``None``, and the assertions
    below pass whatever the dataclass does (security review R3-S2). The two
    preconditions pin that this cell is looking at real bytes.
    """

    # Repr safety applies even to arbitrary bytes held by these dataclasses;
    # image validation is independently pinned at the resolver boundary.
    from local_operator.team_bridge.cos_name import PublishedAgent

    pixels = b"SECRETPIXELS"
    agent = PublishedAgent(local_agent_id="aida", avatar_data=pixels)
    plan = avatar_publish.AvatarPlan(data=pixels)
    assert agent.avatar_data == pixels and plan.data == pixels

    assert "SECRETPIXELS" not in repr(agent)
    assert str(tmp_path) not in repr(agent)
    assert "SECRETPIXELS" not in repr(plan)
    assert "SECRETPIXELS" not in str(plan) and "SECRETPIXELS" not in f"{agent!r}"


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("garbage", id="string"),
        pytest.param("[]", id="empty-list"),
        pytest.param("[1, 2]", id="list"),
        pytest.param("0", id="zero"),
        pytest.param("false", id="false"),
        pytest.param("''", id="empty-string"),
    ],
)
async def test_a_malformed_avatar_value_is_a_doubt_and_never_a_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str
):
    """R3-S4: present-but-not-a-mapping once read as "no block" and DELETEd the hub copy."""

    _preserves_avatar(monkeypatch)  # the store DOES keep a block: absence would be a clear
    _enrolled(tmp_path)
    _agent(tmp_path, block=f"avatar: {raw}\n", files={"avatar.png": PNG})

    agent = published_agent(tmp_path)
    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True

    # Through the real boundary, against a hub that DOES hold a picture.
    seen = await _boundary_requests(tmp_path)
    assert [r.method for r in seen if r.url.path.endswith("/avatar")] == []


async def test_an_account_and_an_agent_with_the_same_id_never_share_cached_bytes():
    """R1-F9: the two reads must key the cache apart, or one answers the other."""

    from local_operator.team_bridge import avatar_cache

    avatar_cache.AVATAR_CACHE._entries.clear()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        body = (
            b"\x89PNG\r\n\x1a\naccount"
            if "/accounts/" in request.url.path
            else b"\x89PNG\r\n\x1a\nagent"
        )
        return httpx.Response(
            200, content=body, headers={"content-type": "image/png", "etag": f'"{len(calls):016d}"'}
        )

    client = _avatar_client(handler)
    account_body, _ = await client.fetch_avatar(
        "acc_2", v="0123456789abcdef", size=64, expected_account_id="acc_1"
    )
    agent_body, _ = await client.fetch_endpoint_avatar(
        "bep_1", v="0123456789abcdef", size=64, expected_account_id="acc_1"
    )

    assert account_body != agent_body
    keys = list(avatar_cache.AVATAR_CACHE._entries)
    assert len(keys) == 2
    assert {key[2] for key in keys} == {"acc_2", "endpoint:bep_1"}


async def test_a_dot_slash_prefixed_name_is_a_doubt(tmp_path: Path):
    """R3-F1: the store compares ``filename`` EXACTLY, so this side does too.

    ``./avatar.png`` was normalised away by the private resolver this replaced;
    the store reads it as "no picture", and a publisher more lenient than the
    store is the shape of the disclosure bug.
    """

    _enrolled(tmp_path)
    _agent(tmp_path, block=_png_block(filename="./avatar.png"), files={"avatar.png": PNG})

    agent = published_agent(tmp_path)

    assert agent is not None and agent.avatar_data is None and agent.avatar_broken is True


def test_a_nul_in_the_stored_name_is_a_doubt(tmp_path: Path):
    """A NUL cannot ride a YAML scalar, so it is asserted at the reader itself."""

    from local_operator.team_bridge.cos_name import _stored_avatar_bytes

    agent_dir = tmp_path / "agents" / "aida"
    agent_dir.mkdir(parents=True)
    (agent_dir / "avatar.png").write_bytes(PNG)
    block = {
        "sha256": PNG_SHA,
        "mime": "image/png",
        "bytes": len(PNG),
        "updated_at": "2026-10-30T00:00:00+00:00",
        "filename": "avatar\x00.png",
    }

    assert _stored_avatar_bytes(agent_dir, block) is None
    assert _stored_avatar_bytes(agent_dir, {**block, "filename": "avatar.png"}) == PNG


@pytest.mark.parametrize(
    "kind",
    ["text", "png-tail", "jpeg-tail", "png-text", "jpeg-exif", "png", "jpeg", "mime-mismatch"],
)
async def test_only_normalized_pixels_can_leave_the_publisher(tmp_path: Path, kind: str):
    """Q1: matching metadata is not proof of an image; no ancillary bytes leave."""
    from io import BytesIO
    from PIL import Image, PngImagePlugin
    from local_operator import agent_avatar

    sentinel = b"PRIVATE_PROMPT_SENTINEL"
    image = Image.new("RGB", (512, 512), (20, 40, 80))
    buffer = BytesIO()
    jpeg = kind.startswith("jpeg")
    mime = "image/jpeg" if jpeg else "image/png"
    name = "avatar.jpg" if jpeg else "avatar.png"
    options: dict[str, Any] = {}
    if kind == "png-text":
        info = PngImagePlugin.PngInfo()
        info.add_text("prompt", sentinel.decode())
        options["pnginfo"] = info
    if kind == "jpeg-exif":
        exif = Image.Exif()
        exif[270] = sentinel.decode()
        options["exif"] = exif
    image.save(buffer, format="JPEG" if jpeg else "PNG", **options)
    data = buffer.getvalue()
    if kind.endswith("tail"):
        data += sentinel
    if kind == "text":
        data = sentinel
    if kind == "mime-mismatch":
        mime, name = "image/jpeg", "avatar.jpg"
    sha = hashlib.sha256(data).hexdigest()
    block = _png_block(filename=name, sha=sha, size=len(data)).replace("image/png", mime)
    _enrolled(tmp_path)
    _agent(tmp_path, block=block, files={name: data})
    client = FakeClient(_status(hub_sha="0" * 64).payload)
    await avatar_publish.reconcile(client, _status(hub_sha="0" * 64), "tnt_1", tmp_path)
    if kind in {"text", "mime-mismatch"}:
        assert client.puts == [] and client.deletes == []
        return
    expected = agent_avatar.normalise(data).data
    assert client.puts == [("bep_1", expected, hashlib.sha256(expected).hexdigest())]
    assert sentinel not in expected
    # The hub echoes the WIRE digest, not the pre-normalization file's digest.
    client.puts.clear()
    await avatar_publish.reconcile(
        client, _status(hub_sha=hashlib.sha256(expected).hexdigest()), "tnt_1", tmp_path
    )
    assert client.puts == [] and client.deletes == []


@pytest.mark.parametrize(
    "code", ["injected\nFAKE_LOG_ENTRY", "x" * 1000, "", "a\x1b[31m", None, 12]
)
async def test_malformed_hub_codes_are_generic_at_the_shared_parser(code: Any):
    from local_operator.team_bridge.client import _refusal

    refusal = _refusal(httpx.Response(403, json={"code": code, "error": "refused"}))
    assert refusal.code == "bridge_error"


async def test_normal_hub_codes_survive_the_shared_parser():
    from local_operator.team_bridge.client import _refusal

    for code in ["avatar_type_unsupported", "rate_limited", "endpoint_avatar_not_found"]:
        assert _refusal(httpx.Response(403, json={"code": code})).code == code
