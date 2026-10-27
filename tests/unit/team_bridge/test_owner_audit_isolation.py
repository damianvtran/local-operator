"""The owner-audit body path is HUMAN-ONLY: no agent-context path can reach it.

The owner audit (agent-server #221, docs/TEAM-BRIDGE-CONVERSATIONS.md §16)
serves FULL bodies including messages the agent's screening rejected or never
screened. That is correct for the human owner and catastrophic for the agent:
an unscreened body in a model's context is exactly the injection screening
exists to stop. The contract (v2 M4) therefore requires a test proving the
`team_bridge` tool, read_context, model_boundary and every other agent-context
path cannot reach the audit client methods. Two halves:

1. STATIC. Every module under ``local_operator`` is parsed; the audit method
   names, the hub's ``agents/owned`` path and the audit contract module may
   appear ONLY in the allowlisted human-only modules (the client that defines
   them and the desktop route family that serves the renderer). A new caller —
   the tool growing an op, a context builder "helpfully" showing held bodies —
   fails here and has to argue its way into the allowlist in review.
2. BEHAVIOURAL. Every op the bridge tool exposes is executed against a REAL
   ``TeamBridgeClient`` whose transport records each request and whose audit
   methods trip a wire. No op reaches an audit method or an audit URL, while a
   positive control proves the ops really did talk to the hub (so the absence
   is not vacuous), and the agent's history read still names its endpoint (the
   admitted-only predicate is unchanged).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import local_operator
from local_operator.harness.types import ToolContext
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerSnapshot
from local_operator.team_bridge.tool import (
    BRIDGE_OPS,
    BridgeRuntime,
    BridgeToolParams,
    build_bridge_tool,
    execute_bridge,
)
from tests.unit.team_bridge.test_client import FakeStore

PACKAGE = Path(local_operator.__file__).parent

#: The ONLY modules that may name the audit surface. Both are human-only: the
#: client defines the methods, the desktop routes hand bytes to the renderer.
HUMAN_ONLY_MODULES = {
    "team_bridge/client.py",
    "server/routes/desktop_team_bridge.py",
    "server/models/desktop_team_bridge_owner_audit.py",
}
AUDIT_METHODS = (
    "owner_audit_conversations",
    "owner_audit_messages",
    "owner_audit_replies",
    "owner_audit_attachment",
)
#: ``owner_audit_`` (with the underscore) is the client methods' and route
#: helpers' prefix; the bare capability key ``team_bridge_owner_audit`` in
#: ``server/features.py`` is metadata the renderer reads, not a reach, and does
#: not match it.
#: ``owner-audit`` / ``owner_audit/`` catch the DESKTOP route path spelled in
#: code (an agent calling its own daemon's route), and ``owned/conversations``
#: a hub path concatenated from halves (#3022 R1-1). A path built from smaller
#: pieces still is what the client-side gate below exists for.
AUDIT_MARKERS = (
    "owner_audit_",
    "owner-audit",
    "agents/owned",
    "owned/conversations",
    "desktop_team_bridge_owner_audit",
)


def _references(tree: ast.AST) -> list[str]:
    """Every identifier, attribute, import name and string literal in ``tree``."""

    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            found.append(node.attr)
        elif isinstance(node, ast.Name):
            found.append(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append(node.value)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                found.append(node.module)
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(node.name)
    return found


def test_the_audit_methods_exist_on_the_client() -> None:
    """The negative control for the scan: the names it looks for are real."""

    for name in AUDIT_METHODS:
        assert callable(getattr(TeamBridgeClient, name))


def test_no_module_outside_the_human_only_routes_names_the_audit_surface() -> None:
    offenders: dict[str, set[str]] = {}
    scanned = 0
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).as_posix()
        source = path.read_text(encoding="utf-8")
        scanned += 1
        # Cheap prefilter: a file that never spells a marker cannot reach the
        # audit by name, so only the few that do pay for a parse.
        if not any(marker in source for marker in AUDIT_MARKERS):
            continue
        tree = ast.parse(source, filename=str(path))
        # Comments and docstrings may DESCRIBE the rule (app.py's cache note
        # does); only code is a reach, so docstrings are dropped from the hits.
        doc_text = {
            node.body[0].value.value
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        hits = {
            ref
            for ref in _references(tree)
            if any(marker in ref for marker in AUDIT_MARKERS) and ref not in doc_text
        }
        if hits and relative not in HUMAN_ONLY_MODULES:
            offenders[relative] = hits
    assert scanned > 500, "the scan did not walk the package"
    assert (
        offenders == {}
    ), f"owner-audit surface reachable outside the human-only routes: {offenders}"


def test_the_human_only_modules_are_the_ones_that_name_it() -> None:
    """Positive control: the scan really finds the routes' use of the methods."""

    routes = (PACKAGE / "server/routes/desktop_team_bridge.py").read_text(encoding="utf-8")
    refs = _references(ast.parse(routes))
    for name in AUDIT_METHODS:
        assert name in refs


def test_the_tool_schema_offers_no_audit_operation() -> None:
    schema_ops = set(BridgeToolParams.model_json_schema()["properties"]["op"]["enum"])
    assert schema_ops == set(BRIDGE_OPS)
    blob = json.dumps(BridgeToolParams.model_json_schema())
    for marker in ("audit", "owned", "owner_view"):
        assert marker not in blob


TENANT = "ten_1"
CID = "bcv_abcd1234"
ROOT = "bcmsg_root0001"

#: Minimal args that drive each tool op to its client call. Every BRIDGE_OP
#: must be here: a new op without an entry fails ``test_every_op_is_driven``.
OP_ARGS: dict[str, dict[str, Any]] = {
    "take_up": {"message_id": "bmsg_1", "body_sha256": "0" * 64},
    "ack": {"message_id": "bmsg_1", "state": "done"},
    "send": {"to": {"account_id": "acc_b"}, "text": "hi"},
    "reply": {"thread_id": "bthr_1", "to": {"account_id": "acc_b"}, "text": "hi"},
    "channels": {"limit": 5},
    "channel_create": {"kind": "channel", "name": "audit-probe", "visibility": "tenant"},
    "channel_post": {"channel_id": CID, "body": "hello"},
    "channel_history": {"channel_id": CID, "limit": 5},
    "react": {"channel_id": CID, "message_id": ROOT, "emoji": "👍"},
    "channel_thread": {"channel_id": CID, "thread_root_id": ROOT},
}


class TrippedClient(TeamBridgeClient):
    """The real client, with every audit method a tripwire."""

    tripped: list[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tripped = []

    def __getattribute__(self, name: str) -> Any:
        if name in AUDIT_METHODS:
            object.__getattribute__(self, "tripped").append(name)
        return object.__getattribute__(self, name)


def _hub_answer(request: httpx.Request) -> httpx.Response:
    """Plausible hub answers so each op runs to completion where it can."""

    path = request.url.path
    if path.endswith("/replies"):
        result: Any = {"root": {"id": ROOT, "seq": 1}, "replies": [], "has_more": False}
    elif path.endswith("/messages") and request.method == "GET":
        result = {"messages": [], "has_more": False}
    elif path.endswith("/conversations") and request.method == "GET":
        result = {"conversations": [], "has_more": False}
    elif path.endswith("/conversations"):
        result = {"id": CID, "kind": "channel", "member_count": 1}
    else:
        result = {"message": {"id": "bcmsg_new0001", "seq": 2}}
    return httpx.Response(200, json={"msg": "ok", "result": result})


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the fences so every op actually reaches its wire call.

    The fences have their own tests; closed, they would make this test pass
    vacuously by refusing every op before it touched the client.
    """

    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def test_every_op_is_driven() -> None:
    assert set(OP_ARGS) == set(BRIDGE_OPS)


@pytest.mark.asyncio
async def test_no_bridge_tool_op_reaches_the_owner_audit(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _hub_answer(request)

    client = TrippedClient(
        tmp_path,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )
    runtime = BridgeRuntime(
        client=client,
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=OwnerSnapshot(
            credential_id=1,
            account_id="acc_1",
            tenant_id=TENANT,
            endpoint_id="bep_1",
            canonical_base_url="https://hub.example",
            enrollment_generation=1,
        ),
        session_id="sess_1",
    )
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None

    for op, args in OP_ARGS.items():
        try:
            await execute_bridge(runtime, {"op": op, **args})
        except Exception:  # noqa: BLE001 — an op failing locally is not a reach
            pass

    paths = [request.url.path for request in requests]
    assert client.tripped == [], f"a tool op reached {client.tripped}"
    assert not any("/agents/owned" in path for path in paths), paths
    # Positive control: the ops DID talk to the hub through this client.
    assert f"/v1/messaging/conversations/{CID}/messages" in paths
    assert f"/v1/messaging/conversations/{CID}/messages/{ROOT}/replies" in paths
    # The agent's own read is unchanged: it names its endpoint, so the hub
    # applies the admitted-only predicate (never the human/audit one).
    history = [
        r
        for r in requests
        if r.method == "GET" and r.url.path == f"/v1/messaging/conversations/{CID}/messages"
    ]
    assert history and all(r.url.params.get("endpoint_id") == "bep_1" for r in history)


def test_the_tripwire_fires_on_an_audit_method(tmp_path: Path) -> None:
    """Negative control for the behavioural half: the wire really detects a reach."""

    client = TrippedClient(tmp_path, base_url="https://hub.example/v1", store=FakeStore())
    getattr(client, "owner_audit_messages")
    assert client.tripped == ["owner_audit_messages"]


# -- runtime gate (#3022 R1-1) ------------------------------------------------
#
# The static scan sees names and literals; it cannot see
# ``getattr(client, "owner" + "_audit_messages")`` or
# ``_send("GET", "/messaging/agents/" + "owned/conversations")``. The client
# therefore refuses ANY hub request on the owner-audit family that is not made
# from inside one of its four methods, at the one dispatch every request uses.


@pytest.mark.parametrize(
    "path",
    [
        "/messaging/agents/" + "owned/conversations",
        "/messaging/agents/owned/conversations/bcv_abcd1234/messages",
        "//messaging//Agents/Owned/conversations",
        "/messaging/agents/%6Fwned/conversations",
        "/messaging/agents/%256Fwned/conversations",
    ],
)
@pytest.mark.asyncio
async def test_a_built_audit_path_is_refused_before_any_request(tmp_path: Path, path: str) -> None:
    requests: list[httpx.Request] = []
    client = TeamBridgeClient(
        tmp_path,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(lambda r: (requests.append(r), _hub_answer(r))[1]),
    )
    with pytest.raises(tb_client.BridgeRefusal) as refused:
        await client._send("GET", path)
    assert refused.value.code == "owner_audit_not_in_scope"
    assert requests == []


@pytest.mark.asyncio
async def test_the_audit_methods_themselves_still_dispatch(tmp_path: Path) -> None:
    """Positive control: the gate admits the four methods (and only by scope)."""

    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"msg": "ok", "result": {"messages": [], "has_more": False}}
        )

    client = TeamBridgeClient(
        tmp_path,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(answer),
    )
    dynamic = getattr(client, "owner" + "_audit" + "_messages")
    await dynamic(CID, expected_account_id="acc_1")
    assert [r.url.path for r in requests] == [
        f"/v1/messaging/agents/owned/conversations/{CID}/messages"
    ]
    # The scope does not leak past the call: the same client, a raw path, refused.
    with pytest.raises(tb_client.BridgeRefusal):
        await client._send("GET", f"/messaging/agents/owned/conversations/{CID}/messages")
