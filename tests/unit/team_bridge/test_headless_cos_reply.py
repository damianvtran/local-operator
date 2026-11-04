"""A Team Bridge message reaches a chief of staff that NO UI has open, and it answers.

WHY THIS FILE EXISTS. The operator's directive: backend sessions must answer
bridge messages with no desktop app, no TUI, no ``lop serve`` and no live
session in sight. Each hop below is already covered in isolation elsewhere;
nothing proved the CHAIN holds when the designated chief-of-staff (CoS) session
has never run and nobody is watching. These tests pin that chain, each naming
the hop it owns:

1. ``deliver_wake`` (cold, no durable history) → the REAL ``engage_runtime`` →
   the REAL ``_spawn_runtime``: what gets started is the detached runtime
   module for the CoS session id, with no desktop/TUI module dragged in.
2. The UNSTARTED runtime's REAL ``RuntimeServer._dispatch("peer_message")``
   admits exactly the bridge card for the designated session under
   ``bridge_consent_v2`` — and refuses every near miss with the ordinary
   unengaged refusal (``_bridge_card_may_engage``, server.py).
3. The wake turn the admitted card starts can call the session's ``bridge``
   tool ``reply`` with only ``message_id`` + ``text``; the POST targets the
   received message's sender in its thread (PR #3023).

Hops 1-3 run as ONE chain in the first test, because the claim is about the
chain: a fake socket hands the frame ``engage_runtime`` delivers to the very
server under test. The second test pins the refusals around that admission.

SEAMS FAKED, all at a process/transport boundary and nothing above it:
- ``launch.spawn_disclaimed`` — the ``Popen`` itself. ``_spawn_runtime`` (env,
  argv, capture file, capability handoff) runs for real; only the fork is
  recorded. The narrowest seam: everything above it is product logic.
- ``attach_client.find_runtime_record`` — the on-disk discovery record a real
  child would publish once it booted. It answers only AFTER the recorded spawn,
  so delivery cannot happen without one.
- ``peer_client.asyncio.open_connection`` — the loopback socket, replayed into
  the REAL ``_dispatch`` (``test_h5_full_wire._wire``).
- the hub's HTTP transport (``httpx.MockTransport``) and the OAuth destination
  allowlist — never a network call.

``runtime.admission = "admitted"`` stands in for the hub status read that sets
it in production (network-dependent, not UI-dependent: tool.py refuses
``bridge_not_ready`` until then). Approval is ``auto`` here; see the note at the
gate for what happens under the default ``ask``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE
from local_operator.harness.types import (
    StreamEndEvent,
    StreamToolCallDelta,
    ToolContext,
)
from local_operator.session.runtime import launch
from local_operator.session.runtime import standby as standby_mod
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import (
    ENGAGED_ENV,
    RUNTIME_MODULE,
    RUNTIME_RECORD_KIND,
)
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.connector import wake as wake_mod
from local_operator.team_bridge.store import OwnerReceipt, Store
from local_operator.team_bridge.tool import BridgeRuntime, build_bridge_tool
from local_operator.team_bridge.types import MessageRef
from tests.unit.session.test_session import ScriptedStream, make_session, wait_for
from tests.unit.team_bridge.test_h5_full_wire import _tasks_pending, _wire
from tests.unit.team_bridge.test_reply_resolution import OWNER
from tests.unit.team_bridge.test_reply_resolution import _runtime as _reply_runtime

pytestmark = pytest.mark.asyncio

COS = "s1"
#: Modules that only a desktop, ``lop serve`` or TUI front end loads. The cold
#: path must not need any of them to start, deliver or answer.
_FRONT_END_MODULES = ("local_operator.server.app", "local_operator.tui.app", "textual.app")


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Home, config and cwd out of the way of the developer's real ones.

    Same shape as ``test_h5_full_wire.isolated_config``; the engagement check
    and the authority fence read ``config_dir()`` from this variable.
    """

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    return tmp_path


def _enroll(
    root: Path,
    *,
    consent: str = "bridge_consent_v2",
    designated: str = COS,
    enabled: bool = True,
) -> Store:
    """A committed owner, its designation, and ONE received message ``bmsg_in1``.

    Written through the store's own API — the same calls the settings commit and
    the connector make — not by SQL, so the engagement check reads production
    rows. Nothing here goes through a desktop route.
    """

    (root / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": enabled}}}), encoding="utf-8"
    )
    store = Store(root)
    store.commit_enrollment(
        OWNER, OwnerReceipt(consent, "consent-ref-1"), cos_session_id=designated
    )
    ref = MessageRef(
        message_id="bmsg_in1",
        thread_id="bth_1",
        tenant_id=OWNER.tenant_id,
        message_type="note",
        body_hash=None,
        seq=3,
        origin="agent",
        delivery_state="unread",
        state_at=None,
        sender_account_id="acc_2",
        sender_endpoint_id="bep_2",
        recipient_account_id=OWNER.account_id,
        recipient_endpoint_id=OWNER.endpoint_id,
    )
    store.merge_page(OWNER, [ref], None)
    return store


def _bridge_frame(**overrides: Any) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "op": "peer_message",
        "req": 1,
        "text": "[team_bridge_message] message_id: bmsg_in1",
        "mode": "mailbox",
        "wake": True,
        "sender": {"account_id": OWNER.account_id, "source": TEAM_BRIDGE_MESSAGE_TYPE},
        "custom_type": TEAM_BRIDGE_MESSAGE_TYPE,
        "bridge_meta": {"v": "1", "message_id": "bmsg_in1", "thread_id": "bth_1"},
    }
    frame.update(overrides)
    return frame


def _reply_turns() -> ScriptedStream:
    """The CoS model's wake turn: one ``bridge reply`` naming only the message."""

    args = json.dumps({"op": "reply", "message_id": "bmsg_in1", "text": "on it"})
    return ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="bridge", argument_delta=args),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
            [StreamEndEvent(stop_reason="stop")],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )


async def _unstarted_cos(
    root: Path, stream: ScriptedStream, requests: list[httpx.Request], *, session_id: str = COS
) -> tuple[Any, RuntimeServer]:
    """The cold runtime's in-process half: Session + serving handle + server.

    Built the way ``process.amain`` builds a spawned runtime (``kind`` is the
    detached runtime's own), with NO ``set_record_started(True)``: the server
    stays unengaged, so only the CoS engagement exception can admit a frame.
    The ``bridge`` tool comes from the production createIf (``build_bridge_tool``)
    over the same runtime the session holds, as ``session_factory`` wires it.
    """

    runtime: BridgeRuntime = _reply_runtime(root, requests)
    runtime.session_id = session_id
    # In production the hub status read sets this (``admit_soon``); it is the
    # network gate, not a UI one, and it is out of this file's scope.
    runtime.admission = "admitted"
    tool = build_bridge_tool(ToolContext(bridge_runtime=runtime))
    assert tool is not None
    session = make_session(
        root,
        stream,
        tools=[tool],
        session_id=session_id,
        bridge_runtime=runtime,
    )
    # THE APPROVAL GATE. ``reply`` is tier "write" (tool.py ``approval_tier``),
    # and the serving handle installs the gate a spawned runtime decides on
    # (serving.py ``_install_gates``), replacing any gate the Session was built
    # with. ``spawn_owned_session`` passes ``auto_approve`` from the saved
    # ``tool_approval_mode``; ``auto`` is mirrored here. Under the DEFAULT
    # ``ask`` this call parks on an approval card that no UI is present to
    # answer, until ``runtime.unattended_gate_timeout`` (24 h) denies it. That
    # is the reported gap and is deliberately NOT tested here.
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(root), auto_approve=True
    )
    server = RuntimeServer(handle, kind=RUNTIME_RECORD_KIND)
    assert server._started is False, "the fixture must leave the CoS runtime unengaged"
    return session, server


async def test_cold_delivery_engages_the_designated_cos_headlessly_and_it_replies(
    tmp_path: Path, isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """deliver_wake → spawn → admitted on an UNSTARTED server → wake turn → reply POST."""

    _enroll(tmp_path)
    requests: list[httpx.Request] = []
    session, server = await _unstarted_cos(tmp_path, _reply_turns(), requests)
    _wire(server, monkeypatch)

    # Keep the spawn deterministic on any host: no warmed standby to adopt and
    # no memory-floor refusal. Both are switches the product itself reads.
    monkeypatch.setenv(standby_mod.DISABLE_ENV, "1")
    monkeypatch.setenv("LOCAL_OPERATOR_ADMISSION_FLOOR_MB", "0")

    spawns: list[dict[str, Any]] = []

    class _Child:
        """A started child that never exits; no ``pid``, so nothing reaps it."""

        returncode = None

        def poll(self) -> None:
            return None

    def _fake_spawn(argv: list[str], **kwargs: Any) -> _Child:
        spawns.append({"argv": list(argv), "env": dict(kwargs.get("env") or {})})
        return _Child()

    monkeypatch.setattr(launch, "spawn_disclaimed", _fake_spawn)

    # The discovery record the booted child would publish; only after a spawn.
    record = SimpleNamespace(control_port=0, control_key="k", session_id=COS, leaving="")

    def _find(_config_dir: Path, session_id: str, *, check_zombie: bool = True):
        assert session_id == COS
        return (record, 4242) if spawns else (None, None)

    monkeypatch.setattr("local_operator.mobile.attach_client.find_runtime_record", _find)

    before = {name for name in _FRONT_END_MODULES if name in sys.modules}
    detail = await wake_mod.deliver_wake(
        SimpleNamespace(account_id=OWNER.account_id),
        "[team_bridge_message] message_id: bmsg_in1",
        session_id=COS,
        config_dir=tmp_path,
        bridge_meta={"v": "1", "message_id": "bmsg_in1", "thread_id": "bth_1"},
    )
    assert detail.startswith(f"engaged {COS}"), detail

    # HOP 1 — what was started: the detached runtime module for the CoS id.
    assert len(spawns) == 1, spawns
    argv, env = spawns[0]["argv"], spawns[0]["env"]
    assert argv[argv.index("-m") + 1] == RUNTIME_MODULE, argv
    assert env["LOP_MOBILE_CHILD_RESUME"] == COS
    assert ENGAGED_ENV not in env, "a carried message is not a warm engage"
    # Only modules THIS path imported: one a previous test loaded proves nothing.
    newly = {n for n in _FRONT_END_MODULES if n in sys.modules} - before
    assert newly == set(), f"the headless path imported front-end modules: {newly}"

    # HOP 2 — the unstarted server admitted it (else deliver_wake raised above)
    # and the row is the bridge card, not a generic peer note.
    await wait_for(lambda: len(requests) >= 1 and not _tasks_pending(session), 10.0)
    cards = [
        m for m in session.history() if getattr(m, "custom_type", None) == TEAM_BRIDGE_MESSAGE_TYPE
    ]
    assert len(cards) == 1, [getattr(m, "custom_type", None) for m in session.history()]

    # HOP 3 — the reply reaches the original sender in the received thread.
    assert [r.url.path for r in requests] == ["/v1/messaging/messages"]
    body = json.loads(requests[0].content)
    assert body["to"] == {"endpoint_id": "bep_2"}, body
    assert body["thread_id"] == "bth_1", body
    await session.dispose()


@pytest.mark.parametrize(
    ("label", "enroll", "server_session", "frame_overrides"),
    [
        ("not the designated session", {}, "s9", {}),
        ("consent v1 never disclosed engagement", {"consent": "bridge_consent_v1"}, COS, {}),
        ("bridge turned off", {"enabled": False}, COS, {}),
        ("not a bridge card", {}, COS, {"custom_type": None}),
    ],
)
async def test_unstarted_cos_runtime_refuses_every_near_miss(
    tmp_path: Path,
    isolated_config: Path,
    label: str,
    enroll: dict[str, Any],
    server_session: str,
    frame_overrides: dict[str, Any],
) -> None:
    """The engagement exception is exactly one door; each near miss stays refused."""

    _enroll(tmp_path, **enroll)
    requests: list[httpx.Request] = []
    session, server = await _unstarted_cos(
        tmp_path, _reply_turns(), requests, session_id=server_session
    )
    frame = {k: v for k, v in _bridge_frame(**frame_overrides).items() if v is not None}
    with pytest.raises(ValueError, match="has not been engaged yet") as refused:
        await server._dispatch("peer_message", frame)
    assert refused.value is not None, label
    assert requests == [], f"{label}: a refused card must not start a turn"
    assert not any(
        getattr(m, "custom_type", None) == TEAM_BRIDGE_MESSAGE_TYPE for m in session.history()
    ), label
    await session.dispose()
