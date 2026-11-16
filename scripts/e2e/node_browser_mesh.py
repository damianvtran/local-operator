"""Harness-only mesh legs for node_browser_harness; never imported by product code.

Both relays bind loopback inside the caller's private network namespace. Their
stores are separate, but they share the existing fixture's actual invite/SAS
ceremony (only the two synthetic humans are supplied). No catalogue, remote
viewer, decision handler or operator-signature verification is replaced.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def require_operator_member(record: Any, device_id: str) -> list[str]:
    """Fail preflight if the synthetic operator cannot reach the signed door.

    A drive member can forward ask_respond, but operator_challenge costs approve
    at the receiving relay. Testing only the former silently omits a prerequisite
    of the HTTP approval path; derive the required caps from its production table.
    """
    from local_operator.network.types import INNER_OP_CAPABILITY

    member = record.member(device_id)
    if member is None or not member.active:
        raise RuntimeError("synthetic operator is not an active mesh member")
    required = {
        INNER_OP_CAPABILITY[op] for op in ("operator_challenge", "bridge_decide", "ask_respond")
    }
    missing = required - set(member.capabilities)
    if missing:
        raise RuntimeError(f"synthetic operator lacks required capabilities: {sorted(missing)}")
    return sorted(member.capabilities)


class MeshRig:
    """Own exactly two relays and one operator inbox for the enclosing run."""

    def __init__(self, node_root: Path, operator_root: Path):
        self.node_root = node_root
        self.operator_root = operator_root
        self.servers: list[Any] = []
        self.link: Any = None
        self.discovery_patch: Any = None

    def start(self) -> dict[str, Any]:
        import pytest

        from local_operator.network import (
            approvals,
            identity,
            operator_binding,
            relay,
            store,
        )
        from local_operator.network.credentials import github
        from local_operator.network.onboard_approvals import request_device
        from local_operator.notifications import remote_attention
        from tests.unit.network.test_approvals_store import _sign_decision
        from tests.unit.network.test_relay_e2e import _pair

        # Native gh discovery is not part of this proof and must not reach a
        # user's credential helper. Same OS-discovery boundary as mesh_process_node.
        self.discovery_patch = pytest.MonkeyPatch()
        self.discovery_patch.setattr(github, "_resolve_program", lambda name, path=None: None)
        with pytest.MonkeyPatch.context() as patch:
            for root, name in (
                (self.node_root, "harness-node"),
                (self.operator_root, "harness-operator"),
            ):
                patch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
                server = relay.RelayServer(
                    settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
                    identity=identity.load_or_mint(root, name=name),
                )
                assert server.root == root
                self.servers.append(server)
            node, operator = self.servers
            host, port = node.bind()
            node.bind_control()
            node.start()
            patch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(self.operator_root))
            _, operator_port = operator.bind()
            operator.bind_control()
            operator.start()
            # The synthetic operator must hold approve as well as prompt. The
            # shared _live_link helper defaults its joiner to drive, which
            # correctly cannot carry an operator_challenge across the relay.
            record, _, _ = _pair(
                (node, operator, host, port),
                patch,
                role="admin",
                settings=relay.NetworkSettings(port=operator_port, listen_address="127.0.0.1"),
            )
            member_capabilities = require_operator_member(
                store.load(record.network_id, node.root), operator.identity.device_id
            )
            self.link, reason = operator.dial(
                record.network_id, host=f"{host}:{port}", epoch=record.epoch
            )
            assert self.link is not None, f"operator member handshake failed: {reason}"
            card = request_device(
                host="synthetic-node.invalid",
                user="harness",
                host_key_fp="SHA256:fixture",
                network=record.network_id,
                credential_ref="synthetic-unused-ssh",
                root=self.operator_root,
            )
            at = time.time()
            approvals.approve(
                card["approval_id"],
                signature_hex=_sign_decision(self.operator_root, card, "approve", at),
                decided_at=at,
                root=self.operator_root,
            )
            assert operator_binding.record_onboarded(
                self.operator_root,
                approval_id=card["approval_id"],
                network_id=record.network_id,
                device_id=node.identity.device_id,
            )
            # Present over the authenticated peer link, not accept_presented()
            # directly: the node must verify the actual transported signed bytes.
            binding_reply = operator_binding.present(operator, self.link)
            assert binding_reply is not None, "operator binding was not presented"
            assert binding_reply.get("detail", {}).get("accepted"), binding_reply
            self.network_id = record.network_id
            self.node_id = node.identity.device_id
            self.operator_id = operator.identity.device_id
            member = store.load(record.network_id, node.root).member(self.operator_id)
            assert member is not None and member.active
        remote_attention.origin_baseline_ms(self.operator_root, self.node_id)
        return {
            "network_id": self.network_id,
            "node_device": self.node_id,
            "operator_device": self.operator_id,
            "relay_ports": [server.settings.port for server in self.servers],
            "signed_binding": operator_binding.operator_device(self.node_root),
            "operator_member_capabilities": member_capabilities,
        }

    def observe(self) -> list[dict[str, Any]]:
        from local_operator.network.projection import RelayPeerCatalog
        from local_operator.notifications.remote_attention import RemoteAttentionWatcher
        from local_operator.notifications.store import NotificationStore
        from local_operator.session.peer_rows import peer_session_rows

        rows = peer_session_rows(
            self.operator_root, ttl_s=0, catalog=RelayPeerCatalog(self.operator_root)
        )
        RemoteAttentionWatcher(self.operator_root).observe(rows)
        items, _ = NotificationStore(self.operator_root).list_items(state="all", limit=100)
        return items

    def notice(self, sid: str, ask_id: str) -> dict[str, Any]:
        from local_operator.notifications.remote_attention import (
            ASK_NOTICE_PREFIX,
            notice_key,
        )

        expected = notice_key(self.node_id, sid, f"{ASK_NOTICE_PREFIX}{ask_id}")
        items = self.observe()
        matched = [item for item in items if item["key"] == expected and item["session_id"] == sid]
        return {"expected_key": expected, "count": len(matched), "items": matched}

    def close(self) -> None:
        if self.link is not None:
            self.link.close("harness complete")
        try:
            for server in reversed(self.servers):
                server.stop()
        finally:
            if self.discovery_patch is not None:
                self.discovery_patch.undo()


async def answer_over_http(
    operator_root: Path,
    sid: str,
    ask: dict[str, Any],
    word: str,
    *,
    plain: bool,
    backstop: float,
) -> dict[str, Any]:
    """Real TCP HTTP -> desktop router -> real mesh-backed AttachedSession.

    The config root used by the pool is always the operator's relay root; the
    caller's isolated ambient root supplies the signing key. Thus the wrong-key
    control reaches the same remote surface without replacing any signing code.
    """
    import asyncio
    import os
    import socket
    import uuid

    import httpx
    import uvicorn
    from fastapi import FastAPI

    from local_operator.server.routes import desktop_sessions
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    answers = {str(ask["questions"][0].get("id") or ""): [word]}
    pool = DesktopSessions(operator_root)
    try:
        if plain:
            # /answers deliberately upgrades a signed card to bridge_decide.
            # Test the *unsigned* remote facade door separately so it cannot
            # accidentally borrow that upgrade and turn the negative green.
            async with pool.session(sid) as bridge:
                assert bridge.remote_row is not None
                assert bridge.remote is not None
                try:
                    detail = await bridge.remote.ask_respond(ask["ask_id"], answers)
                except Exception as error:  # noqa: BLE001 — refusal is the evidence
                    return {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                        "surface": "remote-facade",
                    }
                return {"ok": True, "detail": str(detail), "surface": "remote-facade"}
        token = uuid.uuid4().hex
        os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = token
        app = FastAPI()
        app.state.desktop_sessions = pool
        app.state.config_manager = type("HarnessConfig", (), {"config_dir": operator_root})()
        app.include_router(desktop_sessions.router)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
            task = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                async with asyncio.timeout(backstop):
                    while not server.started:
                        if task.done():
                            await task
                            raise RuntimeError("desktop HTTP server exited before listening")
                        await asyncio.sleep(0)
                path = f"/v1/desktop/sessions/{sid}/answers"
                body = {"ask_id": ask["ask_id"], "answers": answers}
                async with httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{port}", timeout=backstop
                ) as client:
                    response = await client.post(
                        path, json=body, headers={"Authorization": f"Bearer {token}"}
                    )
                payload = response.json()
                bridge = pool.bridges.get(sid)
                remote_row = getattr(bridge, "remote_row", None)
                assert remote_row is not None, "HTTP did not resolve the remote-session surface"
                return {
                    "ok": response.status_code == 200,
                    "error": "" if response.status_code == 200 else str(payload),
                    "http_status": response.status_code,
                    "http_path": path,
                    "request": body,
                    "response": payload,
                    "surface": "desktop-http-remote-session",
                    "remote_owner": remote_row.owner_device,
                }
            finally:
                server.should_exit = True
                async with asyncio.timeout(backstop):
                    await task
    finally:
        await pool.close()
