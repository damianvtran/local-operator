"""Peer stored ordering crosses real relays; no viewer disk or runtime authority."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import projection, relay, store
from local_operator.session import transcript_index as ti
from local_operator.wakes import store as wake_store
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixture registration
    devices,
)
from tests.unit.network.test_session_plane import (
    SESSION,
    Devices,
    _call,
    _dial_to,
    _listen,
    _pair,
    _seed_journal,
)


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    return request.getfixturevalue("devices")


def _ready(root: Path) -> None:
    async def build() -> None:
        task, _ = ti.start_refresh(root, SESSION)
        await task

    asyncio.run(build())
    assert ti.fresh_resident(root, SESSION) is not None


def _read(server: relay.RelayServer, **fields: Any) -> dict[str, Any]:
    return server._op_session_history(
        relay.PeerLink.__new__(relay.PeerLink),
        {"session_id": SESSION, "limit": 2, "req": 1, **fields},
    )


@pytest.mark.parametrize("open_frame", [False, True])
def test_real_pair_ready_legacy_and_old_owner(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, open_frame: bool
) -> None:
    a, b, _host, _port = peer_pair
    ids = _seed_journal(b.root, SESSION, ["", "", ""])
    _ready(b.root)
    wake_store.write_entry(
        b.root,
        SESSION,
        cwd="",
        schedules=[{"id": "no-fire", "state": "scheduled"}],
        preserve={"stopped_at": 1234},
    )
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    seen: list[dict[str, Any]] = []
    original = b._handlers["net_session_history"]

    def observe(owner_link: relay.PeerLink, frame: dict[str, Any]) -> Any:
        seen.append(dict(frame))
        return original(owner_link, frame)

    monkeypatch.setitem(b._handlers, "net_session_history", observe)
    try:
        legacy = projection.peer_stored_history_page(
            a.root,
            device_id=b.identity.device_id,
            session_id=SESSION,
            limit=2,
            open_frame=open_frame,
        )
        assert legacy is not None
        assert "journal_order" not in seen[-1] and "order_ids" not in seen[-1]
        # JOURNAL_ORDER ALONE IS NO LONGER AN ACTIVATION: the complete
        # declaration gates the whole peer stored-read ordering path (the
        # transport slice's activation is re-gated), so a capable owner's
        # ready facts must NOT surface to a journal_order-only request — the
        # answer is byte-identically today's legacy ``unsupported``.
        partial = projection.peer_stored_history_page(
            a.root,
            device_id=b.identity.device_id,
            session_id=SESSION,
            limit=2,
            open_frame=open_frame,
            journal_order=True,
            order_ids=[ids[0], "absent"],
        )
        assert partial is not None
        assert seen[-1]["journal_order"] is True
        assert seen[-1]["order_ids"] == [ids[0], "absent"]
        assert "peer_journal_source" not in seen[-1]
        assert partial["journal_order_state"] == "unsupported"
        assert "journal_generation" not in partial
        assert json.dumps({k: v for k, v in partial.items() if k != "journal_order_state"}) == (
            json.dumps({k: v for k, v in legacy.items() if k != "journal_order_state"})
        )
        # THE COMPLETE DECLARATION IS THE ACTIVATION.
        page = projection.peer_stored_history_page(
            a.root,
            device_id=b.identity.device_id,
            session_id=SESSION,
            limit=2,
            open_frame=open_frame,
            journal_order=True,
            order_ids=[ids[0], "absent"],
            peer_journal_source=True,
        )
        assert page is not None
        assert seen[-1]["peer_journal_source"] is True
        assert page["journal_order_state"] == "ready"
        assert page["peer_journal_source"] is True
        index = ti.fresh_resident(b.root, SESSION)
        assert index is not None
        assert page["journal_generation"] == index.journal_generation
        assert page["journal_ordinals"] == {ids[0]: index.journal_rows[ids[0]]}
        assert page["journal_missing_ids"] == ["absent"]
        assert all(row["journal_seq"] == index.journal_rows[row["id"]] for row in page["entries"])
        # No flag means every byte remains the same, even after an ordered read.
        repeat = projection.peer_stored_history_page(
            a.root,
            device_id=b.identity.device_id,
            session_id=SESSION,
            limit=2,
            open_frame=open_frame,
        )
        assert json.dumps(repeat) == json.dumps(legacy)

        def old_owner(owner_link: relay.PeerLink, frame: dict[str, Any]) -> Any:
            # An OLD owner ignores the whole new vocabulary: it answers the
            # request it has always answered. Its reply carries no ACK, which
            # is exactly why a complete viewer must read absence as
            # ``unsupported`` — never infer completeness from any other key.
            return original(
                owner_link,
                {
                    k: v
                    for k, v in frame.items()
                    if k
                    not in {
                        "journal_order",
                        "order_ids",
                        "peer_journal_source",
                        "source_read",
                        "expected_journal_generation",
                    }
                },
            )

        monkeypatch.setitem(b._handlers, "net_session_history", old_owner)
        old = projection.peer_stored_history_page(
            a.root,
            device_id=b.identity.device_id,
            session_id=SESSION,
            limit=2,
            open_frame=open_frame,
            journal_order=True,
            order_ids=ids,
            peer_journal_source=True,
        )
        assert old == {**legacy, "journal_order_state": "unsupported"}
        # Stored reads take no runtime lease: no registry/runtime artifact appears.
        assert not list(b.root.glob("sessions/*/runtime*"))
        stopped = wake_store.read_entry(b.root, SESSION)
        assert stopped is not None and stopped["stopped_at"] == 1234
        states = [row["state"] for row in b.local_session_rows() if row["session_id"] == SESSION]
        assert states == ["stored"]
        # Exercise the desktop's real public page method and DTO, not merely
        # a projection helper: this is the page the installed UI will consume.
        from local_operator.resume import SessionRow
        from local_operator.server.models.desktop_sessions import HistoryPage
        from local_operator.server.utils.desktop_sessions import DesktopSessionBridge

        row = SessionRow(
            id=SESSION,
            mtime=0,
            name="",
            live_state="idle",
            locality="remote",
            owner_device=b.identity.device_id,
            owner_device_name="device-b",
            reachable=True,
            unreachable_reason="",
        )
        bridge = DesktopSessionBridge(a.root, SESSION, "", remote_row=row)
        monkeypatch.setitem(b._handlers, "net_session_history", observe)
        # LEGACY: journal_order alone keeps the wire page's own unsupported —
        # the transport slice's activation is re-gated behind the complete
        # declaration, so the installed UI's bytes are untouched.
        legacy_desktop = asyncio.run(
            bridge.history(journal_order=True, order_ids=ids, limit=2, open_frame=open_frame)
        )
        assert legacy_desktop.get("journal_order_state") == "unsupported"
        assert "journal_generation" not in legacy_desktop
        # COMPLETE: the stored owner's ready facts ride the page, validated
        # through the real DTO the installed UI consumes.
        desktop = asyncio.run(
            bridge.history(
                journal_order=True,
                order_ids=ids,
                limit=2,
                open_frame=open_frame,
                peer_journal_source=True,
            )
        )
        assert HistoryPage.model_validate(desktop).journal_generation == index.journal_generation
        assert desktop["journal_ordinals"] == {id_: index.journal_rows[id_] for id_ in ids}
        assert desktop["peer_journal_source"] is True
        # A ready cold page must remain repairable after the facade hydrates.
        # This is a real stored owner hop with a synthetic hydrated wire window;
        # no strict display-window DTO or recovery publication changes are needed.
        from types import SimpleNamespace
        from typing import cast

        from local_operator.harness.types import Message
        from local_operator.server.utils import desktop_sessions as desktop_module

        bridge.remote = cast(
            Any,
            SimpleNamespace(
                is_cold=False,
                history=lambda: [Message.user("")],
                history_before_token=None,
                history_entry_times=lambda: {},
            ),
        )
        warm = asyncio.run(
            bridge.history(
                journal_order=True,
                order_ids=ids,
                limit=2,
                open_frame=open_frame,
                peer_journal_source=True,
            )
        )
        assert warm == desktop
        monkeypatch.setattr(desktop_module.time, "time", lambda: 1700000000.0)
        # Use one stable wire identity so exact JSON compares source selection,
        # not a newly minted Message ID or serve-time clock.
        wire = Message.user("")
        assert bridge.remote is not None
        bridge.remote.history = lambda: [wire]
        prior_calls = len(seen)
        unasked = asyncio.run(bridge.history(limit=2, open_frame=open_frame))
        assert len(seen) == prior_calls, "absent opt-in must not add a stored hop"
        monkeypatch.setitem(b._handlers, "net_session_history", old_owner)
        old_warm = asyncio.run(
            bridge.history(
                journal_order=True,
                order_ids=ids,
                limit=2,
                open_frame=open_frame,
                peer_journal_source=True,
            )
        )
        assert json.dumps(old_warm) == json.dumps({**unasked, "journal_order_state": "unsupported"})
        # A not-ready or unservable owner cannot replace the legacy wire page.
        for unavailable in ({"entries": [], "journal_order_state": "unsupported"}, None):
            monkeypatch.setattr(
                projection, "peer_stored_history_page", lambda *args, **kwargs: unavailable
            )
            fallback = asyncio.run(
                bridge.history(
                    journal_order=True,
                    order_ids=ids,
                    limit=2,
                    open_frame=open_frame,
                    peer_journal_source=True,
                )
            )
            assert fallback == old_warm
    finally:
        link.close()


@pytest.mark.parametrize("ids", [["x"] * 513, [""], ["x" * 129], "abc", {"x": 1}, [1], [None]])
@pytest.mark.parametrize("boundary", ["owner", "control"])
def test_lookup_shape_rejected_on_both_boundaries(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, ids: Any, boundary: str
) -> None:
    _a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])

    def no_forward(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid IDs reached the peer hop")

    monkeypatch.setattr(b, "_local_peer_call", no_forward)
    with pytest.raises(relay.MeshRefusal, match="order_ids"):
        if boundary == "owner":
            _read(b, journal_order=True, order_ids=ids)
        else:
            b._ctl_peer_history({"journal_order": True, "order_ids": ids})


@pytest.mark.parametrize("open_frame", [False, True])
def test_building_and_changed_page_witness_stay_unsupported(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, open_frame: bool
) -> None:
    _a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, ["", "", ""])
    # Hold the real refresh scheduling edge still, not a fabricated index.
    monkeypatch.setattr(ti, "start_refresh", lambda *args: None)
    # AN UNDECLARED journal_order read is today's plain page: NO state key at
    # all (the VIEWER half states the legacy ``unsupported``), no coordinates,
    # and the refresh is not even started for a caller that can never receive
    # its facts.
    page = _read(b, journal_order=True, **({"open_frame": True} if open_frame else {}))
    assert "journal_order_state" not in page
    assert "journal_generation" not in page and "journal_ordinals" not in page
    assert all("journal_seq" not in row for row in page["entries"])
    # THE COMPLETE DECLARATION GETS THE HONEST BUILDING — the wake is the
    # availability lane — and never a fabricated ordinal or generation.
    declared = _read(
        b,
        journal_order=True,
        peer_journal_source=True,
        **({"open_frame": True} if open_frame else {}),
    )
    assert declared["journal_order_state"] == "building"
    assert declared["peer_journal_source"] is True
    assert "journal_generation" not in declared and "journal_ordinals" not in declared
    assert all("journal_seq" not in row for row in declared["entries"])


def test_shared_join_refuses_a_different_read_witness(peer_pair: Devices) -> None:
    _a, b, _, _ = peer_pair
    ids = _seed_journal(b.root, SESSION, [""])
    _ready(b.root)
    page = {"entries": [{"id": ids[0]}]}
    assert not ti.apply_journal_order(b.root, SESSION, page, (0, 0, 0.0), ids)
    assert page == {"entries": [{"id": ids[0]}]}


def test_size_bound_counts_metadata_and_refuses_metadata_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = {
        "entries": [{"id": "a", "journal_seq": 1}, {"id": "b", "journal_seq": 2}],
        "has_more": False,
        "journal_generation": "g",
        "journal_order_state": "ready",
        "journal_ordinals": {"outside": 3},
    }
    budget = relay.session_history_reply_bytes(
        1, {**page, "entries": page["entries"][-1:], "has_more": True}
    )
    monkeypatch.setattr(relay, "SESSION_HISTORY_REPLY_BUDGET_BYTES", budget)
    fitted = relay.fit_session_history_reply(1, page)
    assert fitted["entries"] == page["entries"][-1:]
    assert fitted["journal_ordinals"] == page["journal_ordinals"]
    assert relay.session_history_reply_bytes(1, fitted) <= budget
    with pytest.raises(relay.MeshRefusal, match="metadata"):
        relay.fit_session_history_reply(1, {"entries": [], "journal_ordinals": {"x": "x" * budget}})


def test_real_pair_wrong_owner_and_denied_read_leak_no_metadata(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b, _, _ = peer_pair
    _seed_journal(b.root, SESSION, [""])
    _ready(b.root)
    wake_store.write_entry(
        b.root,
        SESSION,
        cwd="",
        schedules=[{"id": "no-fire", "state": "scheduled"}],
        preserve={"stopped_at": 1234},
    )
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    try:
        wrong = _call(
            a.root,
            "peer_session_history",
            peer=b.identity.device_id,
            session_id="ffffffffffe1",
            journal_order=True,
            order_ids=["x"],
        )
        assert wrong.get("op") == "error" and wrong.get("code") == "peer_refused", wrong
        assert "journal_generation" not in json.dumps(wrong)
        # The real grant check runs before owner metadata construction.
        local = store.load(record.network_id, b.root)
        assert local is not None
        member = next(row for row in local.members if row.device_id == a.identity.device_id)
        member.capabilities = []
        store.save(local, b.root)
        denied = _call(
            a.root,
            "peer_session_history",
            peer=b.identity.device_id,
            session_id=SESSION,
            journal_order=True,
            order_ids=["x"],
        )
        assert denied.get("op") == "error" and denied.get("code") == "peer_refused", denied
        assert "journal_generation" not in json.dumps(denied)
    finally:
        link.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capabilities, events",
    [
        ([], True),  # an old runtime: NEITHER key is declared
        (["journal-order-v1"], True),  # partial: still NEITHER (the correction)
        (["journal-order-v1", "peer-journal-source-v1"], True),  # complete
        (["journal-order-v1", "peer-journal-source-v1"], False),  # events off
    ],
)
async def test_remote_attach_declares_complete_order_only_when_negotiated(
    tmp_path, capabilities, events
):
    from local_operator.session.runtime.types import SessionRecord

    frames = []
    done = asyncio.Event()

    async def control(reader, writer):
        try:
            await reader.readline()  # local control auth is never part of evidence
            frames.append(json.loads(await reader.readline()))
            writer.write(b'{"op":"ack","detail":{"stream":"synthetic-stream"}}\n')
            writer.write(
                json.dumps({"op": "welcome", "data": {"session_id": SESSION, "pid": 0}}).encode()
                + b"\n"
            )
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            done.set()

    listener = await asyncio.start_server(control, "127.0.0.1", 0)
    facts = projection.RemoteSessionFacts(
        session_id=SESSION,
        device_id="synthetic-owner",
        control_port=listener.sockets[0].getsockname()[1],
        control_key="synthetic-control-key",
    )
    record = SessionRecord(
        pid=1,
        kind="tui",
        session_id=SESSION,
        conversation_name="",
        cwd=str(tmp_path),
        model_label="",
        control_port=0,
        control_key="synthetic-unused-key",
        protocol=4,
        capabilities=list(capabilities),
    )
    client = projection.RemoteSessionClient(
        lambda value: None, lambda reason: None, facts=facts, events=events
    )
    complete = "peer-journal-source-v1" in capabilities
    try:
        await client.connect(record, SESSION)
        declared = True if complete and events else None
        assert frames[0]["auth"].get("journal_order") is declared
        assert frames[0]["auth"].get("peer_journal_source") is declared
    finally:
        client.close()
        await asyncio.wait_for(done.wait(), 2)
        listener.close()
        await listener.wait_closed()
