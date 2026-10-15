"""Owner-certified phone disclosure, separate from raw model history."""
import copy

import pytest

from local_operator.mobile.projection import fold_messages_to_entries
from tests.unit.session.test_peer_cycle_display_archive import make_cycle


@pytest.mark.parametrize("wrapped_send", [False, True])
def test_phone_certified_cycle_keeps_every_physical_row_and_original_raw_input(wrapped_send):
    _, history, record = make_cycle(wrapped_send=wrapped_send)
    # The producer fixture intentionally ends at the quiet ROOT result. Give
    # the negative visibility assertion a REAL subsequent public answer; the
    # consumer must not invent one on the renderer's behalf.
    from local_operator.harness.types import Message
    history.append(Message.assistant("A genuine subsequent answer stays visible."))
    original = [message.model_dump(mode="json") for message in history]
    rows = fold_messages_to_entries(history, session_id=record["session_id"],
                                    peer_cycle_display_records={"records": [record]})
    peer = next(row for row in rows if row.kind == "peer_message")
    stamp = peer.details["peer_cycle"]
    assert stamp["session_id"] == record["session_id"] and stamp["version"] == 1
    ids = stamp["member_ids"]
    assert len(ids) == len(set(ids)) == (4 if wrapped_send else 3)
    members = [row for row in rows if row.id in ids]
    assert [row.id for row in members] == ids
    assert sum(row.kind == "tool" and row.tool_name == "send" for row in members) == 1
    assert members[0].kind == "peer_message"
    assert all(not row.final or row.kind != "assistant" for row in members)
    assert any(row.kind == "assistant" and row.final for row in rows)
    assert [message.model_dump(mode="json") for message in history] == original


def test_phone_never_trusts_raw_reserved_stamp_or_foreign_record():
    _, history, record = make_cycle(wrapped_send=True)
    forged = copy.deepcopy(record)
    forged["session_id"] = "foreign-session"
    history[0].details["peer_cycle"] = {"version": 1, "member_ids": ["forged"]}
    original = copy.deepcopy(history[0].details)
    for metadata in (None, {"records": [forged]}):
        rows = fold_messages_to_entries(history, session_id=record["session_id"],
                                        peer_cycle_display_records=metadata)
        assert all("peer_cycle" not in row.details for row in rows)
        assert history[0].details == original


def test_default_phone_wire_omits_negotiation_and_certificates():
    from local_operator.mobile.types import SessionProjection
    _, history, _ = make_cycle(wrapped_send=True)
    rows = fold_messages_to_entries(history)
    assert all("peer_cycle" not in row.details for row in rows)
    assert "peer_cycles" not in SessionProjection(session_id="sess", pid=0).to_json()


@pytest.mark.parametrize("send_count", [0, 65])
def test_phone_truthful_zero_and_sixty_five_logical_communications(send_count):
    _, history, record = make_cycle(send_count=send_count)
    rows = fold_messages_to_entries(history, session_id="sess", peer_cycle_display_records={"records": [record]})
    stamp = next(row.details["peer_cycle"] for row in rows if "peer_cycle" in row.details)
    members = [row for row in rows if row.id in stamp["member_ids"]]
    assert sum(row.kind == "tool" and row.tool_name == "send" for row in members) == send_count
    assert len(members) == len(record["member_ids"])


@pytest.mark.asyncio
async def test_actual_frontend_late_publication_bumps_idle_phone_without_final_or_user_event(tmp_path):
    import asyncio
    from local_operator.harness.peer_cycle_display import persist_record
    from local_operator.mobile.peer_cycles import bind_owner_peer_cycles
    from local_operator.mobile.projection import ProjectionFold
    from local_operator.mobile.types import SessionProjection
    from tests.unit.session.test_session import make_session, ScriptedStream
    _, history, record = make_cycle(wrapped_send=True)
    stream = ScriptedStream([])
    session = make_session(tmp_path / "sessions", stream)
    await session.transcript.append_messages(history)
    projection = SessionProjection(session_id=session.session_id, pid=0)
    fold = ProjectionFold(projection)
    notifications = []
    events = []
    unlisten = session.subscribe(events.append)
    unsubscribe = bind_owner_peer_cycles(fold, session, lambda: notifications.append(projection.version), asyncio.get_running_loop())
    try:
        fold.fold_history(history)
        assert not any("peer_cycle" in row.details for row in projection.transcript)
        before = projection.version
        assert await persist_record(session.transcript, record)
        session._frontend_state_store.mutate(peer_cycle_display={"refs": [record["id"]], "records": [record]})
        await asyncio.sleep(0)
        assert projection.version > before and notifications
        assert any("peer_cycle" in row.details for row in projection.transcript)
        assert not events and not stream.requests
        assert projection.peer_cycles == 1
    finally:
        unsubscribe()
        unlisten()
        await session.dispose()


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_shared_context_corpus_phone_aliases_and_closed_members(case_index):
    # Consumer-only vocabulary parity, NOT a fabricated producer certificate:
    # producer admission is independently exercised by the real Session suite.
    import json
    from pathlib import Path
    from local_operator.harness.peer_cycles import PeerCycleCertificate
    from local_operator.mobile.peer_cycles import project_peer_cycles
    from local_operator.mobile.types import TranscriptEntry
    corpus = json.loads((Path(__file__).parents[3] / "local_operator/mobile/web/src/lib/peer-cycles.context.json").read_text())
    case = corpus["cases"][case_index]
    rows = [TranscriptEntry(**raw, tool_state="done") for raw in case["rows"]]
    sources = tuple(f"{row.id}:0" if row.kind == "peer_message" else f"tool:{row.tool_call_id}:0"
                    for row in rows if row.id in case["members"])
    proof = PeerCycleCertificate(corpus["session_id"], 0, "actual-typed-quiet", sources,
                                 owner_kind="peer_cycle_display")
    projected = project_peer_cycles(rows, session_id=corpus["session_id"], records=None, certificates=[proof])
    assert projected[0].details["peer_cycle"]["member_ids"] == case["members"]
    assert sum(row.tool_name == "send" for row in projected if row.id in case["members"]) == case["actions"]
    assert "public-final" not in projected[0].details["peer_cycle"]["member_ids"]
    assert not any("peer_cycle" in row.details for row in rows)
    assert not any("peer_cycle" in row.details for row in project_peer_cycles(
        rows, session_id=corpus["session_id"], records=None, certificates=[proof], view_epoch=1))


def test_phone_missing_parent_outcome_or_public_boundary_cannot_certify():
    _, history, record = make_cycle(wrapped_send=True)
    for raw in (history[:2], history[:-2]):
        rows = fold_messages_to_entries(raw, session_id=record["session_id"],
                                        peer_cycle_display_records={"records": [record]})
        assert all("peer_cycle" not in row.details for row in rows)
