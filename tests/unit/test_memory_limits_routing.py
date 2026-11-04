"""Memory slice A: content-token item cap, session-default routing, actionable quota
refusals and the human global-bank triage.

The defects these pin (operator report, 2026-10-27): a 689-character goal was refused
because the 256-token item cap charged the ~75-token JSON envelope; the retry then
hit a value-free "Memory quota reached" because the ``memory`` tool always wrote the
GLOBAL bank, which misrouted work state had filled to 99%.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import local_operator.memory as memory
from local_operator.compaction.tokens import count_text_tokens
from local_operator.harness.types import ToolContext
from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore, _dump
from local_operator.tools.memory_tools import execute_fact_memory

#: The F3 text: one line, no tracker ref / commit / CI verb, yet plainly one
#: session's work. Its session-ID self-reference is what global now refuses.
F3_GOAL = (
    "Session 25ee8675d7f7 goal (2026-10-27): ship the context-reduction POC behind the "
    "Radient proxy so long agent sessions stop resending the same tool schemas and "
    "stale transcript windows on every turn; measure cached versus uncached prompt "
    "tokens on three real sessions before and after, keep the operator-visible "
    "behaviour identical, record the numbers in the project notes, and leave the "
    "proxy disabled by default until the operator reviews the measurements and "
    "decides whether the reduction is worth the extra moving part in the request "
    "path for every provider we route through today, including the two that "
    "still bill cache writes separately; owners: lane-a (proxy), lane-b (metrics), "
    "lane-c (docs); checkpoints 10:30/14:00/17:15 EDT; cap 2,000 req/min, 4 KiB "
    "schema delta, 12 % budget."
)


def _ctx(root: Path, session: str = "s1") -> ToolContext:
    directory = root / "sessions" / session
    directory.mkdir(parents=True, exist_ok=True)
    return ToolContext(session_id=session, memory_config_dir=str(root), session_dir=str(directory))


def _call(args: dict[str, Any], context: ToolContext):
    result = asyncio.run(execute_fact_memory("c", args, context=context))
    return result, str(getattr(result.content[0], "text", ""))


def _store(root: Path, session: str = "s1") -> MemoryStore:
    return MemoryStore(root, MemoryIdentity(session), root / "sessions" / session)


# --- A1: item cap ---------------------------------------------------------------


def test_the_f3_goal_fits_the_content_cap_although_its_wire_form_exceeded_256():
    assert len(F3_GOAL) > 600
    fact = memory.validate_fact(
        {
            "title": F3_GOAL.splitlines()[0][:100],
            "body": F3_GOAL,
            "source": {"kind": "reported", "session": "25ee8675d7f7", "author": "a"},
            "created_at": 1.0,
        }
    )
    assert count_text_tokens(_dump(fact)) > memory.LEGACY_ITEM_WIRE_TOKENS


def test_content_cap_accepts_450_and_refuses_600_naming_the_overage(tmp_path):
    store = _store(tmp_path)
    # title "t" + newlines + "alpha"*n count n+2 tokens (measured): 450 and 600.
    ok = store.put("session", {"title": "t", "body": "alpha " * 448})
    assert ok["status"] == "active"
    with pytest.raises(MemoryError) as caught:
        store.put("session", {"title": "t", "body": "alpha " * 597})
    message = str(caught.value)
    assert "600 content tokens; the limit is 512 (88 over)" in message
    assert "alpha" not in message


def test_the_byte_bound_is_still_the_storage_guard(tmp_path):
    """Bytes stay a separate bound from content tokens.

    ``guard`` already refuses a RAW payload over 8192 bytes; the stored-bytes check
    covers what the defaults add (status, tags, timestamps ...). Size the body so the
    raw payload passes and the stored fact does not: a whitespace run is ~67 tokens
    per 8 KB, so the token cap cannot be what refuses it.
    """
    store = _store(tmp_path)
    raw = {"title": "t", "body": "a b"}
    stamped = store.prepare_fact(raw)
    pad = 8192 - len(_dump(stamped).encode()) + 20  # stored fact 20 bytes over
    raw["body"] = "a" + " " * pad + "b"
    assert count_text_tokens(raw["body"]) < memory.ITEM_TOKENS
    with pytest.raises(MemoryError, match=r"bytes stored; the limit is 8192 \(\d+ over\)"):
        store.put("session", raw)


def test_current_tokens_still_count_wire_tokens(tmp_path):
    store = _store(tmp_path)
    item = store.put("session", {"title": "t", "body": "alpha " * 20})
    with store.connection() as db:
        row = db.execute(
            "SELECT fact,tokens FROM current WHERE entity_id=?", (item["id"],)
        ).fetchone()
    assert row[1] == count_text_tokens(row[0])


# --- A2: routing ---------------------------------------------------------------


def test_no_scope_stores_to_session_and_leaves_global_unchanged(tmp_path):
    result, text = _call({"op": "store", "text": F3_GOAL}, _ctx(tmp_path))
    assert not result.is_error, text
    receipt = json.loads(text)
    assert receipt["scope"] == "session"
    assert receipt["lifetime"] == "session/this conversation"
    assert receipt["uri"].startswith("memory://session/")
    assert _store(tmp_path).inventory("global")["items"] == []
    exported = (tmp_path / "sessions" / "s1" / memory.SESSION_EXPORT_NAME).read_text()
    assert receipt["id"] in exported


def test_the_f3_goal_is_refused_in_global_naming_the_marker_and_the_remedy(tmp_path):
    result, text = _call({"op": "store", "scope": "global", "text": F3_GOAL}, _ctx(tmp_path))
    assert result.is_error
    assert "session-self-reference" in text and 'scope="session"' in text
    assert "Radient" not in text and "25ee8675d7f7" not in text
    assert not (tmp_path / "memory" / "memory.db").exists() or (
        _store(tmp_path).inventory("global")["items"] == []
    )


def test_session_scope_without_a_bound_session_refuses_rather_than_promotes(tmp_path):
    context = ToolContext(memory_config_dir=str(tmp_path))
    result, text = _call({"op": "store", "text": "Lane note"}, context)
    assert result.is_error and 'scope="global"' in text
    assert not (tmp_path / "memory").exists()
    # Recall with no scope still works unbound: it searches global.
    recall, _ = _call({"op": "recall", "text": "Elara"}, context)
    assert not recall.is_error


def test_scoped_recall_restricts_to_that_bank(tmp_path):
    context = _ctx(tmp_path)
    _call({"op": "store", "text": "Kestrel lane decision: use the proxy"}, context)
    _call({"op": "store", "scope": "global", "text": "Kestrel is my cat's name"}, context)
    _, both = _call({"op": "recall", "text": "Kestrel"}, context)
    _, only_global = _call({"op": "recall", "scope": "global", "text": "Kestrel"}, context)
    _, only_session = _call({"op": "recall", "scope": "session", "text": "Kestrel"}, context)
    assert {row["scope"] for row in json.loads(both)["items"]} == {"session", "global"}
    assert json.loads(only_global)["bank"] == "global/across sessions"
    assert {row["scope"] for row in json.loads(only_global)["items"]} == {"global"}
    assert {row["scope"] for row in json.loads(only_session)["items"]} == {"session"}


@pytest.mark.parametrize(
    ("text", "marker"),
    [
        ("Notes for session 25ee8675d7f7 on the proxy", "session-self-reference"),
        ("Proxy notes (25ee8675d7f7) for later", "session-self-reference"),
        ("Session goal (Oct): ship the proxy", "goal-statement"),
        ("Lane goal: finish the i18n sweep", "goal-statement"),
        ("2026-10-27 merged the proxy and ran QA", "dated-work-log"),
        ("Shipped v3.4.1 on 2026-10-27", "dated-work-log"),
    ],
)
def test_new_global_markers_refuse_work_shapes(text, marker):
    with pytest.raises(MemoryError, match=marker):
        memory.scope_fit("global", {"title": "x", "body": text, "tags": [], "aliases": []})


@pytest.mark.parametrize(
    "text",
    [
        "My colleague Elara lives in Dunedin",
        "Elara's birthday is 1990-04-02",
        "Call Mira at (021) 555 0134 about dinner",
        "Account (123456789012) is the joint savings one",
        "My goal is to run a marathon next year",
        "Prefers dark mode and British spelling",
    ],
)
def test_new_global_markers_leave_personal_facts_alone(text):
    memory.scope_fit("global", {"title": "x", "body": text, "tags": [], "aliases": []})


# --- A3: quota messages --------------------------------------------------------


def test_token_quota_names_bank_usage_need_and_triage(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "GLOBAL_TOKENS", 200)
    store = _store(tmp_path)
    store.put("global", {"title": "Elara", "body": "Elara lives in Dunedin " * 4})
    with pytest.raises(MemoryError) as caught:
        store.put("global", {"title": "Mira", "body": "Mira lives in Nelson " * 8})
    message = str(caught.value)
    assert message.startswith("Memory quota reached: global bank ")
    assert "/200 tokens" in message and "this item needs " in message
    assert "Nothing was stored or evicted." in message
    assert "`lop memory triage --scope global`" in message
    assert "Nelson" not in message and "Dunedin" not in message
    assert len(store.inventory("global")["items"]) == 1


def test_the_tool_adds_the_session_route_to_a_global_quota_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "GLOBAL_TOKENS", 80)
    context = _ctx(tmp_path)
    _call({"op": "store", "scope": "global", "text": "Elara lives in Dunedin"}, context)
    result, text = _call(
        {"op": "store", "scope": "global", "text": "Mira lives in Nelson near the sea"}, context
    )
    assert result.is_error
    assert "global bank" in text and text.endswith('(scope="session").')


def test_op_cap_names_history_and_compact(tmp_path, monkeypatch):
    store = _store(tmp_path, "25ee8675d7f7")
    store.put("session", {"title": "a", "body": "first"})
    monkeypatch.setattr(memory, "MAX_OWNER_OPS", 1)
    with pytest.raises(MemoryError) as caught:
        store.put("session", {"title": "b", "body": "second"})
    message = str(caught.value)
    assert "session bank 25ee8675 has 2/1 history operations" in message
    assert "`lop memory compact`" in message


def test_tombstone_cap_names_tombstones_and_export(tmp_path, monkeypatch):
    store = _store(tmp_path, "25ee8675d7f7")
    first = store.put("session", {"title": "a", "body": "first"})
    store.put("session", {}, entity=first["id"], expected_version=first["version"], forget=True)
    monkeypatch.setattr(memory, "MAX_TOMBSTONES", 0)
    with pytest.raises(MemoryError) as caught:
        store.put("session", {"title": "b", "body": "second"})
    message = str(caught.value)
    assert "has 1/0 tombstones" in message and "export" in message


# --- A4: triage ----------------------------------------------------------------


def _conversation(root: Path, session: str) -> Path:
    """A real conversation directory: triage/move target only sessions with a transcript."""
    directory = root / "sessions" / session
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("")
    return directory


def _seed_misrouted(root: Path) -> dict[str, str]:
    """Global facts written by two sessions, as the old global-only tool did."""
    ids = {}
    for session in ("aaaa00000001", "bbbb00000002"):
        _conversation(root, session)
    work = MemoryStore(root, MemoryIdentity("aaaa00000001"))
    with pytest.MonkeyPatch.context() as patch:
        # Seed as the OLD build did: global without the new markers.
        patch.setattr(memory, "scope_fit", lambda scope, fact: None)
        ids["work1"] = work.put("global", {"title": "Lane goal: ship", "body": "Lane goal: ship"})[
            "id"
        ]
        ids["work2"] = work.put(
            "global", {"title": "log", "body": "2026-10-20 merged the PR after QA"}
        )["id"]
        other = MemoryStore(root, MemoryIdentity("bbbb00000002"))
        ids["work3"] = other.put(
            "global", {"title": "self", "body": "Notes from session bbbb00000002 today"}
        )["id"]
        ids["gone"] = MemoryStore(root, MemoryIdentity("cccc00000003")).put(
            "global", {"title": "orphan", "body": "Lane goal: orphan work"}
        )["id"]
        ids["personal"] = work.put(
            "global", {"title": "Elara", "body": "My colleague Elara lives in Dunedin"}
        )["id"]
    return ids


def _cli(root: Path, text: str, prefix: str = "lop memory"):
    from local_operator.memory_cli import render_human, run

    raw, failed = run(text, config=root, identity=MemoryIdentity(), prefix=prefix)
    return raw, failed, (raw if failed else render_human(raw, failed, prefix=prefix))


def test_triage_preview_groups_by_source_session_and_is_read_only(tmp_path):
    ids = _seed_misrouted(tmp_path)
    before = (tmp_path / "memory" / "memory.db").read_bytes()
    raw, failed, human = _cli(tmp_path, "triage --scope global")
    assert not failed, raw
    value = json.loads(raw)
    groups = {group["session"]: group for group in value["groups"]}
    assert set(groups) == {"aaaa00000001", "bbbb00000002"}
    assert {fact["id"] for fact in groups["aaaa00000001"]["facts"]} == {
        ids["work1"],
        ids["work2"],
    }
    assert value["stranded"]["facts"] == 1 and value["kept"]["facts"] == 1
    assert value["moving"]["facts"] == 3
    assert "Read-only preview" in human and value["fingerprint"] in human
    assert (tmp_path / "memory" / "memory.db").read_bytes() == before


def test_triage_apply_refuses_a_stale_fingerprint_and_moves_nothing(tmp_path):
    _seed_misrouted(tmp_path)
    preview = json.loads(_cli(tmp_path, "triage --scope global")[0])
    (
        MemoryStore(tmp_path, MemoryIdentity("aaaa00000001")).put(
            "global", {"title": "x", "body": "Lane goal: a new one"}
        )
        if False
        else None
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(memory, "scope_fit", lambda scope, fact: None)
        MemoryStore(tmp_path, MemoryIdentity("aaaa00000001")).put(
            "global", {"title": "x", "body": "Lane goal: a new one"}
        )
    raw, failed, _ = _cli(
        tmp_path, f"triage --scope global --apply --expect {preview['fingerprint']}"
    )
    assert failed and "fingerprint mismatch" in raw and "Nothing was moved" in raw
    assert len(MemoryStore(tmp_path, MemoryIdentity()).inventory("global")["items"]) == 6


def test_triage_apply_moves_to_source_sessions_and_never_deletes(tmp_path):
    ids = _seed_misrouted(tmp_path)
    preview = json.loads(_cli(tmp_path, "triage --scope global")[0])
    raw, failed, human = _cli(
        tmp_path, f"triage --scope global --apply --expect {preview['fingerprint']}"
    )
    assert not failed, raw
    receipt = json.loads(raw)
    assert sum(group["moved"] for group in receipt["groups"]) == 3
    left = {
        row["id"] for row in MemoryStore(tmp_path, MemoryIdentity()).inventory("global")["items"]
    }
    assert left == {ids["gone"], ids["personal"]}
    for session, count in (("aaaa00000001", 2), ("bbbb00000002", 1)):
        owner = MemoryStore(tmp_path, MemoryIdentity(session), tmp_path / "sessions" / session)
        rows = owner.inventory("session")["items"]
        assert len(rows) == count
        assert owner.show("session", rows[0]["id"])["moved_from"]["scope"] == "global"
        exported = (tmp_path / "sessions" / session / memory.SESSION_EXPORT_NAME).read_text()
        assert all(row["id"] in exported for row in rows)
    assert "Global bank now" in human


def test_triage_only_and_include_select_the_plan(tmp_path):
    ids = _seed_misrouted(tmp_path)
    only = json.loads(_cli(tmp_path, "triage --scope global --only bbbb00000002")[0])
    assert [group["session"] for group in only["groups"]] == ["bbbb00000002"]
    included = json.loads(_cli(tmp_path, f"triage --scope global --include {ids['personal']}")[0])
    moving = {fact["id"] for group in included["groups"] for fact in group["facts"]}
    assert ids["personal"] in moving
    assert "--include" in _cli(tmp_path, f"triage --scope global --include {ids['personal']}")[2]
    raw, failed, _ = _cli(tmp_path, "triage --scope global --include deadbeefdeadbeef")
    assert failed and "match no global fact" in raw


def test_triage_apply_is_terminal_only_and_flags_are_scoped(tmp_path):
    _seed_misrouted(tmp_path)
    fingerprint = json.loads(_cli(tmp_path, "triage --scope global")[0])["fingerprint"]
    raw, failed, _ = _cli(
        tmp_path, f"triage --scope global --apply --expect {fingerprint}", prefix="/memory"
    )
    assert failed and "terminal only" in raw
    for text, needle in (
        ("triage", "--scope global"),
        ("triage --scope global --apply", "fingerprint"),
        ("triage --scope global --expect abc", "--apply"),
        ("list --apply", "`triage` only"),
    ):
        raw, failed, _ = _cli(tmp_path, text)
        assert failed and needle in raw, (text, raw)


def test_a_session_bound_store_cannot_name_another_destination_owner(tmp_path):
    _conversation(tmp_path, "victim")
    agent = MemoryStore(tmp_path, MemoryIdentity("agent-session"))
    item = agent.put("global", {"title": "Elara", "body": "Elara lives in Dunedin"})
    with pytest.raises(MemoryError, match="human `lop memory triage`"):
        agent.move(f"global/{item['id']}", "session", destination_owner="victim")
    human = MemoryStore(tmp_path, MemoryIdentity())
    with pytest.raises(MemoryError, match="human `lop memory triage`"):
        human.move(f"global/{item['id']}", "team", destination_owner="victim")
    with pytest.raises(MemoryError, match="not a conversation on this device"):
        human.move(f"global/{item['id']}", "session", destination_owner="missing")
    # A bare directory (no transcript) is not a session either (review m2).
    (tmp_path / "sessions" / "bare").mkdir()
    with pytest.raises(MemoryError, match="not a conversation on this device"):
        human.move(f"global/{item['id']}", "session", destination_owner="bare")
    assert agent.show("global", item["id"])["status"] == "active"


def test_a_named_destination_folds_its_portable_log_in_before_exporting(tmp_path):
    """Rows only the target session's log holds must survive the move's re-export."""
    directory = _conversation(tmp_path, "target")
    owner = MemoryStore(tmp_path / "elsewhere", MemoryIdentity("target"), directory)
    kept = owner.put("session", {"title": "kept", "body": "Only in the portable log"})
    item = MemoryStore(tmp_path, MemoryIdentity()).put(
        "global", {"title": "Elara", "body": "Elara lives in Dunedin"}
    )
    MemoryStore(tmp_path, MemoryIdentity()).move(
        f"global/{item['id']}", "session", scope="global", destination_owner="target"
    )
    exported = (directory / memory.SESSION_EXPORT_NAME).read_text()
    assert kept["id"] in exported


# --- A1 mesh compatibility ------------------------------------------------------


def test_an_old_peer_gets_a_withheld_skip_for_an_item_over_the_legacy_cap(tmp_path):
    from local_operator.network.memory import batch

    sender = MemoryStore(tmp_path / "a", MemoryIdentity("s"), origin="device-a")
    small = sender.put("global", {"title": "Elara", "body": "Elara lives in Dunedin"})
    big = sender.put("global", {"title": "Mira", "body": "Mira likes long walks " * 60})
    after = sender.put("global", {"title": "Kai", "body": "Kai lives in Nelson"})
    scopes = ["global"]
    withheld: list[str] = []
    legacy: list[str] = []
    old = batch(
        sender, {}, scopes, {"device-a"}, 0, withheld, legacy_item_cap=True, legacy_withheld=legacy
    )
    sent = {op["entity_id"] for op in old["ops"]}
    assert sent == {small["id"], after["id"]}
    # Reported as OWED (for redelivery), not as an unknown-origin withholding.
    assert withheld == [] and legacy == [big["version"]] and old["after"]["device-a"] == 3
    # The skip is a content-free range the receiver's gap check accepts.
    assert [["device-a", 2, 2]] == [skip for skip in old["skips"]]
    # An old receiver's validator (256 wire tokens) accepts every op it is sent.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(memory, "ITEM_TOKENS", 10**6)
        for op in old["ops"]:
            assert count_text_tokens(_dump(op["fact"])) <= memory.LEGACY_ITEM_WIRE_TOKENS
    new = batch(sender, {}, scopes, {"device-a"}, 0, [])
    assert {op["entity_id"] for op in new["ops"]} == {small["id"], big["id"], after["id"]}
    receiver = MemoryStore(tmp_path / "b", MemoryIdentity("s"), origin="device-b")
    receiver.apply_ops(old["ops"], origins={"device-a"}, contiguous=False)
    assert {row["id"] for row in receiver.inventory("global")["items"]} == sent


def test_the_capability_is_advertised():
    from local_operator.network import wire

    assert wire.MEMORY_ITEM_512 in wire.LINK_CAPABILITIES


# --- A1 prompt bound ------------------------------------------------------------


def test_twenty_max_size_decisions_render_within_the_tier_budget():
    from local_operator.compaction.carry import (
        CARRIED_DECISION_CHARS,
        render_carried_tier,
    )

    body = "We decided to route the proxy through the gateway because " * 40
    assert count_text_tokens(body) >= 400
    message = render_carried_tier([], [{"body": body, "tag": "decision"}] * 20)
    assert message is not None
    text = "".join(getattr(block, "text", "") for block in message.content)
    assert count_text_tokens(text) <= 2200
    rows = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(rows) == 20 and all(len(row) <= CARRIED_DECISION_CHARS + 3 for row in rows)
    assert all(row.endswith("…") for row in rows)
    # Characters alone do not bound tokens: dense identifier text and CJK are
    # clamped by the token bound too, so the tier worst case holds for any script.
    for dense in (
        " ".join(f"{n:040x}" for n in range(10**30, 10**30 + 60)),
        "决定通过网关路由代理因为缓存命中率更高" * 100,
    ):
        message = render_carried_tier([], [{"body": dense, "tag": "decision"}] * 20)
        assert message is not None
        text = "".join(getattr(block, "text", "") for block in message.content)
        assert count_text_tokens(text) <= 2200


def test_contract_validator_accepts_only_the_two_scopes():
    valid = memory.valid_memory_tool_arguments
    assert valid({"op": "store", "text": "x"})
    assert valid({"op": "store", "text": "x", "scope": "session"})
    assert valid({"op": "recall", "text": "x", "scope": "global", "i": "Recalling"})
    for scope in ("team", "agent", "", None, 1):
        assert not valid({"op": "store", "text": "x", "scope": scope})


def test_no_sqlite_handle_leaks_from_triage_preview(tmp_path):
    _seed_misrouted(tmp_path)
    _cli(tmp_path, "triage --scope global")
    # A leaked read connection on an immutable URI would keep the file busy; a
    # writer must still get its lock immediately.
    db = sqlite3.connect(tmp_path / "memory" / "memory.db", timeout=0.1)
    db.execute("BEGIN IMMEDIATE")
    db.rollback()
    db.close()


# --- Round 1 remediation ----------------------------------------------------------


def test_the_global_cap_stays_16000_until_receivers_stop_enforcing_it():
    # Review M2: a receiver enforces the SENDER's quota on apply, so raising the
    # cap here would stall every un-updated peer's stream at 16,000.
    assert memory.GLOBAL_TOKENS == 16000


def test_a_title_repeating_the_body_opening_is_not_charged_twice(tmp_path):
    """QA Q1: the tool derives title from the first line; that is not new content."""
    line = "alpha " * 500
    fact = memory.validate_fact(
        {"title": line.strip()[:100], "body": line, "source": {"kind": "reported"}}
    )
    assert fact["body"] == line
    # A distinct title IS content and is still charged.
    with pytest.raises(MemoryError, match="content tokens"):
        memory.validate_fact({"title": "beta " * 30, "body": line, "source": {"kind": "reported"}})


def test_triage_trusts_source_session_only_for_this_devices_writes(tmp_path):
    """Review m1: ``source.session`` is writer-asserted; a synced fact naming a local
    session must not be moved into it."""
    ids = _seed_misrouted(tmp_path)
    remote = MemoryStore(tmp_path / "remote", MemoryIdentity("zzzz00000009"), origin="device-r")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(memory, "scope_fit", lambda scope, fact: None)
        planted = remote.put(
            "global",
            {
                "title": "plant",
                "body": "Lane goal: plant into aaaa",
                "source": {"kind": "reported", "session": "aaaa00000001"},
            },
        )
    with remote.connection() as db:
        ops = [json.loads(row[0]) for row in db.execute("SELECT data FROM ops")]
    MemoryStore(tmp_path, MemoryIdentity()).apply_ops(ops, origins={"device-r"}, contiguous=False)
    raw, failed, human = _cli(tmp_path, "triage --scope global")
    assert not failed, raw
    plan = json.loads(raw)
    planned = {fact["id"] for group in plan["groups"] for fact in group["facts"]}
    assert planned == {ids["work1"], ids["work2"], ids["work3"]}
    assert planted["id"] not in planned and plan["foreign"]["facts"] == 1
    assert "another device wrote them" in human


def test_triage_apply_receipt_names_old_and_new_ids_and_warns_on_legacy_size(tmp_path):
    """QA Q3 (old -> new references) and review M3 (legacy-size warning)."""
    ids = _seed_misrouted(tmp_path)
    work = MemoryStore(tmp_path, MemoryIdentity("aaaa00000001"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(memory, "scope_fit", lambda scope, fact: None)
        big = work.put("global", {"title": "Lane goal: big", "body": "Lane goal: " + "x y " * 200})
    assert count_text_tokens(_dump(work.show("global", big["id"])["fact"])) > 256
    raw, failed, human = _cli(tmp_path, "triage --scope global")
    assert "exceed the item limit of builds before memory-item-512" in human
    fp = json.loads(raw)["fingerprint"]
    raw, failed, human = _cli(tmp_path, f"triage --scope global --apply --expect {fp}")
    assert not failed, raw
    receipt = json.loads(raw)
    moves = {m["from"]: m["to"] for group in receipt["groups"] for m in group["moves"]}
    assert f"global/{ids['work1']}" in moves and f"global/{big['id']}" in moves
    assert all(target.startswith("session/") for target in moves.values())
    assert f"global/{ids['work1']} → {moves['global/' + ids['work1']]}" in human
    assert "run lop-update there" in human


def test_a_session_move_of_a_legacy_size_fact_carries_the_compatibility_note(tmp_path):
    store = _store(tmp_path)
    big = store.put("global", {"title": "Mira", "body": "Mira likes long walks " * 60})
    small = store.put("global", {"title": "Kai", "body": "Kai lives in Nelson"})
    receipt = store.move([f"global/{big['id']}"], "session")
    assert "1 fact(s) here exceed" in receipt["compatibility"]
    assert "compatibility" not in store.move([f"global/{small['id']}"], "session")


def test_an_overflowing_legacy_withheld_list_rebaselines_the_stream(tmp_path, monkeypatch):
    from local_operator.network import memory as net

    store = MemoryStore(tmp_path, MemoryIdentity("s"), origin="device-a")
    monkeypatch.setattr(net, "LEGACY_WITHHELD_CAP", 2)
    # The overflow arm is about a list that is genuinely longer than the cap, so the
    # ids must be OPS this store still holds: a dead id is dropped before the cap now
    # (``test_stale_ids_alone_do_not_force_a_full_resend``), which is the fix for
    # PR #3057 review r6, n1. This test used 300 made-up ids, all of them dead.
    for n in range(3):
        store.put("global", {"title": f"x{n}", "body": f"y{n}"})
    with store.connection() as db:
        held = [row[0] for row in db.execute("SELECT op_id FROM ops")]
    assert len(held) == 3
    net._save_legacy_pending(store, "d_peer", held, False)
    assert net.legacy_pending(store, "d_peer") == {"ids": [], "overflow": True}
    net._save_legacy_pending(store, "d_peer", [], False)
    assert net.legacy_pending(store, "d_peer") == {"ids": [], "overflow": False}


def test_a_redelivered_op_must_sit_behind_the_receivers_cursor(tmp_path):
    """The redelivery list bypasses contiguity accounting, so it may only carry ops
    the receiver's cursor has already passed — never a way around the gap check."""
    import types

    from local_operator.network import memory as net

    sender = MemoryStore(tmp_path / "s", MemoryIdentity("s"), origin="d_aaaaaaaa")
    item = sender.put("global", {"title": "Mira", "body": "Mira likes long walks " * 60})
    with sender.connection() as db:
        op = json.loads(
            db.execute("SELECT data FROM ops WHERE op_id=?", (item["version"],)).fetchone()[0]
        )
    payload = {
        "v": 1,
        "generation": 0,
        "scopes": ["global"],
        "after": {},
        "ops": [],
        "skips": [],
        "proofs": [],
        "redeliver": [op],
    }
    server = types.SimpleNamespace(root=tmp_path / "r")
    with pytest.raises(MemoryError, match="not behind this peer's cursor"):
        net.receive(server, "d_aaaaaaaa", payload, ["global"], {"d_aaaaaaaa"})
    assert not MemoryStore(tmp_path / "r", MemoryIdentity("s")).search("Mira", "global")


# --- Round 2 remediation ----------------------------------------------------------


def test_a_single_line_text_at_exactly_the_limit_is_stored(tmp_path):
    """QA Q4: empty title/subject parts no longer add separator tokens."""
    text = " ".join(f"w{n}" for n in range(2000))
    while count_text_tokens(text) > memory.ITEM_TOKENS:
        text = text.rsplit(" ", 1)[0]
    while count_text_tokens(text) < memory.ITEM_TOKENS:
        text += " z"
    assert count_text_tokens(text) == memory.ITEM_TOKENS
    stored, raw = _call({"op": "store", "text": text}, _ctx(tmp_path))
    assert not stored.is_error, raw


def test_only_a_session_store_over_the_legacy_bound_carries_the_note(tmp_path):
    """QA Q5: the note appears where an old importer would refuse, nowhere else."""
    context = _ctx(tmp_path)
    big = json.loads(_call({"op": "store", "text": "alpha beta " * 160}, context)[1])
    assert "exceed the item limit of builds before memory-item-512" in big["compatibility"]
    small = json.loads(_call({"op": "store", "text": "Kestrel goal: ship the proxy"}, context)[1])
    assert "compatibility" not in small
    personal = json.loads(
        _call({"op": "store", "scope": "global", "text": "Elara likes tea " * 40}, context)[1]
    )
    assert "compatibility" not in personal


def test_an_overflowed_peer_reports_overflow_not_zero(tmp_path):
    """QA Q6: ``pending`` must not read as nothing owed while the list overflowed."""
    from local_operator.network import memory as net

    assert net._owed({"ids": [], "overflow": True}) == "overflow"
    assert net._owed({"ids": ["a", "b"], "overflow": False}) == 2
