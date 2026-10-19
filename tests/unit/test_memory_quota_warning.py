"""Quota headroom is visible, and 80/90% crossings warn exactly once.

The defect these tests pin: the global bank filled to 99.4% and every write
failed with a bare quota error that read like an unrelated bug. A scope must
report headroom and percent on writes and in `status`, and a crossing must warn
the agent once — with an action it can take — instead of staying silent until
the cap refuses a write.
"""

import json

import pytest

import local_operator.memory as memory
from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore
from local_operator.memory_cli import render_human, run


def fact(body="Release evidence lives at docs/release.md", **kwargs):
    return dict(title="Release evidence", body=body, tags=["release"], **kwargs)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        tmp_path / "cfg",
        MemoryIdentity("session-a", "team-a", "agent-a"),
        tmp_path / "sessions" / "session-a",
        origin="device-a",
    )


def _fill(store, scope, cap, target_percent):
    """Write small facts until usage crosses ``target_percent`` of ``cap``."""
    warnings = []
    while True:
        used = store.inventory(scope)["active_tokens"]
        if used >= cap * target_percent / 100:
            return warnings
        result = store.put(scope, fact(f"Fact for crossing at {used} tokens"))
        if result.get("warning"):
            warnings.append(result)
        if len(warnings) > 5:
            return warnings


def test_every_write_reports_headroom_and_percent(store):
    result = store.put("session", fact())
    quota = result["quota"]
    assert quota["scope"] == "session"
    assert quota["cap_tokens"] == memory.OWNER_TOKENS
    assert quota["used_tokens"] > 0
    assert quota["headroom_tokens"] == quota["cap_tokens"] - quota["used_tokens"]
    assert 0 < quota["percent"] < 100
    assert "warning" not in result


def test_80_percent_crossing_warns_once_with_actionable_guidance(store, monkeypatch):
    monkeypatch.setattr(memory, "OWNER_TOKENS", 2000)
    warnings = _fill(store, "session", 2000, 80)
    assert len(warnings) == 1
    warned = warnings[0]
    assert "80" in warned["warning"]
    assert "onsolidate" in warned["warning"] and "move" in warned["warning"]
    assert 80 <= warned["quota"]["percent"] < 90
    # Subsequent writes under 90% do not re-fire the same threshold.
    later = store.put("session", fact("ok"))
    assert "warning" not in later


def test_90_percent_crossing_warns_once_and_never_again(store, monkeypatch):
    monkeypatch.setattr(memory, "OWNER_TOKENS", 1200)
    _fill(store, "session", 1200, 80)
    at_ninety = None
    for n in range(6):
        result = store.put("session", fact(f"Ninety crossing fact {n}"))
        if result.get("warning"):
            at_ninety = result
            break
    assert at_ninety is not None
    assert "90" in at_ninety["warning"]
    assert at_ninety["quota"]["percent"] >= 90
    # A later write that still fits does not re-fire; the cap still refuses the
    # write that would exceed it, and that refusal is not a threshold warning.
    again = store.put("session", fact("Still under the cap after ninety"))
    assert "warning" not in again
    with pytest.raises(MemoryError):
        store.put("session", fact("word " * 250))


def test_a_scope_far_from_its_cap_never_warns(store):
    results = [store.put("global", fact(f"Global fact number {n}")) for n in range(3)]
    assert all("warning" not in result for result in results)
    assert all(result["quota"]["percent"] < 5 for result in results)


def test_forgetting_drops_the_state_so_a_recrossing_warns_again(store, monkeypatch):
    monkeypatch.setattr(memory, "OWNER_TOKENS", 1000)
    _fill(store, "session", 1000, 80)
    for row in store.inventory("session")["items"]:
        current = store.show("session", row["id"])
        store.put("session", {}, entity=row["id"], expected_version=current["version"], forget=True)
    assert store.inventory("session")["active_tokens"] == 0
    # Crossing back over 80% after a reset is a new crossing and warns again.
    monkeypatch.setattr(memory, "OWNER_TOKENS", 200)
    results = []
    for n in range(5):
        try:
            results.append(store.put("session", fact(f"Recrossing fact {n} " + "word " * 20)))
        except MemoryError:
            break
    assert sum(1 for result in results if result.get("warning")) == 1


def test_guidance_is_scope_aware(store, monkeypatch):
    monkeypatch.setattr(memory, "OWNER_TOKENS", 400)
    warnings = _fill(store, "session", 400, 86)
    assert warnings, "the session crossing should warn"
    session_warning = warnings[-1]["warning"]
    # A session->session move is refused, so session advice must not recommend it.
    assert "move work-state facts to session" not in session_warning
    assert "forget" in session_warning
    # A global crossing recommends the move to session, which IS available there.
    monkeypatch.setattr(memory, "GLOBAL_TOKENS", 400)
    global_warnings = _fill(store, "global", 400, 86)
    assert global_warnings, "the global crossing should warn"
    assert "--to session" in global_warnings[-1]["warning"]


def test_cli_edit_and_forget_receipts_carry_quota(store, monkeypatch):
    from local_operator.memory_cli import _quota_receipt, render_human, run

    item = store.put("session", fact())
    text, failed = run(
        f"edit session/{item['id']} --revision {item['version']} --body 'Corrected body text'",
        config=store.root.parent,
        identity=MemoryIdentity("session-a", "team-a", "agent-a"),
        session_dir=store.session_dir,
    )
    assert not failed, text
    assert "Quota:" in render_human(text, failed)
    current = store.show("session", item["id"])
    text, failed = run(
        f"forget session/{item['id']} --revision {current['version']} --yes",
        config=store.root.parent,
        identity=MemoryIdentity("session-a", "team-a", "agent-a"),
        session_dir=store.session_dir,
    )
    assert not failed, text
    assert "Quota:" in render_human(text, failed)
    # A warning, when the store returns one, is rendered by the same helper.
    assert any("crossed" in line for line in _quota_receipt({"warning": "crossed 80%"}))


def test_status_rejects_the_move_only_to_flag(store):
    text, failed = run(
        "status --to session",
        config=store.root.parent,
        identity=MemoryIdentity("session-a", "team-a", "agent-a"),
        session_dir=store.session_dir,
    )
    assert failed
    assert "--to applies" in text


def test_inventory_reports_a_quota_row_per_authorized_scope(store):
    store.put("global", fact())
    store.put("session", fact())
    quotas = store.inventory()["quotas"]
    by_scope = {row["scope"]: row for row in quotas}
    assert set(by_scope) >= {"global", "session", "team", "agent"}
    assert by_scope["global"]["cap_tokens"] == memory.GLOBAL_TOKENS
    assert by_scope["session"]["cap_tokens"] == memory.OWNER_TOKENS
    for row in quotas:
        assert row["headroom_tokens"] == row["cap_tokens"] - row["used_tokens"]
        assert 0 <= row["percent"] <= 100
    # A read path never carries a threshold warning.
    assert "warning" not in store.inventory()


def test_status_output_shows_headroom_percent_and_the_move_hint(store, monkeypatch):
    # Global is the scope where moving to session IS the available relief; the
    # session case (whose move is refused) is pinned separately below.
    monkeypatch.setattr(memory, "GLOBAL_TOKENS", 400)
    _fill(store, "global", 400, 80)
    text, failed = run(
        "status",
        config=store.root.parent,
        identity=MemoryIdentity("session-a", "team-a", "agent-a"),
        session_dir=store.session_dir,
    )
    assert not failed
    human = render_human(text, failed)
    assert "Quota:" in human and "left" in human and "%" in human
    assert "global" in human and "lop memory move" in human


def test_the_memory_tool_result_carries_quota_and_the_crossing_warning(tmp_path, monkeypatch):
    import asyncio

    import local_operator.memory as memory_module
    from local_operator.harness.types import ToolContext
    from local_operator.tools.memory_tools import execute_fact_memory

    monkeypatch.setattr(memory_module, "OWNER_TOKENS", 300)
    context = ToolContext(
        session_id="tiny",
        memory_config_dir=str(tmp_path / "cfg"),
        session_dir=str(tmp_path / "cfg" / "sessions" / "tiny"),
    )
    values = []
    for n in range(5):
        response = asyncio.run(
            execute_fact_memory(
                "s",
                {"op": "store", "text": f"Elara fact number {n} with several ordinary words"},
                context=context,
            )
        )
        if response.is_error:
            break
        values.append(json.loads(str(getattr(response.content[0], "text", ""))))
    assert values, "at least one stored fact"
    assert all("quota" in value for value in values)
    assert all("headroom_tokens" in value["quota"] for value in values)
    assert values[-1]["quota"]["percent"] > 0


def test_status_session_near_cap_never_prescribes_a_session_move(store, monkeypatch):
    """D1: at session scope the status page must not offer a refused move.

    The per-write warning has been scope-aware since round 1; `status` kept
    printing "move work-state facts to session scope" and a `--to session`
    command even when session was the full scope, which the CLI then refuses.
    """
    monkeypatch.setattr(memory, "OWNER_TOKENS", 400)
    _fill(store, "session", 400, 86)
    text, failed = run(
        "status",
        config=store.root.parent,
        identity=MemoryIdentity("session-a", "team-a", "agent-a"),
        session_dir=store.session_dir,
    )
    assert not failed, text
    human = render_human(text, failed)
    assert "Near cap: session" in human
    assert "--to session" not in human
    assert "forget" in human


def test_status_mixed_near_caps_advise_each_scope_separately(store, monkeypatch):
    monkeypatch.setattr(memory, "GLOBAL_TOKENS", 400)
    monkeypatch.setattr(memory, "OWNER_TOKENS", 400)
    _fill(store, "global", 400, 86)
    _fill(store, "session", 400, 86)
    text, failed = run(
        "status",
        config=store.root.parent,
        identity=MemoryIdentity("session-a", "team-a", "agent-a"),
        session_dir=store.session_dir,
    )
    assert not failed, text
    lines = render_human(text, failed).splitlines()
    global_at = next(i for i, line in enumerate(lines) if "Near cap: global" in line)
    session_at = next(i for i, line in enumerate(lines) if "Near cap: session" in line)
    # Each scope gets its own advice, and only the global block offers the move;
    # the session block's command is the forget ticket.
    global_block, session_block = lines[global_at:session_at], lines[session_at:]
    assert any("--to session" in line for line in global_block)
    assert not any("--to session" in line for line in session_block)
    assert any("forget" in line for line in session_block)
