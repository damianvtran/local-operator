"""Compaction memory follows committed history, not speculative model calls."""

import asyncio

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.memory import MemoryIdentity, MemoryStore
from local_operator.memory_capture import (
    extract,
    finish,
    hint,
    persist,
    snapshot,
    watermark,
)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        tmp_path / "cfg",
        MemoryIdentity("capture-session", "t", "a"),
        tmp_path / "sessions" / "capture-session",
        origin="capture-device",
    )


def test_extractor_returns_data_only_and_refuses_sensitive_locators(tmp_path):
    rows = extract(
        "See https://example.invalid/thread/123 and /src/task.py and ~/.ssh/id_rsa; "
        "secret PASSPORT_NO"
    )
    assert any("https://example.invalid" in row["body"] for row in rows)
    assert any("PASSPORT_NO" in row["body"] for row in rows)
    assert not any("id_rsa" in row["body"] for row in rows)
    assert list(tmp_path.iterdir()) == []
    assert len(extract(" ".join("/src/item%d.py" % n for n in range(100)))) <= 40


def test_bounded_snapshot_ignores_images():
    messages = [
        Message(role="user", content=[TextContent(text="/src/needle.py " + "x" * 1000000)])
    ] * 100
    bounded = snapshot(messages)
    assert len(bounded.encode()) <= 2000000
    assert "/src/needle.py" in bounded


def test_capture_idempotence_watermark_and_semantic_reuse(store):
    rows = extract("https://example.invalid/file") + extract(
        "",
        "# Goal\n- Complete release\n# Key Decisions\n- Keep stable prefix\n"
        "# Other\n- not semantic",
    )
    persist(store, "marker-1", rows, "ok")
    count = store.inventory()["ops"]
    persist(store, "marker-1", rows, "ok")
    assert store.inventory()["ops"] == count
    assert watermark(store, "marker-1") == ("ok", 3)
    assert watermark(store, "uncommitted") is None
    assert hint(store, "uncommitted") == ""
    assert "3 candidates" in hint(store, "marker-1")
    assert store.search("stable prefix")[0]["status"] == "candidate"


@pytest.mark.asyncio
async def test_capture_late_completion_never_blocks_commit_or_wakes(store):
    gate = asyncio.Event()

    async def capture():
        await gate.wait()
        return extract("/src/late.py")

    task = asyncio.create_task(capture())
    writer = asyncio.create_task(finish(store, "committed-marker", task, ""))
    await asyncio.sleep(0)
    assert not writer.done()
    assert not store.path.exists()  # speculative extractor and waiting writer wrote nothing
    gate.set()
    await writer
    assert watermark(store, "committed-marker") == ("ok", 1)
    assert store.search("late")


@pytest.mark.asyncio
async def test_failure_is_terminal_not_false_capture_success(store):
    async def broken():
        raise RuntimeError("synthetic extractor failure")

    await finish(store, "failed-marker", asyncio.create_task(broken()), "")
    assert watermark(store, "failed-marker") == ("failed", 0)
    assert "did not complete" in hint(store, "failed-marker")
    assert not store.search("synthetic")


@pytest.mark.asyncio
async def test_real_committed_compaction_finishes_capture_while_idle_without_waking(
    tmp_path, monkeypatch
):
    import threading

    import local_operator.memory_capture as capture
    from local_operator.compaction.api import CompactionSettings
    from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(
        '{"values":{"memory":{"enabled":true,"capture":true,"suggest":true}}}'
    )
    gate = threading.Event()
    entered = threading.Event()
    original = capture.extract_messages

    def gated(messages):
        entered.set()
        assert gate.wait(5), "capture setup gate was never released"
        return original(messages)

    monkeypatch.setattr(capture, "extract_messages", gated)
    requests = []

    def stream(request, signal):
        requests.append(request)

        async def events():
            yield StreamTextDelta(delta="Recorded reply")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    transcript = Transcript(root / "sessions" / "idle-native")
    session = Session(
        model=ModelSpec(
            provider="test", model_id="deterministic", context_window=100000, supports_images=False
        ),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=40, strategy="context-full"),
    )
    session._config_dir = root
    events = []
    session.subscribe(events.append)
    try:
        for n in range(3):
            await session.prompt(f"Turn {n} docs/idle.md " + "detail " * 35)
        outcome = await session.compact_now()
        assert outcome.ran
        marker = transcript.latest_entry("compaction")
        assert await asyncio.to_thread(entered.wait, 2)
        memory = MemoryStore(root, MemoryIdentity("idle-native", "", "main"), transcript.directory)
        assert watermark(memory, marker.id) is None
        before = (len(requests), sum(event.type == "agent_start" for event in events))
        gate.set()
        pending = list(session._background_tasks)
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), 3)
        assert watermark(memory, marker.id)[0] == "ok"
        assert before == (len(requests), sum(event.type == "agent_start" for event in events))
        assert memory.search("idle")
        await session.prompt("Deliberate wake instrument control")
        with pytest.raises(AssertionError):
            assert before == (len(requests), sum(event.type == "agent_start" for event in events))
    finally:
        gate.set()
        await session.dispose()


@pytest.mark.asyncio
async def test_restart_recovers_committed_marker_without_waking(tmp_path):
    import local_operator.memory_capture as capture
    from local_operator.harness.types import ModelSpec, StreamEndEvent
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(
        '{"values":{"memory":{"enabled":true,"capture":true,"suggest":true}}}'
    )
    transcript = Transcript(root / "sessions" / "restart-session")
    await transcript.append_message(Message.user("Old source docs/recovered.md"))
    kept = await transcript.append_message(Message.user("Kept turn"))
    marker = await transcript.append_compaction("# Goal\n- Recover committed goal", kept.id, 100)
    # No capture writes: this is the exact crash window after marker fsync.
    requests = []

    def stream(request, signal):
        requests.append(request)

        async def events():
            yield StreamEndEvent(stop_reason="stop")

        return events()

    reopened = Session(
        model=ModelSpec(provider="test", model_id="deterministic", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(transcript.directory),
        system_blocks_provider=lambda: ["stable"],
    )
    reopened._config_dir = root
    assert not (root / "memory").exists()
    assert await capture.session_tail(reopened) == ""
    pending = list(reopened._background_tasks)
    await asyncio.wait_for(asyncio.gather(*pending), 3)
    memory = MemoryStore(root, MemoryIdentity("restart-session", "", "main"), transcript.directory)
    assert watermark(memory, marker.id)[0] == "ok"
    assert memory.search("recovered")
    assert requests == []
    assert "memory_hint" in await capture.session_tail(reopened)
    await reopened.dispose()


def test_suggestion_guard_suppresses_generic_one_word_and_filters_scope(store):
    store.put(
        "session",
        {
            "title": "Mercury release",
            "body": "Mercury deployment decisions",
            "tags": ["mercury", "deployment"],
        },
    )
    assert store.search("Mercury deployment", suggest=True, tokens=180)
    assert not store.search("how do I create a new branch", suggest=True, tokens=180)
    foreign = MemoryStore(store.root.parent, MemoryIdentity("other", "other", "other"))
    assert not foreign.search("Mercury deployment", suggest=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy,images",
    [("context-full", False), ("snapcompact", True)],
    ids=["context-full", "snapcompact"],
)
async def test_three_chained_real_compactions_capture_store_restart_retrieve(
    tmp_path, strategy, images
):
    """store -> compact x3 -> restart -> retrieve over a REAL Session, both strategies.

    snapcompact returns no model text, so only locators may be captured there; the
    explicit store must survive either way, and no chained pass may wake the owner.
    """
    from local_operator.compaction.api import CompactionSettings
    from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(
        '{"values":{"memory":{"enabled":true,"capture":true,"suggest":true}}}'
    )
    requests = []

    def stream(request, signal):
        requests.append(request)

        async def events():
            yield StreamTextDelta(delta="# Goal\n- Ship release\n# Key Decisions\n- Keep prefix")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    directory = root / "sessions" / f"chain-{strategy}"
    transcript = Transcript(directory)
    model = ModelSpec(
        provider="test", model_id="deterministic", context_window=100000, supports_images=images
    )
    session = Session(
        model=model,
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id=f"chain-{strategy}",
        compaction_settings=CompactionSettings(keep_recent_tokens=40, strategy=strategy),
    )
    session._config_dir = root
    memory = MemoryStore(root, MemoryIdentity(f"chain-{strategy}", "", "main"), directory)
    explicit = memory.put("session", {"title": "Release decision", "body": "Decision: ship Friday"})
    markers = []
    try:
        for n in range(3):
            for m in range(3):
                await session.prompt(f"Round {n}.{m} see docs/chain{n}.md " + "detail " * 35)
            outcome = await session.compact_now()
            assert outcome.ran, (strategy, n)
            entry = transcript.latest_entry("compaction")
            markers.append(entry.id)
            # Control that the arms really differ: snapcompact's summary is the
            # deterministic archive (no model text); context-full is model-written.
            archived = str(entry.payload.get("summary", "")).startswith("Resume prior conversation")
            assert archived == (strategy == "snapcompact"), entry.payload.get("summary", "")[:80]
            for task in list(session._background_tasks):
                await asyncio.wait_for(task, 5)
        assert len(set(markers)) == 3
        assert all(watermark(memory, marker) for marker in markers)
        before = len(requests)
    finally:
        await session.dispose()
    # Restart: a fresh store handle over the same disk state still retrieves.
    reopened = MemoryStore(root, MemoryIdentity(f"chain-{strategy}", "", "main"), directory)
    assert reopened.show("session", explicit["id"])["fact"]["body"] == "Decision: ship Friday"
    assert reopened.search("chain0")
    assert before >= 9
    print("CHAIN", strategy, "markers", len(markers), "captured", len(reopened.search("docs")))


@pytest.mark.asyncio
async def test_cancelled_capture_is_retried_at_the_next_boundary_without_a_restart(
    tmp_path, monkeypatch
):
    """QA Q5: a capture cancelled in-process left no watermark until the process restarted.

    The committed marker is the authority, so the next prompt boundary re-runs the same
    recovery a restart would. No wake: the retry rides an ordinary user turn.
    """
    import local_operator.memory_capture as capture
    from local_operator.compaction.api import CompactionSettings
    from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text('{"values":{"memory":{"enabled":true,"capture":true}}}')

    def stream(request, signal):
        async def events():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    directory = root / "sessions" / "cancel-s"
    transcript = Transcript(directory)
    session = Session(
        model=ModelSpec(provider="test", model_id="d", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id="cancel-s",
        compaction_settings=CompactionSettings(keep_recent_tokens=40, strategy="context-full"),
    )
    session._config_dir = root
    real_finish = capture.finish
    calls = {"n": 0}

    async def cancelled_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise asyncio.CancelledError
        return await real_finish(*args, **kwargs)

    monkeypatch.setattr(capture, "finish", cancelled_once)
    store = MemoryStore(root, MemoryIdentity("cancel-s", "", "main"), directory)
    try:
        for m in range(4):
            await session.prompt(f"Round {m} see docs/cancel.md " + "detail " * 35)
        assert (await session.compact_now()).ran
        marker = transcript.latest_entry("compaction")
        for task in list(session._background_tasks):
            try:
                await asyncio.wait_for(task, 5)
            except asyncio.CancelledError:
                pass
        assert calls["n"] == 1 and watermark(store, marker.id) is None, "cancelled: no watermark"
        await session.prompt("Ordinary next turn")  # the next boundary re-arms recovery
        for task in list(session._background_tasks):
            await asyncio.wait_for(task, 5)
        assert calls["n"] == 2
        assert watermark(store, marker.id) is not None, "retried without a process restart"
    finally:
        await session.dispose()
