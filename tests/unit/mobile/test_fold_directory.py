"""The archive fold needs indexed proof, not a whole-journal Transcript replay."""

from pathlib import Path

import pytest

from local_operator.harness.types import AgentMessage, Message
from local_operator.mobile.durable import _fold


def test_fold_uses_directory_for_discovery_and_strong_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_operator.harness.peer_cycle_display as peer_display
    import local_operator.mobile.projection as projection
    import local_operator.session.transcript as transcript

    directory = tmp_path / "sessions" / "synthetic-session"
    history: list[AgentMessage] = [Message.user("hello", id="root")]
    records = {"records": [], "remaining": [], "unavailable": [], "bytes": 0}
    observed = {}

    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("archive fold must not replay the whole journal")

    def discover(source, messages, session_id):
        assert source == directory
        assert isinstance(source, Path)
        assert messages == history
        assert session_id == directory.name
        return records

    def fold(messages, **kwargs):
        observed.update(kwargs)
        return ["projected"]

    monkeypatch.setattr(transcript, "Transcript", forbidden_constructor)
    monkeypatch.setattr(peer_display, "discover_records", discover)
    monkeypatch.setattr(projection, "fold_messages_to_entries", fold)
    assert _fold(history, directory) == ["projected"]
    # None silently selects weaker record validation, so path forwarding is a
    # correctness constraint as well as the constructor-avoidance optimization.
    assert observed["proof_transcript"] == directory
    assert observed["peer_cycle_display_records"] is records
    assert observed["session_id"] == directory.name


def _legacy_fold(history, directory=None):
    """Frozen pre-change control; only the unnecessary constructor differs."""
    from local_operator.harness.peer_cycle_display import discover_records
    from local_operator.mobile.projection import fold_messages_to_entries
    from local_operator.session.transcript import Transcript

    if directory is None:
        return fold_messages_to_entries(history)
    transcript = Transcript(directory, read_only=True)
    return fold_messages_to_entries(
        history,
        session_id=directory.name,
        peer_cycle_display_records=discover_records(transcript, history[-4096:], directory.name),
        proof_transcript=transcript,
    )


def _fixture(config, *, marker_first=False, certificate="valid"):
    import asyncio

    from local_operator.harness.peer_cycle_display import persist_record
    from local_operator.harness.types import TextContent, ToolCall
    from local_operator.session.transcript import Transcript
    from tests.unit.session.test_peer_cycle_display_archive import make_cycle

    directory = config / "sessions" / "sess"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    _, cycle, record = make_cycle(wrapped_send=True)

    async def write():
        await transcript.append_message(Message.user("opening", id="opening"))
        await transcript.append_messages(cycle[:-1])
        if certificate != "unavailable":
            assert await persist_record(transcript, record)
        closing = cycle[-1]
        if certificate == "invalid":
            # The cycle's closing entry is a tool result (a Message); narrow so
            # the mutation is typed against the class that owns ``content``.
            assert isinstance(closing, Message)
            closing.content = [TextContent(text="changed result invalidates certificate")]
        await transcript.append_message(closing)
        await transcript.append_message(
            Message.assistant(
                "",
                id="empty-assistant",
                tool_calls=[
                    ToolCall(id="secret-call", name="bash", arguments={"command": "synthetic"})
                ],
            )
        )
        await transcript.append_message(Message.assistant("ASSISTANT_SENTINEL", id="secret-answer"))
        await transcript.append_message(
            Message(
                role="tool",
                id="secret-result",
                tool_call_id="secret-call",
                tool_name="bash",
                content=[TextContent(text="PAYLOAD_SENTINEL")],
            )
        )
        # Padding makes the avoided full replay observable without live data.
        for i in range(240):
            await transcript.append_message(Message.user("x" * 4096, id=f"u{i}"))
        if marker_first:
            # Deliberate historical edge, NOT the production writer order.
            await transcript.append_compaction("summary", "kept", 1000)
        await transcript.append_message(Message.user("kept", id="kept"))
        await transcript.append_message(Message.assistant("tail", id="tail"))
        if not marker_first:
            # session.py computes kept[0].id from already journaled history.
            await transcript.append_compaction("summary", "kept", 1000)
        await transcript.append_prune("secret-result", "[pruned]")
        await transcript.append_prune("secret-answer", "")

    asyncio.run(write())
    transcript.close()
    return directory, cycle, record


class _ReadMeter:
    """Count bytes returned by journal handles, not inferred file sizes."""

    def __init__(self, monkeypatch, directory):
        self.bytes = 0
        self.calls = 0
        original = Path.open
        meter = self

        class Handle:
            def __init__(self, handle):
                self.handle = handle

            def __getattr__(self, key):
                return getattr(self.handle, key)

            def __enter__(self):
                self.handle.__enter__()
                return self

            def __exit__(self, *args):
                return self.handle.__exit__(*args)

            def __iter__(self):
                return self

            def __next__(self):
                value = next(self.handle)
                meter.record(value)
                return value

            def read(self, *args):
                value = self.handle.read(*args)
                meter.record(value)
                return value

            def readline(self, *args):
                value = self.handle.readline(*args)
                meter.record(value)
                return value

        def opened(path, *args, **kwargs):
            handle = original(path, *args, **kwargs)
            mode = args[0] if args else kwargs.get("mode", "r")
            if path.parent == directory and path.name == "transcript.jsonl" and "r" in mode:
                return Handle(handle)
            return handle

        monkeypatch.setattr(Path, "open", opened)

    def record(self, value):
        self.calls += 1
        self.bytes += len(value.encode("utf-8") if isinstance(value, str) else value)


@pytest.mark.parametrize("certificate", ["valid", "invalid", "unavailable"])
@pytest.mark.parametrize("compressed", [False, True])
def test_directory_fold_preserves_indexed_certificate_and_compressed_source(
    tmp_path, monkeypatch, certificate, compressed
):
    import json

    from local_operator.session import compression
    from local_operator.session.transcript import Transcript

    directory, history, _ = _fixture(tmp_path, certificate=certificate)
    if compressed:
        assert compression.compress_session(directory, codec="gzip").status == "compressed"
    files_before = {p.name: p.read_bytes() for p in directory.iterdir() if "transcript" in p.name}
    old = _legacy_fold(history, directory)
    new = _fold(history, directory)
    assert [row.to_json() for row in new] == [row.to_json() for row in old]
    certified = any("peer_cycle" in row.details for row in new)
    # The existing refresh_index contract only indexes plain journals. Both
    # controls therefore withhold compressed-source certificates; do not turn
    # parity into a claim that this constructor-only change adds that support.
    assert certified == (certificate == "valid" and not compressed)
    if compressed:
        from local_operator.harness.peer_cycle_display import discover_records

        discovered = discover_records(directory, history, directory.name)
        assert discovered is not None and discovered["unavailable"]
        assert not discovered["records"]
    assert "SYNTHETIC_SECRET" not in json.dumps([row.to_json() for row in new])
    files_after = {p.name: p.read_bytes() for p in directory.iterdir() if "transcript" in p.name}
    assert files_after == files_before
    if compressed:
        assert not compression.plain_path(directory).exists()
    # A separately reopened read-only control confirms retained canonical data,
    # rather than treating projection equality as evidence of physical retention.
    control = Transcript(directory, read_only=True)
    assert any(message.id == "kept" for message in control.build_llm_history())


@pytest.mark.parametrize(
    "marker_first", [False, True], ids=["production-order", "marker-first-edge"]
)
@pytest.mark.parametrize("certificate", ["valid", "invalid", "unavailable"])
def test_authenticated_archive_route_matches_control_without_full_replay(
    tmp_path, monkeypatch, marker_first, certificate, capsys
):
    import json
    import time

    from starlette.testclient import TestClient

    import local_operator.mobile.durable as durable
    from local_operator.mobile.daemon import (
        MobileDaemon,
        _durable_projection,
        build_app,
    )

    config = tmp_path / "config"
    directory, _, _ = _fixture(config, marker_first=marker_first, certificate=certificate)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    client = TestClient(
        build_app(MobileDaemon(port=0, password="synthetic-password")), follow_redirects=False
    )
    route = "/api/sessions/sess/history"
    assert client.get(route).status_code == 401
    assert client.post("/login", data={"password": "synthetic-password"}).status_code == 303
    assert client.get("/api/sessions/unknown/history").status_code == 404
    assert client.get("/api/sessions/%2E%2E%2Foutside/history").status_code == 404

    measurements = {}
    responses = {}
    # One fixed cache and index prewarm, explicitly outside warm request costs.
    with monkeypatch.context() as cold_patch:
        cold_meter = _ReadMeter(cold_patch, directory)
        assert _durable_projection("sess") is not None
        measurements["cold_new"] = {"bytes": cold_meter.bytes, "calls": cold_meter.calls}
    for name, fold in (("old", _legacy_fold), ("new", _fold)):
        monkeypatch.setattr(durable, "_fold", fold)
        with monkeypatch.context() as patch:
            meter = _ReadMeter(patch, directory)
            start = time.perf_counter()
            response = client.get(route, params={"before": "kept", "limit": 200})
            elapsed = time.perf_counter() - start
            assert response.status_code == 200
            responses[name] = response.json()
            measurements[name] = {
                "bytes": meter.bytes,
                "calls": meter.calls,
                "seconds": elapsed,
            }
        # Reach the oldest payload as well, not just the padded page.
        response = client.get(route, params={"before": "u0", "limit": 200})
        assert response.status_code == 200
        assert "PAYLOAD_SENTINEL" not in response.text
        assert "ASSISTANT_SENTINEL" not in response.text
        assert "[pruned]" in response.text
        oldest = response.json()
        assert not any(row["id"] == "secret-answer" for row in oldest["entries"])
        certified = any("peer_cycle" in row.get("details", {}) for row in oldest["entries"])
        assert certified == (certificate == "valid")
        responses[name + "_oldest"] = oldest
    assert responses["new"] == responses["old"]
    assert responses["new_oldest"] == responses["old_oldest"]
    assert measurements["old"]["bytes"] > measurements["new"]["bytes"]
    assert measurements["old"]["calls"] > measurements["new"]["calls"]
    with capsys.disabled():
        print(
            "ARCHIVE_ROUTE_MEASUREMENT "
            + json.dumps({"marker_first": marker_first, "certificate": certificate, **measurements})
        )
