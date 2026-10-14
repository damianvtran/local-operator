"""Direct convergence defects: alias identity, callback claims, and metadata privacy."""

from pathlib import Path

import pytest

from local_operator.harness.types import Message, ToolCall, ToolResult
from local_operator.session.attachments import AttachmentStore
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.supplements.candidates import prefilter
from local_operator.supplements.decision import Decision
from local_operator.supplements.prepared import promote
from local_operator.supplements.trigger import snapshot_messages
from tests.unit.supplements.test_prepared import DOC
from tests.unit.supplements.test_runner_generator import _make_session


@pytest.mark.parametrize("mutation", ["edit", "uncaptured_write", "captured_write"])
def test_alias_mutation_invalidates_continuation_snapshot(tmp_path: Path, mutation: str) -> None:
    target = tmp_path / "document.html"
    target.write_text(DOC)
    alias = tmp_path / "alias.html"
    alias.symlink_to(target)
    session = _make_session(tmp_path / "session", tmp_path)
    write = ToolCall(id="w", name="write", arguments={"path": str(target), "content": DOC})
    session._capture_prepared_write(
        write, ToolResult(tool_call_id="w", tool_name="write", content=[])
    )
    first = snapshot_messages(
        [
            Message.assistant(tool_calls=[write]),
            Message(role="tool", tool_call_id="w", tool_name="write"),
        ],
        prepared_writes=session._prepared_writes,
    )
    revised = DOC.replace("Ready", "Revised")
    content = (
        revised
        if mutation == "captured_write"
        else ("x" * (48 * 1024 + 1) if mutation == "uncaptured_write" else revised)
    )
    alias.write_text(content)
    name = "edit" if mutation == "edit" else "write"
    call = ToolCall(id="m", name=name, arguments={"path": str(alias), "content": content})
    session._capture_prepared_write(call, ToolResult(tool_call_id="m", tool_name=name, content=[]))
    later = snapshot_messages(
        [
            Message.assistant(tool_calls=[call]),
            Message(role="tool", tool_call_id="m", tool_name=name),
        ],
        prepared_writes=session._prepared_writes,
    )
    pre = prefilter(first + later, "Done.", cwd=str(tmp_path), home=str(tmp_path.parent))
    if mutation == "captured_write":
        assert len(pre.prepared) == 1 and pre.prepared[0][1] == revised
        assert pre.prepared[0][0].path == "alias.html"
    else:
        assert not pre.prepared


def test_sensitive_resolved_alias_is_still_fenced(tmp_path: Path) -> None:
    target = tmp_path / ".env.html"
    target.write_text(DOC)
    alias = tmp_path / "preview.html"
    alias.symlink_to(target)
    session = _make_session(tmp_path / "session", tmp_path)
    call = ToolCall(id="w", name="write", arguments={"path": str(alias), "content": DOC})
    session._capture_prepared_write(
        call, ToolResult(tool_call_id="w", tool_name="write", content=[])
    )
    items = snapshot_messages(
        [
            Message.assistant(tool_calls=[call]),
            Message(role="tool", tool_call_id="w", tool_name="write"),
        ],
        prepared_writes=session._prepared_writes,
    )
    assert not prefilter(items, "Done.", cwd=str(tmp_path), home=str(tmp_path.parent)).prepared


@pytest.mark.parametrize("timer", ["setTimeout", "setInterval"])
@pytest.mark.parametrize(
    "callback",
    [
        '()=>{document.getElementById("detail").textContent="Revenue 999"}',
        (
            "()=>{\n"
            ' const object={nested:{label:"Revenue 999"}};\n'
            ' document.getElementById("detail").textContent=object.nested.label;\n'
            "}"
        ),
        '()=>{setTimeout(()=>{document.getElementById("detail").textContent="Revenue 999"},400)}',
    ],
)
def test_timer_wrapper_cannot_hide_fabricated_callback_claim(timer: str, callback: str) -> None:
    result, refusal = promote(
        DOC.replace("</script>", f";{timer}({callback},300)</script>"), "preview.html", ()
    )
    assert result is not None and not result.components and refusal


@pytest.mark.parametrize(
    "script",
    [
        "setTimeout(()=>{},300)",
        'setInterval(()=>{document.getElementById("detail").hidden=false},300)',
        "setTimeout(()=>{setTimeout(()=>{},400)},300)",
    ],
)
def test_proven_delay_literal_is_not_a_claim(script: str) -> None:
    result, refusal = promote(DOC.replace("</script>", f";{script}</script>"), "preview.html", ())
    assert result is not None and result.components and not refusal


@pytest.mark.asyncio
async def test_metadata_scrub_does_not_reintroduce_private_authored_bytes(tmp_path: Path) -> None:
    import asyncio

    shaped = "AKIA" + "IOSFODNN7EXAMPLE"
    known = "fictional-registered-private-value"
    doc = DOC.replace("<title>Workflow</title>", f"<title>{shaped}</title>").replace(
        "Ready", shaped
    )
    promoted, refusal = promote(doc, known, ())
    assert promoted is not None and promoted.components and not refusal
    session = _make_session(tmp_path / "session", tmp_path)
    # Same host callback the journal uses; no real secret-store write or credential access.
    session._redact_tool_result_text = lambda text: text.replace(known, "[masked]")
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), install_gates=False, cwd=str(tmp_path)
    )
    events = []
    session.subscribe(events.append)
    try:
        await handle._supplements._write("anchor", Decision(), components=promoted.components)
        rows = [
            entry.payload["details"]
            for entry in session.transcript.entries()
            if entry.type == "custom" and entry.payload.get("custom_type") == "supplement_v1"
        ]
        assert rows and rows[0]["components"][0]["title"] == "Supporting visual"
        assert rows[0]["components"][0]["source"] == "Local turn evidence"
        assert shaped not in str(rows) and known not in str(rows)
        assert shaped not in str([event.model_dump() for event in events])
        assert known not in str([event.model_dump() for event in events])
        blob = AttachmentStore().get_bytes(rows[0]["components"][0]["attachment"])
        assert blob is not None and shaped.encode() in blob[0]
    finally:
        await handle.dispose()
