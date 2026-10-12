"""Round-one defects: byte fidelity, dependency honesty, and supported numeric claims."""

from pathlib import Path

import pytest

from local_operator.harness.types import Message, TextContent, ToolCall, ToolResult
from local_operator.supplements.candidates import prefilter
from local_operator.supplements.prepared import promote
from local_operator.supplements.trigger import snapshot_messages
from tests.unit.supplements.test_generator import DATASETS
from tests.unit.supplements.test_prepared import DOC
from tests.unit.supplements.test_runner_generator import _make_session


def test_raw_script_scaffold_literals_are_never_removed() -> None:
    script = (
        "<script>const sample='<meta charset=\"utf-8\">';"
        'const title="<title>Other</title>";</script>'
    )
    result, refusal = promote(DOC.replace("</head>", script + "</head>"), "preview.html", ())
    assert result is not None and result.components and not refusal
    assert script in result.components[0].blob


@pytest.mark.parametrize(
    "resource",
    [
        "<img src=sidecar.png>",
        "<a href=other.html>Open</a>",
        '<img src="sidecar&#46;png">',
        "<img srcset=sidecar.png>",
        '<img srcset="sidecar.png">',
    ],
)
def test_all_real_attribute_forms_refuse_dependencies(resource: str) -> None:
    result, refusal = promote(DOC.replace("Ready", "Ready" + resource), "preview.html", ())
    assert refusal == "prepared:unsupported"
    assert result is None or not result.components


def test_local_svg_fragment_is_self_contained() -> None:
    svg = (
        '<svg viewBox="0 0 20 20"><defs><linearGradient id="g">'
        '<stop offset="0%" stop-color="red"/></linearGradient></defs>'
        '<rect width="20" height="20" fill="url(#g)"/></svg>'
    )
    result, refusal = promote(DOC.replace("Ready", "Ready" + svg), "preview.html", ())
    assert result is not None and result.components and not refusal


@pytest.mark.parametrize(
    "claim,accepted",
    [
        ("120 ms", True),
        ("9.5 ms", False),
        ("Revenue 1", False),
        ("Step 1", True),
        ("Revenue 999", False),
    ],
)
def test_authored_claims_use_turn_evidence_not_proprietary_markup(
    claim: str, accepted: bool
) -> None:
    result, refusal = promote(DOC.replace("Ready", claim), "preview.html", DATASETS)
    assert bool(result is not None and result.components) == accepted
    assert bool(refusal) != accepted


def test_harmless_timer_does_not_disable_data_truth_gate() -> None:
    result, refusal = promote(
        DOC.replace("</script>", ";setTimeout(()=>{},300)</script>"), "preview.html", ()
    )
    assert result is not None and result.components and not refusal
    bad, refusal = promote(
        DOC.replace(
            "</script>", ';document.getElementById("detail").textContent="Revenue 999"</script>'
        ),
        "preview.html",
        (),
    )
    assert bad is not None and not bad.components and refusal


def test_exact_private_capture_does_not_read_scrubbed_history(tmp_path: Path) -> None:
    target = tmp_path / "preview.html"
    original = DOC.replace("Ready", "Private exact bytes")
    target.write_text(original)
    from local_operator.harness.loop import _scrub_history_arguments

    executed = ToolCall(id="w", name="write", arguments={"path": str(target), "content": original})
    session = _make_session(tmp_path / "private-session", tmp_path)
    session._capture_prepared_write(
        executed, ToolResult(tool_call_id="w", tool_name="write", content=[])
    )
    scrubbed_message = _scrub_history_arguments(
        Message.assistant(tool_calls=[executed]),
        lambda text: text.replace("Private exact bytes", "[masked]"),
    )
    scrubbed = scrubbed_message.tool_calls[0].arguments["content"]
    assert scrubbed != original
    message = Message.assistant(
        tool_calls=[
            ToolCall(id="w", name="write", arguments={"path": str(target), "content": scrubbed})
        ]
    )
    end = Message(
        role="tool", tool_call_id="w", tool_name="write", content=[TextContent(text="written")]
    )
    assert not snapshot_messages([message, end])[0].tool_calls[0].prepared_html
    items = snapshot_messages([message, end], prepared_writes=session._prepared_writes)
    assert items[0].tool_calls[0].prepared_html == original
    assert "content" not in items[0].tool_calls[0].args
    # Later successful edits cannot publish an earlier version, including continuation items.
    later = snapshot_messages(
        [
            Message.assistant(
                tool_calls=[ToolCall(id="e", name="edit", arguments={"path": str(target)})]
            ),
            Message(role="tool", tool_call_id="e", tool_name="edit"),
        ]
    )
    assert not prefilter(items + later, "", cwd=str(tmp_path), home=str(tmp_path.parent)).prepared


def test_private_success_hook_invalidates_mutated_paths(tmp_path: Path) -> None:
    session = _make_session(tmp_path / "session", tmp_path)
    target = tmp_path / "preview.html"
    call = ToolCall(id="w", name="write", arguments={"path": str(target), "content": DOC})
    result = ToolResult(tool_call_id="w", tool_name="write", content=[])
    session._capture_prepared_write(call, result)
    assert session._prepared_writes[str(target)] == DOC
    session._capture_prepared_write(
        ToolCall(id="e", name="edit", arguments={"path": str(target)}), result
    )
    assert not session._prepared_writes
    session._capture_prepared_write(call, result)
    session._capture_prepared_write(
        ToolCall(id="b", name="bash", arguments={"command": "mutate"}), result
    )
    assert not session._prepared_writes
