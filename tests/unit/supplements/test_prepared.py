"""Exact current-turn standalone docs share the component boundary, never fetch resources."""

from pathlib import Path

import pytest

from local_operator.harness.types import Message, TextContent, ToolCall
from local_operator.supplements.candidates import prefilter
from local_operator.supplements.prepared import promote
from local_operator.supplements.trigger import snapshot_messages

DOC = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Workflow</title>'
    '<style>button{color:var(--lo-text)}</style></head><body><button id="toggle">Expand</button>'
    '<p id="detail" hidden>Ready</p><script>document.getElementById("toggle").onclick=()=>{'
    'document.getElementById("detail").hidden=false}</script></body></html>'
)


def test_standard_document_preserves_inline_order_and_interaction() -> None:
    result, refusal = promote(DOC, "docs/workflow.html", ())
    assert result is not None and result.components and not refusal
    only = result.components[0]
    assert only.title == "Workflow" and only.source == "docs/workflow.html"
    assert only.blob.index("<style>") < only.blob.index("<button") < only.blob.index("<script>")
    assert "onclick" in only.blob


@pytest.mark.parametrize(
    "resource",
    [
        '<script src="sidecar.js"></script>',
        '<link rel="stylesheet" href="sidecar.css">',
        '<style>@import "sidecar.css";</style>',
        "<style>div{background:url(./art.png)}</style>",
        '<script>import("sidecar.js")</script>',
        "<script>import (name)</script>",
        '<base href="./">',
        '<meta http-equiv="refresh" content="0">',
        '<meta http-equiv="Content-Security-Policy" content="default-src *">',
    ],
)
def test_missing_dependencies_and_document_policies_are_refused(resource: str) -> None:
    result, refusal = promote(DOC.replace("</head>", resource + "</head>"), "preview.html", ())
    assert refusal and (result is None or not result.components)


def test_numeric_claims_still_need_existing_evidence() -> None:
    result, refusal = promote(DOC.replace("Ready", "Revenue 999"), "preview.html", ())
    assert result is not None and not result.components and refusal


def test_capture_is_exact_success_bounded_and_not_a_classifier_argument(tmp_path: Path) -> None:
    target = tmp_path / "preview.html"
    target.write_text(DOC)
    assistant = Message.assistant(
        "",
        tool_calls=[
            ToolCall(id="w", name="write", arguments={"path": str(target), "content": DOC})
        ],
    )
    success = Message(
        role="tool", tool_call_id="w", tool_name="write", content=[TextContent(text="written")]
    )
    items = snapshot_messages([assistant, success], prepared_writes={str(target): DOC})
    call = items[0].tool_calls[0]
    assert call.prepared_html == DOC and "content" not in call.args
    pre = prefilter(items, f"[Preview]({target})", cwd=str(tmp_path), home=str(tmp_path.parent))
    assert len(pre.prepared) == 1 and len(pre.candidates) == 1
    denied = prefilter(
        items, "", cwd=str(tmp_path), home=str(tmp_path.parent), deny_prefixes=("preview.html",)
    )
    assert not denied.prepared
    assert not prefilter(
        items, "", cwd=str(tmp_path), home=str(tmp_path.parent), want_graphics=False
    ).prepared
    failure = Message(
        role="tool",
        tool_call_id="w",
        tool_name="write",
        content=[TextContent(text="denied")],
        is_error=True,
    )
    assert not snapshot_messages([assistant, failure])[0].tool_calls[0].prepared_html
    assert (
        not snapshot_messages([assistant, success], max_prepared=0)[0].tool_calls[0].prepared_html
    )
