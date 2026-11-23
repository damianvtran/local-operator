"""Compatibility contract for versioned inline UI snapshots."""

import json
from typing import Any

from local_operator.session.inline_ui_text import INLINE_UI_V1, project_inline_ui

_RICH = frozenset({INLINE_UI_V1})
_LEGACY = frozenset()
_WIDGET = {"type": "bar", "title": "Revenue", "data": [["January", 6]]}


def _fence(value: dict[str, Any]) -> str:
    return f"```lop-ui\n{json.dumps(value)}\n```\n"


def test_mixed_versions_replay_same_prose_without_leaking_payload() -> None:
    before = "Revenue 📈\n\n"
    source = before + _fence({**_WIDGET, "unused": {"url": "not part of the catalog"}}) + "End."
    # Stand in for a marked journal record; the JSON round trip deliberately
    # exercises persisted provenance separately from the reader capability.
    record = json.loads(json.dumps({"source": source, "source_version": 1}))
    legacy = project_inline_ui(**record, streaming=False, capabilities=_LEGACY)
    rich = project_inline_ui(**record, streaming=False, capabilities=_RICH)
    assert legacy == {"text": before + "\nEnd."}
    assert rich == {
        **legacy,
        "inline_ui": {"version": 1, "widgets": [{"offset": len(before) + 1, "payload": _WIDGET}]},
    }
    assert (
        project_inline_ui(**record, streaming=False, capabilities=frozenset({"inline-ui-v2"}))
        == legacy
    )


def test_stream_contains_no_partial_payload_and_widgets_keep_order() -> None:
    before = "Start.\n\n"
    first = _fence(_WIDGET)
    second_value = {**_WIDGET, "title": "Next", "data": [["February", 7]]}
    second = _fence(second_value)
    previous = before
    for n in range(len(first)):
        rich = project_inline_ui(
            before + first[:n], streaming=True, source_version=1, capabilities=_RICH
        )
        assert "inline_ui" not in rich
        assert rich["text"].startswith(previous)
        previous = rich["text"]
    result = project_inline_ui(
        before + first + "\nBetween.\n\n" + second + "\nEnd.",
        streaming=True,
        source_version=1,
        capabilities=_RICH,
    )
    assert "inline_ui" in result
    assert result["inline_ui"]["widgets"] == [
        {"offset": len(before), "payload": _WIDGET},
        {"offset": len(before + "\n\nBetween.\n\n"), "payload": second_value},
    ]


def test_unmarked_history_keeps_literal_markdown_and_invalid_widget_is_dropped() -> None:
    code = _fence(_WIDGET)
    assert project_inline_ui(code, streaming=False, source_version=None, capabilities=_RICH) == {
        "text": code
    }
    assert project_inline_ui(
        "Text.\n\n" + _fence({**_WIDGET, "data": [["January", None]]}) + "End.",
        streaming=False,
        source_version=1,
        capabilities=_RICH,
    ) == {"text": "Text.\n\n\nEnd."}


def test_grid_payload_drops_unvalidated_optional_fields() -> None:
    table = {"type": "table", "title": "Results", "columns": ["Name"], "rows": [["A"]]}
    source = _fence({**table, "rowLabels": {"style": "not part of the catalog"}})
    rich = project_inline_ui(source, streaming=False, source_version=1, capabilities=_RICH)
    assert "inline_ui" in rich
    assert rich["inline_ui"]["widgets"] == [{"offset": 0, "payload": table}]


def test_unsafe_json_numbers_use_exact_table_text_and_never_plot() -> None:
    def project(kind: str, fields: str):
        source = f'```lop-ui\n{{"type":"{kind}","title":"Exact",{fields}}}\n```'
        return project_inline_ui(source, streaming=False, source_version=1, capabilities=_RICH)

    counts = "[9007199254740991],[9007199254740993],[9007199254740995.0],[1e30]"
    table = project("table", f'"columns":["A"],"rows":[{counts}]')
    assert "inline_ui" in table
    wire = json.loads(json.dumps(table["inline_ui"]))
    assert wire["widgets"][0]["payload"]["rows"] == [
        [9007199254740991],
        ["9007199254740993"],
        ["9007199254740995"],
        ["1000000000000000000000000000000"],
    ]
    assert "inline_ui" in project("bar", '"data":[["A",9007199254740991]]')
    for unsafe in ("9007199254740992", "9007199254740993.0", "1e30", "1e-9999"):
        assert "inline_ui" not in project("bar", f'"data":[["A",{unsafe}]]')
