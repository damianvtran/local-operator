"""Cold/SSE carriage is per-viewer negotiated metadata, not arbitrary custom content."""

from types import SimpleNamespace
from typing import cast

import pytest

from local_operator.server.utils.desktop_sessions import DesktopSessionBridge
from local_operator.session import open_frame as of
from local_operator.session.attached import AttachedSession
from tests.unit.session.test_open_frame import custom, message


def supplement():
    return custom(
        "s",
        "supplement_v1",
        {
            "anchor": "answer",
            "job": "job",
            "version": 1,
            "state": "done",
            "at": 1,
            "files": [
                {
                    "path": "preview.html",
                    "name": "preview.html",
                    "kind": "html",
                    "size_bytes": 42,
                    "mtime": 1,
                    "why": "written",
                }
            ],
            "components": [
                {
                    "attachment": "a" * 32,
                    "title": "Preview",
                    "source": "preview.html",
                    "mime": "text/html",
                    "height_hint": 320,
                }
            ],
        },
    )


def test_only_explicitly_negotiated_known_custom_rows_paint() -> None:
    entry = supplement()
    assert of.strip_entry(entry) is None
    assert of.strip_entry(entry, supplements=False) is None
    assert of.strip_entry(entry, supplements=True) == entry
    assert (
        of.strip_entry(
            custom("unknown", "arbitrary_private_type", {"html": "private"}), supplements=True
        )
        is None
    )


def test_markup_unknown_fields_and_errand_prompt_never_cross_paint_ipc() -> None:
    entry = supplement()
    entry["html"] = "private outer markup"
    entry["payload"]["html"] = "private envelope markup"
    details = entry["payload"]["details"]
    details["body"] = "private body"
    details["instruction"] = "private errand prompt"
    details["decision"] = {"unknown": "private analytics"}
    details["components"][0]["html"] = "private component markup"
    details["components"][0]["data"] = {"private": "body"}
    painted = of.strip_entry(entry, supplements=True)
    assert painted is not None and "private" not in str(painted)
    assert painted["payload"]["details"]["components"][0]["attachment"] == "a" * 32


def test_negotiated_rows_participate_in_cut_and_byte_budget() -> None:
    rows = [message("user", "user"), message("answer", "assistant"), supplement()]
    old = of.build_frame(rows)
    new = of.build_frame(rows, supplements=True)
    assert len(old.entries) == 2 and len(new.entries) == 3
    assert of.served_bytes(new.entries) > of.served_bytes(old.entries)
    kept, reached = of.align_page(rows, 1, supplements=True, extra_rows=0)
    assert len(kept) == 1 and not reached and kept[0]["id"] == "s"
    assert of.paintable(rows, supplements=True)[-1]["id"] == "s"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "viewer,owner,expected", [(False, True, False), (True, False, False), (True, True, True)]
)
async def test_history_requires_this_viewer_and_negotiated_owner(
    viewer: bool, owner: bool, expected: bool
) -> None:
    bridge = object.__new__(DesktopSessionBridge)
    bridge.remote_row = None
    bridge.remote = cast(
        AttachedSession,
        SimpleNamespace(
            supplements_for=lambda anchors: {},
            _runtime_record=SimpleNamespace(capabilities=("supplements-v1",) if owner else ()),
        ),
    )
    observed = {}

    async def reader(**kwargs):
        observed.update(kwargs)
        return {"entries": []}

    bridge._open_frame_history = reader
    await bridge.history(open_frame=True, supplements=viewer)
    assert observed["supplements"] is expected


def test_component_array_and_label_sizes_are_bounded() -> None:
    entry = supplement()
    component = entry["payload"]["details"]["components"][0]
    component["title"] = "x" * 50000
    entry["payload"]["details"]["components"] = [component] * 100
    result = of.strip_entry(entry, supplements=True)
    assert result is not None
    components = result["payload"]["details"]["components"]
    assert len(components) == 3 and len(components[0]["title"]) <= 2048
    assert of.served_bytes([result]) < 32768
