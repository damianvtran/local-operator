"""The viewer-declared ``diff_bindings`` lane of the open-frame strip (QA F5 on #3762).

A nested eval write/edit's full-diff binding is a host-written custom row. An
older client cannot paint it, so by DEFAULT it stays outside the allow-list and
spends no row budget; a viewer that declares ``diff_bindings`` gets it back
rebuilt from validated, whitelisted fields only, scoped to the VIEWED session.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.session import open_frame as of
from local_operator.tools import diff_archive
from tests.unit.session.test_open_frame import custom, message

SID = "sess-view-1"
OUTER = "call-outer-1"
NESTED = f"{OUTER}:nested-2"


def _details(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "v": 1,
        "session_id": SID,
        "outer_call_id": OUTER,
        "call_id": NESTED,
        "tool": "write",
        "diff_full": {"v": 1, "state": "archived", "rows": 503, "digest": "d" * 32},
    }
    base.update(over)
    return base


def _binding(**over: Any) -> dict[str, Any]:
    return custom("b-1", diff_archive.NESTED_DIFF_BINDING_TYPE, _details(**over))


def test_type_literal_is_pinned_to_the_archive_constant():
    assert of._DIFF_BINDING_TYPE == diff_archive.NESTED_DIFF_BINDING_TYPE


def test_default_off_drops_the_binding_exactly_as_before():
    rows = [message("u-1", "user"), _binding()]
    assert of.strip_entry(_binding()) is None
    assert [r["id"] for r in of.paintable(rows)] == ["u-1"]


def test_declared_viewer_gets_only_the_sanitized_binding():
    out = of.strip_entry(_binding(), diff_bindings=SID)
    assert out is not None
    assert out["payload"] == {
        "custom_type": diff_archive.NESTED_DIFF_BINDING_TYPE,
        "details": _details(),
    }
    assert set(out) <= {"id", "ts", "ts_source", "type", "payload"}


def test_host_extras_and_descriptor_extras_never_reach_the_wire():
    out = of.strip_entry(_binding(path="/secret/p", content="RAW"), diff_bindings=SID)
    assert out is not None and "path" not in out["payload"]["details"]
    assert "RAW" not in repr(out) and "/secret/p" not in repr(out)
    bad = _details()
    bad["diff_full"] = {**bad["diff_full"], "rows_raw": ["+secret"]}
    forged = custom("b-2", diff_archive.NESTED_DIFF_BINDING_TYPE, bad)
    assert of.strip_entry(forged, diff_bindings=SID) is None


@pytest.mark.parametrize(
    "over",
    [
        {"session_id": "other-session"},
        {"call_id": "elsewhere:nested-2"},
        {"outer_call_id": "not-the-prefix"},
        {"v": 2},
        {"tool": "bash"},
        {"diff_full": "archived"},
        {"diff_full": {"v": "1", "state": "archived", "rows": 1}},
        {"diff_full": {"v": 1, "state": "archived", "rows": True}},
        {"diff_full": {"v": 1, "state": ""}},
    ],
)
def test_forged_or_malformed_binding_drops(over):
    assert of.strip_entry(_binding(**over), diff_bindings=SID) is None


def test_binding_is_judged_against_the_viewed_session_not_its_own_claim():
    # A row self-consistent for ANOTHER session is not paintable in this view.
    assert of.strip_entry(_binding(), diff_bindings="sess-view-2") is None


def test_other_custom_types_still_drop_with_the_flag_on():
    assert of.strip_entry(custom("c-1", "stt_transcript_v1", {"x": 1}), diff_bindings=SID) is None


def test_memo_carries_the_view_and_refuses_a_mismatch():
    memo = of.StripMemo(diff_bindings=SID)
    assert [r["id"] for r in of.paintable([_binding()], memo=memo, diff_bindings=SID)] == ["b-1"]
    with pytest.raises(ValueError):
        of.paintable([_binding()], memo=memo)
    with pytest.raises(ValueError):
        of.paintable([_binding()], memo=of.StripMemo(), diff_bindings=SID)
