"""Bounded receipts for inline ``data:…;base64,…`` runs in a message BODY.

WHY THESE ASSERTIONS AND NOT OTHERS. The complaint is concrete — one ~600 KB
base64 string in a body reaching the model — so the tests are written against
what the MODEL sees, not against the implementation: every case asserts that no
long base64 run survives and that the withheld receipt names the type and the
size. The last test drives the real tool path so the choke point in
``tool.py`` is exercised rather than the helper alone; a helper that nothing
calls is exactly the failure mode ``models.BridgeFilesHashV1`` already has.
"""

from __future__ import annotations

import base64
import re
from typing import Any, cast

import pytest

from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE
from local_operator.harness.types import ToolContext
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.inline_data import (
    INLINE_DATA_SCAN_MAX_CHARS,
    bound_inline_data_uris,
    inline_data_marker,
)
from local_operator.team_bridge.tool import BridgeRuntime, build_bridge_tool
from tests.unit.team_bridge.test_channel_tool import RecordingClient, _owner

TENANT = "ten_1"

#: Anything at or above this is a blob, not prose. 64 is far below any real
#: base64 payload and far above anything a human sentence contains (the
#: sampled bodies below carry no such run outside the injected URIs).
_RUN = re.compile(r"[A-Za-z0-9+/]{64,}")

#: A plausible PNG payload whose LENGTH is what the receipt must report.
_PNG_PAYLOAD = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4).decode("ascii")


def _runs(text: str) -> list[str]:
    return _RUN.findall(text)


def test_plain_text_is_returned_unchanged_and_is_not_truncated() -> None:
    """The no-op path is load-bearing: ordinary long bodies keep every byte.

    A filter that windowed every body would silently shorten large legitimate
    messages even when they carry no data URI at all.
    """

    line = "a normal sentence with spaces. "
    body = "standup notes\n" + line * (INLINE_DATA_SCAN_MAX_CHARS // len(line))
    assert len(body) > INLINE_DATA_SCAN_MAX_CHARS
    assert bound_inline_data_uris(body) is body


def test_typical_body_uri_becomes_a_receipt_naming_mime_and_size() -> None:
    body = f"look at this chart\ndata:image/png;base64,{_PNG_PAYLOAD}\nthanks"
    bounded = bound_inline_data_uris(body)

    # The prose around the blob survives; only the blob is gone.
    assert bounded.startswith("look at this chart\ndata:") is False
    assert bounded.startswith("look at this chart\n")
    assert bounded.endswith("\nthanks")
    # Names the type and the TRUE decoded size, and says the bytes were withheld.
    assert inline_data_marker("image/png", len(base64.b64decode(_PNG_PAYLOAD))) in bounded
    assert "image/png" in bounded
    assert "withheld" in bounded
    assert str(len(base64.b64decode(_PNG_PAYLOAD))) in bounded
    # No base64 run of any length survives.
    assert _runs(bounded) == []


def test_multiple_uris_in_one_body_are_each_receipted() -> None:
    first = _PNG_PAYLOAD
    second = base64.b64encode(b"second payload, different length").decode("ascii")
    body = (
        f"two shots:\n1) data:image/png;base64,{first}\n"
        f"2) data:image/jpeg;base64,{second}\nend"
    )
    bounded = bound_inline_data_uris(body)

    assert bounded.count("[inline data withheld:") == 2
    assert "image/png" in bounded
    assert "image/jpeg" in bounded
    # Prose between and after the blobs is intact.
    assert "two shots:" in bounded
    assert "1) " in bounded and "2) " in bounded
    assert bounded.endswith("\nend")
    assert _runs(bounded) == []


def test_uri_inside_a_code_fence_is_receipted_and_the_fence_survives() -> None:
    """A fenced block is still a body: the fence is text, the blob is not.

    This is the shape the operator actually hit — someone pasted an image as
    Markdown inside a fence, so nothing upstream treats it as an attachment.
    """

    body = f"here is the payload:\n```\ndata:image/png;base64,{_PNG_PAYLOAD}\n```\ndone"
    bounded = bound_inline_data_uris(body)

    assert bounded.startswith("here is the payload:\n```\n")
    assert bounded.endswith("\n```\ndone")
    assert "withheld" in bounded
    assert _runs(bounded) == []


def test_malformed_base64_still_loses_the_blob() -> None:
    """Invalid characters mid-payload must not become a pass-through.

    The regex stops at the first character outside the base64 alphabet. What
    matters is that the run that WAS a blob is receipted and what remains is
    not a long base64 run.
    """

    body = "prefix data:image/png;base64," + ("A" * 200) + "!!!trailing-junk!!! suffix"
    bounded = bound_inline_data_uris(body)

    assert bounded.startswith("prefix ")
    assert bounded.endswith(" suffix")
    assert "image/png" in bounded
    assert _runs(bounded) == []
    # The junk was never base64 and is left as the (harmless) text it is.
    assert "trailing-junk" in bounded


def test_truncated_base64_reports_a_size_floor_not_a_wrong_exact_size() -> None:
    """A payload cut mid-quantum has an incomplete byte count — say "at least".

    Reporting an exact number would be a lie in the one direction that matters:
    the model would size what it is missing too small.
    """

    body = "x data:image/png;base64,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA suffix"
    bounded = bound_inline_data_uris(body)

    assert "at least" in bounded
    assert "image/png" in bounded
    assert _runs(bounded) == []


def test_header_without_a_payload_is_left_alone() -> None:
    """``data:image/png;base64,`` alone is a mention, not bytes to withhold.

    Claiming bytes were dropped when none were would mislead the model about
    what the sender said.
    """

    body = "the format is data:image/png;base64,<payload> — that is the shape"
    bounded = bound_inline_data_uris(body)

    assert bounded == body


def test_very_large_body_is_windowed_and_the_omission_is_stated() -> None:
    """Over the window the tail is withheld WITH a count, never forwarded.

    Forwarding an unscanned tail is the exact unbounded-blob failure this
    module exists to stop, so the assertion is both "no base64 survives" and
    "the reader is told how much was dropped".
    """

    head = f"big one\ndata:image/png;base64,{_PNG_PAYLOAD}\n"
    filler = "y " * INLINE_DATA_SCAN_MAX_CHARS
    body = head + filler
    bounded = bound_inline_data_uris(body)

    assert len(body) > INLINE_DATA_SCAN_MAX_CHARS
    assert "withheld" in bounded
    assert "not scanned for inline data and are withheld" in bounded
    # The stated count is exactly the number of characters dropped.
    dropped = len(body) - INLINE_DATA_SCAN_MAX_CHARS
    assert f"{dropped} more bytes" in bounded
    # Bounded output: the window plus the receipts, never the whole body.
    assert len(bounded) < INLINE_DATA_SCAN_MAX_CHARS + 4096
    assert _runs(bounded) == []


def test_uri_cut_by_the_window_reports_at_least() -> None:
    """A payload running past the window boundary has an unknown true length."""

    body = "data:image/png;base64," + ("Q" * (INLINE_DATA_SCAN_MAX_CHARS * 2))
    bounded = bound_inline_data_uris(body)

    assert "at least" in bounded
    assert "image/png" in bounded
    assert _runs(bounded) == []


def test_decoded_size_is_arithmetic_and_matches_a_real_decode() -> None:
    """The reported size must be right without the codec ever running.

    ``0``/`1`/`2` padding bytes are all exercised because they are the only way
    the ``len * 3 // 4`` form is wrong if the padding is miscounted.
    """

    for raw in (b"a", b"ab", b"abc", b"abcd", bytes(range(251))):
        payload = base64.b64encode(raw).decode("ascii")
        bounded = bound_inline_data_uris(f"data:application/octet-stream;base64,{payload}")
        assert f"{len(raw)} bytes decoded" in bounded, (raw, bounded)


def test_marker_is_stable_so_a_renderer_can_match_it() -> None:
    assert inline_data_marker("image/png", 12) == (
        "[inline data withheld: image/png, 12 bytes decoded]"
    )
    assert inline_data_marker("", 0) == "[inline data withheld: text/plain, 0 bytes decoded]"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


@pytest.mark.asyncio
async def test_the_tool_result_the_model_receives_carries_no_base64() -> None:
    """The REAL path: one channel_history body with a blob, built by the tool.

    This is the assertion the task turns on — not "the helper works" but "the
    content blocks handed to the provider are clean". ``details`` is
    deliberately NOT asserted clean: it is never serialized to providers and it
    is what the transcript and UI render for humans.
    """

    blob = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 2048).decode("ascii")
    assert len(blob) > 600_000, "the operator's repro was ~600 KB; keep this shape"
    client = RecordingClient(
        messages={
            "messages": [
                {
                    "id": "cmsg_1",
                    "seq": 6,
                    "sender": {"kind": "agent", "display_label": "Aida"},
                    "body": {"text": f"chart\ndata:image/png;base64,{blob}\nend"},
                }
            ]
        }
    )
    runtime = BridgeRuntime(
        client=cast(TeamBridgeClient, client),
        config_dir=".",
        tenant_id=TENANT,
        owner=_owner(),
        session_id="sess_1",
    )
    context = ToolContext(bridge_runtime=runtime)
    tool = build_bridge_tool(context)
    assert tool is not None

    result = await tool.execute(
        "synthetic-call",
        {"op": "channel_history", "channel_id": "bcv_abcd1234", "limit": 20},
        None,
        None,
        context,
    )

    assert result.is_error is False
    model_visible = "\n".join(getattr(block, "text", "") or "" for block in result.content)
    # The clause still leads: this change must not un-mark untrusted content.
    assert model_visible.startswith(BRIDGE_UNTRUSTED_CLAUSE)
    assert "chart" in model_visible and "end" in model_visible
    assert "withheld" in model_visible
    assert _runs(model_visible) == []
    assert blob[:80] not in model_visible
    # ``details`` is never serialized to providers, so it still carries the raw
    # row — the transcript and UI a human reads are unchanged by this filter.
    details = result.details or {}
    assert blob in str(details["content"])
