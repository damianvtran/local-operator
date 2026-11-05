"""Bounded receipts for inline ``data:…;base64,…`` runs in a message BODY.

WHY THESE ASSERTIONS AND NOT OTHERS. The complaint is concrete — one ~600 KB
base64 string in a body reaching the model — so the tests are written against
what the MODEL sees, not against the implementation: every case asserts that no
long base64 run survives and that the withheld receipt names the type and the
size. The last test drives the real tool path so the choke point in
``tool.py`` is exercised rather than the helper alone; a helper that nothing
calls is exactly the failure mode ``models.BridgeFilesHashV1`` already has.

Round-1 findings shape the adversarial half of this file. ``test_scan_cost_*``
is a HARD-RATIO complexity assertion, not a micro-benchmark: the old pattern
was quadratic (3.5 s at 25 KB), so a ratio bound between two sizes fails it
regardless of how fast the host is. The wrapped / URL-safe / uppercase cases
each kept 100 % of the payload before the fix.
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, cast

import pytest

from local_operator.harness.message_types import BRIDGE_UNTRUSTED_CLAUSE
from local_operator.harness.types import ToolContext
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.inline_data import (
    bound_inline_data_payload,
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

#: The operator's repro scale, so the real-path test keeps its shape.
_BIG_PAYLOAD = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 2048).decode("ascii")


def _runs(text: str) -> list[str]:
    """Long standard-base64 runs. URL-safe chars are folded in as well, because
    a payload that kept its ``-_`` spelling must still not survive."""
    return re.findall(r"[A-Za-z0-9+/_-]{64,}", text)


def test_plain_text_is_returned_unchanged_and_is_not_truncated() -> None:
    """The no-op path is load-bearing: ordinary long bodies keep every byte.

    A filter that windowed every body would silently shorten large legitimate
    messages even when they carry no data URI at all. Identity (``is``) is the
    assertion, not equality: it proves no copy and no rebuild happened.
    """

    line = "a normal sentence with spaces. "
    body = "standup notes\n" + line * 70000
    assert len(body) > 2_000_000
    assert bound_inline_data_uris(body) is body


def test_a_long_body_that_only_mentions_data_colon_is_untouched() -> None:
    """R-3's control: the substring ``data:`` is not a data URI.

    ``"data:" in "metadata:"`` is True, and the previous version withheld
    everything past 1 MiB on that alone — 41 % of an ordinary log page lost.
    """

    body = ("2026-11-05 metadata: request_id=abc status=ok balanced\n" * 32000)
    assert len(body) > 1_700_000
    assert "data:" in body
    assert bound_inline_data_uris(body) is body


def test_typical_body_uri_becomes_a_receipt_naming_mime_and_size() -> None:
    body = f"look at this chart\ndata:image/png;base64,{_PNG_PAYLOAD}\nthanks"
    bounded = bound_inline_data_uris(body)

    # The prose around the blob survives; only the blob is gone.
    assert bounded.startswith("look at this chart\n")
    assert bounded.endswith("\nthanks")
    decoded = len(base64.b64decode(_PNG_PAYLOAD))
    assert inline_data_marker("image/png", decoded) in bounded
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
    # Prose between and after the blobs is intact. (The ``2) `` prefix is
    # swallowed only where a marker replaced the line before it, so assert the
    # surviving text, not a position.)
    assert "two shots:" in bounded
    assert "1) [inline data withheld:" in bounded
    assert "2) [inline data withheld:" in bounded
    assert bounded.endswith("\nend")
    assert _runs(bounded) == []


def test_uri_inside_a_code_fence_is_receipted_and_the_fence_survives() -> None:
    """A fenced block is still a body: the fence is text, the blob is not."""

    body = f"here is the payload:\n```\ndata:image/png;base64,{_PNG_PAYLOAD}\n```\ndone"
    bounded = bound_inline_data_uris(body)

    assert bounded.startswith("here is the payload:\n```\n")
    assert bounded.endswith("\n```\ndone")
    assert "withheld" in bounded
    assert _runs(bounded) == []


def test_mime_wrapped_payload_is_receipted_as_one_blob_with_the_true_size() -> None:
    """R-2a: 76-column wrapping is the shape a shell tool actually emits.

    Before the fix each 76-char line matched separately, so the receipt claimed
    an exact 57 bytes while the model was billed for the whole blob.
    """

    wrapped = "\n".join(_BIG_PAYLOAD[i : i + 76] for i in range(0, len(_BIG_PAYLOAD), 76))
    body = f"chart\ndata:image/png;base64,{wrapped}\nend"
    bounded = bound_inline_data_uris(body)

    assert len(_runs(bounded)) == 0
    # ONE receipt, and it names the WHOLE wrapped payload, not one 76-char line.
    assert bounded.count("[inline data withheld:") == 1
    expected = len(base64.b64decode(_BIG_PAYLOAD))
    match = re.search(r"at least (\d+) bytes decoded", bounded)
    assert match is not None, bounded
    reported = int(match.group(1))
    # The final wrap line is deliberately left in place, so the figure is a floor
    # at most one line-width short — never the "57 bytes" the old code claimed.
    assert expected - 60 <= reported <= expected, (reported, expected)
    assert "at least" in bounded
    assert bounded.startswith("chart\n")
    assert bounded.endswith("\nend")
    assert bounded.startswith("chart\n")
    assert bounded.endswith("\nend")
    assert len(bounded) < 200


def test_url_safe_alphabet_payload_is_receipted() -> None:
    """R-2b: ``-_`` instead of ``+/`` is what a URL-safe encoder emits."""

    standard = base64.b64encode(bytes(range(256)) * 4000).decode("ascii")
    payload = standard.replace("+", "-").replace("/", "_")
    assert "-" in payload and "_" in payload, "the sample must exercise both"
    bounded = bound_inline_data_uris(f"blob data:application/octet-stream;base64,{payload} end")

    assert _runs(bounded) == []
    assert bounded.count("[inline data withheld:") == 1
    assert bounded.endswith(" end")
    assert f"{len(base64.b64decode(standard))} bytes decoded" in bounded


def test_uppercase_header_is_receipted() -> None:
    """R-2c: RFC 2397/5234 ABNF literals are case-insensitive; browsers agree."""

    for header in ("DATA:image/png;base64,", "data:image/png;BASE64,", "Data:image/png;Base64,"):
        bounded = bound_inline_data_uris(f"x {header}{_PNG_PAYLOAD} y")
        assert _runs(bounded) == [], header
        assert "withheld" in bounded, header
        assert "image/png" in bounded, header


def test_malformed_base64_still_loses_the_blob() -> None:
    """Invalid characters mid-payload must not become a pass-through."""

    body = "prefix data:image/png;base64," + ("A" * 200) + "!!!trailing-junk!!! suffix"
    bounded = bound_inline_data_uris(body)

    assert bounded.startswith("prefix ")
    assert bounded.endswith(" suffix")
    assert "image/png" in bounded
    assert _runs(bounded) == []
    # The junk was never base64 and is left as the (harmless) text it is.
    assert "trailing-junk" in bounded


def test_truncated_base64_reports_a_size_floor_not_a_wrong_exact_size() -> None:
    """A payload cut mid-quantum has an incomplete byte count — say "at least"."""

    body = "x data:image/png;base64,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA suffix"
    bounded = bound_inline_data_uris(body)

    assert "at least" in bounded
    assert "image/png" in bounded
    assert _runs(bounded) == []


def test_header_without_a_payload_is_left_alone() -> None:
    """``data:image/png;base64,`` alone is a mention, not bytes to withhold."""

    body = "the format is data:image/png;base64,<payload> — that is the shape"
    assert bound_inline_data_uris(body) == body


def test_decoded_size_is_arithmetic_and_matches_a_real_decode() -> None:
    """The reported size must be right without the codec ever running.

    ``0``/``1``/``2`` padding bytes are all exercised because they are the only
    way the ``len * 3 // 4`` form is wrong if the padding is miscounted.
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


def test_scan_cost_is_linear_not_quadratic_on_adversarial_repeats() -> None:
    """R-1's regression: a HARD RATIO, not a micro-benchmark.

    The old pattern retried an unbounded mime group from every start position,
    so ``"data:" * n`` cost O(n²): measured 3.5 s at 25 KB. A wall-clock
    ceiling alone would be host-dependent; requiring ``t(4n) / t(n)`` to stay
    near the linear factor of 4 fails the quadratic shape on any machine. The
    absolute ceiling is a second, generous backstop for the largest body a
    single call is realistically handed.
    """

    def elapsed(chars: int) -> float:
        body = "data:" * (chars // 5)
        best = min(
            _timed(lambda: bound_inline_data_uris(body)) for _ in range(3)
        )
        return best

    small = elapsed(25_000)
    large = elapsed(100_000)
    # Quadratic would be ~16x; the bound leaves generous headroom for noise.
    assert large / max(small, 1e-6) < 8.0, (small, large)
    # And a megabyte of adversarial input must stay comfortably interactive.
    worst = _timed(lambda: bound_inline_data_uris("data:" * 210_000))
    assert worst < 5.0, worst


def _timed(call: Any) -> float:
    started = time.perf_counter()
    call()
    return time.perf_counter() - started


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

    assert len(_BIG_PAYLOAD) > 600_000, "the operator's repro was ~600 KB; keep this shape"
    client = RecordingClient(
        messages={
            "messages": [
                {
                    "id": "cmsg_1",
                    "seq": 6,
                    "sender": {"kind": "agent", "display_label": "Aida"},
                    "body": {"text": f"chart\ndata:image/png;base64,{_BIG_PAYLOAD}\nend"},
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
    assert _BIG_PAYLOAD[:80] not in model_visible
    # ``details`` is never serialized to providers, so it still carries the raw
    # row — the transcript and UI a human reads are unchanged by this filter.
    details = result.details or {}
    assert _BIG_PAYLOAD in str(details["content"])


# ---------------------------------------------------------------------------
# Round-2 findings. The JSON page is THE shipping path: every history op builds
# its model-visible content with ``json.dumps``, which turns the newline inside
# a MIME-wrapped payload into the two characters ``\`` ``n``. A scan of that
# serialised text therefore never sees the fold, and the previous version kept
# the whole blob while claiming an exact 57 bytes. These drive the real tool
# because that is the only test that would have caught it.
# ---------------------------------------------------------------------------


def _wrapped(payload: str, width: int = 76) -> str:
    return "\n".join(payload[i : i + width] for i in range(0, len(payload), width))


def _assert_no_payload(visible: str) -> None:
    """No PAYLOAD survives. The receipts are stripped first: their media type is
    a bounded label (clamped to 64 chars), not withheld bytes, and a real one —
    ``application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`` — is a
    long alphanumeric run that a naive detector would call a blob."""

    stripped = re.sub(r"\[inline data withheld: [^\]]*\]", "", visible)
    assert _runs(stripped) == [], "a base64 run survived into model content"
    assert "base64," not in visible, "a data URI payload survived"


@pytest.mark.asyncio
async def test_channel_history_json_page_withholds_a_wrapped_payload() -> None:
    """The shipping path: a 76-column wrap inside a ``json.dumps`` page."""

    body = f"chart\ndata:image/png;base64,{_wrapped(_BIG_PAYLOAD)}\nend"
    client = RecordingClient(
        messages={
            "messages": [
                {
                    "id": "cmsg_1",
                    "seq": 6,
                    "sender": {"kind": "agent", "display_label": "Aida"},
                    "body": {"text": body},
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
    visible = "\n".join(getattr(block, "text", "") or "" for block in result.content)
    _assert_no_payload(visible)
    # Truthful floor, not the old false exact "57 bytes" for one wrapped line.
    assert "at least" in visible
    # The page the model receives is still valid JSON — the cut lands inside a
    # string, never across the object.
    json.loads(visible[len(BRIDGE_UNTRUSTED_CLAUSE) + 2 :])
    # Humans still get the raw row: ``details`` is not serialized to providers.
    # Raw row preserved for renderers — the wrapped payload is present with its
    # newlines escaped, so compare a prefix rather than the unwrapped string.
    assert _BIG_PAYLOAD[:64] in str((result.details or {})["content"])


@pytest.mark.asyncio
async def test_channel_thread_json_page_withholds_a_wrapped_payload() -> None:
    """``channel_thread`` builds its page the same way and gets the same bound."""

    row = {
        "id": "cmsg_1",
        "seq": 6,
        "sender": {"kind": "agent"},
        "body": {"text": f"chart\ndata:image/png;base64,{_wrapped(_BIG_PAYLOAD)}\nend"},
    }
    client = RecordingClient()
    client.replies_result = {"root": row, "replies": [], "has_more": False}
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
        {
            "op": "channel_thread",
            "channel_id": "bcv_abcd1234",
            "thread_root_id": "bcmsg_abcd1234",
            "limit": 20,
        },
        None,
        None,
        context,
    )

    assert result.is_error is False
    visible = "\n".join(getattr(block, "text", "") or "" for block in result.content)
    _assert_no_payload(visible)
    assert "at least" in visible
    assert _BIG_PAYLOAD[:64] in str((result.details or {}).get("content", ""))


def test_escaped_newlines_are_a_second_layer() -> None:
    """If a caller only has the SERIALISED text, the fold is still understood.

    The shipping fix bounds the plain string before serialisation, so this is
    the defensive half: a future page builder that hands over only JSON text
    must not silently reopen the hole that shipped 700 KB.
    """

    dumped = json.dumps({"body": {"text": f"x\ndata:image/png;base64,{_wrapped(_BIG_PAYLOAD)}\ny"}})
    assert "\\n" in dumped, "the fold must appear escaped, or this tests nothing"
    bounded = bound_inline_data_uris(dumped)

    assert _runs(bounded) == []
    assert "at least" in bounded
    # The escaped page is still valid JSON after the substitution.
    json.loads(bounded)


def test_payload_walker_passes_an_unaffected_page_through_by_identity() -> None:
    """Copy-on-write: the common case allocates nothing and keeps the object."""

    page = {"messages": [{"id": "cmsg_1", "body": {"text": "no inline data here"}}]}
    assert bound_inline_data_payload(page) is page
    assert bound_inline_data_payload(page["messages"]) is page["messages"]
    body = page["messages"][0]["body"]["text"]
    assert bound_inline_data_payload(body) is body


def test_payload_walker_bounds_a_body_it_does_have_to_change() -> None:
    page = {"messages": [{"body": {"text": f"a\ndata:image/png;base64,{_PNG_PAYLOAD}\nb"}}]}
    bounded = bound_inline_data_payload(page)
    text = bounded["messages"][0]["body"]["text"]
    assert text is not page["messages"][0]["body"]["text"]
    assert "withheld" in text
    # The untouched sibling is still the same object.
    assert bound_inline_data_payload(page)["messages"] is not page["messages"]


# ---------------------------------------------------------------------------
# QA round 3. Every case here is a header or payload SHAPE that made the old
# header-grammar matcher fail OPEN — the payload shipped unmarked, which is the
# one failure mode this module exists to prevent. The matcher no longer reads
# the header at all (it locates the literal `;base64,` and scans the alphabet
# forward), so these can only fail if that stops being true.
# ---------------------------------------------------------------------------


def _assert_withheld(body: str, *, label: str) -> str:
    """No `;base64,` content survives, and a receipt is present and honest."""

    bounded = bound_inline_data_uris(body)
    assert bounded != body, f"{label}: unchanged — the blob shipped"
    assert "base64," not in bounded, f"{label}: payload survived"
    assert "[inline data withheld:" in bounded, f"{label}: no receipt"
    _assert_no_payload(bounded)
    return bounded


def test_q1_a_300_char_parameter_does_not_fail_open() -> None:
    """Q-1: a long parameter used to exceed the cap and skip the whole URI."""

    bounded = _assert_withheld(
        f"x data:image/png;name={'a' * 300};base64,{_BIG_PAYLOAD} y",
        label="300-char parameter",
    )
    assert "image/png" in bounded


def test_q1_a_300_char_media_type_does_not_fail_open() -> None:
    """Q-1: an over-long media type is decoration, never a gate."""

    _assert_withheld(
        f"x data:{'a' * 300};base64,{_BIG_PAYLOAD} y", label="300-char media type"
    )


def test_q1_twenty_parameters_do_not_fail_open() -> None:
    """Q-1: a parameter COUNT past the old cap must not skip the URI either."""

    params = "".join(f";p{i}=v" for i in range(20))
    _assert_withheld(f"x data:image/png{params};base64,{_BIG_PAYLOAD} y", label="20 params")


def test_q2_a_line_break_directly_after_the_marker_does_not_fail_open() -> None:
    """Q-2: whitespace after `;base64,` used to read as an empty payload.

    The empty-payload rule exists so a genuine MENTION (`data:…;base64,<payload>`
    in prose) is left alone. Whitespace after the marker is not a mention, it is
    a wrapped payload starting on the next line — so the scanner skips it first.
    """

    for gap in ("\r\n", "\n", "\n   ", " \t"):
        wrapped = "\n".join(_BIG_PAYLOAD[i : i + 76] for i in range(0, len(_BIG_PAYLOAD), 76))
        bounded = _assert_withheld(
            f"x data:image/png;base64,{gap}{wrapped}\nend", label=f"marker then {gap!r}"
        )
        # Cut at the final short wrap line, so the size is a floor and says so.
        assert "at least" in bounded


def test_q3_a_forty_column_wrap_is_consumed_as_one_blob() -> None:
    """Q-3: a narrow wrap used to stop after the first 40-char line."""

    wrapped = "\n".join(_BIG_PAYLOAD[i : i + 40] for i in range(0, len(_BIG_PAYLOAD), 40))
    bounded = _assert_withheld(f"x data:image/png;base64,{wrapped}\nend", label="40-col wrap")

    assert "at least" in bounded
    # ONE receipt carrying the whole payload, not one per line.
    assert bounded.count("[inline data withheld:") == 1
    reported = re.search(r"at least (\d+) bytes decoded", bounded)
    assert reported is not None, bounded
    expected = len(base64.b64decode(_BIG_PAYLOAD))
    assert expected - 60 <= int(reported.group(1)) <= expected


def test_the_fold_floor_keeps_a_short_word_but_eats_a_long_run() -> None:
    """The folding tradeoff, asserted in BOTH directions.

    A short word after a blob is prose and must survive; a run of 16+ base64
    characters after a line break is indistinguishable from a wrapped line, so it
    is consumed. Stating the second half as a test is the point: it is a known,
    accepted over-delete, not a surprise to be rediscovered in review.
    """

    kept = bound_inline_data_uris(f"data:image/png;base64,{_BIG_PAYLOAD}\nthanks")
    assert kept.endswith("thanks"), kept[-40:]

    eaten = bound_inline_data_uris(f"data:image/png;base64,{_BIG_PAYLOAD}\ninternationalized")
    assert "internationalized" not in eaten
    _assert_no_payload(kept)


def test_q4_the_receipt_never_echoes_a_media_type_at_blob_length() -> None:
    """Q-4: the echoed type must stay under the 64-char blob-detector run."""

    bounded = bound_inline_data_uris(f"data:application/{'x' * 400};base64,{_BIG_PAYLOAD}")
    receipt = re.search(r"\[inline data withheld: ([^,]*), [^\]]*\]", bounded)
    assert receipt is not None, bounded
    assert len(receipt.group(1)) < 64, receipt.group(1)
