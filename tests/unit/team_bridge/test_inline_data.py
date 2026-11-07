"""Withholding inline base64 bodies from model context.

THE RULE UNDER TEST, and why the tests are shaped this way. If a body contains
``;base64,`` anywhere, the whole body is replaced by a bounded prefix plus a
truthful statement; if it does not, the body is returned byte-identical. No
header grammar is left to assert on, so these tests pin the DECISION (marker
present / absent) and the CONTRACT (prefix bound, honest count, no payload past
the prefix, byte-identical pass-through) rather than the shape of any parse.

The old matrix tested header spellings one by one because each was a separate
bypass. That is the trap this design removes: a new spelling can only ever
produce the same truncation, so one marker case plus the false-positive control
covers the class.
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, cast

import pytest

from local_operator.harness.message_types import (
    BRIDGE_BODY_READ_HEAD,
    BRIDGE_UNTRUSTED_CLAUSE,
)
from local_operator.harness.types import ToolContext
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.inline_data import (
    MAX_PREFIX_CHARS,
    bound_inline_data_payload,
    bound_inline_data_uris,
)
from local_operator.team_bridge.tool import BridgeRuntime, build_bridge_tool
from tests.unit.team_bridge.test_channel_tool import RecordingClient, _owner

TENANT = "ten_1"

#: The operator's repro scale, so the real-path test keeps its shape.
_BIG_PAYLOAD = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 2048).decode("ascii")

#: Any long base64 run. URL-safe characters are folded in: a payload that kept
#: its ``-_`` spelling must not survive either.
_RUN = re.compile(r"[A-Za-z0-9+/_-]{64,}")


def _withheld(body: str) -> str:
    """Assert the body was withheld per the rule, and return the result.

    The kept part must be a VERBATIM prefix of the original and the marker must
    state the original length. Deliberately not a "no long alphanumeric run"
    check: a 300-character media type or parameter is the sender's own header
    text and legitimately sits in the prefix, so a run detector would flag the
    body's own words. What matters is that nothing PAST the prefix survives.
    """

    bounded = bound_inline_data_uris(body)
    assert bounded != body, "the blob shipped unmarked"
    marker = (
        f"\n…[inline base64 data withheld: {len(body)} characters in this message; body truncated]"
    )
    assert bounded.endswith(marker), bounded[-200:]
    prefix = bounded[: -len(marker)]
    assert body.startswith(prefix), "the kept part is not a verbatim prefix"
    assert len(prefix) <= MAX_PREFIX_CHARS, len(prefix)
    assert ";base64," not in prefix.lower(), "the marker itself leaked into the prefix"
    return bounded


def test_a_body_without_the_marker_is_returned_by_identity() -> None:
    """The control, and the identity assertion is the point.

    A body merely mentioning ``metadata:`` must not be touched; neither must a
    very long ordinary message. Returning the SAME object proves no copy was
    rebuilt, so a future implementation cannot shorten one unnoticed.
    """

    prose = "2026-11-05 metadata: request_id=abc status=ok balanced\n" * 32000
    assert len(prose) > 1_700_000
    assert "data:" in prose
    assert bound_inline_data_uris(prose) is prose

    plain = "standup notes\n" + ("a normal sentence with spaces. " * 70000)
    assert len(plain) > 2_000_000
    assert bound_inline_data_uris(plain) is plain


def test_the_false_positive_is_real_and_stated() -> None:
    """A message that only MENTIONS ``;base64,`` is truncated too — by design.

    This is the accepted cost of a rule that cannot fail open. It is asserted so
    it is a documented decision rather than a surprise in review: the prose
    BEFORE the marker survives and the rest is replaced by the statement.
    """

    body = "how do I decode this? the flag is ;base64, and then a path follows\n" + "x" * 4000
    bounded = _withheld(body)
    assert bounded.startswith("how do I decode this? the flag is ")
    assert "and then a path follows" not in bounded


@pytest.mark.parametrize(
    "label,body",
    [
        ("single-line payload", f"chart\ndata:image/png;base64,{_BIG_PAYLOAD}\nend"),
        (
            "76-column wrap",
            "chart\ndata:image/png;base64,"
            + "\n".join(_BIG_PAYLOAD[i : i + 76] for i in range(0, len(_BIG_PAYLOAD), 76))
            + "\nend",
        ),
        (
            "40-column wrap",
            "chart\ndata:image/png;base64,"
            + "\n".join(_BIG_PAYLOAD[i : i + 40] for i in range(0, len(_BIG_PAYLOAD), 40))
            + "\nend",
        ),
        ("CRLF after marker", "chart\ndata:image/png;base64,\r\n" + _BIG_PAYLOAD + "\nend"),
        ("spaces after marker", f"chart\ndata:image/png;base64,   {_BIG_PAYLOAD}\nend"),
        ("300-char parameter", f"x data:image/png;name={'a' * 300};base64,{_BIG_PAYLOAD} y"),
        ("300-char media type", f"x data:{'a' * 300};base64,{_BIG_PAYLOAD} y"),
        (
            "20 parameters",
            f"x data:image/png{''.join(f';p{i}=v' for i in range(20))};base64,{_BIG_PAYLOAD} y",
        ),
        ("upper-case DATA:", f"chart\nDATA:image/png;base64,{_BIG_PAYLOAD}\nend"),
        ("upper-case BASE64:", f"chart\ndata:image/png;BASE64,{_BIG_PAYLOAD}\nend"),
        (
            "base64url alphabet",
            f"chart\ndata:image/png;base64,{_BIG_PAYLOAD.replace('+', '-').replace('/', '_')}\nend",
        ),
    ],
)
def test_every_marker_shape_is_withheld(label: str, body: str) -> None:
    """One decision, so every spelling lands on the same outcome.

    These were eleven separate bypasses across four review rounds. Under the
    marker rule they are one case: the marker is present, therefore the body is
    replaced. A new spelling cannot produce a new outcome.
    """

    bounded = _withheld(body)
    # The prose that PRECEDED the URI is kept, which is what makes the receipt
    # useful rather than just safe.
    assert bounded.startswith("chart") or bounded.startswith("x "), bounded[:40]


def test_a_one_megabyte_marker_dense_body_stays_fast() -> None:
    """Cost is one substring search plus one slice, on attacker-sized input.

    The bound is generous on purpose — this is a ceiling that fails a return to
    per-URI parsing, not a micro-benchmark. Earlier designs measured 3.5 s here.
    """

    body = "data:image/png;base64," * 50_000
    assert len(body) > 1_000_000
    started = time.perf_counter()
    bounded = bound_inline_data_uris(body)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05, elapsed
    assert "inline base64 data withheld:" in bounded


def test_the_payload_walker_passes_an_unaffected_page_through_by_identity() -> None:
    page = {"messages": [{"id": "cmsg_1", "body": {"text": "no inline data here"}}]}
    assert bound_inline_data_payload(page) is page
    assert bound_inline_data_payload(page["messages"]) is page["messages"]
    body = page["messages"][0]["body"]["text"]
    assert bound_inline_data_payload(body) is body


def test_the_payload_walker_replaces_a_body_it_does_have_to_change() -> None:
    page = {"messages": [{"body": {"text": f"a\ndata:image/png;base64,{_BIG_PAYLOAD}\nb"}}]}
    bounded = bound_inline_data_payload(page)
    assert "inline base64 data withheld:" in bounded["messages"][0]["body"]["text"]
    # Untouched siblings keep their identity even when a cousin changed.
    untouched = {"messages": [{"body": {"text": "clean"}}, page["messages"][0]]}
    assert bound_inline_data_payload(untouched)["messages"][0] is untouched["messages"][0]


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _runtime(client: Any) -> BridgeRuntime:
    return BridgeRuntime(
        client=cast(TeamBridgeClient, client),
        config_dir=".",
        tenant_id=TENANT,
        owner=_owner(),
        session_id="sess_1",
    )


@pytest.mark.asyncio
async def test_channel_history_withholds_and_leaves_details_and_the_wire_alone() -> None:
    """The real path: the page the model gets is clean, everything else is not."""

    wrapped = "\n".join(_BIG_PAYLOAD[i : i + 76] for i in range(0, len(_BIG_PAYLOAD), 76))
    body = f"chart\ndata:image/png;base64,{wrapped}\nend"
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
    context = ToolContext(bridge_runtime=_runtime(client))
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
    # The clause still leads: nothing here may un-mark untrusted content.
    assert visible.startswith(BRIDGE_UNTRUSTED_CLAUSE)
    assert "inline base64 data withheld:" in visible
    assert _RUN.findall(visible) == []
    assert _BIG_PAYLOAD[:64] not in visible
    # The page the model receives is still valid JSON.
    json.loads(visible[len(BRIDGE_BODY_READ_HEAD) + 2 :])

    # ``details`` is never serialized to providers: the raw row a human's
    # renderer reads is byte-for-byte unchanged by this filter.
    details = result.details or {}
    assert _BIG_PAYLOAD[:64] in str(details["content"])
    assert "inline base64 data withheld:" not in str(details["content"])

    # The WIRE is unchanged: the hub call carries exactly what it always did.
    assert client.calls == [
        {
            "method": "conversation_messages",
            "conversation_id": "bcv_abcd1234",
            "tenant_id": TENANT,
            "limit": 20,
            "before_seq": None,
            "after_seq": None,
            "endpoint_id": "bep_1",
        }
    ]
