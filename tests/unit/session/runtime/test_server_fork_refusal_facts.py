"""The fork refusal's CAUSE across the attach transport.

``ForkRefused`` says a fork the conversation's own state would not allow, and it
carries WHICH cause as one token from a closed set. That token is the whole
point of the classification: without it the owner raised a bare ``ValueError``,
the attach client re-raised it as an untyped ``RuntimeError``, and the control
plane's ladder could only read that as an unreachable owner — a 503 telling the
operator to reconnect and reconcile, for a request that was answered promptly
and deliberately (measured on PR #1917).

So the token travels in its own bounded field (``error_reason``) rather than
inside the message, and the decoder rebuilds the sentence locally from it — the
same closed-shape carriage ``error_count``/``error_trigger``/``error_model``
established. A reason this build does not know (a NEWER owner) or no reason at
all (a bare raise) degrades to the generic sentence rather than rendering, and a
frame that is not a recognised category keeps the pre-existing path exactly.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from local_operator.session.errors import ForkRefused, admission_error
from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from tests.unit.session.runtime.test_server import FakeHandle

#: Every reason ``ForkRefused`` enumerates, with the sentence that reason must
#: rebuild. Pinned in full rather than by keyword: the surface renders this
#: string, so a re-wording is a user-visible change and should fail here first.
REASONS: dict[str, str] = {
    "entry_unknown": (
        "that message is not part of this conversation; "
        "pick a message from this session to fork from"
    ),
    "before_anchor": (
        "that message sits before the conversation's last summary; "
        "fork from a message after the summary instead"
    ),
    "unfinished_batch": (
        "compaction boundary is in an unfinished tool batch; "
        "retry /fork after the original finishes that batch"
    ),
    "history_rewriting": "history is being rewritten; retry /fork when compaction finishes",
    "compaction_pending": "Wait for compaction to finish before forking",
    "fork_pending": "A fork is already waiting for a safe boundary",
    "unmatched_tool_result": "history has an unmatched tool result; cannot fork safely",
    "incomplete_tool_calls": "history has incomplete tool calls before later messages",
}


def _rig(exc: Exception) -> tuple[RuntimeServer, list[dict[str, Any]], _ClientConn]:
    """A server whose dispatch raises ``exc``, with its socket writes captured."""
    server = RuntimeServer(FakeHandle(), kind="tui")
    sent: list[dict[str, Any]] = []

    async def capture(target, frame):  # noqa: ANN001
        sent.append(frame)

    async def failing_dispatch(op, frame, **_kwargs):  # noqa: ANN001
        raise exc

    server._send_to = capture  # type: ignore[assignment]
    server._dispatch = failing_dispatch  # type: ignore[assignment]
    conn = _ClientConn(writer=cast(Any, object()), kind=cast(Any, "attach"))
    server._clients[id(conn.writer)] = conn
    return server, sent, conn


async def _error_frame(exc: Exception) -> dict[str, Any]:
    server, sent, conn = _rig(exc)
    await server._on_request({"op": "prompt", "req": 1}, conn)
    errors = [f for f in sent if f.get("op") == "error"]
    assert errors, f"expected an error frame, got {sent}"
    # Round-trip through JSON: the frame really goes over a socket.
    return cast(dict[str, Any], json.loads(json.dumps(errors[0])))


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", sorted(REASONS))
async def test_each_reason_crosses_and_rebuilds_its_own_sentence(reason: str) -> None:
    """End to end: raise on the owner, decode on the client, keep the cause."""
    frame = await _error_frame(ForkRefused(reason=reason))

    assert frame["error_code"] == ForkRefused.code
    assert frame["error_reason"] == reason

    # Exactly what attach_client.py does with the reply.
    known = admission_error(
        str(frame.get("error_code", "")),
        frame.get("error_count"),
        frame.get("error_trigger"),
        "",
        reason=frame.get("error_reason"),
    )
    assert isinstance(known, ForkRefused)
    assert known.reason == reason
    assert str(known) == REASONS[reason]


@pytest.mark.asyncio
async def test_a_bare_refusal_sends_no_reason_and_reads_as_the_generic_sentence() -> None:
    """A raise with no cause named is one sentence, not an empty field rendered."""
    frame = await _error_frame(ForkRefused())

    assert frame["error_code"] == ForkRefused.code
    assert "error_reason" not in frame, "no cause named, no token on the wire"
    known = admission_error(str(frame["error_code"]), reason=frame.get("error_reason"))
    assert isinstance(known, ForkRefused)
    assert known.reason == ""
    assert str(known) == ForkRefused.fallback


def test_a_reason_outside_the_set_is_never_rendered() -> None:
    """The decoder is the boundary: a peer's string keys a table or is dropped."""
    # A newer owner's token this build has never heard of: the fail-safe is the
    # generic sentence, not a raise and not the peer's text.
    unknown = admission_error(ForkRefused.code, reason="a_cause_this_build_lacks")
    assert isinstance(unknown, ForkRefused)
    assert unknown.reason == ""
    assert str(unknown) == ForkRefused.fallback

    # And nothing a peer controls can reach the sentence, in any type at all.
    for hostile in ("../../etc/passwd", "\x00\x1b[31mred", "entry_unknown; rm -rf /", 7, None):
        decoded = admission_error(ForkRefused.code, reason=hostile)  # type: ignore[arg-type]
        assert isinstance(decoded, ForkRefused)
        assert decoded.reason in ("", "entry_unknown")
        assert str(decoded) in (ForkRefused.fallback, REASONS["entry_unknown"])


def test_a_bare_value_error_is_not_a_category_at_all() -> None:
    """The untyped path is UNCHANGED: no code, so the client raises as before.

    This is the fail-safe half of the seam. An owner that raises an ordinary
    ``ValueError`` (or any un-enumerated exception) must still reach the client
    as the owner's own sentence via ``RuntimeError``, so the route's ladder keeps
    reading it as it always has — the classification narrows nothing that was
    not enumerated here.
    """
    assert admission_error("", None, None, "") is None
    assert admission_error("some_invented_code") is None


@pytest.mark.asyncio
async def test_other_categories_carry_no_fork_reason() -> None:
    """The field belongs to one category, not to error frames generally."""
    from local_operator.session.errors import AttachmentUnavailable

    frame = await _error_frame(AttachmentUnavailable())
    assert frame["error_code"] == AttachmentUnavailable.code
    assert "error_reason" not in frame, "only ForkRefused names a reason"
