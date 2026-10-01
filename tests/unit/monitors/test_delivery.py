"""The delivery envelope (contract §9.1): its clauses, and nothing else."""

from __future__ import annotations

from typing import Any

from local_operator.monitors.delivery import (
    NOTICE_KINDS,
    NOTICE_MAX_CHARS,
    MonitorDelivery,
    MonitorNotice,
    format_monitor_delivery_text,
    format_monitor_notice_text,
)


def delivery(**overrides: Any) -> MonitorDelivery:
    base = dict(
        monitor_id="m1",
        name="loom-pr-1710",
        tool="bash",
        changes=3,
        checks=41,
        skipped=0,
        delta_text="+2/-1 changed lines\n- FAILURE\n+ SUCCESS",
        at_ms=1_756_000_000_000,
        held_by_cap=0,
        final=False,
        description="",
    )
    base.update(overrides)
    return MonitorDelivery(**base)  # type: ignore[arg-type]


def test_the_envelope_names_id_counts_clock_and_cancel() -> None:
    from datetime import datetime

    clock = datetime.fromtimestamp(1_756_000_000_000 / 1000).strftime("%H:%M")
    text = format_monitor_delivery_text(delivery())
    lines = text.splitlines()
    assert lines[0] == (f"(monitor) 'loom-pr-1710' m1 (via bash): 3 changes at {clock} — check 41.")
    assert lines[1] == 'Cancel with monitor({op:"cancel",id:"m1"}) once its goal is met.'
    assert lines[2] == ""
    assert lines[3] == "Diff vs the previous check:"
    assert lines[4].startswith("+2/-1 changed lines")


def test_a_bash_named_monitor_omits_the_source_note() -> None:
    text = format_monitor_delivery_text(delivery(name="bash-pr-check"))
    assert "(via bash)" not in text


def test_skipped_and_held_clauses_ride_the_first_line() -> None:
    text = format_monitor_delivery_text(delivery(skipped=2, held_by_cap=3))
    first = text.splitlines()[0]
    assert "(2 skipped while the session was down;" in first
    assert "3 earlier changes were held by the hourly cap)" in first


def test_the_cancel_hint_is_dropped_on_the_final_delivery() -> None:
    text = format_monitor_delivery_text(delivery(final=True))
    assert "Cancel with monitor" not in text


def test_the_description_gets_its_own_line_when_set() -> None:
    with_description = format_monitor_delivery_text(delivery(description="tell me when CI flips"))
    assert "Watching for: tell me when CI flips" in with_description
    without = format_monitor_delivery_text(delivery())
    assert "Watching for" not in without


def test_singular_counts_and_the_no_clock_dependency() -> None:
    text = format_monitor_delivery_text(
        delivery(changes=1, delta_text="+1/-0 changed lines\n+ x", at_ms=0)
    )
    assert "1 change at" in text


# ---------------------------------------------------------------------------
# §D4: lifecycle notices — bounded, and never mistakable for a delivery
# ---------------------------------------------------------------------------


def notice(kind: str = "disabled", **overrides: Any) -> MonitorNotice:
    fields: dict[str, Any] = {
        "monitor_id": "m1",
        "name": "watch",
        "tool": "bash",
        "kind": kind,
        "at_ms": 1_756_000_000_000,
        "checks": 7,
        "deliveries": 0,
        "failures": 5,
        "detail": "check timed out after 120s",
    }
    fields.update(overrides)
    return MonitorNotice(**fields)


def test_the_disable_notice_names_the_id_the_cause_and_the_restore_path() -> None:
    text = format_monitor_notice_text(notice())
    assert "was DISABLED" in text
    assert "5 consecutive failed checks" in text
    assert "Last error: check timed out after 120s" in text
    assert 'monitor({op:"cancel",id:"m1"})' in text
    # The hosting caveat rides the notice as well as the arm receipt (§D8):
    # this is the message an operator reads when a watch stops silently.
    assert "Monitors tick only while this session is open." in text


def test_the_zero_delivery_clause_appears_only_at_zero() -> None:
    never = format_monitor_notice_text(notice(deliveries=0))
    assert "It never delivered a change since arming (7 checks)." in never
    with_deliveries = format_monitor_notice_text(notice(deliveries=3))
    assert "never delivered" not in with_deliveries
    assert "3 delivers of change so far." in with_deliveries
    assert "1 deliver of change so far." in format_monitor_notice_text(notice(deliveries=1))


def test_the_stalled_notice_names_the_remedy_and_the_baseline() -> None:
    text = format_monitor_notice_text(
        notice(
            kind="stalled",
            detail='MCP server "datadog" needs re-authentication — run /mcp reauth',
        )
    )
    assert "could not run its check" in text
    assert "/mcp reauth" in text
    assert "without counting failures" in text
    assert "one delta" in text


def test_the_restored_notice_closes_the_episode() -> None:
    text = format_monitor_notice_text(notice(kind="restored"))
    assert "is running again" in text
    assert "old baseline" in text


def test_every_notice_kind_is_bounded_and_one_block() -> None:
    """A notice is a PUSH into the conversation, so its size is a contract: a
    long failure reason is clipped rather than allowed to turn a disable into a
    wall of text.
    """
    for kind in NOTICE_KINDS:
        text = format_monitor_notice_text(
            notice(
                kind,
                name="x" * 200,
                tool="mcp__" + "y" * 200,
                detail="boom " * 500,
            )
        )
        assert len(text) <= NOTICE_MAX_CHARS, (kind, len(text))
        assert text.count("\n") <= 6, kind


def test_the_notice_error_is_clipped_to_one_line() -> None:
    text = format_monitor_notice_text(notice(detail="first line\nsecond line\t" + "z" * 400))
    assert "\n" not in text.split("Last error: ")[1].split("\n")[0]
    assert "second line" in text  # whitespace-collapsed, not dropped
    assert len(text) <= NOTICE_MAX_CHARS
