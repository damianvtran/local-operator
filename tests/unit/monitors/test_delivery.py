"""The delivery envelope (contract §9.1): its clauses, and nothing else."""

from __future__ import annotations

from typing import Any

from local_operator.monitors.delivery import (
    MonitorDelivery,
    format_monitor_delivery_text,
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
