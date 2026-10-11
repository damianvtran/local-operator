"""The shared quiet-group parity fixture, replayed through the TUI derivation
(quiet-turn design §5 + §8 S5).

WHY THIS FILE EXISTS. A quiet group is client-derived: there is no wire kind
and no capability flag, so the risk this suite closes is silent drift between
surfaces — the UI's ``quietGroupsOf``/``quietGroupOfSegment`` (local-operator-ui,
``scripts/turn-collapse-model.test.mjs``), the relay-web port (S4) and this TUI
port are three implementations of one definition. The fixture is the contract:
every suite reads the same bytes, so a change to either side fails in its own
tree rather than diverging with both green. The copy here is byte-identical to
the shared file and hash-pinned below, not a fork.

ONE CASE IS NOT REPLAYED, and it is named rather than filtered silently: the
fixture's job-only case describes ``custom`` job-result rows, and this surface
has no job receipt row — a delivered job result paints nothing and a held one
paints a generic notice, so it splits and stays visible (pinned by the
skip-list test below; the fixture keeps the case for the surfaces that can
express it). The monitor-only case IS replayed: a monitor delta is its own
receipt row here (``MonitorDeltaBlock``), so unlike the relay the TUI can
positively classify it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.tui.quiet_groups import (
    QuietGroup,
    QuietGroupRecord,
    quiet_group_label,
    quiet_group_of_segment,
    quiet_groups_of,
    quiet_sender_label,
)

#: The shared fixture's copy, at the same path the relay-web slice (S4) uses —
#: ONE path for the one contract in this repository. When both slices are on
#: main the byte-identical copies fold to a single file; the hash pin below is
#: what makes that fold provable rather than assumed.
FIXTURE_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "local_operator"
    / "mobile"
    / "web"
    / "src"
    / "lib"
    / "quiet-groups.parity.json"
)
FIXTURE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf8"))

#: The fixture cases this surface cannot express: the case describes `custom`
#: job-result rows, and the TUI has no job receipt row (see the module doc).
NOT_EXPRESSIBLE_CASES = [
    "a job-only run states the job family",
]

EXPRESSIBLE_CASES = [case for case in FIXTURE["cases"] if case["name"] not in NOT_EXPRESSIBLE_CASES]


def _record_of(row: dict[str, Any]) -> QuietGroupRecord:
    """The TUI mapping of one fixture row (the fixture's own instruction: "A
    client with a different record type maps these onto its own shape").

    Deviations from the row, and why:

    - ``sender.conversationName`` becomes the TUI sender mapping's
      ``conversation_name`` (the TUI reads the persisted snake-case fields);
    - a tool's ``isError`` becomes ``tool_state: "failed"`` — this surface has
      no isError flag, the card's settled ``error`` mark is the failure
      verdict, and ``interrupted`` stays its own state (not a failure);
    - a ``custom`` row is only ever the monitor receipt here
      (``customType: monitor_prompt``); the job case is skipped before mapping,
      so reaching the raise means the skip list drifted;
    - ``notice``/``compaction``/``user``/``assistant`` map by kind alone —
      this surface splits on all of them, which is exactly what the fixture
      pins.
    """
    kind = row["kind"]
    record_id = row["id"]
    ts = row.get("ts")
    if kind == "peer":
        sender = row.get("sender") or {}
        mapped_sender = (
            {"conversation_name": sender["conversationName"]}
            if "conversationName" in sender
            else {}
        )
        return QuietGroupRecord(kind="peer", id=record_id, ts=ts, sender=mapped_sender)
    if kind == "wake":
        return QuietGroupRecord(kind="wake", id=record_id, ts=ts, text=row.get("text", ""))
    if kind == "tool":
        return QuietGroupRecord(
            kind="tool",
            id=record_id,
            ts=ts,
            tool_name=row.get("toolName", ""),
            tool_state="failed" if row.get("isError") else "done",
        )
    if kind == "custom":
        if row.get("customType") == "monitor_prompt":
            return QuietGroupRecord(kind="monitor", id=record_id, ts=ts, text=row.get("text", ""))
        raise AssertionError(f"no TUI mapping for a custom row of type {row.get('customType')!r}")
    return QuietGroupRecord(kind=kind, id=record_id, ts=ts, text=row.get("text", ""))


def _as_fixture(group: QuietGroup | None) -> Any:
    """The fixture's own JSON shape for a derived group, so the comparison is
    against the shared contract rather than against this port's field names."""
    if group is None:
        return None
    return {
        "key": group.key,
        "family": group.family,
        "count": group.count,
        "firstTs": group.first_ts,
        "lastTs": group.last_ts,
        "senders": [{"label": sender.label, "count": sender.count} for sender in group.senders],
        "actions": group.actions,
        "failed": group.failed,
        "open": group.open,
        "rowIds": list(group.row_ids),
    }


def test_the_copied_fixture_stays_byte_identical_to_the_shared_contract() -> None:
    """The copy IS the contract: every suite must replay the same bytes, so a
    local edit of the fixture (rather than a coordinated cross-client re-pin)
    fails here. The digest pins the copy taken from local-operator pull request
    #2160's head (``feat/quiet-turn-relay-group``), itself copied from
    local-operator-ui PR #945's branch (``f9b041f8``); the two copies are
    byte-identical by construction."""
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == "3e5dca2158ef4103ff0e905e979daae2cb4dd61f4e379fecfb03cb9a53631dac"


@pytest.mark.parametrize(
    "case", EXPRESSIBLE_CASES, ids=[case["name"] for case in EXPRESSIBLE_CASES]
)
def test_fixture_case_replays(case: dict[str, Any]) -> None:
    """ONE TEST PER CASE, named with the fixture's own name: a case that fails
    carries its scenario in the report instead of a bare index."""
    records = [_record_of(row) for row in case["rows"]]
    if "span" in case:
        group = quiet_group_of_segment(
            records,
            tuple(case["span"]),
            span_head_loaded=case.get("spanHeadLoaded", True),
            open=case.get("open"),
        )
        assert _as_fixture(group) == case["expected"]
        return
    assert [_as_fixture(group) for group in quiet_groups_of(records)] == case["expected"]


def test_skips_exactly_the_cases_this_surface_cannot_express() -> None:
    """The skip list must name EXACTLY the unexpressible cases — no more
    (nothing here proves another case skippable), no fewer (a skipped case
    proves nothing). Both halves read the same fixture."""
    unexpressible = [
        case["name"]
        for case in FIXTURE["cases"]
        if any(
            row["kind"] == "custom" and row.get("customType") != "monitor_prompt"
            for row in case["rows"]
        )
    ]
    assert unexpressible == NOT_EXPRESSIBLE_CASES
    assert len(EXPRESSIBLE_CASES) + len(NOT_EXPRESSIBLE_CASES) == len(FIXTURE["cases"])


def test_a_held_job_result_splits_rather_than_folding() -> None:
    """The surface's own shape of the skipped case: a held job result paints a
    generic notice, and an unclassifiable receipt must stay visible rather than
    hiding inside a group — so it splits the span it sits in."""
    records = [
        _peer("p1"),
        QuietGroupRecord(kind="notice", id="j2", text="job done"),
        _peer("p3"),
        _peer("p4"),
    ]
    groups = quiet_groups_of(records)
    assert [group.key for group in groups] == ["qg:p3"]


def test_an_inside_row_sits_inside_without_being_counted() -> None:
    """This surface's ``inside`` kind (an image a folded tool row produced) may
    sit inside a group — it is neither a trigger nor countable work, so it
    must not split the span and must not move the count."""
    records = [
        _peer("p1"),
        QuietGroupRecord(kind="inside", id="i2"),
        _peer("p3"),
    ]
    (group,) = quiet_groups_of(records)
    assert group.row_ids == ("p1", "i2", "p3")
    assert group.count == 2
    assert group.actions == 0


def test_the_sender_ladder_falls_through_rather_than_raising() -> None:
    """The ladder is total over partial senders: every missing field falls to
    the next rung, and the last rung is the shared ``another session``."""
    assert quiet_sender_label({}) == "another session"
    assert quiet_sender_label({"pid": 42}) == "pid 42"
    assert quiet_sender_label({"session_id": "abcdefghijkl"}) == "abcdefgh"
    assert quiet_sender_label({"cwd": "/home/dev/tools/"}) == "tools/"
    assert quiet_sender_label({"conversation_name": "release-window"}) == '"release-window"'
    assert quiet_sender_label({"conversation_name": "named", "cwd": "/a/b", "pid": 1}) == '"named"'


def test_facts_freeze_when_closed_and_only_the_open_tail_grows() -> None:
    """The latch, expressed as the derivation's own property: a closed group's
    boundary is fixed — later appends land outside it — so its facts cannot
    move again, and the open tail is the one that grows."""
    rows = [_peer("p1"), _peer("p2"), _user("u3")]
    before = quiet_groups_of(rows)
    assert len(before) == 1 and before[0].open is False

    after = quiet_groups_of([*rows, _peer("p4"), _peer("p5")])
    assert len(after) == 2
    assert after[0] == before[0], "the closed group's facts moved"
    assert after[1].key == "qg:p4" and after[1].open is True

    grown = quiet_groups_of([*rows, _peer("p4"), _peer("p5"), _peer("p6")])
    assert grown[0] == before[0]
    assert grown[1].key == "qg:p4", "the open tail's key moved while growing"
    assert grown[1].count == 3


def test_derives_twice_identically() -> None:
    """Stability: the derivation is a pure function of the rows on hand, so
    replaying the same rows yields identical facts — and re-deriving over a
    grown list keeps the closed prefix identical (the property the on-screen
    fold's latch mirrors)."""
    rows = [
        _peer("p1"),
        _peer("p2"),
        _user("u3"),
        _peer("p4"),
        QuietGroupRecord(kind="tool", id="t5", tool_name="bash", tool_state="done"),
        _peer("p6"),
    ]
    first = quiet_groups_of(rows)
    second = quiet_groups_of([_record_like(row) for row in rows])
    assert first == second

    grown = quiet_groups_of([*rows, _peer("p7")])
    assert grown[0] == first[0], "the closed group's facts moved"
    assert grown[-1].key == first[-1].key, "the open tail's key moved"
    assert grown[-1].count == 3


def _peer(record_id: str) -> QuietGroupRecord:
    return QuietGroupRecord(kind="peer", id=record_id)


def _user(record_id: str) -> QuietGroupRecord:
    return QuietGroupRecord(kind="user", id=record_id, text="steer")


def _record_like(record: QuietGroupRecord) -> QuietGroupRecord:
    """A fresh but equal record — so the second derive starts from its own
    objects and a shared-mutable-state bug cannot hide in an alias."""
    return QuietGroupRecord(
        kind=record.kind,
        id=record.id,
        ts=record.ts,
        text=record.text,
        sender=dict(record.sender),
        tool_name=record.tool_name,
        tool_state=record.tool_state,
    )


def test_states_the_family_words_exactly_once_for_every_family() -> None:
    assert quiet_group_label("peer") == "Peer messages"
    assert quiet_group_label("wake") == "Wake messages"
    assert quiet_group_label("monitor") == "Monitor messages"
    assert quiet_group_label("job") == "Job results"
    assert quiet_group_label("mixed") == "Messages"
