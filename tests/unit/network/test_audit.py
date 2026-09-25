"""The audit log: one record per semantic event, bounded, and never key material."""

from __future__ import annotations

import ast
import gzip
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import relay as relay_mod
from local_operator.network.audit import AuditEvent, AuditLog
from local_operator.network.store import audit_path

NETWORK = "n_0123456789abcdef01234567"


#: One parsed record. ``Any`` and not ``object``: this is ``json.loads`` output, so the
#: values demonstrably have the shape the assertions read (``record["ts_iso"].endswith``,
#: ``record["detail"].get``) -- ``object`` was the annotation that made every reader of
#: this helper a checker error while saying nothing about the data.
def _lines(root: Path) -> list[dict[str, Any]]:
    path = audit_path(root)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_one_line_per_semantic_event(root: Path) -> None:
    log = AuditLog(root)
    for index in range(5):
        log.record(AuditEvent(event="handshake_refused", network_id=NETWORK, detail={"cause": "x"}))
        assert len(_lines(root)) >= index + 1  # durable events land at once; others batch
    log.close()
    records = _lines(root)
    assert len(records) == 5
    assert [record["seq"] for record in records] == [1, 2, 3, 4, 5]
    assert records[0]["schema"] == audit_mod.SCHEMA
    assert records[0]["ts_iso"].endswith("Z")
    assert len(records[0]["ts_iso"]) == len("2026-01-01T00:00:00.000Z")


def test_a_flush_inside_the_write_window_publishes_the_record_once(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One event, one line, even when a second flush lands mid-write.

    THE INSTRUMENT IS THE EVENT, NOT A COUNTED RACE. The relay flushes this writer
    from four threads — the accept loop, each handshake thread, the control loop and
    the heartbeat — and they used to read ONE buffer into two payloads, so the log
    held the same record twice with an identical ``seq`` and ``ts``. That was the
    whole of the cap-drop e2e flake
    (``test_relay_e2e.py::test_a_cap_drop_names_itself_in_the_local_audit``, 3 of 30
    isolated runs) and it is a corruption of the artifact an incident is
    reconstructed from: "this happened twice" is a different story from "this
    happened once". Firing threads at the writer would catch it only when the
    scheduler cooperated, so the interleaving is driven instead — the second flush
    is issued from INSIDE the window between the payload being taken and the file
    being opened, which is where two concurrent flushers actually meet.

    The file is the assertion: one ``record`` call, one line. A writer that clears
    its buffer after the disk write fails this deterministically; so does one that
    joins the buffer without a lock.
    """
    log = AuditLog(root)
    path = audit_path(root)
    real_open = Path.open
    reentered = False

    def gated_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        """``Path.open``, with a flush issued inside the audit writer's write.

        Guarded on the FIRST open of the audit path so the re-entrant flush's own
        open goes straight through — one interleaving, not a recursion.
        """
        nonlocal reentered
        handle = real_open(self, *args, **kwargs)
        if self == path and not reentered:
            reentered = True
            log.flush()
        return handle

    monkeypatch.setattr(Path, "open", gated_open)
    log.record(
        AuditEvent(event="handshake_refused", network_id=NETWORK, detail={"cause": "handshake_cap"})
    )
    assert reentered, "the interleaving never happened, so this test proved nothing"

    records = _lines(root)
    assert len(records) == 1, records
    assert records[0]["event"] == "handshake_refused"
    log.close()


def test_cost_is_per_semantic_event_and_not_per_frame(root: Path) -> None:
    """A7's prohibition, as arithmetic: N events write N records, whatever the
    traffic that produced them was. A per-frame writer would produce frames × records,
    and the relay's frame path is asserted to emit nothing at all in the e2e test."""
    log = AuditLog(root)
    events = 7
    frames = 50_000
    log.record(AuditEvent(event="link_opened", network_id=NETWORK, detail={"role": "listener"}))
    for _ in range(events - 1):
        log.record(AuditEvent(event="link_closed", network_id=NETWORK, detail={"cause": "x"}))
    log.close()
    records = _lines(root)
    assert len(records) == events
    assert frames > len(records)
    # The log's size is a function of the events, not of the frame count: 7 records
    # of a few hundred bytes each, not 50,000.
    assert audit_path(root).stat().st_size < 4 * 1024


def test_the_taxonomy_is_closed_and_self_consistent() -> None:
    """Every event with a detail whitelist is in the taxonomy, so a typo in an
    emitter's event name is caught by the drift guard rather than by a reader."""
    assert set(audit_mod.DETAIL_KEYS) <= audit_mod.EVENT_KINDS
    for kind in audit_mod.EVENT_KINDS:
        assert kind == kind.strip().lower()
    assert "handshake_refused" in audit_mod.EVENT_KINDS


def test_a_stream_close_renders_a_machine_cause_from_the_enum(root: Path) -> None:
    """Every close word in the table renders as a COUNTABLE cause, not ``internal``.

    ``_render`` substitutes ``internal`` for anything outside ``CAUSES``, so the nine
    words the stream-close call sites passed made every ``session_stream_closed`` row
    read as an internal fault on a working mesh — and the enum had zero literal
    outliers before that (agent review round 1, MAJOR 1). Written through the real
    writer rather than asserted on the table, because the substitution IS the writer.
    """
    log = AuditLog(root)
    for word in relay_mod.STREAM_CLOSE_MACHINE_CAUSES:
        log.record(
            AuditEvent(
                event="session_stream_closed",
                network_id=NETWORK,
                cause=relay_mod.stream_close_machine_cause(word),
                detail={"stream": "s_1", "peer": "d_x", "role": "owner", "cause": word},
            )
        )
    log.close()
    rows = _lines(root)
    assert len(rows) == len(relay_mod.STREAM_CLOSE_MACHINE_CAUSES)
    for row, word in zip(rows, relay_mod.STREAM_CLOSE_MACHINE_CAUSES):
        assert row["cause"] == relay_mod.STREAM_CLOSE_MACHINE_CAUSES[word], row
        assert row["cause"] in audit_mod.CAUSES, row
        assert row["cause"] not in ("", "internal"), row
        assert row["detail"]["cause"] == word, row
    # An unmapped word is the one case that DOES read as internal, and that is the whole
    # reason the table has to be complete: assert the fallback so it stays a deliberate
    # last resort rather than something a reader might mistake for a mapped value.
    assert relay_mod.stream_close_machine_cause("invented-by-a-future-call-site") == "internal"


def test_every_stream_close_word_a_call_site_passes_is_mapped() -> None:
    """The table is CLOSED against the call sites, read from relay.py's own source.

    The failure this replaces was a call site introducing a word nobody registered, so
    asserting over the table's own keys would assert nothing. An AST walk over the two
    close paths collects the literal words and the enum's values are read from the
    audit module, which is the pair that has to agree.
    """
    source = (Path(relay_mod.__file__)).read_text(encoding="utf-8")
    passed: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in {"_close_stream", "_report_stream_closed"}:
            continue
        for keyword in node.keywords:
            if keyword.arg == "cause" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str) and keyword.value.value:
                    passed.add(keyword.value.value)
        # ``_report_stream_closed(stream, cause)`` takes it positionally.
        if name == "_report_stream_closed" and len(node.args) > 1:
            second = node.args[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, str) and second.value:
                passed.add(second.value)
    # Non-empty, or the scan found nothing and would pass vacuously.
    assert passed, "no stream-close words found in relay.py: the scan lost its targets"
    unmapped = sorted(word for word in passed if word not in relay_mod.STREAM_CLOSE_MACHINE_CAUSES)
    assert not unmapped, f"close words with no machine cause: {unmapped}"
    for word, machine in relay_mod.STREAM_CLOSE_MACHINE_CAUSES.items():
        assert machine in audit_mod.CAUSES, (word, machine)


def test_unknown_detail_keys_are_dropped_and_never_raise(root: Path) -> None:
    """A lost audit record is worse than a dropped key, so an unknown key is
    DROPPED rather than fatal."""
    log = AuditLog(root)
    log.record(
        AuditEvent(
            event="link_opened",
            network_id=NETWORK,
            detail={"role": "listener", "epoch": 1, "made_up_key": "value"},
        )
    )
    log.close()
    detail = _lines(root)[0]["detail"]
    assert detail["role"] == "listener"
    assert "made_up_key" not in detail


def test_key_material_is_dropped_even_when_a_whitelist_entry_names_it(root: Path) -> None:
    """The second enforcement point: the never-list is dropped on sight, whatever a
    whitelist says, so a future whitelist mistake cannot write key material into a
    log that gets exported during an incident review."""
    log = AuditLog(root)
    detail: dict[str, object] = {"role": "listener", "epoch": 1}
    # EVERY member of the never-list, taken FROM the list rather than re-spelled
    # here: a second spelling is a key this test would silently stop covering.
    detail.update({key: "LEAKED-MATERIAL" for key in audit_mod.FORBIDDEN_DETAIL_KEYS})
    log.record(AuditEvent(event="link_opened", network_id=NETWORK, detail=detail))
    log.close()
    raw = audit_path(root).read_text(encoding="utf-8")
    assert "LEAKED-MATERIAL" not in raw
    stored = _lines(root)[0]["detail"]
    assert set(stored) == {"role", "epoch"}
    for forbidden in audit_mod.FORBIDDEN_DETAIL_KEYS:
        assert forbidden not in stored


def test_a_long_detail_map_is_truncated_and_says_so(root: Path) -> None:
    log = AuditLog(root)
    log.record(
        AuditEvent(
            event="epoch_rotated",
            network_id=NETWORK,
            detail={"removed": ["d_" + "a" * 200 for _ in range(40)]},
        )
    )
    log.close()
    detail = _lines(root)[0]["detail"]
    assert detail.get("truncated") is True
    assert len(json.dumps(detail)) <= audit_mod.MAX_DETAIL_BYTES


def test_control_characters_are_stripped_from_a_string_field(root: Path) -> None:
    log = AuditLog(root)
    log.record(
        AuditEvent(event="link_closed", network_id=NETWORK, detail={"cause": "a\nb\r\u0000c"})
    )
    log.close()
    assert _lines(root)[0]["detail"]["cause"] == "abc"


def test_durable_events_are_written_through_immediately(root: Path) -> None:
    """A panic record sitting in a buffer when the machine is cut is exactly the
    record an incident review needed."""
    log = AuditLog(root)
    log.record(AuditEvent(event="panic_raised", network_id=NETWORK, detail={"epoch_after": 9}))
    assert _lines(root) != []
    # A non-durable event waits for the tick or the byte budget.
    log.record(AuditEvent(event="link_closed", network_id=NETWORK, detail={"cause": "x"}))
    assert len(_lines(root)) == 1
    log.close()
    assert len(_lines(root)) == 2


def test_the_log_is_0600_and_its_directory_0700(root: Path) -> None:
    log = AuditLog(root)
    log.record(AuditEvent(event="panic_raised", network_id=NETWORK))
    path = audit_path(root)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_rotation_closes_a_generation_and_records_that_it_did(root: Path) -> None:
    """The log records its own truncation — the one thing an attacker with disk
    access would most like to hide."""
    log = AuditLog(root, max_bytes=200, generations=3)
    for index in range(40):
        log.record(
            AuditEvent(event="link_closed", network_id=NETWORK, detail={"cause": f"c{index}"})
        )
    log.close()
    generations = sorted(audit_path(root).parent.glob("audit.jsonl.*.gz"))
    assert generations, "no generation was closed"
    with gzip.open(generations[0], "rb") as handle:
        assert b"link_closed" in handle.read()
    assert "audit_rotated" in [record["event"] for record in _lines(root)]


def test_retention_is_bounded_by_construction(root: Path) -> None:
    log = AuditLog(root, max_bytes=120, generations=2)
    for index in range(200):
        log.record(
            AuditEvent(event="link_closed", network_id=NETWORK, detail={"cause": f"c{index}"})
        )
    log.close()
    generations = sorted(audit_path(root).parent.glob("audit.jsonl.*.gz"))
    assert len(generations) <= 2 + 1  # the cap, plus the one being written
    # And the pruned generation is itself recorded.
    assert "audit_pruned" in [record["event"] for record in _lines(root)]


def test_the_sequence_resumes_after_a_restart(root: Path) -> None:
    first = AuditLog(root)
    first.record(AuditEvent(event="panic_raised", network_id=NETWORK))
    first.close()
    second = AuditLog(root)
    second.record(AuditEvent(event="panic_raised", network_id=NETWORK))
    second.close()
    assert [record["seq"] for record in _lines(root)] == [1, 2]


def test_a_write_failure_degrades_the_writer_without_raising(root: Path) -> None:
    """A relay that died when its log filled would be a relay an attacker can kill
    with a full disk."""
    # The LOG PATH is a directory, so the append cannot succeed and nothing about
    # the failure is undoable by the writer (an owner always re-chmods their own
    # directory, which would have made a chmod-based version of this test vacuous).
    log = AuditLog(root)
    audit_path(root).parent.mkdir(parents=True, exist_ok=True)
    audit_path(root).mkdir()
    log.record(AuditEvent(event="panic_raised", network_id=NETWORK))
    log.close()
    assert log.degraded is True
    assert log.degraded_reason != ""


def test_tail_filters_by_network_and_time(root: Path) -> None:
    log = AuditLog(root)
    log.record(AuditEvent(event="panic_raised", network_id="n_one", ts=1000.0))
    log.record(AuditEvent(event="panic_raised", network_id="n_two", ts=2000.0))
    log.close()
    assert len(log.tail(10)) == 2
    assert len(log.tail(10, network_id="n_one")) == 1
    assert len(log.tail(10, since=1500.0)) == 1
    assert log.tail(1)[0]["network_id"] == "n_two"
