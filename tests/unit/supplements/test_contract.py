"""The frozen contract: rows, events, the stale-row rule, the wire (lane C0).

Every fixture under ``tests/fixtures/supplements`` is parsed by the REAL type and
re-serialised, and must come back byte-stable. That is the property the other lanes rely
on: a fixture a lane can load is a fixture the producer's own types accept, and one that
re-serialises differently is a shape the producer would not have written.
"""

from __future__ import annotations

import json
from typing import Any, get_args, get_type_hints

import pytest

from local_operator.harness.types import AgentEvent, SupplementProgressEvent
from local_operator.session.attached import deserialize_event
from local_operator.session.transcript import ENTRY_CUSTOM, TranscriptEntry
from local_operator.supplements import contract
from local_operator.supplements.contract import (
    JOURNALED_STATES,
    LIVE_ONLY_STATES,
    SUPPLEMENT_CUSTOM_TYPE,
    SupplementDetails,
    SupplementRowState,
    SupplementState,
    accept_frame_message,
    is_digest,
    negotiated,
    newest_per_anchor,
    reader_disposition,
)
from tests.unit.supplements.conftest import FIXTURES, load, raw

ROW_NAMES = [
    "decided", "queued_stale", "done_populated", "done_files_only", "done_empty",
    "failed", "cancelled", "superseded", "dismissed",
]  # fmt: skip
EVENT_NAMES = sorted(p.stem for p in (FIXTURES / "events").glob("*.json"))


def _canonical(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


# --- rows -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROW_NAMES)
def test_a_row_fixture_round_trips_through_the_transcript_entry_byte_stable(name: str) -> None:
    text = raw(f"rows/{name}.json")
    data = json.loads(text)
    # the REAL journal line type: what the writer would append and the reader would parse
    entry = TranscriptEntry.from_json(json.dumps(data))
    assert entry is not None
    assert entry.type == ENTRY_CUSTOM == data["type"]
    assert entry.payload["custom_type"] == SUPPLEMENT_CUSTOM_TYPE == "supplement_v1"
    again = json.loads(entry.to_json())
    assert again == data
    assert _canonical(again) == text


@pytest.mark.parametrize("name", ROW_NAMES)
def test_a_row_carries_the_contract_keys_and_no_unknown_ones(name: str) -> None:
    details = load(f"rows/{name}.json")["payload"]["details"]
    hints = get_type_hints(SupplementDetails)
    assert set(details) <= set(hints), set(details) - set(hints)
    assert SupplementDetails.__required_keys__ <= set(details)
    assert details["state"] in get_args(SupplementRowState)
    assert details["state"] in JOURNALED_STATES
    assert details["state"] not in LIVE_ONLY_STATES
    assert len(details["job"]) == 12 and int(details["job"], 16) >= 0


@pytest.mark.parametrize("name", ROW_NAMES)
def test_row_paths_are_never_absolute_and_digests_use_the_literal_attachment_key(
    name: str,
) -> None:
    details = load(f"rows/{name}.json")["payload"]["details"]
    for item in details["files"]:
        assert not item["path"].startswith("/"), "absolute paths replicate the operator's layout"
    for path in details.get("more", []):
        assert not path.startswith("/")
    for component in details["components"]:
        assert is_digest(component["attachment"])
        assert component["mime"] == "text/html"
        assert contract.HEIGHT_HINT_MIN <= component["height_hint"] <= contract.HEIGHT_HINT_MAX
    assert len(details.get("more", [])) <= contract.MORE_MAX
    assert len(details["components"]) <= contract.MAX_COMPONENTS


def test_sync_finds_the_component_digest_by_its_literal_key() -> None:
    """Memo §2.4: move/sync's byte regex ships the blob with no sync change."""
    import re

    pattern = re.compile(rb'"attachment"\s*:\s*"([0-9a-f]{32})"')
    line = json.dumps(load("rows/done_populated.json"), separators=(",", ":")).encode()
    assert pattern.findall(line) == [load("components/digests.json")["populated"].encode()]


def test_the_component_digest_is_the_content_address_of_the_stored_blob() -> None:
    """The fixture digests are computed the way ``AttachmentStore.put_bytes`` does."""
    import tempfile
    from pathlib import Path

    from local_operator.session.attachments import AttachmentStore

    digests = load("components/digests.json")
    with tempfile.TemporaryDirectory() as home:
        store = AttachmentStore(Path(home))
        for name, digest in digests.items():
            blob = (FIXTURES / "components" / f"{name}.html").read_bytes()
            if not blob:
                continue  # put_bytes refuses empty input by contract
            ref = store.put_bytes(blob, "text/html")
            assert ref is not None and ref.digest == digest, name


def test_newest_version_per_anchor_wins_and_older_rows_remain_an_audit_trail() -> None:
    journal = load("rows/journal_versions.json")
    assert [r["payload"]["details"]["version"] for r in journal] == [1, 2]
    newest = newest_per_anchor(r["payload"]["details"] for r in journal)
    assert list(newest) == [journal[0]["payload"]["details"]["anchor"]]
    only = next(iter(newest.values()))
    assert only["version"] == 2 and only["state"] == "done"


def test_the_disposition_fixture_is_the_reader_rule_and_covers_the_stale_row() -> None:
    expected = load("rows/dispositions.json")
    assert set(expected) == set(ROW_NAMES)
    for name in ROW_NAMES:
        details = load(f"rows/{name}.json")["payload"]["details"]
        assert reader_disposition(details, job_live=True) == expected[name]["live"], name
        assert reader_disposition(details, job_live=False) == expected[name]["cold"], name


def test_the_stale_row_rule_queued_with_no_live_job_reads_cancelled_retry() -> None:
    """THE fixture every lane asserts against (memo §2.4, round-1 R4)."""
    stale = load("rows/queued_stale.json")["payload"]["details"]
    assert stale["state"] == "queued"
    assert reader_disposition(stale, job_live=False) == "cancelled_retry"
    assert reader_disposition(stale, job_live=True) == "preparing"


def test_a_superseded_row_renders_nothing_whatever_its_state() -> None:
    row = load("rows/superseded.json")["payload"]["details"]
    assert row["error"] == contract.SUPERSEDED_ERROR
    assert reader_disposition(row, job_live=False) == "nothing"
    assert reader_disposition({**row, "state": "failed"}, job_live=False) == "nothing"


# --- events -----------------------------------------------------------------------------


def test_every_state_of_the_vocabulary_has_an_event_fixture() -> None:
    seen = {load(f"events/{name}.json")["state"] for name in EVENT_NAMES}
    assert seen == set(get_args(SupplementState)) == JOURNALED_STATES | LIVE_ONLY_STATES


@pytest.mark.parametrize("name", EVENT_NAMES)
def test_an_event_fixture_round_trips_through_the_event_class_byte_stable(name: str) -> None:
    text = raw(f"events/{name}.json")
    event = SupplementProgressEvent.model_validate(json.loads(text))
    assert _canonical(event.model_dump(mode="json")) == text


@pytest.mark.parametrize("name", EVENT_NAMES)
def test_a_follower_rehydrates_the_event_as_its_concrete_class(name: str) -> None:
    """`_EVENT_TYPES` registration: without it a relayed event degrades to the base class."""
    event = deserialize_event(load(f"events/{name}.json"))
    assert type(event) is SupplementProgressEvent


def test_an_old_follower_keeps_the_event_as_a_tolerant_base_event() -> None:
    """The additive-event story: `extra="allow"`, no PROTOCOL_VERSION bump."""
    payload = load("events/running_generating.json")
    base = AgentEvent.model_validate(payload)
    assert base.type == "supplement_progress"
    assert base.model_dump()["anchor"] == payload["anchor"]


def test_the_event_rejects_a_state_outside_the_vocabulary() -> None:
    payload = {**load("events/queued.json"), "state": "thinking"}
    with pytest.raises(ValueError):
        SupplementProgressEvent.model_validate(payload)


def test_live_only_states_never_appear_in_a_row_and_rows_never_use_other_words() -> None:
    assert set(get_args(SupplementRowState)) == JOURNALED_STATES
    assert set(get_args(SupplementState)) - set(get_args(SupplementRowState)) == LIVE_ONLY_STATES


def test_an_event_does_not_reach_exec_json_producers() -> None:
    """Rule 6: `lop exec` never produces one. C0 adds no producer at all."""
    import local_operator.headless_print as printing

    source = open(printing.__file__, encoding="utf-8").read()
    assert "supplement" not in source


# --- the wire ---------------------------------------------------------------------------


MESSAGES = load("messages/messages.json")
#: The nonce the fixture host minted in its first theme push.
NONCE = MESSAGES["host"]["theme"]["nonce"]


@pytest.mark.parametrize("name", sorted(MESSAGES["frame_accepted"]))
def test_the_accepted_frame_shapes_are_exactly_ready_resize_error_pong(name: str) -> None:
    assert name in contract.FRAME_MESSAGE_TYPES
    message = MESSAGES["frame_accepted"][name]
    # `ready` is the only shape accepted before a nonce exists
    assert accept_frame_message(message, None) == (message if name == "ready" else None)
    assert accept_frame_message(message, NONCE) == message


def test_the_accepted_list_is_closed() -> None:
    assert contract.FRAME_MESSAGE_TYPES == ("ready", "resize", "error", "pong")
    assert set(MESSAGES["frame_accepted"]) == set(contract.FRAME_MESSAGE_TYPES)
    assert contract.NONCE_REQUIRED_TYPES == ("resize", "error", "pong")


@pytest.mark.parametrize("name", sorted(MESSAGES["frame_rejected"]))
def test_every_rejected_frame_shape_is_dropped(name: str) -> None:
    assert accept_frame_message(MESSAGES["frame_rejected"][name], NONCE) is None


def test_a_non_mapping_is_dropped() -> None:
    for junk in (None, "pong", 3, [], ["supplement"]):
        assert accept_frame_message(junk, NONCE) is None


def test_the_host_messages_are_the_theme_push_and_the_ping() -> None:
    theme, ping = MESSAGES["host"]["theme"], MESSAGES["host"]["ping"]
    assert theme["lo"] == ping["lo"] == contract.HOST_MESSAGE_TAG
    assert theme["t"] == "theme" and theme["mode"] in ("light", "dark")
    assert theme["nonce"] == NONCE
    assert ping == {"lo": "supplement-host", "t": "ping"}
    import re

    for name, value in theme["vars"].items():
        assert re.fullmatch(r"--(lo|font)-[a-z0-9-]+", name) and len(value) <= 120


# --- capability strings and the gate ----------------------------------------------------


def test_the_capability_strings_are_the_memos() -> None:
    assert contract.SUPPLEMENTS_CAPABILITY == "supplements-v1"
    assert contract.SUPPLEMENTS_FEATURE_KEY == "supplements"
    assert contract.SUPPLEMENTS_FEATURE_VERSION == 1
    assert contract.SUPPLEMENT_CONTROL_OPS == (
        "supplement_cancel", "supplement_steer", "supplement_restart", "supplement_dismiss",
    )  # fmt: skip
    assert contract.SUPPLEMENTS_READ_OP == "supplements_for"


def test_the_attach_gate_needs_both_halves() -> None:
    owner = ["display-history-window-v1", "supplements-v1"]
    assert negotiated(owner, True) is True
    assert negotiated(owner, False) is False  # an older viewer never declared it
    assert negotiated(["display-history-window-v1"], True) is False  # an older owner
    assert negotiated([], False) is False


def test_the_spending_ops_are_not_sync_priority_ops() -> None:
    """Memo §2.7: steer/restart spend money, so they must not jump the sync queue."""
    from local_operator.session.runtime.server import _SYNC_PRIORITY_OPS

    assert not set(contract.SUPPLEMENT_OPS) & set(_SYNC_PRIORITY_OPS)


def test_the_viewer_declaration_is_forwarded_across_the_mesh() -> None:
    """The entry-times declaration once shipped inert for want of this allowlist line."""
    from local_operator.network import dial

    assert contract.SUPPLEMENTS_AUTH_FIELD in dial.AUTH_FIELDS


def test_the_contract_module_is_a_stdlib_only_leaf() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse(Path(contract.__file__).read_text(encoding="utf-8"))
    roots = {
        (node.module or "").split(".")[0]
        if isinstance(node, ast.ImportFrom)
        else alias.name.split(".")[0]
        for node in ast.walk(tree)
        for alias in (node.names if isinstance(node, (ast.Import, ast.ImportFrom)) else [])
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }  # fmt: skip
    assert roots <= {"__future__", "math", "re", "typing"}, roots
