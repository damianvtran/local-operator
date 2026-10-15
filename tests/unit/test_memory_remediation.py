"""Round-1 remediation regressions for PR #2414 (each fails on head 08e6f2e0).

R1 locator ReDoS, R2 fail-open prompt tail + FTS5 probe, Q1 invisible-character guard
bypass across every carrier, R3 forget at the op cap, Q3 alias stoplist, Q7 single-slash
``memory:`` routing. Timing is never asserted: R1 is pinned structurally.
"""

import json
import logging
import sqlite3

import pytest

import local_operator.memory as memory
from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    ToolContext,
)
from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore
from local_operator.memory_capture import _LOCATOR, extract
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.memory_tools import execute_memory


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        tmp_path / "cfg",
        MemoryIdentity("rem-session", "team-r", "agent-r"),
        tmp_path / "sessions" / "rem-session",
        origin="rem-device",
    )


# ---- R1 ---------------------------------------------------------------------------------


def test_locator_regex_is_bounded_and_scans_in_fixed_windows():
    from local_operator.memory_capture import SCAN_WINDOW, windows

    # Structural invariant instead of a wall-clock assertion: no unbounded quantifier may
    # sit in front of the "@" (that was the quadratic shape), and no regex call may see
    # more than SCAN_WINDOW characters whatever the input size.
    email = next(alt for alt in _LOCATOR.pattern.split("|") if "@" in alt)
    assert "+]+" not in email and "]+@" not in email and "]+\\." not in email, email
    pieces = list(windows("a" * 40_000 + " " + "b" * 40_000))
    assert pieces and all(len(piece) <= SCAN_WINDOW for piece in pieces)
    assert "".join(pieces).replace(" ", "") == "a" * 40_000 + "b" * 40_000


def test_pathological_inputs_extract_nothing_and_do_not_blow_up():
    for text in ("a" * 32768, "a@" + "b" * 32768, "a" * 16384 + "b" * 16384, "a.-" * 11000):
        assert extract(text) == []
    # Legitimate locators are still found, including inside a long run of filler.
    found = extract("x" * 30000 + " see docs/plan.md and ops@example.com " + "y" * 30000)
    assert {row["body"] for row in found} == {"docs/plan.md", "ops@example.com"}


def test_capture_keeps_short_shas_and_drops_bare_long_hex():
    found = {
        row["body"]
        for row in extract("fix in a3bf4f1b2c3d; key 0123456789abcdef0123456789abcdef01234567")
    }
    assert "a3bf4f1b2c3d" in found
    assert not any(len(body) > 12 and body.isalnum() for body in found), found


# ---- R2 ---------------------------------------------------------------------------------


def test_missing_fts5_is_a_clear_refusal_and_leaves_no_database(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "fts5_available", lambda: False)
    store = MemoryStore(tmp_path / "cfg", MemoryIdentity("s1"), origin="d1")
    with pytest.raises(MemoryError, match="FTS5"):
        store.put("global", {"title": "t", "body": "docs/a.md"})
    assert not store.path.exists() and not store.root.exists() or not list(store.root.iterdir())


def test_failed_schema_does_not_leave_an_initialised_looking_database(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "cfg", MemoryIdentity("s1"), origin="d1")
    monkeypatch.setattr(memory, "_SCHEMA", "CREATE VIRTUAL TABLE x USING nonexistent_module(y);")
    with pytest.raises(MemoryError, match="unavailable"):
        store.put("global", {"title": "t", "body": "docs/a.md"})
    assert not store.path.exists()


def _session(tmp_path, requests):
    root = tmp_path / "cfg"
    root.mkdir(exist_ok=True)
    (root / "config.yml").write_text(
        json.dumps({"values": {"memory": {"enabled": True, "capture": True, "suggest": True}}})
    )
    transcript = Transcript(root / "sessions" / "tail-session")

    def stream(request, signal):
        requests.append(request)

        async def events():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    session = Session(
        model=ModelSpec(provider="test", model_id="deterministic", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id="tail-session",
    )
    session._config_dir = root
    return session, root, transcript


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["operational", "locked", "corrupt", "full"])
async def test_store_failure_never_aborts_the_provider_request(
    tmp_path, monkeypatch, caplog, failure
):
    requests: list = []
    session, root, transcript = _session(tmp_path, requests)
    seed = MemoryStore(root, MemoryIdentity("tail-session"), transcript.directory, origin="d")
    seed.put("global", {"title": "Mercury deployment", "body": "Mercury deploy docs/m.md"})
    errors = {
        "operational": sqlite3.OperationalError("no such module: fts5"),
        "locked": sqlite3.OperationalError("database is locked"),
        "corrupt": sqlite3.DatabaseError("database disk image is malformed"),
        "full": sqlite3.OperationalError("database or disk is full"),
    }

    def broken(self):
        raise errors[failure]

    monkeypatch.setattr(MemoryStore, "connection", broken)
    import local_operator.memory_capture as capture

    monkeypatch.setattr(capture, "_warned", False)
    with caplog.at_level(logging.WARNING):
        await session.prompt("Where is the Mercury deployment documented?")
        await session.prompt("And the Mercury deployment again?")
    assert len(requests) == 2, "the provider request must still go out"
    warnings = [r for r in caplog.records if "memory prompt tail" in r.getMessage()]
    assert len(warnings) == 1, "logged once per process"
    assert "Mercury" not in warnings[0].getMessage()
    await session.dispose()


# ---- Q1 ---------------------------------------------------------------------------------

ZW_SSN = "SSN 1\u200b23-45-6789"


@pytest.mark.parametrize(
    "variant",
    [
        ZW_SSN,
        "SSN 1\u200d23\u201045\u22126789",
        "SSN \uff11\uff12\uff13-\uff14\uff15-\uff16\uff17\uff18\uff19",
        "SSN 123\u00a045\u00a06789",
        "SSN 123 - 45 - 6789",
        "SSN 123\u200e-45-6789",
    ],
)
def test_invisible_and_lookalike_ssn_variants_leave_no_residue_anywhere(tmp_path, variant):
    store = MemoryStore(
        tmp_path / "cfg", MemoryIdentity("zw"), tmp_path / "sessions" / "zw", origin="zw-device"
    )
    (tmp_path / "sessions" / "zw").mkdir(parents=True)
    with pytest.raises(MemoryError, match="Sensitive value refused"):
        store.put("session", {"title": "t", "body": variant})
    with pytest.raises(MemoryError):
        store.put("session", {"title": variant, "body": "x docs/x.md"})
    with pytest.raises(MemoryError):
        store.put("session", {"title": "t", "body": "x docs/x.md", "tags": [variant]})
    assert not store.path.exists()
    assert list((tmp_path / "sessions" / "zw").iterdir()) == []


def test_plain_ssn_control_and_benign_unicode_still_behave(tmp_path):
    store = MemoryStore(tmp_path / "cfg", MemoryIdentity("zw"), origin="zw-device")
    with pytest.raises(MemoryError):
        store.put("global", {"title": "t", "body": "SSN 123-45-6789"})
    # Zero-width characters in ordinary text are not themselves a reason to refuse.
    assert store.put("global", {"title": "Caf\u00e9 notes", "body": "re\u200bview docs/r.md"})


@pytest.mark.parametrize("path", ["write", "edit"])
@pytest.mark.asyncio
async def test_real_tool_path_refuses_zero_width_ssn_without_a_database(tmp_path, path):
    root = tmp_path / "cfg"
    root.mkdir()
    (root / "config.yml").write_text(json.dumps({"values": {"memory": {"enabled": True}}}))
    context = ToolContext(session_id="zw-tool", memory_config_dir=str(root))
    args = {"path": "memory://global", "content": json.dumps({"title": "t", "body": ZW_SSN})}
    if path == "edit":
        args = {"path": "memory://global/abc", "old_text": "rev", "new_text": args["content"]}
    result = await execute_memory("c1", path, args, context)
    assert result.is_error and "Sensitive value refused" in result.content[0].text
    assert not (root / "memory").exists()


def test_retained_message_redacts_zero_width_ssn_in_the_journal_copy():
    from local_operator.harness.types import Message, ToolCall
    from local_operator.memory_privacy import retained_message
    from local_operator.session.transcript import encode_message_payload

    args = {"path": "memory://session", "content": json.dumps({"title": "t", "body": ZW_SSN})}
    message = Message(
        role="assistant",
        content=[],
        tool_calls=[ToolCall(name="write", arguments=args, raw_arguments=json.dumps(args))],
    )
    encoded = json.dumps(encode_message_payload(message), ensure_ascii=False)
    assert "6789" not in encoded and "memory_arguments_redacted" in encoded
    assert "6789" not in json.dumps(retained_message(message).tool_calls[0].arguments)


# ---- R3 ---------------------------------------------------------------------------------


def test_forget_is_exempt_from_the_op_cap_but_a_put_is_not(store, monkeypatch):
    first = store.put("global", {"title": "Keep", "body": "docs/keep.md"})
    victim = store.put("global", {"title": "Victim", "body": "docs/victim.md"})
    monkeypatch.setattr(memory, "MAX_OWNER_OPS", 2)
    with pytest.raises(MemoryError, match="quota"):
        store.put("global", {"title": "Third", "body": "docs/third.md"})
    store.put("global", {}, entity=victim["id"], expected_version=victim["version"], forget=True)
    assert store.show("global", victim["id"])["status"] == "forgotten"
    assert store.show("global", first["id"])["status"] == "active"
    assert not store.search("victim")


# ---- Q3 ---------------------------------------------------------------------------------


def test_two_letter_stoplisted_alias_is_findable_by_explicit_search_only(store):
    store.put(
        "global",
        {"title": "Indigo Seal cadence", "body": "weekly cadence docs/c.md", "aliases": ["IS"]},
    )
    store.put("global", {"title": "Other", "body": "unrelated weekly note docs/o.md"})
    # The alias is the ONLY informative term, so before the exemption this was [] .
    assert [row["title"] for row in store.search("what is IS")] == ["Indigo Seal cadence"]
    # No alias match, no stop-word match: a generic word never retrieves the bank.
    assert store.search("what about it") == []
    # The suggestion path keeps its generic-word suppression.
    assert store.search("what is IS", suggest=True) == []


# ---- Q7 ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_slash_memory_path_is_refused_and_creates_no_file(tmp_path, monkeypatch):
    from local_operator.tools.builtin import execute_write

    monkeypatch.chdir(tmp_path)
    result = await execute_write("w", {"path": "memory:/x", "content": "{}"}, context=None)
    assert result.is_error
    assert "memory://" in result.content[0].text
    assert not (tmp_path / "memory:").exists()
    assert not list(tmp_path.iterdir())
