"""In-thread find: tiers, ranking, snippets/ranges, and the warm/cold view.

The pipeline is exercised over synthetic ``MessageDoc`` lists (no journals at
all), and the view over synthetic journals written in the journal's own row
format — the same shape the index's own suite writes — so nothing here reads
the operator's store.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session import transcript_find as tf
from local_operator.session import transcript_index as ti
from local_operator.session.goal_judge import goal_continuation_prompt
from local_operator.session.transcript import encode_message_payload
from local_operator.session.transcript_index import MessageDoc

SID = "aabbccddee02"


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Both modules keep process-wide state; no test may inherit another's."""
    ti._reset_for_tests()
    tf._reset_for_tests()
    yield
    ti._reset_for_tests()
    tf._reset_for_tests()


@pytest.fixture()
def hide_cross_session(tmp_path, monkeypatch):
    """The display flag, flipped through the real config file and reader.

    The mobile fold suite's fixture (``tests/unit/mobile/test_projection.py``)
    in this file's terms: the write goes through ``settings_io``, so the
    flat-dotted key the reader actually looks up is the one exercised, and a
    test that never calls ``set_hidden`` leaves the default (off) alone.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.tui.settings import settings_reload

    settings_reload()

    def set_hidden(on: bool) -> None:
        from local_operator import settings_io
        from local_operator.config import ConfigManager

        settings_io.write_setting(
            ConfigManager(tmp_path), settings_io.BY_KEY["display.hide_cross_session"], on
        )

    yield set_hidden
    settings_reload()


def doc(
    id_: str,
    seq: int,
    text: str,
    *,
    role: str = "user",
    injected: bool = False,
    ts: float = 0.0,
) -> MessageDoc:
    return MessageDoc(id=id_, ts=ts, role=role, text=text, injected=injected, seq=seq)


def journal_path(root: Path) -> Path:
    return root / "sessions" / SID / "transcript.jsonl"


def write_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    path = journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def user(id_: str, ts: float, text: str = "hello") -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "user", "content": [{"text": text}]},
    }


def assistant(id_: str, ts: float, text: str = "answer") -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "assistant", "content": [{"text": text}]},
    }


def peer_message(id_: str, ts: float, body: str) -> dict[str, Any]:
    """The real ``peer_message`` row shape: its text lives under ``details.body``.

    Copied from the index suite's own extraction fixture
    (``test_injected_rows_extract_text_beyond_the_text_key``) so the two files
    cannot drift on what a peer message looks like on disk.
    """
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {
            "kind": "custom",
            "custom_type": "peer_message",
            "details": {"body": body, "sender": "w1"},
        },
    }


def inject(id_: str, ts: float, custom_type: str, text: str) -> dict[str, Any]:
    """Any other custom row, the index suite's ``inject`` shape.

    ``custom_type`` is the discriminator the hidden-cross-session filter must
    match EXACTLY: only ``peer_message`` is in the hidden set, so a row built
    through this helper (hub, wake, compaction, ...) has to stay findable.
    """
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "custom", "custom_type": custom_type, "details": {"text": text}},
    }


#: The goal judge's continuation, through its own producer (the index suite's
#: fixture, so the two files cannot drift on the exact text).
GOAL_CONTINUATION = goal_continuation_prompt("make the fold hold")


def stamped_user(id_: str, ts: float, text: str) -> dict[str, Any]:
    """A harness row as the renderer writes it: a ``Message`` carrying the stamp."""
    message = Message(
        role="user",
        content=[TextContent(text=text)],
        provider_payload={"harness_injected": True},
    )
    return {"id": id_, "ts": ts, "type": "message", "payload": encode_message_payload(message)}


# ---------------------------------------------------------------------------
# Tiers and ranking
# ---------------------------------------------------------------------------


def test_exact_tier_is_casefolded_and_snippet_keeps_original_text():
    hits, truncated = tf.search_messages([doc("m1", 0, "We DePloy the thing")], "deploy", 10)
    assert [(h.tier, h.id, h.role) for h in hits] == [("exact", "m1", "user")]
    assert hits[0].snippet == "We DePloy the thing"
    assert hits[0].ranges == ((3, 9),)
    assert truncated is False


def test_literal_occurrences_are_non_overlapping():
    hits, _ = tf.search_messages([doc("m1", 0, "aaaa")], "aa", 10)
    assert hits[0].ranges == ((0, 2), (2, 4))


def test_soft_pass_runs_below_the_floor_only():
    below = [
        doc("m1", 0, "classifer one"),
        doc("m2", 1, "classifer two"),
        doc("m9", 5, "the classifier is here"),
    ]
    hits, _ = tf.search_messages(below, "classifer", 10)
    assert [h.id for h in hits] == ["m1", "m2", "m9"]
    assert hits[-1].tier == "soft"

    # A third exact hit reaches PRECISE_HITS_ENOUGH; the typo doc drops out.
    at_floor = below[:2] + [doc("m3", 2, "classifer three"), below[2]]
    hits, _ = tf.search_messages(at_floor, "classifer", 10)
    assert [h.id for h in hits] == ["m1", "m2", "m3"]


def test_soft_search_called_only_below_the_floor():
    calls: list[int] = []

    def spy(digests: dict[str, str]) -> set[str]:
        calls.append(len(digests))
        return set()

    three = [doc(f"m{i}", i, "needle") for i in range(3)]
    tf.search_messages(three, "needle", 10, soft_search=spy)
    assert calls == []
    tf.search_messages(three[:2], "needle", 10, soft_search=spy)
    assert calls == [2]


def test_ranking_exact_then_soft_then_genuine_then_injected_then_seq():
    docs = [
        doc("i0", 0, "classifer from hub", injected=True),
        doc("g1", 1, "the classifier here"),
        doc("g2", 2, "classifer genuine"),
        doc("i3", 3, "classifier too", injected=True),
    ]
    hits, _ = tf.search_messages(docs, "classifer", 10)
    assert [h.id for h in hits] == ["g2", "i0", "g1", "i3"]
    assert [h.tier for h in hits] == ["exact", "exact", "soft", "soft"]


def test_injected_docs_rank_after_genuine_within_a_tier():
    docs = [doc("i0", 0, "needle hub", injected=True), doc("g1", 1, "needle genuine")]
    hits, _ = tf.search_messages(docs, "needle", 10)
    assert [h.id for h in hits] == ["g1", "i0"]


def test_injected_docs_carry_their_text_into_find_results():
    """A peer message is a find result like any other injected doc.

    Injected rows are conversation INPUTS — an incoming peer message is text
    the reader may well want to find again — so the pipeline must never treat
    "injected" as "empty". This is the pipeline-level contract;
    :func:`test_a_peer_message_row_is_findable_end_to_end` drives the same
    property off a real ``details.body`` journal row.
    """
    peer = doc("p1", 1, "peer says the deploy is green", injected=True)
    genuine = doc("g1", 2, "a genuine note about the deploy")
    hits, _ = tf.search_messages([peer, genuine], "deploy", 10)
    assert [h.id for h in hits] == ["g1", "p1"]  # genuine first: D3's demotion
    peer_hit = hits[1]
    assert peer_hit.tier == "exact"
    assert peer_hit.snippet == "peer says the deploy is green"
    start, end = peer_hit.ranges[0]
    assert peer_hit.snippet[start:end] == "deploy"


def test_journal_order_is_oldest_first_within_a_tier():
    docs = [doc("c", 5, "needle"), doc("a", 1, "needle again"), doc("b", 3, "needle more")]
    hits, _ = tf.search_messages(docs, "needle", 10)
    assert [h.id for h in hits] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Snippets and ranges
# ---------------------------------------------------------------------------


def test_ranges_are_windowed_and_snippet_relative():
    text = "x" * 200 + "NEEDLE" + "y" * 200
    hits, _ = tf.search_messages([doc("m1", 0, text)], "needle", 10)
    assert hits[0].ranges == ((60, 66),)
    assert hits[0].snippet == text[140:266]
    start, end = hits[0].ranges[0]
    assert hits[0].snippet[start:end] == "NEEDLE"


def test_ranges_capped_at_five():
    hits, _ = tf.search_messages([doc("m1", 0, "needle " * 9)], "needle", 10)
    assert len(hits[0].ranges) == 5
    assert hits[0].ranges[0] == (0, 6)


def test_soft_hit_carries_head_window_and_empty_ranges():
    text = "classifier " + "z" * 400
    hits, _ = tf.search_messages([doc("m1", 0, text)], "classifer", 10)
    assert hits[0].tier == "soft"
    assert hits[0].snippet == text[:120]
    assert hits[0].ranges == ()


def test_casefold_expansion_offsets_cover_the_original_characters():
    text = "Die Straße liegt dort"
    hits, _ = tf.search_messages([doc("m1", 0, text)], "STRASSE", 10)
    assert hits[0].tier == "exact"
    start, end = hits[0].ranges[0]
    assert (start, end) == (4, 10)
    assert hits[0].snippet[start:end] == "Straße"


# ---------------------------------------------------------------------------
# Limits and the wire shape
# ---------------------------------------------------------------------------


def test_limit_caps_hits_and_reports_truncation():
    docs = [doc(f"m{i}", i, "needle") for i in range(5)]
    hits, truncated = tf.search_messages(docs, "needle", 3)
    assert ([h.id for h in hits], truncated) == (["m0", "m1", "m2"], True)
    hits, truncated = tf.search_messages(docs, "needle", 5)
    assert (len(hits), truncated) == (5, False)
    hits, truncated = tf.search_messages(docs, "needle", 0)
    assert (hits, truncated) == ([], False)


def test_empty_query_matches_nothing():
    hits, truncated = tf.search_messages([doc("m1", 0, "needle")], "   ", 10)
    assert (hits, truncated) == ([], False)


def test_hit_payload_is_json_shaped():
    hits, _ = tf.search_messages([doc("m1", 0, "deploy")], "deploy", 10)
    assert hits[0].to_payload() == {
        "id": "m1",
        "role": "user",
        "ts": 0.0,
        "snippet": "deploy",
        "ranges": [[0, 6]],
        "tier": "exact",
    }


def test_assistant_role_maps_to_agent_on_the_wire():
    hits, _ = tf.search_messages([doc("m1", 0, "deploy ok", role="assistant")], "deploy", 10)
    assert hits[0].to_payload()["role"] == "agent"


def test_soft_cache_is_bounded_and_lru(tmp_path):
    for i in range(5):
        tf._soft_for(tmp_path, f"s{i}")
    assert len(tf._SOFT_LRU) == tf._SOFT_SESSIONS
    assert (str(tmp_path), "s0") not in tf._SOFT_LRU
    assert (str(tmp_path), "s4") in tf._SOFT_LRU


# ---------------------------------------------------------------------------
# The view: ready / building / error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_view_ready_then_resident_reuse(tmp_path, monkeypatch):
    write_rows(
        tmp_path,
        [user("u1", 1.0, "deploy the target"), assistant("a1", 1.1, "deploying now")],
    )
    view = await tf.find_view(tmp_path, SID, query="deploy", limit=10)
    assert view["state"] == "ready"
    assert view["partial"] is False
    assert [h["id"] for h in view["hits"]] == ["u1", "a1"]
    assert view["hits"][1]["role"] == "agent"

    def boom(config_dir, session_id):
        raise AssertionError("the warm path re-read the cache")

    monkeypatch.setattr(ti, "read_index", boom)
    again = await tf.find_view(tmp_path, SID, query="deploy", limit=10)
    assert again == view


@pytest.mark.asyncio
async def test_find_view_building_serves_the_previous_scan_as_partial(tmp_path, monkeypatch):
    write_rows(tmp_path, [user("u1", 1.0, "deploy now")])
    first = await tf.find_view(tmp_path, SID, query="deploy", limit=10)
    assert first["state"] == "ready"

    # Grow the journal so the cache is stale, and hold the rescan so the ask
    # lands inside the build.
    write_rows(tmp_path, [assistant("a2", 2.0, "the zebra escaped")])
    entered, release = threading.Event(), threading.Event()
    real = ti.refresh_index

    def slow(config_dir, session_id):
        entered.set()
        assert release.wait(30), "test never released the build"
        return real(config_dir, session_id)

    monkeypatch.setattr(ti, "refresh_index", slow)
    view = await tf.find_view(tmp_path, SID, query="zebra", limit=10, wait_s=0.05)
    assert view["state"] == "building"
    assert view["partial"] is True
    assert view["hits"] == []  # the previous scan cannot see the appended row

    assert await asyncio.to_thread(entered.wait, 30)
    release.set()
    deadline = asyncio.get_running_loop().time() + 30.0
    while True:
        view = await tf.find_view(tmp_path, SID, query="zebra", limit=10, wait_s=0.05)
        if view["state"] == "ready":
            break
        assert asyncio.get_running_loop().time() < deadline, view
        await asyncio.sleep(0.02)
    assert [h["id"] for h in view["hits"]] == ["a2"]
    assert view["partial"] is False


@pytest.mark.asyncio
async def test_find_view_error_state(tmp_path, monkeypatch):
    write_rows(tmp_path, [user("u1", 1.0, "deploy now")])

    def broken(config_dir, session_id):
        raise OSError("journal on fire")

    monkeypatch.setattr(ti, "refresh_index", broken)
    view = await tf.find_view(tmp_path, SID, query="deploy", limit=10, wait_s=5)
    assert (view["state"], view["hits"], view["partial"], view["truncated"]) == (
        "error",
        [],
        False,
        False,
    )


@pytest.mark.asyncio
async def test_find_view_missing_journal_is_ready_empty(tmp_path):
    view = await tf.find_view(tmp_path, SID, query="deploy", limit=10)
    assert view == {
        "query": "deploy",
        "state": "ready",
        "partial": False,
        "hits": [],
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_a_peer_message_row_is_findable_end_to_end(tmp_path):
    """The row-format pin: a peer's message is searchable, not empty.

    ``peer_message`` carries its text under ``details.body`` (the extraction
    the index slice's remediation added); this drives the REAL journal →
    scanner → find chain so this slice cannot silently regress to indexing a
    peer's message as "" — the shape that made it unfindable. Demotion and
    snippet come from the same run, against text that actually came off disk.
    """
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "deploy the staging target"),
            peer_message("p1", 1.1, "peer says the deploy is green"),
        ],
    )
    view = await tf.find_view(tmp_path, SID, query="deploy", limit=10)
    assert view["state"] == "ready"
    assert [h["id"] for h in view["hits"]] == ["u1", "p1"]  # genuine first: D3
    peer = view["hits"][1]
    assert peer["tier"] == "exact"
    assert peer["snippet"] == "peer says the deploy is green"
    start, end = peer["ranges"][0]
    assert peer["snippet"][start:end] == "deploy"


@pytest.mark.asyncio
async def test_a_harness_continuation_is_not_findable_as_the_operators(tmp_path):
    """F1 of local-operator-ui#670, find half: harness chrome is not the user's.

    The wire has no injected demotion to hide behind — ``FindHit`` carries
    only id/role/snippet/ranges/tier, so a chrome doc kept as "injected" would
    still arrive as a user-role snippet, and a hit's reveal jump must land on
    a row a surface paints (these render nowhere). So the continuation's own
    boilerplate — text only IT carried — must answer nothing, and the
    operator's words must not be doubled by the harness echoing them.
    """
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "make the fold hold"),
            stamped_user("c1", 1.1, GOAL_CONTINUATION),
            assistant("a1", 1.2, "working on it"),
        ],
    )
    # Build the cache synchronously first: the ladder's first-paint wait is a
    # load-sensitive race this cell does not test (the peer cell above drives
    # the live ladder); the find decision under test must not flake on it.
    assert ti.refresh_index(tmp_path, SID) is not None
    view = await tf.find_view(tmp_path, SID, query="state plainly what advanced", limit=10)
    assert view["state"] == "ready"
    assert view["hits"] == []
    view = await tf.find_view(tmp_path, SID, query="make the fold hold", limit=10)
    assert [h["id"] for h in view["hits"]] == ["u1"]


# ---------------------------------------------------------------------------
# Hidden cross-session docs (display.hide_cross_session)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hidden_cross_session_drops_peer_docs_and_keeps_the_rest(
    tmp_path, hide_cross_session
):
    """The flag gates peer docs at the docs-selection seam, both polarities.

    OFF is the byte-identical default: every injected doc is a hit, exactly as
    ``test_a_peer_message_row_is_findable_end_to_end`` pins. ON drops the peer
    doc — including on text only the peer body carries — because a result the
    overlay would reveal cannot point at a row no surface paints (design
    §5.3). A ``hub_message`` is NOT in the frozen hidden set and stays
    findable, which guards the filter against being written as "injected".
    """
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "deploy the staging target"),
            peer_message("p1", 1.1, "peer says the deploy is green"),
            inject("i1", 1.2, "hub_message", "deploy hub note"),
        ],
    )
    view = await tf.find_view(tmp_path, SID, query="deploy", limit=10)
    assert view["state"] == "ready"
    assert [h["id"] for h in view["hits"]] == ["u1", "p1", "i1"]  # genuine, then injected

    hide_cross_session(True)
    view = await tf.find_view(tmp_path, SID, query="deploy", limit=10)
    assert [h["id"] for h in view["hits"]] == ["u1", "i1"]
    # Peer-only text returns nothing under any tier: the doc leaves before the
    # exact and the soft pass both.
    view = await tf.find_view(tmp_path, SID, query="green", limit=10)
    assert view["hits"] == []

    hide_cross_session(False)
    view = await tf.find_view(tmp_path, SID, query="green", limit=10)
    assert [h["id"] for h in view["hits"]] == ["p1"]


@pytest.mark.asyncio
async def test_hidden_peer_docs_are_dropped_from_the_building_partial_too(
    tmp_path, hide_cross_session, monkeypatch
):
    """``building`` serves the PREVIOUS scan's docs — the second selection site.

    A gate that only covered the ready path would leak the peer doc for the
    whole scan window, and the overlay reveals hits the moment they arrive, so
    the partial answer must be filtered too. Same machinery as
    ``test_find_view_building_serves_the_previous_scan_as_partial``.
    """
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "deploy the staging target"),
            peer_message("p1", 1.1, "peer says the deploy is green"),
        ],
    )
    first = await tf.find_view(tmp_path, SID, query="deploy", limit=10)
    assert first["state"] == "ready"

    hide_cross_session(True)

    # Grow the journal so the cache is stale, and hold the rescan so the ask
    # lands inside the build.
    write_rows(tmp_path, [assistant("a2", 2.0, "the zebra escaped")])
    entered, release = threading.Event(), threading.Event()
    real = ti.refresh_index

    def slow(config_dir, session_id):
        entered.set()
        assert release.wait(30), "test never released the build"
        return real(config_dir, session_id)

    monkeypatch.setattr(ti, "refresh_index", slow)
    view = await tf.find_view(tmp_path, SID, query="deploy", limit=10, wait_s=0.05)
    assert view["state"] == "building"
    assert view["partial"] is True
    # The previous scan holds the peer doc; nothing it returns may carry it.
    assert [h["id"] for h in view["hits"]] == ["u1"]

    assert await asyncio.to_thread(entered.wait, 30)
    release.set()
    deadline = asyncio.get_running_loop().time() + 30.0
    while True:
        view = await tf.find_view(tmp_path, SID, query="deploy", limit=10, wait_s=0.05)
        if view["state"] == "ready":
            break
        assert asyncio.get_running_loop().time() < deadline, view
        await asyncio.sleep(0.02)
    # And the refilled scan stays filtered: the re-derived doc carries
    # custom_type, so the gate covers it by the same field.
    assert [h["id"] for h in view["hits"]] == ["u1"]
