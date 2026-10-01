"""The badge aggregate (push/ack-sync S1): one read, two surfaces, one population.

WHAT THIS FILE PINS. ADR 0006 §1.1/§1.2 (@5acb2331) freezes an additive read --
``GET /api/attention/unread`` -- plus a top-level ``unread`` block on the list
payload, ONE implementation behind both, computed over EXACTLY the rows the
listing serves in the same snapshot. The failure modes this file exists to make
impossible, each of which was reachable in an earlier shape of the design:

* **The badge disagrees with the list.** The obvious definition of "unread" is a
  census of ``attention.db``, and on a real machine that number is dominated by
  subagent receipts, ``agent/<id>`` namespaces and conversations that no longer
  exist -- the operator's own store measured 6,392 unread against a list that
  renders none of them [ADR §1.2]. So the population is pinned as the listing's
  own rows filtered to the user's conversations, and pinned BY CONSTRUCTION: the
  route's count must equal the number of ``unseen: true`` rows in the
  ``/api/sessions`` body captured in the same pass, and the payload block must
  equal the route. "The user's conversations" is ONE predicate, asked by BOTH
  halves (the scan of the durable rows and the merge of the live entries) -- the
  divergence review round 1 (MAJOR-1) reproduced was exactly a live entry one
  half painted and the other excluded.
* **A store that could not be read served as an empty pile.** A non-empty
  ``degraded`` withholds ``count``, never sends 0 -- on the route AND on the
  block, and for EITHER failed read behind the build (the attention read's
  ``["attention"]``, review round 1's original shape; the durable walk's
  ``["sessions"]``, review round 1 MINOR-2, which used to answer
  ``count: 0``). Clearing a badge on a read that never happened is the lie the
  absence exists to prevent [ADR §1.4, docs/ATTENTION.md].
* **The read writes.** The store's own rule: frontend reads open ``mode=ro``
  and create nothing (``attention.py:1541-1580`` pins the storage side; the
  mtime/mode cells here pin it at the daemon's seam).

Isolated config dir per test (the suite's autouse HOME isolation, plus the
config-dir monkeypatch below); nothing here touches a live daemon or store.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionRecord
from local_operator.session.attention import AttentionReadDeferred, AttentionStore
from tests.unit.session.test_catalog_read_failures import _store


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *session_ids: str
) -> tuple[Path, MobileDaemon]:
    """An isolated config root with the named listable sessions, and a daemon.

    Sessions are built through the catalogue suite's own helper so this file and
    the listing tests agree about what a listable conversation IS.
    """
    cfg = tmp_path / "config"
    if session_ids:
        _store(cfg, *session_ids)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    return cfg, MobileDaemon(port=0, password="pw123")


def _logged_in(daemon: MobileDaemon) -> TestClient:
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": "pw123"})
    return client


def _publish(session_id: str, *, kind: str = "complete") -> str:
    """One completion for ``session/<id>``; returns its token."""
    token = str(uuid.uuid4())
    AttentionStore().publish(f"session/{session_id}", token, "result", kind)
    return token


def _live(daemon: MobileDaemon, session_id: str, pid: int) -> SessionEntry:
    """A live generation for ``session_id``, as the scan would have adopted it."""
    record = SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name=session_id,
        cwd="/tmp",
        model_label="fixture",
        control_port=1,
        control_key="fixture-key",
    )
    entry = SessionEntry(record)
    daemon.table.entries[pid] = entry
    return entry


def _refresh(daemon: MobileDaemon) -> None:
    """Force the next read to rebuild, as a structural change does.

    The aggregate is served from the listing's snapshot, so a test that writes
    the store must invalidate it or the next read serves the cached build for up
    to its TTL -- the same seam the daemon's own scan loop and ack route use.
    """
    daemon.table.invalidate_summaries_cache()


# ---------------------------------------------------------------------------
# The count: conversations, not completions; and the shape of the route
# ---------------------------------------------------------------------------


def test_count_progresses_from_zero_and_counts_a_conversation_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0, 1, n -- and three unread completions on ONE conversation count once.

    The operator's rule, in the ADR's words: ``count`` counts conversations; a
    conversation with three unread completions contributes 1. The progression
    also pins the two directions of the ack: publishing raises the count, an
    acknowledgement lowers it -- with no push machinery involved at all, which
    is S1's whole point.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    client = _logged_in(daemon)

    # Zero: an existing, unread-free store is an EMPTY pile -- healthy, count 0.
    body = client.get("/api/attention/unread").json()
    assert body == {"count": 0, "revision": [0, 0, 0], "degraded": [], "conversations": []}

    # One conversation, three completions: still one.
    _publish("aaaaaaaaaaaa")
    _publish("aaaaaaaaaaaa")
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)
    body = client.get("/api/attention/unread").json()
    assert body["count"] == 1
    assert [c["session_id"] for c in body["conversations"]] == ["aaaaaaaaaaaa"]
    # The conversation carries its CURRENT completion: token, kind and the
    # row's existing [sequence, acknowledged] pair (three publishes, no reads).
    entry = body["conversations"][0]
    assert entry["kind"] == "complete"
    assert entry["revision"] == [3, 0]
    state = AttentionStore().state("session/aaaaaaaaaaaa")
    assert entry["completion_token"] == state["completion_token"]
    # ``revision`` is the store-wide equality token, served as the store states
    # it -- never an order (the third term moves on heals that move neither of
    # the first two).
    assert body["revision"] == list(AttentionStore().revision())

    # A second conversation: two.
    token_b = _publish("bbbbbbbbbbbb")
    _refresh(daemon)
    body = client.get("/api/attention/unread").json()
    assert body["count"] == 2

    # An acknowledgement clears that conversation: back to one, and the read
    # side of the rule is the store's own (reciprocal token, no second state).
    AttentionStore().acknowledge("session/bbbbbbbbbbbb", token_b)
    _refresh(daemon)
    body = client.get("/api/attention/unread").json()
    assert body["count"] == 1
    assert [c["session_id"] for c in body["conversations"]] == ["aaaaaaaaaaaa"]


def test_the_route_and_the_list_block_agree_by_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ADR's two equality assertions, over one captured pass.

    (a) route ``count`` == the number of ``unseen: true`` rows in the
    ``/api/sessions`` body captured in the same pass; (b) the payload's
    ``unread.count`` == the route's. Plus the ordering contract (the
    conversations come in the listing's order) and the not-store-wide proof: a
    subagent-origin directory with its own unseen receipt is in the store and
    NOT in any count here -- a census-shaped implementation fails this cell.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc")
    client = _logged_in(daemon)

    # a: unread (latest kind "error"); b: unread; c: read (acked).
    _publish("aaaaaaaaaaaa", kind="error")
    _publish("bbbbbbbbbbbb")
    token_c = _publish("cccccccccccc")
    AttentionStore().acknowledge("session/cccccccccccc", token_c)
    # A subagent directory with an unseen receipt: the store holds it, the
    # listing never shows it, and the aggregate must not count it. (The
    # exclusions cell below builds the rest of the store-side population.)
    subagent = cfg / "sessions" / "dddddddddddd"
    subagent.mkdir(parents=True)
    (subagent / "transcript.jsonl").write_text("{}\n")
    (subagent / "origin.json").write_text('{"origin": "subagent"}')
    _publish("dddddddddddd")
    _refresh(daemon)

    body = client.get("/api/sessions").json()
    route = client.get("/api/attention/unread").json()

    listed_unread = [row["session_id"] for row in body["sessions"] if row["unseen"]]
    assert set(listed_unread) == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}, (
        "fixture sanity: the subagent receipt must not surface as a listing row "
        f"at all, got {listed_unread!r}"
    )
    assert route["count"] == len(listed_unread) == 2
    assert body["unread"]["count"] == route["count"] == 2
    assert [
        c["session_id"] for c in route["conversations"]
    ] == listed_unread, "the conversations ride the listing's order, not the store's"
    # And the two readers are ONE implementation: the block IS the route minus
    # the per-conversation list.
    assert body["unread"] == {k: v for k, v in route.items() if k != "conversations"}


@pytest.mark.asyncio
async def test_overlapping_reads_publish_one_builds_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1 MINOR-1 / round 2 R2-2: ONE build feeds both halves.

    Two claims, each with an assertion that fails when it breaks:

    * **Overlapping callers share ONE build.** The second caller joins the
      in-flight build through ``summaries``' single-flight; a second build
      would be a second snapshot, so the merge call count must stay at one.
      This is the assertion round 2's R2-2 asked for: a faithful pre-fix
      joiner (joins, returns rows, publishes nothing) passed the tuple-shape
      assertions, so the count is what discriminates a lost join.
    * **The block names the build the cached rows came from** -- ``tagged-N``
      per build, so a pair left behind by an earlier build fails on the next
      rebuild. The two-build INTERLEAVE (the race itself) is scheduler-ordered
      and not deterministically reproducible here; the one-assignment-site
      structure is its guarantee, and these assertions are the observable
      contract around it.
    """
    import asyncio

    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    table = daemon.table
    real_merge = table._merge_summaries
    merges: list[int] = []

    def tagged(durable):
        rows = real_merge(durable)
        tag = len(merges)
        merges.append(tag)
        for row in rows:
            row["session_id"] = f"tagged-{tag}"
            row["unseen"] = True
        return rows

    monkeypatch.setattr(table, "_merge_summaries", tagged)

    def pair_tags() -> tuple[str, str]:
        """``(tag on the published rows, tag on the published block)``."""
        cached = table._summaries_cache or []
        conversations = table.unread_snapshot().get("conversations") or [{}]
        return (
            cached[0]["session_id"] if cached else "<no rows>",
            conversations[0].get("session_id", "<no block>"),
        )

    real_refresh = table._refresh_durable_rows
    entered = asyncio.Event()

    async def slow_refresh():
        entered.set()
        await asyncio.sleep(0.05)
        return await real_refresh()

    monkeypatch.setattr(table, "_refresh_durable_rows", slow_refresh)
    _refresh(daemon)

    first = asyncio.ensure_future(table.summaries())
    await entered.wait()
    second = asyncio.ensure_future(table.summaries())
    rows_a, rows_b = await asyncio.gather(first, second)

    assert merges == [0], (
        "overlapping callers must share ONE build; a second build would be a " "second snapshot"
    )
    assert rows_a is rows_b, "the second caller must join the in-flight build"
    assert pair_tags() == ("tagged-0", "tagged-0")

    # And on every rebuild, the pair moves together: a block (or a cached row
    # set) left behind by an earlier build fails here -- the closest a
    # deterministic test gets to the two-build race the one-assignment site
    # exists for.
    for depth in (1, 2):
        _refresh(daemon)
        rows = await table.summaries()
        assert [row["session_id"] for row in rows] == [f"tagged-{depth}"]
        assert pair_tags() == (
            f"tagged-{depth}",
            f"tagged-{depth}",
        ), "the block must describe the build the cached rows came from"
    assert merges == [0, 1, 2]


def test_exclusions_are_not_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hidden origins and deleted conversations never raise the badge -- NOR paint.

    Every excluded thing below carries a real unseen receipt -- each one is a
    state a store-wide census WOULD count (4 here), and the live ones are states
    an unfiltered live half WOULD paint. The badge must be 1, and the assertion
    that matters is read from the LISTING BODY: the excluded conversations do
    not appear there either, because count and paint are one set by construction
    (review round 1, MAJOR-1 -- the earlier round of this cell asserted only the
    count, which is how the live-half divergence shipped).

    The live hidden/deleted rows are deliberately the sharp case: a record
    carries no origin, so the live half asks the session's own marker through
    the SAME predicate the scan applies (``resume.is_user_session_origin``),
    and a directory that is gone is not a row -- a deleted conversation's
    receipt must not raise the badge on either surface.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)

    # Counted: the user's own live conversation.
    _publish("aaaaaaaaaaaa")

    # A scheduled origin (``agent-config``) and a subagent origin: durable
    # directories the scan excludes, so they never appear in ANY listing. Both
    # have receipts, so a store-wide count would see them.
    for session_id, origin in (
        ("cccccccccccc", "agent-config"),
        ("dddddddddddd", "subagent"),
    ):
        directory = cfg / "sessions" / session_id
        directory.mkdir(parents=True)
        (directory / "transcript.jsonl").write_text("{}\n")
        (directory / "origin.json").write_text(f'{{"origin": "{origin}"}}')
        _publish(session_id)

    # A deleted conversation: a receipt for a session with no directory left.
    _publish("eeeeeeeeeeee")

    # The LIVE shapes of the same exclusions: the daemon saw a generation for a
    # hidden-origin session, and one for a conversation whose directory is gone.
    _live(daemon, "cccccccccccc", 2101)
    _live(daemon, "eeeeeeeeeeee", 2102)
    _refresh(daemon)

    body = client.get("/api/sessions").json()
    route = client.get("/api/attention/unread").json()

    # NEITHER surface offers the excluded conversations...
    painted = [row["session_id"] for row in body["sessions"]]
    assert painted == ["aaaaaaaaaaaa"], painted
    assert route["count"] == 1
    assert [c["session_id"] for c in route["conversations"]] == ["aaaaaaaaaaaa"]
    # ...and the by-construction equality holds ON THIS STATE -- the state where
    # the two sets would part if the halves asked different questions.
    assert route["count"] == len([row for row in body["sessions"] if row["unseen"]])
    assert body["unread"]["count"] == route["count"]
    # The degraded branch is NOT what excluded them: this is a healthy read.
    assert route["degraded"] == []
    # Fixture sanity: the four receipts really are unseen in the store, so the
    # count above is reading the exclusion rule and not a missing state.
    states = AttentionStore().state_many(
        f"session/{sid}" for sid in ("aaaaaaaaaaaa", "cccccccccccc", "dddddddddddd", "eeeeeeeeeeee")
    )
    assert sum(1 for state in states.values() if state["unseen"]) == 4


def test_a_hidden_live_generation_neither_paints_nor_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1 MAJOR-1, the reviewer's reproduced state.

    A live runtime with a hidden origin -- ``agent-config`` (scheduled) and
    ``agent-shell`` are real runtimes that register a mobile record like any
    other -- used to be PAINTED by the listing (the live half had no origin
    filter) while the aggregate excluded it, or vice versa, so the badge and the
    list could carry different conversation sets. The population rule now holds
    by construction, and this pins it on the reviewer's shape: the hidden
    runtime beside TWO ORDINARY live sessions, each with a receipt, asserting
    absence from the LISTING BODY and the count, with the equality assertions
    passing on the same captured pass.

    The hidden session's marker is written BEFORE its record is published (the
    ordering ``server/utils/desktop_sessions`` documents), and it keeps a real
    directory and transcript so the ONLY reason it is excluded is its origin --
    a cell that dropped it for a missing directory would pass for the wrong
    reason.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    client = _logged_in(daemon)

    hidden = cfg / "sessions" / "cccccccccccc"
    hidden.mkdir(parents=True)
    (hidden / "transcript.jsonl").write_text("{}\n")
    (hidden / "origin.json").write_text('{"origin": "agent-shell"}')

    for session_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"):
        _publish(session_id)
    for pid, session_id in enumerate(("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"), start=3101):
        _live(daemon, session_id, pid)
    _refresh(daemon)

    body = client.get("/api/sessions").json()
    route = client.get("/api/attention/unread").json()

    painted = [row["session_id"] for row in body["sessions"]]
    assert "cccccccccccc" not in painted, (
        "a live generation for a hidden origin must not paint: the live half "
        "asks the same predicate the scan applies"
    )
    assert set(painted) == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert route["count"] == 2
    assert [c["session_id"] for c in route["conversations"]] == [
        row["session_id"] for row in body["sessions"] if row["unseen"]
    ], "the counted set and the painted set are one set by construction"
    assert body["unread"]["count"] == route["count"] == 2


def test_a_live_only_row_with_a_directory_but_no_transcript_paints_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 2 R2-1: the §1.2 shape the one-predicate decision exists for.

    The ONE shape on which "the scan's predicate, asked of the marker" and
    round 1's ``_durable_user_session_dir`` detail check differ: a LIVE-ONLY,
    user-owned conversation whose directory EXISTS with NO transcript and NO
    marker -- the "mail-spool conversation with no transcript yet" §1.2 names
    as the reason the detail check is not the predicate. At this head it must
    PAINT and COUNT: no marker reads as the user's own, and a directory with
    no materialised transcript is still a conversation with an unseen receipt.
    Substituting the detail check (which additionally requires
    ``transcript.jsonl``) would make it vanish from both surfaces -- and this
    is the only cell that fails, which is exactly what round 2 measured when it
    swapped the check and watched all eleven cells stay green.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch)
    client = _logged_in(daemon)

    directory = cfg / "sessions" / "ffffffffffff"
    directory.mkdir(parents=True)
    # The shape, stated in the fixture itself: nothing to read a transcript or
    # a marker from -- only the directory, a live record and a receipt.
    assert not (directory / "transcript.jsonl").exists()
    assert not (directory / "origin.json").exists()

    _publish("ffffffffffff")
    _live(daemon, "ffffffffffff", 4101)
    _refresh(daemon)

    body = client.get("/api/sessions").json()
    route = client.get("/api/attention/unread").json()

    painted = [row["session_id"] for row in body["sessions"]]
    assert painted == [
        "ffffffffffff"
    ], "a live-only user row with no materialised transcript is still a row"
    assert route["count"] == 1
    assert [c["session_id"] for c in route["conversations"]] == ["ffffffffffff"]
    assert route["count"] == len([row for row in body["sessions"] if row["unseen"]])
    assert body["unread"]["count"] == route["count"] == 1


def test_a_live_only_conversation_the_scan_has_not_seen_yet_is_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction of the live probe: a user's row must not be LOST.

    A conversation with a LIVE generation that the durable scan did not return
    is still a listing row, and the aggregate must probe it and COUNT it. The
    shape is real in both directions: every just-started session has it until
    its first scan lands, and an archived conversation keeps it (archived rows
    are dropped from ``recent_session_rows``' rows while the live generation
    still paints). A filter written as "durable rows only" would silently drop
    the badge for the conversation the user just started -- the severe
    direction of this bug -- so this cell is the counterweight to the
    exclusion cell above.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _live(daemon, "aaaaaaaaaaaa", 2301)

    from local_operator.session.archived import set_archived

    assert set_archived(cfg, "aaaaaaaaaaaa", True)
    _refresh(daemon)

    body = client.get("/api/sessions").json()
    route = client.get("/api/attention/unread").json()
    assert [row["session_id"] for row in body["sessions"]] == ["aaaaaaaaaaaa"], (
        "fixture sanity: the scan drops archived rows, so this row can only "
        "come from the LIVE generation"
    )
    assert route["count"] == 1, (
        "a live-only row the user owns must be counted: the probe decides by "
        "the session directory, not by the scan's absence"
    )
    assert [c["session_id"] for c in route["conversations"]] == ["aaaaaaaaaaaa"]


# ---------------------------------------------------------------------------
# Degradation: absent, never 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "refusal",
    [
        pytest.param(
            lambda: AttentionReadDeferred("attention store stayed busy through 2 attempts"),
            id="deferred",
        ),
        pytest.param(lambda: OSError("store cannot be opened"), id="unopenable"),
    ],
)
def test_a_store_that_cannot_be_read_serves_degraded_without_a_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal
) -> None:
    """``degraded: ["attention"]`` and NO ``count`` -- on both surfaces at once.

    Both arms of the build's except clause, because each is a real store
    failure: a deferred read (contention that outlasted the retry budget, which
    rides the read as an ``sqlite3.Error``) and a store that cannot be OPENED
    (``OSError``). Injected at the class seam the repo's own degradation cells
    use, which pins THIS call site: the aggregate must observe the same failed
    read the rows do, and it must publish an ABSENCE. ``count: 0`` here is the
    lie that clears the operator's badge on a machine that never answered -- the
    exact defect ``docs/ATTENTION.md`` names ("a store that could not be read is
    not an empty pile").

    The read is ONE call (``state_many_and_revision`` -- see its docstring for
    why the count and its token share a connection), so there is no
    revision-only failure to simulate separately: the seam IS the failure
    domain, which is the property the listing's own ``degraded`` marker leans on
    below.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")
    _refresh(daemon)
    assert client.get("/api/attention/unread").json()["count"] == 1

    def refuses(_store: AttentionStore, _conversations: object) -> dict[str, object]:
        raise refusal()

    monkeypatch.setattr(AttentionStore, "state_many_and_revision", refuses)
    _refresh(daemon)

    route = client.get("/api/attention/unread")
    assert route.status_code == 200
    body = route.json()
    assert body["degraded"] == ["attention"]
    assert "count" not in body
    assert "revision" not in body
    assert "conversations" not in body

    listed = client.get("/api/sessions").json()
    assert listed["unread"] == {"degraded": ["attention"]}
    assert "count" not in listed["unread"]
    # The listing's OWN marker names the same failure, so the two markers a
    # client can read cannot disagree.
    assert listed["degraded"] == ["attention"]


def test_a_failed_durable_walk_degrades_the_block_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, MINOR-2: the OTHER half's failure is not a zero either.

    The reviewer's repro: with the store's directory unreadable, the build had
    no rows and the block answered ``{"count": 0, "degraded": []}`` -- the
    silent zero this workstream exists to eliminate, reached through the
    durable walk. The block now seeds ``degraded`` from the build's WHOLE read
    verdict, so this failure withholds ``count`` on both surfaces exactly as a
    failed attention read does -- and the healing is real: once the walk reads
    again, the same state's receipt answers with a count.
    """
    import errno

    from tests.unit.mobile.test_daemon import _listing_rows
    from tests.unit.session.test_catalog_read_failures import _failing_open

    cfg = tmp_path / "config"
    store = _listing_rows(cfg, "aaaaaaaaaaaa")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: cfg)
    daemon = MobileDaemon(port=0, password="pw123")
    client = _logged_in(daemon)
    _publish("aaaaaaaaaaaa")

    _failing_open(monkeypatch, store, OSError(errno.EIO, "Input/output error"))
    _refresh(daemon)

    _resp = client.get("/api/sessions")
    listed = _resp.json()
    assert listed["degraded"] == ["sessions"]
    assert "count" not in listed["unread"], listed["unread"]
    assert listed["unread"]["degraded"] == ["sessions"], listed["unread"]

    route = client.get("/api/attention/unread").json()
    assert route["degraded"] == ["sessions"]
    assert "count" not in route

    # The seam fails once (``nth=1``), so the next walk reads: a transient
    # failure must not latch the badge at "unknown".
    _refresh(daemon)
    healed = client.get("/api/sessions").json()
    assert healed["degraded"] == []
    assert healed["unread"]["count"] == 1
    assert client.get("/api/attention/unread").json()["count"] == 1


# ---------------------------------------------------------------------------
# The read path writes nothing; the gate is the listing's gate
# ---------------------------------------------------------------------------


def test_the_read_path_never_writes_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both surfaces read; neither creates, migrates or touches the store.

    The store's rule is ``mode=ro`` on the read path (``attention.py``), pinned
    at the daemon's seam: after a publish, the file's mtime and mode are
    unchanged by the two reads; on a machine with NO store, the reads leave no
    file behind -- a GET that created the database would make every clean
    install look like it had unread state to report.
    """
    cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = _logged_in(daemon)
    # The store's OWN resolved path, not a reconstruction: ``attention.py``
    # binds ``config_dir`` at import (module-level from-import), so the store
    # follows the suite's isolated HOME rather than the monkeypatched
    # ``local_operator.paths.config_dir`` the daemon's lazy imports read.
    # Asserting against a rebuilt path would test the wrong file.
    db = AttentionStore().path

    # No store yet: reading must not bring one into being.
    assert not db.exists()
    client.get("/api/attention/unread")
    client.get("/api/sessions")
    assert not db.exists(), "the badge read created the store it only reads"

    _publish("aaaaaaaaaaaa")
    _refresh(daemon)
    before = db.stat().st_mtime_ns
    client.get("/api/attention/unread")
    client.get("/api/sessions")
    after = db.stat().st_mtime_ns
    assert after == before, "the read path moved the store's mtime"
    assert db.stat().st_mode & 0o777 == 0o600


def test_the_route_is_gated_exactly_like_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No cookie, no read -- with the SAME refusal the listing gives.

    One ``gate`` call, one vocabulary: a client that handles the list's 401
    handles this route's, which is what lets the app's fetch layer reuse one
    branch. The comparison is direct (same status AND same body) so the two
    cannot drift apart silently.
    """
    _cfg, daemon = _fixture(tmp_path, monkeypatch, "aaaaaaaaaaaa")
    client = TestClient(build_app(daemon), follow_redirects=False)

    listing = client.get("/api/sessions")
    unread = client.get("/api/attention/unread")
    assert listing.status_code == unread.status_code == 401
    assert unread.json() == listing.json() == {"error": "authentication required"}

    # The gate is a gate, not a blanket refusal: with the cookie, the same
    # daemon serves the read.
    authed = _logged_in(daemon)
    served = authed.get("/api/attention/unread")
    assert served.status_code == 200
    assert served.json()["degraded"] == []
