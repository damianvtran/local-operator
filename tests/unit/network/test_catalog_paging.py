"""Paged and searched ``net_catalog`` reads over two REAL relays.

Operator report (2026-11-25): a node with 417 session dirs listed 223 — its
live rows plus ``CATALOG_SCAN_LIMIT`` (200) stored — and nothing could reach
the rest; a remote ``query`` was refused outright. These cells pin the owner's
paged read (``RelayServer._catalog_page``) over an authenticated link, with
the production op answering:

* a walk at ``limit=100`` reaches EVERY live and stored row exactly once, past
  the 200 stored cap;
* a frame naming none of ``limit``/``cursor``/``query`` is answered exactly as
  before (the pollers' contract, the cap included);
* paged and searched rows have exactly the unpaged rows' keys and values (the
  read-role invariant: no new field, no transcript content);
* a malformed cursor/limit is refused with a sentence, never a crash;
* the viewer's relay forwards the fields only to an owner advertising
  ``REMOTE_CATALOG_PAGE_V1`` and labels an older owner ``unsupported``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import relay, wire
from local_operator.resume import write_session_title
from local_operator.session.catalog import CATALOG_SCAN_LIMIT
from local_operator.session.runtime.types import SessionRecord
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _live_link,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]

#: Past the stored cap by a margin, and not a multiple of the page size.
STORED = CATALOG_SCAN_LIMIT + 37
LIVE = 3


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture under a name these tests can take (F811)."""
    value: Devices = request.getfixturevalue("devices")
    return value


def _stored(root: Path, count: int, *, needle_every: int = 0) -> list[str]:
    """``count`` real stored conversations; every ``needle_every``-th is titled ``needle``."""
    ids: list[str] = []
    for index in range(count):
        session_id = f"{index:012x}"
        directory = root / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "created_at.json").write_text(str(1_700_000_000.0 + index), encoding="utf-8")
        (directory / "transcript.jsonl").write_text(
            '{"id":"e1","ts":1,"type":"message",'
            '"payload":{"kind":"message","role":"user","content":[{"text":"go"}]}}\n',
            encoding="utf-8",
        )
        needle = needle_every and index % needle_every == 0
        title = f"needle report {index}" if needle else f"plain chat {index}"
        write_session_title(directory, title, user_set=False, past_names=[])
        ids.append(session_id)
    return ids


def _live(monkeypatch: pytest.MonkeyPatch, count: int) -> list[str]:
    """``count`` live registry records, through the scan the owner reads."""
    from local_operator.session.runtime import registry

    records = [
        SessionRecord(
            pid=40_000 + index,
            kind="tui",
            session_id=f"live{index:08x}",
            conversation_name=f"live one {index}",
            cwd="/w",
            model_label="mock",
            control_port=1,
            control_key="k",
        )
        for index in range(count)
    ]
    monkeypatch.setattr(registry, "scan", lambda root=None: [(r, "live") for r in records])
    return [record.session_id for record in records]


def _ask(link: relay.PeerLink, **fields: Any) -> dict[str, Any]:
    reply = link.request({"op": "net_catalog", "req": 7, "locality": "remote", **fields})
    assert reply is not None, "no answer"
    return reply


def _detail(link: relay.PeerLink, **fields: Any) -> dict[str, Any]:
    reply = _ask(link, **fields)
    assert reply.get("op") == "ack", reply
    return dict(reply["detail"])


def _strip_clock(row: dict[str, Any]) -> dict[str, Any]:
    """Rows compared across two reads; ``age_s`` is measured per read."""
    return {key: value for key, value in row.items() if key != "age_s"}


def test_a_limit_100_walk_reaches_every_row_once_past_the_stored_cap(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, _server_b, _host, _port = pair
    stored = _stored(server_a.root, STORED)
    live = _live(monkeypatch, LIVE)
    _record, link = _live_link(pair, monkeypatch)
    try:
        assert wire.REMOTE_CATALOG_PAGE_V1 in link.capabilities
        unpaged = _detail(link)
        seen: list[str] = []
        cursor: str | None = None
        pages = 0
        while True:
            fields: dict[str, Any] = {"limit": 100}
            if cursor is not None:
                fields["cursor"] = cursor
            page = _detail(link, **fields)
            pages += 1
            assert page["total"] == STORED + LIVE
            assert len(page["sessions"]) <= 100
            seen.extend(row["session_id"] for row in page["sessions"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
            assert pages < 10, "the walk did not terminate"
    finally:
        link.close("test")
    # The unpaged answer is still the capped one (pollers' contract)...
    assert len(unpaged["sessions"]) == CATALOG_SCAN_LIMIT + LIVE
    assert "next_cursor" not in unpaged and "total" not in unpaged
    # ...and the walk reaches everything, once, live rows first.
    assert pages == 3
    assert len(seen) == len(set(seen)) == STORED + LIVE
    assert set(seen) == set(stored) | set(live)
    assert seen[:LIVE] == live


def test_paged_rows_are_the_unpaged_rows_key_for_key(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-role invariant: a page exposes no field the unpaged listing does not."""
    server_a, _server_b, _host, _port = pair
    _stored(server_a.root, 40)
    _live(monkeypatch, 2)
    _record, link = _live_link(pair, monkeypatch)
    try:
        unpaged = {row["session_id"]: row for row in _detail(link)["sessions"]}
        paged = _detail(link, limit=100)["sessions"]
    finally:
        link.close("test")
    assert [row["session_id"] for row in paged] == list(unpaged)
    for row in paged:
        assert set(row) == set(unpaged[row["session_id"]])
        assert _strip_clock(row) == _strip_clock(unpaged[row["session_id"]])


def test_a_remote_query_answers_only_matching_viewer_rows_with_no_extra_keys(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, _server_b, _host, _port = pair
    ids = _stored(server_a.root, 30, needle_every=5)
    _record, link = _live_link(pair, monkeypatch)
    try:
        unpaged = {row["session_id"]: row for row in _detail(link)["sessions"]}
        found = _detail(link, query="needle", limit=4)
        rest = _detail(link, query="needle", limit=4, cursor=found["next_cursor"])
        miss = _detail(link, query="no-such-words-anywhere")
    finally:
        link.close("test")
    wanted = {ids[index] for index in range(0, 30, 5)}
    assert found["total"] == len(wanted) == 6
    assert len(found["sessions"]) == 4 and rest["next_cursor"] is None
    got = [row["session_id"] for row in found["sessions"] + rest["sessions"]]
    assert set(got) == wanted and len(got) == len(set(got))
    for row in found["sessions"] + rest["sessions"]:
        # No snippet, digest or match flag: exactly the listing row.
        assert set(row) == set(unpaged[row["session_id"]])
        assert "needle" in row["conversation_name"]
    assert miss["sessions"] == [] and miss["total"] == 0 and miss["next_cursor"] is None


@pytest.mark.parametrize(
    "fields",
    [
        {"limit": 0},
        {"limit": 101},
        {"limit": True},
        {"limit": "5"},
        {"cursor": "not-a-token!!"},
        {"cursor": "x" * 600},
        {"cursor": 5},
        {"query": "q" * 600},
    ],
)
def test_a_malformed_page_frame_is_refused_with_a_sentence(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any]
) -> None:
    server_a, _server_b, _host, _port = pair
    _stored(server_a.root, 3)
    _record, link = _live_link(pair, monkeypatch)
    try:
        reply = _ask(link, **fields)
    finally:
        link.close("test")
    assert reply.get("op") != "ack", reply
    assert reply.get("message"), reply


def test_the_viewer_relay_pages_an_owner_and_labels_an_older_one(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``peer_session_rows`` with a page forwards it; an owner lacking the
    capability is not sent the fields and is marked ``unsupported``."""
    server_a, server_b, _host, _port = pair
    _stored(server_a.root, CATALOG_SCAN_LIMIT + 5)
    _record, link = _live_link(pair, monkeypatch)
    link.close("test")
    device_a = server_a.identity.device_id
    first = server_b._ctl_peer_session_rows({"limit": 100, "peer": device_a})
    block = first["peers"][device_a]
    assert first["paged"] is True
    assert block["paging"] == "paged" and block["total"] == CATALOG_SCAN_LIMIT + 5
    assert len(first["sessions"]) == 100 and isinstance(block["next_cursor"], str)
    # A cursor names one device: refused without it.
    with pytest.raises(relay.MeshRefusal):
        server_b._ctl_peer_session_rows({"cursor": block["next_cursor"]})
    # The federated listing is untouched by the new op's routing.
    assert "paged" not in server_b._ctl_peer_session_rows({})

    # An OLDER owner: the viewer sees no capability on the link.
    real_request = relay.PeerLink.request

    def old_owner(self: relay.PeerLink, frame: dict[str, Any], **kwargs: Any) -> Any:
        assert not any(key in frame for key in ("limit", "cursor", "query")), frame
        return real_request(self, frame, **kwargs)

    monkeypatch.setattr(
        relay.PeerLink,
        "capabilities",
        property(lambda self: frozenset()),
        raising=False,
    )
    monkeypatch.setattr(relay.PeerLink, "request", old_owner)
    searched = server_b._ctl_peer_session_rows({"query": "chat", "peer": device_a})
    assert searched["peers"][device_a]["paging"] == "unsupported"
    # A capped list is not the search that was asked for.
    assert searched["sessions"] == []
    limited = server_b._ctl_peer_session_rows({"limit": 5, "peer": device_a})
    assert limited["peers"][device_a]["paging"] == "unsupported"
    assert len(limited["sessions"]) == CATALOG_SCAN_LIMIT  # its capped default, labelled


def test_a_cursor_resumes_after_its_row_when_rows_arrive_above_it(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New sessions ranking above the position neither repeat nor skip a row."""
    server_a, _server_b, _host, _port = pair
    _stored(server_a.root, 10)
    _record, link = _live_link(pair, monkeypatch)
    try:
        first = _detail(link, limit=4)
        _live(monkeypatch, 2)  # two sessions start, ranked first
        time.sleep(0)
        second = _detail(link, limit=100, cursor=first["next_cursor"])
    finally:
        link.close("test")
    head = [row["session_id"] for row in first["sessions"]]
    tail = [row["session_id"] for row in second["sessions"]]
    assert not set(head) & set(tail)
    assert len(head) + len(tail) == 10


def test_the_cli_pages_a_real_owner_and_prints_the_next_page(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``lop network sessions --peer X --limit N [--cursor T]`` through the real
    parser, with this device's relay answering over a real link to the owner."""
    import argparse
    import json

    from local_operator.network import cli as net_cli

    server_a, server_b, _host, _port = pair
    _stored(server_a.root, CATALOG_SCAN_LIMIT + 5)
    _record, link = _live_link(pair, monkeypatch)
    link.close("test")
    name = server_a._own_label()
    monkeypatch.setattr(
        net_cli,
        "_relay_answer",
        lambda op, timeout=0.0, **fields: server_b._ctl_peer_session_rows(fields),
    )
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="subcommand"))

    def run(*argv: str) -> tuple[int, str]:
        rc = net_cli.main(parser.parse_args(["network", "sessions", *argv]))
        return rc, capsys.readouterr().out

    rc, text = run("--peer", name, "--limit", "100")
    assert rc == 0
    assert (
        f"{name}: 100 of {CATALOG_SCAN_LIMIT + 5} shown — next page: "
        f"lop network sessions --peer {name} --cursor "
    ) in text and text.rstrip().endswith("--limit 100")
    seen: list[str] = []
    cursor: str | None = None
    while True:
        argv = ["--json", "--peer", name, "--limit", "100"] + (
            ["--cursor", cursor] if cursor else []
        )
        rc, out = run(*argv)
        doc = json.loads(out)
        assert rc == 0 and doc["relay_paging"] is True
        seen += [row["session_id"] for row in doc["sessions"]]
        cursor = doc["peers"][server_a.identity.device_id]["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == CATALOG_SCAN_LIMIT + 5
    with pytest.raises(SystemExit):
        parser.parse_args(["network", "sessions", "--limit", "x"])
    assert (
        net_cli.main(parser.parse_args(["network", "sessions", "--all-peers", "--cursor", "t"]))
        == 2
    )


def _body_session(root: Path, session_id: str, title: str, body: str) -> None:
    """One stored conversation whose TRANSCRIPT alone holds ``body``."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "created_at.json").write_text("1700000999.0", encoding="utf-8")
    (directory / "transcript.jsonl").write_text(
        '{"id":"e1","ts":1,"type":"message","payload":{"kind":"message","role":"user",'
        f'"content":[{{"text":"please remember {body} for later"}}]}}}}\n',
        encoding="utf-8",
    )
    write_session_title(directory, title, user_set=False, past_names=[])


#: Sessions whose TRANSCRIPT alone holds the body-only term (their titles do not).
BODY_ONLY = 3


@pytest.mark.parametrize("can_view", [False, True])
def test_a_list_only_link_searches_names_not_transcript_content(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, can_view: bool
) -> None:
    """Agent review round 1, R1 on #4592 (red first on 9d7cd1e2b, list-only half).

    ``net_catalog`` needs only ``list``, so a link WITHOUT ``view`` (what peek
    needs) must not learn what a body holds from which rows a search returns.
    Same seeded store on both arms: the body-only term returns EXACTLY
    ``BODY_ONLY`` rows to a ``view`` link and NONE (nor its prefix) to a
    list-only one; a title term matches for both."""
    from local_operator.network import store
    from local_operator.network.relay import set_member_capabilities

    server_a, server_b, _host, _port = pair
    _stored(server_a.root, 5)
    body_ids = {f"b{index:011x}" for index in range(BODY_ONLY)}
    for index, session_id in enumerate(sorted(body_ids)):
        _body_session(server_a.root, session_id, f"plain chat body {index}", "zebra42secret")
    _body_session(server_a.root, "cccccccccccc", "quokka titled one", "nothing special")
    record, link = _live_link(pair, monkeypatch)
    try:
        if not can_view:
            with store.mutate(record.network_id, server_a.root) as rec:
                set_member_capabilities(rec, device_id=server_b.identity.device_id, revoke=["view"])
                store.save(rec, server_a.root)
        body = _detail(link, query="zebra42secret")
        prefix = _detail(link, query="zebra4")
        title = _detail(link, query="quokka titled")
    finally:
        link.close("test")
    got = {row["session_id"] for row in body["sessions"]}
    if can_view:
        assert got == body_ids and body["total"] == BODY_ONLY
    else:
        assert got == set() and body["total"] == 0
        assert prefix["sessions"] == [] and prefix["total"] == 0
    # A title term matches either way: names are what the row already shows.
    assert [row["session_id"] for row in title["sessions"]] == ["cccccccccccc"]
    assert body.get("search") == ("content" if can_view else "names")


class _Link:
    def __init__(self, capabilities: Any) -> None:
        self._capabilities = capabilities

    def role_capabilities(self) -> Any:
        if isinstance(self._capabilities, Exception):
            raise self._capabilities
        return self._capabilities


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        (None, False),
        (_Link(RuntimeError("store unreadable")), False),
        (_Link(None), False),
        (_Link("view"), False),
        (_Link(["view"]), False),
        (_Link({b"view"}), False),
        (_Link(set()), False),
        (_Link({"list"}), False),
        (_Link({"list", "view"}), True),
        (_Link(frozenset({"view"})), True),
    ],
)
def test_the_content_search_fails_closed(link: Any, expected: bool) -> None:
    """Delegator condition on R1: content only when ``view`` is POSITIVELY held."""
    assert relay._link_may_search_content(link) is expected


def test_a_cursor_whose_row_moved_up_resumes_at_its_offset() -> None:
    """Agent review round 1, R2: the anchor row moving UP (that session went
    live) must not restart the walk near the top and re-serve a page."""
    rows = [{"session_id": f"s{index:03d}"} for index in range(250)]
    token = relay._catalog_cursor_encode(100, "s099")
    assert relay._catalog_cursor_start(token, rows) == 100
    moved_up = [rows[99]] + rows[:99] + rows[100:]
    assert relay._catalog_cursor_start(token, moved_up) == 100
    # Rows arriving ABOVE it still resume right after the anchor row.
    arrived = [{"session_id": "new0"}, {"session_id": "new1"}] + rows
    assert relay._catalog_cursor_start(token, arrived) == 102
    # Anchor gone: the offset stands.
    assert relay._catalog_cursor_start(token, rows[:99] + rows[100:]) == 100


def test_an_owner_refusing_the_ask_is_answered_not_unreachable(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1, Q1: a malformed cursor is the OWNER's answered refusal —
    ``reachable`` stays true and the sentence rides ``refused``; the CLI with
    ``--peer`` exits as a refusal naming it, not as an empty listing."""
    import argparse

    from local_operator.network import cli as net_cli

    server_a, server_b, _host, _port = pair
    _stored(server_a.root, 3)
    _record, link = _live_link(pair, monkeypatch)
    link.close("test")
    device_a = server_a.identity.device_id
    answer = server_b._ctl_peer_session_rows({"cursor": "not-a-token!!", "peer": device_a})
    block = answer["peers"][device_a]
    assert block["reachable"] is True
    assert block["refused"] == "that cursor is not one this listing minted"
    assert block["refused_code"] == "bad_cursor" and answer["sessions"] == []
    monkeypatch.setattr(
        net_cli,
        "_relay_answer",
        lambda op, timeout=0.0, **fields: server_b._ctl_peer_session_rows(fields),
    )
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="subcommand"))
    args = parser.parse_args(
        ["network", "sessions", "--peer", server_a._own_label(), "--cursor", "not-a-token!!"]
    )
    with pytest.raises(relay.MeshRefusal) as refusal:
        net_cli._cmd_sessions(args)
    assert refusal.value.code == "bad_cursor"
    assert "not one this listing minted" in refusal.value.sentence
    # Through ``main`` it is a refusal exit, never rc 0 with an empty listing.
    assert net_cli.main(args) == 1


def test_the_printed_next_page_command_continues_the_same_search(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Design round 1, D1/D3: the footer's pointer, pasted VERBATIM through the
    real parser, continues the FILTERED walk at the same size (300 rows, half
    titled ``needle``) — it used to drop ``--query`` and turn the search into
    an unfiltered listing on page two."""
    import argparse
    import shlex

    from local_operator.network import cli as net_cli

    server_a, server_b, _host, _port = pair
    ids = _stored(server_a.root, 300, needle_every=2)
    needles = {ids[index] for index in range(0, 300, 2)}
    _record, link = _live_link(pair, monkeypatch)
    link.close("test")
    name = server_a._own_label()
    monkeypatch.setattr(
        net_cli,
        "_relay_answer",
        lambda op, timeout=0.0, **fields: server_b._ctl_peer_session_rows(fields),
    )
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="subcommand"))

    def run(argv: list[str]) -> str:
        assert net_cli.main(parser.parse_args(argv)) == 0
        return capsys.readouterr().out

    argv = ["network", "sessions", "--peer", name, "--limit", "40", "--query", "needle"]
    seen: list[str] = []
    for _ in range(10):
        out = run(argv)
        seen += [sid for sid in ids if sid in out.split()]
        pointer = [line for line in out.splitlines() if "next page: " in line]
        if not pointer:
            break
        assert "matching shown" in pointer[0]
        command = pointer[0].split("next page: ", 1)[1]
        words = shlex.split(command)
        assert words[:3] == ["lop", "network", "sessions"]
        assert "--query" in words and "--limit" in words
        argv = words[1:]
    assert len(seen) == len(set(seen)) == 150
    assert set(seen) == needles


@pytest.mark.parametrize("asked", [["--query", "x"], ["--limit", "5"]])
def test_an_older_owner_line_says_whether_the_search_ran(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], asked: list[str]
) -> None:
    """Design round 1, D5: with ``--query`` the older-owner line says the
    search did not run there, not merely that the listing is partial."""
    import argparse

    from local_operator.network import cli as net_cli

    block = {"device_id": "d_x", "name": "node2", "reachable": True, "paging": "unsupported"}
    monkeypatch.setattr(
        net_cli,
        "_relay_answer",
        lambda op, timeout=0.0, **fields: {"paged": True, "sessions": [], "peers": {"d_x": block}},
    )
    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="subcommand"))
    rc = net_cli.main(parser.parse_args(["network", "sessions", "--peer", "node2", *asked]))
    out = capsys.readouterr().out
    assert rc == 0
    if asked[0] == "--query":
        assert "node2: its build predates remote search, so the search did not run there" in out
    else:
        assert "node2: its build predates remote paging, so this is not its whole listing" in out
