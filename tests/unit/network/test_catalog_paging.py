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
    assert f"{name}: 100 of {CATALOG_SCAN_LIMIT + 5} shown — next page: --peer {name}" in text
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
