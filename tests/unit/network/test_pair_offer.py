"""The both-ends join order: what will be shared, shown before anyone confirms.

THE PROPERTY THIS FILE PINS. A pairing used to be a code both humans compared and
nothing else: the joiner learned what it could borrow only after admission, and the
owner had no screen at all. The design (moment: the both-ends join screen) adds one
sealed frame — ``net_pair_offer`` — as the FIRST record after ``welcome``, gated on
a capability both sides must advertise (``pair-offer-v1``), so an old peer never
sees it and neither protocol version moves. The offer is display data: the owner
may only REDUCE the list it sent (``PairDecision.shares``), the grants are written
at admission from the owner's own store, and a refused/widened decision admits
nothing extra.

WHERE THE CELLS RUN. The module cells here exercise ``credentials/offers.py``
directly; the relay-side cells drive the real ``_run_pair_listener`` pair path and
the CLI's ``_join_one`` over real loopback sockets (the ``devices`` fixture), the
way ``test_join_park.py`` drives the two-phase pair. A cell that needs an "old
build" simulates it the only honest way available in-repo: by removing the
capability the old build would not have advertised, from the ONE tuple both
handshake paths read (``wire.LINK_CAPABILITIES``).

Isolation: every cell runs against the pytest ``root`` config dir (``conftest``),
an AuthStore is only ever opened under it, and nothing here touches the operator's
live config.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store, types, wire
from local_operator.network.credentials import offers
from local_operator.network.handshake import (
    PAIR_OFFER_WAIT_S,
    Credential,
    Handshake,
    pair_abort_frame,
    pair_ready_frame,
    pair_timeout_seconds,
    sas_matches,
)
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_join_park import _minted, _park_args, _wait_for_parked
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _answer_confirmation,
    _init_network,
    _type_the_code,
    devices,
)

# ---------------------------------------------------------------------------
# Seeding helpers — a root with a real auth.db (and optionally mcp.json)
# ---------------------------------------------------------------------------


def _store(root: Path) -> Any:
    """A real AuthStore at ``root`` (explicit db_path: never the ambient home)."""
    from local_operator.providers.auth_store import AuthStore

    return AuthStore(db_path=root / "auth.db", config_dir=root)


def _seed(root: Path, provider: str, payload: dict[str, Any]) -> None:
    auth = _store(root)
    try:
        auth.upsert_credential(provider, payload)
    finally:
        auth.close()


def _seed_mcp_json(root: Path, servers: dict[str, dict[str, Any]]) -> None:
    (root / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


# ---------------------------------------------------------------------------
# The candidate list (memo §7 cell 2), masking, defaults
# ---------------------------------------------------------------------------


def test_offer_carries_candidates_kinds_labels_defaults(root: Path) -> None:
    """One row per serveable credential, each classified and defaulted per §2.3.

    Seeded: an OAuth provider login (default YES), a pasted static key (default
    NO), a device-bound provider (excluded by name — a grant would be refused),
    the Radient org login (never auto-offered — §1.1 "there is no join-time
    default"), an MCP login (its kind is conservative in v1) and an MCP server
    with NO login row (excluded — offering it would promise a row the admission
    re-check drops).
    """
    _seed(root, "openai", {"refresh": "r", "access": "a", "email": "damian@example.com"})
    _seed(root, "anthropic", {"key": "sk-ant-1", "type": "api_key"})
    _seed(root, "kimi", {"key": "sk-kimi", "type": "api_key"})
    _seed(root, "radient", {"refresh": "r2", "access": "a2", "email": "ops@corp.example"})
    _seed(
        root,
        "mcp-oauth",
        {"project_id": "https://mcp.example/sse", "refresh": "r3", "access": "a3"},
    )
    _seed_mcp_json(
        root,
        {
            "tools": {"type": "sse", "url": "https://mcp.example/sse"},
            "cold": {"type": "http", "url": "https://cold.example/mcp"},
            "local": {"command": "npx", "args": ["-y", "something"]},
        },
    )

    items = offers.build_items(root)

    assert items == [
        {
            "key": "anthropic",
            "kind": "api-key-static",
            "label": "",
            "share": False,
        },
        {
            "key": "mcp:https://mcp.example/sse",
            "kind": "mcp-rotating",
            "label": "",
            "share": False,
        },
        {
            "key": "openai",
            "kind": "oauth-rotating",
            "label": "d***@example.com",
            "share": True,
        },
    ]
    # The exclusions are pinned BY NAME so a future filter change re-reads the
    # reason comment rather than flipping silently.
    keys = [item["key"] for item in items]
    assert "kimi" not in keys and "radient" not in keys
    assert not any(key.startswith("mcp:https://cold.example") for key in keys)


def test_offer_masks_labels_and_never_carries_a_raw_identity(root: Path) -> None:
    """The label's whole job is "which of my two logins is this" — masked.

    The frame carries the label MASKED (the memo's §5.1 label row: shown masked,
    never in the audit), so the raw address must not be findable in the wire
    bytes the joiner's screen renders from.
    """
    _seed(root, "openai", {"refresh": "r", "access": "a", "email": "damian@gmail.com"})
    items = offers.build_items(root)
    assert items[0]["label"] == "d***@gmail.com"
    assert "damian@gmail.com" not in json.dumps(items)

    assert offers.mask_label("") == ""
    assert offers.mask_label("abc123") == "a***"
    assert offers.mask_label("a@b") == "a***@b"


def test_offer_bounded_sorted_digest_matches_canonical_items(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Size, sort and digest: the three pins that make two implementations agree.

    A hostile or broken store cannot strain the record budget (the cap), the list
    is sorted by key (a stable digest input), and the digest is sha256 over the
    canonical serialisation of exactly the items sent.
    """
    from local_operator.network import wire

    many = [
        {"key": f"provider-{index:03d}", "kind": "api-key-static", "label": ""}
        for index in range(offers.MAX_OFFER_ITEMS + 6)
    ]
    monkeypatch.setattr(offers, "enumerate_candidates", lambda config: list(reversed(many)))

    items = offers.build_items(root)
    assert len(items) == offers.MAX_OFFER_ITEMS
    assert [item["key"] for item in items] == sorted(item["key"] for item in items)

    digest = offers.digest_of(items)
    assert digest == hashlib.sha256(wire.canonical_json(items)).hexdigest()
    # One changed byte in one row re-pins the digest.
    mutated = [dict(item) for item in items]
    mutated[-1]["share"] = not mutated[-1]["share"]
    assert offers.digest_of(mutated) != digest

    # And a list that is exactly at the cap is kept whole.
    monkeypatch.setattr(
        offers, "enumerate_candidates", lambda config: many[: offers.MAX_OFFER_ITEMS]
    )
    assert len(offers.build_items(root)) == offers.MAX_OFFER_ITEMS


def test_offer_validation_refuses_unusable_items() -> None:
    """A frame the parser cannot trust is refused, never shown partially."""
    good = {"key": "openai", "kind": "oauth-rotating", "label": "d***@x", "share": True}
    assert offers.validate_items([good]) == [good]

    for bad in (
        "not a list",
        [good] * (offers.MAX_OFFER_ITEMS + 1),
        [{"key": "", "kind": "oauth-rotating", "label": "", "share": True}],
        [{"key": "x", "kind": "secret", "label": "", "share": True}],
        [{"key": "x", "kind": "oauth-rotating", "label": "", "share": "yes"}],
        [
            {
                "key": "x",
                "kind": "oauth-rotating",
                "label": "x" * (offers.MAX_OFFER_LABEL + 1),
                "share": True,
            }
        ],
        [["nope"]],
    ):
        with pytest.raises(ValueError):
            offers.validate_items(bad)


def test_offer_reads_degrade_by_absence_and_raise_by_unreadability(root: Path) -> None:
    """No store is an honest empty offer; an unreadable store is SAYABLE.

    The two must not collapse: ``_pair_offer_for`` sends an empty offer either
    way, but only the unreadable case records ``enumeration: "unreadable"``, so
    "why was my list empty" has an answer.
    """
    assert offers.build_items(root) == []

    (root / "auth.db").mkdir()
    with pytest.raises(offers.OfferEnumerationError):
        offers.build_items(root)


# ---------------------------------------------------------------------------
# The rendered lines (one contract, four surfaces)
# ---------------------------------------------------------------------------


def test_offer_lines_and_states_are_one_contract() -> None:
    """The words both humans read, exactly — they are a contract, not formatting.

    §3.1's joiner block, §3.3's owner block, and §4.3/§4.4's two DIFFERENT
    "nothing" lines ("did not offer" is about a build; "offered nothing" is
    about a device's store).
    """
    items = [
        {"key": "openai", "kind": "oauth-rotating", "label": "d***@gmail.com", "share": True},
        {"key": "anthropic", "kind": "api-key-static", "label": "", "share": False},
    ]
    assert offers.render_joiner_block(items, inviter="damian-mbp") == [
        "Credentials damian-mbp will serve to this device:",
        "  openai (OAuth, d***@gmail.com)   will be served",
        "  anthropic (API key)   not offered",
        "the inviter can remove items before admitting; nothing else will be served.",
    ]
    assert offers.render_owner_block(items, joiner="laptop") == [
        "Credentials this device will serve to laptop:",
        "  openai (OAuth, d***@gmail.com)   will be served",
        "  anthropic (API key)   not offered",
        "the list can only shrink; nothing else will be served.",
    ]
    assert offers.render_state_line(offers.OFFER_EMPTY) == "no credentials were offered"
    assert "older build" in offers.render_state_line(offers.OFFER_ABSENT)

    assert offers.served_keys(items) == ["openai"]
    assert offers.owner_default_shares(items) == ["openai"]
    assert offers.sentence_clause(offers.OFFER_LISTED, items, inviter="damian-mbp") == (
        "damian-mbp will serve: openai."
    )
    assert offers.sentence_clause(offers.OFFER_EMPTY, [], inviter="x").startswith("No credentials")
    assert offers.sentence_clause(offers.OFFER_ABSENT, [], inviter="x").startswith(
        "The other device did not offer"
    )


# ---------------------------------------------------------------------------
# The drain: capability gate, bound, refusal mapping (memo §7 cells 6 and 12)
# ---------------------------------------------------------------------------


class _StubHandshake:
    """Just enough of a Handshake for the drain: the owner's advertised caps."""

    def __init__(self, caps: list[str]) -> None:
        self.peer_capabilities = list(caps)


class _StubReader:
    def __init__(self, payloads: list[Any]) -> None:
        self.payloads = list(payloads)
        self.deadlines: list[float] = []

    def read_record_payload(self, deadline: float) -> Any:
        self.deadlines.append(deadline)
        return self.payloads.pop(0)


class _NeverReader:
    """A reader that FAILS THE CELL if the drain reads it — the old-owner path
    must do zero reads, and "did not read" is asserted structurally, not by a wait."""

    def read_record_payload(self, deadline: float) -> Any:  # pragma: no cover - must not run
        raise AssertionError("the drain read a record from an owner without the capability")


class _StubCodec:
    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.frames = list(frames)

    def open(self, payload: Any) -> dict[str, Any]:
        return self.frames.pop(0)

    def seal(self, frame: dict[str, Any]) -> bytes:
        return b"sealed"


def _offer_frame(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"op": "net_pair_offer", "items": items, "digest": offers.digest_of(items)}


def _advertised() -> _StubHandshake:
    return _StubHandshake([wire.MESH_NET_V1, wire.PAIR_OFFER_V1])


def test_the_drain_reads_one_record_within_its_bound() -> None:
    """THE BOUND IS ASSERTED ON THE DEADLINE VALUE, never on elapsed time.

    The drain must hand the reader a deadline no further out than
    ``PAIR_OFFER_WAIT_S`` (capped by what is left of the invite) — a rendezvous
    bound, not a human budget: the owner sends the offer immediately after
    ``welcome`` with no human step in between, and a slow frame is still absorbed
    by ``_finish_pairing``'s tolerant first read, so the bound only decides how
    long the ON-SCREEN list waits to appear.
    """
    items = [{"key": "openai", "kind": "oauth-rotating", "label": "", "share": True}]
    reader = _StubReader([b"one sealed record"])
    before = time.monotonic()
    state, got = net_cli._drain_pair_offer(  # noqa: SLF001 — the CLI's own helper
        handshake=_advertised(),
        codec=_StubCodec([_offer_frame(items)]),
        reader=reader,
        remaining_s=30.0,
    )
    assert (state, got) == ("listed", items)
    assert before <= reader.deadlines[0] <= before + PAIR_OFFER_WAIT_S + 0.5

    # The invite's own remaining life caps it when that is smaller.
    capped = _StubReader([b"one sealed record"])
    before = time.monotonic()
    net_cli._drain_pair_offer(  # noqa: SLF001
        handshake=_advertised(),
        codec=_StubCodec([_offer_frame(items)]),
        reader=capped,
        remaining_s=0.4,
    )
    assert capped.deadlines[0] - before <= 0.5


def test_the_drain_does_no_read_for_an_owner_without_the_capability() -> None:
    """An older owner: no read, no wait — the legacy sequence is byte-identical.

    The gate is the owner's OWN advertisement from its challenge. A read attempt
    here would be a two-second stall on every pairing with an old build, which is
    exactly the cost the capability gate exists to avoid.
    """
    state, items = net_cli._drain_pair_offer(  # noqa: SLF001
        handshake=_StubHandshake([wire.MESH_NET_V1]),
        codec=_StubCodec([]),
        reader=_NeverReader(),
        remaining_s=30.0,
    )
    assert (state, items) == ("absent", [])


def test_the_drain_maps_an_abort_and_refuses_a_malformed_or_unverifiable_frame() -> None:
    """Anything other than an offer on this record is a refusal, never a downgrade.

    ``net_pair_abort`` goes through the ONE reason-to-sentence map; a frame of the
    wrong shape is a protocol error; and an offer whose digest does not match its
    items is refused too — the digest is a checksum, but the end that receives a
    list it cannot verify must say so rather than show it (§6.2: a corrupt offer
    is a hard refusal).
    """
    abort = {"op": "net_pair_abort", "reason": "declined_remote", "detail": "not this time"}
    with pytest.raises(types.PairingRefusal) as refused:
        net_cli._drain_pair_offer(  # noqa: SLF001
            handshake=_advertised(),
            codec=_StubCodec([abort]),
            reader=_StubReader([b"abort"]),
            remaining_s=30.0,
        )
    assert "not this time" in refused.value.sentence

    wrong_op = {"op": "net_pair_ready", "req": 1, "sas": "123456"}
    with pytest.raises(types.MeshRefusal) as refused2:
        net_cli._drain_pair_offer(  # noqa: SLF001
            handshake=_advertised(),
            codec=_StubCodec([wrong_op]),
            reader=_StubReader([b"ready"]),
            remaining_s=30.0,
        )
    assert refused2.value.code == "protocol_error"

    items = [{"key": "openai", "kind": "oauth-rotating", "label": "", "share": True}]
    bad_digest = {"op": "net_pair_offer", "items": items, "digest": "00" * 32}
    with pytest.raises(types.MeshRefusal) as refused3:
        net_cli._drain_pair_offer(  # noqa: SLF001
            handshake=_advertised(),
            codec=_StubCodec([bad_digest]),
            reader=_StubReader([b"offer"]),
            remaining_s=30.0,
        )
    assert refused3.value.code == "protocol_error"


# ---------------------------------------------------------------------------
# The wire: the offer is the FIRST sealed record after welcome (memo §7 cell 1)
# ---------------------------------------------------------------------------


class _RecordingCrypto:
    """The real codec, recording every frame OPENED (order = arrival order)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.opened: list[dict[str, Any]] = []

    def open(self, payload: Any) -> dict[str, Any]:
        frame = self.inner.open(payload)
        self.opened.append(frame)
        return frame

    def seal(self, frame: dict[str, Any]) -> bytes:
        return self.inner.seal(frame)


class _RecordingHandshake(Handshake):
    """The real Handshake, handing out a recording codec.

    The joiner reads records STRICTLY in arrival order, so what this records is
    the socket's record order from the side the claim is about — no second client
    and no relay surgery needed. ``codec()`` is called once per join, by
    ``_join_one``.
    """

    last: Any = None

    def codec(self) -> Any:
        crypto = _RecordingCrypto(super().codec())
        type(self).last = crypto
        return crypto


def _join_with_recorder(server_b: relay.RelayServer, *, host: str, port: int, minted: Any) -> Any:
    """``test_relay_e2e._join``, with the recording handshake class swapped in."""
    args = Namespace(
        sas_stdin=True,
        verify=False,
        emit_sas=True,
        name=server_b.identity.name,
        json=True,
    )
    return net_cli._join_one(  # noqa: SLF001 — the CLI's own driver, as the CLI runs it
        host=f"{host}:{port}",
        token=minted.token,
        envelope=minted.envelope,
        identity=server_b.identity,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        args=args,
        wire=wire,
        Handshake=_RecordingHandshake,
        Credential=Credential,
        pair_abort_frame=pair_abort_frame,
        pair_timeout_seconds=pair_timeout_seconds,
        sas_matches=sas_matches,
        invite_mod=invite_mod,
        store=store,
        relay_mod=relay,
    )


def test_offer_is_the_first_sealed_record_after_welcome(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record order AND its content: welcome (plaintext), offer, result.

    The owner seeds one OAuth login, so the offer is non-empty and its row carries
    the masked label and the §2.3 default (``share: true``). The digest is
    recomputed from the received items, which is what the joiner's parser does.
    """
    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "damian@example.com"})
    record = _init_network(server_a)
    state = store.load_secrets(record.network_id, server_a.root)
    minted = invite_mod.mint(record, state.secret, role="drive", ttl_s=600.0)
    record.invites.append(minted.record)
    store.save(record, server_a.root)
    store.save_invite_token(minted.record.invite_id, minted.token, server_a.root)
    _type_the_code(monkeypatch)
    answered: dict[str, Any] = {}
    thread = threading.Thread(
        target=lambda: answered.update(_answer_confirmation(server_a) or {}), daemon=True
    )
    thread.start()
    try:
        _join_with_recorder(server_b, host=host, port=port, minted=minted)
    finally:
        thread.join(20)

    crypto = _RecordingHandshake.last
    assert crypto is not None
    # EXACTLY the op sequence, in order: the offer first, then the result. No
    # other sealed record was read (an extra read would be a second frame on the
    # wire this joiner was not built for).
    assert [frame.get("op") for frame in crypto.opened] == [
        "net_pair_offer",
        "net_pair_result",
    ]
    offer = crypto.opened[0]
    assert offer["items"] == [
        {
            "key": "openai",
            "kind": "oauth-rotating",
            "label": "d***@example.com",
            "share": True,
        }
    ]
    assert offer["digest"] == offers.digest_of(offer["items"])


# ---------------------------------------------------------------------------
# Phase one carries the list (memo §7 cells 4, 5, 7)
# ---------------------------------------------------------------------------


def _park(
    server_b: relay.RelayServer, *, host: str, port: int, minted: Any, json_mode: bool
) -> Any:
    """Phase one, driven exactly as ``lop network join --park`` drives it."""
    return net_cli._join_one(  # noqa: SLF001 — the CLI's own driver, run as the CLI runs it
        host=f"{host}:{port}",
        token=minted.token,
        envelope=minted.envelope,
        identity=server_b.identity,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        args=_park_args(json=json_mode, name=server_b.identity.name),
        wire=wire,
        Handshake=Handshake,
        Credential=Credential,
        pair_abort_frame=pair_abort_frame,
        pair_timeout_seconds=pair_timeout_seconds,
        sas_matches=sas_matches,
        invite_mod=invite_mod,
        store=store,
        relay_mod=relay,
    )


def _park_in_background(
    server_b: relay.RelayServer, *, host: str, port: int, minted: Any, json_mode: bool
) -> tuple[threading.Thread, dict[str, Any], list[BaseException]]:
    parked: dict[str, Any] = {}
    failures: list[BaseException] = []

    def _run() -> None:
        try:
            parked["result"] = _park(
                server_b, host=host, port=port, minted=minted, json_mode=json_mode
            )
        except BaseException as exc:  # noqa: BLE001 — asserted by the caller
            failures.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, parked, failures


def _printed(capsys: pytest.CaptureFixture[str], *, until: Any, timeout_s: float = 15.0) -> str:
    """Accumulate captured stdout until ``until`` accepts it (the print happens
    in the parked thread, after the record is written — wait on the OUTPUT too)."""
    buf = ""

    def _ready() -> bool:
        nonlocal buf
        buf += capsys.readouterr().out
        return until(buf)

    assert net_fixtures.wait_for(_ready, timeout_s=timeout_s), f"the output never appeared: {buf!r}"
    return buf


def test_join_park_payload_carries_offers_when_the_owner_advertises(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The park payload: the same rows as data, and a sentence that says them in words.

    The record must carry them too, so the phase-two invocation and any later
    reader see the list the human was shown rather than a re-enumeration.
    """
    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "damian@example.com"})
    _record, minted = _minted(server_a, ttl_s=3.0)
    thread, _parked, _failures = _park_in_background(
        server_b, host=host, port=port, minted=minted, json_mode=True
    )
    try:
        row = _wait_for_parked(server_b.root)
        assert row.offer_state == offers.OFFER_LISTED
        assert [item["key"] for item in row.offers] == ["openai"]
        assert row.offers[0]["share"] is True
        raw = _printed(capsys, until=lambda text: """}""" in text)
        payload = json.loads(raw[: raw.rindex("}") + 1])
        assert payload["offers"] == row.offers
        assert "will serve: openai" in payload["sentence"]
    finally:
        thread.join(15)


def test_join_prompt_prints_the_offer_rows(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The TTY prompt path shows the rows before the human types — the join-side
    screen the design puts under the code block (§3.1)."""
    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "damian@example.com"})
    _record, minted = _minted(server_a, ttl_s=3.0)
    thread, _parked, _failures = _park_in_background(
        server_b, host=host, port=port, minted=minted, json_mode=False
    )
    try:
        _wait_for_parked(server_b.root)
        out = _printed(capsys, until=lambda text: "will be served" in text)
        assert "will serve to this device:" in out
        assert "openai (OAuth, d***@example.com)   will be served" in out
        assert "the inviter can remove items before admitting; nothing else will be served." in out
    finally:
        thread.join(15)


def test_old_owner_path_shows_the_not_offered_line_and_changes_nothing(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """§4.3, pinned the only honest way in-repo: remove the capability BOTH ends
    would have advertised after an upgrade, and the ceremony is exactly today's —
    no offer on the wire, the joiner says so, nothing is admitted, the invite is
    untouched (§G1's both-sides half; the one-sided halves are the mixed cells).
    """
    old = tuple(name for name in wire.LINK_CAPABILITIES if name != wire.PAIR_OFFER_V1)
    monkeypatch.setattr(wire, "LINK_CAPABILITIES", old)
    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "damian@example.com"})
    _record, minted = _minted(server_a, ttl_s=2.0)
    thread, _parked, failures = _park_in_background(
        server_b, host=host, port=port, minted=minted, json_mode=True
    )
    try:
        row = _wait_for_parked(server_b.root)
        assert row.offer_state == offers.OFFER_ABSENT
        assert row.offers == []
        raw = _printed(capsys, until=lambda text: """}""" in text)
        payload = json.loads(raw[: raw.rindex("}") + 1])
        assert "offers" not in payload
        assert "did not offer credentials" in payload["sentence"]
        # The OWNER was never asked anything: no transcription ever reached it
        # (the joiner types nothing while parked, and this ceremony has no answer).
        assert server_a._ctl_pair_pending({}) == []  # noqa: SLF001 — the CLI's own op
    finally:
        thread.join(15)
    assert failures and isinstance(failures[0], types.JoinParkUnanswered)
    assert store.list_networks(server_b.root) == []


# ---------------------------------------------------------------------------
# One offer per ceremony (memo §7 cell 12)
# ---------------------------------------------------------------------------


class _StubSock:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def sendall(self, payload: bytes) -> None:
        self.sent.append(payload)


def test_second_offer_is_a_protocol_error(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """§6.6: a SECOND offer is a protocol error, never a second screen.

    Driven where the race lives: the drain already saw one offer, and the tolerant
    first read of the result half then receives another. One offer exists per
    ceremony; a peer that sends two is driving a state machine this side does not
    have, and tolerating it again would let a peer append rows nobody reviewed.
    """
    server_a, _server_b, _host, _port = devices
    record = _init_network(server_a)
    state = store.load_secrets(record.network_id, server_a.root)
    minted = invite_mod.mint(record, state.secret, role="drive", ttl_s=600.0)
    items = [{"key": "openai", "kind": "oauth-rotating", "label": "", "share": True}]
    second = _offer_frame(items)
    with pytest.raises(types.MeshRefusal) as refused:
        net_cli._finish_pairing(  # noqa: SLF001 — the joining half of the ceremony
            typed="123456",
            sock=_StubSock(),
            codec=_StubCodec([second]),
            reader=_StubReader([b"late offer"]),
            envelope=minted.envelope,
            host="127.0.0.1:1",
            identity=server_a.identity,
            store=store,
            handshake=None,
            advertised=[],
            fingerprint="AAAA-BBBB-CCCC-DDDD",
            offer_view={"state": offers.OFFER_LISTED, "items": list(items)},
            helpers={
                "invite_mod": invite_mod,
                "pair_timeout_seconds": pair_timeout_seconds,
                "pair_ready_frame": pair_ready_frame,
            },
        )
    assert refused.value.code == "protocol_error"
    assert "second share list" in refused.value.sentence
