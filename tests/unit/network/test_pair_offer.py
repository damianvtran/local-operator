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
    """One row per serveable credential, each classified and defaulted per §1.4.

    Seeded: an OAuth provider login (default YES), a pasted static key (default
    YES since the §1.4 flip — the class that works from a second device), a
    device-bound provider (excluded by name — a grant would be refused), the
    Radient org login (offered by default since §1.4, reduce-only), an MCP login
    (default YES since §1.4) and an MCP server with NO login row (excluded —
    offering it would promise a row the admission re-check drops).
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
            "share": True,
        },
        {
            "key": "mcp:https://mcp.example/sse",
            "kind": "mcp-rotating",
            "label": "",
            "share": True,
        },
        {
            "key": "openai",
            "kind": "oauth-rotating",
            "label": "d***@example.com",
            "share": True,
        },
        {
            "key": "radient",
            "kind": "oauth-rotating",
            "label": "o***@corp.example",
            "share": True,
        },
    ]
    # The remaining exclusion is pinned BY NAME so a future filter change re-reads
    # the reason comment rather than flipping silently.
    keys = [item["key"] for item in items]
    assert "kimi" not in keys
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
    """A frame the parser cannot trust is refused, never shown partially.

    Note what is NOT in the refusal list: an unknown KIND. Tolerating it is the
    row-schema rule (``canonical_rows``) and cell-pinned below.
    """
    good = {"key": "openai", "kind": "oauth-rotating", "label": "d***@x", "share": True}
    assert offers.validate_items([good]) == [good]

    for bad in (
        "not a list",
        [good] * (offers.MAX_OFFER_ITEMS + 1),
        [{"key": "", "kind": "oauth-rotating", "label": "", "share": True}],
        [{"key": "x", "kind": "", "label": "", "share": True}],
        [{"key": "x", "kind": "k" * (offers.MAX_OFFER_KIND + 1), "label": "", "share": True}],
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


def test_the_row_schema_evolves_additively_only() -> None:
    """M1's rule, pinned: additive fields and kinds ride pair-offer-v1, and the
    digest still catches REWRITES (so tampering fails even as additions pass).

    The failure this prevents: a future add-a-field change holds a legitimate
    ``pair-offer-v1``, and old joiners refuse EVERY pairing with it — while the
    code reads as if the only risk were a classifier bug.
    """
    from local_operator.network.handshake import parse_pair_offer

    good = {"key": "openai", "kind": "oauth-rotating", "label": "d***@x", "share": True}

    # An additive per-row FIELD: same digest (the projection drops it), and the
    # parser returns the canonical row rather than refusing the frame.
    richer = dict(good, pool="team-a")
    assert offers.digest_of([richer]) == offers.digest_of([good])
    assert offers.validate_items([richer]) == [good]
    frame = {"op": "net_pair_offer", "items": [richer], "digest": offers.digest_of([richer])}
    assert parse_pair_offer(frame) == [good]

    # An additive KIND: validated, displayed through the fallback, never able to
    # grant (grants are the owner's, key-based, against its own store).
    future = {"key": "newthing", "kind": "future-kind", "label": "", "share": True}
    assert offers.validate_items([future]) == [future]
    assert offers.kind_label("future-kind") == "future-kind"
    rows = offers.render_rows([future])
    assert any("future-kind" in row and "will be served" in row for row in rows)
    frame = {"op": "net_pair_offer", "items": [future], "digest": offers.digest_of([future])}
    assert parse_pair_offer(frame) == [future]

    # But MEANING is frozen: a rewrite of an existing field fails the digest —
    # that is the half that makes the tolerance above safe.
    rewritten = [dict(good, share=False)]
    assert offers.digest_of(rewritten) != offers.digest_of([good])
    with pytest.raises(types.MeshRefusal) as rewritten_refusal:
        parse_pair_offer(
            {"op": "net_pair_offer", "items": rewritten, "digest": offers.digest_of([good])}
        )
    assert rewritten_refusal.value.code == "protocol_error"


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
        "  anthropic (API key)              not offered",
        "the inviter can remove items before admitting; nothing else will be served.",
    ]
    assert offers.render_owner_block(items, joiner="laptop") == [
        "Credentials this device will serve to laptop:",
        "  openai (OAuth, d***@gmail.com)   will be served",
        "  anthropic (API key)              not offered",
        "the list can only shrink; nothing else will be served.",
    ]
    # The confirm screen's re-render (D1/U1): same block, states from the CURRENT
    # decision — a served key the person removed reads `no longer served`.
    reprint = offers.render_rows(items, decision=[])
    assert reprint[0] == "  openai (OAuth, d***@gmail.com)   no longer served"
    assert reprint[1].endswith("(API key)" + " " * 14 + "not offered")
    assert (
        offers.render_rows(items, decision=["openai"])[0]
        == "  openai (OAuth, d***@gmail.com)   will be served"
    )
    assert offers.render_state_line(offers.OFFER_EMPTY) == "no credentials were offered"
    # §4.3 names the two causes the state actually covers — an older build, or an
    # offer that did not arrive; "has none to share" is §4.4's fact (D4).
    absent = offers.render_state_line(offers.OFFER_ABSENT)
    assert "older build" in absent and "offer did not arrive" in absent
    assert "has none to share" not in absent

    assert offers.served_keys(items) == ["openai"]
    assert offers.owner_default_shares(items) == ["openai"]
    assert offers.sentence_clause(offers.OFFER_LISTED, items, inviter="damian-mbp") == (
        "damian-mbp will serve: openai."
    )
    assert offers.sentence_clause(offers.OFFER_EMPTY, [], inviter="x").startswith("No credentials")
    assert offers.sentence_clause(offers.OFFER_ABSENT, [], inviter="x").startswith(
        "The other device did not offer"
    )


def test_the_receipt_delta_keeps_a_decision_apart_from_a_failure() -> None:
    """U2's sentence contract, both halves — and the pronoun follows the count.

    A deliberate reduction gets NO remedy: pointing the joiner at the share verb
    would re-open a decision the owner already took. The failed path keeps the
    verb, which is the one thing they can act on.
    """
    assert offers.missing_share_lines(["openai"], [], ["openai"]) == [
        "not available here: openai — the other device chose not to share it"
    ]
    assert offers.missing_share_lines(["a", "b"], [], ["a", "b"]) == [
        "not available here: a, b — the other device chose not to share them"
    ]
    assert offers.missing_share_lines(["openai"], [], []) == [
        "not available here: openai — ask the other device to run "
        "`lop network credential share <key> --with <device>` to lend it after the join"
    ]
    # Both causes at once, deliberately first; nothing that WAS served is named.
    assert offers.missing_share_lines(["a", "b", "c"], ["a"], ["c"]) == [
        "not available here: c — the other device chose not to share it",
        "not available here: b — ask the other device to run "
        "`lop network credential share <key> --with <device>` to lend it after the join",
    ]


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


class _FailingCodec:
    """The real codec with its FIRST ``open`` refused — a forced sealed-phase failure.

    A real authentication failure needs a key divergence, so the inviter-side cell
    corrupts bytes in flight; here the point is the JOINER's own catch-and-record
    path, so the codec is the real one and only ``open`` refuses — with the real
    exception class, message and kind, which is exactly what a wire failure raises.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.refused = False

    def open(self, payload: Any) -> dict[str, Any]:
        if not self.refused:
            self.refused = True
            raise wire.LinkCryptoError(
                "a record failed authentication: the link is closed and nothing in it "
                "is repaired",
                kind="auth",
            )
        return self.inner.open(payload)

    def seal(self, frame: dict[str, Any]) -> bytes:
        return self.inner.seal(frame)


class _FailingHandshake(Handshake):
    """The real ``Handshake``, handing out a codec whose first open refuses."""

    def codec(self) -> Any:
        return _FailingCodec(super().codec())


def test_a_forced_sealed_failure_leaves_the_joiner_its_own_record(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """F4 (drill 2026-10-04): a sealed record that fails authentication names its
    stage and class LOCALLY on the joining end.

    Run 10 stopped at this layer and read as "no entries": the payload said only
    ``(LinkCryptoError)`` and nothing said WHERE (which read) or WHICH kind. The
    forced failure lands on the FIRST sealed read — the offer drain — because both
    ends advertise ``pair-offer-v1``, so "the first sealed read failed" is the
    reading a re-run must find in the record.
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a)
    args = _park_args(park=False, automated=True, name=server_b.identity.name)

    result = net_cli._join_one(  # noqa: SLF001 — the CLI's own driver, run as the CLI runs it
        host=f"{host}:{port}",
        token=minted.token,
        envelope=minted.envelope,
        identity=server_b.identity,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        args=args,
        wire=wire,
        Handshake=_FailingHandshake,
        Credential=Credential,
        pair_abort_frame=pair_abort_frame,
        pair_timeout_seconds=pair_timeout_seconds,
        sas_matches=sas_matches,
        invite_mod=invite_mod,
        store=store,
        relay_mod=relay,
    )

    # 1. The sentence is the one run 10 saw, UNCHANGED (the wire copy is pinned).
    assert isinstance(result, str)
    assert "(LinkCryptoError)" in result

    # 2. The local record names stage + class + kind, read back from the store...
    attempt = store.join_attempt()
    assert attempt is not None
    assert attempt["ok"] is False
    assert attempt["stage"] == "offer_read"
    assert attempt["class"] == "link_crypto"
    assert attempt["kind"] == "auth"
    assert attempt["host"] == f"{host}:{port}"
    assert attempt["invite_id"] == minted.envelope.invite_id
    assert attempt["network_id"] == record.network_id
    assert attempt["records_sent"] == 0
    assert attempt["records_received"] == 0
    # ...and the block is marked local-only, which is what keeps an F4 reader from
    # treating it as something the peer said.
    assert attempt["local_only"] is True
    # NEGATIVE: nothing was admitted; the failure did not sneak a member in.
    assert store.load(record.network_id, server_a.root).member(server_b.identity.device_id) is None


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


# ---------------------------------------------------------------------------
# The owner's confirm screen (memo §7 cells 8 and 9)
# ---------------------------------------------------------------------------

_OWNER_ITEMS: list[dict[str, Any]] = [
    {"key": "openai", "kind": "oauth-rotating", "label": "d***@example.com", "share": True},
    {"key": "anthropic", "kind": "oauth-rotating", "label": "", "share": True},
    {"key": "legacy-key", "kind": "api-key-static", "label": "", "share": False},
]


def _confirm_args(**overrides: Any) -> Namespace:
    base: dict[str, Any] = {
        "json": True,
        "invite_id": "",
        "list_pending": False,
        "decline": False,
        "sas_stdin": False,
    }
    base.update(overrides)
    return Namespace(**base)


def _pending_with_offer(
    root: Path, *, items: list[dict[str, Any]] | None = None
) -> types.PendingPairing:
    """The record a relay parks once it has SENT a share list (state ``sent``)."""
    pending = types.PendingPairing(
        invite_id="i_offer1",
        network_id="n_0123456789abcdef01234567",
        network_name="home-net",
        joiner_device_id="d_" + "b" * 32,
        joiner_name="laptop",
        sas="481926",
        fingerprint="K7QM-3XPD-4WZ9-8NRB",
        transcribed="481926",
        peer_addr="127.0.0.1:4097",
        expires_at=time.time() + 120,
        prompt=(
            'd_bbbb… ("laptop", new device) transcribed 481 926 to join home-net as drive.\n'
            "YOUR screen shows 481 926.\n\n"
            "Credentials this device will serve to laptop:\n"
            "  openai (OAuth, d***@example.com)   will be served\n"
            "  anthropic (OAuth)                  will be served\n"
            "  legacy-key (API key)               not offered\n"
            "the list can only shrink; nothing else will be served.\n\n"
            "Do they match? Confirm only if the other device shows the same code."
        ),
        offer=list(items if items is not None else _OWNER_ITEMS),
        offer_state="sent",
    )
    store.save_pending_pairing(pending, root)
    return pending


def test_confirm_screen_shows_the_offer_and_a_reduction_lands_in_the_decision(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§3.3: the owner's question carries the list, ``t`` unchecks a served row, and
    ONLY the reduced set is written into the decision — the thing admission reads."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _pending_with_offer(root)
    monkeypatch.setattr(net_cli, "_has_terminal", lambda: True)
    prompts: list[str] = []
    answers = iter(["t", "openai", "y"])

    def _answer(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", _answer)
    assert net_cli._cmd_confirm(_confirm_args()) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    assert "openai (OAuth, d***@example.com)   will be served" in out
    assert any("t to remove what will be served" in prompt for prompt in prompts)
    # The re-render (D1/U1): the block prints again from the CURRENT decision, so
    # the frame the final `y` lands on shows openai as removed — not as served.
    assert out.count("Credentials this device will serve") == 2
    assert "openai (OAuth, d***@example.com)   no longer served" in out
    # D9 (design round 2): the re-asked question gets its blank line back — the
    # re-printed block ends with the tail plus one blank, like the first frame.
    assert out.count("nothing else will be served.\n\n") == 2
    assert "anthropic (OAuth)" + " " * 18 + "will be served" in out
    decision = store.pair_decision("i_offer1", root)
    assert decision is not None and decision.matched and decision.decision == "admit"
    assert decision.shares == ["anthropic"]


def test_confirm_refuses_an_unoffered_addition_by_name(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§3.3 + design round 1, D8: nothing can be ADDED at confirm — the CLI refuses
    the key BY NAME and RE-ASKS in place (the command is not exited, and nothing is
    lost but the screen), while the relay still refuses a widening that arrives in a
    frame (a frame is not a place to re-ask)."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _pending_with_offer(root)
    monkeypatch.setattr(net_cli, "_has_terminal", lambda: True)
    answers = iter(["t", "t", "ghost-key", "openai", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert net_cli._cmd_confirm(_confirm_args()) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    # U8 (round 2): a stray `t` answered here is the prompt's own word, not a key
    # — it must not be pointed at `credential share t`.
    assert "'t' is not a known key here" in out
    assert "credential share t " not in out
    assert "'ghost-key' is not currently being served" in out
    assert "credential share ghost-key" in out
    assert "nothing can be added in this ceremony" in out
    decision = store.pair_decision("i_offer1", root)
    assert decision is not None and decision.matched and decision.decision == "admit"
    # The refusal re-asked; the SECOND edit attempt landed and wrote the reduction.
    assert decision.shares == ["anthropic"]

    # The daemon side refuses the same widening, with the same remedy, and writes
    # nothing — a frame gets no interactive re-ask.
    server = relay.RelayServer(settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"))
    try:
        with pytest.raises(types.MeshRefusal) as daemon_refusal:
            server._ctl_pair_confirm(  # noqa: SLF001 — the relay's own control op
                {
                    "invite_id": "i_offer1",
                    "decision": "admit",
                    "matched": True,
                    "shares": ["ghost-key"],
                }
            )
        assert daemon_refusal.value.code == "shares_not_offered"
        assert "ghost-key" in daemon_refusal.value.sentence
        untouched = store.pair_decision("i_offer1", root)
        assert untouched is not None and untouched.shares == [
            "anthropic"
        ], "the refused widening frame rewrote the decision"
    finally:
        server.stop()


def test_the_owner_receipt_names_what_it_served(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """U6 (design-confirmed fold, round 2): the owner's own receipt names the
    final set — the same fact the joiner reads — or says nothing was served; a
    ceremony with no offer sent keeps the bare line."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _pending_with_offer(root)
    monkeypatch.setattr(net_cli, "_has_terminal", lambda: True)
    joiner = "d_" + "b" * 32

    # Run A: keep openai, stop anthropic -> "— serving: openai".
    answers = iter(["t", "anthropic", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert net_cli._cmd_confirm(_confirm_args(json=False)) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    assert f"admitted {joiner} to home-net — serving: openai" in out

    # Run B: stop both -> "— serving nothing".
    answers = iter(["t", "openai anthropic", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert net_cli._cmd_confirm(_confirm_args(json=False)) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    assert f"admitted {joiner} to home-net — serving nothing" in out

    # Run C: no offer sent (an older peer) -> the line is unchanged.
    skipped = types.PendingPairing(
        invite_id="i_offer1",
        network_id="n_0123456789abcdef01234567",
        network_name="home-net",
        joiner_device_id=joiner,
        joiner_name="laptop",
        sas="481926",
        fingerprint="K7QM-3XPD-4WZ9-8NRB",
        transcribed="481926",
        peer_addr="127.0.0.1:4097",
        expires_at=time.time() + 120,
        prompt="no share list travelled here.",
        offer=[],
        offer_state="skipped_peer_unsupported",
    )
    store.save_pending_pairing(skipped, root)
    answers = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert net_cli._cmd_confirm(_confirm_args(json=False)) == 0  # noqa: SLF001
    out = capsys.readouterr().out
    assert f"admitted {joiner} to home-net" in out
    assert "— serving" not in out


def test_the_tool_finished_receipt_mounts_the_same_delta_lines() -> None:
    """N3 (agent review round 2): the tool's finished-receipt mount renders the
    SAME subjects, sentences and order the CLI does — one helper, pinned where
    it is used."""
    from local_operator.network import tool as net_tool

    payload = {
        "name": "lab",
        "network_id": "n_0123456789abcdef01234567",
        "epoch": 1,
        "role": "drive",
        "members": 2,
        "fingerprint": "K7QM-3XPD-4WZ9-8NRB",
        "shares": ["openai"],
        "reduced": ["azon"],
        "offers": [
            {"key": "azon", "kind": "oauth-rotating", "label": "", "share": True},
            {"key": "openai", "kind": "oauth-rotating", "label": "", "share": True},
            {"key": "legacy-key", "kind": "api-key-static", "label": "", "share": False},
        ],
    }
    lines = net_tool._render("join", payload)  # noqa: SLF001 — the mount itself
    assert lines[0].startswith("joined lab ")
    assert lines[1] == "available here: openai"
    assert lines[2] == "not available here: azon — the other device chose not to share it"
    assert lines[-1].startswith("fingerprint ")
    assert all("legacy-key" not in line for line in lines), "an unoffered key entered the receipt"


# ---------------------------------------------------------------------------
# Admission: grants, capability, audit, receipt (memo §7 cells 10 and 11)
# ---------------------------------------------------------------------------


def _pair_with_credential(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    *,
    seeds: list[tuple[str, dict[str, Any]]] | None = None,
    answer_shares: list[str] | None = None,
) -> tuple[Any, list[str], dict[str, Any]]:
    """A full ceremony where A holds oauth logins and answers as its person.

    ``seeds`` overrides the single-openai store; ``answer_shares`` drives the
    reduce-only choice at the confirm seam (omitted = the y/N default, the whole
    offered set — exactly what a person who types `y` sends).
    """
    server_a, server_b, host, port = devices
    for provider, payload in seeds or [
        ("openai", {"refresh": "r", "access": "a", "email": "damian@example.com"})
    ]:
        _seed(server_a.root, provider, payload)
    record = _init_network(server_a)
    state = store.load_secrets(record.network_id, server_a.root)
    minted = invite_mod.mint(record, state.secret, role="drive", ttl_s=600.0)
    record.invites.append(minted.record)
    store.save(record, server_a.root)
    store.save_invite_token(minted.record.invite_id, minted.token, server_a.root)
    _type_the_code(monkeypatch)
    answered: dict[str, Any] = {}
    thread = threading.Thread(
        target=lambda: answered.update(_answer_confirmation(server_a, shares=answer_shares) or {}),
        daemon=True,
    )
    thread.start()
    try:
        import tests.unit.network.test_relay_e2e as relay_e2e

        result = relay_e2e._join(
            server_b,
            host=host,
            port=port,
            token=minted.token,
            envelope=minted.envelope,
        )
    finally:
        thread.join(20)
    assert answered, "the inviter's person never answered"
    lines, payload = result
    return record, lines, payload


def test_admission_applies_the_granted_shares(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§5.3: one lock, admit THEN grant — the placement holder row, the
    ``broker_credential`` capability, and one ``credential.placement`` audit row,
    all from the offered set. The member row exists first; the grant rides it."""
    server_a, server_b, _host, _port = devices
    record, _lines, _payload = _pair_with_credential(devices, monkeypatch)
    # The member row (admitted first) and the grant that rides it.
    fresh = store.load(record.network_id, server_a.root)
    member = fresh.member(server_b.identity.device_id)
    assert member is not None and member.active
    assert "broker_credential" in member.capabilities
    # The placement holder: scope session — the smallest useful authority.
    from local_operator.network.credentials import placement as placement_mod

    document = placement_mod.PlacementDocument.load(
        record.network_id, server_a.root, self_device=server_a.identity.device_id
    )
    entry = document.entry("openai")
    assert entry is not None, "the admission wrote no placement for the offered key"
    holder = entry.holder(server_b.identity.device_id)
    assert holder is not None and holder.scope == "session"
    # The audit trail: one placement row for the grant, and the admission itself.
    import tests.unit.network.test_relay_e2e as relay_e2e

    events = relay_e2e._events(server_a)
    assert "credential.placement" in events
    assert "member_admitted" in events
    assert "pairing_confirmed" in events


def test_result_frame_carries_the_granted_shares_and_the_receipt_shows_them(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§3.5: the joiner's receipt states the final set — the line a person reads
    and the payload key an agent reads."""
    _record, lines, payload = _pair_with_credential(devices, monkeypatch)
    assert payload["shares"] == ["openai"]
    assert payload["reduced"] == []
    assert "available here: openai" in lines


def test_a_reduction_cannot_widen_at_admission(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """G4: even a decision written OUT OF BAND (shares ⊃ the offer) can only grant
    what the wire offered — the intersection is structural, not a convention."""
    server_a, server_b, _host, _port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "d@e"})
    record = _init_network(server_a)
    decision = types.PairDecision(
        invite_id="i_x",
        decision="admit",
        matched=True,
        shares=["openai", "ghost-key"],
    )
    offer_items = [{"key": "openai", "kind": "oauth-rotating", "label": "", "share": True}]
    granted, reduced = relay._grant_pair_shares(  # noqa: SLF001 — the admission helper itself
        record,
        joiner_id=server_b.identity.device_id,
        decision=decision,
        offer_items=offer_items,
        owner_name=server_a.identity.name,
        root=server_a.root,
        audit=server_a.audit,
    )
    assert granted == ["openai"]
    assert reduced == []
    from local_operator.network.credentials import placement as placement_mod

    document = placement_mod.PlacementDocument.load(record.network_id, server_a.root)
    assert document.entry("openai") is not None
    assert document.entry("ghost-key") is None, "an out-of-band share reached the document"


def test_a_deliberate_reduction_reaches_the_joiner_as_a_decision_not_a_failure(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UX round 1, U2, end to end: the owner removes `azon` at the confirm screen
    and the joiner's receipt says the other device CHOSE NOT TO — not the remedy,
    which would re-open a decision already taken. The result frame's ``reduced``
    is what makes the two causes distinguishable on this side at all."""
    _record, lines, payload = _pair_with_credential(
        devices,
        monkeypatch,
        seeds=[
            ("openai", {"refresh": "r", "access": "a", "email": "d@e"}),
            ("azon", {"refresh": "r", "access": "a", "email": "a@e"}),
        ],
        answer_shares=["openai"],
    )
    assert payload["shares"] == ["openai"]
    assert payload["reduced"] == ["azon"], "the deliberate half did not ride the result"
    # Both delta lines sit beside the serving line, right after `joined …` (D5),
    # and the deliberate one carries no remedy (U2).
    assert lines[0].startswith("joined ")
    assert lines[1] == "available here: openai"
    assert lines[2] == "not available here: azon — the other device chose not to share it"
    assert not any(
        "credential share" in line for line in lines
    ), "a deliberate reduction still points the joiner at the share verb"


def test_empty_and_skipped_offers_render_their_lines() -> None:
    """§4.2/§4.4's owner-side lines, and the rows when there are any — the same
    renderer `inviter_prompt_for` threads through all four owner surfaces."""
    from local_operator.network.invite import inviter_prompt_for

    empty = inviter_prompt_for(
        network_name="lab",
        role="drive",
        device_id="d_" + "b" * 32,
        name="laptop",
        transcribed="481926",
        derived="481926",
        offer_items=[],
        offer_state="sent",
    )
    assert "nothing will be served in this ceremony" in empty
    skipped = inviter_prompt_for(
        network_name="lab",
        role="drive",
        device_id="d_" + "b" * 32,
        name="laptop",
        transcribed="481926",
        derived="481926",
        offer_state="skipped_peer_unsupported",
    )
    assert "older build" in skipped
    rows = inviter_prompt_for(
        network_name="lab",
        role="drive",
        device_id="d_" + "b" * 32,
        name="laptop",
        transcribed="481926",
        derived="481926",
        offer_items=list(_OWNER_ITEMS),
        offer_state="sent",
    )
    assert "Credentials this device will serve to laptop:" in rows
    assert "openai (OAuth, d***@example.com)   will be served" in rows
    assert "legacy-key (API key)               not offered" in rows


# ---------------------------------------------------------------------------
# Mixed builds (memo §4: G1's other direction, and the mixed-daemon echo)
# ---------------------------------------------------------------------------


def test_a_new_owner_sends_nothing_to_an_old_joiner(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """G1(a): the gate is the JOINER'S OWN hello, so an old build's wire is
    untouched — its first sealed record is the RESULT, never an unknown op.

    Played by the real protocol code: a handshake whose advertised capabilities
    simply lack ``pair-offer-v1`` (the frame never existed for that build). The
    owner must skip the send AND say why on its own side §4.2's line).
    """
    import socket as socket_mod

    from local_operator.network.identity import mint_instance_id

    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "d@e"})
    _record, minted = _minted(server_a)
    answered: dict[str, Any] = {}
    failures: list[BaseException] = []

    def _answer_and_record() -> None:
        try:
            answered.update(_answer_confirmation(server_a) or {})
        except BaseException as exc:  # noqa: BLE001 — reported below
            failures.append(exc)

    thread = threading.Thread(target=_answer_and_record, daemon=True)
    thread.start()
    sock = socket_mod.create_connection((host, port), timeout=30)
    try:
        handshake = Handshake.new(
            role="dialer",
            identity=server_b.identity,
            network_id=minted.envelope.network_id,
            epoch=minted.envelope.epoch,
            instance_id=mint_instance_id(),
            session_protocol=net_cli._session_protocol(),  # noqa: SLF001 — the dialer's own
            mode="join",
            # AN OLD BUILD'S HELLO, spelled as absence: one string less than this
            # build advertises, which is exactly what the gate reads.
            capabilities=[n for n in wire.LINK_CAPABILITIES if n != wire.PAIR_OFFER_V1],
            build={},
            endpoints=[],
        )
        handshake.join_block = {
            "invite_id": minted.envelope.invite_id,
            "joiner_public_key": server_b.identity.public_key,
            "joiner_name": server_b.identity.name,
        }
        handshake.send_hello(sock)
        reader = wire.FrameReader(sock)
        handshake.read_challenge(reader, wire.deadline_in(30))
        # The dialer's own steps, exactly as `_join_one` runs them: the invite-derived
        # credential, the MAC-ed auth frame, then the plaintext welcome.
        credential = Credential(
            "invite",
            minted.envelope.epoch,
            wire.invite_key(
                minted.envelope.material, minted.envelope.network_id, minted.envelope.invite_id
            ),
        )
        handshake.send_auth(sock, credential)
        handshake.read_welcome(reader, wire.deadline_in(30))
        result = handshake.establish()
        codec = handshake.codec()
        sock.sendall(codec.seal(pair_ready_frame(req=1, typed_sas=result.sas)))
        frame = codec.open(reader.read_record_payload(wire.deadline_in(30)))
        assert (
            frame.get("op") == "net_pair_result"
        ), "an old joiner's first sealed record was not the result — the offer gate leaked"
        assert frame.get("admit") is True
        assert frame.get("shares") == [], "a mixed pair has nothing to report"
        assert frame.get("reduced") == []
    finally:
        sock.close()
        thread.join(20)
    assert not failures, failures[0]
    assert answered, "the inviter's person was never asked"
    # §4.2 on the owner's own record: the silence is SKIPPED, not EMPTY — the
    # distinction the confirm screen and the audit both read.
    assert answered.get("offer_state") == "skipped_peer_unsupported"
    assert answered.get("offer") == []
    assert "older build" in str(answered.get("prompt") or "")


def test_a_stale_relay_response_is_named_not_hidden(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§4.5's skew: a relay that does not echo ``shares`` is older than this CLI.

    The old daemon's reply is simulated at the CLI's call seam (its own shape,
    minus the echo: that is the only difference the check can see). The warning
    goes to stderr and the command still succeeds — the admission lands; what the
    operator learns is why the share choices did not.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _pending_with_offer(root)
    monkeypatch.setattr(net_cli, "_has_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    monkeypatch.setattr(
        net_cli,
        "_relay_call",
        lambda op, **fields: {
            "invite_id": "i_offer1",
            "decision": "admit",
            "matched": True,
            "joiner_device_id": "d_" + "b" * 32,
            # NO "shares" key: an older relay, whose state machine predates them.
        },
    )
    assert net_cli._cmd_confirm(_confirm_args()) == 0  # noqa: SLF001
    err = capsys.readouterr().err
    assert "older build" in err
    assert "restart" in err
