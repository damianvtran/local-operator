"""Slice B's two-root cells — E1-E4 (mesh credential binding).

The memo's Q6 table is the source; this file is the receipt. E1-E4 ride the
suite's REAL two-root rigs — the ``test_credentials_real_link`` pair (A owns the
stub provider, B borrows over real relays and a real rotating IdP whose POST
count is a fact) for E1/E3/E4, and the ``test_mobility`` pair (both relays live,
``lop sessions move`` driven through the relay's control socket the way the CLI
drives it) for the move round-trip. No test here builds a frame by hand or
calls a handler directly.

Isolation is the fixtures' own: each root is the rig's temp config dir, every
store is opened with an EXPLICIT ``db_path`` (``AuthStore`` otherwise derives its
database from the ambient ``LOCAL_OPERATOR_CONFIG_DIR``), and the operator's
live stores are never read or written — the cells run under the same ``env -i``
roots as the rest of ``tests/unit/network``.

RED on the slice-B base: every cell drives a seam Slice B adds
(``set_binding_reader``, ``set_change_handler``, ``recall_for`` as a reader) or
asserts a row/notice only Slice B produces; on the A-only tree the file fails at
its first seam, by construction.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message
from local_operator.network import relay
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.store import MeshAwareAuthStore
from local_operator.providers.auth_store import AuthStore
from local_operator.session.credential_binding import (
    CredentialBinding,
    CredentialBindingRecorder,
    recall_for,
    record,
)
from local_operator.session.placement import MeshStamp, SessionPlacement, write_stamp
from local_operator.session.transcript import Transcript
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    RotatingIdP,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _audit,
    _borrower_client,
    _lop_network,
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures by import
    _move,
    pair,
)
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixtures by import

#: The stub IdP pins this; the identity the owner serves with.
OWNER_EMAIL = "owner@example.test"
#: A session id shaped like the ones the rigs use, unique to E2.
MOVE_SESSION = "9f3ac1e0b7e2"


def _dump(db: Path) -> str:
    """The LOGICAL dump of a store — rows and schema, not file bytes.

    ``§5.2``'s method: an ``auth.db`` asserted unchanged is asserted about what
    a reader sees, so a WAL checkpoint moving bytes around is not a failure.
    """
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return "\n".join(conn.iterdump())


def _lease_count(db: Path) -> int:
    """Refresh leases outstanding on the owner (``§9.2``: 0 after a clean lend)."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        try:
            row = conn.execute("select count(*) from auth_credential_refresh_leases").fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row[0])


@pytest.fixture()
def cred_mesh(request: pytest.FixtureRequest) -> Any:
    """The real-link pair, reached by NAME.

    Imported fixtures are registered by import but pulled with
    ``getfixturevalue`` — the convention the import block's note points at, and
    what keeps flake8 from reading the fixture name as a shadowed redefinition.
    """
    return request.getfixturevalue("mesh")


@pytest.fixture()
def mobility_pair(request: pytest.FixtureRequest) -> Any:
    """The mobility pair fixture, reached by name (same note as ``cred_mesh``)."""
    return request.getfixturevalue("pair")


def _binding_lines(path: Path) -> list[str]:
    """Every ``mesh_credential_binding.v1`` line of a transcript, verbatim."""
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "mesh_credential_binding.v1" in line
    ]


def _borrower_store(root: Path) -> MeshAwareAuthStore:
    """B's real store: explicit ``db_path``, the real per-device client."""
    client = MeshCredentialClient.for_this_device(root)
    assert client is not None, "this root cannot borrow; the share never reached it"
    auth = AuthStore(db_path=root / "auth.db", config_dir=root)
    return MeshAwareAuthStore(auth, mesh=client, config_dir=root)


async def _seed_owned_session(
    server: relay.RelayServer, session_id: str, binding: CredentialBinding
) -> Path:
    """A session this device owns, whose transcript already carries a binding row.

    The transcript is written by the REAL ``Transcript``/``record`` writers (the
    row must be a genuine journal line for the byte comparison to mean
    anything), then the store marker, title and stamp are written exactly as
    ``session_factory`` and the mobility rig's ``_owned_session`` write them —
    without the store marker a move's commit cannot delete the source and the
    cell would test the wrong failure.
    """
    from local_operator.session.cleanup import mark_store

    directory = server.root / "sessions" / session_id
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("bound elsewhere"))
    await record(transcript, binding)
    directory.joinpath("title.json").write_text(json.dumps({"title": "slice B"}), "utf-8")
    mark_store(server.root / "sessions")
    write_stamp(
        server.root,
        MeshStamp(
            session_id=session_id,
            network_id="n_test",
            home_device=server.identity.device_id,
            placement=SessionPlacement(
                mode="local", home_device=server.identity.device_id, stamp_revision=1
            ),
        ),
    )
    return directory


# ---------------------------------------------------------------------------
# E1 — a real turn on B binds the row (the §9.2 turn, in-process)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e1_a_borrow_on_b_binds_the_row_over_the_real_link(cred_mesh: Any) -> None:
    """E1: B borrows through its real store; ONE row citing dev_A; B untouched.

    Asserts exactly what §9.2's table reads: ``grep -c mesh_credential_binding.v1``
    on B's transcript is 1 and cites the owner; B's ``auth.db`` dump is unchanged
    (a borrow is not a local login); A refreshed exactly ONCE, its refresh lease
    is released, and one ``credential.grant`` names this session.
    """
    _share(cred_mesh)
    _pull(cred_mesh)
    root_b = cred_mesh.b.root
    transcript = Transcript(root_b / "sessions" / "sess-e1")
    recorder = CredentialBindingRecorder(
        transcript, session_id="sess-e1", device_id=cred_mesh.borrower, device_name="device-b"
    )
    store_b = _borrower_store(root_b)
    store_b.set_serve_sink(recorder.observe_serve)
    store_b.set_binding_reader(recorder.recall_for)
    before = _dump(root_b / "auth.db")
    try:
        token = await store_b.get_api_key(STUB_PROVIDER, "sess-e1")
        await recorder.drain()
        assert token, "the borrow did not serve"
        rows = _binding_lines(transcript.path)
        assert len(rows) == 1, rows
        details = json.loads(rows[0])["payload"]["details"]
        assert details["owner_device"] == cred_mesh.owner
        assert details["identity_label"] == OWNER_EMAIL
        assert details["policy"] == "local-first"

        # B's own store never took a row: the borrow is not a local login.
        assert _dump(root_b / "auth.db") == before
        # A: exactly one refresh POST, the lease released, one grant for this session.
        assert len(cred_mesh.idp.posts) == 1
        assert _lease_count(cred_mesh.a.root / "auth.db") == 0
        grants = [
            row
            for row in _audit(cred_mesh.a, "credential.grant")
            if row.get("session_id") == "sess-e1"
        ]
        assert len(grants) == 1, grants
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# E2 — a move carries the row byte-identical; the destination re-resolves to
#      the same owner without the move being a credential event (§5.5 a-c)
# ---------------------------------------------------------------------------


class _MeshView:
    """The two attributes ``_pull`` reads off the real-link ``mesh`` fixture."""

    def __init__(self, server_a: relay.RelayServer, server_b: relay.RelayServer) -> None:
        self.a = server_a
        self.b = server_b


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [False, True], ids=["move", "keep"])
async def test_e2_a_move_carries_the_row_and_re_resolves_to_the_same_owner(
    keep: bool,
    mobility_pair: Any,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """E2: ``--keep`` off and on; (a) row byte-identical at the destination,
    (b) the next resolve picks the same owner, (c) the MOVE never touched A's
    credentials (dump and POST counter unmoved — a move is not a credential
    event; the resolve that follows is a borrow, which is)."""
    from tests.unit.network.test_relay_e2e import _pair_settled

    # Registered by import, pulled by name (see ``cred_mesh``'s note); the IdP's
    # POST count is the "is the owner spending anything" fact.
    request.getfixturevalue("stub_provider")
    stub_idp: RotatingIdP = request.getfixturevalue("idp")
    server_a, server_b = mobility_pair[0], mobility_pair[1]
    _pair_settled(mobility_pair, monkeypatch, role="admin")
    # The share is typed by a human on A; the credential the broker serves.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    auth_a = AuthStore(config_dir=server_a.root)
    try:
        auth_a.upsert_credential(
            STUB_PROVIDER,
            {
                "type": "oauth",
                "access": "-".join(("stale", "access")),
                "expires": int(time.time() * 1000) - 60_000,
                "refresh": stub_idp.current_refresh,
                "email": OWNER_EMAIL,
            },
        )
        row_id = auth_a.list_credentials(STUB_PROVIDER)[0].id
    finally:
        auth_a.close()
    assert _lop_network("credential", "share", STUB_PROVIDER, "--with", server_b.identity.name) == 0
    _pull(mesh=_MeshView(server_a, server_b))

    # The session was served on A before the move: the row is what travels.
    binding = CredentialBinding(
        provider=STUB_PROVIDER,
        owner_device=server_a.identity.device_id,
        owner_device_name=server_a.identity.name,
        credential_id=row_id,
        identity_label=OWNER_EMAIL,
        writer="1000:1000",
    )
    await _seed_owned_session(server_a, MOVE_SESSION, binding)
    source_lines = _binding_lines(server_a.root / "sessions" / MOVE_SESSION / "transcript.jsonl")
    assert len(source_lines) == 1

    dump_before = _dump(server_a.root / "auth.db")
    posts_before = len(stub_idp.posts)

    result = _move(server_b, MOVE_SESSION, keep=keep, monkeypatch=monkeypatch)
    assert result["ok"] is True, result
    dest_id = result["new_session_id"]
    dest_transcript = server_b.root / "sessions" / dest_id / "transcript.jsonl"

    # (a) the row travels byte-identical — the ROW LINE, not the whole file
    # (the destination legitimately rewrites nothing of this line either, but
    # the comparison the design asks for is the row's).
    assert _binding_lines(dest_transcript) == source_lines

    # (c) the move itself was not a credential event.
    assert _dump(server_a.root / "auth.db") == dump_before
    assert len(stub_idp.posts) == posts_before, "a move spent the owner's refresh token"

    # (b) the destination's next resolve picks the same owner_device.
    recorder = CredentialBindingRecorder(
        Transcript(server_b.root / "sessions" / dest_id),
        session_id=dest_id,
        device_id=server_b.identity.device_id,
        device_name="device-b",
    )
    # The owner's broker resolves ITS OWN store from the ambient root (the
    # ``test_credentials_real_link`` module docstring's constraint: ``AuthStore``
    # derives its database from ``LOCAL_OPERATOR_CONFIG_DIR``, not from its
    # argument). The move's CLI pointed the ambient root at B, so point it back
    # at A before the borrowed serve — otherwise the broker reads B's empty
    # auth.db and refuses ``no_local_credential``.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    store_b = _borrower_store(server_b.root)
    store_b.set_serve_sink(recorder.observe_serve)
    store_b.set_binding_reader(recorder.recall_for)
    try:
        grant_token = await store_b.get_api_key(STUB_PROVIDER, dest_id)
        await recorder.drain()
        assert grant_token, "the moved session could not resolve after the move"
        bound = recall_for(Transcript(server_b.root / "sessions" / dest_id), STUB_PROVIDER)
        assert bound is not None and bound.owner_device == server_a.identity.device_id
        # Still exactly the carried row: same owner, same row id, same label —
        # nothing to append (replacement state), so the line is untouched.
        assert _binding_lines(dest_transcript) == source_lines
        served = [
            row for row in _audit(server_a, "credential.grant") if row.get("session_id") == dest_id
        ]
        assert len(served) == 1, served
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# E3 — an account switch mid-session: new row + the notice sentence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e3_an_account_switch_mid_session_writes_the_row_and_one_notice(
    cred_mesh: Any,
) -> None:
    """E3: B gains its own login mid-session → row ``owner=self`` + ONE change.

    The capture is allowed (D1: record + notify, never block); the durable row
    and the ONE notice signal are the whole difference between this build and
    the silent switch the audit found. The notice row itself (operator-only,
    never model context) is pinned by ``test_credential_binding.py``'s U5b;
    here the signal carries the production sentence.
    """
    from local_operator.network.credentials.messages import render_binding_change_notice

    _share(cred_mesh)
    _pull(cred_mesh)
    root_b = cred_mesh.b.root
    transcript = Transcript(root_b / "sessions" / "sess-e3")
    changes: list[tuple[CredentialBinding, CredentialBinding | None]] = []
    recorder = CredentialBindingRecorder(
        transcript,
        session_id="sess-e3",
        device_id=cred_mesh.borrower,
        device_name="device-b",
        on_change=lambda new, previous: changes.append((new, previous)),
    )
    auth_b = AuthStore(db_path=root_b / "auth.db", config_dir=root_b)
    client = _borrower_client(cred_mesh)
    store_b = MeshAwareAuthStore(auth_b, mesh=client, config_dir=root_b)
    store_b.set_serve_sink(recorder.observe_serve)
    store_b.set_binding_reader(recorder.recall_for)
    try:
        first = await store_b.get_api_key(STUB_PROVIDER, "sess-e3")
        await recorder.drain()
        assert first, "the borrow did not serve"
        posts_after_borrow = len(cred_mesh.idp.posts)

        # The device gains its own login mid-session: the next serve is local.
        auth_b.upsert_credential(STUB_PROVIDER, {"type": "api_key", "key": "sk-own-login"})
        second = await store_b.get_api_key(STUB_PROVIDER, "sess-e3")
        await recorder.drain()
        assert second == "sk-own-login"
        assert len(cred_mesh.idp.posts) == posts_after_borrow, "the switch dialled the owner"

        rows = _binding_lines(transcript.path)
        assert len(rows) == 2, rows
        newest = json.loads(rows[1])["payload"]["details"]
        assert newest["owner_device"] == cred_mesh.borrower

        assert len(changes) == 2, "one first-serve signal, one CHANGE signal"
        new, previous = changes[-1]
        assert new is not None and previous is not None
        notice = render_binding_change_notice(new, previous, self_device=cred_mesh.borrower)
        assert "your login on this device" in notice
        assert previous.identity_label == OWNER_EMAIL and OWNER_EMAIL in notice
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# E4 — a report after a sibling pick targets the row the binding names
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e4_a_report_after_a_sibling_pick_corroborates_the_binding_row(
    cred_mesh: Any,
) -> None:
    """E4: the report path and the binding row name the SAME lent row.

    Extends ``test_the_stores_report_carries_the_lent_rows_id_over_the_link``'s
    shape: the owner has two rows, row 1 is model-blocked so the lend lands on
    row 2 (the sibling pick), and the binding row records row 2. The report
    raised through the store must then spend row 2's refresh token — the two
    readers of "which row served" corroborate each other.
    """
    from local_operator.providers.failover import ProviderError

    _share(cred_mesh)
    _pull(cred_mesh)

    store_a = AuthStore(config_dir=cred_mesh.a.root)
    try:
        first = store_a.list_credentials(STUB_PROVIDER)[0]
        token0 = cred_mesh.idp.current_refresh
        second = store_a.upsert_credential(
            STUB_PROVIDER,
            {
                "type": "oauth",
                "access": "-".join(("row2", "stale")),
                "expires": int(time.time() * 1000) - 60_000,
                "refresh": token0,
                "email": "row2@example.test",
            },
        )
        assert second.id != first.id
        store_a.block_credential(
            first.id, STUB_PROVIDER, block_scope="model:fable", block_ms=600_000
        )

        root_b = cred_mesh.b.root
        transcript = Transcript(root_b / "sessions" / "sess-e4")
        recorder = CredentialBindingRecorder(
            transcript,
            session_id="sess-e4",
            device_id=cred_mesh.borrower,
            device_name="device-b",
        )
        store_b = _borrower_store(root_b)
        store_b.set_serve_sink(recorder.observe_serve)
        store_b.set_binding_reader(recorder.recall_for)
        try:
            token = await store_b.get_api_key(STUB_PROVIDER, "sess-e4", model_id="claude-fable-5")
            await recorder.drain()
            assert token, "the lend did not serve"
            bound = recall_for(transcript, STUB_PROVIDER)
            assert bound is not None and bound.credential_id == second.id, "the sibling pick"
            assert cred_mesh.idp.posts == [token0]

            row2 = next(r for r in store_a.list_credentials(STUB_PROVIDER) if r.id == second.id)
            token_after_lend = str(row2.data.get("refresh") or "")
            assert token_after_lend and token_after_lend != token0
            # Expire row 2 again through the product's own write path, so a
            # refresh aimed at it POSTs — and the token it posts is by
            # construction the one the report resolved.
            store_a.upsert_credential(
                STUB_PROVIDER,
                {
                    "type": "oauth",
                    "access": "-".join(("row2", "expired-again")),
                    "expires": int(time.time() * 1000) - 60_000,
                    "refresh": token_after_lend,
                    "email": "row2@example.test",
                },
            )

            rotated = store_b.rotate_sibling(
                STUB_PROVIDER,
                "sess-e4",
                ProviderError(401, "invalid_grant"),
                str(token),
            )
            assert rotated is False, rotated  # no local sibling: the report IS the answer
            assert cred_mesh.idp.posts == [token0, token_after_lend], (
                "the owner refreshed a row other than the one the binding names: "
                f"{cred_mesh.idp.posts!r} vs binding row {second.id}"
            )
            # Both readers still agree after the round trip.
            after = recall_for(transcript, STUB_PROVIDER)
            assert after is not None and after.credential_id == second.id
        finally:
            store_b.close()
    finally:
        store_a.close()
