"""S4 cells: class-2 copies — selection defaults, re-sealing, provenance.

WHAT THIS FILE PINS, and why each cell exists:

- the DEFAULT-SET INTERSECTION (§4.2): a key is preselected exactly when the
  pushed bundles declare it (``ref:<NAME>``) or the operator marked it ``sync``;
  an unmarked key is OFFERED with ``share`` false — never copied by default —
  and a ``local-only`` key is not a candidate at all;
- ``local-only`` never crosses, even where a grant raced the mark;
- the copy on the member is RE-SEALED under the member's own key — the owner's
  master key cannot open it and the plaintext appears nowhere under the
  member's root — and it carries the provenance marker the wipe will scan;
- a payload that does not name the key it claims is dropped whole.

The delivery cells run two REAL relays on loopback roots (the ``devices``
fixture in ``test_relay_e2e``), so the copy is exercised over real relays and
the real store/sealing path, never a direct function call. **The unit run takes
the documented KEYFILE FALLBACK, not a broker-served key**: no lop session is
registered with a broker, so a broker would deny the caller and
``master_key_for`` reads the key file. Broker custody is covered by the broker
suite; the relay, re-sealing and provenance coverage here is not broker-specific.
The member side uses the same ``auth_store_factory`` seam S3's cells use (one
process, two roots).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import store
from local_operator.network.credentials import sync
from local_operator.network.credentials.sync import SyncState
from tests.conftest import _stop_brokers_in
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_real_link import (  # noqa: F401
    _lop_network,
    _pull,
)
from tests.unit.network.test_credentials_sync import _open_member_store
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

_LOG = logging.getLogger(__name__)

# The unit run takes the keyfile fallback: no session is registered, so a
# broker would deny every read. The autouse gate stays on -- review round 2
# measured zero successful broker RPCs even with the daemons spawned, so there
# was no broker-served coverage to restore.

#: The secret both relay cells copy: a name that is not a provider's, so nothing
#: else in the tree claims it.
SECRET_NAME = "STORE_COPY_TOKEN"
KEY = f"secret:{SECRET_NAME}"
VALUE_1 = b"class2-copy-value-one"
VALUE_2 = b"class2-copy-value-two-rotated"


def _secret_store(root: Path, *, create: bool = False) -> Any:
    from local_operator.secrets import access

    return access.open_store(root, create=create)


def _member_value(root: Path, name: str) -> bytes | None:
    """The member's stored value, or ``None`` when it holds nothing under ``name``."""
    from local_operator.secrets.errors import SecretNotFound, SecretStoreError

    try:
        handle = _secret_store(root)
    except Exception:  # noqa: BLE001 — no store yet: nothing held
        return None
    try:
        _record, value = handle.read_for_copy(name)
        return value
    except (SecretNotFound, SecretStoreError):
        return None


def _member_has(root: Path, name: str) -> bool:
    return _member_value(root, name) is not None


def _probe(fn: Any) -> str:
    """One diagnostic read, rendered; a failing probe names its exception type only.

    WHY: the snapshot runs while a wait has ALREADY failed, on a mesh that may be
    half torn down. A probe that raised would replace the original failure with
    its own and defeat the point, so every read reports ``<unavailable: Type>``
    instead (the type, never ``str(exc)``: an exception message can carry a path
    or a value).
    """
    try:
        return str(fn())
    except Exception as exc:  # noqa: BLE001 — diagnostics never mask the real failure
        return f"<unavailable: {type(exc).__name__}>"


def _link_summary(server: Any) -> list[str]:
    """Per live link: peer short id, phase, frames received, seconds since last frame."""
    now = time.time()
    with server._links_lock:
        links = list(server.links.items())
    return [
        f"{str(device)[-8:]}:{getattr(link, 'phase', '?')}"
        f"/frames_in={getattr(link, 'frames_in', '?')}"
        f"/idle={now - float(getattr(link, 'last_frame_at', now)):.1f}s"
        for device, link in links
    ]


def _engine_summary(engine: Any) -> str:
    """The sync engine's scheduling state: in-flight guards and the loop thread's life.

    ``inflight_members`` still holding the member after a long wait means the
    exchange coroutine never finished (stuck dial/serve); an empty set with no
    ack means it finished without delivering. ``loop_thread_alive`` False means
    nothing could have run the scheduled work at all.
    """
    loop = engine._loop
    thread = getattr(loop, "_thread", None)
    return (
        f"inflight_members={len(engine._inflight_members)} pulling={len(engine._pulling)} "
        f"closed={engine._closed} loop_started={loop is not None} "
        f"loop_thread_alive={thread.is_alive() if thread is not None else None}"
    )


def _ledger_summary(mesh: Any) -> str:
    """Owner ledger (generation + the member's ack) and the member's applied row.

    Reads the sync documents from disk exactly as the product does. Only ints
    and field NAMES are rendered: the ack/applied rows carry digests and record
    ids, which stay out of the log.
    """
    owner_state = SyncState.load(mesh.network_id, mesh.a.root)
    member_state = SyncState.load(mesh.network_id, mesh.b.root)
    generation = owner_state.generation(KEY) or {}
    ack = owner_state.ack_for(mesh.member, KEY)
    applied = member_state.applied_for(KEY)
    ack_text = "none" if ack is None else f"gen={ack.get('gen')} fields={sorted(ack)}"
    applied_text = "none" if applied is None else f"gen={applied.get('gen')}"
    return (
        f"owner_gen={generation.get('gen')} owner_ack_for_member={ack_text} "
        f"member_applied={applied_text} owner_mark={owner_state.mark_for(KEY) or 'none'!r}"
    )


def _store_names(root: Path) -> list[str]:
    """Secret NAMES held under ``root`` — the listing returns metadata, never values."""
    return sorted(record.name for record in _secret_store(root).list())


def _sync_snapshot(mesh: Any, step_code: Any = None, started: float | None = None) -> str:
    """A compact, REDACTED account of where a copy stalled, for an expired wait.

    Built lazily (only as an ``assert`` message, i.e. on failure) and also logged
    at ERROR so it lands in CI's "Captured log call" next to conftest's
    "net wait expired" line. One expiry of the member-has wait in
    ``test_a_shared_secret_copies_re_sealed_and_marked`` (CI run 43570943897)
    left no evidence to tell a loaded runner from a lost wake from a delivery
    defect; each section below answers one of those:

    - ``step``: what ``credentials_sync_step`` returned (scheduled/in_sync/...);
    - ``owner_engine`` / ``member_engine``: was the exchange still in flight, and
      was the loop thread that runs it alive;
    - ``owner_links`` / ``member_links``: is the relay link up, and is it still
      carrying frames (idle seconds);
    - ``ledger``: did the owner record a generation and an ack, did the member
      record an applied row (a value without an applied row = the ack/ledger
      write lagged, the sibling wait's race);
    - ``member_store_names``: what the member's store holds, names only;
    - ``elapsed``: seconds since ``started`` when the caller passes it.

    NEVER add values, keys, tokens or digests here: names, counts, states and
    generations only. To extend, add one more ``_probe(lambda: ...)`` entry.
    """
    parts = [
        f"step={step_code!r}",
        f"elapsed={'n/a' if started is None else '%.1fs' % (time.monotonic() - started)}",
        f"owner_engine=[{_probe(lambda: _engine_summary(mesh.engine_a))}]",
        f"member_engine=[{_probe(lambda: _engine_summary(mesh.engine_b))}]",
        f"owner_links={_probe(lambda: _link_summary(mesh.a))}",
        f"member_links={_probe(lambda: _link_summary(mesh.b))}",
        f"ledger=[{_probe(lambda: _ledger_summary(mesh))}]",
        f"member_store_names={_probe(lambda: _store_names(mesh.b.root))}",
    ]
    text = "sync snapshot: " + " ".join(parts)
    _LOG.error("%s", text)
    return text


@pytest.fixture()
def secret_mesh(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Owner A (holds one store secret) and member B (drive), both relays up.

    A's secret is SHARED with B before the cells run — the same shape
    ``sync_mesh`` uses for the api-key class — so the copy itself arrives over
    the real relay on the cells' own exchange. The store side takes the keyfile
    fallback (see the module docstring); no broker session is registered here.
    """
    pair_devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _host, _port = _pair(pair_devices, monkeypatch, role="drive")
    # The ambient root is the OWNER's from here on (see the sibling modules).
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    # B must be able to DIAL A: the pairing leaves A's row on B with no endpoint.
    with store.mutate(record.network_id, server_b.root) as copy:
        owner_row = copy.member(server_a.identity.device_id)
        assert owner_row is not None
        owner_row.endpoints = [f"{host}:{port}"]
        store.save(copy, server_b.root)
    server_b.start()
    # A must be able to DIAL B for the announce push.
    with store.mutate(record.network_id, server_a.root) as copy:
        member_row = copy.member(server_b.identity.device_id)
        assert member_row is not None
        member_row.endpoints = [f"127.0.0.1:{server_b.settings.port}"]
        store.save(copy, server_a.root)

    store_a = _secret_store(server_a.root, create=True)
    record_a = store_a.set(SECRET_NAME, VALUE_1, description="copied by S4's cells")
    assert record_a.origin is None, "a local secret starts with no mesh marker"

    assert _lop_network("credential", "share", KEY, "--with", server_b.identity.name) == 0
    pulled = _pull(SimpleNamespace(b=server_b, borrower=server_b.identity.device_id))
    # ``credential share`` now PUSHES the row (``share_push``), so the pull may find it
    # already merged; what matters is that the member holds it after the pull.
    from local_operator.network.credentials import placement as placement_mod

    assert KEY in pulled.get("changed", []) or placement_mod.PlacementDocument.load(
        record.network_id, server_b.root, self_device=server_b.identity.device_id
    ).is_holder(KEY, server_b.identity.device_id), pulled

    engine_b = sync.SyncEngine(
        server_b,
        root=server_b.root,
        self_device=server_b.identity.device_id,
        self_device_name=server_b.identity.name,
        audit=server_b.audit,
        auth_store_factory=lambda: _open_member_store(server_b.root),
    )
    setattr(server_b, sync._SYNC_ATTR, engine_b)
    engine_a = sync.sync_for_relay(server_a)
    assert engine_a is not None
    mesh = SimpleNamespace(
        a=server_a,
        b=server_b,
        engine_a=engine_a,
        engine_b=engine_b,
        network_id=record.network_id,
        store_a=store_a,
        owner=server_a.identity.device_id,
        member=server_b.identity.device_id,
    )
    try:
        yield mesh
    finally:
        server_b.stop()
        # THE PER-SIDE STOPS ARE A BACKSTOP, because nothing else can reach these
        # roots. The copy path takes the keyfile fallback (no session is registered,
        # so a broker would deny and `master_key_for` reads the key file) and the
        # autouse gate starts no daemon; but if a future cell opts into a real broker
        # under these roots, the suite-wide sweep in
        # ``tests/conftest.py`` cannot see those roots — its candidates come from
        # ``tmp_path``, and this module only ever reaches ``tmp_path`` through fixtures
        # (``secret_mesh`` and the network suite's ``root``), so ``item.funcargs``
        # never holds it and the sweep's list comes back as ``home/.local-operator``
        # alone. Measured on this fleet: a ``-n0`` run of this file left NINE live
        # key-holding daemons even though the sweep ran (its own probe logged
        # ``candidates=1, with_socket=0``, and each two-device cell added two). The
        # socket lives INSIDE each root (pytest paths here are short enough that no
        # TMPDIR fallback is in play), and pytest reclaims the directories at
        # ``tmp_path``'s own teardown, after this finaliser — so this is the last
        # moment `_stop_brokers_in` (the sweep's own stop, candidate by candidate,
        # refusing anything it cannot confirm dead) can reach them. Cell 5
        # (``needs_list``) leaked nothing and needs none of this; the bare-secret cell
        # shares this fixture and is covered with the rest.
        _stop_brokers_in([server_a.root, server_b.root])


def test_a_shared_secret_copies_re_sealed_and_marked(secret_mesh: Any) -> None:
    """The headline: copy -> the member holds it, under ITS key, with provenance.

    Three assertions the design fixes: the member's value round-trips; the copy
    is SEALED under the member's own key (the owner's key cannot open the
    member's store, and the plaintext appears nowhere under the member's root);
    and the marker names the owner and the key — the two fields the wipe scans.
    """
    mesh = secret_mesh
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: _member_has(mesh.b.root, SECRET_NAME)
    ), _sync_snapshot(mesh, code)

    member_store = _secret_store(mesh.b.root)
    record, value = member_store.read_for_copy(SECRET_NAME)
    assert value == VALUE_1
    origin = record.origin or {}
    assert origin.get("owner_device") == mesh.owner
    assert origin.get("key") == KEY
    assert int(origin.get("gen") or 0) >= 1

    # CANARY: no plaintext of the copied value anywhere under the member's root.
    for path in Path(mesh.b.root).rglob("*"):
        if path.is_file():
            assert VALUE_1 not in path.read_bytes(), f"plaintext found in {path}"

    # RE-SEALED UNDER THE MEMBER'S OWN KEY: the owner's key cannot open it.
    from local_operator.secrets.keys import load_master_key
    from local_operator.secrets.store import SecretStore

    owner_key = load_master_key(mesh.a.root)
    member_key = load_master_key(mesh.b.root)
    assert bytes(owner_key) != bytes(member_key)
    alien = SecretStore(owner_key, base=mesh.b.root)
    opened = False
    try:
        alien.describe(SECRET_NAME)
        opened = True
    except Exception:  # noqa: BLE001 — the expected end: the foreign key opens nothing
        opened = False
    assert not opened, "the owner's key must not open the member's store"


def test_a_rotated_value_reaches_the_member_as_a_new_generation(secret_mesh: Any) -> None:
    """Lifecycle 2 through the copy path: update on the owner -> the member converges."""
    mesh = secret_mesh
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: _member_value(mesh.b.root, SECRET_NAME) == VALUE_1
    ), _sync_snapshot(mesh, code)

    mesh.store_a.update(SECRET_NAME, VALUE_2)
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: _member_value(mesh.b.root, SECRET_NAME) == VALUE_2
    ), _sync_snapshot(mesh, code)
    # WHY wait on the ledger, not just the value: the copy path writes the
    # member's store first and the applied-generation ledger second
    # (``_pull_blocking``: ``_apply_value`` then ``record_applied``), so the
    # value becoming readable does NOT prove the generation has been recorded.
    # Reading the ledger immediately after the value wait races that second
    # write on a loaded shard (CI 38863032433: value converged, applied gen
    # still 1). CONTRIBUTING's flake rule is the same point: wait on the
    # acknowledgement/ledger before asserting it.
    assert net_fixtures.wait_for(
        lambda: int(
            (SyncState.load(mesh.network_id, mesh.b.root).applied_for(KEY) or {}).get("gen") or 0
        )
        >= 2
    ), _sync_snapshot(mesh, code)
    applied = SyncState.load(mesh.network_id, mesh.b.root).applied_for(KEY) or {}
    assert int(applied.get("gen") or 0) >= 2
    assert str(applied.get("record_id") or ""), "the class-2 record id rides the sidecar"


def test_local_only_never_crosses_even_with_a_grant(secret_mesh: Any) -> None:
    """§4.2's kill switch, both halves: the serve path refuses and nothing lands.

    The grant exists (the fixture wrote it), so this is exactly the race the
    design says the mark must win: set AFTER the grant, the key neither
    announces nor copies, ``owner_copy`` refuses it by name, and it is gone
    from the candidate list.
    """
    mesh = secret_mesh
    with sync.mutate(mesh.network_id, mesh.a.root) as state:
        state.record_mark(KEY, "local-only")

    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert not _member_has(mesh.b.root, SECRET_NAME), "a local-only key must never copy"

    network_id, frames = mesh.engine_a._pending_announces(mesh.member)
    positives = [frame for frame in frames if frame.get("value_state") == "present"]
    assert positives == [], frames

    from local_operator.network.credentials import offers as offers_mod
    from local_operator.network.credentials import owner as owner_mod

    broker = owner_mod.broker_for_relay(mesh.a)
    assert broker is not None
    link = SimpleNamespace(device_id=mesh.member)
    detail = sync.owner_copy(
        broker, link, {"kind": "copy", "key": KEY, "gen": 1, "held": 0, "from_device": mesh.member}
    )
    assert detail.get("code") == "local_only", detail
    keys = {row["key"] for row in offers_mod.build_items(mesh.a.root, network_id=mesh.network_id)}
    assert KEY not in keys, "a local-only key is not a candidate"


def test_a_payload_naming_another_secret_is_dropped(secret_mesh: Any) -> None:
    """Defence in depth under the digest pin: a mis-addressed reply writes nothing."""
    mesh = secret_mesh
    entry = SimpleNamespace(kind="store-secret", provider=SECRET_NAME, key=KEY)
    out = mesh.engine_b._apply_value(
        KEY,
        entry,
        {
            "type": "store-secret",
            "name": "SOMETHING_ELSE",
            "value": b"wrong-address".hex(),
            "description": "",
        },
        mesh.owner,
        3,
    )
    assert out is None
    assert not _member_has(mesh.b.root, SECRET_NAME)
    assert not _member_has(mesh.b.root, "SOMETHING_ELSE")


def test_the_needs_list_and_sync_marks_select_the_default_set(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.2's intersection, as arithmetic: needs ∪ ``sync``; everything else offered.

    A key no bundle declares and nobody marked is OFFERED with ``share`` false —
    the "not copied" half of the default; a needs-list name and a ``sync`` mark
    are on; ``local-only`` is not a candidate at all.
    """
    config = root / "defaults"
    config.mkdir(parents=True, exist_ok=True)
    # The device's pushed bundles declare ONE ref: the needs-list, written the
    # way the device's own writers write it (``mcp.json``).
    (config / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "hub": {
                        "type": "http",
                        "url": "https://mcp.example/sse",
                        "env": {"TOKEN": "${NEEDED_TOKEN}"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    handle = _secret_store(config, create=True)
    for name in ("NEEDED_TOKEN", "MARKED_TOKEN", "PLAIN_TOKEN", "PRIVATE_TOKEN"):
        handle.set(name, b"v-" + name.encode())

    network_id = "n_defaults"
    with sync.mutate(network_id, config) as state:
        state.record_mark("secret:MARKED_TOKEN", "sync")
        state.record_mark("secret:PRIVATE_TOKEN", "local-only")

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    from local_operator.network.credentials import offers as offers_mod

    assert sync.needs_names(config) == frozenset({"NEEDED_TOKEN"})
    items = offers_mod.build_items(config, network_id=network_id)
    by_key = {row["key"]: row for row in items}
    assert by_key["secret:NEEDED_TOKEN"]["share"] is True
    assert by_key["secret:MARKED_TOKEN"]["share"] is True
    assert by_key["secret:PLAIN_TOKEN"]["share"] is False  # offered, not copied
    assert by_key["secret:PLAIN_TOKEN"]["kind"] == "store-secret"
    assert "secret:PRIVATE_TOKEN" not in by_key


def test_the_bare_secret_name_resolves_for_share_and_revoke(secret_mesh: Any) -> None:
    """Q2 (review round 1): ``share``/``revoke`` take the name ``lop secret list`` prints.

    Both verbs used to read a bare name as a PROVIDER, so a store secret answered
    a login-flavored remedy (share) or ``no placement`` while the placement existed
    under ``secret:<NAME>`` (revoke). The bare spelling now canonicalises to the
    secret key — and ONLY when no provider of that name is held, which is why this
    rig (no provider logins) exercises exactly the fallback reading. The assertion
    is on the document the relays actually read, not on the receipt wording.
    """
    from local_operator.network.credentials import placement as placement_mod

    mesh = secret_mesh

    def _holder() -> bool:
        doc = placement_mod.PlacementDocument.resolve(
            mesh.a.root, self_device=mesh.owner, network_id=mesh.network_id
        )
        assert doc is not None
        entry = doc.entry(KEY)
        return entry is not None and any(h.device == mesh.member for h in entry.holders)

    assert _holder(), "the fixture's share must be present to start from"
    assert _lop_network("credential", "revoke", SECRET_NAME, "--from", mesh.b.identity.name) == 0
    assert not _holder(), "the bare name did not revoke the store secret"
    from tests.unit.network.finite_recovery import restore_finite

    restore_finite(mesh.a.root, SECRET_NAME, mesh.b.identity.name)
    assert _holder(), "the bare name did not share the store secret"


# -- the expiry diagnostics themselves ---------------------------------------


def test_an_expired_copy_wait_carries_a_redacted_snapshot(
    secret_mesh: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A forced expiry fails with the snapshot, logs it, and leaks no secret material.

    The predicate never holds and the budget is tiny, so the wait expires in
    ~0.2 s. The ``assert ..., msg`` form is what the real cells use; the cell
    runs it and inspects the AssertionError. It also proves the message is
    built only on expiry: a passing wait never calls the helper.
    """
    from local_operator.secrets.keys import load_master_key

    mesh = secret_mesh
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    calls: list[int] = []

    # PASSING PATH: the message expression must not run (laziness of assert msg).
    assert net_fixtures.wait_for(lambda: True, timeout_s=0.2), calls.append(1)
    assert calls == []

    caplog.set_level(logging.ERROR)
    with pytest.raises(AssertionError) as failure:
        assert net_fixtures.wait_for(lambda: False, timeout_s=0.2), _sync_snapshot(mesh, code)
    message = str(failure.value)
    for field in (
        "sync snapshot:",
        f"step={code!r}",
        "owner_engine=[",
        "member_engine=[",
        "owner_links=",
        "member_links=",
        "ledger=[",
        "owner_gen=",
        "member_store_names=",
        "net wait EXPIRED",
    ):
        assert field in message, (field, message)
    assert any("sync snapshot:" in record.getMessage() for record in caplog.records)

    # REDACTION: neither the value, nor either master key, nor the owner's
    # device-id-free secret record content appears in the message or the log.
    haystack = message + "\n" + caplog.text
    assert VALUE_1.decode() not in haystack
    assert VALUE_2.decode() not in haystack
    for root in (mesh.a.root, mesh.b.root):
        key = bytes(load_master_key(root))
        assert key.hex() not in haystack
        assert key.decode("latin-1") not in haystack


def test_the_snapshot_survives_a_half_built_mesh() -> None:
    """Every probe is guarded: a broken mesh yields ``<unavailable: Type>``, never a raise."""
    text = _sync_snapshot(SimpleNamespace(), step_code=None)
    assert text.startswith("sync snapshot:")
    assert text.count("<unavailable: AttributeError>") >= 5, text
