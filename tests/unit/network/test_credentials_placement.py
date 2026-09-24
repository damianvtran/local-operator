"""Placement, observations, and the predicate that keeps a 0-peer device untouched.

The document rules are the whole authorisation model for the broker, so they are
checked as RULES rather than through one happy path: who may write a row, what a
merged row may say, what may never appear in the file at all, and what a device with
no placement gets (the unchanged ``AuthStore``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials import store as mesh_store
from local_operator.network.credentials.state import (
    DEFAULT_OBSERVATION_TTL_MS,
    PlacementState,
)
from local_operator.network.credentials.types import (
    NO_RETRY,
    BrokerError,
    CredentialRef,
    Grant,
    synthetic_credential_id,
)
from local_operator.network.types import MeshRefusal

OWNER = "d_00000000000000000000000000000001"
OTHER = "d_00000000000000000000000000000002"
THIRD = "d_00000000000000000000000000000003"
NETWORK = "n_placement_test"


def _document(root: Path) -> placement_mod.PlacementDocument:
    return placement_mod.PlacementDocument(NETWORK, root=root, written_by=OWNER)


def _entry(document: placement_mod.PlacementDocument, key: str) -> Any:
    """``document.entry(key)`` with the absence asserted, once.

    ``entry`` returns ``None`` for a key the document does not carry, and a test that
    dereferenced the result directly would be asserting through a ``None`` that a
    checker cannot rule out. Narrowing here keeps every call site readable and means a
    typo in the key fails as ``AssertionError: deepseek`` rather than as
    ``AttributeError: 'NoneType'``.
    """
    entry = document.entry(key)
    assert entry is not None, key
    return entry


def _declared(root: Path) -> placement_mod.PlacementDocument:
    document = _document(root)
    document.declare(
        "openai",
        owner_device=OWNER,
        owner_device_name="damian-mbp",
        provider="openai",
        identity_label="damian@example.test",
        by=OWNER,
    )
    return document


# ---------------------------------------------------------------------------
# Who may write a row
# ---------------------------------------------------------------------------


def test_only_the_owner_may_declare_a_credential(tmp_path: Path) -> None:
    """A device may declare only what IT holds — checked, not trusted to the caller."""
    document = _document(tmp_path)
    with pytest.raises(MeshRefusal) as refusal:
        document.declare("openai", owner_device=OWNER, provider="openai", by=OTHER)
    assert refusal.value.code == "not_owner"


def test_only_the_owner_may_change_who_borrows(tmp_path: Path) -> None:
    document = _declared(tmp_path)
    with pytest.raises(MeshRefusal) as refusal:
        document.grant("openai", OTHER, by=OTHER)
    assert refusal.value.code == "not_owner"
    with pytest.raises(MeshRefusal):
        document.revoke("openai", OTHER, by=OTHER)
    assert _entry(document, "openai").holders == [] or document.is_holder("openai", OWNER)


def test_the_owner_is_always_its_own_holder(tmp_path: Path) -> None:
    """A document whose owner could not use what it holds is one no reader can act on."""
    entry = _declared(tmp_path).entry("openai")
    assert entry is not None
    assert entry.is_holder(OWNER)
    own_holder = entry.holder(OWNER)
    assert own_holder is not None and own_holder.scope == "device"


def test_the_owner_cannot_revoke_itself(tmp_path: Path) -> None:
    with pytest.raises(MeshRefusal) as refusal:
        _declared(tmp_path).revoke("openai", OWNER, by=OWNER)
    assert refusal.value.code == "not_owner"


def test_absence_is_a_refusal_not_a_default_allow(tmp_path: Path) -> None:
    """``holders`` IS the authorisation: a device not in it is not a holder."""
    document = _declared(tmp_path)
    assert not document.is_holder("openai", OTHER)
    assert not document.is_holder("deepseek", OWNER)


def test_revoking_a_device_that_never_held_it_is_refused(tmp_path: Path) -> None:
    with pytest.raises(MeshRefusal) as refusal:
        _declared(tmp_path).revoke("openai", OTHER, by=OWNER)
    assert refusal.value.code == "not_a_holder"


def test_a_regrant_widens_and_never_narrows(tmp_path: Path) -> None:
    """A re-share is an increase; narrowing is ``revoke``. Silently reducing a
    working peer's scope on a typo would break a device that is mid-turn."""
    document = _declared(tmp_path)
    document.grant("openai", OTHER, scope="device", by=OWNER)
    granted = _entry(document, "openai").holder(OTHER)
    assert granted is not None and granted.scope == "device"
    document.grant("openai", OTHER, scope="session", by=OWNER)
    still = _entry(document, "openai").holder(OTHER)
    assert still is not None and still.scope == "device"


# ---------------------------------------------------------------------------
# What may never appear in the file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forbidden", ["access_token", "refresh_token", "secret", "access"])
def test_the_writer_refuses_a_document_carrying_material(tmp_path: Path, forbidden: str) -> None:
    """A gate that refuses the write, because a dropped field is a silent lie.

    The list is this document's own, NOT ``audit.FORBIDDEN_DETAIL_KEYS``: in the
    audit log ``key`` means KEY MATERIAL, while here it is the credential's NAME and
    the first field of every entry — a shared list would have refused every legitimate
    document, which is how a safety check gets deleted instead of fixed.
    """
    payload = {"schema": 1, "credentials": [{"key": "openai", forbidden: "x"}]}
    with pytest.raises(ValueError) as refusal:
        placement_mod._assert_no_material(payload)  # noqa: SLF001 — the guard itself
    assert forbidden in str(refusal.value)


def test_a_legitimate_document_passes_the_material_check(tmp_path: Path) -> None:
    """The other direction, so the check above cannot be satisfied by refusing all."""
    entry = _declared(tmp_path).entry("openai")
    assert entry is not None
    json.dumps(entry.to_json())  # the `key` field is a NAME and is allowed


# ---------------------------------------------------------------------------
# Merging a peer's document
# ---------------------------------------------------------------------------


def test_a_merge_accepts_only_rows_the_sender_owns(tmp_path: Path) -> None:
    """A claim about a THIRD device is a rumour, not a membership statement."""
    document = _document(tmp_path)
    incoming = {
        "epoch": 7,
        "credentials": [
            _row("openai", owner=OTHER, rev=2),
            _row("deepseek", owner=THIRD, rev=2),
        ],
    }
    changed = document.merge(incoming, from_device=OTHER)
    assert changed == ["openai"]
    assert document.owner_of("deepseek") == ""


def test_a_merge_never_overwrites_a_row_we_own(tmp_path: Path) -> None:
    document = _declared(tmp_path)
    document.grant("openai", OTHER, by=OWNER)
    incoming = {"credentials": [{**_row("openai", owner=OTHER, rev=9), "holders": []}]}
    assert document.merge(incoming, from_device=OTHER) == []
    assert document.is_holder("openai", OTHER), "a peer rewrote our own row"


def test_a_replayed_row_cannot_revert_a_newer_revoke(tmp_path: Path) -> None:
    """Higher ``doc_rev`` wins that key; equal loses, so a replay is a no-op."""
    document = _document(tmp_path)
    document.merge(
        {"credentials": [{**_row("openai", owner=OTHER, rev=5), "holders": []}]}, from_device=OTHER
    )
    assert document.owner_of("openai") == OTHER
    assert (
        document.merge({"credentials": [_row("openai", owner=OTHER, rev=5)]}, from_device=OTHER)
        == []
    )
    assert (
        document.merge({"credentials": [_row("openai", owner=OTHER, rev=4)]}, from_device=OTHER)
        == []
    )
    assert document.merge(
        {"credentials": [_row("openai", owner=OTHER, rev=6)]}, from_device=OTHER
    ) == ["openai"]


def _row(key: str, *, owner: str, rev: int) -> dict[str, Any]:
    return {
        "key": key,
        "provider": key,
        "kind": "oauth-rotating",
        "owner_device": owner,
        "owner_device_name": owner[-4:],
        "identity_label": "",
        "holders": [{"device": owner, "scope": "device", "granted_at": 0.0, "granted_by": owner}],
        "declared_at": 0.0,
        "doc_rev": rev,
    }


def test_a_document_round_trips_through_disk(tmp_path: Path) -> None:
    document = _declared(tmp_path)
    document.grant("openai", OTHER, scope="session", by=OWNER)
    document.save()
    reloaded = placement_mod.PlacementDocument.load(NETWORK, tmp_path)
    assert reloaded.owner_of("openai") == OWNER
    assert reloaded.is_holder("openai", OTHER)
    assert _entry(reloaded, "openai").identity_label == "damian@example.test"


def test_the_document_is_written_privately(tmp_path: Path) -> None:
    """0600 on the file, 0700 on the directory, checked on the real write."""
    _declared(tmp_path).save()
    path = placement_mod.placement_path(NETWORK, tmp_path)
    parent = path.parent
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct(parent.stat().st_mode & 0o777) == "0o700"
    network_dir = parent.parent
    assert (
        oct(network_dir.stat().st_mode & 0o777) == "0o700"
    ), "an intermediate directory took the umask's mode and published the document's name"


# ---------------------------------------------------------------------------
# Kimi, and the other device-bound providers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["kimi"])
def test_a_device_bound_provider_cannot_be_declared(tmp_path: Path, key: str) -> None:
    """Refused on DECLARE and on GRANT, because either one alone leaves a way in."""
    document = _document(tmp_path)
    with pytest.raises(MeshRefusal) as refusal:
        document.declare(key, owner_device=OWNER, provider=key, by=OWNER)
    assert refusal.value.code == "device_bound"


def test_a_device_bound_row_already_on_disk_is_refused_at_grant(tmp_path: Path) -> None:
    """A document written before the rule existed is still refused — at the second gate.

    This is why the check is in two places: a file is durable, and the code that wrote
    it may have been an earlier build.
    """
    document = _document(tmp_path)
    entry = document.entry("kimi")
    assert entry is None
    document.entries["kimi"] = placement_mod.CredentialPlacementEntry(
        key="kimi", provider="kimi", kind="oauth-rotating", owner_device=OWNER
    )
    with pytest.raises(MeshRefusal) as refusal:
        document.grant("kimi", OTHER, by=OWNER)
    assert refusal.value.code == "device_bound"


# ---------------------------------------------------------------------------
# The device-local observation document
# ---------------------------------------------------------------------------


def test_an_observation_expires_on_its_ttl(tmp_path: Path) -> None:
    """The refusal cache: believed for its TTL and no longer.

    A cache that outlived its TTL would keep a recovered owner marked offline, which
    is a capability the operator cannot explain and cannot clear.
    """
    state = PlacementState(NETWORK, root=tmp_path)
    state.observe("openai", "owner_offline", owner_device=OWNER, retry_after_ms=1000)
    now = time.time()
    assert state.status("openai", now=now) == "owner_offline"
    assert state.status("openai", now=now + 5.0) == ""
    assert state.observation("openai", now=now + 5.0) is None


def test_an_observation_with_no_ttl_gets_the_default(tmp_path: Path) -> None:
    state = PlacementState(NETWORK, root=tmp_path)
    state.observe("openai", "not_a_holder")
    observed = state.observation("openai")
    assert observed is not None
    assert observed["retry_after_ms"] == DEFAULT_OBSERVATION_TTL_MS


def test_a_successful_borrow_clears_a_standing_refusal(tmp_path: Path) -> None:
    state = PlacementState(NETWORK, root=tmp_path)
    state.observe("openai", "not_a_holder", owner_device=OWNER, retry_after_ms=300_000)
    state.note_grant("openai", "g_1", owner_device=OWNER)
    assert state.status("openai") == "active"
    assert state.last_grant_at("openai") is not None


def test_a_refusal_is_cached_for_the_code_s_own_ttl(tmp_path: Path) -> None:
    """The codes are a closed set WITH TTLs, so a borrower never has to guess."""
    from local_operator.network.credentials.types import BROKER_ERROR_TTL_MS

    assert BrokerError(code="owner_offline").cache_ttl_ms == 15_000
    assert BrokerError(code="not_a_holder").cache_ttl_ms == 60_000
    assert BrokerError(code="grant_invalid").cache_ttl_ms == 300_000
    assert BrokerError(code="unsupported").cache_ttl_ms == NO_RETRY
    assert BrokerError(code="rate_limited", retry_after_ms=7_000).cache_ttl_ms == 7_000
    assert set(BROKER_ERROR_TTL_MS) >= {"owner_offline", "not_a_holder", "revoked"}


def test_an_observation_document_is_written_privately_and_round_trips(tmp_path: Path) -> None:
    state = PlacementState(NETWORK, root=tmp_path)
    state.observe("openai", "grant_invalid", owner_device=OWNER, retry_after_ms=60_000)
    state.save()
    path = placement_mod.placement_state_path(NETWORK, tmp_path)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    reloaded = PlacementState.load(NETWORK, tmp_path)
    assert reloaded.status("openai") == "grant_invalid"


def test_the_state_writer_refuses_material_too(tmp_path: Path) -> None:
    """The cheapest proof that a bearer cannot become durable: refuse it at the writer."""
    state = PlacementState(NETWORK, root=tmp_path)
    state._observations["openai"] = {"key": "openai", "access_token": "x"}  # noqa: SLF001
    with pytest.raises(ValueError):
        state.to_json()


# ---------------------------------------------------------------------------
# The predicate: a device that borrows nothing runs the OLD code
# ---------------------------------------------------------------------------


def _point_config_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """``AuthStore``'s database follows ``paths.config_dir()``, not the argument."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))


def test_a_device_with_no_placement_reads_nothing_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bare"
    root.mkdir()
    _point_config_at(monkeypatch, root)
    assert not placement_mod.has_any_placement(root)
    # THE SIDE EFFECT IS THE POINT: a read that mkdirs would put a `<config>/network`
    # footprint on every machine that so much as constructs a session.
    assert not (root / "network").exists()
    assert placement_mod.placement_for_store(root) is None


def test_never_minted_identity_is_not_minted_by_the_predicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 0-peer device must not grow a keypair because a session was constructed."""
    root = tmp_path / "bare"
    root.mkdir()
    _point_config_at(monkeypatch, root)
    _declared(root).save()  # a document, but no identity
    assert placement_mod.placement_for_store(root) is None
    assert not (root / "network" / "identity" / "device.json").exists()


def test_build_auth_store_returns_the_plain_store_with_nothing_borrowable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 0-peer guarantee, asserted on the TYPE the session path receives.

    ``is AuthStore`` — not "behaves like" — because that is the object every caller
    had before this slice existed, and the whole non-regression argument is that the
    mesh adds a rung rather than replacing the store.
    """
    from local_operator.providers.auth_store import AuthStore

    root = tmp_path / "bare"
    root.mkdir()
    _point_config_at(monkeypatch, root)
    store = mesh_store.build_auth_store(root)
    assert type(store) is AuthStore

    # A document we own entirely is still not borrowable: the broker rung could only
    # ever delegate to ourselves, so the wrapper would be a layer with no work to do.
    _declared(root).save()
    from local_operator.network.identity import mint

    identity = mint(root, name="solo")
    own = placement_mod.PlacementDocument.load(NETWORK, root, self_device=identity.device_id)
    own.entries["openai"].owner_device = identity.device_id
    own.save()
    plain = mesh_store.build_auth_store(root)
    assert type(plain) is AuthStore
    plain.close()


def test_build_auth_store_returns_the_mesh_store_when_borrowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the swap happens exactly when another device owns something we may borrow."""
    root = tmp_path / "borrower"
    root.mkdir()
    _point_config_at(monkeypatch, root)
    from local_operator.network.identity import mint

    identity = mint(root, name="borrower")
    document = placement_mod.PlacementDocument(NETWORK, root=root, written_by=OWNER)
    document.declare("openai", owner_device=OWNER, provider="openai", by=OWNER)
    document.grant("openai", identity.device_id, scope="session", by=OWNER)
    document.save()
    store = mesh_store.build_auth_store(root)
    try:
        assert isinstance(store, mesh_store.MeshAwareAuthStore)
        assert store.mesh_client is not None
        assert store.mesh_client.should_borrow("openai")
        assert not store.mesh_client.should_borrow("deepseek")
        # The synthetic row is what makes "why is there no bearer" answerable at all.
        rows = store.list_credentials("openai")
        assert [row.id for row in rows] == [synthetic_credential_id("openai", OWNER)]
    finally:
        store.close()


def test_a_borrowed_credential_is_never_refreshed_and_is_never_blocked_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The borrower's refusal to act, asserted on the methods themselves.

    ``ensure_oauth_fresh`` on a synthetic id answers ``None`` because there IS nothing
    to refresh; ``is_blocked_for_model`` answers ``False`` because the verdict belongs
    to the owner's row; ``get_credential`` returns a descriptor with no payload.
    """
    import asyncio

    root = tmp_path / "borrower"
    root.mkdir()
    _point_config_at(monkeypatch, root)
    from local_operator.providers.auth_store import AuthStore

    wrapper = mesh_store.MeshAwareAuthStore(AuthStore(config_dir=root), mesh=None, config_dir=root)
    try:
        synthetic = synthetic_credential_id("openai", OWNER)
        assert asyncio.run(wrapper.ensure_oauth_fresh(synthetic)) is None
        assert asyncio.run(wrapper.ensure_oauth_fresh_or_raise(synthetic)) is None
        assert wrapper.is_blocked_for_model(synthetic, "openai", "gpt-5") is False
        assert wrapper.is_blocked(synthetic, "openai") is False
        assert wrapper.grant_is_dead(synthetic) is False
        assert wrapper.get_credential(synthetic) is None
    finally:
        wrapper.close()


# ---------------------------------------------------------------------------
# The CLI surfaces
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    """The real ``lop network`` parser, built the way ``lop`` builds it.

    ``add_parser`` registers the ``network`` group ON the subparsers it is handed, so
    the top level has to exist and the verbs are reached through it — which is also
    what makes this a test of the shipped wiring rather than of a copy of it.
    """
    from local_operator.network import cli as network_cli

    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="command")
    network_cli.add_parser(subparsers)
    return parser


def test_the_credential_verbs_parse() -> None:
    parser = _parser()
    share = parser.parse_args(["network", "credential", "share", "openai", "--with", "peer-b"])
    assert (share.credential_command, share.key, share.device, share.scope) == (
        "share",
        "openai",
        "peer-b",
        "session",
    )
    revoke = parser.parse_args(
        ["network", "credential", "revoke", "mcp:https://x.test/mcp", "--from", "peer-b"]
    )
    assert revoke.credential_command == "revoke"
    assert parser.parse_args(["network", "credentials", "--json"]).json is True


def _bare_args_reads(function: Any, verb: str, seen: set[str] | None = None) -> set[str]:
    """Every bare ``args.<name>`` ``function`` can reach for ``verb`` — through helpers too.

    THREE REFINEMENTS OVER A FLAT SCAN, each for a way a real gap could hide:

    * **Per verb.** ``_cmd_credential`` serves ``share`` and ``revoke``, and a read inside
      ``if verb == "share":`` is not one ``revoke`` ever makes. Branches guarded by the
      OTHER verb are skipped, so each verb is checked against exactly its own reads
      and a future revoke-only read cannot hide behind ``share`` (review round 3, F4).
    * **Through helpers.** A module function the handler hands ``args`` to is followed
      (``_emit``, ``_cmd_credential`` from the guard), so a read moved into a helper is
      still seen.
    * **Bare reads only.** ``getattr(args, name, default)`` is a deliberate optional read
      and cannot crash; ``args.name`` raises ``AttributeError`` when the parser never
      defined it, which is the Q1 crash.
    """
    import ast
    import inspect
    import textwrap

    from local_operator.network import cli as network_cli

    seen = set() if seen is None else seen
    if function.__name__ in seen:
        return set()
    seen.add(function.__name__)
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))

    def _verb_test(node: ast.AST) -> str | None:
        """The verb an ``if verb == "<x>":`` names, else ``None``."""
        test = getattr(node, "test", None)
        if (
            isinstance(node, ast.If)
            and isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "verb"
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
        ):
            return str(test.comparators[0].value)
        return None

    names: set[str] = set()

    def _walk(node: ast.AST) -> None:
        named = _verb_test(node)
        if named is not None and isinstance(node, ast.If):
            # ``if verb == X`` with two verbs: the body is X's, the ``else`` the other's.
            # Only the half that belongs to THIS verb is walked.
            for child in node.body if named == verb else node.orelse:
                _walk(child)
            return
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "args":
                names.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            helper = getattr(network_cli, node.func.id, None)
            passes_args = any(isinstance(a, ast.Name) and a.id == "args" for a in node.args)
            if passes_args and inspect.isfunction(helper):
                names.update(_bare_args_reads(helper, verb, seen))
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)
    return names


@pytest.mark.parametrize(
    ("argv", "handler_name"),
    [
        (["credential", "share", "openai", "--with", "peer-b"], "_guard_credential_subcommand"),
        (["credential", "revoke", "openai", "--from", "peer-b"], "_guard_credential_subcommand"),
        (["credentials"], "_cmd_credentials"),
    ],
)
def test_every_argument_the_credential_handlers_read_is_one_the_parser_defines(
    argv: list[str], handler_name: str
) -> None:
    """THE CLASS OF GAP BEHIND Q1, checked for every credential verb at once.

    ``_cmd_credential`` read ``args.network`` and no parser defined it, so the only
    share/revoke surface raised ``AttributeError`` for every input; the test that should
    have caught it set the attribute by hand. Here the handler's own source — followed
    into every helper it hands ``args`` to, and narrowed to the verb under test — says
    which attributes it reads, and the namespace the SHIPPED parser produces for that
    verb must carry each one.
    """
    from local_operator.network import cli as network_cli

    namespace = vars(_parser().parse_args(["network", *argv]))
    verb = argv[1] if argv[0] == "credential" else argv[0]
    handler = getattr(network_cli, handler_name)
    reads = _bare_args_reads(handler, verb)
    if argv[0] == "credential":
        # A LIVENESS CHECK ON THE INSTRUMENT: share/revoke really do read ``key`` and
        # ``device`` bare, so a scan that returned nothing would be a scan that stopped
        # looking. ``credentials`` reads ``args`` only through ``getattr``, so nothing
        # bare is the correct answer there.
        assert {"key", "device", "network"} <= reads, reads
    missing = reads - set(namespace)
    assert (
        not missing
    ), f"{handler_name} reads {sorted(missing)} for {argv} but the parser defines none"


def test_a_bare_credential_verb_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Neither verb is safe as a default: one widens authority, one cuts a device off."""
    from local_operator.network import cli as network_cli

    args = _parser().parse_args(["network", "credential"])
    assert network_cli.main(args) == 2
    captured = capsys.readouterr()
    assert "usage: lop network credential share" in captured.err


def test_sharing_something_this_device_does_not_hold_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A share is only meaningful on the device that HOLDS the login.

    Declaring a row this device cannot serve would put an entry in every peer's
    document whose owner refuses, and the sentence the borrower reads would then tell
    the operator to sign in somewhere they already are.
    """
    from local_operator.network import cli as network_cli
    from local_operator.network import store as network_store
    from local_operator.network.identity import mint
    from local_operator.network.types import NetworkRecord, SecretState

    root = tmp_path / "owner"
    root.mkdir()
    _point_config_at(monkeypatch, root)
    identity = mint(root, name="owner-laptop")
    record = NetworkRecord(
        network_id=NETWORK,
        name="testnet",
        created_by=identity.device_id,
        self_device_id=identity.device_id,
        self_role="admin",
        self_capabilities=["admin"],
    )
    from local_operator.network import relay as relay_mod
    from local_operator.network import wire

    relay_mod.admit(
        record,
        device_id=identity.device_id,
        public_key=identity.public_key,
        name=identity.name,
        role="admin",
        capabilities=["admin"],
        added_by=identity.device_id,
        added_via="self",
        root=root,
        persist=False,
    )
    relay_mod.admit(
        record,
        device_id=OTHER,
        public_key="a" * 43,
        name="peer-b",
        role="drive",
        capabilities=["drive"],
        added_by=identity.device_id,
        added_via="invite",
        root=root,
        persist=False,
    )
    network_store.save(record, root)
    network_store.save_secrets(
        SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"0" * 32)), root
    )

    # EXACTLY AS TYPED, with nothing set on the namespace by hand: the first version
    # of this test set ``args.network`` itself, which is how a parser that never
    # defined ``--network`` crashed every real share while CI stayed green (QA round
    # 1, Q1). One network here, so no ``--network`` is needed — the common case.
    args = _parser().parse_args(["network", "credential", "share", "openai", "--with", "peer-b"])
    assert network_cli.main(args) == 1
    captured = capsys.readouterr()
    assert "no credential for 'openai'" in captured.err
    # Nothing was written: a refused share must not leave a placement row behind.
    assert not placement_mod.has_any_placement(root)


def test_one_contended_merge_does_not_stop_the_other_members_being_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 2, m3: the lock's timeout is a ``MeshRefusal``, not an ``OSError``.

    ``pull_placement`` caught only ``OSError``, so one member whose merge hit the 10 s
    lock timeout ended the loop and every later member's document went unmerged. The
    first member's merge is made to time out here; the second must still land.
    """
    from local_operator.network.credentials import client as client_mod

    asked: list[str] = []
    merged: list[str] = []

    def _merge(
        network_id: str, document: dict[str, Any], *, from_device: str, **_: Any
    ) -> list[str]:
        if from_device == OWNER:
            raise MeshRefusal("busy", "another writer is changing who may borrow")
        merged.append(from_device)
        return ["deepseek"]

    client = client_mod.MeshCredentialClient(
        root=tmp_path, self_device=THIRD, network_id=NETWORK, placement=_document(tmp_path)
    )
    monkeypatch.setattr(client_mod, "merge_from_peer", _merge)
    monkeypatch.setattr(client, "_other_active_members", lambda: [OWNER, OTHER])

    def _ask(device: str, frame: dict[str, Any]) -> dict[str, Any]:
        asked.append(device)
        return {"op": "ack", "detail": {"kind": "placement", "document": {"credentials": []}}}

    monkeypatch.setattr(client, "_ask_owner_directly", _ask)
    result = client.pull_placement()
    assert asked == [OWNER, OTHER]
    assert merged == [OTHER], "the contended merge stopped the loop"
    assert result["changed"] == ["deepseek"] and result["owners"] == 2


def test_forgetting_a_network_removes_its_placement_and_state(tmp_path: Path) -> None:
    """``disconnect`` must not leave a list of who may borrow what behind."""
    _declared(tmp_path).save()
    PlacementState(NETWORK, root=tmp_path).save()
    assert placement_mod.has_any_placement(tmp_path)
    assert placement_mod.forget_network(NETWORK, tmp_path) == 2
    assert not placement_mod.has_any_placement(tmp_path)


def test_a_grant_is_the_only_thing_with_a_serializer_named_after_one() -> None:
    """A borrowed bearer has no durable form, asserted structurally.

    ``GrantCache`` has ``get``/``put``/``drop``/``clear`` and no writer: there is no
    ``to_json``, no path and no pickle hook, so a future caller cannot make a borrowed
    token durable without adding one — which is a change review would see.
    """
    from local_operator.network.credentials.client import GrantCache

    cache = GrantCache()
    assert not [
        name for name in ("to_json", "save", "dump", "pickle", "path") if hasattr(cache, name)
    ]
    grant = Grant(
        access_token="a",
        kind="bearer",
        token_expires_at_ms=0,
        grant_expires_at_ms=int((time.time() + 600) * 1000),
        credential_ref=CredentialRef(
            owner_device=OWNER,
            owner_device_name="o",
            provider="openai",
            kind="oauth",
            credential_id=1,
        ),
        served_by=OWNER,
    )
    cache.put("openai", "sess", grant)
    assert cache.get("openai", "sess") is grant
    cache.drop("openai")
    assert cache.get("openai", "sess") is None
    assert "access_token" not in json.dumps({"cache": str(cache)})


# ---------------------------------------------------------------------------
# The borrower's own database
# ---------------------------------------------------------------------------


class _Wire:
    """The ONE fake in this section, and it stands in for the wire, not the store.

    The wrapper, the local ``AuthStore``, its SQLite file and the grant objects are
    all real; only the request that would cross to the owner is replaced, because
    what is under test here is what the BORROWER writes down — which a live owner
    would only obscure.
    """

    def __init__(
        self,
        document: placement_mod.PlacementDocument,
        grant: Grant,
        *,
        self_device: str,
    ) -> None:
        from local_operator.network.credentials.client import GrantCache

        self.placement = document
        # The BORROWER'S REAL id, minted by ``identity.mint`` above: a fake that
        # invented one would make ``should_borrow`` answer for a device the document
        # has never heard of, and the test would pass by refusing to borrow at all.
        self.self_device = self_device
        self.grants = GrantCache()
        self._grant = grant
        self.reports: list[tuple[str, str]] = []

    def should_borrow(self, key: str) -> bool:
        return (
            self.placement.is_holder(key, self.self_device)
            and self.placement.owner_of(key) != self.self_device
        )

    def owner_of(self, key: str) -> str:
        return self.placement.owner_of(key)

    def owner_label(self, key: str) -> str:
        return "owner-laptop"

    def owner_last_seen_s(self, device: str) -> float | None:
        return 4.0

    async def grant_async(self, key: str, **_: Any) -> Grant:
        return self._grant

    def report_sync(self, key: str, *, kind: str, **_: Any) -> None:
        self.reports.append((key, kind))

    def close(self) -> None:
        self.grants.clear()


#: The bearer this section's fake wire hands back. Assembled from parts rather than
#: written as one literal: it is not a secret, and a fully spelled-out token-shaped
#: string in a test file is the shape every reviewer has to stop and check.
BORROWED_BEARER = "-".join(("borrowed", "bearer"))
LOCAL_KEY = "-".join(("local", "key"))


def _borrower_grant() -> Grant:
    return Grant(
        access_token=BORROWED_BEARER,
        kind="bearer",
        token_expires_at_ms=int((time.time() + 3600) * 1000),
        grant_expires_at_ms=int((time.time() + 900) * 1000),
        credential_ref=CredentialRef(
            owner_device=OWNER,
            owner_device_name="owner-laptop",
            provider="openai",
            kind="oauth",
            credential_id=7,
        ),
        served_by=OWNER,
        grant_id="g_test",
    )


def _dump(db_path: Path) -> str:
    """``sqlite3``'s own dump — the design's stated instrument, not a row count."""
    import sqlite3

    connection = sqlite3.connect(str(db_path))
    try:
        return "\n".join(connection.iterdump())
    finally:
        connection.close()


def _borrower_wrapper(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, placement_mod.PlacementDocument, Path]:
    _point_config_at(monkeypatch, root)
    from local_operator.network.identity import mint
    from local_operator.providers.auth_store import AuthStore

    identity = mint(root, name="borrower")
    document = placement_mod.PlacementDocument(NETWORK, root=root, written_by=OWNER)
    document.declare("openai", owner_device=OWNER, provider="openai", by=OWNER)
    document.grant("openai", identity.device_id, scope="session", by=OWNER)
    document.save()
    local = AuthStore(config_dir=root)
    wrapper = mesh_store.MeshAwareAuthStore(
        local,
        mesh=_Wire(document, _borrower_grant(), self_device=identity.device_id),
        config_dir=root,
    )
    return wrapper, document, local.db_path


def test_a_brokered_turn_leaves_the_borrowers_database_logically_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The designer's instrument: ``sqlite3 .dump`` before and after a brokered turn.

    This is the assertion behind "the borrower holds only a short-lived access token,
    in memory". A dump rather than a row count, because what must not change is the
    SCHEMA and every value in it: a block, a sticky pointer or a tombstone written
    against a row this device does not own would all show up here.
    """
    root = tmp_path / "borrower"
    root.mkdir()
    wrapper, _document, db_path = _borrower_wrapper(root, monkeypatch)
    try:
        before = _dump(db_path)
        key = asyncio.run(wrapper.get_api_key("openai", "sess-1"))
        assert key == BORROWED_BEARER
        assert _dump(db_path) == before
    finally:
        wrapper.close()


def test_an_offline_owner_reaches_the_user_in_the_brokers_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1, Q6: the turn said "rate limit or quota exceeded" for an offline owner.

    The synthetic row this store lists for a borrowable key made the failover driver's
    diagnosis read "rate limited, a token refresh failed…". The broker's own sentence
    names the owner and the local remedy, and is what the user must see.
    """
    from local_operator.providers.failover import _no_credential_error

    root = tmp_path / "borrower"
    root.mkdir()
    wrapper, _document, _db = _borrower_wrapper(root, monkeypatch)
    sentence = (
        "No credential for 'openai' is reachable: owner-laptop owns it and was last seen "
        "4 min ago. Reconnect that device, or run 'lop login openai' here to use your own "
        "account."
    )

    async def _offline(key: str, **_: Any) -> BrokerError:
        return BrokerError(code="owner_offline", key=key, owner_device=OWNER, message=sentence)

    wrapper._mesh.grant_async = _offline  # type: ignore[union-attr]  # noqa: SLF001
    try:
        assert asyncio.run(wrapper.get_api_key("openai", "sess-1")) is None
        error = _no_credential_error(wrapper, "openai")
        assert str(error) == sentence
        assert "rate limit" not in str(error)
        # A plain store, and the same wrapper before any refusal, keep the old wording.
        assert wrapper.no_credential_reason("deepseek") is None
    finally:
        wrapper.close()


def test_foreground_exec_preflight_accepts_a_credential_this_device_borrows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA round 1, Q3: ``lop exec`` on a borrower refused to start.

    ``_preflight_api_key`` built a plain ``AuthStore`` and saw no row, so a foreground
    exec said "OPENAI_API_KEY is required" on a device the owner lends to, while
    ``exec --background`` and the TUI — which build the session's store — ran. The
    preflight is a presence check and must stay one: no borrow is attempted here, so a
    POST, a dial or a relay is not needed to pass it.
    """
    from local_operator import cli

    root = tmp_path / "borrower"
    root.mkdir()
    _point_config_at(monkeypatch, root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from local_operator.network.identity import mint

    identity = mint(root, name="borrower")
    document = placement_mod.PlacementDocument(NETWORK, root=root, written_by=OWNER)
    document.declare("openai", owner_device=OWNER, provider="openai", by=OWNER)
    document.grant("openai", identity.device_id, scope="session", by=OWNER)
    document.save()
    assert cli._preflight_api_key("openai", root) is None  # noqa: SLF001
    assert capsys.readouterr().err == ""

    # THE CONTROL: the same device with the share revoked is refused as before, so
    # the pass above is the borrow and not a preflight that stopped checking.
    with placement_mod.mutate(NETWORK, root, self_device=OWNER) as current:
        current.revoke("openai", identity.device_id, by=OWNER)
    assert cli._preflight_api_key("openai", root) == 1  # noqa: SLF001
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_a_local_login_still_wins_over_a_borrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOCAL-FIRST, asserted where it matters: the device's own account is kept.

    A wrapper that consulted the mesh first would quietly move a session onto someone
    else's account — the operator's own login is the account they chose here, and
    spending another device's quota without being asked is the whole reason the rung
    is LAST rather than first.
    """
    root = tmp_path / "borrower"
    root.mkdir()
    wrapper, _document, _db = _borrower_wrapper(root, monkeypatch)
    try:
        wrapper.local.upsert_credential(
            "openai",
            {"type": "api_key", "key": LOCAL_KEY, "account_id": "local-account"},
        )
        assert asyncio.run(wrapper.get_api_key("openai", "sess-1")) == LOCAL_KEY
    finally:
        wrapper.close()


def test_a_failed_borrow_is_reported_to_the_owner_and_not_repaired_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 401 on a borrowed bearer becomes a REPORT, never a local block or disable.

    ``rotate_sibling`` is what the failover driver calls, and on a borrowed credential
    it must not reach the local rotation at all: the row it would edit belongs to the
    owner. The return of ``False`` tells the driver there is no sibling HERE, which is
    true and lets it move to another provider.
    """
    root = tmp_path / "borrower"
    root.mkdir()
    wrapper, _document, _db = _borrower_wrapper(root, monkeypatch)
    try:
        wire = wrapper.mesh_client
        assert isinstance(wire, _Wire)
        asyncio.run(wrapper.get_api_key("openai", "sess-1"))

        class _Unauthorized(Exception):
            pass

        monkeypatch.setattr(
            "local_operator.providers.failover.is_invalidated_credential_error",
            lambda _exc: True,
        )
        remaining = wrapper.rotate_sibling("openai", "sess-1", _Unauthorized(), BORROWED_BEARER)
        assert remaining is False
        assert wire.reports == [("openai", "invalid")]
        # NOTHING was disabled here: the borrower has no row for this provider, and the
        # report is the only thing that crossed.
        assert wrapper.local.list_credentials("openai", include_disabled=True) == []
    finally:
        wrapper.close()


def test_the_wrapper_takes_the_logout_signature_its_callers_use(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5: the logout method's signature IS the base store's, keyword included.

    Both real callers (``ProviderController.logout`` and the auth CLI) pass
    ``disabled_cause``, and the first version's narrower signature raised ``TypeError``
    there — on a path no test reached, because every ``ProviderController`` today is
    built over a plain ``AuthStore``. So the protection this method carries ("``/logout``
    on this device must not delete the OWNER's row") was on code nothing called.
    """
    root = root / "logout"
    root.mkdir()
    _point_config_at(monkeypatch, root)
    from local_operator.providers.auth_store import AuthStore

    wrapper = mesh_store.MeshAwareAuthStore(AuthStore(config_dir=root), mesh=None, config_dir=root)
    try:
        wrapper.local.upsert_credential(
            "openai", {"type": "api_key", "key": LOCAL_KEY, "account_id": "local-account"}
        )
        # The keyword is the whole point: this line raised TypeError before the fix.
        assert wrapper.delete_credentials_for_provider("openai", disabled_cause="logged-out") == 1
        assert wrapper.local.list_credentials("openai", include_disabled=True) == []
        # Idempotent on a provider with nothing stored, like the base store.
        assert wrapper.delete_credentials_for_provider("openai", disabled_cause="logged-out") == 0
    finally:
        wrapper.close()


def test_a_logout_drops_the_borrowed_bearer_and_leaves_the_owner_alone(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of F5: what a logout may and may not reach on a borrower.

    A logout must drop this device's cached borrow — the operator asked for it — and
    must NOT touch the owner's row, which is un-shared by ``credential revoke`` on the
    device that holds it.
    """
    import asyncio

    root = root / "borrower-logout"
    root.mkdir()
    wrapper, document, _db_path = _borrower_wrapper(root, monkeypatch)
    try:
        borrowed = asyncio.run(wrapper.get_api_key("openai", "sess-1"))
        wire = wrapper.mesh_client
        assert wire is not None
        # The REAL client caches the grant it is served (``_from_reply`` ->
        # ``GrantCache.put``); this fake hands the grant back without touching the
        # cache, so the line below stands in for that one write and nothing else.
        wire.grants.put("openai", "sess-1", _borrower_grant())
        assert borrowed and wire.grants.get("openai", "sess-1") is not None
        assert wrapper.delete_credentials_for_provider("openai", disabled_cause="logged-out") == 0
        assert wire.grants.get("openai", "sess-1") is None, "the borrowed bearer survived a logout"
        # The OWNER's row — the only copy of the sharing decision this device has — is
        # untouched, and the peer's own document still names the borrow as granted.
        document = placement_mod.PlacementDocument.load(
            document.network_id, root, self_device=wrapper.mesh_client.self_device
        )
        assert document.entry("openai") is not None
        assert document.is_holder("openai", wire.self_device)
    finally:
        wrapper.close()


def test_the_credentials_listing_names_the_owner_and_who_may_borrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The listing is driven by the DOCUMENTS, and it says who owns what.

    Driven by the placement files rather than by the network records on purpose: a
    device that has been shared a credential but whose record has not been re-read
    would otherwise print "nothing is shared" — the one state this listing exists to
    make visible.
    """
    from local_operator.network import cli as network_cli
    from local_operator.network.credentials.state import PlacementState
    from local_operator.network.identity import mint

    root = tmp_path / "borrower"
    root.mkdir()
    _point_config_at(monkeypatch, root)
    identity = mint(root, name="borrower-laptop")
    document = placement_mod.PlacementDocument(NETWORK, root=root, written_by=OWNER)
    document.declare(
        "openai",
        owner_device=OWNER,
        owner_device_name="damian-mbp",
        provider="openai",
        identity_label="damian@example.test",
        by=OWNER,
    )
    document.grant("openai", identity.device_id, scope="session", by=OWNER)
    document.save()
    # ONE instance, then save: two would be two documents and the second would write
    # the empty one.
    state = PlacementState(NETWORK, root=root)
    state.observe("openai", "owner_offline", owner_device=OWNER, retry_after_ms=600_000)
    state.save()

    args = _parser().parse_args(["network", "credentials"])
    assert network_cli.main(args) == 0
    out = capsys.readouterr().out
    assert "owner: damian-mbp (damian@example.test)" in out
    assert "borrowed: owner_offline" in out
    # The owner's own row is not a share, and this device is the reader.
    assert "shared with" not in out.split("owner:")[1].split("\n")[0]


#: ``10**400`` is the shape the first helper missed (review round 4, R4-M1): ``json``
#: decodes a 401-digit number to an ``int``, and ``float()`` of it raised
#: ``OverflowError``. ``2**53 + 1`` is the first integer a float round-trip loses.
_GARBLED = (
    "abc",
    -5,
    10**18,
    10**400,
    "1" + "0" * 400,
    2**53 + 1,
    float("nan"),
    float("inf"),
    True,
    [1],
    {"x": 1},
    None,
)


@pytest.mark.parametrize("value", _GARBLED)
def test_every_number_a_peer_sends_is_validated_at_the_boundary(value: Any) -> None:
    """QA round 2's question, answered for EVERY sibling field, not the one reported.

    A refusal, a grant and a placement document are all parsed from a peer's bytes.
    Each numeric field, garbled, must yield a defined value — never an exception — and
    the value must fail SAFE for what that field controls:

    * a refusal's ``retry_after_ms`` is bounded, so no peer can silence a borrower for
      longer than the borrower's own longest refusal TTL;
    * a grant's expiries read as 0, i.e. already expired, so a garbled grant is never
      cached as a live bearer;
    * a placement row's ``doc_rev`` reads as the floor (1), so a garbled revision can
      never outrank one this device already holds; the epoch reads as 0 for the same
      reason, since the merge takes the max.
    """
    from local_operator.network.credentials.placement import PlacementDocument
    from local_operator.network.credentials.types import (
        MAX_PEER_RETRY_AFTER_MS,
        BrokerError,
        CredentialPlacementEntry,
        Grant,
    )

    error = BrokerError.from_detail({"code": "quota_blocked", "retry_after_ms": value})
    assert 0 <= error.retry_after_ms <= MAX_PEER_RETRY_AFTER_MS
    assert 0 < error.cache_ttl_ms <= MAX_PEER_RETRY_AFTER_MS

    grant = Grant.from_detail(
        {
            "kind": "grant",
            "token_expires_at_ms": value,
            "grant_expires_at_ms": value,
            "latency_ms": value,
            "credential_ref": {"credential_id": value},
            "identity": value,
        }
    )
    # NO EXEMPTION (review round 4, R4-m1): the old test skipped ``10**18`` — exactly
    # the value that was cached as live. Above ``PEER_NUMBER_CEILING`` it is garbage
    # like any other, and a large value BELOW it is capped by the borrower (next test).
    assert grant.grant_expires_at_ms == 0 and grant.token_expires_at_ms == 0
    assert grant.latency_ms >= 0 and grant.credential_ref.credential_id >= 0
    assert isinstance(grant.identity, dict)

    entry = CredentialPlacementEntry.from_json(
        {
            "key": "k",
            "doc_rev": value,
            "declared_at": value,
            "holders": [{"device": "d_x", "granted_at": value}],
        }
    )
    assert entry.doc_rev >= 1 and entry.declared_at >= 0
    assert entry.holders[0].granted_at >= 0

    document = PlacementDocument(network_id="n")
    changed = document.merge(
        {"epoch": value, "credentials": [{"key": "k", "owner_device": "d_peer"}]},
        from_device="d_peer",
    )
    assert changed == ["k"]
    assert document.epoch >= 0


def test_a_numeric_string_from_a_peer_still_reads_as_its_number() -> None:
    """The boundary check refuses garbage, not a well-formed number in string form."""
    from local_operator.network.credentials.types import BrokerError

    assert BrokerError.from_detail({"code": "x", "retry_after_ms": "1500"}).retry_after_ms == 1500
    assert BrokerError.from_detail({"code": "x"}).retry_after_ms == 0


@pytest.mark.parametrize("value", [300_001, 10**18, 10**400, "10000000"])
def test_an_over_cap_retry_is_clamped_to_the_cap_not_dropped_to_the_default(value: Any) -> None:
    """R4-m2: the clamp is the property, and nothing pinned it.

    Turning "over the cap → the cap" into "over the cap → the default" left every test
    green, yet it SHORTENS the backoff: a ``quota_blocked`` refusal asking for more than
    300 s was then cached for the 60 s default, and the borrower re-asked a rate-limited
    owner five times sooner than the cap allows. Exact equality with the cap is the
    only assertion that tells the two apart.
    """
    from local_operator.network.credentials.types import (
        DEFAULT_RETRY_AFTER_MS,
        MAX_PEER_RETRY_AFTER_MS,
        BrokerError,
    )

    assert MAX_PEER_RETRY_AFTER_MS != DEFAULT_RETRY_AFTER_MS, "the test would be vacuous"
    error = BrokerError.from_detail({"code": "quota_blocked", "retry_after_ms": value})
    assert error.retry_after_ms == MAX_PEER_RETRY_AFTER_MS
    assert error.cache_ttl_ms == MAX_PEER_RETRY_AFTER_MS


def test_a_peer_integer_is_never_silently_rounded_to_a_neighbour() -> None:
    """R4-n1: above ``2**53`` a float round-trip turned ``2**53 + 1`` into ``2**53``.

    Every accepted value is at most :data:`PEER_NUMBER_CEILING` (``2**53``), and every
    integer up to it is exact — so an accepted number is always the number sent. A
    value above the ceiling is REFUSED (the default, or the clamp), never rounded to an
    adjacent one: a rounded ``doc_rev`` or epoch would be a plausible wrong number.
    """
    from local_operator.network.credentials.types import PEER_NUMBER_CEILING, peer_int

    assert peer_int(PEER_NUMBER_CEILING) == PEER_NUMBER_CEILING
    assert peer_int(str(PEER_NUMBER_CEILING - 1)) == PEER_NUMBER_CEILING - 1
    assert peer_int(1_700_000_000_123) == 1_700_000_000_123
    assert peer_int(PEER_NUMBER_CEILING + 1) == 0
    assert peer_int(str(PEER_NUMBER_CEILING + 1), default=1) == 1
    assert peer_int(PEER_NUMBER_CEILING + 1, maximum=300_000) == 300_000


def test_a_far_future_grant_is_cached_no_longer_than_grant_ttl(tmp_path: Path) -> None:
    """R4-m1: the borrower enforces "a lent grant is dropped within grant_ttl_s" itself.

    An expiry ten years out is a VALID number (below the peer ceiling), so the boundary
    helper keeps it. Only the owner capped a grant's life until now; an owner clock far
    ahead, or an owner that just said so, got a bearer cached as live for years.
    """
    import time

    from local_operator.network.credentials import client as client_mod
    from local_operator.network.credentials import grant_ttl_s
    from local_operator.network.credentials.types import Grant

    client = client_mod.MeshCredentialClient(
        root=tmp_path, self_device=THIRD, network_id=NETWORK, placement=_document(tmp_path)
    )
    ten_years_ms = int(time.time() * 1000) + 10 * 365 * 86_400_000
    before_ms = int(time.time() * 1000)
    grant = client._from_reply(  # noqa: SLF001 — the leg-1 reply path the cache is fed from
        {
            "op": "ack",
            "detail": {
                "kind": "grant",
                "access_token": "-".join(("synthetic", "bearer")),
                "grant_expires_at_ms": ten_years_ms,
                "token_expires_at_ms": ten_years_ms,
            },
        },
        "deepseek",
        "deepseek",
        "sess-far",
    )
    ceiling_ms = int(time.time() * 1000) + int(grant_ttl_s(tmp_path) * 1000)
    assert isinstance(grant, Grant), grant
    cached = client.grants.get("deepseek", "sess-far")
    assert cached is grant
    assert before_ms < grant.grant_expires_at_ms <= ceiling_ms


def test_a_member_whose_document_cannot_be_read_is_named_and_the_rest_still_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-M1, the pull half: one paired member's bytes must not stop everyone else's.

    A 401-digit ``doc_rev`` raised ``OverflowError`` in the merge, and the loop caught
    only ``(OSError, MeshRefusal)`` — so every member after that one went unmerged, for
    every borrower, from one field any invited device could send. The helper is now
    total; THIS pins the loop's own guarantee against the next shape nobody foresaw:
    the member is refused BY NAME and the others merge.
    """
    from local_operator.network.credentials import client as client_mod

    merged: list[str] = []

    def _merge(
        network_id: str, document: dict[str, Any], *, from_device: str, **_: Any
    ) -> list[str]:
        if from_device == OWNER:
            raise OverflowError("int too large to convert to float")
        merged.append(from_device)
        return ["deepseek"]

    client = client_mod.MeshCredentialClient(
        root=tmp_path, self_device=THIRD, network_id=NETWORK, placement=_document(tmp_path)
    )
    monkeypatch.setattr(client_mod, "merge_from_peer", _merge)
    monkeypatch.setattr(client, "_other_active_members", lambda: [OWNER, OTHER])
    monkeypatch.setattr(
        client,
        "_ask_owner_directly",
        lambda device, frame: {"op": "ack", "detail": {"kind": "placement", "document": {}}},
    )
    result = client.pull_placement()
    assert merged == [OTHER], "the unreadable document stopped the loop"
    assert result["changed"] == ["deepseek"] and result["owners"] == 2
    assert result["skipped"] == [{"device": OWNER, "reason": "malformed_document"}]


def test_a_real_merge_of_a_401_digit_revision_reads_it_as_the_floor(tmp_path: Path) -> None:
    """R4-M1 without a stub: the REAL merge, fed the document a peer could send."""
    import json

    from local_operator.network.credentials import placement as placement_mod

    wire = json.loads(
        '{"epoch": 1'
        + "0" * 400
        + ', "credentials": [{"key": "deepseek", "owner_device": "'
        + OWNER
        + '", "doc_rev": 1'
        + "0" * 400
        + ', "holders": [{"device": "'
        + THIRD
        + '"}]}]}'
    )
    assert isinstance(wire["credentials"][0]["doc_rev"], int)
    changed = placement_mod.merge_from_peer(
        NETWORK, wire, from_device=OWNER, self_device=THIRD, root=tmp_path
    )
    assert changed == ["deepseek"]
    merged = placement_mod.PlacementDocument.load(NETWORK, tmp_path).entries["deepseek"]
    assert merged.doc_rev == 1
