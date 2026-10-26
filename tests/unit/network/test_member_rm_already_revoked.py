"""``lop network member rm <already-removed id>``: answered, and repaired.

THE DEFECT. ``network show`` lists a removed member (``active:false``,
``lifecycle:expired``) with its id, but ``member rm`` resolved its argument
through the ACTIVE-ONLY device resolver, so that exact id answered
``unknown_member``. Meanwhile this device's credential metadata could still name
the removed device as a holder — the removal prune runs only where the removal
was made, so a removal applied from a peer's rotation frame (or a prune that
failed) leaves the references behind, and nothing else ever removes them
(``store.prune`` has no production caller).

WHAT IS PINNED: the exact removed id answers ``already_revoked`` and prunes the
stale references with NO second epoch rotation; a repeat is idempotent; the
tombstone and the burned id survive; a genuinely unknown id still refuses
``unknown_member``; an ACTIVE member that shares the removed one's display name
is untouched (ids are the identity, names are not); and a repair that cannot
write is reported as a failure rather than a success.

Synthetic ids and a per-test config root only — never a live network.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import relay, store, types, wire
from local_operator.network.credentials import mcp_state
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.messages import COPY_CEILING_LINES

NETWORK = "n_" + "1" * 32
SELF = "d_" + "a" * 32
GONE = "d_" + "b" * 32
LIVE = "d_" + "c" * 32
#: A THIRD device that owns a credential the removed device was a holder of.
PEER = "d_" + "f" * 32
NEVER = "d_" + "e" * 32


def _member(device_id: str, name: str, role: str = "drive") -> types.MemberRecord:
    return types.MemberRecord(
        device_id=device_id,
        public_key=wire.b64u(device_id[-1].encode() * 32),
        name=name,
        role=role,
        capabilities=sorted(types.capabilities_for_role(role)),
        added_via="self" if device_id == SELF else "invite",
    )


def _network(root: Path, *, live_same_name: bool = False) -> None:
    """A network whose member ``GONE`` is removed but still referenced locally.

    The removal goes through the real ``remove_member`` (tombstone, burned id,
    rotation). The stale references are then written the way a removal applied
    from a PEER's rotation frame leaves them: that path tombstones the row but
    never runs this device's placement prune.
    """
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        epoch=1,
        self_device_id=SELF,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    record.members.append(_member(SELF, "desk", role="admin"))
    record.members.append(_member(GONE, "laptop"))
    if live_same_name:
        record.members.append(_member(LIVE, "laptop"))
    state = types.SecretState(network_id=NETWORK, epoch=1, secret=wire.b64u(b"s" * 32))
    store.save(record, root)
    store.save_secrets(state, root)
    relay.remove_member(record, state, device_id=GONE, by=SELF, root=root)
    store.save(record, root)
    store.save_secrets(state, root)

    with placement_mod.mutate(NETWORK, root, self_device=SELF) as document:
        document.declare("openai", owner_device=SELF, provider="openai", by=SELF)
        document.grant("openai", GONE, scope="device", by=SELF)
        if live_same_name:
            document.grant("openai", LIVE, scope="device", by=SELF)
    mcp_state.record(
        NETWORK, "mcp:notion", state="synced", owner_device=GONE, owner_name="laptop", root=root
    )
    if live_same_name:
        mcp_state.record(
            NETWORK, "mcp:linear", state="synced", owner_device=LIVE, owner_name="laptop", root=root
        )


def _rm(device: str, *, json_out: bool = True) -> Namespace:
    return Namespace(
        network_command="member",
        member_command="rm",
        network="home-net",
        device=device,
        capabilities=[],
        json=json_out,
    )


def _references(root: Path, device: str) -> list[str]:
    """Keys whose on-disk metadata names ``device``, read straight off the two files.

    Independent of the product's own reader on purpose, so these tests discriminate
    on BEHAVIOUR against a tree that does not have that reader at all.
    """
    document = placement_mod.PlacementDocument.load(NETWORK, root)
    keys = {
        key
        for key, entry in document.entries.items()
        if entry.owner_device == device or entry.is_holder(device)
    }
    keys |= {
        key
        for key, row in mcp_state.load(NETWORK, root).items()
        if row.get("owner_device") == device
    }
    return sorted(keys)


@pytest.fixture()
def config(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


def test_an_already_removed_id_is_reported_and_its_references_pruned(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _network(config)
    epoch_before = store.load(NETWORK, config).epoch
    assert _references(config, GONE) == ["mcp:notion", "openai"]

    assert net_cli.main(_rm(GONE)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "already_revoked", payload
    assert payload["removed"] == GONE and payload["rotated"] is False
    assert payload["repaired"] == ["mcp:notion", "openai"]
    assert _references(config, GONE) == []

    after = store.load(NETWORK, config)
    # NO second rotation, and the history the burned id enforces is intact.
    assert after.epoch == epoch_before == payload["epoch"]
    assert store.load_secrets(NETWORK, config).epoch == epoch_before
    assert GONE in after.removed_ids
    tombstone = after.member(GONE)
    assert tombstone is not None and not tombstone.active and tombstone.lifecycle == "expired"
    # The owner's own holder row is not collateral damage.
    document = placement_mod.PlacementDocument.load(NETWORK, config)
    assert document.is_holder("openai", SELF)


def test_the_human_receipt_says_already_removed_and_keeps_the_copy_ceiling(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No erasure claim: a copy already on the removed device is out of reach."""
    _network(config)
    assert net_cli.main(_rm(GONE, json_out=False)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("laptop (d_bbbbbb) was already removed from home-net")
    assert "not rotated again" in lines[0]
    assert lines[1] == "removed 2 stale credential reference(s) to it: mcp:notion, openai"
    assert tuple(lines[2:]) == COPY_CEILING_LINES


def test_a_repeat_is_idempotent(config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _network(config)
    assert net_cli.main(_rm(GONE)) == 0
    first = json.loads(capsys.readouterr().out)
    document_before = placement_mod.placement_path(NETWORK, config).read_bytes()

    assert net_cli.main(_rm(GONE)) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["code"] == "already_revoked" and second["repaired"] == []
    assert second["epoch"] == first["epoch"]
    # Nothing left to prune means nothing rewritten.
    assert placement_mod.placement_path(NETWORK, config).read_bytes() == document_before
    assert GONE in store.load(NETWORK, config).removed_ids


def test_a_burned_id_whose_tombstone_row_is_gone_is_still_already_revoked(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``removed_ids`` outlives the tombstone row by design; it alone proves removal."""
    _network(config)
    with store.mutate(NETWORK, config) as record:
        record.members = [row for row in record.members if row.device_id != GONE]
        store.save(record, config)  # ``mutate`` holds the lock; the body saves
    assert store.load(NETWORK, config).member(GONE) is None
    assert net_cli.main(_rm(GONE)) == 0
    assert json.loads(capsys.readouterr().out)["code"] == "already_revoked"
    assert _references(config, GONE) == []
    assert GONE in store.load(NETWORK, config).removed_ids

    # R3: with no row there is no name, and the receipt must not print the id twice.
    assert net_cli.main(_rm(GONE, json_out=False)) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith(f"{GONE[:8]} was already removed from home-net"), first
    assert GONE not in first, first


def test_an_active_member_with_the_same_name_is_untouched(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _network(config, live_same_name=True)
    assert net_cli.main(_rm(GONE)) == 0
    assert json.loads(capsys.readouterr().out)["code"] == "already_revoked"

    after = store.load(NETWORK, config)
    live = after.member(LIVE)
    assert live is not None and live.active and live.name == "laptop"
    assert LIVE not in after.removed_ids
    assert _references(config, LIVE) == ["mcp:linear", "openai"]
    assert _references(config, GONE) == []


def test_a_genuinely_unknown_id_still_refuses_and_repairs_nothing(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A preservation guard (passes before and after): the new branch is exact-id only."""
    _network(config)
    assert net_cli.main(_rm(NEVER)) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["code"] == "unknown_member"
    assert "is not a device in home-net" in captured.err
    # A TAIL of the removed id is not the identity either.
    assert net_cli.main(_rm(GONE[-8:])) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "unknown_member"
    assert _references(config, GONE) == ["mcp:notion", "openai"]


def test_a_repair_that_cannot_write_is_reported_as_a_failure(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _network(config)

    def refuse(*_args: object, **_kwargs: object) -> list[str]:
        raise PermissionError("read-only placement document")

    monkeypatch.setattr(placement_mod, "revoke_device_everywhere", refuse)
    assert net_cli.main(_rm(GONE)) == 1
    captured = capsys.readouterr()
    body = json.loads(captured.out)
    assert body["code"] == "already_revoked_repair_failed", body
    assert "PermissionError" in body["message"] and "openai" in body["message"]
    assert "re-run" in body["message"]
    assert "openai" in _references(config, GONE)


# ---------------------------------------------------------------------------
# #2991 round 1 remediation: R1, R2, Q-1
# ---------------------------------------------------------------------------


def _peer_document(rev_after: int, *, holders: tuple[str, ...]) -> dict[str, Any]:
    """``PEER``'s own document for ``peer:gh``, at an exact ``doc_rev``, as it sends it."""
    document = placement_mod.PlacementDocument(NETWORK, written_by=PEER)
    entry = document.declare("anthropic", owner_device=PEER, provider="anthropic", by=PEER)
    for holder in holders:
        document.grant("anthropic", holder, scope="device", by=PEER)
    entry.doc_rev = rev_after
    return document.to_json()


def _merge(config: Path, payload: dict[str, Any]) -> list[str]:
    return placement_mod.merge_from_peer(
        NETWORK, payload, from_device=PEER, self_device=SELF, root=config
    )


def test_a_row_another_device_owns_is_reported_not_rewritten(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R1: the owner's revisions must keep landing through the real ``merge_from_peer``.

    The removal reached this device through a rotation frame, so PEER — which owns
    ``anthropic`` and lent it to the removed device and to this one — has not revised
    its row yet. Rewriting that row here bumped its
    ``doc_rev`` on a copy PEER never saw: PEER's next revision (revoking SELF) then
    arrived at an EQUAL rev and was dropped as a replay, and the one after brought the
    removed holder straight back.
    """
    _network(config)
    assert _merge(config, _peer_document(5, holders=(GONE, SELF))) == ["anthropic"]
    before = placement_mod.PlacementDocument.load(NETWORK, config).entry("anthropic")
    assert before is not None and before.doc_rev == 5

    assert net_cli.main(_rm(GONE)) == 0
    payload = json.loads(capsys.readouterr().out)
    # Reported, attributed to its owner, and NOT counted as repaired.
    assert "anthropic" not in payload["repaired"], payload
    assert payload["owner_must_revoke"] == [
        {"key": "anthropic", "owner_device": PEER, "owner_name": ""}
    ], payload
    after = placement_mod.PlacementDocument.load(NETWORK, config).entry("anthropic")
    assert after is not None and after.doc_rev == 5 and after.is_holder(GONE)

    # The owner's real next revision (it revokes SELF and keeps GONE for now) lands...
    assert _merge(config, _peer_document(6, holders=(GONE,))) == ["anthropic"]
    landed = placement_mod.PlacementDocument.load(NETWORK, config).entry("anthropic")
    assert landed is not None and landed.doc_rev == 6 and not landed.is_holder(SELF)
    # ...and so does the one where the owner itself prunes the removed holder.
    assert _merge(config, _peer_document(7, holders=())) == ["anthropic"]
    pruned = placement_mod.PlacementDocument.load(NETWORK, config).entry("anthropic")
    assert pruned is not None and pruned.doc_rev == 7 and not pruned.is_holder(GONE)


def test_the_human_receipt_names_the_owner_that_must_revoke(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _network(config)
    _merge(config, _peer_document(5, holders=(GONE,)))
    assert net_cli.main(_rm(GONE, json_out=False)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "removed 2 stale credential reference(s) to it: mcp:notion, openai"
    assert lines[2].startswith(f"anthropic is still listed by {PEER[:8]}, which owns it")
    assert tuple(lines[3:]) == COPY_CEILING_LINES


def test_this_devices_own_id_is_refused_and_its_placement_kept(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R2: a device a peer removed carries its OWN id in ``removed_ids``.

    The prune deletes every row its target owns, so ``member rm <own id>`` used to
    empty this device's whole placement document and answer success.
    """
    _network(config)
    with store.mutate(NETWORK, config) as record:
        # What ``apply_epoch``'s ``removed_by_this_rotation`` arm installs: the
        # rotator's table, with this device's own row tombstoned and its id burned.
        own = record.member(SELF)
        assert own is not None
        own.removed_at, own.lifecycle = 1.0, "expired"
        record.removed_ids = sorted({*record.removed_ids, SELF})
        # No rotation lock in the way: the refusal under test is the self-id one.
        record.rotation_lock_until = 0.0
        store.save(record, config)
    assert net_cli.main(_rm(SELF)) == 1
    body = json.loads(capsys.readouterr().out)
    assert body["code"] == "cannot_remove_self", body
    assert "lop network disconnect" in body["message"]
    assert sorted(placement_mod.PlacementDocument.load(NETWORK, config).entries) == ["openai"]


def test_a_removal_refused_by_the_rotation_lock_prunes_nothing(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-1 (pre-existing): the lock refusal must come before the credential prune.

    ``remove_member`` pruned the target's placement and MCP evidence, THEN
    ``rotate_epoch`` refused ``rotation_in_progress`` — leaving an ACTIVE member with
    its shares and evidence deleted. Driven through the real CLI's local path.
    """
    _network(config, live_same_name=True)
    with store.mutate(NETWORK, config) as record:
        record.rotation_lock_until = 4_000_000_000.0
        store.save(record, config)
    before = _references(config, LIVE)
    assert before == ["mcp:linear", "openai"]

    assert net_cli.main(_rm(LIVE)) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "rotation_in_progress"
    live = store.load(NETWORK, config).member(LIVE)
    assert live is not None and live.active
    assert _references(config, LIVE) == before


def test_the_evidence_update_reads_inside_the_lock(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4: the read that feeds an evidence write must happen while the lock is held.

    The writers used to ``load`` first and take the lock only to write, so a relay
    ``record`` that read the file, then waited while ``forget_device`` wrote, put
    its stale copy — the forgotten device's row included — straight back. Each
    load is stamped with whether the lock was held at that moment; the LAST read a
    writer makes before writing is the one its write is built from.
    """
    from contextlib import contextmanager
    from typing import Iterator

    _network(config)
    held = [0]
    events: list[tuple[str, int]] = []
    real_lock = mcp_state._cross_process_lock  # noqa: SLF001 — the exact lock under test
    real_load = mcp_state.load
    real_write = store._write_private_json  # noqa: SLF001

    @contextmanager
    def tracking_lock(*args: Any, **kwargs: Any) -> Iterator[None]:
        with real_lock(*args, **kwargs):
            held[0] += 1
            try:
                yield
            finally:
                held[0] -= 1

    def tracking_load(*args: Any, **kwargs: Any) -> Any:
        events.append(("load", held[0]))
        return real_load(*args, **kwargs)

    def tracking_write(*args: Any, **kwargs: Any) -> Any:
        events.append(("write", held[0]))
        return real_write(*args, **kwargs)

    monkeypatch.setattr(mcp_state, "_cross_process_lock", tracking_lock)
    monkeypatch.setattr(mcp_state, "load", tracking_load)
    monkeypatch.setattr(store, "_write_private_json", tracking_write)

    for write in (
        lambda: mcp_state.forget_device(NETWORK, GONE, root=config),
        lambda: mcp_state.record(NETWORK, "mcp:x", state="synced", owner_device=LIVE, root=config),
        lambda: mcp_state.clear(NETWORK, "mcp:x", root=config),
    ):
        events.clear()
        write()
        assert events[-1] == ("write", 1), events
        feeding = [event for event in events if event[0] == "load"][-1]
        assert feeding == ("load", 1), events
    assert "mcp:notion" not in real_load(NETWORK, config)
