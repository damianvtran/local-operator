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
    assert net_cli.main(_rm(GONE)) == 0
    assert json.loads(capsys.readouterr().out)["code"] == "already_revoked"
    assert _references(config, GONE) == []
    assert GONE in store.load(NETWORK, config).removed_ids


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
