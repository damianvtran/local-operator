"""The operator's failover routing over the mesh (`network/routing_sync.py`).

WHY THESE CELLS EXIST: ``retry.fallbackChains`` and the opt-ins that shape the walk
reached no mesh node, so a node's effective chain was primary-only and a borrowed
rate limit ended the turn there while the same turn on the head walked on. The
definitions sync now carries the head's routing as per-unit mirrors. Each cell pins
one claim:

* a node with no routing (or only the shipped defaults written out) gets the head's
  chain and opt-ins, and a second push sends nothing;
* a node's own explicit value wins, is reported, and is not re-offered every tick;
  clearing it lets the head's value in on the next push;
* a chain the head deletes is removed from an untouched mirror; an edited one stays;
* an older peer (state reply without the key) is never sent it, a mirror is never
  re-sent as the node's own, and a credential-shaped value never travels;
* readiness names the effective chain and whose it is.

The "peer" is a second config root behind a fake link that routes the two phases to
the real ``definition_state`` / ``apply_bundle`` (the bodies ``make_handler`` serves),
as in ``test_definitions_operator_instructions``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from local_operator.config import read_config_values
from local_operator.network import definitions, mirror_authority, routing_sync

HEAD_DEVICE = "d_" + "a" * 32
NODE_DEVICE = "d_" + "b" * 32
OTHER_DEVICE = "d_" + "c" * 32

#: The operator's Mac chain, as the brief recorded it.
HEAD_CHAIN = [
    "anthropic/claude-opus-5-5",
    "openai/gpt-6.1-sol",
    "deepseek/deepseek-flash",
    "auto",
]


#: The synthetic operator anchor every root here is enrolled under.
OP_KEY = "k_" + "0" * 16
#: Roots (by directory name) that hold the anchor's PRIVATE half: the operator's
#: device. Every other root is a node holding the anchor verify-only, except
#: ``FOREIGN_ROOTS`` (another operator's anchor) and ``UNANCHORED_ROOTS`` (none).
SIGNING_ROOTS = {"head"}
FOREIGN_ROOTS = {"foreign"}
UNANCHORED_ROOTS = {"unanchored"}


@pytest.fixture(autouse=True)
def _operator_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve each config root's operator authority from its name (M1).

    The real probe reads the HOST's anchor, which a test running head and nodes in
    one process cannot tell apart (and must never read off this machine).
    """

    def _resolve(root: Path) -> mirror_authority.Authority:
        name = Path(root).name
        if name in UNANCHORED_ROOTS:
            return mirror_authority.Authority()
        key = "k_" + "f" * 16 if name in FOREIGN_ROOTS else OP_KEY
        return mirror_authority.Authority(key_id=key, signs=name in SIGNING_ROOTS)

    monkeypatch.setattr(mirror_authority, "_resolve", _resolve)
    mirror_authority.reset_cache()


def _root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_retry(root: Path, retry: dict[str, Any], **extra: Any) -> None:
    values = {"retry": retry, **extra}
    (root / "config.yml").write_text(yaml.safe_dump({"values": values}), encoding="utf-8")


def _retry(root: Path) -> dict[str, Any]:
    values = read_config_values(root) or {}
    return dict(values.get("retry") or {})


class _PeerLink:
    def __init__(self, peer_root: Path, *, legacy: bool = False, origin: str = HEAD_DEVICE) -> None:
        self.peer_root = peer_root
        self.legacy = legacy
        self.origin = origin
        self.frames: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], timeout: float = 0) -> Any:
        self.frames.append(frame)
        if frame["phase"] == "state":
            detail = definitions.definition_state(self.peer_root, names=frame.get("names"))
            if self.legacy:
                detail.pop(routing_sync.ROUTING_KIND)
            return {"op": "ok", "detail": detail}
        summary = definitions.apply_bundle(
            self.peer_root, frame["bundle"], origin_device=self.origin
        )
        return {"op": "ok", "detail": summary}

    def applies(self) -> list[dict[str, Any]]:
        return [frame for frame in self.frames if frame["phase"] == "apply"]


class _Server:
    def __init__(self, root: Path, link: _PeerLink) -> None:
        self.root = root
        self._link = link

    def _ensure_link(self, device_id: str) -> Any:
        return self._link

    def _next_relay_req(self) -> int:
        return 1

    def _own_label(self) -> str:
        return "head"


def _push(head: Path, link: _PeerLink) -> dict[str, Any]:
    return definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]


def _head(tmp_path: Path, monkeypatch: Any) -> Path:
    head = _root(tmp_path, "head")
    _write_retry(
        head,
        {
            "fallbackChains": {"default": list(HEAD_CHAIN)},
            "usageAwareFallback": True,
            "usageReservePercent": 15,
            "maxRetries": 3,  # device-local transport tuning: never mirrored
        },
    )
    # ``local_bundle`` names this device by its identity; pin it so the node's
    # recorded origin is a known id rather than a freshly minted one.
    from local_operator.network import identity as identity_mod

    real = identity_mod.load_or_mint

    def _load(root: Path, *a: Any, **k: Any) -> Any:
        if Path(root) == head:
            return SimpleNamespace(device_id=HEAD_DEVICE)
        return real(root, *a, **k)

    monkeypatch.setattr(identity_mod, "load_or_mint", _load)
    return head


def test_a_bare_node_gets_the_heads_chain_and_opt_ins(tmp_path: Path, monkeypatch: Any) -> None:
    head = _head(tmp_path, monkeypatch)
    node = _root(tmp_path, "node")
    # A generated config with the shipped defaults written out is still "no
    # routing of its own": a default carries no intent to protect.
    _write_retry(node, {"fallbackChains": {}, "usageAwareFallback": False, "maxRetries": 10})
    link = _PeerLink(node)

    result = _push(head, link)

    assert result["ok"] is True, result
    assert result[routing_sync.ROUTING_KIND]["outcome"] == "installed", result
    retry = _retry(node)
    assert retry["fallbackChains"] == {"default": HEAD_CHAIN}, retry
    assert retry["usageAwareFallback"] is True
    assert float(retry["usageReservePercent"]) == 15.0
    assert retry["maxRetries"] == 10, "transport tuning must stay the node's own"
    state = routing_sync.routing_state(node)
    assert state["units"]["fallbackChains.default"] == {"source": "mirror", "origin": HEAD_DEVICE}

    # The runtime reads it: the node's effective chain is the operator's.
    from local_operator.providers.failover import RetrySettings, resolve_chain

    settings = RetrySettings.from_settings(read_config_values(node))
    assert resolve_chain("anthropic/claude-opus-5-5", settings.fallback_chains) == HEAD_CHAIN
    assert settings.usage_aware_fallback is True

    # Idempotent: the node settled this revision, so nothing is applied again.
    link.frames.clear()
    again = _push(head, link)
    assert again["code"] == "in_sync", again
    assert again[routing_sync.ROUTING_KIND] == {"outcome": "in_sync"}
    assert link.applies() == []


def test_a_nodes_own_value_wins_is_reported_and_not_re_offered(
    tmp_path: Path, monkeypatch: Any
) -> None:
    head = _head(tmp_path, monkeypatch)
    node = _root(tmp_path, "node")
    own_chain = ["openai/gpt-6.1-sol"]
    _write_retry(node, {"fallbackChains": {"default": own_chain}})
    link = _PeerLink(node)

    result = _push(head, link)

    assert result["ok"] is True, result  # a kept unit never fails the push
    report = result[routing_sync.ROUTING_KIND]
    assert [entry["unit"] for entry in report["kept"]] == ["fallbackChains.default"], report
    retry = _retry(node)
    assert retry["fallbackChains"] == {"default": own_chain}
    # The other units still landed: per-unit ownership, not all-or-nothing.
    assert retry["usageAwareFallback"] is True
    assert routing_sync.routing_state(node)["units"]["fallbackChains.default"]["source"] == "local"

    link.frames.clear()
    assert _push(head, link)[routing_sync.ROUTING_KIND] == {"outcome": "in_sync"}
    assert link.applies() == []

    # The node's operator clears their chain: the head's lands on the next push.
    _write_retry(node, {**_retry(node), "fallbackChains": {}})
    cleared = _push(head, link)
    assert "fallbackChains.default" in cleared[routing_sync.ROUTING_KIND]["installed"], cleared
    assert _retry(node)["fallbackChains"] == {"default": HEAD_CHAIN}


def test_a_chain_deleted_on_the_head_is_removed_from_an_untouched_mirror(
    tmp_path: Path, monkeypatch: Any
) -> None:
    head = _head(tmp_path, monkeypatch)
    _write_retry(
        head,
        {
            "fallbackChains": {
                "default": list(HEAD_CHAIN),
                "openai/*": ["deepseek/deepseek-flash"],
            },
            "usageAwareFallback": True,
        },
    )
    node = _root(tmp_path, "node")
    link = _PeerLink(node)
    assert _push(head, link)["ok"] is True
    assert set(_retry(node)["fallbackChains"]) == {"default", "openai/*"}

    # The operator deletes one chain and turns the opt-in back off on the head.
    _write_retry(head, {"fallbackChains": {"default": list(HEAD_CHAIN)}})
    result = _push(head, link)

    report = result[routing_sync.ROUTING_KIND]
    assert set(report["removed"]) == {"fallbackChains.openai/*", "usageAwareFallback"}, report
    retry = _retry(node)
    assert retry["fallbackChains"] == {"default": HEAD_CHAIN}
    assert retry.get("usageAwareFallback", False) is False


def test_a_mirror_edited_on_the_node_survives_the_heads_deletion(
    tmp_path: Path, monkeypatch: Any
) -> None:
    head = _head(tmp_path, monkeypatch)
    node = _root(tmp_path, "node")
    link = _PeerLink(node)
    assert _push(head, link)["ok"] is True
    edited = ["deepseek/deepseek-flash"]
    _write_retry(node, {**_retry(node), "fallbackChains": {"default": edited}})

    _write_retry(head, {"usageAwareFallback": True, "usageReservePercent": 15})
    result = _push(head, link)

    assert "fallbackChains.default" not in (result[routing_sync.ROUTING_KIND].get("removed") or [])
    assert _retry(node)["fallbackChains"] == {"default": edited}
    assert routing_sync.routing_state(node)["units"]["fallbackChains.default"]["source"] == "local"


def test_a_mirror_is_never_re_sent_as_the_nodes_own(tmp_path: Path, monkeypatch: Any) -> None:
    head = _head(tmp_path, monkeypatch)
    node = _root(tmp_path, "node")
    assert _push(head, _PeerLink(node))["ok"] is True
    row = routing_sync.local_row(node)
    assert row is not None and row["units"] == {}, row


def test_an_older_peer_is_never_sent_routing(tmp_path: Path, monkeypatch: Any) -> None:
    head = _head(tmp_path, monkeypatch)
    node = _root(tmp_path, "node")
    link = _PeerLink(node, legacy=True)

    result = _push(head, link)

    assert result[routing_sync.ROUTING_KIND] == {"outcome": "unsupported"}
    assert all(routing_sync.ROUTING_KIND not in f["bundle"] for f in link.applies())
    assert read_config_values(node) == {}


def test_credential_shaped_routing_is_withheld_and_refused(
    tmp_path: Path, monkeypatch: Any
) -> None:
    head = _head(tmp_path, monkeypatch)
    secretish = "sk-ant-api03-" + "A" * 80
    _write_retry(head, {"fallbackChains": {"default": [f"anthropic/{secretish}"]}})
    node = _root(tmp_path, "node")

    result = _push(head, _PeerLink(node))
    assert result[routing_sync.ROUTING_KIND]["outcome"] == "withheld", result

    index: dict[str, Any] = {}
    refused = routing_sync.apply_routing_locked(
        node,
        {"units": {"fallbackChains.default": [f"anthropic/{secretish}"]}},
        HEAD_DEVICE,
        index,
    )
    assert refused["outcome"] == "refused", refused
    assert (
        secretish not in (node / "config.yml").read_text(encoding="utf-8")
        if (node / "config.yml").exists()
        else True
    )


def test_a_mirror_from_another_device_is_replaced_by_the_authoritys(tmp_path: Path) -> None:
    """M1 / R3 tail: a mirror some other origin left is not the operator's intent."""
    node = _root(tmp_path, "node")
    index: dict[str, Any] = {}
    stray = ["deepseek/deepseek-flash"]
    first = routing_sync.apply_routing_locked(
        node, {"authority": OP_KEY, "units": {"fallbackChains.default": stray}}, OTHER_DEVICE, index
    )
    assert first["outcome"] == "installed", first
    second = routing_sync.apply_routing_locked(
        node,
        {"authority": OP_KEY, "units": {"fallbackChains.default": HEAD_CHAIN}},
        HEAD_DEVICE,
        index,
    )
    assert second["updated"] == ["fallbackChains.default"], second
    assert not second["kept"], second
    assert _retry(node)["fallbackChains"] == {"default": HEAD_CHAIN}
    definitions.write_index(node, index)
    unit = routing_sync.routing_state(node)["units"]["fallbackChains.default"]
    assert unit == {"source": "mirror", "origin": HEAD_DEVICE}, unit


def test_an_unknown_or_unusable_unit_is_refused_whole(tmp_path: Path) -> None:
    node = _root(tmp_path, "node")
    for units in (
        {"maxRetries": 1},
        {"usageReservePercent": 400},
        {"fallbackChains.default": "not-a-list"},
    ):
        result = routing_sync.apply_routing_locked(
            node, {"authority": OP_KEY, "units": units}, HEAD_DEVICE, {}
        )
        assert result["outcome"] == "refused", (units, result)
    assert read_config_values(node) == {}


def test_readiness_names_the_effective_chain_and_whose_it_is(
    tmp_path: Path, monkeypatch: Any
) -> None:
    head = _head(tmp_path, monkeypatch)
    node = _root(tmp_path, "node")
    _write_retry(node, {}, hosting="anthropic", model_name="claude-opus-5-5")
    assert _push(head, _PeerLink(node))["ok"] is True

    fact = routing_sync.failover_fact(node, hosting="anthropic", model_name="claude-opus-5-5")
    assert fact is not None
    assert fact["hops"] == HEAD_CHAIN and fact["source"] == "mirror", fact
    assert fact["origin"] == HEAD_DEVICE
    sentence = routing_sync.failover_sentence(fact, origin_label="mac")
    assert (
        sentence == "Fails over to: " + " → ".join(HEAD_CHAIN) + " (mirrored from mac)."
    ), sentence

    bare = _root(tmp_path, "bare")
    empty = routing_sync.failover_fact(bare, hosting="anthropic", model_name="claude-opus-5-5")
    assert empty is not None and empty["hops"] == []
    assert "ends the turn" in routing_sync.failover_sentence(empty)

    from local_operator.network import readiness

    facts = readiness.collect_peer_facts(node, home=tmp_path)
    assert facts["failover"]["hops"] == HEAD_CHAIN, facts["failover"]


def test_a_device_with_no_routing_of_its_own_sends_no_apply(tmp_path: Path) -> None:
    """The common case — no chain anywhere — costs the peer no apply frame."""
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    link = _PeerLink(node)

    result = _push(head, link)

    assert result["code"] == "in_sync", result
    assert result[routing_sync.ROUTING_KIND] == {"outcome": "none"}
    assert link.applies() == []


def test_the_model_credential_row_names_the_chain_only_when_there_is_one() -> None:
    from types import SimpleNamespace as NS

    from local_operator.network import readiness

    server = NS(identity=NS(device_id=HEAD_DEVICE), _member_name=lambda _d: "node")
    bare_row = {"detail": "node is served by its own login"}
    readiness._attach_failover(  # noqa: SLF001
        bare_row, {"failover": {"hops": [], "model_fallback": True}}, server=server
    )
    assert bare_row["detail"] == "node is served by its own login"
    assert bare_row["observed"]["failover"]["hops"] == []

    row = {"detail": "node is served by its own login"}
    fact = {"hops": HEAD_CHAIN, "source": "mirror", "origin": OTHER_DEVICE, "model_fallback": True}
    readiness._attach_failover(row, {"failover": fact}, server=server)  # noqa: SLF001
    assert row["detail"].endswith(" (mirrored from node)."), row
    assert "Fails over to: " + " → ".join(HEAD_CHAIN) in row["detail"]

    legacy = {"detail": "unchanged"}
    readiness._attach_failover(legacy, {}, server=server)  # noqa: SLF001
    assert legacy == {"detail": "unchanged"}


# ---------------------------------------------------------------------------
# Review round 1 regressions (PR #3905): M1 direction, M2 lost mirror, m1, m2, Q-1.
# ---------------------------------------------------------------------------


def _push_from(sender: Path, receiver: Path, origin: str) -> tuple[dict[str, Any], _PeerLink]:
    link = _PeerLink(receiver, origin=origin)
    result = definitions.push_to_peer(_Server(sender, link), NODE_DEVICE)  # type: ignore[arg-type]
    return result, link


def test_r2_a_nodes_own_routing_never_reaches_the_head(tmp_path: Path, monkeypatch: Any) -> None:
    """M1 / R2: the node's override stays on the node, by both ends' rule."""
    head = _head(tmp_path, monkeypatch)
    before = _retry(head)
    node = _root(tmp_path, "node")
    _write_retry(
        node,
        {"usageReservePercent": 40, "fallbackChains": {"[redacted]/*": ["openai/gpt-6.1-sol"]}},
    )

    # Sender side: the node does not sign for the operator, so it sends nothing.
    result, link = _push_from(node, head, NODE_DEVICE)
    assert result[routing_sync.ROUTING_KIND]["outcome"] == "not_authority", result
    assert all(routing_sync.ROUTING_KIND not in f["bundle"] for f in link.applies())

    # Receiver side: a forged row claiming the operator's anchor is refused by the
    # head, which signs for that anchor itself.
    row = routing_sync.local_row(node)
    assert row is not None and row["units"], row
    forged = routing_sync.apply_routing_locked(head, {**row, "authority": OP_KEY}, NODE_DEVICE, {})
    assert forged["outcome"] == "refused", forged
    assert _retry(head) == before


def test_r3_a_node_never_authors_for_a_third_device(tmp_path: Path, monkeypatch: Any) -> None:
    """M1 / R3: node2's chain does not reach node3, and the head's lands there."""
    head = _head(tmp_path, monkeypatch)
    node2, node3 = _root(tmp_path, "node2"), _root(tmp_path, "node3")
    _write_retry(node2, {"fallbackChains": {"default": ["deepseek/deepseek-flash"]}})

    result, link = _push_from(node2, node3, OTHER_DEVICE)
    assert result[routing_sync.ROUTING_KIND]["outcome"] == "not_authority", result
    assert link.applies() == []
    # A row without the operator's claim (what node2 could hand-craft) is refused.
    row = routing_sync.local_row(node2)
    unclaimed = routing_sync.apply_routing_locked(node3, row, OTHER_DEVICE, {})
    assert unclaimed["outcome"] == "refused", unclaimed
    assert read_config_values(node3) == {}

    assert _push(head, _PeerLink(node3))[routing_sync.ROUTING_KIND]["outcome"] == "installed"
    assert _retry(node3)["fallbackChains"] == {"default": HEAD_CHAIN}


def test_only_a_node_under_the_same_operator_is_sent_or_accepts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    head = _head(tmp_path, monkeypatch)
    for name in ("foreign", "unanchored"):
        peer = _root(tmp_path, name)
        link = _PeerLink(peer)
        result = _push(head, link)
        assert result[routing_sync.ROUTING_KIND]["outcome"] == "not_authority", (name, result)
        assert link.applies() == []
        claimed = routing_sync.apply_routing_locked(
            peer,
            {"authority": OP_KEY, "units": {"fallbackChains.default": HEAD_CHAIN}},
            HEAD_DEVICE,
            {},
        )
        assert claimed["outcome"] == "refused", (name, claimed)
        assert read_config_values(peer) == {}


def test_r1_a_lost_mirror_is_re_offered_and_re_installed(tmp_path: Path, monkeypatch: Any) -> None:
    """M2 / R1: deleting the installed chain on the node is not ``in_sync``."""
    head = _head(tmp_path, monkeypatch)
    node = _root(tmp_path, "node")
    link = _PeerLink(node)
    assert _push(head, link)[routing_sync.ROUTING_KIND]["outcome"] == "installed"
    assert _push(head, link)[routing_sync.ROUTING_KIND] == {"outcome": "in_sync"}

    retry = _retry(node)
    retry.pop("fallbackChains")
    _write_retry(node, retry)
    assert HEAD_DEVICE not in routing_sync.routing_state(node)["offered"]

    link.frames.clear()
    again = _push(head, link)
    assert again[routing_sync.ROUTING_KIND]["outcome"] in ("installed", "updated"), again
    assert "fallbackChains.default" in again[routing_sync.ROUTING_KIND]["installed"], again
    assert _retry(node)["fallbackChains"] == {"default": HEAD_CHAIN}
    assert _push(head, link)[routing_sync.ROUTING_KIND] == {"outcome": "in_sync"}


def test_m1_a_nodes_own_invalid_scalar_does_not_block_the_chain(
    tmp_path: Path, monkeypatch: Any
) -> None:
    head = _head(tmp_path, monkeypatch)
    node = _root(tmp_path, "node")
    _write_retry(node, {"usageReservePercent": "abc"})

    report = _push(head, _PeerLink(node))[routing_sync.ROUTING_KIND]

    assert "fallbackChains.default" in report["installed"], report
    assert [k["unit"] for k in report["kept"]] == ["usageReservePercent"], report
    assert _retry(node)["usageReservePercent"] == "abc"


def test_m2_a_crash_between_index_and_config_never_turns_a_mirror_local(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from local_operator import settings_io

    node = _root(tmp_path, "node")
    row = {"authority": OP_KEY, "units": {"fallbackChains.default": HEAD_CHAIN}}
    index = definitions.read_index(node)
    assert (
        routing_sync.apply_routing_locked(node, row, HEAD_DEVICE, index)["outcome"] == "installed"
    )
    definitions.write_index(node, index)

    # Crash AFTER the config write, before the caller's final index write: the
    # caller's index is simply never written.
    changed = ["openai/gpt-6.1-sol", "auto"]
    routing_sync.apply_routing_locked(
        node,
        {**row, "units": {"fallbackChains.default": changed}},
        HEAD_DEVICE,
        definitions.read_index(node),
    )
    assert _retry(node)["fallbackChains"] == {"default": changed}
    unit = routing_sync.routing_state(node)["units"]["fallbackChains.default"]
    assert unit == {"source": "mirror", "origin": HEAD_DEVICE}, unit
    assert routing_sync.local_row(node)["units"] == {}  # type: ignore[index]

    # Crash BEFORE the config write: the old value is still the mirror.
    def _boom(*_a: Any, **_k: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(settings_io, "_store", _boom)
    with pytest.raises(OSError):
        routing_sync.apply_routing_locked(node, row, HEAD_DEVICE, definitions.read_index(node))
    assert _retry(node)["fallbackChains"] == {"default": changed}
    unit = routing_sync.routing_state(node)["units"]["fallbackChains.default"]
    assert unit == {"source": "mirror", "origin": HEAD_DEVICE}, unit
    assert routing_sync.local_row(node)["units"] == {}  # type: ignore[index]


def test_q1_the_failover_sentence_starts_its_own_sentence() -> None:
    from types import SimpleNamespace as NS

    from local_operator.network import readiness

    server = NS(identity=NS(device_id=HEAD_DEVICE), _member_name=lambda _d: "node")
    fact = {"hops": HEAD_CHAIN, "source": "mirror", "origin": OTHER_DEVICE, "model_fallback": True}
    row = {"detail": "a create there without --model is refused"}
    readiness._attach_failover(row, {"failover": fact}, server=server)  # noqa: SLF001
    assert row["detail"].startswith("a create there without --model is refused. Fails over to: ")
    ended = {"detail": "node is served by its own login."}
    readiness._attach_failover(ended, {"failover": fact}, server=server)  # noqa: SLF001
    assert ended["detail"].startswith("node is served by its own login. Fails over to: "), ended


# ---------------------------------------------------------------------------
# Review round 2 regressions (PR #3905): M3 a foreign signer is not authority,
# m3 the state text names routing authority.
# ---------------------------------------------------------------------------

#: The anchor every host below has installed, and a second Mac's OWN operator key.
ANCHOR_KEY = "k_" + "a" * 16
OWN_KEY = "k_" + "b" * 16


class _ProbeSigner:
    def __init__(self, key_id: str) -> None:
        self.handle = SimpleNamespace(key_id=key_id)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _host_with_signer(monkeypatch: pytest.MonkeyPatch, signer_key: str | None) -> list[Any]:
    """Install ``ANCHOR_KEY`` as this host's anchor and make ``signer_key`` loadable.

    Patches the operator module's own seams (the anchor load and the keychain
    backend) so the REAL ``_probe_host`` / ``_can_sign_here`` run end to end
    without touching this machine's keychain or ``/Library`` anchor.
    """
    from local_operator import operator as operator_mod
    from local_operator.operator import keychain, trust

    loaded = SimpleNamespace(
        usable=True,
        exists=True,
        root_owned=True,
        anchor=SimpleNamespace(key_id=ANCHOR_KEY, backend="file-only", presence=False),
    )
    monkeypatch.setattr(trust, "load_anchor", lambda *a, **k: loaded)
    monkeypatch.setattr(
        operator_mod, "authority_level_load", lambda **k: ("operator-file-only", loaded)
    )
    made: list[Any] = []

    def _load() -> Any:
        if signer_key is None:
            return None
        signer = _ProbeSigner(signer_key)
        made.append(signer)
        return signer

    monkeypatch.setattr(keychain, "choose_backend", lambda *a, **k: SimpleNamespace(load=_load))
    mirror_authority.reset_cache()
    return made


@pytest.mark.parametrize(
    ("signer_key", "signs"),
    [(OWN_KEY, False), (ANCHOR_KEY, True), (None, False)],
)
def test_r4_only_the_anchors_own_key_signs_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signer_key: str | None, signs: bool
) -> None:
    """M3 / R4: another Mac's own operator key is verify-only under the anchor."""
    from local_operator import operator as operator_mod
    from local_operator.network import readiness

    made = _host_with_signer(monkeypatch, signer_key)
    authority = mirror_authority._probe_host(tmp_path)  # noqa: SLF001 — the real probe
    assert authority == mirror_authority.Authority(key_id=ANCHOR_KEY, signs=signs)
    assert all(s.closed for s in made), "a probe must leave nothing open"

    # readiness reads the same answer: a foreign key is "verify-only", not "signs".
    monkeypatch.setattr(
        operator_mod, "operator_authority_report", lambda: {"level": "operator-file-only"}
    )
    assert readiness.operator_fact()["verify_only"] is (not signs)


def _with_probed(monkeypatch: pytest.MonkeyPatch, probed: dict[str, mirror_authority.Authority]):
    table = mirror_authority._resolve  # noqa: SLF001 — the autouse fixture's table

    def _resolve(root: Path) -> mirror_authority.Authority:
        return probed.get(Path(root).name) or table(root)

    monkeypatch.setattr(mirror_authority, "_resolve", _resolve)
    mirror_authority.reset_cache()


def test_r4_a_node_with_its_own_key_accepts_the_heads_row_and_sends_nothing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    head = _head(tmp_path, monkeypatch)
    node3, node4 = _root(tmp_path, "node3"), _root(tmp_path, "node4")
    _host_with_signer(monkeypatch, OWN_KEY)
    probed = mirror_authority._probe_host(node3)  # noqa: SLF001
    assert probed == mirror_authority.Authority(key_id=ANCHOR_KEY, signs=False)
    # Head and the other node share node3's anchor, so the comparison is real.
    _with_probed(
        monkeypatch,
        {
            "node3": probed,
            "head": mirror_authority.Authority(key_id=ANCHOR_KEY, signs=True),
            "node4": mirror_authority.Authority(key_id=ANCHOR_KEY, signs=False),
        },
    )
    _write_retry(node3, {"fallbackChains": {"openai/*": ["deepseek/deepseek-flash"]}})

    assert _push(head, _PeerLink(node3))[routing_sync.ROUTING_KIND]["outcome"] == "installed"
    assert _retry(node3)["fallbackChains"]["default"] == HEAD_CHAIN

    result, link = _push_from(node3, node4, OTHER_DEVICE)
    assert result[routing_sync.ROUTING_KIND]["outcome"] == "not_authority", result
    assert link.applies() == []
    assert read_config_values(node4) == {}


def test_r5_a_node_with_its_own_key_does_not_flap_across_ticks(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """M3 / R5: alternating head and node3 ticks converge once and then stay put."""
    head = _head(tmp_path, monkeypatch)
    node3, node4 = _root(tmp_path, "node3"), _root(tmp_path, "node4")
    _host_with_signer(monkeypatch, OWN_KEY)
    _with_probed(
        monkeypatch,
        {
            "node3": mirror_authority._probe_host(node3),  # noqa: SLF001
            "head": mirror_authority.Authority(key_id=ANCHOR_KEY, signs=True),
            "node4": mirror_authority.Authority(key_id=ANCHOR_KEY, signs=False),
        },
    )
    _write_retry(node3, {"fallbackChains": {"openai/*": ["deepseek/deepseek-flash"]}})
    head_before = _retry(head)
    to3, to4 = _PeerLink(node3), _PeerLink(node4)

    outcomes = []
    for _tick in range(3):
        outcomes.append(_push(head, to3)[routing_sync.ROUTING_KIND]["outcome"])
        from_3, link = _push_from(node3, node4, OTHER_DEVICE)
        assert from_3[routing_sync.ROUTING_KIND]["outcome"] == "not_authority", from_3
        assert link.applies() == []
        back, back_link = _push_from(node3, head, OTHER_DEVICE)
        assert back[routing_sync.ROUTING_KIND]["outcome"] == "not_authority", back
        assert back_link.applies() == []
        outcomes.append(_push(head, to4)[routing_sync.ROUTING_KIND]["outcome"])
        assert _retry(node3)["fallbackChains"]["default"] == HEAD_CHAIN
        assert _retry(node4)["fallbackChains"] == {"default": HEAD_CHAIN}
        assert _retry(head) == head_before

    assert outcomes == ["installed", "installed"] + ["in_sync"] * 4, outcomes


@pytest.mark.parametrize(
    ("authority", "line"),
    [
        (
            mirror_authority.Authority(),
            "routing authority: no operator anchor (the operator's failover routing "
            "is not accepted here)",
        ),
        (
            mirror_authority.Authority(key_id=ANCHOR_KEY, signs=True),
            f"routing authority: signs for {ANCHOR_KEY} (this device sends routing)",
        ),
        (
            mirror_authority.Authority(key_id=ANCHOR_KEY, signs=False),
            f"routing authority: verify-only under {ANCHOR_KEY} (accepts that operator's routing)",
        ),
    ],
)
def test_m3_definitions_state_names_routing_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    authority: mirror_authority.Authority,
    line: str,
) -> None:
    """m3: ``lop network definitions state`` says why a node stays primary-only."""
    import argparse

    from local_operator.network import cli as network_cli

    node = _root(tmp_path, "node")
    monkeypatch.setattr(mirror_authority, "_resolve", lambda _root: authority)
    mirror_authority.reset_cache()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(node))

    args = argparse.Namespace(definitions_command="state", json=False)
    assert network_cli._cmd_definitions(args) == 0  # noqa: SLF001
    text = capsys.readouterr().out.splitlines()
    assert line in text, text
    # A bare device still says it has nothing, before the authority line.
    assert text[0] == "no agent or team definitions on this device", text
