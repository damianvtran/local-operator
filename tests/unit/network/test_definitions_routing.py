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

import yaml

from local_operator.config import read_config_values
from local_operator.network import definitions, routing_sync

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


def test_a_mirror_from_another_device_is_not_taken_over(tmp_path: Path) -> None:
    node = _root(tmp_path, "node")
    index: dict[str, Any] = {}
    first = routing_sync.apply_routing_locked(
        node, {"units": {"fallbackChains.default": HEAD_CHAIN}}, HEAD_DEVICE, index
    )
    assert first["outcome"] == "installed", first
    second = routing_sync.apply_routing_locked(
        node, {"units": {"fallbackChains.default": ["auto"]}}, OTHER_DEVICE, index
    )
    assert [entry["unit"] for entry in second["kept"]] == ["fallbackChains.default"], second
    assert _retry(node)["fallbackChains"] == {"default": HEAD_CHAIN}


def test_an_unknown_or_unusable_unit_is_refused_whole(tmp_path: Path) -> None:
    node = _root(tmp_path, "node")
    for units in (
        {"maxRetries": 1},
        {"usageReservePercent": 400},
        {"fallbackChains.default": "not-a-list"},
    ):
        result = routing_sync.apply_routing_locked(node, {"units": units}, HEAD_DEVICE, {})
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
