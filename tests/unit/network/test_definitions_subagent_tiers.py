"""The operator's subagent model tiers over the mesh (`network/definitions.py`).

WHY THESE CELLS EXIST: a role or team slot pinned to ``effort: hi`` resolves that
tier against the config of the device the session runs on. On lop-remote-node-2 the
config had no ``subagents.models``, so every pinned launch refused with "effort tier
'hi' is unavailable: no model configured at subagents.models.hi" (designer and
ux-reviewer could not launch there). The definitions bundle now carries the tier map
as a singleton row, under the operator-instructions policy (#3829). Each cell pins
one claim:

* a node with no tiers gets the head's map from a push, a second push sends nothing,
  and ``effort='hi'`` then resolves on the node through the real launch resolver;
* the install goes through the settings write path, so every other key survives;
* a node's own tiers, or an edited mirror, are kept and the conflict recorded, and
  the push still succeeds;
* a malformed row (unknown tier, non-string id, oversize, bad model_choice) is
  refused on receipt and writes nothing;
* nothing but the tier map travels: no other config key, and a credential-shaped
  selector is withheld at the sender and refused at the receiver;
* the map flows ONE way, by the failover routing's rule (``mirror_authority``,
  #3905 M1): only the operator's signing device sends it, only a node holding the
  operator's anchor verify-only takes it, and a node never authors the head's.

The "peer" is a second config root behind a fake link routed to the real
``definition_state`` / ``apply_bundle``, as in the operator-instructions tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.network import definitions, mirror_authority

HEAD_DEVICE = "d_" + "a" * 32
NODE_DEVICE = "d_" + "b" * 32
HEAD_TIERS = {
    "lo": "deepseek/deepseek-flash",
    "med": "deepseek/deepseek-flash",
    "hi": "anthropic/claude-opus-5-5",
}
#: A credential SHAPE (GitHub classic PAT), synthetic. Used to prove nothing but the
#: tier map leaves the head, and that a credential-shaped selector never travels.
FAKE_TOKEN = "ghp_" + "A1b2C3d4E5" * 3 + "f6G7h8"

#: The synthetic operator anchor every root is enrolled under, resolved from the
#: root's directory name exactly as in ``test_definitions_routing``: ``head`` holds
#: the private half (the operator's device), ``foreign`` another operator's anchor,
#: ``unanchored`` none, and every other root the anchor verify-only (a node).
OP_KEY = "k_" + "0" * 16
SIGNING_ROOTS = {"head"}
FOREIGN_ROOTS = {"foreign"}
UNANCHORED_ROOTS = {"unanchored"}


@pytest.fixture(autouse=True)
def _operator_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real probe reads the HOST's anchor, which a test running head and nodes
    in one process cannot tell apart (and must never read off this machine)."""

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


def _set_subagents(root: Path, section: dict[str, Any], *, drop: tuple[str, ...] = ()) -> None:
    # ``set_config_value`` keeps every disk leaf the payload omits unless dropped.
    ConfigManager(root).set_config_value(
        "subagents", section, drop={("subagents", *leaf.split(".")): "test" for leaf in drop}
    )


def _subagents(root: Path) -> dict[str, Any]:
    raw = ConfigManager(root).get_config_value("subagents", None)
    return dict(raw) if isinstance(raw, dict) else {}


class _PeerLink:
    def __init__(self, peer_root: Path, *, legacy: bool = False) -> None:
        self.peer_root = peer_root
        self.legacy = legacy
        self.frames: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], timeout: float = 0) -> Any:
        self.frames.append(frame)
        if frame["phase"] == "state":
            detail = definitions.definition_state(self.peer_root, names=frame.get("names"))
            if self.legacy:
                detail.pop(definitions.SUBAGENT_TIERS_KIND)
            return {"op": "ok", "detail": detail}
        summary = definitions.apply_bundle(
            self.peer_root, frame["bundle"], origin_device=HEAD_DEVICE
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


def test_a_push_installs_the_heads_tiers_and_keeps_every_other_key(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS), "model_choice": "operator"})
    # The node's own unrelated settings, inside AND outside ``subagents``.
    _set_subagents(node, {"max_running": 3})
    ConfigManager(node).set_config_value("tool_approval_mode", "ask")
    link = _PeerLink(node)

    result = _push(head, link)

    assert result["ok"] is True, result
    assert result[definitions.SUBAGENT_TIERS_KIND] == {"outcome": "installed"}, result
    assert "the subagent model tiers" in result["message"], result
    section = _subagents(node)
    assert section["models"] == HEAD_TIERS
    assert section["model_choice"] == "operator"
    assert section["max_running"] == 3
    assert ConfigManager(node).get_config_value("tool_approval_mode", None) == "ask"
    state = definitions.subagent_tiers_state(node)
    assert state["status"] == "mirror" and state["origin"] == HEAD_DEVICE, state

    # Idempotent through the node's state digest.
    link.frames.clear()
    again = _push(head, link)
    assert again["code"] == "in_sync", again
    assert again[definitions.SUBAGENT_TIERS_KIND] == {"outcome": "in_sync"}
    assert link.applies() == []

    # The head changes hi and drops lo: an untouched mirror follows exactly.
    _set_subagents(
        head,
        {"models": {"med": "deepseek/deepseek-flash", "hi": "openai/gpt-6"}},
        drop=("models.lo", "model_choice"),
    )
    updated = _push(head, link)
    assert updated[definitions.SUBAGENT_TIERS_KIND] == {"outcome": "updated"}, updated
    section = _subagents(node)
    assert section["models"] == {"med": "deepseek/deepseek-flash", "hi": "openai/gpt-6"}
    assert "model_choice" not in section
    assert definitions.subagent_tiers_state(node)["status"] == "mirror"


def test_effort_hi_resolves_on_the_node_after_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incident's real path: the tool-argument resolver ``builtin.py`` uses
    (``effort_tier_rejection``) and the launch resolver, read on the node's root."""
    from local_operator.harness.subagent import (
        effort_tier_rejection,
        resolve_member_model,
    )

    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(node))

    before = effort_tier_rejection("hi")
    assert before is not None and "unavailable" in before, before
    assert resolve_member_model("designer", launch_tier="hi").unavailable

    assert _push(head, _PeerLink(node))["ok"] is True

    assert effort_tier_rejection("hi") is None
    resolved = resolve_member_model("designer", launch_tier="hi")
    assert resolved.unavailable == "", resolved
    assert resolved.selector == HEAD_TIERS["hi"], resolved


def test_a_nodes_own_tiers_are_kept_and_the_conflict_recorded(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    _set_subagents(node, {"models": {"hi": "openai/gpt-6"}})
    link = _PeerLink(node)

    result = _push(head, link)

    # A conflict never fails the push.
    assert result["ok"] is True, result
    report = result[definitions.SUBAGENT_TIERS_KIND]
    assert report["outcome"] == "conflict" and report["code"] == "own", report
    assert "keeps its own subagent model tiers" in report["notice"], report
    assert _subagents(node)["models"] == {"hi": "openai/gpt-6"}
    state = definitions.subagent_tiers_state(node)
    assert state["status"] == "authored"
    assert state["conflict"]["origin"] == HEAD_DEVICE and state["conflict"]["code"] == "own"

    # The refused revision is not re-sent every tick.
    link.frames.clear()
    again = _push(head, link)
    assert again[definitions.SUBAGENT_TIERS_KIND]["outcome"] == "conflict", again
    assert link.applies() == []


def test_a_mirror_edited_on_the_node_is_kept(tmp_path: Path) -> None:
    from local_operator import settings_io

    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    link = _PeerLink(node)
    assert _push(head, link)[definitions.SUBAGENT_TIERS_KIND] == {"outcome": "installed"}

    # The node's operator edits hi through the Settings write path.
    row = next(s for s in settings_io.SETTINGS if s.key == "subagents.models.hi")
    settings_io.write_setting(ConfigManager(node), row, "openai/gpt-6", notify=False)
    assert definitions.subagent_tiers_state(node)["status"] == "edited"

    _set_subagents(head, {"models": {**HEAD_TIERS, "lo": "openai/gpt-6-mini"}})
    result = _push(head, link)
    assert result["ok"] is True, result
    assert result[definitions.SUBAGENT_TIERS_KIND]["code"] == "edited", result
    assert _subagents(node)["models"]["hi"] == "openai/gpt-6"
    assert _subagents(node)["models"]["lo"] == HEAD_TIERS["lo"]


@pytest.mark.parametrize(
    "tiers, needle",
    [
        ({"models": {"ultra": "openai/gpt-6"}}, "not a model tier"),
        ({"models": {"hi": 5}}, "not a usable model id"),
        ({"models": {"hi": {"model": "openai/gpt-6", "effort": {"x": 1}}}}, "not a usable"),
        ({"models": {"hi": ["openai/gpt-6"]}}, "not a usable"),
        ({"models": {"hi": "x" * 300}}, "not a usable"),
        ({"models": {"hi": "openai/gpt-6"}, "model_choice": "always"}, "model_choice"),
        ({"models": {"hi": "openai/gpt-6"}, "pad": "y" * 5000}, "bytes"),
        ({"models": "openai/gpt-6"}, "not a mapping"),
        ({}, "empty"),
    ],
)
def test_a_malformed_row_is_refused_and_writes_nothing(
    tmp_path: Path, tiers: dict[str, Any], needle: str
) -> None:
    node = _root(tmp_path, "node")
    _set_subagents(node, {"max_running": 2})
    bundle = {
        "kind": definitions.BUNDLE_KIND,
        "version": definitions.BUNDLE_VERSION,
        "origin_device": HEAD_DEVICE,
        "agents": [],
        "teams": [],
        definitions.SUBAGENT_TIERS_KIND: {"kind": definitions.SUBAGENT_TIERS_KIND, **tiers},
    }

    summary = definitions.apply_bundle(node, bundle, origin_device=HEAD_DEVICE)

    result = summary[definitions.SUBAGENT_TIERS_KIND]
    assert result["outcome"] == "refused" and needle in result["reason"], result
    # The agent/team half is untouched, and nothing was written.
    assert summary["conflicts"] == [] and summary["refused"] == []
    assert _subagents(node) == {"max_running": 2}
    assert definitions.subagent_tiers_state(node)["status"] == "absent"


def test_only_the_tier_map_travels_and_a_credential_shape_never_does(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    # Credential-shaped values elsewhere in the head's config, inside and outside
    # ``subagents``: none of them may reach the wire.
    _set_subagents(head, {"models": dict(HEAD_TIERS), "github_token": FAKE_TOKEN})
    ConfigManager(head).set_config_value("api_token", FAKE_TOKEN)
    link = _PeerLink(node)

    assert _push(head, link)[definitions.SUBAGENT_TIERS_KIND] == {"outcome": "installed"}
    sent = link.applies()[0]["bundle"][definitions.SUBAGENT_TIERS_KIND]
    # ``authority`` is the sender's anchor claim beside the map, never config.
    assert set(sent) == {"kind", "name", "models", "authority"}, sent
    assert sent["authority"] == OP_KEY, sent
    assert FAKE_TOKEN not in json.dumps(link.frames)
    assert "github_token" not in _subagents(node)

    # A credential-shaped SELECTOR is withheld at the sender ...
    _set_subagents(head, {"models": {"hi": f"openai/{FAKE_TOKEN}"}})
    other = _root(tmp_path, "other")
    other_link = _PeerLink(other)
    withheld = _push(head, other_link)
    assert withheld[definitions.SUBAGENT_TIERS_KIND]["outcome"] == "withheld", withheld
    assert FAKE_TOKEN not in json.dumps(other_link.frames)
    assert definitions.subagent_tiers_state(other)["status"] == "absent"

    # ... and refused at the receiver whatever an older sender put on the wire.
    summary = definitions.apply_bundle(
        other,
        {
            "kind": definitions.BUNDLE_KIND,
            "version": definitions.BUNDLE_VERSION,
            "agents": [],
            "teams": [],
            definitions.SUBAGENT_TIERS_KIND: {"models": {"hi": f"openai/{FAKE_TOKEN}"}},
        },
        origin_device=HEAD_DEVICE,
    )
    result = summary[definitions.SUBAGENT_TIERS_KIND]
    assert result["outcome"] == "refused" and "credential" in result["reason"], result
    assert definitions.subagent_tiers_state(other)["status"] == "absent"


def test_an_older_peer_is_never_sent_the_tiers(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    link = _PeerLink(node, legacy=True)

    result = _push(head, link)

    assert result[definitions.SUBAGENT_TIERS_KIND] == {"outcome": "unsupported"}, result
    assert link.applies() == []
    assert _subagents(node) == {}


def test_a_mirror_is_never_re_sent_as_the_nodes_own(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    _push(head, _PeerLink(node))
    assert definitions.SUBAGENT_TIERS_KIND not in definitions.local_bundle(node)


def test_mapping_tiers_round_trip_without_a_resend(tmp_path: Path) -> None:
    """The ``{model, effort}`` form (#2428) installs and re-reads to the same digest."""
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    tiers = {"hi": {"model": "anthropic/claude-opus-5-5", "effort": "High"}}
    _set_subagents(head, {"models": tiers})
    link = _PeerLink(node)

    assert _push(head, link)[definitions.SUBAGENT_TIERS_KIND] == {"outcome": "installed"}
    assert _subagents(node)["models"] == {
        "hi": {"model": "anthropic/claude-opus-5-5", "effort": "high"}
    }
    link.frames.clear()
    assert _push(head, link)["code"] == "in_sync"


def test_state_and_readiness_name_the_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    from local_operator.network import cli as network_cli
    from local_operator.network import readiness

    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    _set_subagents(node, {"models": {"hi": "openai/gpt-6"}})
    _push(head, _PeerLink(node))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(node))

    args = argparse.Namespace(definitions_command="state", json=False)
    assert network_cli._cmd_definitions(args) == 0
    text = capsys.readouterr().out
    assert "subagent model tiers: yours (hi=openai/gpt-6)" in text, text
    assert f"subagent model tiers from {HEAD_DEVICE} not installed" in text, text

    args = argparse.Namespace(definitions_command="state", json=True)
    assert network_cli._cmd_definitions(args) == 0
    payload = json.loads(capsys.readouterr().out)
    row = payload.get(definitions.SUBAGENT_TIERS_KIND) or payload.get("detail", {}).get(
        definitions.SUBAGENT_TIERS_KIND
    )
    assert row["status"] == "authored" and row["conflict"]["code"] == "own", payload

    member = SimpleNamespace(device_id=NODE_DEVICE, name="device-b")
    facts = {"subagent_tiers": readiness.subagent_tiers_fact(node)}
    ready_row = readiness.subagent_tiers_row(member, facts, peer_label="device-b")
    assert ready_row is not None
    assert readiness.row_state(ready_row) == readiness.ROW_STATE_WARN, ready_row
    assert ready_row["detail"].startswith("device-b keeps its own subagent model tiers")
    assert readiness.onboarding_failures([ready_row]) == []
    # A clean mirror has nothing to name.
    mirror = _root(tmp_path, "mirror")
    _push(head, _PeerLink(mirror))
    facts = {"subagent_tiers": readiness.subagent_tiers_fact(mirror)}
    assert readiness.subagent_tiers_row(member, facts, peer_label="x") is None


def test_the_state_reply_never_carries_this_devices_own_tier_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA round 4, Q-3: the ``state`` phase answers every admin peer, every tick.

    A node's own tier text is read off disk unvalidated, so before the fix a
    credential-shaped selector the push would withhold still left the device in
    the reply. The wire carries digest/status/origin only; the local CLI keeps
    showing the values.
    """
    from types import SimpleNamespace

    from local_operator.network import cli as network_cli

    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    own = {"hi": "openai/gpt-6", "med": FAKE_TOKEN}
    _set_subagents(node, {"models": dict(own)})

    # The real ``net_definitions`` handler, as a peer's state request reaches it.
    handler = definitions.make_handler(SimpleNamespace(root=node))  # type: ignore[arg-type]
    reply = handler(None, {"phase": "state"})
    wire = json.dumps(reply, default=str)
    assert FAKE_TOKEN not in wire and "openai/gpt-6" not in wire, reply
    row = reply[definitions.SUBAGENT_TIERS_KIND]
    assert "models" not in row, row
    assert row["status"] == "authored" and row["digest"], row

    # The sender decides on what the reply still carries: the node keeps its own
    # map, and no frame in either direction carries the token.
    link = _PeerLink(node)
    _push(head, link)
    assert _subagents(node)["models"] == own
    assert all(FAKE_TOKEN not in json.dumps(f, default=str) for f in link.frames)

    # The node's own operator still sees which model each tier runs on, locally.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(node))
    args = argparse.Namespace(definitions_command="state", json=True)
    assert network_cli._cmd_definitions(args) == 0
    payload = json.loads(capsys.readouterr().out)
    local = payload.get(definitions.SUBAGENT_TIERS_KIND) or payload.get("detail", {}).get(
        definitions.SUBAGENT_TIERS_KIND
    )
    assert local["models"]["hi"] == "openai/gpt-6", payload


def _tiers_bundle(tiers: dict[str, Any], origin: str = HEAD_DEVICE) -> dict[str, Any]:
    return {
        "kind": definitions.BUNDLE_KIND,
        "version": definitions.BUNDLE_VERSION,
        "origin_device": origin,
        "agents": [],
        "teams": [],
        # Signed for by the operator's anchor, as the head's push sends it.
        definitions.SUBAGENT_TIERS_KIND: {
            "kind": definitions.SUBAGENT_TIERS_KIND,
            "authority": OP_KEY,
            **tiers,
        },
    }


def test_a_broken_config_is_reported_unreadable_and_never_moved_aside(tmp_path: Path) -> None:
    """Review round 1, M1: peer-reachable reads never let ConfigManager rename config.

    ``ConfigManager`` recovers a malformed file by renaming it ``config.yml.bad.*``
    and loading defaults; a peer's state probe used to trigger that, and the next
    apply then installed over defaults. Every peer path must leave the bytes alone.
    """
    from local_operator.network import readiness

    node = _root(tmp_path, "node")
    config = node / "config.yml"
    broken = b"subagents:\n  models: {hi: openai/gpt-6\n\tmax_running: [\n"
    config.write_bytes(broken)

    def untouched() -> None:
        assert config.read_bytes() == broken
        assert sorted(p.name for p in node.iterdir() if ".bad" in p.name) == []

    state = definitions.subagent_tiers_state(node)
    assert state["status"] == "unreadable", state
    untouched()
    detail = definitions.definition_state(node, names=None)
    assert detail[definitions.SUBAGENT_TIERS_KIND]["status"] == "unreadable"
    untouched()
    fact = readiness.subagent_tiers_fact(node)
    assert fact is not None and fact["status"] == "unreadable", fact
    untouched()

    summary = definitions.apply_bundle(
        node, _tiers_bundle({"models": dict(HEAD_TIERS)}), origin_device=HEAD_DEVICE
    )
    result = summary[definitions.SUBAGENT_TIERS_KIND]
    assert result["outcome"] == "conflict" and result["code"] == "unreadable", result
    untouched()


def test_a_peer_readiness_probe_never_moves_a_broken_config_aside(tmp_path: Path) -> None:
    """QA round 2, Q-2: the WHOLE ``net_readiness`` answer leaves the bytes alone.

    The M1 guard covered the tier reader, but ``collect_peer_facts`` runs
    ``default_model_fact`` first, which built ``ConfigManager(root)`` after only
    an existence check: the probe renamed config.yml aside, and the tiers fact
    read in the same answer then saw defaults instead of ``unreadable``.
    """
    from local_operator.network import readiness

    node = _root(tmp_path, "node")
    home = _root(tmp_path, "home")
    config = node / "config.yml"
    broken = b"subagents:\n  models: {hi: openai/gpt-6\n\tmax_running: [\n"
    config.write_bytes(broken)

    facts = readiness.collect_peer_facts(node, home=home)

    assert config.read_bytes() == broken
    assert sorted(p.name for p in node.iterdir() if ".bad" in p.name) == []
    assert facts["subagent_tiers"] is not None
    assert facts["subagent_tiers"]["status"] == "unreadable", facts["subagent_tiers"]
    # #3951's pins fact reads the same config: guarded, so no fact (and no row)
    # rather than a rename plus a "nothing configured" claim read off defaults.
    assert facts["subagent_tier_pins"] is None
    assert facts["default_model"]["resolved"] is False
    assert "could not be read" in facts["default_model"]["reason"], facts["default_model"]


def test_a_tiers_mirror_from_another_origin_is_kept(tmp_path: Path) -> None:
    """Review round 1, m2: a copy of device C's tiers is not replaced by A's."""
    other = "d_" + "c" * 32
    node = _root(tmp_path, "node")
    first = definitions.apply_bundle(
        node, _tiers_bundle({"models": {"hi": "openai/gpt-6"}}, other), origin_device=other
    )
    assert first[definitions.SUBAGENT_TIERS_KIND] == {"outcome": "installed"}, first

    summary = definitions.apply_bundle(
        node, _tiers_bundle({"models": dict(HEAD_TIERS)}), origin_device=HEAD_DEVICE
    )
    result = summary[definitions.SUBAGENT_TIERS_KIND]
    assert result["outcome"] == "conflict" and result["code"] == "other_origin", result
    assert _subagents(node)["models"] == {"hi": "openai/gpt-6"}
    state = definitions.subagent_tiers_state(node)
    assert state["status"] == "mirror" and state["origin"] == other, state
    assert state["conflict"]["code"] == "other_origin", state
    notice = definitions.tiers_conflict_notice(state["conflict"], peer_label="device-b")
    assert notice.startswith("device-b keeps a copy of another device's"), notice


def test_a_settings_save_racing_the_mirror_write_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, m1: the policy read is re-checked inside the write lock.

    The node had no tiers when apply decided to install, then its operator saved
    ``hi`` through ``/settings`` before the mirror took the lock. That save wins.
    """
    import contextlib

    from local_operator import config_writer, settings_io

    node = _root(tmp_path, "node")
    _set_subagents(node, {"max_running": 2})
    real_lock = config_writer.config_write_lock
    raced = {"done": False}

    @contextlib.contextmanager
    def lock_after_a_settings_save(config_dir: Path, **kwargs: Any) -> Any:
        if not raced["done"]:
            raced["done"] = True
            row = next(s for s in settings_io.SETTINGS if s.key == "subagents.models.hi")
            settings_io.write_setting(ConfigManager(node), row, "openai/gpt-6", notify=False)
        with real_lock(config_dir, **kwargs):
            yield

    monkeypatch.setattr(config_writer, "config_write_lock", lock_after_a_settings_save)
    summary = definitions.apply_bundle(
        node, _tiers_bundle({"models": dict(HEAD_TIERS)}), origin_device=HEAD_DEVICE
    )

    assert raced["done"]
    result = summary[definitions.SUBAGENT_TIERS_KIND]
    assert result["outcome"] == "conflict" and result["code"] == "edited", result
    assert _subagents(node)["models"] == {"hi": "openai/gpt-6"}
    assert _subagents(node)["max_running"] == 2
    assert definitions.subagent_tiers_state(node)["status"] == "authored"


def test_an_in_sync_push_after_a_tiers_conflict_does_not_claim_the_same_definitions(
    tmp_path: Path,
) -> None:
    """QA round 1, Q-1: a refused tiers revision is not "the same definitions"."""
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    _set_subagents(node, {"models": {"hi": "openai/gpt-6"}})
    link = _PeerLink(node)
    _push(head, link)

    again = _push(head, link)

    assert again["code"] == "in_sync", again
    assert again[definitions.SUBAGENT_TIERS_KIND]["outcome"] == "conflict", again
    assert again["message"].endswith("hold the same agent and team definitions"), again


def test_a_node_with_an_empty_subagents_section_resolves_hi_after_a_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lop-remote-node-2 shape exactly: ``subagents: {}`` on disk (not an absent
    section). After the push the node's own resolvers, read against its root, see
    the head's ``hi`` selector."""
    from local_operator.harness.subagent import (
        read_effort_tier_targets,
        resolve_member_model,
    )

    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    (node / "config.yml").write_text("subagents: {}\n", encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(node))
    assert "hi" not in read_effort_tier_targets()
    assert resolve_member_model("designer", launch_tier="hi").unavailable

    result = _push(head, _PeerLink(node))
    assert result[definitions.SUBAGENT_TIERS_KIND] == {"outcome": "installed"}, result

    assert read_effort_tier_targets()["hi"][0] == HEAD_TIERS["hi"]
    resolved = resolve_member_model("designer", launch_tier="hi")
    assert resolved.unavailable == "" and resolved.selector == HEAD_TIERS["hi"], resolved


def test_a_node_never_sends_its_tiers_to_the_operators_device(tmp_path: Path) -> None:
    """#3905 M1 for this lane: an operator device with no tiers of its own used to
    take a node's map on the node's cadence push. The node does not sign for the
    anchor, so it sends nothing and the head's config is untouched."""
    node, head = _root(tmp_path, "node"), _root(tmp_path, "head")
    _set_subagents(node, {"models": {"hi": "openai/gpt-6"}})
    link = _PeerLink(head)

    result = definitions.push_to_peer(_Server(node, link), HEAD_DEVICE)  # type: ignore[arg-type]

    report = result[definitions.SUBAGENT_TIERS_KIND]
    assert report["outcome"] == "not_authority", report
    assert "does not sign for the operator" in report["reason"], report
    assert link.applies() == []
    assert "models" not in _subagents(head)


@pytest.mark.parametrize(
    ("peer", "needle"),
    [
        ("unanchored", "does not hold this operator's authority"),
        ("foreign", "does not hold this operator's authority"),
    ],
)
def test_the_head_sends_only_to_a_node_under_its_own_anchor(
    tmp_path: Path, peer: str, needle: str
) -> None:
    head, other = _root(tmp_path, "head"), _root(tmp_path, peer)
    _set_subagents(head, {"models": dict(HEAD_TIERS)})
    link = _PeerLink(other)

    report = _push(head, link)[definitions.SUBAGENT_TIERS_KIND]

    assert report["outcome"] == "not_authority" and needle in report["reason"], report
    assert definitions.SUBAGENT_TIERS_KIND not in json.dumps(link.applies())
    assert _subagents(other) == {}


@pytest.mark.parametrize(
    ("root_name", "claim", "needle"),
    [
        ("node", None, "does not sign for this device's operator"),
        ("node", "k_" + "f" * 16, "does not sign for this device's operator"),
        ("unanchored", OP_KEY, "no operator authority installed"),
        ("head", OP_KEY, "signs for the operator itself"),
    ],
)
def test_the_receiver_takes_tiers_only_from_its_operator_device(
    tmp_path: Path, root_name: str, claim: str | None, needle: str
) -> None:
    """Receiver half, whatever the sender's build put on the wire: a missing or
    foreign claim, a device with no anchor, and the operator's own device all
    refuse, write nothing and record no conflict."""
    root = _root(tmp_path, root_name)
    bundle = _tiers_bundle({"models": dict(HEAD_TIERS)})
    row = bundle[definitions.SUBAGENT_TIERS_KIND]
    if claim is None:
        row.pop("authority")
    else:
        row["authority"] = claim

    summary = definitions.apply_bundle(root, bundle, origin_device=HEAD_DEVICE)

    result = summary[definitions.SUBAGENT_TIERS_KIND]
    assert result["outcome"] == "refused" and needle in result["reason"], result
    assert "models" not in _subagents(root)
    state = definitions.subagent_tiers_state(root)
    assert state["status"] == "absent" and not state.get("conflict"), state
