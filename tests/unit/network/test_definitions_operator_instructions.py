"""The operator's custom instructions over the mesh (`network/definitions.py`).

WHY THESE CELLS EXIST: ``<config>/system_prompt.md`` carries the operator's standing
rules (the review gate, release rules, reviewer rules) and the prompt builder reads
it per device, so a session on a mesh node ran without them. The definitions bundle
now carries it as a singleton row. Each cell pins one claim the change makes:

* a peer with no file gets the head's text from a push, and a second push sends
  nothing (idempotent through the peer's state digest);
* a node's OWN file, or an edited mirror, is never overwritten, and the refusal is
  recorded where ``lop network definitions state`` reads it — without failing the
  push, so creates and the agent/team cadence are unaffected;
* the credential-shape scan withholds it at the sender and refuses it at the
  receiver, exactly as for agent and team rows;
* an older peer (state reply without the key) is never sent it;
* a symlinked instructions file is never written through.

The "peer" is a second config root behind a fake link that routes the two phases to
the real ``definition_state`` / ``apply_bundle`` — the same handler bodies
``make_handler`` serves — so the sender's and receiver's code both run for real.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import definitions

HEAD_RULES = "# Personal Defaults\n\n- Never merge without an agent-review round.\n"
NODE_RULES = "# Node rules\n\n- This machine's own operator wrote these.\n"
HEAD_DEVICE = "d_" + "a" * 32
NODE_DEVICE = "d_" + "b" * 32


def _root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return root


class _PeerLink:
    """A link whose far end is ``peer_root``, served by the real handler bodies."""

    def __init__(self, peer_root: Path, *, legacy: bool = False) -> None:
        self.peer_root = peer_root
        self.legacy = legacy
        self.frames: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], timeout: float = 0) -> Any:
        self.frames.append(frame)
        if frame["phase"] == "state":
            detail = definitions.definition_state(self.peer_root, names=frame.get("names"))
            if self.legacy:
                # An older build's reply: agents and teams only.
                detail.pop(definitions.OPERATOR_INSTRUCTIONS_KIND)
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


def _instructions(root: Path) -> Path:
    return root / "system_prompt.md"


def test_a_peer_with_no_instructions_gets_the_heads_after_a_push(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _instructions(head).write_text(HEAD_RULES, encoding="utf-8")
    link = _PeerLink(node)

    result = definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]

    assert result["ok"] is True, result
    assert result[definitions.OPERATOR_INSTRUCTIONS_KIND] == {"outcome": "installed"}
    assert _instructions(node).read_text(encoding="utf-8") == HEAD_RULES
    state = definitions.operator_instructions_state(node)
    assert state["status"] == "mirror" and state["origin"] == HEAD_DEVICE, state

    # Idempotent: the peer's state digest now matches, so nothing is applied again.
    link.frames.clear()
    again = definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]
    assert again["code"] == "in_sync", again
    assert again[definitions.OPERATOR_INSTRUCTIONS_KIND] == {"outcome": "in_sync"}
    assert link.applies() == []

    # The head edits its rules: an untouched mirror follows its origin.
    _instructions(head).write_text(HEAD_RULES + "- Release once per window.\n", encoding="utf-8")
    updated = definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]
    assert updated[definitions.OPERATOR_INSTRUCTIONS_KIND] == {"outcome": "updated"}
    assert "Release once per window" in _instructions(node).read_text(encoding="utf-8")


def test_an_unnamed_create_push_still_carries_the_instructions(tmp_path: Path) -> None:
    """``names`` with no agents and no teams narrows those kinds, not the singleton."""
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _instructions(head).write_text(HEAD_RULES, encoding="utf-8")
    link = _PeerLink(node)

    result = definitions.push_to_peer(
        _Server(head, link),  # type: ignore[arg-type]
        NODE_DEVICE,
        names={"agents": [], "teams": []},
    )

    assert result[definitions.OPERATOR_INSTRUCTIONS_KIND] == {"outcome": "installed"}, result
    assert _instructions(node).read_text(encoding="utf-8") == HEAD_RULES


def test_a_nodes_own_instructions_are_preserved_and_reported(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _instructions(head).write_text(HEAD_RULES, encoding="utf-8")
    _instructions(node).write_text(NODE_RULES, encoding="utf-8")
    link = _PeerLink(node)

    result = definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]

    # Never overwritten, and never a failed push: creates and the cadence carry on.
    assert _instructions(node).read_text(encoding="utf-8") == NODE_RULES
    assert result["ok"] is True, result
    assert result["conflicts"] == [] and result["refused"] == []
    report = result[definitions.OPERATOR_INSTRUCTIONS_KIND]
    assert report["outcome"] == "conflict" and "own custom instructions" in report["reason"]
    assert definitions.create_refusal(result, peer_label="node") == ""

    # Recorded on the NODE, where `definitions state` reads it with no peer up.
    state = definitions.operator_instructions_state(node)
    assert state["status"] == "authored"
    assert state["conflict"]["origin"] == HEAD_DEVICE
    assert state["conflict"]["offered"] == definitions.digest_of(
        definitions._operator_instructions_row(HEAD_RULES)
    )

    # The same revision is not re-sent every tick to a node that already refused it.
    link.frames.clear()
    again = definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]
    assert link.applies() == [], "a refused revision was shipped again"
    assert again[definitions.OPERATOR_INSTRUCTIONS_KIND]["outcome"] == "conflict"


def test_an_edited_mirror_is_preserved_and_reported(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _instructions(head).write_text(HEAD_RULES, encoding="utf-8")
    link = _PeerLink(node)
    definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]

    edited = HEAD_RULES + "- A rule only this node follows.\n"
    _instructions(node).write_text(edited, encoding="utf-8")
    assert definitions.operator_instructions_state(node)["status"] == "edited"
    _instructions(head).write_text(HEAD_RULES + "- A newer head rule.\n", encoding="utf-8")

    result = definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]

    assert _instructions(node).read_text(encoding="utf-8") == edited
    report = result[definitions.OPERATOR_INSTRUCTIONS_KIND]
    assert report["outcome"] == "conflict" and "local edits" in report["reason"], report
    assert definitions.operator_instructions_state(node)["conflict"]["reason"] == report["reason"]


def test_a_mirror_is_never_re_sent_as_the_nodes_own(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _instructions(head).write_text(HEAD_RULES, encoding="utf-8")
    definitions.push_to_peer(_Server(head, _PeerLink(node)), NODE_DEVICE)  # type: ignore[arg-type]

    assert definitions.OPERATOR_INSTRUCTIONS_KIND not in definitions.local_bundle(node)


def test_credential_shaped_instructions_are_withheld_and_refused(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    poisoned = "Deploy with " + "sk_live_" + "A" * 40 + " when asked.\n"
    _instructions(head).write_text(poisoned, encoding="utf-8")

    # Sender: withheld under its own key, value nowhere in the payload.
    bundle = definitions.local_bundle(head)
    assert definitions.OPERATOR_INSTRUCTIONS_KIND not in bundle
    assert bundle["operator_instructions_withheld"]
    assert bundle["withheld"] == [], "the singleton must not abort every create"
    assert "sk_live_" not in json.dumps(bundle)

    link = _PeerLink(node)
    result = definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]
    assert result[definitions.OPERATOR_INSTRUCTIONS_KIND]["outcome"] == "withheld"
    assert link.applies() == []
    assert definitions.create_refusal(result, peer_label="node") == ""

    # Receiver: an older or hostile sender's credential-shaped row is refused.
    summary = definitions.apply_bundle(
        node,
        {
            "kind": definitions.BUNDLE_KIND,
            "version": definitions.BUNDLE_VERSION,
            "agents": [],
            "teams": [],
            definitions.OPERATOR_INSTRUCTIONS_KIND: {"text": poisoned},
        },
        origin_device=HEAD_DEVICE,
    )
    assert summary[definitions.OPERATOR_INSTRUCTIONS_KIND]["outcome"] == "refused"
    assert "credential" in summary[definitions.OPERATOR_INSTRUCTIONS_KIND]["reason"]
    assert not _instructions(node).exists()


def test_an_older_peer_is_never_sent_the_singleton(tmp_path: Path) -> None:
    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _instructions(head).write_text(HEAD_RULES, encoding="utf-8")
    link = _PeerLink(node, legacy=True)

    result = definitions.push_to_peer(_Server(head, link), NODE_DEVICE)  # type: ignore[arg-type]

    assert result["code"] == "in_sync", result
    assert result[definitions.OPERATOR_INSTRUCTIONS_KIND] == {"outcome": "unsupported"}
    assert link.applies() == []


def test_a_symlinked_instructions_file_is_never_written_through(tmp_path: Path) -> None:
    node = _root(tmp_path, "node")
    target = tmp_path / "elsewhere.md"
    target.write_text("", encoding="utf-8")
    _instructions(node).symlink_to(target)

    summary = definitions.apply_bundle(
        node,
        {
            "kind": definitions.BUNDLE_KIND,
            "version": definitions.BUNDLE_VERSION,
            definitions.OPERATOR_INSTRUCTIONS_KIND: {"text": HEAD_RULES},
        },
        origin_device=HEAD_DEVICE,
    )

    assert summary[definitions.OPERATOR_INSTRUCTIONS_KIND]["outcome"] == "conflict"
    assert _instructions(node).is_symlink() and target.read_text(encoding="utf-8") == ""


def test_definitions_state_lists_the_operator_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``lop network definitions state`` names the row, its origin and the conflict."""
    from local_operator.network import cli as network_cli

    head, node = _root(tmp_path, "head"), _root(tmp_path, "node")
    _instructions(head).write_text(HEAD_RULES, encoding="utf-8")
    _instructions(node).write_text(NODE_RULES, encoding="utf-8")
    definitions.push_to_peer(_Server(head, _PeerLink(node)), NODE_DEVICE)  # type: ignore[arg-type]
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(node))

    args = argparse.Namespace(definitions_command="state", json=False)
    assert network_cli._cmd_definitions(args) == 0
    text = capsys.readouterr().out
    assert "custom instructions: yours" in text, text
    assert f"custom instructions from {HEAD_DEVICE} not installed" in text, text

    args = argparse.Namespace(definitions_command="state", json=True)
    assert network_cli._cmd_definitions(args) == 0
    payload = json.loads(capsys.readouterr().out)
    row = payload.get(definitions.OPERATOR_INSTRUCTIONS_KIND) or payload.get("detail", {}).get(
        definitions.OPERATOR_INSTRUCTIONS_KIND
    )
    assert row["status"] == "authored" and row["conflict"]["origin"] == HEAD_DEVICE, payload


def test_an_unnamed_create_still_refreshes_the_instructions_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A create naming no agent or team still runs under the operator's rules, so
    ``_ctl_peer_create`` pushes (instructions only) before anything else is sent,
    and a failing push never fails the create on its own."""
    from local_operator.network import relay

    root = _root(tmp_path, "head")
    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    calls: list[tuple[str, Any]] = []

    def push(srv: Any, device_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((device_id, kwargs.get("names")))
        raise RuntimeError("peer went away mid-push")

    class _Stop(Exception):
        pass

    def answer(*args: Any, **kwargs: Any) -> Any:
        # The step AFTER the push: reaching it proves the push failure did not
        # abort the create.
        raise _Stop

    monkeypatch.setattr(definitions, "push_to_peer", push)
    monkeypatch.setattr(server, "_resolve_peer", lambda peer: NODE_DEVICE)
    monkeypatch.setattr(relay, "_relay_answer", answer, raising=False)
    monkeypatch.setattr(server, "_relay_answer", answer, raising=False)
    monkeypatch.setattr(server, "_ensure_link", lambda *a, **k: (_ for _ in ()).throw(_Stop()))

    try:
        server._ctl_peer_create({"peer": "node", "prompt": "hi"})
    except RuntimeError as exc:  # pragma: no cover - the regression this cell pins
        pytest.fail(f"a failed instructions push aborted the create: {exc}")
    except Exception:  # noqa: BLE001 - the stubbed send after the push stops the rig
        pass

    assert calls == [(NODE_DEVICE, {"agents": [], "teams": []})], calls


def test_the_move_receipt_prints_the_stale_definitions_notice() -> None:
    """``lop sessions move`` prints the carry's ``definitions_notice`` with the others."""
    from local_operator.cli import _carry_lines

    notice = "team 'lopdev' and the custom instructions on node may be an older copy"
    assert _carry_lines({"carry": {"definitions_notice": notice}}) == [notice]
    assert _carry_lines({"carry": {}}) == []
