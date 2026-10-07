"""MCP server definitions, unit half: the bundle, the gate, the merge, the push.

The link-level half (two real relays, the real parser) lives in
``test_mcpdefs_link.py``. What this file pins is the CONTENT rules and the merge
matrix, because those are where the slice's promises live:

* a literal ``env``/``headers`` value never reaches the bundle (the escape
  ``$${X}`` included — it is a held value, never a reference);
* a row whose text trips the ONE shape table is withheld and NAMED at the
  sender and refused by name at the receiver;
* the conflict policy is definitions': authored wins, a mirror follows its
  origin only, an edited mirror is a conflict, a deleted mirror is re-installed,
  and a second apply touches no file;
* an old peer is skipped BY CAPABILITY — no request is sent, so no slow-op
  deadline is ever paid (the measured trap this is written against).

It also carries the two hardenings the live-config incident forced (see the
module docstring's ``_assert_inside_root``): the path resolver must honour its
``root`` argument, and every write target must assert containment in the root it
serves before anything is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import definitions, mcpdefs, types, wire

PEER = "d_" + "b" * 32
OWNER = "d_" + "a" * 32


def _write_servers(root: Path, servers: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n", encoding="utf-8")
    return path


def _read_servers(root: Path) -> dict[str, Any]:
    return json.loads((root / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]


# ---------------------------------------------------------------------------
# The bundle builder: what travels and what never does
# ---------------------------------------------------------------------------


def test_the_bundle_carries_reference_names_and_no_literal_value_bytes(root: Path) -> None:
    _write_servers(
        root,
        {
            "gl": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "gitlab-mcp"],
                "env": {"GITLAB_TOKEN": "${GITLAB_TOKEN}", "PLAIN": "hunter2"},
            },
            "crm": {
                "type": "http",
                "url": "https://example.test/mcp",
                "headers": {"Authorization": "Bearer ${CRM_KEY}", "X-Tenant": "acme"},
            },
        },
    )
    bundle = mcpdefs.local_bundle(root)
    flat = json.dumps(bundle, sort_keys=True)
    # THE LITERAL BYTES ARE ABSENT — both a plain literal and the tail of a
    # partial reference (which could hide one) are only ever states.
    assert "hunter2" not in flat
    assert "acme" not in flat
    assert "Bearer" not in flat
    gl = next(row for row in bundle["servers"] if row["name"] == "gl")
    assert gl["raw"]["env"] == {"GITLAB_TOKEN": "ref:GITLAB_TOKEN", "PLAIN": "literal-held"}
    crm = next(row for row in bundle["servers"] if row["name"] == "crm")
    assert crm["raw"]["headers"] == {
        "Authorization": "literal-held",
        "X-Tenant": "literal-held",
    }


def test_a_whole_value_reference_travels_as_its_name(root: Path) -> None:
    _write_servers(root, {"gl": {"type": "stdio", "command": "x", "env": {"K": "${NAME}"}}})
    row = mcpdefs.local_bundle(root)["servers"][0]
    assert row["raw"]["env"] == {"K": "ref:NAME"}


def test_the_escape_stays_a_held_value_never_a_reference(root: Path) -> None:
    """``$${X}`` is the literal text ``${X}`` and must not become ``ref:X``."""
    _write_servers(root, {"gl": {"type": "stdio", "command": "x", "env": {"K": "$${X}"}}})
    row = mcpdefs.local_bundle(root)["servers"][0]
    assert row["raw"]["env"] == {"K": "literal-held"}


def test_withheld_rows_are_named_and_never_sent(root: Path) -> None:
    key_shaped = "sk-live-" + "a" * 24  # trips the shape table on the ARG
    _write_servers(
        root,
        {
            "good": {"type": "stdio", "command": "ok", "args": []},
            "bad": {"type": "stdio", "command": "run", "args": [f"--token={key_shaped}"]},
        },
    )
    bundle = mcpdefs.local_bundle(root)
    assert [row["name"] for row in bundle["servers"]] == ["good"]
    assert [row["name"] for row in bundle["withheld"]] == ["bad"]
    assert key_shaped not in json.dumps(bundle, sort_keys=True)


def test_shape_likeness_chooses_the_article() -> None:
    """D5 (design round 1): "looks like a authorization-bearer" — the renderer
    hardcoded the article, and several shape labels are vowel-initial."""
    assert mcpdefs.shape_likeness("authorization-bearer") == "looks like an authorization-bearer"
    assert mcpdefs.shape_likeness("github-token") == "looks like a github-token"
    # The receipt's own fallback when a row carries no shape label at all.
    assert mcpdefs.shape_likeness("") == "looks like a credential"


def test_transport_owned_headers_and_cwd_never_travel(root: Path) -> None:
    _write_servers(
        root,
        {
            "crm": {
                "type": "http",
                "url": "https://example.test/mcp",
                "headers": {"Content-Type": "application/json", "X-Keep": "v"},
            },
            "local": {"type": "stdio", "command": "x", "cwd": "/machine/local/path"},
        },
    )
    rows = {row["name"]: row for row in mcpdefs.local_bundle(root)["servers"]}
    assert "Content-Type" not in rows["crm"]["raw"]["headers"]
    assert rows["crm"]["raw"]["headers"] == {"X-Keep": "literal-held"}
    assert "cwd" not in rows["local"]["raw"]


def test_auth_type_is_the_only_oauth_field_carried(root: Path) -> None:
    _write_servers(
        root,
        {
            "crm": {
                "type": "http",
                "url": "https://example.test/mcp",
                "auth": {"type": "oauth", "client_secret": "shhh"},
            }
        },
    )
    row = mcpdefs.local_bundle(root)["servers"][0]
    assert row["raw"]["auth"] == {"type": "oauth"}
    assert "shhh" not in json.dumps(row, sort_keys=True)


def test_run_shaping_flags_travel_and_snake_case_is_canonicalised(root: Path) -> None:
    _write_servers(
        root,
        {
            "s": {
                "type": "stdio",
                "command": "x",
                "enabled": False,
                "timeout": 30000,
                "enabled_tools": ["a"],
                "own_turn_only": True,
            }
        },
    )
    row = mcpdefs.local_bundle(root)["servers"][0]
    assert row["raw"]["enabled"] is False
    assert row["raw"]["timeout"] == 30000
    assert row["raw"]["enabledTools"] == ["a"]
    assert row["raw"]["ownTurnOnly"] is True


def test_a_mirror_this_device_holds_is_not_re_sent_as_its_own(root: Path) -> None:
    source = root / "source"
    _write_servers(source, {"crm": {"type": "http", "url": "https://example.test/mcp"}})
    bundle = mcpdefs.local_bundle(source)
    mcpdefs.apply_bundle(root, bundle, origin_device=PEER)
    # ``crm`` here is now a mirror; a bundle built HERE must not claim it.
    assert mcpdefs.local_bundle(root)["servers"] == []


# ---------------------------------------------------------------------------
# The receiver's gate
# ---------------------------------------------------------------------------


def _bundle(servers: list[dict[str, Any]], *, kind: str = mcpdefs.BUNDLE_KIND) -> dict[str, Any]:
    return {
        "kind": kind,
        "version": mcpdefs.BUNDLE_VERSION,
        "origin_device": PEER,
        "servers": servers,
    }


def _row(name: str, **raw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"type": "http", "url": "https://example.test/mcp"}
    base.update(raw)
    transport = base["type"]
    return {"kind": "server", "name": name, "transport": transport, "raw": base}


def test_a_foreign_kind_or_version_is_refused_whole(root: Path) -> None:
    with pytest.raises(types.MeshRefusal) as kind_exc:
        mcpdefs.apply_bundle(root, _bundle([], kind="lop.mesh.agents.v1"), origin_device=PEER)
    assert kind_exc.value.code == "unknown_bundle"
    bad = _bundle([])
    bad["version"] = mcpdefs.BUNDLE_VERSION + 1
    with pytest.raises(types.MeshRefusal) as ver_exc:
        mcpdefs.apply_bundle(root, bad, origin_device=PEER)
    assert ver_exc.value.code == "unknown_bundle_version"
    assert not (root / "mcp.json").exists()


def test_a_credential_shaped_row_is_refused_by_name(root: Path) -> None:
    shape = "ghp_" + "a" * 30
    summary = mcpdefs.apply_bundle(
        root, _bundle([_row("bad", url=f"https://example.test/{shape}")]), origin_device=PEER
    )
    assert [row["name"] for row in summary["refused"]] == ["bad"]
    assert not (root / "mcp.json").exists()


def test_an_unusable_reference_state_is_refused_by_name(root: Path) -> None:
    bad = _row("bad", type="stdio", command="x", env={"K": "ref:not a name"})
    summary = mcpdefs.apply_bundle(root, _bundle([bad]), origin_device=PEER)
    assert [row["name"] for row in summary["refused"]] == ["bad"]
    # A state string that is neither of the two the wire defines is refused too.
    other = _row("bad2", type="stdio", command="x", env={"K": "value:oops"})
    summary2 = mcpdefs.apply_bundle(root, _bundle([other]), origin_device=PEER)
    assert [row["name"] for row in summary2["refused"]] == ["bad2"]


def test_a_transport_owned_header_and_an_unknown_transport_are_refused(root: Path) -> None:
    owned = _row("owned", headers={"Content-Type": "text/plain"})
    summary = mcpdefs.apply_bundle(root, _bundle([owned]), origin_device=PEER)
    assert [row["name"] for row in summary["refused"]] == ["owned"]
    weird = _row("weird", type="carrier-pigeon")
    summary2 = mcpdefs.apply_bundle(root, _bundle([weird]), origin_device=PEER)
    assert [row["name"] for row in summary2["refused"]] == ["weird"]


def test_a_name_the_product_cannot_hold_is_refused(root: Path) -> None:
    summary = mcpdefs.apply_bundle(root, _bundle([_row("bad name")]), origin_device=PEER)
    assert [row["name"] for row in summary["refused"]] == ["bad name"]
    assert not (root / "mcp.json").exists()


def test_an_unreadable_mcpjson_refuses_the_whole_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mcp.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(types.MeshRefusal) as exc:
        mcpdefs.apply_bundle(root, _bundle([_row("a")]), origin_device=PEER)
    assert exc.value.code == "unreadable_config"
    assert (root / "mcp.json").read_text() == "{ not json"


# ---------------------------------------------------------------------------
# The merge matrix (definitions' conflict policy)
# ---------------------------------------------------------------------------


def test_authored_here_wins_and_the_conflict_names_the_row(root: Path) -> None:
    _write_servers(root, {"crm": {"type": "http", "url": "https://mine.example/mcp"}})
    before = (root / "mcp.json").read_bytes()
    summary = mcpdefs.apply_bundle(root, _bundle([_row("crm")]), origin_device=PEER)
    assert [row["name"] for row in summary["conflicts"]] == ["crm"]
    assert (root / "mcp.json").read_bytes() == before


def test_a_mirror_follows_its_origin_and_a_second_apply_touches_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle([_row("crm")])
    first = mcpdefs.apply_bundle(root, bundle, origin_device=PEER)
    assert [row["name"] for row in first["installed"]] == ["crm"]
    before = (root / "mcp.json").read_bytes()

    written: list[Path] = []
    real_write = mcpdefs._write_document

    def _spy(root_arg: Path, document: dict[str, Any]) -> None:
        written.append(root_arg)
        real_write(root_arg, document)

    monkeypatch.setattr(mcpdefs, "_write_document", _spy)
    second = mcpdefs.apply_bundle(root, bundle, origin_device=PEER)
    assert [row["name"] for row in second["unchanged"]] == ["crm"]
    assert not second["conflicts"] and not second["refused"]
    assert written == [], "the second apply wrote the file"
    assert (root / "mcp.json").read_bytes() == before

    # And the source itself can now update its mirror.
    updated = mcpdefs.apply_bundle(
        root, _bundle([_row("crm", url="https://new.example/mcp")]), origin_device=PEER
    )
    assert [row["name"] for row in updated["updated"]] == ["crm"]
    assert _read_servers(root)["crm"]["url"] == "https://new.example/mcp"


def test_an_edited_mirror_is_a_conflict_by_name(root: Path) -> None:
    bundle = _bundle([_row("crm")])
    mcpdefs.apply_bundle(root, bundle, origin_device=PEER)
    servers = _read_servers(root)
    servers["crm"]["url"] = "https://hand-edited.example/mcp"
    _write_servers(root, servers)
    summary = mcpdefs.apply_bundle(root, bundle, origin_device=PEER)
    assert [row["name"] for row in summary["conflicts"]] == ["crm"]
    assert _read_servers(root)["crm"]["url"] == "https://hand-edited.example/mcp"


def test_a_foreign_origin_cannot_take_a_mirror(root: Path) -> None:
    mcpdefs.apply_bundle(root, _bundle([_row("crm")]), origin_device=PEER)
    other = _bundle([_row("crm", url="https://other.example/mcp")])
    summary = mcpdefs.apply_bundle(root, other, origin_device=OWNER)
    assert [row["name"] for row in summary["conflicts"]] == ["crm"]
    assert _read_servers(root)["crm"]["url"] == "https://example.test/mcp"


def test_a_deleted_mirror_is_reinstalled(root: Path) -> None:
    bundle = _bundle([_row("crm")])
    mcpdefs.apply_bundle(root, bundle, origin_device=PEER)
    _write_servers(root, {})
    summary = mcpdefs.apply_bundle(root, bundle, origin_device=PEER)
    assert [row["name"] for row in summary["installed"]] == ["crm"]
    assert "crm" in _read_servers(root)


def test_a_literal_held_key_lands_as_a_placeholder_the_store_can_satisfy(
    root: Path,
) -> None:
    row = _row("s", type="stdio", command="x", env={"SOME_KEY": "literal-held"})
    mcpdefs.apply_bundle(root, _bundle([row]), origin_device=PEER)
    assert _read_servers(root)["s"]["env"] == {"SOME_KEY": "${SOME_KEY}"}
    # And the round trip is idempotent: the placeholder re-reads as the landed
    # digest, so a re-apply reports unchanged rather than conflict (a mirror of
    # a value key is a REFERENCE here — provisioning ``SOME_KEY`` in the store
    # is what makes the server work on this device).
    again = mcpdefs.apply_bundle(root, _bundle([row]), origin_device=PEER)
    assert [row["name"] for row in again["unchanged"]] == ["s"]


def test_top_level_install_lists_are_never_clobbered(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {},
                "enabledServers": ["keep-me"],
                "disabledServers": ["also-keep"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mcpdefs.apply_bundle(root, _bundle([_row("crm")]), origin_device=PEER)
    doc = json.loads((root / "mcp.json").read_text(encoding="utf-8"))
    assert doc["enabledServers"] == ["keep-me"]
    assert doc["disabledServers"] == ["also-keep"]


# ---------------------------------------------------------------------------
# The hardenings the live-config incident forced
# ---------------------------------------------------------------------------


def test_the_path_resolver_honours_its_root_not_the_ambient_config_dir(
    root: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug that clobbered the operator's mcp.json, pinned shut.

    Pre-fix, ``_global_path`` resolved ``config_dir()/mcp.json`` — the ambient
    install — so a rig acting for ``root`` read (and, on apply, wrote) the
    PROCESS's own file. The cell makes the ambient dir a DECOY with a different
    server set and asserts the root argument is what was read.
    """
    decoy = tmp_path_factory.mktemp("ambient-decoy")
    _write_servers(decoy, {"decoy": {"type": "http", "url": "https://decoy.example/mcp"}})
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(decoy))
    _write_servers(root, {"mine": {"type": "http", "url": "https://mine.example/mcp"}})
    assert mcpdefs._global_path(root) == root / "mcp.json"
    assert sorted(mcpdefs.server_state(root)["servers"]) == ["mine"]
    assert [row["name"] for row in mcpdefs.local_bundle(root)["servers"]] == ["mine"]
    # And an apply for ``root`` writes ROOT's file, never the decoy's.
    decoy_before = (decoy / "mcp.json").read_bytes()
    mcpdefs.apply_bundle(root, _bundle([_row("crm")]), origin_device=PEER)
    assert "crm" in _read_servers(root)
    assert (decoy / "mcp.json").read_bytes() == decoy_before


def test_a_write_outside_the_root_fails_hard_before_writing(
    root: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.mcp import config as mcp_config

    outside = tmp_path_factory.mktemp("outside") / "mcp.json"
    calls: list[Path] = []

    def _never(path: Path, document: dict[str, Any]) -> None:  # pragma: no cover
        calls.append(path)

    monkeypatch.setattr(mcpdefs, "_global_path", lambda _root: outside)
    monkeypatch.setattr(mcp_config, "write_scope_document", _never)
    with pytest.raises(RuntimeError, match="outside the root"):
        mcpdefs._write_document(root, {"mcpServers": {}})
    assert calls == []
    assert not outside.exists()


def test_the_containment_guard_resolves_both_sides(root: Path) -> None:
    mcpdefs._assert_inside_root(root / "mcp.json", root)
    with pytest.raises(RuntimeError):
        mcpdefs._assert_inside_root(root.parent / "elsewhere.json", root)


# ---------------------------------------------------------------------------
# State, rows, and the push (fake link)
# ---------------------------------------------------------------------------


def test_server_state_is_a_digest_manifest(root: Path) -> None:
    _write_servers(root, {"crm": {"type": "http", "url": "https://example.test/mcp"}})
    state = mcpdefs.server_state(root)
    row = mcpdefs.local_bundle(root)["servers"][0]
    assert state == {"servers": {"crm": definitions.digest_of(row)}}


def test_an_unedited_mirror_reports_its_origins_digest_an_edited_one_the_disk_form(
    root: Path,
) -> None:
    """The held-value asymmetry, pinned: the mirror cannot hash to its source
    form, so an unedited mirror reports the SOURCE digest (its origin sees the
    revision as arrived) while an edited mirror reports the disk digest (the
    origin sees a conflict to report, not silence).
    """
    bundle = _bundle(
        [
            {
                "kind": "server",
                "name": "gl",
                "transport": "stdio",
                "raw": {
                    "type": "stdio",
                    "command": "npx",
                    "args": [],
                    "env": {"PLAIN": "literal-held"},
                },
            }
        ]
    )
    mcpdefs.apply_bundle(root, bundle, origin_device=PEER)
    source_digest = definitions.digest_of(bundle["servers"][0])
    assert mcpdefs.server_state(root) == {"servers": {"gl": source_digest}}

    # An edit to a CARRIED field (a held-value edit is invisible by design —
    # values do not travel, so the canonical row does not change).
    document = _read_servers(root)
    document["gl"]["command"] = "other-command"
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": document}, indent=2) + "\n", encoding="utf-8"
    )
    state = mcpdefs.server_state(root)
    disk_row = mcpdefs.server_row("gl", document["gl"])
    assert state == {"servers": {"gl": definitions.digest_of(disk_row)}}
    assert state["servers"]["gl"] != source_digest


def test_state_rows_mark_origins_and_missing_reference_keys(root: Path) -> None:
    # ``borrowed`` does not exist here yet: the bundle installs it as a MIRROR,
    # then a second, authored server joins it in the same file. The reference
    # rides a HEADER — the only fields the secret-ref parser classifies (a url
    # is carried verbatim and is not a reference site).
    mcpdefs.apply_bundle(
        root,
        _bundle([_row("borrowed", headers={"Authorization": "ref:B_KEY"})]),
        origin_device=PEER,
    )
    document = _read_servers(root)
    document["mine"] = {"type": "stdio", "command": "x", "env": {"MINE": "${MINE}"}}
    (root / "mcp.json").write_text(
        json.dumps({"mcpServers": document}, indent=2) + "\n", encoding="utf-8"
    )
    rows = {row["name"]: row for row in mcpdefs.state_rows(root)}
    assert rows["borrowed"]["origin"] == PEER
    assert rows["mine"]["origin"] == ""
    borrowed_refs = {ref["id"]: ref["set"] for ref in rows["borrowed"]["refs"]}
    assert borrowed_refs == {"B_KEY": False}
    mine_refs = {ref["id"]: ref["set"] for ref in rows["mine"]["refs"]}
    assert mine_refs == {"MINE": False}


def test_state_rows_mark_a_row_that_will_never_travel(root: Path) -> None:
    """D4 (design round 1): a shape-tripping row rendered identical to a clean
    one, so the local ledger gave no signal for "this row will never reach any
    peer" — the operator learned it only by pushing, or never, via the cadence.
    """
    _write_servers(
        root,
        {
            "fine": {"type": "http", "url": "https://fine.example/mcp"},
            "leaky": {"type": "http", "url": "https://x.example/ghp_" + "a" * 36},
        },
    )
    rows = {row["name"]: row for row in mcpdefs.state_rows(root)}
    assert rows["fine"]["withheld"] == ""
    assert rows["leaky"]["withheld"] == "github-token"


class _FakeLink:
    def __init__(self, *, capabilities: Any, replies: list[Any]) -> None:
        self.capabilities = frozenset(capabilities)
        self.device_id = PEER
        self.network_id = "n_" + "0" * 24
        self.epoch = 1
        self._replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], *, timeout: float) -> Any:
        self.requests.append(frame)
        return self._replies.pop(0)


class _FakeServer:
    def __init__(self, root: Path, link: _FakeLink | None) -> None:
        self.root = root
        self._link = link
        self.identity = SimpleNamespace(device_id=OWNER)
        self._req = 0

    def _ensure_link(self, device_id: str) -> Any:
        return self._link

    def _member_name(self, device_id: str) -> str:
        return "device-b"

    def _next_relay_req(self) -> int:
        self._req += 1
        return self._req


def _fake_server(root: Path, link: Any = None) -> Any:
    """A duck-typed server for the pure push/step cells (never a real relay).

    ``Any`` on purpose: the seams take a ``RelayServer``, and these cells drive
    them with the four attributes they actually read (``root``,
    ``_ensure_link``, ``_member_name``, ``_next_relay_req``) — a real relay here
    would make every cell a link test, which is the link file's job.
    """
    return _FakeServer(root, link)


def _apply_reply(**summary: Any) -> dict[str, Any]:
    base = {"installed": [], "updated": [], "unchanged": [], "conflicts": [], "refused": []}
    base.update(summary)
    return {"op": "ack", "req": 2, "detail": {"phase": "apply", **base}}


def test_push_reports_in_sync_without_sending_a_bundle(root: Path) -> None:
    _write_servers(root, {"crm": {"type": "http", "url": "https://example.test/mcp"}})
    digest = definitions.digest_of(mcpdefs.local_bundle(root)["servers"][0])
    link = _FakeLink(
        capabilities=wire.LINK_CAPABILITIES,
        replies=[{"op": "ack", "req": 1, "detail": {"phase": "state", "servers": {"crm": digest}}}],
    )
    result = mcpdefs.push_to_peer(_fake_server(root, link), PEER)
    assert result["code"] == "in_sync" and result["ok"] is True
    assert len(link.requests) == 1, "an in-sync peer must cost one round trip"


def test_push_sends_only_missing_rows(root: Path) -> None:
    _write_servers(
        root,
        {
            "kept": {"type": "http", "url": "https://kept.example/mcp"},
            "sent": {"type": "http", "url": "https://sent.example/mcp"},
        },
    )
    kept_digest = {
        row["name"]: definitions.digest_of(row) for row in mcpdefs.local_bundle(root)["servers"]
    }["kept"]
    link = _FakeLink(
        capabilities=wire.LINK_CAPABILITIES,
        replies=[
            {"op": "ack", "req": 1, "detail": {"phase": "state", "servers": {"kept": kept_digest}}},
            _apply_reply(installed=[{"kind": "server", "name": "sent"}]),
        ],
    )
    result = mcpdefs.push_to_peer(_fake_server(root, link), PEER)
    assert result["code"] == "applied" and result["ok"] is True
    # D2 (design round 1): the receipt is a person-facing register — "sent 1
    # MCP server definition", never "definition(s)".
    assert result["message"] == "sent 1 MCP server definition"
    frames = [frame for frame in link.requests if frame["phase"] == "apply"]
    assert [row["name"] for row in frames[0]["bundle"]["servers"]] == ["sent"]


def test_push_to_an_old_peer_is_capability_first_and_pays_no_request(root: Path) -> None:
    """The measured trap: an unknown op must NOT become a slow-op timeout.

    The capability gate runs BEFORE any request is sent, so the peer's 60 s
    deadline is never touched — structurally pinned by ``link.requests`` being
    empty rather than by a wall-clock bound.
    """
    _write_servers(root, {"crm": {"type": "http", "url": "https://example.test/mcp"}})
    old_capabilities = [cap for cap in wire.LINK_CAPABILITIES if cap != wire.MCP_DEFS_V1]
    link = _FakeLink(capabilities=old_capabilities, replies=[])
    result = mcpdefs.push_to_peer(_fake_server(root, link), PEER)
    assert result["code"] == "peer_too_old" and result["ok"] is False
    assert link.requests == [], "an old peer was asked anyway"
    assert "lop-update" in result["message"]


def test_push_reports_unreachable_when_there_is_no_link(root: Path) -> None:
    result = mcpdefs.push_to_peer(_fake_server(root, None), PEER)
    assert result["code"] == "unreachable"


def test_push_surfaces_a_peer_refusal_with_its_code(root: Path) -> None:
    _write_servers(root, {"crm": {"type": "http", "url": "https://example.test/mcp"}})
    link = _FakeLink(
        capabilities=wire.LINK_CAPABILITIES,
        replies=[
            {
                "op": "error",
                "req": 1,
                "code": "not_authorised",
                "message": "this device may not install MCP servers on that one",
            }
        ],
    )
    result = mcpdefs.push_to_peer(_fake_server(root, link), PEER)
    assert (result["code"], result["ok"]) == ("not_authorised", False)


def test_push_to_peer_files_no_answer_off_the_policy_codes(root: Path) -> None:
    """A NO-ANSWER IS A TRANSPORT FAILURE, NOT A POLICY ANSWER (definitions' Q-1).

    ``link.request`` returning None is a timeout or a dead-link send — nobody
    answered — and its old collapse into the ``refused`` default below filed it
    with the ANSWERED refusals, which PARK the member for
    REFUSED_MIN_INTERVAL_S (1800 s) with no retry and no log line. Read from the
    real ``push_to_peer`` rather than the syncer, because the collapse lived in
    the mapping: the ``refused`` default below it remains for the codeless
    ANSWERED case, and that half must keep parking.
    """
    link = _FakeLink(capabilities=wire.LINK_CAPABILITIES, replies=[None])
    result = mcpdefs.push_to_peer(_fake_server(root, link), PEER)
    assert result["code"] == "no_answer", result
    assert "no_answer" not in definitions.POLICY_REFUSAL_CODES
    assert "refused" in definitions.POLICY_REFUSAL_CODES  # the answered case still parks


def test_push_reports_a_conflict_by_name(root: Path) -> None:
    _write_servers(root, {"crm": {"type": "http", "url": "https://example.test/mcp"}})
    link = _FakeLink(
        capabilities=wire.LINK_CAPABILITIES,
        replies=[
            {"op": "ack", "req": 1, "detail": {"phase": "state", "servers": {}}},
            _apply_reply(
                conflicts=[{"kind": "server", "name": "crm", "reason": "authored there already"}]
            ),
        ],
    )
    result = mcpdefs.push_to_peer(_fake_server(root, link), PEER)
    assert result["code"] == "conflict" and result["ok"] is False
    assert "crm" in result["message"]


# ---------------------------------------------------------------------------
# The cadence step and the seam it rides
# ---------------------------------------------------------------------------


def test_the_step_skips_a_member_without_the_capability(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(definitions, "unholdable_capability", lambda *_a: "admin")
    assert mcpdefs.mesh_tick_step(_fake_server(root, None), PEER) == "skipped:no_admin"


def test_the_step_answers_in_sync_without_dialing_when_there_is_nothing_to_send(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(definitions, "unholdable_capability", lambda *_a: "")

    def _must_not_run(*_a: Any, **_k: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("the step dialed for an empty bundle")

    monkeypatch.setattr(mcpdefs, "push_to_peer", _must_not_run)
    assert mcpdefs.mesh_tick_step(_fake_server(root, None), PEER) == "in_sync"


def test_the_step_returns_the_push_code(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_servers(root, {"gl": {"type": "stdio", "command": "npx"}})
    monkeypatch.setattr(definitions, "unholdable_capability", lambda *_a: "")
    monkeypatch.setattr(mcpdefs, "push_to_peer", lambda _s, _d: {"code": "applied"})
    assert mcpdefs.mesh_tick_step(_fake_server(root, None), PEER) == "applied"


def _member_record(root: Path, *, self_role: str) -> Any:
    """One network on disk with a peer member and NO endpoints.

    The cadence walks member RECORDS (not links), so a record is all a tick
    needs; the peer having no endpoints makes the definitions push resolve to
    ``unreachable`` without a socket, which keeps these cells hermetic.
    """
    from local_operator.network import identity, relay, store
    from local_operator.network import wire as wire_mod

    own = identity.mint(root, name="cadence-a")
    other = identity.mint(root / "b", name="cadence-b")
    network_id = store.new_network_id()
    record = types.NetworkRecord(
        network_id=network_id,
        name="cadence",
        epoch=1,
        created_by=own.device_id,
        self_device_id=own.device_id,
        self_role=self_role,
        self_capabilities=sorted(types.capabilities_for_role(self_role)),
        listen={"address": "127.0.0.1", "port": 1, "advertised": []},
    )
    relay.admit(
        record,
        device_id=own.device_id,
        public_key=own.public_key,
        name=own.name,
        role=self_role,
        capabilities=sorted(types.capabilities_for_role(self_role)),
        added_by=own.device_id,
        added_via="self",
        endpoints=[],
        root=root,
        persist=False,
    )
    relay.admit(
        record,
        device_id=other.device_id,
        public_key=other.public_key,
        name=other.name,
        role="drive",
        capabilities=sorted(types.capabilities_for_role("drive")),
        added_by=own.device_id,
        added_via="invite",
        endpoints=[],
        root=root,
        persist=False,
    )
    store.save(record, root)
    store.save_secrets(
        types.SecretState(network_id=network_id, epoch=1, secret=wire_mod.b64u(bytes(32))), root
    )
    return own


def test_add_tick_step_is_idempotent(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(definitions, "_TICK_STEPS", [])

    def _step(_server: Any, _device_id: str) -> str:
        return ""

    definitions.add_tick_step(_step)
    definitions.add_tick_step(_step)
    assert definitions._tick_steps() == (_step,)


def test_a_step_runs_after_the_push_and_a_policy_code_parks_the_member(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(definitions, "_TICK_STEPS", [])
    seen: list[str] = []

    def _step(_server: Any, device_id: str) -> str:
        seen.append(device_id)
        return "capability_denied"

    definitions.add_tick_step(_step)
    own = _member_record(root, self_role="admin")
    pushes: list[str] = []

    def _push(server: Any, device_id: str, **fields: Any) -> dict[str, Any]:
        pushes.append(device_id)
        return {"ok": False, "code": "unreachable", "message": "not answering"}

    monkeypatch.setattr(definitions, "push_to_peer", _push)
    server = SimpleNamespace(root=root, identity=own)
    syncer = definitions.DefinitionsSyncer(server)  # type: ignore[arg-type]
    syncer.tick(now=1000.0)
    assert seen and pushes, "the step must run after the member's push attempt"
    # The policy code from the STEP parked the member on the refused floor: a
    # tick five minutes later asks again, a tick one second later does not.
    outcomes = syncer.tick(now=1001.0)
    assert all(device_id != pushes[0] for device_id, _code in outcomes), outcomes
    assert len(pushes) == 1
    syncer.tick(now=1000.0 + definitions.REFUSED_MIN_INTERVAL_S + 1)
    assert len(pushes) == 2


def test_an_unanswered_push_is_not_a_refusal_and_keeps_the_fast_retry(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NO-ANSWER IS A TRANSPORT FAILURE, NOT A POLICY ANSWER (definitions' Q-1).

    The same collapse as definitions': ``link.request`` returning None (a
    timeout or a dead link) fell into the ``refused`` default, ``refused`` is a
    policy code, and the syncer parked the member for REFUSED_MIN_INTERVAL_S
    with no retry and no log line. This cell drives the REAL step over the REAL
    push and a link that never answers, so the mapping and the fast retry are
    pinned end to end: one tick later the cadence asks again, exactly as it
    does after ``unreachable``.
    """
    monkeypatch.setattr(definitions, "_TICK_STEPS", [])
    _write_servers(root, {"gl": {"type": "stdio", "command": "npx"}})
    link = _FakeLink(capabilities=wire.LINK_CAPABILITIES, replies=[None, None])
    monkeypatch.setattr(definitions, "unholdable_capability", lambda *_a: "")
    # The syncer's OWN push is not under test here; keep it a transient failure
    # so the only refusal-shaped answer could be the step's.
    monkeypatch.setattr(
        definitions,
        "push_to_peer",
        lambda _s, _d, **fields: {"ok": False, "code": "unreachable", "message": "not answering"},
    )
    definitions.add_tick_step(mcpdefs.mesh_tick_step)
    syncer = definitions.DefinitionsSyncer(_fake_server(root, link))
    monkeypatch.setattr(syncer, "_targets", lambda: [PEER])
    syncer.tick(now=1000.0)
    assert len(link.requests) == 1, "the step must have dialed once"
    assert syncer._refused_at.get(PEER, 0.0) == 0.0, "a no-answer must not park"
    # One tick later it asks again — the same fast retry a transport failure gets.
    syncer.tick(now=1016.0)
    assert len(link.requests) == 2, "a no-answer must be re-contacted on the next tick"


def test_a_refused_mcp_push_still_parks_the_member(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AN ANSWERED REFUSAL STILL PARKS (the other half of definitions' Q-1).

    The codeless ANSWERED case keeps the ``refused`` default, and ``refused``
    is a policy code: the no-answer fix must not un-park it. The answer to "may
    I write MCP servers here" does not change between ticks, so the member
    waits REFUSED_MIN_INTERVAL_S — then re-attempts, because a capability
    granted on the peer's side must be discovered rather than hidden.
    """
    monkeypatch.setattr(definitions, "_TICK_STEPS", [])
    _write_servers(root, {"gl": {"type": "stdio", "command": "npx"}})
    link = _FakeLink(
        capabilities=wire.LINK_CAPABILITIES,
        replies=[{"op": "error", "req": 1}, {"op": "error", "req": 2}],
    )
    monkeypatch.setattr(definitions, "unholdable_capability", lambda *_a: "")
    monkeypatch.setattr(
        definitions,
        "push_to_peer",
        lambda _s, _d, **fields: {"ok": False, "code": "unreachable", "message": "not answering"},
    )
    definitions.add_tick_step(mcpdefs.mesh_tick_step)
    syncer = definitions.DefinitionsSyncer(_fake_server(root, link))
    monkeypatch.setattr(syncer, "_targets", lambda: [PEER])
    syncer.tick(now=1000.0)
    assert len(link.requests) == 1
    assert syncer._refused_at.get(PEER, 0.0) == 1000.0 + definitions.REFUSED_MIN_INTERVAL_S
    # Parked: every tick in the next half hour asks nobody ...
    assert syncer.tick(now=1016.0) == []
    syncer.tick(now=2000.0)
    assert len(link.requests) == 1, "a refusal must park the member, not re-ask every tick"
    # ... and half an hour later it asks again — a re-attempt, not a permanent skip.
    syncer.tick(now=1000.0 + definitions.REFUSED_MIN_INTERVAL_S + 16.0)
    assert len(link.requests) == 2


# ---------------------------------------------------------------------------
# The sync verb's own receipt (D2 — the register a person reads)
# ---------------------------------------------------------------------------


def test_the_sync_receipt_pluralises_by_device_count(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2 (design round 1): "1 of 1 device(s) hold" was the register defect.

    The verb agrees with the DEVICE count, and the singular is the form a
    one-peer mesh actually reads: "1 of 1 device holds", "0 of 1 device holds".
    """
    own = _member_record(root, self_role="admin")
    server = SimpleNamespace(root=root, identity=own)
    monkeypatch.setattr(mcpdefs, "push_to_peer", lambda _s, _d: {"ok": True, "code": "applied"})
    handler = mcpdefs.local_sync_handler(server)  # type: ignore[arg-type]
    detail = handler({"peer": ""})
    assert detail["message"] == "1 of 1 device holds this device's MCP servers"
    monkeypatch.setattr(
        mcpdefs, "push_to_peer", lambda _s, _d: {"ok": False, "code": "unreachable"}
    )
    detail = handler({"peer": ""})
    assert detail["ok"] is False
    assert detail["message"] == "0 of 1 device holds this device's MCP servers"
