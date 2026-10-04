"""Peer readiness, unit half: the evaluators, their sentences and their codes.

The link-level half (two real relays) lives in ``test_readiness_link.py``; this
file drives the pure functions — the peer-side fact collectors and the L-side
verdict builders — over fakes and tmp roots, because those are where the
verdict MATRIX lives and a matrix belongs in cells, not in one end-to-end run.

Two properties are pinned here rather than only observed in the link tests:

* every row that could not be established is ``ok: False`` with a reason
  ("unknown is not ok" — an absent answer is never rendered as a pass);
* the refusal/silence discriminator: a connection that was REFUSED reads as
  "the host is up; nothing is listening", never as "nothing answered" — the
  conflation this report exists to end, and the reason its renderer is its own
  function rather than ``resume.doctor_detail_words``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import readiness, wire

PEER = "d_" + "b" * 32
VIEWER = "d_" + "a" * 32
THIRD = "d_" + "c" * 32


def _member(device_id: str = PEER, name: str = "cloud-node-1") -> SimpleNamespace:
    return SimpleNamespace(device_id=device_id, name=name)


class _FakeViewer:
    """The viewer-side facts a verdict builder reads, as plain answers."""

    def __init__(
        self,
        *,
        rows: list[str] | None = None,
        holds_mcp: bool | None = False,
        placement: dict[str, Any] | None = None,
        device_id: str = VIEWER,
    ) -> None:
        self.device_id = device_id
        self._rows = rows
        self._holds_mcp = holds_mcp
        self._placement = dict(placement or {})

    def provider_rows(self, provider: str) -> list[str] | None:
        return self._rows

    def holds_mcp_login(self, url: str) -> bool | None:
        return self._holds_mcp

    def placement(self, *, provider: str = "", mcp_url: str = "") -> dict[str, Any]:
        return dict(self._placement)

    def close(self) -> None:  # pragma: no cover — duck-typed parity
        pass


def _facts(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": readiness.READINESS_SCHEMA,
        "default_model": {
            "hosting": "openai",
            "model_name": "gpt-5",
            "resolved": True,
            "reason": "",
        },
        "provider": "openai",
        "has_local": False,
        "credential_placement": {},
        "mcp": {"servers": [], "withheld": [], "config_path": "/home/x/.local-operator/mcp.json"},
        "git": {"user_name": "", "user_email": ""},
        "operator": {"level": "spawn-capability-only", "reason": "no anchor is installed"},
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# (a) Operator authority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fact", "ok", "code"),
    [
        ({"level": "operator-presence", "reason": "ok"}, True, ""),
        ({"level": "operator-file-only", "reason": "ok"}, True, ""),
        (
            {"level": "spawn-capability-only", "reason": "no anchor is installed"},
            False,
            readiness.CODE_NOT_INSTALLED,
        ),
        (
            {"level": "anchor-unpinned", "reason": "the anchor is owned by uid 501, not root"},
            False,
            readiness.CODE_ANCHOR_UNPINNED,
        ),
        (
            {"level": "unreported", "reason": "the anchor is not readable JSON"},
            False,
            readiness.CODE_UNUSABLE,
        ),
    ],
    ids=["presence", "file-only", "spawn-only", "unpinned", "unreported"],
)
def test_operator_levels_map_to_codes(fact: dict[str, Any], ok: bool, code: str) -> None:
    row = readiness.operator_row(_member(), _facts(operator=fact), peer_label="cloud-node-1")
    assert row["ok"] is ok, row
    assert row.get("code", "") == code, row
    assert row["capability"] == readiness.CAPABILITY_OPERATOR_AUTHORITY
    if not ok:
        assert any("ask Local Operator to set up" in remedy for remedy in row["remedies"]), row
        assert "parks" in row["detail"], row
    if ok:
        assert row["remedies"] == []


def test_operator_file_only_carries_the_presence_caveat() -> None:
    row = readiness.operator_row(
        _member(),
        _facts(operator={"level": "operator-file-only", "reason": "ok"}),
        peer_label="box",
    )
    assert row["ok"] is True
    assert "0600 file" in row["detail"] and "presence is not enforced" in row["detail"]


def test_operator_not_installed_names_a_staged_anchor() -> None:
    row = readiness.operator_row(
        _member(),
        _facts(
            operator={
                "level": "spawn-capability-only",
                "reason": "no anchor is installed, so the runtime trusts no key yet",
                "anchor_installed": True,
                "anchor_root_owned": False,
            }
        ),
        peer_label="cloud-node-1",
    )
    assert row["ok"] is False
    assert "staged but not installed" in row["detail"], row


def test_operator_row_is_unknown_when_the_answer_lacked_the_section() -> None:
    row = readiness.operator_row(_member(), _facts(operator=None), peer_label="cloud-node-1")
    assert row["ok"] is False
    assert row["code"] == readiness.CODE_UNKNOWN
    assert "did not carry this check" in row["detail"]


# ---------------------------------------------------------------------------
# (c) git identity — the file read, git's own precedence
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_git_identity_reads_both_global_files_with_git_precedence(tmp_path: Path) -> None:
    """``~/.config/git/config`` is read first and ``~/.gitconfig`` wins."""
    _write(
        tmp_path / ".config" / "git" / "config",
        "[user]\n\tname = XDG Name\n\temail = xdg@example.test\n",
    )
    _write(tmp_path / ".gitconfig", "[user]\n\tname = Home Name\n\temail = home@example.test\n")
    fact = readiness.git_identity_fact(home=tmp_path)
    assert fact == {"user_name": "Home Name", "user_email": "home@example.test"}

    (tmp_path / ".gitconfig").unlink()
    assert readiness.git_identity_fact(home=tmp_path) == {
        "user_name": "XDG Name",
        "user_email": "xdg@example.test",
    }


def test_git_identity_skips_an_unparsable_file_and_reads_the_other(tmp_path: Path) -> None:
    _write(
        tmp_path / ".config" / "git" / "config", "[user]\n\tname = Kept\n\temail = kept@e.test\n"
    )
    _write(tmp_path / ".gitconfig", "[user\n\tthis is not gitconfig at all =\n")
    assert readiness.git_identity_fact(home=tmp_path) == {
        "user_name": "Kept",
        "user_email": "kept@e.test",
    }


def test_git_identity_is_absent_not_wrong_when_no_files_exist(tmp_path: Path) -> None:
    assert readiness.git_identity_fact(home=tmp_path) == {"user_name": "", "user_email": ""}


def test_git_row_says_what_commits_will_do() -> None:
    ok_row = readiness.git_row(
        _member(),
        _facts(git={"user_name": "Damian Tran", "user_email": "damian@gominerva.com"}),
        peer_label="cloud-node-1",
    )
    assert ok_row["ok"] is True
    assert "commits as Damian Tran <damian@gominerva.com>" in ok_row["detail"]
    assert "push credentials are a separate question" in ok_row["detail"]
    # The github clause, per the desk call (2026-10-03): the ACTUAL state in the
    # reader's terms — GitHub push waits on a configured App, one step — never
    # "implemented" and never "blocked".
    assert "GitHub push through the mesh waits on a configured GitHub App" in ok_row["detail"]

    bad_row = readiness.git_row(
        _member(), _facts(git={"user_name": "", "user_email": "someone@e.test"}), peer_label="box"
    )
    assert bad_row["ok"] is False
    assert bad_row["code"] == readiness.CODE_NO_GIT_IDENTITY
    assert "user.name" in bad_row["detail"]
    assert any("git config --global user.name" in remedy for remedy in bad_row["remedies"])


def test_git_row_fills_the_remedy_with_this_devices_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design §3: the one-time setup is surfaced with REAL values — this device's
    own global git config — because the suggestion is the point. It stays a
    suggestion: the operator may want a different identity on that device, and
    ``ready`` still writes nothing on either side."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write(
        tmp_path / ".gitconfig",
        "[user]\n\tname = Damian Tran\n\temail = damian@gominerva.com\n",
    )
    row = readiness.git_row(
        _member(), _facts(git={"user_name": "", "user_email": ""}), peer_label="cloud-node-1"
    )
    remedy = " ".join(row["remedies"])
    assert '`git config --global user.name "Damian Tran"`' in remedy
    assert '`git config --global user.email "damian@gominerva.com"`' in remedy
    assert "…" not in remedy


def test_git_row_keeps_the_placeholder_for_values_this_device_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value this device does not have keeps the shipped ``"…"`` placeholder —
    inventing one for the peer would be a suggestion that cannot run."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _write(tmp_path / ".gitconfig", "[user]\n\tname = Damian Tran\n")
    row = readiness.git_row(
        _member(), _facts(git={"user_name": "", "user_email": "someone@e.test"}), peer_label="box"
    )
    remedy = " ".join(row["remedies"])
    assert '`git config --global user.name "Damian Tran"`' in remedy
    assert '`git config --global user.email "…"`' in remedy


# ---------------------------------------------------------------------------
# (d)+(f) the user-scope MCP surface
# ---------------------------------------------------------------------------


def test_mcp_servers_fact_reads_the_user_scope_file_only(tmp_path: Path) -> None:
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "notion": {"type": "http", "url": "https://mcp.notion.com/mcp"},
                    "fs": {"command": "npx", "args": ["-y", "x"]},
                }
            }
        ),
        encoding="utf-8",
    )
    fact = readiness.mcp_servers_fact(tmp_path)
    by_name = {row["name"]: row for row in fact["servers"]}
    assert by_name["notion"]["transport"] == "http"
    assert by_name["notion"]["url"] == "https://mcp.notion.com/mcp"
    assert by_name["fs"]["transport"] == "stdio"
    assert fact["withheld"] == []
    # No store exists here, so a http/sse row is a definite False and the run
    # created nothing (the read-only promise).
    assert by_name["notion"]["has_row"] is False
    assert not (tmp_path / "auth.db").exists()

    assert readiness.mcp_servers_fact(tmp_path / "missing-root")["servers"] == []


def test_mcp_servers_fact_withholds_and_names_a_shaped_row(tmp_path: Path) -> None:
    """A row whose text LOOKS like a credential is withheld and named."""
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "leaky": {"type": "http", "url": "https://user:s3cr3t-token-abcd@e.test/mcp"},
                    "fine": {"type": "http", "url": "https://mcp.linear.app/mcp"},
                }
            }
        ),
        encoding="utf-8",
    )
    fact = readiness.mcp_servers_fact(tmp_path)
    assert [row["name"] for row in fact["servers"]] == ["fine"]
    assert fact["withheld"] == ["leaky"]
    serialised = json.dumps(fact)
    assert "s3cr3t" not in serialised and "user:" not in serialised


def test_mcp_has_row_is_three_valued_and_never_reads_as_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"notion": {"type": "http", "url": "https://mcp.notion.com/mcp"}}}
        ),
        encoding="utf-8",
    )
    rows = [SimpleNamespace(id=1, identity_key="https://mcp.notion.com/mcp", updated_at=0)]

    class _Store:
        def __init__(self, outcome: Any) -> None:
            self.outcome = outcome

        def list_credentials(self, provider: str) -> Any:
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome

        def close(self) -> None:
            pass

    monkeypatch.setattr(readiness, "_open_store", lambda root: _Store(rows))
    assert readiness.mcp_servers_fact(tmp_path)["servers"][0]["has_row"] is True

    monkeypatch.setattr(readiness, "_open_store", lambda root: _Store(OSError("unreadable")))
    assert readiness.mcp_servers_fact(tmp_path)["servers"][0]["has_row"] is None


def test_a_missing_store_is_read_without_creating_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_open_store``'s absence answer is a definite False and must not be
    re-resolved into an ambient store: ``McpTokenStorage(url, store=None)`` builds
    ``AuthStore()`` for the process's ambient root — which CREATES the database,
    so a read-only report would write one, and read the wrong root whenever the
    two differ (found by the shareability ledger's creates-nothing cell)."""
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"notion": {"type": "http", "url": "https://mcp.notion.com/mcp"}}}
        ),
        encoding="utf-8",
    )
    ambient = tmp_path / "ambient-config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(ambient))
    fact = readiness.mcp_servers_fact(tmp_path)
    assert fact["servers"][0]["has_row"] is False
    assert not (ambient / "auth.db").exists()


def test_an_unopenable_store_answers_could_not_be_read(tmp_path: Path) -> None:
    """An ``auth.db`` that exists but cannot be opened is "could not be read",
    never a crash (convergence round 2, MAJOR class): every open keeps
    ``_open_store``'s absence answer (definite False / no rows) for an ABSENT
    store, and answers the tri-state ``None`` for one that fails to open — the
    same answer a store that fails mid-read already gets.
    """
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"notion": {"type": "http", "url": "https://mcp.notion.com/mcp"}}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "auth.db").write_bytes(b"not a database")
    fact = readiness.mcp_servers_fact(tmp_path)
    assert fact["servers"][0]["has_row"] is None
    assert readiness.has_local_provider_credential(tmp_path, "openai") is None
    viewer = readiness.ViewerFacts(tmp_path)
    try:
        assert viewer.holds_mcp_login("https://mcp.notion.com/mcp") is None
        assert viewer.provider_rows("openai") is None
    finally:
        viewer.close()


def test_mcp_servers_row_names_the_missing_file(tmp_path: Path) -> None:
    fact = readiness.mcp_servers_fact(tmp_path / "none")
    row = readiness.mcp_servers_row(_member(), _facts(mcp=fact), peer_label="cloud-node-1")
    assert row["ok"] is False
    assert row["code"] == readiness.CODE_NO_MCP_SERVERS
    assert "no user-scope MCP servers" in row["detail"]
    remedy = " ".join(row["remedies"])
    assert "/mcp add" in remedy
    # D1 (design round 1): the remedy must name the verb that makes these rows
    # travel, and must not keep the clause this slice falsified ("server config
    # is per device and is not copied over the mesh") — a shipped surface may
    # not lie about its own sibling verb.
    assert "lop network mcp push --peer cloud-node-1" in remedy
    assert "per device" not in remedy and "not copied" not in remedy


@pytest.mark.parametrize(("count", "phrase"), [(1, "1 row withheld"), (2, "2 rows withheld")])
def test_mcp_servers_row_pluralises_its_withheld_tail(
    tmp_path: Path, count: int, phrase: str
) -> None:
    """Round 2, D5: the withheld tail is a person-facing register — "1 row",
    never "1 row(s)" (the same class as D4, one branch below it)."""
    fact = readiness.mcp_servers_fact(tmp_path / "none")
    fact["withheld"] = [f"shaped-{index}" for index in range(count)]
    row = readiness.mcp_servers_row(_member(), _facts(mcp=fact), peer_label="cloud-node-1")
    assert phrase + " from this report" in row["detail"]
    assert "row(s)" not in row["detail"]


def test_mcp_credential_rows_walk_the_verdict_chain(tmp_path: Path) -> None:
    servers = {
        "servers": [
            {
                "name": "mine",
                "url": "https://a.example/mcp",
                "transport": "http",
                "auth_declared": True,
                "has_row": True,
            },
            {
                "name": "borrowed",
                "url": "https://b.example/mcp",
                "transport": "http",
                "auth_declared": True,
                "has_row": False,
                "has_placement": True,
            },
            {
                "name": "shared-to-me",
                "url": "https://c.example/mcp",
                "transport": "sse",
                "auth_declared": True,
                "has_row": False,
                "has_placement": True,
            },
            {
                # SHARED ON THIS DEVICE'S BOOKS, NOT PULLED THERE: the owner's
                # document says the peer is a holder, the peer's own answer says
                # it has no placement entry — the row must report the GAP.
                "name": "not-pulled",
                "url": "https://i.example/mcp",
                "transport": "http",
                "auth_declared": True,
                "has_row": False,
                "has_placement": False,
            },
            {
                "name": "unshared",
                "url": "https://d.example/mcp",
                "transport": "http",
                "auth_declared": True,
                "has_row": False,
            },
            {
                "name": "we-hold",
                "url": "https://e.example/mcp",
                "transport": "http",
                "auth_declared": True,
                "has_row": False,
            },
            {
                "name": "unreadable",
                "url": "https://f.example/mcp",
                "transport": "http",
                "auth_declared": True,
                "has_row": None,
            },
            {
                "name": "declared",
                "url": "https://g.example/mcp",
                "transport": "http",
                "auth_declared": True,
                "has_row": False,
            },
            {
                "name": "soft",
                "url": "https://h.example/mcp",
                "transport": "http",
                "auth_declared": False,
                "has_row": False,
            },
            {
                "name": "stdio-one",
                "url": "",
                "transport": "stdio",
                "auth_declared": False,
                "has_row": None,
            },
        ],
        "withheld": [],
        "config_path": "/r/mcp.json",
    }

    class _PlacementViewer(_FakeViewer):
        def __init__(self, entries: dict[str, dict[str, Any]], *, holds_urls: set[str]) -> None:
            super().__init__(holds_mcp=False)
            self._entries = entries
            self._holds_urls = holds_urls

        def placement(self, *, provider: str = "", mcp_url: str = "") -> dict[str, Any]:
            return dict(self._entries.get(mcp_url, {}))

        def holds_mcp_login(self, url: str) -> bool | None:
            # Per-URL: "this device holds a login for the url" is a fact about
            # ONE server, and a viewer-wide True made every empty case read as
            # "we hold it, not shared".
            return url in self._holds_urls

    viewer = _PlacementViewer(
        {
            "https://b.example/mcp": {
                "owner_device": VIEWER,
                "holders": [{"device": PEER, "scope": "session"}],
            },
            "https://c.example/mcp": {
                "owner_device": THIRD,
                "holders": [{"device": PEER, "scope": "session"}],
            },
            "https://d.example/mcp": {
                "owner_device": VIEWER,
                "holders": [{"device": THIRD, "scope": "session"}],
            },
            "https://i.example/mcp": {
                "owner_device": VIEWER,
                "holders": [{"device": PEER, "scope": "session"}],
            },
        },
        holds_urls={"https://e.example/mcp"},
    )
    rows = readiness.mcp_credential_rows(
        _member(), _facts(mcp=servers), viewer=viewer, peer_label="cloud-node-1"
    )
    by_server = {(row.get("observed") or {}).get("server", ""): row for row in rows}
    assert "stdio-one" not in by_server, "stdio servers are listed, not credential-checked"
    assert by_server["mine"]["ok"] is True
    assert by_server["borrowed"]["ok"] is True and "borrow" in by_server["borrowed"]["detail"]
    assert (
        by_server["shared-to-me"]["ok"] is True
        and "not verified from here" in by_server["shared-to-me"]["detail"]
    )
    assert (
        by_server["unshared"]["ok"] is False
        and by_server["unshared"]["code"] == readiness.CODE_NOT_SHARED
    )
    assert (
        "credential share mcp:https://d.example/mcp --with cloud-node-1"
        in by_server["unshared"]["remedies"][0]
    )
    assert by_server["not-pulled"]["ok"] is False
    assert by_server["not-pulled"]["code"] == readiness.CODE_NOT_SHARED
    assert "has not pulled the placement" in by_server["not-pulled"]["detail"]
    assert "lop network credentials" in by_server["not-pulled"]["remedies"][0]
    assert by_server["we-hold"]["ok"] is False
    assert "mcp:https://e.example/mcp" in by_server["we-hold"]["remedies"][0]
    assert (
        by_server["unreadable"]["ok"] is False
        and by_server["unreadable"]["code"] == readiness.CODE_UNKNOWN
    )
    assert "'/mcp login https://g.example/mcp' here first" in by_server["declared"]["detail"]
    assert by_server["soft"]["detail"].startswith("if the `soft` server needs a sign-in")
    for row in by_server.values():
        assert row["capability"] == readiness.CAPABILITY_MCP_CREDENTIAL


def test_shareable_lines_render_every_login_state_once() -> None:
    """One renderer for the CLI and the agent digest (design §2), so the two cannot
    drift the way the readiness rows once did.

    Both row kinds render (Radient org projection): MCP-server rows keep their
    three-valued login state, and a PROVIDER-LOGIN row — which only exists when the
    login is held — renders the held sentence plus the identity it was read from —
    and, for Radient only, its person-scope caution line (design review round 1,
    D1)."""
    rows = [
        {
            "server": "slack",
            "transport": "http",
            "login_here": True,
            "remedy": "lop network credential share mcp:https://h.example/mcp --with <device>",
            "shared_with": [{"device": "d_1", "name": "cloud-node-1", "scope": "session"}],
        },
        {
            "server": "notion",
            "transport": "sse",
            "login_here": False,
            "remedy": "run '/mcp login https://n.example/mcp' here first",
            "shared_with": [],
        },
        {"server": "odd", "transport": "http", "login_here": None, "remedy": "", "shared_with": []},
        {
            "provider": "radient",
            "kind": "oauth-rotating",
            "identity_label": "owner@example.test",
            "remedy": "lop network credential share radient --with <device>",
            "shared_with": [{"device": "d_2", "name": "cloud-node-1", "scope": "session"}],
        },
    ]
    assert readiness.shareable_lines(rows) == [
        "shareable here:",
        "  slack  http  login held — share: lop network credential share mcp:https://h.example/mcp"
        " --with <device>",
        "      shared with cloud-node-1 (session)",
        "  notion  sse  no login here yet — run '/mcp login https://n.example/mcp' here first",
        "  odd  http  login state not known — this device's credential store could not be read",
        "  radient  oauth-rotating  login held — share: lop network credential share radient"
        " --with <device>",
        "      organization account — share only to your own devices",
        "      signed in as owner@example.test",
        "      shared with cloud-node-1 (session)",
    ]
    # A provider row without a label renders the single line; the identity line is
    # optional, not a placeholder that prints as empty.
    assert readiness.shareable_lines(
        [
            {
                "provider": "openai",
                "kind": "api-key-static",
                "identity_label": "",
                "remedy": "lop network credential share openai --with <device>",
                "shared_with": [],
            }
        ]
    ) == [
        "shareable here:",
        "  openai  api-key-static  login held — share: lop network credential share openai"
        " --with <device>",
    ]
    # The Radient caution is not an identity line: it renders with the label
    # absent too. The openai cell above pins that no other provider carries it.
    assert readiness.shareable_lines(
        [
            {
                "provider": "radient",
                "kind": "oauth-rotating",
                "identity_label": "",
                "remedy": "lop network credential share radient --with <device>",
                "shared_with": [],
            }
        ]
    ) == [
        "shareable here:",
        "  radient  oauth-rotating  login held — share: lop network credential share radient"
        " --with <device>",
        "      organization account — share only to your own devices",
    ]
    assert readiness.shareable_lines([]) == []


# ---------------------------------------------------------------------------
# (b) build parity
# ---------------------------------------------------------------------------


def test_build_row_compares_versions_and_degrades() -> None:
    equal = readiness.build_row(
        _member(),
        peer_build={"version": "0.64.1", "source_ref": "abc"},
        own_build={"version": "0.64.1", "source_ref": "def"},
        peer_label="cloud-node-1",
    )
    assert equal["ok"] is True
    assert "0.64.1" in equal["detail"]

    behind = readiness.build_row(
        _member(),
        peer_build={"version": "0.63.2", "source_ref": "abc"},
        own_build={"version": "0.64.1", "source_ref": "def"},
        peer_label="cloud-node-1",
    )
    assert behind["ok"] is False and behind["code"] == readiness.CODE_BEHIND
    assert "runs 0.63.2" in behind["detail"] and "0.64.1" in behind["detail"]
    assert any(
        "ask Local Operator to update it there" in remedy and "cloud-node-1" in remedy
        for remedy in behind["remedies"]
    )

    ahead = readiness.build_row(
        _member(),
        peer_build={"version": "0.66.0", "source_ref": "abc"},
        own_build={"version": "0.64.1", "source_ref": "def"},
        peer_label="cloud-node-1",
    )
    assert ahead["ok"] is False and ahead["code"] == readiness.CODE_AHEAD
    assert any("this device" in remedy for remedy in ahead["remedies"])

    unknown = readiness.build_row(
        _member(), peer_build={}, own_build={"version": "0.64.1"}, peer_label="cloud-node-1"
    )
    assert unknown["ok"] is False and unknown["code"] == readiness.CODE_UNKNOWN
    assert "did not arrive" in unknown["detail"]

    unparsable = readiness.build_row(
        _member(),
        peer_build={"version": "0.28.0rc1"},
        own_build={"version": "0.64.1"},
        peer_label="cloud-node-1",
    )
    assert unparsable["ok"] is False and unparsable["code"] == readiness.CODE_UNKNOWN


def test_compare_builds_is_the_one_version_ordering() -> None:
    """The ordering ``peers`` and ``ready`` share, so the two cannot disagree about
    which side is behind; unknown covers absent, unparsable, and neither-side-stated."""
    assert readiness.compare_builds({"version": "0.64.1"}, {"version": "0.64.1"}) == (
        readiness.BuildComparison("equal", "0.64.1", "0.64.1")
    )
    behind = readiness.compare_builds({"version": "0.63.2"}, {"version": "0.64.1"})
    assert (behind.state, behind.peer_version, behind.own_version) == (
        "behind",
        "0.63.2",
        "0.64.1",
    )
    assert readiness.compare_builds({"version": "0.66.0"}, {"version": "0.64.1"}).state == "ahead"
    assert readiness.compare_builds({}, {"version": "0.64.1"}).state == "unknown"
    assert (
        readiness.compare_builds({"version": "0.28.0rc1"}, {"version": "0.64.1"}).state == "unknown"
    )
    assert readiness.compare_builds(None, None).state == "unknown"
    assert readiness.compare_builds({"version": "0.64.1"}, "not-a-mapping").state == "unknown"


def test_build_suffix_speaks_only_when_a_version_is_known() -> None:
    assert readiness.build_suffix(readiness.BuildComparison("unknown", "", "")) == ""
    assert (
        readiness.build_suffix(readiness.BuildComparison("equal", "0.64.1", "0.64.1"))
        == "  build 0.64.1"
    )
    assert (
        readiness.build_suffix(readiness.BuildComparison("ahead", "0.66.0", "0.64.1"))
        == "  build 0.66.0"
    )
    assert readiness.build_suffix(readiness.BuildComparison("behind", "0.63.2", "0.64.1")) == (
        "  build 0.63.2 — behind this device (0.64.1); ask Local Operator to update it there"
    )


# ---------------------------------------------------------------------------
# (e) model credential
# ---------------------------------------------------------------------------


def test_model_credential_row_resolves_no_default_first() -> None:
    row = readiness.model_credential_row(
        _member(),
        _facts(
            default_model={
                "hosting": "",
                "model_name": "",
                "resolved": False,
                "reason": "no config.yml",
            }
        ),
        viewer=_FakeViewer(),
        peer_label="cloud-node-1",
    )
    assert row["ok"] is False and row["code"] == readiness.CODE_NOT_CONFIGURED
    assert "/model default" in row["remedies"][0]


@pytest.mark.parametrize(
    ("over", "viewer_kwargs", "ok", "needle"),
    [
        ({"has_local": True}, {}, True, "own login"),
        (
            {
                "credential_placement": {
                    "owner_device": VIEWER,
                    "holders": [{"device": PEER, "scope": "session"}],
                }
            },
            {},
            True,
            "grant is issued per request",
        ),
        (
            {
                "credential_placement": {
                    "owner_device": THIRD,
                    "holders": [{"device": PEER, "scope": "session"}],
                }
            },
            {},
            True,
            "not verified from here",
        ),
        ({}, {"rows": ["row"]}, False, "share it"),
        ({}, {"rows": []}, False, "lop login openai"),
    ],
    ids=["own-login", "borrowed-here", "borrowed-third", "signed-in-here", "neither"],
)
def test_model_credential_verdicts(
    over: dict[str, Any], viewer_kwargs: dict[str, Any], ok: bool, needle: str
) -> None:
    viewer = _FakeViewer(**viewer_kwargs)
    row = readiness.model_credential_row(
        _member(), _facts(**over), viewer=viewer, peer_label="cloud-node-1"
    )
    assert row["ok"] is ok, row
    haystack = row["detail"] + " ".join(row["remedies"])
    assert needle in haystack, row


def test_a_share_that_has_not_been_pulled_is_not_denied() -> None:
    """The harness's own finding, pinned: a borrower LEARNS a share on a pull.

    ``pull_placement`` is paid on an explicit act (``lop network credentials``),
    never on the provider path, so between the owner's ``share`` and the
    borrower's pull the borrower's OWN answer carries no placement entry. A
    report that stopped at "not shared" would deny a share that is on THIS
    device's books and send the operator to run a command that already ran; the
    row names the missing STEP instead.
    """
    viewer = _FakeViewer(
        rows=["row"],
        placement={
            "owner_device": VIEWER,
            "holders": [{"device": PEER, "scope": "session"}],
        },
    )
    row = readiness.model_credential_row(
        _member(), _facts(), viewer=viewer, peer_label="cloud-node-1"
    )
    assert row["ok"] is False and row["code"] == readiness.CODE_NOT_SHARED
    assert "has not pulled the placement" in row["detail"]
    assert "lop network credentials" in row["remedies"][0]


def test_model_credential_row_reads_the_observation_memory() -> None:
    placement = {
        "owner_device": THIRD,
        "owner_device_name": "third-box",
        "holders": [{"device": PEER, "scope": "session"}],
        "observation": {
            "status": "owner_offline",
            "reason": "no answer",
            "owner_device": THIRD,
            "observed_at": 1_700_000_000.0,
            "retry_after_ms": 15_000,
        },
    }
    row = readiness.model_credential_row(
        _member(),
        _facts(credential_placement=placement),
        viewer=_FakeViewer(),
        peer_label="cloud-node-1",
    )
    assert row["ok"] is False
    assert row["code"] == readiness.CODE_OBSERVED_FAILURE
    assert "is not reachable" in row["detail"]
    assert "last seen" in row["detail"]


def test_observation_failure_is_empty_for_the_positive_and_unknown_states() -> None:
    assert readiness._observation_failure({"status": "active"}) == ""
    assert readiness._observation_failure({}) == ""
    assert readiness._observation_failure({"status": "grant_invalid"}) != ""


# ---------------------------------------------------------------------------
# Reachability readings — the discriminator the operator asked for
# ---------------------------------------------------------------------------


def _row(  # noqa: PLR0913 — one builder for the reading matrix's rows
    outcome: str,
    *,
    detail: str = "ok",
    ok: bool = False,
    source: str | None = "203.0.113.7",
    interface: str | None = "utun4",
    winner: str = "",
    winner_verified: bool = False,
    link_address: str = "",
    link_unpinned: bool = False,
) -> dict[str, Any]:
    observed: dict[str, Any] = {
        "outcome": outcome,
        "source_address": source,
        "interface": interface,
        "elapsed_ms": 3000.2,
        "budget_s": 3.0,
        "attempted": True,
        "last_seen_at": 170,
    }
    # Mirrors the composer: the winner keys exist only when something won, and
    # ``winner_verified`` only ever rides with a winner (design round 1, D1).
    if winner:
        observed["winner"] = winner
        observed["winner_verified"] = winner_verified
    if link_address:
        observed["link_address"] = link_address
    if link_unpinned:
        # Mirrors the composer: a live link this report cannot pin to a declared
        # address (round 2, R2-1/Q-3). The key names the CONDITION, never a
        # direction (round 3, R3-2).
        observed["link_unpinned"] = True
    return {
        "check": "reachability",
        "device_id": PEER,
        "device_name": "cloud-node-1",
        "endpoint": "198.51.100.7:7777",
        "ok": ok,
        "detail": detail,
        "observed": observed,
    }


def test_reachability_reading_distinguishes_refused_from_silence() -> None:
    refused = readiness.reachability_reading(
        _row("refused", detail="connect_failed:ConnectionRefusedError")
    )
    assert "refused the connection" in refused
    assert "the host is up" in refused
    assert "nothing answered" not in refused

    silent = readiness.reachability_reading(_row("no_answer", detail="no_answer"))
    assert silent.startswith("nothing answered this address before the budget ran out")
    assert "(this device routes to it from 203.0.113.7 via utun4)" in silent

    silent_no_route_facts = readiness.reachability_reading(
        _row("no_answer", detail="no_answer", source=None, interface=None)
    )
    assert "this device routes" not in silent_no_route_facts

    verified_elsewhere = readiness.reachability_reading(
        _row(
            "no_answer_elsewhere",
            detail="no_answer",
            winner="10.0.0.9:4097",
            winner_verified=True,
        )
    )
    assert "the peer is up (it answered 10.0.0.9:4097)" in verified_elsewhere
    assert "this address did not answer" in verified_elsewhere

    # AN UNVERIFIED WINNER CLAIMS NOTHING (design round 1, D1): a bare accept
    # is not a peer answer, so the row says only what the row knows.
    unverified_elsewhere = readiness.reachability_reading(
        _row("no_answer_elsewhere", detail="no_answer", winner="192.0.2.9:4097")
    )
    assert unverified_elsewhere == "this address did not answer"

    link_elsewhere = readiness.reachability_reading(
        _row(
            "no_answer_elsewhere",
            detail="no_answer",
            winner="10.0.0.9:4097",
            winner_verified=True,
            link_address="10.0.0.9:4097",
        )
    )
    assert "the peer is up (its link runs at 10.0.0.9:4097)" in link_elsewhere

    no_route = readiness.reachability_reading(_row("no_route", detail="connect_failed:OSError"))
    assert "routing said the address is unreachable" in no_route

    connected = readiness.reachability_reading(_row("connected", ok=True))
    assert "answered" in connected

    bad = readiness.reachability_reading(_row("bad_endpoint", detail="bad_endpoint"))
    assert "cannot be dialled" in bad


def test_a_bare_accept_is_never_reported_as_the_peer() -> None:
    """Design round 1, D1: any listener satisfies a TCP connect.

    The unverified states render the observed fact — "something accepted a TCP
    connection at this address; it was not identified as the peer" — and the
    verified states keep their peer claims, so the counter-direction is pinned
    in the same cell.
    """
    unverified_winner = readiness.reachability_reading(
        _row("connected_unverified", winner="192.0.2.9:4097", link_address="10.0.0.9:4097")
    )
    assert "was not identified as the peer" in unverified_winner
    assert "the peer answered" not in unverified_winner
    assert "(the peer's link runs at 10.0.0.9:4097)" in unverified_winner

    elsewhere = readiness.reachability_reading(
        _row("connected_elsewhere", winner="10.0.0.9:4097", winner_verified=True)
    )
    assert "was not identified as the peer" in elsewhere
    assert "(the peer answered on 10.0.0.9:4097)" in elsewhere

    elsewhere_unverified = readiness.reachability_reading(
        _row("connected_elsewhere", winner="192.0.2.9:4097")
    )
    assert "was not identified as the peer" in elsewhere_unverified
    assert "the peer answered" not in elsewhere_unverified


def test_an_unpinned_link_reads_as_the_link_fact_not_an_address() -> None:
    """Round 2, R2-1/Q-3: a live link whose address cannot be pinned to a
    declared endpoint (an inbound source socket; round 3, R3-2 renamed the
    condition straight).

    The row presents the LINK fact (the peer is up) with no address named,
    and the accept is not turned into a peer-answer claim either — a false red
    on the peer's own reachable address is the defect this pins shut.
    """
    reading = readiness.reachability_reading(
        _row("connected_unpinned", detail="accepted_unpinned_link", ok=True)
    )
    assert reading == ("the peer is up (its link is live); this address accepted a TCP connection")
    assert "the peer answered" not in reading
    assert "not identified" not in reading

    # The same state on an address that did NOT answer names no address either.
    silent = readiness.reachability_reading(
        _row("no_answer_elsewhere", detail="no_answer", link_unpinned=True)
    )
    assert silent == "the peer is up (its link is live); this address did not answer"


def test_a_report_budget_tail_reads_in_this_verbs_clock_not_the_doctors() -> None:
    """Round 2, R2-2: the clock that expired is THIS report's.

    Doctor's gloss for the same state names the doctor's own budget ("the
    doctor ran out of time before the handshake"), which would send the reader
    to the wrong command; the ready register composes its own sentence from
    its own clock, while REFUSAL tails keep reading in doctor's refusal words.
    """
    from local_operator.network import relay as relay_mod

    detail = relay_mod.handshake_not_attempted_reason("203.0.113.7:4097", budget="report")
    reading = readiness.reachability_reading(_row("handshake_failed", detail=detail))
    assert reading == "the address answered, but the report ran out of time before the handshake"
    assert "doctor" not in reading

    refusal = readiness.reachability_reading(
        _row("handshake_failed", detail="handshake_refused:TimeoutError")
    )
    assert "the link was refused" in refusal

    link_row = readiness.reachability_reading(_row("connected_link", ok=True))
    assert link_row == "the peer's link runs at this address"

    verified = readiness.reachability_reading(_row("connected", ok=True))
    assert verified == "the peer answered at this address"

    no_endpoint = readiness.reachability_reading(_row("no_endpoint", detail="no_endpoint"))
    assert no_endpoint == "no address published for it — nothing was dialled"


def test_a_refused_handshake_reads_in_words_not_a_wire_code() -> None:
    """Round 1 reconciliation: doctor's own refusal words, never a bare code.

    ``epoch_stale`` is a ``handshake.REASON_*`` identifier — a refusal from a
    peer that ANSWERED — and a bare wire token on a human line is what doctor's
    renderer exists to avoid.
    """
    row = readiness.reachability_reading(_row("handshake_failed", detail="epoch_stale"))
    assert "no link came up" in row
    assert "epoch_stale" not in row


def test_remedies_do_not_point_at_unverified_addresses() -> None:
    """Design round 1, D1: the "point it at the working address" advice is a
    PEER claim, so it rides only on an identified answer."""
    unverified = readiness._reachability_remedies(
        "refused",
        peer_label="cloud-node-1",
        observed={"winner": "192.0.2.9:4097"},
    )
    assert unverified == ["start cloud-node-1's relay (`lop network start` there), then re-check"]

    verified = readiness._reachability_remedies(
        "refused",
        peer_label="cloud-node-1",
        observed={"winner": "10.0.0.9:4097", "winner_verified": True},
    )
    assert "point cloud-node-1 at the working address: it answered on 10.0.0.9:4097" in verified[0]

    link_only = readiness._reachability_remedies(
        "no_answer_elsewhere",
        peer_label="cloud-node-1",
        observed={
            "winner": "10.0.0.9:4097",
            "winner_verified": True,
            "link_address": "10.0.0.9:4097",
        },
    )
    assert "point the peer at the working address: it answered on 10.0.0.9:4097" in link_only[0]

    nothing = readiness._reachability_remedies(
        "no_answer_elsewhere",
        peer_label="cloud-node-1",
        observed={"winner": "192.0.2.9:4097"},
    )
    assert "re-check once cloud-node-1 answers at this address" in nothing[0]


def test_an_unpinned_link_pivots_the_address_remedies_off_the_ephemeral() -> None:
    """Round 2, R2-1/Q-3: the peer is up (its link is live) but the link names
    no address this report may cite — so no remedy cites one, and "start the
    relay" is not advice for a peer that is demonstrably running."""
    for outcome in ("refused", "no_answer_elsewhere"):
        remedies = readiness._reachability_remedies(
            outcome, peer_label="cloud-node-1", observed={"link_unpinned": True}
        )
        joined = " ".join(remedies)
        assert "its link is live" in joined, outcome
        assert "start" not in joined, outcome
        assert "127." not in joined and "10." not in joined, outcome


def test_a_report_budget_remedy_says_re_run_not_the_peers_handshake() -> None:
    """Round 3, D6: when the tail says the report's OWN budget ran out, the
    remedy must not send anyone to the peer's relay — the address answered."""
    from local_operator.network import relay as relay_mod

    re_run = readiness._reachability_remedies(
        "handshake_failed",
        peer_label="cloud-node-1",
        observed={"dial_stage": relay_mod.HANDSHAKE_NOT_ATTEMPTED},
    )
    assert "re-run this report" in re_run[0]
    assert "relay answers a handshake" not in re_run[0]

    refusal = readiness._reachability_remedies(
        "handshake_failed", peer_label="cloud-node-1", observed={"dial_stage": "unreachable"}
    )
    assert "re-check once cloud-node-1's relay answers a handshake again" in refusal[0]


def test_reachability_remedies_name_the_discriminating_checks() -> None:
    row = _row("no_answer", detail="no_answer")
    remedies = readiness._reachability_remedies(
        "no_answer", peer_label="cloud-node-1", observed=row["observed"]
    )
    joined = " ".join(remedies)
    assert "lop network status --json" in joined
    assert "203.0.113.7" in joined


# ---------------------------------------------------------------------------
# The informational flip (drill finding, 2026-10-03)
# ---------------------------------------------------------------------------


def _reach_row(
    endpoint: str,
    outcome: str,
    *,
    ok: bool,
    detail: str = "ok",
    winner: str = "",
    winner_verified: bool = False,
) -> dict[str, Any]:
    """A full reachability row shaped as ``_reachability_rows`` builds it."""
    observed: dict[str, Any] = {"outcome": outcome, "attempted": True}
    if winner:
        observed["winner"] = winner
        observed["winner_verified"] = winner_verified
    return {
        "check": "reachability",
        "device_id": PEER,
        "device_name": "cloud-node-1",
        "endpoint": endpoint,
        "ok": ok,
        "detail": detail,
        "observed": observed,
        "remedies": ["a remedy that must not read as an action item under an ok row"],
    }


def test_a_remote_unusable_address_is_informational_when_the_peer_answers_elsewhere() -> None:
    """The drill's exact shape: the VPC-private address refused, the public one
    connected — and the report must not read unhealthy because of the address
    no remote device can use (``172.31.22.23`` vs ``99.79.190.164``)."""
    rows = [
        _reach_row(
            "172.31.22.23:4097",
            "refused",
            ok=False,
            detail="connect_failed:ConnectionRefusedError",
        ),
        _reach_row(
            "99.79.190.164:4097",
            "connected",
            ok=True,
            winner="99.79.190.164:4097",
            winner_verified=True,
        ),
    ]
    readiness.mark_informational(rows)
    dead, live = rows
    assert dead["ok"] is True
    assert dead["observed"]["informational"] is True
    assert dead["observed"]["usable_elsewhere"] == ["99.79.190.164:4097"]
    # The row keeps its observed facts, and the action bullet is gone.
    assert dead["observed"]["outcome"] == "refused"
    assert dead["remedies"] == []
    assert live["ok"] is True and live["observed"].get("informational") is None
    reading = readiness.reachability_reading(dead)
    # The raw failure clause is REPLACED, never parroted (design round 1, D1):
    # "nothing is listening on that port" would name a wrong action for the one
    # row this flip exists for — and the reading IS the shared clause (D3), so
    # the doctor renderers cannot drift from it.
    assert reading == readiness.informational_clause(dead)
    assert "nothing is listening on that port" not in reading
    assert "not remote-usable from this device" in reading
    assert "the peer is reachable at 99.79.190.164:4097" in reading
    assert "the machine's own private address" in reading
    # The note names the KIND; the clause carries the scope (D8).
    assert "remote devices cannot use it" not in reading


def test_without_a_usable_address_the_report_keeps_its_failures() -> None:
    """THE HONEST NEGATIVE: with no address answering anywhere, nothing flips —
    a member nothing answers for must never read healthy because its addresses
    "look" private."""
    rows = [
        _reach_row("172.31.22.23:4097", "refused", ok=False, detail="connect_failed"),
        _reach_row("99.79.190.164:4097", "no_answer", ok=False, detail="no_answer"),
    ]
    readiness.mark_informational(rows)
    assert [row["ok"] for row in rows] == [False, False]
    assert all(row["observed"].get("informational") is None for row in rows)
    assert all(row["remedies"] for row in rows)


def test_an_unidentified_accept_never_vouches_for_the_member() -> None:
    """``connected_unverified`` (a stranger answered the port) is not a usable
    address: it must neither flip other rows nor ride someone else's accept."""
    rows = [
        _reach_row(
            "203.0.113.9:4097", "connected_unverified", ok=False, detail="accepted_unverified"
        ),
        _reach_row("99.79.190.164:4097", "connected", ok=True, winner="99.79.190.164:4097"),
    ]
    readiness.mark_informational(rows)
    stranger, live = rows
    assert stranger["ok"] is False
    assert stranger["observed"].get("informational") is None
    assert live["ok"] is True

    # And the reverse shape: an unidentified accept ALONE vouches for nothing.
    only_stranger = [_reach_row("203.0.113.9:4097", "connected_unverified", ok=False)]
    readiness.mark_informational(only_stranger)
    assert only_stranger[0]["ok"] is False


def test_a_public_dead_address_reads_informational_without_the_private_note() -> None:
    rows = [
        _reach_row("198.51.100.7:7777", "no_route", ok=False, detail="connect_failed:OSError"),
        _reach_row("99.79.190.164:4097", "connected_link", ok=True, winner="99.79.190.164:4097"),
    ]
    readiness.mark_informational(rows)
    reading = readiness.reachability_reading(rows[0])
    assert "not remote-usable from this device" in reading
    assert "the peer is reachable at 99.79.190.164:4097" in reading
    assert "private address" not in reading


def test_the_renderer_reads_an_informational_row_as_ok_and_keeps_failures_failing() -> None:
    flipped = _reach_row("172.31.22.23:4097", "refused", ok=False, detail="connect_failed")
    connected = _reach_row("99.79.190.164:4097", "connected", ok=True)
    readiness.mark_informational([flipped, connected])
    still_failing = {
        "check": "readiness",
        "capability": "mcp_login",
        "device_name": "cloud-node-1",
        "ok": False,
        "detail": "no login for https://slack.example",
        "remedies": [],
    }
    lines = readiness.render_check_lines([flipped, connected, still_failing])
    assert lines[0].startswith("ok  reachability cloud-node-1 172.31.22.23:4097:")
    assert "not remote-usable from this device" in lines[0]
    assert lines[-1].startswith("FAIL readiness mcp_login cloud-node-1:")


def _doctor_row(
    endpoint: str, *, check: str = "reachability", ok: bool, detail: str = "ok"
) -> dict[str, Any]:
    """A doctor-dialect row as ``RelayServer._probe_member`` builds it."""
    return {"check": check, "device_id": PEER, "endpoint": endpoint, "ok": ok, "detail": detail}


def test_doctor_marks_a_dead_address_informational_when_the_handshake_verified_elsewhere() -> None:
    """Same semantics, doctor's row shape (drill finding, 2026-10-03): the dead
    row is a fact about the address once the handshake row proves the member."""
    rows = [
        _doctor_row("172.31.22.23:4097", ok=False, detail="connect_failed:ConnectionRefusedError"),
        _doctor_row("99.79.190.164:4097", check="handshake", ok=True),
    ]
    readiness.mark_informational(rows)
    dead = rows[0]
    assert dead["ok"] is True
    assert dead["observed"]["informational"] is True
    assert dead["observed"]["usable_elsewhere"] == ["99.79.190.164:4097"]
    assert readiness.informational_clause(dead) == (
        "not remote-usable from this device (the machine's own private address); "
        "the peer is reachable at 99.79.190.164:4097"
    )
    # Every other row gets no clause; the handshake row keeps its own shape.
    assert readiness.informational_clause(rows[1]) == ""


def test_doctor_keeps_failing_when_nothing_verified() -> None:
    """THE HONEST NEGATIVE: no verified address anywhere ⇒ nothing flips, and a
    handshake that did NOT pass verifies nothing."""
    rows = [
        _doctor_row("172.31.22.23:4097", ok=False, detail="connect_failed:ConnectionRefusedError"),
        _doctor_row("99.79.190.164:4097", check="handshake", ok=False, detail="no_answer"),
    ]
    readiness.mark_informational(rows)
    assert [row["ok"] for row in rows] == [False, False]
    assert "observed" not in rows[0]


def test_doctor_keeps_not_attempted_failing_and_repair_rows_untouched() -> None:
    rows = [
        _doctor_row("172.31.22.23:4097", ok=False, detail="not_attempted"),
        _doctor_row("198.51.100.7:7777", ok=False, detail="bad_endpoint"),
        _doctor_row("99.79.190.164:4097", check="handshake", ok=True),
        {
            "check": "credential_repair",
            "device_id": PEER,
            "ok": False,
            "detail": "the owner's minerva-qa login died; the owner must sign in again",
        },
    ]
    readiness.mark_informational(rows)
    assert rows[0]["ok"] is False  # unproven is not a fact about the address
    assert rows[1]["ok"] is True  # bad_endpoint IS one: it cannot be dialled
    assert rows[2]["ok"] is True  # the verification row itself, untouched
    assert rows[3]["ok"] is False  # a different check; the flip never touches it


# ---------------------------------------------------------------------------
# Degradation rows and sanitisation
# ---------------------------------------------------------------------------


def test_not_asked_and_too_old_rows_cover_the_checklist() -> None:
    member = _member()
    not_asked = readiness.not_asked_rows(member, detail="not asked: the peer did not answer")
    assert [row["capability"] for row in not_asked] == list(readiness.PEER_SIDE_CHECKS)
    assert all(row["ok"] is False and row["code"] == readiness.CODE_NOT_ASKED for row in not_asked)

    too_old = readiness._peer_too_old_rows(member)
    assert [row["capability"] for row in too_old] == list(readiness.PEER_SIDE_CHECKS)
    assert all(row["code"] == readiness.CODE_PEER_TOO_OLD for row in too_old)
    assert all(
        any("ask Local Operator to update it" in remedy for remedy in row["remedies"])
        for row in too_old
    )


def test_safe_text_drops_credential_shaped_text() -> None:
    assert readiness._safe_text("mcp.linear.app") == "mcp.linear.app"
    # The canonical AWS documentation example, assembled at RUNTIME: a single
    # literal of this shape is exactly what the session's own output scrubber
    # rewrites to a marker, so spelling it whole here would pin nothing. It is
    # a published value (AWS docs), and the table's `aws-access-key-id` shape.
    example = "".join(("AKIA", "IOSFODNN7EXAMPLE"))
    assert readiness._safe_text(example) == ""


def test_peer_facts_never_carry_a_scrubber_marker_or_a_shaped_value(tmp_path: Path) -> None:
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ok": {"type": "http", "url": "https://mcp.example.test/mcp"},
                    "shaped": {"type": "http", "url": "https://user:t-1234567890abcd@e.test/mcp"},
                }
            }
        ),
        encoding="utf-8",
    )
    facts = readiness.collect_peer_facts(tmp_path)
    serialised = json.dumps(facts)
    assert "t-1234567890abcd" not in serialised
    # The agent tool scrubs by NAME substrings; the payload's field names avoid
    # them so nothing legitimate is dropped on the way to a model.
    for marker in ("token", "secret", "password"):
        assert marker not in serialised.lower(), marker
    assert facts["mcp"]["withheld"] == ["shaped"]


# ---------------------------------------------------------------------------
# The read-only discipline, applied to the verb
# ---------------------------------------------------------------------------


def test_compose_on_a_fresh_root_creates_nothing(tmp_path: Path) -> None:
    """``ready`` must never be the reason anything exists (test_reads_create_nothing).

    Driven against a REAL RelayServer (no control socket, no start): the
    composer reads and probes nothing here, but it must not even create the
    directories a ConfigManager or an AuthStore would.
    """
    from local_operator.network import identity, relay

    root = tmp_path / "root"
    root.mkdir(parents=True)
    ident = identity.mint(root, name="fresh-device")
    server = relay.RelayServer(
        identity=ident, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))

    payload = readiness.compose(server)

    after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert after == before, f"compose created files: {sorted(set(after) - set(before))}"
    assert payload["checks"] == []
    assert payload["identity_present"] is True
    assert payload["relay"].startswith("running, pid ")


def test_compose_names_the_running_build_where_one_can_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5 (design round 1, F6): the answering path names its own build.

    The drill's own shape is a relay that ANSWERS while a build behind — and this
    row's plain ``running, pid N`` gave it a clean bill in ``doctor`` and
    ``ready``. The reading is the shipped one (``relay.generation_reading``) with
    the update probes and the version reader as the seams, so the clause cannot
    keep passing against a row that stopped carrying it.
    """
    import os

    from local_operator import update as update_mod
    from local_operator.network import identity, relay

    root = tmp_path / "root-build"
    root.mkdir(parents=True)
    ident = identity.mint(root, name="fresh-device")
    server = relay.RelayServer(
        identity=ident, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    old = "20260921T125352Z-0.61.12"
    new = "20260924T103058Z-509c7450dbf6"
    generations = tmp_path / "generations"
    monkeypatch.setattr(update_mod, "current_generation", lambda: generations / new)
    monkeypatch.setattr(update_mod, "generation_of_process", lambda _pid: generations / old)
    monkeypatch.setattr(update_mod, "stale_generation_of_process", lambda _pid: generations / old)
    monkeypatch.setattr(
        update_mod,
        "generation_version",
        lambda gen: {old: "0.61.12", new: "0.67.4"}.get(gen.name, ""),
    )

    payload = readiness.compose(server)

    assert payload["relay"] == (
        f"running, pid {os.getpid()}, build 0.61.12 — behind 0.67.4; " "run `lop network restart`"
    ), payload["relay"]


def test_wire_capability_is_the_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    assert wire.PEER_READINESS_V1 in wire.LINK_CAPABILITIES
