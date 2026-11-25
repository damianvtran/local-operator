"""A provable ``sqlite3`` read is not a dangerous delete (field case 0.113.2).

WHY. In the Team Bridge lane a read-only query on
``~/.local-operator/team_bridge/state.sqlite3`` was refused CRITICAL as a
delete: a read cannot delete. ``resolve._sqlite_read_only`` now proves a read
(``-readonly``, a ``file:…?mode=ro`` URI, or only read statements on an existing
file) and treats it like ``cat``. The parse is deliberately conservative, so the
second half of this file pins every write shape at its pre-change rank: a
``DELETE``, ``-cmd``/``-init``, ``.output``/``.dump``, ATTACH/VACUUM/writefile,
a missing file (sqlite3 creates it), a glob, an opaque expansion, a redirect.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.guardrails.check import SQLITE_READ_HINT, CallContext, GuardSettings
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import preview_tool_call

FIELD = 'sqlite3 ~/.local-operator/team_bridge/state.sqlite3 "SELECT * FROM t"'
LEDGER = "~/.local-operator/guardrails/created.sqlite"
BRIDGE = "~/.local-operator/team_bridge/state.sqlite3"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / "own" / "scratchpad").mkdir(parents=True)
    (state / "team_bridge").mkdir()
    (state / "team_bridge" / "state.sqlite3").write_bytes(b"")
    (state / "guardrails").mkdir()
    (state / "guardrails" / "created.sqlite").write_bytes(b"")
    (home / "repo").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    return home


def _env(home: Path) -> dict[str, str]:
    pad = home / ".local-operator" / "sessions" / "own" / "scratchpad"
    return {"HOME": str(home), "LOCAL_OPERATOR_SCRATCHPAD": str(pad)}


def _risk(home: Path, command: str) -> Risk:
    env = _env(home)
    return classify_bash(
        command,
        cwd=str(home / "repo"),
        env=env,
        scratchpad=env["LOCAL_OPERATOR_SCRATCHPAD"],
        session_id="own",
    ).risk


#: (command, rank on origin/main before this change). Each now ranks LOW.
_ACCEPTED = [
    (f'sqlite3 -readonly {LEDGER} "select 1"', Risk.HIGH),
    (f'sqlite3 --readonly {LEDGER} "select 1"', Risk.HIGH),
    (
        'sqlite3 "file:$HOME/.local-operator/team_bridge/state.sqlite3?mode=ro" "SELECT 1"',
        Risk.CRITICAL,
    ),
    (
        'sqlite3 "file:$HOME/.local-operator/team_bridge/state.sqlite3?immutable=1" "SELECT 1"',
        Risk.CRITICAL,
    ),
    (
        'cd ~/.local-operator/team_bridge && sqlite3 state.sqlite3 "SELECT 1"',
        Risk.CRITICAL,
    ),
    (
        "cd ~/.local-operator/team_bridge && "
        'sqlite3 -readonly state.sqlite3 "PRAGMA table_info(t)"',
        Risk.CRITICAL,
    ),
    (f'sqlite3 {LEDGER} "select count(*) from created"', Risk.CRITICAL),
    (f'sqlite3 {LEDGER} "WITH x AS (SELECT 1) SELECT * FROM x"', Risk.CRITICAL),
    (f'sqlite3 {LEDGER} "EXPLAIN SELECT 1"', Risk.CRITICAL),
    (f'sqlite3 -header -column {LEDGER} "PRAGMA user_version"', Risk.CRITICAL),
    (f"sqlite3 -readonly {LEDGER} .tables", Risk.HIGH),
    (f'sqlite3 -json {LEDGER} "select 1; select 2"', Risk.CRITICAL),
]


@pytest.mark.parametrize(("command", "before"), _ACCEPTED)
def test_provable_sqlite_reads_are_low(home: Path, command: str, before: Risk) -> None:
    assert before is not Risk.LOW  # documents the false positive this fixes
    assert _risk(home, command) is Risk.LOW, command


def test_field_command_stays_low(home: Path) -> None:
    assert _risk(home, FIELD) is Risk.LOW


#: (command, rank, unchanged from origin/main): every write shape keeps today's rank.
_REFUSED = [
    (f'sqlite3 {LEDGER} "insert into created values (1)"', Risk.CRITICAL),
    (f'sqlite3 {LEDGER} "DELETE FROM created"', Risk.CRITICAL),
    (f'sqlite3 {LEDGER} "select 1; delete from created"', Risk.CRITICAL),
    (f'sqlite3 {LEDGER} "select replace(a, b, c) from created"', Risk.CRITICAL),
    (f'sqlite3 -cmd "delete from created" {LEDGER} "select 1"', Risk.CRITICAL),
    (f'sqlite3 -init /x {LEDGER} "select 1"', Risk.CRITICAL),
    (f'sqlite3 {LEDGER} "PRAGMA journal_mode=wal"', Risk.CRITICAL),
    (f"sqlite3 {LEDGER}", Risk.CRITICAL),  # commands come from stdin: unseen
    ('sqlite3 ~/.local-operator/guardrails/missing.sqlite "select 1"', Risk.CRITICAL),
    ('sqlite3 ~/.local-operator/guardrails/*.sqlite "select 1"', Risk.CRITICAL),
    ('sqlite3 ~/.local-operator/guardrails/$(echo x).sqlite "select 1"', Risk.CRITICAL),
    (f'sqlite3 -readonly {LEDGER} "select 1" > {LEDGER}', Risk.CRITICAL),
    (f"sqlite3 -readonly {LEDGER} \"select writefile('/x','y')\"", Risk.HIGH),
    (f"sqlite3 -readonly {LEDGER} \"vacuum into '/x'\"", Risk.HIGH),
    (f"sqlite3 -readonly {LEDGER} \"attach '/x' as y\"", Risk.HIGH),
    (f'sqlite3 -readonly {LEDGER} ".output /x" "select 1"', Risk.HIGH),
    (f'sqlite3 -readonly {LEDGER} ".once /x" "select 1"', Risk.HIGH),
    (f"sqlite3 -readonly {LEDGER} .dump", Risk.HIGH),
    (f'sqlite3 -readonly {LEDGER} ".shell touch /x"', Risk.HIGH),
    (
        'sqlite3 "file:$HOME/.local-operator/guardrails/created.sqlite?mode=rw" "select 1"',
        Risk.HIGH,
    ),
    (f'sqlite3 -readonly -cmd ".output /x" {LEDGER} "select 1"', Risk.HIGH),
    (f'X=1 sqlite3 {LEDGER} "select 1"', Risk.CRITICAL),
    (f'python3 -c "x" {LEDGER}', Risk.HIGH),  # not sqlite3: unchanged
]


@pytest.mark.parametrize(("command", "risk"), _REFUSED)
def test_sqlite_write_shapes_keep_their_rank(
    home: Path, command: str, risk: Risk
) -> None:
    assert _risk(home, command) is risk, command


def test_preview_of_field_command_allows(home: Path) -> None:
    ctx = CallContext(
        cwd=str(home / "repo"),
        env=_env(home),
        scratchpad=_env(home)["LOCAL_OPERATOR_SCRATCHPAD"],
        session_id="own",
    )
    rules = home / "rules"
    rules.mkdir()
    settings = GuardSettings(rules_dirs=(rules,))
    for command in (FIELD, f'sqlite3 -readonly {BRIDGE} "SELECT 1"'):
        report = preview_tool_call("bash", {"command": command}, ctx, settings=settings)
        assert report.action == "allow", (command, report.render())
    # Negative control: the same preview still refuses a write to the ledger,
    # and its refusal names the read-only form an agent can use instead.
    write = f'sqlite3 {LEDGER} "DELETE FROM created"'
    report = preview_tool_call("bash", {"command": write}, ctx, settings=settings)
    assert report.action == "refuse"  # a CRITICAL built-in is a refusal
    assert SQLITE_READ_HINT in report.builtin_text
    assert "sqlite3 -readonly" in SQLITE_READ_HINT and "mode=ro" in SQLITE_READ_HINT


def test_hint_only_on_sqlite_refusals(home: Path) -> None:
    ctx = CallContext(
        cwd=str(home / "repo"), env=_env(home), scratchpad=None, session_id="own"
    )
    rules = home / "rules"
    rules.mkdir()
    report = preview_tool_call(
        "bash",
        {"command": f"rm -f {LEDGER}"},
        ctx,
        settings=GuardSettings(rules_dirs=(rules,)),
    )
    assert report.action == "refuse"  # a CRITICAL built-in is a refusal
    assert SQLITE_READ_HINT not in report.builtin_text and report.builtin_text
