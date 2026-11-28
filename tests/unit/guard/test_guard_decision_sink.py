"""Guard L1: every guardrail verdict reaches the decision sink, enum-only, safely.

Driven through ``check_tool_call`` -- the function ``harness/loop.py`` calls --
not through ``_journal`` directly, so a call site that stops recording (or a
new one that forgets to) shows up here. Pins the RFC v0.6 L1 acceptance rows:
block/warn/allow/pass are recorded; a user rule is ``custom`` whatever it is
named; a raising sink never changes the call's outcome; and no socket is
touched while recording.
"""

from __future__ import annotations

import ast
import dataclasses
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator.guard import decisions as gd
from local_operator.guardrails import check as check_mod
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call

GUARD_PKG = Path(gd.__file__).parent


class _Collect:
    def __init__(self) -> None:
        self.records: list[gd.GuardDecisionRecord] = []

    def emit(self, record: gd.GuardDecisionRecord) -> None:
        self.records.append(record)

    def triples(self) -> list[tuple[str, str, str]]:
        return [(r.outcome, r.rule_class, r.tool_class) for r in self.records]


class _Raising:
    calls = 0

    def emit(self, record: gd.GuardDecisionRecord) -> None:
        type(self).calls += 1
        raise RuntimeError("sink exploded with /secret/path and a token sk-live-123")


@pytest.fixture
def sink() -> Iterator[_Collect]:
    collect = _Collect()
    previous = gd.set_decision_sink(collect)
    try:
        yield collect
    finally:
        gd.set_decision_sink(previous)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fail loudly on any socket use and record the attempt."""
    attempts: list[str] = []

    def refuse(name: str):
        def _refuse(*_a: object, **_k: object) -> None:
            attempts.append(name)
            raise AssertionError(f"network touched via {name}")

        return _refuse

    monkeypatch.setattr(socket.socket, "connect", refuse("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    return attempts


@pytest.fixture
def rules_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "state" / "rules"
    directory.mkdir(parents=True)
    return directory


def _rule(name: str, action: str, condition: str, body: str = "Body.") -> str:
    return (
        f"---\nname: {name}\nscope: tool:bash\ncondition: ['{condition}']\n"
        f"action: {action}\n---\n{body}\n"
    )


def _ctx(tmp_path: Path) -> CallContext:
    scratch = tmp_path / "scratchpad"
    scratch.mkdir(exist_ok=True)
    return CallContext(
        cwd=str(tmp_path),
        env={},
        scratchpad=str(scratch),
        session_id="own",
        user_request="tidy up",
    )


def _settings(rules_dir: Path) -> GuardSettings:
    return GuardSettings(rules_dirs=(rules_dir,))


async def _allow(system: str, prompt: str) -> str:
    return "ALLOW"


@pytest.mark.asyncio
async def test_block_warn_allow_and_pass_are_all_recorded(
    sink: _Collect, no_network: list[str], rules_dir: Path, tmp_path: Path
) -> None:
    (rules_dir / "warn.md").write_text(_rule("w-push", "warn", r"git\s+push\b"), "utf-8")
    (rules_dir / "block.md").write_text(_rule("b-force", "block", r"--force\b"), "utf-8")
    (rules_dir / "review.md").write_text(_rule("r-tag", "review", r"git\s+tag\b"), "utf-8")
    ctx, settings = _ctx(tmp_path), _settings(rules_dir)
    (Path(ctx.scratchpad or "") / "x.txt").write_text("x", "utf-8")

    warned = await check_tool_call(
        "bash", {"command": "git push origin f"}, ctx, judge=None, settings=settings
    )
    blocked = await check_tool_call(
        "bash", {"command": "git reset --force"}, ctx, judge=None, settings=settings
    )
    allowed = await check_tool_call(
        "bash", {"command": "git tag v1"}, ctx, judge=_allow, settings=settings
    )
    passed = await check_tool_call(
        "bash", {"command": f"rm {ctx.scratchpad}/x.txt"}, ctx, judge=None, settings=settings
    )

    assert warned is not None and not warned.blocked
    assert blocked is not None and blocked.blocked
    assert allowed is None or not allowed.blocked
    assert passed is not None and not passed.blocked and passed.destructive
    assert sink.triples() == [
        ("warn", "custom", "shell"),
        ("block", "custom", "shell"),
        ("allow", "custom", "shell"),
        ("pass", "dangerous_delete", "shell"),
    ]
    assert no_network == []


@pytest.mark.asyncio
async def test_builtin_block_is_dangerous_delete_and_file_tools_class(
    sink: _Collect, no_network: list[str], rules_dir: Path, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    outcome = await check_tool_call(
        "bash", {"command": "rm -rf ~"}, ctx, judge=None, settings=_settings(rules_dir)
    )
    assert outcome is not None and outcome.blocked
    assert sink.triples() == [("block", "dangerous_delete", "shell")]
    assert no_network == []


@pytest.mark.asyncio
async def test_a_user_rule_named_like_a_builtin_is_still_custom(
    sink: _Collect, rules_dir: Path, tmp_path: Path
) -> None:
    # The name is operator-authored text; it must never pick the class.
    (rules_dir / "spoof.md").write_text(_rule("dangerous-delete", "block", r"echo\b"), "utf-8")
    outcome = await check_tool_call(
        "bash", {"command": "echo hi"}, _ctx(tmp_path), judge=None, settings=_settings(rules_dir)
    )
    assert outcome is not None and outcome.blocked
    assert sink.triples() == [("block", "custom", "shell")]


@pytest.mark.asyncio
async def test_a_call_no_rule_evaluates_records_nothing(
    sink: _Collect, rules_dir: Path, tmp_path: Path
) -> None:
    # Not a decision: recording every unmatched call would make the count a
    # raw activity meter rather than the guard's verdicts.
    outcome = await check_tool_call(
        "bash", {"command": "ls"}, _ctx(tmp_path), judge=None, settings=_settings(rules_dir)
    )
    assert outcome is None
    assert sink.records == []


@pytest.mark.asyncio
async def test_records_carry_only_enums_never_call_content(
    sink: _Collect, rules_dir: Path, tmp_path: Path
) -> None:
    (rules_dir / "b.md").write_text(
        _rule("acme-prod-secret-rule", "block", r"deploy\b", "Never deploy acme."), "utf-8"
    )
    await check_tool_call(
        "bash",
        {"command": "deploy /Users/someone/acme --token sk-live-123"},
        _ctx(tmp_path),
        judge=None,
        settings=_settings(rules_dir),
    )
    (record,) = sink.records
    fields = {f.name for f in dataclasses.fields(record)}
    assert fields == {"outcome", "rule_class", "tool_class", "day", "record_id", "schema"}
    blob = repr(dataclasses.asdict(record))
    for leaked in ("acme", "deploy", "sk-live", "/Users", "Never"):
        assert leaked not in blob


@pytest.mark.asyncio
async def test_a_raising_sink_never_changes_the_outcome(
    no_network: list[str], rules_dir: Path, tmp_path: Path
) -> None:
    (rules_dir / "block.md").write_text(_rule("b", "block", r"git\s+push\b"), "utf-8")
    ctx, settings = _ctx(tmp_path), _settings(rules_dir)
    (Path(ctx.scratchpad or "") / "y.txt").write_text("y", "utf-8")
    _Raising.calls = 0
    previous = gd.set_decision_sink(_Raising())
    try:
        blocked = await check_tool_call(
            "bash", {"command": "git push"}, ctx, judge=None, settings=settings
        )
        passed = await check_tool_call(
            "bash", {"command": f"rm {ctx.scratchpad}/y.txt"}, ctx, judge=None, settings=settings
        )
        unmatched = await check_tool_call(
            "bash", {"command": "ls"}, ctx, judge=None, settings=settings
        )
    finally:
        gd.set_decision_sink(previous)
    assert _Raising.calls == 2  # the sink WAS reached, and its raise went nowhere
    # The user-rule block, not the fail-closed ``guardrail-error`` refusal.
    assert blocked is not None and blocked.blocked and blocked.rule == "b"
    assert passed is not None and not passed.blocked
    assert unmatched is None
    assert no_network == []


@pytest.mark.asyncio
async def test_the_fail_closed_error_refusal_is_recorded(
    sink: _Collect, monkeypatch: pytest.MonkeyPatch, rules_dir: Path, tmp_path: Path
) -> None:
    async def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("inner failure")

    monkeypatch.setattr(check_mod, "_check_tool_call", boom)
    outcome = await check_tool_call(
        "write", {"path": "x"}, _ctx(tmp_path), judge=None, settings=_settings(rules_dir)
    )
    assert outcome is not None and outcome.blocked and outcome.rule == "guardrail-error"
    assert sink.triples() == [("block", "guard_error", "file_write")]


@pytest.mark.asyncio
async def test_default_sink_is_noop_and_builds_no_record(
    monkeypatch: pytest.MonkeyPatch, rules_dir: Path, tmp_path: Path
) -> None:
    assert gd.get_decision_sink() is gd.NOOP_SINK

    def must_not_build(*_a: object, **_k: object) -> None:
        raise AssertionError("a record was built with no sink installed")

    monkeypatch.setattr(gd, "GuardDecisionRecord", must_not_build)
    (rules_dir / "b.md").write_text(_rule("b", "block", r"git\b"), "utf-8")
    outcome = await check_tool_call(
        "bash", {"command": "git push"}, _ctx(tmp_path), judge=None, settings=_settings(rules_dir)
    )
    assert outcome is not None and outcome.blocked and outcome.rule == "b"


def test_record_refuses_values_outside_the_vocabulary() -> None:
    gd.GuardDecisionRecord("pass", "dangerous_delete", "shell")
    with pytest.raises(ValueError):
        gd.GuardDecisionRecord("git push --force", "custom", "shell")
    with pytest.raises(ValueError):
        gd.GuardDecisionRecord("block", "no-force-push", "shell")
    with pytest.raises(ValueError):
        gd.GuardDecisionRecord("block", "custom", "mcp__acme__deploy")
    with pytest.raises(ValueError):
        gd.GuardDecisionRecord("block", "custom", "shell", record_id="not-hex")
    assert gd.tool_class_of("mcp__acme__deploy") == "other"


_NETWORK_MODULES = {
    "socket",
    "ssl",
    "http",
    "urllib",
    "httpx",
    "requests",
    "aiohttp",
    "websockets",
    "websocket",
}


#: The L1 decision modules, which stay inert by construction. Slice L2a added
#: ``consent.py``/``consent_op.py``/``install_key.py``, the admission half that
#: MAY reach the network -- only from an admission point for a Team member or an
#: admitted accept, which ``test_guard_consent.py`` pins with a fixture server.
_INERT_MODULES = ("__init__.py", "decisions.py", "outbox.py")


def test_guard_package_imports_no_network_client() -> None:
    """Inert by construction: no L1 decision module can reach a network client."""
    seen: set[str] = set()
    for path in sorted(GUARD_PKG / name for name in _INERT_MODULES):
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                seen.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                seen.add(node.module.split(".")[0])
                # Any absolute import into local_operator could drag a client in.
                assert node.module.split(".")[0] != "local_operator", (path, node.module)
    assert seen.isdisjoint(_NETWORK_MODULES), seen & _NETWORK_MODULES
