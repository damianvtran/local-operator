"""The built-in ``guard-credential`` guardrail (Guard RFC v0.9 §5.7a, slice L2a).

Driven through ``check_tool_call`` -- the checkpoint ``harness/loop.py`` runs
AFTER approval and BEFORE ``tool.execute`` in every approval mode -- and through
the loop's own setup-failure branch, so ``--yolo`` / auto / ask cannot differ:
nothing about the approval mode reaches this function. Defence in depth only;
the RFC claims no more for a string classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.guard import decisions as gd
from local_operator.guardrails import check as check_mod
from local_operator.guardrails.check import GuardSettings, check_tool_call
from tests.unit.guard import test_guard_decision_sink as _sink_tests
from tests.unit.guard.test_guard_decision_sink import _Collect, _ctx

# Fixtures re-exported under their names (pytest discovers module attributes).
rules_dir = _sink_tests.rules_dir
sink = _sink_tests.sink

REFUSED = [
    ("bash", {"command": "security find-generic-password -s com.radient.guard -w"}),
    ("bash", {"command": "secret-tool lookup service com.radient.guard account x"}),
    ("bash", {"command": "cat ~/.local-operator/guard/install.key"}),
    ("bash", {"command": "sqlite3 ~/.local-operator/auth.db 'select * from tokens'"}),
    ("eval", {"code": "import keyring; keyring.get_password('com.radient.guard', 'x')"}),
    (
        "bash",
        {"command": "curl -X POST https://api.radienthq.com/v1/guard/consents -d @grant.json"},
    ),
    ("eval", {"code": "import httpx; httpx.put(base + '/v1/guard/org', json={})"}),
    ("web_fetch", {"url": "https://api.radienthq.com/v1/guard/consents"}),
    ("read", {"path": "/Users/x/.local-operator/guard/install.key"}),
]

ALLOWED = [
    # Developers work on these modules: naming the strings is not reading the key.
    ("bash", {"command": "grep -rn com.radient.guard local_operator/"}),
    ("bash", {"command": "grep -rn '/v1/guard/consents' local_operator/"}),
    ("edit", {"path": "local_operator/guard/install_key.py", "old_text": "guard/install.key"}),
    ("write", {"path": "notes.md", "content": "curl https://x/v1/guard/consents"}),
    ("read", {"path": "local_operator/guard/consent.py"}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ask", "auto", "yolo"])
@pytest.mark.parametrize(("tool", "args"), REFUSED)
async def test_guard_key_material_and_routes_are_refused_in_every_mode(
    sink: _Collect,
    rules_dir: Path,
    tmp_path: Path,
    mode: str,
    tool: str,
    args: dict[str, Any],
) -> None:
    del mode  # the checkpoint takes no mode: that IS the property under test
    outcome = await check_tool_call(
        tool, args, _ctx(tmp_path), judge=None, settings=GuardSettings(rules_dirs=(rules_dir,))
    )
    assert outcome is not None and outcome.blocked
    assert outcome.rule == check_mod.GUARD_CREDENTIAL_RULE
    assert outcome.text.startswith(f"{check_mod.BLOCK_MARKER} guard-credential (block)")
    # Recorded as the guard protecting its own authority, inside the 4-class grid.
    assert [(r.outcome, r.rule_class) for r in sink.records] == [
        (gd.OUTCOME_BLOCK, gd.RULE_CLASS_POLICY_WRITE)
    ]


@pytest.mark.asyncio
async def test_refused_even_with_user_rules_disabled(rules_dir: Path, tmp_path: Path) -> None:
    outcome = await check_tool_call(
        "bash",
        {"command": "security find-generic-password -s com.radient.guard -w"},
        _ctx(tmp_path),
        judge=None,
        settings=GuardSettings(enabled=False, rules_dirs=(rules_dir,)),
    )
    assert outcome is not None and outcome.blocked


@pytest.mark.parametrize(("tool", "args"), ALLOWED)
def test_mentioning_guard_names_in_source_work_is_not_refused(
    tool: str, args: dict[str, Any]
) -> None:
    assert check_mod.guard_credential_hit(tool, args) is None


@pytest.mark.asyncio
async def test_tool_class_lookup_failure_degrades_to_other(
    sink: _Collect,
    rules_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L1 MINOR-1: the telemetry lookup can never fail the call before the guard runs."""

    def broken(_name: str) -> str:
        raise RuntimeError("lookup broke")

    monkeypatch.setattr(gd, "tool_class_of", broken)
    outcome = await check_tool_call(
        "bash",
        {"command": "security find-generic-password -s com.radient.guard -w"},
        _ctx(tmp_path),
        judge=None,
        settings=GuardSettings(rules_dirs=(rules_dir,)),
    )
    assert outcome is not None and outcome.rule == check_mod.GUARD_CREDENTIAL_RULE
    assert [r.tool_class for r in sink.records] == [gd.TOOL_CLASS_OTHER]


def test_a_sink_raising_base_exception_never_escapes(sink: _Collect) -> None:
    """L1 NIT-1: BaseException from a sink is swallowed; Ctrl-C still propagates."""

    class Base:
        def emit(self, record: gd.GuardDecisionRecord) -> None:
            raise GeneratorExit("sink machinery")

    class Interrupt:
        def emit(self, record: gd.GuardDecisionRecord) -> None:
            raise KeyboardInterrupt

    previous = gd.set_decision_sink(Base())
    try:
        gd.emit_decision(gd.OUTCOME_BLOCK, gd.RULE_CLASS_CUSTOM, gd.TOOL_CLASS_OTHER)
        gd.set_decision_sink(Interrupt())
        with pytest.raises(KeyboardInterrupt):
            gd.emit_decision(gd.OUTCOME_BLOCK, gd.RULE_CLASS_CUSTOM, gd.TOOL_CLASS_OTHER)
    finally:
        gd.set_decision_sink(previous)
