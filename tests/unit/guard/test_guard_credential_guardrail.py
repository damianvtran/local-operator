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
    # Round 1 (review MINOR-1): a monitor create is judged as the call it schedules.
    (
        "monitor",
        {"op": "create", "tool": "bash", "arguments": {"command": "cat ~/.local-operator/auth.db"}},
    ),
    (
        "monitor",
        {
            "op": "create",
            "tool": "bash",
            "arguments": {"command": "cat ~/.local-operator/guard/install.key"},
        },
    ),
    # Round 1 (security S-2): the in-process engine and the desktop signing route.
    (
        "eval",
        {
            "code": "from local_operator.guard.consent import GuardConsentHost as H; "
            "H(root).accept('ten', 1, capability=True, signature=None, source='x')"
        },
    ),
    ("bash", {"command": "python -c 'from local_operator.guard import consent_op'"}),
    (
        "bash",
        {
            "command": 'curl -X POST -H "Authorization: Bearer $(jq -r .token run.json)" '
            "http://127.0.0.1:1/v1/desktop/approvals/ap_1/approve-operation"
        },
    ),
    (
        "bash",
        {"command": "curl -X POST http://127.0.0.1:1/v1/desktop/approvals/ap_1/apply-operation"},
    ),
    # Round 1 (QA Q2): glob, cd and string-built spellings of the key file.
    ("bash", {"command": "cat ~/.local-operator/guard/inst*.key"}),
    ("bash", {"command": "cat ~/.local-operator/guard/*"}),
    ("bash", {"command": "cd ~/.local-operator/guard && cat install.key"}),
    ("eval", {"code": "open(os.path.expanduser('~/.local-operator/guard/'+'install'+'.key'))"}),
    ("eval", {"code": "open(os.path.join(home, '.local-operator', 'guard', 'install.key'))"}),
    ("bash", {"command": "security find-generic-password -s com.radient.gu''ard -w"}),
    # L2b (L2a security S-7): the file store's per-writer temp holds the key too.
    ("bash", {"command": "cat ~/.local-operator/guard/install.key.0a1b2c3d.tmp"}),
    # L2b (L2a NIT-R2-1, narrowed): a pipe into a bare python RUNS what flows in.
    ("bash", {"command": "echo 'from local_operator.guard import consent' | python3"}),
    (
        "bash",
        {"command": "rg x | python3 -c 'from local_operator.guard import consent_op'"},
    ),
]

ALLOWED = [
    # Developers work on these modules: naming the strings is not reading the key.
    ("bash", {"command": "grep -rn com.radient.guard local_operator/"}),
    ("bash", {"command": "grep -rn '/v1/guard/consents' local_operator/"}),
    ("edit", {"path": "local_operator/guard/install_key.py", "old_text": "guard/install.key"}),
    ("write", {"path": "notes.md", "content": "curl https://x/v1/guard/consents"}),
    ("read", {"path": "local_operator/guard/consent.py"}),
    # Round 1 widening must not refuse source work on the Guard package.
    ("bash", {"command": "grep -rn GuardConsentHost local_operator/"}),
    ("bash", {"command": "grep -n accept_signed local_operator/guard/*"}),
    ("bash", {"command": "cat local_operator/guard/install_key.py"}),
    ("bash", {"command": ".venv/bin/python -m pytest tests/unit/guard/ -q"}),
    ("bash", {"command": "cd tests/unit/guard && ls"}),
    ("bash", {"command": "git diff -- local_operator/server/routes/desktop_credentials.py"}),
    ("monitor", {"op": "create", "tool": "bash", "arguments": {"command": "git status"}}),
    # L2b (L2a review NIT-R2-2): the key name is matched exactly, not by prefix.
    ("bash", {"command": "cat ~/.local-operator/guard/install.keyring"}),
    # L2b (L2a review NIT-R2-1): a pipe into ``python -c`` reads the pipe as DATA.
    (
        "bash",
        {
            "command": "rg -n GuardConsentHost local_operator | "
            "python3 -c 'import sys; print(len(sys.stdin.read()))'"
        },
    ),
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
