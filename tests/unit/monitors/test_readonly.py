"""The §6.9 matrix: every row of the read-only contract, as a test.

The matrix IS this file — it is the safety core's deliverable. Accept rows
assert ``None`` (the call is monitorable); reject rows assert the refusal's
DISCRIMINATING phrase, so a refusal that drifts to the wrong clause, or a rule
that fires in the wrong order, fails a test rather than passing both ways.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from local_operator.harness.types import AgentTool, ToolResult
from local_operator.monitors import readonly


def fake_tool(
    name: str,
    tier: Literal["read", "write", "exec"] = "read",
    *,
    call_tier: Any = None,
    mcp_annotations: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> AgentTool:
    async def _exec(*_args: Any, **_kwargs: Any) -> ToolResult:  # pragma: no cover
        return ToolResult(tool_call_id="x")

    return AgentTool(
        name=name,
        approval_tier=tier,
        call_approval_tier=call_tier,
        parameters=parameters or {"type": "object", "properties": {}},
        mcp_annotations=mcp_annotations,
        execute=_exec,
    )


def verdict(tool: AgentTool, args: dict[str, Any] | None = None) -> str | None:
    return readonly.readonly_verdict(tool, args or {})


# ---------------------------------------------------------------------------
# Accept rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name",
    ["read", "grep", "glob", "web_fetch", "web_search"],
)
def test_read_tier_tools_are_monitorable(tool_name: str) -> None:
    assert verdict(fake_tool(tool_name)) is None


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("hub", {"op": "list"}),
        ("hub", {"op": "peek"}),
        ("jobs", {"op": "list"}),
        ("jobs", {"op": "peek"}),
        ("todo", {"op": "view"}),
        ("agent", {"op": "list"}),
        ("agent", {"op": "show"}),
        ("team", {"op": "list"}),
        ("team", {"op": "show"}),
        ("project", {"op": "list"}),
        ("project", {"op": "show"}),
        ("lsp", {"action": "definitions"}),
        ("network", {"action": "status"}),
        ("network", {"action": "show"}),
        ("network", {"action": "peers"}),
        ("network", {"action": "log"}),
        ("console", {"method": "read"}),
        ("console", {"method": "status"}),
        ("console", {"method": "list"}),
    ],
)
def test_observing_verbs_of_op_bearing_read_tools_are_monitorable(
    tool_name: str, args: dict[str, Any]
) -> None:
    assert verdict(fake_tool(tool_name, tier="read"), args) is None


def test_mcp_tool_with_read_only_hint_true_is_monitorable() -> None:
    tool = fake_tool(
        "mcp__slack__conversations_history",
        tier="exec",
        mcp_annotations={"readOnlyHint": True, "title": "History"},
    )
    assert verdict(tool, {"channel": "C1"}) is None


@pytest.mark.parametrize(
    "command",
    [
        "gh pr view 1710 --json state,reviews",
        "git status --porcelain",
        "git branch -vv",
        "git branch --merged main",
        "git branch --no-merged main",
        "git branch --format '%(refname)'",
        "git log --oneline -n 5",
        "git log --pretty",
        "git log --pretty=%H -n 1",
        "git log --pretty --oneline -n1",
        "git diff --stat --cached",
        "git diff -U",
        "git diff -U3",
        "git diff --color",
        "git diff --color=never --stat",
        "git show --stat HEAD",
        "git show --pretty=%H",
        "git status -u --short",
        "git status --untracked-files=no --short",
        "git blame -L 1,5 f.py",
        "git rev-parse --verify HEAD",
        "git rev-parse --short HEAD",
        "git rev-parse --short=7 HEAD",
        "git ls-files --stage",
        "git grep -n pattern",
        "ls -la",
        "cat f | grep x | wc -l",
        "find . -name '*.md'",
        "ls x|grep foo",
        "rg 'a|b' f",
        "ls \\-la",
        "date +%s",
        "date -u",
        "df -h .",
        "du -sh .",
        "stat -f %z f",
        "head -n 5 f",
        "tail -c 100 f",
        "wc -l f",
        "file --mime-type f",
        "ps -ef",
        "uname -a",
    ],
)
def test_provably_read_only_bash_commands_are_monitorable(command: str) -> None:
    assert verdict(fake_tool("bash", tier="exec"), {"command": command}) is None


# ---------------------------------------------------------------------------
# Reject rows — tool classes
# ---------------------------------------------------------------------------


def test_eval_is_refused_by_construction() -> None:
    reason = verdict(fake_tool("eval", tier="exec"))
    assert reason is not None
    assert "arbitrary Python" in reason


def test_static_write_tools_are_refused() -> None:
    # Every write-tier tool is refused for the same reason: the tier check.
    # `bash` is NOT among them (its static exec tier is answered by §6.4), and
    # a name that does not exist is not a special case either — the refusal
    # is the tier's sentence, never a missing-name crash.
    for name in ("task", "browser", "edit", "write"):
        reason = verdict(fake_tool(name, tier="write"))
        assert reason is not None, name
        assert 'tier is "write"' in reason, name
    reason = verdict(fake_tool("task", tier="write"))
    assert reason is not None and "delegates" in reason
    reason = verdict(fake_tool("ask", tier="read"))
    assert reason is not None and "sends a message" in reason
    reason = verdict(fake_tool("wait", tier="read"))
    assert reason is not None and "parks the turn" in reason


def test_a_write_tier_op_of_a_read_tier_tool_is_refused() -> None:
    def hub_tier(args: dict[str, Any]) -> str:
        return "read" if str(args.get("op") or "") in ("list", "peek") else "write"

    hub = fake_tool("hub", tier="write", call_tier=hub_tier)
    reason = verdict(hub, {"op": "send", "target": "x"})
    assert reason is not None
    assert 'tier is "write"' in reason


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("jobs", {"op": "cancel", "job_id": "j1"}),
        ("todo", {"op": "add", "items": ["x"]}),
        ("agent", {"op": "create", "name": "x"}),
        ("team", {"op": "create", "name": "x"}),
        ("project", {"op": "update", "name": "x"}),
        ("network", {"action": "init", "network": "n"}),
        ("network", {"action": "invite"}),
    ],
)
def test_mutating_verbs_under_a_read_tier_are_refused(tool_name: str, args: dict[str, Any]) -> None:
    reason = verdict(fake_tool(tool_name, tier="read"), args)
    assert reason is not None
    assert "monitor can't watch" in reason


def test_console_screenshot_is_refused_even_though_its_tier_is_read() -> None:
    # The tier vocabulary calls screenshot a read; the monitor definition is
    # stricter ("no filesystem writes") — documented divergence, test-pinned.
    reason = verdict(fake_tool("console", tier="read"), {"method": "screenshot"})
    assert reason is not None
    assert "writes a file to disk" in reason


def test_an_op_bearing_read_tool_outside_the_map_is_refused_fail_closed() -> None:
    """A NEW op-bearing read-tier tool must add itself and its test.

    Nothing about the map should admit an unknown tool whose op surface is not
    wholly observing (round-1 review F6: the class, not one tool).
    """
    schema = {"type": "object", "properties": {"op": {"type": "string"}}}
    tool = fake_tool("brand_new_tool", tier="read", parameters=schema)
    reason = readonly.readonly_verdict(tool, {"op": "list"})
    assert reason is not None
    assert "not in the monitor observing map" in reason
    assert "add it and its test" in reason


def test_a_plain_read_tool_with_no_op_parameter_is_monitorable() -> None:
    # The fail-closed rule keys on an ``op`` argument being SENTable, not on
    # every read-tier tool needing a map row.
    tool = fake_tool("ls", tier="read")
    assert readonly.readonly_verdict(tool, {"path": "/tmp"}) is None


# ---------------------------------------------------------------------------
# Reject rows — MCP
# ---------------------------------------------------------------------------


def test_mcp_tool_without_hint_is_refused() -> None:
    reason = verdict(fake_tool("mcp__slack__send_message", tier="exec"))
    assert reason is not None
    assert "readOnlyHint" in reason


def test_mcp_tool_with_false_hint_is_refused() -> None:
    reason = verdict(fake_tool("mcp__x__y", tier="exec", mcp_annotations={"readOnlyHint": False}))
    assert reason is not None
    assert "readOnlyHint" in reason


def test_mcp_hint_must_be_literally_true() -> None:
    # "true" (a string), 1, or a missing key are all absent/false per §6.5.
    for hint in ("true", 1, None):
        tool = fake_tool("mcp__x__y", tier="exec", mcp_annotations={"readOnlyHint": hint})
        assert verdict(tool) is not None


# ---------------------------------------------------------------------------
# Reject rows — bash (each names its rule's discriminating phrase)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "phrase"),
    [
        ("rm -rf build", "provably read-only"),
        ("python3 -c 'print(1)'", "provably read-only"),
        ("sed -i s/a/b/ f", "provably read-only"),
        ("ls /dev/null|sed 1p", "provably read-only"),
        ("ls x|rm y", "provably read-only"),
        ("git push", "git allow-list"),
        ("git checkout main", "git allow-list"),
        ("git fetch --all", "git allow-list"),
        ("git log --output=/tmp/x", "not an allowed flag"),
        ("git log --output /tmp/x", "not an allowed flag"),
        # Round-1 review F1: an OPTIONAL-value flag (git OPTARG) does not
        # consume the next token, so the refused flag after it is judged as
        # its own word — before the fix these six smuggled a real write
        # (--output) or a real program run (--ext-diff) past the evaluator.
        ("git log --pretty --output=/tmp/x", '"--output" is not an allowed flag'),
        ("git log --pretty --output /tmp/x", '"--output" is not an allowed flag'),
        ("git diff --color --output=FILE", '"--output" is not an allowed flag'),
        ("git diff -U --output=FILE", '"--output" is not an allowed flag'),
        ("git show --pretty --output=FILE", '"--output" is not an allowed flag'),
        ("git log --pretty --ext-diff -p -n 1 HEAD", '"--ext-diff" is not an allowed flag'),
        # Siblings in the same class: the glued-only --format has no separate
        # form, and the other optional-value spellings never swallow either.
        ("git log --format --output=/tmp/x", "glued"),
        ("git show --format --output=/tmp/x", "glued"),
        ("git status -u --show-signature", "not an allowed flag"),
        ("git status --untracked-files --output=/tmp/x", "not an allowed flag"),
        ("git rev-parse --short --output=/tmp/x", "not an allowed flag"),
        # Round-1 review F2: the accidental value-taking shorts are denied and
        # the real short --short is allowed (accept rows above).
        ("git rev-parse -s HEAD", "not an allowed flag"),
        ("git rev-parse -h", "not an allowed flag"),
        # Round-1 review F5: a dangling required value is refused at arm time
        # (every allow-listed tool errors on one; the fail-closed posture
        # names the flag instead of counting a mystery check failure).
        ("head -n", "expects a value"),
        ("git log -n", "expects a value"),
        ("git log --date", "expects a value"),
        ("git log --since", "expects a value"),
        ("git blame -L", "expects a value"),
        ("git diff --ext-diff", "not an allowed flag"),
        ("git show --textconv", "not an allowed flag"),
        ("git log --show-signature", "not an allowed flag"),
        ("git grep -O pager", "not an allowed flag"),
        ("git branch -d x", "not an allowed flag"),
        ("git branch x", "list-only"),
        ("gh pr view --web", "launches a browser"),
        ("gh pr diff -w", "launches a browser"),
        ("gh api repos/x/y", "not a read-only gh subcommand"),
        ("gh pr checkout 1", "not read-only"),
        ("tree -o out.txt", "not an allowed flag"),
        ("file -C", "not an allowed flag"),
        ("date -s '2020-01-01'", "not an allowed flag"),
        ("date 2020-01-01", "SET form"),
        ("head -f x", "not an allowed flag"),
        ("tail -f x", "not an allowed flag"),
        ("find . -delete", "makes find a write"),
        ("find . -fprintf out '%p'", "makes find a write"),
        ("find . -exec rm x \\;", "makes find a write"),
        # The `-exec … {} …` idiom stacks two refusals — the brace rule fires
        # in the character scan before find's primary is reached. Pinned so a
        # reorder that lets `{}` through while find's rule answers fails here.
        ("find . -exec rm {} \\;", "brace expansion"),
        ("rg --pre 'wc -l' .", "runs a program"),
        ("rg --pre-glob '*.py' --pre x .", "runs a program"),
        ("rg '--pre=<cmd>' .", "runs a program"),
        ("rg \\--pre=x .", "runs a program"),
        ("git log '--output=/tmp/x'", "not an allowed flag"),
        ("file '-C'", "not an allowed flag"),
        ("file \\-C", "not an allowed flag"),
        ('cat "$F"', "substitution"),
        ("echo $(date)", "substitution"),
        ("ls > f", "writes to disk"),
        ("cat f > out", "writes to disk"),
        ("ls >> f", "writes to disk"),
        ("cat f < g", "redirects input"),
        ("ls x;rm y", "chains a second command"),
        ("ls x&&rm y", "chains a second command"),
        ("ls x||rm y", "not a pipeline"),
        ("ls x|&rm y", "not a pipeline"),
        ("rg {--pre=./pre.sh,content} f.txt", "brace expansion"),
        ("git log {--output=./out.txt,HEAD}", "brace expansion"),
        ("ls {a,b}", "brace expansion"),
        ("echo {1..9}", "brace expansion"),
        ("ls {,--probe=/tmp/x}", "brace expansion"),
        ("ls {{a,b},c}", "brace expansion"),
        ("rg 'a{2,3}' f", "brace expansion"),
        ("ls 'unclosed", "does not parse"),
        ("ls x y\\", "final character"),
    ],
)
def test_refused_bash_commands_name_their_rule(command: str, phrase: str) -> None:
    reason = verdict(fake_tool("bash", tier="exec"), {"command": command})
    assert reason is not None, command
    assert phrase in reason, f"{command!r}: {reason!r} lacks {phrase!r}"
    assert command.split("\n")[0][:40] in reason  # the command is quoted


def test_git_optional_value_flags_do_not_swallow_a_following_flag() -> None:
    """Round-1 review F1, pinned end to end.

    ``git log --pretty oneline`` was probed against git 2.55.0: git did NOT
    consume ``oneline`` (it errored \"ambiguous argument\"), so a model that
    fed the next token to ``--pretty`` was lying — and the lie let
    ``--output=/tmp/x`` (a real write) and ``--ext-diff`` (a real program run)
    ride through as \"data\". Every smuggled spelling must refuse, naming the
    smuggled flag; every genuine spelling must accept.
    """
    tool = fake_tool("bash", tier="exec")
    smuggled = [
        "git log --pretty --output=/tmp/x",
        "git log --pretty --output /tmp/x",
        "git diff --color --output=FILE",
        "git diff -U --output=FILE",
        "git show --pretty --output=FILE",
        "git log --pretty --ext-diff -p -n 1 HEAD",
    ]
    for command in smuggled:
        reason = verdict(tool, {"command": command})
        assert reason is not None, command
        smuggled_flag = "--ext-diff" if "--ext-diff" in command else "--output"
        assert f'"{smuggled_flag}" is not an allowed flag' in reason, f"{command!r}: {reason!r}"
    genuine = [
        "git log --pretty=%H -n 1",
        "git log --pretty",
        "git log --pretty --oneline -n1",
        "git log --oneline",
        "git diff -U3",
        "git diff -U",
        "git diff --color=never",
        "git show --pretty=%H",
        "git status -u --short",
        "git rev-parse --short HEAD",
    ]
    for command in genuine:
        assert verdict(tool, {"command": command}) is None, command


def test_a_multiline_command_is_refused_naming_the_newline() -> None:
    reason = verdict(
        fake_tool("bash", tier="exec"),
        {"command": "ls /dev/null\nprintf INJECTED"},
    )
    assert reason is not None
    assert "raw newline" in reason and "second command" in reason


def test_a_bare_dash_and_quoted_operators_survive() -> None:
    # `-` alone is an operand (stdin) and a quoted `>` is data: neither is a
    # refusal, or every `git log -` / `grep '>'` watch would be blocked.
    assert verdict(fake_tool("bash", tier="exec"), {"command": "cat -"}) is None
    assert verdict(fake_tool("bash", tier="exec"), {"command": "grep '>' f"}) is None
    assert verdict(fake_tool("bash", tier="exec"), {"command": 'grep ";" f'}) is None


def test_an_empty_or_missing_bash_command_is_refused() -> None:
    reason = verdict(fake_tool("bash", tier="exec"), {"command": "   "})
    assert reason is not None
    reason = verdict(fake_tool("bash", tier="exec"), {})
    assert reason is not None
    assert 'needs a "command"' in reason


# ---------------------------------------------------------------------------
# The external resolver: the same verdicts, reached WITHOUT a live session
# ---------------------------------------------------------------------------
#
# The arm routes resolve a tool from ``TOOL_BUILDERS`` cold, so its sentences
# must be ``readonly_verdict``'s own for every call the evaluator can judge —
# and the hard rejects especially, because three of them (task/ask/wait) are
# session-only builders: resolving them through the builder branch answered
# "not available without a running session", which sends the caller to a
# session that STILL cannot watch the call. The fragments below are the
# reasons, not the transport (round-1 review F1).


def test_hard_rejects_keep_their_own_sentence_outside_a_session() -> None:
    """task/ask/wait refuse with the REASON, never the session transport.

    The module docstring promises ``readonly_verdict``'s sentence for
    hard-rejected tools; this is that promise as assertions — the
    discriminating phrase of each entry, and the absence of the cold-path
    wording that would bury it.
    """
    cases = [
        ("task", "delegates autonomous work"),
        ("ask", "sends a message to the parent"),
        ("wait", "parks the turn"),
    ]
    for name, phrase in cases:
        reason = readonly.external_monitor_verdict(name, {})
        assert reason is not None, name
        assert phrase in reason, f"{name}: {reason!r}"
        assert "not available without a running session" not in reason, name


def test_the_external_resolver_refuses_by_class() -> None:
    """One request per refusal class, so a branch reordered in the resolver
    fails here rather than shifting which sentence a class receives."""
    # eval: the evaluator's own sentence (its builder resolves cold).
    reason = readonly.external_monitor_verdict("eval", {"code": "1"})
    assert reason is not None, "eval"
    assert "arbitrary Python" in reason, reason

    # bash: the §6.4 verdict, not a tier complaint.
    reason = readonly.external_monitor_verdict("bash", {"command": "rm -rf /"})
    assert reason is not None, "bash"
    assert "is not on the read-only allow-list" in reason, reason

    # A read-tier tool that resolves cold: accepted.
    assert readonly.external_monitor_verdict("network", {"action": "status"}) is None

    # A session-only builder with no evaluator sentence (hub): the transport
    # sentence is the true one, and it is kept distinct from the hard rejects.
    reason = readonly.external_monitor_verdict("hub", {"op": "list"})
    assert reason is not None, "hub"
    assert "not available without a running session" in reason, reason

    # Unknown name / MCP: the two builder-table refusals.
    reason = readonly.external_monitor_verdict("totally_unknown_tool", {})
    assert reason is not None, "unknown"
    assert "not a tool every session builds" in reason, reason
    reason = readonly.external_monitor_verdict("mcp__srv__thing", {})
    assert reason is not None, "mcp"
    assert "read-only hint can only be checked inside a running session" in reason, reason


# ---------------------------------------------------------------------------
# §D2: the shape check — one validator for arm and tick
# ---------------------------------------------------------------------------


def real_tool(name: str) -> AgentTool:
    """A tool built by its own real builder, schema and all.

    The shape check's whole point is the tool's OWN params model, so these rows
    use the shipped builders rather than a hand-written schema: a fake would
    let the check pass while the real ``extra="forbid"`` model still rejected
    the call at tick, which is the bug this closes.
    """
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import TOOL_BUILDERS

    tool = TOOL_BUILDERS[name](ToolContext())
    assert tool is not None, name
    return tool


def test_glob_with_path_is_refused_at_arm() -> None:
    """The arm-time half of the `glob({path: ...})` report: the call was
    accepted, then died on every tick with ``Extra inputs are not permitted``
    and disabled the monitor after five strikes.
    """
    glob_tool = real_tool("glob")
    assert readonly.monitor_call_verdict(glob_tool, {"pattern": "*.py"}) is None
    reason = readonly.monitor_call_verdict(glob_tool, {"pattern": "*.py", "path": "/tmp"})
    assert reason is not None
    assert 'unknown argument(s) "path"' in reason, reason
    assert "glob accepts: pattern." in reason, reason


def test_injected_intent_key_is_refused() -> None:
    """``i`` is declared in the schema the MODEL sees but is stripped by the
    loop before ``execute``; the monitor path calls ``execute`` directly, so a
    nested ``i`` would pass a naive schema check and fail at tick.
    """
    glob_tool = real_tool("glob")
    reason = readonly.monitor_call_verdict(glob_tool, {"pattern": "x", "i": "narrate"})
    assert reason is not None
    assert 'unknown argument(s) "i"' in reason, reason


def test_mcp_extras_are_not_refused() -> None:
    """Arm refuses exactly what a tick refuses. The manager's
    ``prepare_outbound_args`` drops undeclared extras before the call, so a
    tick tolerates them and arm must not be stricter than the run.
    """
    tool = fake_tool(
        "mcp__datadog_search_datadog_hosts",
        mcp_annotations={"readOnlyHint": True},
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "additionalProperties": False,
        },
    )
    assert readonly.monitor_call_verdict(tool, {"query": "x", "extra": 1}) is None


def test_mcp_without_the_read_only_hint_is_still_refused_first() -> None:
    """Order: the safety verdict runs first, so its sentence is byte-identical
    and the shape check can only ADD refusals.
    """
    tool = fake_tool(
        "mcp__srv_thing",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    reason = readonly.monitor_call_verdict(tool, {"anything": 1})
    assert reason is not None
    assert "readOnlyHint" in reason, reason


def test_required_and_type_errors_refused() -> None:
    read_tool = real_tool("read")
    missing = readonly.monitor_call_verdict(read_tool, {"range": "1-5"})
    assert missing is not None and "missing required argument 'path'" in missing, missing
    wrong_type = readonly.monitor_call_verdict(read_tool, {"path": 5})
    assert wrong_type is not None and "does not match type" in wrong_type, wrong_type


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("glob", {"pattern": "*.py"}),
        ("glob", {"pattern": "*.py", "path": "/tmp"}),
        ("read", {"path": "/x"}),
        ("read", {"range": "1-5"}),
    ],
)
def test_in_session_and_external_verdicts_agree(tool_name: str, args: dict[str, Any]) -> None:
    """PARITY: one call must not be describable two ways. The agent's own tool
    goes through ``monitor_call_verdict``; the CLI and the desktop route reach
    the same call through ``external_monitor_verdict``.
    """
    in_session = readonly.monitor_call_verdict(real_tool(tool_name), args)
    external = readonly.external_monitor_verdict(tool_name, args)
    if in_session is None:
        assert external is None
    else:
        assert external == in_session, (tool_name, args, in_session, external)
