"""Cache-aware pruning: superseded reads, useless blanking, and every guard."""

from local_operator.compaction.pruning import (
    MIN_PRUNE_TOKENS,
    SUPERSEDED_NOTICE,
    USELESS_NOTICE,
    compute_suffix_tokens,
    prune_tool_outputs,
)
from local_operator.compaction.tokens import estimate_tokens
from local_operator.harness.types import Message

NOW = 10_000_000
ACTIVE = NOW  # not idle
IDLE_AGO = 6_000_000  # > 5_400_000 ms idle window


def _read_result(path: str, words: int = 300) -> Message:
    """A tool-role message shaped like the loop converts a read ToolResult."""
    message = Message(role="tool", tool_call_id=f"call-{path}", tool_name="read")
    message.content = Message.user(f"content of {path} " + "data " * words).content
    message.provider_payload = {"details": {"path": path}}
    return message


def _assistant_big(words: int = 30_000) -> Message:
    return Message.assistant("filler " * words)


def test_superseded_read_blanked_with_pairing_intact():
    """A later read of the same path blanks the earlier result in place."""
    old_read = _read_result("/repo/a.py")
    messages = [Message.user("hi"), old_read, Message.user("again"), _read_result("/repo/a.py")]
    out, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is True
    assert len(out) == 4  # never deleted
    assert old_read.text == SUPERSEDED_NOTICE
    assert old_read.provider_payload is not None
    assert old_read.provider_payload["pruned"] is True
    assert old_read.provider_payload["details"]["path"] == "/repo/a.py"  # old details kept
    # The newer read is untouched.
    assert out[3].text.startswith("content of /repo/a.py")
    # Pairing survives.
    assert old_read.tool_call_id == "call-/repo/a.py"


def test_distinct_paths_do_not_supersede():
    messages = [_read_result("/a.py"), Message.user("x"), _read_result("/b.py")]
    _, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is False


def test_pathless_results_never_supersede():
    """Two bash outputs with no path are distinct results, not stale copies."""
    first = Message(role="tool", tool_call_id="b1", tool_name="bash")
    first.content = Message.user("out " * 300).content
    second = Message(role="tool", tool_call_id="b2", tool_name="bash")
    second.content = Message.user("more " * 300).content
    _, changed = prune_tool_outputs([first, Message.user("x"), second], NOW, ACTIVE)
    assert changed is False


def test_warm_suffix_guard_blocks_then_idle_flushes():
    """A superseded result with a large suffix sits in the warm cache prefix
    and is skipped — until the idle window proves the cache cold."""
    old_read = _read_result("/repo/a.py")
    messages = [
        old_read,
        _assistant_big(),  # large suffix after the victim
        _read_result("/repo/a.py"),
    ]
    suffix = compute_suffix_tokens(messages)
    assert suffix[0] > 8000  # victim is in the warm prefix

    out, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is False
    assert out[0].text.startswith("content of")

    # Same state, idle past the flush window: everything flushes.
    _, changed = prune_tool_outputs(messages, NOW, NOW - IDLE_AGO)
    assert changed is True
    assert messages[0].text == SUPERSEDED_NOTICE


def test_useless_flag_blanked():
    useless = Message(role="tool", tool_call_id="g1", tool_name="grep")
    useless.content = Message.user("nothing matched " * 100).content
    useless.provider_payload = {"useless": True, "details": {"pattern": "foo"}}
    messages = [Message.user("q"), useless, Message.user("next")]
    _, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is True
    assert useless.text == USELESS_NOTICE
    assert useless.provider_payload["pruned"] is True
    assert useless.provider_payload["useless"] is True  # prior payload preserved


def test_error_results_never_blanked():
    """is_error wins over both supersede and useless flags."""
    err_superseded = _read_result("/repo/a.py")
    err_superseded.is_error = True
    useless_err = Message(role="tool", tool_call_id="e2", tool_name="bash")
    useless_err.content = Message.user("boom " * 300).content
    useless_err.is_error = True
    useless_err.provider_payload = {"useless": True}
    messages = [
        err_superseded,
        Message.user("retry"),
        _read_result("/repo/a.py"),
        useless_err,
    ]
    _, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is False
    assert err_superseded.text.startswith("content of")
    assert useless_err.text.startswith("boom")


def test_min_prune_floor_skips_tiny_results():
    """Blanking below MIN_PRUNE_TOKENS costs more than it saves — skip."""
    tiny = _read_result("/repo/small.py", words=1)  # well under 50 tokens
    assert estimate_tokens(tiny) < MIN_PRUNE_TOKENS
    messages = [tiny, Message.user("reread"), _read_result("/repo/small.py", words=1)]
    _, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is False
    assert tiny.text.startswith("content of")


def test_skill_reads_protected():
    """skill:// reads and the skill tool are exempt — a pruned skill loops.

    Fixtures use the REAL read-tool shape: the loop stores
    ``provider_payload = {"details": result.details, ...}``, and builtin read
    reports internal-URL targets under ``details['url']`` (tools/builtin.py).
    Protection must match on BOTH ``path`` and ``url`` for read results, in
    BOTH prune passes (protected tools apply to superseded and useless).
    """
    # Internal-URL read: details['url'], not details['path'].
    skill_read = Message(role="tool", tool_call_id="call-s1", tool_name="read")
    skill_read.content = Message.user("skill content " * 400).content
    skill_read.provider_payload = {"details": {"url": "skill://deploy-guide"}}
    later_skill = Message(role="tool", tool_call_id="call-s1b", tool_name="read")
    later_skill.content = Message.user("skill content " * 400).content
    later_skill.provider_payload = {"details": {"url": "skill://deploy-guide"}}

    skill_tool = Message(role="tool", tool_call_id="s2", tool_name="skill")
    skill_tool.content = Message.user("skill body " * 300).content
    skill_tool.provider_payload = {"details": {"name": "x"}}
    later_skill_tool = Message(role="tool", tool_call_id="s3", tool_name="skill")
    later_skill_tool.content = Message.user("skill body " * 300).content

    # A USELESS-flagged skill read is equally protected (useless pass).
    useless_skill = Message(role="tool", tool_call_id="call-s2", tool_name="read")
    useless_skill.content = Message.user("stale skill " * 300).content
    useless_skill.provider_payload = {"useless": True, "details": {"url": "skill://other"}}

    messages = [
        skill_read,
        Message.user("x"),
        later_skill,
        skill_tool,
        later_skill_tool,
        useless_skill,
    ]
    _, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is False
    assert skill_read.text.startswith("skill content")
    assert skill_tool.text.startswith("skill body")
    assert useless_skill.text.startswith("stale skill")


def test_seconds_based_gap_does_not_idle_flush():
    """RC-8 regression: ``now_ms``/``last_activity_ms`` are MILLISECONDS. A
    90-minute gap expressed in seconds is far below the ms idle window, so it
    must NOT flush the warm prefix."""
    old_read = _read_result("/repo/a.py")
    messages = [
        old_read,
        _assistant_big(),  # large suffix: victim sits in the warm prefix
        _read_result("/repo/a.py"),
    ]
    assert compute_suffix_tokens(messages)[0] > 8000
    # 90 minutes in SECONDS = 5400 << idle_flush_ms; must stay gated.
    _, changed = prune_tool_outputs(messages, 1_000_000, 1_000_000 - 5_400)
    assert changed is False
    assert old_read.text.startswith("content of")
    # The SAME gap in milliseconds does flush.
    _, changed = prune_tool_outputs(messages, 1_000_000, 1_000_000 - 5_400_000)
    assert changed is True
    assert old_read.text == SUPERSEDED_NOTICE


def test_already_pruned_not_reprocessed():
    """A second pass is a no-op: changed=False, notice unchanged."""
    messages = [_read_result("/repo/a.py"), Message.user("x"), _read_result("/repo/a.py")]
    _, first = prune_tool_outputs(messages, NOW, ACTIVE)
    assert first is True
    notice = messages[0].text
    _, second = prune_tool_outputs(messages, NOW, ACTIVE)
    assert second is False
    assert messages[0].text == notice


def test_empty_input():
    out, changed = prune_tool_outputs([], NOW, ACTIVE)
    assert out == []
    assert changed is False


def test_suffix_tokens_strictly_after():
    """suffix[i] counts only messages strictly after i (reversed accumulate)."""
    a, b, c = Message.user("one"), Message.user("two two"), Message.user("three three three")
    suffix = compute_suffix_tokens([a, b, c])
    assert suffix[2] == 0
    assert suffix[1] == estimate_tokens(c)
    assert suffix[0] == estimate_tokens(b) + estimate_tokens(c)


def _ranged_read(path: str, range_spec: str, words: int = 120) -> Message:
    """A ranged read result (details carry path + range, like the real tool)."""
    message = Message(role="tool", tool_call_id=f"call-{path}-{range_spec}", tool_name="read")
    message.content = Message.user(f"content of {path}[{range_spec}] " + "data " * words).content
    message.provider_payload = {"details": {"path": path, "range": range_spec}}
    return message


def test_nested_range_read_supersedes_the_covered_earlier_range():
    """A later range that fully covers an earlier range of the same file
    blanks the earlier result — the model's re-read of a span already served."""
    covered = _ranged_read("/repo/a.py", "100-500")
    messages = [covered, _ranged_read("/repo/a.py", "1-800")]
    out, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is True
    assert covered.text == SUPERSEDED_NOTICE
    assert out[1].text.startswith("content of /repo/a.py[1-800]")


def test_adjacent_and_partial_overlap_ranges_are_both_kept():
    """Paging ranges (1-500 then 501-900) and partial overlaps (100-400 then
    300-700) carry distinct content — neither must be blanked."""
    a = _ranged_read("/repo/a.py", "1-500")
    b = _ranged_read("/repo/a.py", "501-900")
    out, changed = prune_tool_outputs([a, b], NOW, ACTIVE)
    assert changed is False
    assert a.text.startswith("content of /repo/a.py[1-500]")
    assert b.text.startswith("content of /repo/a.py[501-900]")

    c = _ranged_read("/repo/a.py", "100-400")
    d = _ranged_read("/repo/a.py", "300-700")
    out2, changed2 = prune_tool_outputs([c, d], NOW, ACTIVE)
    assert changed2 is False
    assert c.text.startswith("content of /repo/a.py[100-400]")


def test_identical_ranged_read_is_superseded_like_identical_full():
    """The identical-key supersede still applies to ranged reads."""
    first = _ranged_read("/repo/a.py", "100-500")
    messages = [first, _ranged_read("/repo/a.py", "100-500")]
    out, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is True
    assert first.text == SUPERSEDED_NOTICE
