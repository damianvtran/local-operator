"""Cache-aware pruning: superseded reads, useless blanking, and every guard."""

from typing import Any

from local_operator.compaction.pruning import (
    MIN_PRUNE_TOKENS,
    STALE_FRAME_NOTICE,
    SUPERSEDED_NOTICE,
    USELESS_NOTICE,
    apply_capture_elisions,
    compute_suffix_tokens,
    partition_captures,
    prune_stale_frames_in_place,
    prune_tool_outputs,
    reclaim_captures_to_wire_budget,
)
from local_operator.compaction.tokens import estimate_tokens, estimate_wire_bytes
from local_operator.harness.types import (
    CapturedFrame,
    ImageContent,
    Message,
    TextContent,
)

_PAYLOAD_FIELD = "".join(["[redacted]", "_payload"])

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
    assert old_read.text == "[Superseded by a newer result for the same resource]"
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


def _observer(call_id: str, name: str, details: dict[str, Any], body: str = "x" * 1200) -> Message:
    return Message(
        role="tool",
        tool_call_id=call_id,
        tool_name=name,
        content=[TextContent(text=f"{call_id} {body}")],
        provider_payload={"details": details},
    )


def _text_len(messages: list[Message]) -> int:
    """Total characters of TEXT content, ignoring non-text blocks."""
    return sum(
        len(block.text)
        for message in messages
        for block in (message.content if isinstance(message.content, list) else [])
        if isinstance(block, TextContent)
    )


def _first_text(message: Message) -> str:
    blocks = message.content if isinstance(message.content, list) else []
    return next((b.text for b in blocks if isinstance(b, TextContent)), "")


def test_a_declared_key_supersedes_re_reads_of_one_resource() -> None:
    """A tool may declare that its result is a re-read of one thing.

    ``details['supersede_key']`` names the CONTENT, so a later result carrying
    the same key describes the same resource in a newer state and the older one
    is dead weight. Polling one endpoint is the common case.
    """
    key = "https://api/status\x00rendered"
    messages = [Message.user("poll")] + [
        _observer(f"c{i}", "web_fetch", {"url": "https://api/status", "supersede_key": key})
        for i in range(6)
    ]
    before = _text_len(messages)

    pruned, changed = prune_tool_outputs(messages, now_ms=0, last_activity_ms=0)

    assert changed
    assert _text_len(pruned) < before * 0.35
    # The newest result must survive intact: it alone describes reality.
    assert "x" * 1200 in _first_text(pruned[-1])


def test_path_and_declared_key_namespaces_cannot_collide() -> None:
    """A path that resembles a declaration must remain a different resource."""
    path = _observer("path", "read", {"path": "declared:/etc/passwd"})
    declared = _observer("declared", "read", {"supersede_key": "/etc/passwd:full"})

    before = _text_len([path, declared])
    pruned, changed = prune_tool_outputs([path, declared], now_ms=0, last_activity_ms=0)

    assert changed is False
    assert _text_len(pruned) == before


def test_a_declared_key_never_blanks_a_different_result() -> None:
    """The saving must come from redundancy only, never from real content.

    This is the axis that makes inference unsafe and the opt-in necessary. A
    browser stamps ONE surface handle on every action it performs, so a key
    inferred from that handle collapses an accessibility snapshot, the console
    log, a scroll and a click into a single group -- and a 40-character click
    result then blanks the snapshot and the errors the agent gathered to decide
    what to click. Only results the tool DECLARES to be the same content may be
    grouped, and a differing declaration must keep them apart.
    """
    surface = {"surface_id": "s:1", "url": "https://app/form", "title": "Form"}
    flow = [
        Message.user("fill the form"),
        _observer("snapshot", "browser", dict(surface)),
        _observer("console", "browser", dict(surface)),
        _observer("scrolled", "browser", dict(surface)),
        _observer("clicked", "browser", dict(surface), body="y" * 40),
    ]
    before = _text_len(flow)
    pruned, _changed = prune_tool_outputs(flow, now_ms=0, last_activity_ms=0)
    assert _text_len(pruned) == before, (
        "a browser surface handle is stamped on every action, so it must not "
        "group them; blanking here loses the snapshot the click depended on"
    )

    # Renditions of one URL are different answers and must not blank each other.
    raw = _observer("raw", "web_fetch", {"url": "https://a", "supersede_key": "https://a\x00raw"})
    rendered = _observer(
        "rendered", "web_fetch", {"url": "https://a", "supersede_key": "https://a\x00rendered"}
    )
    mixed = [Message.user("go"), raw, rendered]
    before_mixed = _text_len(mixed)
    pruned_mixed, _ = prune_tool_outputs(mixed, now_ms=0, last_activity_ms=0)
    assert _text_len(pruned_mixed) == before_mixed

    # And results with no declaration at all stay exempt, as before.
    bare = [Message.user("go")] + [_observer(f"b{i}", "bash", {}) for i in range(8)]
    before_bare = _text_len(bare)
    pruned_bare, _ = prune_tool_outputs(bare, now_ms=0, last_activity_ms=0)
    assert _text_len(pruned_bare) == before_bare


# ---------------------------------------------------------------------------
# The session-side stale-frame sweep: retention window, narrowings, guards
# ---------------------------------------------------------------------------


def _frame(path: str, call_id: str | None = None) -> Message:
    """A tool-role message shaped like the loop converts a read image result."""
    message = Message(role="tool", tool_call_id=call_id or f"call-{path}", tool_name="read")
    message.content = [
        TextContent(text=f"Image {path} (640x480 png, 12 KB)"),
        ImageContent(data="aGVsbG8=", mime_type="image/png"),
    ]
    message.provider_payload = {"details": {"path": path, "mime_type": "image/png"}}
    return message


def test_a_stale_frame_is_blanked_and_its_caption_kept():
    """The sweep drops pixels, never the row: caption and pairing survive."""
    oldest, middle, newest = [_frame(f"/shots/{i}.png") for i in range(3)]
    messages = [Message.user("look"), oldest, middle, newest]

    dropped = prune_stale_frames_in_place(messages, now_ms=NOW, last_activity_ms=ACTIVE)

    assert dropped == 1
    assert not any(isinstance(block, ImageContent) for block in oldest.content)
    assert STALE_FRAME_NOTICE in oldest.text
    assert "Image /shots/0.png (640x480 png, 12 KB)" in oldest.text
    assert oldest.provider_payload is not None
    assert oldest.provider_payload["pruned"] is True
    assert oldest.provider_payload["details"]["path"] == "/shots/0.png"
    assert oldest.tool_call_id == "call-/shots/0.png"


def test_keep_newest_frames_survive_the_sweep():
    """The retention window is the module's own rule: newest two frames kept."""
    frames = [_frame(f"/shots/{i}.png") for i in range(5)]
    dropped = prune_stale_frames_in_place(frames, now_ms=NOW, last_activity_ms=ACTIVE)

    assert dropped == 3
    for victim in frames[:3]:
        assert not any(isinstance(block, ImageContent) for block in victim.content)
    for survivor in frames[3:]:
        assert any(isinstance(block, ImageContent) for block in survivor.content)
        assert STALE_FRAME_NOTICE not in survivor.text


def test_the_retention_window_is_the_callers_choice():
    """``keep_recent_frames`` is honored when a caller pins it wider."""
    frames = [_frame(f"/shots/{i}.png") for i in range(4)]
    dropped = prune_stale_frames_in_place(
        frames, now_ms=NOW, last_activity_ms=ACTIVE, keep_recent_frames=3
    )

    assert dropped == 1
    for survivor in frames[1:]:
        assert any(isinstance(block, ImageContent) for block in survivor.content)


def test_warm_suffix_defers_frames_until_idle():
    """A stale frame inside the warm cache prefix waits for the idle flush."""
    oldest, middle, newest = [_frame(f"/shots/{i}.png") for i in range(3)]
    messages = [oldest, _assistant_big(), middle, newest]
    suffix = compute_suffix_tokens(messages)
    assert suffix[0] > 8000  # the victim sits in the warm prefix

    dropped = prune_stale_frames_in_place(messages, now_ms=NOW, last_activity_ms=ACTIVE)
    assert dropped == 0
    assert any(isinstance(block, ImageContent) for block in oldest.content)

    # Same state, idle past the flush window: the cache is provably cold and
    # everything flushes.
    dropped = prune_stale_frames_in_place(messages, now_ms=NOW, last_activity_ms=NOW - IDLE_AGO)
    assert dropped == 1
    assert not any(isinstance(block, ImageContent) for block in oldest.content)


def test_user_frames_are_never_swept():
    """A paste is not a view of a surface the session drove.

    Never a candidate, even when it sits outside the newest window — and it
    still consumes a slot, exactly as it does in the module's own walk, so a
    newer paste shifts the window the way the module defines it.
    """
    read_a, read_b = [_frame(f"/shots/{i}.png") for i in range(2)]
    paste = Message.user("see this", images=[ImageContent(data="aGVsbG8=", mime_type="image/png")])

    # Newest frame is the user paste: one read falls out of the window and
    # folds; the paste and the other read survive.
    dropped = prune_stale_frames_in_place(
        [read_a, read_b, paste], now_ms=NOW, last_activity_ms=ACTIVE
    )
    assert dropped == 1
    assert not any(isinstance(block, ImageContent) for block in read_a.content)
    assert any(isinstance(block, ImageContent) for block in read_b.content)
    assert any(isinstance(block, ImageContent) for block in paste.content)

    # An OLDER paste is a victim by the retention rule, skipped by the role
    # filter: its pixels stay.
    older_paste = Message.user(
        "older", images=[ImageContent(data="aGVsbG8=", mime_type="image/png")]
    )
    read_c, read_d = [_frame(f"/other/{i}.png") for i in range(2)]
    dropped = prune_stale_frames_in_place(
        [older_paste, read_c, read_d], now_ms=NOW, last_activity_ms=ACTIVE
    )
    assert dropped == 0
    assert any(isinstance(block, ImageContent) for block in older_paste.content)
    assert any(isinstance(block, ImageContent) for block in read_c.content)


def test_error_frames_are_never_swept():
    """Errors are signal: the sweep takes the same never-errors guard
    ``prune_tool_outputs`` takes."""
    failed = _frame("/shots/failed.png")
    failed.is_error = True
    messages = [failed, _frame("/shots/1.png"), _frame("/shots/2.png")]

    dropped = prune_stale_frames_in_place(messages, now_ms=NOW, last_activity_ms=ACTIVE)

    assert dropped == 0
    assert any(isinstance(block, ImageContent) for block in failed.content)


def test_oversized_superseded_victim_blanks_despite_warm_suffix():
    """Fix C: a victim bigger than the warm suffix it invalidates is blanked.

    The warm-suffix guard defers small victims because re-writing the suffix
    costs more than they save. A victim larger than that suffix inverts the
    arithmetic — it would be re-billed on every request until compaction,
    while the suffix is re-written only once.
    """
    big = _read_result("/repo/big.py", words=30000)
    messages = [big, _assistant_big(words=10000), _read_result("/repo/big.py")]
    suffix = compute_suffix_tokens(messages)
    victim_tokens = estimate_tokens(big)
    assert suffix[0] > 8000
    assert victim_tokens > suffix[0], f"victim {victim_tokens} must exceed suffix {suffix[0]}"

    _, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is True
    assert big.text == SUPERSEDED_NOTICE


def test_oversized_victim_smaller_than_suffix_stays_despite_exceeding_floor():
    """R1 discriminating case: victim >= floor but victim < suffix must stay.

    This is the band the fixed-threshold rule got wrong: a 9k-token victim
    behind a 20k-token warm suffix was blanked even though the re-write it
    forces costs more than the victim saves on the next request. The two
    rules (fixed floor vs victim-vs-suffix) disagree ONLY here, so this is
    the test that distinguishes them. Word counts are tuned so the measured
    estimates land in the band; the asserts pin the band itself so a
    tokenizer change that moves the fixture out of it fails loudly instead
    of silently testing the wrong rule.
    """
    victim = _read_result("/repo/big.py", words=9000)
    messages = [victim, _assistant_big(words=20000), _read_result("/repo/big.py")]
    suffix = compute_suffix_tokens(messages)
    victim_tokens = estimate_tokens(victim)
    assert victim_tokens >= 8000, f"victim {victim_tokens} below the floor band"
    assert suffix[0] > victim_tokens, f"suffix {suffix[0]} must exceed victim {victim_tokens}"
    assert suffix[0] > 8000

    _, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is False
    assert victim.text.startswith("content of")


def test_oversized_useless_victim_stays_inside_warm_suffix():
    """R2: the oversized exception is supersede-only.

    A useless-flagged result is worthless once consumed by definition, so
    blanking it inside the warm suffix buys no re-billing repayment the way
    a superseded victim does. Even a useless victim larger than its suffix
    keeps the plain warm-suffix guard (it still blanks once idle, like every
    other candidate — that path is covered by the idle-flush tests).
    """
    useless = Message(role="tool", tool_call_id="g-big", tool_name="grep")
    useless.content = Message.user("nothing matched " * 8000).content
    useless.provider_payload = {"useless": True, "details": {"pattern": "foo"}}
    messages = [useless, _assistant_big(words=10000)]
    suffix = compute_suffix_tokens(messages)
    victim_tokens = estimate_tokens(useless)
    assert victim_tokens >= 8000
    assert suffix[0] > 8000
    assert victim_tokens > suffix[0], (
        f"useless victim {victim_tokens} must outweigh suffix {suffix[0]} — "
        "otherwise this does not prove the exception is supersede-only"
    )

    _, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is False
    assert useless.text.startswith("nothing matched")


def test_guide_reads_protected():
    """guide:// reads share the skill exemption (R4/R5-review).

    A filesystem guide read now carries ``details['url'] = guide://...``
    (Fix A); before the exemption was extended it stayed supersede-prunable
    under ``details['path']`` and a superseded shaped guide blanked into the
    re-read loop the skill exemption exists to prevent.
    """
    guide_read = Message(role="tool", tool_call_id="call-g1", tool_name="read")
    guide_read.content = Message.user("guide content " * 400).content
    guide_read.provider_payload = {
        "details": {"path": "/guides/browser/GUIDE.md", "url": "guide://browser"}
    }
    later_guide = Message(role="tool", tool_call_id="call-g1b", tool_name="read")
    later_guide.content = Message.user("guide content " * 400).content
    later_guide.provider_payload = {
        "details": {"path": "/guides/browser/GUIDE.md", "url": "guide://browser"}
    }
    messages = [guide_read, Message.user("x"), later_guide]
    _, changed = prune_tool_outputs(messages, NOW, ACTIVE)
    assert changed is False
    assert guide_read.text.startswith("guide content")


def _capture(*, stream: str, revision: str, scope: str) -> ImageContent:
    """One image carrying harness-stamped capture provenance."""
    return ImageContent(
        data="aGVsbG8=",
        mime_type="image/png",
        captured_frame=CapturedFrame(stream_id=stream, revision_id=revision, producer_scope=scope),
    )


def test_partition_reclaims_only_superseded_captures_and_keeps_the_latest():
    """Oldest-first, per stream: the newest capture of every stream stays."""
    messages = [
        Message.user("go", images=[_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.assistant("step"),
        Message.user("see", images=[_capture(stream="screen", revision="2", scope="mcp:s:a")]),
        Message.user("doc", images=[ImageContent(data="aGVsbG8=", mime_type="image/png")]),
        Message.user("poll", images=[_capture(stream="camera", revision="9", scope="mcp:s:a")]),
        Message.user("again", images=[_capture(stream="screen", revision="3", scope="mcp:s:a")]),
    ]

    partition = partition_captures(messages)

    # ``Message.user`` orders text then images, so each image sits at index 1.
    assert [(e.message_index, e.content_index) for e in partition.reclaimable] == [(0, 1), (2, 1)]
    assert [e.revision_id for e in partition.reclaimable] == ["1", "2"]
    assert all(e.producer_scope == "mcp:s:a" for e in partition.reclaimable)
    protected = set(partition.protected)
    # The latest screen capture, the camera capture and the unannotated image stay.
    assert {(3, 1), (4, 1), (5, 1)} <= protected
    assert (0, 1) not in protected and (2, 1) not in protected


def test_partition_scopes_supersession_per_producer():
    """Another producer's newer claim never supersedes this producer's captures.

    Identical stream/revision strings across scopes are deliberately used: the
    identity that matters is the scope the harness stamped, not the strings.
    """
    messages = [
        Message.user("a", images=[_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user("b", images=[_capture(stream="screen", revision="1", scope="mcp:s:b")]),
        Message.user("a2", images=[_capture(stream="screen", revision="2", scope="mcp:s:a")]),
    ]

    partition = partition_captures(messages)

    assert [(e.message_index, e.producer_scope) for e in partition.reclaimable] == [(0, "mcp:s:a")]
    assert (1, 1) in set(partition.protected)  # B's own capture is B's latest


def test_partition_orders_by_arrival_not_revision_values():
    """Canonical arrival decides "latest"; revision strings are opaque."""
    messages = [
        Message.user("x", images=[_capture(stream="screen", revision="9", scope="mcp:s:a")]),
        Message.user("y", images=[_capture(stream="screen", revision="1", scope="mcp:s:a")]),
    ]

    partition = partition_captures(messages)

    assert [e.revision_id for e in partition.reclaimable] == ["9"]
    assert (1, 1) in set(partition.protected)


def test_partition_handles_several_blocks_in_one_message():
    """Two captures in one message: the older block is the reclaimable one."""
    messages = [
        Message.user(
            "pair",
            images=[
                _capture(stream="s", revision="1", scope="p"),
                _capture(stream="s", revision="2", scope="p"),
            ],
        )
    ]

    partition = partition_captures(messages)

    assert [(e.content_index, e.revision_id) for e in partition.reclaimable] == [(1, "1")]
    assert (0, 2) in set(partition.protected)


def test_partition_is_pure_and_deterministic():
    """Same source, same plan, no mutation: replay must reproduce the projection."""
    messages = [
        Message.user("go", images=[_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user("again", images=[_capture(stream="screen", revision="2", scope="mcp:s:a")]),
    ]
    before = [message.model_dump() for message in messages]

    first = partition_captures(messages)
    second = partition_captures(messages)

    assert first == second
    assert [message.model_dump() for message in messages] == before


def _big_capture(*, stream: str, revision: str, scope: str) -> ImageContent:
    """A capture whose payload dominates the wire estimate."""
    return ImageContent(
        data="aGVsbG8=" * 30000,
        mime_type="image/png",
        captured_frame=CapturedFrame(stream_id=stream, revision_id=revision, producer_scope=scope),
    )


def test_reclaim_elides_superseded_captures_until_under_budget():
    """One superseded capture is enough: elide it, stop, leave the rest alone."""
    messages = [
        Message.user("go", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user(
            "again", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:a")]
        ),
    ]
    full = estimate_wire_bytes(messages)
    # Mirror exactly what one elision produces, measured by the SAME ruler, so
    # the budget sits strictly between "before" and "after".
    elided_once = [
        messages[0].model_copy(update={"content": [messages[0].content[0]]}),
        messages[1],
    ]
    after_one = estimate_wire_bytes(elided_once)
    assert after_one < full, "the capture payload must dominate the estimate"
    budget = (after_one + full) // 2

    out, elided = reclaim_captures_to_wire_budget(messages, budget=budget)

    assert len(elided) == 1
    assert (elided[0].stream_id, elided[0].revision_id) == ("screen", "1")
    assert estimate_wire_bytes(out) <= budget
    assert [type(block).__name__ for block in out[0].content] == ["TextContent"]
    assert out[1] is messages[1], "untouched messages keep sharing their objects"
    assert len(messages[0].content) == 2, "the SOURCE is never mutated"


def test_reclaim_below_budget_is_byte_identical_and_free():
    messages = [
        Message.user("go", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user(
            "again", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:a")]
        ),
    ]
    full = estimate_wire_bytes(messages)

    out, elided = reclaim_captures_to_wire_budget(messages, budget=full + 1)

    assert elided == ()
    assert out[0] is messages[0] and out[1] is messages[1]


def test_reclaim_refuses_when_only_protected_images_remain():
    """A single latest capture and an unannotated image are untouchable."""
    messages = [
        Message.user("only", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user("doc", images=[ImageContent(data="aGVsbG8=" * 30000, mime_type="image/png")]),
    ]

    out, elided = reclaim_captures_to_wire_budget(messages, budget=1)

    assert elided == ()
    assert estimate_wire_bytes(out) == estimate_wire_bytes(messages)
    assert out[0] is messages[0] and out[1] is messages[1]


def test_reclaim_never_empties_a_message():
    """A superseded capture that is its message's ONLY block is skipped.

    An empty content list breaks wire and transcript invariants, so the
    function refuses deterministically and the caller records no-relief
    rather than receiving a message nothing can carry.
    """
    only = Message(role="user")
    only.content = [_big_capture(stream="screen", revision="1", scope="mcp:s:a")]
    latest = Message.user(
        "x", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:a")]
    )
    messages = [only, latest]

    out, elided = reclaim_captures_to_wire_budget(messages, budget=1)

    assert elided == ()
    assert len(out[0].content) == 1
    assert out[0] is only


def test_reclaim_is_deterministic_and_pure():
    messages = [
        Message.user("go", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user("mid", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:a")]),
        Message.user("now", images=[_big_capture(stream="screen", revision="3", scope="mcp:s:a")]),
    ]
    before = [message.model_dump() for message in messages]

    first, first_elided = reclaim_captures_to_wire_budget(messages, budget=1)
    second, second_elided = reclaim_captures_to_wire_budget(messages, budget=1)

    assert len(first_elided) == len(second_elided) == 2  # superseded pair; latest stays
    assert [capture.revision_id for capture in first_elided] == ["1", "2"]
    assert first == second
    assert [message.model_dump() for message in messages] == before


def test_apply_capture_elisions_reproduces_the_byte_rung_projection():
    """The recorded addresses replay to the live projection: same source, same plan."""
    messages = [
        Message.user("go", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user("mid", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:a")]),
        Message.user("now", images=[_big_capture(stream="screen", revision="3", scope="mcp:s:a")]),
    ]
    rung, elided = reclaim_captures_to_wire_budget(messages, budget=1)
    assert [capture.revision_id for capture in elided] == ["1", "2"]
    identities = tuple(
        (capture.producer_scope, capture.stream_id, capture.revision_id) for capture in elided
    )

    replayed, applied_back = apply_capture_elisions(messages, identities=identities)

    assert replayed == rung
    assert [capture.revision_id for capture in applied_back] == ["1", "2"]


def test_apply_capture_elisions_never_empties_a_message() -> None:
    """A recorded address whose block is its message's ONLY block is skipped,
    not turned into an empty content list (the same guard the live rung has)."""
    only = Message(role="user")
    only.content = [_big_capture(stream="screen", revision="1", scope="mcp:s:a")]
    latest = Message.user(
        "x", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:a")]
    )
    messages = [only, latest]

    out, elided = apply_capture_elisions(messages, identities=[("mcp:s:a", "screen", "1")])

    assert elided == ()
    assert len(out[0].content) == 1
    assert out[0] is only


def test_apply_capture_elisions_empty_or_unmatched_and_purity() -> None:
    messages = [
        Message.user("go", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user("mid", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:a")]),
        Message.user("now", images=[_big_capture(stream="screen", revision="3", scope="mcp:s:a")]),
    ]
    before = [message.model_dump() for message in messages]

    untouched, elided_none = apply_capture_elisions(messages, identities=())
    assert elided_none == ()
    assert untouched[0] is messages[0]

    unmatched, elided_unmatched = apply_capture_elisions(
        messages, identities=[("mcp:s:a", "screen", "99")]
    )
    assert elided_unmatched == ()
    assert unmatched[2] is messages[2]

    again, elided_again = apply_capture_elisions(
        messages,
        identities=[("mcp:s:a", "screen", "1"), ("mcp:s:a", "screen", "2")],
    )
    assert [capture.revision_id for capture in elided_again] == ["1", "2"]
    assert again[2] is messages[2]
    assert [message.model_dump() for message in messages] == before


def test_apply_capture_elisions_bounds_duplicate_identities_by_occurrence() -> None:
    """A repeated claim elides only as many occurrences as the record carries."""
    messages = [
        Message.user("a", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user("b", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:a")]),
        Message.user("c", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:a")]),
    ]
    identity = ("mcp:s:a", "screen", "1")
    block_count = lambda source: sum(  # noqa: E731 -- assertion shorthand, local to the test
        1 for message in source for block in message.content if isinstance(block, ImageContent)
    )

    one, elided_one = apply_capture_elisions(messages, identities=[identity])
    assert len(elided_one) == 1
    assert block_count(one) == 2, "one duplicate occurrence must remain"

    both, elided_both = apply_capture_elisions(messages, identities=[identity, identity])
    assert len(elided_both) == 2
    assert block_count(both) == 1


def test_apply_capture_elisions_uses_recorded_addresses_not_a_fresh_prefix() -> None:
    """The reviewer's case: a later turn retro-supersedes a capture the pass
    protected. A count re-derived over the resumed segment would pick a
    different first-N -- the retro-superseded block PRECEDES the elided one
    in canonical arrival order -- so the recorded addresses must win."""
    b1 = Message.user("b1", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:b")])
    a1 = Message.user("a1", images=[_big_capture(stream="screen", revision="1", scope="mcp:s:a")])
    a2 = Message.user("a2", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:a")])
    at_pass = [b1, a1, a2]

    _, elided = reclaim_captures_to_wire_budget(at_pass, budget=1)
    assert [(capture.producer_scope, capture.revision_id) for capture in elided] == [
        ("mcp:s:a", "1")
    ]
    identities = tuple(
        (capture.producer_scope, capture.stream_id, capture.revision_id) for capture in elided
    )

    # A later turn supersedes B1 -- the block the pass protected.
    b2 = Message.user("b2", images=[_big_capture(stream="screen", revision="2", scope="mcp:s:b")])
    resumed_segment = [b1, a1, a2, b2]
    assert (
        partition_captures(resumed_segment).reclaimable[0].producer_scope == "mcp:s:b"
    ), "the fixture must make the retro-superseded block the first fresh candidate"

    replayed, applied_back = apply_capture_elisions(resumed_segment, identities=identities)

    assert [(capture.producer_scope, capture.revision_id) for capture in applied_back] == [
        ("mcp:s:a", "1")
    ]
    kept_blocks = [
        (block.captured_frame.producer_scope, block.captured_frame.revision_id)
        for message in replayed
        if isinstance(message, Message)
        for block in message.content
        if isinstance(block, ImageContent)
    ]
    assert kept_blocks == [("mcp:s:b", "1"), ("mcp:s:a", "2"), ("mcp:s:b", "2")]
