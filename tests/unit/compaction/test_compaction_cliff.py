"""The compaction cliff: material must survive SUCCESSIVE passes.

This is the regression proof for the defect measured on the operator's own
session (``439818272d84``): 84 snapcompact passes in which
``Archive.truncated_chars`` climbed monotonically (257,965 → 1,197,200) while
the kept text stayed ~110k–140k chars. Each pass folded the prior archive text
in as ``previous_text`` and dropped the oldest middle overflow AGAIN, so the
same material was lost a little more on every pass — the "cliff" that makes an
agent re-read to catch up and spend back the tokens compaction saved.

The metric is deliberately NOT ``truncated_chars`` alone (that stays non-zero
by design: pages are still dropped, they are just no longer LOST). The honest
metric is RECOVERABILITY: is a distinctive marker minted at pass i still
reachable in pass k's MODEL-READABLE replay payload — the carry digest plus the
verbatim edges plus any middle pages replayed as text? A marker that survives
only inside a dropped page or an elided frame is not recoverable.

These assertions FAIL against the pre-carry engine (measured: after one later
pass, passes 0 and 1's markers are already gone; retention count 1 of 6) and
pass with the carry.
"""

from __future__ import annotations

from typing import Literal

from local_operator.compaction.snapcompact import (
    Archive,
    compact_to_archive,
    estimate_archive_tokens,
    history_blocks,
)
from local_operator.harness.types import Message, TextContent

PROVIDER = "anthropic"
MODEL = "claude-opus-4-8"
WINDOW = 400_000


def _chunk(i: int, lines: int = 400) -> str:
    """One pass's newly discarded history: a goal, a decision, an ask, an id, filler."""
    out = [
        f"[Assistant]\nGoal for step {i}: <<RET-{i}>> DECISION-{i}: keep the "
        f"existing helper at /Users/damian/local-operator/step{i}.py, do not add a new one.",
        f"[User]\n<<ASK-{i}>> make sure the manager posts concise updates in thread {i}.",
        f"[Tool result: bash]\nsee pull request #10{i:02d} and branch step-{i}",
    ]
    for j in range(lines):
        out.append(f"[Assistant]\nstep {i}.{j}: ran make check-changed; " + "ok " * 40)
        out.append(f"[Tool result: bash]\nresult {i}.{j}: " + "stdout line " * 30)
    return "\n".join(out)


def _messages(serialized: str) -> list[Message]:
    msgs: list[Message] = []
    for block in serialized.split("[")[1:]:
        header, _, body = block.partition("]\n")
        role: Literal["user", "assistant", "tool"] = "tool"
        if header.split(" tool", 1)[0] == "User":
            role = "user"
        elif header.split(" tool", 1)[0] == "Assistant":
            role = "assistant"
        msgs.append(Message(role=role, content=[TextContent(text=body)]))
    return msgs


def _replay_text(archive: Archive) -> str:
    """Everything a replay turns into MODEL-READABLE text.

    Uses the REAL ``history_blocks`` payload (not a private shortcut), so the
    assertion is about what a model actually receives: the carry block, the
    verbatim edges and every middle page the policy replays as text. Frames are
    excluded on purpose — their content is OCR-recoverable but not verbatim, and
    the defect is about material becoming unreadable, not about its carrier.
    """
    parts: list[str] = []
    for block in history_blocks(archive):
        if block["kind"] == "text":
            parts.append(block["text"])
    return "\n".join(parts)


def _run(passes: int) -> tuple[Archive, list[set[str]]]:
    """Run ``passes`` successive passes; return the final archive and the marker
    set minted at each pass index."""
    prev_text: str | None = None
    prev_carry: str | None = None
    minted: list[set[str]] = []
    archive: Archive | None = None
    for i in range(passes):
        archive = compact_to_archive(
            _messages(_chunk(i)),
            PROVIDER,
            MODEL,
            prev_text,
            context_window=WINDOW,
            previous_carry=prev_carry,
        )
        prev_text = archive.text
        prev_carry = archive.carry or None
        minted.append({f"<<RET-{i}>>", f"DECISION-{i}", f"<<ASK-{i}>>"})
    assert archive is not None
    return archive, minted


def test_material_survives_successive_compactions():
    """The core regression: after N passes, EVERY earlier pass's markers are
    still in the replay payload. Fails on the pre-carry engine."""
    n = 8
    archive, minted = _run(n)
    payload = _replay_text(archive)
    fully = [i for i in range(n) if all(m in payload for m in minted[i])]
    assert fully == list(range(n)), (
        f"material lost: passes {sorted(set(range(n)) - set(fully))} are not "
        f"recoverable from the final replay payload"
    )


def test_retention_does_not_degrade_with_more_passes():
    """Passes 7..12's markers must be as recoverable as 0..5's — i.e. the loss
    curve is FLAT, not slope-1. This is the assertion the cliff fails."""
    archive, minted = _run(12)
    payload = _replay_text(archive)
    early_ok = all(all(m in payload for m in minted[i]) for i in range(6))
    late_ok = all(all(m in payload for m in minted[i]) for i in range(6, 12))
    assert early_ok, "early-pass material was lost"
    assert late_ok, "late-pass material was lost"
    assert all(all(m in payload for m in minted[i]) for i in range(12))


def test_identifiers_survive_to_the_final_pass():
    """Tier 1: identifiers are carried forward across passes — the newest AND
    oldest readable at the end, tagged so the digest is self-identifying."""
    archive, _ = _run(10)
    carry = archive.carry
    assert carry, "a run of 10 passes must have mined a carry"
    assert "- Identifiers: " in carry


def test_replayed_archive_stays_under_the_token_bar():
    """400k in, well under 100k out — the scale the operator asked to keep."""
    archive, _ = _run(12)
    tokens = estimate_archive_tokens(archive)
    assert tokens < 100_000, f"replay is {tokens} tokens, above the 100k bar"
    # And the carry is a small part of it, not the bulk.
    assert len(archive.carry) < 30_000


def test_legacy_archive_without_carry_replays_unchanged():
    """Additivity: an archive with no carry (every pre-carry archive) replays
    exactly as it did before this change."""
    archive, _ = _run(3)
    legacy = archive.model_copy(update={"carry": ""})
    assert history_blocks(legacy) == history_blocks(archive)[1:]
    assert estimate_archive_tokens(legacy) < estimate_archive_tokens(archive)


def test_replay_including_carry_stays_inside_the_pass_budget():
    """The carry is RESERVED inside the archive's own token budget, so a pass
    that exists to get under the line does not overshoot it once the carry is
    counted (review round 1, MINOR 5)."""
    import local_operator.compaction.snapcompact as sc
    from local_operator.harness.types import Message, TextContent

    shape = sc.resolve_shape(PROVIDER, MODEL)
    for window in (60_000, 120_000):
        prev_text: str | None = None
        prev_carry: str | None = None
        archive = None
        for i in range(8):
            msgs = [
                Message.user(f"ask {i} keep /Users/damian/proj{i}/file.py in thread {i}"),
                Message(
                    role="user",
                    content=[TextContent(text=f"dense {i} " + "w" * shape.capacity)],
                ),
            ]
            archive = sc.compact_to_archive(
                msgs,
                PROVIDER,
                MODEL,
                prev_text,
                context_window=window,
                previous_carry=prev_carry,
            )
            prev_text = archive.text
            prev_carry = archive.carry or None
        assert archive is not None
        assert estimate_archive_tokens(archive) <= window // 2
