"""The cold tail's proof of absence (L7 design R1).

WHAT THIS PINS. ``read_cold_tail`` claims a fact either because it MET that
fact's newest row in the bounded tail, or because the fact's byte spelling is
ABSENT from every byte below the tail. The second claim is only sound if the
scan cannot miss a real row. Each test here removes one way it could, and each
compares against eager on the same bytes rather than against a hand-written
expectation:

* a real row buried below the tail must decline, never be "proven absent";
* the type name quoted in a message's TEXT is escaped, so it does not match and
  the journal is still claimed, equal to eager;
* an unescaped spelling inside some other row is a false PRESENCE: safe, declines;
* a spelling split across a scan window boundary is still found;
* a journal whose below-tail span exceeds the scan cap declines unscanned;
* a newest usage receipt that does not validate never lets the tail claim a seed
  eager would not give.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent, Usage
from local_operator.session import replay_cache
from local_operator.session import transcript as transcript_module
from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE
from local_operator.session.transcript import (
    Transcript,
    TranscriptEntry,
    _custom_type_spelling,
    _spellings_absent,
    read_cold_tail,
    read_replay_suffix,
    usages_since_newest_shrink,
)
from local_operator.session.usage_seed import seed_reported_usage

WANTED = (FRONTEND_CHECKPOINT_CUSTOM_TYPE, SESSION_SPEND_CUSTOM_TYPE)


def _facts(entries: Any, checkpoints: dict[str, Any], order: dict[str, int]) -> dict[str, Any]:
    seed = seed_reported_usage(usages_since_newest_shrink(entries))
    return {
        "checkpoint": checkpoints.get(FRONTEND_CHECKPOINT_CUSTOM_TYPE),
        "spend": checkpoints.get(SESSION_SPEND_CUSTOM_TYPE),
        "order": dict(order),
        "seed": seed.model_dump(mode="json") if seed else None,
    }


def eager_facts(directory: Path) -> dict[str, Any]:
    """The cold open's facts as eager computes them, on the same bytes."""
    suffix = read_replay_suffix(
        directory,
        checkpoint_types=WANTED[:1],
        opportunistic_types=WANTED[1:],
    )
    return _facts(suffix.entries, suffix.checkpoints, suffix.checkpoint_order)


def tail_facts(directory: Path, **kwargs: Any) -> dict[str, Any] | None:
    suffix = read_cold_tail(directory, checkpoint_types=WANTED, **kwargs)
    if suffix is None:
        return None
    return _facts(suffix.entries, suffix.checkpoints, suffix.checkpoint_order)


async def _journal(
    tmp_path: Path,
    *,
    turns: int,
    checkpoint_first: bool = False,
    usage: bool = False,
    text: str = "filler",
) -> Path:
    """A legacy-shaped journal: messages only, optionally one old checkpoint.

    Written through ``Transcript``, so ``to_json`` is the writer, which is the
    assumption the scan rests on.
    """
    directory = tmp_path / "session"
    transcript = Transcript(directory)
    if checkpoint_first:
        await transcript.append_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"checkpoint_id": "buried"})
    for turn in range(turns):
        await transcript.append_message(
            Message(id=f"u{turn}", role="user", content=[TextContent(text=f"{text} {turn}")])
        )
        answer = Message(id=f"a{turn}", role="assistant", content=[TextContent(text="ok")])
        if usage and turn == 0:
            answer = answer.model_copy(update={"usage": Usage(input_tokens=1, output_tokens=2)})
        await transcript.append_message(answer)
    transcript.flush()
    return directory


@pytest.mark.asyncio
async def test_a_legacy_journal_is_claimed_with_eager_facts(tmp_path: Path) -> None:
    directory = await _journal(tmp_path, turns=400)
    suffix = read_cold_tail(directory, checkpoint_types=WANTED)
    assert suffix is not None, "nothing exists below the tail, so absence is proven"
    assert suffix.rows_read == 256, "the walk stopped at the row bound"
    assert suffix.usage_proven_absent
    assert 0 < suffix.absence_scan_bytes < (directory / "transcript.jsonl").stat().st_size
    assert tail_facts(directory) == eager_facts(directory)


@pytest.mark.asyncio
@pytest.mark.parametrize("buried", ["checkpoint", "usage"])
async def test_a_fact_buried_below_the_tail_declines(tmp_path: Path, buried: str) -> None:
    """The one false absence the design must make impossible."""
    directory = await _journal(
        tmp_path,
        turns=400,
        checkpoint_first=buried == "checkpoint",
        usage=buried == "usage",
    )
    eager = eager_facts(directory)
    if buried == "checkpoint":
        assert eager["checkpoint"] == {"checkpoint_id": "buried"}, "eager finds it"
    else:
        assert eager["seed"] is not None, "eager seeds from it"
    assert read_cold_tail(directory, checkpoint_types=WANTED) is None


@pytest.mark.asyncio
async def test_the_type_name_inside_message_text_is_not_a_row(tmp_path: Path) -> None:
    """Quoted in TEXT, the type's quotes are escaped, so the spelling does not match.

    The tail claims the journal, and it agrees with eager, which also sees no row.
    """
    text = f'see "custom_type":"{FRONTEND_CHECKPOINT_CUSTOM_TYPE}" in the docs'
    directory = await _journal(tmp_path, turns=400, text=text)
    raw = (directory / "transcript.jsonl").read_bytes()
    spelling = _custom_type_spelling(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert spelling is not None
    assert FRONTEND_CHECKPOINT_CUSTOM_TYPE.encode() in raw
    assert spelling not in raw
    assert read_cold_tail(directory, checkpoint_types=WANTED) is not None
    assert tail_facts(directory) == eager_facts(directory)


@pytest.mark.asyncio
async def test_an_unescaped_spelling_in_another_row_is_a_safe_false_presence(
    tmp_path: Path,
) -> None:
    """A nested object carrying the spelling is no checkpoint row, but it matches.

    The scan cannot tell, so it declines. A false presence costs an eager read and
    nothing else.
    """
    directory = tmp_path / "session"
    transcript = Transcript(directory)
    await transcript.append_custom(
        "some_other_type", {"echo": {"custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE}}
    )
    for turn in range(300):
        await transcript.append_message(
            Message(id=f"m{turn}", role="user", content=[TextContent(text="x")])
        )
    transcript.flush()
    assert eager_facts(directory)["checkpoint"] is None, "eager sees no checkpoint row"
    assert read_cold_tail(directory, checkpoint_types=WANTED) is None


@pytest.mark.parametrize("split", range(1, 30))
def test_a_spelling_split_across_a_window_boundary_is_found(split: int) -> None:
    """Every split point of the needle across the window edge is still a hit."""
    needle = _custom_type_spelling(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert needle is not None and split < len(needle)
    window = 64
    data = b"x" * (window - split) + needle + b"x" * 200
    handle = io.BytesIO(data)
    assert not _spellings_absent(handle, 0, len(data), (needle,), window_bytes=window)
    # The negative control: one byte of the needle damaged and it is absent, so
    # the positive above is the needle being found and not a scan that always
    # answers "present".
    damaged = data.replace(needle, needle[:-2] + b"X" + needle[-1:])
    assert _spellings_absent(io.BytesIO(damaged), 0, len(damaged), (needle,), window_bytes=window)


def test_a_spelling_only_reaching_past_the_range_end_does_not_count() -> None:
    """The overlap never reads past ``end``: a hit there is outside the range."""
    needle = b'"usage":{'
    data = b"x" * 100 + needle
    assert _spellings_absent(io.BytesIO(data), 0, 104, (needle,), window_bytes=16)
    assert not _spellings_absent(io.BytesIO(data), 0, len(data), (needle,), window_bytes=16)


@pytest.mark.asyncio
async def test_a_journal_over_the_scan_cap_declines_without_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = await _journal(tmp_path, turns=400)
    assert read_cold_tail(directory, checkpoint_types=WANTED) is not None

    def no_scan(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("a journal over the cap must not be scanned")

    monkeypatch.setattr(transcript_module, "_spellings_absent", no_scan)
    assert read_cold_tail(directory, checkpoint_types=WANTED, max_scan_bytes=1024) is None


@pytest.mark.asyncio
async def test_a_journal_over_128_mib_declines_at_the_default_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real 130 MiB journal at the DEFAULT cap, not a lowered parameter.

    The head is a sparse hole, so this costs no disk. The tail rows are valid
    and need a scan, which is refused before one byte below the tail is read.
    """
    assert transcript_module.ABSENCE_SCAN_MAX_BYTES == 128 * 1024 * 1024
    small = await _journal(tmp_path, turns=20)
    rows = (small / "transcript.jsonl").read_bytes()
    directory = tmp_path / "big"
    directory.mkdir()
    with (directory / "transcript.jsonl").open("wb") as out:
        out.seek(130 * 1024 * 1024)
        out.write(b"\n" + rows)

    def no_scan(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("a journal over the cap must not be scanned")

    monkeypatch.setattr(transcript_module, "_spellings_absent", no_scan)
    assert read_cold_tail(directory, checkpoint_types=WANTED) is None


@pytest.mark.asyncio
async def test_an_archived_journal_declines(tmp_path: Path) -> None:
    import gzip

    from local_operator.session import compression

    directory = await _journal(tmp_path, turns=50)
    plain = directory / "transcript.jsonl"
    archive = directory / "transcript.jsonl.gz"
    archive.write_bytes(gzip.compress(plain.read_bytes()))
    plain.unlink()
    assert compression.transcript_source(directory)[1] is not None
    assert read_cold_tail(directory, checkpoint_types=WANTED) is None


@pytest.mark.asyncio
async def test_a_torn_tail_declines(tmp_path: Path) -> None:
    directory = await _journal(tmp_path, turns=50)
    with (directory / "transcript.jsonl").open("ab") as out:
        out.write(b'{"id":"torn","ts":1.0,"type":"message"')
    assert read_cold_tail(directory, checkpoint_types=WANTED) is None


@pytest.mark.asyncio
async def test_a_shrink_row_inside_the_tail_before_any_seed_declines(tmp_path: Path) -> None:
    directory = await _journal(tmp_path, turns=50)
    with (directory / "transcript.jsonl").open("a") as out:
        out.write(json.dumps({"id": "c", "ts": 1.0, "type": "compaction", "payload": {}}) + "\n")
    assert read_cold_tail(directory, checkpoint_types=WANTED) is None


@pytest.mark.asyncio
async def test_a_shrink_older_than_the_seed_stops_the_walk_and_matches_eager(
    tmp_path: Path,
) -> None:
    """A shrink BELOW a parseable receipt cannot change the seed, so the walk stops.

    The checkpoint and spend types are then proven absent from everything below,
    the shrink row included. This is the spike's stop rule (``spike4c``/``4d``).
    """
    directory = tmp_path / "session"
    transcript = Transcript(directory)
    await transcript.append_message(Message(id="old", role="user", content=[TextContent(text="x")]))
    transcript.flush()
    with (directory / "transcript.jsonl").open("a") as out:
        out.write(
            TranscriptEntry(
                id="cmp", ts=1.0, type="compaction", payload={"first_kept_entry_id": "old"}
            ).to_json()
            + "\n"
        )
    transcript = Transcript(directory)
    await transcript.append_message(
        Message(
            id="a",
            role="assistant",
            content=[TextContent(text="ok")],
            usage=Usage(input_tokens=3, output_tokens=4),
        )
    )
    transcript.flush()
    suffix = read_cold_tail(directory, checkpoint_types=WANTED)
    assert suffix is not None
    assert [entry.id for entry in suffix.entries] == ["a"]
    assert tail_facts(directory) == eager_facts(directory)


@pytest.mark.asyncio
async def test_an_unparseable_newest_receipt_with_none_older_declines(tmp_path: Path) -> None:
    """Design row 15: a receipt was MET, so usage is not absent, but none parses."""
    directory = await _journal(tmp_path, turns=20)
    with (directory / "transcript.jsonl").open("a") as out:
        out.write(
            TranscriptEntry(
                id="bad",
                ts=1.0,
                type="message",
                payload={"role": "assistant", "content": [], "usage": {"input_tokens": "abc"}},
            ).to_json()
            + "\n"
        )
    assert read_cold_tail(directory, checkpoint_types=WANTED) is None


@pytest.mark.asyncio
async def test_an_unparseable_newest_receipt_keeps_walking_to_the_parseable_one(
    tmp_path: Path,
) -> None:
    """Design row 14: the seed is the newest PARSEABLE receipt, as eager's is."""
    directory = await _journal(tmp_path, turns=20, usage=True)
    with (directory / "transcript.jsonl").open("a") as out:
        out.write(
            TranscriptEntry(
                id="bad",
                ts=1.0,
                type="message",
                payload={"role": "assistant", "content": [], "usage": {"input_tokens": "abc"}},
            ).to_json()
            + "\n"
        )
    facts = tail_facts(directory)
    assert facts is not None and facts["seed"] is not None
    assert facts == eager_facts(directory)


@pytest.mark.asyncio
async def test_met_checkpoints_do_not_stop_the_walk_at_an_unparseable_receipt(
    tmp_path: Path,
) -> None:
    """With both types met, an unparseable receipt alone must not end the walk.

    Stopping there would hand the caller entries with no usable seed while eager
    seeds from the older parseable receipt. Kills the "stop on any usage dict"
    mutation.
    """
    directory = await _journal(tmp_path, turns=20, usage=True)
    transcript = Transcript(directory)
    await transcript.append_custom(SESSION_SPEND_CUSTOM_TYPE, {"usd": 0.5})
    await transcript.append_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"checkpoint_id": "new"})
    transcript.flush()
    with (directory / "transcript.jsonl").open("a") as out:
        out.write(
            TranscriptEntry(
                id="bad",
                ts=1.0,
                type="message",
                payload={"role": "assistant", "content": [], "usage": {"input_tokens": "abc"}},
            ).to_json()
            + "\n"
        )
    facts = tail_facts(directory)
    assert facts is not None and facts["seed"] is not None
    assert facts == eager_facts(directory)


@pytest.mark.asyncio
async def test_a_proven_usage_free_journal_files_a_record_with_no_seed(tmp_path: Path) -> None:
    """``seed is None`` is eager's answer once usage is PROVEN absent: file it."""
    directory = await _journal(tmp_path, turns=400)
    key = replay_cache.replay_key(
        directory,
        through_id=None,
        strict_cut=False,
        checkpoint_types=WANTED[:1],
        opportunistic_types=WANTED[1:],
    )
    assert key is not None
    record = replay_cache.build_cold_tail_metadata(key, tmp_path / "record.json")
    assert record is not None
    assert record.seed_usage is None and record.checkpoint is None
    assert (tmp_path / "record.json").exists()
