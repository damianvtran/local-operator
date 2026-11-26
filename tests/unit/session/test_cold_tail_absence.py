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

import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent, Usage
from local_operator.session import cold_metadata, replay_cache
from local_operator.session import transcript as transcript_module
from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE, SessionSpend
from local_operator.session.transcript import (
    Transcript,
    TranscriptEntry,
    _custom_type_spellings,
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
async def test_desktop_recovers_checkpoint_just_below_row_bound(tmp_path: Path) -> None:
    directory = await _journal(tmp_path, turns=150, checkpoint_first=True)
    writer = Transcript(directory)
    await writer.append_message(
        Message(id="receipt", role="assistant", usage=Usage(input_tokens=1, output_tokens=2))
    )
    writer.flush()
    writer.close()
    key = replay_cache.replay_key(
        directory,
        through_id=None,
        strict_cut=False,
        checkpoint_types=WANTED[:1],
        opportunistic_types=WANTED[1:],
    )
    assert key is not None
    expected = eager_facts(directory)
    assert expected["checkpoint"] == {"checkpoint_id": "buried"}
    target = tmp_path / "record.json"
    stats: dict[str, Any] = {}
    actual = replay_cache.build_cold_tail_metadata(key, target, stats)
    assert actual is not None, "desktop must not replay eager for a provably recoverable checkpoint"
    assert actual.checkpoint == expected["checkpoint"]
    assert actual.checkpoints.get(SESSION_SPEND_CUSTOM_TYPE) == expected["spend"]
    assert actual.order == expected["order"]
    assert actual.seed_usage is not None
    assert actual.seed_usage.model_dump(mode="json") == expected["seed"]
    assert cold_metadata.read_cold_metadata(target, replay_cache.witness_for(key)) == actual
    assert stats["proof_mode"] == "recovered"


def _encoded(id_: str, kind: str, **payload: Any) -> bytes:
    return TranscriptEntry(id_, 1.0, kind, payload).to_json().encode() + b"\n"


def _custom_row(name: str, details: Any, *, id_: str = "custom") -> bytes:
    return _encoded(id_, "custom", custom_type=name, details=details)


_CHECKPOINT = _custom_row(WANTED[0], {"checkpoint_id": "newest"})
_SPEND = _custom_row(WANTED[1], {"version": 1, "micro": 310, "calls": 1, "priced_calls": 1})
_RECEIPT = _encoded("receipt", "message", role="assistant", usage={"context_tokens": 125})
_LOOKALIKE = _custom_row("unrelated", {"custom_type": WANTED[0]})
_DENSE_LINE = _encoded("filler", "message", role="user")
_FILLER = b"".join(_encoded(f"row-{i}", "message", role="user") for i in range(256))
_SHRINK = _encoded("compact", "compaction", first_kept_entry_id="row-0")


def _raw_journal(tmp_path: Path, data: bytes) -> tuple[Path, Path]:
    directory = tmp_path / "session"
    directory.mkdir()
    journal = directory / "transcript.jsonl"
    journal.write_bytes(data)
    return directory, journal


def _desktop_key(directory: Path) -> replay_cache.ReplayKey:
    key = replay_cache.replay_key(
        directory,
        through_id=None,
        strict_cut=False,
        checkpoint_types=WANTED[:1],
        opportunistic_types=WANTED[1:],
    )
    assert key is not None
    return key


@pytest.mark.parametrize("met_spend", [False, True], ids=["absent-spend", "met-spend"])
@pytest.mark.parametrize("in_window", [False, True], ids=["beyond-window", "within-window"])
def test_desktop_checkpoint_recovery_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, met_spend: bool, in_window: bool
) -> None:
    # Exercise default json.dumps as well as the compact encoding (the older
    # duplicate). Include blank lines, a shrink, and an old receipt that must
    # not seed the recovered first frame. A no-hit window is full and dense.
    newest = json.dumps(json.loads(_CHECKPOINT)).encode() + b"\n"
    older_receipt = _encoded(
        "before-shrink", "message", role="assistant", usage={"context_tokens": 999}
    )
    dense = _DENSE_LINE * (((4 << 20) // len(_DENSE_LINE)) + 1) if not in_window else b""
    directory, _ = _raw_journal(
        tmp_path,
        _custom_row(WANTED[0], {"checkpoint_id": "older"}, id_="older")
        + newest
        + dense
        + b"\n \n"
        + older_receipt
        + _SHRINK
        + _FILLER
        + (_SPEND if met_spend else b"")
        + _RECEIPT,
    )
    key = _desktop_key(directory)
    target = tmp_path / "metadata.json"
    expected = eager_facts(directory)
    assert expected["checkpoint"] == {"checkpoint_id": "newest"}
    assert expected["seed"]["context_tokens"] == 125
    calls: list[tuple[bytes, ...]] = []
    original_absence = transcript_module._spellings_absent
    original_parse = TranscriptEntry.from_json
    parses = 0

    def count_absence(handle: Any, start: int, end: int, needles: tuple[bytes, ...]) -> bool:
        calls.append(needles)
        return original_absence(handle, start, end, needles)

    def count_parse(line: str) -> TranscriptEntry | None:
        nonlocal parses
        parses += 1
        return original_parse(line)

    monkeypatch.setattr(transcript_module, "_spellings_absent", count_absence)
    monkeypatch.setattr(TranscriptEntry, "from_json", count_parse)
    stats: dict[str, Any] = {}
    actual = replay_cache.build_cold_tail_metadata(key, target, stats)
    front = _custom_type_spellings(WANTED[0])
    spend = _custom_type_spellings(WANTED[1])
    assert front is not None and spend is not None
    if not in_window:
        assert actual is None
        # The cold tail parses 256 rows as before. The dense continuation adds
        # bytes, not even one parse or a second prefix scan.
        assert parses == 256
        assert stats["lookup_rows"] == 0 and stats["lookup_bytes"] == 4 << 20
        assert stats["lookup_outcome"] == "no_spelling"
        assert stats["proof_mode"] == "declined"
        assert calls == [front if met_spend else front + spend]
        assert not target.exists() and not replay_cache.absence_proof_path(target).exists()
    else:
        assert actual is not None
        assert calls == ([] if met_spend else [spend]), "no combined prefix scan on a recovery"
        assert stats["proof_mode"] == "recovered" and stats["lookup_rows"] > 0
        assert actual.checkpoint == expected["checkpoint"]
        assert actual.checkpoints.get(WANTED[1]) == expected["spend"]
        if met_spend:
            assert SessionSpend.from_details(actual.checkpoints[WANTED[1]]) is not None
        assert actual.order == expected["order"]
        assert actual.seed_usage is not None
        assert actual.seed_usage.model_dump(mode="json") == expected["seed"]
        assert cold_metadata.read_cold_metadata(target, replay_cache.witness_for(key)) == actual
        sidecar = replay_cache.absence_proof_path(target)
        if met_spend:
            assert not sidecar.exists()
        else:
            assert set(json.loads(sidecar.read_text())["needles"]) == {n.decode() for n in spend}
    assert replay_cache.cached_replay(key) is None


@pytest.mark.parametrize(
    ("prefix", "tip", "reason", "lookup_outcome"),
    [
        (_CHECKPOINT + b"{not json}\n", _SPEND + _RECEIPT, "below_tail_presence", "malformed"),
        (_CHECKPOINT + b"\xff\n", _SPEND + _RECEIPT, "below_tail_presence", "malformed"),
        (_custom_row(WANTED[0], None), _SPEND + _RECEIPT, "below_tail_presence", "malformed"),
        (
            _CHECKPOINT,
            _custom_row(WANTED[1], {"version": 999}) + _RECEIPT,
            "below_tail_presence",
            "invalid_spend",
        ),
        (_SPEND + _CHECKPOINT, _RECEIPT, "spend_unproven", "matched"),
        (_CHECKPOINT, _SPEND, "below_tail_presence", None),
        (_CHECKPOINT, _RECEIPT + _SHRINK, "shrink_before_seed", None),
        (_CHECKPOINT, _RECEIPT[:-1], "torn_tail", None),
    ],
)
def test_desktop_recovery_declines_uncertain_facts_without_writes(
    tmp_path: Path, prefix: bytes, tip: bytes, reason: str, lookup_outcome: str | None
) -> None:
    directory, journal = _raw_journal(tmp_path, prefix + _FILLER + tip)
    key = _desktop_key(directory)
    old = journal.read_bytes()
    target = tmp_path / "metadata.json"
    stats: dict[str, Any] = {}
    assert replay_cache.build_cold_tail_metadata(key, target, stats) is None
    assert stats["proof_mode"] == "declined" and stats["decline_reason"] == reason
    assert stats.get("lookup_outcome") == lookup_outcome
    assert not target.exists() and not replay_cache.absence_proof_path(target).exists()
    assert replay_cache.cached_replay(key) is None and journal.read_bytes() == old


def test_false_presence_stops_at_the_oldest_possible_row(tmp_path: Path) -> None:
    directory, _ = _raw_journal(
        tmp_path, _CHECKPOINT + _DENSE_LINE * 75_000 + _LOOKALIKE + _FILLER + _SPEND + _RECEIPT
    )
    stats: dict[str, Any] = {}
    key = _desktop_key(directory)
    result = replay_cache.build_cold_tail_metadata(key, tmp_path / "meta", stats)
    assert result is None
    assert stats["lookup_outcome"] == "not_checkpoint"
    assert stats["lookup_rows"] == 3  # two rows below the cold tail, plus the lookalike


@pytest.mark.parametrize("complete", [False, True], ids=["partial-floor", "crosses-chunk"])
def test_preflight_needs_the_whole_checkpoint_row(tmp_path: Path, complete: bool) -> None:
    needle = _custom_type_spellings(WANTED[0])
    assert needle is not None
    if complete:
        # The needle straddles the usual backward tail chunk boundary: a
        # byte scan split with no overlap would miss this complete row.
        filler = _encoded("long", "message", role="user", text="")
        target_distance = (64 << 10) + len(needle[0]) // 2
        distance = len(_CHECKPOINT) - _CHECKPOINT.index(needle[0]) + len(filler)
        older = _CHECKPOINT + _encoded(
            "long", "message", role="user", text="x" * (target_distance - distance)
        )
    else:
        # The needle itself fits; its row start does not. The legacy absence
        # proof still sees it but the recovery cannot count this partial row.
        older = _encoded(
            "custom", "custom", details={"text": "x" * (256 << 10)}, custom_type=WANTED[0]
        )
    directory, _ = _raw_journal(tmp_path, older + _RECEIPT)
    key = _desktop_key(directory)
    stats: dict[str, Any] = {}
    suffix = read_cold_tail(
        directory,
        checkpoint_types=WANTED,
        desktop_source=(key.inode, key.size, key.mtime_ns, key.ctime_ns),
        max_bytes=128 << 10,
        max_rows=1,
        stats=stats,
    )
    if complete:
        assert suffix is not None and suffix.recovered_checkpoint
        assert suffix.checkpoint_order[WANTED[0]] == 3
        assert stats["lookup_rows"] == 2  # the long row and the checkpoint
    else:
        assert suffix is None
        assert needle[0] in older[-(128 << 10) :]
        assert stats["lookup_outcome"] == "no_spelling" and stats["lookup_rows"] == 0


def test_recovery_cannot_extend_the_spend_absence_cap(tmp_path: Path) -> None:
    directory, _ = _raw_journal(tmp_path, _CHECKPOINT + _FILLER + _RECEIPT)
    key = _desktop_key(directory)
    stats: dict[str, Any] = {}
    result = read_cold_tail(
        directory,
        checkpoint_types=WANTED,
        desktop_source=(key.inode, key.size, key.mtime_ns, key.ctime_ns),
        max_scan_bytes=1,
        stats=stats,
    )
    assert result is None
    assert stats == {"decline_reason": "absence_cap"}


def test_untraced_recovery_does_not_read_a_lookup_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, _ = _raw_journal(tmp_path, _CHECKPOINT + _FILLER + _SPEND + _RECEIPT)

    def unexpected() -> float:
        pytest.fail("disabled lookup tracing sampled a clock")

    monkeypatch.setattr(transcript_module, "time", SimpleNamespace(monotonic=unexpected))
    key = _desktop_key(directory)
    assert replay_cache.build_cold_tail_metadata(key, tmp_path / "meta") is not None


@pytest.mark.parametrize("move", ["append", "replace"])
def test_recovered_checkpoint_must_pass_both_journal_witnesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, move: str
) -> None:
    # With spend absent, an accepted result would file both metadata and the
    # spend-only sidecar. Change the descriptor during lookup, or the pathname
    # after descriptor verification while building the selection reading.
    data = _CHECKPOINT + _FILLER + _RECEIPT
    directory, journal = _raw_journal(tmp_path, data)
    key = _desktop_key(directory)
    if move == "append":
        original_lookup = transcript_module._recover_cold_checkpoint

        def grow(*args: Any, **kwargs: Any) -> Any:
            hit, found = original_lookup(*args, **kwargs)
            assert hit is not None and found is not None
            with journal.open("ab") as handle:
                handle.write(_DENSE_LINE)
            return hit, found

        monkeypatch.setattr(transcript_module, "_recover_cold_checkpoint", grow)
    else:
        original_selection = replay_cache.read_selection_reading

        def replace(*args: Any, **kwargs: Any) -> Any:
            selection = original_selection(*args, **kwargs)
            new = directory / "replacement.jsonl"
            new.write_bytes(data)
            new.replace(journal)
            return selection

        monkeypatch.setattr(replay_cache, "read_selection_reading", replace)
    target = tmp_path / "metadata.json"
    stats: dict[str, Any] = {}
    assert replay_cache.build_cold_tail_metadata(key, target, stats) is None
    assert stats["lookup_outcome"] == "matched"
    assert stats["decline_reason"] == "source_changed"
    fresh = _desktop_key(directory)
    assert fresh != key and (fresh.inode == key.inode) == (move == "append")
    assert not target.exists() and not replay_cache.absence_proof_path(target).exists()


def test_archive_declines_read_only(tmp_path: Path) -> None:
    directory = tmp_path / "session"
    directory.mkdir()
    archive = directory / "transcript.jsonl.gz"
    archive.write_bytes(gzip.compress(_CHECKPOINT + _FILLER + _RECEIPT))
    key = _desktop_key(directory)
    stats: dict[str, Any] = {}
    assert replay_cache.build_cold_tail_metadata(key, directory / "metadata.json", stats) is None
    assert stats["decline_reason"] == "archive" and "lookup_bytes" not in stats
    assert _desktop_key(directory) == key and set(directory.iterdir()) == {archive}


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
@pytest.mark.parametrize("buried", ["checkpoint", "usage", "spend"])
async def test_a_fact_buried_below_the_tail_declines(tmp_path: Path, buried: str) -> None:
    """The one false absence the design must make impossible.

    ``spend`` pins QA round 1 Q1: the spend type reaches the scan because
    ``build_cold_tail_metadata`` passes the opportunistic types too.
    """
    if buried == "spend":
        directory = tmp_path / "session"
        transcript = Transcript(directory)
        await transcript.append_custom(SESSION_SPEND_CUSTOM_TYPE, {"usd": 4.2, "calls": 7})
        for turn in range(300):
            await transcript.append_message(
                Message(id=f"m{turn}", role="user", content=[TextContent(text="x")])
            )
        transcript.flush()
        assert eager_facts(directory)["spend"] == {"usd": 4.2, "calls": 7}
        assert read_cold_tail(directory, checkpoint_types=WANTED) is None
        return
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
    spellings = _custom_type_spellings(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert spellings is not None
    assert FRONTEND_CHECKPOINT_CUSTOM_TYPE.encode() in raw
    assert not any(spelling in raw for spelling in spellings)
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


@pytest.mark.parametrize("spaced", [False, True])
@pytest.mark.parametrize("split", range(1, 30))
def test_a_spelling_split_across_a_window_boundary_is_found(split: int, spaced: bool) -> None:
    """Every split point of the needle across the window edge is still a hit."""
    spellings = _custom_type_spellings(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert spellings is not None
    needle = spellings[int(spaced)]
    assert split < len(needle)
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


def _spaced_row(row: TranscriptEntry) -> bytes:
    """The row as ``json.dumps`` writes it with its DEFAULT separators.

    No repo writer does this today (every row goes through ``to_json``); the
    point is that soundness must not rest on that convention (review round 1,
    MAJOR-1). A spaced row the scan cannot see would be "proven absent" and its
    wrong facts filed as a durable record.
    """
    return (
        json.dumps({"id": row.id, "ts": row.ts, "type": row.type, "payload": row.payload}) + "\n"
    ).encode("utf-8")


async def _journal_with_spaced_row_below_the_tail(tmp_path: Path, buried: str) -> Path:
    directory = tmp_path / "session"
    directory.mkdir(parents=True)
    if buried == "usage":
        row = TranscriptEntry(
            id="buried",
            ts=1.0,
            type="message",
            payload={
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 111, "output_tokens": 222},
            },
        )
    else:
        row = TranscriptEntry(
            id="buried",
            ts=1.0,
            type="custom",
            payload={"custom_type": buried, "details": {"checkpoint_id": "buried"}},
        )
    spaced = _spaced_row(row)
    assert b'": ' in spaced and row.to_json().encode() not in spaced
    (directory / "transcript.jsonl").write_bytes(spaced)
    transcript = Transcript(directory)
    for turn in range(300):
        await transcript.append_message(
            Message(id=f"m{turn}", role="user", content=[TextContent(text="x")])
        )
    transcript.flush()
    return directory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "buried", ["usage", FRONTEND_CHECKPOINT_CUSTOM_TYPE, SESSION_SPEND_CUSTOM_TYPE]
)
async def test_a_spaced_separator_row_below_the_tail_is_never_proven_absent(
    tmp_path: Path, buried: str
) -> None:
    directory = await _journal_with_spaced_row_below_the_tail(tmp_path, buried)
    eager = eager_facts(directory)
    assert eager != tail_facts_if_absent(directory), "eager sees the buried row"
    facts = tail_facts(directory)
    assert facts is None or facts == eager
    key = replay_cache.replay_key(
        directory,
        through_id=None,
        strict_cut=False,
        checkpoint_types=WANTED[:1],
        opportunistic_types=WANTED[1:],
    )
    assert key is not None
    record_path = tmp_path / "record.json"
    record = replay_cache.build_cold_tail_metadata(key, record_path)
    assert record is None and not record_path.exists(), "no record of wrong facts is filed"


def tail_facts_if_absent(directory: Path) -> dict[str, Any]:
    """What the tail answered on b3fc4d605d8 for every case above: the buried
    row "proven absent". Eager must differ from it, or the test proves nothing."""
    return {"checkpoint": None, "spend": None, "order": {}, "seed": None}
