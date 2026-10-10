"""The cold-replay cache: identity, bound, sharing, and the append that lies about mtime.

The audit's C3 lane names one regression risk above the rest, and it is the
reason ``size`` is in the key: ``Transcript._write_entries`` RESTORES the file's
mtime after a bookkeeping batch (``preserve_mtime``, the activity clock), so a
spend record or a checkpoint can be appended inside the same mtime instant —
even the same nanosecond, after a restore — while the bytes on disk moved. A key
of ``(inode, mtime_ns)`` alone would then answer a request from the older parse,
and the phone or desktop would paint a conversation missing its newest row.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from local_operator.session import replay_cache as cache_module
from local_operator.session.replay_cache import (
    ColdReplay,
    cached_replay,
    load_cold_replay,
    publish_replay,
    replay_cache,
    replay_key,
)
from local_operator.session.transcript import Transcript, read_replay_suffix


def write_journal(directory, rows: int = 3) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        '{"id":"m%d","ts":1.0,"type":"message","payload":{"kind":"message","role":"user",'
        '"content":[{"text":"turn %d"}]}}' % (index, index)
        for index in range(rows)
    ]
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def clean_cache():
    replay_cache().clear()
    yield
    replay_cache().clear()


def test_the_key_carries_the_file_version_and_the_request(tmp_path):
    """Size is the half that a restored mtime would otherwise hide."""
    directory = tmp_path / "sess"
    write_journal(directory)

    first = replay_key(directory, through_id=None, strict_cut=False, checkpoint_types=())
    assert first is not None
    (directory / "transcript.jsonl").write_text('{"id":"x"', encoding="utf-8")  # mid-append
    os.utime(directory / "transcript.jsonl", ns=(first.mtime_ns, first.mtime_ns))
    second = replay_key(directory, through_id=None, strict_cut=False, checkpoint_types=())

    assert second is not None
    assert second.mtime_ns == first.mtime_ns, "precondition: the mtime was held still"
    assert second.size != first.size
    assert second != first, "a same-mtime append must not reuse the older parse"


def test_the_key_separates_the_request_not_only_the_file(tmp_path):
    directory = tmp_path / "sess"
    write_journal(directory)
    base = replay_key(directory, through_id=None, strict_cut=False, checkpoint_types=())

    assert replay_key(directory, through_id="m1", strict_cut=False, checkpoint_types=()) != base
    assert replay_key(directory, through_id=None, strict_cut=True, checkpoint_types=()) != base
    assert replay_key(
        directory, through_id=None, strict_cut=False, checkpoint_types=("a",)
    ) != replay_key(directory, through_id=None, strict_cut=False, checkpoint_types=("a", "b"))
    # A bare string and a one-element tuple are one request, not two.
    assert replay_key(
        directory, through_id=None, strict_cut=False, checkpoint_types="a"
    ) == replay_key(directory, through_id=None, strict_cut=False, checkpoint_types=("a",))


def test_an_absent_journal_has_no_key(tmp_path):
    assert replay_key(tmp_path / "nope", through_id=None, strict_cut=False) is None


def test_a_published_replay_is_served_from_the_cache(tmp_path):
    directory = tmp_path / "sess"
    write_journal(directory)
    key = replay_key(directory, through_id=None, strict_cut=False, checkpoint_types=())
    assert key is not None

    published = publish_replay(
        key,
        messages=("a", "b"),
        checkpoint=None,
        checkpoints={},
        order={},
        seed_usage=None,
        bytes_read=42,
    )
    assert cached_replay(key) is published
    assert published.messages == ("a", "b") and published.bytes_read == 42


def test_an_oversized_replay_is_answered_but_never_stored(tmp_path, monkeypatch):
    """The bound must not change the ANSWER, only whether it is held.

    An entry bigger than the whole budget is refused rather than admitted and
    immediately evicted: a cache that thrashes on one session pays the parse
    again on every request, which is worse than never holding it.
    """
    monkeypatch.setattr(cache_module, "REPLAY_CACHE_BYTES", 4096)
    directory = tmp_path / "sess"
    write_journal(directory)
    key = replay_key(directory, through_id=None, strict_cut=False, checkpoint_types=())
    assert key is not None
    huge = {"blob": "x" * 8192}

    value = publish_replay(
        key, messages=(huge,), checkpoint=None, checkpoints={}, order={}, seed_usage=None
    )

    assert value.messages == (huge,), "the caller still gets the conversation"
    assert cached_replay(key) is None, "an over-budget replay must not be pinned"
    assert replay_cache().entry_count == 0


def test_the_cache_is_bounded_by_entries_and_by_bytes(tmp_path):
    """Both bounds, held at once: entries alone is unbounded bytes, bytes alone
    would evict to a single entry on a long conversation."""
    directory = tmp_path / "sess"
    write_journal(directory)
    replay_cache().max_entries = 3
    replay_cache().max_bytes = 1 << 20
    body = "y" * 100_000

    for index in range(6):
        key = replay_key(
            directory, through_id=f"cursor-{index}", strict_cut=False, checkpoint_types=()
        )
        assert key is not None
        publish_replay(
            key,
            messages=(body,),
            checkpoint=None,
            checkpoints={},
            order={},
            seed_usage=None,
        )

    assert replay_cache().entry_count <= 3
    assert replay_cache().accounted_bytes <= (1 << 20)


@pytest.mark.asyncio
async def test_a_bookkeeping_append_invalidates_the_cached_replay(tmp_path):
    """THE case the size-in-key rule exists for, driven through the real writer.

    A spend record is appended with ``preserve_mtime=True``, so the file's mtime
    does not move while its bytes do. The next cold read must see the new row: a
    cache keyed on the timestamp alone would serve the pre-append replay and the
    conversation would be missing its newest receipt.
    """
    from local_operator.harness.types import Message
    from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE

    session_dir = tmp_path / "sess"
    transcript = Transcript(session_dir)
    await transcript.append_message(Message.user("do the thing"))

    key = replay_key(session_dir, through_id=None, strict_cut=False, checkpoint_types=())
    assert key is not None
    suffix = read_replay_suffix(session_dir, opportunistic_types=(SESSION_SPEND_CUSTOM_TYPE,))
    publish_replay(
        key,
        messages=tuple(suffix.entries),
        checkpoint=None,
        checkpoints={},
        order={},
        seed_usage=None,
    )
    assert cached_replay(key) is not None

    stamp = (session_dir / "transcript.jsonl").stat()
    await transcript.append_custom(SESSION_SPEND_CUSTOM_TYPE, {"usd": 1.0}, preserve_mtime=True)
    after = (session_dir / "transcript.jsonl").stat()

    # ``os.utime`` restores the mtime through a float, so a few hundred
    # nanoseconds of precision are lost; the exemption is about the CLOCK not
    # moving (minutes), not about bit-exact recovery.
    assert after.st_mtime == pytest.approx(
        stamp.st_mtime, abs=1e-5
    ), "precondition: the clock must not move"
    assert after.st_size > stamp.st_size, "precondition: the bytes must move"
    moved = replay_key(session_dir, through_id=None, strict_cut=False, checkpoint_types=())
    assert moved is not None and moved != key
    assert cached_replay(moved) is None, "the append was served from the stale parse"


def test_a_second_caller_for_one_key_shares_the_read(tmp_path):
    """An open issues its snapshot and its stream back to back; both must not parse."""
    directory = tmp_path / "sess"
    write_journal(directory)
    key = replay_key(directory, through_id=None, strict_cut=False, checkpoint_types=())
    assert key is not None
    builds = 0

    async def build() -> ColdReplay:
        nonlocal builds
        builds += 1
        await asyncio.sleep(0.05)
        publish_replay(
            key,
            messages=("m",),
            checkpoint=None,
            checkpoints={},
            order={},
            seed_usage=None,
        )
        return ColdReplay(
            messages=("m",), checkpoint=None, checkpoints={}, order={}, seed_usage=None
        )

    async def scenario() -> None:
        first, second = await asyncio.gather(
            load_cold_replay(key, build), load_cold_replay(key, build)
        )
        assert first.messages == second.messages == ("m",)

    asyncio.run(scenario())
    assert builds == 1, "the same key was parsed twice"
    assert cached_replay(key) is not None


def test_an_uncached_key_still_builds():
    """``None`` is the absent-journal case: the read happens, nothing is stored."""
    calls = 0

    async def build() -> ColdReplay:
        nonlocal calls
        calls += 1
        return ColdReplay(messages=(), checkpoint=None, checkpoints={}, order={}, seed_usage=None)

    asyncio.run(load_cold_replay(None, build))
    asyncio.run(load_cold_replay(None, build))
    assert calls == 2
    assert replay_cache().entry_count == 0
