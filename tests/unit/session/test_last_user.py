"""``session.last_user``: the Running order's clock, and what it may cost.

WHY THESE TESTS EXIST, in the module's own terms. The tracker answers one
question — when did the PERSON last send this session a message — and every
property asserted here is one the Running section's stability depends on:

* it FINDS a typed row buried behind megabytes of assistant/tool output (the
  distance measured on the operator's store: median 733 KB, p90 2.77 MB, max
  12.2 MB, all beyond the 64 KB preview window);
* it SKIPS everything that merely looks like a user row — harness chrome, the
  notice heads, stamped renders, and every ``kind == "custom"`` injection — so
  a wake or a peer message cannot order the section as the operator's own;
* it does NOT move when a response streams (the whole point: the value changes
  on a person's send and on nothing else), and moving must be picked up;
* the INCREMENTAL read is asserted on BYTES, not on a clock — a poll must read
  the appended bytes and no more, because the property is what makes this
  affordable on the 2 s listing loop, and a timing assertion on this fleet
  measures contention rather than the algorithm;
* the failure modes degrade to ``None`` (unknown) — a torn tail is never
  half-consumed, the cap gives up rather than reading forever and remembers
  that it did, and a missing or unreadable file is not an exception on a
  listing path.

The rows are hand-written in the WRITER's own shape (``TranscriptEntry.to_json``:
compact separators, id/ts/type/payload) rather than through ``Transcript``, so
the tests stay pure-file and each row's shape is visible where the reader's
decision about it is asserted.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import last_user

# ---------------------------------------------------------------------------
# Row builders: the writer's compact shape, one function per family the reader
# must discriminate between.
# ---------------------------------------------------------------------------


def _line(entry: dict[str, Any], *, spaced: bool = False) -> bytes:
    separators = None if spaced else (",", ":")
    return json.dumps(entry, separators=separators).encode("utf-8") + b"\n"


def _user(text: str, ts: float, *, entry_id: str = "u" * 24, **payload_extra: Any) -> bytes:
    payload: dict[str, Any] = {"kind": "message", "role": "user", "content": [{"text": text}]}
    payload.update(payload_extra)
    return _line({"id": entry_id, "ts": ts, "type": "message", "payload": payload})


def _assistant(text: str, ts: float) -> bytes:
    return _line(
        {
            "id": "a" * 24,
            "ts": ts,
            "type": "message",
            "payload": {"kind": "message", "role": "assistant", "content": [{"text": text}]},
        }
    )


def _tool(text: str, ts: float) -> bytes:
    return _line(
        {
            "id": "t" * 24,
            "ts": ts,
            "type": "message",
            "payload": {"kind": "message", "role": "tool", "content": [{"text": text}]},
        }
    )


def _custom(custom_type: str, ts: float) -> bytes:
    return _line(
        {
            "id": "c" * 24,
            "ts": ts,
            "type": "message",
            "payload": {
                "kind": "custom",
                "custom_type": custom_type,
                "details": {"text": f"a {custom_type}"},
            },
        }
    )


def _stamped(text: str, ts: float) -> bytes:
    """A rendered harness injection: a user-role row carrying the provenance stamp."""
    return _line(
        {
            "id": "s" * 24,
            "ts": ts,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "user",
                "content": [{"text": text}],
                "provider_payload": {"harness_injected": True},
            },
        }
    )


def _notice(text: str, ts: float) -> bytes:
    """A pre-stamp harness notice: a user-role row recognised by its head."""
    return _user(text, ts)


def _session(tmp_path: Path, name: str = "s") -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write(directory: Path, *chunks: bytes) -> Path:
    path = directory / "transcript.jsonl"
    with path.open("wb") as handle:
        for chunk in chunks:
            handle.write(chunk)
    return path


def _append(path: Path, *chunks: bytes) -> None:
    with path.open("ab") as handle:
        for chunk in chunks:
            handle.write(chunk)


@pytest.fixture(autouse=True)
def _fresh_memo():
    """The memo is process-global; every test starts and ends with it empty."""
    with last_user._MEMO_GUARD:
        last_user._MEMO.clear()
    yield
    with last_user._MEMO_GUARD:
        last_user._MEMO.clear()


def _counting_reads(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count the bytes ``_read_range`` is asked to read, via the one read site."""
    reads = {"bytes": 0}
    real = last_user._read_range

    def counting(handle, start, end):  # type: ignore[no-untyped-def]
        reads["bytes"] += end - start
        return real(handle, start, end)

    monkeypatch.setattr(last_user, "_read_range", counting)
    return reads


# ---------------------------------------------------------------------------
# Finding the row.
# ---------------------------------------------------------------------------


class TestItFindsTheNewestTypedRow:
    @pytest.mark.parametrize("distance", [70_000, 1_200_000])
    def test_a_typed_row_far_from_the_end_is_found(self, tmp_path: Path, distance: int) -> None:
        """Beyond both the 64 KB preview window and the 256 KB picker window."""
        directory = _session(tmp_path)
        chunks = [_user("the question", 1000.0)]
        filler = 0
        while filler < distance:
            row = _assistant("a streamed answer " * 40, 1100.0 + filler)
            filler += len(row)
            chunks.append(row)
        chunks.append(_tool("a tool result", 9000.0))
        path = _write(directory, *chunks)
        typed_end = len(chunks[0])
        assert path.stat().st_size - typed_end > distance, "the row is further from EOF than asked"

        assert last_user.last_user_at(directory) == 1000.0

    def test_the_value_is_the_newest_typed_rows_ts(self, tmp_path: Path) -> None:
        directory = _session(tmp_path)
        _write(directory, _user("first", 1000.0), _user("second", 2000.0), _assistant("hi", 3000.0))
        assert last_user.last_user_at(directory) == 2000.0

    def test_a_spaced_hand_written_row_is_still_read(self, tmp_path: Path) -> None:
        """``json.dumps``' default separators, for hand-edited or foreign rows."""
        directory = _session(tmp_path)
        _write(directory, _user("typed", 1500.0, spaced=True))
        assert last_user.last_user_at(directory) == 1500.0


class TestWhatDoesNotCount:
    def test_stamped_notice_and_custom_rows_are_skipped(self, tmp_path: Path) -> None:
        """Every family newer than the typed row, and the typed row is returned."""
        from local_operator.harness.rows import harness_chrome_prompts

        directory = _session(tmp_path)
        _write(
            directory,
            _user("the real question", 1000.0),
            # A pre-stamp notice (recognised by its head).
            _notice("[session warning] the MCP pool was rebuilt", 1100.0),
            # The chrome prompt families, and a stamped render.
            _user(harness_chrome_prompts()[0], 1200.0),
            _stamped("Continue working toward this goal", 1300.0),
            # Every kind == "custom" injection the sidebar must not attribute.
            _custom("wake_prompt", 1400.0),
            _custom("peer_message", 1410.0),
            _custom("hub_message", 1420.0),
            _custom("job_result", 1430.0),
            _custom("session_state", 1440.0),
            _custom("session_incident", 1450.0),
            _custom("selected_model", 1460.0),
            # Assistant and tool rows never count.
            _assistant("an answer", 1500.0),
            _tool("a result", 1510.0),
        )
        assert last_user.last_user_at(directory) == 1000.0

    def test_a_typed_row_with_no_usable_ts_is_skipped_and_an_older_one_returned(
        self, tmp_path: Path
    ) -> None:
        directory = _session(tmp_path)
        missing = _line(
            {
                "id": "m" * 24,
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": [{"text": "no ts"}]},
            }
        )
        zero = _user("zero ts", 0.0)
        negative = _user("negative ts", -5.0)
        _write(directory, _user("usable", 1000.0), zero, missing, negative)
        assert last_user.last_user_at(directory) == 1000.0

    def test_a_steer_row_counts(self, tmp_path: Path) -> None:
        """A mid-turn steer IS the person's message; its ts is the drain time.

        The row is written by ``_drain_steering`` when the running turn next
        yields it — an ordinary unstamped user row, possibly carrying a
        producer identity — so it counts exactly like a typed prompt, and the
        value moves at the DRAIN, not at the keystroke. That lag is inherent
        to reading the transcript and is stated in the module docstring.
        """
        directory = _session(tmp_path)
        _write(
            directory,
            _user("the original prompt", 1000.0),
            _assistant("working", 1100.0),
            _user(
                "actually, do it the other way",
                1200.0,
                entry_id="d" * 24,
                producer_command_id="cid-1",
            ),
        )
        assert last_user.last_user_at(directory) == 1200.0


# ---------------------------------------------------------------------------
# Stability: what the Running order exists FOR.
# ---------------------------------------------------------------------------


class TestStability:
    def test_rows_appended_after_a_typed_row_do_not_move_the_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE stability property: the agent answering must not re-sort Running."""
        directory = _session(tmp_path)
        path = _write(directory, _user("do the thing", 1000.0))
        assert last_user.last_user_at(directory) == 1000.0

        reads = _counting_reads(monkeypatch)
        _append(
            path,
            _assistant("working on it", 1100.0),
            _tool("a tool result", 1110.0),
            _assistant("still working", 1120.0),
            _custom("job_result", 1130.0),
        )
        assert last_user.last_user_at(directory) == 1000.0
        assert reads["bytes"] > 0, "the append was actually seen (a nonzero read)"

    def test_an_appended_typed_row_moves_it(self, tmp_path: Path) -> None:
        directory = _session(tmp_path)
        path = _write(directory, _user("first", 1000.0), _assistant("hi", 1100.0))
        assert last_user.last_user_at(directory) == 1000.0
        _append(path, _user("second", 2000.0))
        assert last_user.last_user_at(directory) == 2000.0


# ---------------------------------------------------------------------------
# The incremental read: bytes, not clocks.
# ---------------------------------------------------------------------------


class TestTheIncrementalRead:
    def test_an_incremental_append_reads_only_the_appended_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _session(tmp_path)
        path = _write(directory, _user("one", 1000.0))
        assert last_user.last_user_at(directory) == 1000.0

        reads = _counting_reads(monkeypatch)
        appended = _assistant("two", 1100.0) + _tool("three", 1200.0) + _user("four", 1300.0)
        _append(path, appended)
        assert last_user.last_user_at(directory) == 1300.0
        assert reads["bytes"] == len(appended), "exactly the appended bytes were read"

    def test_an_unchanged_file_is_answered_without_reading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _session(tmp_path)
        _write(directory, _user("one", 1000.0), _assistant("two", 1100.0))
        assert last_user.last_user_at(directory) == 1000.0
        reads = _counting_reads(monkeypatch)
        assert last_user.last_user_at(directory) == 1000.0
        assert reads["bytes"] == 0

    def test_a_torn_last_line_is_not_consumed_and_is_consumed_once_completed(
        self, tmp_path: Path
    ) -> None:
        directory = _session(tmp_path)
        path = _write(directory, _user("old", 1000.0) + _user("new", 2000.0).rstrip(b"\n"))
        assert last_user.last_user_at(directory) == 1000.0, "the torn row is not counted yet"
        _append(path, b"\n")
        assert last_user.last_user_at(directory) == 2000.0

    def test_a_torn_first_row_is_not_consumed_and_is_consumed_once_completed(
        self, tmp_path: Path
    ) -> None:
        """The whole file is one unterminated row: nothing is covered yet."""
        directory = _session(tmp_path)
        path = _write(directory, _user("only", 2000.0).rstrip(b"\n"))
        assert last_user.last_user_at(directory) is None
        _append(path, b"\n")
        assert last_user.last_user_at(directory) == 2000.0

    def test_an_inode_replace_recools(self, tmp_path: Path) -> None:
        """``compact_file`` replaces the journal with ``os.replace``."""
        directory = _session(tmp_path)
        path = _write(directory, _user("v1", 1000.0))
        assert last_user.last_user_at(directory) == 1000.0

        staged = directory / "replacement.jsonl"
        staged.write_bytes(_user("v2", 2000.0) + _assistant("resumed", 2100.0))
        os.replace(staged, path)
        assert last_user.last_user_at(directory) == 2000.0

    def test_a_shrunk_file_recools(self, tmp_path: Path) -> None:
        directory = _session(tmp_path)
        path = _write(directory, _user("v1", 1000.0), _assistant("lots of it", 1100.0))
        assert last_user.last_user_at(directory) == 1000.0
        with path.open("wb") as handle:  # same inode, smaller file
            handle.write(_user("v2", 2000.0))
        assert last_user.last_user_at(directory) == 2000.0


# ---------------------------------------------------------------------------
# The cap, and the answers that are remembered.
# ---------------------------------------------------------------------------


class TestTheColdCap:
    def test_the_cold_cap_returns_none_without_rescanning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(last_user, "COLD_SCAN_CAP_BYTES", 4096)
        monkeypatch.setattr(last_user, "_CHUNK_BYTES", 1024)
        directory = _session(tmp_path)
        chunks = [_user("out of budget", 1000.0)]
        filler = 0
        while filler < 20_000:
            row = _assistant("y" * 200, 1100.0 + filler)
            filler += len(row)
            chunks.append(row)
        path = _write(directory, *chunks)

        reads = _counting_reads(monkeypatch)
        assert last_user.last_user_at(directory) is None
        assert reads["bytes"] == 4096, "the cap bounds the cold read"

        reads["bytes"] = 0
        assert last_user.last_user_at(directory) is None
        assert reads["bytes"] == 0, "the same bytes are never rescanned"

        # The cap is a statement about the OLD bytes, not a permanent verdict:
        # a typed row appended later is the newest row and is found.
        _append(path, _user("typed now", 3000.0))
        assert last_user.last_user_at(directory) == 3000.0


class TestTheAnswersThatAreRemembered:
    def test_a_file_with_no_typed_row_returns_none_and_that_is_remembered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _session(tmp_path)
        _write(directory, _assistant("only", 1000.0), _custom("wake_prompt", 1100.0))
        assert last_user.last_user_at(directory) is None
        reads = _counting_reads(monkeypatch)
        assert last_user.last_user_at(directory) is None
        assert reads["bytes"] == 0

    def test_a_missing_or_unreadable_file_returns_none_without_raising(
        self, tmp_path: Path
    ) -> None:
        assert last_user.last_user_at(tmp_path / "no-such-session") is None
        empty = _session(tmp_path, "empty")
        assert last_user.last_user_at(empty) is None

        if os.geteuid() != 0:  # root reads anything; the permission arm needs a non-root run
            locked = _session(tmp_path, "locked")
            _write(locked, _user("unreadable", 1000.0))
            (locked / "transcript.jsonl").chmod(0)
            try:
                assert last_user.last_user_at(locked) is None
            finally:
                (locked / "transcript.jsonl").chmod(0o600)


# ---------------------------------------------------------------------------
# The memo: bounded, shared, and never pruned to whatever is live.
# ---------------------------------------------------------------------------


class TestTheMemo:
    def test_the_memo_is_a_bounded_lru(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(last_user, "_MEMO_MAX", 4)
        keys = []
        for index in range(6):
            directory = _session(tmp_path, f"s{index}")
            _write(directory, _user("x", 1000.0 + index))
            assert last_user.last_user_at(directory) == 1000.0 + index
            keys.append(str(directory / "transcript.jsonl"))

        assert len(last_user._MEMO) == 4
        assert keys[0] not in last_user._MEMO and keys[1] not in last_user._MEMO
        assert list(last_user._MEMO) == keys[2:], "oldest evicted first, newest kept"

        # A hit MOVES the entry to the most-recent end, so the next insert
        # evicts the least recently USED entry rather than the oldest one.
        assert last_user.last_user_at(tmp_path / "s2") == 1002.0
        assert list(last_user._MEMO)[-1] == keys[2]
        directory = _session(tmp_path, "s6")
        _write(directory, _user("x", 2000.0))
        assert last_user.last_user_at(directory) == 2000.0
        assert keys[2] in last_user._MEMO and keys[3] not in last_user._MEMO

    def test_the_memo_is_not_pruned_per_poll(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session that goes busy -> idle -> busy must not pay a cold scan again.

        Entries are keyed per session and only ever evicted by the LRU bound —
        never because a poll happened, or because the session is not running
        (the tracker is not consulted about liveness at all).
        """
        directory = _session(tmp_path, "the-one")
        path = _write(directory, _user("busy question", 1000.0))
        assert last_user.last_user_at(directory) == 1000.0

        for index in range(30):  # other sessions come and go
            other = _session(tmp_path, f"other{index}")
            _write(other, _user("x", 1000.0))
            last_user.last_user_at(other)

        reads = _counting_reads(monkeypatch)
        _append(path, _assistant("after the idle", 1100.0))
        assert last_user.last_user_at(directory) == 1000.0
        assert reads["bytes"] == len(
            _assistant("after the idle", 1100.0)
        ), "still the warm path, not a cold re-scan"

    def test_the_memo_survives_concurrent_use(self, tmp_path: Path) -> None:
        """The lock exists because the listing runs on worker threads."""
        shared = _session(tmp_path, "shared")
        _write(shared, _user("shared", 1000.0))
        others = []
        for index in range(8):
            directory = _session(tmp_path, f"t{index}")
            _write(directory, _user("x", 2000.0 + index))
            others.append(directory)

        errors: list[BaseException] = []
        results: dict[int, list[float | None]] = {index: [] for index in range(6)}

        def worker(index: int) -> None:
            try:
                for step in range(30):
                    directory = shared if step % 2 else others[(index + step) % len(others)]
                    results[index].append(last_user.last_user_at(directory))
            except BaseException as exc:  # noqa: BLE001 - reported by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        for index, values in results.items():
            assert values, f"thread {index} made calls"
            for step, value in enumerate(values):
                directory = shared if step % 2 else others[(index + step) % len(others)]
                expected = 1000.0 if directory == shared else 2000.0 + (index + step) % len(others)
                assert value == expected
