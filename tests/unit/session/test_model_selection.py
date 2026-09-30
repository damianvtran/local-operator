"""``read_model_selection``: the backward settle scan against the fold it replaced.

The rewrite is a pure optimisation — same rows read, same answer — so the tests
here are DIFFERENTIAL first: the pre-rewrite implementation is kept verbatim as
:func:`_forward_fold` and every case is asserted against it, over a matrix of
journals that covers each branch of the fold. Targeted assertions then pin the
fields the differential alone would report as "equal, whatever they are" — the
settled row, and ``recovered`` for the one case that distinguishes the scan from a
naive "first v2 row": an invalid v2 row ABOVE a valid one.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.model_selection import (
    SELECTED_MODEL_CUSTOM_TYPE,
    SELECTION_VERSION,
    read_model_selection,
    refused_decision_only_selection,
    selection_from_payloads,
)


@pytest.fixture(autouse=True)
def _clean_selection_memo():
    """The selection memo is process-wide; no test may inherit another's entry."""
    from local_operator.session import model_selection as module

    module._reset_for_tests()
    yield
    module._reset_for_tests()


PROVIDER_A = "test/conversation-a"
PROVIDER_B = "test/default-b"
CHECKPOINT_TYPE = "frontend_state_checkpoint_v1"


def _forward_fold(directory: Path):
    """The pre-rewrite implementation, VERBATIM, as the oracle.

    Copied from ``git show HEAD:local_operator/session/model_selection.py`` rather
    than re-expressed through the module's own helpers: an oracle built out of the
    code under test proves nothing about the code under test.
    """

    def payloads():
        try:
            with (directory / "transcript.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    # Large transcripts mostly contain messages. Do not parse
                    # their payloads just to recover a two-field selector.
                    if '"custom_type"' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if (
                        isinstance(row, dict)
                        and row.get("type") == "custom"
                        and isinstance(row.get("payload"), dict)
                    ):
                        yield row["payload"]
        except (OSError, UnicodeError):
            return

    return selection_from_payloads(payloads())


def _row(payload: Any, *, kind: str = "custom") -> dict[str, Any]:
    return {"id": uuid.uuid4().hex, "ts": 1787000000.0, "type": kind, "payload": payload}


def _selection_row(
    selector: str | None,
    *,
    version: int | None = SELECTION_VERSION,
    effort: Any = None,
    boot: str | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if selector is not None:
        details["selector"] = selector
    if version is not None:
        details["version"] = version
    if effort is not None:
        details["effort"] = effort
    if boot is not None:
        details["boot"] = boot
    return _row({"custom_type": SELECTED_MODEL_CUSTOM_TYPE, "details": details})


def _checkpoint_row(provider: str = "test", model_id: str = "default-b") -> dict[str, Any]:
    return _row(
        {
            "custom_type": CHECKPOINT_TYPE,
            "details": {"state": {"selected_model": {"provider": provider, "model_id": model_id}}},
        }
    )


def _message_row(text: str) -> dict[str, Any]:
    return _row({"kind": "message", "role": "assistant", "content": [{"text": text}]})


def _journal(directory: Path, rows: list[Any], *, newline: bool = True) -> Path:
    """Write raw rows, so malformed and non-dict shapes are expressible."""
    directory.mkdir(parents=True, exist_ok=True)
    body = "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows)
    (directory / "transcript.jsonl").write_text(body + ("\n" if newline else ""), encoding="utf-8")
    return directory


CASES: dict[str, list[Any]] = {
    "empty journal": [],
    "messages only": [_message_row("hello"), _message_row("world")],
    "one valid v2 row": [_selection_row(PROVIDER_A)],
    # THE CASE THE SCAN EXISTS FOR: the invalid row is NEWER, so the answer is the
    # older valid row and `recovered` must be true. A "take the first v2 row"
    # implementation gets the selector right and `recovered` wrong.
    "invalid v2 above a valid v2": [
        _selection_row(PROVIDER_A),
        _selection_row("missing-provider/model"),
    ],
    "invalid v2 above two valid v2 rows": [
        _selection_row(PROVIDER_A),
        _selection_row(PROVIDER_B),
        _selection_row("missing-provider/model"),
    ],
    "invalid v2 above a valid v2, then a valid v2": [
        _selection_row(PROVIDER_A),
        _selection_row("missing-provider/model"),
        _selection_row(PROVIDER_B),
    ],
    "invalid v2 only": [_selection_row("missing-provider/model")],
    "empty selector v2 above a valid v2": [
        _selection_row(PROVIDER_A),
        _selection_row(None),
        _selection_row("no-slash"),
    ],
    "legacy row only": [_selection_row(PROVIDER_A, version=None)],
    "legacy row then a checkpoint": [
        _selection_row(PROVIDER_A, boot="test/old-boot", version=None),
        _checkpoint_row(),
    ],
    "checkpoint newer than a valid v2 row is discarded": [
        _selection_row(PROVIDER_A),
        _checkpoint_row(provider="test", model_id="conversation-a"),
    ],
    "unversioned row after a valid v2 row": [
        _selection_row(PROVIDER_A),
        _selection_row("missing-provider/model", version=None),
    ],
    "unknown version is ignored": [_selection_row(PROVIDER_A, version=3)],
    "malformed line between rows": [
        _selection_row(PROVIDER_A),
        "{not json",
        _selection_row(PROVIDER_B),
    ],
    "details is not a mapping": [
        _row({"custom_type": SELECTED_MODEL_CUSTOM_TYPE, "details": "nope"}),
        _selection_row(PROVIDER_A),
    ],
    "payload is not a mapping": [
        _row("not-a-mapping"),
        _selection_row(PROVIDER_A),
    ],
    "row is not a mapping": ["[1, 2, 3]", _selection_row(PROVIDER_A)],
    "non-custom row with the marker": [
        _row({"custom_type": SELECTED_MODEL_CUSTOM_TYPE, "details": {}}, kind="message"),
        _selection_row(PROVIDER_A),
    ],
    "no trailing newline": [_selection_row(PROVIDER_A)],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_settle_scan_answers_exactly_what_the_fold_answered(tmp_path, name):
    directory = _journal(
        tmp_path / "sessions" / "case",
        CASES[name],
        newline=name != "no trailing newline",
    )
    assert read_model_selection(directory) == _forward_fold(directory)


def test_a_missing_or_empty_journal_answers_none(tmp_path):
    absent = tmp_path / "sessions" / "absent"
    assert read_model_selection(absent) is None
    assert _forward_fold(absent) is None
    empty = _journal(tmp_path / "sessions" / "empty", [])
    assert read_model_selection(empty) is None
    assert _forward_fold(empty) is None


def test_an_invalid_v2_row_above_a_valid_one_is_recovered(tmp_path):
    """The named regression: newest valid v2 wins, and `recovered` is true.

    `recovered` is what makes the cold viewer tell the user its saved model
    information was incomplete, so getting it from the wrong side of the settled
    row is a user-visible difference rather than a bookkeeping one.
    """
    directory = _journal(
        tmp_path / "sessions" / "recover",
        [
            _selection_row(PROVIDER_B, version=None),
            _selection_row(PROVIDER_A),
            _selection_row("missing-provider/model"),
        ],
    )
    saved = read_model_selection(directory)
    assert saved is not None
    assert saved.selector == PROVIDER_A
    assert saved.authoritative
    assert saved.recovered is True
    assert saved == _forward_fold(directory)


def test_a_valid_v2_row_above_an_invalid_one_is_not_recovered(tmp_path):
    directory = _journal(
        tmp_path / "sessions" / "clean",
        [
            _selection_row("missing-provider/model"),
            _selection_row(PROVIDER_A),
        ],
    )
    saved = read_model_selection(directory)
    assert saved is not None
    assert saved.selector == PROVIDER_A
    assert saved.recovered is False


def test_a_v2_row_below_the_settled_one_is_never_read(tmp_path, monkeypatch):
    """The saving, stated as a test: rows under the settled row are not visited.

    ``_settled_selection`` may stop at the first valid v2 row, so anything older
    must be irrelevant by construction. A row that would RAISE if parsed is the
    sharpest way to say so — if the scan ever reads past the settled row this
    test fails with the exception rather than with a comparison.
    """
    from local_operator.session import model_selection as module

    directory = _journal(
        tmp_path / "sessions" / "below",
        [_selection_row(PROVIDER_B), _selection_row(PROVIDER_A)],
    )
    real = module._selection
    seen: list[Any] = []

    def watching(selector, effort, *, authoritative=False, boot=None):
        seen.append(selector)
        return real(selector, effort, authoritative=authoritative, boot=boot)

    monkeypatch.setattr(module, "_selection", watching)
    saved = read_model_selection(directory)
    assert saved is not None and saved.selector == PROVIDER_A
    assert seen == [PROVIDER_A], "the scan resolved rows below the settled v2 row"


def test_the_fallback_fold_still_answers_a_journal_with_no_v2_row(tmp_path, monkeypatch):
    """103 of the store's transcripts take this path; it must stay the old code.

    The fold is asserted to RUN (not merely to agree), because the agreement is
    what would hide a scan that silently treated "no v2 row" as "no selection".
    """
    from local_operator.session import model_selection as module

    directory = _journal(
        tmp_path / "sessions" / "legacy", [_selection_row(PROVIDER_A, version=None)]
    )
    calls: list[Path] = []
    real = module._forward_payloads

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(module, "_forward_payloads", counting)
    saved = read_model_selection(directory)
    assert saved is not None and saved.selector == PROVIDER_A
    assert calls == [directory]
    assert saved == _forward_fold(directory)


def test_a_byte_corrupt_journal_is_answered_rather_than_raising(tmp_path):
    """The named divergence: rows decode with ``errors="replace"``.

    ``_forward_payloads`` decodes strictly, so one bad byte raises out of the
    iteration and its fold ends at the corruption, silently answering from the
    prefix ABOVE it — here, from nothing at all. The scan answers from the newest
    valid v2 row instead, which is the same direction the module's other backward
    readers already took, and the one a reader would expect from a journal with one
    torn row in it. Both halves are asserted so the docstring's claim cannot drift
    from the behaviour in either direction.
    """
    directory = tmp_path / "sessions" / "corrupt"
    directory.mkdir(parents=True)
    torn = b'{"id":"\xff\xfe"}\n'
    valid = json.dumps(_selection_row(PROVIDER_A)).encode("utf-8")
    (directory / "transcript.jsonl").write_bytes(torn + valid + b"\n")
    saved = read_model_selection(directory)
    assert saved is not None and saved.selector == PROVIDER_A
    assert _forward_fold(directory) is None, "the fold no longer stops at the torn row"


# ---------------------------------------------------------------------------
# The stat-keyed memo (the 2026-09-30 open-cost audit): what it must NOT
# re-read, what it must re-read, and that it never changes an answer
# ---------------------------------------------------------------------------


def test_an_unchanged_journal_is_not_re_walked_for_the_settle_path(tmp_path, monkeypatch):
    """The whole point: a second read of an unchanged journal costs one stat.

    Counted at the scan, not by wall clock — a clock bound here would be a bet
    on machine load, and the scan is exactly what the memo exists to skip.
    """
    from local_operator.session import model_selection as module

    directory = _journal(tmp_path / "sessions" / "memo-settle", [_selection_row(PROVIDER_A)])
    calls: list[Path] = []
    real = module._settled_selection

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(module, "_settled_selection", counting)
    first = read_model_selection(directory)
    second = read_model_selection(directory)
    assert first is not None and first.selector == PROVIDER_A
    assert second == first
    assert calls == [directory], "the second call re-walked an unchanged journal"


def test_the_fold_fallback_is_memoised_even_when_it_answers_none(tmp_path, monkeypatch):
    """Both outcomes are cached, including the fold's ``None``.

    The fold is the path most of the store's transcripts take; a journal whose
    fold answers ``None`` must not re-walk on every open either.
    """
    from local_operator.session import model_selection as module

    directory = _journal(
        tmp_path / "sessions" / "memo-fold", [_selection_row(PROVIDER_A, version=None)]
    )
    calls: list[Path] = []
    real = module._forward_payloads

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(module, "_forward_payloads", counting)
    first = read_model_selection(directory)
    second = read_model_selection(directory)
    assert first is not None and first.selector == PROVIDER_A
    assert second == first
    assert calls == [directory]

    empty = _journal(tmp_path / "sessions" / "memo-none", [_message_row("hello")])
    assert read_model_selection(empty) is None
    assert read_model_selection(empty) is None
    assert calls == [directory, empty], "the None fold was re-walked"


def test_an_append_recomputes_once_then_memoises_again(tmp_path, monkeypatch):
    """A grown journal (size/mtime moved) is recomputed, and the new answer cached."""
    from local_operator.session import model_selection as module

    directory = _journal(tmp_path / "sessions" / "memo-append", [_selection_row(PROVIDER_A)])
    calls: list[Path] = []
    real = module._settled_selection

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(module, "_settled_selection", counting)
    before = read_model_selection(directory)
    assert before is not None and before.selector == PROVIDER_A

    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_selection_row(PROVIDER_B)) + "\n")

    after = read_model_selection(directory)
    again = read_model_selection(directory)
    assert after is not None and after.selector == PROVIDER_B
    assert again == after
    assert calls == [directory, directory], "append: one recompute, then cached again"

    module._reset_for_tests()
    assert after == read_model_selection(directory)  # equals the uncached answer


def test_a_replacement_at_identical_size_and_mtime_is_detected_by_inode(tmp_path):
    """``os.replace`` can keep size and mtime; only ``st_ino`` moves — so the
    memo must key on it (the same term ``transcript_index``'s scan signature
    carries for exactly this case).
    """
    import os

    directory = _journal(
        tmp_path / "sessions" / "memo-swap", [_selection_row("test/conversation-a")]
    )
    path = directory / "transcript.jsonl"
    first = read_model_selection(directory)
    assert first is not None and first.selector == "test/conversation-a"
    st = path.stat()

    swap = directory / "swap.tmp"
    swap.write_text(json.dumps(_selection_row("test/conversation-z")) + "\n", encoding="utf-8")
    os.utime(swap, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(swap, path)

    replaced = path.stat()
    assert replaced.st_ino != st.st_ino, "premise: os.replace gave a new inode"
    assert replaced.st_size == st.st_size
    assert replaced.st_mtime_ns == st.st_mtime_ns, "premise: only the inode moved"

    second = read_model_selection(directory)
    assert second is not None and second.selector == "test/conversation-z"


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_memoised_answer_equals_an_uncached_one(tmp_path, name):
    """The memo may change WHAT is read, never WHAT is answered."""
    from local_operator.session import model_selection as module

    directory = _journal(
        tmp_path / "sessions" / "case", CASES[name], newline=name != "no trailing newline"
    )
    cached = read_model_selection(directory)
    module._reset_for_tests()
    assert cached == read_model_selection(directory)


def test_the_refusal_reader_is_memoised_on_the_same_key(tmp_path, monkeypatch):
    """The refusal walk rides the same memo — it is the same per-open cost."""
    from local_operator.session import model_selection as module

    directory = _journal(tmp_path / "sessions" / "memo-refusal", [_selection_row(PROVIDER_A)])
    calls: list[int] = []
    real = module._iter_complete_lines_backward

    def counting(handle, end):
        calls.append(end)
        return real(handle, end)

    monkeypatch.setattr(module, "_iter_complete_lines_backward", counting)
    assert refused_decision_only_selection(directory) is None
    assert refused_decision_only_selection(directory) is None
    assert len(calls) == 1, "the second call re-walked the journal for a refusal"


def test_the_memo_is_bounded_and_evicts_least_recently_used(tmp_path):
    """128 entries, LRU: the hot session survives a sweep of one-off reads."""
    from local_operator.session import model_selection as module

    directories = []
    for i in range(module._SELECTION_MEMO_MAX):
        directory = _journal(tmp_path / "sessions" / f"memo-{i:03d}", [_selection_row(PROVIDER_A)])
        directories.append(directory)
        assert read_model_selection(directory) is not None
    assert len(module._SELECTION_MEMO) == module._SELECTION_MEMO_MAX

    # Re-open the OLDEST entry, then add one more: an LRU evicts the
    # second-oldest; a FIFO would have dropped the re-opened one instead.
    assert read_model_selection(directories[0]) is not None
    newest = _journal(tmp_path / "sessions" / "memo-extra", [_selection_row(PROVIDER_A)])
    assert read_model_selection(newest) is not None

    keys = list(module._SELECTION_MEMO)
    assert len(keys) == module._SELECTION_MEMO_MAX
    assert str(directories[0] / "transcript.jsonl") in keys
    assert str(directories[1] / "transcript.jsonl") not in keys
    assert keys[-1] == str(newest / "transcript.jsonl")
