"""The ``/resume`` preview's text layer, over a fixture transcript.

Pure functions only — no Textual, no app. The predicates here were derived by
measuring the real store (1,711 content blocks across the 25 newest
transcripts), and each test pins one measurement that a plausible-looking
rewrite would silently break.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from local_operator.session.preview import (
    CHECKPOINT_CUSTOM_TYPE,
    PREVIEW_TAIL_BYTES,
    SessionPreviews,
    condense_entries,
    demark,
    grep_context,
    wrap_turns,
)


def _message(role: str, text: str, ts: float = 1.0, kind: str = "message") -> dict[str, Any]:
    """A transcript message entry. Blocks carry NO ``type`` key, as real ones do not."""
    return {
        "type": "message",
        "ts": ts,
        "payload": {"kind": kind, "role": role, "content": [{"text": text}]},
    }


def _write(tmp_path: Path, session_id: str, entries: list[dict[str, Any]]) -> Path:
    session = tmp_path / "sessions" / session_id
    session.mkdir(parents=True, exist_ok=True)
    with (session / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return session


def test_only_user_and_assistant_message_entries_survive_the_condense() -> None:
    """``role == "tool"`` is noise, ``kind == "custom"`` is journal, empty is nothing."""
    turns = condense_entries(
        [
            _message("user", "the real question"),
            _message("tool", "tool chatter"),
            _message("assistant", "", ts=2.0),
            _message("assistant", "a real answer", ts=3.0),
            _message("user", "journal noise", kind="custom"),
        ]
    )
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert [turn.text for turn in turns] == ["the real question", "a real answer"]


def test_a_content_block_without_a_type_key_is_still_read() -> None:
    """Verified across 1,711 real blocks: key-sets are ``('text',)`` and
    ``('attachment','mime_type')``. NO block carries a ``type`` key, so a
    predicate on ``block["type"]`` matches NOTHING. This proves there is none.
    """
    turns = condense_entries([_message("user", "no type key here")])
    assert [turn.text for turn in turns] == ["no type key here"]


def test_an_attachment_block_renders_a_placeholder_instead_of_raising() -> None:
    """``attachment`` blocks have no ``text``, so a bare ``b["text"]`` raises."""
    entry = {
        "type": "message",
        "ts": 1.0,
        "payload": {
            "kind": "message",
            "role": "user",
            "content": [{"attachment": "a.png", "mime_type": "image/png"}, {"text": " look"}],
        },
    }
    turns = condense_entries([entry])
    assert turns[0].text == "[attachment] look"


def test_the_preview_opens_on_the_first_user_turn() -> None:
    """D18. Measured: 36 of 141 transcripts open on mid-session assistant
    narration, and those 36 included the default cursor row — which is why no
    round-2 frame contained a single ``▸ you``. Leading assistant turns are
    DROPPED, not scrolled past: at 80x24 the pane shows one turn, so "somewhere
    below" is the same as absent.
    """
    turns = condense_entries(
        [
            _message("assistant", "Fixing the stale import first", ts=1.0),
            _message("assistant", "still working", ts=2.0),
            _message("user", "what I actually asked", ts=3.0),
            _message("assistant", "the reply", ts=4.0),
        ]
    )
    lines = wrap_turns(turns, width=60, height=40)
    gutters = [text for kind, text in lines if kind == "gutter"]
    assert gutters[0] == "▸ you"
    assert "Fixing the stale import first" not in "\n".join(text for _kind, text in lines)


def test_markdown_emphasis_is_stripped_and_bullets_survive() -> None:
    """D5. Bullets are STRUCTURE the reader wants, not emphasis, so ``* item``
    must never match the ``*em*`` pattern; ``_`` only strips when it is not
    flanked by word characters, so ``snake_case`` survives.
    """
    assert demark("## Heading") == "Heading"
    assert demark("a `code` span") == "a code span"
    assert demark("**bold** and __also__") == "bold and also"
    assert demark("*em* and _em_") == "em and em"
    assert demark("- item one") == "- item one"
    assert demark("* item two") == "* item two"
    assert demark("snake_case_name") == "snake_case_name"


def test_no_line_breaks_inside_a_word() -> None:
    """D10: word-boundary wrap, so a continuation never splits a word that FITS.

    A word LONGER than the pane is broken deliberately — that is what keeps a
    200-char URL from overflowing — so the invariant is asserted over words
    that fit, which is every word a prose turn actually contains.
    """
    width = 24
    text = "the quick brown fox jumps over the lazy dog and then keeps running onward"
    assert all(len(word) <= width for word in text.split())
    turns = condense_entries([_message("user", text)])
    lines = [line for kind, line in wrap_turns(turns, width=width, height=40) if kind == "user"]
    assert len(lines) > 1
    assert all(len(line) <= width for line in lines)
    rejoined = " ".join(lines)
    for word in text.split():
        assert word in rejoined, f"{word!r} was split across lines"


def test_a_role_header_is_never_the_last_line() -> None:
    """D30. Three round-3 frames ended on a bare ``▪ lop`` with nothing beneath,
    which reads as a turn that failed to load. A truncated sentence reads as
    continuation and is correct; a label with nothing after it looks broken.
    """
    turns = condense_entries(
        [
            _message("user", "one two three four five six seven", ts=1.0),
            _message("assistant", "the answer body", ts=2.0),
        ]
    )
    for height in range(1, 12):
        lines = wrap_turns(turns, width=30, height=height)
        if not lines:
            continue
        assert lines[-1][0] != "gutter", f"orphan role header at height={height}: {lines}"


def test_the_checkpoint_is_found_by_custom_type_not_by_kind(tmp_path: Path) -> None:
    """The discriminator is ``payload["custom_type"]`` with ``entry["type"] ==
    "custom"``. It is NOT ``payload["kind"]`` — that is ``None`` on these
    entries, so this fails immediately if anyone keys on ``kind``.
    """
    _write(
        tmp_path,
        "aa11bb22cc33",
        [
            _message("user", "hello"),
            {
                "type": "custom",
                "ts": 2.0,
                "payload": {
                    "kind": None,
                    "custom_type": CHECKPOINT_CUSTOM_TYPE,
                    "details": {
                        "state": {
                            "cwd": "/tmp/x",
                            "effective_model": {"model_id": "anthropic/claude-opus-5"},
                        }
                    },
                },
            },
        ],
    )
    previews = SessionPreviews(tmp_path / "sessions")
    state = previews.checkpoint("aa11bb22cc33")
    assert state["cwd"] == "/tmp/x"
    assert state["effective_model"]["model_id"] == "anthropic/claude-opus-5"


def test_a_session_without_a_checkpoint_omits_the_model_line_entirely(tmp_path: Path) -> None:
    """D7: no bare ``· · ·`` placeholder that reads as a load that never resolved."""
    _write(tmp_path, "bb22cc33dd44", [_message("user", "no checkpoint here")])
    previews = SessionPreviews(tmp_path / "sessions")
    assert previews.checkpoint("bb22cc33dd44") == {}


def test_the_tail_read_is_bounded_by_the_window_not_the_file(tmp_path: Path) -> None:
    """Bytes read are counted through a wrapper, so this asserts the BOUND
    rather than a wall-clock proxy. The cost of a first preview is independent
    of file size: one ``stat``, one ``seek``, one 256 KB read.
    """
    filler = "x" * 4_000
    entries = [_message("user", f"{index} {filler}", ts=float(index)) for index in range(200)]
    entries.append(_message("user", "THE TAIL MARKER", ts=999.0))
    session = _write(tmp_path, "cc33dd44ee55", entries)
    transcript = session / "transcript.jsonl"
    assert transcript.stat().st_size > PREVIEW_TAIL_BYTES

    read_bytes = 0
    real_open = Path.open

    def counting_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        handle = real_open(self, *args, **kwargs)
        if self == transcript:
            real_read = handle.read

            def read(*read_args):  # type: ignore[no-untyped-def]
                nonlocal read_bytes
                data = real_read(*read_args)
                read_bytes += len(data)
                return data

            handle.read = read  # type: ignore[method-assign]
        return handle

    Path.open = counting_open  # type: ignore[method-assign]
    try:
        turns = SessionPreviews(tmp_path / "sessions").condensed("cc33dd44ee55")
    finally:
        Path.open = real_open  # type: ignore[method-assign]

    assert read_bytes <= PREVIEW_TAIL_BYTES
    assert read_bytes < transcript.stat().st_size
    assert turns[-1].text.endswith("THE TAIL MARKER")
    assert not any(turn.text.startswith("0 ") for turn in turns)


def test_grep_context_is_centred_on_the_hit_at_the_width_requested() -> None:
    """D17. The prototype asked for ``width=150`` and then truncated that
    snippet into a ~55-cell pane FROM THE LEFT, cutting the match off the right
    end: the query sat at index 73 of a 152-char snippet, so 0 of 9 context
    lines contained the query. Requesting the width actually drawn fixes it.
    """
    digest = ("lorem ipsum dolor sit amet " * 3) + "picker" + (" consectetur adipiscing elit" * 5)
    assert digest.index("picker") > 70
    snippet = grep_context(digest, "picker", width=55)
    assert snippet is not None
    assert "picker" in snippet
    assert len(snippet) <= 55 + 2  # the two ellipsis marks


def test_a_fuzzy_only_match_returns_no_context() -> None:
    """A fuzzy hit has no literal substring, which is exactly why it gets ``~``."""
    assert grep_context("a digest about sessions", "pikcer", width=55) is None
    assert grep_context("", "picker", width=55) is None


def test_created_at_and_verbose_read_the_same_bounded_window(tmp_path: Path) -> None:
    """``verbose`` keeps ``role == "tool"`` and custom entries the condense drops."""
    _write(
        tmp_path,
        "dd44ee55ff66",
        [
            _message("user", "ask", ts=1.0),
            _message("tool", "tool output", ts=2.0),
            _message("assistant", "answer", ts=3.0),
        ],
    )
    previews = SessionPreviews(tmp_path / "sessions")
    assert [turn.role for turn in previews.condensed("dd44ee55ff66")] == ["user", "assistant"]
    assert [turn.role for turn in previews.verbose("dd44ee55ff66")] == [
        "user",
        "tool",
        "assistant",
    ]
    created = previews.created_at("dd44ee55ff66")
    assert 0.0 < created <= time.time() + 1


def test_a_missing_transcript_returns_empty_rather_than_raising(tmp_path: Path) -> None:
    """The picker paints on a cursor move; an unreadable session must not crash it."""
    previews = SessionPreviews(tmp_path / "sessions")
    assert previews.condensed("no-such-session") == []
    assert previews.verbose("no-such-session") == []
    assert previews.checkpoint("no-such-session") == {}
    assert previews.created_at("no-such-session") == 0.0
