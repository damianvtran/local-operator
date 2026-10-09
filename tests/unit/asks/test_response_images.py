"""Image answers on a queued ask: from the queue's write to the model's turn.

WHY THIS FILE EXISTS. An ask answer was strings at every hop; the Other door's
attachments make it carry pictures (the ``Answer.images`` wire, capability
``ask-attachments-v1``). The properties worth pinning are all about LOSS, because
an answer is terminal and an image that vanishes is the one failure the user can
neither see nor repair:

* the bytes go to the AttachmentStore FIRST and ``asks.jsonl`` carries refs only;
* a picture the queue cannot keep REFUSES the whole answer and leaves the ask
  open (secret question, unknown question, too many, a store that will not write);
* the fold keeps the FIRST answer's refs across a revision, and a revision that
  tries to carry images is refused in words;
* the model's turn is ``[Text, Image...]`` exactly as a composer send is, and the
  durable row stays tiny because the bytes ride the transcript's own
  externalisation (``content`` blocks), not ``details``.

The session is the real one (``make_session``) and so is the queue; the stream
scripts only what the provider would see. Spike B in the design doc is the
template for the transcript cells.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import queue as queue_module
from local_operator.asks import render, store
from local_operator.harness.render import _default_convert_to_llm
from local_operator.harness.types import (
    ImageContent,
    Message,
    StreamEndEvent,
    TextContent,
)
from local_operator.session.attachments import store_for_transcript_dir
from tests.unit.session.test_session import make_session


def _png_b64(width: int = 64, height: int = 64) -> str:
    """A real PNG of random noise: big enough to cross the store's 1 KiB floor."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + os.urandom(width * 3) for _ in range(height))
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(blob).decode("ascii")


def _image() -> ImageContent:
    return ImageContent(data=_png_b64(), mime_type="image/png")


def _questions(*, secret_id: str = "") -> list[dict[str, Any]]:
    out = [
        {
            "id": "q1",
            "question": "Which one?",
            "options": [{"label": "yes"}, {"label": "no"}],
            "multi": False,
            "secret": False,
            "persist": False,
            "recommended": None,
        }
    ]
    if secret_id:
        out.append(
            {
                "id": secret_id,
                "question": "Paste the key",
                "options": [],
                "multi": False,
                "secret": True,
                "persist": False,
                "recommended": None,
            }
        )
    return out


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


class _Stream:
    """Records every request the provider would have seen."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> Any:
        self.requests.append(request)

        async def gen() -> Any:
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def _ask_session(tmp_path: Path, stream: _Stream | None = None) -> tuple[Any, Any, _Stream]:
    stream = stream or _Stream()

    async def handler(questions: Any) -> Any:
        return None

    session = make_session(tmp_path, stream)
    session.set_ask_handler(handler)
    queue = session.ask_queue()
    assert queue is not None
    return session, queue, stream


def _enqueue(queue: Any, questions: list[dict[str, Any]]) -> str:
    outcome = queue.enqueue(questions, None)
    assert outcome["ok"] is True, outcome
    return str(outcome["details"]["ask_id"])


async def _drain(session: Any) -> None:
    for _ in range(10):
        pending = [t for t in list(session._background_tasks) if not t.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
    raise AssertionError("delivery turns never settled")


def _answered_event(queue: Any, ask_id: str) -> dict[str, Any]:
    events = [
        e
        for e in store.read_events(queue.session_dir)
        if e["ask_id"] == ask_id and e["kind"] == store.EVENT_ANSWERED
    ]
    assert len(events) == 1, events
    return events[0]


# ---------------------------------------------------------------------------
# The write: bytes first, refs in the log, never base64
# ---------------------------------------------------------------------------


def test_the_bytes_go_to_the_store_and_the_log_carries_refs_only(
    isolated_config: Path, tmp_path: Path
) -> None:
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())
    image = _image()

    outcome = session.respond_ask(ask_id, {"q1": ["see screenshot"]}, attachments={"q1": [image]})

    assert outcome == {"ok": True}
    event = _answered_event(queue, ask_id)
    (ref,) = event["attachments"]["q1"]
    assert set(ref) == {"attachment", "mime_type", "bytes"}
    # The ref resolves to the exact bytes in the session's OWN store root.
    found = store_for_transcript_dir(queue.session_dir).get(ref["attachment"])
    assert found is not None and found[0] == image.data
    # And the log row never contains the pixels (one-line O_APPEND rows; the fold
    # reads the whole file on every refresh).
    raw = store.asks_log_path(queue.session_dir).read_text()
    assert image.data[:64] not in raw
    assert len(raw) < 2_000, "an image answer must not inflate the ask log"


def test_a_text_only_answer_writes_the_event_it_always_did(
    isolated_config: Path, tmp_path: Path
) -> None:
    """No ``attachments`` key at all: the row is byte-identical to before the field."""
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())

    assert session.respond_ask(ask_id, {"q1": ["yes"]})["ok"] is True

    assert "attachments" not in _answered_event(queue, ask_id)
    record = queue.find(ask_id)
    assert record is not None and "attachments" not in record
    (row,) = [r for r in queue.projection() if r["ask_id"] == ask_id]
    assert "attachments" not in row, "a text-only ask's wire row must stay byte-identical"


def test_an_image_on_a_secret_question_refuses_the_whole_answer(
    isolated_config: Path, tmp_path: Path
) -> None:
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions(secret_id="API_KEY"))

    outcome = session.respond_ask(
        ask_id,
        {"q1": ["yes"], "API_KEY": ["API_KEY"]},
        attachments={"API_KEY": [_image()]},
    )

    assert outcome["ok"] is False
    assert "secret" in outcome["error"]
    # Refused whole: nothing was recorded, nothing was stored, the ask is open.
    assert [e["kind"] for e in store.read_events(queue.session_dir)] == [store.EVENT_QUEUED]
    assert not (isolated_config / "attachments").exists()


def test_a_secret_is_not_stored_for_an_answer_refused_over_its_images(
    isolated_config: Path, tmp_path: Path
) -> None:
    """The image probe runs BEFORE the secret hop: a refused answer leaves no credential."""
    from local_operator.variables import VariableStore

    session, queue, _ = _ask_session(tmp_path)
    session._variables = VariableStore(cwd=str(tmp_path))  # noqa: SLF001
    ask_id = _enqueue(queue, _questions(secret_id="API_KEY"))

    outcome = session.respond_ask(
        ask_id,
        {"q1": ["yes"], "API_KEY": ["sk-live-do-not-store"]},
        attachments={"q1": [_image()] * (queue_module.MAX_ANSWER_IMAGES + 1)},
    )

    assert outcome["ok"] is False
    assert "API_KEY" not in session._variables.credential_names()  # noqa: SLF001


def test_an_image_for_an_unknown_question_is_refused(isolated_config: Path, tmp_path: Path) -> None:
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())

    outcome = session.respond_ask(ask_id, {"q1": ["yes"]}, attachments={"nope": [_image()]})

    assert outcome["ok"] is False and "'nope'" in outcome["error"]


def test_more_than_eight_images_are_refused_not_truncated(
    isolated_config: Path, tmp_path: Path
) -> None:
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())
    # A LITERAL 9, not ``MAX_ANSWER_IMAGES + 1``: the latter moves with the constant,
    # so a limit silently raised to 80 would still pass. The wire contract is 8.
    images = [_image() for _ in range(9)]

    outcome = queue.respond(ask_id, {"q1": ["yes"]}, attachments={"q1": images})

    assert outcome["ok"] is False and "at most 8" in outcome["error"]
    assert [e["kind"] for e in store.read_events(queue.session_dir)] == [store.EVENT_QUEUED]


def test_a_store_that_will_not_write_refuses_instead_of_recording_text_only(
    isolated_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no degraded form for this log (no base64 in it), so loss is refusal."""
    from local_operator.session import attachments as attachments_module

    monkeypatch.setattr(attachments_module.AttachmentStore, "put", lambda self, d, m: None)
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())

    outcome = session.respond_ask(ask_id, {"q1": ["yes"]}, attachments={"q1": [_image()]})

    assert outcome["ok"] is False and "could not be saved" in outcome["error"]
    assert [e["kind"] for e in store.read_events(queue.session_dir)] == [store.EVENT_QUEUED]


def test_a_store_failure_after_the_secret_hop_leaves_a_recoverable_ask(
    isolated_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one refusal the PROBE cannot see: a store that fails AFTER the secret hop.

    ``attachment_refusal`` (run before the hop) covers every refusal that is a pure
    function of the answer -- a secret question's image, an unknown question, too
    many. Whether the AttachmentStore will WRITE is only known by writing, which
    happens in ``AskQueue.respond`` after ``Session.respond_ask`` has stored the
    secret value. So "an answer refused for its pictures never stores a credential"
    holds for the probe-class refusals and NOT for this one; it is the same class
    ``AskQueue.revision_refusal`` records as deferred (PR #1954) -- recoverable, and
    no value reaches a durable surface (the value lives in the memory-only store,
    the log never carries it).

    What this cell pins is the recovery, which is the contract that matters: the
    refusal records NOTHING, the ask stays open, and the resend with a working
    store lands with the images. It deliberately does not assert the credential's
    presence -- that residue is a known limit, not a behaviour to freeze.
    """
    from local_operator.session import attachments as attachments_module
    from local_operator.variables import VariableStore

    session, queue, _ = _ask_session(tmp_path)
    session._variables = VariableStore(cwd=str(tmp_path))  # noqa: SLF001
    ask_id = _enqueue(queue, _questions(secret_id="API_KEY"))
    answers = {"q1": ["yes"], "API_KEY": ["sk-live-do-not-log"]}

    with monkeypatch.context() as broken:
        broken.setattr(attachments_module.AttachmentStore, "put", lambda self, d, m: None)
        outcome = session.respond_ask(ask_id, answers, attachments={"q1": [_image()]})

    assert outcome["ok"] is False and "could not be saved" in outcome["error"]
    assert [e["kind"] for e in store.read_events(queue.session_dir)] == [store.EVENT_QUEUED]
    assert "sk-live-do-not-log" not in json.dumps(store.read_events(queue.session_dir))

    retry = session.respond_ask(ask_id, answers, attachments={"q1": [_image()]})
    assert retry["ok"] is True, retry
    assert "q1" in queue.find(ask_id)["attachments"]


def test_an_image_only_answer_is_a_legal_answer(isolated_config: Path, tmp_path: Path) -> None:
    """Completeness is on KEYS: an empty cell plus a picture is a complete answer."""
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())

    assert session.respond_ask(ask_id, {"q1": []}, attachments={"q1": [_image()]})["ok"] is True

    record = queue.find(ask_id)
    assert record is not None and record["answers"] == {"q1": []}
    assert "q1" in record["attachments"]


# ---------------------------------------------------------------------------
# Revision (D5: text only) and the fold
# ---------------------------------------------------------------------------


def test_a_revision_that_carries_images_is_refused_in_words(
    isolated_config: Path, tmp_path: Path
) -> None:
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())
    assert session.respond_ask(ask_id, {"q1": ["yes"]})["ok"] is True

    refused = session.revise_ask(ask_id, {"q1": ["no"]}, attachments={"q1": [_image()]})

    assert refused == {"ok": False, "error": queue_module.REVISION_IMAGES_REFUSED}
    refused_at_queue = queue.revise(ask_id, {"q1": ["no"]}, attachments={"q1": [_image()]})
    assert refused_at_queue["error"] == queue_module.REVISION_IMAGES_REFUSED
    # Nothing was written by either refusal.
    assert [e["kind"] for e in store.read_events(queue.session_dir)] == [
        store.EVENT_QUEUED,
        store.EVENT_ANSWERED,
    ]


def test_a_text_revision_keeps_the_first_answers_images(
    isolated_config: Path, tmp_path: Path
) -> None:
    """The fold copies attachments from the FIRST answered event only."""
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())
    assert session.respond_ask(ask_id, {"q1": ["yes"]}, attachments={"q1": [_image()]})["ok"]
    first = queue.find(ask_id)["attachments"]

    revised = session.revise_ask(ask_id, {"q1": ["no"]})

    assert revised["ok"] is True and revised["revised"] is True
    record = queue.find(ask_id)
    assert record["answers"] == {"q1": ["no"]}
    assert record["attachments"] == first, "a revision changes the text, never the pictures"
    (row,) = [r for r in queue.projection() if r["ask_id"] == ask_id]
    assert row["attachments"] == first


def test_the_fold_tolerates_a_malformed_attachments_cell() -> None:
    """Reader tolerance: one bad cell must not blank every ask on the surface."""
    queued = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_QUEUED,
        "ask_id": "a-1",
        "at": 1_000_000,
        "expires_at": 1_000_000 + 3_600_000,
        "timeout_s": 3600,
        "questions": [{"id": "q", "question": "?", "options": [], "multi": False}],
    }
    answered = {
        "v": store.EVENT_SCHEMA,
        "kind": store.EVENT_ANSWERED,
        "ask_id": "a-1",
        "at": 1_000_500,
        "answers": {"q": ["x"]},
        "attachments": {"q": [{"no_digest": 1}, "junk"], "r": "not-a-list"},
    }
    (record,) = store.fold([queued, answered], 1_001_000)
    assert "attachments" not in record


# ---------------------------------------------------------------------------
# Delivery: the model's turn and the durable row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_model_receives_the_image_in_the_answer_turn(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE REAL PATH: answer with a picture, run the delivery turn, read the request."""
    session, queue, stream = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())
    image = _image()

    assert session.respond_ask(ask_id, {"q1": ["see screenshot"]}, attachments={"q1": [image]})[
        "ok"
    ]
    await session.reconcile_asks()
    await _drain(session)

    assert stream.requests, "the answer never produced a model turn"
    user = [m for m in stream.requests[-1].messages if m.role == "user"][-1]
    kinds = [type(block).__name__ for block in user.content]
    assert kinds == ["TextContent", "ImageContent"], kinds
    assert user.content[1].data == image.data
    assert "see screenshot (+1 image, shown below)" in user.content[0].text


@pytest.mark.asyncio
async def test_the_durable_row_is_small_and_replays_the_bytes(
    isolated_config: Path, tmp_path: Path
) -> None:
    """Spike B's claim, kept as a test: refs in the row, bytes in the store, replay exact."""
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())
    image = _image()
    assert session.respond_ask(ask_id, {"q1": ["yes"]}, attachments={"q1": [image]})["ok"]
    await session.reconcile_asks()
    await _drain(session)

    rows = [
        json.loads(line)
        for line in session._transcript.path.read_text().splitlines()  # noqa: SLF001
    ]
    (row,) = [r for r in rows if r["id"] == store.response_row_id(ask_id)]
    line = json.dumps(row)
    assert image.data[:64] not in line, "base64 must not stay inline in the transcript row"
    assert len(line) < 2_000
    (block,) = row["payload"]["content"]
    assert "attachment" in block and "data" not in block
    # Refs for the cards ride ``details``; they are refs, not bytes.
    assert row["payload"]["details"]["attachments"]["q1"][0]["attachment"] == block["attachment"]

    # A cold replay of the journal resolves the same bytes back into the render.
    from local_operator.session.transcript import Transcript

    history = Transcript(session._transcript.directory).build_llm_history()  # noqa: SLF001
    rendered = _default_convert_to_llm(history)
    user = [m for m in rendered if m.role == "user" and len(m.content) == 2][-1]
    assert isinstance(user.content[1], ImageContent) and user.content[1].data == image.data


def test_a_text_only_answer_still_renders_a_single_text_block(
    isolated_config: Path, tmp_path: Path
) -> None:
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())
    assert session.respond_ask(ask_id, {"q1": ["yes"]})["ok"]
    record = queue.find(ask_id)

    message = queue._response_message(record)  # noqa: SLF001
    rendered = _default_convert_to_llm([message])

    assert (message.model_extra or {}).get("content") is None
    assert "attachments" not in message.details
    assert len(rendered) == 1 and [type(b) for b in rendered[0].content] == [TextContent]


def test_an_unresolvable_attachment_degrades_one_picture_and_says_so(
    isolated_config: Path, tmp_path: Path
) -> None:
    """A pruned store must not take the turn down, and the text must not lie."""
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _questions())
    assert session.respond_ask(ask_id, {"q1": ["yes"]}, attachments={"q1": [_image()]})["ok"]
    for blob in (isolated_config / "attachments").iterdir():
        blob.unlink()

    message = queue._response_message(queue.find(ask_id))  # noqa: SLF001
    rendered = _default_convert_to_llm([message])

    assert [type(b) for b in rendered[0].content] == [TextContent]
    assert "1 attached image could not be loaded" in message.details["text"]
    assert "shown below" not in message.details["text"]


# ---------------------------------------------------------------------------
# The report text
# ---------------------------------------------------------------------------


def test_the_report_points_at_the_pictures_per_question() -> None:
    record = {
        "status": store.STATUS_ANSWERED,
        "questions": [
            {"id": "a", "question": "First?", "options": [], "multi": False},
            {"id": "b", "question": "Second?", "options": [], "multi": False},
            {"id": "c", "question": "Third?", "options": [], "multi": False},
        ],
        "answers": {"a": ["see this"], "b": [], "c": []},
    }

    text = render.response_text(record, image_counts={"a": 2, "b": 1})

    assert "answer: see this (+2 images, shown below)" in text
    assert "answer: (image attached, shown below)" in text
    assert "answer: (not answered)" in text  # c has neither text nor a picture


def _two_questions(first: str, second: str) -> list[dict[str, Any]]:
    out = []
    for qid in (first, second):
        question = dict(_questions()[0])
        question["id"] = qid
        question["question"] = f"Question {qid}?"
        out.append(question)
    return out


@pytest.mark.parametrize(
    ("first", "second"),
    [("q1", "a0"), ("q2", "q10"), ("b", "a")],
    ids=["q1-then-a0", "q2-then-q10", "b-then-a"],
)
def test_the_images_follow_the_questions_not_the_alphabet(
    isolated_config: Path, tmp_path: Path, first: str, second: str
) -> None:
    """R1 (review round 1): position is the model's only attribution of image to question.

    ``asks.jsonl`` is written ``sort_keys=True``, so the folded attachments map is
    alphabetical by id. The report text lists questions in ASK order and points at
    "the image below" for each, so the blocks must follow ASK order too -- or the
    model reads the second question's picture as the first's. The ids are chosen to
    sort AGAINST the ask order (``q1``/``a0``; ``q2``/``q10`` is the lexicographic
    trap a numeric-looking id falls into). The dict is passed in reverse ask order
    on purpose, so insertion order cannot be what makes this pass.
    """
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _two_questions(first, second))
    pictures = {first: _image(), second: _image()}
    assert pictures[first].data != pictures[second].data

    outcome = session.respond_ask(
        ask_id,
        {first: ["one"], second: ["two"]},
        attachments={second: [pictures[second]], first: [pictures[first]]},
    )
    assert outcome["ok"] is True, outcome

    record = queue.find(ask_id)
    assert record is not None
    assert list(record["attachments"]) == sorted([first, second]), "the fold is alphabetical"
    message = queue._response_message(record)  # noqa: SLF001
    content = (message.model_extra or {})["content"]
    assert [block["data"] for block in content] == [
        pictures[first].data,
        pictures[second].data,
    ]
    text = message.details["text"]
    assert text.index(f"{first} \u2014") < text.index(f"{second} \u2014")


def test_several_images_per_question_keep_question_then_attachment_order(
    isolated_config: Path, tmp_path: Path
) -> None:
    session, queue, _ = _ask_session(tmp_path)
    ask_id = _enqueue(queue, _two_questions("q1", "a0"))
    q1_images = [_image(), _image()]
    a0_images = [_image()]

    assert session.respond_ask(
        ask_id,
        {"q1": ["x"], "a0": ["y"]},
        attachments={"a0": a0_images, "q1": q1_images},
    )["ok"]

    record = queue.find(ask_id)
    assert record is not None
    content = (queue._response_message(record).model_extra or {})["content"]  # noqa: SLF001
    assert [b["data"] for b in content] == [i.data for i in q1_images + a0_images]


def test_the_report_is_unchanged_without_images() -> None:
    from local_operator.tools.builtin import _ask_report

    record = {
        "status": store.STATUS_ANSWERED,
        "questions": [{"id": "a", "question": "First?", "options": [], "multi": False}],
        "answers": {"a": ["yes"]},
    }
    assert render.response_text(record) == _ask_report(
        render._question_models(record["questions"]), {"a": ["yes"]}  # noqa: SLF001
    )


def test_the_injected_image_turn_is_a_composer_shaped_user_message() -> None:
    from local_operator.compaction.cutpoint import RENDERED_INJECTION_KEY
    from local_operator.harness.render import _injected_user_message

    message = _injected_user_message(
        "t", "row-1", [ImageContent(data="QUJD", mime_type="image/png")]
    )

    assert isinstance(message, Message) and message.id == "row-1"
    assert [b.type for b in message.content] == ["text", "image"]
    assert (message.provider_payload or {}).get(RENDERED_INJECTION_KEY) is True
