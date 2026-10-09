"""Saved display must be bounded by the rows it needs, without inventing a cut.

The reader used to take a fixed 256 KiB tail window and refuse the whole preview
when that window cut a row, missed a compaction boundary, or ended on a live
append. On this store those are the COMMON shapes (checkpoint rows reach
0.9 MB, compaction summaries average 568 KB), so the reader painted a blank
pane on exactly the large sessions it exists for. The tests below pin the
replacement contract: the walk is bounded by ROWS, an oversized bookkeeping row
is stepped over, a torn append is one dropped row, and anything the reader
cannot prove is reported through ``partial`` rather than by returning nothing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness.types import Message
from local_operator.session.history_window import DISPLAY_HISTORY_MESSAGES
from local_operator.session.saved_preview import PREVIEW_SCAN_BYTES, read_saved_preview
from local_operator.session.transcript import (
    Transcript,
    TranscriptEntry,
    encode_message_payload,
)


def message(identifier: str, text: str) -> TranscriptEntry:
    return TranscriptEntry(identifier, 0, "message", encode_message_payload(Message.user(text)))


def journal(path: Path, entries: list[TranscriptEntry]) -> None:
    (path / "transcript.jsonl").write_text(
        "".join(entry.to_json() + "\n" for entry in entries), encoding="utf-8"
    )


def test_missing_journal_is_not_a_complete_empty_saved_conversation(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_saved_preview(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        read_saved_preview(tmp_path)
    assert not (tmp_path / "missing").exists()


def test_existing_empty_journal_is_a_valid_empty_preview(tmp_path):
    (tmp_path / "transcript.jsonl").touch()
    result = read_saved_preview(tmp_path)
    assert result.messages == [] and not result.partial


@pytest.mark.asyncio
async def test_unmaterialized_empty_view_requires_a_discoverable_owner(tmp_path, monkeypatch):
    from local_operator.session.attached import AttachedSession

    record = SimpleNamespace(cwd="/synthetic-owner")
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record", lambda *args: (record, 12345)
    )

    async def no_takeover():
        raise AssertionError("a preview must not become an execution owner")

    remote = await AttachedSession.saved_preview(
        "unstarted", config_dir=tmp_path, cwd="/other", takeover_factory=no_takeover
    )
    try:
        assert remote.is_cold
        assert remote.frontend_state.cwd == "/synthetic-owner"
        assert remote.display_history_window() == []
        assert not (tmp_path / "sessions" / "unstarted").exists()
    finally:
        await remote.dispose()


@pytest.mark.asyncio
async def test_absent_owner_and_journal_refuse_before_building_facade(tmp_path, monkeypatch):
    from local_operator.session.attached import AttachedSession

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record", lambda *args: (None, None)
    )

    async def no_takeover():
        raise AssertionError("a missing target must not launch an owner")

    with pytest.raises(FileNotFoundError, match="no longer available"):
        await AttachedSession.saved_preview(
            "deleted", config_dir=tmp_path, cwd="/other", takeover_factory=no_takeover
        )
    assert not (tmp_path / "sessions" / "deleted").exists()


def test_preview_replays_correct_session_and_prunes(tmp_path):
    journal(
        tmp_path,
        [
            message("old", "original tool output"),
            message("new", "Useful saved answer"),
            TranscriptEntry("prune", 0, "prune", {"target": "old", "notice": "Removed"}),
        ],
    )
    result = read_saved_preview(tmp_path)
    assert not result.partial
    assert result.messages[-1].text == "Useful saved answer"
    assert "original tool output" not in result.messages[0].text


def test_preview_walks_by_rows_not_a_byte_window(tmp_path, monkeypatch):
    """The tail row is reached across an oversized row, in bounded reads.

    A row far larger than any window this reader could pick is the shape that
    used to fail: the window landed inside it, found no newline, and returned
    nothing. The assertion on the read sizes is the other half — every read is
    a chunk of the shared backward walker, never an unbounded ``read()`` of a
    file a writer can grow while this call runs.
    """
    journal(
        tmp_path,
        [message("huge", "x" * (PREVIEW_SCAN_BYTES // 8)), message("tail", "Useful tail")],
    )
    original = Path.open
    reads = []

    class BoundedRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def seek(self, *args):
            return self.handle.seek(*args)

        def tell(self):
            return self.handle.tell()

        def read(self, count=-1):
            reads.append(count)
            assert 0 <= count <= PREVIEW_SCAN_BYTES
            return self.handle.read(count)

    monkeypatch.setattr(Path, "open", lambda path, *a, **kw: BoundedRead(original(path, *a, **kw)))
    result = read_saved_preview(tmp_path)
    assert [row.text for row in result.messages] == ["x" * (PREVIEW_SCAN_BYTES // 8), "Useful tail"]
    assert reads, "precondition: the reader must read the file through this handle"
    assert max(reads) < PREVIEW_SCAN_BYTES


@pytest.mark.parametrize("suffix", [b'{"type":"prune"', b"not json\n"])
def test_a_torn_or_malformed_tail_row_is_dropped_not_the_preview(tmp_path, suffix):
    """An append in progress costs a row, never the pane.

    Both suffixes are rows no reader can use: the first is a live append caught
    mid-write, the second a complete but unparseable line. Neither is durable
    evidence about the row above it — the writer fsyncs whole lines and
    truncates on failure, and every other reader (``read_replay_suffix``, the
    resident ``Transcript``) drops exactly these — so the preview shows the
    conversation and reports the incompleteness through ``partial``.

    The previous revision returned an EMPTY preview here, which is the
    blank-pane failure one append wide: a running session is mid-append often
    enough that this was reachable by hovering a sidebar.
    """
    journal(tmp_path, [message("saved", "May have been retracted")])
    with (tmp_path / "transcript.jsonl").open("ab") as handle:
        handle.write(suffix)
    result = read_saved_preview(tmp_path)
    assert [row.text for row in result.messages] == ["May have been retracted"]
    # A torn row means the file's newest row is unreadable, so the preview is an
    # excerpt; a complete-but-malformed row is fully accounted for and is not.
    assert result.partial is suffix.startswith(b'{"type"')


def test_oversized_bookkeeping_rows_do_not_blank_the_pane(tmp_path):
    """AUDIT F9, reproduced: the real-shape blank pane.

    The journal here is S7's shape — the row sizes the audit measured on the
    operator's store: a ~0.9 MB checkpoint row and ~600 KB compaction summaries
    as the newest rows, with the conversation underneath them. A 256 KiB tail
    lands inside the newest checkpoint (no newline at all in the window) and the
    old reader returned ``[]``, so the sidebar painted nothing on exactly the
    sessions most worth previewing. Row-wise walking steps over them.
    """
    big = "p" * 600_000
    journal(
        tmp_path,
        [
            message("m0", "earlier answer"),
            TranscriptEntry("c0", 0, "compaction", {"summary": big, "first_kept_entry_id": "m0"}),
            message("m1", "the newest thing the user said"),
            TranscriptEntry(
                "k0",
                0,
                "custom",
                {"custom_type": "frontend_state_checkpoint_v1", "details": {"state": big}},
            ),
        ],
    )
    result = read_saved_preview(tmp_path)
    # ``CustomMessage`` (the compaction marker) has no ``text``; the rows this
    # preview is about do.
    texts = [row.text for row in result.messages if isinstance(row, Message)]
    assert "the newest thing the user said" in texts
    # The compaction marker is in the window and the kept window is resolved, so
    # the preview is the journal's own replay of it, not an excerpt.
    assert not result.partial


def test_an_unresolved_compaction_cut_matches_the_canonical_replay(tmp_path):
    """A compaction naming a row that is gone is NOT this reader's to invent around.

    The row it names may have been dropped as malformed, or the journal may have
    been written by a converter that minted ids elsewhere. What this reader owes
    is the SAME answer every other reader gives for that journal rather than a
    private fallback: the canonical replay logs and replays everything it has
    (``context_cut_index`` returning 0), and so does this preview, on the one
    shared implementation. It is pinned against a real ``Transcript`` rather
    than against a restatement of the rule, so a future divergence fails here.
    """
    journal(
        tmp_path,
        [
            message("saved", "Do not resurrect"),
            TranscriptEntry("compact", 0, "compaction", {"first_kept_entry_id": "missing"}),
        ],
    )
    result = read_saved_preview(tmp_path)
    canonical = Transcript(tmp_path).build_llm_history()
    assert [getattr(row, "text", "") for row in result.messages] == [
        getattr(row, "text", "") for row in canonical
    ]


def test_an_oversized_final_message_row_is_shown_whole(tmp_path):
    """A big MESSAGE row is content, not a reason to withdraw the preview.

    Distinct from the bookkeeping case above: this row is a turn the reader can
    display, at 512 KB of text. A row ceiling that refused it would be a size
    policy hiding a real conversation, and the widget renders one message row
    whatever its length; the walk's own ``PREVIEW_SCAN_BYTES`` ceiling is where
    cost is bounded instead.
    """
    big = "x" * (512 * 1024)
    journal(tmp_path, [message("huge", big)])
    result = read_saved_preview(tmp_path)
    assert [row.text for row in result.messages] == [big]
    assert not result.partial


def test_the_walk_stops_at_the_display_budget_not_the_file_start(tmp_path):
    """The budget counts MESSAGES the reader can show, and reports the excerpt.

    Row counting is the point of the change: 300 display rows on disk must not
    become 300 journal rows read when 120 of them are what a pane can use, and
    the ``partial`` flag is what tells the sidebar it is showing an excerpt.
    """
    journal(tmp_path, [message(f"m{index}", f"turn {index}") for index in range(300)])
    result = read_saved_preview(tmp_path)
    assert len(result.messages) == DISPLAY_HISTORY_MESSAGES
    assert result.partial
    # The newest rows are the ones kept — a preview is for the tail.
    assert result.messages[-1].text == "turn 299"
    assert result.messages[0].text == f"turn {300 - DISPLAY_HISTORY_MESSAGES}"


def test_a_scan_ceiling_hit_is_an_honest_excerpt_not_a_silent_blank(tmp_path, monkeypatch):
    """The ceiling's answer is ``partial``, which is what the TUI paints.

    ``PREVIEW_SCAN_BYTES`` is a bound this reader imposes on itself, so the one
    case it can still refuse is a journal whose display rows all sit deeper than
    the ceiling. Returning nothing there is honest ONLY because it is flagged:
    the sidebar's lease turns ``partial`` with no blocks into the explicit
    "Saved preview unavailable. Connect to load the conversation." notice rather
    than an empty pane, which is the difference this whole change turns on.
    """
    monkeypatch.setattr("local_operator.session.saved_preview.PREVIEW_SCAN_BYTES", 1024)
    # Two 600 KB bookkeeping rows newest, so the first chunk the walker can
    # yield holds NOTHING displayable — the one shape where the ceiling binds
    # before any display row is in hand.
    pad = "z" * 600_000
    journal(
        tmp_path,
        [
            message("m0", "the turn behind them"),
            TranscriptEntry(
                "k0",
                0,
                "custom",
                {"custom_type": "frontend_state_checkpoint_v1", "details": {"state": pad}},
            ),
            TranscriptEntry(
                "k1",
                0,
                "custom",
                {"custom_type": "frontend_state_checkpoint_v1", "details": {"state": pad}},
            ),
        ],
    )
    result = read_saved_preview(tmp_path)
    assert result.messages == []
    assert result.partial, "an empty preview must say it is an excerpt, or the pane is silent"


async def _seeded_image_journal(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    """A journal in its OWNER config dir whose image row is externalized.

    Returns ``(session_dir, pasted_base64)``. The owning config dir is the env
    config dir *at write time*, which is what ``Transcript`` externalizes
    through; the caller relocates the env afterwards to prove the reader uses
    the journal's own store rather than its environment.
    """
    import base64

    from local_operator.harness.types import ImageContent, Message
    from local_operator.session.transcript import Transcript

    owner_cfg = tmp_path / "owner"
    session_dir = owner_cfg / "sessions" / "preview01"
    session_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(owner_cfg))

    # >1 KB base64 so the row references the store instead of carrying the
    # bytes inline (an inline row resolves with no store at all).
    pasted = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096).decode("ascii")
    image = ImageContent(data=pasted, mime_type="image/png")
    await Transcript(session_dir).append_message(
        Message.user("previewed [Image #1]", images=[image])
    )
    return session_dir, pasted


@pytest.mark.asyncio
async def test_preview_hydrates_an_externalized_image_from_the_owning_store(tmp_path, monkeypatch):
    """The sidebar's default surface must not paint a false "unavailable".

    ``read_saved_preview`` is the sidebar's cold excerpt seam
    (``AttachedSession.saved_preview``), so an unhydrated reference there paints
    the unavailable receipt for a picture the session actually has.
    """
    from local_operator.harness.types import ImageContent
    from local_operator.session.attachments import (
        AttachmentStore,
        store_for_transcript_dir,
    )

    session_dir, pasted = await _seeded_image_journal(tmp_path, monkeypatch)

    # The reader's env config dir must NOT be the answer: point it at an empty
    # directory, so a rewire to the env-default store fails this test.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "reader-env"))
    assert store_for_transcript_dir(session_dir).root == tmp_path / "owner" / "attachments"
    assert store_for_transcript_dir(session_dir).root != AttachmentStore().root

    blocks = [
        block
        for message in read_saved_preview(session_dir).messages
        for block in (message.content or [])
        if isinstance(block, ImageContent)
    ]
    assert len(blocks) == 1
    assert blocks[0].data == pasted
    assert len(blocks[0].data) > 0


@pytest.mark.asyncio
async def test_preview_degrades_to_the_receipt_when_the_store_blob_is_gone(tmp_path, monkeypatch):
    """A genuinely missing blob still degrades — never raises, never blanks.

    The hydration above must not be able to turn a pruned store into a failed
    preview: the row keeps its prompt text and an empty image the widget renders
    as the deliberate unavailable receipt.
    """
    from local_operator.harness.types import ImageContent

    session_dir, _ = await _seeded_image_journal(tmp_path, monkeypatch)
    blobs = list((tmp_path / "owner" / "attachments").glob("*.bin"))
    assert blobs, "precondition: the write path must have stored the bytes"
    for blob in blobs:
        blob.unlink()

    preview = read_saved_preview(session_dir)
    blocks = [
        block
        for message in preview.messages
        for block in (message.content or [])
        if isinstance(block, ImageContent)
    ]
    assert len(blocks) == 1
    assert blocks[0].data == ""
    assert any("previewed [Image #1]" in message.text for message in preview.messages)


@pytest.mark.asyncio
async def test_preview_degrades_when_the_store_sidecar_is_not_an_object(tmp_path, monkeypatch):
    """A JSON sidecar that is not an object must degrade, not raise.

    ``AttachmentStore.get`` documents "Callers must treat that as ordinary and
    degrade to a placeholder, never raise", and this reader reaches the store
    only because it now hydrates. A sidecar of ``[]``/``null`` parses fine but
    is not an object, so an unguarded ``meta.get`` raised AttributeError out
    through ``read_saved_preview`` and into the sidebar's lease.
    """
    from local_operator.harness.types import ImageContent

    session_dir, _ = await _seeded_image_journal(tmp_path, monkeypatch)
    sidecars = list((tmp_path / "owner" / "attachments").glob("*.json"))
    assert len(sidecars) == 1, "precondition: the write path stored one sidecar"
    sidecars[0].write_text("null", encoding="utf-8")

    preview = read_saved_preview(session_dir)
    blocks = [
        block
        for message in preview.messages
        for block in (message.content or [])
        if isinstance(block, ImageContent)
    ]
    assert len(blocks) == 1
    assert blocks[0].data == ""
    assert any("previewed [Image #1]" in message.text for message in preview.messages)


def test_the_row_cap_bounds_a_pathological_journal_and_says_so(tmp_path, monkeypatch):
    """F6: the byte ceiling does not bound the DECODES, so a row cap does.

    Review round 1 measured 37-106 ms of this reader on a 132.5 MB journal whose
    tail is a long run of bookkeeping rows, against 1.2-2.1 ms before the reader
    existed. Both ceilings are checked at CHUNK boundaries, so the bound is "the
    cap plus the rows of one chunk" — the fixture below is deliberately more than
    one chunk (2.5 MB of 4 KB bookkeeping rows) with the only message at the very
    top, so a walk that respected the cap cannot reach it.
    """
    monkeypatch.setattr("local_operator.session.saved_preview.PREVIEW_SCAN_ROWS", 200)
    pad = "z" * 4096
    entries = [message("m0", "an older turn")]
    entries.extend(
        TranscriptEntry(
            f"k{index}",
            0,
            "custom",
            {"custom_type": "frontend_state_checkpoint_v1", "details": {"state": pad}},
        )
        for index in range(600)
    )
    journal(tmp_path, entries)
    assert (tmp_path / "transcript.jsonl").stat().st_size > (2 << 20)

    result = read_saved_preview(tmp_path)

    # Nothing displayable within the cap, so the answer is the honest excerpt —
    # flagged, which is what the sidebar renders as "connect to load", never a
    # silent empty pane.
    assert result.messages == []
    assert result.partial is True
