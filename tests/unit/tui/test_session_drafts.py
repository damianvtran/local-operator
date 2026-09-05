"""Draft cache limits are never permissions to discard unsubmitted input."""

from __future__ import annotations

import stat

import pytest
from textual.widgets.text_area import Selection

from local_operator.harness.types import ImageContent
from local_operator.tui import session_drafts
from local_operator.tui.session_drafts import SessionDraftStore
from local_operator.tui.session_interaction import SessionDraft
from local_operator.tui.widgets.editor import Attachment, PastedText


@pytest.mark.asyncio
async def test_unused_store_does_no_io_and_spill_preserves_exact_draft(monkeypatch):
    monkeypatch.setattr(session_drafts, "DRAFT_RESIDENT_BYTES", 32)
    store = SessionDraftStore()
    assert store._directory is None
    assert await store.get("missing") == SessionDraft()
    assert store._directory is None
    draft = SessionDraft(
        text="編輯草稿 [Paste #3] [Image #4]",
        attachments={
            3: PastedText("long pasted text " * 20, "[Paste #3]"),
            4: Attachment(ImageContent(data="YWJj", mime_type="image/png"), "[Image #4]"),
        },
        selection=Selection((0, 1), (0, 3)),
        shell_mode=True,
        focus_id="@editor",
    )
    await store.put("../../outside", draft)
    assert store.resident_count == 0
    assert store.resident_bytes == 0
    assert await store.get("../../outside") == draft
    path = store._path("../../outside")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert len(path.stem) == 64
    directory = path.parent
    await store.close()
    assert not directory.exists()


@pytest.mark.asyncio
async def test_count_limit_spills_oldest_without_losing_any_draft(monkeypatch):
    monkeypatch.setattr(session_drafts, "DRAFT_RESIDENT_COUNT", 2)
    store = SessionDraftStore()
    try:
        for index in range(10):
            await store.put(str(index), SessionDraft(text=f"draft {index}"))
        assert store.resident_count == 2
        for index in range(10):
            assert (await store.get(str(index))).text == f"draft {index}"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failed_spill_keeps_input_in_memory(monkeypatch):
    monkeypatch.setattr(session_drafts, "DRAFT_RESIDENT_BYTES", 1)
    store = SessionDraftStore()

    def full_disk(*args):
        raise OSError("no space")

    monkeypatch.setattr(store, "_write", full_disk)
    try:
        with pytest.raises(OSError, match="no space"):
            await store.put("source", SessionDraft(text="never discard this input"))
        assert (await store.get("source")).text == "never discard this input"
    finally:
        await store.close()
