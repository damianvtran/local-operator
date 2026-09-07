"""Bound draft RAM without ever evicting the user's unsubmitted input.

Only this viewer's overflow goes to a private temporary directory. No pickle,
shared session file, transcript write or acknowledgement is involved. The store
is lazy: a hidden, unused sidebar performs no filesystem work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from local_operator.harness.types import ImageContent
from local_operator.tui.session_interaction import SessionDraft

if TYPE_CHECKING:
    from local_operator.tui.widgets.transcript import NoticeKind

DRAFT_RESIDENT_COUNT = 16
DRAFT_RESIDENT_BYTES = 1024 * 1024


class SessionDraftStore:
    def __init__(self) -> None:
        self._memory: OrderedDict[str, SessionDraft] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def resident_bytes(self) -> int:
        return sum(self._sizes.values())

    @property
    def resident_count(self) -> int:
        return len(self._memory)

    @staticmethod
    def _size(draft: SessionDraft) -> int:
        from local_operator.tui.widgets.editor import Attachment, PastedText

        size = len(draft.text.encode("utf-8")) + len((draft.aside_main_text or "").encode("utf-8"))
        for attachment in (*draft.attachments.values(), *draft.aside_main_images.values()):
            if isinstance(attachment, Attachment):
                size += len(attachment.image.data)
            elif isinstance(attachment, PastedText):
                size += len(attachment.text.encode("utf-8"))
        if draft.aside is not None:
            for turn in draft.aside.turns:
                size += len((turn.question + turn.answer + turn.error).encode("utf-8"))
        size += sum(SessionDraftStore._size(item) for item in draft.recoveries)
        size += sum(len(text.encode("utf-8")) for text, _kind in draft.notices)
        return size

    async def put(self, session_id: str, draft: SessionDraft) -> None:
        async with self._lock:
            if self._closed:
                return
            self._memory[session_id] = draft
            self._memory.move_to_end(session_id)
            self._sizes[session_id] = self._size(draft)
            if self._directory is not None:
                # A dismissed aside or edited draft must not leave its older
                # plaintext snapshot behind merely because the new one fits RAM.
                await asyncio.to_thread(self._path(session_id).unlink, missing_ok=True)
            while (
                self.resident_count > DRAFT_RESIDENT_COUNT
                or self.resident_bytes > DRAFT_RESIDENT_BYTES
            ):
                key = next(iter(self._memory))
                # Remove only AFTER the write succeeds. A full disk can exceed
                # the desired RAM budget but may never discard accepted input.
                write = asyncio.create_task(asyncio.to_thread(self._write, key, self._memory[key]))
                try:
                    await asyncio.shield(write)
                except asyncio.CancelledError:
                    # A worker cancellation cannot interrupt a filesystem write.
                    # Join it before releasing the lock so close cannot remove
                    # its private directory while the writer still owns it.
                    await write
                    raise
                del self._memory[key]
                self._sizes.pop(key)

    async def get(self, session_id: str) -> SessionDraft:
        async with self._lock:
            draft = self._memory.get(session_id)
            if draft is not None:
                self._memory.move_to_end(session_id)
                return draft
            if self._directory is None:
                return SessionDraft()
            return await asyncio.to_thread(self._read, session_id)

    def _path(self, session_id: str) -> Path:
        if self._directory is None:
            self._directory = tempfile.TemporaryDirectory(prefix="lop-viewer-drafts-")
            os.chmod(self._directory.name, 0o700)
        name = hashlib.sha256(session_id.encode()).hexdigest()
        return Path(self._directory.name) / f"{name}.json"

    @staticmethod
    def _encode_attachments(values: dict[int, Any]) -> dict[str, Any]:
        from local_operator.tui.widgets.editor import Attachment, PastedText

        attachments: dict[str, Any] = {}
        for index, attachment in values.items():
            if isinstance(attachment, Attachment):
                attachments[str(index)] = {
                    "kind": "image",
                    "image": attachment.image.model_dump(),
                    "marker": attachment.marker,
                }
            elif isinstance(attachment, PastedText):
                attachments[str(index)] = {
                    "kind": "text",
                    "text": attachment.text,
                    "marker": attachment.marker,
                }
            else:
                raise ValueError("unsupported draft attachment type")
        return attachments

    def _encode_draft(self, draft: SessionDraft) -> dict[str, Any]:
        from dataclasses import asdict

        selection = draft.selection
        payload = {
            "text": draft.text,
            "attachments": self._encode_attachments(draft.attachments),
            "selection": (
                {"start": selection.start, "end": selection.end} if selection is not None else None
            ),
            "shell_mode": draft.shell_mode,
            "focus_id": draft.focus_id,
            "scroll_anchor_id": draft.scroll_anchor_id,
            "scroll_anchor_part": draft.scroll_anchor_part,
            "scroll_offset": draft.scroll_offset,
            "following_tail": draft.following_tail,
            "aside": asdict(draft.aside) if draft.aside is not None else None,
            "aside_main_text": draft.aside_main_text,
            "aside_main_images": self._encode_attachments(draft.aside_main_images),
            "aside_main_shell_mode": draft.aside_main_shell_mode,
            "approve_all": draft.approve_all,
            "recoveries": [self._encode_draft(item) for item in draft.recoveries],
            "notices": draft.notices,
        }
        return payload

    def _write(self, session_id: str, draft: SessionDraft) -> None:
        payload = self._encode_draft(draft)
        target = self._path(session_id)
        fd, temporary = tempfile.mkstemp(prefix=".draft-", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _decode_attachments(values: dict[str, Any]) -> dict[int, Any]:
        from local_operator.tui.widgets.editor import Attachment, PastedText

        attachments = {}
        for index, item in values.items():
            attachments[int(index)] = (
                Attachment(ImageContent.model_validate(item["image"]), item["marker"])
                if item["kind"] == "image"
                else PastedText(item["text"], item["marker"])
            )
        return attachments

    def _decode_draft(self, data: dict[str, Any]) -> SessionDraft:
        from textual.widgets.text_area import Selection

        from local_operator.tui.widgets.aside_panel import AsideSnapshot, AsideTurn

        if not isinstance(data.get("text"), str):
            raise ValueError("Saved draft text is invalid")
        for field in ("shell_mode", "following_tail", "aside_main_shell_mode"):
            if not isinstance(data.get(field), bool):
                raise ValueError("Saved draft mode is invalid")
        if data.get("approve_all") is not None and not isinstance(data["approve_all"], bool):
            raise ValueError("Saved approval mode is invalid")
        selection = data["selection"]
        notices = []
        for item in data.get("notices", []):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(part, str) for part in item)
                or item[1] not in {"info", "note", "success", "warning", "error"}
            ):
                raise ValueError("Saved draft notice is invalid")
            notices.append((item[0], cast("NoticeKind", item[1])))
        return SessionDraft(
            text=data["text"],
            attachments=self._decode_attachments(data["attachments"]),
            selection=(
                Selection(tuple(selection["start"]), tuple(selection["end"]))
                if selection is not None
                else None
            ),
            shell_mode=data["shell_mode"],
            focus_id=data["focus_id"],
            scroll_anchor_id=data["scroll_anchor_id"],
            scroll_anchor_part=data["scroll_anchor_part"],
            scroll_offset=data["scroll_offset"],
            following_tail=data["following_tail"],
            aside=(
                AsideSnapshot(
                    **{
                        **data["aside"],
                        "turns": [AsideTurn(**turn) for turn in data["aside"]["turns"]],
                    }
                )
                if data["aside"] is not None
                else None
            ),
            aside_main_text=data["aside_main_text"],
            aside_main_images=self._decode_attachments(data["aside_main_images"]),
            aside_main_shell_mode=data["aside_main_shell_mode"],
            approve_all=data.get("approve_all"),
            recoveries=[self._decode_draft(item) for item in data.get("recoveries", [])],
            notices=notices,
        )

    def _read(self, session_id: str) -> SessionDraft:
        path = self._path(session_id)
        if not path.exists():
            return SessionDraft()
        return self._decode_draft(json.loads(path.read_text(encoding="utf-8")))

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            if self._directory is not None:
                await asyncio.to_thread(self._directory.cleanup)
                self._directory = None
            self._memory.clear()
            self._sizes.clear()
