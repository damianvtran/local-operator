"""Dark, send-only generic files: exact bytes and one immutable actor tuple.

This is NOT recipient file admission. Names/descriptors/originals remain outside
agent history and model context. The hub owns classification, quotas and binding;
its existing `other` bind/replay policy remains an activation blocker.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import agent_attachments
from .client import FILE_MAX_BYTES, BridgeRefusal, agent_files_capability

_LOG = logging.getLogger(__name__)
MAX_FILES = 10
UNCERTAIN_MESSAGE = "delivery is uncertain; do not restage or resend with a new reference"


@dataclass(frozen=True)
class PreparedFile:
    """Transient, bounded upload buffer; never a receipt or retry journal."""

    name: str
    _bytes: bytes = field(repr=False)

    @property
    def size(self) -> int:
        return len(self._bytes)

    def payload(self) -> bytes:
        return self._bytes


def _owner(runtime: Any) -> tuple[Any, ...]:
    return tuple(
        getattr(runtime.owner, key, None)
        for key in (
            "account_id",
            "tenant_id",
            "endpoint_id",
            "credential_id",
            "enrollment_generation",
            "canonical_base_url",
        )
    )


class FileBatch:
    """Call-local staging and uncertainty, never a session/global payload cache."""

    def __init__(self, runtime: Any, endpoint_id: str) -> None:
        self.runtime = runtime
        self.client = runtime.client
        self.origin = getattr(self.client, "base_url", None)
        self.owner = _owner(runtime)
        self.account_id = self.owner[0]
        self.tenant_id = runtime.tenant_id
        self.endpoint_id = endpoint_id
        self.conversation_id: str | None = None
        self.staged: list[str] = []
        self.images: list[str] = []
        self.uncertain = False
        if (
            not self.account_id
            or not self.tenant_id
            or not endpoint_id
            or self.owner[1] != self.tenant_id
            or self.owner[2] != endpoint_id
            or not isinstance(self.origin, str)
            or not self.origin
            or self.owner[5] != self.origin
        ):
            raise agent_attachments.AttachmentRefused(
                "owner_changed", "file-send authority is unavailable"
            )

    def bind_conversation(self, conversation_id: str) -> None:
        if self.conversation_id is None:
            self.conversation_id = conversation_id
        elif self.conversation_id != conversation_id:
            raise agent_attachments.AttachmentRefused(
                "owner_changed", "file-send conversation changed"
            )

    def check(self) -> None:
        # Re-run at the physical HTTP seam, after OAuth/refresh waits, rather
        # than trusting the owner that happened to be present at tool entry.
        from .host import outbound_tool_fence

        if (
            self.runtime.client is not self.client
            or getattr(self.client, "base_url", None) != self.origin
            or _owner(self.runtime) != self.owner
            or self.runtime.tenant_id != self.tenant_id
            or outbound_tool_fence(self.runtime) is not None
        ):
            raise agent_attachments.AttachmentRefused(
                "owner_changed", "file-send authority changed"
            )

    async def prepare(
        self, paths: Sequence[str], images: Sequence[Any]
    ) -> tuple[PreparedFile, ...]:
        self.check()
        try:
            payload = await self.client.status(expected_account_id=self.account_id)
        except Exception:
            payload = None
        self.check()
        capability = agent_files_capability(payload)
        if capability is None:
            raise agent_attachments.AttachmentRefused(
                "files_unavailable", "this hub does not accept agent files; nothing was sent"
            )
        cap = min(MAX_FILES, capability.get("max_items", MAX_FILES))
        if len(paths) + len(images) > cap or len(images) > capability.get(
            "max_images", agent_attachments.ATTACHMENT_MAX_COUNT
        ):
            raise agent_attachments.AttachmentRefused(
                "attachment_invalid", "too many combined attachments"
            )
        prepared = []
        seen = {hashlib.sha256(image.payload()).digest() for image in images}
        for index, path in enumerate(paths):
            try:
                data, name = await asyncio.to_thread(
                    agent_attachments.read_path_bytes,
                    path,
                    index,
                    str(self.runtime.session_id),
                    label_prefix="files",
                    max_bytes=FILE_MAX_BYTES,
                    anchor_parents=True,
                )
            except agent_attachments.AttachmentRefused:
                raise
            except Exception:
                raise agent_attachments.AttachmentRefused(
                    "attachment_invalid", "a file could not be prepared"
                ) from None
            digest = hashlib.sha256(data).digest()
            if digest in seen:
                raise agent_attachments.AttachmentRefused(
                    "attachment_invalid", "duplicate attachment bytes"
                )
            seen.add(digest)
            prepared.append(PreparedFile(name, data))
        self.check()
        return tuple(prepared)

    async def stage(self, conversation_id: str, prepared: Sequence[PreparedFile]) -> None:
        self.bind_conversation(conversation_id)
        try:
            for item in prepared:
                self.check()
                result = await self.client.upload_attachment(
                    conversation_id,
                    item.payload(),
                    mode="file",
                    name=item.name,
                    tenant_id=self.tenant_id,
                    actor_endpoint_id=self.endpoint_id,
                    expected_account_id=self.account_id,
                    pre_http_guard=self.check,
                )
                row = result.get("file") if isinstance(result, dict) else None
                identifier = row.get("id") if isinstance(row, dict) else None
                if (
                    not isinstance(identifier, str)
                    or agent_attachments._ATTACHMENT_ID.fullmatch(identifier) is None
                ):
                    raise agent_attachments.AttachmentRefused(
                        "attachment_failed", "the file upload did not confirm"
                    )
                if identifier in self.staged or identifier in self.images:
                    raise agent_attachments.AttachmentRefused(
                        "attachment_invalid", "duplicate staged file"
                    )
                self.staged.append(identifier)
        except BaseException:
            await self.cleanup(conversation_id)
            raise

    async def send(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        """Replay only this call's EXACT ref/IDs; never delete an uncertain send.

        The guard marks the last boundary before HTTP. Failures before it are
        local/definitive; after it an unconfirmed response may hide a commit.
        There is one in-call replay, no re-stage/new-ref and no durable payload.
        """
        self.check()
        self.bind_conversation(conversation_id)
        # Legacy channel sends rely on their dispatch fence rather than an
        # explicit account kwarg. Files additionally bind that same owner into
        # OAuth selection; absence never means permission to select the human.
        kwargs.setdefault("expected_account_id", self.account_id)
        for key, value in (
            ("tenant_id", self.tenant_id),
            ("actor_endpoint_id", self.endpoint_id),
            ("expected_account_id", self.account_id),
        ):
            if kwargs.get(key) != value:
                raise agent_attachments.AttachmentRefused(
                    "owner_changed", "file-send authority changed"
                )
        original_guard = kwargs.get("pre_http_guard")
        had_unknown = False
        for attempt in range(2):
            entered = False

            def guard() -> None:
                nonlocal entered
                self.check()
                if original_guard is not None:
                    original_guard()
                entered = True
                self.uncertain = True

            try:
                result = await self.client.conversation_send(
                    conversation_id,
                    **{**kwargs, "files": list(self.staged), "pre_http_guard": guard},
                )
                row = result.get("message", result) if isinstance(result, dict) else None
                if not isinstance(row, dict) or not str(row.get("id", "")).startswith("bcmsg_"):
                    raise ValueError("unconfirmed message receipt")
                self.uncertain = False
                return result
            except Exception as failure:
                definitive = (
                    isinstance(failure, BridgeRefusal) and 400 <= (failure.http_status or 0) < 500
                )
                if definitive:
                    self.uncertain = had_unknown
                if not entered or definitive:
                    raise
                had_unknown = True
                self.uncertain = True
                if attempt:
                    raise agent_attachments.AttachmentRefused(
                        "delivery_uncertain",
                        UNCERTAIN_MESSAGE,
                    ) from None
        raise AssertionError("unreachable")

    async def cleanup(self, conversation_id: str) -> None:
        if self.uncertain:
            return
        self.bind_conversation(conversation_id)

        # Cleanup keeps the original tuple even if the live endpoint is paused;
        # it grants no new send and bound rows remain undeletable on the hub.
        async def delete_known() -> None:
            failures = 0
            for identifier in (*self.staged, *self.images):
                try:
                    await self.client.delete_attachment(
                        conversation_id,
                        identifier,
                        tenant_id=self.tenant_id,
                        actor_endpoint_id=self.endpoint_id,
                        expected_account_id=self.account_id,
                    )
                except Exception:
                    failures += 1
            if failures:
                _LOG.warning("Agent file staged cleanup incomplete: %d", failures)

        task = asyncio.ensure_future(delete_known())
        # Reuse the existing strong in-flight cleanup ownership: cancellation
        # must not garbage-collect a shielded delete. This holds IDs, never bytes.
        agent_attachments._CLEANUPS.add(task)
        task.add_done_callback(agent_attachments._CLEANUPS.discard)
        await asyncio.shield(task)
