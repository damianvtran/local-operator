"""Project runtime events into prose and an optional validated widget sidecar."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    HistoryDeltaEvent,
    Message,
    MessageUpdateEvent,
)
from local_operator.session.inline_ui_text import (
    INLINE_UI_V1,
    InlineUiProjection,
    project_inline_ui,
)


def _project(text: str, *, streaming: bool, sidecars: bool) -> InlineUiProjection:
    return project_inline_ui(
        text,
        streaming=streaming,
        source_version=1,
        capabilities=frozenset({INLINE_UI_V1}) if sidecars else frozenset(),
    )


_MAX_SIDECAR_BYTES = 48 * 1024


def _attach(
    data: dict[str, Any],
    message_id: str,
    projection: InlineUiProjection,
    *,
    include_text: bool = False,
) -> None:
    envelope = projection.get("inline_ui")
    if not envelope:
        return
    # Agent-end can repeat a whole run. Bound the optional augmentation, not
    # the prose frame. Preserve a prefix of complete widgets and their offsets.
    previous = data.get("inline_ui", {}).get("messages", {})
    widgets = []
    # Live phone rows use a different text path. The history caller already
    # has the exact basis in the projected row and must not duplicate it.
    basis = {"text": projection["text"]} if include_text else {}
    for widget in envelope["widgets"]:
        candidate = {
            "version": 1,
            "messages": {
                **previous,
                message_id: {
                    "version": 1,
                    **basis,
                    "widgets": [*widgets, widget],
                },
            },
        }
        if len(json.dumps(candidate, separators=(",", ":"))) > _MAX_SIDECAR_BYTES:
            break
        widgets.append(widget)
        data["inline_ui"] = candidate


def _marked(message: object) -> bool:
    return (
        isinstance(message, Message)
        and message.role == "assistant"
        and message._inline_ui_source_version == 1
    )


def is_text_block(block: dict[str, Any]) -> bool:
    # Journal encodings omit defaults, including TextContent's discriminator.
    return block.get("type", "text") == "text" and isinstance(block.get("text"), str)


def replace_text_content(data: dict[str, Any], text: str) -> None:
    # Strip from the dump, not from the stored or model-visible message. Join
    # text blocks first so a fence split across blocks cannot escape projection.
    content = data.get("content", [])
    if any(is_text_block(block) for block in content):
        out = []
        first_text = True
        for block in content:
            if not is_text_block(block):
                out.append(block)
            elif first_text:
                if text:
                    out.append({**block, "text": text})
                first_text = False
        data["content"] = out


class InlineUiEvents:
    """Project append-only live updates and authoritative final messages.

    The provider's message object doesn't hold accumulated text on every
    update. Track the active marked message locally rather than forwarding its
    raw provider delta. Turns have only one active assistant text stream.
    """

    def __init__(self, *, sidecars: bool = False) -> None:
        self._sidecars = sidecars
        self._message_id = ""
        self._source = ""
        self._visible = ""
        self._published: set[str] = set()

    def dump(self, event: AgentEvent[Any], *, reconnect_seed: bool = False) -> dict[str, Any]:
        data = event.model_dump(mode="json")
        message = getattr(event, "message", None)
        if isinstance(message, Message) and _marked(message):
            streaming = isinstance(event, MessageUpdateEvent)
            projection = _project(message.text, streaming=streaming, sidecars=self._sidecars)
            replace_text_content(data["message"], projection["text"])
            if isinstance(event, MessageUpdateEvent):
                if self._message_id != message.id:
                    self._reset()
                    self._message_id = message.id
                self._source += event.delta
                projection = _project(self._source, streaming=True, sidecars=self._sidecars)
                visible = projection["text"]
                delta = ""
                if visible.startswith(self._visible):
                    delta = visible[len(self._visible) :]
                    self._visible = visible
                # A reconnect seed retains only the latest update. It must be
                # self-contained; an ordinary subscriber still gets only deltas.
                data["delta"] = self._visible if reconnect_seed else delta
                if visible == self._visible:
                    _attach(data, message.id, projection, include_text=True)
                    if "inline_ui" in data:
                        self._published.add(message.id)
            elif event.type == "message_end":
                self._finish(data, message.id, projection)
        elif isinstance(message, Message) and event.type == "message_end":
            # Recovery providers can finalize with a replacement Message that
            # has lost its source marker. Retire the prior provisional widget.
            self._finish(data, message.id, {"text": message.text})
        if isinstance(event, (AgentEndEvent, HistoryDeltaEvent)):
            for item, payload in zip(event.messages, data["messages"], strict=True):
                if isinstance(item, Message) and _marked(item):
                    projection = _project(item.text, streaming=False, sidecars=self._sidecars)
                    replace_text_content(payload, projection["text"])
                    self._finish(data, item.id, projection)
                elif self._sidecars and isinstance(item, Message) and item._inline_ui_overlay:
                    # A bridge-local reconnect delta may already hold the
                    # projected row. Forward its admitted overlay, never try
                    # to recover source fences from the prose.
                    row = item._inline_ui_overlay["messages"].get(item.id)
                    if (
                        row
                        and row.get("text") == item.text
                        and isinstance(row.get("widgets"), list)
                    ):
                        self._finish(
                            data,
                            item.id,
                            {
                                "text": item.text,
                                "inline_ui": {"version": 1, "widgets": row["widgets"]},
                            },
                        )
        if event.type == "agent_end" and self._published:
            for message_id in tuple(self._published):
                self._finish(data, message_id, {"text": ""})
        if event.type in {"message_start", "message_end", "agent_start", "agent_end"}:
            self._reset()
        return data

    def _finish(
        self, data: dict[str, Any], message_id: str, projection: InlineUiProjection
    ) -> None:
        _attach(data, message_id, projection, include_text=True)
        if message_id in self._published:
            envelope = data.setdefault("inline_ui", {"version": 1, "messages": {}})
            # A prior complete widget is provisional until the provider's
            # authoritative message seals it. Empty is an explicit deletion.
            envelope["messages"].setdefault(message_id, None)
            self._published.discard(message_id)

    def _reset(self) -> None:
        self._message_id = ""
        self._source = ""
        self._visible = ""


def journal_inline_ui(entries: Iterable[Any], row_ids: set[str]) -> dict[str, Any] | None:
    """Sidecar for served rows only; the durable payload already holds legacy prose.

    The marked text lives outside the public row payload. A retained ordinary
    Markdown example, an unknown version, or a clipped row is not a widget.
    """
    envelope: dict[str, Any] = {}
    for entry in entries:
        if entry.id not in row_ids or entry.inline_ui_source_version != 1:
            continue
        blocks = entry.inline_ui_source_blocks
        if blocks:
            projection = _project(
                "".join(text for _, text in blocks), streaming=False, sidecars=True
            )
            _attach(envelope, entry.id, projection, include_text=True)
    return envelope.get("inline_ui")


def attach_journal_inline_ui(answer: dict[str, Any], entries: Iterable[Any]) -> dict[str, Any]:
    """Augment one already bounded, negotiated journal page without changing its rows."""
    ids = {
        row["id"]
        for row in answer.get("entries", [])
        if isinstance(row.get("id"), str) and not row.get("truncated")
    }
    if sidecar := journal_inline_ui(entries, ids):
        answer["inline_ui"] = sidecar
    return answer


def without_inline_ui(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the legacy shape without mutating the negotiated stream."""
    if "inline_ui" not in payload:
        return payload
    return {key: value for key, value in payload.items() if key != "inline_ui"}


def without_live_inline_ui(state: dict[str, Any]) -> dict[str, Any]:
    events = state.get("live_events")
    if not isinstance(events, list) or not any(
        isinstance(event, dict) and "inline_ui" in event for event in events
    ):
        return state
    return {
        **state,
        "live_events": [without_inline_ui(e) if isinstance(e, dict) else e for e in events],
    }


def without_desktop_inline_ui(frame: dict[str, Any]) -> dict[str, Any]:
    """Filter a shared desktop replay frame without mutating another viewer's cut."""
    payload = frame.get("payload")
    if not isinstance(payload, dict):
        return frame
    original = payload
    if frame.get("type") == "event":
        payload = without_inline_ui(payload)
    elif frame.get("type") == "frontend.update" and isinstance(
        changes := payload.get("changes"), dict
    ):
        clean = without_live_inline_ui(changes)
        if clean is not changes:
            payload = {**payload, "changes": clean}
    frontend = payload.get("frontend")
    if isinstance(frontend, dict) and isinstance(snapshot := frontend.get("snapshot"), dict):
        clean = without_live_inline_ui(snapshot)
        if clean is not snapshot:
            payload = {**payload, "frontend": {**frontend, "snapshot": clean}}
    history = payload.get("history")
    if isinstance(history, dict) and "inline_ui" in history:
        payload = {**payload, "history": without_inline_ui(history)}
    return frame if payload is original else {**frame, "payload": payload}
