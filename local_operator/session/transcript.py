"""Append-only JSONL transcript store.

The transcript is the session's durable memory: every LLM-visible message,
every compaction marker, and every host custom entry (wake schedules, skill
prompts, …) is one line in ``<dir>/transcript.jsonl``.

Entry shape::

    {"id": ..., "ts": ..., "type": "message" | "compaction" | "custom" | "prune",
     "payload": {...}}

Replay semantics: the latest
compaction entry wins; ``build_llm_history`` returns a marker message for that
summary plus every entry from ``first_kept_entry_id`` onward — never anything
before it. Custom entries are ignored by replay, so bookkeeping like wake
schedules never enters LLM context.

Footprint. Two things keep the file from growing the way an unbounded harness
transcript does (the failure mode that fills a volume: 200 MB single files).

- **Slim rows.** Message payloads are written with ``exclude_defaults``, the
  entry id is not repeated inside the payload, and a ``raw_arguments`` string
  that round-trips to the structured ``arguments`` is dropped. Together those
  are ~26% of a real transcript, measured on a 66-entry deepseek run. Reading
  is unchanged: every omitted field is a pydantic default, so rows written by
  older builds still load.
- **Prune entries.** Compaction blanks superseded and useless tool outputs in
  the LIVE context; without a record of that the transcript keeps the original
  multi-kilobyte text and a resumed session pays for output the session it
  resumes had already thrown away. ``append_prune`` journals the blanking,
  replay applies it, and :meth:`Transcript.compact_file` folds the journal
  into the message rows so the dead bytes leave the disk too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from local_operator.harness.types import (
    AgentMessage,
    CustomMessage,
    Message,
    TextContent,
)
from local_operator.session.attachments import AttachmentStore

logger = logging.getLogger(__name__)

TRANSCRIPT_FILENAME = "transcript.jsonl"

ENTRY_MESSAGE = "message"
ENTRY_COMPACTION = "compaction"
ENTRY_CUSTOM = "custom"

CUSTOM_KIND_MESSAGE = "message"
CUSTOM_KIND_CUSTOM = "custom"

#: Journal entry recording that compaction blanked a tool result in the live
#: context. Replay applies it; :meth:`Transcript.compact_file` folds it away.
ENTRY_PRUNE = "prune"

#: Rewrite the file only once this many bytes are provably reclaimable. A
#: prune pass runs on most turns, and rewriting a multi-megabyte transcript
#: every turn would cost far more I/O than the blanking saves. 256 KiB makes
#: the rewrite amortized-free while still bounding a long session's file.
COMPACT_FILE_THRESHOLD_BYTES = 256 * 1024

#: Marks the entry that sat at or before a prune whose journal entry has since
#: been folded away. Read by :meth:`Transcript.usages_since_compaction` as the
#: surviving evidence of WHERE the blanking happened; see :func:`_shrink_marked`.
#: Lives under ``provider_payload`` so an older build simply ignores it.
SHRUNK_KEY = "context_shrunk_here"


@dataclass
class TranscriptEntry:
    """One line of the transcript. ``id`` is stable and referenced by
    compaction entries (``first_kept_entry_id``)."""

    id: str
    ts: float
    type: str  # ENTRY_MESSAGE | ENTRY_COMPACTION | ENTRY_CUSTOM
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"id": self.id, "ts": self.ts, "type": self.type, "payload": self.payload},
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(line: str) -> "TranscriptEntry | None":
        """Parse one line; malformed lines are dropped individually (a corrupt
        row must never destroy the whole replay)."""
        try:
            raw = json.loads(line)
            return TranscriptEntry(
                id=str(raw["id"]),
                ts=float(raw.get("ts", 0.0)),
                type=str(raw["type"]),
                payload=dict(raw.get("payload", {})),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("dropping malformed transcript line: %.120s", line)
            return None


def encode_message_payload(
    message: Message | CustomMessage, attachments: AttachmentStore | None = None
) -> dict[str, Any]:
    """Serialize a message for one transcript row, omitting what a reader can
    reconstruct.

    Three redundancies, measured at 26% of a real 66-entry transcript:

    - ``exclude_defaults`` drops the ``"is_error":false``/``"usage":null``/
      ``"tool_calls":[]`` filler that ``model_dump`` writes on every row.
      Reading is unaffected because every omitted value IS the pydantic
      default, so rows from older builds and rows from this one both load.
    - The message ``id`` is already the entry id; repeating it inside the
      payload buys nothing and ``_entry_to_message`` overwrites it from the
      entry anyway.
    - ``raw_arguments`` is the provider's verbatim JSON argument string. When
      it round-trips to the same object as ``arguments`` it is a second,
      *escape-inflated* copy of data already on the row — on the measured
      transcript, 22.9 KB of escaped duplicate against 22.0 KB of structure.
      Dropped in that case only: wire clients fall back to
      ``json.dumps(arguments)``, which is the same call, and a string that
      does NOT round-trip (a model emitting non-canonical or repaired JSON) is
      kept verbatim because that is exactly where byte fidelity matters.
    """
    payload = message.model_dump(exclude_defaults=True, exclude={"id"})
    _externalize_attachments(payload, attachments)
    for call in payload.get("tool_calls") or ():
        raw = call.get("raw_arguments")
        if raw is None:
            continue
        try:
            redundant = json.loads(raw) == call.get("arguments", {})
        except (json.JSONDecodeError, TypeError):
            redundant = False
        if redundant:
            call.pop("raw_arguments")
    return payload


#: Key a content block carries when its media lives in the attachment store
#: instead of inline. Deliberately NOT a pydantic field of ``ImageContent``:
#: the wire model never sees it, because :func:`_resolve_attachments` inlines
#: the bytes back before replay, and an older build reading a row that
#: carries it simply ignores the unknown key — the block still parses as an
#: image with an empty ``data``, which degrades like any missing media.
ATTACHMENT_KEY = "attachment"

#: Placeholder ``data`` an unresolvable reference is rehydrated with. Empty
#: string would parse identically; the marker exists so a host that inspects
#: the replayed message can tell "no image" from "image we could not load".
ATTACHMENT_MISSING = ""


def _externalize_attachments(payload: dict[str, Any], attachments: AttachmentStore | None) -> None:
    """Move inline base64 image payloads out of ``payload`` into the store.

    In place, on the encoded payload dict. A block whose write fails keeps
    its inline data — the fallback is larger on disk, never wrong to read.
    Only blocks above a small floor are externalized: a 200-byte inline
    image costs less than the reference plus the store round-trip, and
    churning the store for favicon-sized images buys nothing.
    """
    if attachments is None:
        return
    content = payload.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        # Identify image blocks by the ``data`` key, NOT by ``type``: the
        # encoder dumps with ``exclude_defaults``, and ``type`` IS the
        # pydantic default on both content models, so the discriminant is
        # absent from the encoded row. A text block never carries ``data``.
        data = block.get("data")
        if not isinstance(data, str) or len(data) < _ATTACHMENT_FLOOR_BYTES:
            continue
        ref = attachments.put(data, str(block.get("mime_type", "image/png")))
        if ref is None:
            continue
        block.pop("data", None)
        block[ATTACHMENT_KEY] = ref.digest
        block["mime_type"] = ref.mime_type


def _resolve_attachments(payload: dict[str, Any], attachments: AttachmentStore) -> None:
    """Inline the bytes an externalized block references, in place.

    The mirror of :func:`_externalize_attachments` on the read path. A
    reference that no longer resolves degrades to an empty image rather than
    raising, for the same reason malformed transcript lines are dropped
    individually: one missing attachment must not take down a resume.
    """
    content = payload.get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        digest = block.pop(ATTACHMENT_KEY, None)
        if not isinstance(digest, str):
            continue
        resolved = attachments.get(digest)
        if resolved is None:
            logger.warning("transcript references missing attachment %s", digest)
            block["data"] = ATTACHMENT_MISSING
            continue
        data, mime_type = resolved
        block["data"] = data
        block["mime_type"] = mime_type


#: Below this many base64 characters an image stays inline. The reference
#: itself is ~60 bytes of JSON plus a store round-trip on every replay, so
#: the break-even is well under a kilobyte; 1 KiB is the conservative round
#: number that keeps genuinely tiny images (thumbnails, 1px spacers) out of
#: the store entirely.
_ATTACHMENT_FLOOR_BYTES = 1024


@dataclass(frozen=True)
class TranscriptPage:
    """One stable-ID page read from the durable JSONL transcript.

    ``before_id`` is deliberately an entry ID rather than a byte offset. File
    compaction replaces the JSONL atomically, so offsets become lies while IDs
    remain meaningful and let callers reconcile a replacement without keeping
    an unbounded mirror in memory.
    """

    entries: tuple[TranscriptEntry, ...]
    has_more: bool
    reconciled: bool = False


def read_transcript_page(
    directory: str | Path, *, before_id: str | None = None, limit: int = 100
) -> TranscriptPage:
    """Read a tail page, or the page immediately before ``before_id``.

    The scan is streaming and retains at most ``limit + 1`` parsed rows. This
    costs a sequential disk pass for older pages, which is intentional: the UI
    runs it in a thread, transcripts are append-mostly, and permanent offsets
    cannot survive ``compact_file`` replacing the file. Malformed rows are
    skipped independently, matching normal replay. If a requested ID vanished
    during replacement, return the current tail with ``reconciled=True`` so a
    reader can dedupe by stable ID instead of getting stuck on a stale cursor.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    path = Path(directory) / TRANSCRIPT_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    retained: deque[TranscriptEntry] = deque(maxlen=limit + 1)
    found = before_id is None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = TranscriptEntry.from_json(line)
            if entry is None:
                continue
            if before_id is not None and entry.id == before_id:
                found = True
                break
            retained.append(entry)
    if before_id is not None and not found:
        tail = read_transcript_page(directory, limit=limit)
        return TranscriptPage(tail.entries, tail.has_more, True)
    rows = tuple(retained)
    return TranscriptPage(entries=rows[-limit:], has_more=len(rows) > limit)


class Transcript:
    """Append-only JSONL store, thread/async-safe via an asyncio lock.

    Message entries reuse the message's own ``id`` as the entry id so
    compaction can reference ``first_kept_entry_id`` without a second mapping.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / TRANSCRIPT_FILENAME
        self._lock = asyncio.Lock()
        self._entries: list[TranscriptEntry] = []
        # Command identities live beside the append-only rows that prove their
        # admission.  Replay builds this once; hot-path lookups must not scan a
        # transcript whose model-facing window may also have been compacted.
        self._admitted_command_ids: set[str] = set()
        self._admission_handlers: list[Callable[[str], None]] = []
        #: Shared content-addressed media store. Write path: image payloads
        #: are externalized to it on append. Read path: references are
        #: resolved back to inline base64 on replay, so anything downstream
        #: of :meth:`build_llm_history` sees the same ``ImageContent`` it
        #: always has. Owned by the transcript rather than passed per call
        #: because both paths need the SAME store.
        self._attachments = AttachmentStore()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = TranscriptEntry.from_json(line)
                    if entry is not None:
                        self._entries.append(entry)
                        command_id = _admitted_command_id(entry)
                        if command_id is not None:
                            self._admitted_command_ids.add(command_id)

    # -- append -------------------------------------------------------------

    async def append_message(
        self,
        message: Message | CustomMessage,
        *,
        producer_command_id: str | None = None,
    ) -> TranscriptEntry:
        """Append one LLM-visible message (or a custom transcript message).

        BOTH kinds persist the message's own ``id`` as the entry id (never
        mint a new one): compaction's ``first_kept_entry_id`` must be able to
        reference a custom entry that renders into LLM context.

        ``producer_command_id`` is deliberately separate from that message id.
        Hosts use message ids for several unrelated purposes, so treating every
        user row as producer admission lets imported or seeded history suppress
        a later command whose namespace happens to collide. The marker lives on
        the transcript envelope, where replay preserves it without exposing
        transport bookkeeping to either model-facing history or the UI.
        """
        kind = CUSTOM_KIND_MESSAGE if isinstance(message, Message) else CUSTOM_KIND_CUSTOM
        payload: dict[str, Any] = {
            "kind": kind,
            **encode_message_payload(message, self._attachments),
        }
        if producer_command_id is not None:
            payload["producer_command_id"] = producer_command_id
        return await self._append(ENTRY_MESSAGE, payload, message.id)

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        preserve_data: dict[str, Any] | None = None,
        preserved_user_turns: list[dict[str, str]] | None = None,
    ) -> TranscriptEntry:
        """Record a compaction marker. Replay treats the LATEST one as the
        boundary: summary marker + entries from ``first_kept_entry_id`` on.

        ``preserve_data`` carries strategy-specific replay payloads (e.g.
        ``{"snapcompact": Archive.model_dump()}``) that replay renders back
        into LLM context instead of plain text.

        ``preserved_user_turns`` carries the VERBATIM text of every
        user-authored turn that fell into the summarized partition, so replay
        can re-inject them unparaphrased between the marker and the kept
        window. This is the structural half of "never summarize a user turn":
        a summarizer paraphrases, and a paraphrased constraint is how an agent
        later does the forbidden thing, so the user's own words must survive a
        compaction byte for byte rather than as a best-effort summary clause.
        Replay reconstructs ``[marker, *preserved_user_turns, *kept]`` — the
        contiguous-suffix replay (``first_kept_entry_id`` onward) alone would
        drop these turns on the next resume, since they sit BEFORE the cut in
        the transcript, so they have to ride the marker payload instead.
        """
        payload: dict[str, Any] = {
            "summary": summary,
            "first_kept_entry_id": first_kept_entry_id,
            "tokens_before": tokens_before,
        }
        if preserve_data is not None:
            payload["preserve_data"] = preserve_data
        if preserved_user_turns:
            payload["preserved_user_turns"] = preserved_user_turns
        return await self._append(ENTRY_COMPACTION, payload)

    async def append_custom(self, custom_type: str, details: dict[str, Any]) -> TranscriptEntry:
        """Append a host bookkeeping entry (wake schedules, checkpoints, …).
        Custom entries never enter LLM context."""
        return await self._append(ENTRY_CUSTOM, {"custom_type": custom_type, "details": details})

    async def append_prune(self, target_entry_id: str, notice: str) -> TranscriptEntry:
        """Journal that compaction blanked the tool result at ``target_entry_id``.

        Without this the transcript and the live context disagree the moment
        pruning fires: the session drops a 12 KB tool output down to a short
        notice, then a resume replays the 12 KB back in and the resumed turn
        costs more tokens than the turn it resumed from. Journalling instead
        of rewriting the row keeps the store append-only on the hot path;
        :meth:`compact_file` folds the journal in later, off the hot path.
        """
        return await self._append(ENTRY_PRUNE, {"target": target_entry_id, "notice": notice})

    async def _append(
        self, type: str, payload: dict[str, Any], entry_id: str | None = None
    ) -> TranscriptEntry:
        entry = TranscriptEntry(
            id=entry_id or uuid.uuid4().hex,
            ts=time.time(),
            type=type,
            payload=payload,
        )
        async with self._lock:
            # DELIBERATELY SYNCHRONOUS, and not an oversight — do not "fix"
            # this into a to_thread the way :meth:`compact_file` legitimately
            # is. This append has callers that never await it:
            # ``Session.queue_aside`` persists a materialized aside through
            # ``_spawn_background``, i.e. fire-and-forget. Every await inside
            # this method widens the window between "the message reached the
            # model" and "the message is on disk", and a reader that opens the
            # transcript in that window sees a child's context missing a
            # message it has already acted on.
            #
            # Measured, because the tradeoff was real: offloading this cut the
            # worst loop stall at one workload (56 ms -> 22 ms) and made it
            # WORSE at a larger one (298 ms -> 655 ms), while CI caught the
            # visibility regression it introduced. The tokenizer offload in
            # ``Session._offloaded`` is where the loop-responsiveness win
            # actually comes from (1360 ms -> 56 ms); this write is a few
            # hundred microseconds and is not worth a correctness hazard.
            #
            # Self-healing against a vanished directory: another process's
            # startup sweep (or a user tidying by hand) can remove a session
            # directory while it still looks empty — the gap between Session
            # construction and the first turn is as long as the user takes to
            # type their first message, and the retention sweep's ``live_dir``
            # only protects the sweeping process's OWN session. Dying on
            # FileNotFoundError here costs the whole session for a directory
            # one mkdir restores; the candidate joins ``self._entries`` only
            # after the rebuilt file reaches its durability boundary, so a
            # failed rebuild cannot resurrect a row on the next append.
            #
            # The FILE alone vanishing (directory intact) is the same wound
            # with a quieter symptom: ``"a"`` mode recreates the file without
            # raising, so the append succeeds and the transcript silently
            # holds one row while memory holds the whole session — a resume
            # would then replay a single message as if the rest never
            # happened. Checked BEFORE the append because that path never
            # raises; both cases rebuild from the committed rows plus this
            # unpublished candidate under the same lock.
            rebuild = not self.path.exists()
            if not rebuild:
                try:
                    previous_size = self.path.stat().st_size
                except FileNotFoundError:
                    rebuild = True
                else:
                    try:
                        with self.path.open("a", encoding="utf-8") as handle:
                            handle.write(entry.to_json() + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                    except FileNotFoundError:
                        # Directory removed between the exists() check and the
                        # open — the race is real, so the window is too.
                        rebuild = True
                    except BaseException:
                        # write/flush/fsync may raise after changing the file.
                        # The append-only framing makes its old byte boundary
                        # the only safe rollback point; the lock prevents a
                        # successor from being truncated with this candidate.
                        try:
                            os.truncate(self.path, previous_size)
                        except FileNotFoundError:
                            pass
                        raise
            if rebuild:
                self.directory.mkdir(parents=True, exist_ok=True)
                try:
                    with self.path.open("w", encoding="utf-8") as handle:
                        for row in (*self._entries, entry):
                            handle.write(row.to_json() + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                except BaseException:
                    # A failed rebuild started from a missing file. Removing
                    # its partial journal preserves that pre-call state and
                    # prevents a parseable prefix from becoming admitted after
                    # restart even though this call failed.
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    raise
            self._entries.append(entry)
            command_id = _admitted_command_id(entry)
            if command_id is not None:
                # Update only after the append's flush boundary.  A crash after
                # this point can replay the row; an ACK or callback before it
                # would claim durability the transcript did not yet have.
                self._admitted_command_ids.add(command_id)
                for handler in list(self._admission_handlers):
                    try:
                        handler(command_id)
                    except Exception:  # noqa: BLE001 - observers cannot undo durability
                        logger.exception("transcript admission handler failed")
        return entry

    def has_admitted_command(self, command_id: str) -> bool:
        """Return whether an append-only user row already owns ``command_id``."""
        return command_id in self._admitted_command_ids

    def subscribe_admitted_commands(self, handler: Callable[[str], None]) -> Callable[[], None]:
        """Observe user command IDs after their durable append boundary."""
        self._admission_handlers.append(handler)

        def unsubscribe() -> None:
            try:
                self._admission_handlers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    # -- replay -------------------------------------------------------------

    def entries(self) -> list[TranscriptEntry]:
        """All entries in append order (in-memory snapshot)."""
        return list(self._entries)

    def latest_custom(self, custom_type: str) -> dict[str, Any] | None:
        """Details of the newest custom entry of ``custom_type`` (backward
        scan, first hit wins — each change appends a full snapshot)."""
        for entry in reversed(self._entries):
            if entry.type == ENTRY_CUSTOM and entry.payload.get("custom_type") == custom_type:
                return dict(entry.payload.get("details", {}))
        return None

    def usages_since_compaction(self) -> list[dict[str, Any]]:
        """Every message entry's ``usage`` payload recorded AFTER the newest
        compaction, oldest first (all of them when nothing compacted).

        Append order is the only place the "after" in that sentence exists.
        :meth:`build_llm_history` cannot answer it: it puts the compaction
        marker at the HEAD of what it returns and the kept window after it, so
        the replayed list carries pre-pass readings ahead of post-pass ones with
        nothing to tell them apart. The entries know, because they are in the
        order they happened.

        Why the distinction is worth an accessor: a kept message still carries
        the ``usage`` it had before the pass shrank the context, and a host that
        seeds a status readout from one reports a context that no longer exists
        (measured at 900k against a real 1.7k). But a session that compacted and
        then ran ten more turns has a perfectly good newest reading, and
        refusing that would be the opposite error. Entries after the marker are
        exactly the readings that survived the pass.

        PRUNING moves the boundary too, and it is the case that makes this more
        than a compaction concern. Blanking a tool result shrinks the live
        context exactly as a compaction pass does, but leaves no marker: the
        journal entry is folded away by :meth:`compact_file` and the message
        rows survive untouched. Measured against the real pruner: a reading of
        640_000 restored for a true context of 31_715, which a host installs as
        exact and hands to the compaction gate.

        So the boundary is POSITIONAL for both passes — the newest compaction
        marker or the newest pruned row, whichever is later — and that is not a
        stylistic choice. A filter that merely skipped pruned rows would be dead
        code: pruning only ever blanks ``role == "tool"`` messages
        (``compaction.pruning``) and ``usage`` is only ever set on the
        ASSISTANT message of a turn (``harness.loop``), so the two sets are
        disjoint by construction and no usage-carrying row is ever flagged. That
        version passed a test which pruned an assistant message — a state the
        production pruner cannot produce — and fixed nothing. What is wrong with
        a pre-prune reading is not the row it sits on, it is that it describes a
        context measured before the shrink, so position is the only thing that
        can express it.

        Both spellings of "pruned" are consulted for the same reason
        :meth:`compact_file` exists: the journal entry before a fold, the
        ``provider_payload`` flag on the row afterwards.

        Returns the raw payload dicts rather than ``Usage`` objects: this module
        is the persistence layer and does not own the harness's models, and the
        caller is already parsing them.
        """
        start = 0
        for index in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[index]
            if entry.type in (ENTRY_COMPACTION, ENTRY_PRUNE):
                # A journal entry sits at the moment the shrink HAPPENED, which
                # is what the boundary must be drawn on — not at the row it
                # targets. The targeted tool result may be hundreds of entries
                # older, and the readings in between were all measured before
                # the blanking and so describe the pre-shrink context. Seen in
                # the wild: one real transcript on this machine has its newest
                # prune at entry 88 with its newest usage at 82, and taking the
                # target's position instead restored two stale readings.
                start = index + 1
                break
            if (entry.payload.get("provider_payload") or {}).get(SHRUNK_KEY) or (
                entry.type == ENTRY_MESSAGE
                and (entry.payload.get("provider_payload") or {}).get("pruned")
            ):
                # The FOLDED forms, newest-position-wins by virtue of the scan
                # order. Two of them, because two kinds of file exist:
                #
                # * ``SHRUNK_KEY`` is the mark `compact_file` leaves at the
                #   position the journal entry held (see `_shrink_marked`), so
                #   the boundary lands where the prune actually was. This is the
                #   accurate one and the only one written from now on.
                # * The ``pruned`` flag on the target row is the FALLBACK, for
                #   transcripts folded by a build that predates the mark. Those
                #   exist on disk already — `main` folds journals writing only
                #   this flag — and without it the scan would match nothing,
                #   fall through to the start of the file, and restore every
                #   reading in it.
                #
                #   It is the weaker signal: it marks WHICH row was blanked, not
                #   when, and the target sits EARLIER than the prune that
                #   blanked it. So the boundary lands too early and the window
                #   is too WIDE — it can still admit readings taken between the
                #   target and the prune, which is fewer stale figures than
                #   admitting the whole file but not zero. Exact only where the
                #   two coincide.
                #
                #   Kept anyway, because the alternative for those files is
                #   restoring everything, and improved only by the mark, which
                #   every fold from here writes.
                start = index + 1
                break
        return [
            dict(entry.payload["usage"])
            for entry in self._entries[start:]
            if entry.type == ENTRY_MESSAGE and isinstance(entry.payload.get("usage"), dict)
        ]

    def pending_prunes(self) -> dict[str, str]:
        """``{target entry id: notice}`` for every un-folded prune entry.

        Later entries win, so a target blanked twice keeps the newest notice.
        """
        return {
            str(entry.payload.get("target")): str(entry.payload.get("notice", ""))
            for entry in self._entries
            if entry.type == ENTRY_PRUNE and entry.payload.get("target")
        }

    def build_llm_history(self) -> list[AgentMessage]:
        """Replay the transcript into LLM-visible messages.

        Latest compaction entry wins: a marker for its summary followed by the
        entries from ``first_kept_entry_id`` onward. Without compaction, every
        message entry replays. Custom entries are ignored.

        Prune entries are applied last, so a replayed history matches the
        pruned live context byte for byte rather than resurrecting the tool
        output compaction already decided was dead weight.
        """
        entries = self._entries
        compaction_index: int | None = None
        for i in range(len(entries) - 1, -1, -1):
            if entries[i].type == ENTRY_COMPACTION:
                compaction_index = i
                break

        start = 0
        prefix: list[AgentMessage] = []
        if compaction_index is not None:
            compaction = entries[compaction_index]
            details: dict[str, Any] = {"summary": compaction.payload.get("summary", "")}
            preserve_data = compaction.payload.get("preserve_data")
            if preserve_data is not None:
                details["preserve_data"] = preserve_data
            prefix.append(
                CustomMessage(
                    custom_type="compaction_summary",
                    attribution="system",
                    details=details,
                )
            )
            # User-authored turns that fell into the summarized partition are
            # re-injected VERBATIM right after the marker, so the user's own
            # words survive the pass byte for byte instead of being paraphrased
            # away (see :meth:`append_compaction`). They sit before the cut in
            # the transcript, so the contiguous ``first_kept_entry_id`` suffix
            # below never replays them — the marker payload is the only carrier
            # that reaches a resumed session. Reusing each turn's original id is
            # safe: those message entries live before ``start`` and are not
            # replayed, so there is no duplicate, and the guard that expires
            # stale asides keys on ``CustomMessage`` (a real ``user`` Message is
            # never mistaken for one).
            preserved_turns = compaction.payload.get("preserved_user_turns") or ()
            if preserved_turns:
                # Lazy import keeps this low-level store free of a hard compaction
                # dependency (the session imports compaction lazily for the same
                # reason). The flag marks these as already-compacted content so a
                # post-resume pass does not re-count them as fresh history.
                from local_operator.compaction.cutpoint import PRESERVED_USER_TURN_KEY

                for turn in preserved_turns:
                    if not isinstance(turn, dict):
                        continue
                    message = Message.user(str(turn.get("text", "")))
                    turn_id = turn.get("id")
                    if isinstance(turn_id, str) and turn_id:
                        message.id = turn_id
                    message.provider_payload = {PRESERVED_USER_TURN_KEY: True}
                    prefix.append(message)
            first_kept_id = compaction.payload.get("first_kept_entry_id")
            # The first kept entry normally sits BEFORE the compaction marker
            # (messages are persisted as they happen; the marker comes last),
            # so scan the whole transcript. If the id no longer resolves (a
            # dropped malformed line, a converter-minted id), replaying from
            # compaction_index + 1 would point PAST the kept window and lose
            # every message compaction promised to preserve. Replaying too
            # much is recoverable at the next compaction; silent amnesia is
            # not, so fall back to the full history with an error.
            start = 0
            if first_kept_id is None:
                start = compaction_index + 1
            else:
                for i in range(len(entries)):
                    if entries[i].id == first_kept_id:
                        start = i
                        break
                else:
                    logger.error(
                        "first_kept_entry_id %s not found in transcript; " "replaying full history",
                        first_kept_id,
                    )

        prunes = self.pending_prunes()
        out: list[AgentMessage] = list(prefix)
        for entry in entries[start:]:
            if entry.type != ENTRY_MESSAGE:
                continue
            message = _entry_to_message(entry, self._attachments)
            if message is None:
                continue
            notice = prunes.get(entry.id)
            if notice is not None and isinstance(message, Message):
                _apply_prune(message, notice)
            out.append(message)
        return out

    # -- lifecycle ----------------------------------------------------------

    def reclaimable_bytes(self) -> int:
        """Bytes :meth:`compact_file` would free right now.

        The dead weight is the difference between each pruned row as written
        and the one-line notice that replaces it, plus the journal entries
        themselves — MINUS the few bytes the fold adds back, which is the
        boundary mark it leaves in place of the journal entry's position (see
        :func:`_shrink_marked`). Small, but this figure is compared for equality
        against what :meth:`compact_file` actually reclaims, and an estimate
        that ignores a cost the fold really pays is simply wrong.
        """
        prunes = self.pending_prunes()
        if not prunes:
            return 0
        total = 0
        boundary: TranscriptEntry | None = None
        newest_prune = max(
            (index for index, entry in enumerate(self._entries) if entry.type == ENTRY_PRUNE),
            default=-1,
        )
        for index, entry in enumerate(self._entries):
            if entry.type == ENTRY_PRUNE:
                total += len(entry.to_json()) + 1
                continue
            if entry.type == ENTRY_MESSAGE and entry.id in prunes:
                entry = _pruned_entry(entry, prunes[entry.id])
                total += len(self._entries[index].to_json()) - len(entry.to_json())
            if index <= newest_prune:
                boundary = entry
        if boundary is not None:
            total -= len(_shrink_marked(boundary).to_json()) - len(boundary.to_json())
        return max(total, 0)

    async def compact_file(self, min_reclaim_bytes: int = COMPACT_FILE_THRESHOLD_BYTES) -> int:
        """Fold the prune journal into the message rows and rewrite the file.

        This is the half of the footprint story that the in-context caps do
        not cover. A tool output the session already blanked still occupies
        its original kilobytes on disk, forever, in a file that only ever
        grows — the exact shape that let a sibling harness accumulate 233 MB
        single-file transcripts and fill a volume. Folding is semantically
        invisible: entry ids, order and types are unchanged, the journal
        entries disappear because their effect is now materialized, and
        :meth:`build_llm_history` produces the same messages either way.

        Returns the bytes reclaimed (0 when below ``min_reclaim_bytes``, so
        the caller can invoke it every turn without rewriting a large file
        for a few hundred bytes). Crash-safe: the new file is written beside
        the old one and moved over it with an atomic ``os.replace``, so an
        interrupted compaction leaves the original transcript intact.
        """
        async with self._lock:
            prunes = self.pending_prunes()
            if not prunes:
                return 0
            before = self.path.stat().st_size if self.path.exists() else 0
            # WHERE the newest prune sat, before the entries carrying that fact
            # are dropped. Folding is meant to be semantically invisible, and
            # without this it is not: a prune's position is what tells a reader
            # which usage readings predate the blanking, and the target row it
            # points at can be hundreds of entries older than the prune itself.
            # Discarding it silently promoted every reading in between back to
            # "current" — measured as three stale figures restored on a folded
            # transcript that correctly reported none before the fold.
            newest_prune = max(
                (index for index, entry in enumerate(self._entries) if entry.type == ENTRY_PRUNE),
                default=-1,
            )
            folded: list[TranscriptEntry] = []
            boundary = -1
            for index, entry in enumerate(self._entries):
                if entry.type == ENTRY_PRUNE:
                    continue
                if entry.type == ENTRY_MESSAGE and entry.id in prunes:
                    entry = _pruned_entry(entry, prunes[entry.id])
                folded.append(entry)
                # Remember which SURVIVING entry is the last one at or before
                # the newest prune. Marked after the loop rather than inside it,
                # because only the final such entry is the boundary — marking
                # every one of them puts the mark on the newest row in the file
                # and a backward scan then stops at the end, refusing the whole
                # history instead of the part that predates the blanking.
                if index <= newest_prune:
                    boundary = len(folded) - 1
            if boundary >= 0:
                folded[boundary] = _shrink_marked(folded[boundary])
            # Re-serializing and rewriting the WHOLE transcript is proportional
            # to session length (hundreds of ms on a long one), and it runs on
            # the loop every session shares. Off to a worker: the ``_lock`` held
            # across the await keeps it exclusive against appends, and the
            # ``os.replace`` is still atomic, so an interrupted rewrite leaves
            # the original intact exactly as before. The boundary mark is
            # applied ABOVE, so the worker serializes the marked list.
            payload, reclaimed = await asyncio.to_thread(self._render_folded, folded, before)
            if reclaimed < min_reclaim_bytes:
                return 0
            await asyncio.to_thread(self._replace_file, payload)
            self._entries = folded
            return reclaimed

    @staticmethod
    def _render_folded(folded: list[TranscriptEntry], before: int) -> tuple[str, int]:
        """Serialize the folded entries and price the rewrite.

        Worker-thread half of :meth:`compact_file`: this is the O(session)
        string building, kept out of the loop. Returns the payload alongside
        the bytes it reclaims so the caller can decide whether the rewrite is
        worth doing without re-measuring.
        """
        payload = "".join(entry.to_json() + "\n" for entry in folded)
        return payload, before - len(payload.encode("utf-8"))

    def _replace_file(self, payload: str) -> None:
        """Write the compacted transcript beside the old one and move it over.

        Worker-thread half of :meth:`compact_file`. Crash safety is the whole
        point of the temp-file dance: ``os.replace`` is atomic, so an
        interrupted compaction leaves the original transcript readable.
        """
        tmp = self.path.with_suffix(self.path.suffix + ".compact")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    def flush(self) -> None:
        """Writes are flushed per append; provided for dispose() symmetry."""
        return None


def _pruned_entry(entry: TranscriptEntry, notice: str) -> TranscriptEntry:
    """``entry`` with its content folded down to ``notice``.

    A fresh entry rather than a mutation: ``reclaimable_bytes`` measures the
    original against the folded form and must not destroy the original to do
    it. ``tool_calls`` are left alone — pruning only ever blanks tool RESULTS,
    and dropping an assistant message's calls would break call/result pairing
    for every provider.
    """
    payload = dict(entry.payload)
    payload["content"] = [TextContent(text=notice).model_dump()]
    provider_payload = dict(payload.get("provider_payload") or {})
    provider_payload["pruned"] = True
    payload["provider_payload"] = provider_payload
    return TranscriptEntry(id=entry.id, ts=entry.ts, type=entry.type, payload=payload)


def _shrink_marked(entry: TranscriptEntry) -> TranscriptEntry:
    """``entry`` marked as sitting at or before a prune that has been folded away.

    :meth:`Transcript.compact_file` materializes the prune journal into the rows
    it targets and drops the journal entries, which is semantically invisible for
    replay — and was NOT invisible for the usage boundary, because a prune's
    POSITION is what says which readings predate the blanking. The target row is
    no substitute: it can be hundreds of entries older than the prune that
    blanked it.

    So the position is preserved as a flag on the last entry that sat at or
    before it. ``provider_payload`` rather than a new entry type, because a new
    type would have to be understood by every reader of the file including older
    builds, where an unknown key on a payload is ignored by construction.
    """
    payload = dict(entry.payload)
    provider_payload = dict(payload.get("provider_payload") or {})
    if provider_payload.get(SHRUNK_KEY):
        return entry
    provider_payload[SHRUNK_KEY] = True
    payload["provider_payload"] = provider_payload
    return TranscriptEntry(id=entry.id, ts=entry.ts, type=entry.type, payload=payload)


def _admitted_command_id(entry: TranscriptEntry) -> str | None:
    """Extract an explicitly marked producer identity from a transcript row.

    Message IDs belong to the conversation namespace and can collide with
    seeded, imported, compacted, custom, or locally-authored rows. Only the
    transport marker proves that a producer command crossed the durable append
    boundary; released transcripts and malformed fragments have no marker and
    therefore cannot poison admission.
    """
    if entry.type != ENTRY_MESSAGE:
        return None
    payload = entry.payload
    # Producer provenance was introduced with an explicit kind marker. Treating
    # the legacy missing-kind default as producer-eligible lets imported rows
    # claim IDs merely by carrying an unknown extra envelope field.
    if payload.get("kind") != CUSTOM_KIND_MESSAGE:
        return None
    command_id = payload.get("producer_command_id")
    if not isinstance(command_id, str) or not command_id.strip():
        return None
    message = _entry_to_message(entry)
    if not isinstance(message, Message) or message.role != "user":
        return None
    return command_id


def _entry_to_message(
    entry: TranscriptEntry, attachments: AttachmentStore | None = None
) -> AgentMessage | None:
    """Rehydrate one message entry; malformed rows are dropped individually."""
    payload = dict(entry.payload)
    kind = payload.pop("kind", CUSTOM_KIND_MESSAGE)
    # Producer identity belongs to the transcript envelope, never the Message
    # schema. Leaving it in makes strict Message decoding reject every valid
    # producer row as an unknown field during both replay and admission checks.
    payload.pop("producer_command_id", None)
    # The entry id IS the message id for BOTH kinds — the writer passes
    # ``message.id`` as the entry id and the encoder omits it from the
    # payload, so a custom entry that read its id from the payload would come
    # back with a converter-minted uuid and break the ``first_kept_entry_id``
    # reference the transcript exists to keep stable.
    payload["id"] = entry.id
    if attachments is not None:
        _resolve_attachments(payload, attachments)
    try:
        if kind == CUSTOM_KIND_CUSTOM:
            return CustomMessage.model_validate(payload)
        return Message.model_validate(payload)
    except Exception:
        logger.warning("dropping unparseable transcript message entry %s", entry.id)
        return None


def _apply_prune(message: Message, notice: str) -> None:
    """Blank ``message`` the way ``compaction.pruning._blank`` blanked it live.

    Deliberately mirrors that function's shape (notice as the whole content,
    ``provider_payload["pruned"] = True`` layered over the existing payload)
    so a replayed message is indistinguishable from one the running session
    pruned — including for the pruning pass itself, which skips anything
    already flagged and would otherwise re-blank it every turn.
    """
    message.content = [TextContent(text=notice)]
    message.provider_payload = {**(message.provider_payload or {}), "pruned": True}
