"""The core import service every surface calls.

One service owns selection, idempotency, target creation, batching, memory and
cancellation. Routes/CLI/TUI/agent tools are thin adapters over this class; none
of them may parse a foreign store or write a target transcript themselves.

**Idempotency is derived from the manifest, not from luck.** A complete manifest
with an unchanged source fingerprint is a no-op. A grown source appends only
rows whose deterministic ids are missing, but ONLY while the target's byte
prefix still matches the manifest: if the user edited/compacted the target, the
service imports a fresh copy with a deterministic suffix instead of appending to
their work. A rotated source (shorter file or a changed prefix) likewise gets a
new copy, never a splice.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from local_operator.harness_import import manifest as manifest_store
from local_operator.harness_import import memory as memory_import
from local_operator.harness_import import store
from local_operator.harness_import.mapping import MappedSession, build_mapped_session
from local_operator.harness_import.redact import redact_text
from local_operator.harness_import.sources.base import SourceAdapter
from local_operator.harness_import.sources.registry import build_adapters
from local_operator.harness_import.types import (
    ImportPreview,
    ImportProgress,
    ImportRequest,
    ImportResult,
    ImportWarning,
    SessionResult,
    SourceInfo,
    SourceSession,
)
from local_operator.session.creation import ensure_session_created_at
from local_operator.session.transcript import ENTRY_COMPACTION, Transcript

#: Bounded write batch: rows and bytes. A batch is always extended past an
#: anchor that lands later, so replay never sees a dangling ``first_kept_entry_id``.
BATCH_ROWS = 200
BATCH_BYTES = 4 * 1024 * 1024
TITLE_MAX = 120
#: Source discovery parses every listing once; the Settings page polls it. A
#: short cache keeps a repeat call cheap without making an import decision on
#: stale counts for long.
SOURCE_CACHE_SECONDS = 30.0


class _ImportCancelled(RuntimeError):
    """Raised between write batches when the operator cancelled the job."""


def normalise_title(text: str | None) -> str:
    if not text:
        return ""
    # Drop control/format/private-use/surrogate characters: omp titles can
    # begin with a private-use glyph that renders as tofu in lop's pickers.
    text = "".join(char for char in str(text) if unicodedata.category(char)[0] != "C")
    text = " ".join(text.split())
    text, _ = redact_text(text)
    return text[:TITLE_MAX].strip()


def target_session_id(source: str, source_id: str) -> str:
    payload = f"lop-harness-import/v1\0{source}\0{source_id}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


class HarnessImportService:
    def __init__(self, config: Path, *, importing_session_id: str = "harness-import"):
        self.config = Path(config)
        self.importing_session_id = re.sub(r"[^A-Za-z0-9_.-]", "-", importing_session_id)[:64]
        self.adapters: dict[str, SourceAdapter] = build_adapters(self.config)
        self._cancel = threading.Event()
        self._lock = store.ImportLock(self.config)
        self._sources_cache: tuple[float, list[SourceInfo]] | None = None

    # -- discovery ----------------------------------------------------------

    async def sources(self) -> list[SourceInfo]:
        def detect() -> list[SourceInfo]:
            now = time.monotonic()
            if (
                self._sources_cache is not None
                and now - self._sources_cache[0] < SOURCE_CACHE_SECONDS
            ):
                return list(self._sources_cache[1])
            rows = [adapter.detect() for adapter in self.adapters.values()]
            self._sources_cache = (now, rows)
            return list(rows)

        return await asyncio.to_thread(detect)

    async def list_sessions(
        self, source: str, *, include_subagents: bool = False
    ) -> list[SourceSession]:
        adapter = self.adapters.get(source)
        if adapter is None:
            raise ValueError(f"unknown import source: {source}")
        return await asyncio.to_thread(adapter.list_sessions, include_subagents=include_subagents)

    def _select(self, request: ImportRequest) -> list[tuple[SourceAdapter, SourceSession]]:
        selected: list[tuple[SourceAdapter, SourceSession]] = []
        names = request.sources or list(self.adapters)
        for name in names:
            adapter = self.adapters.get(name)
            if adapter is None:
                continue
            wanted = request.session_ids.get(name, [])
            sessions = adapter.list_sessions(include_subagents=request.include_subagents)
            if not request.all and wanted:
                by_id = {session.source_id: session for session in sessions}
                sessions = [by_id[sid] for sid in wanted if sid in by_id]
            elif not request.all:
                sessions = []
            selected.extend((adapter, session) for session in sessions)
        # Same source session selected twice in one job: one target, one row set.
        deduped: dict[tuple[str, str], tuple[SourceAdapter, SourceSession]] = {}
        for adapter, session in selected:
            deduped[(adapter.source, session.source_id)] = (adapter, session)
        ordered = sorted(
            deduped.values(), key=lambda item: (item[1].source, item[1].updated_ts), reverse=False
        )
        if request.max_sessions is not None:
            ordered = ordered[: max(0, request.max_sessions)]
        return ordered

    # -- preview ------------------------------------------------------------

    async def preview(self, request: ImportRequest) -> ImportPreview:
        return await asyncio.to_thread(self._preview_sync, request)

    def _preview_sync(self, request: ImportRequest) -> ImportPreview:
        preview = ImportPreview()
        selected = self._select(request)
        preview.warnings.extend(self._option_warnings(request, selected))
        for adapter, session in selected:
            row = session.to_dict()
            # Titles are a result surface too: credential-shaped text must not
            # reach the agent from preview while `sessions` redacts it, and
            # private-use glyphs render as tofu in lop's pickers.
            row["title"] = normalise_title(session.title)
            row["target_session_id"] = target_session_id(adapter.source, session.source_id)
            manifests = self._existing_manifests(session, adapter.source)
            row["already_imported"] = any(
                manifest.get("status") == "complete" for manifest in manifests
            )
            warnings = list(session.warnings)
            messages = session.message_count or 0
            compactions = session.compaction_count or 0
            # Preview must stay a metadata/count pass: it never builds the full
            # mapped object graph a real import needs (design §6).
            counter = getattr(adapter, "preview_counts", None)
            if session.availability == "full" and counter is not None:
                try:
                    messages, compactions = counter(session)
                except Exception as exc:  # noqa: BLE001 - a preview reports, never crashes
                    warnings.append(f"body unreadable: {type(exc).__name__}")
                    preview.warnings.append(
                        ImportWarning(
                            source=adapter.source,
                            session_id=session.source_id,
                            code="preview_failed",
                            message=f"body unreadable: {type(exc).__name__}",
                        )
                    )
            row["message_count"] = messages
            row["message_count_estimated"] = True
            preview.message_counts_estimated = True
            row["compaction_count"] = compactions
            row["warnings"] = warnings
            preview.total_messages += messages
            preview.total_compactions += compactions
            preview.total_sessions += 1
            preview.sessions.append(row)
        if preview.total_sessions:
            preview.warnings.append(
                ImportWarning(
                    source="",
                    session_id="",
                    code="preview_counts_estimated",
                    message=(
                        "Message counts are estimates from source metadata; the "
                        "import may emit fewer after branch/segment selection."
                    ),
                )
            )
        if request.include_memory:
            for source, sessions in self._sessions_by_source(selected).items():
                adapter = self.adapters[source]
                try:
                    records = adapter.read_memory(sessions)
                except Exception as exc:  # noqa: BLE001
                    preview.warnings.append(
                        ImportWarning(
                            source=source,
                            session_id="",
                            code="memory_preview_failed",
                            message=f"memory unreadable: {type(exc).__name__}",
                        )
                    )
                    continue
                total, global_tokens = memory_import.project_memory(records)
                preview.total_memory_records += len(records)
                preview.projected_memory_tokens += total
                preview.projected_global_tokens += global_tokens
                if any(row.scope_hint == "instruction" for row in records):
                    preview.instruction_review_paths.append(
                        str(memory_import.instruction_path(self.config, source))
                    )
        preview.sources = [adapter.detect().to_dict() for adapter in self.adapters.values()]
        return preview

    # -- import -------------------------------------------------------------

    async def import_sessions(
        self,
        request: ImportRequest,
        progress: Callable[[ImportProgress], None] | None = None,
    ) -> ImportResult:
        result = ImportResult(job_id=uuid.uuid4().hex[:12])
        state = ImportProgress(
            job_id=result.job_id,
            state="running",
            phase="selecting",
            started_at=time.time(),
            updated_at=time.time(),
        )
        if not self._lock.acquire():
            result.state = "failed"
            result.error = "another import is already running"
            state.state = "failed"
            state.error = result.error
            if progress:
                progress(state)
            return result

        def emit(phase: str, **changes: Any) -> None:
            state.phase = phase
            for key, value in changes.items():
                setattr(state, key, value)
            state.updated_at = time.time()
            if progress:
                progress(state)

        try:
            selected = await asyncio.to_thread(self._select, request)
            result.warnings.extend(self._option_warnings(request, selected))
            emit("importing", total=len(selected))
            imported_by_thread: dict[str, str] = {}
            imported_by_cwd: dict[str, list[str]] = {}
            session_dirs: dict[str, Path] = {}
            for index, (adapter, session) in enumerate(selected):
                if self._cancel.is_set():
                    result.state = "cancelled"
                    state.state = "cancelled"
                    emit("cancelled")
                    break
                emit(
                    "session",
                    source=adapter.source,
                    session_id=session.source_id,
                    completed=index,
                    total=len(selected),
                )
                outcome = await asyncio.to_thread(
                    self._import_one, adapter, session, request, result
                )
                result.sessions.append(outcome)
                if outcome.target_session_id and outcome.outcome in ("imported", "resumed"):
                    imported_by_thread[session.source_id] = outcome.target_session_id
                    session_dirs[outcome.target_session_id] = (
                        self.config / "sessions" / outcome.target_session_id
                    )
                    if session.cwd:
                        imported_by_cwd.setdefault(session.cwd, []).append(
                            outcome.target_session_id
                        )
                emit(
                    "session",
                    source=adapter.source,
                    session_id=session.source_id,
                    completed=index + 1,
                    total=len(selected),
                    results=[row.to_dict() for row in result.sessions],
                )
                if self._cancel.is_set():
                    result.state = "cancelled"
                    state.state = "cancelled"
                    emit("cancelled")
                    break
                if outcome.outcome == "failed":
                    continue
            result.messages = sum(row.messages for row in result.sessions)
            result.compactions = sum(row.compactions for row in result.sessions)
            result.redactions = sum(row.redactions for row in result.sessions)
            if request.include_memory and result.state != "cancelled":
                emit("memory")
                await asyncio.to_thread(
                    self._import_memory,
                    request,
                    selected,
                    imported_by_thread,
                    imported_by_cwd,
                    session_dirs,
                    result,
                )
            if self._cancel.is_set():
                result.state = "cancelled"
                state.state = "cancelled"
            elif result.state != "cancelled":
                result.state = "done"
                state.state = "done"
            emit("done")
            return result
        except Exception as exc:  # noqa: BLE001 - the job reports its own failure
            result.state = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            state.state = "failed"
            state.error = result.error
            emit("failed")
            return result
        finally:
            self._lock.release()

    def clear_cancel(self) -> None:
        """Clear a previous/cancel-in-queue request before a NEW job starts.

        This lives on the service but is called by ``JOBS.start`` before the
        task is scheduled: clearing inside ``import_sessions`` would discard a
        cancel that arrived while the job was still queued.
        """
        self._cancel.clear()

    def cancel(self) -> bool:
        self._cancel.set()
        return True

    # -- per-session --------------------------------------------------------

    def _import_one(
        self,
        adapter: SourceAdapter,
        session: SourceSession,
        request: ImportRequest,
        result: ImportResult,
    ) -> SessionResult:
        outcome = SessionResult(source=adapter.source, source_id=session.source_id)
        if session.availability != "full":
            outcome.outcome = "metadata_only"
            outcome.warnings.extend(session.warnings)
            return outcome
        try:
            fingerprint = store.fingerprint(self._primary_path(session))
            manifests = self._existing_manifests(session, adapter.source)
            latest = manifests[0] if manifests else None
            if latest is not None:
                directory = Path(str(latest["target_session_dir"]))
                prefix_ok = manifest_store.prefix_sha256(directory) == latest.get(
                    "target_prefix_sha256"
                )
                same_source = latest.get("source_fingerprint") == fingerprint
                committed = {entry.id for entry in Transcript(directory).entries()}
                if latest.get("status") == "partial":
                    # A crash left the rows we managed to write. Re-enter the
                    # SAME directory when the source is still a superset of
                    # those rows; a rewritten source needs a fresh copy.
                    mapped = self._map(adapter, session)
                    if committed - {row.id for row in mapped.rows}:
                        return self._create(
                            adapter,
                            session,
                            self._allocate_directory(
                                adapter.source, session.source_id, force_new=True
                            ),
                            fingerprint,
                            outcome,
                            latest,
                            mapped,
                        )
                    return self._create(
                        adapter,
                        session,
                        directory,
                        fingerprint,
                        outcome,
                        self._carried_predecessor(latest),
                        mapped,
                    )
                if prefix_ok and same_source:
                    outcome.outcome = "skipped_already_imported"
                    outcome.target_session_id = str(latest.get("target_session_id") or "")
                    return outcome
                if prefix_ok and self._source_grew(latest, session):
                    mapped = self._map(adapter, session)
                    # SOURCE-GENERIC rewrite check: a linear append keeps every
                    # already-committed row id in the new mapping. A Claude
                    # /rewind branch or a Codex thread_rolled_back drops rows
                    # from the active path, so those ids disappear. Never
                    # splice a rewritten source into the existing transcript:
                    # allocate a new deterministic target and record the old
                    # one as predecessor.
                    if committed - {row.id for row in mapped.rows}:
                        return self._create(
                            adapter,
                            session,
                            self._allocate_directory(
                                adapter.source, session.source_id, force_new=True
                            ),
                            fingerprint,
                            outcome,
                            latest,
                            mapped,
                        )
                    return self._resume(
                        adapter, session, latest, directory, fingerprint, outcome, mapped
                    )
                # Rotated source or user-edited target: a fresh deterministic
                # copy, never an append to bytes we do not own.
                return self._create(
                    adapter,
                    session,
                    self._allocate_directory(adapter.source, session.source_id, force_new=True),
                    fingerprint,
                    outcome,
                    latest,
                )
            return self._create(
                adapter,
                session,
                self._allocate_directory(adapter.source, session.source_id),
                fingerprint,
                outcome,
                None,
            )
        except _ImportCancelled:
            outcome.outcome = "failed"
            outcome.warnings.append("import cancelled")
            return outcome
        except Exception as exc:  # noqa: BLE001 - one bad session cannot kill the job
            outcome.outcome = "failed"
            outcome.warnings.append(f"{type(exc).__name__}: {exc}")
            result.warnings.append(
                ImportWarning(
                    source=adapter.source,
                    session_id=session.source_id,
                    code="import_failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            return outcome

    def _map(self, adapter: SourceAdapter, session: SourceSession) -> MappedSession:
        events = list(adapter.read_session(session))
        return build_mapped_session(
            events,
            source=adapter.source,
            source_session_id=session.source_id,
            default_created_ts=session.created_ts,
        )

    def _primary_path(self, session: SourceSession) -> Path:
        if session.path.exists():
            return session.path
        for path in session.alternate_paths:
            if path.exists():
                return path
        return session.path

    def _source_grew(self, manifest: dict[str, Any], session: SourceSession) -> bool:
        previous = manifest.get("source_fingerprint") or {}
        previous_size = int(previous.get("size") or 0)
        current = self._primary_path(session).stat().st_size
        if current < previous_size:
            return False
        head_length = min(previous_size, store.FINGERPRINT_PREFIX)
        return store.sha256_prefix(self._primary_path(session), head_length) == previous.get(
            "sha256_prefix"
        )

    def _create(
        self,
        adapter: SourceAdapter,
        session: SourceSession,
        directory: Path,
        fingerprint: dict[str, Any],
        outcome: SessionResult,
        predecessor: dict[str, Any] | None,
        mapped: MappedSession | None = None,
    ) -> SessionResult:
        target_id = directory.name
        outcome.target_session_id = target_id
        mapped = mapped or self._map(adapter, session)
        self._materialise_target(directory, session, mapped, adapter.source)
        manifest_store.write_manifest(
            directory,
            self._manifest(
                source=adapter.source,
                session=session,
                directory=directory,
                fingerprint=fingerprint,
                mapped=mapped,
                status="partial",
                counts=self._counts(mapped),
                predecessor=predecessor,
            ),
        )
        transcript = Transcript(directory, defer_materialise=True)
        committed = {entry.id for entry in transcript.entries()}
        rows = [row for row in mapped.rows if row.id not in committed]
        if rows:
            self._append(transcript, mapped, outcome, rows=rows)
        manifest_store.write_manifest(
            directory,
            self._manifest(
                source=adapter.source,
                session=session,
                directory=directory,
                fingerprint=fingerprint,
                mapped=mapped,
                status="complete",
                counts=self._counts(mapped),
                predecessor=predecessor,
            ),
        )
        self._apply_stats(outcome, mapped)
        outcome.outcome = "imported"
        self._update_index(adapter.source, session, target_id, fingerprint, outcome)
        return outcome

    def _resume(
        self,
        adapter: SourceAdapter,
        session: SourceSession,
        existing: dict[str, Any],
        directory: Path,
        fingerprint: dict[str, Any],
        outcome: SessionResult,
        mapped: MappedSession,
    ) -> SessionResult:
        target_id = directory.name
        committed = {entry.id for entry in Transcript(directory).entries()}
        new_rows = [row for row in mapped.rows if row.id not in committed]
        outcome.target_session_id = target_id
        if not new_rows:
            manifest_store.write_manifest(
                directory,
                self._manifest(
                    source=adapter.source,
                    session=session,
                    directory=directory,
                    fingerprint=fingerprint,
                    mapped=mapped,
                    status="complete",
                    counts=self._counts(mapped),
                    predecessor=self._carried_predecessor(existing),
                ),
            )
            self._apply_stats(outcome, mapped)
            outcome.outcome = "skipped_already_imported"
            return outcome
        transcript = Transcript(directory)
        self._append(transcript, mapped, outcome, rows=new_rows)
        self._apply_stats(outcome, mapped)
        manifest_store.write_manifest(
            directory,
            self._manifest(
                source=adapter.source,
                session=session,
                directory=directory,
                fingerprint=fingerprint,
                mapped=mapped,
                status="complete",
                counts=self._counts(mapped),
                predecessor=self._carried_predecessor(existing),
            ),
        )
        self._update_index(adapter.source, session, target_id, fingerprint, outcome)
        outcome.outcome = "resumed"
        return outcome

    # -- target construction ------------------------------------------------

    def _allocate_directory(self, source: str, source_id: str, *, force_new: bool = False) -> Path:
        sessions = self.config / "sessions"
        sessions.mkdir(parents=True, exist_ok=True, mode=0o700)
        base = target_session_id(source, source_id)
        for attempt in range(100):
            name = base if attempt == 0 else f"{base}-{attempt + 1}"
            directory = sessions / name
            existing = manifest_store.read_manifest(directory)
            if not directory.exists():
                return directory
            if (
                not force_new
                and existing
                and existing.get("source") == source
                and existing.get("source_id") == source_id
                and existing.get("status") in ("partial", "complete")
            ):
                return directory
            if existing is None:
                # A directory we did not create is not ours to resume.
                continue
        raise RuntimeError("could not allocate a target session directory")

    @staticmethod
    def _carried_predecessor(existing: dict[str, Any]) -> dict[str, Any] | None:
        """The predecessor to carry on an in-place update, never self."""
        prior = existing.get("predecessor_target_session_id")
        return {"target_session_id": str(prior)} if prior else None

    def _existing_manifests(self, session: SourceSession, source: str) -> list[dict[str, Any]]:
        """Every target this source session owns, NEWEST first.

        A rotated source or an edited target creates -2, -3, ...; checking only
        the first manifest made every later run allocate another copy. The
        current-state match is chosen by the caller, so this returns the whole
        family (newest first) rather than a single arbitrary member.
        """
        base = target_session_id(source, session.source_id)
        sessions = self.config / "sessions"
        found: list[dict[str, Any]] = []
        for attempt in range(100):
            name = base if attempt == 0 else f"{base}-{attempt + 1}"
            directory = sessions / name
            value = manifest_store.read_manifest(directory)
            if (
                value
                and value.get("source") == source
                and value.get("source_id") == session.source_id
            ):
                value = dict(value)
                value["target_session_id"] = name
                value["target_session_dir"] = str(directory)
                found.append(value)
            if not directory.exists():
                break
        found.reverse()
        return found

    def _option_warnings(
        self,
        request: ImportRequest,
        selected: Sequence[tuple[SourceAdapter, SourceSession]],
    ) -> list[ImportWarning]:
        """Surface options this slice cannot apply, instead of ignoring them."""
        warnings: list[ImportWarning] = []
        if request.include_cleared_segments:
            warnings.append(
                ImportWarning(
                    source="",
                    session_id="",
                    code="option_unsupported",
                    message=(
                        "include_cleared_segments is not supported by this slice; "
                        "only each source's active segment is imported"
                    ),
                )
            )
        if request.include_subagents and any(adapter.source == "claude" for adapter, _ in selected):
            warnings.append(
                ImportWarning(
                    source="claude",
                    session_id="",
                    code="option_unsupported",
                    message=(
                        "Claude sidechain/subagent inclusion is not supported by "
                        "this slice; sidechain records are skipped"
                    ),
                )
            )
        return warnings

    def _materialise_target(
        self,
        directory: Path,
        session: SourceSession,
        mapped: MappedSession,
        source: str,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        ensure_session_created_at(directory, session.created_ts or 0.0)
        title = (
            normalise_title(mapped.title)
            or normalise_title(session.title)
            or normalise_title(mapped.first_user_text)
            or f"{source} {session.source_id}"[:TITLE_MAX]
        )
        from local_operator.resume import write_session_title

        write_session_title(directory, title, user_set=mapped.title_user_set, past_names=[])

    def _manifest(
        self,
        *,
        source: str,
        session: SourceSession,
        directory: Path,
        fingerprint: dict[str, Any],
        mapped: MappedSession,
        status: str,
        counts: dict[str, int],
        predecessor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "v": 1,
            "source": source,
            "source_id": session.source_id,
            "source_path": str(session.path),
            "alternate_paths": [str(path) for path in session.alternate_paths],
            "fork_parent_source_id": session.fork_parent_source_id,
            "source_fingerprint": fingerprint,
            "source_created_at": session.created_ts,
            "imported_at": time.time(),
            "status": status,
            "active_leaf_source_id": mapped.active_leaf_source_id,
            "active_branch_source_ids": list(mapped.active_branch_source_ids),
            "last_source_offset": fingerprint.get("size", 0),
            "imported_entries": len(mapped.rows),
            "counts": counts,
            "compaction_anchors": mapped.compaction_anchors,
            "cwd": session.cwd,
            "parent_source_session_id": session.parent_source_id,
            "model_history": mapped.model_history,
            "warnings": [*session.warnings, *mapped.warnings],
            "capture_memory": False,
            "target_prefix_sha256": manifest_store.prefix_sha256(directory),
            "predecessor_target_session_id": (
                str(predecessor.get("target_session_id") or "") or None
                if predecessor is not None
                else None
            ),
        }

    @staticmethod
    def _apply_stats(outcome: SessionResult, mapped: MappedSession) -> None:
        outcome.messages = int(mapped.stats["messages"])
        outcome.tool_calls = int(mapped.stats["tool_calls"])
        outcome.tool_results = int(mapped.stats["tool_results"])
        outcome.compactions = int(mapped.stats["compactions"])
        outcome.redactions = int(mapped.stats["redactions"])
        outcome.skipped = int(mapped.stats["skipped"])
        outcome.warnings.extend(mapped.warnings)

    def _counts(self, mapped: MappedSession) -> dict[str, int]:
        return {
            "messages": mapped.stats["messages"],
            "tool_calls": mapped.stats["tool_calls"],
            "tool_results": mapped.stats["tool_results"],
            "duplicate_results_skipped": mapped.stats.get("duplicate_results", 0),
            "compactions": mapped.stats["compactions"],
            "skipped": mapped.stats["skipped"],
            "redactions": mapped.stats["redactions"],
        }

    def _append(
        self,
        transcript: Transcript,
        mapped: MappedSession,
        outcome: SessionResult,
        *,
        rows: Sequence[Any] | None = None,
    ) -> None:
        selected = list(rows if rows is not None else mapped.rows)
        batch: list[Any] = []
        size = 0

        async def flush() -> None:
            nonlocal batch, size
            if not batch:
                return
            if self._cancel.is_set():
                raise _ImportCancelled("import cancelled between write batches")
            batch_ids = {row.id for row in batch}
            await transcript.append_imported(
                batch,
                messages={key: value for key, value in mapped.messages.items() if key in batch_ids},
            )
            batch = []
            size = 0

        async def run() -> None:
            nonlocal size
            index_by_id = {row.id: index for index, row in enumerate(selected)}
            index = 0
            while index < len(selected):
                end = min(index + BATCH_ROWS, len(selected))
                # Never end a batch before an anchor it references.
                changed = True
                while changed:
                    changed = False
                    for row in selected[index:end]:
                        if row.type != ENTRY_COMPACTION:
                            continue
                        anchor = str(row.payload.get("first_kept_entry_id") or "")
                        position = index_by_id.get(anchor)
                        if position is not None and position >= end:
                            end = position + 1
                            changed = True
                for row in selected[index:end]:
                    batch.append(row)
                    size += len(row.to_json())
                if len(batch) >= BATCH_ROWS or size >= BATCH_BYTES:
                    await flush()
                index = end
            await flush()

        asyncio.run(run())

    def _update_index(
        self,
        source: str,
        session: SourceSession,
        target_id: str,
        fingerprint: dict[str, Any],
        outcome: SessionResult,
    ) -> None:
        index = manifest_store.read_index(self.config)
        sessions = index.setdefault("sessions", {})
        sessions[f"{source}:{session.source_id}"] = {
            "target_session_id": target_id,
            "source": source,
            "source_id": session.source_id,
            "updated_ts": session.updated_ts,
            "fingerprint": fingerprint,
            "outcome": outcome.outcome,
        }
        manifest_store.write_index(self.config, index)

    # -- memory -------------------------------------------------------------

    def _sessions_by_source(
        self, selected: Iterable[tuple[SourceAdapter, SourceSession]]
    ) -> dict[str, list[SourceSession]]:
        grouped: dict[str, list[SourceSession]] = {}
        for adapter, session in selected:
            grouped.setdefault(adapter.source, []).append(session)
        return grouped

    def _import_memory(
        self,
        request: ImportRequest,
        selected: Sequence[tuple[SourceAdapter, SourceSession]],
        imported_by_thread: dict[str, str],
        imported_by_cwd: dict[str, list[str]],
        session_dirs: dict[str, Path],
        result: ImportResult,
    ) -> None:
        for source, sessions in self._sessions_by_source(selected).items():
            adapter = self.adapters[source]
            try:
                records = adapter.read_memory(sessions)
                outcome = memory_import.import_memory(
                    self.config,
                    source,
                    records,
                    target_by_thread=imported_by_thread,
                    target_by_cwd=imported_by_cwd,
                    session_dirs=session_dirs,
                    importing_session_id=self.importing_session_id,
                    include_instructions=request.include_instructions,
                    memory_status=request.memory_status,
                )
            except Exception as exc:  # noqa: BLE001 - memory never fails a conversation import
                result.warnings.append(
                    ImportWarning(
                        source=source,
                        session_id="",
                        code="memory_import_failed",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            result.memory_imported += outcome.imported
            result.memory_skipped += outcome.skipped
            if outcome.review_path:
                result.memory_review_path = outcome.review_path
            if outcome.instruction_path and not result.instruction_review_path:
                result.instruction_review_path = outcome.instruction_path
            for warning in outcome.warnings:
                result.warnings.append(
                    ImportWarning(
                        source=source,
                        session_id="",
                        code="memory_notice",
                        message=warning,
                    )
                )
