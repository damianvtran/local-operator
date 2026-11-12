"""The viewer's half of a peer session's Files panel: listing, content, download.

docs/design/remote-files-parity.md §2-§3. The owner answers the index
(``network/file_index.py``'s ``net_session_files``) through this device's relay
(``peer_session_files``); bytes go through the same fetch-on-miss as transcript
attachments (``desktop_sessions.peer_blob``/``_fetch_peer``), so statuses follow
§1's table: 504 retryable, 409 permanent.

``remote``/``origin_device`` are stamped HERE from the catalogue row, never
trusted from the owner. A remote row's ``ref.path`` is the owner's DISPLAY path:
nothing on this device ever opens it (I3-1/I5-1); the only local path a remote
file gets is the one :func:`download` lands.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.network.types import MeshRefusal

#: The download copy's block: RAM stays flat however large the file is.
COPY_BLOCK = 1024 * 1024
#: ``name (1).ext`` … ``name (99).ext``, then refuse.
MAX_NUMBERED_COPIES = 99
#: Passive types the content route serves as themselves; anything else (HTML, SVG
#: scripts…) is served as opaque bytes, beside ``nosniff`` and a sandbox CSP.
_PASSIVE_TEXT = frozenset(
    {"application/pdf", "text/plain", "text/markdown", "text/csv", "application/json"}
)


def _row_for(config_dir: Path, session_id: str) -> Any:
    from local_operator.session.remote_open import remote_row_for

    row = remote_row_for(session_id, config_dir)
    if row is None:
        raise KeyError("Unknown session")
    return row


def device_of(row: Any) -> dict[str, str]:
    from local_operator.resume import UNNAMED_DEVICE

    return {
        "id": str(row.owner_device or ""),
        "name": str(row.owner_device_name or row.owner_label or UNNAMED_DEVICE),
    }


def _unreachable(device: str) -> HTTPException:
    return HTTPException(
        504,
        _wd.session_files_owner_unreachable(device=device).envelope(
            code="session_files_owner_unreachable"
        ),
    )


async def ask(config_dir: Path, session_id: str, **fields: Any) -> tuple[dict[str, Any], Any]:
    """One ``peer_session_files`` phase for ``session_id``: ``(detail, catalogue row)``.

    Raises :class:`MeshRefusal` with the relay's code; ``KeyError`` when the id is no
    peer's session.
    """
    from local_operator.network.file_index import INDEX_CLIENT_TIMEOUT_S
    from local_operator.server.utils import desktop_mesh

    row = await asyncio.to_thread(_row_for, config_dir, session_id)
    detail = await asyncio.to_thread(
        desktop_mesh._call,  # noqa: SLF001 — the desktop's one relay seam
        config_dir,
        "peer_session_files",
        timeout=INDEX_CLIENT_TIMEOUT_S,
        peer=row.owner_device,
        session_id=session_id,
        **fields,
    )
    return detail, row


def _stamp(items: list[dict[str, Any]], device: dict[str, str]) -> None:
    for item in items:
        item["remote"] = True
        item["origin_device"] = dict(device)
        item.setdefault("remote_read", None)
        item.setdefault("remote_unreadable", None)


async def listing(
    config_dir: Path, session_id: str, *, index: str, query: dict[str, Any], empty: dict[str, Any]
) -> dict[str, Any]:
    """The peer's listing with the remote fields, or ``empty`` marked why it is empty."""
    from local_operator.resume import UNNAMED_DEVICE

    try:
        detail, row = await ask(config_dir, session_id, phase="list", index=index, query=query)
    except MeshRefusal as refusal:
        row = await asyncio.to_thread(_row_for, config_dir, session_id)
        device = device_of(row)["name"] or UNNAMED_DEVICE
        if refusal.code == "predates_file_index":
            return {
                **empty,
                "state": "unsupported",
                "unsupported_reason": "peer_update_required",
                "device_name": device,
            }
        if refusal.code in ("peer_unreachable", "relay_unavailable"):
            raise _unreachable(device) from None
        if refusal.code == "bad_cursor":
            raise HTTPException(
                422, _wd.session_files_bad_cursor().envelope(code="session_files_bad_cursor")
            ) from None
        from local_operator.server.utils.desktop_sessions import (
            PeerAttachmentUnavailable,
        )

        raise PeerAttachmentUnavailable(session_id, row) from None
    device = device_of(row)
    _stamp(list(detail.get("rows") or detail.get("items") or []), device)
    detail["unsupported_reason"] = None
    detail["device_name"] = device["name"]
    detail["remote"] = True
    return detail


async def _resolve(config_dir: Path, session_id: str, file_id: str) -> tuple[Any, Any, Any]:
    """``(blob, row dict, catalogue row)`` for one readable remote row, fetched if needed."""
    from local_operator.server.utils.desktop_sessions import _fetch_peer, peer_blob

    try:
        detail, row = await ask(config_dir, session_id, phase="row", file_id=file_id)
    except MeshRefusal as refusal:
        row = await asyncio.to_thread(_row_for, config_dir, session_id)
        if refusal.code == "not_found":
            raise HTTPException(
                404,
                _wd.session_files_file_not_found().envelope(code="session_files_file_not_found"),
            ) from None
        if refusal.code in ("peer_unreachable", "relay_unavailable"):
            from local_operator.server.utils.desktop_sessions import (
                PeerAttachmentRetryable,
            )

            raise PeerAttachmentRetryable(session_id, row, pending=False) from None
        if refusal.code == "predates_file_index":
            from local_operator.server.utils.desktop_sessions import (
                PeerAttachmentOutdated,
            )

            raise PeerAttachmentOutdated(session_id, row) from None
        from local_operator.server.utils.desktop_sessions import (
            PeerAttachmentUnavailable,
        )

        raise PeerAttachmentUnavailable(session_id, row) from None
    file_row = detail.get("row") or {}
    source = detail.get("source")
    reason = file_row.get("remote_unreadable")
    if reason or not isinstance(source, dict):
        raise HTTPException(
            409,
            _wd.session_files_remote_unreadable(
                reason=str(reason or "outside_session_files"), device=device_of(row)["name"]
            ).envelope(code="session_files_remote_unreadable"),
        )
    member_id = str(source.get("session_id") or session_id)
    if source.get("attachment"):
        blob = await peer_blob(
            config_dir, session_id, row, str(source["attachment"]), member_id=member_id
        )
    else:
        path = str(source.get("path") or "")
        key = f"path:{path}:{file_row.get('size_bytes')}:{file_row.get('modified_at')}"
        blob = await _fetch_peer(
            config_dir,
            session_id,
            row,
            member_id,
            {"path": path},
            key,
            tuple(dict.fromkeys((member_id, session_id))),
        )
    return blob, file_row, row


def _open(blob: Any) -> Any:
    from local_operator.remote_files import cache as remote_cache

    handle = open(blob.path, "rb")  # noqa: SIM115 — the response body closes it
    if blob.source == "cache":
        remote_cache.touch(blob.path)
    return handle


async def content(config_dir: Path, session_id: str, file_id: str, range_header: str | None):
    """The bytes of one remote row, streamed from an fd opened before answering."""
    from local_operator.server.routes.desktop_sessions import (
        ATTACHMENT_SERVABLE_MIME_TYPES,
    )
    from local_operator.server.utils import byte_range

    blob, _file_row, _row = await _resolve(config_dir, session_id, file_id)
    handle = await asyncio.to_thread(_open, blob)
    mime = blob.mime if blob.mime in ATTACHMENT_SERVABLE_MIME_TYPES | _PASSIVE_TEXT else None
    return byte_range.handle_response(
        handle,
        media_type=mime or "application/octet-stream",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox",
            "Cache-Control": "no-store",
        },
        range_header=range_header,
    )


def downloads_dir(config_dir: Path) -> Path:
    """``~/Downloads`` when it is a directory, else ``<config>/downloads`` (no config key)."""
    home_downloads = Path.home() / "Downloads"
    if home_downloads.is_dir():
        return home_downloads
    fallback = Path(config_dir) / "downloads"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


class DownloadRefused(Exception):
    def __init__(self, status: int, envelope: dict[str, Any]) -> None:
        super().__init__(envelope.get("code"))
        self.status = status
        self.envelope = envelope


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(COPY_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _candidates(name: str) -> list[str]:
    stem, ext = os.path.splitext(name)
    return [name, *(f"{stem} ({n}){ext}" for n in range(1, MAX_NUMBERED_COPIES + 1))]


def land(blob: Any, *, name: str, folder: Path, device: str) -> dict[str, Any]:
    """Copy ``blob`` into ``folder`` verified and without overwriting. Blocking.

    I3-2: an ``O_EXCL`` temp in the target folder, a streamed 1 MiB-block copy from an
    fd opened once, ``fsync``, the full sha256 checked against the blob's content
    address, then ``os.link`` (a no-replace publish) and an unlink of the temp. A
    mismatch lands nothing. A same-bytes file at a candidate name is reused
    (``created: false``); a different one moves the landing to ``name (n).ext``.
    """
    expected = Path(blob.path).stem
    with open(blob.path, "rb") as source:
        temp = folder / f".{name}.{secrets.token_hex(6)}.part"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as sink:
                while block := source.read(COPY_BLOCK):
                    digest.update(block)
                    sink.write(block)
                    size += len(block)
                sink.flush()
                os.fsync(sink.fileno())
            hexdigest = digest.hexdigest()
            if hexdigest[:32] != expected:
                raise DownloadRefused(
                    409,
                    _wd.session_files_download_mismatch(device=device).envelope(
                        code="session_files_download_mismatch"
                    ),
                )
            for candidate in _candidates(name):
                final = folder / candidate
                if final.exists():
                    try:
                        same = final.is_file() and _sha256_file(final) == hexdigest
                    except OSError:
                        same = False
                    if same:
                        return _receipt(final, size, hexdigest, created=False)
                    continue
                try:
                    os.link(temp, final)
                except FileExistsError:
                    continue
                return _receipt(final, size, hexdigest, created=True)
            raise DownloadRefused(
                409,
                _wd.session_files_download_exhausted(name=name, folder=str(folder)).envelope(
                    code="session_files_download_exhausted"
                ),
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                temp.unlink()


def _receipt(path: Path, size: int, hexdigest: str, *, created: bool) -> dict[str, Any]:
    return {
        "path": str(path),
        "name": path.name,
        "bytes": size,
        "sha256": f"sha256:{hexdigest}",
        "created": created,
    }


async def download(config_dir: Path, session_id: str, file_id: str) -> dict[str, Any]:
    """Fetch one remote row and land it on THIS device; the receipt names where."""
    from local_operator.session_files import service

    blob, file_row, row = await _resolve(config_dir, session_id, file_id)
    device = device_of(row)
    raw_name = str(file_row.get("name") or "")
    name = service.sanitise_name(raw_name, Path(blob.path).stem, blob.mime)
    if service._bare_name_denied(name) or (  # noqa: SLF001 — the service's name rule
        raw_name and service._bare_name_denied(os.path.basename(raw_name))  # noqa: SLF001
    ):
        raise HTTPException(
            409,
            _wd.session_files_sensitive_name(name=name).envelope(
                code="session_files_sensitive_name"
            ),
        )
    try:
        folder = await asyncio.to_thread(downloads_dir, config_dir)
        receipt = await asyncio.to_thread(
            land, blob, name=name, folder=folder, device=device["name"]
        )
    except DownloadRefused as refusal:
        raise HTTPException(refusal.status, refusal.envelope) from None
    except OSError as error:
        raise HTTPException(
            409,
            _wd.session_files_materialise_failed(reason=error.strerror or "write failed").envelope(
                code="session_files_download_failed"
            ),
        ) from None
    return {**receipt, "origin_device": device}
