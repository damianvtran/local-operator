#!/usr/bin/env python3
"""Externalize inline image payloads already sitting in session transcripts.

Transcripts written before the attachment store existed carry their images
as base64 inline in ``transcript.jsonl``. On a screenshot-heavy install that
payload dominates the session store (measured: 102 of 134 MB across 142
sessions) and is exactly duplicated across sessions (~20 MB of identical
screenshots stored repeatedly). The live write path externalizes new images
as they are appended; this script applies the same transform to history.

**Quiescent install only.** The rewrite is a read-transform-replace of the
whole file. A line a live session appends between the read and the replace
would be silently overwritten, so a transcript whose mtime is newer than
:data:`QUIESCENT_S` is skipped (and the script refuses to rewrite anything
if any session looks live, unless ``--force``). Run this with no
local-operator sessions open.

Per transcript: every image content block over the inline floor is written
to the content-addressed store (``<config>/attachments/``) and replaced with
a reference. Rewrites are crash-safe — the new file is written beside the
old and ``os.replace``d over it, and a transcript is only rewritten when
something actually changed, so a no-op run touches nothing.

Nothing is deleted: the bytes move from ``sessions/<id>/transcript.jsonl``
to ``attachments/<digest>.bin``, deduplicated by content. Both remain on
disk and replay produces the same messages either way.

Run:
    .venv/bin/python scripts/migrate_transcript_attachments.py --dry-run
    .venv/bin/python scripts/migrate_transcript_attachments.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.attachments import AttachmentStore  # noqa: E402
from local_operator.session.transcript import _ATTACHMENT_FLOOR_BYTES  # noqa: E402

#: A transcript whose mtime is newer than this many seconds is treated as
#: live and skipped. Sessions here write every few seconds on a turn, so
#: five minutes is well past "the user just closed the window" and well
#: short of "this is history we can rewrite".
QUIESCENT_S = 5 * 60


class SkipLive(Exception):
    """Raised when a transcript looks like a live session is still appending."""


def migrate_transcript(path: Path, store: AttachmentStore, *, dry_run: bool) -> tuple[int, int]:
    """Externalize one transcript's inline images.

    Returns ``(bytes_before, bytes_after)``. Rewrites atomically; a
    transcript with nothing over the floor is returned untouched. Raises
    :class:`SkipLive` if the file's mtime moved between the read and the
    replace — that is the window in which a live session would have
    appended a line this rewrite would silently drop.
    """
    before_stat = path.stat()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    any_changed = False
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # A malformed line is carried through verbatim, the same policy
            # the live reader applies: drop nothing, break nothing.
            out.append(line)
            continue
        line_changed = False
        content = entry.get("payload", {}).get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                data = block.get("data")
                if not isinstance(data, str) or len(data) < _ATTACHMENT_FLOOR_BYTES:
                    continue
                ref = store.put(data, str(block.get("mime_type", "image/png")))
                if ref is None:
                    continue
                block.pop("data", None)
                block["attachment"] = ref.digest
                block["mime_type"] = ref.mime_type
                line_changed = True
        if line_changed:
            any_changed = True
            out.append(json.dumps(entry, separators=(",", ":")) + "\n")
        else:
            out.append(line)

    before = before_stat.st_size
    if not any_changed or dry_run:
        return before, before

    # Refuse to overwrite a file that grew or was rewritten while we
    # were transforming it: that growth is a live session's append.
    after_stat = path.stat()
    if after_stat.st_mtime != before_stat.st_mtime or after_stat.st_size != before_stat.st_size:
        raise SkipLive(str(path))

    payload = "".join(out)
    # Same-directory temp + os.replace: an interrupted run leaves the
    # original transcript intact, exactly like Transcript.compact_file.
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".migrate",
        delete=False,
        encoding="utf-8",
    ) as stream:
        tmp = Path(stream.name)
        stream.write(payload)
    os.replace(tmp, path)
    return before, path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=config_dir() / "sessions",
        help="directory of session transcripts (default: the live install's)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite even transcripts whose mtime is newer than the "
        "quiescence window (unsafe if a session is still appending)",
    )
    args = parser.parse_args()

    store = AttachmentStore()
    total_before = 0
    total_after = 0
    touched = 0
    skipped_live = 0
    transcripts = sorted(args.sessions_dir.glob("*/transcript.jsonl"))
    now = time.time()
    for path in transcripts:
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age < QUIESCENT_S and not args.force and not args.dry_run:
            skipped_live += 1
            print(f"  {path.parent.name}: skipped (mtime {age:.0f}s ago; looks live)")
            continue
        try:
            before, after = migrate_transcript(path, store, dry_run=args.dry_run)
        except SkipLive:
            skipped_live += 1
            print(f"  {path.parent.name}: skipped (changed while we were reading it)")
            continue
        total_before += before
        total_after += after
        if after < before:
            touched += 1
            print(f"  {path.parent.name}: {before / 1e6:6.1f} MB -> {after / 1e6:6.1f} MB")

    store_bytes = (
        sum(f.stat().st_size for f in store.root.glob("*.bin")) if store.root.is_dir() else 0
    )
    print()
    print(f"transcripts scanned : {len(transcripts)}")
    print(f"transcripts changed : {touched}")
    print(f"transcripts skipped : {skipped_live} (looked live)")
    print(f"transcript bytes    : {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")
    print(f"attachment store    : {store_bytes / 1e6:.1f} MB in {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
