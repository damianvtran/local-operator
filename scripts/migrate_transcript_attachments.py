#!/usr/bin/env python3
"""Externalize inline image payloads already sitting in session transcripts.

Transcripts written before the attachment store existed carry their images
as base64 inline in ``transcript.jsonl``. On a screenshot-heavy install that
payload dominates the session store (measured: 102 of 134 MB across 142
sessions) and is exactly duplicated across sessions (~20 MB of identical
screenshots stored repeatedly). The live write path externalizes new images
as they are appended; this script applies the same transform to history.

Per transcript: every image content block over the inline floor is written
to the content-addressed store (``<config>/attachments/``) and replaced with
a reference. Rewrites are crash-safe — the new file is written beside the
old and ``os.replace``d over it, and a transcript is only rewritten when the
saving clears a floor, so a no-op run touches nothing.

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
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.attachments import AttachmentStore  # noqa: E402
from local_operator.session.transcript import _ATTACHMENT_FLOOR_BYTES  # noqa: E402


def migrate_transcript(path: Path, store: AttachmentStore, *, dry_run: bool) -> tuple[int, int]:
    """Externalize one transcript's inline images.

    Returns ``(bytes_before, bytes_after)``. Rewrites atomically; a
    transcript with nothing over the floor is returned untouched.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    moved = 0
    changed = False
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
                moved += len(data)
                block.pop("data", None)
                block["attachment"] = ref.digest
                block["mime_type"] = ref.mime_type
                changed = True
        out.append(json.dumps(entry, separators=(",", ":")) + "\n" if changed else line)

    before = path.stat().st_size
    if not changed or dry_run:
        return before, before - moved + moved // 2  # rough: ref replaces payload

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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=config_dir() / "sessions",
        help="directory of session transcripts (default: the live install's)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    args = parser.parse_args()

    store = AttachmentStore()
    total_before = 0
    total_after = 0
    touched = 0
    transcripts = sorted(args.sessions_dir.glob("*/transcript.jsonl"))
    for path in transcripts:
        before, after = migrate_transcript(path, store, dry_run=args.dry_run)
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
    print(f"transcript bytes    : {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")
    print(f"attachment store    : {store_bytes / 1e6:.1f} MB in {store.root}")
    saved = (total_before - total_after) - (0 if args.dry_run else store_bytes)
    print(f"net disk saving     : ~{saved / 1e6:.1f} MB {'(projected)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
