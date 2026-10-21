"""The TUI's durable "group the sidebar by primary tag" preference.

WHY A FILE AND NOT ``config.yml``. The setting is a per-terminal presentation
choice with exactly one writer — the TUI's own ``/tags on|off`` — and adding a
key to the audited ``wire.settings`` catalogue voids every translated locale's
ledger hash (the shipped set is computed from those hashes). The sidebar's pin
index already keeps deliberate, durable view state in a small config-root file
for the same reason, so this follows that precedent rather than spending the
translation audit on one boolean.

A SINGLE BOOLEAN, last writer wins. There is no read-modify-write window to
protect the way the pin index needs a lock: two ``lop`` processes toggling the
preference within one window can only disagree about which value is stored, and
the next reader takes the file as it finds it. An unreadable or malformed file
reads as the shipped default, OFF — grouping adds headings the flat list never
had, and a presentation mode has to be opted into.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

#: Dot-free so a config-directory scan can tell it apart from a lock or cache.
TAG_GROUPS_FILE = "sidebar-tags.json"

_DEFAULT = False


def _file(root: Path) -> Path:
    return Path(root) / TAG_GROUPS_FILE


def read_tag_groups(root: Path) -> bool:
    """Whether sidebar tag grouping is on. Any read failure means ``False``."""
    try:
        raw = json.loads(_file(root).read_text())
    except (OSError, ValueError):
        return _DEFAULT
    if not isinstance(raw, dict) or raw.get("version") != 1:
        return _DEFAULT
    value = raw.get("group_by_tag")
    return value if isinstance(value, bool) else _DEFAULT


def write_tag_groups(root: Path, enabled: bool) -> bool:
    """Store the preference atomically; True iff the bytes on disk changed."""
    path = _file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"version": 1, "group_by_tag": bool(enabled)}, sort_keys=True) + "\n"
    try:
        if path.read_text() == data:
            return False
    except OSError:
        pass
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".sidebar-tags-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return True
