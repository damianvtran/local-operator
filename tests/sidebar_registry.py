"""Session-only discovery re-keying for fixtures with same-process owners."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def rekey_session_registry(monkeypatch: Any) -> None:
    """Let multiple fixture sessions share a PID without changing other registries.

    The runtime registry also publishes peers and serve daemons, whose protocol
    deliberately has no ``session_id``. Only session records need the fixture's
    alternate key; capturing both original operations keeps every other namespace
    on its real publication/cleanup path, including its return values and errors.
    """
    from local_operator.session.runtime import registry

    original_publish = registry.publish
    original_unpublish = registry.unpublish

    def publish(record: Any, root: Path | None = None, dirname: str = registry.RUN_DIRNAME) -> Path:
        if dirname != registry.RUN_DIRNAME:
            return original_publish(record, root, dirname=dirname)
        directory = registry.ensure_run_dir(root, dirname)
        record.heartbeat_at = time.time()
        target = directory / f"{record.pid}-{record.session_id}.json"
        # Reuse the real atomic, 0600 writer rather than maintaining another
        # staged-write implementation in each assembled-app fixture.
        registry._staged_write(target, record.to_json(), prefix=f".{record.pid}.")
        return target

    def unpublish(pid: int, root: Path | None = None, dirname: str = registry.RUN_DIRNAME) -> None:
        if dirname != registry.RUN_DIRNAME:
            return original_unpublish(pid, root, dirname=dirname)
        # Session owners share one PID here: one owner's shutdown must not erase
        # another's record. The fixture owns the re-keyed files' final cleanup.
        return None

    monkeypatch.setattr(registry, "publish", publish)
    monkeypatch.setattr(registry, "unpublish", unpublish)
