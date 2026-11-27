"""One opener for the slice-L3 Guard SQLite files (house rules of ``guard/outbox.py``).

WHY A SHARED OPENER. L3 adds three small files beside L1's outbox and L2a's
consent store, and each must obey the same rules: construction does no I/O
(a non-Team install builds no Guard file), the directory is re-asserted 0700
and the file 0600 on EVERY open, a file that is not SQLite or carries an
unknown schema generation is refused -- never re-created over the member's
data -- and the DDL runs once, inside ``BEGIN IMMEDIATE`` so two lop
processes cannot both initialise it. Writing that four times invites the
copy that forgets the chmod.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Final

_BUSY_TIMEOUT_S: Final = 5.0


class GuardStoreRefused(Exception):
    """A Guard L3 store is damaged or from an unknown generation; nothing was written."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def open_store(path: Path, *, version: int, ddl: Sequence[str]) -> sqlite3.Connection:
    """Open (creating on first use) ``path`` in autocommit mode with ``ddl`` at ``version``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    existed = path.exists() and path.stat().st_size > 0
    conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        os.chmod(path, 0o600)
        try:
            found = conn.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise GuardStoreRefused("store_damaged", f"{path.name} is not a database") from exc
        if existed and found != version:
            raise GuardStoreRefused(
                "store_damaged", f"{path.name} has an unknown schema generation; refusing it"
            )
        if not found:
            conn.execute("BEGIN IMMEDIATE")
            if not conn.execute("PRAGMA user_version").fetchone()[0]:
                for statement in ddl:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {int(version)}")
            conn.execute("COMMIT")
    except BaseException:
        conn.close()
        raise
    return conn


__all__ = ["GuardStoreRefused", "open_store"]
