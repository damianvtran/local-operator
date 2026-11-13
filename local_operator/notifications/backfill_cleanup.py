"""Inspect or undo the one-time v0.102.2 remote-backfill cleanup.

``python -m local_operator.notifications.backfill_cleanup [--dry-run] [--undo]``
(for the installed runtime: ``~/.local/share/uv/tools/local-operator/bin/python
-m local_operator.notifications.backfill_cleanup``).

WHY THIS EXISTS (operator rule, 2026-11-13). Opening the notifications store
marks READ, once, every unread ``remote-done:``/``remote-notice:`` row the
v0.102.2 backfill flooded in (``NotificationStore._cleanup_remote_backfill``).
The operator asked for that cleanup to be reversible: this is the undo. It
restores only the rows the cleanup itself marked and that nobody has touched
since (``NotificationStore.undo_remote_backfill_cleanup``); it never deletes
and never touches parks, asks or any other item.

A MAINTENANCE ENTRY POINT, not product surface: no ``lop`` subcommand, no
desktop control, so its few report lines are developer output, not catalogued
copy (the ``play`` module's ``python -m`` precedent). ``--config-dir`` defaults
to the runtime's own resolution, so it reads the store the harness writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_operator.notifications.backfill_cleanup")
    parser.add_argument("--config-dir", default="", help="config root (default: the runtime's)")
    parser.add_argument("--undo", action="store_true", help="restore the rows the cleanup marked")
    parser.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    args = parser.parse_args(argv)

    from local_operator.notifications.store import NotificationStore

    if args.config_dir:
        root = Path(args.config_dir).expanduser()
    else:
        from local_operator.paths import config_dir

        root = config_dir()
    store = NotificationStore(root)
    record = store.remote_backfill_cleanup_record() or {}
    marked = record.get("rows")
    rows: dict[str, object] = marked if isinstance(marked, dict) else {}
    print(f"store: {store.path}")  # i18n: ignore maintenance output, not product copy
    print(f"cleanup marked read: {len(rows)} rows")  # i18n: ignore maintenance output
    if record.get("undone_at"):
        print(f"undone at (epoch ms): {record['undone_at']}")  # i18n: ignore maintenance output
    if args.undo:
        restored = store.undo_remote_backfill_cleanup(dry_run=args.dry_run)
        verb = "would restore" if args.dry_run else "restored"
        print(f"{verb}: {restored} rows")  # i18n: ignore maintenance output
    return 0


if __name__ == "__main__":
    sys.exit(main())
