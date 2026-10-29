#!/usr/bin/env python3
"""One-shot migration: split ``i18n/ledger.json`` into per-locale shards.

WHY. The aggregate was one ``{"entries": [...]}`` array every locale PR
appended to, so concurrent locale work conflicted on the same tail (and on
every other locale PR). Sharding per locale means two locales edit different
files and never conflict; the aggregate remains as a generated compatibility
mirror (``generate.py`` re-renders and drift-checks it).

PRESERVATION RULES (both measured against the pre-migration tree):

* Append-only history is kept verbatim — rows for one namespace at several
  ``source_sha256`` values are legitimate supersession history, and deduping
  on ``(locale, namespace)`` or ``(locale, namespace, None)`` corrupts it.
* Each shard preserves the aggregate's relative row order FOR THAT LOCALE,
  because eligibility is last-wins per ``(locale, namespace)`` in file order.
  Cross-locale interleaving never fed the gate, so grouping by locale is
  semantics-preserving.

Idempotent: re-running rewrites the same shard bytes. Run once, then
``generate.py`` to re-render the aggregate mirror.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "i18n" / "ledger.json"
SHARDS = REPO / "i18n" / "ledger"


def shard_entries(aggregate: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by locale, preserving per-locale file order exactly."""
    by_locale: dict[str, list[dict[str, Any]]] = {}
    for row in aggregate.get("entries", []):
        if not isinstance(row, dict):
            continue
        locale = row.get("locale")
        if isinstance(locale, str):
            by_locale.setdefault(locale, []).append(row)
    return by_locale


def render_shard(locale: str, rows: list[dict[str, Any]]) -> str:
    payload = {"schema": 1, "locale": locale, "entries": rows}
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER, help="aggregate to read")
    parser.add_argument("--out", type=Path, default=SHARDS, help="shard directory to write")
    args = parser.parse_args(argv)

    aggregate = json.loads(args.ledger.read_text(encoding="utf-8"))
    by_locale = shard_entries(aggregate)
    args.out.mkdir(parents=True, exist_ok=True)
    for locale in sorted(by_locale):
        path = args.out / f"{locale}.json"
        path.write_text(render_shard(locale, by_locale[locale]), encoding="utf-8")
        print(f"wrote {path} ({len(by_locale[locale])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
