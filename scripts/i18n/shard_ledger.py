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

TWO MODES, because there are two situations and confusing them loses audits:

* **First migration** (no shards yet): ``shard_ledger.py`` splits the
  aggregate into shards. Idempotent.
* **A branch that appended to the aggregate** (shards already exist — the
  state after rebasing an open locale PR onto a sharded ``main``):
  ``shard_ledger.py --fold-stranded`` moves the aggregate-only rows into
  their locale's shard, preserving aggregate order, so they are not stranded.
  Without the flag the script REFUSES, because rewriting shards from the
  aggregate would drop every shard row the aggregate does not carry.

After either mode, run ``scripts/i18n/generate.py` to re-render the mirror.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "i18n" / "ledger.json"
SHARDS = REPO / "i18n" / "ledger"


def _rows_by_locale(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Rows grouped by locale, preserving their order in ``rows``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("locale"), str):
            grouped.setdefault(row["locale"], []).append(row)
    return grouped


def render_shard(locale: str, rows: list[dict[str, Any]]) -> str:
    payload = {"schema": 1, "locale": locale, "entries": rows}
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _read_shards(directory: Path) -> dict[str, list[dict[str, Any]]]:
    shards: dict[str, list[dict[str, Any]]] = {}
    if not directory.is_dir():
        return shards
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("entries", []) if isinstance(data, dict) else []
        shards[path.stem] = [row for row in rows if isinstance(row, dict)]
    return shards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER,
        help="aggregate to read",  # i18n: ignore operator tooling diagnostic
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SHARDS,
        help="shard directory to write",  # i18n: ignore operator tooling diagnostic
    )
    parser.add_argument(
        "--fold-stranded",
        action="store_true",
        help=(
            # i18n: ignore operator tooling diagnostic (English-only)
            "move aggregate-only rows into their locale's shard (the rebase case: "
            "a branch that appended to the aggregate after sharding landed)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            # i18n: ignore operator tooling diagnostic (English-only)
            "re-split the aggregate over EXISTING shards "
            "(drops shard-only rows; not a rebase fix)"
        ),
    )
    args = parser.parse_args(argv)

    aggregate = json.loads(args.ledger.read_text(encoding="utf-8"))
    aggregate_rows = [row for row in aggregate.get("entries", []) if isinstance(row, dict)]
    existing = _read_shards(args.out)

    if existing and not (args.fold_stranded or args.force):
        print(
            # i18n: ignore operator tooling diagnostic (English-only)
            "shard_ledger.py: shards already exist under "
            f"{args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}.\n"
            "Re-splitting the aggregate now would DROP every shard row the aggregate does not\n"
            "carry — normally the newest audits. Pick the situation you are in:\n"
            "  rebased a branch that appended to i18n/ledger.json  ->  --fold-stranded\n"
            "  deliberately re-splitting from scratch (reviewed)   ->  --force"
        )
        return 2

    if args.fold_stranded:
        stranded_by_locale = _stranded_by_locale(aggregate_rows, existing)
        if not any(stranded_by_locale.values()):
            print(
                # i18n: ignore operator tooling diagnostic (English-only)
                "shard_ledger.py --fold-stranded: nothing stranded; "
                "aggregate agrees with shards"
            )
            return 0
        args.out.mkdir(parents=True, exist_ok=True)
        for locale in sorted(stranded_by_locale):
            extra = stranded_by_locale[locale]
            if not extra:
                continue
            rows = existing.get(locale, []) + extra
            (args.out / f"{locale}.json").write_text(render_shard(locale, rows), encoding="utf-8")
            print(
                f"folded {len(extra)} stranded row(s) into {locale}.json"
            )  # i18n: ignore operator tooling diagnostic
            for row in extra:
                print(
                    # i18n: ignore operator tooling diagnostic (English-only)
                    f"  {row.get('locale')}/{row.get('namespace')} {str(row.get('note', ''))[:60]}"
                )
        print(
            # i18n: ignore operator tooling diagnostic (English-only)
            "now run scripts/i18n/generate.py to re-render the aggregate mirror"
        )
        return 0

    by_locale = _rows_by_locale(aggregate_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for locale in sorted(by_locale):
        path = args.out / f"{locale}.json"
        path.write_text(render_shard(locale, by_locale[locale]), encoding="utf-8")
        print(
            f"wrote {path} ({len(by_locale[locale])} rows)"
        )  # i18n: ignore operator tooling diagnostic
    return 0


def _stranded_by_locale(
    aggregate_rows: list[dict[str, Any]], existing: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate rows absent from the shards, grouped by locale, in aggregate order."""
    counts = Counter(json.dumps(row, sort_keys=True) for rows in existing.values() for row in rows)
    stranded: dict[str, list[dict[str, Any]]] = {}
    for row in aggregate_rows:
        key = json.dumps(row, sort_keys=True)
        if counts[key] > 0:
            counts[key] -= 1
            continue
        stranded.setdefault(row["locale"], []).append(row)
    return stranded


if __name__ == "__main__":
    raise SystemExit(main())
