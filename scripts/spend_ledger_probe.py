"""Probe: the store-wide spend picture, before and after the ledger.

Read-only against the operator's store by default. Two jobs:

1. **The census the PR quotes.** How many sessions paint a cost at all, how many
   paint it as a lower bound (``≥``), and how many paint NOTHING because the
   newest reading is unpriceable — reported twice: as the code behaves on
   ``main`` (price the newest surviving receipt, mark it FLOOR) and as it
   behaves with the ledger (a record's own knowledge state; otherwise the
   one-time rebuild's classification). Plus the ratio that motivates the whole
   change: the newest surviving reading against the sum of the session's own
   turn rows.
2. **One session in detail** (``--session``), for the headline claims — e.g. the
   session whose turn rows total $324.99 while the pre-ledger restore shows
   $36.99, or one whose transcript can only account for $9.39 of a $67.89 bill.

Never writes to the store: it opens transcripts with ``read_replay_suffix`` and
parses nothing it does not have to, and it opens ``analytics.db`` read-only.
``--turn``/``--resume`` are the only modes that write, and they demand an
explicit directory (point them at a throwaway ``HOME``).

Usage::

    .venv/bin/python scripts/spend_ledger_probe.py --census
    .venv/bin/python scripts/spend_ledger_probe.py --session 28a800c6a783
    .venv/bin/python scripts/spend_ledger_probe.py --turn /tmp/iso/sessions/p1
    .venv/bin/python scripts/spend_ledger_probe.py --resume /tmp/iso/sessions/p1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.session.spend import (  # noqa: E402
    SESSION_SPEND_CUSTOM_TYPE,
    SessionSpend,
)
from local_operator.session.transcript import (  # noqa: E402
    ENTRY_COMPACTION,
    ENTRY_PRUNE,
    Transcript,
    all_usage_rows,
    read_replay_suffix,
    usages_since_newest_shrink,
)
from local_operator.session.usage_seed import seed_reported_usage  # noqa: E402
from local_operator.tui.costs import turn_cost  # noqa: E402

DEFAULT_STORE = Path.home() / ".local-operator" / "sessions"
DEFAULT_DB = Path.home() / ".local-operator" / "analytics.db"


def _no_stream(request=None, signal=None):  # noqa: ANN001
    """An explicit empty async stream: a resume never calls the provider."""

    async def gen():
        return
        yield  # pragma: no cover - an async generator that yields nothing

    return gen()


def _label(row: dict[str, Any], default: str) -> str:
    provider = row.get("provider") or default.split("/")[0]
    model_id = row.get("model_id") or default.split("/")[-1]
    return f"{provider}/{model_id}"


def _price(row: dict[str, Any], default_label: str) -> float | None:
    """Paint-grade price of one usage row: the pre-ledger restore's own path."""
    recorded = row.get("usd_cost")
    if isinstance(recorded, (int, float)):
        return float(recorded)
    return turn_cost(_label(row, default_label), row)


def census(store: Path, db_path: Path | None, session_dir: Path | None) -> dict[str, Any]:
    directories = sorted(p for p in store.iterdir() if (p / "transcript.jsonl").exists())
    if session_dir is not None:
        directories = [session_dir]
    stats: dict[str, Any] = {
        "sessions": len(directories),
        "with_cost": 0,
        "paint_floor": 0,
        "paint_nothing": 0,
        "with_record": 0,
        "record_exact": 0,
        "record_partial": 0,
        "record_floor": 0,
        "record_unknown": 0,
        "rebuild_would_be_exact": 0,
        "rebuild_would_be_partial": 0,
        "rebuild_would_be_floor": 0,
        "rebuild_empty": 0,
        "ratios": [],
        "undercount": [],
        "record_micro": 0,
    }
    for directory in directories:
        try:
            suffix = read_replay_suffix(
                directory,
                checkpoint_types=(SESSION_SPEND_CUSTOM_TYPE,),
            )
        except Exception:  # noqa: BLE001 — a broken session must not stop the census
            continue
        entries = suffix.entries
        rows = all_usage_rows(entries)
        survivors = usages_since_newest_shrink(entries)
        shrunk = any(entry.type in (ENTRY_COMPACTION, ENTRY_PRUNE) for entry in entries)
        default_label = "unknown/unknown"
        priced_survivors = [p for p in (_price(r, default_label) for r in survivors) if p]
        priced_all = [p for p in (_price(r, default_label) for r in rows) if p]
        newest = seed_reported_usage(survivors)
        newest_cost = _price(newest.model_dump(mode="json"), default_label) if newest else None

        record = SessionSpend.from_details(suffix.checkpoints.get(SESSION_SPEND_CUSTOM_TYPE))
        if priced_all:
            stats["with_cost"] += 1
        # BEFORE: the restore prices the newest surviving reading and marks it
        # FLOOR; an unpriceable newest reading paints no cost segment at all.
        if newest_cost is not None:
            stats["paint_floor"] += 1
        elif newest is not None:
            stats["paint_nothing"] += 1
        if priced_all and priced_survivors:
            total = sum(priced_all)
            if total > 0:
                stats["ratios"].append(sum(priced_survivors) / total)
        # THE OPERATOR'S NUMBER: how far the pre-ledger restore understates the
        # session. It prices only the newest surviving reading, so the factor is
        # "what the turn rows say" over "what the band will show".
        if priced_all and newest_cost:
            stats["undercount"].append(sum(priced_all) / newest_cost)
        # AFTER: a record answers for itself; otherwise the rebuild classifies.
        if record is not None:
            stats["with_record"] += 1
            stats["record_micro"] += record.micro
            stats[f"record_{record.knowledge().value}"] += 1
        elif not rows:
            stats["rebuild_empty"] += 1
        elif any(
            _price(row, default_label) is None
            and (row.get("input_tokens") or row.get("output_tokens"))
            for row in rows
        ):
            stats["rebuild_would_be_partial"] += 1
        elif shrunk:
            stats["rebuild_would_be_floor"] += 1
        else:
            stats["rebuild_would_be_exact"] += 1
    if stats["ratios"]:
        stats["survivor_share_median"] = round(statistics.median(stats["ratios"]), 3)
        stats["survivor_share_mean"] = round(statistics.mean(stats["ratios"]), 3)
    stats.pop("ratios", None)
    undercounts = sorted(stats.pop("undercount", []))
    if undercounts:
        stats["undercount_median"] = round(statistics.median(undercounts), 1)
        stats["undercount_mean"] = round(statistics.mean(undercounts), 1)
        stats["undercount_max"] = round(undercounts[-1], 1)
        stats["undercount_over_10x"] = sum(1 for value in undercounts if value > 10)
        stats["undercount_over_100x"] = sum(1 for value in undercounts if value > 100)
    cost_sessions = stats["with_cost"]
    stats["paint_floor_share_of_cost_sessions"] = (
        round(stats["paint_floor"] / cost_sessions, 4) if cost_sessions else 0.0
    )
    if db_path is not None and db_path.exists():
        stats["ledger"] = _ledger_totals(db_path)
    return stats


def session_detail(directory: Path, db_path: Path | None) -> dict[str, Any]:
    transcript = Transcript(directory)
    rows = transcript.all_usage_rows()
    survivors = transcript.usages_since_compaction()
    entries = transcript.entries()
    default_label = "unknown/unknown"
    record = SessionSpend.from_details(transcript.latest_custom(SESSION_SPEND_CUSTOM_TYPE))
    newest = seed_reported_usage(survivors)
    detail = {
        "session": directory.name,
        "usage_rows": len(rows),
        "surviving_rows": len(survivors),
        "compaction_markers": sum(1 for e in entries if e.type == ENTRY_COMPACTION),
        "prune_markers": sum(1 for e in entries if e.type == ENTRY_PRUNE),
        "journal_shrank": transcript.journal_shrank(),
        "all_rows_priced_usd": round(
            sum(p for p in (_price(r, default_label) for r in rows) if p), 4
        ),
        "surviving_rows_priced_usd": round(
            sum(p for p in (_price(r, default_label) for r in survivors) if p), 4
        ),
        "newest_reading_usd": (
            round(_price(newest.model_dump(mode="json"), default_label) or 0.0, 6)
            if newest
            else None
        ),
        "record": record.to_details() if record is not None else None,
    }
    if db_path is not None and db_path.exists():
        detail["ledger_usd"] = _ledger_session_sum(db_path, directory.name)
    return detail


def _ledger_connection(db_path: Path):  # type: ignore[no-untyped-def]
    import sqlite3

    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _ledger_totals(db_path: Path) -> dict[str, Any]:
    try:
        with _ledger_connection(db_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*), SUM(cost_micro) FROM calls WHERE cost_known = 1"
            ).fetchone()
        return {"known_calls": int(row[0] or 0), "known_micro": int(row[1] or 0)}
    except Exception as error:  # noqa: BLE001 — a probe must not fail on the oracle
        return {"error": type(error).__name__}


def _ledger_session_sum(db_path: Path, session_id: str) -> float | None:
    try:
        with _ledger_connection(db_path) as connection:
            row = connection.execute(
                "SELECT SUM(cost_micro) FROM calls WHERE session_id = ?", (session_id,)
            ).fetchone()
        return round((row[0] or 0) / 1_000_000.0, 4)
    except Exception:  # noqa: BLE001
        return None


async def write_turn(directory: Path) -> None:
    """One scripted turn whose usage carries a provider receipt (write mode)."""
    from local_operator.harness.types import (
        Message,
        ModelSpec,
        StreamEndEvent,
        StreamTextDelta,
        Usage,
    )
    from local_operator.session.session import Session

    directory.mkdir(parents=True, exist_ok=True)
    usage = Usage(
        provider="openrouter",
        model_id="deepseek/deepseek-v4.1-flash",
        input_tokens=1_000,
        output_tokens=200,
        context_tokens=1_000,
        usd_cost=0.0021,
    )

    def stream(request, signal=None):
        async def gen():
            yield StreamTextDelta(delta="ack")
            yield StreamEndEvent(stop_reason="stop", usage=usage)

        return gen()

    session = Session(
        model=ModelSpec(
            provider="deepseek",
            model_id="deepseek-chat",
            display_name="DeepSeek",
            context_window=64_000,
        ),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        yolo=True,
        cwd=str(directory),
    )
    await session._run_turn([Message.user("probe")])
    for _ in range(200):
        await asyncio.sleep(0.01)
        if session._spend_recorded:
            break
    print(
        json.dumps(
            {
                "turn_spend": session.spend.to_details(),
                "knowledge": session.spend.knowledge().value,
                "usd": round(session.spend.usd, 6),
            },
            indent=2,
            sort_keys=True,
        )
    )


def resume_show(directory: Path) -> None:
    from local_operator.harness.types import ModelSpec
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript as _Transcript

    session = Session(
        model=ModelSpec(
            provider="deepseek",
            model_id="deepseek-chat",
            display_name="DeepSeek",
            context_window=64_000,
        ),
        stream_fn=_no_stream,
        tools=[],
        transcript=_Transcript(directory),
        system_blocks_provider=lambda: [],
        yolo=True,
        cwd=str(directory),
    )
    spend = session.restored_spend()
    print(
        json.dumps(
            {
                "recalled_record": spend.to_details() if spend is not None else None,
                "knowledge": spend.knowledge().value if spend is not None else None,
                "restored_usage_present": session.restored_usage() is not None,
                "record_rows_in_journal": sum(
                    1
                    for entry in session._transcript.entries()
                    if entry.payload.get("custom_type") == SESSION_SPEND_CUSTOM_TYPE
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--census", action="store_true")
    parser.add_argument("--session", type=str, default="")
    parser.add_argument("--turn", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.quiet:
        logging.disable(logging.CRITICAL)

    if args.turn is not None:
        asyncio.run(write_turn(args.turn))
        return
    if args.resume is not None:
        resume_show(args.resume)
        return
    if args.session:
        detail = session_detail(args.store / args.session, args.db)
        print(json.dumps(detail, indent=2, sort_keys=True))
        return
    print(json.dumps(census(args.store, args.db, None), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
