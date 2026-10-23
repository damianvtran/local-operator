"""The byte cap and the token quota are separate budgets; both must be visible.

Measured defect this file pins: the live store reached its 64 MiB page cap, so
every write was refused, while `lop memory status` printed only the token quota
(33% headroom) and the refusal was the generic "refused or store unavailable".
The two regressions are (1) status not reporting bytes against the cap and (2)
the cap refusal not naming its cause or remedy. The store is driven to the cap
with a small override plus synthetic facts, never 64 MiB of real data.
"""

import json
import sqlite3

import pytest

import local_operator.memory as memory
from local_operator.memory import MAX_DB_BYTES, MemoryError, MemoryIdentity, MemoryStore
from local_operator.memory_cli import render_human, run


def human_bytes(value):
    """The exact rendering the status page must use for a byte figure.

    Kept local because these tests double as the defect reproduction: against
    the pre-change module an imported helper would abort collection instead of
    showing which assertion the old code fails.
    """
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)}B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def make_store(tmp_path):
    identity = MemoryIdentity("cap-session", "", "")
    session_dir = tmp_path / "sessions" / "cap-session"
    return MemoryStore(tmp_path / "cfg", identity, session_dir, origin="cap-device"), identity


def status(tmp_path, identity):
    out, failed = run(
        "status",
        config=tmp_path / "cfg",
        identity=identity,
        session_dir=tmp_path / "sessions" / "cap-session",
    )
    assert not failed, out
    return json.loads(out)["usage"], render_human(out, failed)


def test_status_reports_bytes_against_the_cap_alongside_the_token_quota(tmp_path):
    store, identity = make_store(tmp_path)
    store.put("global", {"title": "Byte budget", "body": "A fact that uses a little space."})

    usage, text = status(tmp_path, identity)
    assert usage["storage_cap_bytes"] == MAX_DB_BYTES
    assert 0 < usage["storage_bytes"] < MAX_DB_BYTES
    assert usage["storage_percent"] == pytest.approx(
        round(100.0 * usage["storage_bytes"] / MAX_DB_BYTES, 1)
    )
    assert usage["storage_full"] is False
    assert usage["database_path"].endswith("memory/memory.db")
    # The human page must carry BOTH budgets: the token quota alone is exactly
    # what reported headroom while the store was byte-full.
    assert f"Storage: {human_bytes(usage['storage_bytes'])} of {human_bytes(MAX_DB_BYTES)}" in text
    assert "% of the store cap" in text
    assert "Quota: global" in text


def test_page_cap_refusal_is_distinct_actionable_and_names_the_store(tmp_path, monkeypatch):
    # A 32-page store (128 KiB) fills in seconds. Small bodies keep the write
    # under the global token quota, so the cap is the budget that refuses first.
    monkeypatch.setattr(memory, "MAX_DB_BYTES", 4096 * 32)
    store, _ = make_store(tmp_path)

    writes = 0
    with pytest.raises(MemoryError) as refused:
        while True:
            writes += 1
            store.put(
                "global",
                {"title": f"Fill fact {writes}", "body": f"payload {writes} " + "z" * 200},
            )

    message = str(refused.value)
    assert writes > 1
    assert "Memory store is full" in message
    assert f"reached its {human_bytes(4096 * 32)} cap" in message
    assert str(store.path) in message
    # Every remedy the operator needs, including the file the VACUUM acts on.
    assert "lop memory status" in message
    assert "lop memory forget SCOPE/ID --revision REV --yes" in message
    assert "VACUUM" in message
    # NOT the generic refusal: that sentence is for genuinely unavailable
    # stores and carried no cause after 41 hours of silent refusals.
    assert "request refused or store unavailable" not in message


def test_full_store_status_says_so_plainly_and_offers_the_way_out(tmp_path, monkeypatch):
    # Pre-fill the pages directly rather than writing 64 MiB: the status surface
    # reads page_count, which is what the cap is enforced on.
    monkeypatch.setattr(memory, "MAX_DB_BYTES", 4096 * 32)
    store, identity = make_store(tmp_path)
    with store.connection() as db:
        db.commit()
        for size in (3000, 128):
            index = 0
            while True:
                try:
                    db.execute(
                        "INSERT INTO meta VALUES (?,?)",
                        (f"fill-{size}-{index}", "x" * size),
                    )
                    db.commit()
                except sqlite3.OperationalError:
                    db.rollback()
                    break
                index += 1

    usage, text = status(tmp_path, identity)

    assert usage["storage_full"] is True
    assert usage["storage_percent"] == 100.0
    # The exact contradiction the defect reported: the TOKEN quota is empty of
    # use while the byte cap refuses writes. Both must be on the page.
    assert usage["active_tokens"] == 0
    assert "Quota: global 0/16000" in text
    assert "Store FULL" in text
    assert "Reclaim: `lop memory forget SCOPE/ID --revision REV --yes`." in text
    assert f"Compact: `sqlite3 {store.path} 'VACUUM'` — run this first" in text


def test_non_cap_sqlite_errors_keep_the_generic_refusal(tmp_path):
    store, _ = make_store(tmp_path)
    store.put("global", {"title": "Present", "body": "Keeps the store initialised."})

    # A locked store is unavailable, not full: it must not be retranslated.
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with store.connection():
            raise sqlite3.OperationalError("database is locked")

    # SQLITE_FULL with the store nowhere near its cap is the host disk, not the
    # cap; the page-cap message would send the operator to the wrong remedy.
    with pytest.raises(sqlite3.OperationalError, match="database or disk is full"):
        with store.connection():
            raise sqlite3.OperationalError("database or disk is full")
