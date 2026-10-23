"""The byte cap and the token quota are separate budgets; both must be visible.

Measured defect this file pins: the live store reached its 64 MiB page cap, so
every write was refused, while `lop memory status` printed only the token quota
(33% headroom) and the refusal was the generic "refused or store unavailable".
The two regressions are (1) status not reporting bytes against the cap and (2)
the cap refusal not naming its cause or remedy. The store is driven to the cap
with a small override plus synthetic facts, never 64 MiB of real data.

Round 2 adds the recovery state: forget/VACUUM leave reusable pages in the
freelist, and a store at max_page_count with freelist > 0 still accepts writes.
Status must not call that FULL (`storage_full` is "no free page AND no headroom")
and the near-cap warning must exist for the 90-99.9% state where refusals land.
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


def enforced_cap():
    """The byte cap SQLite is actually given: MAX_DB_BYTES floored to a page.

    The denominator is repeated rather than imported so the file can run
    against the pre-round-2 module and fail on assertions, not collection.
    """
    return (MAX_DB_BYTES // 4096) * 4096


def unwrapped(text):
    """Status prose wraps at 80 columns; assert on the sentence, not the layout."""
    return " ".join(text.split())


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


def fill_meta_to_page_cap(db):
    """Consume pages until SQLite refuses one; leaves page_count == max_page_count.

    Bounded on purpose: an unbounded loop would hang the suite if a future
    change made the store grow without limit. 8 KiB rows reach a 32-page cap
    in single-digit inserts (measured), so 500 is a wide ceiling, not a budget.
    """
    for index in range(500):
        try:
            db.execute("INSERT INTO meta VALUES (?,?)", (f"fill-{index}", "x" * 8192))
            db.commit()
        except sqlite3.OperationalError:
            db.rollback()
            return
    pytest.fail("store never reached its page cap")


def test_status_reports_bytes_against_the_cap_alongside_the_token_quota(tmp_path):
    store, identity = make_store(tmp_path)
    store.put("global", {"title": "Byte budget", "body": "A fact that uses a little space."})

    usage, text = status(tmp_path, identity)
    assert usage["storage_cap_bytes"] == enforced_cap()
    assert 0 < usage["storage_bytes"] < usage["storage_cap_bytes"]
    assert usage["storage_percent"] == pytest.approx(
        round(100.0 * usage["storage_bytes"] / usage["storage_cap_bytes"], 1)
    )
    assert usage["storage_free_bytes"] == 0
    assert usage["storage_full"] is False
    assert usage["database_path"].endswith("memory/memory.db")
    # The human page must carry BOTH budgets: the token quota alone is exactly
    # what reported headroom while the store was byte-full.
    assert (
        f"Storage: {human_bytes(usage['storage_bytes'])} of {human_bytes(enforced_cap())}" in text
    )
    assert "% of the store cap" in text
    assert "Quota: global" in text


def test_page_cap_refusal_is_distinct_actionable_and_names_the_store(tmp_path, monkeypatch):
    # A 32-page store (128 KiB) fills in seconds. Small bodies keep the write
    # under the global token quota, so the cap is the budget that refuses first.
    monkeypatch.setattr(memory, "MAX_DB_BYTES", 4096 * 32)
    store, _ = make_store(tmp_path)

    writes = 0
    with pytest.raises(MemoryError) as refused:
        for _ in range(200):
            writes += 1
            store.put(
                "global",
                {"title": f"Fill fact {writes}", "body": f"payload {writes} " + "z" * 200},
            )

    message = str(refused.value)
    assert 1 < writes < 200
    # The fact fill refuses below the page cap (measured: 30 of 32 pages), so
    # this is the near-cap arm: it must state the headroom, not claim the cap
    # was reached.
    assert "Memory store is near its cap" in message
    assert "needs more pages than the" in message and "remaining under the" in message
    assert "a smaller write may still succeed" in message
    assert "has reached its cap" not in message.lower()
    assert "no new fact can be saved" not in message.lower()
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
        fill_meta_to_page_cap(db)
        assert db.execute("PRAGMA page_count").fetchone()[0] == 32
        assert db.execute("PRAGMA freelist_count").fetchone()[0] == 0

    usage, text = status(tmp_path, identity)

    assert usage["storage_full"] is True
    assert usage["storage_percent"] == 100.0
    assert usage["storage_free_bytes"] == 0
    # The exact contradiction the defect reported: the TOKEN quota is empty of
    # use while the byte cap refuses writes. Both must be on the page.
    assert usage["active_tokens"] == 0
    assert "Quota: global 0/16000" in text
    assert "Store FULL" in text
    flat = unwrapped(text)
    assert "writes that need a new page are refused until space is freed" in flat
    # The reviewer measured writes succeeding at this state until the intra-page
    # budget is spent: the sentence must not say nothing can be saved.
    assert "no new fact can be saved" not in flat.lower()
    assert "Reclaim: `lop memory forget SCOPE/ID --revision REV --yes`." in text
    assert f"Compact: `sqlite3 {store.path} 'VACUUM'`." in text
    assert "Run the Compact command first if the forget is also refused." in text

    # At the cap exactly, the refusal keeps the at-cap wording: a write that
    # needs a new page is refused until space is freed.
    with pytest.raises(MemoryError) as refused:
        for index in range(50):
            # Distinct bodies: identical facts are deduplicated, not written.
            store.put(
                "global",
                {"title": f"After the full report {index}", "body": f"Needs a page {index}."},
            )
    at_cap = str(refused.value)
    assert "Memory store is full" in at_cap and "has reached its" in at_cap
    assert "writes that need a new page are refused until space is freed" in at_cap
    assert "no new fact can be saved" not in at_cap.lower()


def test_free_pages_at_the_cap_are_not_reported_full_and_still_write(tmp_path, monkeypatch):
    """Round-2 MAJOR: the state a successful remedy leaves behind must not say FULL.

    forget/VACUUM leave reusable pages in the freelist. A store at
    max_page_count with freelist > 0 accepts writes, so status must not claim
    "nothing is saved" while the very next small write succeeds.
    """
    monkeypatch.setattr(memory, "MAX_DB_BYTES", 4096 * 32)
    store, identity = make_store(tmp_path)
    with store.connection() as db:
        db.commit()
        fill_meta_to_page_cap(db)
        assert db.execute("PRAGMA page_count").fetchone()[0] == 32
        db.execute("DELETE FROM meta WHERE key IN ('fill-0','fill-1','fill-2')")
        db.commit()
        pages = db.execute("PRAGMA page_count").fetchone()[0]
        free = db.execute("PRAGMA freelist_count").fetchone()[0]
    assert pages == 32 and free > 0

    usage, text = status(tmp_path, identity)

    assert usage["storage_full"] is False
    assert usage["storage_percent"] == 100.0
    assert usage["storage_free_bytes"] == free * 4096
    assert "Store FULL" not in text
    assert "Reusing freed pages" in text
    # The claim has to be true, not merely softer: the next write must work.
    item = store.put("global", {"title": "After the remedy", "body": "Reuses a freed page."})
    assert item["status"] == "active"


def test_near_cap_warning_covers_the_90_to_99_percent_state(tmp_path, monkeypatch):
    """QA measured refusals at 96.9% (31/32 pages); that branch needs coverage.

    Deleting one filled row and VACUUMing leaves a compacted store below the
    page cap with no freelist and no headroom — the state the warning names.
    """
    monkeypatch.setattr(memory, "MAX_DB_BYTES", 4096 * 32)
    store, identity = make_store(tmp_path)
    with store.connection() as db:
        db.commit()
        fill_meta_to_page_cap(db)
        db.execute("DELETE FROM meta WHERE key='fill-0'")
        db.commit()
    raw = sqlite3.connect(store.path)
    try:
        raw.execute("VACUUM")
        raw.commit()
    finally:
        raw.close()

    usage, text = status(tmp_path, identity)

    assert usage["storage_full"] is False
    assert usage["storage_free_bytes"] == 0
    assert 90 <= usage["storage_percent"] < 100
    flat = unwrapped(text)
    assert "Near the storage cap" in flat
    assert "with no free pages left" in flat
    assert "needs more pages than remain" in flat
    assert "Store FULL" not in text


def test_batch_refusal_below_the_cap_states_headroom_and_allows_a_smaller_write(
    tmp_path, monkeypatch
):
    """Round-3 MAJOR: the batch arm must not borrow the at-cap wording.

    28 of 32 pages with an empty freelist: a multi-page batch is refused, but
    four pages remain, so "has reached its cap" and "writes that need a new page
    are refused" would both be false — the next small write was measured to
    succeed.
    """
    monkeypatch.setattr(memory, "MAX_DB_BYTES", 4096 * 32)
    store, identity = make_store(tmp_path)
    with store.connection() as db:
        db.commit()
        fill_meta_to_page_cap(db)
        db.execute("DELETE FROM meta WHERE key IN ('fill-0','fill-1')")
        db.commit()
    raw = sqlite3.connect(store.path)
    try:
        raw.execute("VACUUM")
        raw.commit()
    finally:
        raw.close()
    with store.connection() as db:
        db.commit()
        pages = db.execute("PRAGMA page_count").fetchone()[0]
        cap = db.execute("PRAGMA max_page_count").fetchone()[0]
        free = db.execute("PRAGMA freelist_count").fetchone()[0]
    assert (pages, cap, free) == (28, 32, 0)

    with pytest.raises(MemoryError) as refused:
        with store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for index in range(3):
                db.execute("INSERT INTO meta VALUES (?,?)", (f"batch-{index}", "x" * 8192))
            db.commit()

    message = str(refused.value)
    assert "has reached its cap" not in message.lower()
    assert f"{human_bytes(4 * 4096)} remaining under" in message
    assert "a smaller write may still succeed" in message
    # Status is not FULL, and the claim is true: a small write goes through.
    usage, _ = status(tmp_path, identity)
    assert usage["storage_full"] is False
    item = store.put("global", {"title": "Small after batch", "body": "Fits the headroom."})
    assert item["status"] == "active"


def test_reported_cap_is_the_enforced_page_aligned_cap(tmp_path, monkeypatch):
    """QA Q3: a non-page-aligned override must not read 99.9% beside FULL.

    The module constant is reported NOWHERE — status and the refusal both use
    max_page_count * page_size, the number SQLite enforces.
    """
    unaligned = 4096 * 32 + 1000
    monkeypatch.setattr(memory, "MAX_DB_BYTES", unaligned)
    store, _ = make_store(tmp_path)

    usage = store.inventory()
    assert usage["storage_cap_bytes"] == 32 * 4096
    assert usage["storage_percent"] == 0.0

    writes = 0
    with pytest.raises(MemoryError) as refused:
        for _ in range(200):
            writes += 1
            store.put(
                "global",
                {"title": f"Fill fact {writes}", "body": f"payload {writes} " + "z" * 200},
            )

    assert f"remaining under the {human_bytes(32 * 4096)} cap" in str(refused.value)
    assert human_bytes(unaligned) not in str(refused.value)
    usage = store.inventory()
    assert usage["storage_cap_bytes"] == 32 * 4096
    assert usage["storage_percent"] == pytest.approx(
        round(100.0 * usage["storage_bytes"] / (32 * 4096), 1)
    )


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
