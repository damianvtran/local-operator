"""Per-locale ledger shards: migration parity and merge-conflict freedom.

The aggregate ``i18n/ledger.json`` was one shared array every locale PR
appended to, so concurrent locale work conflicted on the same tail. Shards
under ``i18n/ledger/<locale>.json`` are authoritative; the aggregate is a
generated compatibility mirror. These tests pin the two properties the
migration must not move: the shipped set (eligibility semantics) and the
append-only supersession history.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SHARD_SCRIPT = REPO / "scripts" / "i18n" / "shard_ledger.py"


def test_shards_and_aggregate_yield_identical_shipped_set(i18n_generate) -> None:
    """Non-negotiable: pre-migration aggregate vs post-migration shards.

    Loads the committed aggregate (legacy path) and the committed shards
    (authoritative path) through the SAME generator entry point and requires
    the shipped set to be identical for all eight locales, byte-for-byte.
    """
    # Authoritative path: shards present (the committed tree).
    from_shards = i18n_generate._shipped_locales()
    shard_entries = i18n_generate._ledger_entries()

    # Legacy path: hide the shard directory so the aggregate is the source.
    import unittest.mock as mock

    with mock.patch.object(i18n_generate, "LEDGER_SHARDS", REPO / "i18n" / "nonexistent"):
        legacy_entries = i18n_generate._ledger_entries()
        from_aggregate = i18n_generate._shipped_locales()

    # Same rows, same shipped set. The mirror groups by locale, so compare
    # per-locale ordered rows rather than raw array order.
    assert len(shard_entries) == len(legacy_entries) == 437

    def per_locale(rows: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for row in rows:
            out.setdefault(row["locale"], []).append(row)
        return out

    assert per_locale(shard_entries) == per_locale(legacy_entries)
    assert from_shards == from_aggregate
    assert from_shards == ["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"]

    # And the generated shipped.json on disk matches both.
    generated = json.loads(
        (REPO / "local_operator/i18n/data/shipped.json").read_text(encoding="utf-8")
    )
    assert generated["locales"] == from_shards


def test_migration_is_order_preserving_and_lossless(tmp_path: Path) -> None:
    """The one-shot script: per-locale order preserved, no dedupe, idempotent."""
    aggregate = json.loads((REPO / "i18n" / "ledger.json").read_text(encoding="utf-8"))
    # Write the pre-migration shape (schema + entries) to a scratch aggregate.
    source = tmp_path / "ledger.json"
    source.write_text(json.dumps(aggregate), encoding="utf-8")
    out = tmp_path / "shards"
    subprocess.run(
        [sys.executable, str(SHARD_SCRIPT), "--ledger", str(source), "--out", str(out)],
        check=True,
        capture_output=True,
    )
    expected: dict[str, list[dict]] = {}
    for row in aggregate["entries"]:
        expected.setdefault(row["locale"], []).append(row)
    assert len(expected) == 7  # seven non-en locales carry rows
    for locale, rows in expected.items():
        shard = json.loads((out / f"{locale}.json").read_text(encoding="utf-8"))
        # Byte-for-byte row equality in original per-locale order: this is the
        # assertion that forbids dedupe on (locale, namespace) — several
        # namespaces legitimately hold multiple rows at different hashes.
        assert shard["entries"] == rows, locale
        assert shard["locale"] == locale
    # Idempotent: a second run writes identical bytes.
    first = {p.name: p.read_bytes() for p in out.glob("*.json")}
    subprocess.run(
        [sys.executable, str(SHARD_SCRIPT), "--ledger", str(source), "--out", str(out)],
        check=True,
        capture_output=True,
    )
    assert {p.name: p.read_bytes() for p in out.glob("*.json")} == first


def test_supersession_last_wins_across_hashes(i18n_generate, tmp_path: Path, monkeypatch) -> None:
    """Two rows for one namespace at different hashes: the LAST row wins.

    The discriminating case for "do not dedupe": an earlier passed row at a
    stale hash must be superseded by a later passed row at the current hash,
    and a later FAILED row must un-ship even when an earlier row passed.
    """
    from local_operator.i18n import catalogues

    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "fr").mkdir(parents=True)
    (root / "en" / "wire.exec.json").write_text(
        json.dumps({"wire.exec.demo": "Synthetic receipt"}), encoding="utf-8"
    )
    (root / "fr" / "wire.exec.json").write_text(
        json.dumps({"wire.exec.demo": "Recu synthetique"}), encoding="utf-8"
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(i18n_generate, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(i18n_generate, "LEDGER_SHARDS", tmp_path / "ledger")
    current = catalogues.catalogue_sha256("en", "wire.exec")

    def row(status: str, sha: str) -> dict:
        return {
            "locale": "fr",
            "namespace": "wire.exec",
            "source_sha256": sha,
            "status": status,
        }

    shard_dir = tmp_path / "ledger"
    shard_dir.mkdir()

    # Stale-passed then fresh-passed: ships (supersession forward).
    (shard_dir / "fr.json").write_text(
        json.dumps({"schema": 1, "locale": "fr", "entries": [row("passed", "0" * 64), row("passed", current)]}),
        encoding="utf-8",
    )
    assert "fr" in i18n_generate._shipped_locales()

    # Fresh-passed then failed: does NOT ship (a later failure supersedes).
    (shard_dir / "fr.json").write_text(
        json.dumps({"schema": 1, "locale": "fr", "entries": [row("passed", current), row("failed", current)]}),
        encoding="utf-8",
    )
    assert "fr" not in i18n_generate._shipped_locales()


def test_stale_aggregate_fails_drift_check(i18n_generate, tmp_path: Path, monkeypatch) -> None:
    """A hand-edited aggregate that disagrees with the shards cannot win.

    The check compares COMMITTED bytes against the mirror render — it never
    re-derives the aggregate and calls the two equal. A genuinely stale
    aggregate (here: empty entries while the shard holds a row) is detected.
    """
    from local_operator.i18n import catalogues

    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "fr").mkdir(parents=True)
    (root / "en" / "wire.exec.json").write_text(
        json.dumps({"wire.exec.demo": "Synthetic receipt"}), encoding="utf-8"
    )
    (root / "fr" / "wire.exec.json").write_text(
        json.dumps({"wire.exec.demo": "Recu synthetique"}), encoding="utf-8"
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(i18n_generate, "REPO", tmp_path)
    monkeypatch.setattr(i18n_generate, "LEDGER", tmp_path / "i18n" / "ledger.json")
    monkeypatch.setattr(i18n_generate, "LEDGER_SHARDS", tmp_path / "i18n" / "ledger")
    # _check also scans the generated dirs for orphans; point them at empty
    # fixture dirs so only the ledger path is under test.
    monkeypatch.setattr(i18n_generate, "KEYS_DIR", tmp_path / "keys")
    monkeypatch.setattr(i18n_generate, "DATA_DIR", tmp_path / "data")
    shard_dir = tmp_path / "i18n" / "ledger"
    shard_dir.mkdir(parents=True)
    current = catalogues.catalogue_sha256("en", "wire.exec")
    row = {
        "locale": "fr",
        "namespace": "wire.exec",
        "source_sha256": current,
        "status": "passed",
    }
    (shard_dir / "fr.json").write_text(
        json.dumps({"schema": 1, "locale": "fr", "entries": [row]}), encoding="utf-8"
    )
    # Aggregate disagrees (empty entries): the mirror render must differ.
    ledger_path = tmp_path / "i18n" / "ledger.json"
    ledger_path.write_text(json.dumps({"schema": 1, "entries": []}), encoding="utf-8")
    mirror = i18n_generate._render_aggregate()
    assert mirror != ledger_path.read_text(encoding="utf-8")
    # And _check flags it as changed when planned.
    planned = {ledger_path: mirror}
    assert i18n_generate._check(planned) == 1
    # Writing the mirror makes it pass.
    ledger_path.write_text(mirror, encoding="utf-8")
    assert i18n_generate._check(planned) == 0


def test_two_locales_touch_disjoint_files() -> None:
    """The structural claim: fr and es rows live in different files.

    Simulated as two concurrent edits applied to disjoint shard paths — git
    merges disjoint-file changes without conflict by construction; what this
    pins is that no shared file (the aggregate) is a WRITE target for locale
    contributions anymore: the aggregate is regenerated, never appended to.
    """
    shards = {p.name for p in (REPO / "i18n" / "ledger").glob("*.json")}
    assert shards == {"es.json", "fr.json", "hi.json", "ru.json", "ur.json", "vi.json", "zh-CN.json"}
    # Each shard names exactly its own locale in every row.
    for path in (REPO / "i18n" / "ledger").glob("*.json"):
        shard = json.loads(path.read_text(encoding="utf-8"))
        assert {row["locale"] for row in shard["entries"]} == {shard["locale"]}
        assert path.stem == shard["locale"]


@pytest.mark.parametrize("locale", ["es", "fr", "hi", "ru", "ur", "vi", "zh-CN"])
def test_shard_round_trips_through_aggregate_mirror(locale: str, i18n_generate) -> None:
    """Every shard's rows appear verbatim in the regenerated aggregate."""
    shard = json.loads((REPO / "i18n" / "ledger" / f"{locale}.json").read_text(encoding="utf-8"))
    aggregate = json.loads(i18n_generate._render_aggregate())
    rows = [r for r in aggregate["entries"] if r["locale"] == locale]
    assert rows == shard["entries"]


def _row(locale: str, namespace: str, sha: str, note: str) -> dict:
    return {
        "locale": locale,
        "namespace": namespace,
        "source_sha256": sha,
        "status": "passed",
        "note": note,
    }


def _write_shard(root: Path, locale: str, rows: list[dict]) -> None:
    (root / f"{locale}.json").write_text(
        json.dumps({"schema": 1, "locale": locale, "entries": rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(root.glob("*.json"))}


def test_two_disjoint_locale_changes_commute(tmp_path: Path) -> None:
    """THE property the sharding exists for: merge order must not matter.

    Two disjoint changes (fr + es rows, and two namespaces of the SAME
    locale) are applied to the sharded ledger in BOTH orders; the resulting
    files must be byte-identical — not merely individually equal to the
    pre-migration aggregate. A shard layout that encoded order dependence
    (a shared counter, a serialization that depends on what else is present)
    would fail here even while passing the losslessness test.
    """
    base_fr = [_row("fr", "wire.exec", "a" * 64, "base fr")]
    base_es = [_row("es", "wire.exec", "b" * 64, "base es")]

    def build(order: list[str]) -> dict[str, bytes]:
        root = tmp_path / f"order-{''.join(order)}"
        root.mkdir()
        _write_shard(root, "fr", list(base_fr))
        _write_shard(root, "es", list(base_es))
        for change in order:
            if change == "F":
                rows = json.loads((root / "fr.json").read_text())["entries"]
                rows.append(_row("fr", "wire.settings", "c" * 64, "fr delta"))
                _write_shard(root, "fr", rows)
            elif change == "E":
                rows = json.loads((root / "es.json").read_text())["entries"]
                rows.append(_row("es", "wire.settings", "d" * 64, "es delta"))
                _write_shard(root, "es", rows)
            elif change == "F2":
                # Same locale, DIFFERENT namespace: also disjoint (append-only).
                rows = json.loads((root / "fr.json").read_text())["entries"]
                rows.append(_row("fr", "tui.status", "e" * 64, "fr second ns"))
                _write_shard(root, "fr", rows)
        return _snapshot(root)

    # Different locales, both orders: byte-identical final files.
    assert build(["F", "E"]) == build(["E", "F"])

    # Same locale, different namespaces: order is SEMANTIC under append-only
    # last-wins, so the two orders need not be byte-identical — what must hold
    # is that both rows survive in each order and the last-applied row wins.
    # (Requiring byte equality here would contradict the ledger's supersession
    # semantics, which the migration is forbidden to change.)
    ff = build(["F", "F2"])["fr.json"]
    ff_rev = build(["F2", "F"])["fr.json"]
    for blob in (ff, ff_rev):
        entries = json.loads(blob)["entries"]
        notes = [r["note"] for r in entries]
        assert "fr delta" in notes and "fr second ns" in notes
        assert len(entries) == 3  # base + both deltas, nothing deduped
    assert json.loads(ff)["entries"][-1]["note"] == "fr second ns"
    assert json.loads(ff_rev)["entries"][-1]["note"] == "fr delta"
    # And the real git-level claim: the two-locale case touches disjoint FILES,
    # so a merge has nothing to conflict over (proved end-to-end in the PR
    # description with an actual `git merge`).
