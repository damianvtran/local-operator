"""Shard migration proof: what the aggregate held BEFORE sharding, unchanged.

This compares the shards against a PINNED description of the pre-migration
aggregate (per-locale row count + SHA-256 of the locale's ordered rows), not
against the aggregate file itself — that file is now generated FROM the shards,
so comparing the two would only prove the generator is deterministic.

The pins were captured from ``git show b12173b:i18n/ledger.json`` — the parent
commit of the sharding change — and cover the FIRST N rows of each shard:
append-only growth (every locale PR appends) leaves the prefix untouched, so a
new audit wave does not redden this test while a rewritten row does.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
SHARD_SCRIPT = REPO / "scripts" / "i18n" / "shard_ledger.py"

#: ``locale -> (row count, sha256 of that locale's first-count rows)``, taken from
#: the pre-migration aggregate at ``b12173b``. Pinned literals, never derived.
PRE_MIGRATION: dict[str, tuple[int, str]] = {
    "es": (96, "652ea3bdcdd47dc7c5ddd4192ce5254827d7269f76158a649d2d256834b50e8f"),
    "fr": (86, "0d691cab5d74056ec7914eecfcd6611994662dfbca68bbbdedb693519af4b918"),
    "hi": (52, "bf1771f9010d095b6c932fa66800dab479dee730cf45f9aebf31cb5ca73b4ac8"),
    "ru": (52, "21bbaade5c818419b31e2784fb6a61e25c04d8f43e2bacd630db7c2091377f80"),
    "ur": (45, "4a9194a519b85e40589a3b2e6286d7459c262c869c4b7b044161af68b5c2a4ca"),
    "vi": (52, "659fc832a6f155f4f1e3b70186af1a1e9def74e201571a8b021383045fd32e39"),
    "zh-CN": (98, "8a0045c4258adc4157af2ab463d2bb276594859dc869af578f387ca7ed229e7e"),
}


def _digest(rows: list[dict[str, Any]]) -> str:
    canon = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canon).hexdigest()


def test_shards_preserve_the_pre_migration_rows_exactly() -> None:
    """Every shard's history PREFIX is the pre-migration aggregate, row for row."""
    for locale, (count, digest) in PRE_MIGRATION.items():
        shard = json.loads(
            (REPO / "i18n" / "ledger" / f"{locale}.json").read_text(encoding="utf-8")
        )
        rows = shard["entries"]
        assert len(rows) >= count, (locale, "shard shrank below the migration baseline")
        assert _digest(rows[:count]) == digest, (locale, "pre-migration rows were altered")
        assert {row["locale"] for row in rows} == {locale}, locale


def test_shards_and_aggregate_yield_identical_shipped_set(i18n_generate) -> None:
    """Non-negotiable: aggregate path and shard path ship the same locales.

    The aggregate is a generated mirror, so this pins the two READ PATHS to each
    other (a mirror that lost a row would change the set); that the rows
    themselves are the pre-migration ones is
    ``test_shards_preserve_the_pre_migration_rows_exactly`` above.
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
    # per-locale ordered rows rather than raw array order. NO row-count literal:
    # the ledger grows on every locale wave, so a pinned total turns an
    # unrelated contribution into a red mechanism test. The invariant is that
    # both read paths see the same rows, whatever today's count is.
    assert len(shard_entries) == len(legacy_entries)
    assert len(shard_entries) > 0

    def per_locale(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            out.setdefault(row["locale"], []).append(row)
        return out

    assert per_locale(shard_entries) == per_locale(legacy_entries)
    assert from_shards == from_aggregate

    # The generated shipped.json on disk must agree with BOTH read paths. WHO
    # ships is product state that a locale wave legitimately changes; the
    # property here is that the two paths and the committed artifact cannot
    # disagree about it, so no literal locale list is pinned.
    generated = json.loads(
        (REPO / "local_operator/i18n/data/shipped.json").read_text(encoding="utf-8")
    )
    assert generated["locales"] == from_shards
    assert "en" in from_shards  # the one invariant that is not product state


def test_migration_is_order_preserving_and_lossless(tmp_path: Path) -> None:
    """The one-shot script: per-locale order preserved, no dedupe, idempotent."""
    aggregate = json.loads((REPO / "i18n" / "ledger.json").read_text(encoding="utf-8"))
    source = tmp_path / "ledger.json"
    source.write_text(json.dumps(aggregate), encoding="utf-8")
    out = tmp_path / "shards"

    def split(*extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SHARD_SCRIPT), *extra, "--ledger", str(source), "--out", str(out)],
            capture_output=True,
            text=True,
        )

    first_run = split()
    assert first_run.returncode == 0, first_run.stdout + first_run.stderr

    expected: dict[str, list[dict[str, Any]]] = {}
    for row in aggregate["entries"]:
        expected.setdefault(row["locale"], []).append(row)
    # Derived, not counted: every locale present in the aggregate must appear,
    # and no other. A new locale is a contribution, not a test failure.
    assert set(expected) == {row["locale"] for row in aggregate["entries"]}
    for locale, rows in expected.items():
        shard = json.loads((out / f"{locale}.json").read_text(encoding="utf-8"))
        # Row-for-row equality in the original per-locale order: this is the
        # assertion that forbids dedupe on (locale, namespace) — several
        # namespaces legitimately hold multiple rows at different hashes.
        assert shard["entries"] == rows, locale
        assert shard["locale"] == locale

    # Over EXISTING shards a plain re-split refuses: rewriting shards from the
    # aggregate would drop every shard row the aggregate does not carry.
    refused = split()
    assert refused.returncode == 2, refused.stdout
    assert "--fold-stranded" in refused.stdout

    # Idempotent with --force: identical bytes.
    first = {p.name: p.read_bytes() for p in out.glob("*.json")}
    forced = split("--force")
    assert forced.returncode == 0, forced.stdout + forced.stderr
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

    def row(status: str, sha: str) -> dict[str, Any]:
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
        json.dumps(
            {
                "schema": 1,
                "locale": "fr",
                "entries": [row("passed", "0" * 64), row("passed", current)],
            }
        ),
        encoding="utf-8",
    )
    assert "fr" in i18n_generate._shipped_locales()

    # Fresh-passed then failed: does NOT ship (a later failure supersedes).
    (shard_dir / "fr.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "locale": "fr",
                "entries": [row("passed", current), row("failed", current)],
            }
        ),
        encoding="utf-8",
    )
    assert "fr" not in i18n_generate._shipped_locales()

    # Fresh-passed then a LATER passed row at a DIFFERENT hash: does NOT ship.
    # This is the shape of the real multi-hash blocks (a re-audit appended after
    # a good one): the newest row decides, so a stale hash cannot resurrect an
    # earlier approval. The mirrored direction ships — together they prove the
    # rule keys on the LAST row rather than on any row.
    (shard_dir / "fr.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "locale": "fr",
                "entries": [row("passed", current), row("passed", "b" * 64)],
            }
        ),
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


def test_a_row_appended_to_the_aggregate_is_refused_not_dropped(
    i18n_generate: Any, tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """The mirror must never silently overwrite a contributor's row.

    `--check` already fails on drift, but a plain `generate.py` run WRITES the
    mirror — so a row appended to the aggregate (the pre-migration workflow)
    would be dropped without a word. `main()` refuses instead, naming the
    stranded (locale, namespace). Pure reordering is not stranded.
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
    monkeypatch.setattr(i18n_generate, "KEYS_DIR", tmp_path / "keys")
    monkeypatch.setattr(i18n_generate, "DATA_DIR", tmp_path / "data")
    shard_dir = tmp_path / "i18n" / "ledger"
    shard_dir.mkdir(parents=True)
    current = catalogues.catalogue_sha256("en", "wire.exec")
    shard_row = {
        "locale": "fr",
        "namespace": "wire.exec",
        "source_sha256": current,
        "status": "passed",
    }
    (shard_dir / "fr.json").write_text(
        json.dumps({"schema": 1, "locale": "fr", "entries": [shard_row]}), encoding="utf-8"
    )
    ledger_path = tmp_path / "i18n" / "ledger.json"
    # Reordering only (mirror content in a different order) is NOT stranded.
    ledger_path.write_text(json.dumps({"schema": 1, "entries": [shard_row]}), encoding="utf-8")
    assert i18n_generate._aggregate_rows_missing_from_shards() == []

    # A row present only in the aggregate IS stranded, and main() refuses.
    stray = {
        "locale": "fr",
        "namespace": "wire.settings",
        "source_sha256": current,
        "status": "passed",
    }
    ledger_path.write_text(
        json.dumps({"schema": 1, "entries": [stray, shard_row]}), encoding="utf-8"
    )
    assert i18n_generate._aggregate_rows_missing_from_shards() == ["fr/wire.settings"]
    # non-check mode: refusal, exit 2, and the row is still on disk afterwards.
    assert i18n_generate.main([]) == 2
    assert "wire.settings" in ledger_path.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "stranded aggregate row: fr/wire.settings" in out

    # The reviewed exception names what it drops and does not refuse. Needed for
    # the one legitimate removal direction (an authorised row deletion): the
    # ledger is append-only, so this flag is never a rebase tool.
    assert i18n_generate.main(["--accept-stranded"]) == 0
    dropped = capsys.readouterr().out
    assert "dropped aggregate row: fr/wire.settings" in dropped
    assert "wire.settings" not in ledger_path.read_text(encoding="utf-8")


def test_two_locales_touch_disjoint_files() -> None:
    """The structural claim: fr and es rows live in different files.

    Simulated as two concurrent edits applied to disjoint shard paths — git
    merges disjoint-file changes without conflict by construction; what this
    pins is that no shared file (the aggregate) is a WRITE target for locale
    contributions anymore: the aggregate is regenerated, never appended to.
    """
    # Derived from the ledger itself: one shard per locale the aggregate
    # carries, named for that locale. A hard-coded file list would have to be
    # edited every time a locale lands, which is the defect this whole change
    # exists to remove.
    aggregate_locales = {
        row["locale"]
        for row in json.loads((REPO / "i18n" / "ledger.json").read_text(encoding="utf-8"))[
            "entries"
        ]
    }
    shard_locales = {p.stem for p in (REPO / "i18n" / "ledger").glob("*.json")}
    assert shard_locales == aggregate_locales
    # Each shard names exactly its own locale in every row.
    for path in (REPO / "i18n" / "ledger").glob("*.json"):
        shard = json.loads(path.read_text(encoding="utf-8"))
        assert {row["locale"] for row in shard["entries"]} == {shard["locale"]}
        assert path.stem == shard["locale"]


# Derived at collection time, so a new locale shard is covered without editing
# this file — the point of a mechanism test.
SHARD_LOCALES = sorted(p.stem for p in (REPO / "i18n" / "ledger").glob("*.json"))


@pytest.mark.parametrize("locale", SHARD_LOCALES)
def test_shard_round_trips_through_aggregate_mirror(locale: str, i18n_generate) -> None:
    """Every shard's rows appear verbatim in the regenerated aggregate."""
    shard = json.loads((REPO / "i18n" / "ledger" / f"{locale}.json").read_text(encoding="utf-8"))
    aggregate = json.loads(i18n_generate._render_aggregate())
    rows = [r for r in aggregate["entries"] if r["locale"] == locale]
    assert rows == shard["entries"]


def _row(locale: str, namespace: str, sha: str, note: str) -> dict[str, Any]:
    return {
        "locale": locale,
        "namespace": namespace,
        "source_sha256": sha,
        "status": "passed",
        "note": note,
    }


def _stage_scripts(root: Path) -> None:
    """Copy the REAL i18n scripts into a scratch tree so REPO resolves there.

    ``generate.py`` derives every path from ``Path(__file__).parents[2]``, so
    running the checkout's script against a temp tree would read and WRITE the
    checkout itself. Staged copies keep the run hermetic while executing the real
    code; ``PYTHONPATH`` still points at the checkout so the ``local_operator``
    package imports.
    """
    target = root / "scripts" / "i18n"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("generate.py", "shard_ledger.py", "emit.mjs"):
        shutil.copy2(REPO / "scripts" / "i18n" / name, target / name)


def _run(
    root: Path, script: str, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a staged script with the scratch tree as its REPO."""
    env = {**env, "PYTHONPATH": str(REPO)}
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "i18n" / script), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _git_env(tmp_path: Path) -> dict[str, str]:
    """A hermetic git env: no user/system config, no terminal or credential prompts.

    A redirected HOME with the system gitconfig in play makes git ask macOS
    Keychain Services for a login keychain (the credential helper), which pops a
    system dialog at the operator's desk; GIT_CONFIG_SYSTEM=/dev/null plus
    GIT_TERMINAL_PROMPT=0 removes that path entirely.
    """
    return {
        **os.environ,
        "HOME": str(tmp_path),
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "sim",
        "GIT_AUTHOR_EMAIL": "sim@example.invalid",
        "GIT_COMMITTER_NAME": "sim",
        "GIT_COMMITTER_EMAIL": "sim@example.invalid",
    }


def test_two_locale_branches_merge_in_either_order(tmp_path: Path) -> None:
    """Existing-locale appends commute, proved with real git and real scripts.

    Two branches each append an audit row to their OWN EXISTING locale's shard
    and run the REAL ``generate.py`` (staged into a scratch tree); they are
    merged in both orders in a scratch git repo. Asserted: zero conflicts,
    identical merged shard AND aggregate blobs, and both appends present
    afterwards — nothing here approximates the tools.

    SCOPE, precisely: this covers two changes to locales that already have a
    shard block in the mirror, because the mirror is locale-grouped and each
    append lands in its own block. It does NOT claim that ADDING a locale is
    conflict-free: a new shard contributes a new block to `i18n/ledger.json`,
    whose insertion point sits next to another locale's rows, so a concurrent
    append there can still collide on the mirror (measured by QA with a new
    `fo.json` against an `es` append). The shards themselves never collide; the
    generated mirror is the only file that can, and regeneration — never a
    hand-merge — is its resolution.
    """
    if not shutil.which("git"):
        pytest.skip("git is required to prove the merge claim")
    work = tmp_path / "repo"
    (work / "i18n" / "ledger").mkdir(parents=True)
    _stage_scripts(work)
    for locale in ("fr", "es"):
        # A few rows each, so the mirror has a realistic per-locale block: the
        # conflict-freedom claim is about appends landing in DISTANT blocks of
        # the generated aggregate, and an empty aggregate would put both
        # appends at the same line.
        (work / "i18n" / "ledger" / f"{locale}.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "locale": locale,
                    "entries": [
                        _row(locale, "wire.exec", f"{i:064x}", f"base {locale} {i}")
                        for i in range(3)
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (work / "i18n" / "ledger.json").write_text(
        json.dumps({"schema": 1, "entries": []}) + "\n", encoding="utf-8"
    )
    env = _git_env(tmp_path)

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=work, capture_output=True, text=True, check=False, env=env
        )

    assert git("init", "--quiet", "-b", "main").returncode == 0
    # Render the BASE mirror with the real generator, so both branches start
    # from the file shape production actually commits.
    seeded = _run(work, "generate.py", [], env)
    assert seeded.returncode == 0, seeded.stdout + seeded.stderr
    assert len(json.loads((work / "i18n" / "ledger.json").read_text())["entries"]) == 6
    assert git("add", "-A").returncode == 0
    assert git("commit", "--quiet", "-m", "base").returncode == 0

    def add_branch(name: str, locale: str, note: str) -> None:
        assert git("checkout", "--quiet", "-b", name, "main").returncode == 0
        path = work / "i18n" / "ledger" / f"{locale}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "locale": locale,
                    "entries": [*data["entries"], _row(locale, "wire.settings", "c" * 64, note)],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        done = _run(work, "generate.py", [], env)
        assert done.returncode == 0, done.stdout + done.stderr
        assert git("add", "-A").returncode == 0
        assert git("commit", "--quiet", "-m", note).returncode == 0

    add_branch("fr-branch", "fr", "fr delta")
    add_branch("es-branch", "es", "es delta")
    branches = ("fr-branch", "es-branch")

    def merge_and_hash(first: str, second: str) -> dict[str, str]:
        assert git("checkout", "--quiet", first).returncode == 0
        merged = git("merge", "--no-edit", second)
        assert merged.returncode == 0, f"merge conflict:\n{merged.stdout}{merged.stderr}"
        out: dict[str, str] = {}
        for path in ("i18n/ledger/fr.json", "i18n/ledger/es.json", "i18n/ledger.json"):
            blob = git("rev-parse", f"HEAD:{path}")
            assert blob.returncode == 0, path
            out[path] = blob.stdout.strip()
        return out

    forward = merge_and_hash(*branches)
    reverse = merge_and_hash(branches[1], branches[0])
    assert forward == reverse, "merge order changed the merged ledger blobs"

    # Both appends survived, in the mirror as well as in the shards.
    merged_aggregate = json.loads((work / "i18n" / "ledger.json").read_text(encoding="utf-8"))
    notes = {row.get("note") for row in merged_aggregate["entries"]}
    assert {"fr delta", "es delta"} <= notes


def test_rebased_branch_rows_are_folded_not_lost(tmp_path: Path) -> None:
    """An open locale PR (#3131 shape) rebases onto a sharded main: rows survive.

    The old workflow appended to the aggregate. After sharding, such a branch's
    aggregate is the mirror PLUS its own rows and the shards know nothing about
    them. The documented sequence is proved end to end: ``generate.py`` refuses
    (F1), ``shard_ledger.py --fold-stranded`` moves the rows into their locale's
    shard preserving their order, and both the shard and the mirror carry them
    afterwards.
    """
    work = tmp_path / "tree"
    (work / "i18n" / "ledger").mkdir(parents=True)
    _stage_scripts(work)
    env = {**os.environ, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_TERMINAL_PROMPT": "0"}
    shard = [_row("fr", "wire.exec", "a" * 64, "pre-existing fr row")]
    (work / "i18n" / "ledger" / "fr.json").write_text(
        json.dumps({"schema": 1, "locale": "fr", "entries": shard}, indent=2) + "\n",
        encoding="utf-8",
    )
    appended = [_row("fr", "wire.errors", f"{i:064d}", f"branch row {i}") for i in range(21)]
    (work / "i18n" / "ledger.json").write_text(
        json.dumps({"schema": 1, "entries": [*shard, *appended]}, indent=2) + "\n",
        encoding="utf-8",
    )

    refused = _run(work, "generate.py", [], env)
    assert refused.returncode == 2, refused.stdout
    assert "fr/wire.errors (21 rows)" in refused.stdout
    assert len(json.loads((work / "i18n" / "ledger.json").read_text())["entries"]) == 22

    folded = _run(work, "shard_ledger.py", ["--fold-stranded"], env)
    assert folded.returncode == 0, folded.stdout + folded.stderr
    assert "folded 21 stranded row(s) into fr.json" in folded.stdout
    shard_after = json.loads((work / "i18n" / "ledger" / "fr.json").read_text())["entries"]
    assert len(shard_after) == 22, "a folded row was dropped"
    assert [row["note"] for row in shard_after[-21:]] == [f"branch row {i}" for i in range(21)]

    again = _run(work, "shard_ledger.py", ["--fold-stranded"], env)
    assert again.returncode == 0
    assert "nothing stranded" in again.stdout
