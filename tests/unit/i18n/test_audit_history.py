"""Audit storage preserves ordered receipts, not timestamp-selected winners."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.i18n import generate


def receipt(locale="fr", namespace="wire.test", *, note="first"):
    return {
        "locale": locale,
        "namespace": namespace,
        "source_sha256": "a" * 64,
        "translator": {"agent": "translator", "model": "test"},
        "auditor": {"agent": "auditor", "model": "test"},
        "status": "passed",
        "note": note,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def legacy(root, rows=None):
    rows = [receipt()] if rows is None else rows
    write_json(root / "i18n/ledger.json", {"schema": 1, "entries": rows})
    for locale in sorted({row["locale"] for row in rows}):
        write_json(
            root / f"i18n/ledger/{locale}.json",
            {
                "schema": 1,
                "entries": [row for row in rows if row["locale"] == locale],
            },
        )
    return rows


def test_unclaimed_suffix_refuses_before_generator_output(tmp_path, monkeypatch):
    """An old reader must not silently ignore new authority without its marker."""
    legacy(tmp_path)
    write_json(
        tmp_path / "i18n/audit-history/fr/wire.test.json",
        {
            "schema": 1,
            "locale": "fr",
            "namespace": "wire.test",
            "history_sha256": hashlib.sha256(b"[]").hexdigest(),
            "entries": [receipt(note="unclaimed")],
        },
    )
    monkeypatch.setattr(generate, "LEDGER", tmp_path / "i18n/ledger.json")
    monkeypatch.setattr(generate, "LEDGER_SHARDS", tmp_path / "i18n/ledger")
    import pytest

    with pytest.raises(ValueError, match="marker"):
        generate._ledger_entries()


def test_migration_is_marker_only_and_lossless(tmp_path):
    from scripts.i18n import ledger

    rows = legacy(tmp_path, [receipt(), receipt(note="failed", locale="es")])
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")}
    report = ledger.migrate(tmp_path, apply=False)
    assert report["receipts"] == 2
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    ledger.migrate(tmp_path, apply=True)
    assert ledger.load_entries(tmp_path) == [rows[1], rows[0]]
    for path, data in before.items():
        assert (tmp_path / path).read_bytes() == data
    marker = (tmp_path / "i18n/audit-history.json").read_bytes()
    ledger.migrate(tmp_path, apply=True)
    assert (tmp_path / "i18n/audit-history.json").read_bytes() == marker
    assert json.loads(ledger.export_aggregate(tmp_path))["entries"] == [rows[1], rows[0]]


def test_append_import_and_stale_expected_history(tmp_path):
    import pytest

    from scripts.i18n import ledger

    rows = legacy(tmp_path)
    ledger.migrate(tmp_path, apply=True)
    key = ("fr", "wire.test")
    first_digest = ledger.history_sha256(rows)
    second = receipt(note="failed")
    second["status"] = "failed"
    second["unknown"] = {"retain": [1, "é"]}
    ledger.append(tmp_path, second, expected_history_sha=first_digest, apply=False)
    assert ledger.load_entries(tmp_path) == rows
    ledger.append(tmp_path, second, expected_history_sha=first_digest, apply=True)
    with pytest.raises(ValueError, match="stale"):
        ledger.append(
            tmp_path, receipt(note="third"), expected_history_sha=first_digest, apply=True
        )
    with pytest.raises(ValueError, match="duplicate"):
        ledger.append(
            tmp_path,
            second,
            expected_history_sha=ledger.history_sha256(rows + [second]),
            apply=True,
        )
    # A detached stale branch is a no-op; a longer prefix imports only its suffix.
    source = tmp_path / "detached.json"
    write_json(source, {"schema": 1, "entries": rows})
    assert ledger.import_legacy(tmp_path, source, "aggregate", apply=True) == []
    third = receipt(note="third")
    write_json(source, {"schema": 1, "entries": rows + [second, third]})
    assert len(ledger.import_legacy(tmp_path, source, "aggregate", apply=True)) == 1
    assert ledger.load_entries(tmp_path) == rows + [second, third]
    assert ledger.import_legacy(tmp_path, source, "aggregate", apply=True) == []
    write_json(source, {"schema": 1, "entries": rows + [receipt(note="competing")]})
    with pytest.raises(ValueError, match="non-prefix.*fr/wire.test"):
        ledger.import_legacy(tmp_path, source, "aggregate", apply=True)
    assert key == (third["locale"], third["namespace"])


def test_invalid_storage_refuses(tmp_path):
    import pytest

    from scripts.i18n import ledger

    rows = legacy(tmp_path)
    ledger.migrate(tmp_path, apply=True)
    marker = tmp_path / "i18n/audit-history.json"
    marker_bytes = marker.read_bytes()
    snapshot = tmp_path / "i18n/ledger/fr.json"
    original = snapshot.read_bytes()
    snapshot.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="baseline.*import-legacy"):
        ledger.load_entries(tmp_path)
    snapshot.write_bytes(original)
    extra = tmp_path / "i18n/ledger/xx.json"
    write_json(extra, {"schema": 1, "entries": []})
    with pytest.raises(ValueError, match="inventory"):
        ledger.load_entries(tmp_path)
    extra.unlink()
    ledger.append(
        tmp_path,
        receipt(note="second"),
        expected_history_sha=ledger.history_sha256(rows),
        apply=True,
    )
    suffix = tmp_path / "i18n/audit-history/fr/wire.test.json"
    payload = json.loads(suffix.read_text())
    payload["history_sha256"] = "0" * 64
    write_json(suffix, payload)
    with pytest.raises(ValueError, match="digest"):
        ledger.load_entries(tmp_path)
    marker.unlink()
    with pytest.raises(ValueError, match="marker"):
        ledger.load_entries(tmp_path)
    marker.write_bytes(marker_bytes)


def test_stranded_aggregate_rows_never_disappear(tmp_path):
    import pytest

    from scripts.i18n import ledger

    legacy(tmp_path)
    write_json(
        tmp_path / "i18n/ledger.json",
        {"schema": 1, "entries": [receipt(), receipt(note="stranded")]},
    )
    with pytest.raises(ValueError, match="stranded"):
        ledger.load_entries(tmp_path)
    with pytest.raises(ValueError, match="stranded"):
        ledger.migrate(tmp_path, apply=True)
    assert not (tmp_path / "i18n/audit-history.json").exists()


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/i18n/ledger.py"


def isolated_env(root):
    home = root / "home"
    home.mkdir(exist_ok=True)
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(home / "config"),
        "TMPDIR": str(root),
        "TERM": "xterm-256color",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
    }


def git(root, *args, ok=True):
    result = subprocess.run(
        ["git", "-c", "user.name=Audit fixture", "-c", "user.email=audit@example.invalid", *args],
        cwd=root,
        env=isolated_env(root),
        capture_output=True,
        text=True,
    )
    if ok:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def cli(root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        cwd=root,
        env=isolated_env(root),
        capture_output=True,
        text=True,
    )


def commit(root, message):
    git(root, "add", "i18n")
    git(root, "commit", "-qm", message)
    return git(root, "rev-parse", "HEAD").stdout.strip()


def git_fixture(root, *, migrate=True):
    from scripts.i18n import ledger

    legacy(root)
    git(root, "init", "-q", "-b", "main")
    old = commit(root, "legacy")
    if migrate:
        ledger.migrate(root, apply=True)
        return commit(root, "migrate")
    return old


@pytest.mark.parametrize(
    "other_key", [("fr", "wire.other"), ("es", "wire.test"), ("fr", "wire.test")]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_real_git_merge_orders(tmp_path, other_key, reverse):
    from scripts.i18n import ledger

    git_fixture(tmp_path)
    # Exercise an EXISTING suffix too: same-key conflict must come from the
    # effective-history digest, not merely Git's easy add/add conflict case.
    ledger.append(
        tmp_path,
        receipt(note="common suffix"),
        expected_history_sha=ledger.history_sha256([receipt()]),
        apply=True,
    )
    base = commit(tmp_path, "shared suffix")
    for branch, key, note in [("left", ("fr", "wire.test"), "left"), ("right", other_key, "right")]:
        git(tmp_path, "checkout", "-qb", branch, base)
        current = [r for r in ledger.load_entries(tmp_path) if (r["locale"], r["namespace"]) == key]
        ledger.append(
            tmp_path,
            receipt(*key, note=note),
            expected_history_sha=ledger.history_sha256(current),
            apply=True,
        )
        commit(tmp_path, note)
    first, second = ("right", "left") if reverse else ("left", "right")
    git(tmp_path, "checkout", "-q", first)
    result = git(tmp_path, "merge", "--no-edit", second, ok=False)
    if other_key == ("fr", "wire.test"):
        assert result.returncode == 1
        assert "CONFLICT" in result.stdout
        assert "<<<<<<<" in (tmp_path / "i18n/audit-history/fr/wire.test.json").read_text()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        ledger.validate_base(tmp_path, base)
        assert len(ledger.load_entries(tmp_path)) == 4


def test_two_all_locale_namespace_waves_merge_identically_in_both_orders(tmp_path):
    from scripts.i18n import ledger

    exports = []
    for reverse in (False, True):
        work = tmp_path / str(reverse)
        work.mkdir()
        git_fixture(work)
        keys = [
            (locale, namespace)
            for locale in ("fr", "es", "hi", "ru", "ur", "vi", "zh-CN")
            for namespace in ("wire.wave_a", "wire.wave_b")
        ]
        for key in keys:
            ledger.append(
                work,
                receipt(*key, note="common"),
                expected_history_sha=ledger.history_sha256([]),
                apply=True,
            )
        base = commit(work, "common existing partitions")
        pinned = {p: p.read_bytes() for p in (work / "i18n/ledger").glob("*.json")}
        pinned[work / "i18n/ledger.json"] = (work / "i18n/ledger.json").read_bytes()
        pinned[work / "i18n/audit-history.json"] = (work / "i18n/audit-history.json").read_bytes()
        for branch, namespace in (("a", "wire.wave_a"), ("b", "wire.wave_b")):
            git(work, "checkout", "-qb", branch, base)
            for locale, ns in keys:
                if ns == namespace:
                    history = [receipt(locale, ns, note="common")]
                    ledger.append(
                        work,
                        receipt(locale, ns, note=branch),
                        expected_history_sha=ledger.history_sha256(history),
                        apply=True,
                    )
            commit(work, namespace)
        first, second = ("b", "a") if reverse else ("a", "b")
        git(work, "checkout", "-q", first)
        result = git(work, "merge", "--no-edit", second)
        assert result.returncode == 0 and "CONFLICT" not in result.stdout
        ledger.validate_base(work, base)
        assert len(ledger.load_entries(work)) == 29
        assert all(path.read_bytes() == data for path, data in pinned.items())
        exports.append(ledger.export_aggregate(work))
    assert exports[0] == exports[1]


def test_old_append_auto_merges_but_stale_migration_fails_closed(tmp_path):
    """Green Git merge is NOT a valid migration integration preflight."""
    from scripts.i18n import ledger

    base = git_fixture(tmp_path, migrate=False)
    git(tmp_path, "checkout", "-qb", "migration")
    ledger.migrate(tmp_path, apply=True)
    commit(tmp_path, "one-time migration")
    git(tmp_path, "checkout", "-q", "main")
    rows = [receipt(), receipt(note="old main append")]
    legacy(tmp_path, rows)
    actual_main = commit(tmp_path, "old writer on main")
    git(tmp_path, "checkout", "-q", "migration")
    assert git(tmp_path, "merge", "--no-edit", "main").returncode == 0
    result = cli(tmp_path, "validate", "--base", actual_main)
    assert result.returncode == 2 and "stale audit baseline" in result.stderr
    for path in ("i18n/ledger.json", "i18n/ledger/fr.json"):
        assert json.loads((tmp_path / path).read_text())["entries"] == rows
    # Legitimate pre-integration recovery: rebuild the one-time migration from
    # ACTUAL legacy main, not a digest repair on the mixed merge result.
    git(tmp_path, "checkout", "-qb", "reconstructed", actual_main)
    ledger.migrate(tmp_path, apply=True)
    assert ledger.load_entries(tmp_path) == rows
    ledger.validate_base(tmp_path, actual_main)
    assert len(rows) > len(
        json.loads(git(tmp_path, "show", f"{base}:i18n/ledger.json").stdout)["entries"]
    )


def test_base_check_rejects_recomputed_rewrite_and_marker_removal(tmp_path):
    from scripts.i18n import ledger

    base = git_fixture(tmp_path)
    ledger.append(
        tmp_path,
        receipt(note="second"),
        expected_history_sha=ledger.history_sha256([receipt()]),
        apply=True,
    )
    suffix_head = commit(tmp_path, "append")
    path = tmp_path / "i18n/audit-history/fr/wire.test.json"
    payload = json.loads(path.read_text())
    payload["entries"][0]["note"] = "rewritten"
    payload["history_sha256"] = ledger.history_sha256([receipt(), *payload["entries"]])
    write_json(path, payload)
    assert len(ledger.load_entries(tmp_path)) == 2
    with pytest.raises(ValueError, match="append-only"):
        ledger.validate_base(tmp_path, suffix_head)
    path.unlink()
    with pytest.raises(ValueError, match="deletion"):
        ledger.validate_base(tmp_path, suffix_head)
    (tmp_path / "i18n/audit-history.json").unlink()
    with pytest.raises(ValueError, match="marker removal"):
        ledger.validate_base(tmp_path, base)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-member",
        "symlink",
        "wrong-key",
        "traversal",
        "duplicate-record",
        "bad-provenance",
        "unsupported-schema",
    ],
)
def test_malformed_history_refuses(tmp_path, mutation):
    from scripts.i18n import ledger

    legacy(tmp_path)
    ledger.migrate(tmp_path, apply=True)
    ledger.append(
        tmp_path,
        receipt(note="second"),
        expected_history_sha=ledger.history_sha256([receipt()]),
        apply=True,
    )
    path = tmp_path / "i18n/audit-history/fr/wire.test.json"
    payload = json.loads(path.read_text())
    if mutation == "duplicate-member":
        path.write_text(path.read_text().replace('"schema": 1,', '"schema": 1, "schema": 1,'))
    elif mutation == "symlink":
        target = tmp_path / "detached.json"
        path.rename(target)
        path.symlink_to(target)
    else:
        if mutation == "wrong-key":
            payload["namespace"] = "wrong"
        elif mutation == "traversal":
            payload["entries"][0]["namespace"] = "../escape"
        elif mutation == "duplicate-record":
            payload["entries"] = [receipt()]
        elif mutation == "bad-provenance":
            payload["entries"][0]["translator"] = []
        else:
            payload["schema"] = 2
        write_json(path, payload)
    with pytest.raises(ValueError):
        ledger.load_entries(tmp_path)


def test_cli_dry_run_export_protection_and_old_writer_refusal(tmp_path):
    from scripts.i18n import ledger

    rows = legacy(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert cli(tmp_path, "migrate", "--dry-run").returncode == 0
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    assert cli(tmp_path, "migrate", "--apply").returncode == 0
    assert json.loads(cli(tmp_path, "export").stdout)["entries"] == rows
    for protected in (
        "i18n/ledger.json",
        "i18n/ledger/fr.json",
        "i18n/audit-history.json",
        "i18n/audit-history/fr/new.json",
    ):
        result = cli(tmp_path, "export", "--output", str(tmp_path / protected))
        assert result.returncode == 2 and "cannot overwrite" in result.stderr
    alias = tmp_path / "snapshot-alias.json"
    os.link(tmp_path / "i18n/ledger.json", alias)
    result = cli(tmp_path, "export", "--output", str(alias))
    assert result.returncode == 2 and "cannot overwrite" in result.stderr
    for flag in ([], ["--force"], ["--fold-stranded"]):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT.with_name("shard_ledger.py")),
                "--ledger",
                str(tmp_path / "i18n/ledger.json"),
                "--out",
                str(tmp_path / "i18n/ledger"),
                *flag,
            ],
            env=isolated_env(tmp_path),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2 and "import-legacy" in result.stderr
    # A modified destination snapshot can never masquerade as detached input.
    with pytest.raises(ValueError, match="detached"):
        ledger.import_legacy(tmp_path, tmp_path / "i18n/ledger.json", "aggregate", apply=True)


def test_old_writer_refuses_dangling_migration_marker(tmp_path):
    legacy(tmp_path)
    marker = tmp_path / "i18n/audit-history.json"
    marker.symlink_to(tmp_path / "missing-marker.json")
    before = (tmp_path / "i18n/ledger/fr.json").read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT.with_name("shard_ledger.py")),
            "--ledger",
            str(tmp_path / "i18n/ledger.json"),
            "--out",
            str(tmp_path / "i18n/ledger"),
            "--force",
        ],
        env=isolated_env(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "import-legacy" in result.stderr
    assert (tmp_path / "i18n/ledger/fr.json").read_bytes() == before


@pytest.mark.parametrize("layout", ["aggregate", "locale-shards"])
def test_import_layouts_and_all_or_nothing_preflight(tmp_path, layout):
    from scripts.i18n import ledger

    destination, detached = tmp_path / "destination", tmp_path / "detached"
    legacy(destination)
    ledger.migrate(destination, apply=True)
    incoming = [receipt(), receipt(note="imported"), receipt("es", note="other key")]
    legacy(detached, incoming)
    source = detached / "i18n/ledger.json" if layout == "aggregate" else detached
    before = {p: p.read_bytes() for p in destination.rglob("*.json")}
    paths = ledger.import_legacy(destination, source, layout, apply=False)
    assert len(paths) == 2
    assert {p: p.read_bytes() for p in destination.rglob("*.json")} == before
    assert ledger.import_legacy(destination, source, layout, apply=True) == paths
    assert ledger.import_legacy(destination, source, layout, apply=True) == []
    # A later competing key invalidates the whole plan BEFORE any earlier key
    # may be written. Cross-key source ordering never establishes authority.
    legacy(detached, [receipt("es", note="new earlier key"), receipt(note="conflict")])
    before = {p: p.read_bytes() for p in destination.rglob("*.json")}
    with pytest.raises(ValueError, match="non-prefix"):
        ledger.import_legacy(destination, source, layout, apply=True)
    assert {p: p.read_bytes() for p in destination.rglob("*.json")} == before
    # Explicit selected format is not permission to ignore stranded rows.
    write_json(detached / "i18n/ledger.json", {"schema": 1, "entries": [receipt(note="stranded")]})
    with pytest.raises(ValueError, match="stranded"):
        ledger.import_legacy(destination, source, layout, apply=True)


@pytest.mark.parametrize("aggregate_only", [False, True])
def test_first_migration_base_proof_and_snapshot_bytes(tmp_path, aggregate_only):
    from scripts.i18n import ledger

    base = git_fixture(tmp_path, migrate=False)
    if aggregate_only:
        git(tmp_path, "rm", "i18n/ledger/fr.json")
        base = commit(tmp_path, "aggregate-only legacy")
    ledger.migrate(tmp_path, apply=True)
    ledger.validate_base(tmp_path, base)
    # A self-consistent newly recomputed marker may not bless a rewritten
    # baseline, even if that rewrite preserves all per-key values.
    (tmp_path / "i18n/audit-history.json").unlink()
    snapshot = tmp_path / "i18n/ledger.json"
    snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    ledger.migrate(tmp_path, apply=True)
    with pytest.raises(ValueError, match="snapshot bytes"):
        ledger.validate_base(tmp_path, base)


def test_normal_check_requires_readable_base_and_forbids_ci_escape(tmp_path):
    base = git_fixture(tmp_path)

    def gate(*, base_ref=None, structural_only=False, ci_env=None, checkout=tmp_path):
        # Per-subprocess overlays model local vs real CI explicitly. Never clear
        # the test runner's CI environment or borrow its GitHub event/base.
        environment = {
            **isolated_env(tmp_path),
            "PYTHONPATH": str(SCRIPT.parents[2]),
            **(ci_env or {}),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,sys; from pathlib import Path; from scripts.i18n import check; "
                "check.REPO=Path(sys.argv[1]); "
                "print(json.dumps(check.check_audit_history(sys.argv[2] or None, "
                "structural_only=sys.argv[3]=='true')))",
                str(checkout),
                base_ref or "",
                str(structural_only).lower(),
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    assert gate(base_ref=base) == []
    assert gate(base_ref="definitely-missing-ref")
    assert gate(structural_only=True) == []
    assert "forbidden in CI" in gate(structural_only=True, ci_env={"CI": "true"})[0]
    pr_environment = {"CI": "true", "GITHUB_EVENT_NAME": "pull_request"}
    assert "actual PR audit base" in gate(ci_env=pr_environment)[0]
    event = tmp_path / "event.json"
    write_json(event, {"pull_request": {"base": {"sha": base, "ref": "main"}}})
    pr_environment["GITHUB_EVENT_PATH"] = str(event)
    assert gate(ci_env=pr_environment) == []
    assert "actual PR base" in gate(base_ref="definitely-missing-ref", ci_env=pr_environment)[0]
    repo_less = tmp_path / "repo-less"
    legacy(repo_less)
    assert "readable audit base required" in gate(checkout=repo_less, ci_env={"CI": "true"})[0]


def test_missing_marker_and_stale_baseline_refuse_before_generator_writes(tmp_path, monkeypatch):
    from scripts.i18n import ledger

    legacy(tmp_path)
    ledger.migrate(tmp_path, apply=True)
    monkeypatch.setattr(generate, "LEDGER", tmp_path / "i18n/ledger.json")
    monkeypatch.setattr(generate, "LEDGER_SHARDS", tmp_path / "i18n/ledger")
    called = []
    monkeypatch.setattr(generate, "_run_emitter", lambda path: called.append(path))
    snapshot = tmp_path / "i18n/ledger.json"
    snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    for args in ([], ["--check"], ["--accept-stranded"]):
        assert generate.main(args) == 2
    assert not called
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
