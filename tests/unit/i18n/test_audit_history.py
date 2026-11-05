"""Audit storage preserves ordered receipts, not timestamp-selected winners."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
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


@pytest.mark.parametrize(
    "first_action,second_action",
    [("append", "append"), ("append", "import-legacy"), ("import-legacy", "append")],
)
def test_process_mutations_never_acknowledge_lost_receipts(tmp_path, first_action, second_action):
    """Hold only A before replacement; B must refuse, never wait at a two-party barrier."""
    from scripts.i18n import ledger

    rows = legacy(tmp_path)
    ledger.migrate(tmp_path, apply=True)
    expected = ledger.history_sha256(rows)
    worker = tmp_path / "worker.py"
    worker.write_text("""
import sys,time
from pathlib import Path
from scripts.i18n import ledger
root=Path(sys.argv[1]); label=sys.argv[2]; action=sys.argv[3]
if label == 'a':
    original=ledger._write_suffix
    def held_replace(*args):
        (root/'ready').write_text('ready')
        deadline=time.monotonic()+15
        while not (root/'release').exists():
            if time.monotonic()>deadline: raise RuntimeError('parent failed to release A')
            time.sleep(.01)
        return original(*args)
    ledger._write_suffix=held_replace
args=['--root',str(root),action]
if action=='append':
    args += ['--receipt',str(root/(label+'-receipt.json'))]
    args += ['--expected-history-sha',sys.argv[4]]
else: args += ['--source',str(root/(label+'-source.json')),'--format','aggregate']
sys.exit(ledger.main(args+['--apply']))
""")
    for label in ("a", "b"):
        row = receipt(note=f"writer-{label}")
        write_json(tmp_path / f"{label}-receipt.json", row)
        write_json(tmp_path / f"{label}-source.json", {"schema": 1, "entries": rows + [row]})
    environment = {**isolated_env(tmp_path), "PYTHONPATH": str(SCRIPT.parents[2])}
    processes = []
    try:
        first = subprocess.Popen(
            [sys.executable, str(worker), str(tmp_path), "a", first_action, expected],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(first)
        deadline = time.monotonic() + 15
        while not (tmp_path / "ready").exists():
            assert first.poll() is None, first.communicate()
            assert time.monotonic() < deadline, "A never reached replacement boundary"
            time.sleep(0.01)
        second = subprocess.Popen(
            [sys.executable, str(worker), str(tmp_path), "b", second_action, expected],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(second)
        second_output = second.communicate(timeout=15)
    finally:
        (tmp_path / "release").write_text("release")
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    first_output = first.communicate()
    notes = [row["note"] for row in ledger.load_entries(tmp_path)]
    print(
        "PROCESS MUTATIONS",
        first_action,
        second_action,
        "exits",
        first.returncode,
        second.returncode,
        "notes",
        notes,
        "second",
        second_output,
    )
    assert first.returncode == 0, first_output
    assert second.returncode == 2 and (
        "busy" in second_output[1] or "stale" in second_output[1]
    ), second_output
    assert notes == ["first", "writer-a"]
    # Once A's transaction is finished, the old expected digest is rechecked,
    # not accepted from a read that preceded ownership of the exclusion.
    if second_action == "append":
        result = cli(
            tmp_path,
            "append",
            "--receipt",
            str(tmp_path / "b-receipt.json"),
            "--expected-history-sha",
            expected,
            "--apply",
        )
        assert result.returncode == 2 and "stale" in result.stderr


@pytest.mark.parametrize("original,replacement", [(True, 1), (1, 1.0)])
def test_json_typed_prefix_import_refuses_without_writes(tmp_path, original, replacement):
    from scripts.i18n import ledger

    rows = legacy(tmp_path)
    ledger.migrate(tmp_path, apply=True)
    earlier = receipt(note="suffix")
    earlier["second_audit"] = {"passed": original}
    ledger.append(tmp_path, earlier, expected_history_sha=ledger.history_sha256(rows), apply=True)
    changed = {**earlier, "second_audit": {"passed": replacement}}
    source = tmp_path / "detached.json"
    write_json(source, {"schema": 1, "entries": [*rows, changed, receipt(note="new")]})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = cli(
        tmp_path, "import-legacy", "--source", str(source), "--format", "aggregate", "--apply"
    )
    print("TYPED IMPORT", repr(original), repr(replacement), result.returncode, result.stderr)
    assert result.returncode == 2 and "non-prefix" in result.stderr
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    # Exact canonical prefixes still append, and an older exact source no-ops.
    write_json(source, {"schema": 1, "entries": [*rows, earlier, receipt(note="new")]})
    exact = cli(
        tmp_path, "import-legacy", "--source", str(source), "--format", "aggregate", "--apply"
    )
    assert exact.returncode == 0 and len(json.loads(exact.stdout)["paths"]) == 1
    write_json(source, {"schema": 1, "entries": rows})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    older = cli(
        tmp_path, "import-legacy", "--source", str(source), "--format", "aggregate", "--apply"
    )
    assert older.returncode == 0 and json.loads(older.stdout) == {"paths": []}
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


@pytest.mark.parametrize("original,replacement", [(True, 1), (1, 1.0)])
def test_json_typed_committed_history_rewrite_refuses(tmp_path, original, replacement):
    from scripts.i18n import ledger

    git_fixture(tmp_path)
    earlier = receipt(note="suffix")
    earlier["extension"] = {"value": original}
    ledger.append(
        tmp_path, earlier, expected_history_sha=ledger.history_sha256([receipt()]), apply=True
    )
    base = commit(tmp_path, "typed history")
    path = tmp_path / "i18n/audit-history/fr/wire.test.json"
    payload = json.loads(path.read_text())
    payload["entries"][0]["extension"]["value"] = replacement
    payload["history_sha256"] = ledger.history_sha256([receipt(), *payload["entries"]])
    write_json(path, payload)
    result = cli(tmp_path, "validate", "--base", base)
    print("TYPED BASE", repr(original), repr(replacement), result.returncode, result.stderr)
    assert result.returncode == 2 and "append-only" in result.stderr


@pytest.mark.parametrize("strict", [False, True])
def test_json_typed_legacy_authority_mismatch_refuses(tmp_path, strict):
    from scripts.i18n import ledger

    baseline = receipt()
    baseline["extension"] = {"value": True}
    legacy(tmp_path, [baseline])
    changed = {**baseline, "extension": {"value": 1}}
    write_json(tmp_path / "i18n/ledger.json", {"schema": 1, "entries": [changed]})
    with pytest.raises(ValueError, match="stranded"):
        if strict:
            ledger.migrate(tmp_path, apply=True)
        else:
            ledger.load_entries(tmp_path)
    assert not (tmp_path / "i18n/audit-history.json").exists()


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "directory"])
def test_mutation_lock_refuses_unsafe_targets_before_storage_writes(tmp_path, alias):
    from scripts.i18n import ledger

    legacy(tmp_path)
    target = tmp_path / "do-not-touch"
    target.write_bytes(b"untouched")
    lock = tmp_path / ".audit-history-write.lock"
    if alias == "symlink":
        lock.symlink_to(target)
    elif alias == "hardlink":
        os.link(target, lock)
    else:
        lock.mkdir()
    with pytest.raises((ValueError, OSError), match="lock"):
        ledger.migrate(tmp_path, apply=True)
    assert target.read_bytes() == b"untouched"
    assert not (tmp_path / "i18n/audit-history.json").exists()


def test_mutation_lock_inode_persists_outside_tracked_history(tmp_path, monkeypatch):
    from scripts.i18n import ledger

    git_fixture(tmp_path)
    lock = tmp_path / ".git/audit-history-write.lock"
    assert lock.is_file()
    identity = (lock.stat().st_dev, lock.stat().st_ino)
    # Ambient Git discovery must not send another caller of the SAME root to a
    # different coordination inode (or fall back because Git is off its PATH).
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated-git-dir"))
    ledger.append(
        tmp_path,
        receipt(note="append"),
        expected_history_sha=ledger.history_sha256([receipt()]),
        apply=True,
    )
    source = tmp_path / "detached.json"
    write_json(
        source,
        {"schema": 1, "entries": [receipt(), receipt(note="append"), receipt(note="import")]},
    )
    ledger.import_legacy(tmp_path, source, "aggregate", apply=True)
    ledger.migrate(tmp_path, apply=True)
    assert (lock.stat().st_dev, lock.stat().st_ino) == identity
    assert not git(tmp_path, "ls-files", "--", "*lock*").stdout.strip()
    assert not (tmp_path / ".audit-history-write.lock").exists()


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_extension_json_refuses_before_receipt_writes(tmp_path, literal):
    from scripts.i18n import ledger

    legacy(tmp_path)
    ledger.migrate(tmp_path, apply=True)
    row = receipt(note="nonfinite")
    row["extension"] = "VALUE"
    source = tmp_path / "detached.json"
    source.write_text(
        json.dumps({"schema": 1, "entries": [receipt(), row]}).replace('"VALUE"', literal)
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = cli(
        tmp_path, "import-legacy", "--source", str(source), "--format", "aggregate", "--apply"
    )
    assert result.returncode == 2
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    assert not (tmp_path / "i18n/audit-history").exists()


@pytest.mark.parametrize("event", ["push", "release", "workflow_dispatch"])
def test_release_shallow_checkout_needs_the_actual_parent(tmp_path, monkeypatch, event):
    """Release pushes have no PR-base payload or origin/main ref. Fetching the
    event's parent must enable append-only validation, not just structure."""
    from scripts.i18n import check, ledger

    parent = git_fixture(tmp_path)
    git(tmp_path, "checkout", "-qb", "release-next-fixture")
    git(tmp_path, "branch", "-D", "main")
    ledger.append(
        tmp_path,
        receipt(note="genuine append"),
        expected_history_sha=ledger.history_sha256([receipt()]),
        apply=True,
    )
    git(tmp_path, "add", "i18n")
    git(tmp_path, "commit", "-qm", "append before release")
    head = git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    shallow = tmp_path / ".git/shallow"
    shallow.write_text(head + "\n")
    monkeypatch.setattr(check, "REPO", tmp_path)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    problems = check.check_audit_history()
    assert any("readable audit base required" in p for p in problems), problems
    assert git(tmp_path, "rev-parse", "--verify", "HEAD^", ok=False).returncode != 0

    # The objects already exist locally, so moving the shallow boundary to the
    # actual parent models checkout's depth=2 fetch without a network or clone.
    shallow.write_text(parent + "\n")
    assert git(tmp_path, "rev-parse", "HEAD^").stdout.strip() == parent
    assert check.check_audit_history("HEAD^") == []

    # A readable base is meaningful: recomputing a valid head-only digest must
    # NOT make an immutable baseline rewrite pass the recovered gate.
    snapshot = tmp_path / "i18n/ledger.json"
    snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    marker_path = tmp_path / "i18n/audit-history.json"
    marker = json.loads(marker_path.read_text())
    marker["baseline"]["i18n/ledger.json"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    write_json(marker_path, marker)
    assert len(ledger.load_entries(tmp_path)) == 2  # head-only structure is valid
    problems = check.check_audit_history("HEAD^")
    assert problems and any("baseline" in p or "snapshot" in p for p in problems), problems


def provenance_graph(root):
    """Queued event base predates main's lossless migration; merge has new P1."""
    from scripts.i18n import ledger

    stale = git_fixture(root, migrate=False)
    git(root, "checkout", "-qb", "feature", stale)
    git(root, "commit", "--allow-empty", "-qm", "feature")
    feature = git(root, "rev-parse", "HEAD").stdout.strip()
    git(root, "checkout", "-q", "main")
    legacy(root, [receipt(), receipt(note="new main audit")])
    commit(root, "main audit advances after queued event")
    ledger.migrate(root, apply=True)
    actual = commit(root, "main lossless migration")
    git(root, "merge", "--no-ff", "-qm", "synthetic pull merge", "feature")
    merged = git(root, "rev-parse", "HEAD").stdout.strip()
    event = root / ".gha-event.json"
    write_json(
        event,
        {
            "number": 4242,
            "pull_request": {
                "base": {"sha": stale, "ref": "main"},
                "head": {"sha": feature},
                # This field has the SAME stale race and is deliberately not authority.
                "merge_commit_sha": "0" * 40,
            },
        },
    )
    environment = {
        **isolated_env(root),
        "PYTHONPATH": str(SCRIPT.parents[2]),
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_REF": "refs/pull/4242/merge",
        "GITHUB_SHA": merged,
    }
    return stale, actual, feature, merged, event, environment


def provenance_gate(root, environment, override="", *, allowance=False):
    call = "check.check_allowance_growth" if allowance else "check.check_audit_history"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from pathlib import Path; from scripts.i18n import check; "
            "check.REPO=Path(sys.argv[1]); print(json.dumps(" + call + "(sys.argv[2] or None)))",
            str(root),
            override,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("shallow", [False, True])
@pytest.mark.parametrize("allowance", [False, True])
@pytest.mark.parametrize("merge_field", ["absent", "null", "stale"])
def test_gha_merged_checkout_uses_actual_parent(tmp_path, shallow, allowance, merge_field):
    stale, actual, feature, merged, event, environment = provenance_graph(tmp_path)
    payload = json.loads(event.read_text())
    if merge_field == "absent":
        payload["pull_request"].pop("merge_commit_sha")
    elif merge_field == "null":
        payload["pull_request"]["merge_commit_sha"] = None
    write_json(event, payload)
    if shallow:
        (tmp_path / ".git/shallow").write_text(merged + "\n")
        assert len(git(tmp_path, "rev-list", "--parents", "-1", "HEAD").stdout.split()) == 1
    before = {p.relative_to(tmp_path): p.read_bytes() for p in (tmp_path / "i18n").rglob("*.json")}
    assert provenance_gate(tmp_path, environment, allowance=allowance) == []
    assert provenance_gate(tmp_path, environment, actual, allowance=allowance) == []
    assert provenance_gate(tmp_path, environment, stale, allowance=allowance)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in (tmp_path / "i18n").rglob("*.json")}
    assert before == after  # No history rewrite/repair is involved in selection.


@pytest.mark.parametrize(
    "damage", ["head", "ref", "second_parent", "nonmerge", "missing_first", "event"]
)
@pytest.mark.parametrize("allowance", [False, True])
def test_gha_provenance_contradictions_fail_closed(tmp_path, damage, allowance):
    stale, actual, feature, merged, event, environment = provenance_graph(tmp_path)
    if damage == "head":
        environment["GITHUB_SHA"] = actual
    elif damage == "ref":
        environment["GITHUB_REF"] = "refs/pull/4242/head"
    elif damage == "second_parent":
        data = json.loads(event.read_text())
        data["pull_request"]["head"]["sha"] = stale
        write_json(event, data)
    elif damage == "nonmerge":
        git(tmp_path, "checkout", "--detach", "-q", actual)
        environment["GITHUB_SHA"] = actual
    elif damage == "missing_first":
        tree = git(tmp_path, "rev-parse", "HEAD^{tree}").stdout.strip()
        body = (
            f"tree {tree}\nparent {'1' * 40}\nparent {feature}\n"
            "author Fixture <fixture@example.test> 1 +0000\n"
            "committer Fixture <fixture@example.test> 1 +0000\n\nmissing parent fixture\n"
        )
        result = subprocess.run(
            ["git", "hash-object", "-t", "commit", "-w", "--stdin"],
            input=body,
            cwd=tmp_path,
            env=isolated_env(tmp_path),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        fake = result.stdout.strip()
        git(tmp_path, "update-ref", "HEAD", fake)
        environment["GITHUB_SHA"] = fake
    else:
        environment.pop("GITHUB_EVENT_PATH")
    problems = provenance_gate(tmp_path, environment, allowance=allowance)
    assert problems, (damage, allowance)
    assert any("provenance" in problem for problem in problems), problems


def test_gha_actual_base_preserves_genuine_history_denial(tmp_path):
    _, actual, _, _, _, environment = provenance_graph(tmp_path)
    path = tmp_path / "i18n/ledger/fr.json"
    path.write_bytes(path.read_bytes() + b"\n")
    problems = provenance_gate(tmp_path, environment)
    assert problems and any("baseline" in problem for problem in problems)


@pytest.mark.parametrize("event_name", ["push", "workflow_dispatch", ""])
def test_non_pr_explicit_base_unchanged(tmp_path, event_name):
    _, actual, _, _, _, environment = provenance_graph(tmp_path)
    environment["GITHUB_EVENT_NAME"] = event_name
    assert provenance_gate(tmp_path, environment, actual) == []
