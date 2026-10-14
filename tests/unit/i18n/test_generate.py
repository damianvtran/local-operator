"""The generator: per-key function shape, shipped-set computation, drift."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.i18n import catalogues
from local_operator.i18n import format as fmt

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture()
def generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, i18n_generate):
    """A fixture catalogue root + emitted tables wired into the generator."""
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    emitted = tmp_path / "emitted"
    emitted.mkdir()
    data = Path(fmt.__file__).parent / "data"
    for name in ("plural_rules.json", "formats.json"):
        shutil.copy(data / name, emitted / name)
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(i18n_generate, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(i18n_generate, "KEYS_DIR", tmp_path / "keys")
    monkeypatch.setattr(i18n_generate, "LEDGER", tmp_path / "ledger.json")
    # `_check` relativises planned paths against REPO for its messages (and
    # only for that); point it at the temp root so the fixture paths resolve.
    monkeypatch.setattr(i18n_generate, "REPO", tmp_path)
    return root, emitted


def test_ts_key_map_reuses_icu_argument_types(i18n_generate) -> None:
    output = i18n_generate._render_ts_key_map(
        {
            "wire.demo": {
                "wire.demo.plain": "Plain",
                "wire.demo.string": "Hello {name}",
                "wire.demo.count": "{count, plural, one {# file} other {# files}}",
                "wire.demo.number": "{amount, number}",
                "wire.demo.date": "{when, date, short}",
                "wire.demo.time": "{when, time, short}",
                "wire.demo.select": "{mode, select, one {First} other {Other}}",
            }
        }
    )
    assert '"wire.demo.plain": undefined;' in output
    assert '"name": string' in output
    assert '"count": number' in output
    assert '"amount": number' in output
    assert '"when": Date | number' in output
    assert '"mode": string' in output
    assert "<K extends WithParams>(key: K, params: MessageParams[K]): string;" in output
    assert "(key: WithoutParams): string;" in output
    assert "bundledEnglish" not in output


def test_bundled_spa_module_is_empty_when_no_owned_namespace_is_present(
    generated, i18n_generate
) -> None:
    """The bundle is driven by the EXPLICIT owned list, not by whatever `relay.*`
    files a fixture tree happens to hold.

    ``relay.login`` is SSR-only (the daemon paints it before the SPA exists) and
    ``wire.*`` is served, so neither may land in the client bundle. A fixture
    that carries only those must emit an empty ``BUNDLED_OWNED`` — proving the
    emission is namespace-list driven, not a glob over ``relay.*``.
    """
    root, emitted = generated
    for namespace in ["relay.login", "wire.exec", "relay.demo"]:
        (root / "en" / f"{namespace}.json").write_text(json.dumps({f"{namespace}.label": "Label"}))
    planned = i18n_generate._planned_files(emitted)
    bundle = planned[i18n_generate.REPO / "local_operator/mobile/web/src/i18n/bundled.gen.ts"]
    assert "export const BUNDLED_OWNED" in bundle
    assert "relay.login" not in bundle and "wire.exec" not in bundle and "relay.demo" not in bundle
    # The owned child namespace reaches it only through the explicit list.
    (root / "en" / "relay.shell.json").write_text(
        json.dumps({"relay.shell.list.empty": "no sessions running"})
    )
    with_shell = i18n_generate._planned_files(emitted)
    shell_bundle = with_shell[
        i18n_generate.REPO / "local_operator/mobile/web/src/i18n/bundled.gen.ts"
    ]
    assert '"relay.shell"' in shell_bundle and "no sessions running" in shell_bundle
    assert "relay.login" not in shell_bundle and "wire." not in shell_bundle


def test_committed_bundle_carries_exactly_the_owned_shell_namespace() -> None:
    """Drift/ownership pin against the COMMITTED tree (not a fixture).

    The bundle that ships must be the explicit owned namespace, whole, and
    nothing else — the client-side half of "no implicit wire prose bundled".
    """
    bundle = (REPO / "local_operator/mobile/web/src/i18n/bundled.gen.ts").read_text(
        encoding="utf-8"
    )
    en = json.loads((REPO / "local_operator/i18n/catalogues/en/relay.shell.json").read_text())
    namespace = "relay.shell"
    assert '"relay.shell": {' in bundle
    for key, message in en.items():
        assert key.startswith(f"{namespace}.")
        assert json.dumps(message) in bundle, key
    assert "wire." not in bundle and "relay.login" not in bundle


def test_ts_plan_only_owns_wire_and_relay_types(generated, i18n_generate) -> None:
    root, emitted = generated
    for namespace in ["wire.demo", "relay.demo", "tui.demo"]:
        (root / "en" / f"{namespace}.json").write_text(json.dumps({f"{namespace}.label": "Label"}))
    planned = i18n_generate._planned_files(emitted)
    output = planned[i18n_generate.REPO / "local_operator/mobile/web/src/i18n/keys.gen.ts"]
    assert "wire.demo.label" in output and "relay.demo.label" in output
    assert "tui.demo.label" not in output
    assert planned == i18n_generate._planned_files(emitted)


def test_per_key_functions_are_emitted_with_typed_kwargs(generated, i18n_generate) -> None:
    root, emitted = generated
    (root / "en" / "demo.words.json").write_text(
        json.dumps(
            {
                "demo.words.hello": "Hello, {name}!",
                "demo.words.files": "{count, plural, one {# file} other {# files}}",
                "demo.words.updated": "Updated {when, date, short}",
                "demo.words.plain": "plain text, no params",
                "demo.words.import": "keyword collision",
                # round-1 m2: shapes that emit SYNTAX ERRORS without
                # sanitisation — a digit-leading suffix and a keyword ARGUMENT.
                "demo.words.2fa": "two-factor prompts",
                "demo.words.klass": "{class, select, login {Sign in} other {Continue}}",
                # Multi-argument message: the keyword-only marker `*` is emitted
                # ONCE for the block. Emitting it per argument (its own earlier
                # shape) is a SyntaxError, caught when tui.analytics landed the
                # first two-argument message.
                "demo.words.pair": "Pair {left} and {right}",
            }
        ),
        encoding="utf-8",
    )
    planned = i18n_generate._planned_files(emitted)
    module = planned[Path(i18n_generate.KEYS_DIR) / "demo_words.py"]
    expected_body = """\


def _2fa() -> Msg:
    return Msg("demo.words.2fa", {})


def files(*, count: int | float) -> Msg:
    return Msg("demo.words.files", {"count": count})


def hello(*, name: str) -> Msg:
    return Msg("demo.words.hello", {"name": name})


def import_() -> Msg:
    return Msg("demo.words.import", {})


def klass(*, class_: str) -> Msg:
    return Msg("demo.words.klass", {"class": class_})


def pair(*, left: str, right: str) -> Msg:
    return Msg("demo.words.pair", {"left": left, "right": right})


def plain() -> Msg:
    return Msg("demo.words.plain", {})


def updated(*, when: datetime) -> Msg:
    return Msg("demo.words.updated", {"when": when})
"""
    assert module == i18n_generate.GENERATED_HEADER + (
        "from datetime import datetime\n\nfrom ..messages import Msg\n\n\n"
    ) + expected_body.lstrip("\n")
    # The sanitised shapes COMPILE — the regression m2 recorded was a
    # SyntaxError the text-comparing drift check could not see.
    compile(module, "demo_words.py", "exec")
    # The generated module is what black/isort would leave alone.
    check = subprocess.run(
        [sys.executable, "-m", "black", "--check", "-"],
        input=module,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0 and "No module named" not in check.stderr:
        pytest.fail(f"generated module is not black-clean:\n{check.stdout}\n{check.stderr}")


def test_empty_namespace_module_has_no_dangling_import(generated, i18n_generate) -> None:
    root, emitted = generated
    (root / "en" / "wire.empty.json").write_text("{}", encoding="utf-8")
    planned = i18n_generate._planned_files(emitted)
    module = planned[Path(i18n_generate.KEYS_DIR) / "wire_empty.py"]
    assert module == i18n_generate.GENERATED_HEADER.rstrip() + "\n"
    assert "from ..messages import Msg" not in module
    assert "from datetime" not in module


def test_identifier_collisions_are_refused(generated, i18n_generate) -> None:
    # round-2 m5: two keys that sanitise to one FUNCTION NAME are refused too
    # (the kwargs case was guarded in round 1; neither may silently shadow).
    root, emitted = generated
    (root / "en" / "demo.words.json").write_text(
        json.dumps({"demo.words.class": "a", "demo.words.class_": "b"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="same function name"):
        i18n_generate._planned_files(emitted)
    (root / "en" / "demo.words.json").write_text(
        json.dumps({"demo.words.k": "{class} {class_}"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="duplicate identifiers"):
        i18n_generate._planned_files(emitted)


def test_check_flags_orphaned_generated_files(generated, i18n_generate, capsys) -> None:
    # round-1 n4: a namespace rename/removal leaves keys/<old>.py behind; the
    # plan-vs-plan diff never saw it, so --check now flags files in the
    # generated dirs that are not part of the plan.
    root, emitted = generated
    (root / "en" / "demo.json").write_text('{"demo.x": "x"}', encoding="utf-8")
    planned = i18n_generate._planned_files(emitted)
    for path, content in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    assert i18n_generate._check(planned) == 0
    orphan = Path(i18n_generate.KEYS_DIR) / "orphan.py"
    orphan.write_text("", encoding="utf-8")
    assert i18n_generate._check(planned) == 1
    out = capsys.readouterr().out
    assert "orphaned" in out and "orphan.py" in out
    orphan.unlink()
    assert i18n_generate._check(planned) == 0


def test_shipped_set_requires_fresh_passed_entries(generated, i18n_generate) -> None:
    root, emitted = generated
    demo = {"demo.x": "x"}
    (root / "en" / "demo.json").write_text(json.dumps(demo), encoding="utf-8")
    fresh = catalogues.catalogue_sha256("en", "demo")
    # Each locale CARRYING the namespace (its own file on disk) is what makes
    # the per-namespace gate judge it; see the 2026-10-11 amendment test below.
    for locale in ("fr", "es", "ru"):
        (root / locale).mkdir(parents=True, exist_ok=True)
        (root / locale / "demo.json").write_text(json.dumps(demo), encoding="utf-8")
    (root / "vi").mkdir(parents=True, exist_ok=True)
    (root / "vi" / "other.json").write_text(json.dumps({"other.y": "t"}), encoding="utf-8")
    ledger = {
        "schema": 1,
        "entries": [
            {"locale": "fr", "namespace": "demo", "status": "passed", "source_sha256": fresh},
            {"locale": "es", "namespace": "demo", "status": "passed", "source_sha256": "stale"},
            {"locale": "ru", "namespace": "demo", "status": "failed", "source_sha256": fresh},
            {"locale": "vi", "namespace": "other", "status": "passed", "source_sha256": "x"},
        ],
    }
    Path(i18n_generate.LEDGER).write_text(json.dumps(ledger), encoding="utf-8")
    assert i18n_generate._shipped_locales() == ["en", "fr"]
    # No ledger at all: en alone, not an empty set.
    Path(i18n_generate.LEDGER).unlink()
    assert i18n_generate._shipped_locales() == ["en"]


def test_a_new_en_namespace_does_not_un_ship_carried_locales(generated, i18n_generate) -> None:
    """The E2a tree, exactly (2026-10-11 amendment): `wire.errors.desktop` is
    born after wave 1.

    Before the per-namespace gate this was the failure mode — one new en
    namespace dropped the generated set es/fr → en only (measured on the E2a
    worktree). A namespace the locale does not carry is not judged, so the
    locale keeps shipping the ones it has; clients read the newborn namespace
    from en until its own translation wave.
    """
    root, emitted = generated
    (root / "en" / "demo.json").write_text(json.dumps({"demo.x": "x"}), encoding="utf-8")
    (root / "en" / "newborn.json").write_text(json.dumps({"newborn.y": "y"}), encoding="utf-8")
    (root / "fr").mkdir(parents=True, exist_ok=True)
    (root / "fr" / "demo.json").write_text(json.dumps({"demo.x": "t"}), encoding="utf-8")
    ledger = {
        "schema": 1,
        "entries": [
            {
                "locale": "fr",
                "namespace": "demo",
                "status": "passed",
                "source_sha256": catalogues.catalogue_sha256("en", "demo"),
            }
        ],
    }
    Path(i18n_generate.LEDGER).write_text(json.dumps(ledger), encoding="utf-8")
    assert i18n_generate._shipped_locales() == ["en", "fr"]


def test_a_carried_namespace_without_a_ledger_entry_un_ships(generated, i18n_generate) -> None:
    """The flip side of the amendment: completeness is still the gate's job.

    A file the locale carries with no passing, fresh ledger entry — including
    a MISSING entry — keeps that locale out; the per-namespace scope narrows
    WHAT is judged, not whether judgement happens.
    """
    root, emitted = generated
    (root / "en" / "demo.json").write_text(json.dumps({"demo.x": "x"}), encoding="utf-8")
    (root / "fr").mkdir(parents=True, exist_ok=True)
    (root / "fr" / "demo.json").write_text(json.dumps({"demo.x": "t"}), encoding="utf-8")
    (root / "fr" / "extra.json").write_text(json.dumps({"extra.y": "y"}), encoding="utf-8")
    (root / "en" / "extra.json").write_text(json.dumps({"extra.y": "y"}), encoding="utf-8")
    ledger = {
        "schema": 1,
        "entries": [
            {
                "locale": "fr",
                "namespace": "demo",
                "status": "passed",
                "source_sha256": catalogues.catalogue_sha256("en", "demo"),
            }
        ],
    }
    Path(i18n_generate.LEDGER).write_text(json.dumps(ledger), encoding="utf-8")
    assert i18n_generate._shipped_locales() == ["en"]


def test_a_missing_sha_on_an_en_less_namespace_is_not_fresh(generated, i18n_generate) -> None:
    """``None == None`` must not read as fresh (round-1 minor M1).

    A locale carrying a namespace ``en`` does not ship (a renamed child still
    on disk) with a ``passed`` entry that OMITS ``source_sha256`` used to ship:
    the entry is read with ``.get``, so both sides of the freshness comparison
    were ``None``. The fix requires a REAL ``en`` hash before comparing; this
    pins that exact shape.
    """
    root, emitted = generated
    (root / "en" / "demo.json").write_text(json.dumps({"demo.x": "x"}), encoding="utf-8")
    (root / "fr").mkdir(parents=True, exist_ok=True)
    (root / "fr" / "demo.json").write_text(json.dumps({"demo.x": "t"}), encoding="utf-8")
    # The carried child `en` does not have, with a passed entry naming NO hash.
    (root / "fr" / "ghost.json").write_text(json.dumps({"ghost.y": "t"}), encoding="utf-8")
    ledger = {
        "schema": 1,
        "entries": [
            {
                "locale": "fr",
                "namespace": "demo",
                "status": "passed",
                "source_sha256": catalogues.catalogue_sha256("en", "demo"),
            },
            {"locale": "fr", "namespace": "ghost", "status": "passed"},
        ],
    }
    Path(i18n_generate.LEDGER).write_text(json.dumps(ledger), encoding="utf-8")
    assert i18n_generate._shipped_locales() == ["en"]


def test_a_locale_directory_with_no_namespaces_is_not_shipped(generated, i18n_generate) -> None:
    """Pin for the empty-carried reading (round-1 nit N1).

    A locale DIRECTORY that exists but carries no namespace has nothing to
    serve, so it is skipped rather than shipped as an empty locale; a wave
    mid-checkout cannot put a blank entry in the picker. (Round 1 asked for
    either a pin or a recorded reject; this is the pin.)
    """
    root, emitted = generated
    (root / "en" / "demo.json").write_text(json.dumps({"demo.x": "x"}), encoding="utf-8")
    (root / "fr").mkdir(parents=True, exist_ok=True)  # exists, carries nothing
    ledger = {
        "schema": 1,
        "entries": [
            {
                "locale": "fr",
                "namespace": "demo",
                "status": "passed",
                "source_sha256": catalogues.catalogue_sha256("en", "demo"),
            }
        ],
    }
    Path(i18n_generate.LEDGER).write_text(json.dumps(ledger), encoding="utf-8")
    assert i18n_generate._shipped_locales() == ["en"]


def test_drift_check_passes_on_the_committed_tree() -> None:
    if shutil.which("node") is None:
        pytest.skip("node is required for the emitter; CI's i18n job covers it")
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "i18n" / "generate.py"), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "byte-identical" in result.stdout
