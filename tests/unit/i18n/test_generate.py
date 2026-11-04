"""The generator: per-key function shape, shipped-set computation, drift."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.i18n import catalogues
from local_operator.i18n import format as fmt

REPO = Path(__file__).resolve().parents[3]


def test_long_keyword_signature_wraps_without_losing_required_parameters(i18n_generate) -> None:
    """Several typed placeholders must not make generated source fail its own lint."""
    import ast

    module = i18n_generate._render_keys_module(
        "demo",
        {
            "demo.session.tray.agents_failed": (
                "Agents {running, number}/{total, number}: {failed, number} failed"
            )
        },
    )
    assert all(len(line) <= 100 for line in module.splitlines())
    tree = ast.parse(module)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assert {arg.arg for arg in function.args.kwonlyargs} == {"failed", "running", "total"}
    assert function.args.kw_defaults == [None, None, None]
    assert not function.args.args and function.args.kwarg is None
    compile(module, "demo.py", "exec")


def test_long_placeholder_free_name_stays_valid_source(i18n_generate) -> None:
    """The OTHER long-header shape: no placeholders at all (agent review C1-2).

    Wrapping an empty parameter list emitted `def name(\n    ,\n) -> Msg:`, a
    SyntaxError for a key the generator has always accepted. A function name can
    be split by no formatter, so this shape must stay parseable — the long-line
    lint concern cannot justify emitting invalid Python.
    """
    import ast

    key = "demo." + "one_" * 24 + "label"
    module = i18n_generate._render_keys_module("demo", {key: "A fixed label"})
    tree = ast.parse(module)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    # The generated name is `one_` x 24 plus `label`: split only for the
    # 100-column gate. Implicit concatenation keeps the asserted text
    # byte-identical (one Constant after parsing).
    assert function.name == (
        "one_one_one_one_one_one_one_one_"
        "one_one_one_one_one_one_one_one_"
        "one_one_one_one_one_one_one_one_label"
    )
    assert not function.args.args and not function.args.kwonlyargs
    assert function.args.kw_defaults == []
    compile(module, "demo.py", "exec")


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
    # No shard dir in fixture trees: the aggregate fallback is under test, and
    # the real tree's shards must not shadow the fixture ledger.
    monkeypatch.setattr(i18n_generate, "LEDGER_SHARDS", tmp_path / "ledger")
    # `_check` relativises planned paths against REPO for its messages (and
    # only for that); point it at the temp root so the fixture paths resolve.
    monkeypatch.setattr(i18n_generate, "REPO", tmp_path)
    return root, emitted


def test_account_policy_missing_generated_key_fails_counterfactual(
    generated, i18n_generate
) -> None:
    """The new English-only child is still subject to canonical drift checks."""
    root, emitted = generated
    (root / "en" / "wire.settings.account_policy.json").write_text(
        json.dumps(
            {
                "wire.settings.account_policy.label": "account selection policy",
                "wire.settings.account_policy.help": "Synthetic policy help",
            }
        )
    )
    planned = i18n_generate._planned_files(emitted)
    for path, text in planned.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert i18n_generate._check(planned) == 0
    module = i18n_generate.KEYS_DIR / "wire_settings_account_policy.py"
    assert "def label()" in module.read_text()
    module.write_text(module.read_text().replace("def label()", "def missing_label()"))
    assert i18n_generate._check(planned) == 1


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


def test_bundled_spa_module_emits_only_shipped_localized_owned_copy(
    generated, i18n_generate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eager map is English; each complete locale is its own lazy chunk.

    The synthetic Urdu text is a generation fixture, not a translation or an
    audit. An unshipped locale beside it must not leak into the client package,
    and the eager map must not carry the other locales' bytes: that is the whole
    point of the chunk split, where a chunk is fetched only for the locale in use.
    """
    root, emitted = generated
    monkeypatch.setattr(i18n_generate, "_shipped_locales", lambda: ("en", "ur"))
    for locale, text in (("en", "English"), ("ur", "اردو"), ("fr", "Français")):
        target = root / locale
        target.mkdir(exist_ok=True)
        (target / "relay.shell.json").write_text(json.dumps({"relay.shell.list.empty": text}))
    planned = i18n_generate._planned_files(emitted)
    bundle_path = i18n_generate._bundled_dir() / "bundled.gen.ts"
    chunk_path = i18n_generate._bundled_dir() / "bundled.ur.gen.ts"
    bundle, chunk = planned[bundle_path], planned[chunk_path]
    # Eager: English only, and the manifest names the chunk.
    assert json.dumps("English") in bundle
    assert json.dumps("اردو") not in bundle
    assert '"ur": () => import("./bundled.ur.gen")' in bundle
    # Lazy: the shipped locale's own file carries it; an unshipped one has none.
    assert json.dumps("اردو") in chunk
    assert json.dumps("Français") not in chunk and json.dumps("Français") not in bundle
    assert i18n_generate._bundled_dir() / "bundled.fr.gen.ts" not in planned


def test_committed_bundle_carries_exactly_the_owned_shell_namespace() -> None:
    """Drift/ownership pin against the COMMITTED tree (not a fixture).

    The bundle that ships must be the explicit owned namespaces, whole, and
    nothing else — the client-side half of "no implicit wire prose bundled".
    C2 adds ``relay.session`` beside ``relay.shell``; both are the SPA's own
    copy and both must be present and complete.
    """
    bundle = (REPO / "local_operator/mobile/web/src/i18n/bundled.gen.ts").read_text(
        encoding="utf-8"
    )
    for namespace in ("relay.shell", "relay.session"):
        en = json.loads((REPO / f"local_operator/i18n/catalogues/en/{namespace}.json").read_text())
        assert f'"{namespace}": {{' in bundle
        for key, message in en.items():
            assert key.startswith(f"{namespace}.")
            assert json.dumps(message) in bundle, key
    assert "wire." not in bundle and "relay.login" not in bundle


def test_committed_relay_session_namespace_is_whole_and_generated() -> None:
    """The relay.session key module, TS key map and en catalogue agree exactly.

    Drift/ownership pin against the COMMITTED tree: every authored key must be
    reachable as a generated Python ``Msg`` function and a typed TS key, and no
    generated artifact may carry a key the catalogue does not (a hand edit that
    desynced one of the three would otherwise ship a dead key or a missing one).
    """
    namespace = "relay.session"
    en = json.loads((REPO / f"local_operator/i18n/catalogues/en/{namespace}.json").read_text())
    module = (REPO / "local_operator/i18n/keys/relay_session.py").read_text(encoding="utf-8")
    ts = (REPO / "local_operator/mobile/web/src/i18n/keys.gen.ts").read_text(encoding="utf-8")
    for key in en:
        assert key.startswith(f"{namespace}.")
        assert f'Msg("{key}",' in module, key
        assert f'"{key}":' in ts, key
    # The Python module owns ONLY this namespace's keys.
    import re as _re

    for match in _re.finditer(r'Msg\("([^"]+)"', module):
        assert match.group(1).startswith(f"{namespace}."), match.group(1)
    # relay.session is registered in the generated package __init__.
    init = (REPO / "local_operator/i18n/keys/__init__.py").read_text(encoding="utf-8")
    assert "relay_session" in init


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


@pytest.mark.parametrize("slash_status", [None, "failed", "passed"])
@pytest.mark.parametrize("candidate", ["zh-CN", "zh-cn", "zh-Hans", "zh_CN"])
def test_security_hold_cannot_be_bypassed_by_partial_or_forged_passes(
    generated, i18n_generate, monkeypatch, slash_status, candidate
) -> None:
    """Synthetic bytes only: omission is the bypass the carried gate cannot see.

    The real disposition is EMPTY for zh-CN as of 2026-10-24, so the
    canonical tag is held here in the loaded module's isolated state: these
    synthetic rows (including the fully passed ones) must still be excluded,
    which is what proves the mechanism is live rather than zh-CN's old status
    baked into a fixture. The alias candidates are not SUPPORTED tags, so they
    can never ship however complete their directory looks.
    """
    root, _ = generated
    entries = []
    for locale in ("en", "fr", candidate):
        (root / locale).mkdir(exist_ok=True)
        (root / locale / "wire.exec.json").write_text(
            json.dumps({"wire.exec.demo": "Synthetic receipt"})
        )
    (root / "en" / "wire.slash.json").write_text(
        json.dumps({"wire.slash.demo": "Synthetic action"})
    )
    for locale in ("fr", candidate):
        entries.append(
            {
                "locale": locale,
                "namespace": "wire.exec",
                "status": "passed",
                "source_sha256": catalogues.catalogue_sha256("en", "wire.exec"),
            }
        )
    if slash_status is not None:
        (root / candidate / "wire.slash.json").write_text(
            json.dumps({"wire.slash.demo": "Synthetic action"})
        )
        entries.append(
            {
                "locale": candidate,
                "namespace": "wire.slash",
                "status": slash_status,
                "source_sha256": catalogues.catalogue_sha256("en", "wire.slash"),
            }
        )
    i18n_generate.LEDGER.write_text(json.dumps({"entries": entries}))
    # The committed disposition is empty; inject the canonical hold under test.
    assert i18n_generate.SECURITY_HELD_LOCALES == frozenset()
    monkeypatch.setattr(i18n_generate, "SECURITY_HELD_LOCALES", frozenset({"zh-CN"}))
    # fr proves that ordinary partial namespace eligibility remains supported.
    # Even fabricated passes do not constitute explicit security disposition.
    assert i18n_generate._shipped_locales() == ["en", "fr"]


def _bundle_locale_maps(bundle: str) -> dict[str, set[str]]:
    """`namespace -> locales` from a generated namespace/locale map module.

    Parsed by indentation rather than imported: the artifact is TypeScript and
    this test judges the COMMITTED bytes, not a re-import of them. Works on the
    eager `bundled.gen.ts` (English only) and on every `bundled.<locale>.gen.ts`
    chunk, which share one namespace -> locale -> messages shape.
    """
    namespaces = re.findall(r'^  "([^"]+)": \{$', bundle, re.MULTILINE)
    blocks = re.split(r'^  "[^"]+": \{$', bundle, flags=re.MULTILINE)[1:]
    found: dict[str, set[str]] = {}
    for namespace, block in zip(namespaces, blocks):
        found[namespace] = set(re.findall(r'^    "([^"]+)": \{$', block, re.MULTILINE))
    return found


def _chunk_locales(bundle: str) -> set[str]:
    """Locales `BUNDLED_CHUNKS` lists, i.e. the ones the SPA can lazily load."""
    return set(
        re.findall(
            r'^  "([^"]+)": \(\) => import\("\./bundled\.[^"]+\.gen"\),$', bundle, re.MULTILINE
        )
    )


def test_committed_bundle_emits_every_shipped_locale_that_is_complete(i18n_generate) -> None:
    """A locale complete in EVERY owned namespace must be a lazy chunk.

    Completeness is the only door to a non-English SPA: `generate.py` emits a
    locale only when it cleared every owned namespace (the `covering`
    intersection), and `uiLocaleFor` re-checks the same precondition at render
    time. So the failure that no green suite would otherwise see is a COMPLETE
    locale that the emitter skipped — every catalogue audited, every gate
    satisfied, and the surface silently English anyway.

    Since the per-locale chunk split the eager `bundled.gen.ts` carries English
    only; every other locale lives in its own `bundled.<locale>.gen.ts` reached
    through `BUNDLED_CHUNKS`. The invariant is therefore read from three
    outputs that must agree: the `BUNDLED_CHUNKS` registry, the chunk files on
    disk, and the content of each chunk.

    The expectation is derived from the committed catalogues and shipped
    manifest, never from today's artifact: a new language enters the bundle by
    landing its translations, with no edit here.
    """
    bundled_dir = REPO / "local_operator/mobile/web/src/i18n"
    bundle = (bundled_dir / "bundled.gen.ts").read_text(encoding="utf-8")
    shipped = json.loads((REPO / "local_operator/i18n/data/shipped.json").read_text())["locales"]
    # The owned list and the shipped manifest are the two inputs the emitter
    # reads; re-declaring either here would make the test agree with its own
    # copy of the rule instead of with the rule.
    owned = i18n_generate.OWNED_SPA_NAMESPACES
    catalogues_dir = REPO / "local_operator/i18n/catalogues"
    complete: set[str] = set(shipped)
    for namespace in owned:
        english = json.loads((catalogues_dir / "en" / f"{namespace}.json").read_text())
        for locale in shipped:
            path = catalogues_dir / locale / f"{namespace}.json"
            carried = set(json.loads(path.read_text())) if path.is_file() else set()
            if locale == "en" or set(english) <= carried:
                continue
            complete.discard(locale)
    assert "en" in complete
    expected_chunks = complete - {"en"}

    # The eager module is English-only: a second locale there is the pre-split
    # shape, which ships every language's bytes to every visitor.
    eager = _bundle_locale_maps(bundle)
    # Equality, not subset: a namespace dropped from the eager module is a relay
    # surface that paints raw keys, and nothing else would notice.
    assert set(eager) == set(owned)
    for namespace in eager:
        assert eager[namespace] == {"en"}, namespace

    assert _chunk_locales(bundle) == expected_chunks
    # A chunk file with no registry entry is unreachable weight; a registry
    # entry with no file is a build break. Both are compared to the same set.
    on_disk = {
        path.name.removeprefix("bundled.").removesuffix(".gen.ts")
        for path in bundled_dir.glob("bundled.*.gen.ts")
    }
    assert on_disk == expected_chunks
    for locale in expected_chunks:
        chunk = _bundle_locale_maps(
            (bundled_dir / f"bundled.{locale}.gen.ts").read_text(encoding="utf-8")
        )
        # Every owned namespace, and only this locale in each: the whole-surface
        # rule means the chunk is all of a language or nothing of it.
        assert chunk == {ns: {locale} for ns in owned}, locale


def test_bundled_spa_module_drops_a_locale_complete_for_only_some_owned_namespaces(
    generated, i18n_generate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One missing owned namespace drops the locale ENTIRELY, not partially.

    The runtime refuses a per-key mix of two languages under one page language,
    so an incomplete locale's bytes are weight with no rendering behind them.
    Pinned at the emitter because the alternative — bundling what exists and
    trusting the runtime to refuse it — ships a client that grows with every
    partial translation wave and one that flips language mid-namespace the day
    someone "fixes" the gate.

    Judged on the planned outputs: no `BUNDLED_CHUNKS` entry and no
    `bundled.fr.gen.ts` chunk while fr misses a namespace, both once it is whole.
    """
    root, emitted = generated
    monkeypatch.setattr(i18n_generate, "_shipped_locales", lambda: ("en", "fr"))
    for namespace in ("relay.shell", "relay.session"):
        (root / "en" / f"{namespace}.json").write_text(json.dumps({f"{namespace}.label": "Label"}))
    (root / "fr").mkdir(exist_ok=True)
    (root / "fr" / "relay.shell.json").write_text(json.dumps({"relay.shell.label": "Étiquette"}))
    bundled = i18n_generate._bundled_dir()
    partial_plan = i18n_generate._planned_files(emitted)
    assert _chunk_locales(partial_plan[bundled / "bundled.gen.ts"]) == set()
    assert bundled / "bundled.fr.gen.ts" not in partial_plan
    # Dropped entirely: not even the namespace it does carry leaks into the
    # eager module (English only) or into any chunk.
    assert '"fr"' not in partial_plan[bundled / "bundled.gen.ts"]
    (root / "fr" / "relay.session.json").write_text(
        json.dumps({"relay.session.label": "Étiquette"})
    )
    whole_plan = i18n_generate._planned_files(emitted)
    assert _chunk_locales(whole_plan[bundled / "bundled.gen.ts"]) == {"fr"}
    chunk = _bundle_locale_maps(whole_plan[bundled / "bundled.fr.gen.ts"])
    assert chunk == {"relay.session": {"fr"}, "relay.shell": {"fr"}}
