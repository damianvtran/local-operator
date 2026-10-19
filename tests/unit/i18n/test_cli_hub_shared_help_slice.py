"""Shared hub helper help: independent BASE bytes, per-call locale, and ratchet.

BASE 51a85d820e017af61932d41e3eaaa7b44c5ca0ee:
both one-argument helpers are shared by teams and agents.
The helper resolves read-only pre-startup locale on each call, not at import.
"""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_hub_shared_help as _hs

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 467
PINS = [
    ("_add_hub_publish_flags", "publish_yes"),
    ("_add_hub_publish_flags", "publish_preview_only"),
    ("_add_hub_publish_flags", "publish_json"),
    ("_add_hub_publish_flags", "publish_accept_unresolved"),
    ("_add_hub_publish_flags", "publish_allow_internal_ops"),
    ("_add_hub_sync_flags", "sync_check"),
    ("_add_hub_sync_flags", "sync_prefer"),
    ("_add_hub_sync_flags", "sync_accept_unknown_baseline"),
    ("_add_hub_sync_flags", "sync_replace"),
    ("_add_hub_sync_flags", "sync_yes"),
    ("_add_hub_sync_flags", "sync_dry_run"),
    ("_add_hub_sync_flags", "sync_json"),
]
HELPERS = {"_add_hub_publish_flags", "_add_hub_sync_flags"}


def _fixture():
    return json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_hub_shared_help_base_ast.json").read_text()
    )


def test_base_bytes_catalogue_golden_context_and_factory_parity():
    expected = _fixture()["messages"]
    live = catalogues.load_catalogue("en", "cli.hub_shared_help")
    golden = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_hub_shared_help_en_golden.json").read_text()
    )
    assert live == golden == expected
    context = catalogues.context("en", "cli.hub_shared_help")
    keys = {key for _, key in PINS}
    assert len(PINS) == len(keys) == 12
    assert set(context) == set(expected) == {"cli.hub_shared_help." + key for key in keys}
    factories = {
        key
        for key, value in vars(_hs).items()
        if callable(value) and getattr(value, "__module__", None) == _hs.__name__
    }
    assert factories == keys
    for key in keys:
        assert not inspect.signature(getattr(_hs, key)).parameters
        assert inspect.signature(getattr(_hs, key)).return_annotation == "Msg"
        message = getattr(_hs, key)()
        assert message.params == {}
        assert message.envelope(locale="en")["text"] == expected["cli.hub_shared_help." + key]
        assert all(
            context["cli.hub_shared_help." + key].get(field) for field in ("role", "tone", "note")
        )


def _shape(key):
    return f'_hs.{key}().envelope(locale=help_locale)["text"]'


def _helper_nodes():
    return [
        node
        for node in ast.parse((REPO / CLI).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    ]


def test_every_owned_copy_site_has_the_exact_pinned_shape():
    actual = []
    for helper in _helper_nodes():
        calls = []
        for call in ast.walk(helper):
            if isinstance(call, ast.Call):
                calls.extend(
                    (kw.value.lineno, kw.value)
                    for kw in call.keywords
                    if kw.arg in {"help", "description", "epilog"}
                )
        actual.extend((helper.name, ast.dump(value)) for _, value in sorted(calls))
        snippet = ast.get_source_segment((REPO / CLI).read_text(), helper)
        assert snippet is not None
        assert "# i18n: ignore" not in snippet
    assert actual == [
        (owner, ast.dump(ast.parse(_shape(key), mode="eval").body)) for owner, key in PINS
    ]


def _live_cli_count(i18n_check) -> int:
    entries = i18n_check.load_allowlist(i18n_check.ALLOWLIST)
    scanned = i18n_check.scan_tree(i18n_check.REPO)
    return i18n_check._counts(scanned, entries).get(CLI, 0)


def test_the_ceiling_has_no_slack_and_stays_unactivated(i18n_check) -> None:
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    live = _live_cli_count(i18n_check)
    assert baseline["files"].get(CLI) == live, (baseline["files"].get(CLI), live)
    assert live <= CEILING, live
    assert not i18n_check.in_enforced_scope(
        CLI, baseline["enforced"]
    ), "cli.py is activated without the file being clean"


def test_a_planted_literal_fails_the_checker(tmp_path, monkeypatch, i18n_check, capsys) -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    target = tmp_path / CLI
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    # Later tranches lower the live count; this fixture enforces that count,
    # not this tranche's historical ceiling, so the planted +1 stays meaningful.
    count = _live_cli_count(i18n_check)
    baseline.write_text(
        json.dumps(
            {
                "note": "slice fixture",
                "enforced": [CLI],
                "files": {CLI: count},
            }
        )
    )
    allowlist = baseline_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", baseline)
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
    assert i18n_check.main([]) == 0
    target.write_text(source + '\nprint("planted shared hub helper prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    output = capsys.readouterr().out
    assert f"{CLI}: {count + 1} literal(s) against a ceiling of {count}" in output
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


def test_helpers_resolve_locale_each_call_and_direct_parsers_preserve_protocol(monkeypatch):
    import local_operator.cli as cli
    from local_operator.i18n import cli_help
    from local_operator.i18n.cli_help import help_locale
    from local_operator.i18n.messages import Msg

    seen = []
    resolutions = []

    def record_resolution():
        locale = help_locale()
        resolutions.append(locale)
        return locale

    original = Msg.envelope

    def record(self, *, locale=None):
        if self.code.startswith("cli.hub_shared_help."):
            seen.append((self.code, locale))
        return original(self, locale=locale)

    monkeypatch.setattr(Msg, "envelope", record)
    monkeypatch.setattr(cli_help, "help_locale", record_resolution)
    for locale in ("en", "fr", "en"):
        monkeypatch.setenv("LOP_LANG", locale)
        assert help_locale() == locale
        seen.clear()
        resolutions.clear()
        for name in ("_add_hub_publish_flags", "_add_hub_sync_flags"):
            parser = argparse.ArgumentParser(prog=name)
            getattr(cli, name)(parser)
            assert all(
                getattr(_hs, key)().envelope(locale=locale)["text"]
                in [action.help for action in parser._actions]
                for owner, key in PINS
                if owner == name
            )
        # The explicit renders above also exercise fallback, so pin locale values
        # independently and check exactly one helper render per key below.
        assert {value for _, value in seen} == {locale}
        assert len(seen) == 24
        assert resolutions == [locale, locale]
    assert (REPO / "local_operator/i18n/catalogues/fr/cli.hub_shared_help.json").is_file()
    assert set(catalogues.load_catalogue("fr", "cli.hub_shared_help")) == set(
        catalogues.load_catalogue("en", "cli.hub_shared_help")
    )
    monkeypatch.setenv("LOP_LANG", "fr")
    seen.clear()
    resolutions.clear()
    cli.build_cli_parser()
    assert len(seen) == 24
    assert {locale for _, locale in seen} == {"fr"}
    assert resolutions == ["fr"]


def test_one_build_resolves_once_and_reuses_its_snapshot_for_all_shared_copy(monkeypatch):
    from local_operator import cli
    from local_operator.i18n import cli_help
    from local_operator.i18n.messages import Msg

    resolutions = []
    renders = []
    original = Msg.envelope

    def resolve_once():
        resolutions.append("fr")
        assert len(resolutions) == 1, "one parser build must use one locale snapshot"
        return "fr"

    def record(self, *, locale=None):
        if self.code.startswith("cli.hub_shared_help."):
            renders.append(locale)
        return original(self, locale=locale)

    monkeypatch.setattr(cli_help, "help_locale", resolve_once)
    monkeypatch.setattr(Msg, "envelope", record)
    cli.build_cli_parser()
    assert resolutions == ["fr"]
    assert renders == ["fr"] * 24


def test_explicit_helper_locale_is_honored_without_reading_environment(monkeypatch):
    from local_operator import cli
    from local_operator.i18n import cli_help
    from local_operator.i18n.messages import Msg

    renders = []
    original = Msg.envelope

    def forbidden():
        raise AssertionError("an explicit parser snapshot must not resolve again")

    def record(self, *, locale=None):
        if self.code.startswith("cli.hub_shared_help."):
            renders.append(locale)
        return original(self, locale=locale)

    monkeypatch.setenv("LOP_LANG", "en")
    monkeypatch.setattr(cli_help, "help_locale", forbidden)
    monkeypatch.setattr(Msg, "envelope", record)
    for name in ("_add_hub_publish_flags", "_add_hub_sync_flags"):
        getattr(cli, name)(argparse.ArgumentParser(prog=name), help_locale="fr")
    assert renders == ["fr"] * 12


def _restore_helpers():
    fixture = _fixture()
    replacements = {
        ast.dump(ast.parse(_shape(key), mode="eval").body): ast.Constant(
            fixture["messages"]["cli.hub_shared_help." + key]
        )
        for _, key in PINS
    }

    class Restore(ast.NodeTransformer):
        count = 0

        def visit_Subscript(self, node):
            shape = ast.dump(node)
            if shape in replacements:
                self.count += 1
                return replacements[shape]
            return self.generic_visit(node)

    tree = ast.parse((REPO / CLI).read_text())
    restore = Restore()
    tree = restore.visit(tree)
    assert restore.count == 12
    # Reverse only the deliberate optional-locale plumbing. Keep the original
    # BASE hashes: rehashing the new signature would hide a protocol regression.
    guard = ast.parse(
        "if help_locale is None:\n"
        "    from local_operator.i18n.cli_help import help_locale as resolve_help_locale\n"
        "    help_locale = resolve_help_locale()"
    ).body[0]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            assert len(node.args.kwonlyargs) == len(node.args.kw_defaults) == 1
            argument = node.args.kwonlyargs[0]
            assert argument.arg == "help_locale"
            assert argument.annotation is not None
            assert ast.dump(argument.annotation) == ast.dump(
                ast.parse("str | None", mode="eval").body
            )
            assert isinstance(node.args.kw_defaults[0], ast.Constant)
            assert node.args.kw_defaults[0].value is None
            node.args.kwonlyargs = []
            node.args.kw_defaults = []
            matches = [
                statement for statement in node.body if ast.dump(statement) == ast.dump(guard)
            ]
            assert len(matches) == 1
            node.body = [statement for statement in node.body if statement not in matches]
    calls = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in HELPERS
        ):
            assert len(node.keywords) == 1
            keyword = node.keywords[0]
            assert keyword.arg == "help_locale"
            assert isinstance(keyword.value, ast.Name) and keyword.value.id == "help_locale"
            node.keywords = []
            calls += 1
    assert calls == 4
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "local_operator.i18n.keys"
            and any(alias.name == "cli_hub_shared_help" for alias in node.names)
        )
    ]
    return tree


def test_restoring_copy_and_locale_bootstrap_preserves_helper_protocol():
    restored = _restore_helpers()
    for node in restored.body:
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            assert (
                hashlib.sha256(_canonical_ast(node).encode()).hexdigest()
                == _fixture()["helpers_ast_sha256"][node.name]
            )


def _canonical_ast(node):
    # ast.dump started omitting empty fields in Python 3.13. Serialize fields
    # explicitly so a BASE hash derived on 3.14 also verifies on CI's 3.12.
    def normalize(value):
        if isinstance(value, ast.AST):
            return {
                "type": type(value).__name__,
                "fields": {key: normalize(item) for key, item in ast.iter_fields(value)},
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if value is Ellipsis or isinstance(value, (bytes, complex)):
            return {"literal": repr(value)}
        return value

    return json.dumps(normalize(node), sort_keys=True, ensure_ascii=False)
