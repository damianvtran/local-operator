"""Global/root help only: exact BASE bytes and one-bootstrap parser contract.

Immutable BASE 786b95b7ae5eaaecb2db5bd34e64e4dc893d00f1; output copy remains outside this tranche.
"""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_global_help as _gh

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 451
PINS = [
    ("parent_parser", "debug"),
    ("parent_parser", "agent"),
    ("parent_parser", "train"),
    ("parser", "description"),
    ("parser", "resume"),
    ("parser", "version"),
    ("parser", "hosting"),
    ("parser", "model"),
    ("parser", "run_in"),
    ("parser", "yolo"),
    ("parser", "no_tui"),
    ("parser", "tui"),
    ("_propagate_global_flags", "yolo_conditional"),
    ("_propagate_global_flags", "yolo"),
    ("_propagate_global_flags", "propagated_resume"),
    ("_propagate_global_flags", "propagated_hosting"),
    ("_propagate_global_flags", "propagated_model"),
    ("_propagate_global_flags", "propagated_run_in"),
]


def _fixture():
    return json.loads((REPO / "tests/unit/i18n/fixtures/cli_global_help_base_ast.json").read_text())


def _shape(key):
    return f'_gh.{key}().envelope(locale=help_locale)["text"]'


def test_base_bytes_catalogue_golden_context_and_factory_parity():
    expected = _fixture()["messages"]
    live = catalogues.load_catalogue("en", "cli.global_help")
    golden = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_global_help_en_golden.json").read_text()
    )
    assert live == golden == expected
    context = catalogues.context("en", "cli.global_help")
    keys = {key for _, key in PINS}
    assert len(PINS) == 18 and len(keys) == 17
    assert set(context) == set(expected) == {"cli.global_help." + key for key in keys}
    factories = {
        key
        for key, value in vars(_gh).items()
        if callable(value) and getattr(value, "__module__", None) == _gh.__name__
    }
    assert factories == keys
    for key in keys:
        assert not inspect.signature(getattr(_gh, key)).parameters
        assert inspect.signature(getattr(_gh, key)).return_annotation == "Msg"
        assert (
            getattr(_gh, key)().envelope(locale="en")["text"] == expected["cli.global_help." + key]
        )
        assert all(
            context["cli.global_help." + key].get(field) for field in ("role", "tone", "note")
        )


def test_all_pinned_copy_sites_render_from_the_existing_snapshot():
    source = (REPO / CLI).read_text()
    module = ast.parse(source)
    build = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_cli_parser"
    )
    propagated = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_propagate_global_flags"
    )
    actual = []
    for node in ast.walk(build):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or not isinstance(node.func.value, ast.Name)
        ):
            continue
        owner = node.func.value.id
        for kw in node.keywords:
            if (
                owner in {"parser", "parent_parser"}
                and kw.arg == "help"
                and node.func.attr == "add_argument"
            ):
                actual.append((kw.value.lineno, kw.value.col_offset, owner, kw.value))
            elif (
                kw.arg == "description"
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "ArgumentParser"
            ):
                actual.append((kw.value.lineno, kw.value.col_offset, "parser", kw.value))
    for node in ast.walk(propagated):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_add_global_flag"
        ):
            for kw in node.keywords:
                if kw.arg == "help":
                    values = (
                        [kw.value.body, kw.value.orelse]
                        if isinstance(kw.value, ast.IfExp)
                        else [kw.value]
                    )
                    actual.extend(
                        (value.lineno, value.col_offset, "_propagate_global_flags", value)
                        for value in values
                    )
    actual.sort(key=lambda row: row[:2])
    assert [(owner, ast.dump(value)) for _, _, owner, value in actual] == [
        (owner, ast.dump(ast.parse(_shape(key), mode="eval").body)) for owner, key in PINS
    ]
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CLI_DESCRIPTION"
            for target in node.targets
        )
        for node in module.body
    )


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
    target.write_text(source + '\nprint("planted global flag prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    output = capsys.readouterr().out
    assert f"{CLI}: {count + 1} literal(s) against a ceiling of {count}" in output
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


def test_one_build_snapshot_covers_global_and_recursive_help(monkeypatch):
    from local_operator import cli
    from local_operator.i18n import cli_help
    from local_operator.i18n.messages import Msg

    resolutions = []
    propagated = []
    renders = []
    original_envelope = Msg.envelope
    original_propagation = cli._propagate_global_flags

    def once():
        resolutions.append("fr")
        assert len(resolutions) == 1
        return "fr"

    def record(self, *, locale=None):
        if self.code.startswith("cli.global_help."):
            renders.append(locale)
        return original_envelope(self, locale=locale)

    def record_propagation(parser, **kwargs):
        propagated.append(kwargs.get("help_locale"))
        return original_propagation(parser, **kwargs)

    monkeypatch.setattr(cli_help, "help_locale", once)
    monkeypatch.setattr(Msg, "envelope", record)
    monkeypatch.setattr(cli, "_propagate_global_flags", record_propagation)
    cli.build_cli_parser()
    assert resolutions == ["fr"]
    assert len(propagated) > 1 and set(propagated) == {"fr"}
    assert len(renders) == 12 + 5 * (len(propagated) - 1)
    assert set(renders) == {"fr"}
    assert not (REPO / "local_operator/i18n/catalogues/fr/cli.global_help.json").exists()


def _standalone_tree():
    root = argparse.ArgumentParser(prog="synthetic")
    sub = root.add_subparsers()
    ordinary = sub.add_parser("ordinary")
    conditional = sub.add_parser("conditional")
    setattr(conditional, "yolo_needs_grant", True)
    return root, ordinary, conditional


def test_standalone_propagation_is_fresh_and_explicit_locale_never_reads(monkeypatch):
    from local_operator import cli
    from local_operator.i18n import cli_help
    from local_operator.i18n.messages import Msg

    resolutions = []
    original_resolve = cli_help.help_locale
    original_envelope = Msg.envelope
    renders = []

    def resolve():
        value = original_resolve()
        resolutions.append(value)
        return value

    def record(self, *, locale=None):
        if self.code.startswith("cli.global_help."):
            renders.append(locale)
        return original_envelope(self, locale=locale)

    monkeypatch.setattr(cli_help, "help_locale", resolve)
    monkeypatch.setattr(Msg, "envelope", record)
    for locale in ("en", "fr", "en"):
        monkeypatch.setenv("LOP_LANG", locale)
        resolutions.clear()
        renders.clear()
        root, ordinary, conditional = _standalone_tree()
        cli._propagate_global_flags(root)
        assert resolutions == [locale]
        assert renders == [locale] * 10
        assert (
            ordinary._option_string_actions["--yolo"].help
            == _fixture()["messages"]["cli.global_help.yolo"]
        )
        assert (
            conditional._option_string_actions["--yolo"].help
            == _fixture()["messages"]["cli.global_help.yolo_conditional"]
        )

    def forbidden():
        raise AssertionError("an explicit snapshot must not read locale/config")

    monkeypatch.setenv("LOP_LANG", "en")
    monkeypatch.setattr(cli_help, "help_locale", forbidden)
    renders.clear()
    cli._propagate_global_flags(_standalone_tree()[0], help_locale="fr")
    assert renders == ["fr"] * 10


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


def _restore_global_copy():
    fixture = _fixture()
    replacements = {}
    for _, key in PINS:
        value = (
            ast.Name(id="CLI_DESCRIPTION", ctx=ast.Load())
            if key == "description"
            else ast.Constant(fixture["messages"]["cli.global_help." + key])
        )
        replacements[ast.dump(ast.parse(_shape(key), mode="eval").body)] = value

    class Restore(ast.NodeTransformer):
        count = 0

        def visit_Subscript(self, node):
            key = ast.dump(node)
            if key in replacements:
                self.count += 1
                return replacements[key]
            return self.generic_visit(node)

    tree = ast.parse((REPO / CLI).read_text())
    restore = Restore()
    tree = restore.visit(tree)
    assert restore.count == 18
    guard = ast.parse(
        "if help_locale is None:\n"
        "    from local_operator.i18n.cli_help import help_locale as resolve_help_locale\n"
        "    help_locale = resolve_help_locale()"
    ).body[0]
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_propagate_global_flags"
    )
    assert len(fn.args.kwonlyargs) == len(fn.args.kw_defaults) == 1
    argument = fn.args.kwonlyargs[0]
    assert argument.arg == "help_locale" and argument.annotation is not None
    assert ast.dump(argument.annotation) == ast.dump(ast.parse("str | None", mode="eval").body)
    default = fn.args.kw_defaults[0]
    assert isinstance(default, ast.Constant) and default.value is None
    fn.args.kwonlyargs = []
    fn.args.kw_defaults = []
    matches = [statement for statement in fn.body if ast.dump(statement) == ast.dump(guard)]
    assert len(matches) == 1
    fn.body = [statement for statement in fn.body if statement not in matches]
    calls = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_propagate_global_flags"
        ):
            assert len(node.keywords) == 1
            keyword = node.keywords[0]
            assert keyword.arg == "help_locale"
            assert isinstance(keyword.value, ast.Name) and keyword.value.id == "help_locale"
            node.keywords = []
            calls += 1
    assert calls == 2
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "local_operator.i18n.keys"
            and any(alias.name == "cli_global_help" for alias in node.names)
        )
    ]
    tree.body.insert(
        fixture["description_index"], ast.parse(fixture["description_assignment"]).body[0]
    )
    return tree


def _root_calls(module):
    fn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_cli_parser"
    )
    result = []
    for node in ast.walk(fn):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or not isinstance(node.func.value, ast.Name)
        ):
            continue
        receiver = node.func.value.id
        if (receiver in {"parser", "parent_parser"} and node.func.attr == "add_argument") or (
            receiver == "argparse" and node.func.attr == "ArgumentParser"
        ):
            result.append(node)
    return sorted(result, key=lambda node: (node.lineno, node.col_offset))


def test_restored_root_and_propagation_protocol_match_original_base():
    restored = _restore_global_copy()
    prop = next(
        node
        for node in restored.body
        if isinstance(node, ast.FunctionDef) and node.name == "_propagate_global_flags"
    )
    fixture = _fixture()
    assert (
        hashlib.sha256(_canonical_ast(prop).encode()).hexdigest()
        == fixture["propagator_ast_sha256"]
    )
    assert (
        hashlib.sha256(_canonical_ast(_root_calls(restored)).encode()).hexdigest()
        == fixture["root_calls_ast_sha256"]
    )
