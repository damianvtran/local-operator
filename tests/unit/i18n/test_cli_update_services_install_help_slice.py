"""Update/services/install extraction: base-AST bytes, exact call shape, and ratchet.

The independent oracle was frozen from d8a799489a's owned argparse AST before
extraction; catalogue/generator changes cannot silently redefine expected English.
"""

import argparse
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path

from local_operator.i18n import catalogues

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 479
PINS = [
    ("_up", "update"),
    ("_up", "check"),
    ("_up", "from_snapshot"),
    ("_up", "allow_downgrade"),
    ("_up", "no_services"),
    ("_up", "no_roll"),
    ("_sr", "services"),
    ("_sr", "services_description"),
    ("_sr", "status"),
    ("_sr", "status_description"),
    ("_sr", "restart"),
    ("_sr", "wait"),
    ("_sr", "reclaim"),
    ("_sr", "reclaim_description"),
    ("_sr", "pid"),
    ("_in", "install"),
    ("_in", "prune"),
    ("_in", "keep"),
    ("_in", "migrate"),
    ("_in", "status"),
]

MODULES = {
    alias: importlib.import_module("local_operator.i18n.keys.cli_" + name + "_help")
    for alias, name in (("_up", "update"), ("_sr", "services"), ("_in", "install"))
}


def _message(alias, key):
    # Keep the authored g-format contract while the catalogue owns the prose.
    params = {"wait_seconds": "45"} if (alias, key) == ("_sr", "wait") else {}
    return getattr(MODULES[alias], key)(**params)


def _shape(alias, key):
    args = 'wait_seconds=f"{DEFAULT_SERVICES_WAIT_S:g}"' if (alias, key) == ("_sr", "wait") else ""
    return f'{alias}.{key}({args}).envelope(locale=help_locale)["text"]'


def test_base_ast_bytes_catalogue_context_and_function_parity():
    expected = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_update_services_install_base_ast.json").read_text()
    )["messages"]
    for alias, name in (("_up", "update"), ("_sr", "services"), ("_in", "install")):
        namespace = "cli." + name + "_help"
        oracle = {k: v for k, v in expected.items() if k.startswith(namespace + ".")}
        live = catalogues.load_catalogue("en", namespace)
        golden = json.loads(
            (
                REPO / "tests/unit/i18n/fixtures" / ("cli_" + name + "_help_en_golden.json")
            ).read_text()
        )
        assert live == golden
        assert {k: v.replace("{wait_seconds}", "45") for k, v in live.items()} == oracle
        context = catalogues.context("en", namespace)
        keys = {key for owner, key in PINS if owner == alias}
        assert set(context) == set(oracle) == {namespace + "." + key for key in keys}
        funcs = {
            key
            for key, value in vars(MODULES[alias]).items()
            if callable(value) and getattr(value, "__module__", None) == MODULES[alias].__name__
        }
        assert funcs == keys
        for key in keys:
            assert (
                _message(alias, key).envelope(locale="en")["text"] == oracle[namespace + "." + key]
            )
            assert all(
                context[namespace + "." + key].get(field) for field in ("role", "tone", "note")
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
    target.write_text(
        source + '\nprint("planted update services install prose")\n', encoding="utf-8"
    )
    assert i18n_check.main([]) == 1
    output = capsys.readouterr().out
    assert f"{CLI}: {count + 1} literal(s) against a ceiling of {count}" in output
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


def test_owned_block_copy_is_exactly_the_pinned_envelopes():
    # Scope by receiver, not by scanner findings: variable/epilog copy is also
    # owned, even when the ratchet cannot discover it.
    source = (REPO / CLI).read_text()
    tree = ast.parse(source)
    calls = _owned_copy_keywords(tree)
    expected = [ast.dump(ast.parse(_shape(alias, key), mode="eval").body) for alias, key in PINS]
    actual = [ast.dump(kw.value) for _, kw in sorted(calls, key=lambda row: row[0])]
    assert actual == expected
    assert len(actual) == 20


def test_french_locale_reaches_each_owned_message_and_falls_back(monkeypatch):
    from local_operator.cli import build_cli_parser
    from local_operator.i18n.messages import Msg

    seen = []
    original = Msg.envelope

    def record(self, *, locale=None):
        if self.code.startswith(("cli.update_help.", "cli.services_help.", "cli.install_help.")):
            seen.append((self.code, locale))
        return original(self, locale=locale)

    monkeypatch.setenv("LOP_LANG", "fr")
    monkeypatch.setattr(Msg, "envelope", record)
    parser = build_cli_parser()
    assert len(seen) == len(PINS)
    assert {locale for _, locale in seen} == {"fr"}
    for namespace in ("update", "services", "install"):
        assert (
            REPO / "local_operator/i18n/catalogues/fr" / ("cli." + namespace + "_help.json")
        ).is_file()
        en_values = catalogues.load_catalogue("en", "cli." + namespace + "_help")
        fr_values = catalogues.load_catalogue("fr", "cli." + namespace + "_help")
        assert set(fr_values) == set(en_values)
        alias = {"update": "_up", "services": "_sr", "install": "_in"}[namespace]
        for key, value in en_values.items():
            rendered_fr = _message(alias, key.rsplit(".", 1)[1]).envelope(locale="fr")["text"]
            expected_fr = fr_values[key].replace("{wait_seconds}", "45")
            assert rendered_fr == expected_fr
            assert rendered_fr != value.replace("{wait_seconds}", "45")
            # A shipped locale without this namespace still reads the source.
            assert _message(alias, key.rsplit(".", 1)[1]).envelope(locale="es")[
                "text"
            ] == value.replace("{wait_seconds}", "45")
    assert parser.prog


def test_wait_parameter_contract_tracks_the_actual_constant(monkeypatch):
    import local_operator.cli as cli

    signature = inspect.signature(MODULES["_sr"].wait)
    assert list(signature.parameters) == ["wait_seconds"]
    assert signature.parameters["wait_seconds"].annotation == "str"
    assert signature.return_annotation == "Msg"
    assert (
        "1.5s; must be positive"
        in MODULES["_sr"].wait(wait_seconds="1.5").envelope(locale="en")["text"]
    )
    monkeypatch.setattr(cli, "DEFAULT_SERVICES_WAIT_S", 1.5)
    parser = cli.build_cli_parser()
    # Narrow the yielded action itself: a generator's choices predicate does
    # not narrow Action.choices (which may be an Iterable or None) after next().
    service_actions = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    services = service_actions.choices["services"]
    assert isinstance(services, argparse.ArgumentParser)
    restart_actions = next(
        action for action in services._actions if isinstance(action, argparse._SubParsersAction)
    )
    restart = restart_actions.choices["restart"]
    assert isinstance(restart, argparse.ArgumentParser)
    assert "1.5s; must be positive" in restart.format_help()


def test_restoring_owned_help_preserves_the_owned_parser_protocol():
    # Freeze this slice's parser structure, not all of cli.py: a whole-file
    # assertion would incorrectly fail the next unrelated extraction tranche.
    fixture = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_update_services_install_base_ast.json").read_text()
    )
    restored = _restore_owned_help()
    assert (
        hashlib.sha256(_canonical_ast(_owned_parser_calls(restored)).encode()).hexdigest()
        == fixture["owned_parser_ast_sha256"]
    )


def _restore_owned_help():
    fixture = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_update_services_install_base_ast.json").read_text()
    )
    namespaces = {"_up": "update", "_sr": "services", "_in": "install"}
    replacements = {}
    for alias, key in PINS:
        shape = ast.dump(ast.parse(_shape(alias, key), mode="eval").body)
        if (alias, key) == ("_sr", "wait"):
            original = ast.parse("(" + fixture["wait_source"] + "\n)", mode="eval").body
        else:
            original = ast.Constant(fixture["messages"][f"cli.{namespaces[alias]}_help.{key}"])
        replacements[shape] = original

    class Restore(ast.NodeTransformer):
        count = 0

        def visit_Subscript(self, node):
            shape = ast.dump(node)
            if shape in replacements:
                self.count += 1
                return replacements[shape]
            return self.generic_visit(node)

    restored = ast.parse((REPO / CLI).read_text())
    restore = Restore()
    restored = restore.visit(restored)
    restored.body = [
        node
        for node in restored.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "local_operator.i18n.keys"
            and any(
                alias.name in {"cli_update_help", "cli_services_help", "cli_install_help"}
                for alias in node.names
            )
        )
    ]
    assert restore.count == len(PINS)
    return restored


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


def _owned_copy_keywords(tree):
    owners = {
        "update_parser",
        "services_parser",
        "services_subparsers",
        "services_restart",
        "services_reclaim",
        "install_parser",
        "install_subparsers",
        "prune_parser",
    }
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        receiver = node.func.value
        if not isinstance(receiver, ast.Name):
            continue
        owned = receiver.id in owners
        if receiver.id == "subparsers" and node.args and isinstance(node.args[0], ast.Constant):
            owned = node.args[0].value in {"update", "services", "install"}
        if owned:
            calls.extend(
                (node.lineno, kw)
                for kw in node.keywords
                if kw.arg in {"help", "description", "epilog"}
                and not (
                    isinstance(kw.value, ast.Attribute)
                    and isinstance(kw.value.value, ast.Name)
                    and kw.value.value.id == "argparse"
                    and kw.value.attr == "SUPPRESS"
                )
            )
    return calls


def _owned_parser_calls(tree):
    owners = {
        "update_parser",
        "services_parser",
        "services_subparsers",
        "services_restart",
        "services_reclaim",
        "install_parser",
        "install_subparsers",
        "prune_parser",
    }
    selected = []
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or not isinstance(node.func.value, ast.Name)
        ):
            continue
        receiver = node.func.value.id
        owned = receiver in owners
        if receiver == "subparsers" and node.args and isinstance(node.args[0], ast.Constant):
            owned = node.args[0].value in {"update", "services", "install"}
        if owned:
            selected.append(node)
    return sorted(selected, key=lambda node: (node.lineno, node.col_offset))
