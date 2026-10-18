"""Stop/refresh/monitor extraction: base-AST bytes, exact call shape, and ratchet.

The independent oracle was frozen from a39684d55f's owned argparse AST before
extraction; catalogue/generator changes cannot silently redefine expected English.
"""

import ast
import importlib
import json
from pathlib import Path

from local_operator.i18n import catalogues

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 496
PINS = [
    ("_st", "stop"),
    ("_st", "target"),
    ("_st", "pid"),
    ("_st", "session"),
    ("_st", "all"),
    ("_st", "yes"),
    ("_st", "json"),
    ("_st", "timeout"),
    ("_st", "force"),
    ("_re", "refresh"),
    ("_re", "target"),
    ("_re", "pid"),
    ("_re", "session"),
    ("_re", "all"),
    ("_re", "json"),
    ("_re", "timeout"),
    ("_mn", "monitor"),
    ("_mn", "status"),
    ("_mn", "json"),
    ("_mn", "cancel"),
    ("_mn", "session"),
    ("_mn", "monitor_id"),
    ("_mn", "json"),
]
MODULES = {
    alias: importlib.import_module("local_operator.i18n.keys.cli_" + name + "_help")
    for alias, name in (("_st", "stop"), ("_re", "refresh"), ("_mn", "monitor"))
}


def test_base_ast_bytes_catalogue_context_and_function_parity():
    expected = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_stop_refresh_monitor_base_ast.json").read_text()
    )
    source = (REPO / CLI).read_text()
    for alias, name in (("_st", "stop"), ("_re", "refresh"), ("_mn", "monitor")):
        namespace = "cli." + name + "_help"
        oracle = {k: v for k, v in expected.items() if k.startswith(namespace + ".")}
        live = catalogues.load_catalogue("en", namespace)
        golden = json.loads(
            (
                REPO / "tests/unit/i18n/fixtures" / ("cli_" + name + "_help_en_golden.json")
            ).read_text()
        )
        assert live == golden == oracle
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
            assert f'{alias}.{key}().envelope(locale=help_locale)["text"]' in source
            assert (
                getattr(MODULES[alias], key)().envelope(locale="en")["text"]
                == oracle[namespace + "." + key]
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
    target.write_text(source + '\nprint("planted stop refresh monitor prose")\n', encoding="utf-8")
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
    owners = {
        "stop_parser",
        "refresh_parser",
        "monitor_parser",
        "monitor_status",
        "monitor_cancel",
        "monitor_sub",
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
            owned = node.args[0].value in {"stop", "refresh", "monitor"}
        if owned:
            calls.extend(
                (node.lineno, kw)
                for kw in node.keywords
                if kw.arg in {"help", "description", "epilog"}
            )
    expected = [f'{alias}.{key}().envelope(locale=help_locale)["text"]' for alias, key in PINS]
    actual = [
        ast.get_source_segment(source, kw.value) for _, kw in sorted(calls, key=lambda row: row[0])
    ]
    assert actual == expected
    assert len(actual) == 23


def test_french_locale_reaches_each_owned_message_and_falls_back(monkeypatch):
    from local_operator.cli import build_cli_parser
    from local_operator.i18n.messages import Msg

    seen = []
    original = Msg.envelope

    def record(self, *, locale=None):
        if self.code.startswith(("cli.stop_help.", "cli.refresh_help.", "cli.monitor_help.")):
            seen.append((self.code, locale))
        return original(self, locale=locale)

    monkeypatch.setenv("LOP_LANG", "fr")
    monkeypatch.setattr(Msg, "envelope", record)
    parser = build_cli_parser()
    assert len(seen) == len(PINS)
    assert {locale for _, locale in seen} == {"fr"}
    for namespace in ("stop", "refresh", "monitor"):
        assert not (
            REPO / "local_operator/i18n/catalogues/fr" / ("cli." + namespace + "_help.json")
        ).exists()
        for key, value in catalogues.load_catalogue("en", "cli." + namespace + "_help").items():
            alias = {"stop": "_st", "refresh": "_re", "monitor": "_mn"}[namespace]
            assert (
                getattr(MODULES[alias], key.rsplit(".", 1)[1])().envelope(locale="fr")["text"]
                == value
            )
    assert parser.prog
