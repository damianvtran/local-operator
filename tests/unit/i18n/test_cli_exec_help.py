"""CLI20's read-only boundary and zero-argument copy contract.

Help must not repair a half-written config or activate the process-wide locale:
argparse exits before ordinary startup has authority to do either. These tests
use real reader bytes, not a mocked YAML result, to cover that boundary.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import gc
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from local_operator import config
from local_operator.i18n import messages, resolve, runtime, text


def _bootstrap():
    # An assertion, not an import error, makes the first RED pin the missing
    # safe entrypoint rather than accidentally testing import machinery.
    spec = importlib.util.find_spec("local_operator.i18n.cli_help")
    assert spec is not None, "Help needs a read-only locale bootstrap"
    return importlib.import_module("local_operator.i18n.cli_help")


@pytest.mark.parametrize(
    "payload, expected",
    [
        (b"values:\n  language: fr\n", "fr"),
        (None, "es"),
        (b"values: [", "es"),
        (b"- not-a-mapping\n", "es"),
        (b"values:\n  language: 7\n", "es"),
        (b"values: []\n", "es"),
        (b"\xff\xfe", "es"),
    ],
)
def test_bootstrap_never_repairs_or_activates(tmp_path, monkeypatch, payload, expected):
    helper = _bootstrap()
    path = tmp_path / "config.yml"
    if payload is not None:
        path.write_bytes(payload)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOP_LANG", "es")
    monkeypatch.delenv("LOP_I18N", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Help must not construct ConfigManager or read implicit config")

    monkeypatch.setattr(config.ConfigManager, "__init__", forbidden)
    monkeypatch.setattr(resolve, "config_value", forbidden)
    previous = text.active_locale()
    assert helper.help_locale() == expected
    assert text.active_locale() == previous
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def _run_real_entrypoint(
    tmp_path, monkeypatch, capsys, payload, expected, argv, *, gc_in_main=False
):
    helper = _bootstrap()
    from local_operator import cli

    path = tmp_path / "config.yml"
    path.write_bytes(payload)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOP_LANG", "es")
    monkeypatch.delenv("LOP_I18N", raising=False)
    monkeypatch.setattr(sys, "argv", ["lop", *argv])

    def forbidden(*args, **kwargs):
        pytest.fail("argparse exit must precede normal config/credential startup")

    monkeypatch.setattr(config.ConfigManager, "__init__", forbidden)
    monkeypatch.setattr(resolve, "config_value", forbidden)
    actual_resolve = helper.help_locale
    resolved = []

    def record_locale():
        if gc_in_main:
            # Models the collector firing at an arbitrary allocation inside the
            # entrypoint, which is when CPython finalises whatever garbage is
            # still waiting from earlier tests.
            gc.collect()
        locale = actual_resolve()
        resolved.append(locale)
        return locale

    monkeypatch.setattr(helper, "help_locale", record_locale)
    previous = text.active_locale()
    # cli.main() replaces the root logger's handlers with a stderr StreamHandler
    # (configure_cli_logging) whose stream is capsys's buffer for this test. A
    # coroutine/Task leaked by an EARLIER test (e.g. a pending Session
    # "persist-subagent-roster" task) is only finalised when the collector next
    # runs, and asyncio reports that through the root logger: collected before
    # main() it lands in pytest's own capture handlers, collected after it lands
    # in the captured stderr this test asserts is empty. Drain that garbage now,
    # while logging still routes to pytest, so the assertion below sees only what
    # the entrypoint itself wrote. Nothing is silenced: a record the entrypoint
    # emits still fails the test.
    gc.collect()
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    output = capsys.readouterr()
    assert output.out and not output.err
    assert resolved == [expected]
    assert text.active_locale() == previous
    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes() == payload


@pytest.mark.parametrize(
    "payload, expected", [(b"values:\n  language: fr\n", "fr"), (b"\xff\xfe", "es")]
)
@pytest.mark.parametrize("argv", [["--help"], ["exec", "--help"], ["--version"]])
def test_real_entrypoint_exits_without_startup_side_effects(
    tmp_path, monkeypatch, capsys, payload, expected, argv
):
    _run_real_entrypoint(tmp_path, monkeypatch, capsys, payload, expected, argv)


@pytest.fixture
def leaked_pending_roster_task():
    """Leave a pending, unreachable-but-uncollected Task, as a prior test might.

    Mirrors CI 38630407658: ``Session._persist_subagent_roster`` scheduled as
    ``persist-subagent-roster`` on a loop that is then closed, so its finalisation
    logs "Task was destroyed but it is pending!". Cyclic so only ``gc.collect()``
    frees it, and the collector is held off until the test decides when it runs.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    loop = asyncio.new_event_loop()

    async def roster():
        await asyncio.sleep(3600)

    task = loop.create_task(roster(), name="persist-subagent-roster")
    loop.run_until_complete(asyncio.sleep(0))
    holder: dict[str, object] = {"task": task}
    holder["self"] = holder
    del task, holder
    loop.close()
    try:
        yield
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


def test_real_entrypoint_stderr_is_not_polluted_by_prior_tests_garbage(
    tmp_path, monkeypatch, capsys, leaked_pending_roster_task
):
    _run_real_entrypoint(
        tmp_path,
        monkeypatch,
        capsys,
        b"values:\n  language: fr\n",
        "fr",
        ["--help"],
        gc_in_main=True,
    )


def test_bootstrap_kill_switch_and_eligibility(tmp_path, monkeypatch):
    helper = _bootstrap()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOP_LANG", "zh-CN")
    assert helper.help_locale() == "en"
    (tmp_path / "config.yml").write_text("values:\n  language: fr\n")
    monkeypatch.setenv("LOP_I18N", "0")
    assert helper.help_locale() == "en"


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("fr", "fr"),
        ("es", "es"),
        ("ru", "ru"),
        ("vi", "vi"),
        ("hi", "hi"),
        ("ur", "en"),
        ("zh-CN", "en"),
        ("unknown", "en"),
    ],
)
def test_bootstrap_uses_eligible_not_prompt_preference(tmp_path, monkeypatch, tag, expected):
    helper = _bootstrap()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_I18N", raising=False)
    monkeypatch.setenv("LOP_LANG", tag)
    monkeypatch.setattr(resolve, "os_language", lambda **kwargs: None)
    assert helper.help_locale() == expected


def test_linux_os_chain_with_explicit_auto(tmp_path, monkeypatch):
    helper = _bootstrap()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_LANG", raising=False)
    monkeypatch.delenv("LOP_I18N", raising=False)
    monkeypatch.setenv("LC_ALL", "vi_VN.UTF-8")
    monkeypatch.setattr(resolve.platform, "system", lambda: "Linux")
    assert helper.help_locale() == "vi"


def test_schema_raw_icu_is_not_literal_text():
    with pytest.raises(runtime.MessageSyntaxError):
        runtime.parse_message("a JSON object {'required_sections': [...]}")


def test_twenty_generated_zero_argument_messages_and_english_fallback():
    _bootstrap()
    from local_operator.i18n import catalogues
    from local_operator.i18n.keys import cli_exec_help as keys

    source = catalogues.load_catalogue("en", "cli.exec_help")
    functions = [f for _, f in inspect.getmembers(keys, inspect.isfunction)]
    assert len(source) == len(functions) == 20
    for function in functions:
        assert not inspect.signature(function).parameters
        msg = function()
        assert msg.params == {}
        en = messages.render(msg.code, msg.params, locale="en")
        assert messages.render(msg.code, msg.params, locale="fr") == en
    assert "{'required_sections': [...]}" in keys.output_schema().envelope(locale="en")["text"]


def _help_calls():
    root = Path(__file__).resolve().parents[3]
    calls = []
    for rel, receiver in [("cli.py", "exec_parser"), ("exec_startup.py", None)]:
        tree = ast.parse((root / "local_operator" / rel).read_text())
        scope = (
            tree
            if receiver
            else next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "add_startup_arguments"
            )
        )
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument":
                continue
            if receiver and (
                not isinstance(node.func.value, ast.Name) or node.func.value.id != receiver
            ):
                continue
            keyword = next((k for k in node.keywords if k.arg == "help"), None)
            if keyword is not None and not isinstance(keyword.value, ast.Attribute):
                calls.append(node)
    return calls


def test_extracted_sites_ratchet_without_activating_legacy_cli(i18n_check):
    # cli.py remains advisory: activating its 772 legacy literals would expand
    # CLI20 into a whole-CLI extraction. Scan only these existing sink calls
    # with the production scanner and prove a reintroduced literal is rejected.
    calls = _help_calls()
    assert len(calls) == 20
    source = "\n".join(ast.unparse(call) for call in calls)
    assert i18n_check.scan_source("local_operator/cli.py", source) == []
    next(k for k in calls[0].keywords if k.arg == "help").value = ast.Constant(
        "New uncatalogued help text"
    )
    counterfactual = "\n".join(ast.unparse(call) for call in calls)
    violations = i18n_check.scan_source("local_operator/cli.py", counterfactual)
    assert len(violations) == 1
    with pytest.raises(AssertionError):
        assert violations == [], "CLI20's new help site must fail the scoped ratchet"
    baseline = i18n_check.load_baseline(i18n_check.BASELINE)
    assert not i18n_check.in_enforced_scope("local_operator/cli.py", baseline["enforced"])
    # 772 -> 807 in the scanner-coverage change that taught the checker to fold
    # `+` and the ternary in argument position: the legacy CLI's own literal
    # count is unchanged, but the improved scanner now SEES more of it. The
    # pin moves with the detector, which is the point of pinning it here.
    #
    # 807 -> 746 in the CLI tranche 1 (#2552): the `browser_command` region's
    # findings were extracted (55 sites) or pragma'd with a reason (6), so the
    # ceiling moved by exactly the count the tranche cleared. The pin moves
    # with the extraction too -- same discipline, opposite direction.
    assert baseline["files"]["local_operator/cli.py"] == 746
    assert baseline["files"]["local_operator/exec_startup.py"] == 0


def test_standalone_startup_defaults_to_english_without_global_activation():
    _bootstrap()
    from local_operator.exec_startup import add_startup_arguments

    previous = text.active_locale()
    try:
        text.set_active_locale("fr")
        parser = argparse.ArgumentParser()
        add_startup_arguments(parser)
        assert "Attach a saved team" in parser.format_help()
        assert parser.parse_args([]).output_retries is None
        with pytest.raises(SystemExit) as error:
            parser.parse_args(["--goal", "goal", "--clear-goal"])
        assert error.value.code == 2
    finally:
        text.set_active_locale(previous)
