"""Offline contract checks for the optional Harbor plugin boundary, not native QA."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class _Agent:
    def __init__(self, *args, **kwargs):
        self.options = kwargs

    async def exec_as_agent(self, environment, command, **kwargs):
        return None

    async def exec_as_root(self, environment, command):
        return None

    def _get_env(self, name):
        return None


def _load(agent: object = _Agent, decorator: object = lambda function: function):
    # Inject only the optional kit boundary; execute the real adapter file so
    # class construction and decorators cannot be accidentally bypassed.
    spec = importlib.util.spec_from_file_location(
        "coding_adapter_contract", Path(__file__).with_name("adapters.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    boundary = SimpleNamespace(
        BaseInstalledAgent=agent,
        ClaudeCode=agent,
        Codex=agent,
        Hermes=agent,
        OpenCode=agent,
        with_prompt_template=decorator,
    )
    with patch("importlib.import_module", return_value=boundary):
        spec.loader.exec_module(module)
    return module


class AdapterContractTests(unittest.TestCase):
    def test_concrete_bases_constructor_and_decorator_are_retained(self):
        decorated = []

        def decorate(function):
            decorated.append(function)
            return function

        module = _load(decorator=decorate)
        for name in ("FrozenClaudeCode", "FrozenCodex", "FrozenHermes", "OpenCodeV2"):
            self.assertIs(getattr(module, name).__bases__[0], _Agent)
        self.assertIs(module._Native.__bases__[0], _Agent)
        self.assertEqual(
            module.LocalOperator().options["version"], module.VERSIONS["local-operator"]
        )
        self.assertEqual(module.LocalOperator(version="explicit").options["version"], "explicit")
        self.assertEqual(len(decorated), 5)

    def test_nonclass_and_incomplete_agent_are_rejected(self):
        for candidate in (object(), object):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(TypeError, "not an installed Harbor agent"):
                    _load(agent=candidate)

    def test_present_but_noncallable_agent_method_is_rejected(self):
        class InvalidAgent(_Agent):
            pass

        # Structural Protocol checks prove attribute presence, not signatures or
        # callability. The boundary must independently reject unusable methods.
        setattr(InvalidAgent, "exec_as_agent", None)
        with self.assertRaisesRegex(TypeError, "not an installed Harbor agent"):
            _load(agent=InvalidAgent)

    def test_noncallable_decorator_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be callable"):
            _load(decorator=object())


class ServerToolDenialTests(unittest.TestCase):
    """R14 M2: no arm gets an Anthropic-executed (server) tool; each deny is native."""

    def test_claude_code_denies_websearch_and_keeps_client_webfetch(self):
        module = _load()
        self.assertEqual(module.FrozenClaudeCode().options["disallowed_tools"], "WebSearch")
        merged = module.FrozenClaudeCode(disallowed_tools="Bash,WebSearch").options
        self.assertEqual(merged["disallowed_tools"], "Bash,WebSearch")
        self.assertEqual(module.FrozenClaudeCode(disallowed_tools="Bash").options[
            "disallowed_tools"], "Bash,WebSearch")
        self.assertNotIn("WebFetch", module.FrozenClaudeCode().options["disallowed_tools"])

    def test_omp_run_loads_the_web_search_deny_overlay(self):
        import asyncio

        module = _load()
        commands = []

        class Omp(module.Omp):
            async def exec_as_agent(self, environment, command, **kwargs):
                commands.append(command)

            def _get_env(self, name):
                return "http://bench-gate:8787/r/run" if name == "BENCH_ANTHROPIC_URL" else None

        asyncio.run(Omp().run("do it", SimpleNamespace(), None))
        self.assertEqual(module.OMP_DENY_CONFIG, "web_search:\n  enabled: false\n")
        write, run = commands
        self.assertIn(module.OMP_DENY_PATH, write)
        self.assertIn("enabled: false", write)
        self.assertIn("--config " + module.OMP_DENY_PATH, run)
        self.assertLess(run.index("--config"), run.index("do it"))

    def test_every_arm_has_an_audited_server_tool_disposition(self):
        module = _load()
        import driver

        self.assertEqual(set(module.SERVER_TOOL_AUDIT), set(driver.HARNESS))
        for arm, rows in module.SERVER_TOOL_AUDIT.items():
            for tool, side, action in rows:
                with self.subTest(arm=arm, tool=tool):
                    self.assertIn(side, ("client", "server"))
                    if side == "server":
                        self.assertTrue(action.startswith("denied"), action)


if __name__ == "__main__":
    unittest.main()
