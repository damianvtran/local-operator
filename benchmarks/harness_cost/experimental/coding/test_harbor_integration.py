"""Real Harbor 0.25 construction paths (S9e); no Docker, model, network or credentials.

WHY: the stdlib tests use stand-in Harbor bases, so they could not see that
Harbor 0.25 constructs an import-path agent WITHOUT the oracle kwargs
(d0b2b32c sweep: TypeError on every qualification row), nor what Harbor's
Codex adapter renders into ``config.toml``. These tests drive Harbor's own
``Trial.create`` -> ``Trial._init_agent`` -> ``AgentFactory`` and its own
Codex config builder. ``Trial.create`` loads a local task and builds the agent
and environment OBJECTS only; nothing is started.

Needs Harbor 0.25.0 importable (the tool's interpreter, e.g.
``~/.local/share/uv/tools/harbor/bin/python``); it FAILS, never skips, without
it, so it is a separate command from the stdlib CI set (coding/README.md).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harbor.agents.installed.codex import Codex  # noqa: E402
from harbor.agents.oracle import OracleAgent  # noqa: E402
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig  # noqa: E402
from harbor.trial.trial import Trial  # noqa: E402

import driver  # noqa: E402
import oracle_qualification  # noqa: E402

TASK_TOML = """version = "1.0"

[agent]
timeout_sec = 123.0

[verifier]
timeout_sec = 60.0

[environment]
docker_image = "bench/never-pulled:latest"
"""


class NoKwargsOracle(OracleAgent):
    """Negative control: an import-path OracleAgent subclass with no fix."""


class HarborIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(importlib.metadata.version("harbor"), "0.25.0")
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = self.root / "task"
        for rel, text in (("task.toml", TASK_TOML), ("instruction.md", "noop\n"),
                          ("environment/Dockerfile", "FROM scratch\n"),
                          ("tests/test.sh", "#!/bin/sh\nexit 0\n"),
                          ("solution/solve.sh", "#!/bin/sh\nexit 0\n")):
            (self.task / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.task / rel).write_text(text)

    def trial(self, import_path):
        config = TrialConfig(task=TaskConfig(path=self.task), trials_dir=self.root / "trials",
                             trial_name="t", agent=AgentConfig(import_path=import_path))
        return asyncio.run(Trial.create(config))

    def test_stock_harbor_builds_the_qualification_oracle_with_its_oracle_values(self):
        # Negative control first: Harbor's real non-"oracle" branch omits the
        # kwargs, exactly the d0b2b32c TypeError.
        with self.assertRaisesRegex(TypeError, "task_dir"):
            self.trial("test_harbor_integration:NoKwargsOracle")
        trial = self.trial("oracle_qualification:QualificationOracle")
        agent = trial.agent
        self.assertIsInstance(agent, oracle_qualification.QualificationOracle)
        self.assertNotEqual(trial.config.agent.name, "oracle")
        self.assertIs(agent._trial_paths, trial.paths)
        self.assertEqual(Path(agent._task.task_dir), Path(trial.task.task_dir))
        self.assertEqual(agent._agent_timeout_sec, trial.agent_timeout_sec)
        self.assertEqual(agent._agent_timeout_sec, 123.0)
        self.assertEqual(Path(agent.logs_dir), trial.paths.agent_dir)
        self.assertEqual(agent._receipt_path(),
                         trial.paths.trial_dir / oracle_qualification.RECEIPT)

    def test_explicit_oracle_kwargs_are_forwarded_and_partial_or_absent_refuse(self):
        trial = self.trial("oracle_qualification:QualificationOracle")
        agent = oracle_qualification.QualificationOracle(
            logs_dir=trial.paths.agent_dir, task_dir=self.task, trial_paths=trial.paths,
            agent_timeout_sec=7.0)
        self.assertIs(agent._trial_paths, trial.paths)
        self.assertEqual(agent._agent_timeout_sec, 7.0)
        # Outside a Harbor Trial there is nothing to bind to: refuse.
        with self.assertRaisesRegex(TypeError, "Harbor Trial"):
            oracle_qualification.QualificationOracle(logs_dir=trial.paths.agent_dir)
        with self.assertRaisesRegex(TypeError, "both"):
            oracle_qualification.QualificationOracle(
                logs_dir=trial.paths.agent_dir, task_dir=self.task)

    def test_harbor_renders_claude_code_websearch_deny(self):
        # R14 M2: Harbor's own ClaudeCode option builder must put the native
        # deny on the CLI (and into its settings permissions on the ACP path).
        import adapters

        agent = adapters.FrozenClaudeCode(logs_dir=self.root / "agent",
                                          model_name="anthropic/claude-sonnet-5-5")
        self.assertEqual(agent.options.disallowed_tools, "WebSearch")
        flags = agent.build_cli_flags()
        self.assertIn("--disallowedTools WebSearch", flags)
        self.assertNotIn("WebFetch", flags)

    def test_harbor_renders_codex_best_web_search_disabled(self):
        # Harbor's own Codex builder renders the driver's --ak config into the
        # config.toml Codex reads; the mode must survive as "disabled".
        config = driver.codex_config("http://bench-gate:8789/v1")
        agent = Codex(logs_dir=self.root / "agent", model_name="anthropic/claude-sonnet-5-5",
                      config=config)
        self.assertIsNone(agent.options.web_search)  # nothing overrides the base
        effective = agent._build_effective_config("http://bench-gate:8789/v1")
        rendered = tomllib.loads(__import__("toml").dumps(effective))
        self.assertEqual(rendered["web_search"], "disabled")
        self.assertEqual(rendered["model_provider"], "bench")


if __name__ == "__main__":
    unittest.main()
