"""Harbor adapters for the coding cost arm, pinned to the host benchmark builds.

WHY: tasks are stateful Linux environments; a host-side CLI would edit the wrong
filesystem and cannot be scored by the verifier. These adapters execute inside
Harbor's task container. A read-only /opt/bench mount holds the pinned Linux
amd64 toolchain built by build_toolchain.sh (one installation, not one duplicate
dependency tree per trial). Runtime configuration and sessions are fresh in every
container; none of the host user's instructions, skills, MCP or credentials are
copied. The task's cwd and instruction are left unchanged.

Local Operator is pinned to released v0.94.22 (500883ef) and is metered only
through the per-run host edge (../lop_edge.py), reached via the bench gate
(egress.py); its launcher (lop_launch.py, mounted read-only) refuses any other
environment or source. No endpoint shim exists.

Every adapter ends install() with ``_preinference_checks``: before the first
model request, gold_probe.py proves no solution/test bytes are visible and the
egress fence holds (only the gate's listed ports connect). Both inputs come
from the Harbor process's own environment (set by driver.py), never from --ae,
so neither reaches the agent nor Harbor's persisted job config.

Harbor 0.25.0 already supports Claude Code, Codex, OpenCode and Hermes. We reuse
those adapters' run paths except OpenCode v2, whose Go-release CLI differs from
the npm v1 CLI Harbor implements. Its compatibility subclass uses the host
benchmark's exact command/config. Install overrides select the frozen toolchain
instead of mutable bootstrap installers. Codex is catalog+bridge (codex-best),
not its unknown-model fallback; the driver supplies the verified 1M catalog.

Usage through Harbor: -a adapters:LocalOperator (or Omp, Dsh, FrozenClaudeCode,
FrozenCodex, OpenCodeV2, FrozenHermes). Run from this directory, or put it on the
Harbor process's PYTHONPATH. Never put that PYTHONPATH on the agent environment.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

from gold_probe import PREFLIGHT_RECEIPT, preflight_binding, provenance_args


@runtime_checkable
class _InstalledAgent(Protocol):
    def __init__(self, *args: object, **kwargs: object) -> None:
        # A concrete constructor contract permits super() delegation in _Native;
        # runtime classes are still Harbor's, never this structural interface.
        return None

    async def exec_as_agent(
        self,
        environment: object,
        command: str,
        *,
        env: dict[str, str | None] | None = None,
        timeout_sec: int | None = None,
    ) -> object: ...

    async def exec_as_root(self, environment: object, command: str) -> object: ...

    def _get_env(self, name: str) -> str | None: ...


_Run = TypeVar("_Run", bound=Callable[..., Awaitable[object]])


@runtime_checkable
class _PromptTemplate(Protocol):
    def __call__(self, function: _Run) -> _Run: ...


def _agent_class(module: str, name: str) -> type[_InstalledAgent]:
    # Harbor belongs to the standalone task kit, not core dependencies. Keep
    # its concrete inheritance at runtime, validating the methods we rely on
    # rather than shipping analyzer stubs that can drift from that installation.
    candidate = getattr(importlib.import_module(module), name)
    if (
        not isinstance(candidate, type)
        or not issubclass(candidate, _InstalledAgent)
        or not all(
            callable(getattr(candidate, method, None))
            for method in (
                "exec_as_agent",
                "exec_as_root",
                "_get_env",
            )
        )
    ):
        raise TypeError(f"{module}.{name} is not an installed Harbor agent")
    return candidate


def _prompt_template() -> _PromptTemplate:
    candidate = getattr(
        importlib.import_module("harbor.agents.installed.base"), "with_prompt_template"
    )
    if not isinstance(candidate, _PromptTemplate):
        raise TypeError("Harbor with_prompt_template must be callable")
    return candidate


BaseInstalledAgent = _agent_class("harbor.agents.installed.base", "BaseInstalledAgent")
ClaudeCode = _agent_class("harbor.agents.installed.claude_code", "ClaudeCode")
Codex = _agent_class("harbor.agents.installed.codex", "Codex")
Hermes = _agent_class("harbor.agents.installed.hermes", "Hermes")
OpenCode = _agent_class("harbor.agents.installed.opencode", "OpenCode")
with_prompt_template = _prompt_template()

HERE = Path(__file__).resolve().parent
PUBLIC_LOADOUT = HERE / "../../public_loadout"
MODEL = os.environ.get("CODING_MODEL", "claude-sonnet-5-5")
NODE_PATH = "/opt/bench/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
VERSIONS = {
    "local-operator": "500883ef-v0.94.22",
    "omp": "18.11.2",
    "dsh": "0.2.0-rc.2",
    "claude-code": "2.1.274",
    "codex-best": "0.147.0",
    "opencode": "2.0.27",
    "hermes": "0.22.6",
}


SUPPORT = "/opt/bench-support"
PROBE_PYTHON = "/opt/bench/python/cpython-3.14-linux-x86_64-gnu/bin/python3.14"


async def _preinference_checks(agent, environment) -> None:
    """Gold isolation + egress fence, as root, before any model request.

    Without driver-provided inputs (BENCH_GOLD_DIGESTS, BENCH_GATE_PORTS,
    BENCH_EGRESS_CANARY_PORT) this refuses: a paid trial must never start
    without both proofs. The canary is the driver's run-owned OPEN host
    listener; it must be unreachable from ``main`` (R6 F16). The JSON
    results are kept on the host beside the agent logs.
    """
    digests = os.environ.get("BENCH_GOLD_DIGESTS", "")
    ports = os.environ.get("BENCH_GATE_PORTS", "")
    canary = os.environ.get("BENCH_EGRESS_CANARY_PORT", "")
    if (not digests or not ports or not canary.isdigit()
            or not all(p.isdigit() for p in ports.split(","))):
        raise ValueError(
            "BENCH_GOLD_DIGESTS, BENCH_GATE_PORTS and BENCH_EGRESS_CANARY_PORT are required "
            "before inference"
        )
    results = {}
    for check, args in (
        ("isolation", [*digests.split(","), *provenance_args()]),
        ("egress", ["--canary", canary, *ports.split(",")]),
    ):
        result = await environment.exec(
            command=shlex.join([PROBE_PYTHON, f"{SUPPORT}/gold_probe.py", check, *args]),
            user="root",
        )
        results[check] = (result.stdout or "").strip().splitlines()[-1:]
        try:
            parsed = json.loads(results[check][0])
        except (ValueError, IndexError):
            parsed = {}
        if (result.return_code != 0 or not isinstance(parsed, dict)
                or parsed.get("ok") is not True
                or (check == "isolation" and parsed.get("manifest_sha256")
                    != os.environ.get("BENCH_PUBLIC_PROVENANCE_SHA256"))):
            _write_preinference(agent, results)
            _write_preflight_receipt(agent, environment, results, ok=False)
            raise RuntimeError(f"pre-inference {check} check failed")
    _write_preinference(agent, results)
    _write_preflight_receipt(agent, environment, results, ok=True)


def _write_preflight_receipt(agent, environment, results, *, ok: bool) -> None:
    """Host-side record that THIS trial passed the strict pre-inference scan.

    R11 F28: BenchVerifier grades a late hidden match only with this receipt.
    It goes in the trial dir, the parent of the agent log dir: that dir is not
    a container mount, unlike ``/logs/agent`` (bench-preinference.json), so
    nothing in ``main`` can forge it. Bound to trial id, image and manifest.
    """
    logs = getattr(agent, "logs_dir", None)
    if logs is None or Path(logs).name != "agent":
        return
    receipt = {**preflight_binding(getattr(environment, "session_id", "")), "ok": ok,
               "isolation_sha256": hashlib.sha256(
                   "".join(results.get("isolation", [])).encode()).hexdigest()}
    (Path(logs).parent / PREFLIGHT_RECEIPT).write_text(json.dumps(receipt, sort_keys=True) + "\n")


def _write_preinference(agent, results) -> None:
    logs = getattr(agent, "logs_dir", None)
    if logs is not None:
        Path(logs).mkdir(parents=True, exist_ok=True)
        (Path(logs) / "bench-preinference.json").write_text(json.dumps(results, indent=1) + "\n")


async def _link(agent, environment, binary: str, name: str) -> None:
    """Publish a pinned executable into the empty container user's local bin."""
    await agent.exec_as_agent(
        environment,
        f'mkdir -p "$HOME/.local/bin" && ln -s {shlex.quote(binary)} "$HOME/.local/bin/{name}" && '
        f'PATH={NODE_PATH} "$HOME/.local/bin/{name}" --version',
        timeout_sec=120,
    )


class FrozenClaudeCode(ClaudeCode):
    """Harbor's built-in Claude Code run path, byte-pinned installer only."""

    async def install(self, environment):
        await _link(self, environment, "/opt/bench/claude/bin/claude", "claude")
        await _preinference_checks(self, environment)


class FrozenCodex(Codex):
    """Harbor's built-in Codex run path plus the catalog/translator bridge."""

    async def install(self, environment):
        await _link(self, environment, "/opt/bench/codex/bin/codex", "codex")
        await self.exec_as_agent(environment, "mkdir -p /tmp/bench-support")
        await environment.upload_file(
            HERE / "codex-model-catalog.json", "/tmp/bench-support/codex-model-catalog.json"
        )
        # Codex's built-in run command uses the bare binary; make it available
        # without borrowing the host PATH or changing any task Python binary.
        await self.exec_as_root(environment, 'ln -s "$HOME/.local/bin/codex" /usr/local/bin/codex')
        await _preinference_checks(self, environment)


class FrozenHermes(Hermes):
    """Pinned minimal Hermes plus its verified named Anthropic provider route.

    The native built-in provider can silently ignore a custom URL; use the
    named providers/transport surface verified by the host adapter lane.
    A normal developer venv is an upstream-supported bootstrap contract and
    avoids PM's unrelated media/browser tool downloads. No source is patched.
    """

    async def install(self, environment):
        await _link(self, environment, "/opt/bench/hermes/hermes-agent/.venv/bin/hermes", "hermes")
        await _preinference_checks(self, environment)

    @with_prompt_template
    async def run(self, instruction, environment, context):
        setup = """import os,pathlib,yaml
p=pathlib.Path('/tmp/bench-hermes');p.mkdir(exist_ok=True)
src=pathlib.Path('/opt/bench/hermes/hermes-agent/cli-config.yaml.example')
cfg=yaml.safe_load(src.read_text()) if src.exists() else {}
cfg['model']={'default':'claude-sonnet-5-5','provider':'bench','context_length':1000000}
cfg['providers']={'bench':{'api':os.environ['BENCH_ANTHROPIC_URL'],'key_env':'ANTHROPIC_API_KEY','transport':'anthropic_messages','default_model':'claude-sonnet-5-5'}}
cfg.setdefault('terminal',{})['backend']='local'
(p/'config.yaml').write_text(yaml.safe_dump(cfg))
"""
        await self.exec_as_agent(environment, "/opt/bench/lop/bin/python -c " + shlex.quote(setup))
        await self.exec_as_agent(
            environment,
            shlex.join(
                [
                    "/opt/bench/hermes/hermes-agent/.venv/bin/hermes",
                    "--yolo",
                    "chat",
                    "-q",
                    instruction,
                    "-Q",
                    "--model",
                    MODEL,
                    "--provider",
                    "bench",
                    "--reasoning",
                    "high",
                ]
            )
            + " 2>&1 | tee /logs/agent/hermes.txt",
            env={
                "PATH": NODE_PATH,
                "HERMES_HOME": "/tmp/bench-hermes",
                "ANTHROPIC_API_KEY": "bench-dummy",
            },
        )


class OpenCodeV2(OpenCode):
    """Compatibility shim: Harbor targets npm 1.x, host arm is Go v2.0.27."""

    async def install(self, environment):
        await _link(self, environment, "/opt/bench/opencode/bin/opencode", "opencode")
        await _preinference_checks(self, environment)

    @with_prompt_template
    async def run(self, instruction, environment, context):
        url = self._get_env("BENCH_ANTHROPIC_URL")
        if not url:
            raise ValueError("BENCH_ANTHROPIC_URL required")
        cfg = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "provider": {
                "anthropic": {"options": {"baseURL": url + "/v1", "apiKey": "bench-dummy"}}
            },
        }
        await self.exec_as_agent(
            environment,
            f'"$HOME/.local/bin/opencode" run --standalone --auto --format json '
            f"-m anthropic/{MODEL} {shlex.quote(instruction)} 2>&1 | tee /logs/agent/opencode.txt",
            env={
                "PATH": NODE_PATH,
                "OPENCODE_CONFIG_CONTENT": json.dumps(cfg),
                "OPENCODE_DISABLE_AUTOUPDATE": "1",
            },
        )


class _Native(BaseInstalledAgent):
    """Common task-side invocation without an extra harness system prompt."""

    PIN = ""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("version", self.PIN)
        super().__init__(*args, **kwargs)

    async def _upload_support(self, environment, files):
        await self.exec_as_agent(environment, "mkdir -p /tmp/bench-support")
        for name in files:
            await environment.upload_file(HERE / name, "/tmp/bench-support/" + name)

    async def _run(self, environment, argv, env):
        await self.exec_as_agent(
            environment,
            shlex.join(argv) + " 2>&1 | tee /logs/agent/native.jsonl",
            env={"PATH": NODE_PATH, **env},
        )


# The edge's child environment, rewritten for the container: proxy host is the
# gate (same edge port), CA is the read-only mount. The driver passes these to
# the agent via --ae; lop_launch.py refuses any other shape.
EDGE_ENV_KEYS = (
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "RADIENT_API_KEY",
    "PIP_CERT", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS", "BENCH_EDGE_CA_SHA256",
    # The frozen package mirror pin every arm gets (driver.package_agent_env).
    "UV_PYTHON_INSTALL_MIRROR",
)


def _edge_env(agent) -> dict[str, str]:
    env = {k: v for k in EDGE_ENV_KEYS if (v := agent._get_env(k))}
    if "HTTPS_PROXY" not in env:
        raise ValueError("LoP needs the edge environment (HTTPS_PROXY via the bench gate)")
    return env


class LocalOperator(_Native):
    """Released v0.94.22 behind the host edge; the read-only launcher checks it all."""

    PIN = VERSIONS["local-operator"]

    @staticmethod
    def name():
        return "local-operator"

    async def install(self, environment):
        # Import from the frozen overlay, not the venv's own (unreleased
        # candidate) install; lop_launch.py re-verifies RECORD hashes at run.
        await self.exec_as_agent(
            environment,
            "PYTHONPATH=/opt/bench/lop-baseline-500883ef /opt/bench/lop/bin/python "
            "-c 'import local_operator'",
            timeout_sec=120,
        )
        await _preinference_checks(self, environment)

    @with_prompt_template
    async def run(self, instruction, environment, context):
        await self._run(
            environment,
            [
                "/opt/bench/lop/bin/python",
                f"{SUPPORT}/lop_launch.py",
                "exec",
                "--hosting",
                "anthropic",
                "--model",
                MODEL,
                "--yolo",
                "--json",
                instruction,
            ],
            {
                **_edge_env(self),
                "ANTHROPIC_API_KEY": "bench-dummy",
                "LOCAL_OPERATOR_CONFIG_DIR": "/tmp/bench-lop",
                "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1",
                "LOCAL_OPERATOR_ALLOW_NESTED_SESSION": "1",
            },
        )


class Omp(_Native):
    """Released Linux amd64 omp 18.11.2, not Harbor's upstream pi agent."""

    PIN = VERSIONS["omp"]

    @staticmethod
    def name():
        return "omp"

    async def install(self, environment):
        await self.exec_as_agent(environment, "/opt/bench/omp/bin/omp --version", timeout_sec=120)
        await _preinference_checks(self, environment)

    @with_prompt_template
    async def run(self, instruction, environment, context):
        # cwd is Harbor's task cwd; omitting --cwd preserves it exactly.
        await self._run(
            environment,
            [
                "/opt/bench/omp/bin/omp",
                "-p",
                "--mode",
                "json",
                "--model",
                "anthropic/" + MODEL,
                "--auto-approve",
                "--no-session",
                "--allow-home",
                instruction,
            ],
            {
                "ANTHROPIC_BASE_URL": self._get_env("BENCH_ANTHROPIC_URL"),
                "ANTHROPIC_API_KEY": "bench-dummy",
                "PI_CODING_AGENT_DIR": "/tmp/bench-omp",
            },
        )


class Dsh(_Native):
    """Native headless profile with a declared Anthropic provider route."""

    PIN = VERSIONS["dsh"]

    @staticmethod
    def name():
        return "dsh"

    async def install(self, environment):
        await self.exec_as_agent(
            environment,
            "/opt/bench/node/bin/node "
            "/opt/bench/dsh/lib/node_modules/@deepseek-ai/dsh/lib/bin.js --version",
        )
        url = self._get_env("BENCH_ANTHROPIC_URL")
        if not url:
            raise ValueError("BENCH_ANTHROPIC_URL required")
        # Generate only transport config, never copy a host's private profile.
        # Preserve the pinned release's JS URL binding and exact native model
        # metadata; a different config shape would silently select a fallback.
        config = (
            "# Native provider route: this release's catalog predates the measured mo"
            "del.\n"
            "# Declare the verified 1M context and output limit instead of a fallback"
            ", plus\n"
            "# the exposed high reasoning level. The wire probe must confirm its tran"
            "slation.\n"
            "# All keys are dummy; the shared metering proxy owns upstream authentica"
            "tion.\n"
            "- id: llm-pi-ai\n"
            "  config:\n"
            "    providers:\n"
            "      anthropic:\n"
            "        apiKeyEnv: ANTHROPIC_API_KEY\n"
            "        baseURL: !!js process.env.BENCH_ANTHROPIC_URL\n"
            "        models:\n"
            "          - id: claude-sonnet-5-5\n"
            "            contextWindow: 1000000\n"
            "            maxTokens: 128000\n"
            "            compat:\n"
            "              forceAdaptiveThinking: true\n"
            "              supportsStrictTools: true\n"
            "            reasoningEfforts:\n"
            "              low: low\n"
            "              medium: medium\n"
            "              high: high\n"
            "- id: agent-default-model\n"
            "  config:\n"
            "    provider: anthropic\n"
            "    model: claude-sonnet-5-5\n"
            "    reasoningEffort: high\n"
        )
        await self.exec_as_agent(
            environment,
            "mkdir -p /tmp/bench-support && printf %s "
            + shlex.quote(config)
            + " > /tmp/bench-support/dsh-anthropic.yml",
        )
        await _preinference_checks(self, environment)

    @with_prompt_template
    async def run(self, instruction, environment, context):
        await self._run(
            environment,
            [
                "/opt/bench/node/bin/node",
                "/opt/bench/dsh/lib/node_modules/@deepseek-ai/dsh/lib/bin.js",
                "--profile",
                "headless",
                "--patch",
                "/tmp/bench-support/dsh-anthropic.yml",
                "--json",
                instruction,
            ],
            {
                "DSH_HOME": "/tmp/bench-dsh",
                "DSH_TELEMETRY_MODE": "DISABLED",
                "DSH_TELEMETRY_DISABLED": "1",
                "DSH_PERMISSION_MODE": "danger-full-access",
                "ANTHROPIC_API_KEY": "bench-dummy",
            },
        )
