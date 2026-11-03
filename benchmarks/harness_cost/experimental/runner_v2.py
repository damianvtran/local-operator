"""V2 integration runner for seven Claude Sonnet harness adapters, development only.

WHY: historical private-loadout rep-0 results must never absorb new public-arm
trials. This runner accepts only clean/public-full, always prefixes v2dev__, and
writes runs_v2dev.jsonl separately. Public skill/instruction/schema inputs are
byte-checked, not copied from personal configurations. No execution is allowed
without a frozen preregistration and runtime-bound zero-cost wire attestations.
Config-only and preflight-only modes never call a model or fetch credentials.

CONSTRAINTS: no installs, no shared-service restarts, no reusable run ids, no
credential files in artifacts. Hermes' detached gateway and the per-run Codex
search bridge are explicitly reaped. Model usage and Radient classification
spend are separate USD columns. A passed filesystem/config check is NOT wire
validation: missing capture evidence blocks paid execution.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Literal, Mapping, NotRequired, Sequence, TypedDict

# The external task and capture formats have distinct shapes. Keep their
# contracts explicit so a metadata dictionary cannot become a command by accident.
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class TaskParameters(TypedDict):
    files: list[dict[str, str]]
    command: str | list[str]


class TaskConfig(TypedDict):
    type: str
    parameters: TaskParameters


class TaskEvaluator(TypedDict):
    result: dict[str, str]


class TaskIdentity(TypedDict):
    id: str


class PreparedTask(TaskIdentity):
    _domain: str
    config: list[TaskConfig]


class Task(PreparedTask):
    instruction: str
    evaluator: TaskEvaluator


class RuntimeIdentity(TypedDict):
    binary: str
    sha256: str
    entry: NotRequired[str]
    source_head: NotRequired[str]
    interpreter_sha256: NotRequired[str]
    components: NotRequired[dict[str, str]]


class PublicTool(TypedDict):
    name: str
    description: NotRequired[str]
    inputSchema: NotRequired[JSONValue]


class PublicServer(TypedDict):
    tools: list[PublicTool]


class PublicSnapshot(TypedDict):
    servers: dict[str, PublicServer]


class ToolDeclaration(TypedDict, total=False):
    type: str
    name: str


class ToolUse(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, JSONValue]


class ToolResult(TypedDict):
    type: Literal["tool_result"]
    tool_use_id: str
    content: JSONValue | RPCResult


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


ContentPart = ToolUse | ToolResult | TextPart


class Message(TypedDict, total=False):
    role: str
    content: str | list[ContentPart]


class WireBody(TypedDict, total=False):
    model: str
    messages: list[Message]
    tools: list[ToolDeclaration]
    input: list[JSONValue]


class RPCParams(TypedDict, total=False):
    name: str
    arguments: dict[str, JSONValue]


class RPCRequest(TypedDict, total=False):
    id: JSONValue
    method: str
    params: RPCParams


class RPCResult(TypedDict, total=False):
    tools: list[PublicTool]
    isError: bool
    content: list[dict[str, JSONValue]]


class RPCResponse(TypedDict, total=False):
    id: JSONValue
    result: RPCResult


class RPCExchange(TypedDict, total=False):
    server: str
    request: RPCRequest
    response: RPCResponse


HERE = Path(__file__).resolve().parent
PUBLIC_LOADOUT = (
    Path(os.environ.get("BENCH_PUBLIC_LOADOUT", str(HERE.parent / "public_loadout")))
    .expanduser()
    .resolve()
)
TASKS_DIR = (
    Path(os.environ.get("BENCH_TASKS_DIR", str(HERE / "missing-external-tasks")))
    .expanduser()
    .resolve()
)
TASKS_FILE = (
    Path(os.environ.get("BENCH_TASKS_FILE", str(TASKS_DIR / "tasks.json"))).expanduser().resolve()
)
TASK_CACHE = (
    Path(os.environ.get("BENCH_TASK_CACHE", str(TASKS_DIR / "cache"))).expanduser().resolve()
)
EVALUATOR = (
    Path(os.environ.get("BENCH_EVALUATOR", str(HERE / "evaluate.py"))).expanduser().resolve()
)
MODEL = "claude-sonnet-5-5"
CONTEXT = 1000000
LOP_HEAD = "07c324e9f613b378c437936ede1978c9fc0b49db"
HARNESSES = ("claude-code", "codex", "local-operator", "omp", "opencode", "hermes", "dsh")
SCRATCH = (
    Path(
        os.environ.get(
            "BENCH_SCRATCH", str(Path(os.environ.get("TMPDIR", "/var/tmp")) / "harness-cost-bench")
        )
    )
    .expanduser()
    .resolve()
)
PROXY = os.environ.get("BENCH_PROXY", "http://127.0.0.1:8787").rstrip("/")
MCP = os.environ.get("BENCH_MCP_URL", "http://127.0.0.1:8790").rstrip("/")
LITELLM = os.environ.get("BENCH_LITELLM_URL", "http://127.0.0.1:4000/v1").rstrip("/")
TIMEOUT_S = int(os.environ.get("BENCH_TIMEOUT_S", "900"))
PATH = os.environ.get("BENCH_TASK_PATH", os.defpath)
TASK_PY = str(Path(os.environ.get("BENCH_TASK_PY", sys.executable)).expanduser().absolute())
INSTRUCTION_PATHS = {
    "claude-code": ".claude/CLAUDE.md",
    "codex": ".codex/AGENTS.md",
    "local-operator": ".local-operator/system_prompt.md",
    "omp": ".omp/agent/AGENTS.md",
    "opencode": ".config/opencode/AGENTS.md",
    "hermes": ".hermes/SOUL.md",
    "dsh": ".dsh/AGENTS.md",
}
SKILL_ROOTS = (
    ".local-operator/skills",
    ".claude/skills",
    ".codex/skills",
    ".agents/skills",
    ".config/opencode/skills",
    ".omp/agent/skills",
    ".hermes/skills",
    ".dsh/skills",
)
BIN_KEYS = {
    "claude-code": ("BENCH_CLAUDE_BIN", "claude"),
    "codex": ("BENCH_CODEX_BIN", "codex"),
    "omp": ("BENCH_OMP_BIN", "omp"),
    "opencode": ("BENCH_OPENCODE_BIN", "opencode"),
    "hermes": ("BENCH_HERMES_BIN", "hermes"),
}


def resolve_program(key: str, default: str | None = None) -> str:
    """Resolve explicitly configured runtimes or PATH, never a hard-coded account path.

    Shell wrappers that attach to a personal daemon must be overridden with the
    native binary. Runtime hashes in frozen preregistration bind the actual path.
    """
    raw = os.environ.get(key) or (shutil.which(default) if default else None)
    if not raw:
        raise ValueError(f"Set {key}: runtime not found on PATH")
    # Do NOT resolve executable symlinks: invoking a venv's python through
    # its realpath selects the base interpreter and loses the editable package.
    path = Path(raw).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{key}: executable unavailable")
    return str(path)


def sha(path: Path) -> str:
    """Hash bytes rather than trusting version strings or mutable filenames."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_snapshot() -> PublicSnapshot:
    """Load the exact public nine-server surface; reject partial snapshots."""
    d = json.loads((PUBLIC_LOADOUT / "public_mcp_schemas.json").read_text())
    if len(d["servers"]) != 9 or sum(len(s["tools"]) for s in d["servers"].values()) != 493:
        raise ValueError("Expected frozen public snapshot: 9 servers / 493 tools")
    return d


MCP_SERVERS = list(public_snapshot()["servers"])


def expected_skills() -> dict[str, str]:
    """Get the nineteen pinned public skill-body hashes from the fetch manifest."""
    out = {}
    for ln in (PUBLIC_LOADOUT / "public_skills.sha256").read_text().splitlines():
        if not ln or ln.startswith("#"):
            continue
        digest, rel = ln.split(maxsplit=1)
        p = Path(rel.strip())
        if p.name == "SKILL.md":
            out[p.parent.name] = digest
    if len(out) != 19:
        raise ValueError("Public skill manifest must describe exactly 19 skills")
    return out


def public_home_check(home: Path, full: bool, complete: bool = True) -> dict[str, object]:
    """Check exact public bytes and ban imported credential/plugin state.

    Cross-harness instruction paths in a template are expected, but the run
    clone keeps only its own. In particular omp must not read three identical
    copies from .claude, .codex and .omp. A later wire attestation checks that
    the instruction body occurs exactly once in the effective prompt.
    """
    if not home.is_dir():
        raise ValueError("Base home missing; build public/clean templates first")
    instructions = (PUBLIC_LOADOUT / "public_user_instructions.md").read_bytes()
    skills = expected_skills()
    for rel in SKILL_ROOTS:
        root = home / rel
        actual = {p.parent.name: sha(p) for p in root.glob("*/SKILL.md")}
        if full and (complete or rel not in (".hermes/skills", ".dsh/skills")):
            if actual != skills:
                raise ValueError(f"Public skill bytes/count mismatch: {rel}")
        elif not full and actual:
            raise ValueError(f"Clean template contains user skills: {rel}")
    for rel in INSTRUCTION_PATHS.values():
        p = home / rel
        if full and (not p.is_file() or p.read_bytes() != instructions):
            raise ValueError(f"Public instruction bytes differ: {rel}")
        if not full and p.exists():
            raise ValueError(f"Clean template contains instructions: {rel}")
    forbidden = (
        ".env",
        ".hermes/.env",
        ".hermes/auth.json",
        ".dsh/.credentials.yaml",
        ".dsh/.env",
        ".codex/auth.json",
        ".claude/.credentials.json",
        ".claude/plugins",
        ".codex/plugins",
    )
    if any((home / p).exists() for p in forbidden):
        raise ValueError("Template contains credential/private-plugin state")
    return {
        "public_skills": 19 if full else 0,
        "instruction_sha256": sha(PUBLIC_LOADOUT / "public_user_instructions.md") if full else None,
    }


def augment_public_home(home: Path) -> None:
    """Add native skill roots and runtime-only Hermes hooks, without personal config reads.

    BENCH_HERMES_INSTALLS points explicitly at prepared upstream `installs/`
    runtime metadata plus relocatable environments (all+anthropic). Only that
    state is cloned, never owner config/.env/auth/memories/bundled skills.
    No dependency sync/install occurs. The caller must provide the already
    prepared Anthropic extra; otherwise this hook refuses rather than installing.
    """
    public_home_check(home, True, complete=False)
    if shutil.disk_usage(home.parent).free < 15 * 1024**3:
        raise ValueError("Disk below the 15 GiB floor")
    for rel in (".hermes/skills", ".dsh/skills"):
        dest = home / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            clone_path(home / ".agents/skills", dest)
    source = os.environ.get("BENCH_HERMES_INSTALLS")
    if not source:
        raise ValueError("Set BENCH_HERMES_INSTALLS to prepared runtime-only installs directory")
    src = Path(source).expanduser().resolve()
    if src.stat().st_dev != home.stat().st_dev:
        raise ValueError("Runtime clone must share a filesystem; refuse a dependency-tree copy")
    facts = list(src.glob("*/facts.json"))
    if not facts:
        raise ValueError("No prepared Hermes dependency facts")
    for p in facts:
        v = json.loads(p.read_text())["packages"]["venv"]
        if "anthropic" not in v["extras"]:
            raise ValueError(
                "Prepare Hermes anthropic extra outside this runner; no auto-install allowed"
            )
        if "relocatable = true" not in (Path(v["environment"]) / "pyvenv.cfg").read_text():
            raise ValueError("Hermes environment is not relocatable")
    dst = home / ".hermes/installs"
    if dst.exists():
        raise ValueError("Hermes runtime hook already exists; refusing overwrite")
    clone_path(src, dst)
    _hermes_relocate(home)
    public_home_check(home, True)


def build_prompt(task: Task) -> str:
    """OSWorld instruction + a fixed, harness-neutral offline adapter."""
    ev = task["evaluator"]
    result_rel = ev["result"]["path"][len("/home/user/") :]
    if task["_domain"] == "vs_code":
        ctx = (
            "Environment note: there is no GUI. The current working directory is the "
            "user's home directory. VS Code is not running; its user configuration "
            "directory is ./.config/Code/User/ (create files there if needed). "
            "Make the change by editing VS Code's configuration files directly."
        )
    elif task["_domain"] == "multi_apps":
        ctx = (
            "Environment note: there is no GUI. The current working directory is the "
            "user's home directory (/home/user in the task text); the Desktop is ./Desktop. "
            f"The deliverable is ./{result_rel}; create or edit it in place. "
            f"Use the Python interpreter at {TASK_PY} (it has openpyxl, python-docx, "
            "python-pptx and pandas installed; other python3 binaries on this machine do not). "
            "LibreOffice, GIMP, VLC and Thunderbird are not available: do the work with files "
            "and scripts. Spreadsheet formulas will not be recalculated, so store computed values."
        )
    else:
        ctx = (
            "Environment note: there is no GUI. The current working directory is the "
            f"user's home directory and the document is at ./{result_rel}. "
            "Make the change by editing that file in place and saving it under the same "
            f"name and format. Use the Python interpreter at {TASK_PY} (it has openpyxl, "
            "python-docx, python-pptx and pandas installed; other python3 binaries on this machine "
            "do not). LibreOffice is not available, so spreadsheet formulas will not be "
            "recalculated: store computed values in cells, not formulas."
        )
    return f"{task['instruction'].strip()}\n\n{ctx}"


def base_env(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "PATH": PATH,
        "TERM": "dumb",
        "LANG": "en_US.UTF-8",
        "TMPDIR": str(home / "tmp"),
        "USER": os.environ.get("USER", "bench"),
        # Keychain-reaching CLIs: never fall back to the operator's helpers.
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local/share"),
        "XDG_STATE_HOME": str(home / ".local/state"),
        "XDG_CACHE_HOME": str(home / ".cache"),
    }


def _hermes_relocate(home: Path) -> None:
    """Point the cloned Hermes dependency record at THIS run's home.

    WHY: Hermes scopes its dependency environment per data root
    ($HERMES_HOME/installs/<key>/environments/<gen>/venv). A scratch HOME is a
    "borrowing" root, so without a committed environment there every run
    re-syncs a fresh ~370 MB venv (measured: one new generation per launch).
    build_basehomes.sh therefore pre-syncs one relocatable venv
    (pyvenv.cfg relocatable = true) into the base home; `cp -Rc` clones it
    for ~0 bytes. facts.json records that venv by absolute path, and Hermes
    refuses one outside its own install dir ("dependency environment is
    missing or outside this install"), so the prefix is rewritten here.
    """
    import re

    for facts in (home / ".hermes/installs").glob("*/facts.json"):
        text = facts.read_text()
        facts.write_text(
            re.sub(r'"[^"]*/\.hermes/installs/', '"' + str(home) + "/.hermes/installs/", text)
        )


def _hermes_teardown(home: Path, env: dict[str, str]) -> None:
    """Stop the Hermes gateway this run started; it is NOT in our process group.

    WHY: `hermes -z` is a client of a local gateway daemon. With none running
    for $HERMES_HOME it starts `gateway run --idle-exit` as a one-off launchd
    job wrapped in /usr/bin/osascript (macOS Local Network identity), so the
    agent loop and its tool children run there, outside the runner's pgid,
    and idle on for 600 s after the turn (measured 2026-11-03). On timeout
    the turn would even keep spending. So: `gateway stop` for this home, then
    kill the recorded pid (and its osascript parent) if it is still alive,
    then unload the finished launchd label so nothing is left in the domain.
    """
    # configure() may fail before env is returned. No Hermes process can
    # have started then; never invoke vendor cleanup with an empty HOME/env,
    # which could fall back to the account's own gateway/configuration.
    if env.get("HERMES_HOME") != str(home / ".hermes") or env.get("HOME") != str(home):
        return
    state = home / ".hermes/gateway_state.json"
    pid = None
    try:
        rec = json.loads(state.read_text())
        if rec.get("hermes_home") == str(home / ".hermes"):
            pid = int(rec["pid"])
    except Exception:
        pass
    ppid = None
    if pid:
        # A stale pid file must not authorize killing another root's gateway
        # after pid reuse. Inspect environment locally; never print it (it can
        # hold credentials). The per-run HERMES_HOME is the ownership fence.
        identity = subprocess.run(
            ["/bin/ps", "-E", "-p", str(pid), "-o", "command="], capture_output=True, text=True
        ).stdout
        if f"HERMES_HOME={home / '.hermes'}" not in identity:
            pid = None
        else:
            out = subprocess.run(
                ["/bin/ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True
            )
            ppid = int(out.stdout.strip()) if out.stdout.strip().isdigit() else None
    labels = []
    if ppid:
        listed = subprocess.run(
            ["/bin/launchctl", "list"], capture_output=True, text=True
        ).stdout.splitlines()
        labels = [
            ln.split()[2]
            for ln in listed
            if len(ln.split()) == 3
            and ln.split()[0] == str(ppid)
            and ln.split()[2].startswith("com.nousresearch.hermes.runtime.")
        ]
    try:
        subprocess.run(
            [resolve_program("BENCH_HERMES_BIN", "hermes"), "gateway", "stop", "-y"],
            env=env,
            cwd=home,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pass
    for p in (pid, ppid):
        if not p or p == 1:
            continue
        cmd = subprocess.run(
            ["/bin/ps", "-o", "command=", "-p", str(p)], capture_output=True, text=True
        ).stdout
        if "hermes_cli.main" in cmd and "gateway run" in cmd:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for label in labels:
        subprocess.run(["/bin/launchctl", "remove", label], capture_output=True)
    remaining = []
    for p in (pid, ppid):
        if p and p != 1:
            check = subprocess.run(
                ["/bin/ps", "-o", "command=", "-p", str(p)], capture_output=True, text=True
            )
            if "hermes_cli.main" in check.stdout and "gateway run" in check.stdout:
                remaining.append(p)
    (home.parent / "hermes-teardown.json").write_text(
        json.dumps(
            {
                "gateway_pid": pid,
                "launcher_pid": ppid,
                "labels_removed": len(labels),
                "remaining_pids": remaining,
            }
        )
        + "\n"
    )
    if remaining:
        raise RuntimeError(f"Hermes gateway cleanup failed for exact pids {remaining}")


def write_dsh_patch(home: Path, run_id: str, full: bool) -> None:
    """Per-run $DSH_HOME/cordis.patch.yml (DSH_HOME = ~/.dsh under the scratch HOME).

    dsh is configured only through YAML patches (no model/base-URL CLI flags).
    The Anthropic route is the shipped pi-ai adapter (`llm-pi-ai`), whose
    `anthropic` catalog route speaks native Messages; its SDK posts
    <baseURL>/v1/messages, so baseURL is the proxy root. ANTHROPIC_BASE_URL is
    NOT honoured (pi-ai passes the catalog baseUrl explicitly).
    The installed catalog (pi-ai 0.87.1) predates claude-sonnet-5-5; an
    unlisted model falls back to a 262,144 window / 32,768 max_tokens, so the
    entry uses the verified 1,000,000-token window and 128,000 output cap.
    The benchmark requires high/adaptive thinking uniformly; dsh normally
    leaves effort unset (measured: no thinking block), so high is explicit
    here and must be recorded as a departure from its shipped default.
    Full arm: the replayed MCP servers via dsh-mcp-client (streamable-http).
    """
    entry = {
        "id": MODEL,
        "name": "Claude Sonnet 5.5",
        "contextWindow": 1000000,
        "maxTokens": 128000,
        "reasoningEfforts": {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": "max",
        },
        "compat": {"forceAdaptiveThinking": True, "supportsStrictTools": True},
    }
    patch = [
        {
            "id": "llm-pi-ai",
            "config": {
                "providers": {
                    "anthropic": {
                        "apiKeyEnv": "ANTHROPIC_API_KEY",
                        "baseURL": f"{PROXY}/r/{run_id}",
                        "models": [entry],
                    }
                }
            },
        },
        {
            "id": "agent-default-model",
            "config": {"provider": "anthropic", "model": MODEL, "reasoningEffort": "high"},
        },
    ]
    if full:
        # Cordis patches override EXISTING ids; a new id/name row alone is
        # silently skipped. New MCP plugins require an explicit `insert`.
        # Surface connection failures as plugin errors. Cordis still treats
        # these plugins as optional, so this is NOT a global fail-closed
        # gate: validate full-arm schema exposure on the wire after the run.
        patch.append(
            {
                "insert": [
                    {
                        "id": f"mcp-{s}",
                        "name": "@deepseek-ai/dsh-mcp-client",
                        "config": {
                            "serverName": s,
                            "transport": "streamable-http",
                            "url": f"{MCP}/{run_id}/{s}/mcp",
                            "failOnStartupError": True,
                        },
                    }
                    for s in MCP_SERVERS
                ]
            }
        )
    (home / ".dsh").mkdir(exist_ok=True)
    # JSON is a YAML subset: no PyYAML-specific tags reach dsh's loader.
    (home / ".dsh/cordis.patch.yml").write_text(json.dumps(patch, indent=1) + "\n")


def runtime_identity(harness: str) -> RuntimeIdentity:
    """Resolve all runtime inputs without launching a model or reading credentials."""
    if harness == "local-operator":
        binary = resolve_program("BENCH_LOP_PY")
        code = "import inspect,local_operator;print(inspect.getfile(local_operator))"
        p = subprocess.run([binary, "-c", code], capture_output=True, text=True, check=True)
        source = Path(p.stdout.strip()).parent.parent
        head = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if head != LOP_HEAD:
            raise ValueError("BENCH_LOP_PY must import the independently reviewed fixed LOP head")
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            raise ValueError("Reviewed LOP worktree has tracked changes")
        return {"binary": binary, "sha256": sha(Path(binary)), "source_head": head}
    if harness == "dsh":
        binary = resolve_program("BENCH_NODE_BIN", "node")
        raw = os.environ.get("BENCH_DSH_ENTRY")
        if not raw or not Path(raw).expanduser().is_file():
            raise ValueError(
                "Set BENCH_DSH_ENTRY to installed CLI lib/bin.js (not a HOME-relative wrapper)"
            )
        entry = Path(raw).expanduser().resolve()
        package = entry.parent.parent
        components = (
            "node_modules/@deepseek-ai/dsh-llm-pi-ai/lib/index.js",
            "node_modules/@deepseek-ai/dsh-mcp-client/lib/index.js",
            "node_modules/@earendil-works/pi-ai/dist/providers/data/anthropic.json",
        )
        return {
            "binary": binary,
            "entry": str(entry),
            "sha256": sha(entry),
            "interpreter_sha256": sha(Path(binary)),
            "components": {rel: sha(package / rel) for rel in components},
        }
    key, default = BIN_KEYS[harness]
    binary = resolve_program(key, default)
    # A wrapper can attach to a personal service/keychain before the scratch
    # config is read. Refuse implicit script wrappers; explicit env is required.
    if key not in os.environ and Path(binary).read_bytes()[:2] == b"#!":
        raise ValueError(
            f"{default} resolves to a wrapper; set {key} to the actual isolated runtime"
        )
    identity: RuntimeIdentity = {"binary": binary, "sha256": sha(Path(binary))}
    if harness == "hermes":
        if sys.platform != "darwin":
            raise ValueError("Hermes Linux lifecycle cleanup is incomplete; do not dispatch")
        source = Path(os.environ.get("BENCH_HERMES_SOURCE", str(Path(binary).parents[2]))).resolve()
        head = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            raise ValueError("Hermes source contains tracked modifications")
        identity["source_head"] = head
    return identity


def runtime_fingerprint(runtime: RuntimeIdentity) -> str:
    """Ignore machine paths but bind binary, interpreter and source content identities."""
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in runtime.items() if k not in ("binary", "entry")}, sort_keys=True
        ).encode()
    ).hexdigest()


def clone_path(source: Path, destination: Path) -> None:
    """Clone on APFS or require Linux reflinks; never silently duplicate runtimes.

    Linux copy support is an unverified portability path, not runtime wire proof.
    Both filesystems must support the clone operation or the command fails closed.
    """
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError("Clone source/destination must share a filesystem")
    binary = resolve_program("BENCH_CP_BIN", "cp")
    flags = ["-Rc"] if sys.platform == "darwin" else ["--reflink=always", "-a"]
    subprocess.run([binary, *flags, str(source), str(destination)], check=True)


def _prepare(run_id: str, task: PreparedTask, arm: str, harness: str) -> tuple[Path, Path]:
    """Clone only vetted public/clean state and map OSWorld files into scratch HOME."""
    full = arm == "public-full"
    key = "BENCH_PUBLIC_BASEHOME" if full else "BENCH_CLEAN_BASEHOME"
    base = Path(
        os.environ.get(key, str(SCRATCH / ("basehome-public" if full else "basehome")))
    ).expanduser()
    public_home_check(base, full)
    if SCRATCH.is_relative_to(Path.home()):
        raise ValueError(
            "BENCH_SCRATCH must be outside the account HOME to prevent parent-config discovery"
        )
    if shutil.disk_usage(SCRATCH.parent).free < 15 * 1024**3:
        raise ValueError("Disk below the 15 GiB floor")
    root = SCRATCH / "runs" / run_id
    root.mkdir(parents=True, exist_ok=False)  # immutable ids: never merge proxy history
    home = root / "home"
    clone_path(base, home)
    (home / "tmp").mkdir(exist_ok=True)
    # Exactly one global instruction source per harness, not three copies for
    # omp's multi-vendor discovery. Template bytes were checked before clone.
    for name, rel in INSTRUCTION_PATHS.items():
        if name != harness and (home / rel).exists():
            (home / rel).unlink()
    if harness == "hermes":
        facts = list((home / ".hermes/installs").glob("*/facts.json"))
        if not facts:
            raise ValueError(
                "Hermes runtime hook absent; --augment-public needs prepared runtime state"
            )
        _hermes_relocate(home)
        for record in facts:
            v = json.loads(record.read_text())["packages"]["venv"]
            if (
                "anthropic" not in v["extras"]
                or "relocatable = true" not in (Path(v["environment"]) / "pyvenv.cfg").read_text()
            ):
                raise ValueError("Hermes hook lacks prepared relocatable Anthropic dependencies")
        # A scratch profile must not seed the install owner's bundled catalog.
        (home / ".hermes/.no-bundled-skills").touch()
    for c in task["config"]:
        if c["type"] == "download":
            for f in c["parameters"]["files"]:
                target = home / f["path"].removeprefix("/home/user/")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(TASK_CACHE / task["id"] / "input" / Path(f["path"]).name, target)
        elif c["type"] == "execute":
            raw = c["parameters"]["command"]
            text = " ".join(raw) if isinstance(raw, list) else raw
            if "unzip" in text:
                text = text.replace("/home/user", str(home)).removeprefix("/bin/bash -c ")
                subprocess.run(text, shell=True, cwd=home, capture_output=True, check=True)
    if task["_domain"] == "vs_code":
        (home / ".config/Code/User").mkdir(parents=True, exist_ok=True)
    return root, home


def prepare(run_id: str, task: PreparedTask, arm: str, harness: str) -> tuple[Path, Path]:
    """Never delete a prior trial; clean only a new partial clone on failure."""
    root = SCRATCH / "runs" / run_id
    if root.exists():
        raise ValueError("Trial id exists; choose a fresh assigned dev rep")
    try:
        return _prepare(run_id, task, arm, harness)
    except BaseException:
        if (root / "home").exists():
            shutil.rmtree(root / "home")
        raise


def configure(
    harness: str,
    run_id: str,
    prompt: str,
    home: Path,
    full: bool,
    runtime: RuntimeIdentity,
    bridge_port: int = 0,
) -> tuple[list[str], dict[str, str]]:
    """Write native public configs only; never resolve credentials in config-only mode."""
    global MCP_SERVERS
    MCP_SERVERS = tuple(public_snapshot()["servers"])
    env = base_env(home)
    binary = runtime["binary"]
    url = f"{PROXY}/r/{run_id}"
    servers = (
        {s: {"type": "http", "url": f"{MCP}/{run_id}/{s}/mcp"} for s in MCP_SERVERS} if full else {}
    )
    if harness == "claude-code":
        (home / ".claude.json").write_text(
            json.dumps({"hasCompletedOnboarding": True, "mcpServers": servers})
        )
        env.update(
            ANTHROPIC_BASE_URL=url,
            ANTHROPIC_API_KEY="bench-dummy",
            ENABLE_TOOL_SEARCH="true",
            DISABLE_TELEMETRY="1",
            DISABLE_AUTOUPDATER="1",
            DISABLE_ERROR_REPORTING="1",
        )
        return [
            binary,
            "-p",
            prompt,
            "--model",
            MODEL,
            "--effort",
            "high",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            *([] if full else ["--strict-mcp-config"]),
        ], env
    if harness == "codex":
        (home / ".codex").mkdir(exist_ok=True)
        cfg = home / ".codex/config.toml"
        with cfg.open("a") as f:
            for name, spec in servers.items():
                f.write(
                    f'\n[mcp_servers.{name.replace("-", "_")}]\nurl = {json.dumps(spec["url"])}\n'
                )
        env.update(CODEX_HOME=str(home / ".codex"))
        catalog = Path(
            os.environ.get("BENCH_CODEX_CATALOG", str(HERE / "codex-model-catalog.json"))
        ).resolve()
        # Primary Codex is ALWAYS catalog+tool_search bridge, never fallback.
        provider = (
            '{name="bench",base_url="http://127.0.0.1:'
            + str(bridge_port)
            + '/v1",env_key="BENCH_LITELLM_KEY",wire_api="responses"}'
        )
        return [
            binary,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "--dangerously-bypass-approvals-and-sandbox",
            "-m",
            MODEL,
            "-c",
            "model_provider=bench",
            "-c",
            f"model_providers.bench={provider}",
            "-c",
            f"model_catalog_json={json.dumps(str(catalog))}",
            prompt,
        ], env
    if harness == "local-operator":
        d = home / ".local-operator"
        d.mkdir(exist_ok=True)
        (d / "mcp.json").write_text(json.dumps({"mcpServers": servers}))
        env.update(
            LOP_BENCH_PYTHON=binary,
            LOP_BENCH_SOURCE_ROOT=os.environ.get("LOP_BENCH_SOURCE_ROOT", ""),
            LOCAL_OPERATOR_CONFIG_DIR=str(d),
            BENCH_ANTHROPIC_URL=url,
            ANTHROPIC_API_KEY="bench-dummy",
            LOCAL_OPERATOR_NO_NOTIFICATIONS="1",
            LOCAL_OPERATOR_ALLOW_NESTED_SESSION="1",
        )
        return [
            binary,
            str(HERE / "lop_launch.py"),
            "exec",
            "--hosting",
            "anthropic",
            "--model",
            MODEL,
            "--effort",
            "high",
            "--yolo",
            "--json",
            prompt,
        ], env
    if harness == "omp":
        d = home / ".omp/agent"
        d.mkdir(parents=True, exist_ok=True)
        (d / "mcp.json").write_text(json.dumps({"mcpServers": servers}))
        env.update(
            ANTHROPIC_BASE_URL=url, ANTHROPIC_API_KEY="bench-dummy", PI_CODING_AGENT_DIR=str(d)
        )
        return [
            binary,
            "-p",
            "--mode",
            "json",
            "--model",
            f"anthropic/{MODEL}",
            "--thinking",
            "high",
            "--auto-approve",
            "--no-session",
            "--allow-home",
            "--cwd",
            str(home),
            prompt,
        ], env
    if harness == "opencode":
        cfg = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "provider": {
                "anthropic": {"options": {"baseURL": f"{url}/v1", "apiKey": "bench-dummy"}}
            },
            "mcp": {
                s: {"type": "remote", "url": v["url"], "enabled": True} for s, v in servers.items()
            },
        }
        env.update(OPENCODE_CONFIG_CONTENT=json.dumps(cfg), OPENCODE_DISABLE_AUTOUPDATE="1")
        return [
            binary,
            "run",
            "--standalone",
            "--auto",
            "--format",
            "json",
            "-m",
            f"anthropic/{MODEL}#high",
            prompt,
        ], env
    if harness == "hermes":
        d = home / ".hermes"
        d.mkdir(exist_ok=True)
        cfg = {
            "model": {"default": MODEL, "provider": "bench", "context_length": CONTEXT},
            "providers": {
                "bench": {
                    "api": url,
                    "key_env": "ANTHROPIC_API_KEY",
                    "transport": "anthropic_messages",
                }
            },
            "mcp_servers": {s: {"url": v["url"]} for s, v in servers.items()},
        }
        # JSON is also YAML; no dependency needed. No MEMORY.md import, no
        # private config, and no inherited 44-skill catalog in the public arm.
        (d / "config.yaml").write_text(json.dumps(cfg))
        env.update(
            HERMES_HOME=str(d), ANTHROPIC_API_KEY="bench-dummy", HERMES_DISABLE_LAZY_INSTALLS="1"
        )
        return [
            binary,
            "-z",
            prompt,
            "-m",
            MODEL,
            "--provider",
            "bench",
            "--reasoning",
            "high",
            "--usage-file",
            str(home.parent / "hermes-usage.json"),
        ], env
    if harness == "dsh":
        if "entry" not in runtime:
            raise ValueError("DSH runtime identity lacks its installed CLI entry")
        write_dsh_patch(home, run_id, full)
        env.update(
            ANTHROPIC_API_KEY="bench-dummy",
            DSH_PERMISSION_MODE="danger-full-access",
            DSH_TELEMETRY_DISABLED="1",
        )
        return [binary, runtime["entry"], "--profile", "headless", "--json", prompt], env
    raise ValueError("Unsupported harness")


def verify_metadata() -> dict[str, object]:
    """Fail closed on model/catalog metadata before any request can cost money."""
    p = Path(os.environ.get("BENCH_MODEL_METADATA", str(HERE / "model_metadata.json")))
    d = json.loads(p.read_text())
    if d.get("id") != "anthropic/claude-sonnet-5.5" or d.get("context_length") != CONTEXT:
        raise ValueError("Unverified Sonnet 5.5 context metadata")
    if d.get("top_provider", {}).get("max_completion_tokens") != 128000:
        raise ValueError("Unverified Sonnet 5.5 output cap")
    catalog = Path(os.environ.get("BENCH_CODEX_CATALOG", str(HERE / "codex-model-catalog.json")))
    m = next(m for m in json.loads(catalog.read_text())["models"] if m["slug"] == MODEL)
    if (
        m.get("context_window"),
        m.get("max_context_window"),
        m.get("supports_search_tool"),
        m.get("default_reasoning_level"),
    ) != (CONTEXT, CONTEXT, True, "high"):
        raise ValueError(
            "Primary Codex requires best catalog: 1M window, high effort and tool search"
        )
    return {"model_metadata_sha256": sha(p), "codex_catalog_sha256": sha(catalog)}


def verify_replay(run_id: str) -> dict[str, int]:
    """Fetch every tools/list and compare exact definitions, not only total count.

    The legacy private replay on :8790 intentionally fails this check. This
    runner never replaces it: point BENCH_MCP_URL at an owned public replay.
    MCP read-only protocol probes make no model calls and execute no tools.
    """
    expected = public_snapshot()["servers"]
    counts = {}
    for name, rec in expected.items():
        url = f"{MCP}/{run_id}/{name}/mcp"
        data = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        ).encode()
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.load(resp)
        actual = result.get("result", {}).get("tools")
        if actual != rec["tools"]:
            raise ValueError(f"Replay schema mismatch or missing server: {name}")
        counts[name] = len(actual)
    return counts


def integration_inputs() -> dict[str, str]:
    """One shared input digest binds captures and freeze to the same helpers/catalog."""
    out = {
        name: sha(HERE / name)
        for name in ("codex_toolsearch_bridge.py", "lop_launch.py", "lop_shim/sitecustomize.py")
    }
    out["tasks.json"] = sha(TASKS_FILE)
    out.update(
        {
            name: sha(PUBLIC_LOADOUT / name)
            for name in (
                "public_skills.sha256",
                "public_mcp_schemas.json",
                "public_user_instructions.md",
            )
        }
    )
    out["task_manifest"] = sha(TASKS_FILE.parent / "task-manifest.json")
    out["evaluator"] = sha(EVALUATOR)
    out["codex_catalog"] = sha(
        Path(os.environ.get("BENCH_CODEX_CATALOG", str(HERE / "codex-model-catalog.json")))
    )
    out["model_metadata"] = sha(
        Path(os.environ.get("BENCH_MODEL_METADATA", str(HERE / "model_metadata.json")))
    )
    return out


def validate_deferred_exchanges(
    bodies: Sequence[WireBody],
    rpc: Sequence[RPCExchange],
    responses: Sequence[WireBody],
    codex: bool,
) -> None:
    """Validate captured exchanges, not asserted booleans/server-name summaries.

    Messages artifacts must contain the real harness's accumulated conversation:
    a per-server discovery use/result AND a separate per-server call/use/result.
    The paired raw MCP JSON-RPC capture must show byte-identical tools/list and
    an actual known tools/call returning the replay refusal. Codex additionally
    needs native Responses tool_search/tool_search_output plus the translated
    Messages tool_search declaration. These traces are zero-cost capture-lane
    inputs, never synthesized by this runner or inferred from a config dump.
    """
    expected = public_snapshot()["servers"]
    uses, results = {}, {}
    for b in bodies:
        for message in b.get("messages", []):
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if "type" not in part:
                    continue
                if part["type"] == "tool_use":
                    uses[part["id"]] = part
                elif part["type"] == "tool_result":
                    results[part["tool_use_id"]] = part

    def contains_result(value, expected_result, depth=0):
        if value == expected_result or value == expected_result.get("content"):
            return True
        if depth > 8:
            return False
        if isinstance(value, str):
            plain = "".join(
                p.get("text", "") for p in expected_result.get("content", []) if isinstance(p, dict)
            )
            if value == plain:
                return True
            try:
                return contains_result(json.loads(value), expected_result, depth + 1)
            except (ValueError, TypeError):
                return False
        if isinstance(value, dict):
            return any(contains_result(v, expected_result, depth + 1) for v in value.values())
        if isinstance(value, list):
            return any(contains_result(v, expected_result, depth + 1) for v in value)
        return False

    def matches_invocation(use, server, tool, arguments):
        variants = (server, server.replace("-", "_"))
        aliases = {f"mcp__{v}__{tool}" for v in variants} | {f"mcp_{v}_{tool}" for v in variants}
        aliases |= {f"{v}.{tool}" for v in variants}
        name, args = use.get("name", ""), use.get("input", {})
        if name.removeprefix("functions.") in aliases:
            return args == arguments
        # Accept only known structured dispatchers. Merely mentioning a server
        # in terminal/echo/eval text is never evidence of a harness MCP call.
        if name not in ("tool_call", "mcp_call", "mcp", "call") or not isinstance(args, dict):
            return False
        candidates = args.get("calls", [args])
        if not isinstance(candidates, list):
            return False
        for c in candidates:
            if not isinstance(c, dict):
                continue
            target = c.get("name", "")
            addressed = target in aliases or (c.get("server") == server and c.get("tool") == tool)
            if addressed and c.get("arguments", {}) == arguments:
                return True
        return False

    for server, rec in expected.items():
        variants = (server, server.replace("-", "_"))
        discovered = False
        for ident, use in uses.items():
            if ident not in results:
                continue
            name = use.get("name", "").lower()
            args, result = json.dumps(use.get("input", {})), json.dumps(results[ident])
            is_discovery = any(x in name for x in ("search", "describe")) or name in (
                "read",
                "read_file",
            )
            if (
                is_discovery
                and any(v in name or v in args for v in variants)
                and any(v in result for v in variants)
            ):
                discovered = True
        listed, correlated_call = False, False
        known = {t["name"] for t in rec["tools"]}
        for exchange in rpc:
            if exchange.get("server") != server:
                continue
            request, response = exchange.get("request", {}), exchange.get("response", {})
            if request.get("id") is None or request.get("id") != response.get("id"):
                continue
            if (
                request.get("method") == "tools/list"
                and response.get("result", {}).get("tools") == rec["tools"]
            ):
                listed = True
            if request.get("method") != "tools/call":
                continue
            params, result = request.get("params", {}), response.get("result", {})
            if (
                params.get("name") not in known
                or result.get("isError") is not True
                or not result.get("content")
            ):
                continue
            for ident, use in uses.items():
                if ident in results and matches_invocation(
                    use, server, params.get("name"), params.get("arguments", {})
                ):
                    correlated_call |= contains_result(results[ident].get("content"), result)
        if not discovered or not listed or not correlated_call:
            raise ValueError(
                f"Deferred capture lacks correlated discovery/call/arguments/result: {server}"
            )
    if codex:
        search_declared = any(
            t.get("type") == "tool_search" for b in responses for t in b.get("tools", [])
        )
        outputs = [
            i
            for b in responses
            for i in b.get("input", [])
            if isinstance(i, dict) and i.get("type") == "tool_search_output"
        ]
        loaded = json.dumps(outputs, sort_keys=True)
        translated = any(t.get("name") == "tool_search" for b in bodies for t in b.get("tools", []))
        if (
            not search_declared
            or not translated
            or not all(s in loaded or s.replace("-", "_") in loaded for s in expected)
        ):
            raise ValueError(
                "Codex native Responses search/output -> Messages bridge exchanges missing"
            )


def attest(harness: str, arm: str, runtime: RuntimeIdentity) -> dict[str, object]:
    """Require hashed, runtime-bound zero-cost captures; no config-only attestation.

    This manifest is supplied by the independent prereg/capture lane. Lazy tool
    modes require all-server search/call roundtrips, not an optimistic first
    request tool count. Eager mode requires exact public MCP tool count too.
    Nothing here manufactures a zero-cost capture or infers one from a dump.
    """
    root = Path(os.environ.get("BENCH_WIRE_ATTESTATIONS", str(HERE / "wire-attestations")))
    record = json.loads((root / f"{harness}.{arm}.json").read_text())
    full = arm == "public-full"
    if record.get("input_sha256") != integration_inputs():
        raise ValueError("Wire evidence helper/catalog inputs differ from this runner")
    if record.get("runner_sha256") != sha(Path(__file__)):
        raise ValueError("Wire evidence predates this adapter source")
    expected = {k: len(v["tools"]) for k, v in public_snapshot()["servers"].items()} if full else {}
    if record.get("runtime_fingerprint") != runtime_fingerprint(runtime):
        raise ValueError("Wire evidence belongs to a different executable")
    if record.get("public_schema_sha256") != sha(PUBLIC_LOADOUT / "public_mcp_schemas.json"):
        raise ValueError("Wire evidence belongs to another MCP loadout")
    if record.get("context_window") != CONTEXT or record.get("main_effort") != "high":
        raise ValueError("Wire metadata attestation is incomplete")
    if (
        record.get("model") != MODEL
        or record.get("thinking_type") != "adaptive"
        or record.get("main_native_tools", 0) < 1
    ):
        raise ValueError("Missing model/thinking/native-tool evidence")
    if record.get("instruction_occurrences") != int(full):
        raise ValueError("Instruction duplication or missing instructions on wire")
    if record.get("instruction_sha256") != (
        sha(PUBLIC_LOADOUT / "public_user_instructions.md") if full else None
    ):
        raise ValueError("Wire instructions do not match identical public bytes")
    if record.get("skill_count") != (19 if full else 0):
        raise ValueError("Wire skill catalog count differs")
    if record.get("mcp_server_counts") != expected:
        raise ValueError("Wire evidence omits MCP servers/tools")
    if record.get("mcp_mode") == "deferred" and set(record.get("mcp_roundtrip_servers", [])) != set(
        expected
    ):
        raise ValueError("Deferred schemas have no complete tool-search/call roundtrip evidence")
    if full and record.get("mcp_mode") not in ("deferred", "eager"):
        raise ValueError("Unknown MCP exposure mode")
    if harness == "codex" and full and not record.get("codex_bridge_roundtrip"):
        raise ValueError("Codex search bridge has no roundtrip evidence")
    if harness == "local-operator" and record.get("source_head") != LOP_HEAD:
        raise ValueError("LOP capture predates reviewed fix")
    if not record.get("captures"):
        raise ValueError("Wire attestation has no capture artifacts")
    observed_main, message_bodies, rpc_exchanges, response_bodies = [], [], [], []
    for item in record["captures"]:
        p = (root / item["path"]).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p) != item["sha256"]:
            raise ValueError("Wire capture artifact missing, escaped root or changed")
        if item.get("kind") == "messages":
            data = json.loads(p.read_text())
            bodies = data if isinstance(data, list) else [data]
            message_bodies.extend(b.get("body", b) for b in bodies)
            observed_main.extend(
                b.get("body", b)
                for b in bodies
                if b.get("body", b).get("tools") and b.get("body", b).get("model") == MODEL
            )
        elif item.get("kind") in ("mcp-rpc", "responses"):
            data = json.loads(p.read_text())
            values = data if isinstance(data, list) else [data]
            if item["kind"] == "mcp-rpc":
                rpc_exchanges.extend(values)
            else:
                response_bodies.extend(v.get("body", v) for v in values)
    if not observed_main:
        raise ValueError("No actual main Messages body in capture artifacts")
    first = observed_main[0]
    if (
        first.get("thinking", {}).get("type") != "adaptive"
        or first.get("output_config", {}).get("effort") != "high"
    ):
        raise ValueError("Actual captured main wire is not adaptive/high")
    texts = []
    for content in [first.get("system", "")] + [
        m.get("content", "") for m in first.get("messages", [])
    ]:
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(x.get("text", "") for x in content if isinstance(x, dict))
    # Codex may carry global instructions in a developer/user message rather
    # than system. Count exact instruction bytes across prompt text, not the
    # JSON encoding and not only the first system block.
    instruction = (PUBLIC_LOADOUT / "public_user_instructions.md").read_text()
    if sum(text.count(instruction.strip()) for text in texts) != int(full):
        raise ValueError("Actual captured prompt instruction count differs (omp duplication guard)")
    if full and record.get("mcp_mode") == "deferred":
        validate_deferred_exchanges(
            message_bodies, rpc_exchanges, response_bodies, harness == "codex"
        )
    if full and record.get("mcp_mode") == "eager":
        counts = collections.Counter()
        for tool in first["tools"]:
            name = tool.get("name", "")
            for server in expected:
                if name.startswith(f"mcp__{server}__"):
                    counts[server] += 1
        if dict(counts) != expected:
            raise ValueError("Actual eager wire omits public MCP server schemas")
    return {
        "wire_attestation_sha256": sha(root / f"{harness}.{arm}.json"),
        "attested_main_tool_count": len(first["tools"]),
    }


def frozen_prereg(harness: str, arm: str, runtime: RuntimeIdentity) -> dict[str, object]:
    """An explicit frozen, hash-bound prereg is required; never freezes itself."""
    raw = os.environ.get("BENCH_PREREG_FILE")
    if not raw:
        raise ValueError("No frozen preregistration: paid execution prohibited")
    d = json.loads(Path(raw).read_text())
    if d.get("frozen") is not True or d.get("model") != MODEL or d.get("context_window") != CONTEXT:
        raise ValueError("Preregistration is not frozen for this model/context")
    if d.get("public_schema_sha256") != sha(PUBLIC_LOADOUT / "public_mcp_schemas.json"):
        raise ValueError("Preregistered schema changed")
    if d.get("instruction_sha256") != sha(PUBLIC_LOADOUT / "public_user_instructions.md"):
        raise ValueError("Preregistered instructions changed")
    if d.get("runtime_fingerprints", {}).get(harness) != runtime_fingerprint(
        runtime
    ) or arm not in d.get("arms", []):
        raise ValueError("Runtime/arm is not preregistered")
    if d.get("runner_sha256") != sha(Path(__file__)) or d.get("lop_head") != LOP_HEAD:
        raise ValueError("Runner or reviewed LOP head differs from preregistration")
    inputs = integration_inputs()
    if d.get("input_sha256") != inputs:
        raise ValueError("Preregistered task/loadout/bridge/shim/model inputs differ")
    return {"prereg_sha256": sha(Path(raw))}


def radient_token(lop_python: str) -> str:
    """Resolve the operator's credential into memory, never argv/config/artifacts."""
    if os.environ.get("RADIENT_API_KEY"):
        return os.environ["RADIENT_API_KEY"]
    code = (
        "import asyncio,sys\nfrom local_operator.classification.vendors import auth_store_api_key\n"
        "t=asyncio.run(auth_store_api_key(None,'radient'))\nsys.stdout.write(t or '')\n"
    )
    proc = subprocess.run([lop_python, "-c", code], capture_output=True, text=True, timeout=60)
    if proc.returncode or not proc.stdout.strip():
        raise ValueError("Radient login unavailable; sign in outside the runner")
    return proc.stdout.strip()


def money(
    rows: Sequence[Mapping[str, object]], stderr: str, classification: bool
) -> dict[str, float | int | bool | None]:
    """Account for every recorded Messages attempt, independent of HTTP success.

    An error can be billed; a missing/invalid bill is unknown, never zero. Keep
    the measured partial sum separate from nullable complete totals. Discovery
    and count-token routes are not Messages attempts, but their presence alone
    cannot establish a measured model run. Retries remain separate ledger rows;
    do not deduplicate by status, timestamp or request contents. This function
    cannot detect attempts missing entirely from its caller's metering ledger.
    """
    from decimal import InvalidOperation

    messages = [row for row in rows if str(row.get("path", "")).rstrip("/").endswith("v1/messages")]
    measured = Decimal(0)
    unknown = 0
    for row in messages:
        usage = row.get("usage")
        value = usage.get("cost") if isinstance(usage, dict) else None
        try:
            if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
                raise ValueError("Invalid billed cost")
            amount = Decimal(str(value))
            if not amount.is_finite() or amount < 0:
                raise ValueError("Invalid billed cost")
        except (InvalidOperation, ValueError, TypeError):
            unknown += 1
            continue
        measured += amount
    model_complete = bool(messages) and unknown == 0
    charges = re.findall(r"classification:.*?cost=\$([0-9]+(?:\.[0-9]+)?)", stderr)
    radient = (
        sum((Decimal(v) for v in charges), Decimal(0)) if charges or not classification else None
    )
    complete = model_complete and radient is not None
    return {
        "model_cost_usd": float(measured) if model_complete else None,
        "measured_model_cost_usd": float(measured),
        "radient_cost_usd": float(radient) if radient is not None else None,
        "total_cost_usd": float(measured + radient) if complete and radient is not None else None,
        "message_attempts": len(messages),
        "unknown_billing_attempts": unknown,
        "non_200_message_attempts": sum(row.get("status") != 200 for row in messages),
        "non_model_rows": len(rows) - len(messages),
        "classification_calls_charged": len(charges),
        "cost_complete": complete,
    }


def stop_group(proc: subprocess.Popen[str] | subprocess.Popen[bytes] | None) -> None:
    """Reap only an owned exact process group, once, including orphaned children."""
    if proc is None or getattr(proc, "_bench_group_reaped", False):
        return
    setattr(proc, "_bench_group_reaped", True)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit while tool children ignore SIGTERM. Always reap
    # the known group rather than treating leader completion as tree exit.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def execute(
    harness: str,
    run_id: str,
    arm: str,
    rep: int,
    task: Task,
    runtime: RuntimeIdentity,
    preflight: dict[str, object],
) -> dict[str, object]:
    """Run only after freeze/evidence gates; keep credentials in memory and env.

    Codex's shared translator uses current_run: hold an exclusive file lock so
    v2 runs cannot race each other (the legacy runner does not honor this lock,
    therefore coordinator must prohibit concurrent legacy Codex executions).
    All subprocess startup errors still pass through explicit cleanup.
    """
    # The main-model meter cannot bound a separate hosted classifier account.
    # Do not disable classification or accept a boolean attestation in place of
    # a working reservation path. This gate must be implemented and validated
    # before classifier-enabled paid LOP collection becomes available.
    if harness == "local-operator":
        raise ValueError(
            "Paid LOP pending validated classifier reservation/accounting; do not disable Radient"
        )
    import fcntl

    root = home = None
    proc = bridge = None
    hermes_cleaned = False
    token = ""
    lock = None
    lock_acquired = False
    env = {}
    old_current = None
    current_file = Path(os.environ.get("BENCH_CURRENT_RUN_FILE", str(SCRATCH / "current_run")))
    stdout = stderr = ""
    timed_out = False
    try:
        root, home = prepare(run_id, task, arm, harness)
        port = 0
        if harness == "codex":
            lock = (SCRATCH / "codex-v2.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
            old_current = current_file.read_bytes() if current_file.exists() else None
            current_file.write_text(run_id)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
        argv, env = configure(
            harness, run_id, build_prompt(task), home, arm == "public-full", runtime, port
        )
        if harness == "local-operator":
            token = radient_token(runtime["binary"])
            env["RADIENT_API_KEY"] = token
        if harness == "codex":
            key = os.environ.get("BENCH_LITELLM_KEY")
            if not key:
                raise ValueError(
                    "Provide BENCH_LITELLM_KEY in environment; no config credential reads"
                )
            env["BENCH_LITELLM_KEY"] = key
            token = key
            bridge_py = resolve_program("BENCH_BRIDGE_PY", "python3")
            bridge_env = base_env(home) | {
                "BRIDGE_UPSTREAM": LITELLM.removesuffix("/v1"),
                "BRIDGE_LOG": str(root / "bridge-usage.jsonl"),
            }
            bridge = subprocess.Popen(
                [bridge_py, str(HERE / "codex_toolsearch_bridge.py"), str(port)],
                env=bridge_env,
                cwd=HERE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            for _ in range(100):
                if bridge.poll() is not None:
                    raise ValueError("Codex bridge exited during startup")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise ValueError("Codex bridge did not become reachable")
        started = time.time()
        proc = subprocess.Popen(
            argv,
            cwd=home,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_group(proc)
            stdout, stderr = proc.communicate()
        finally:
            stop_group(proc)
            if harness == "hermes":
                _hermes_teardown(home, env)
                hermes_cleaned = True
            stop_group(bridge)
        # Sanitize before persistence, not after credential-bearing logs have
        # already hit disk. Do not write the complete environment or argv.
        for secret in (token, env.get("RADIENT_API_KEY", ""), env.get("BENCH_LITELLM_KEY", "")):
            if secret:
                stdout, stderr = stdout.replace(secret, "[credential removed]"), stderr.replace(
                    secret, "[credential removed]"
                )
        (root / "stdout.log").write_text(stdout)
        (root / "stderr.log").write_text(stderr)
        log = Path(os.environ.get("BENCH_REQUEST_LOG", str(SCRATCH / "logs/requests.jsonl")))
        rows = [r for r in (json.loads(line) for line in log.open()) if r.get("run_id") == run_id]
        main_rows = [r for r in rows if r.get("n_tools", 0) > 0 and r.get("usage")]
        if not main_rows:
            raise ValueError("Run lost all native tools on the actual metered wire")
        if arm == "public-full" and harness == "dsh":
            expected_count = preflight["attested_main_tool_count"]
            first_main = min(main_rows, key=lambda row: row["ts"])
            if first_main["n_tools"] != expected_count:
                raise ValueError("Full dsh main tool count differs from verified wire capture")
            expected = {name: len(v["tools"]) for name, v in public_snapshot()["servers"].items()}
            bodies = Path(os.environ.get("BENCH_BODY_DIR", str(SCRATCH / "logs/bodies")))
            verified = False
            for suffix in ("first", "last"):
                path = bodies / f"{run_id}.{suffix}.json"
                if not path.is_file():
                    continue
                counts = collections.Counter()
                for tool in json.loads(path.read_text()).get("tools", []):
                    for server in expected:
                        if tool.get("name", "").startswith(f"mcp__{server}__"):
                            counts[server] += 1
                verified = verified or dict(counts) == expected
            if not verified:
                raise ValueError(
                    "No actual retained dsh body proves all nine public MCP partitions"
                )
        scored = subprocess.run(
            [resolve_program("BENCH_EVAL_PY"), str(EVALUATOR), task["id"], str(home)],
            capture_output=True,
            text=True,
            check=True,
            env={
                **os.environ,
                "BENCH_TASKS_FILE": str(TASKS_FILE),
                "BENCH_TASK_CACHE": str(TASK_CACHE),
            },
        )
        score = json.loads(scored.stdout.strip().splitlines()[-1])
        rec = {
            "run_id": run_id,
            "harness": harness,
            "adapter_variant": "codex-best" if harness == "codex" else harness,
            "arm": arm,
            "rep": rep,
            "development_only": True,
            "task_id": task["id"],
            **score,
            "exit_code": proc.returncode,
            "timed_out": timed_out,
            "wall_s": round(time.time() - started, 2),
            "lop_head": LOP_HEAD if harness == "local-operator" else None,
            **money(rows, stderr, harness == "local-operator"),
            **preflight,
        }
        result = home / task["evaluator"]["result"]["path"].removeprefix("/home/user/")
        if result.is_file():
            (root / "kept").mkdir(exist_ok=True)
            shutil.copy(result, root / "kept" / result.name)
        logs = SCRATCH / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / "runs_v2dev.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    finally:
        stop_group(proc)
        stop_group(bridge)
        if not hermes_cleaned and home and harness == "hermes" and (home / ".hermes").exists():
            _hermes_teardown(home, env)
        if home and home.exists():
            shutil.rmtree(home)
        if lock:
            if lock_acquired:
                if old_current is None:
                    current_file.unlink(missing_ok=True)
                else:
                    current_file.write_bytes(old_current)
            lock.close()


def validate_external_task(task: TaskIdentity) -> None:
    """Reject unfetched/tampered external assets through the pinned evaluator.

    The evaluator preflight only imports the selected upstream metric and checks
    pins/hashes; it neither scores a task nor calls a harness/model.
    """
    proc = subprocess.run(
        [resolve_program("BENCH_EVAL_PY"), str(EVALUATOR), task["id"], str(SCRATCH), "--preflight"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "BENCH_TASKS_FILE": str(TASKS_FILE),
            "BENCH_TASK_CACHE": str(TASK_CACHE),
        },
    )
    if proc.returncode:
        raise ValueError(
            "External task preflight failed; check pin, fetched hashes and metric dependencies"
        )


def main() -> None:
    """Separate configuration inspection, read-only preflight and paid execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", nargs="?", choices=HARNESSES)
    parser.add_argument("task", nargs="?")
    parser.add_argument("rep", nargs="?", type=int)
    parser.add_argument("arm", nargs="?", choices=("clean", "public-full"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config-only", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--augment-public", type=Path)
    args = parser.parse_args()
    if args.augment_public:
        augment_public_home(args.augment_public.resolve())
        print(json.dumps({"public_hook": "verified"}))
        return
    if not args.harness or not args.task or args.rep not in range(40, 50) or not args.arm:
        parser.error("harness, task, dev rep 40–49 and explicit clean/public-full are required")
    tasks = json.loads((TASKS_FILE).read_text())
    matches = [t for t in tasks if t["id"].startswith(args.task)]
    if len(matches) != 1:
        parser.error("Task prefix is missing or ambiguous")
    task = matches[0]
    validate_external_task(task)
    rid = f"v2dev__{args.arm}__{args.harness}__{task['id'][:8]}__r{args.rep}"
    runtime = runtime_identity(args.harness)
    if args.config_only:
        root, home = prepare(rid, task, args.arm, args.harness)
        try:
            argv, env = configure(
                args.harness,
                rid,
                build_prompt(task),
                home,
                args.arm == "public-full",
                runtime,
                8797,
            )
            # No credential resolution; persist neither environment nor prompt.
            result = {
                "run_id": rid,
                "adapter": "codex-best" if args.harness == "codex" else args.harness,
                "config_only": True,
                "credentials_resolved": False,
                "model_calls": 0,
                "public_schema_sha256": sha(PUBLIC_LOADOUT / "public_mcp_schemas.json"),
            }
            print(json.dumps(result))
        finally:
            shutil.rmtree(home)
        return
    full = args.arm == "public-full"
    base_key = "BENCH_PUBLIC_BASEHOME" if full else "BENCH_CLEAN_BASEHOME"
    base = Path(
        os.environ.get(base_key, str(SCRATCH / ("basehome-public" if full else "basehome")))
    ).expanduser()
    checks = public_home_check(base, full)
    checks.update(verify_metadata())
    if args.arm == "public-full":
        counts = verify_replay(rid)
        checks["verified_mcp_tools"] = sum(counts.values())
    checks.update(attest(args.harness, args.arm, runtime))
    if args.preflight_only:
        print(json.dumps({"run_id": rid, "preflight": checks, "model_calls": 0}))
        return
    checks.update(frozen_prereg(args.harness, args.arm, runtime))
    print(json.dumps(execute(args.harness, rid, args.arm, args.rep, task, runtime, checks)))


if __name__ == "__main__":
    main()
