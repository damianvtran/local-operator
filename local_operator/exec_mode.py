"""Headless ``exec`` mode — run one task and exit.

README contract: ``local-operator exec "<task>"`` executes the task and
exits 0 on success, non-zero on error. Two execution shapes:

- foreground: build a session via the shared factory, run the prompt through
  the print renderer, return 0/1;
- ``--background``: spawn ``python -m local_operator.exec_worker`` detached
  (``start_new_session=True``) with stdout/stderr redirected to a timestamped
  log under ``~/.local-operator/logs/``, record the job in the lightweight
  ``exec-jobs.json`` ledger, print the job id + log path, and return 0
  immediately. The worker is what actually runs the task.

No engine imports at module level — the session factory and renderer are
imported inside the foreground path so ``import local_operator.exec_mode``
stays cheap for the CLI's parser tests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

#: Root for detached-exec logs and the jobs ledger (the legacy config dir).
LOGS_DIR = Path.home() / ".local-operator" / "logs"

#: Ledger of spawned background jobs (lightweight; the harness AsyncJobManager
#: is in-process and cannot track a detached OS process across CLI runs).
JOBS_FILE = "exec-jobs.json"

#: Returns a session or an awaitable session. Tests monkeypatch this to inject
#: scripted fakes without touching the real engine.
SessionFactory = Callable[[], Any]
default_session_factory: SessionFactory | None = None


@dataclass
class ExecArgs:
    """CLI-facing options for one ``exec`` invocation.

    Field names mirror the parser dests; ``agent_id`` is the additive
    ``exec --agent-id`` selector (by id instead of by name).
    """

    background: bool = False
    json_mode: bool = False
    agent_name: str | None = None
    agent_id: str | None = None
    yolo: bool = False
    hosting: str | None = None
    model: str | None = None


def slugify(command: str, max_length: int = 40) -> str:
    """Log-name slug: first ``max_length`` chars, non-alphanumerics -> '-'.

    Spec-literal mapping (no trailing-dash strip) so log names are
    predictable from the prompt; only an empty input needs a fallback.
    """
    chars = []
    for char in command.strip()[:max_length]:
        chars.append(char if char.isalnum() else "-")
    slug = "".join(chars)
    return slug or "task"


def build_worker_argv(command: str, exec_args: ExecArgs) -> list[str]:
    """Serialize the exec request into ``python -m local_operator.exec_worker``
    argv. Only set flags are passed so defaults stay in one place (worker)."""
    argv = [sys.executable, "-m", "local_operator.exec_worker", "--prompt", command]
    if exec_args.json_mode:
        argv.append("--json")
    if exec_args.yolo:
        argv.append("--yolo")
    if exec_args.agent_name:
        argv.extend(["--agent", exec_args.agent_name])
    if exec_args.agent_id:
        argv.extend(["--agent-id", exec_args.agent_id])
    if exec_args.hosting:
        argv.extend(["--hosting", exec_args.hosting])
    if exec_args.model:
        argv.extend(["--model", exec_args.model])
    return argv


def _append_job_record(log_path: Path, prompt: str, pid: int) -> str:
    """Register the detached run in ``exec-jobs.json`` and return the job id.

    The ledger is a JSON array rewritten atomically (temp file + rename in
    the same directory) so concurrent spawns cannot corrupt it.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    jobs_path = LOGS_DIR / JOBS_FILE
    records: list[dict[str, Any]] = []
    if jobs_path.exists():
        try:
            loaded = json.loads(jobs_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = [record for record in loaded if isinstance(record, dict)]
        except (OSError, ValueError):
            records = []

    job_id = uuid4().hex[:12]
    records.append(
        {
            "id": job_id,
            "started_at": datetime.now().astimezone().isoformat(),
            "prompt": prompt,
            "log": str(log_path),
            "pid": pid,
        }
    )

    fd, tmp_name = tempfile.mkstemp(dir=str(LOGS_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        os.replace(tmp_name, jobs_path)
    except OSError:
        # Best-effort ledger: never let bookkeeping kill the spawn.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
    return job_id


def _spawn_background(command: str, exec_args: ExecArgs) -> int:
    """Spawn the detached worker, register the job, report, return 0.

    Detachment semantics: ``start_new_session=True`` on POSIX so the worker
    survives this CLI process exiting (Windows has no sessions — the child is
    detached by virtue of not being waited on). stdout/stderr go to the log
    file so the worker's full run is inspectable after the fact.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_DIR / f"exec-{timestamp}-{slugify(command)}.log"

    argv = build_worker_argv(command, exec_args)
    popen_kwargs: dict[str, Any] = dict(
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    with open(log_path, "ab") as log_handle:
        log_handle.write(
            f"# local-operator exec background job\n# prompt: {command}\n".encode("utf-8")
        )
        log_handle.flush()
        process = subprocess.Popen(argv, stdout=log_handle, **popen_kwargs)

    job_id = _append_job_record(log_path, command, process.pid)
    print(f"Started background job {job_id}")
    print(f"Log: {log_path}")
    return 0


def _make_default_session_factory(exec_args: ExecArgs) -> SessionFactory:
    """Bind the shared composition root to this exec invocation.

    Builds the legacy managers from the fixed config dir and an argparse
    namespace carrying the effective selectors. All engine imports stay
    inside :mod:`local_operator.session_factory`; ``create_session`` is
    async, so this factory returns an awaitable the runner awaits.
    """

    def factory() -> Any:
        from local_operator.agents import AgentRegistry
        from local_operator.config import ConfigManager
        from local_operator.credentials import CredentialManager
        from local_operator.session_factory import create_session

        config_dir = Path.home() / ".local-operator"
        config_manager = ConfigManager(config_dir)
        credential_manager = CredentialManager(config_dir)
        agent_registry = AgentRegistry(config_dir)

        session_args = argparse.Namespace(
            hosting=exec_args.hosting,
            model=exec_args.model,
            agent_name=exec_args.agent_name,
            agent_id=exec_args.agent_id,
            yolo=exec_args.yolo,
            train=False,
        )
        return create_session(session_args, config_manager, credential_manager, agent_registry)

    return factory


def run_exec(command: str, args: ExecArgs) -> int:
    """Entry point for the ``exec`` subcommand (README contract: exit 0 on
    success, non-zero on error).

    ``--background`` detaches and returns 0 immediately; foreground builds a
    session via the shared factory (or the monkeypatched test factory) and
    runs one prompt headless through :func:`run_print_mode` — subscribe
    first, prompt once, map error/abort to exit 1.
    """
    if args.background:
        return _spawn_background(command, args)

    import asyncio

    from local_operator.headless_print import run_print_mode

    factory = default_session_factory or _make_default_session_factory(args)

    async def runner() -> int:
        session = factory()
        if asyncio.iscoroutine(session):
            session = await session
        return await run_print_mode(session, [command], json_mode=args.json_mode)

    return asyncio.run(runner())
