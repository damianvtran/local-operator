"""Headless ``exec`` mode — run one task and exit.

README contract: ``local-operator exec "<task>"`` executes the task and
exits 0 on success, non-zero on error. Two execution shapes:

- foreground: build a session via the shared factory, run the prompt through
  the print renderer, return 0/1;
- ``--background``: spawn ``python -m local_operator.exec_worker`` detached
  (``start_new_session=True``) with stdout/stderr redirected to a timestamped
  log under ``~/.local-operator/logs/``, record the job in the lightweight
  JSONL ledger, print the job id + log path, and return 0 immediately. The
  worker is what actually runs the task; it appends a terminal record
  (``finished_at`` + ``exit_code``) to the same ledger on exit.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

#: Root for detached-exec logs and the jobs ledger (the legacy config dir).
LOGS_DIR = Path.home() / ".local-operator" / "logs"

#: Ledger of spawned background jobs (lightweight; the harness AsyncJobManager
#: is in-process and cannot track a detached OS process across CLI runs).
#: Append-only JSONL (CL-11): one JSON record per line, written with
#: O_APPEND so concurrent spawns never lose each other's records (the old
#: read-modify-write JSON array had a lost-update race).
JOBS_FILE = "exec-jobs.jsonl"

#: Returns a session or an awaitable session. Tests monkeypatch this to inject
#: scripted fakes without touching the real engine.
SessionFactory = Callable[[], Any]
default_session_factory: SessionFactory | None = None


@dataclass
class ExecArgs:
    """CLI-facing options for one ``exec`` invocation.

    Field names mirror the parser dests; ``agent_id`` is the additive
    ``exec --agent-id`` selector (by id instead of by name); ``train``
    carries the legacy ``--train`` flag through to the worker (CL-05).
    """

    background: bool = False
    json_mode: bool = False
    agent_name: str | None = None
    agent_id: str | None = None
    yolo: bool = False
    hosting: str | None = None
    model: str | None = None
    train: bool = False


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
    if exec_args.train:
        argv.append("--train")
    if exec_args.agent_name:
        argv.extend(["--agent", exec_args.agent_name])
    if exec_args.agent_id:
        argv.extend(["--agent-id", exec_args.agent_id])
    if exec_args.hosting:
        argv.extend(["--hosting", exec_args.hosting])
    if exec_args.model:
        argv.extend(["--model", exec_args.model])
    return argv


def _ensure_logs_dir() -> None:
    """Create ``LOGS_DIR`` owner-only (CL-10): the directory holds job logs
    and the ledger, neither of which other users should read or tamper with.
    """
    LOGS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(LOGS_DIR, 0o700)  # umask may have clipped the mode on mkdir
    except OSError:
        pass


def _open_log_file(log_path: Path) -> Any:
    """Open a job log for append, forcing 0600 regardless of umask (CL-10)."""
    return os.fdopen(os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab")


def _append_job_record(log_path: Path, prompt: str, pid: int, job_id: str | None = None) -> str:
    """Append the detached run as one JSONL record and return the job id.

    O_APPEND single-write (CL-11): POSIX guarantees atomicity for small
    writes on O_APPEND fds, so concurrent spawns cannot lose or interleave
    each other's records. Readers tolerate a partial trailing line.
    ``finished_at``/``exit_code`` start unset; the worker appends a terminal
    record carrying both when the run exits (CL-09).
    """
    _ensure_logs_dir()
    jobs_path = LOGS_DIR / JOBS_FILE

    job_id = job_id or uuid4().hex[:12]
    record = {
        "id": job_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "prompt": prompt,
        "log": str(log_path),
        "pid": pid,
        "finished_at": None,
        "exit_code": None,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        fd = os.open(str(jobs_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        # Best-effort ledger: never let bookkeeping kill the spawn.
        pass
    return job_id


def read_job_records() -> list[dict[str, Any]]:
    """Parse the JSONL ledger; tolerate a partial final line and any stray
    corruption by skipping it (the ledger must never break a reader)."""
    records: list[dict[str, Any]] = []
    jobs_path = LOGS_DIR / JOBS_FILE
    if not jobs_path.exists():
        return records
    try:
        text = jobs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            loaded = json.loads(line)
        except ValueError:
            continue  # partial trailing line or corrupt record — skip
        if isinstance(loaded, dict):
            records.append(loaded)
    return records


def update_job_exit(job_id: str, exit_code: int) -> None:
    """Append the terminal record for ``job_id`` (CL-09): ``finished_at`` +
    ``exit_code``. Append-only keeps this race-free; consumers take the
    latest record per id, so the terminal record supersedes the spawn one."""
    _ensure_logs_dir()
    jobs_path = LOGS_DIR / JOBS_FILE
    update = {
        "id": job_id,
        "finished_at": datetime.now().astimezone().isoformat(),
        "exit_code": exit_code,
    }
    try:
        fd = os.open(str(jobs_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(update, ensure_ascii=False) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass


def resolve_hosting_model_dry(exec_args: ExecArgs) -> tuple[str, str]:
    """Preflight hosting/model resolution WITHOUT spawning (CL-09).

    Uses the exact same precedence path the worker will use (agent > flag >
    config) via the composition root's ``resolve_agent``/``resolve_hosting_model``
    plus the registry's agent-id lookup, raising ``ValueError`` with the
    legacy message shapes when unconfigured.
    """
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.session_factory import resolve_agent, resolve_hosting_model

    config_dir = Path.home() / ".local-operator"
    config_manager = ConfigManager(config_dir)
    agent_registry = AgentRegistry(config_dir)
    selector_args = argparse.Namespace(
        hosting=exec_args.hosting,
        model=exec_args.model,
        agent_name=exec_args.agent_name,
        agent_id=exec_args.agent_id,
    )
    agent = resolve_agent(selector_args, agent_registry)
    return resolve_hosting_model(agent, selector_args, config_manager)


def _spawn_background(command: str, exec_args: ExecArgs) -> int:
    """Spawn the detached worker, register the job, report, return 0.

    Detachment semantics: ``start_new_session=True`` on POSIX so the worker
    survives this CLI process exiting (Windows has no sessions — the child is
    detached by virtue of not being waited on). stdout/stderr go to the log
    file so the worker's full run is inspectable after the fact.

    Preflight (CL-09): hosting/model (and agent-id lookup) are validated via
    the same resolution path the worker uses BEFORE any spawn; a failure
    prints the legacy error shape and returns non-zero without spawning.
    """
    try:
        resolve_hosting_model_dry(exec_args)
    except ValueError as exc:
        print(f"\n\033[1;31mError: {exc}\033[0m")
        return -1
    except Exception as exc:  # noqa: BLE001 — never spawn blind
        print(f"\n\033[1;31mError: preflight failed: {exc}\033[0m")
        return -1

    _ensure_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_DIR / f"exec-{timestamp}-{slugify(command)}.log"

    job_id = uuid4().hex[:12]
    argv = build_worker_argv(command, exec_args)
    argv.extend(["--job-id", job_id])
    popen_kwargs: dict[str, Any] = dict(
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    with _open_log_file(log_path) as log_handle:
        log_handle.write(
            f"# local-operator exec background job\n# prompt: {command}\n".encode("utf-8")
        )
        log_handle.flush()
        process = subprocess.Popen(argv, stdout=log_handle, **popen_kwargs)

    _append_job_record(log_path, command, process.pid, job_id=job_id)
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
            train=exec_args.train,
        )
        return create_session(session_args, config_manager, credential_manager, agent_registry)

    return factory


def run_exec(command: str, args: ExecArgs) -> int:
    """Entry point for the ``exec`` subcommand (README contract: exit 0 on
    success, non-zero on error).

    ``--background`` detaches and returns 0 immediately; foreground builds a
    session via the shared factory (or the monkeypatched test factory) and
    runs one prompt headless through :func:`run_print_mode` — subscribe
    first, prompt once, map error/abort to exit 1. A ``prompt()`` that
    RAISES also maps to exit 1 with the error on stderr (CL-19), never the
    interactive red banner: exec is machine-driven.
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

    try:
        return asyncio.run(runner())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 — CL-19: raising prompt = exit 1
        print(f"exec failed: {exc}", file=sys.stderr)
        return 1
