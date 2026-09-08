"""Headless ``exec`` mode — run one task and exit.

README contract: ``local-operator exec "<task>"`` executes the task and
exits 0 on success, non-zero on error. Two execution shapes:

- foreground: build a session via the shared factory, run the prompt through
  the print renderer, return 0/1;
- ``--background``: spawn ``python -m local_operator.exec_worker`` detached
  (``start_new_session=True``) with stdout/stderr redirected to a timestamped
  log under :func:`logs_dir` (``~/.local-operator/logs/`` by default, and the
  override's ``logs/`` when ``LOCAL_OPERATOR_CONFIG_DIR`` is set), record the
  job in the lightweight JSONL ledger, print the job id + log path, and return
  0 immediately. The worker is what actually runs the task; it appends a
  terminal record (``finished_at`` + ``exit_code``) to the same ledger on exit.

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

from local_operator.interpreter import python_argv


def logs_dir() -> Path:
    """Root for detached-exec logs and the jobs ledger: ``config_dir()/logs``.

    Shares :data:`local_operator.paths.LOG_DIRNAME` with the app's rotating
    log rather than spelling ``"logs"`` a second time. Under an override the
    two therefore land in one directory, which is fine and deliberate: the
    filenames are disjoint (``local-operator.log`` vs ``exec-*.log`` /
    ``exec-jobs.jsonl``) and both roots are created 0700.

    A FUNCTION, not the module constant this used to be, for the reason
    :func:`local_operator.paths.config_dir` states in its own docstring: a
    constant freezes whatever the first importer saw, and the override is read
    from the environment on every call. The old constant was
    ``Path.home() / ".local-operator" / "logs"``, written in the original CLI
    rewrite BEFORE ``paths.py`` and ``LOCAL_OPERATOR_CONFIG_DIR`` existed at all
    — which is what its "the legacy config dir" comment recorded. Three sibling
    copies of that same expression in this file (the foreground factory, the
    background worker's factory, and the preflight resolver) have since been
    fixed to resolve through ``config_dir()``; this was the last one, and it
    split the jobs ledger away from the root the rest of exec mode honours: with
    the override set, ``exec --background`` wrote its config and agents under the
    override while its log and ledger landed in the operator's real home.

    Resolution is UNCHANGED when no override is set — ``config_dir()`` is itself
    ``~/.local-operator`` — so an existing install's logs and ledger stay exactly
    where they are. Only an overridden run moves, which is the whole point.
    """
    from local_operator.paths import LOG_DIRNAME, config_dir

    return config_dir() / LOG_DIRNAME


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
    #: Session id to resume, or the `@latest` sentinel. Accepted because the
    #: shared parent parser offers `--resume` on every subcommand; carried
    #: through so `exec` continues a session rather than silently starting a new
    #: one and reporting success against the wrong history.
    resume: str | None = None
    #: Publish a discovery record and serve the control socket for this run, so
    #: an external supervisor can steer, cancel and answer gates mid-run. OFF by
    #: default — the reasoning for opt-in (session-list pollution, daemon
    #: adoption, heartbeat cadence, import weight) is in
    #: :mod:`local_operator.session.runtime.exec_control`, which owns the
    #: mechanism. Carried through to the worker so `--background --control` is
    #: the same request run elsewhere, exactly like ``resume``.
    control: bool = False
    team: str | None = None
    profile: str | None = None
    goal: str | None = None
    clear_goal: bool = False
    loop: int | None = None
    loop_goal: str | None = None
    name: str | None = None
    effort: str | None = None


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
    # ``python_argv``: `exec --background` is launched from wherever the user
    # happens to be standing, and the worker imports the whole harness
    # (session_factory, agents, config) — so a run started inside a checkout of
    # this project would execute that checkout rather than the installed build,
    # silently answering with different code than `lop --version` reports.
    # Same defect as the runtime spawn; see :mod:`local_operator.interpreter`.
    argv = python_argv("-m", "local_operator.exec_worker", "--prompt", command)
    from local_operator.exec_startup import STARTUP_FIELDS

    for field in STARTUP_FIELDS:
        value = getattr(exec_args, field)
        if value is not None and value is not False:
            argv.append("--" + field.replace("_", "-"))
            if value is not True:
                argv.append(str(value))
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
    if exec_args.control:
        # A detached run is the one that most needs steering — nobody is
        # watching its log — so the flag has to survive the process boundary.
        # Dropped here it would be accepted by the front end and silently lost,
        # the identical failure the ``resume`` note below records.
        argv.append("--control")
    if exec_args.resume:
        # Serialized like every other field, because `--background` is supposed to
        # be the same request run elsewhere. Omitted, `exec --background --resume`
        # silently started a FRESH session in the worker and reported success
        # against the wrong history — the failure `ExecArgs.resume` exists to
        # prevent, one process boundary further out.
        argv.extend(["--resume", exec_args.resume])
    return argv


def _ensure_logs_dir() -> Path:
    """Create :func:`logs_dir` owner-only (CL-10): the directory holds job logs
    and the ledger, neither of which other users should read or tamper with.

    Returns the resolved directory so a caller that needs the path uses the
    SAME resolution this created, rather than calling the resolver a second
    time — the override could differ between the two calls, and a log written
    to a directory that was never created is the failure this prevents.
    """
    directory = logs_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)  # umask may have clipped the mode on mkdir
    except OSError:
        pass
    return directory


def _open_log_file(log_path: Path) -> Any:
    """Open a job log for append, forcing 0600 regardless of umask (CL-10)."""
    return os.fdopen(os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab")


def _append_job_record(
    log_path: Path,
    prompt: str,
    pid: int,
    job_id: str | None = None,
    *,
    requested_team: str | None = None,
) -> str:
    """Append the detached run as one JSONL record and return the job id.

    O_APPEND single-write (CL-11): POSIX guarantees atomicity for small
    writes on O_APPEND fds, so concurrent spawns cannot lose or interleave
    each other's records. Readers tolerate a partial trailing line.
    ``finished_at``/``exit_code`` start unset; the worker appends a terminal
    record carrying both when the run exits (CL-09).
    """
    jobs_path = _ensure_logs_dir() / JOBS_FILE

    job_id = job_id or uuid4().hex[:12]
    record = {
        "id": job_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "prompt": prompt,
        "log": str(log_path),
        "pid": pid,
        "process_generation": _process_generation(pid),
        "status": "starting",
        "requested_team": requested_team,
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
    # Resolved, not created: a read must not be the thing that materialises the
    # logs directory on a root that has none.
    jobs_path = logs_dir() / JOBS_FILE
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
    jobs_path = _ensure_logs_dir() / JOBS_FILE
    update = {
        "id": job_id,
        "finished_at": datetime.now().astimezone().isoformat(),
        "exit_code": exit_code,
        "pid": os.getpid(),
        "process_generation": _process_generation(os.getpid()),
        "status": (
            "succeeded" if exit_code == 0 else "cancelled" if exit_code in (130, 143) else "failed"
        ),
    }
    try:
        fd = os.open(str(jobs_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(update, ensure_ascii=False) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass


def _process_generation(pid: int) -> str | None:
    # Reuse the resource reaper's locale-stable start token; bare PID liveness
    # must never credit a recycled process with keeping this job alive.
    from local_operator.tools.group_reaper import _owner_start_token

    return _owner_start_token(pid)


def update_job_running(job_id: str, session: Any, control: Any) -> None:
    from local_operator.session.runtime.registry import record_path

    update = {
        "id": job_id,
        "session_id": session.session_id,
        "pid": os.getpid(),
        "process_generation": _process_generation(os.getpid()),
        "status": "running",
        "team": session.active_team_name,
        "session_directory": str(session._transcript.directory),
        "runtime_path": str(record_path(os.getpid())),
    }
    _append_job_update(update)


def _append_job_update(update: dict[str, Any]) -> None:
    path = _ensure_logs_dir() / JOBS_FILE
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(update, ensure_ascii=False) + "\n").encode())
    finally:
        os.close(fd)


def job_status(job_id: str, *, reconcile: bool = True) -> dict[str, Any]:
    """Fold the existing append-only ledger without allowing late spawn rows
    to regress a worker's ready/terminal record. Dead owners are interrupted,
    never successful, and reconciliation never acts on their resources.
    """
    result: dict[str, Any] = {}
    rank = {
        "starting": 0,
        "running": 1,
        "succeeded": 2,
        "failed": 2,
        "cancelled": 2,
        "interrupted": 2,
    }
    current = -1
    for row in read_job_records():
        if row.get("id") != job_id:
            continue
        level = rank.get(row.get("status", "starting"), 0)
        if level < current:
            for key, value in row.items():
                result.setdefault(key, value)
            continue
        result.update(row)
        current = level
    if reconcile and result.get("status") in ("starting", "running"):
        from local_operator.tools.group_reaper import _owner_is_dead

        generation = result.get("process_generation")
        pid = result.get("pid")
        if pid and generation and _owner_is_dead(pid, generation) is True:
            update = {
                "id": job_id,
                "status": "interrupted",
                "finished_at": datetime.now().astimezone().isoformat(),
            }
            _append_job_update(update)
            result.update(update)
    return result


def resolve_hosting_model_dry(exec_args: ExecArgs) -> tuple[str, str]:
    """Preflight hosting/model resolution WITHOUT spawning (CL-09).

    Uses the exact same precedence path the worker will use (agent > flag >
    config) via the composition root's ``resolve_agent``/``resolve_hosting_model``
    plus the registry's agent-id lookup, raising ``ValueError`` with the
    legacy message shapes when unconfigured.
    """
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.paths import config_dir
    from local_operator.session_factory import resolve_agent, resolve_hosting_model

    # config_dir(), not ``Path.home() / ".local-operator"``: the same missed copy
    # fixed in ``_make_default_session_factory`` below. Preflight's whole purpose
    # is to resolve hosting/model through the EXACT path the worker will use, so
    # reading a different config root than the worker does defeats it — with
    # LOCAL_OPERATOR_CONFIG_DIR set this validated against the developer's real
    # agents and config and then spawned a worker that used the override's.
    base_dir = config_dir()
    config_manager = ConfigManager(base_dir)
    agent_registry = AgentRegistry(base_dir)
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
        print(f"\n\033[1;31mError: {exc}\033[0m", file=sys.stderr)
        # Return 1, not -1: this becomes the process exit code via exit(main()),
        # and a negative return maps to exit 255 (item 18's contract is that a
        # preflight failure exits with a clean non-zero, not a wrapped -1).
        return 1
    except Exception as exc:  # noqa: BLE001 — never spawn blind
        print(f"\n\033[1;31mError: preflight failed: {exc}\033[0m", file=sys.stderr)
        return 1

    logs_root = _ensure_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = uuid4().hex[:12]
    log_path = logs_root / f"exec-{timestamp}-{slugify(command)}-{job_id}.log"
    argv = build_worker_argv(command, exec_args)
    argv.extend(["--job-id", job_id])
    popen_kwargs: dict[str, Any] = dict(
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    # Name the detached worker in the OS process listing, keyed by the job id
    # this call already prints to the user, so `ps` and `lop`'s own job output
    # agree on one handle. Both the image and argv[0] fall back to the bare
    # interpreter when no branded image exists.
    from local_operator import procname

    link = procname.ensure_branded_interpreter()
    if link is not None:
        popen_kwargs["executable"] = str(link)
        argv = list(argv)
        argv[0] = procname.branded_argv0(procname.LABEL_EXEC, job=job_id)

    with _open_log_file(log_path) as log_handle:
        log_handle.write(
            f"# local-operator exec background job\n# prompt: {command}\n".encode("utf-8")
        )
        log_handle.flush()
        process = subprocess.Popen(argv, stdout=log_handle, **popen_kwargs)

    _append_job_record(log_path, command, process.pid, job_id=job_id, requested_team=exec_args.team)
    # --json and --background are independent flags, so these notices must not
    # land on stdout: a consumer parsing the event stream would hit two
    # unparseable lines before any event.
    import time

    # Readiness is not completion: bound the launcher's wait and report an
    # honest 'starting' receipt when provider/session initialization is slow.
    deadline = time.monotonic() + 5.0
    state = job_status(job_id, reconcile=False)
    while state.get("status") == "starting" and time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.05)
        state = job_status(job_id, reconcile=False)
    state = job_status(job_id)
    print(f"Background job {job_id}: {state.get('status', 'starting')}", file=sys.stderr)
    if state.get("session_id"):
        print(
            f"Session: {state['session_id']} (lop --resume {state['session_id']})", file=sys.stderr
        )
    print(f"Status: lop exec --status {job_id}", file=sys.stderr)
    print(f"Log: {log_path}", file=sys.stderr)
    return 1 if state.get("status") in ("failed", "cancelled", "interrupted") else 0


def _make_default_session_factory(exec_args: ExecArgs) -> SessionFactory:
    """Bind the shared composition root to this exec invocation.

    Builds the legacy managers from the app config dir and an argparse
    namespace carrying the effective selectors. All engine imports stay
    inside :mod:`local_operator.session_factory`; ``create_session`` is
    async, so this factory returns an awaitable the runner awaits.
    """

    def factory() -> Any:
        from local_operator.agents import AgentRegistry
        from local_operator.config import ConfigManager
        from local_operator.credentials import CredentialManager
        from local_operator.paths import config_dir
        from local_operator.session_factory import create_session

        # config_dir(), not ``Path.home() / ".local-operator"``: this was the
        # last hardcoded copy of that path, and it made `exec` the one entry
        # point that ignored LOCAL_OPERATOR_CONFIG_DIR — so an exec run wrote
        # its transcript, its autosave agent and its session directory into
        # the developer's real config dir even when the environment pointed
        # somewhere else, which is precisely the divergence paths.py exists
        # to prevent (and which makes exec impossible to isolate in a test or
        # a benchmark).
        base_dir = config_dir()
        config_manager = ConfigManager(base_dir)
        credential_manager = CredentialManager(base_dir)
        agent_registry = AgentRegistry(base_dir)

        session_args = argparse.Namespace(
            hosting=exec_args.hosting,
            model=exec_args.model,
            agent_name=exec_args.agent_name,
            agent_id=exec_args.agent_id,
            yolo=exec_args.yolo,
            train=exec_args.train,
            resume=exec_args.resume,
        )
        return create_session(session_args, config_manager, credential_manager, agent_registry)

    return factory


#: The conventional "read the prompt from stdin" argument, as codex and other
#: CLI harnesses spell it. Supervisors pipe a composed prompt file rather than
#: passing it in argv, which is both visible in ``ps`` and bounded by ARG_MAX.
STDIN_PROMPT_SENTINEL = "-"


def resolve_prompt(command: str | None, *, stdin_text: str | None = None) -> str:
    """Return the prompt to run, reading stdin when ``command`` is ``-``.

    Resolved BEFORE the ``--background`` branch on purpose: the background
    worker receives its prompt through argv (:func:`build_worker_argv`), so a
    stdin prompt left unresolved would spawn a worker whose prompt is the
    literal ``-`` — the same class of silent-wrong-input bug the ``resume``
    field documents having already been fixed once, one process boundary out.

    ``stdin_text`` is injectable so the behaviour is testable without a real
    pipe on the process.
    """
    if command is None:
        if stdin_text is None and sys.stdin.isatty():
            return ""
    elif command != STDIN_PROMPT_SENTINEL:
        return command
    text = sys.stdin.read() if stdin_text is None else stdin_text
    return text.strip()


def run_exec(command: str | None, args: ExecArgs) -> int:
    """Entry point for the ``exec`` subcommand (README contract: exit 0 on
    success, non-zero on error).

    ``--background`` detaches and returns 0 immediately; foreground builds a
    session via the shared factory (or the monkeypatched test factory) and
    runs one prompt headless through :func:`run_print_mode` — subscribe
    first, prompt once, map error/abort to exit 1. A ``prompt()`` that
    RAISES also maps to exit 1 with the error on stderr (CL-19), never the
    interactive red banner: exec is machine-driven.

    A ``command`` of ``-`` means "read the prompt from stdin"; it is resolved
    here so the foreground and ``--background`` paths run the same text.

    ``--control`` wraps the foreground run in a session runtime (record +
    control socket) so a supervisor can steer and cancel it; the endpoint goes
    to STDERR because stdout is the payload stream.
    """
    from local_operator.exec_startup import resolve_startup

    try:
        team = resolve_startup(args)
        command = resolve_prompt(command)
    except (ValueError, OSError) as exc:
        print(f"exec failed: {exc}", file=sys.stderr)
        return 1
    # The positional is optional so a loop-only or piped run can omit it
    # (argparse cannot express "required unless --loop/--loop-goal/stdin"), so
    # the requirement is enforced here — naming the ways to supply one rather
    # than reporting a bare "empty prompt".
    if not command.strip() and args.loop is None and args.loop_goal is None:
        print(
            "exec failed: no prompt. Pass one as an argument, pipe it on stdin "
            "(or '-'), or run a loop with --loop/--loop-goal",
            file=sys.stderr,
        )
        return 1
    if args.background:
        return _spawn_background(command, args)

    import asyncio

    from local_operator.exec_session import run_session

    factory = default_session_factory or _make_default_session_factory(args)

    async def runner() -> int:
        session = factory()
        if asyncio.iscoroutine(session):
            session = await session
        return await run_session(session, command, args, team)

    try:
        return asyncio.run(runner())
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    except Exception as exc:  # noqa: BLE001 — CL-19: raising prompt = exit 1
        print(f"exec failed: {exc}", file=sys.stderr)
        return 1
