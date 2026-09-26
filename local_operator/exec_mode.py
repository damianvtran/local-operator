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

from local_operator import procstate
from local_operator.interpreter import python_argv
from local_operator.procstate import O_BINARY


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
    #: The descriptor a supervised run hands its own capability UP (stage E).
    #: Carried as an INTEGER, not a flag, because the supervisor's socketpair end
    #: is created before this process starts — the number is only meaningful to
    #: the process it was inherited by, which is why it must survive the
    #: ``--background`` argv boundary unaltered if it is serialized at all.
    #: ``None`` (the default) means no supervisor is holding the other end, and
    #: the run then holds no capability and every authority-increasing request
    #: that does not carry a signature is refused — the fail-closed state.
    supervisor_fd: int | None = None
    team: str | None = None
    profile: str | None = None
    #: Comma-separated tools this run may reach, and the only ones. Reaches the
    #: session through ``exec_startup.apply_startup``, so a declaration holds for
    #: the foreground CLI and for the detached worker alike (``STARTUP_FIELDS``
    #: carries it across that boundary). ``None`` — the default — leaves the
    #: session unrestricted, exactly as every run before this flag existed.
    tools: str | None = None
    #: The operator asked for this run as a LONG-LIVED PARALLEL WORKSTREAM, so
    #: it is published rather than hidden: listed in the sidebar, ``/resume`` and
    #: the phone list, labelled with the session that opened it, and steerable.
    #:
    #: OFF by default, and that direction is the whole point: every caller that
    #: predates this flag gets the ephemeral behaviour it already had — an
    #: agent-opened run is stamped ``agent-shell`` and hidden everywhere — so the
    #: list only ever grows by the runs the operator actually asked for.
    #:
    #: INDEPENDENT OF :attr:`control`. Every exec run publishes its discovery
    #: record and serves the control socket already, so the row is followable
    #: and steerable without it; ``control`` changes the APPROVAL posture (gates
    #: park instead of denying, ``tools`` stops standing as the approval), which
    #: is not something choosing a row's visibility may change behind the
    #: caller's back (PR #1436 agent review round 1, F1).
    #: Meaningful only under an agent's shell — that is where the run is stamped
    #: ``agent-workstream``; outside one nothing is stamped and the flag is a
    #: no-op, so the run is an ordinary session either way.
    workstream: bool = False
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
    # EVERY value-carrying option uses the `--opt=value` form, never two argv
    # items. argparse reads a following token that starts with `-` as the next
    # OPTION, so `--name -nightly` (or a prompt phrased `-- verify everything`)
    # dies at the worker's parse_args — before `--job-id` is honoured, so no
    # terminal ledger row is ever written and reconciliation reports the run as
    # `interrupted`: the vocabulary reserved for a worker killed mid-flight,
    # for a run that never started. The `=` form is unambiguous for any value.
    argv = python_argv("-m", "local_operator.exec_worker", f"--prompt={command}")
    from local_operator.exec_startup import STARTUP_FIELDS

    for field in STARTUP_FIELDS:
        value = getattr(exec_args, field)
        if value is None or value is False:
            continue
        option = "--" + field.replace("_", "-")
        # `True` is a store_true flag, which carries no value to attach.
        argv.append(option if value is True else f"{option}={value}")
    if exec_args.json_mode:
        argv.append("--json")
    if exec_args.yolo:
        argv.append("--yolo")
    if exec_args.train:
        argv.append("--train")
    if exec_args.agent_name:
        argv.append(f"--agent={exec_args.agent_name}")
    if exec_args.agent_id:
        argv.append(f"--agent-id={exec_args.agent_id}")
    if exec_args.hosting:
        argv.append(f"--hosting={exec_args.hosting}")
    if exec_args.model:
        argv.append(f"--model={exec_args.model}")
    if exec_args.control:
        # A detached run is the one that most needs steering — nobody is
        # watching its log — so the flag has to survive the process boundary.
        # Dropped here it would be accepted by the front end and silently lost,
        # the identical failure the ``resume`` note below records.
        argv.append("--control")
    if exec_args.supervisor_fd is not None:  # pragma: no cover — refused upstream
        # NEVER REACHED: the front end refuses ``--background --supervisor-fd``
        # (see :func:`reject_detached_supervisor_fd`) because ``--background``
        # detaches the worker and its launcher exits, closing the other end of the
        # supervisor's socketpair. A descriptor serialized past that point names a
        # number the worker does not hold, and the failure would surface as an
        # EPIPE in the run rather than as the configuration error it is. The
        # append stays as the assertion of intent and is guarded so it can never
        # be a silent pass — the branch that would produce it raises first.
        raise AssertionError("a detached run must not be given a supervisor descriptor")
    if exec_args.resume:
        # Serialized like every other field, because `--background` is supposed to
        # be the same request run elsewhere. Omitted, `exec --background --resume`
        # silently started a FRESH session in the worker and reported success
        # against the wrong history — the failure `ExecArgs.resume` exists to
        # prevent, one process boundary further out.
        argv.append(f"--resume={exec_args.resume}")
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
    descriptor = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | O_BINARY, 0o600)
    return os.fdopen(descriptor, "ab")


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
        fd = os.open(str(jobs_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | O_BINARY, 0o600)
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
        fd = os.open(str(jobs_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | O_BINARY, 0o600)
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


def update_job_running(job_id: str, session: Any) -> None:
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
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | O_BINARY, 0o600)
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
    if result.get("status") == "running":
        # "Waiting for you" and "working" are the two states a detached run can
        # be in, and they are the ones a user must tell apart — a supervised run
        # parked on a gate reports `running` forever, and neither --status nor
        # the log said so. `lop sessions` already computes this from the live
        # runtime record; read the same field rather than inventing a second
        # source of truth. NOT persisted to the ledger: it is live state that
        # goes stale the moment the gate is answered, whereas every ledger row
        # is a durable fact about the run.
        result["pending"] = _live_pending(result.get("runtime_path"))
    return result


def _live_pending(runtime_path: str | None) -> str | None:
    """What the live runtime says this run is blocked on, or ``None``.

    Reads the record the run already publishes. Absent/unreadable/older-runtime
    records answer ``None`` — an unknown answer must never be reported as a
    parked gate, and a status read must not fail because a worker just exited.
    """
    if not runtime_path:
        return None
    try:
        with open(runtime_path, encoding="utf-8") as handle:
            return json.load(handle).get("pending") or None
    except (OSError, ValueError):
        return None


def resolve_hosting_model_dry(exec_args: ExecArgs) -> tuple[str, str]:
    """Preflight hosting/model resolution WITHOUT spawning (CL-09).

    Uses the exact same precedence path the worker will use (agent > flag >
    config) via the composition root's ``resolve_agent``/``resolve_hosting_model``
    plus the registry's agent-id lookup, raising ``ValueError`` with the
    legacy message shapes when unconfigured.
    """
    from local_operator.agents import AgentRegistry, agents_store_present
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
    # GUARDED (review round 2, finding 1): build the registry only when it has
    # something to answer — an agent selector, or a store already on disk.
    # ``resolve_agent`` returns ``None`` for a selector-less call WITHOUT
    # reading the registry, so on a truly fresh root the old unconditional
    # construction wrote ``config/agents/`` for an answer nothing consumed,
    # and it did so on BOTH launch arms before the deny-trap advisory could
    # reach its fresh-root branch (foreground: the ``cli.py`` preflight then
    # ``run_exec``; background: ``_spawn_background``). A selector keeps the
    # construction exactly as it was — ``--agent`` creates on a miss, so the
    # registry is load-bearing there — and a store that EXISTS is still
    # constructed (migrations included) even without one, preserving prior
    # behaviour byte for byte.
    selector_args = argparse.Namespace(
        hosting=exec_args.hosting,
        model=exec_args.model,
        agent_name=exec_args.agent_name,
        agent_id=exec_args.agent_id,
    )
    agent_registry: AgentRegistry | None = (
        AgentRegistry(base_dir)
        if (exec_args.agent_name or exec_args.agent_id) or agents_store_present(base_dir)
        else None
    )
    # ``agent_registry`` is None only when no selector is present (the guard
    # above) — which is exactly the case ``resolve_agent`` answers with
    # ``None`` before it reads the registry, so this branch is that same
    # answer without the construction's write.
    agent = resolve_agent(selector_args, agent_registry) if agent_registry is not None else None
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
    argv.append(f"--job-id={job_id}")
    popen_kwargs: dict[str, Any] = dict(
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    # ``start_new_session`` is documented "(POSIX only)" and Windows SILENTLY
    # ignores it — its ``_execute_child`` parameter is literally named
    # ``unused_start_new_session`` — so a hand-rolled POSIX branch here looked
    # like a detached worker on Windows while the child kept this console: a
    # Ctrl-C and a console close both reached it, which is the property this
    # call exists to get. ``procstate.detached_popen_kwargs`` owns the
    # per-platform answer and the reasoning; this is the last call site that
    # spelled its own.
    popen_kwargs.update(procstate.detached_popen_kwargs())

    # Name the detached worker in the OS process listing, keyed by the job id
    # this call already prints to the user, so `ps` and `lop`'s own job output
    # agree on one handle. `spawn_identity` supplies BOTH axes as a pair: the
    # label is `argv[0]` when a branded image was planted, and where one could
    # not be, the interpreter arrives unlabelled — a label on its own would be
    # EXECUTED as a path on POSIX, and on Linux it would also empty the child's
    # `sys.executable` (see `procname.spawn_identity`).
    from local_operator import procname

    argv0, executable = procname.spawn_identity(procname.LABEL_EXEC, job=job_id)
    argv = list(argv)
    argv[0] = argv0
    popen_kwargs["executable"] = executable

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
    status = state.get("status", "starting")
    # The job/session split is the receipt's whole reason for existing, but two
    # twelve-hex ids in the same shape do not carry it on their own: say which
    # is the execution receipt and which is the conversation, and give each an
    # imperative rather than leaving `Session:` a bare parenthetical.
    failure = _worker_failure(log_path) if status == "failed" else ""
    # The reason is already written to a file whose path we hold, and the
    # launcher is still in the foreground: making the user open a log to read a
    # one-line validation error is a gap we can close for free.
    print(
        f"Background job {job_id}: {status} (execution receipt)"
        + (f" \u2014 {failure}" if failure else ""),
        file=sys.stderr,
    )
    if state.get("session_id"):
        print(
            f"Session: {state['session_id']} (the conversation) \u2014 attach or resume: "
            f"lop --resume {state['session_id']}",
            file=sys.stderr,
        )
    print(f"Status: lop exec --status {job_id}", file=sys.stderr)
    print(f"Log: {log_path}", file=sys.stderr)
    if getattr(exec_args, "control", False):
        # A supervised run can park on a gate and sit at `running` indefinitely.
        # `lop sessions` is the one command that shows what a run NEEDS, and it
        # was the one command this receipt never mentioned.
        print("Waiting on you? lop sessions shows what a run needs", file=sys.stderr)
    elif status in ("starting", "running"):
        # The receipt is where a launch is read back, so a run nobody can
        # approve has to say so HERE rather than only in a transcript of
        # denials (see ``_deny_trapped_advisory``). Printed only while the run
        # is live: a launch that already failed carries its own reason above.
        advisory = _deny_trapped_advisory(exec_args)
        if advisory is not None:
            print(advisory, file=sys.stderr)
    return 1 if status in ("failed", "cancelled", "interrupted") else 0


def _worker_failure(log_path: Path) -> str:
    """The worker's own last error line, for a launch that already failed.

    Best-effort by contract: the receipt is strictly better with the reason and
    must never be lost to a race on the log file, so any read problem yields an
    empty string and the caller prints the plain status it already had.
    """
    try:
        lines = [line.strip() for line in log_path.read_text(errors="replace").splitlines()]
    except OSError:
        return ""
    for line in reversed(lines):
        if line and not line.startswith("#"):
            return line[:200]
    return ""


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
        agent_registry = AgentRegistry(base_dir)

        session_args = argparse.Namespace(
            hosting=exec_args.hosting,
            model=exec_args.model,
            agent_name=exec_args.agent_name,
            agent_id=exec_args.agent_id,
            yolo=exec_args.yolo,
            train=exec_args.train,
            resume=exec_args.resume,
            # Read by ``session_factory._prepare`` through the stamp, its only
            # consumer: the value decides which ``origin.json`` the run gets,
            # and therefore whether every listing hides it. Carried in BOTH
            # narrow namespaces (here and the detached worker's) because the
            # factory receives this namespace and nothing wider.
            workstream=exec_args.workstream,
        )
        return create_session(session_args, config_manager, agent_registry)

    return factory


#: The conventional "read the prompt from stdin" argument, as codex and other
#: CLI harnesses spell it. Supervisors pipe a composed prompt file rather than
#: passing it in argv, which is both visible in ``ps`` and bounded by ARG_MAX.
STDIN_PROMPT_SENTINEL = "-"


def resolve_prompt(
    command: str | None, *, stdin_text: str | None = None, has_loop: bool = False
) -> str:
    """Return the prompt to run, reading stdin when ``command`` is ``-``.

    Resolved BEFORE the ``--background`` branch on purpose: the background
    worker receives its prompt through argv (:func:`build_worker_argv`), so a
    stdin prompt left unresolved would spawn a worker whose prompt is the
    literal ``-`` — the same class of silent-wrong-input bug the ``resume``
    field documents having already been fixed once, one process boundary out.

    ``has_loop`` reports that ``--loop``/``--loop-goal`` was given, which is a
    DECLARATION that this run has no prompt. Without it an omitted positional
    fell through to an unbounded ``sys.stdin.read()`` on any non-TTY stdin, and
    a pipe whose writer stays open never sends EOF — so the documented
    ``lop exec --goal X --loop 3`` hung forever, with no output, under every
    supervisor that hands its child an inherited pipe (a CI runner, a cmux
    surface, ``Popen(stdin=PIPE)``). An explicit ``-`` still reads, because
    that is the user asking for stdin rather than merely inheriting one.

    ``stdin_text`` is injectable so the behaviour is testable without a real
    pipe on the process.
    """
    if command is None:
        if stdin_text is None and (has_loop or sys.stdin.isatty()):
            return ""
    elif command != STDIN_PROMPT_SENTINEL:
        return command
    text = sys.stdin.read() if stdin_text is None else stdin_text
    return text.strip()


def reject_detached_supervisor_fd(args: ExecArgs) -> str | None:
    """Why ``--background --supervisor-fd`` is refused, or ``None`` when it is fine.

    THE COMBINATION CANNOT WORK, so it is refused rather than degraded. The
    descriptor names one end of a socketpair the SUPERVISOR holds; ``--background``
    detaches a worker and the launcher that owns that end exits immediately, so the
    run's upward capability write finds a closed peer. The observable outcome would
    be a run that appears supervised and whose cards nobody can approve — the
    failure mode the whole stage exists to remove. The design names the row: a
    detached ``--background --control`` run has no live supervisor, and unattended
    approval there is ``--yolo`` or ``tool_approval_mode: auto``.

    Also refused WITHOUT ``--control``: nothing installs a supervisor's gate on an
    unsupervised run, so a descriptor there asks for authority over a surface that
    does not exist.
    """
    if args.supervisor_fd is None:
        return None
    if args.background:
        return (
            "--supervisor-fd cannot be combined with --background: the background "
            "launcher exits, so nothing holds the other end of the supervisor's "
            "socket and the run could never hand its capability up. Use --yolo or "
            "tool_approval_mode: auto for an unattended run."
        )
    if not args.control:
        return (
            "--supervisor-fd requires --control: only a supervised run installs the "
            "gates a supervisor credential could answer"
        )
    return None


def _declared_by_profile(name: str | None) -> bool:
    """Whether the named role's own allow-list answers the deny trap (M2).

    ``--tools`` is not the only declaration that stands as the approval for
    its own members in an unattended run: a ``lop exec --profile reviewer``
    run resolves its inventory from the seed's ``tools:`` list
    (``exec_startup.declared_tool_inventory``), and that list is exactly what
    approves its write/exec members where nobody can be asked. The advisory
    must consult the same source of truth, or it tells a run that will work
    that it will be denied (review round 1, M2).

    Resolution is best-effort and matches the session side by value, not by
    identity: an unresolvable name resolves to "no declaration" and leaves
    the advisory on — ``resolve_startup`` refuses such runs before this is
    reached anyway.

    DOCUMENTED LIMITS (post-merge review of #1597). This is a launch-time
    oracle, and it is narrower than the session's own resolution in two known
    ways. ``--resume`` restores a stored role attachment INSIDE the session,
    with no ``--profile`` on this command line, so a resumed run whose role
    declares tools still prints the advisory — the copy stays true (calls that
    need approval will be denied; the role's own members do not need it), but
    the suppression is narrower than the session's. Conversely, a role's
    allow-list approves only its members, so a call outside it still denies
    with no advisory: the declaration bounds reach rather than approving the
    run. Sharing one predicate with ``exec_startup.declared_tool_inventory``
    would close both gaps, but that function needs the live session and runs
    after this point; it is a recorded follow-up, not a claim of equivalence.
    """
    if not name:
        return False
    try:
        from local_operator.agent_profiles import resolve_profile_or_specialist
        from local_operator.agents import AgentRegistry, agents_store_present
        from local_operator.paths import config_dir

        config_root = config_dir()
        # GUARDED so a best-effort probe cannot WRITE: ``AgentRegistry``'s
        # constructor creates ``config_dir`` and ``config_dir/agents`` when
        # missing (then runs its migrations), and a launch-path check has no
        # business creating directories (post-merge review of #1597; round 1,
        # finding 2). ``agents_store_present`` counts BOTH shapes — the
        # per-agent tree and a legacy ``agents.json`` — so a legacy root still
        # resolves its registered roles here rather than being reported as
        # "no declaration" (round 1, finding 3); a truly fresh root constructs
        # nothing and the packaged seeds resolve without a registry. The same
        # predicate guards the sibling writer in
        # ``exec_startup.resolve_startup``, which fires before this probe.
        #
        # Flagged for review, not settled here: on a legacy ``agents.json``
        # root the construction this predicate permits still runs the
        # registry's migrations — a write triggered by a name check. That
        # migration is what the session side runs too, so name resolution
        # stays identical; whether a launch-time CHECK should be what triggers
        # a migration is a question for the store's design, not this probe.
        registry = AgentRegistry(config_root) if agents_store_present(config_root) else None
        _kind, profile, _prompt, _display = resolve_profile_or_specialist(name, registry=registry)
    except Exception:  # noqa: BLE001 — an odd registry means "no declaration"
        return False
    return profile is not None and bool(profile.tools)


def _deny_trapped_advisory(args: ExecArgs) -> str | None:
    """The launch advisory for a run whose approval calls nobody can answer.

    A headless run has no terminal to prompt on, and ``--background``'s worker
    is spawned with ``stdin=DEVNULL``, so every write/exec tool call is denied
    by the CLI's headless gate unless one of the flags that makes the run
    answerable is present: ``--control`` (cards park for a supervisor),
    ``--yolo`` (approve every tier inline), or a tool declaration (``--tools``,
    or a role whose own allow-list resolves through ``--profile`` — both are
    the declaration ``exec_startup.apply_startup`` lets stand as the approval
    exactly where nobody can be asked). Before this advisory the operator's
    first evidence of the trap was a transcript of denials for calls no user
    had seen; this is the same news told BEFORE the run, which is the half
    that was missing.

    Returns the two-line advisory to print to stderr, or ``None`` when the run
    can be approved on (a foreground tty) or a flag answers the gate. A tty
    gets ``None`` because its y/N prompt IS the answer, and the advisory must
    not decorate a run that already works.
    """
    if args.control or args.yolo:
        return None
    from local_operator.exec_startup import parse_tool_inventory

    if parse_tool_inventory(getattr(args, "tools", None)):
        return None
    if _declared_by_profile(getattr(args, "profile", None)):
        return None
    if not args.background and (sys.stdin is not None and sys.stdin.isatty()):
        return None
    return (
        "Warning: this run cannot ask for approval (no terminal attached), so "
        "any tool call that needs approval will be denied.\n"
        "  Remedies: --control parks cards for a supervisor; --yolo auto-approves "
        "every tier; --tools NAME[,NAME] pre-approves the listed tools and bounds "
        "this run's reach to them."
    )


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

    refusal = reject_detached_supervisor_fd(args)
    if refusal is not None:
        print(f"exec failed: {refusal}", file=sys.stderr)
        return 1
    has_loop = args.loop is not None or args.loop_goal is not None
    try:
        team = resolve_startup(args)
        command = resolve_prompt(command, has_loop=has_loop)
    except (ValueError, OSError) as exc:
        print(f"exec failed: {exc}", file=sys.stderr)
        return 1
    # The positional is optional so a loop-only or piped run can omit it
    # (argparse cannot express "required unless --loop/--loop-goal/stdin"), so
    # the requirement is enforced here — naming the ways to supply one rather
    # than reporting a bare "empty prompt". Each refusal below answers the
    # BELIEF that produced it, not merely the missing argument: a user who
    # typed --goal expected the TUI's /goal, which also sends the text.
    if not command.strip() and not has_loop:
        if args.clear_goal:
            print(
                "exec failed: --clear-goal adjusts a run, it does not start one. "
                "Pair it with --resume SESSION_ID plus a prompt or --loop.",
                file=sys.stderr,
            )
        elif args.goal:
            print(
                "exec failed: --goal sets the objective but does not start work "
                "(unlike the TUI's /goal). Add a prompt, pipe one on stdin (or "
                "'-'), or run a loop with --loop N.",
                file=sys.stderr,
            )
        else:
            print(
                "exec failed: no prompt. Pass one as an argument, pipe it on stdin "
                "(or '-'), or run a loop with --loop/--loop-goal",
                file=sys.stderr,
            )
        return 1
    if args.background:
        return _spawn_background(command, args)

    # Foreground, non-tty only: say what will happen before it does. The
    # background half prints inside its launch receipt (see
    # ``_spawn_background``); a tty run gets nothing because its y/N prompt is
    # the answer the advisory would otherwise describe as missing.
    advisory = _deny_trapped_advisory(args)
    if advisory is not None:
        print(advisory, file=sys.stderr)

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
