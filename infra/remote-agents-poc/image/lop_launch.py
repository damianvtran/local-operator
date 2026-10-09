#!/opt/lop/bin/python
"""Run a lop CLI in-process with the provider key read from an inherited pipe.

WHY THIS EXISTS — the read path it closes
-----------------------------------------
The boundary is already documented in this repository, in
``local_operator/tools/shell_env.py``: the strict mode removes the key from a
child's OWN environment and explicitly does not make it unreadable, because "the
parent's environment stays readable from it: Linux — ``cat /proc/$PPID/environ``
does the same", and an unset "closes NOTHING, because ``ps`` and
``/proc/PID/environ`` report the environment a process was STARTED with".

So exporting the key into the ``lop exec`` process's launch environment — which is
what this POC did first — hands it to every bash child the model starts, one
command away. This launcher takes it out of that environment entirely:

* the entrypoint writes the key to a pipe and never exports it;
* this process reads it from the pipe and sets ``os.environ[<provider var>]``;
* ``setenv`` writes to the heap copy of the environment, and ``/proc/PID/environ``
  exposes the INITIAL image the kernel copied at ``exec``, so the key is in this
  process's memory and in no process's initial environment.

WHAT THIS DOES NOT CLOSE (stated, not implied)
---------------------------------------------
The key is in this process's memory, and every descendant runs as the same uid
10001, so a same-uid process that can read ``/proc/<pid>/mem`` can still recover
it (subject to ``yama/ptrace_scope``). That residual is measured by the
entrypoint's watcher (probe ``4e_proc_environ``) rather than asserted away.

ORDER MATTERS
-------------
``procname.reexec_branded`` replaces the process with the environment passed
through UNTOUCHED, so it has to run BEFORE the key is set. For a launch through
this script it is a no-op anyway — ``is_own_launch()`` accepts only a real
``lop``/``lo``/``local-operator`` console script, and ``sys.orig_argv[1]`` here is
this file — but the ordering is what keeps that true if the guard ever widens.
The self-check below reads ``/proc/self/environ`` after the key is set and
reports whether the initial image is clean, so a re-exec that would re-expose it
shows up in the container log instead of being taken on trust.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="start a lop CLI with an fd-delivered key")
    parser.add_argument(
        "--key-fd",
        type=int,
        default=3,
        help="inherited descriptor carrying the key; the entrypoint writes it and closes it",
    )
    parser.add_argument(
        "--provider-env",
        default="",
        help="provider variable to set IN-PROCESS (never passed to a child as its own env)",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help=(
            "write this process's pid here so the entrypoint's environ watcher can "
            "aim its memory probe at the agent rather than guess it from a cmdline "
            "that the process renames under itself"
        ),
    )
    parser.add_argument(
        "--selftest-child",
        action="store_true",
        help=(
            "spawn one bash child the way the product's tools do (FILTERED) and report "
            "what it inherited, instead of starting the CLI (the coexistence check)"
        ),
    )
    parser.add_argument(
        "--selftest-child-inherited",
        action="store_true",
        help=(
            "the same child with the environment INHERITED rather than filtered — the "
            "shape group_reaper/memory_guard use — so 4e's red case is measurable "
            "through this same code path"
        ),
    )
    parser.add_argument(
        "--child-sleep",
        type=int,
        default=20,
        help=(
            "seconds the self-test child lives; it must outlive the watcher's sampling "
            "window or the run proves nothing about coexistence (agent review round 3)"
        ),
    )
    parser.add_argument("lop_argv", nargs=argparse.REMAINDER, help="arguments for the lop CLI")
    return parser.parse_args(argv)


def _read_key(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _replace_key_channel(fd: int) -> None:
    """Leave the launched CLI with a clean stdin rather than a half-read pipe.

    ``lop exec`` must read stdin as an ordinary unattended run (``/dev/null``) —
    the ``--tools`` declaration only stands as its own approval where nobody can
    be asked — so the key channel is replaced, not left dangling.
    """
    if fd == 0:
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    else:
        os.close(fd)


def _self_environ_clean(key: str) -> bool:
    """Is the key absent from THIS process's initial environment image?

    The check is the only honest way to know whether a re-exec happened between
    setting the key and now: ``/proc/self/environ`` is the image the kernel copied
    at ``exec``, so a ``True`` here means this process was never STARTED with the
    key, however it was set afterwards.
    """
    try:
        with open("/proc/self/environ", "rb") as handle:
            return key.encode() not in handle.read()
    except OSError:
        return True  # no procfs (a macOS dev box): nothing to claim either way


def _selftest_child(provider_env: str, key: str, *, filtered: bool, sleep_seconds: int) -> int:
    """Spawn one bash child and report what it inherited, from the real code paths.

    WHY THIS EXISTS (agent review rounds 2 and 3): the five acceptance runs are mock, so
    the agent spawns no tool child, and their green 4e reading cannot cover coexistence.
    Two shapes are measured here, through the SAME launcher and the same child argv, so
    "green" and "red" differ in exactly one thing — the environment the child is given:

    * ``filtered`` — ``local_operator.tools.shell_env.child_environment``, the function
      the bash and eval tools build their children with. Exit 0 iff the child ran and
      the key is not in it.
    * ``inherited`` — ``{**os.environ, "LC_ALL": "C"}``, which is what
      ``tools/group_reaper.py:229`` passes to ``ps`` and what
      ``memory_guard._default_runner`` does by passing no ``env=`` at all. Exit 0 iff
      the child ran and the key IS in it, i.e. the shape is as leaky as the probe must
      be able to see. Without this half, a green coexistence reading means nothing.

    The child is a real ``sh -c 'sleep N'`` whose ``/proc/<pid>/environ`` the entrypoint's
    watcher can read while it is alive, and ``--child-sleep`` exists because a child that
    dies before the first sample is not a coexistence case at all.
    """
    if filtered:
        from local_operator.tools.shell_env import child_environment

        env = child_environment()
    else:
        env = {**os.environ, "LC_ALL": "C"}
    argv = ["sh", "-c", f"sleep {sleep_seconds}"]
    child = subprocess.Popen(argv, env=env)
    child_rc = child.wait()
    inherited_by_name = bool(provider_env) and provider_env in env
    inherited_by_value = bool(key) and any(key in value for value in env.values())
    report = {
        "child_argv": argv,
        "child_env_entries": len(env),
        "child_exit": child_rc,
        "child_pid": child.pid,
        "child_sleep_seconds": sleep_seconds,
        "expected_key_in_child_env": not filtered,
        "key_value_in_child_env": inherited_by_value,
        "mode": "filtered" if filtered else "inherited",
        "parent_self_environ_clean": _self_environ_clean(key),
        "provider_var_in_child_env": inherited_by_name,
    }
    report["pass"] = (
        child_rc == 0
        and inherited_by_value == report["expected_key_in_child_env"]
        and report["parent_self_environ_clean"]
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    # BEFORE the key is set, and before anything else can re-exec: see the module
    # docstring. Both the plan and its result are logged, because "it does not
    # re-exec here" is a claim the reader of a container log should be able to check.
    from local_operator.procname import reexec_branded, should_reexec

    print(f"lop-launch: branding re-exec plan: {should_reexec()!r}", file=sys.stderr, flush=True)
    reexec_branded()

    raw = _read_key(args.key_fd)
    _replace_key_channel(args.key_fd)
    key = raw.decode("utf-8", "replace").strip()
    if args.pid_file is not None:
        args.pid_file.write_text(str(os.getpid()), encoding="utf-8")
    # The wording matters: a mock run passes no provider variable, so a delivered key is
    # correctly left unset — which is NOT the same as a key that never arrived, and the
    # first version of this line said "no key delivered" for both (measured, in the
    # container log).
    print(
        f"lop-launch: read {len(raw)} key byte(s) from fd {args.key_fd}; "
        f"provider_env={args.provider_env!r}; set={bool(args.provider_env and key)}",
        file=sys.stderr,
        flush=True,
    )
    if args.provider_env and key:
        os.environ[args.provider_env] = key
        print(
            f"lop-launch: set {args.provider_env} in-process ({len(key)} bytes); "
            f"/proc/self/environ clean: {_self_environ_clean(key)}",
            file=sys.stderr,
            flush=True,
        )

    lop_argv = list(args.lop_argv)
    if args.selftest_child or args.selftest_child_inherited:
        # The key is already set in this process's memory by the block above, which is
        # exactly the state the checks need: a parent holding it, and a child given
        # either the filtered environment or the inherited one.
        return _selftest_child(
            args.provider_env,
            key,
            filtered=args.selftest_child,
            sleep_seconds=args.child_sleep,
        )
    if lop_argv and lop_argv[0] == "--":
        lop_argv = lop_argv[1:]
    # `cli.main()` parses `sys.argv` itself and takes no argument, so the launch
    # line is reconstructed here rather than passed through.
    sys.argv = ["lop", *lop_argv]
    from local_operator.cli import main as cli_main

    return int(cli_main())


if __name__ == "__main__":
    raise SystemExit(main())
