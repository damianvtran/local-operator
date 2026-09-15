"""Give a process that refuses to write bytecode the cache it can still READ.

THE PROBLEM, MEASURED
=====================
A process started with ``PYTHONDONTWRITEBYTECODE=1`` compiles every import and
keeps nothing: ``sys.dont_write_bytecode`` is true before the first import, so
``SourceFileLoader.set_data`` is never reached. Add ``PYTHONPYCACHEPREFIX`` and
the read path moves too — ``cache_from_source`` returns the path under the
prefix, so the ``.pyc`` CPython would otherwise have found beside the source
(stdlib included) is invisible from here. Every fresh process therefore
recompiles its whole import graph from source.

That is the environment the Local Operator desktop app spawns every Python in
(``local-operator-ui``'s ``python-bytecode-cache.ts``; it sets both variables
deliberately, to keep CPython from writing ``__pycache__`` into its code-sealed
``.app``). The app's own note calls the cost "paid on an app start". It is not:
the daemon passes its environment to every runtime child
(``session/runtime/launch.py`` copies ``os.environ``), so the cost is paid once
per ATTACH.

Measured on this machine, one runtime-child-shaped process, import graph and
first turn only:

=========================================  ==========================
configuration                              time to first streamed token
=========================================  ==========================
``PYTHONDONTWRITEBYTECODE=1``, cache cold   **1,277 ms**
same, cache populated once                  **174 ms**
cache populated by this module              137 ms
=========================================  ==========================

501 modules and 749 ms of ``compile`` inside that one turn. The reader is
unaffected by the flag — only the WRITER is refused — so populating the cache
ONCE removes the cost from every later process, which is why this belongs to
the long-lived daemon rather than to each child.

WHY THIS IS SAFE TO WRITE
=========================
Only under a ``PYTHONPYCACHEPREFIX``, and that is the whole safety argument.
The prefix is a redirect: it decides where a bytecode write goes, and the app
sets it precisely so writes land in per-user state instead of inside the signed
bundle. With no prefix configured this module does NOTHING, because then a
write would land in ``__pycache__`` beside the source — which, for an install
inside an ``.app``, is the unsealing the app's variable exists to prevent. An
interpreter that writes its own cache is CPython behaving normally; the flag is
the thing that is unusual here, and this module is its complement, not an
override of it.

The write happens in a SUBPROCESS with only that one variable dropped, so
nothing about in-process flag semantics matters and the caller's event loop is
untouched. ``py_compile`` writes through ``cache_from_source``, i.e. to the
same path the importing process will look in, and writes atomically (temp file
plus rename), so two processes racing this produce one cache, not a torn one.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

#: Modules whose cached bytecode is probed to decide whether the cache is warm.
#: Deliberately spread across the graph — our own package, the two heaviest
#: third-party imports, and the tokenizer — so a partially populated cache (the
#: state a previous partial run or a version bump leaves behind) is reported
#: cold and gets completed rather than trusted.
_PROBE_MODULES: tuple[str, ...] = (
    "local_operator.session.session",
    "local_operator.providers.clients",
    "local_operator.tools.builtin",
    "pydantic",
    "httpx",
    "httpcore",
    "tiktoken",
)

#: A generous ceiling for a one-shot compile of the import graph. It bounds a
#: pathological environment (a network filesystem) rather than a real run:
#: measured at well under a second locally, and the subprocess is killed on
#: expiry so a hung compiler cannot pin a daemon thread for the process
#: lifetime.
_CHILD_TIMEOUT_S = 180.0


#: Run in the subprocess. Imports the graph the real entry points import — this
#: module must never enumerate modules by hand, or the cache would drift behind
#: the code it is supposed to serve — then compiles every loaded source file.
#:
#: NARROW ON PURPOSE, and the reason is measured. Compiling the whole installed
#: package (``rglob``) pulls in the TUI's ~40k-line ``app.py`` and every tool
#: module: 1,341 files and ~40 s, for modules a runtime child never imports. The
#: graph below is what a child actually touches between ``exec`` and its first
#: streamed token, and it is the number the benchmark reports. Anything left out
#: is compiled on first import by whoever needs it — the cost is not lost, only
#: moved off the attach path, which is the entire point.
_CHILD_SOURCE = """
import sys

import local_operator
from local_operator import bytecode

try:
    from local_operator.session_factory import warm_session_imports

    warm_session_imports()
except Exception:
    pass
for name in (
    "local_operator.session.runtime.process",
    "local_operator.session.runtime.serving",
    "local_operator.tools.registry",
    "local_operator.compaction.tokens",
):
    try:
        __import__(name)
    except Exception:
        pass
try:
    from local_operator.compaction.tokens import warm_tokenizer

    warm_tokenizer()
except Exception:
    pass

print(bytecode._compile_loaded_modules())
"""


def _cache_is_current(source: Path) -> bool:
    """Whether ``source``'s cached bytecode exists and is not older than it.

    Freshness, not mere existence, and it matters in exactly one place: an
    install that has moved on (``lop-update``) leaves every ``.pyc`` in place
    while its source files are newer. A process under the refusal flag never
    repairs that — it cannot write — so an existence-only probe would report a
    permanently stale cache as warm and switch this module off for good.

    The comparison is deliberately one-directional (cache at least as new as
    the source). CPython's own rule compares the source's mtime and size
    recorded in the pyc header, which is stricter; being *stricter here* would
    only cost, at worst, one extra compile of a file that was fine.
    """
    cache = Path(importlib.util.cache_from_source(str(source)))
    try:
        return cache.stat().st_mtime >= source.stat().st_mtime
    except OSError:
        return False


def cache_is_cold() -> bool:
    """Whether at least one probed module has no usable bytecode cache.

    Cold means "a process under the refusal flag is compiling this from source
    right now", which is the only condition worth spawning for. An absent
    ``PYTHONPYCACHEPREFIX`` is reported WARM: there is a real cache beside the
    sources and this module has no business writing into it (see the module
    docstring), so declaring it warm keeps the caller's decision in one place
    instead of leaving a second gate to forget.
    """
    if sys.pycache_prefix is None:
        return False
    for name in _PROBE_MODULES:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None) if module is not None else None
        if not isinstance(origin, str) or not origin.endswith(".py"):
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ValueError):
                return True
            origin = getattr(spec, "origin", None) if spec is not None else None
        if not isinstance(origin, str) or not origin.endswith(".py"):
            return True
        if not _cache_is_current(Path(origin)):
            return True
    return False


def _compile_loaded_modules(sources: Iterable[str | Path] | None = None) -> int:
    """Compile every loaded source file; return how many were written.

    Runs in the SUBPROCESS. ``py_compile.compile`` skips a target whose cache is
    already current, so a warm run is a stat per module rather than a
    recompilation — which is what makes it safe to invoke this on a schedule no
    faster than "once per process, in the background".

    Deliberately NOT a directory walk: see the note above ``_CHILD_SOURCE``. The
    subprocess imports the graph first and this compiles exactly what that
    import loaded, so the cache grows to fit the code instead of the code being
    enumerated a second time in here.

    ``sources`` is the TEST SEAM. Production passes nothing and gets the walk
    over ``sys.modules``; a unit test passes one file, because the alternative —
    letting the walk run against a fresh temporary prefix — is a thousand
    compilations of this interpreter's whole graph, which is the very cost the
    module exists to avoid paying twice.
    """
    import py_compile

    if sources is not None:
        candidates = [Path(entry) for entry in sources]
    else:
        candidates = []
        for module in list(sys.modules.values()):
            origin = getattr(module, "__file__", None)
            if isinstance(origin, str) and origin.endswith(".py"):
                candidates.append(Path(origin))

    written = 0
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        try:
            result = py_compile.compile(str(path), doraise=False, quiet=2)
        except Exception:  # noqa: BLE001 — a cache miss is not a failure
            continue
        if result is not None:
            written += 1
    return written


def warm_bytecode_cache_in_background() -> threading.Thread | None:
    """Populate the interpreter's bytecode cache once, off this process's path.

    For long-lived processes: the desktop daemon, and the TUI/CLI through
    ``session_factory.warm_session_imports``. A runtime CHILD should not call
    it — it is short-lived by design, so it would usually die before the
    subprocess finished, and the daemon that spawned it has already done the
    work by then.

    Returns the thread so a caller (or a test) can join it; ``None`` means
    there was nothing to do and no process was started.
    """
    if not sys.dont_write_bytecode:
        # Nothing refuses the write, so the first process to import each module
        # caches it on the way past. There is no cold state to repair.
        return None
    if sys.pycache_prefix is None:
        # No redirect: a write here would land beside the source. See the
        # module docstring — this is the case the app's own variable exists to
        # make impossible, and guessing otherwise is not ours to do.
        return None
    try:
        if not cache_is_cold():
            return None
    except Exception:  # noqa: BLE001 — an unanswerable probe is not a reason to run
        logger.debug("bytecode cache probe failed", exc_info=True)
        return None

    thread = threading.Thread(target=_run_child, name="lop-bytecode-warm", daemon=True)
    thread.start()
    return thread


def _run_child() -> None:
    """Spawn the compiler subprocess. Never raises."""
    try:
        import local_operator

        env = dict(os.environ)
        # The ONE variable dropped, and the reason this is a subprocess at all:
        # the refusal is read by CPython before its first import, so a process
        # that already started with it cannot write bytecode even after the
        # attribute is flipped.
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        # The redirect is RESTATED from ``sys.pycache_prefix`` rather than left
        # to what the child would read from the environment. In production the
        # two agree — the variable is what set the attribute — but ``-X
        # pycache_prefix`` and programmatic callers set only the attribute, and
        # a child that wrote to a DIFFERENT cache than this process reads would
        # be a silent no-op: the compiler would report success and the next
        # process would still recompile.
        if sys.pycache_prefix:
            env["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix
        # The child must import THIS local_operator. An editable install or a
        # source checkout reaches it through the parent's ``sys.path`` entry,
        # which ``-c`` does not replicate.
        package_parent = str(Path(local_operator.__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            package_parent if not existing else package_parent + os.pathsep + existing
        )
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [sys.executable, "-c", _CHILD_SOURCE],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_CHILD_TIMEOUT_S,
            check=False,
        )
        logger.debug("bytecode cache warm wrote %s files", completed.stdout.decode().strip())
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("bytecode cache warm skipped", exc_info=True)
