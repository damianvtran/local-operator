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
import struct
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


def _inside_app_bundle(path: str) -> bool:
    """Whether ``path`` names something inside a macOS ``.app`` bundle.

    Mirrors ``insideAppBundle`` in local-operator-ui's
    ``src/main/python-bytecode-cache.ts``, deliberately: that predicate is the
    other half of this module's safety argument, and the two must agree about
    what "inside a bundle" means or one of them is guarding a door the other
    leaves open.

    A path-SEGMENT test rather than a resolve-and-prefix test, because the
    bundle root is not known here: any segment ending in ``.app`` is a bundle,
    including another application's.
    """
    return any(
        len(segment) > 4 and segment.endswith(".app")
        for segment in path.replace("\\", "/").split("/")
    )


def _pyc_matches_source(source: Path, cache: Path) -> bool:
    """Whether CPython would ACCEPT ``cache`` for ``source``.

    The header, not the mtimes: CPython's ``_validate_timestamp_pyc`` compares
    the source's mtime and size *as recorded inside the ``.pyc``* against the
    source, so a cache can be newer than its source and still be rejected —
    that is exactly what ``cp -p``, ``rsync -a`` and a restored backup produce
    (content changed, mtime moved backwards). A mtime comparison reports those
    "warm", and the consequence is a self-disabling loop rather than a missed
    optimisation: CPython recompiles from source for ever, the probe keeps
    calling the cache current, and nothing repairs it until some unrelated
    edit moves a source mtime forward.

    Returns True for a hash-based ``.pyc`` (invalidated by a source hash this
    function would have to read the whole file to check, and which no tool in
    this repo writes), and for the ``check_source`` flag, which means the
    header asks not to be validated at all.
    """
    try:
        with cache.open("rb") as handle:
            head = handle.read(16)
        stat = source.stat()
    except OSError:
        return False
    if len(head) < 16 or head[:4] != importlib.util.MAGIC_NUMBER:
        return False
    flags = int.from_bytes(head[4:8], "little")
    if flags & 0b11:  # hash-based, or check_source: no timestamp to compare
        return True
    stored_mtime, stored_size = struct.unpack("<II", head[8:16])
    return (
        stored_mtime == int(stat.st_mtime) & 0xFFFFFFFF and stored_size == stat.st_size & 0xFFFFFFFF
    )


def cache_is_cold() -> bool:
    """Whether at least one probed module has no usable bytecode cache.

    Cold means "a process under the refusal flag is compiling this from source
    right now", which is the only condition worth spawning for.

    Two states are NOT cold, and both matter:

    * **No prefix.** A real cache sits beside the sources and this module has no
      business writing into it (see the module docstring), so declaring it warm
      keeps the caller's decision in one place instead of leaving a second gate
      to forget.
    * **A probe module that is not installed.** ``tiktoken`` is the ``tokenizer``
      extra, and a name can also be renamed or removed by a later version. "Not
      there" is *not applicable*, not *cold*: reporting it cold would spawn a
      compiler at every daemon and TUI boot, for ever, on a host whose cache is
      perfectly warm.
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
                continue
            origin = getattr(spec, "origin", None) if spec is not None else None
        if not isinstance(origin, str) or not origin.endswith(".py"):
            continue
        source = Path(origin)
        cache = Path(importlib.util.cache_from_source(str(source)))
        if not _pyc_matches_source(source, cache):
            return True
    return False


def _compile_loaded_modules(sources: Iterable[str | Path] | None = None) -> int:
    """Compile every loaded source file whose cache is missing or stale.

    Runs in the SUBPROCESS.

    THE SKIP IS EXPLICIT, and the first version of this function was wrong to
    assume otherwise: ``py_compile.compile`` has no freshness check at all — it
    recompiles and rewrites the target every time it is called (verified: the
    ``.pyc`` mtime advances on a second call with an unchanged source). So a
    warm run would have been a full recompilation of the loaded graph, the very
    cost the module exists to remove, on every spawn. :func:`_pyc_matches_source`
    is the check that makes the docstring true: a cache CPython will accept is
    left alone, and the run becomes a header read per module.

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
            cache = Path(importlib.util.cache_from_source(str(path)))
            if _pyc_matches_source(path, cache):
                continue
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

    NEVER RAISES, and for every reason rather than the one: the conditions
    below, the probe, the thread, and the subprocess are each wrapped. That is
    load-bearing at the call sites — the daemon runs it inside its FastAPI
    ``lifespan``, where a raise fails startup, and one of the two TUI callers
    reaches it from a function documented as never raising.

    Returns the thread so a caller (or a test) can join it; ``None`` means
    there was nothing to do and no process was started.
    """
    try:
        return _warm_bytecode_cache()
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("bytecode cache warm skipped", exc_info=True)
        return None


def _warm_bytecode_cache() -> threading.Thread | None:
    if not sys.dont_write_bytecode:
        # Nothing refuses the write, so the first process to import each module
        # caches it on the way past. There is no cold state to repair.
        return None
    if sys.pycache_prefix is None:
        # No redirect: a write here would land beside the source. See the
        # module docstring — this is the case the app's own variable exists to
        # make impossible, and guessing otherwise is not ours to do.
        return None
    if _inside_app_bundle(sys.pycache_prefix):
        # The prefix is a REDIRECT, and this module's whole safety argument is
        # that the write it performs lands in per-user state instead of inside
        # a code-sealed bundle. A prefix that points into one would invert
        # that, and the value is reachable: the backend sources the operator's
        # shell rc files, so a stray `export` reaches every python the app
        # runs. local-operator-ui sanitises the same value on its side with the
        # same predicate; refusing here means neither has to trust the other.
        #
        # BEFORE the writability check, which CREATES the directory: a refusal
        # that leaves a new empty directory inside a signed bundle is not a
        # refusal.
        logger.debug("bytecode cache prefix points inside an app bundle; declining")
        return None
    if not _prefix_is_writable(sys.pycache_prefix):
        # AN UNWRITABLE PREFIX IS A PERMANENT COLD STATE, and the probe would
        # report it cold for ever: every daemon and TUI boot would spawn a
        # compiler that writes nothing, which is a cost this module invented.
        # Declining here is the only place that can tell the difference between
        # "cold, and a write would fix it" and "cold, and no write can land".
        logger.debug("bytecode cache prefix is not writable; declining")
        return None
    if not cache_is_cold():
        return None

    try:
        thread = threading.Thread(target=_run_child, name="lop-bytecode-warm", daemon=True)
        thread.start()
    except RuntimeError:
        # `can't start new thread`: thread or fd exhaustion. The runtime child
        # calls this FIRST in `main()` (see session/runtime/process.py), so a
        # refusal here must not become a boot failure — the whole contract is
        # that a warm-up is never the thing that breaks.
        logger.debug("bytecode cache warm could not start a thread", exc_info=True)
        return None
    return thread


def _child_argv(optimize: int | None = None) -> list[str]:
    """How this module re-enters Python: ``-P``, the optimize rung, then ``-c``.

    ``-P`` IS NOT OPTIONAL. A ``-c`` child gets its cwd at ``sys.path[0]``,
    which is searched BEFORE the ``PYTHONPATH`` entry :func:`_run_child`
    prepends — so a parent running from a checkout of this project (the
    operator's ``~/local-operator``, or a daemon spawned with a checkout as its
    cwd) would have the child import and compile a DIFFERENT tree than the one
    the parent reads. The symptom is not an error: the warm reports success, the
    installed tree stays cold, and the optimisation silently achieves nothing.
    This is the same hazard ``local_operator.interpreter`` documents, and the
    flag is taken from there rather than respelled so the two cannot drift.

    The OPTIMIZE RUNG is the other half of "write the cache the parent will
    read". ``-O``/``-OO`` move the cache filename to ``*.opt-1.pyc``/``opt-2``,
    so a child spawned without them writes a file an optimised parent never
    looks for — the probe would find it missing, and every boot would spawn a
    compiler whose output is unreachable.
    """
    from local_operator.interpreter import python_argv

    level = sys.flags.optimize if optimize is None else optimize
    flags = ["-" + "O" * level] if level else []
    return python_argv(*flags, "-c", _CHILD_SOURCE)


def _prefix_is_writable(prefix: str) -> bool:
    """Whether a bytecode write COULD land under ``prefix``.

    Creates the directory on the way, which is not a side effect worth
    avoiding: CPython makes exactly this directory, lazily, on the first write,
    and a cold prefix normally does not exist yet — so an ``os.access`` on it
    would answer "unwritable" for the one case the warm exists to fix. A
    failure to create it is the answer we actually want (a read-only volume, a
    denied parent), and it is reported rather than raised.
    """
    try:
        Path(prefix).mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(prefix, os.W_OK)


def _run_child() -> None:
    """Spawn the compiler subprocess. Never raises.

    REFUSES WITHOUT A REDIRECT, and that refusal is the safety property rather
    than a formality: this function drops ``PYTHONDONTWRITEBYTECODE`` from the
    child's environment (it must, or the child cannot write at all), so a child
    spawned with no ``PYTHONPYCACHEPREFIX`` would write ``__pycache__`` beside
    every module it imports — the repo and the venv included — which is exactly
    the write the module docstring exists to prevent. The public entry point
    cannot reach here without a prefix; this guard is what makes that true of
    ANY caller, including a future one and a test.
    """
    if not sys.pycache_prefix:
        logger.debug("bytecode cache warm needs a redirect; declining")
        return
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
            _child_argv(),
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
