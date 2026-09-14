"""Child-side phase timeline for ``scripts/bench_cold_engage.py``.

Copied onto ``PYTHONPATH`` by the benchmark so it runs before
``-m local_operator.session.runtime.process`` resolves anything: the marks it
writes are the only way to see INSIDE the runtime child, whose construction is
most of a cold engage.

Two properties are load-bearing and easy to get wrong:

* **It never forces an eager import.** Patching is driven by a
  ``builtins.__import__`` shim that wraps a target attribute only once the real
  code has imported the module holding it. Importing a target here would move
  the very cost being measured out of the measured window and warm it, which is
  exactly the artifact this instrument exists to avoid.
* **It marks, it does not measure.** Each entry carries ``time.time_ns()`` for
  cross-process ordering against the parent's clock and ``time.perf_counter()``
  for durations inside this process. Every line is flushed immediately, so a
  run killed mid-construction still leaves every earlier mark on disk.

The instrument is inert unless ``LOP_BENCH_ROLE=child`` and this process is
actually the session runtime: the environment is inherited by everything the
runtime spawns (an MCP stub server, a bash tool child), and those must not
append their own marks to the same timeline.
"""

from __future__ import annotations

import builtins
import functools
import inspect
import json
import os
import sys
import time
from typing import Any

_ACTIVE = os.environ.get("LOP_BENCH_ROLE") == "child" and bool(os.environ.get("LOP_BENCH_TIMELINE"))
if _ACTIVE:
    # Gate on THIS process being the session runtime, not on the env alone.
    _argv = " ".join(getattr(sys, "orig_argv", sys.argv))
    _ACTIVE = "local_operator.session.runtime.process" in _argv

if _ACTIVE:
    _PATH = os.environ["LOP_BENCH_TIMELINE"]
    _handle = open(_PATH, "a", buffering=1, encoding="utf-8")
    _seq = 0

    def mark(label: str, detail: object = None) -> None:
        global _seq
        _seq += 1
        try:
            _handle.write(
                json.dumps(
                    {
                        "seq": _seq,
                        "label": label,
                        "wall_ns": time.time_ns(),
                        "perf_s": time.perf_counter(),
                        "detail": detail if isinstance(detail, dict) else None,
                    }
                )
                + "\n"
            )
        except Exception:  # noqa: BLE001 — the instrument must never break a run
            pass

    #: (module, attribute, marker label). Applied lazily, post-import.
    _FUNCTION_TARGETS = [
        ("local_operator.session.runtime.process", "amain", "child.amain"),
        (
            "local_operator.session.runtime.process",
            "_drain_inbox_into",
            "child.drain_inbox",
        ),
        (
            "local_operator.session.runtime.serving",
            "spawn_owned_session",
            "child.spawn_owned_session",
        ),
        ("local_operator.session_factory", "create_session", "child.create_session"),
        ("local_operator.session_factory", "_prepare", "child.prepare"),
        (
            "local_operator.session_factory",
            "wire_mcp_into_session",
            "child.wire_mcp",
        ),
        (
            "local_operator.session_factory",
            "_start_store_maintenance",
            "child.store_maintenance",
        ),
        ("local_operator.session.runtime.registry", "publish", "child.record_write"),
        ("local_operator.session_lease", "acquire_session_lease", "child.lease_acquire"),
        ("local_operator.session.session", "Session", "child.Session___init__"),
    ]

    #: (module, class, method, label) — class-method targets, patched once the
    #: class's module is imported. ``_open_mcp_wiring_gate`` is the publication
    #: latch a gated deferred wiring waits on; marking it is what separates
    #: "the wiring was gated" from "the wiring happened to run later".
    _METHOD_TARGETS = [
        (
            "local_operator.session.runtime.server",
            "RuntimeServer",
            "start_in_process",
            "child.start_in_process",
        ),
        ("local_operator.session.runtime.server", "RuntimeServer", "_serve", "child.serve"),
        (
            "local_operator.session.runtime.server",
            "RuntimeServer",
            "_open_mcp_wiring_gate",
            "child.mcp_gate_open",
        ),
        (
            "local_operator.session.runtime.registry",
            "RecordPublisher",
            "__init__",
            "child.RecordPublisher_init",
        ),
    ]

    def _wrap(fn, label):  # type: ignore[no-untyped-def]
        if getattr(fn, "_lo_bench_wrapped", False):
            return fn
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                mark(label + ".enter")
                try:
                    return await fn(*args, **kwargs)
                finally:
                    mark(label + ".exit")

            async_wrapper._lo_bench_wrapped = True  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            mark(label + ".enter")
            try:
                return fn(*args, **kwargs)
            finally:
                mark(label + ".exit")

        wrapper._lo_bench_wrapped = True  # type: ignore[attr-defined]
        return wrapper

    _done: set[tuple[str, str, str, int]] = set()
    #: Remaining patches. Once zero, the import shim degenerates to one
    #: attribute test per import. Unconditional patching measured 1.8 s of pure
    #: probe overhead inside the window being measured — the class of artifact
    #: this instrument exists to avoid, so the counter is not an optimisation.
    _pending = [len(_FUNCTION_TARGETS) * 2 + len(_METHOD_TARGETS)]

    def _twin_modules(mod_name: str) -> list[Any]:
        """Every module object that could hold this attribute.

        ``python -m local_operator.session.runtime.process`` loads the module
        TWICE: once under its real name (by the spec lookup that finds its
        code) and once as ``__main__`` (the copy whose globals ``main()`` and
        ``amain()`` actually resolve against). Patching only the real-name copy
        silently misses the entry points that live in it.
        """
        out = []
        named = sys.modules.get(mod_name)
        if named is not None:
            out.append(named)
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and main_mod is not named:
            out.append(main_mod)
        return out

    def _apply_pending() -> None:
        for mod_name, attr, label in _FUNCTION_TARGETS:
            targets = _twin_modules(mod_name)
            if not targets:
                continue
            for index, mod in enumerate(targets):
                key = (mod_name, attr, "fn", index)
                if key in _done:
                    continue
                target = getattr(mod, attr, None)
                if target is None:
                    continue
                if mod_name == "local_operator.session.session" and attr == "Session":
                    init = getattr(target, "__init__", None)
                    if init is None:
                        continue
                    setattr(target, "__init__", _wrap(init, label))
                else:
                    setattr(mod, attr, _wrap(target, label))
                _done.add(key)
                _pending[0] -= 1
        for mod_name, cls_name, method, label in _METHOD_TARGETS:
            key = (mod_name, cls_name + "." + method, "m", 0)
            if key in _done:
                continue
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            target = getattr(cls, method, None)
            if target is None or getattr(target, "_lo_bench_wrapped", False):
                continue
            setattr(cls, method, _wrap(target, label))
            _done.add(key)
            _pending[0] -= 1

    _orig_import = builtins.__import__

    def _probe_import(
        name, globals=None, locals=None, fromlist=(), level=0
    ):  # type: ignore[no-untyped-def]
        module = _orig_import(name, globals, locals, fromlist, level)
        # Two cheap gates, in this order: a bare integer test (the common case
        # once everything is patched, and the only check a third-party import
        # ever pays), then a name test that skips the stdlib and site-packages
        # bulk. Every target lives under ``local_operator``, reached either by
        # an absolute import or by a relative one (``level`` non-zero).
        if _pending[0] > 0 and (
            name.startswith("local_operator")
            or (
                level != 0 and str((globals or {}).get("__name__", "")).startswith("local_operator")
            )
        ):
            _apply_pending()
        return module

    builtins.__import__ = _probe_import
    mark("child.sitecustomize", {"argv": sys.argv[:2], "pid": os.getpid()})
