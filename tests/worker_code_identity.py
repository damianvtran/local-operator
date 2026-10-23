"""Bounded test-only launch-root identities, never worker payloads or names.

Arm BEFORE App construction: CallbackSpans starts too late to see boot workers.
Map the exact native Task synchronously after Worker._start under DEFAULT
(non-eager) task scheduling. Every Handle resume may then look up the same ID.
This does not identify the deepest callee or time across an await.
"""
import ast
import asyncio
from contextlib import ExitStack
import functools
from types import CoroutineType, FunctionType, MethodType
from unittest.mock import patch
import weakref

ACTIVE = None
UNKNOWN = "UNKNOWN"


def source_allowlist(root, relative_paths=("local_operator/tui/app.py",)):
    """Output comes ONLY from pinned repository AST, not runtime code strings."""
    result = {}
    for relative in relative_paths:
        path = root / relative
        tree = ast.parse(path.read_text())
        def visit(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = prefix + node.name
                    if not isinstance(node, ast.ClassDef):
                        line = min([node.lineno] + [d.lineno for d in node.decorator_list])
                        result[(str(path.resolve()), line, name)] = f"{relative}:{line}:{name}"
                    child_prefix = name + ("." if isinstance(node, ast.ClassDef) else ".<locals>.")
                    visit(node.body, child_prefix)
                else:
                    # Definitions under if/try are still static source definitions.
                    for field in ("body", "orelse", "finalbody"):
                        values = vars(node).get(field)
                        if isinstance(values, list):
                            visit(values, prefix)
        visit(tree.body)
    if len(result) > 4096:
        raise ValueError("static allowlist cap")
    return result


class WorkerCodeIdentity:
    def __init__(self, allowlist, cap=256):
        self.allowlist = dict(allowlist)
        self.cap = cap
        self.workers = weakref.WeakKeyDictionary()
        self.tasks = weakref.WeakKeyDictionary()
        self.counts = dict(unknown=0, overflow=0, unsupported=0, eager=0)
        self.stack = None
        self.worker_type = None

    def code_id(self, work):
        # Exact native-type checks: never getattr/hasattr/repr/unwrap an arbitrary
        # callable, nor read partial arguments, coroutine frames or local vars.
        for _ in range(4):
            if type(work) is functools.partial:
                work = work.func
            else:
                break
        if type(work) is CoroutineType:
            code = work.cr_code
        elif type(work) is FunctionType:
            code = work.__code__
        elif type(work) is MethodType and type(work.__func__) is FunctionType:
            code = work.__func__.__code__
        else:
            self.counts["unknown"] += 1
            return UNKNOWN
        answer = self.allowlist.get(
            (code.co_filename, code.co_firstlineno, code.co_qualname), UNKNOWN
        )
        if answer == UNKNOWN:
            self.counts["unknown"] += 1
        return answer

    def retain(self, registry, key, label):
        if key not in registry and len(registry) >= self.cap:
            self.counts["overflow"] += 1
            return False
        registry[key] = label
        return True

    def install(self, Worker):
        global ACTIVE
        if ACTIVE is not None:
            raise RuntimeError("worker identity already armed")
        self.worker_type = Worker
        original_init, original_start = Worker.__init__, Worker._start
        if (type(original_init) is not FunctionType or type(original_start) is not FunctionType
                or original_init.__code__.co_varnames[:3] != ("self", "node", "work")
                or original_start.__code__.co_varnames[:3] != ("self", "app", "done_callback")):
            self.counts["unsupported"] += 1
            ACTIVE = self
            return self  # No guessed patching against a different producer API.
        registry = self
        def init(worker, node, work, *args, **kwargs):
            original_init(worker, node, work, *args, **kwargs)
            if type(worker) is not Worker:
                registry.counts["unsupported"] += 1
                return
            registry.retain(registry.workers, worker, registry.code_id(work))
        def start(worker, app, done_callback=None):
            if type(worker) is not Worker:
                registry.counts["unsupported"] += 1
                return original_start(worker, app, done_callback)
            loop = asyncio.get_running_loop()
            if loop.get_task_factory() is not None:
                # Eager/custom scheduling may resume INSIDE create_task before
                # _start returns. Never invent a complete registration there.
                registry.counts["eager"] += 1
                return original_start(worker, app, done_callback)
            result = original_start(worker, app, done_callback)
            task = object.__getattribute__(worker, "__dict__").get("_task")
            if type(task) is not asyncio.Task:
                registry.counts["unsupported"] += 1
                return result
            if worker not in registry.workers:
                registry.counts["unknown"] += 1
            registry.retain(registry.tasks, task, registry.workers.get(worker, UNKNOWN))
            return result
        self.stack = ExitStack()
        try:
            self.stack.enter_context(patch.object(Worker, "__init__", init))
            self.stack.enter_context(patch.object(Worker, "_start", start))
            ACTIVE = self
        except BaseException:
            self.close()
            raise
        return self

    def label(self, owner):
        if type(owner) is not asyncio.Task:
            return ""
        return self.tasks.get(owner, "")

    def quality(self):
        return dict(qualified=not any(self.counts.values()), **self.counts)

    def close(self):
        global ACTIVE
        try:
            if self.stack is not None:
                self.stack.close()
        finally:
            if ACTIVE is self:
                ACTIVE = None
            self.workers.clear()
            self.tasks.clear()


def task_label(owner):
    return ACTIVE.label(owner) if ACTIVE is not None else ""


def quality():
    return ACTIVE.quality() if ACTIVE is not None else dict(qualified=False, unarmed=True)
