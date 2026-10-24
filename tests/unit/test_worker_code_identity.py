"""Pure producer-shaped controls: stdlib asyncio only, no Textual/App clocks."""

import asyncio
import functools
import tempfile
import unittest
from pathlib import Path

from tests import worker_code_identity as identity


async def first():
    await asyncio.sleep(0)


async def second():
    await asyncio.sleep(0)


def allowlist():
    return {
        (
            f.__code__.co_filename,
            f.__code__.co_firstlineno,
            f.__code__.co_qualname,
        ): "tests/unit/test_worker_code_identity.py:"
        + str(f.__code__.co_firstlineno)
        + ":"
        + f.__name__
        for f in (first, second)
    }


class FakeWorker:
    def __init__(self, node, work, **kwargs):
        self._task = None
        self.work = work
        self.labels = []

    async def _run(self, app):
        self.labels.append(identity.task_label(asyncio.current_task()))
        await self.work()
        self.labels.append(identity.task_label(asyncio.current_task()))

    def _start(self, app, done_callback=None):
        self._task = asyncio.create_task(self._run(app))


class IdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_ids_registered_before_first_and_every_resume(self):
        old_init, old_start = FakeWorker.__init__, FakeWorker._start
        registry = identity.WorkerCodeIdentity(allowlist())
        try:
            registry.install(FakeWorker)
            workers = [FakeWorker(None, f) for f in (first, second)]
            for worker in workers:
                worker._start(None)
                self.assertNotEqual(registry.label(worker._task), "")
                self.assertEqual(worker.labels, [])  # no first resume yet
            await asyncio.gather(*(w._task for w in workers))
            self.assertNotEqual(workers[0].labels[0], workers[1].labels[0])
            self.assertTrue(all(len(set(w.labels)) == 1 and len(w.labels) == 2 for w in workers))
            self.assertTrue(registry.quality()["qualified"])
        finally:
            registry.close()
        self.assertIs(FakeWorker.__init__, old_init)
        self.assertIs(FakeWorker._start, old_start)
        self.assertIsNone(identity.ACTIVE)
        self.assertFalse(registry.tasks)

    async def test_unsupported_eager_and_overflow_fail_qualification(self):
        registry = identity.WorkerCodeIdentity(allowlist(), cap=1)
        loop = asyncio.get_running_loop()
        try:
            registry.install(FakeWorker)

            class Subclass(FakeWorker):
                pass

            Subclass(None, first)
            workers = [FakeWorker(None, first), FakeWorker(None, second)]
            for worker in workers:
                worker._start(None)
            await asyncio.gather(*(w._task for w in workers))
            loop.set_task_factory(lambda loop, coro, **kw: asyncio.Task(coro, loop=loop, **kw))
            extra = FakeWorker(None, first)
            extra._start(None)
            await extra._task
            self.assertFalse(registry.quality()["qualified"])
            self.assertGreater(registry.counts["unsupported"], 0)
            self.assertGreater(registry.counts["overflow"], 0)
            self.assertGreater(registry.counts["eager"], 0)
            self.assertLessEqual(len(registry.tasks), 1)
        finally:
            loop.set_task_factory(None)
            registry.close()

    async def test_hostile_objects_never_inspected_and_native_wrappers(self):
        class Hostile:
            def __getattribute__(self, name):
                raise AssertionError("attribute inspected")

            def __repr__(self):
                raise AssertionError("repr inspected")

        registry = identity.WorkerCodeIdentity(allowlist())
        self.assertEqual(registry.code_id(Hostile()), identity.UNKNOWN)
        self.assertEqual(
            registry.code_id(functools.partial(first, "private-ignored")), registry.code_id(first)
        )
        coroutine = first()
        try:
            self.assertEqual(registry.code_id(coroutine), registry.code_id(first))
        finally:
            coroutine.close()
        self.assertFalse(registry.quality()["qualified"])

    async def test_real_static_allowlist_decorators_bound_method_and_partial(self):
        # Compile only owned pure source, not an imported App/module. Decoration
        # shifts co_firstlineno to the FIRST decorator, not the async-def line.
        source = (
            "def keep(fn): return fn\n\n"
            "@keep\nasync def task(): return None\n\n"
            "class C:\n    @keep\n    async def method(self): return None\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pkg" / "trusted.py"
            path.parent.mkdir()
            path.write_text(source)
            namespace = {}
            exec(compile(source, str(path.resolve()), "exec"), namespace)
            registry = identity.WorkerCodeIdentity(
                identity.source_allowlist(root, ("pkg/trusted.py",))
            )
            task, method = namespace["task"], namespace["C"]().method
            self.assertEqual(task.__code__.co_firstlineno, 3)
            self.assertEqual(registry.code_id(task), "pkg/trusted.py:3:task")
            self.assertEqual(registry.code_id(method), "pkg/trusted.py:7:C.method")
            self.assertEqual(registry.code_id(functools.partial(method)), registry.code_id(method))
            coroutine = task()
            try:
                self.assertEqual(registry.code_id(coroutine), registry.code_id(task))
            finally:
                coroutine.close()
            self.assertTrue(registry.quality()["qualified"])

    async def test_partial_setup_failure_restores_first_hook(self):
        old_init = FakeWorker.__init__
        real_patch, calls = identity.patch, []

        class FaultPatch:
            @staticmethod
            def object(*args):
                calls.append(args[1])
                if len(calls) == 2:
                    raise RuntimeError("synthetic patch setup")
                return real_patch.object(*args)

        identity.patch = FaultPatch
        registry = identity.WorkerCodeIdentity(allowlist())
        try:
            with self.assertRaises(RuntimeError):
                registry.install(FakeWorker)
            self.assertIs(FakeWorker.__init__, old_init)
            self.assertIsNone(identity.ACTIVE)
        finally:
            identity.patch = real_patch
            registry.close()


if __name__ == "__main__":
    unittest.main()
