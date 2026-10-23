"""SOURCE-ONLY real Textual producer fragment; integrate before any offhost GO.
No OperatorApp, production payload, raw callback names, frames or locals exported.
All runtime imports/execution are for the future reviewed offhost child ONLY.
"""
import asyncio
import gc
import importlib.metadata
from pathlib import Path
import threading
import time

from textual.app import App
from textual import work
from textual.worker import Worker, WorkerCancelled, WorkerFailed
from tests.loop_callback_spans import CallbackSpans
from tests.worker_code_identity import WorkerCodeIdentity, source_allowlist, task_label, quality


def burn(seconds=0.02):
    start = time.thread_time()
    while time.thread_time() - start < seconds:
        pass
    return dict(thread=threading.get_ident(), cpu_ms=(time.thread_time()-start)*1000)


def witness(rows, phase):
    rows.append(dict(phase=phase, label=task_label(asyncio.current_task()),
                     thread=threading.get_ident()))


async def cpu_one(rows):
    witness(rows, 'one-before')
    first = burn()
    await asyncio.sleep(0.01)
    witness(rows, 'one-after')
    second = burn()
    return [first, second]


async def cpu_two(rows):
    witness(rows, 'two-before')
    first = burn()
    await asyncio.sleep(0.01)
    witness(rows, 'two-after')
    second = burn()
    return [first, second]


class ProbeApp(App):
    @work(exit_on_error=False)
    async def first_worker(self, rows):
        return await cpu_one(rows)

    @work(exit_on_error=False)
    async def second_worker(self, rows):
        return await cpu_two(rows)


async def wait_only(rows, entered, release):
    witness(rows, 'wait-before')
    entered.set()
    await release.wait()
    witness(rows, 'wait-after')


async def offloaded(rows):
    witness(rows, 'offload-before')
    data = await asyncio.to_thread(burn)
    witness(rows, 'offload-after')
    return data


async def cancelled(rows, entered):
    witness(rows, 'cancel-enter')
    entered.set()
    await asyncio.Event().wait()


async def errored(rows):
    witness(rows, 'error-enter')
    await asyncio.sleep(0)
    raise ValueError('synthetic worker control')


def validate_quality(value, observed):
    # Empty/unarmed/UNKNOWN must not silently pass through a JSON projection.
    if set(value) != {'qualified', 'unknown', 'overflow', 'unsupported', 'eager'}:
        raise ValueError('worker quality schema STOP')
    if value['qualified'] is not True or any(type(value[k]) is not int or value[k] != 0 for k in ('unknown','overflow','unsupported','eager')):
        raise ValueError('worker quality invalid STOP')
    if observed < 1:
        raise ValueError('empty worker attribution STOP')


def project_window(window, labels):
    # Only pinned-source allowlist labels. NEVER serialize recorder timer sites,
    # callback coroutine names, user Worker names or complete raw summary.
    spans = []
    for span in window.top_cpu:
        if span.detail in labels:
            spans.append(dict(label=span.detail, cpu_ms=span.cpu_ms, wall_ms=span.wall_ms))
    return dict(callbacks=window.count, cpu_ms=window.cpu_ms, wall_ms=window.wall_ms,
                worker_spans=spans)


async def overhead_pair(allowlist):
    """Fixed zero-work callback cohort, OFF then ON; NOT unbiased cost estimate.
    Measures an extra identity lookup outside the existing callback clocks too.
    Scheduling/order/host variance confound a single pair; never call it zero.
    """
    out = []
    loop = asyncio.get_running_loop()
    for enabled in (False, True):
        registry = WorkerCodeIdentity(allowlist)
        if enabled:
            registry.install(Worker)
        wall, cpu = time.perf_counter(), time.thread_time()
        try:
            with CallbackSpans(loop) as observer:
                async def resume_steps():
                    for _ in range(128):
                        await asyncio.sleep(0)
                await asyncio.create_task(resume_steps())
                await asyncio.sleep(0)
                bucket = observer.take_window()
            out.append(dict(enabled=enabled, requested_callbacks=128,
                            observed_callbacks=bucket.count,
                            outer_cpu_ms=(time.thread_time()-cpu)*1000,
                            outer_wall_ms=(time.perf_counter()-wall)*1000))
        finally:
            registry.close()
    return out


async def run(root, relative):
    # Integrator MUST bind this path + helper/observer/Textual producer bytes in
    # the parent manifest. No using a self-reported hash as the expected source.
    if importlib.metadata.version('textual') != '8.2.8':
        raise RuntimeError('Textual version STOP')
    if (root / relative).resolve() != Path(__file__).resolve():
        raise RuntimeError('compiled probe source path STOP')
    allowlist = source_allowlist(root, (relative,))
    registry = WorkerCodeIdentity(allowlist)
    original_init, original_start = Worker.__init__, Worker._start
    old_handle = asyncio.Handle._run
    old_gc = tuple(gc.callbacks)
    rows, windows, workers = [], [], []
    loop = asyncio.get_running_loop()
    if loop.get_task_factory() is not None:
        raise RuntimeError('default task factory required STOP')
    ready, stop = asyncio.Event(), asyncio.Event()
    sampler = None
    result = {}
    registry.install(Worker)  # BEFORE even constructing the Generic Textual App.
    try:
        with CallbackSpans(loop) as observer:
            async def sample():
                windows.append(project_window(observer.take_window(), set(allowlist.values())))
                ready.set()
                while not stop.is_set():
                    await asyncio.sleep(0.002)
                    if len(windows) >= 256:
                        raise RuntimeError('window capacity STOP')
                    windows.append(project_window(observer.take_window(), set(allowlist.values())))
            sampler = asyncio.create_task(sample())
            await ready.wait()  # READY before any native Worker is started.
            app = ProbeApp()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                entered, release = asyncio.Event(), asyncio.Event()
                waiter = app.run_worker(wait_only(rows, entered, release), exit_on_error=False)
                workers.append(waiter)
                try:
                    await entered.wait()
                    wait_wall = time.perf_counter()
                    # Keep the waiter suspended WHILE two decorated native
                    # workers consume CPU. Never attribute that CPU to wait.
                    first_workers = [app.first_worker(rows), app.second_worker(rows)]
                    workers.extend(first_workers)
                    cpu_results = await asyncio.gather(*(w.wait() for w in first_workers))
                    await asyncio.sleep(0.08)
                finally:
                    release.set()
                await waiter.wait()
                wait_wall = (time.perf_counter()-wait_wall)*1000
                off = app.run_worker(offloaded(rows), exit_on_error=False)
                workers.append(off)
                offload_data = await off.wait()
                cancel_entered = asyncio.Event()
                cw = app.run_worker(cancelled(rows, cancel_entered), exit_on_error=False)
                workers.append(cw)
                await cancel_entered.wait()
                cw.cancel()
                cancel_seen = False
                try:
                    await cw.wait()
                except WorkerCancelled:
                    cancel_seen = True
                ew = app.run_worker(errored(rows), exit_on_error=False)
                workers.append(ew)
                error_seen = False
                try:
                    await ew.wait()
                except WorkerFailed:
                    error_seen = True
                await pilot.pause()
            await asyncio.sleep(0)  # trailing native callback completion sample.
            stop.set()
            await sampler  # propagate sampler errors, never suppress them.
            windows.append(project_window(observer.take_window(), set(allowlist.values())))
            observed = [s for w in windows for s in w['worker_spans']]
            validate_quality(quality(), len(observed))
            by_phase = {r['phase']:r for r in rows}
            for prefix in ('one', 'two', 'wait', 'offload'):
                a, b = by_phase[prefix+'-before'], by_phase[prefix+'-after']
                if not a['label'] or a['label'] == 'UNKNOWN' or a['label'] != b['label']:
                    raise RuntimeError('per-resume identity STOP')
            if by_phase['one-before']['label'] == by_phase['two-before']['label']:
                raise RuntimeError('two distinct native IDs required STOP')
            if any(r['thread'] != threading.get_ident() for r in rows) or offload_data['thread'] == threading.get_ident():
                raise RuntimeError('native/offload thread proof STOP')
            if not cancel_seen or not error_seen:
                raise RuntimeError('native cancel/error control STOP')
            required = {by_phase[p+'-before']['label'] for p in ('one','two','wait','offload')}
            if not required.issubset({s['label'] for s in observed}):
                raise RuntimeError('missing actual callback span identity STOP')
            cpu_by_id = {label: sum(s['cpu_ms'] for s in observed if s['label'] == label)
                         for label in required}
            one_cpu = cpu_by_id[by_phase['one-before']['label']]
            two_cpu = cpu_by_id[by_phase['two-before']['label']]
            wait_cpu = cpu_by_id[by_phase['wait-before']['label']]
            off_cpu = cpu_by_id[by_phase['offload-before']['label']]
            if not (one_cpu > 0 and two_cpu > 0 and wait_cpu < min(one_cpu, two_cpu)
                    and off_cpu < offload_data['cpu_ms']):
                raise RuntimeError('per-resume CPU attribution control STOP')
            result = dict(quality=quality(), rows=rows, windows=windows,
                          cpu_results=cpu_results, wait_wall_ms=wait_wall,
                          offload=offload_data, cancel_seen=cancel_seen, error_seen=error_seen,
                          ready=True, trailing=True)
    finally:
        stop.set()
        for worker in workers:
            worker.cancel()
        try:
            for worker in workers:
                try:
                    await worker.wait()
                except (WorkerCancelled, WorkerFailed):
                    pass
            if sampler is not None:
                await sampler
        finally:
            registry.close()
        result['restored'] = (Worker.__init__ is original_init and Worker._start is original_start
                              and asyncio.Handle._run is old_handle and tuple(gc.callbacks) == old_gc)
        if not result['restored']:
            raise RuntimeError('producer restoration STOP')
    # Deliberate annotation-on/off callback-only observations, not S2 throughput.
    result['overhead_observations'] = await overhead_pair(allowlist)
    return result
