"""Owned diagnostic plugin: original tests untouched; NOT a product fix.

Only fixed S2/S3 nodeids may run. App/native imports occur OFFHOST in the
selected hook, never in pure oracle tests. Held-worker gates the TEST dependency,
not a product seam. No releaser thread or timed-success release is permitted.
"""
import asyncio
import json
import os
import re
import threading
import time

import pytest

S2 = "tests/unit/test_tui_responsiveness.py::test_s2_send_to_agent_end_keeps_the_loop_under_the_bar"
S3 = "tests/unit/test_tui_responsiveness.py::test_s3_harvest_with_unlisted_model_and_cold_memo_keeps_the_loop_free"
MODE = os.environ.get("S2S3_MODE")
records = []
report = {"call": None, "assertion": None, "setup_ok": False}


def rec(kind, **values):
    row = dict(kind=kind, **values)
    if len(records) >= 250 or len(json.dumps(records + [row]).encode()) > 180000:
        raise RuntimeError("diagnostic output bound")
    records.append(row)


def clock():
    return time.thread_time(), time.perf_counter()


def span(kind, start, **values):
    cpu, wall = clock()
    rec(kind, cpu_ms=(cpu-start[0])*1000, wall_ms=(wall-start[1])*1000,
        start_wall=start[1], end_wall=wall, thread=threading.get_ident(), **values)


def pytest_collection_modifyitems(items):
    target = S2 if MODE == "s2-original" else S3
    if len(items) != 1 or items[0].nodeid != target:
        raise RuntimeError("exact single named original node required")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    value = result.get_result()
    if value.when == "setup":
        report["setup_ok"] = value.passed
    if value.when == "call":
        report["call"] = "pass" if value.passed else "fail"
        if call.excinfo is not None and call.excinfo.type is AssertionError:
            # Persist ONLY these known static oracle fragments, never traceback,
            # arbitrary repr/message, local variables or application content.
            text = value.longreprtext
            match = re.search(r"harvest blocked for [0-9]+ ms|event loop consumed [0-9]+ ms of CPU|event loop probe wake gap was [0-9]+ ms", text)
            report["assertion"] = match.group(0) if match else "UNKNOWN"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if item.nodeid not in (S2, S3):
        raise RuntimeError("unknown diagnostic node")
    from local_operator.tui.app import OperatorApp
    from local_operator.model import configure
    from textual.pilot import Pilot
    module = item.module
    restorations = []
    gate, entered, finished = threading.Event(), threading.Event(), threading.Event()
    finished.set()
    lock = threading.Lock()
    active = 0
    loop = task = None
    loop_thread = None
    injected = False
    manual_harvests = 0
    measured_pauses = 0
    requests = 0
    idle = burst = settle = None

    def patch(owner, name, replacement):
        restorations.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    original_harvest = OperatorApp._harvest_subagent_costs
    original_pause = Pilot.pause
    original_refresh = configure.refresh_model_info_background
    original_listing = module._SlowListing.__call__
    original_start, original_stop = module.StallRecorder.start, module.StallRecorder.stop
    original_begin, original_end = module.StallRecorder.begin_noise_window, module.StallRecorder.end_noise_window

    def refresh(*args, **kwargs):
        nonlocal requests
        requests += 1
        return original_refresh(*args, **kwargs)

    def held_listing(self, *args, **kwargs):
        nonlocal active
        thread = threading.get_ident()
        with lock:
            active += 1
            finished.clear()
        rec("worker_entry", thread=thread)
        entered.set()
        try:
            if loop_thread is None or thread == loop_thread:
                raise RuntimeError("real listing worker is not independently observed off-loop")
            if not gate.wait(3):
                raise RuntimeError("held worker bound: no causal loop release")
            return original_listing(self, *args, **kwargs)
        finally:
            rec("worker_done", thread=thread)
            with lock:
                active -= 1
                if active == 0:
                    finished.set()

    async def held_witness():
        deadline = time.perf_counter() + 2
        ticks = 0
        while time.perf_counter() < deadline:
            await asyncio.sleep(0.005)
            if entered.is_set() and not gate.is_set():
                ticks += 1
                rec("held_loop_tick", thread=threading.get_ident())
                if ticks == 3:
                    rec("held_release", origin="loop-witness", thread=threading.get_ident(), requests=requests)
                    gate.set()
                    return
        raise RuntimeError("held worker/tick witness missing")

    def harvest(self, *args, **kwargs):
        nonlocal manual_harvests
        caller = __import__("sys")._getframe(1).f_code.co_name
        manual = caller == S3.split("::")[1]
        if manual:
            manual_harvests += 1
        start = clock()
        try:
            return original_harvest(self, *args, **kwargs)
        finally:
            span("harvest", start, manual=manual)

    async def pause(self, *args, **kwargs):
        nonlocal loop, loop_thread, task, injected, measured_pauses, burst, settle
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        caller = __import__("sys")._getframe(1).f_code.co_name
        manual = caller == item.nodeid.split("::")[1]
        measured = manual and manual_harvests > measured_pauses
        if MODE == "s3-held" and task is None:
            task = asyncio.create_task(held_witness())
        if burst is not None and settle is None and manual:
            span("s2_emit_burst", burst)
            settle = clock()
        start = clock()
        try:
            if MODE == "s3-delayed" and measured and not injected:
                injected = True
                rec("injected_async_delay", seconds=0.6)
                await asyncio.sleep(0.6)
            return await original_pause(self, *args, **kwargs)
        finally:
            if measured:
                measured_pauses += 1
            if manual:
                span("pause", start, measured=measured)

    async def start(self, *args, **kwargs):
        rec("recorder_start", thread=threading.get_ident())
        return await original_start(self, *args, **kwargs)

    def begin(self):
        nonlocal idle
        idle = clock()
        return original_begin(self)

    def end(self):
        nonlocal burst
        result = original_end(self)
        if idle is not None:
            span("s2_idle", idle)
        burst = clock()
        return result

    async def stop(self, *args, **kwargs):
        try:
            return await original_stop(self, *args, **kwargs)
        finally:
            if settle is not None:
                span("s2_settle", settle)
            rec("recorder", summary=self.measurement_summary())
            if task is not None:
                await task

    patch(OperatorApp, "_harvest_subagent_costs", harvest)
    patch(Pilot, "pause", pause)
    patch(configure, "refresh_model_info_background", refresh)
    patch(module.StallRecorder, "start", start)
    patch(module.StallRecorder, "stop", stop)
    patch(module.StallRecorder, "begin_noise_window", begin)
    patch(module.StallRecorder, "end_noise_window", end)
    if MODE == "s3-held":
        patch(module._SlowListing, "__call__", held_listing)
    try:
        yield
    finally:
        # Cleanup release is NEVER accepted as the causal witness. The real
        # worker's completion precedes restoration and parent sandbox deletion.
        gate.set()
        if task is not None and not task.done():
            task.cancel()
        if task is not None and loop is not None and not loop.is_running() and not loop.is_closed():
            loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        settled = finished.wait(3)
        for owner, name, original in reversed(restorations):
            setattr(owner, name, original)
        rec("cleanup", worker_joined=settled, task_settled=task is None or task.done())
        if not settled:
            raise RuntimeError("listing worker unsettled")


def pytest_sessionfinish(session, exitstatus):
    from scripts.diagnostics.s2s3_6911.driver import identity
    payload = dict(mode=MODE, exitstatus=int(exitstatus), records=records, identity=identity(), **report)
    encoded = json.dumps(payload)
    if len(encoded.encode()) > 200000:
        raise RuntimeError("diagnostic evidence bound")
    with open(os.environ["S2S3_OUT"], "w", encoding="utf8") as stream:
        stream.write(encoded)


async def synthetic():
    from tests.unit.test_tui_responsiveness import StallRecorder
    from tests.loop_callback_spans import CallbackSpans
    output = []
    for mode in ("sync", "offload"):
        windows = 0
        with CallbackSpans(asyncio.get_running_loop()) as callbacks:
            observe = callbacks.observe_window
            def observed(*args):
                nonlocal windows
                windows += 1
                return observe(*args)
            callbacks.observe_window = observed
            recorder = StallRecorder(callbacks=callbacks)
            await recorder.start()
            deadline = time.perf_counter() + 1
            while windows == 0 and time.perf_counter() < deadline:
                await asyncio.sleep(0.005)
            ready = windows > 0
            before = windows
            if mode == "sync":
                cpu = time.thread_time()
                while time.thread_time() - cpu < 0.2:
                    pass
            else:
                await asyncio.to_thread(time.sleep, 0.2)
            deadline = time.perf_counter() + 1
            while windows <= before and time.perf_counter() < deadline:
                await asyncio.sleep(0.005)
            await recorder.stop()
            caught = False
            try:
                recorder.assert_no_stall(cpu_ceiling_ms=50, wall_ceiling_ms=2000)
            except AssertionError as error:
                if not str(error).startswith("event loop consumed "):
                    raise
                caught = True
            summary = recorder.measurement_summary()
            output.append(dict(mode=mode, ready=ready, trailing=windows > before,
                               cpu_trip=caught, worst_cpu_ms=summary["worst_cpu_ms"]))
    with open(os.environ["S2S3_OUT"], "w", encoding="utf8") as stream:
        from scripts.diagnostics.s2s3_6911.driver import identity
        json.dump(dict(mode="synthetic", controls=output, identity=identity()), stream)


if __name__ == "__main__":
    asyncio.run(synthetic())
