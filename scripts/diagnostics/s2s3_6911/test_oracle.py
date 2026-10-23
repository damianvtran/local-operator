import re
import ast
import unittest
from copy import deepcopy
from oracle import final, judge

class Controls(unittest.TestCase):
    def good(self):
        return dict(mode="synthetic", rc=0, reaped=True, timed_out=False, signal=None,
                    cleanup_errors=[], identity_ok=True, data=dict(mode="synthetic", controls=[
                        dict(mode="sync", ready=True, trailing=True, cpu_trip=True, worst_cpu_ms=200),
                        dict(mode="offload", ready=True, trailing=True, cpu_trip=False, worst_cpu_ms=1, worker_cpu_ms=200, worker_thread=2, loop_thread=1)]))

    def test_clock_positive_and_named_negative_oracles(self):
        good = self.good()
        self.assertEqual(judge(good), "qualified clock controls")
        for mutate, message in [
            (lambda r: r.update(reaped=False), "lifecycle"),
            (lambda r: r.update(identity_ok=False), "identity"),
            (lambda r: r["data"].update(locals={}), "privacy"),
            (lambda r: r["data"]["controls"].pop(), "missing clock"),
            (lambda r: r["data"]["controls"][0].update(ready=False), "baseline"),
            (lambda r: r["data"]["controls"][1].update(cpu_trip=True), "CPU"),
        ]:
            row = deepcopy(good)
            mutate(row)
            with self.assertRaisesRegex(ValueError, message):
                judge(row)
        with self.assertRaisesRegex(ValueError, "missing/duplicate"):
            final([good])



class Completion(unittest.TestCase):
    def test_actual_finish_orchestration_healthy_and_faults(self):
        from oracle import finish_child
        for fault in (None, "observe", "kill", "wait"):
            events = []
            class Child:
                def __init__(self): self.rc = None
                def wait(self, timeout):
                    events.append("wait")
                    if fault == "wait": raise RuntimeError("synthetic")
                    if fault == "kill" and len(events) == 1: raise TimeoutError()
                    self.rc = 0
                    return 0
                def poll(self): return self.rc
                def kill(self): events.append("direct-kill"); raise RuntimeError("synthetic")
            def observe():
                if fault == "observe": raise RuntimeError("synthetic")
                return {}
            def kill(): events.append("kill"); raise RuntimeError("synthetic")
            row = finish_child(Child(), observe, kill, lambda: {}, lambda: events.append("delete"))
            if fault is None:
                self.assertEqual(events, ["wait", "wait", "delete"])
            else:
                self.assertGreaterEqual(events.count("wait"), 2)
                self.assertNotIn("delete", events)
                self.assertTrue(row["cleanup_errors"])

    def test_partial_patch_ticker_release_restore_faults(self):
        from oracle import restore_all
        for broken in ("release", "ticker", "complete", "restore"):
            actions = []
            class Owner:
                def __setattr__(self, name, value):
                    actions.append(name)
                    if broken == "restore" and name == "b": raise RuntimeError("synthetic")
            owner = Owner()
            def action(name):
                def run():
                    actions.append(name)
                    if name == broken: raise RuntimeError("synthetic")
                return run
            errors = restore_all([(owner, "a", 1), (owner, "b", 2)], action("release"), action("ticker"), action("complete"))
            self.assertTrue(errors)
            self.assertEqual(actions[-2:], ["b", "a"])
            self.assertTrue({"release", "ticker", "complete"} <= set(actions))

    def protocol(self, mode):
        from oracle import span_record, held_record
        rows = [dict(kind="scope", supplemental_gc="NOT_COLLECTED", excluded_gc=0), dict(kind="outcome", call="fail" if mode == "s3-split" else "pass", exitstatus=1 if mode == "s3-split" else 0), dict(kind="cleanup", worker_function_completed=True, task_settled=True), dict(kind="recorder", summary={})]
        for i in range(3):
            rows.append(span_record("harvest", (0, i), (.001, i+.01), 1, manual=True, measured=True, i=i))
            rows.append(span_record("pause", (.001, i+.01), (.002, i+1), 1, manual=True, measured=True, i=i))
        rows.append(span_record("pause", (0, -1), (.01, 0), 1, manual=True, measured=False))
        rows.append(span_record("harvest", (0, -1), (.01, 0), 1, manual=False, measured=False))
        if mode == "s3-split": rows.append(span_record("controlled_async_delay", (0, .02), (.001, .621), 1))
        else: rows += [held_record(3, 1, 1), dict(kind="worker_entry", thread=2)]
        return dict(mode=mode, rc=1 if mode=="s3-split" else 0, reaped=True, timed_out=False, signal=None, cleanup_errors=[], identity_ok=True, data=dict(mode=mode, setup_ok=True, call="fail" if mode=="s3-split" else "pass", assertion="harvest blocked for 601 ms" if mode=="s3-split" else None, records=rows))

    def test_shared_producer_protocol_and_wrong_oracles(self):
        from oracle import judge, project
        for mode in ("s3-split", "s3-held"):
            row = self.protocol(mode)
            self.assertTrue(judge(project(row)))
            row["data"]["records"][-1]["message"] = "PRIVATE"
            with self.assertRaisesRegex(ValueError, "privacy"): project(row)
        row = self.protocol("s3-split")
        row["data"]["assertion"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "OLD500"): judge(row)
        row = self.protocol("s3-held")
        row["data"]["records"][-2]["loop_ticks"] = 0
        with self.assertRaisesRegex(ValueError, "LOOP witness"): judge(row)




class Capacity(unittest.TestCase):
    def s2(self, records):
        return dict(mode="s2-original", rc=0, reaped=True, timed_out=False, signal=None,
                    cleanup_errors=[], identity_ok=True, data=dict(mode="s2-original",
                    setup_ok=True, call="pass", assertion=None, records=records))

    def suffix(self, budget):
        budget.add("recorder", summary={"callbacks": {"loop_thread_id": 1,
            "worst_cpu_windows": [{"callbacks": {"gc": [], "top_cpu": [], "top_wall": []}}],
            "worst_wall_windows": []}})
        for kind in ("s2_idle", "s2_emit_burst", "s2_settle"):
            budget.add(kind, cpu_ms=1, wall_ms=2)
        budget.add("cleanup", worker_function_completed=True, task_settled=True)
        budget.add("outcome", call="pass", exitstatus=0)

    def test_actual_budget_replay_excludes_startup_gc_retains_m15(self):
        from oracle import RecordBudget, judge, project
        budget = RecordBudget()
        for i in range(244): self.assertFalse(budget.add("gc", thread=1, is_loop=None))
        for i in range(4): budget.add("pause", manual=True, measured=False)
        budget.add("harvest", manual=False, measured=False)
        budget.add("recorder_start", thread=1)
        self.suffix(budget)
        records = budget.snapshot()
        self.assertFalse(any(r["kind"] == "gc" for r in records))
        self.assertEqual(records[-1]["excluded_gc"], 244)
        self.assertEqual(judge(project(self.s2(records))), "PASS original")
        summary = next(r for r in records if r["kind"] == "recorder")
        del summary["summary"]["callbacks"]["worst_cpu_windows"][0]["callbacks"]["gc"]
        with self.assertRaisesRegex(ValueError, "S2 M15/thread GC missing"): judge(self.s2(records))

    def test_optional_capacity_keeps_critical_and_overflow_fails(self):
        from oracle import RecordBudget, judge, project
        budget = RecordBudget()
        for i in range(1000): budget.add("pause", manual=False, measured=False)
        self.suffix(budget)
        records = budget.snapshot()
        self.assertLessEqual(len(records), 250)
        self.assertEqual(sum(r["kind"] == "pause" for r in records), 234)
        self.assertTrue({"cleanup", "recorder", "outcome", "overflow"} <= {r["kind"] for r in records})
        self.assertEqual(records[-1]["dropped_count"], 766)
        with self.assertRaisesRegex(ValueError, "capacity overflow"): judge(project(self.s2(records)))
        for i in range(100): budget.add("error")
        self.assertEqual(len(budget.snapshot()), 250)
        self.assertGreater(budget.snapshot()[-1]["critical_dropped"], 0)

    def test_missing_unknown_critical_and_privacy_fail_right_oracle(self):
        from oracle import RecordBudget, judge, project
        budget = RecordBudget()
        self.suffix(budget)
        records = budget.snapshot()
        with self.assertRaisesRegex(ValueError, "missing cleanup proof"):
            judge(self.s2([r for r in records if r["kind"] != "cleanup"]))
        with self.assertRaisesRegex(ValueError, "scope/outcome"):
            judge(self.s2([r for r in records if r["kind"] != "outcome"]))
        with self.assertRaisesRegex(ValueError, "unknown diagnostic record"):
            budget.add("private-message", value="PRIVATE")
        with self.assertRaisesRegex(ValueError, "unknown measurement record"):
            judge(self.s2(records + [dict(kind="unknown")]))
        records[0]["message"] = "PRIVATE"
        with self.assertRaisesRegex(ValueError, "privacy"): project(self.s2(records))

    def test_bytes_exhaustion_explicit_and_no_global_gc_callback(self):
        from oracle import RecordBudget
        from pathlib import Path
        import ast
        budget = RecordBudget()
        budget.add("harvest", value="x" * 180001)
        self.suffix(budget)
        self.assertTrue(any(r["kind"] == "overflow" for r in budget.snapshot()))
        self.assertTrue(any(r["kind"] == "cleanup" for r in budget.snapshot()))
        tree = ast.parse(Path("plugin-source.txt").read_text())
        self.assertFalse(any(isinstance(n, ast.Import) and any(a.name == "gc" for a in n.names) for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == "callbacks" and isinstance(n.value, ast.Name) and n.value.id == "gc" for n in ast.walk(tree)))


class Budget(unittest.TestCase):
    def test_numeric_budget_single_source_and_arithmetic(self):
        import oracle
        from pathlib import Path
        self.assertEqual((oracle.CHILD_SECONDS, oracle.REAP_SECONDS, oracle.CASE_COUNT, oracle.OUTER_SECONDS, oracle.OUTER_GRACE_SECONDS), (20, 3, 4, 100, 5))
        self.assertEqual(oracle.CASE_COUNT * (oracle.CHILD_SECONDS + oracle.REAP_SECONDS), 92)
        self.assertLess(92, oracle.OUTER_SECONDS)
        # Actual 10.3 s first-case wall + 5 s fixture + 3 s reap fits the child budget.
        self.assertLessEqual(10.3 + 5 + oracle.REAP_SECONDS, oracle.CHILD_SECONDS + oracle.REAP_SECONDS)
        driver = Path("driver-source.txt").read_text()
        self.assertIn("time.monotonic()+CHILD_SECONDS", driver)
        self.assertNotIn("monotonic()+12", driver)
        workflow = Path("workflow-source.txt").read_text()
        self.assertIn("--timeout 100 --grace 5 --memory-mb 1024", workflow)
        self.assertIn("timeout-minutes: 5", workflow)
        self.assertEqual(len(re.findall(r"--timeout", workflow)), 1)
        self.assertGreater(300, 120 + oracle.OUTER_SECONDS + oracle.OUTER_GRACE_SECONDS + 20)


def _plugin_tree():
    from pathlib import Path
    return ast.parse(Path("plugin-source.txt").read_text())


def _literal_record_kinds(tree):
    """Every literal first argument of rec()/span() calls in the real plugin."""
    kinds = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("rec", "span"):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                kinds.add(node.args[0].value)
    return kinds


class PluginShape(unittest.TestCase):
    def test_every_literal_record_kind_is_accepted_by_budget_and_judge(self):
        from oracle import RecordBudget
        kinds = _literal_record_kinds(_plugin_tree())
        self.assertIn("worker_function_completed", kinds)
        for kind in sorted(kinds):
            budget = RecordBudget()
            try:
                budget.add(kind)
            except ValueError as error:
                self.fail("producer kind %r rejected by RecordBudget: %s" % (kind, error))
        # The oracle's `known` allow-list is derived from the same classes.
        self.assertIn("worker_function_completed", RecordBudget.OPTIONAL | RecordBudget.CRITICAL)

    def test_held_worker_completion_record_cannot_gate_completion(self):
        # rec() in the worker's finally must come AFTER the decrement and the
        # event set, so a recording failure can never leave the held case unfinished.
        tree = _plugin_tree()
        held = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "held_listing")
        final = next(n for n in ast.walk(held) if isinstance(n, ast.Try) and n.finalbody)
        order = []
        for stmt in final.finalbody:
            text = ast.unparse(stmt)
            if "worker_function_completed" in text:
                order.append("record")
            elif "finished.set" in text:
                order.append("set")
        self.assertEqual(order, ["set", "record"])

    def test_every_mode_yields_exactly_once_unconditionally(self):
        tree = _plugin_tree()
        hook = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pytest_runtest_call")
        yields = [n for n in ast.walk(hook) if isinstance(n, ast.Yield)]
        self.assertEqual(len(yields), 1)
        # Parent chain from the hook body to the yield must contain no If/Else:
        # a conditional yield means other modes never yield and pluggy rejects the wrapper.
        def conditional(node, target, under_if=False):
            if node is target:
                return under_if
            for child in ast.iter_child_nodes(node):
                found = conditional(child, target, under_if or isinstance(node, ast.If))
                if found is not None:
                    return found
            return None
        self.assertIs(conditional(hook, yields[0]), False)
        # Fake hook-shaped drive: for each mode the generator must reach its yield.
        for mode in ("s2-original", "s3-split", "s3-held", "synthetic"):
            self.assertTrue(self._reaches_yield(hook, mode), mode)

    @staticmethod
    def _reaches_yield(hook, mode):
        """Evaluate only the MODE guards on the path to the yield, never App code."""
        def walk(stmts):
            for stmt in stmts:
                if isinstance(stmt, ast.If):
                    test = ast.unparse(stmt.test)
                    taken = None
                    if test.startswith("MODE =="):
                        taken = mode == ast.literal_eval(stmt.test.comparators[0])
                    elif all(isinstance(n, ast.Raise) for n in stmt.body) and not stmt.orelse:
                        taken = False  # fail-closed precondition guard; not taken for valid nodes
                    if taken is None:
                        raise AssertionError("unexpected guard " + test)
                    branch = stmt.body if taken else stmt.orelse
                    if walk(branch):
                        return True
                elif isinstance(stmt, ast.Try):
                    if walk(stmt.body):
                        return True
                elif any(isinstance(n, ast.Yield) for n in ast.walk(stmt)):
                    return True
            return False
        return walk(hook.body)


def _extract(name, params, setup, returns):
    """Compile ONE nested function of the real plugin text in a throwaway closure.

    The plugin is never imported (it needs pytest/App); only this function's own
    source is executed, against fakes, so the timeline it encodes is exercised.
    """
    from pathlib import Path
    tree = ast.parse(Path("plugin-source.txt").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    body = "\n".join("    " + line for line in ast.unparse(node).splitlines())
    source = "def _make(%s):\n%s\n%s\n    return %s\n" % (params, "\n".join("    " + l for l in setup.splitlines()), body, returns)
    namespace = {}
    return source, namespace


class FakeClock:
    def __init__(self, step):
        self.now, self.step = 0.0, step
    def perf_counter(self):
        return self.now


class Flag:
    def __init__(self, clock=None, at=None):
        self.value, self.clock, self.at = False, clock, at
    def set(self):
        self.value = True
    def is_set(self):
        return self.value or (self.clock is not None and self.at is not None and self.clock.now >= self.at)


class HeldTimeline(unittest.TestCase):
    """Observed run 39960859806: witness created at the first pause, worker entered
    ~1.96 s later (after 3 measured pairs), loop granularity ~22 ms (pause walls)."""

    def run_witness(self, entry_at, step=0.022, need_loop=True):
        import asyncio, threading
        import oracle
        clock = FakeClock(step)
        class FakeAsyncio:
            @staticmethod
            async def sleep(delay):
                clock.now += clock.step
        entered, gate, records = Flag(clock, entry_at), Flag(), []
        source, ns = _extract("held_witness", "asyncio, time, threading, entered, gate, rec, requests, held_record",
                              "causal_released = False", "held_witness, (lambda: causal_released)")
        ns.update(ast=ast)
        exec(source, ns)
        witness, causal = ns["_make"](FakeAsyncio, clock, threading, entered, gate,
                                      lambda kind, **v: records.append(dict(kind=kind, **v)), 1, oracle.held_record)
        try:
            asyncio.run(witness())
            error = None
        except RuntimeError as exc:
            error = str(exc)
        return error, records, gate.is_set(), causal()

    def test_witness_survives_observed_late_worker_entry(self):
        # Entry 1.96 s after creation: the OLD 2 s-from-creation deadline leaves ~2 ticks.
        error, records, gate_set, _ = self.run_witness(entry_at=1.96)
        self.assertIsNone(error, "witness expired before the worker had time to be observed")
        self.assertTrue(gate_set)
        self.assertGreaterEqual(sum(r["kind"] == "held_loop_tick" for r in records), 3)
        self.assertEqual(sum(r["kind"] == "held_witness" for r in records), 1)

    def test_witness_still_fails_closed_when_worker_never_enters(self):
        error, records, gate_set, _ = self.run_witness(entry_at=10 ** 6)
        self.assertIsNotNone(error)
        self.assertFalse(gate_set)
        self.assertFalse(any(r["kind"] == "held_witness" for r in records))

    def test_witness_requires_three_ticks_not_one(self):
        # A stalled loop (one tick inside the window) must NOT qualify.
        error, records, gate_set, _ = self.run_witness(entry_at=0.0, step=1.2)
        self.assertIsNotNone(error)
        self.assertFalse(gate_set)

    def listing(self, causal, latency=5.0, gate_wait_ok=True):
        import threading
        source, ns = _extract("held_listing",
                              "threading, gate, entered, finished, rec, original_listing",
                              "active = 0\nlock = threading.Lock()\nloop_thread = 1\ncausal_released = False\n"
                              "def set_causal():\n    nonlocal causal_released\n    causal_released = True",
                              "held_listing, set_causal")
        exec(source, ns)
        gate, entered, finished, records, seen = threading.Event(), threading.Event(), threading.Event(), [], []
        def original(self, *a, **k):
            seen.append(self.latency_s)
            return [], "ok"
        listing, set_causal = ns["_make"](threading, gate, entered, finished, lambda kind, **v: records.append(dict(kind=kind, **v)), original)
        class Slow:
            latency_s = latency
        gate.set()  # Released: either causally by the witness, or only by cleanup.
        if causal:
            set_causal()
        try:
            listing(Slow())
            error = None
        except RuntimeError as exc:
            error = str(exc)
        return error, seen, records, finished.is_set()

    def test_causal_release_runs_listing_within_completion_wait(self):
        # completed() waits 3 s; the fixture's 5 s latency must not outlive it.
        error, seen, records, finished = self.listing(causal=True)
        self.assertIsNone(error)
        self.assertEqual(len(seen), 1)
        self.assertLessEqual(seen[0], 3.0)
        self.assertTrue(finished)
        self.assertEqual([r["kind"] for r in records][-1], "worker_function_completed")

    def test_cleanup_only_release_does_not_start_the_long_listing(self):
        error, seen, records, finished = self.listing(causal=False)
        self.assertEqual(seen, [], "a non-causal release must not sleep through the 5 s fixture")
        self.assertIsNotNone(error)
        self.assertTrue(finished)

    def test_synthetic_control_runs_before_the_held_protocol(self):
        import oracle
        self.assertLess(oracle.MODES.index("synthetic"), oracle.MODES.index("s3-held"))
        self.assertEqual(sorted(oracle.MODES), sorted(("s2-original", "s3-split", "s3-held", "synthetic")))

    def test_held_protocol_defect_is_labelled_not_product(self):
        import oracle
        row = Completion("test_shared_producer_protocol_and_wrong_oracles").protocol("s3-held")
        row["data"]["records"] = [r for r in row["data"]["records"] if r["kind"] != "held_witness"]
        with self.assertRaisesRegex(ValueError, "^HELD PROTOCOL DEFECT"):
            oracle.judge(row)
        # A lifecycle fault in the same mode is NOT relabelled as a protocol defect.
        row = Completion("test_shared_producer_protocol_and_wrong_oracles").protocol("s3-held")
        row["reaped"] = False
        with self.assertRaises(ValueError) as caught:
            oracle.judge(row)
        self.assertFalse(str(caught.exception).startswith("HELD PROTOCOL DEFECT"))


if __name__ == "__main__":
    unittest.main()
