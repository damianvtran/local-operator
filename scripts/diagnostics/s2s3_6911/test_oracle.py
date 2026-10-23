import re
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


if __name__ == "__main__":
    unittest.main()
