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
        rows = [dict(kind="cleanup", worker_function_completed=True, task_settled=True), dict(kind="recorder", summary={})]
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


if __name__ == "__main__":
    unittest.main()
