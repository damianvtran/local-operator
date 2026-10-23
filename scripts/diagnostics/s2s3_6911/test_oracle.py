import unittest
from copy import deepcopy
from oracle import final, judge

class Controls(unittest.TestCase):
    def good(self):
        return dict(mode="synthetic", rc=0, reaped=True, timed_out=False, signal=None,
                    cleanup_errors=[], identity_ok=True, data=dict(mode="synthetic", controls=[
                        dict(mode="sync", ready=True, trailing=True, cpu_trip=True, worst_cpu_ms=200),
                        dict(mode="offload", ready=True, trailing=True, cpu_trip=False, worst_cpu_ms=1)]))

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

if __name__ == "__main__":
    unittest.main()
