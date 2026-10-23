"""Synthetic data only: this suite never imports SQLite or the product."""
import unittest
from checker import check_case, fixture, negative_controls


class EvidenceControls(unittest.TestCase):
    def test_all_named_negatives_and_complete_positive(self):
        negative_controls()

    def test_a_signal_cannot_be_green_even_with_complete_success_events(self):
        source, row = fixture()
        row["rc"] = -7
        with self.assertRaisesRegex(ValueError, "signal"):
            check_case(row, source)




class FakeProcess:
    pid = 11

    def __init__(self, mode="normal"):
        self.mode = mode
        self.returncode = None
        self.events = []

    def poll(self):
        return self.returncode

    def kill(self):
        self.events.append("kill")

    def wait(self, seconds):
        self.events.append(f"wait{seconds}")
        if seconds == 10 and self.mode in ("timeout", "unsettled"):
            raise TimeoutError("synthetic bound")
        if seconds == 5 and self.mode == "unsettled":
            raise RuntimeError("synthetic final-wait failure")
        self.returncode = -9 if self.mode == "timeout" else 0
        return self.returncode


class MandatoryEvidence(unittest.TestCase):
    def test_getpgid_error_still_waits_and_reports_unknown(self):
        from checker import own_process, settled_for_cleanup
        child = FakeProcess()
        def broken(_pid):
            raise OSError("synthetic observation")
        row = own_process(child, broken, child.wait, lambda pid: {})
        self.assertEqual(child.events, ["wait10", "wait5"])
        self.assertEqual(row["child_pgid"], "UNKNOWN")
        self.assertEqual(row["lifecycle_errors"], ["OSError"])
        self.assertTrue(settled_for_cleanup(row))

    def test_timeout_reaps_but_failed_final_wait_never_deletes(self):
        from checker import own_process, settled_for_cleanup
        for mode in ("timeout", "unsettled"):
            child = FakeProcess(mode)
            row = own_process(child, lambda pid: 10, child.wait, lambda pid: {"pid": pid})
            self.assertEqual(child.events, ["wait10", "kill", "wait5"])
            self.assertTrue(row["timed_out"])
            deletes = []
            if settled_for_cleanup(row):
                deletes.append("delete-owned-root")
            self.assertEqual(bool(deletes), mode == "timeout")
            self.assertEqual(row["reaped"], mode == "timeout")

    def test_required_setup_fields_have_right_missing_or_wrong_oracles(self):
        from checker import check_case, fixture
        for field, oracle in (("executable", "executable"),
                              ("executable_sha256", "executable"),
                              ("python_info", "Python"),
                              ("dependencies", "dependencies"),
                              ("core_limit", "core limit")):
            for wrong in (False, True):
                source, row = fixture()
                if wrong:
                    row["events"][0][field] = "incorrect"
                else:
                    del row["events"][0][field]
                with self.assertRaisesRegex(ValueError, oracle):
                    check_case(row, source)
        for field in ("version", "binary", "binary_sha256", "compile_options", "linked_libraries"):
            source, row = fixture()
            event = next(e for e in row["events"] if e["phase"] == "metadata_sqlite_done")
            del event[field]
            with self.assertRaisesRegex(ValueError, "SQLite"):
                check_case(row, source)

    def test_phase_enum_sqlite_start_order_and_store_key_are_required(self):
        from checker import check_case, fixture
        source, row = fixture()
        row["events"].insert(-1, {"phase": "unexpected"})
        for i, e in enumerate(row["events"]):
            e["index"] = i
        with self.assertRaisesRegex(ValueError, "unknown phase"):
            check_case(row, source)
        source, row = fixture()
        row["events"] = [e for e in row["events"] if e["phase"] != "metadata_sqlite_start"]
        for i, e in enumerate(row["events"]):
            e["index"] = i
        with self.assertRaisesRegex(ValueError, "metadata_sqlite_start"):
            check_case(row, source)
        source, row = fixture("separate-db")
        event = next(e for e in row["events"] if e["phase"] == "before_workers")
        event["synthetic_files"][0]["store_key"] = "private-path-not-permitted"
        with self.assertRaisesRegex(ValueError, "attribution"):
            check_case(row, source)




class AttributionEvidence(unittest.TestCase):
    def summary(self):
        from checker import qualified_exception
        refusal = type("MemoryError", (ValueError,), {"__module__": "local_operator.memory"})
        sql = type("OperationalError", (Exception,), {"__module__": "sqlite3"})
        scope = {"Refusal": refusal, "SqlError": sql}
        exec(compile('def fail(n):\n if n: return fail(n-1)\n err=SqlError("SECRET")\n err.sqlite_errorcode=5\n err.sqlite_errorname="SQLITE_BUSY"\n try: raise err\n except Exception as cause: raise Refusal("PRIVATE") from cause\n', "/repo/local_operator/memory.py", "exec"), scope)
        try:
            scope["fail"](20)
        except Exception as exc:
            return qualified_exception(exc, "/repo")
        self.fail("synthetic exception not raised")

    def failure(self, mode="same-db"):
        from checker import fixture
        source, row = fixture(mode)
        row.update(rc=1, exception_summary=self.summary())
        row["events"] = [e for e in row["events"] if e["phase"] not in ("before_connection", "connection_entered", "connection_closed", "case_done")]
        after = row["events"][-1]
        after.update(threads_joined=True, worker_failures=[dict(slot=0, iteration=0, phase="before_context", exception=self.summary())])
        for i, e in enumerate(row["events"]):
            e["index"] = i
        return source, row

    def test_original_missing_qualification_and_privacy_oracles(self):
        from checker import fixture, check_case
        source, row = fixture()
        row["exception_summary"] = {"message": "PRIVATE"}
        with self.assertRaisesRegex(ValueError, "privacy"):
            check_case(row, source)
        row.update(rc=1, exception_summary={"exception_type": "MemoryError", "errno": None})
        with self.assertRaisesRegex(ValueError, "qualified"):
            check_case(row, source)

    def test_actual_trace_qualification_privacy_and_innermost_cap(self):
        import json
        from checker import check_exception
        value = self.summary()
        check_exception(value)
        encoded = json.dumps(value)
        self.assertNotIn("SECRET", encoded)
        self.assertNotIn("PRIVATE", encoded)
        self.assertNotIn("/repo/", encoded)
        self.assertLessEqual(len(encoded.encode()), 4096)
        self.assertLessEqual(sum(len(e["frames"]) for e in value["chain"]), 12)
        self.assertEqual(value["chain"][0]["type_module"], "local_operator.memory")
        self.assertEqual(value["chain"][1]["type_module"], "sqlite3")
        self.assertEqual(value["chain"][1]["sqlite_errorname"], "SQLITE_BUSY")
        self.assertEqual(value["chain"][1]["frames"][-1]["line"], 6)
        self.assertEqual(value["chain"][1]["frames"][-1]["function"], "fail")

    def test_ordinary_failure_continues_but_final_verdict_remains_fail(self):
        from checker import observation, check_all, fixture, MODES
        source, bad = self.failure()
        self.assertIs(observation(bad, source), False)
        rows = [bad] + [fixture(m)[1] for m in MODES[1:]]
        self.assertTrue(check_all(rows, source).startswith("FAIL:"))
        self.assertEqual(MODES, ("same-db", "separate-db", "same-db-serialized"))

    def test_privacy_unknown_trace_source_signal_limits_and_reap_stop(self):
        from copy import deepcopy
        from checker import observation
        source, good = self.failure()
        for mutate in (
            lambda r: r["exception_summary"].update(message="PRIVATE"),
            lambda r: r["exception_summary"]["chain"][0].update(type_module="UNKNOWN"),
            lambda r: r["exception_summary"]["chain"][0].update(frames=[]),
            lambda r: r["exception_summary"]["chain"][0]["frames"][0].update(file="/private/path"),
            lambda r: r["exception_summary"]["chain"][0].update(sqlite_errorname="SELECT secret"),
            lambda r: r["exception_summary"]["chain"][0]["frames"][0].update(locals={}),
            lambda r: r["exception_summary"].update(truncated=True),
            lambda r: r["events"][0].update(source={}),
            lambda r: r["events"][0]["resources"].update(units="bytes"),
            lambda r: r.update(rc=-7),
            lambda r: r.update(signal=7),
            lambda r: r.update(timed_out=True),
            lambda r: r.update(reaped=False),
            lambda r: r.update(case_root_removed=False),
            lambda r: r.update(lifecycle_errors=["unsettled"]),
            lambda r: r["events"][-1]["worker_failures"][0].update(phase="UNKNOWN"),
        ):
            row = deepcopy(good)
            mutate(row)
            with self.assertRaises(ValueError):
                observation(row, source)

    def test_resource_units_and_unlimited_are_explicit(self):
        from checker import resource_data, check_resources
        limits = {k: (-1, -1) for k in ("AS", "DATA", "STACK")}
        limits["CORE"] = (0, 0)
        value = resource_data("linux", 12345, limits)
        check_resources(value)
        self.assertEqual(value["units"], "KiB")
        self.assertIsNone(value["limits"]["AS"]["soft"])
        self.assertTrue(value["limits"]["AS"]["soft_unlimited"])
        self.assertEqual(resource_data("darwin", 12345, limits)["units"], "bytes")
        value["limits"]["AS"]["soft"] = -1
        with self.assertRaisesRegex(ValueError, "unlimited"):
            check_resources(value)


if __name__ == "__main__":
    unittest.main()
