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


if __name__ == "__main__":
    unittest.main()
