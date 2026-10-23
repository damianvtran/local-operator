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


if __name__ == "__main__":
    unittest.main()
