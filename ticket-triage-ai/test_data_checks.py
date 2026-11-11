"""Проверки исключительно на синтетических данных."""
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from create_synthetic_csv import DEPARTMENTS, generate
from data_checks import inspect_data
from triage import experiment, is_synthetic, split, synthetic, validate


class DataChecks(unittest.TestCase):
    def test_generated_dataset(self):
        frame = generate()
        pd.testing.assert_frame_equal(frame, generate())
        self.assertEqual(frame.shape, (12000, 6))
        self.assertEqual(frame.id.nunique(), 12000)
        self.assertEqual(frame.priority.eq("критический").sum(), 600)
        self.assertTrue(is_synthetic(frame))
        report = inspect_data(frame, DEPARTMENTS)
        self.assertEqual(report["counts"].sum(), 0)
        self.assertTrue(report["department_dictionary_checked"])
        parts = split(validate(frame), True)
        self.assertEqual(tuple(map(len, parts)), (8000, 2000, 2000))
        self.assertEqual(set(parts[0].index) & set(parts[2].index), set())

    def test_audit_reports_defects_without_mutation(self):
        frame = synthetic()
        frame.loc[0, "id"] = frame.loc[1, "id"]
        frame.loc[2, ["subject", "text"]] = ""
        frame.loc[3, ["subject", "text"]] = ["?", ""]
        frame.loc[4, "priority"] = "неизвестно"
        frame.loc[5, "department"] = "неизвестно"
        frame.loc[6] = frame.loc[7]
        before = frame.copy(deep=True)
        report = inspect_data(frame, ["Учебная сеть", "Учебный склад", "Учебный доступ"])
        for name in ["duplicate_id", "duplicate_row", "duplicate_text", "empty_input",
                     "short_input", "unknown_priority", "unknown_department"]:
            self.assertGreater(report["counts"][name], 0, name)
        pd.testing.assert_frame_equal(frame, before)
        self.assertFalse(inspect_data(frame)["department_dictionary_checked"])

    def test_synthetic_marker_prevents_real_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            experiment(synthetic(), out, demo=False)
            info = joblib.load(out / "model.joblib")["info"]
            self.assertEqual(info["mode"], "SYNTHETIC")
            self.assertIsNone(info["acceptance"])
        mixed = synthetic()
        mixed.loc[0, "id"] = "historical-001"
        with self.assertRaises(ValueError):
            is_synthetic(mixed)


if __name__ == "__main__":
    unittest.main()
