"""Все примеры и результаты этих тестов СИНТЕТИЧЕСКИЕ."""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from operator_cli import connect, recommend, resolve
from triage import PRIORITIES, TASKS, experiment, split, synthetic, validate


class Smoke(unittest.TestCase):
    def test_schema(self):
        frame = synthetic()
        for broken in [frame.drop(columns="subject"), frame.assign(id="duplicate"),
                       frame.assign(priority="unknown"), frame.assign(subject="", text="")]:
            with self.subTest(), self.assertRaises(ValueError):
                validate(broken)
        self.assertEqual(len(validate(frame.assign(subject=""))), len(frame))

    def test_legacy_priorities(self):
        frame = synthetic()
        aliases = dict(zip(PRIORITIES, ["low", "medium", "high", "critical"]))
        legacy = frame.assign(priority=frame.priority.map(aliases))
        pd.testing.assert_frame_equal(validate(frame), validate(legacy))
        legacy.loc[legacy.priority.eq("medium"), "priority"] = "средний"
        pd.testing.assert_frame_equal(validate(frame), validate(legacy))

    def test_historical_sizes_on_synthetic_rows(self):
        frame = pd.concat([synthetic()] * 32, ignore_index=True).iloc[:12000].copy()
        frame["id"] = [f"SYN-{i}" for i in range(len(frame))]
        frame["text"] += " " + frame.id
        data = validate(frame)
        ids = list(data.index[-2000:])
        train, val, test = split(data, False, ids)
        self.assertEqual((len(train), len(val), len(test)), (8000, 2000, 2000))
        self.assertEqual(set(test.index), set(ids))
        self.assertFalse(set(train.index) & set(val.index))

    def test_end_to_end(self):
        frame = synthetic()
        train, val, test = split(validate(frame), True)
        frame.loc[frame.id.isin(test.index), "text"] += " holdoutsentinel"
        frame.loc[frame.id.isin(val.index), "text"] += " valsentinel"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            scores = experiment(frame, out, demo=True)
            self.assertEqual(len(scores), 4)
            bundle = joblib.load(out / "model.joblib")
            self.assertIsNone(bundle["info"]["acceptance"])
            for task in TASKS:
                vocab = bundle["models"][task][0].vocabulary_
                self.assertFalse({"holdoutsentinel", "valsentinel"} & set(vocab))
                for method in ["majority", "tfidf_lr"]:
                    m = pd.read_csv(out / f"matrix_{method}_{task}.csv", index_col=0)
                    self.assertEqual(m.to_numpy().sum(), len(test))
                    if task == "priority":
                        missed = pd.read_csv(out / f"missed_{method}.csv")
                        self.assertEqual(len(missed), m.loc["критический", PRIORITIES[:-1]].sum())
            db = connect(Path(tmp) / "log.sqlite")
            try:
                with self.assertRaises(ValueError):
                    recommend(db, bundle, "synthetic", "", " ")
                for correction in [False, True]:
                    key, proposal = recommend(db, bundle, "synthetic", "Учебная сеть", "полная остановка")
                    final = {t: proposal[t]["label"] for t in TASKS}
                    if correction:
                        final["priority"] = next(p for p in PRIORITIES if p != final["priority"])
                    resolve(db, key, final, bundle["labels"])
                    with self.assertRaises(ValueError):
                        resolve(db, key, final, bundle["labels"])
                    saved = db.execute("SELECT proposal,final FROM decisions WHERE id=?", (key,)).fetchone()
                    self.assertEqual(list(map(json.loads, saved)), [proposal, final])
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
