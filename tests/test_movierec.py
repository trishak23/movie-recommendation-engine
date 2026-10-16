"""Unit tests for the movie recommendation engine (stdlib unittest)."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movierec.config import Config  # noqa: E402
from movierec.data import generate, make_split  # noqa: E402
from movierec.evaluate import rmse_mae  # noqa: E402
from movierec.models import ALSRecommender, PopularityRecommender  # noqa: E402
from movierec.predict import recommend_new_user  # noqa: E402

CFG = replace(Config(), n_users=150, n_items=100, n_cold_items=10, n_cold_users=10, density=0.15)


class TestData(unittest.TestCase):
    def test_generation_is_seeded_and_valid(self):
        _, items, r1 = generate(CFG)
        _, _, r2 = generate(CFG)
        self.assertTrue(r1.equals(r2))
        self.assertTrue(r1.rating.between(1, 5).all())
        self.assertFalse(r1.duplicated(["user_id", "item_id"]).any())
        self.assertEqual(len(items), CFG.n_items)

    def test_split_has_no_leakage(self):
        _, _, r = generate(CFG)
        sp = make_split(r, CFG)
        self.assertFalse((sp.M_train & sp.M_test).any())
        self.assertFalse(sp.M_train[:, sp.cold_items].any())
        self.assertFalse(sp.M_train[sp.cold_users].any())
        self.assertFalse(sp.M_test[:, sp.cold_items].any())


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, r = generate(CFG)
        cls.sp = make_split(r, CFG)

    def test_als_beats_global_mean_and_loss_decreases(self):
        sp = self.sp
        als = ALSRecommender(8, 0.1, iters=8, seed=0).fit(sp.R_train, sp.M_train)
        mean_pred = np.full(sp.R_test.shape, sp.R_train[sp.M_train].mean())
        base = rmse_mae(mean_pred, sp.R_test, sp.M_test)["rmse"]
        got = rmse_mae(als.predict_matrix(), sp.R_test, sp.M_test)["rmse"]
        self.assertLess(got, base)
        self.assertLess(als.history[-1], als.history[0])

    def test_predictions_shape_and_finite(self):
        sp = self.sp
        pop = PopularityRecommender().fit(sp.R_train, sp.M_train)
        self.assertEqual(pop.predict_matrix().shape, sp.R_train.shape)
        self.assertTrue(np.isfinite(pop.predict_matrix()).all())

    def test_cold_start_new_user(self):
        sp = self.sp
        als = ALSRecommender(8, 0.1, iters=5, seed=0).fit(sp.R_train, sp.M_train)
        pop = PopularityRecommender().fit(sp.R_train, sp.M_train)
        bundle = {"als": als, "popularity": pop}
        empty = recommend_new_user(bundle, {}, 5)
        self.assertEqual(len(empty), 5)
        rated = {0: 5.0, 1: 4.0, 2: 1.0}
        recs = recommend_new_user(bundle, rated, 5)
        self.assertEqual(len(recs), 5)
        self.assertFalse({i for i, _ in recs} & set(rated))
        self.assertTrue(all(1 <= s <= 5 for _, s in recs))


if __name__ == "__main__":
    unittest.main()
