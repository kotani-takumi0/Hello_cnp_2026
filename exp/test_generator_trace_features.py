"""Focused invariance tests for H90 generator-trace features."""
from __future__ import annotations

import unittest
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator_trace_features import (
    build_generator_trace_features, fit_predict_trace,
    select_trace_feature_group,
)


class GeneratorTraceFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv("data/train.csv")
        cls.test = pd.read_csv("data/test.csv")

    def test_train_test_schema_and_finiteness(self):
        train_features = build_generator_trace_features(self.train)
        test_features = build_generator_trace_features(self.test)
        self.assertEqual(list(train_features.columns), list(test_features.columns))
        self.assertGreater(train_features.shape[1], 100)
        self.assertTrue(np.isfinite(train_features.to_numpy()).all())
        self.assertTrue(np.isfinite(test_features.to_numpy()).all())

    def test_features_are_deterministic_and_target_free(self):
        first = build_generator_trace_features(self.train)
        changed = self.train.copy()
        changed["購入フラグ"] = 1 - changed["購入フラグ"]
        second = build_generator_trace_features(changed)
        np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())

    def test_target_labels_are_not_read(self):
        features = build_generator_trace_features(self.train)
        y = self.train["購入フラグ"].to_numpy(dtype=int)
        train_indices = np.arange(500)
        target_indices = np.arange(500, 550)
        expected = fit_predict_trace(features, y, train_indices, target_indices)
        changed = y.copy()
        changed[target_indices] = 1 - changed[target_indices]
        actual = fit_predict_trace(features, changed, train_indices, target_indices)
        np.testing.assert_array_equal(expected, actual)

    def test_predeclared_groups_are_nonempty_and_disjoint(self):
        features = build_generator_trace_features(self.train)
        numeric = select_trace_feature_group(features, "numeric_digits")
        accounting = select_trace_feature_group(features, "accounting_residuals")
        text = select_trace_feature_group(features, "text_layout")
        self.assertTrue(set(numeric).isdisjoint(accounting))
        self.assertTrue(set(numeric).isdisjoint(text))
        self.assertTrue(set(accounting).isdisjoint(text))
        self.assertEqual(
            set(select_trace_feature_group(features, "numeric_all")),
            set(numeric) | set(accounting),
        )


if __name__ == "__main__":
    unittest.main()
