"""Stdlib-only tests for the churn logistic-scoring prototype.

Verifies: sigmoid bounds, risk thresholds, monotonicity (tenure up =>
probability down in this weight set), one-hot contract effects, and that
the shipped model_weights.json parses and has the documented feature keys.
Run: python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from inference import load_model, predict_churn  # noqa: E402

ROOT = os.path.join(HERE, '..')

class TestModelArtifact(unittest.TestCase):
    def test_weights_file_parses(self):
        with open(os.path.join(ROOT, 'model_weights.json')) as f:
            m = json.load(f)
        self.assertIn('intercept', m)
        for k in ['tenure', 'monthlyCharges', 'totalCharges',
                  'contract_Month-to-month', 'contract_One year', 'contract_Two year']:
            self.assertIn(k, m['features'], k)

    def test_sigmoid_bounds(self):
        weights, intercept = load_model(os.path.join(ROOT, 'model_weights.json'))
        for p in [-100.0, 0.0, 100.0]:
            prob, risk = predict_churn(tenure=p, monthly_charges=60, total_charges=720,
                                       contract_type='Month-to-month',
                                       weights=weights, intercept=intercept)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertIn(risk, ('HIGH', 'MEDIUM', 'LOW'))

    def test_tenure_decreases_probability(self):
        weights, intercept = load_model(os.path.join(ROOT, 'model_weights.json'))
        p_short, _ = predict_churn(1, 60, 720, 'Month-to-month', weights, intercept)
        p_long, _ = predict_churn(72, 60, 720, 'Month-to-month', weights, intercept)
        self.assertLess(p_long, p_short)

    def test_two_year_contract_reduces_probability(self):
        weights, intercept = load_model(os.path.join(ROOT, 'model_weights.json'))
        p_m2m, _ = predict_churn(12, 60, 720, 'Month-to-month', weights, intercept)
        p_yr, _ = predict_churn(12, 60, 720, 'Two year', weights, intercept)
        self.assertLess(p_yr, p_m2m)

    def test_high_charge_m2m_is_elevated(self):
        weights, intercept = load_model(os.path.join(ROOT, 'model_weights.json'))
        p_hi, r_hi = predict_churn(1, 120, 100, 'Month-to-month', weights, intercept)
        self.assertGreater(p_hi, 0.5)  # strongly month-to-month + high charges

if __name__ == '__main__':
    unittest.main()
