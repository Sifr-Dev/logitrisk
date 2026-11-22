"""Stdlib-only checks for the illustrative logistic scorer."""
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from inference import (  # noqa: E402
    InputValidationError,
    ModelConfigError,
    load_model,
    predict_churn,
)

ROOT = os.path.join(HERE, '..')


class TestModelArtifact(unittest.TestCase):
    def setUp(self):
        self.weights, self.intercept = load_model(os.path.join(ROOT, 'model_weights.json'))

    def write_model(self, value):
        temporary = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        with temporary:
            json.dump(value, temporary)
        self.addCleanup(os.unlink, temporary.name)
        return temporary.name

    def test_shipped_artifact_is_exact(self):
        self.assertEqual(self.intercept, -0.85)
        self.assertEqual(
            self.weights,
            {
                'tenure': -0.045,
                'monthlyCharges': 0.018,
                'totalCharges': -0.00015,
                'contract_Month-to-month': 1.45,
                'contract_One year': 0.05,
                'contract_Two year': -1.25,
            },
        )

    def test_missing_file_is_an_os_error(self):
        with self.assertRaises(FileNotFoundError):
            load_model(os.path.join(ROOT, 'does-not-exist.json'))

    def test_malformed_json_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{broken')
        self.addCleanup(os.unlink, f.name)
        with self.assertRaises(json.JSONDecodeError):
            load_model(f.name)

    def test_malformed_model_shapes_and_values_are_rejected(self):
        valid = {'intercept': self.intercept, 'features': dict(self.weights)}
        cases = [
            None,
            [],
            {},
            {'intercept': 0},
            {'features': dict(self.weights)},
            {'intercept': '0', 'features': dict(self.weights)},
            {'intercept': True, 'features': dict(self.weights)},
            {'intercept': math.inf, 'features': dict(self.weights)},
        ]
        missing_feature = {'intercept': 0, 'features': dict(self.weights)}
        del missing_feature['features']['tenure']
        cases.append(missing_feature)
        non_mapping_features = dict(valid)
        non_mapping_features['features'] = []
        cases.append(non_mapping_features)
        nonfinite_weight = {'intercept': 0, 'features': dict(self.weights)}
        nonfinite_weight['features']['tenure'] = math.nan
        cases.append(nonfinite_weight)
        for model in cases:
            with self.subTest(model=model):
                with self.assertRaises(ModelConfigError):
                    load_model(self.write_model(model))

    def test_sigmoid_is_exact_at_zero(self):
        weights = dict.fromkeys(self.weights, 0.0)
        probability, risk = predict_churn(0, 0, 0, 'One year', weights, 0.0)
        self.assertEqual(probability, 0.5)
        self.assertEqual(risk, 'MEDIUM')

    def test_extreme_logits_are_finite_and_in_bounds(self):
        weights = dict.fromkeys(self.weights, 0.0)
        cases = ((-1000.0, 0.0, 'LOW'), (-746.0, 0.0, 'LOW'),
                 (-745.0, 5e-324, 'LOW'), (745.0, 1.0, 'HIGH'),
                 (1000.0, 1.0, 'HIGH'))
        for logit, expected, risk in cases:
            with self.subTest(logit=logit):
                probability, actual_risk = predict_churn(
                    0, 0, 0, 'One year', weights, logit
                )
                self.assertEqual(probability, expected)
                self.assertEqual(actual_risk, risk)
                self.assertTrue(math.isfinite(probability))

    def test_thresholds_are_strict_and_exact(self):
        # The tier boundary is the contract under test, so it is asserted
        # exactly: at the threshold logit the lower tier applies, one ULP above
        # it the higher tier applies, one ULP below it the lower tier applies.
        # The returned *probability* is a separate, rounded quantity: near a
        # threshold one ULP of logit can map to the identical double, so the
        # probability assertions use exact equality where it is representable
        # and monotonic non-strict ordering otherwise. Asserting p > t after a
        # 1-ULP logit step would assert something no double sigmoid can satisfy.
        weights = dict.fromkeys(self.weights, 0.0)
        for threshold, at_boundary, just_above in (
            (0.35, 'LOW', 'MEDIUM'),
            (0.65, 'MEDIUM', 'HIGH'),
        ):
            logit = math.log(threshold / (1.0 - threshold))
            with self.subTest(threshold=threshold, position='at'):
                probability, risk = predict_churn(0, 0, 0, 'One year', weights, logit)
                self.assertTrue(math.isclose(probability, threshold, rel_tol=0.0, abs_tol=1e-15),
                                (probability, threshold))
                self.assertEqual(risk, at_boundary)
            with self.subTest(threshold=threshold, position='above'):
                probability, risk = predict_churn(
                    0, 0, 0, 'One year', weights, math.nextafter(logit, math.inf)
                )
                self.assertGreaterEqual(probability, threshold)
                self.assertEqual(risk, just_above)
            with self.subTest(threshold=threshold, position='below'):
                probability, risk = predict_churn(
                    0, 0, 0, 'One year', weights, math.nextafter(logit, -math.inf)
                )
                self.assertLessEqual(probability, threshold)
                self.assertEqual(risk, at_boundary)

    def test_classification_is_monotonic_across_threshold_neighbourhood(self):
        # Sweep the representable logits around both thresholds: the tier must
        # never move "backwards" as the logit increases.
        weights = dict.fromkeys(self.weights, 0.0)
        order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        for threshold in (0.35, 0.65):
            centre = math.log(threshold / (1.0 - threshold))
            previous = -1
            logit = centre
            for _ in range(64):
                logit = math.nextafter(logit, -math.inf)
            for _ in range(129):
                _, risk = predict_churn(0, 0, 0, 'One year', weights, logit)
                self.assertGreaterEqual(order[risk], previous)
                previous = order[risk]
                logit = math.nextafter(logit, math.inf)

    def test_contract_one_hot_effects(self):
        arguments = (12, 60, 720)
        probabilities = {
            contract: predict_churn(*arguments, contract, self.weights, self.intercept)[0]
            for contract in ('Month-to-month', 'One year', 'Two year')
        }
        self.assertLess(probabilities['Two year'], probabilities['One year'])
        self.assertLess(probabilities['One year'], probabilities['Month-to-month'])

    def test_invalid_direct_inputs_are_rejected(self):
        for field in ('tenure', 'monthly_charges', 'total_charges'):
            values = {
                'tenure': 12,
                'monthly_charges': 60,
                'total_charges': 720,
            }
            for bad_value in (None, '12', True, math.nan, math.inf, -math.inf, 10**400):
                values[field] = bad_value
                with self.subTest(field=field, value=bad_value):
                    with self.assertRaises(InputValidationError):
                        predict_churn(**values, contract_type='One year',
                                      weights=self.weights, intercept=self.intercept)
        for contract in (None, 'month-to-month', 'unknown', 1):
            with self.subTest(contract=contract):
                with self.assertRaises(InputValidationError):
                    predict_churn(12, 60, 720, contract, self.weights, self.intercept)


if __name__ == '__main__':
    unittest.main()
