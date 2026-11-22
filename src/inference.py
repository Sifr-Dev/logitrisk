"""Dependency-free logistic scoring with illustrative, hand-set weights."""

import json
import math
import os


FEATURE_NAMES = (
    'tenure',
    'monthlyCharges',
    'totalCharges',
    'contract_Month-to-month',
    'contract_One year',
    'contract_Two year',
)
CONTRACT_TYPES = ('Month-to-month', 'One year', 'Two year')

# Documented semantics: HIGH when p > 0.65, MEDIUM when p > 0.35.
# Classification is performed on the logit, not on the rounded probability.
# sigmoid is strictly monotonic, so z > log(t/(1-t)) is exactly equivalent to
# p > t in real arithmetic, but it does not inherit the double-rounding of
# 1/(1+exp(-z)): at threshold scale one ULP of logit frequently maps to the
# *same* double probability, which made boundary tiers depend on rounding.
HIGH_LOGIT = math.log(0.65 / 0.35)
MEDIUM_LOGIT = math.log(0.35 / 0.65)


class ModelConfigError(ValueError):
    """The model artifact is missing or contains invalid values."""


class InputValidationError(ValueError):
    """A customer feature is missing, malformed, or unsupported."""


def _finite_number(value, label, error_type=InputValidationError):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f'{label} must be a finite number')
    try:
        value = float(value)
    except OverflowError:
        raise error_type(f'{label} must be a finite number') from None
    if not math.isfinite(value):
        raise error_type(f'{label} must be a finite number')
    return value


def load_model(weights_path=None):
    """Load and validate the six-feature model artifact."""
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'model_weights.json')
    with open(weights_path, 'r', encoding='utf-8') as f:
        model = json.load(f)

    if not isinstance(model, dict) or not isinstance(model.get('features'), dict):
        raise ModelConfigError('model must be an object with a features object')
    if 'intercept' not in model:
        raise ModelConfigError('model.intercept is required')
    intercept = _finite_number(model['intercept'], 'model.intercept', ModelConfigError)

    weights = {}
    for name in FEATURE_NAMES:
        if name not in model['features']:
            raise ModelConfigError(f'model.features.{name} is required')
        weights[name] = _finite_number(
            model['features'][name], f'model.features.{name}', ModelConfigError
        )
    return weights, intercept


def predict_churn(tenure, monthly_charges, total_charges, contract_type, weights, intercept):
    """Return a sigmoid score and a risk tier; this is not a trained predictor."""
    tenure = _finite_number(tenure, 'tenure')
    monthly_charges = _finite_number(monthly_charges, 'monthlyCharges')
    total_charges = _finite_number(total_charges, 'totalCharges')
    if contract_type not in CONTRACT_TYPES:
        raise InputValidationError(
            'contractType must be one of: ' + ', '.join(CONTRACT_TYPES)
        )

    # Validate caller-supplied artifacts too; HTTP uses load_model, while the
    # public scoring function can also be used directly by other programs.
    intercept = _finite_number(intercept, 'intercept')
    z = intercept
    z += tenure * _finite_number(weights['tenure'], 'weights.tenure')
    z += monthly_charges * _finite_number(
        weights['monthlyCharges'], 'weights.monthlyCharges'
    )
    z += total_charges * _finite_number(
        weights['totalCharges'], 'weights.totalCharges'
    )
    z += _finite_number(
        weights[f'contract_{contract_type}'], f'weights.contract_{contract_type}'
    )

    # Rewrite exp(-z) for negative logits to avoid overflow in either tail.
    if z >= 0:
        probability = 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        probability = exp_z / (1.0 + exp_z)

    risk_level = 'LOW'
    if z > HIGH_LOGIT:
        risk_level = 'HIGH'
    elif z > MEDIUM_LOGIT:
        risk_level = 'MEDIUM'
    return probability, risk_level


def demo_prediction():
    weights, intercept = load_model()
    return predict_churn(12, 60, 720, 'Month-to-month', weights, intercept)


if __name__ == '__main__':
    prob, risk = demo_prediction()
    print(f"Churn Probability: {prob:.4f}")
    print(f"Risk Level: {risk}")
