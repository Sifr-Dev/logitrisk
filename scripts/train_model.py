"""Re-export the illustrative coefficient artifact; no training occurs."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'model_weights.json'


def illustrative_model():
    """Return hand-set example values, not coefficients learned from data."""
    return {
        'intercept': -0.85,
        'features': {
            'tenure': -0.045,
            'monthlyCharges': 0.018,
            'totalCharges': -0.00015,
            'contract_Month-to-month': 1.45,
            'contract_One year': 0.05,
            'contract_Two year': -1.25,
        },
    }


def export_model_weights(output_path=DEFAULT_OUTPUT, check=False):
    """Write the illustrative artifact, or verify it without modifying it."""
    output_path = Path(output_path)
    serialized = json.dumps(illustrative_model(), indent=4) + '\n'
    if check:
        if not output_path.exists() or output_path.read_text(encoding='utf-8') != serialized:
            raise SystemExit(f'[FAIL] {output_path} differs from the illustrative export')
        print(f'[OK] {output_path} matches the illustrative export; no training performed')
        return output_path

    output_path.write_text(serialized, encoding='utf-8')
    print(f'[OK] Illustrative hand-set coefficients exported to {output_path}; no training performed')
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check', action='store_true', help='verify the shipped artifact without writing it'
    )
    args = parser.parse_args()
    export_model_weights(check=args.check)
