"""Reproducibility and CLI/HTTP agreement checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inference import demo_prediction  # noqa: E402


class TestCLIAndExport(unittest.TestCase):
    def test_cli_output_matches_library_prediction(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'src' / 'inference.py')],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        probability, risk = demo_prediction()
        self.assertEqual(
            result.stdout,
            f'Churn Probability: {probability:.4f}\nRisk Level: {risk}\n',
        )
        self.assertEqual(result.stderr, '')

    def test_illustrative_export_check_does_not_modify_artifact(self):
        artifact = ROOT / 'model_weights.json'
        before = artifact.read_bytes()
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / 'train_model.py'), '--check'],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        self.assertEqual(artifact.read_bytes(), before)
        self.assertIn('no training performed', result.stdout)
        self.assertNotIn('trained', result.stdout.lower().replace('no training performed', ''))

    def test_export_is_reproducible_outside_repository_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'weights.json'
            sys.path.insert(0, str(ROOT / 'scripts'))
            try:
                from train_model import export_model_weights
                previous = Path.cwd()
                os.chdir(directory)
                try:
                    export_model_weights(output)
                finally:
                    os.chdir(previous)
            finally:
                sys.path.remove(str(ROOT / 'scripts'))
            self.assertEqual(output.read_bytes(), (ROOT / 'model_weights.json').read_bytes())
            self.assertEqual(json.loads(output.read_text())['intercept'], -0.85)


if __name__ == '__main__':
    unittest.main()
