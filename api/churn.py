"""HTTP adapter for the shared, standard-library scoring implementation."""

from http.server import BaseHTTPRequestHandler
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from inference import InputValidationError, load_model, predict_churn  # noqa: E402


MAX_BODY_BYTES = 16 * 1024
# These are example values for a partial request, not imputation learned from data.
DEFAULT_INPUT = {
    'tenure': 12,
    'monthlyCharges': 60,
    'totalCharges': 720,
    'contractType': 'Month-to-month',
}


class NonFiniteConstantError(ValueError):
    """Infinity/NaN are valid JSON decoder tokens but never valid features."""


def _reject_non_finite(token):
    raise NonFiniteConstantError(f'{token} is not a finite feature value')


class handler(BaseHTTPRequestHandler):
    server_version = 'LogitRisk/1.0'

    def _send_json(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status, code, message):
        self._send_json(status, {'error': {'code': code, 'message': message}})

    def do_POST(self):
        media_type = self.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
        if media_type != 'application/json':
            self._send_error(
                415, 'unsupported_media_type', 'Content-Type must be application/json'
            )
            return

        transfer_encoding = self.headers.get('Transfer-Encoding', '').strip().lower()
        if transfer_encoding:
            self.close_connection = True
            self._send_error(400, 'invalid_encoding', 'chunked request bodies are not supported')
            return

        try:
            content_length = int(self.headers.get('Content-Length', ''))
        except ValueError:
            self._send_error(400, 'invalid_content_length', 'Content-Length must be an integer')
            return
        if content_length < 0:
            self._send_error(400, 'invalid_content_length', 'Content-Length must not be negative')
            return
        if content_length > MAX_BODY_BYTES:
            self.close_connection = True
            self._send_error(
                413, 'request_too_large', f'request body must be at most {MAX_BODY_BYTES} bytes'
            )
            return
        if content_length == 0:
            self._send_error(400, 'empty_body', 'request body must contain a JSON object')
            return

        try:
            body = self.rfile.read(content_length)
            data = json.loads(
                body.decode('utf-8'), parse_constant=_reject_non_finite
            )
        except NonFiniteConstantError as exc:
            self._send_error(400, 'invalid_input', str(exc))
            return
        except (UnicodeDecodeError, ValueError):
            self._send_error(400, 'invalid_json', 'request body must be valid UTF-8 JSON')
            return
        if not isinstance(data, dict):
            self._send_error(400, 'invalid_input', 'JSON body must be an object')
            return

        values = {}
        for field in ('tenure', 'monthlyCharges', 'totalCharges'):
            value = data.get(field, DEFAULT_INPUT[field])
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self._send_error(400, 'invalid_input', f'{field} must be a finite number')
                return
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                self._send_error(400, 'invalid_input', f'{field} must be a finite number')
                return
            values[field] = value
        contract_type = data.get('contractType', DEFAULT_INPUT['contractType'])
        if not isinstance(contract_type, str):
            self._send_error(400, 'invalid_input', 'contractType must be a supported string')
            return

        try:
            weights, intercept = load_model()
            probability, risk_level = predict_churn(
                tenure=values['tenure'],
                monthly_charges=values['monthlyCharges'],
                total_charges=values['totalCharges'],
                contract_type=contract_type,
                weights=weights,
                intercept=intercept,
            )
        except InputValidationError as exc:
            self._send_error(400, 'invalid_input', str(exc))
            return
        except (OSError, ValueError):
            self._send_error(
                500, 'model_unavailable', 'the illustrative scoring artifact could not be loaded'
            )
            return

        self._send_json(
            200, {'churnProbability': probability, 'riskLevel': risk_level}
        )

    def do_GET(self):
        self.send_response(405)
        self.send_header('Allow', 'POST, OPTIONS')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '600')
        self.send_header('Content-Length', '0')
        self.end_headers()
