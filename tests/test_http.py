"""Socket-level integration tests for the standard-library HTTP adapter."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
import os
import socket
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'api'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from churn import MAX_BODY_BYTES, handler  # noqa: E402
from inference import demo_prediction  # noqa: E402


class QuietHandler(handler):
    def log_message(self, format, *args):
        pass


class TestHTTPScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, body, method='POST', content_type='application/json', headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        self.addCleanup(connection.close)
        request_headers = {} if content_type is None else {'Content-Type': content_type}
        if headers:
            request_headers.update(headers)
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        connection.request(method, '/', body=body, headers=request_headers)
        response = connection.getresponse()
        response_body = response.read()
        return response, response_body

    def assert_error(self, response, body, status, code):
        self.assertEqual(response.status, status)
        self.assertEqual(response.getheader('Content-Type'), 'application/json; charset=utf-8')
        self.assertEqual(int(response.getheader('Content-Length')), len(body))
        payload = json.loads(body)
        self.assertEqual(payload['error']['code'], code)
        self.assertIsInstance(payload['error']['message'], str)
        self.assertNotIn('traceback', body.decode().lower())

    def test_valid_request_matches_shared_scorer(self):
        response, body = self.request({
            'tenure': 12,
            'monthlyCharges': 60,
            'totalCharges': 720,
            'contractType': 'Month-to-month',
        })
        expected_probability, expected_risk = demo_prediction()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader('Content-Type'), 'application/json; charset=utf-8')
        self.assertEqual(
            json.loads(body),
            {'churnProbability': expected_probability, 'riskLevel': expected_risk},
        )
        self.assertEqual(json.loads(body)['churnProbability'], 0.7373034537835591)

    def test_media_type_parameters_are_accepted(self):
        response, _ = self.request({}, content_type='application/json; charset=utf-8')
        self.assertEqual(response.status, 200)

    def test_empty_object_uses_documented_example_defaults(self):
        response, body = self.request({})
        self.assertEqual(response.status, 200)
        self.assertAlmostEqual(json.loads(body)['churnProbability'], 0.7373034537835591)

    def test_malformed_json_and_encoding_are_client_errors(self):
        cases = (b'{broken', b'[]', b'null', b'"text"', b'\xff', b'', b'{"tenure": NaN}')
        for body in cases:
            with self.subTest(body=body):
                response, response_body = self.request(body)
                self.assertEqual(response.status, 400)
                self.assertIn(json.loads(response_body)['error']['code'],
                              ('invalid_json', 'invalid_input', 'empty_body'))

    def test_missing_content_type_and_wrong_media_type_are_rejected(self):
        for content_type in (None, 'text/plain', 'application/jsonp', ''):
            with self.subTest(content_type=content_type):
                response, body = self.request('{}', content_type=content_type)
                self.assert_error(response, body, 415, 'unsupported_media_type')

    def test_invalid_features_are_client_errors_not_disconnections(self):
        cases = []
        for field in ('tenure', 'monthlyCharges', 'totalCharges'):
            for value in ('12', None, True, [], {}, 1e400, -1e400, 10**400):
                cases.append((field, value))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                response, body = self.request({field: value})
                self.assert_error(response, body, 400, 'invalid_input')
        for value in (None, 1, 'unknown', 'month-to-month'):
            with self.subTest(contract=value):
                response, body = self.request({'contractType': value})
                self.assert_error(response, body, 400, 'invalid_input')

    def test_body_limit_and_content_length_semantics(self):
        response, body = self.request(
            json.dumps({'padding': 'x' * MAX_BODY_BYTES})
        )
        self.assert_error(response, body, 413, 'request_too_large')

        raw_response = self.raw_request(
            b'POST / HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'
            b'Content-Length: nope\r\nConnection: close\r\n\r\n{}'
        )
        self.assertIn(b' 400 ', raw_response.split(b'\r\n', 1)[0])
        self.assertIn(b'invalid_content_length', raw_response)

        raw_response = self.raw_request(
            b'POST / HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'
            b'Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\n{}\r\n0\r\n\r\n'
        )
        self.assertIn(b' 400 ', raw_response.split(b'\r\n', 1)[0])
        self.assertIn(b'invalid_encoding', raw_response)

    def raw_request(self, request):
        with socket.create_connection(self.server.server_address, timeout=3) as client:
            client.sendall(request)
            chunks = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        return b''.join(chunks)

    def test_get_and_preflight_method_semantics(self):
        response, body = self.request(None, method='GET', content_type=None)
        self.assertEqual(response.status, 405)
        self.assertEqual(response.getheader('Allow'), 'POST, OPTIONS')
        self.assertEqual(body, b'')
        response, body = self.request(None, method='OPTIONS', content_type=None)
        self.assertEqual(response.status, 204)
        self.assertEqual(response.getheader('Access-Control-Allow-Methods'), 'POST, OPTIONS')
        self.assertEqual(response.getheader('Access-Control-Allow-Headers'), 'Content-Type')
        self.assertEqual(body, b'')

    def test_persistent_connection_can_be_reused(self):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        self.addCleanup(connection.close)
        for _ in range(3):
            connection.request('POST', '/', body='{}', headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            payload = json.loads(response.read())
            self.assertEqual(payload['riskLevel'], 'HIGH')

    def test_concurrent_requests_are_isolated_and_agree_with_scorer(self):
        requests = [
            {'tenure': index, 'monthlyCharges': 40 + index,
             'totalCharges': index * 50, 'contractType': contract}
            for index, contract in enumerate(
                ('Month-to-month', 'One year', 'Two year') * 8
            )
        ]

        def score(data):
            response, body = self.request(data)
            return response.status, json.loads(body)

        with ThreadPoolExecutor(max_workers=12) as pool:
            actual = list(pool.map(score, requests))
        from inference import load_model, predict_churn
        weights, intercept = load_model()
        for data, (status, payload) in zip(requests, actual):
            probability, risk = predict_churn(
                data['tenure'], data['monthlyCharges'], data['totalCharges'],
                data['contractType'], weights, intercept
            )
            self.assertEqual(status, 200)
            self.assertEqual(payload, {'churnProbability': probability, 'riskLevel': risk})
            self.assertTrue(math.isfinite(payload['churnProbability']))


if __name__ == '__main__':
    unittest.main()
