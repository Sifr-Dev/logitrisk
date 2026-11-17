<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/banner-dark.svg">
    <img alt="churn_analytics_engine — built from zero" src=".github/banner-light.svg" width="100%">
  </picture>
</p>

# churn_analytics_engine

**a logistic-scoring prototype, not a trained model.** a tiny, dependency-free
(standard-library-only) scorer: feature vector × hand-set coefficients →
sigmoid → three-tier risk (HIGH > 0.65, MEDIUM > 0.35). the runtime does not
import scikit-learn, Pandas or NumPy.

## honesty notes (this is the important part)

- **the weights are illustrative, not trained.** `scripts/train_model.py`
  does not call scikit-learn — it writes the shipped `model_weights.json`
  verbatim. no dataset is bundled, no evaluation has been run, there is no
  accuracy figure to report, and none is claimed.
- **`requirements.txt` previously listed** `scikit-learn`, `pandas`, `numpy`
  although nothing imported them; it is now `stdlib only` — see below.
- if you want a real model: bring a labeled dataset (e.g. the public Telco
  Customer Churn dataset), train offline, export the coefficients in the same
  JSON shape, and re-run the tests below.

## run

```sh
python3 src/inference.py          # Churn Probability: 0.7373 / Risk Level: HIGH
python3 -m unittest discover -s tests -v   # 5 tests, stdlib only
```

HTTP handler (same math, one POST):

```python
python3 -c "import sys; sys.path.insert(0,'api'); import churn; from http.server import ThreadingHTTPServer; srv=ThreadingHTTPServer(('127.0.0.1',8899), churn.handler); srv.serve_forever()"
curl -X POST localhost:8899/ -d '{"tenure":12,"monthlyCharges":60,"totalCharges":720,"contractType":"Month-to-month"}'
# {"churnProbability": 0.7373, "riskLevel": "HIGH"}
```

## model artifact

`model_weights.json` (~300 bytes): `intercept` + `features` —
`tenure`, `monthlyCharges`, `totalCharges`, one-hot `contract_Month-to-month`,
`contract_One year`, `contract_Two year`. sigmoid via `math.exp`; overflow-safe.

## architecture

| file | role |
| --- | --- |
| `model_weights.json` | the "model" — hand-set coefficients |
| `src/inference.py` | `predict_churn` + CLI demo (exit 0) |
| `api/churn.py` | `BaseHTTPRequestHandler` POST scorer (serverless-shaped) |
| `scripts/train_model.py` | re-exports the artifact (honest: no training happens) |
| `tests/test_inference.py` | 5 stdlib unittest checks |

## verified (2026-11-17, this machine)

`python3 src/inference.py` → `0.7373 / HIGH` · `python3 -m unittest discover -s tests -v` → `Ran 5 tests ... OK` · handler POST → 200 `{"churnProbability": 0.7373, "riskLevel": "HIGH"}`

## license

`MIT` — see `LICENSE`. built from zero.
