# Fashionista Price Optimization

<p align="center">
  <img src="https://raw.githubusercontent.com/your/repo/main/logo-placeholder.png" alt="Logo" width="200"/>
</p>

A compact, end-to-end demo that finds the profit-maximizing price for a product using a trained Ridge Regression demand model. This repo includes a FastAPI backend that exposes an optimization endpoint and a Streamlit front-end (demo dashboard) so you can show results to stakeholders in seconds.

---

## Highlights

- FastAPI REST API that serves a Ridge Regression demand model (pre-extracted coefficients).
- Streamlit interactive dashboard for rapid demo and client walkthroughs.
- Model trained with a time-based split (train < May 2018, test >= May 2018).
- Defensive production-friendly handling for edge cases (NaN/inf, division-by-zero).

---

## Project structure

- `api.py` — FastAPI application exposing `/optimize_price` (POST) and basic info endpoints.
- `app.py` — Streamlit demo that calls the API and displays results with attractive metrics.
- `price_opt.ipynb` — Jupyter notebook with the training pipeline, scaler params and coefficient extraction.
- `test_api.py` — Simple Python test script that POSTs a sample payload to the API.
- `retail_price.csv` — The dataset used to train the model (feature engineering performed in the notebook).

---

## Model summary

- Model: Ridge Regression (alpha=20)
- Reported R² on the time-split test set: 0.9425 (as printed by the training notebook)
- Feature set: ~38 features including numeric inputs and one-hot product-category dummies.
- Categories included (example): `bed_bath_table`, `computers_accessories`, `consoles_games`, `cool_stuff`, `furniture_decor`, `garden_tools`, `health_beauty`, `perfumery`, `watches_gifts`.

> Note: Exact model coefficients and scaler parameters are embedded in `api.py` and were exported from `price_opt.ipynb`.

---

## Tools & Visuals

Here are the main tools and libraries used in this project — visually highlighted so you can quickly show stakeholders the tech stack.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white" />
  <img alt="pandas" src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" />
  <img alt="uvicorn" src="https://img.shields.io/badge/uvicorn-7A61FF?logo=python&logoColor=white" />
  <img alt="ngrok" src="https://img.shields.io/badge/ngrok-14A0E2?logo=ngrok&logoColor=white" />
</p>

<p align="center">
  <img src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" alt="FastAPI logo" width="140" style="margin:12px;" />
  <img src="https://streamlit.io/images/brand/streamlit-mark-color.svg" alt="Streamlit logo" width="90" style="margin:12px;" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/0a/Python.svg" alt="Python logo" width="70" style="margin:12px;" />
</p>

You can use these visuals in client decks to quickly communicate the architecture: a Python ML model exported to a lightweight FastAPI service, with an interactive Streamlit front-end for demos.

---

## Quick start (local)

These steps assume you are on macOS (zsh) and using the included virtual environment at `.venv`. Adjust commands for your environment if different.

1) Create / activate virtual environment (if not already created):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies (minimal set):

```bash
pip install --upgrade pip
pip install fastapi uvicorn scikit-learn pandas streamlit requests
```

(If you prefer, create a `requirements.txt` from the environment and pin versions.)

3) Start the FastAPI server (the project uses port 8001):

```bash
# from project root
uvicorn api:app --host 127.0.0.1 --port 8001 --reload
```

4) Start the Streamlit demo (default port 8501):

```bash
# from project root; use the venv python if needed
python -m streamlit run app.py
```

5) Open the Streamlit app in your browser: `http://localhost:8501`

---

## API: endpoints & payload

Base URL (local): `http://127.0.0.1:8001`

- GET `/` — basic API info and model summary
- GET `/optimize_price` — docs for the optimization endpoint
- POST `/optimize_price` — compute optimal price. The API expects a JSON body with the following fields:

Request JSON schema (example):

```json
{
  "category": "bed_bath_table",
  "cogs": 45.0,
  "freight": 15.0,
  "comp1": 120.0,
  "comp2": 150.0,
  "comp3": 100.0,
  "score": 4.2,
  "customers": 50
}
```

Response (example):

```json
{
  "optimal_price": 275.5,
  "max_profit": 1445.77,
  "predicted_qty": 6.71
}
```

Notes:
- The API performs internal feature engineering (price ratios/differences) and uses scaler parameters embedded in `api.py`.
- Defensive checks skip non-finite values (NaN / inf) and report errors cleanly.

---

## Streamlit demo

The Streamlit app (`app.py`) includes:

- Input panel for: category, COGS, freight, three competitor prices, product score (1–5) and estimated number of customers.
- An `Optimize Price` button that POSTs to `/optimize_price` and displays:
  - Optimal price, predicted demand, and max profit (attractive metric cards)
  - Markup %, revenue, profit margin and a recommendation box

UX & edge-case handling added:
- Division-by-zero guard (when COGS + freight = 0)
- Graceful handling of API-side errors

---

## How I verified things (quick tests you can re-run)

- Check API info:

```bash
curl http://127.0.0.1:8001/
```

- Run a sample POST (using Python `requests`):

```python
import requests
payload = {
  'category': 'bed_bath_table',
  'cogs': 45,
  'freight': 15,
  'comp1': 120,
  'comp2': 150,
  'comp3': 100,
  'score': 4.2,
  'customers': 50
}
resp = requests.post('http://127.0.0.1:8001/optimize_price', json=payload, timeout=10)
print(resp.status_code, resp.json())
```

---

## Sharing with a client (quick public demo)

If you want to show the Streamlit dashboard to a client without deploying, use `ngrok` to expose the local port.

1) Install `ngrok` (macOS):

```bash
brew install --cask ngrok
```

2) Start Streamlit locally (port 8501), then run:

```bash
ngrok http 8501
```

3) Share the `https://...ngrok.io` address shown by `ngrok` with your client. Note: ngrok free plans sleep sessions after some time.

Security note: only expose demo data or anonymized samples when using public tunnels.

---

## Troubleshooting

- Streamlit command not found: ensure the venv is activated or use `python -m streamlit run app.py`.
- API 422 errors: verify you send the correct field names expected by `api.py` (`comp1`, `comp2`, `comp3`, `score`, `customers`).
- API 500 / non-finite errors: check input values (no negative competitor prices) and review console logs. API includes checks and returns clear `{"error": "..."}` messages on bad inputs.
- Socket/port in use: if port 8001 or 8501 is busy, change the 
port when launching (`--port <N>` for uvicorn or `streamlit run app.py --server.port <N>`).

---

## Next steps (recommended)

- Add a pinned `requirements.txt` or `pyproject.toml` for reproducible installs.
- Add a small `launcher.sh` that activates venv and runs both API and Streamlit (for demo convenience).
- Add unit tests for the API (`pytest`) covering at least: normal case, zero-cost edge case, and invalid payload.
- Consider Dockerizing the stack if you want an easily-shareable, reproducible demo.

---

## Credits & License

Created for a client demo. Feel free to adapt the UI and model exports for your use. Add a LICENSE file if you plan to share publicly.

---

If you'd like, I can:
- create `requirements.txt` and a `launcher.sh` script,
- add a small `README` section for the model interpretability notes,
- or generate a shareable ngrok tutorial with screenshots.


Which of those would you like next?
