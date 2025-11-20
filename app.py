import os
import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/price_model.joblib')
METADATA_PATH = os.environ.get('METADATA_PATH', 'models/metadata.json')
COGS = float(os.environ.get('COGS', '50.0'))

app = FastAPI(title='Price Optimization API')


def try_load_model():
    model = None
    metadata = None
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
    except Exception as e:
        # keep model as None and surface error at call time
        print(f'Warning: failed to load model or metadata: {e}')
    return model, metadata


# Defer loading until first request to make testing and monkeypatching easier
MODEL = None
METADATA = None


def get_model_and_metadata():
    global MODEL, METADATA
    if MODEL is None or METADATA is None:
        MODEL, METADATA = try_load_model()
    return MODEL, METADATA


class PredictRequest(BaseModel):
    unit_price: float
    comp_1: float
    comp_2: float
    comp_3: float
    freight_price: float
    product_category_name: Optional[str] = None


def make_input_df(payload: dict):
    # compute price ratio features
    p = payload.copy()
    p['price_ratio_comp1'] = p['unit_price'] / (p['comp_1'] + 1e-6)
    p['price_ratio_comp2'] = p['unit_price'] / (p['comp_2'] + 1e-6)
    p['price_ratio_comp3'] = p['unit_price'] / (p['comp_3'] + 1e-6)

    # Respect metadata column ordering if available
    if METADATA and 'num_cols' in METADATA:
        num_cols = METADATA['num_cols']
        cat_cols = METADATA.get('cat_cols', [])
        row = {k: p.get(k, 0) for k in num_cols}
        for c in cat_cols:
            row[c] = p.get(c)
        df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([p])
    return df


@app.get('/healthz')
def health():
    model, _ = get_model_and_metadata()
    return {'status': 'ok', 'model_loaded': model is not None}


@app.post('/predict')
def predict(req: PredictRequest):
    model, _ = get_model_and_metadata()
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    df = make_input_df(req.dict())
    log_pred = model.predict(df)[0]
    qty = float(np.expm1(log_pred))
    qty = max(0.0, qty)
    margin = req.unit_price - COGS - req.freight_price
    profit = max(0.0, margin * qty)
    return {'predicted_qty': qty, 'predicted_profit': profit}


@app.post('/optimize')
def optimize(req: PredictRequest, min_price: float = 1.0, max_price: float = 300.0, step: float = 1.0):
    model, _ = get_model_and_metadata()
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    base = req.dict()
    prices = np.arange(min_price, max_price + 1e-9, step)
    rows = []
    for p in prices:
        r = base.copy()
        r['unit_price'] = float(p)
        r['price_ratio_comp1'] = p / (r['comp_1'] + 1e-6)
        r['price_ratio_comp2'] = p / (r['comp_2'] + 1e-6)
        r['price_ratio_comp3'] = p / (r['comp_3'] + 1e-6)
        rows.append(r)

    df = pd.DataFrame(rows)
    log_preds = model.predict(df)
    qtys = np.expm1(log_preds)
    qtys = np.clip(qtys, 0.0, None)
    margins = df['unit_price'] - COGS - df['freight_price']
    profits = np.where(margins > 0, margins * qtys, 0.0)
    best_idx = int(np.argmax(profits))
    return {
        'best_price': float(prices[best_idx]),
        'best_profit': float(profits[best_idx]),
        'best_qty': float(qtys[best_idx])
    }
