import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib


def build_and_train(csv_path: str = 'retail_price.csv', model_path: str = 'models/price_model.joblib', metadata_path: str = 'models/metadata.json'):
    # 1. Load data
    df = pd.read_csv(csv_path)
    # match notebook behavior
    drop_cols = [c for c in ['product_id', 'month_year', 'total_price'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 2. Feature engineering - price ratios
    df['price_ratio_comp1'] = df['unit_price'] / (df['comp_1'] + 1e-6)
    df['price_ratio_comp2'] = df['unit_price'] / (df['comp_2'] + 1e-6)
    df['price_ratio_comp3'] = df['unit_price'] / (df['comp_3'] + 1e-6)

    # 3. Columns
    num_cols = [
        'unit_price', 'comp_1', 'comp_2', 'comp_3', 'freight_price',
        'price_ratio_comp1', 'price_ratio_comp2', 'price_ratio_comp3'
    ]
    cat_cols = [c for c in ['product_category_name'] if c in df.columns]

    X = df[num_cols + cat_cols]
    y = np.log1p(df['qty'])

    # 4. Preprocessor and pipeline
    # Construct OneHotEncoder in a way that's compatible with different sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        # older sklearn versions use `sparse` param
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', ohe, cat_cols),
    ], remainder='drop')

    gbr = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    pipe = Pipeline([
        ('pre', preprocessor),
        ('model', gbr)
    ])

    # 5. Fit
    print('Fitting model...')
    pipe.fit(X, y)

    # 6. Save artifacts
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    meta = {
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'model_path': model_path,
        'metadata_version': 1,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'target': 'log1p(qty)'
    }
    with open(metadata_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Model saved to: {model_path}')
    print(f'Metadata saved to: {metadata_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and save price optimization model')
    parser.add_argument('--csv', default='retail_price.csv', help='Path to training CSV')
    parser.add_argument('--model-path', default='models/price_model.joblib', help='Output path for model')
    parser.add_argument('--metadata-path', default='models/metadata.json', help='Output path for metadata')
    args = parser.parse_args()

    build_and_train(csv_path=args.csv, model_path=args.model_path, metadata_path=args.metadata_path)
