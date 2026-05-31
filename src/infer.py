import argparse
import json
import joblib
import pandas as pd
from typing import Any


def load_model(model_path: str):
    return joblib.load(model_path)


def load_input(input_path: str, input_format: str) -> pd.DataFrame:
    if input_format == 'csv':
        return pd.read_csv(input_path)
    elif input_format == 'json':
        with open(input_path, 'r', encoding='utf-8') as f:
            data: Any = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    else:
        raise ValueError('input_format must be "csv" or "json"')


def predict(
    model_path: str,
    input_path: str,
    input_format: str = 'csv',
    threshold: float = 0.5
) -> pd.DataFrame:
    clf = load_model(model_path)
    X = load_input(input_path, input_format)

    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    res = X.copy()
    res['pred_proba'] = proba
    res['pred_label'] = pred

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/model.pkl')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--input_format', type=str, choices=['csv', 'json'], default='csv')
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    result = predict(
        model_path=args.model_path,
        input_path=args.input_path,
        input_format=args.input_format,
        threshold=args.threshold
    )

    print(result[['pred_label', 'pred_proba']])