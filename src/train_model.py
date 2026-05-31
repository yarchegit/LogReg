import argparse
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data_prep import load_sessions_hits, build_training_table


def train(
    sessions_path: str,
    hits_path: str,
    models_dir: str = 'models'
):
    # 1. Загружаем и подготавливаем данные
    sessions, hits = load_sessions_hits(sessions_path, hits_path)
    df = build_training_table(sessions, hits)
    df = df.sample(min(200000, len(df)), random_state=42)

    target_col = 'target'
    drop_cols = ['session_id', 'client_id']

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols + [target_col]
    ]

    X = df[feature_cols]
    y = df[target_col]

    # 2. Разделяем на числовые/категориальные
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = 'passthrough'
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=True
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ]
    )

    # 3. Модель
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight='balanced'
    )

    clf = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )

    # 4. Train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Обучение
    clf.fit(X_train, y_train)

    # 6. Оценка
    y_pred_proba = clf.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, y_pred_proba)
    print(f'Validation ROC-AUC: {roc_auc:.4f}')

    # 7. Сохранение пайплайна
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'model.pkl')
    joblib.dump(clf, model_path)
    print(f'Model (pipeline) saved to {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions_path', type=str, required=True)
    parser.add_argument('--hits_path', type=str, required=True)
    parser.add_argument('--models_dir', type=str, default='models')

    args = parser.parse_args()

    train(
        sessions_path=args.sessions_path,
        hits_path=args.hits_path,
        models_dir=args.models_dir
    )