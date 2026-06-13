import argparse
import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "sample_infer.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "predictions.csv"


def load_model(model_path: Path):
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    clip_threshold = artifact.get("clip_threshold", None)
    return pipeline, clip_threshold


def prepare_features(df: pd.DataFrame, clip_threshold: float | None):
    df = df.copy()
    if clip_threshold is not None and "visit_number" in df.columns:
        df["visit_number"] = df["visit_number"].clip(upper=clip_threshold)
    return df


def run_inference(model_path: Path, input_path: Path, output_path: Path):
    pipeline, clip_threshold = load_model(model_path)

    df = pd.read_csv(input_path)
    X = prepare_features(df, clip_threshold)

    proba = pipeline.predict_proba(X)[:, 1]

    out = df.copy()
    out["proba"] = proba
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with model.pkl")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model.pkl",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to input CSV with visits",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to save predictions",
    )
    args = parser.parse_args()

    run_inference(Path(args.model), Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()