import json
import os
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def load_cleaned_data(base_dir: str) -> pd.DataFrame:
    cleaned_path = os.path.join(base_dir, "output", "preprocessed_data.csv")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data not found at {cleaned_path}.")
    return pd.read_csv(cleaned_path)


def load_original_data(base_dir: str) -> pd.DataFrame:
    original_path = os.path.join(base_dir, "data", "Tweets.csv")
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original data not found at {original_path}.")
    return pd.read_csv(original_path)


def validate_columns(frame: pd.DataFrame, required: Tuple[str, ...], context: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {context}: {', '.join(missing)}")


def prepare_dataset(cleaned: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    validate_columns(cleaned, ("tweet_id", "cleaned_text"), "cleaned data")
    validate_columns(original, ("tweet_id", "airline_sentiment"), "original data")

    cleaned = cleaned[["tweet_id", "cleaned_text"]].copy()
    original = original[["tweet_id", "airline_sentiment"]].copy()

    merged = cleaned.merge(original, on="tweet_id", how="inner")
    merged = merged.dropna(subset=["cleaned_text", "airline_sentiment"])
    merged["cleaned_text"] = merged["cleaned_text"].astype(str)
    merged["airline_sentiment"] = merged["airline_sentiment"].astype(str)

    if merged.empty:
        raise ValueError("Merged dataset is empty. Check tweet_id alignment.")

    return merged


def split_data(frame: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    X = frame["cleaned_text"]
    y = frame["airline_sentiment"]
    tweet_ids = frame["tweet_id"]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        tweet_ids,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, id_train, id_test


def train_vectorizer(X_train: pd.Series) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    vectorizer.fit(X_train)
    return vectorizer


def train_model(X_train_vec, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_vec, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test_vec, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test_vec)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "f1_score_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
    }
    return metrics


def save_artifacts(base_dir: str, vectorizer: TfidfVectorizer, model: LogisticRegression) -> None:
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(output_dir, "sentiment_model.pkl"))


def save_metrics(base_dir: str, metrics: Dict[str, float]) -> None:
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "sentiment_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_predictions(
    base_dir: str,
    tweet_ids: pd.Series,
    predictions: pd.Series,
) -> None:
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "sentiment_predictions.csv")
    output_frame = pd.DataFrame(
        {
            "tweet_id": tweet_ids.values,
            "predicted_sentiment": predictions,
        }
    )
    output_frame.to_csv(output_path, index=False)


def run() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    cleaned = load_cleaned_data(base_dir)
    original = load_original_data(base_dir)
    dataset = prepare_dataset(cleaned, original)

    X_train, X_test, y_train, y_test, id_train, id_test = split_data(dataset)

    vectorizer = train_vectorizer(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = train_model(X_train_vec, y_train)

    metrics = evaluate_model(model, X_test_vec, y_test)
    save_artifacts(base_dir, vectorizer, model)
    save_metrics(base_dir, metrics)

    predictions = model.predict(X_test_vec)
    save_predictions(base_dir, id_test, predictions)

    print("Training completed. Outputs saved to output/.")


if __name__ == "__main__":
    run()
