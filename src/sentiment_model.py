import json
import os
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

def get_base_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_output_dir(base_dir: str) -> str:
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_cleaned_data(base_dir: str) -> pd.DataFrame:

    path = os.path.join(base_dir, "output", "preprocessed_data.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cleaned data at {path}")

    return pd.read_csv(path)


def load_original_data(base_dir: str) -> pd.DataFrame:

    path = os.path.join(base_dir, "data", "Tweets.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing original data at {path}")

    return pd.read_csv(path)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validate_columns(df: pd.DataFrame, required: Tuple[str, ...], name: str):

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


# ------------------------------------------------------------
# Merge datasets
# ------------------------------------------------------------

def prepare_dataset(cleaned: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:

    validate_columns(cleaned, ("tweet_id", "cleaned_text"), "cleaned_data")
    validate_columns(original, ("tweet_id", "airline_sentiment"), "original_data")

    merged = cleaned.merge(
        original[["tweet_id", "airline_sentiment"]],
        on="tweet_id",
        how="inner"
    )

    merged.dropna(inplace=True)

    if merged.empty:
        raise ValueError("Merged dataset is empty")

    return merged


# ------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------

def split_data(df: pd.DataFrame):

    X = df["cleaned_text"]
    y = df["airline_sentiment"]
    ids = df["tweet_id"]

    return train_test_split(
        X,
        y,
        ids,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


# ------------------------------------------------------------
# Optimized TF-IDF Vectorizer (major accuracy improvement)
# ------------------------------------------------------------

def train_vectorizer(X_train: pd.Series) -> TfidfVectorizer:

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 3),          # unigram + bigram + trigram
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    vectorizer.fit(X_train)

    return vectorizer


# ------------------------------------------------------------
# Optimized Linear SVM model
# ------------------------------------------------------------

def train_model(X_train_vec, y_train: pd.Series) -> LinearSVC:

    model = LinearSVC(
        C=0.75,
        class_weight="balanced",
        random_state=42,
        max_iter=5000
    )

    model.fit(X_train_vec, y_train)

    return model


# ------------------------------------------------------------
# Evaluate model
# ------------------------------------------------------------

def evaluate_model(model, X_test_vec, y_test: pd.Series):

    predictions = model.predict(X_test_vec)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro")),
        "recall_macro": float(recall_score(y_test, predictions, average="macro")),
        "f1_score_macro": float(f1_score(y_test, predictions, average="macro")),
    }

    return metrics, predictions


# ------------------------------------------------------------
# Save artifacts
# ------------------------------------------------------------

def save_vectorizer_and_model(base_dir: str, vectorizer, model):

    output_dir = get_output_dir(base_dir)

    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))

    joblib.dump(model, os.path.join(output_dir, "sentiment_model.pkl"))


def save_metrics(base_dir: str, metrics: Dict[str, float]):

    output_dir = get_output_dir(base_dir)

    with open(os.path.join(output_dir, "sentiment_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def save_predictions(base_dir: str, tweet_ids, predictions):

    output_dir = get_output_dir(base_dir)

    df = pd.DataFrame({
        "tweet_id": tweet_ids.values,
        "predicted_sentiment": predictions
    })

    df.to_csv(
        os.path.join(output_dir, "sentiment_predictions.csv"),
        index=False
    )


# ------------------------------------------------------------
# Main training pipeline
# ------------------------------------------------------------

def run():

    base_dir = get_base_dir()

    print("Loading datasets...")

    cleaned = load_cleaned_data(base_dir)
    original = load_original_data(base_dir)

    dataset = prepare_dataset(cleaned, original)

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test, id_train, id_test = split_data(dataset)

    print("Training TF-IDF vectorizer...")

    vectorizer = train_vectorizer(X_train)

    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Linear SVM model...")

    model = train_model(X_train_vec, y_train)

    print("Evaluating model...")

    metrics, predictions = evaluate_model(model, X_test_vec, y_test)

    print("Saving artifacts...")

    save_vectorizer_and_model(base_dir, vectorizer, model)

    save_metrics(base_dir, metrics)

    save_predictions(base_dir, id_test, predictions)

    print("Training completed successfully")

    print("Metrics:", metrics)


# ------------------------------------------------------------

if __name__ == "__main__":
    run()
