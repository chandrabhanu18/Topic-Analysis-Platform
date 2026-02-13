import os
import re
from typing import Iterable, List

import pandas as pd


def ensure_nltk_resources() -> None:
    try:
        import nltk

        # nltk.download("stopwords", quiet=True)
        # nltk.download("wordnet", quiet=True)
        # nltk.download("omw-1.4", quiet=True)
    except Exception as exc:
        raise RuntimeError("Failed to download NLTK resources.") from exc


def load_dataset(data_dir: str) -> pd.DataFrame:
    data_path = os.path.join(data_dir, "Tweets.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")
    return pd.read_csv(data_path)


def get_text_column(frame: pd.DataFrame) -> str:
    for candidate in ["text", "tweet", "content", "message", "body"]:
        if candidate in frame.columns:
            return candidate
    return frame.columns[0]


def build_stopwords() -> set:
    from nltk.corpus import stopwords

    return set(stopwords.words("english"))


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def clean_text(text: str, stop_words: set) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    tokens = [token for token in text.split() if token not in stop_words]
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    if "tweet_id" not in frame.columns:
        raise ValueError("Required column 'tweet_id' is missing.")

    text_column = get_text_column(frame)
    working = frame[["tweet_id", text_column]].copy()
    working[text_column] = working[text_column].astype(str)

    stop_words = build_stopwords()
    working["cleaned_text"] = working[text_column].apply(lambda value: clean_text(value, stop_words))
    working = working.drop(columns=[text_column])

    working = working[working["cleaned_text"].str.strip().astype(bool)]
    return working.reset_index(drop=True)


def save_output(frame: pd.DataFrame, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "preprocessed_data.csv")
    frame.to_csv(output_path, index=False)
    return output_path


def run() -> None:
    ensure_nltk_resources()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output")

    data = load_dataset(data_dir)
    processed = preprocess(data)
    output_path = save_output(processed, output_dir)

    print(f"Saved preprocessed data to {output_path}")


if __name__ == "__main__":
    run()
