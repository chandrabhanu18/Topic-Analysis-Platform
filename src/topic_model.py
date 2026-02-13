import json
import joblib
from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import pyLDAvis
import pyLDAvis.lda_model


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------
# Load preprocessed data
# -------------------------------------------------------

def load_data():
    data_path = OUTPUT_DIR / "preprocessed_data.csv"

    if not data_path.exists():
        raise FileNotFoundError("preprocessed_data.csv not found")

    df = pd.read_csv(data_path)

    return df["cleaned_text"]


# -------------------------------------------------------
# Train LDA model
# -------------------------------------------------------

def train_lda(text_data, n_topics=5):

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words="english"
    )

    X = vectorizer.fit_transform(text_data)

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch"
    )

    lda_model.fit(X)

    return lda_model, vectorizer, X


# -------------------------------------------------------
# Save model
# -------------------------------------------------------

def save_model(model):

    model_path = OUTPUT_DIR / "lda_model.pkl"

    joblib.dump(model, model_path)


# -------------------------------------------------------
# Save topics JSON
# -------------------------------------------------------

def save_topics(model, vectorizer, n_words=10):

    words = vectorizer.get_feature_names_out()

    topics = {}

    for topic_idx, topic in enumerate(model.components_):

        top_indices = topic.argsort()[-n_words:][::-1]

        topics[f"topic_{topic_idx}"] = [
            words[i] for i in top_indices
        ]

    topics_path = OUTPUT_DIR / "topics.json"

    with open(topics_path, "w") as f:
        json.dump(topics, f, indent=4)


# -------------------------------------------------------
# Save REAL pyLDAvis visualization
# -------------------------------------------------------

def save_visualization(model, vectorizer, X):

    vis = pyLDAvis.lda_model.prepare(
        model,
        X,
        vectorizer,
        mds="pcoa"
    )

    vis_path = OUTPUT_DIR / "lda_visualization.html"

    pyLDAvis.save_html(vis, str(vis_path))


# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------

def run():

    print("Loading data...")

    text_data = load_data()

    print("Training LDA model...")

    lda_model, vectorizer, X = train_lda(text_data)

    print("Saving model...")

    save_model(lda_model)

    print("Saving topics...")

    save_topics(lda_model, vectorizer)

    print("Saving visualization...")

    save_visualization(lda_model, vectorizer, X)

    print("Topic modeling completed. Outputs saved to output/.")


# -------------------------------------------------------

if __name__ == "__main__":

    run()
