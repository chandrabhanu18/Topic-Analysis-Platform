#!/bin/bash

echo "Running preprocessing..."
python -m src.preprocess

echo "Training sentiment model..."
python -m src.sentiment_model

echo "Training topic model..."
python -m src.topic_model

echo "Starting Streamlit..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
