Social Media Sentiment and Topic Analysis Platform
-----------------------------------------------------
Overview
---------
This project implements a complete Natural Language Processing (NLP) pipeline to analyze sentiment and discover discussion topics from social media data. The system processes Twitter airline feedback and transforms it into actionable insights using sentiment classification and topic modeling.

The pipeline includes preprocessing, feature extraction, model training, topic modeling, and visualization, all integrated into a professional interactive dashboard.

The application is fully containerized using Docker to ensure reproducibility, portability, and easy deployment.

Key Features
-------------
Automated text preprocessing pipeline
Feature extraction using TF-IDF vectorization

Sentiment classification using Linear Support Vector Machine (LinearSVC):
Topic modeling using Latent Dirichlet Allocation (LDA)
Interactive topic visualization using pyLDAvis
Professional interactive dashboard built with Streamlit
Fully containerized application using Docker and Docker Compose
Reproducible end-to-end machine learning workflow


Dataset:
-----------
This project uses the Twitter US Airline Sentiment dataset.

Dataset characteristics:
---------------------------
Total tweets: 14,640

Sentiment classes:
-------------------
Positive
Negative
Neutral
Each tweet contains customer feedback directed at US airlines.

Machine Learning Pipeline
1. Data Preprocessing:
Removes URLs, mentions, hashtags, punctuation
Converts text to lowercase
Removes stopwords using NLTK
Applies lemmatization
Saves cleaned data to output/preprocessed_data.csv
2. Feature Extraction:
Uses TF-IDF vectorizer
Converts cleaned text into numerical feature vectors
3. Sentiment Classification:
Uses Linear Support Vector Machine (LinearSVC)
Trained on TF-IDF features
Generates predictions and performance metrics
Saves model and vectorizer
4. Topic Modeling:
Uses Latent Dirichlet Allocation (LDA)
Identifies major discussion topics
Extracts top keywords per topic
Saves model and topics
5. Visualization and Dashboard:
Interactive dashboard built with Streamlit
Displays metrics, sentiment distribution, topics, and LDA visualization
Loads precomputed artifacts from output directory

Model Performance:
-----------------

Sentiment classification results:
--------------------------------
Accuracy: 0.78
Precision (Macro): 0.73
Recall (Macro): 0.72
F1 Score (Macro): 0.72

These results demonstrate strong baseline performance using classical machine learning techniques.

How to Run the Project Using Docker
-----------------------------------------
Step 1: Build and start the application
---------------------------------------
docker-compose up --build

Step 2: Open the dashboard
---------------------------
http://localhost:8501


The dashboard will automatically load the trained models and display results.

Dashboard Features:
---------------------
The Streamlit dashboard provides:

Dataset overview:
--------------------
Sentiment performance metrics
Sentiment distribution chart
Topic keyword exploration
Interactive LDA visualization

Technologies Used:
--------------------
Python
scikit-learn
NLTK
Streamlit
pyLDAvis
Docker
Pandas
NumPy
Gensim
Plotly

Conclusion:
---------------

This project demonstrates a complete, production-ready NLP pipeline for sentiment analysis and topic modeling. The system integrates preprocessing, machine learning, topic modeling, visualization, and containerization into a fully reproducible and deployable application.