Social Media Sentiment and Topic Analysis Platform
--------------------------------------------------
Overview
--------
This project implements a complete Natural Language Processing (NLP)
pipeline to analyze customer sentiment and discover discussion topics
from social media data. Using machine learning techniques, the system
processes raw Twitter airline feedback and transforms it into meaningful
insights through sentiment classification and topic modeling.

The goal of this project is to simulate a real-world analytics system
used by companies to understand customer feedback, identify common
issues, and make data-driven decisions.

The entire pipeline is containerized using Docker to ensure
reproducibility and easy deployment.

------------------------------------------------------------------------

Key Features
---------------
-   Automated text preprocessing pipeline
-   Sentiment classification using TF-IDF and Logistic Regression
-   Topic modeling using Latent Dirichlet Allocation (LDA)
-   Interactive topic visualization using pyLDAvis
-   Professional interactive dashboard built with Streamlit
-   Fully containerized application using Docker and Docker Compose
-   Reproducible machine learning pipeline

------------------------------------------------------------------------

Dataset
--------
This project uses the Twitter US Airline Sentiment Dataset, which
contains real customer feedback tweets directed at major US airlines.

Each tweet is analyzed to determine:

-   Sentiment (positive, negative, neutral)
-   Common discussion topics
-   Overall sentiment trends

------------------------------------------------------------------------

Machine Learning Pipeline
-------------------------
The system follows a structured pipeline:

1.  Data Preprocessing
    -   Removes URLs, mentions, hashtags, punctuation
    -   Converts text to lowercase
    -   Removes stopwords
    -   Applies lemmatization
    -   Saves cleaned data
2.  Feature Extraction
    -   Uses TF-IDF vectorization to convert text into numerical
        features
3.  Sentiment Classification
    -   Logistic Regression classifier
    -   Predicts sentiment labels
    -   Generates evaluation metrics
4.  Topic Modeling
    -   Uses Latent Dirichlet Allocation (LDA)
    -   Identifies key topics discussed in tweets
    -   Extracts top words per topic
5.  Visualization
    -   Interactive topic visualization using pyLDAvis
    -   Professional dashboard using Streamlit

------------------------------------------------------------------------

Model Performance
-----------------------
Sentiment classification results:

-   Accuracy: \~0.76
-   Precision (Macro): \~0.70
-   Recall (Macro): \~0.74
-   F1 Score (Macro): \~0.71

These results demonstrate strong baseline performance using traditional
machine learning techniques.

------------------------------------------------------------------------

How to Run the Project
-----------------------
-Step 1: Build and start the application

docker-compose up --build

-Step 2: Open the dashboard

http://localhost:8501

The dashboard will automatically load the trained models and display the
analysis.

------------------------------------------------------------------------

Technologies Used
------------------
-   Python
-   scikit-learn
-   NLTK
-   Streamlit
-   pyLDAvis
-   Docker
-   Pandas
-   NumPy

------------------------------------------------------------------------

Conclusion
----------
This project demonstrates a complete NLP workflow, from raw text data to
actionable insights presented in an interactive dashboard. It combines
preprocessing, machine learning, topic modeling, and visualization into
a cohesive and production-ready system.
