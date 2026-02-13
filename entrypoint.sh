#!/bin/bash

echo "Starting Streamlit dashboard..."

streamlit run app.py --server.port=8501 --server.address=0.0.0.0
