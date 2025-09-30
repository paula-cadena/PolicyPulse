#!/bin/bash

# Set up Python environment
python -m venv climate_policy_env
source climate_policy_env/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the analysis pipeline
echo "Running data ingestion and cleaning..."
python code/01_ingest_clean.py

echo "Running feature engineering..."
python code/02_feature_engineering.py

echo "Running modeling..."
python code/03_modeling.py

echo "Creating visualizations and evaluation..."
python code/04_eval_visuals.py

echo "Analysis complete! Check outputs/ folder for results."