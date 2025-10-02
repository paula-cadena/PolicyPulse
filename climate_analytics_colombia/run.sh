#!/bin/bash

# Data Scientist Technical Assessment - Pipeline Execution
# Climate Impact on Financial Outcomes in Colombian Municipalities

set -e  # Exit on any error

echo "Starting pipeline execution..."
echo "======================================"

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data_processed
mkdir -p outputs/figures
mkdir -p outputs/tables  
mkdir -p outputs/model_artifacts
mkdir -p outputs/logs

# Check if virtual environment exists, if not create it
if [ ! -d "climate_env" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv climate_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source climate_env/bin/activate

# Upgrade pip and install requirements
echo "Installing/updating Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 1: Data Ingestion and Cleaning
echo "Step 1: Data Ingestion and Cleaning..."
$PYTHON_CMD code/01_ingest_clean.py

# Step 2: Feature Engineering
echo "Step 2: Feature Engineering..."
$PYTHON_CMD code/02_feature_engineering.py

# Step 3: Modeling
echo "Step 3: Modeling..."
$PYTHON_CMD code/03_modeling.py

# Step 4: Evaluation and Visualization
echo "Step 4: Evaluation and Visualization..."
$PYTHON_CMD code/04_eval_visuals.py

echo "======================================"
echo "Pipeline execution completed successfully!"
echo "Check outputs/ and docs/ for results."
echo "Generated files:"
find outputs/ -type f -name "*.png" -o -name "*.csv" -o -name "*.txt" -o -name "*.pkl" | head -10
find docs/ -type f -name "*.md" | head -10