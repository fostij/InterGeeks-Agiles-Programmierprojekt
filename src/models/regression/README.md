# Regression Module - Vehicle Claim Prediction

This module provides a robust, automated machine learning pipeline designed to predict vehicle insurance claim amounts. It follows a modular architecture to ensure data integrity, high predictive performance, and model consistency across the project.

## Pipeline Architecture
The module is structured into distinct, decoupled components to maintain high software engineering standards:

1. **Data Preparation (`prepare_data.py`)**: 
   - Handles ETL (Extract, Transform, Load) processes.
   - Cleans raw data and handles missing values.
   - Performs One-Hot Encoding on categorical variables to ensure compatibility with machine learning algorithms.

2. **Model Training (`train_model.py`)**: 
   - Utilizes a `RandomForestRegressor` to capture complex, non-linear relationships in the data.
   - Ensures high predictive accuracy (currently achieving an **$R^2$ score of 0.9528**).
   - Serializes the trained model into a `model.pkl` artifact for deployment.

3. **Inference (`predict_model.py`)**: 
   - Provides a streamlined interface for predicting claim amounts for new accident entries.
   - Implements automated feature alignment to ensure input data matches the training schema.

4. **Orchestration (`regression.py`)**: 
   - The central controller that executes the full end-to-end pipeline, ensuring consistent model updates.

## Data Setup
Ensure you have the `dataset.csv` file available on your local machine. Place the file in the following directory:
`data/output/dataset.csv`

The pipeline expects this file to be present to execute the training process.

## Project Structure
```text
/InterGeeks-Agiles-Programmierprojekt
├── data/
│   └── output/
│       └── dataset.csv       # Source dataset
├── src/
│   └── models/
│       └── regression/
│           ├── prepare_data.py   # Data engineering module
│           ├── train_model.py    # Training & serialization logic
│           ├── predict_model.py  # Prediction & inference engine
│           └── regression.py     # Main execution pipeline
└── README.md

# Regression Module - Vehicle Claim Prediction

This module provides a robust, automated machine learning pipeline for predicting insurance claims.

## Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install pandas scikit-learn joblib

python regression.py

python predict_model.py

