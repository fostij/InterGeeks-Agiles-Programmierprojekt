# Regression Module - Vehicle Claim Prediction

This module provides an automated machine learning pipeline for predicting
vehicle insurance claim amounts (`vehicle_claim`). It follows a modular
architecture to keep data preparation, training, and inference decoupled.

## Pipeline Architecture

1. **Data Preparation (`prepare_data.py`)**:
   - Loads the dataset from PostgreSQL via `get_insurance_dataset()`
     (`src/utils/data_loader.py`), which joins the 5 normalized tables
     (`kunden`, `policen`, `fahrzeuge`, `unfaelle`, `schaeden`) and maps
     columns back to the original English names.
   - Filters to the relevant fields (`TARGET_FIELDS` in `src/ml_config.py`).
   - Maps `YES`/`NO` fields (`property_damage`, `police_report_available`)
     to `1`/`0`.
   - Encodes missing `collision_type` values as an explicit
     `"NotApplicable"` category. The original `"?"` placeholder from the
     raw CSV is already converted to `NaN` during the CSV→DB ETL step
     (`src/db/load_data.py`, `transform()`), so `get_insurance_dataset()`
     returns `NaN` here, not the literal string `"?"`. This is not random
     noise — it occurs systematically for incident types without a
     collision (e.g. `Vehicle Theft`, `Parked Car`) and is a meaningful
     signal, not a value to discard.
   - Fills remaining missing numeric/binary values with `0`.
   - **Note:** Categorical → numeric encoding (One-Hot) is **not** done
     here. It happens inside the `ColumnTransformer` in `train_model.py`,
     as part of the saved `sklearn` pipeline. This module only cleans and
     selects columns.

2. **Model Training (`train_model.py`)**:
   - Builds an `sklearn` `Pipeline` (`ColumnTransformer` with
     `OneHotEncoder` + `RandomForestRegressor`) so that encoding and model
     are serialized together as one artifact.
   - The R² printed at the end of `train_and_save_model()` comes from a
     single train/test split and is only a quick sanity check at training
     time — see **Model Performance** below for the validated number.
   - Serializes the trained pipeline via `joblib` to `REGRESSION_MODEL_PATH`
     (`src/ml_config.py`).

3. **Inference (`predict_model.py`)**:
   - `predict_new_claim(input_data: dict)` loads the saved pipeline and
     predicts a single claim amount.
   - **Input must be raw, unencoded values** (e.g. `auto_make="Audi"` as a
     plain string). Do not call `pd.get_dummies()` or otherwise pre-encode
     the input — the pipeline's `OneHotEncoder` already does this
     internally. Passing pre-encoded input causes column mismatches and
     produces near-constant, incorrect predictions.

4. **Orchestration (`regression.py`)**:
   - Runs the full training pipeline end-to-end via
     `train_and_save_model()`.

5. **Testing (`test_model.py`)**:
   - Smoke-tests the saved model on two cases: a collision case and a
     theft/parked-car case. `vehicle_claim` is bimodal — collision cases
     average ~44,000-46,000, while theft/parked-car cases average
     ~3,800-4,000. The test warns if both predictions land suspiciously
     close together, which previously indicated a broken input pipeline.

## Model Performance

A single train/test split is sensitive to the small dataset size and is
**not** used as the reported metric. Instead, RandomForest, XGBoost, and
CatBoost were each evaluated with 5-fold cross-validation on the original
dataset (~1,000 rows):

| Model                        | R² (mean) | R² (std) | MAE (mean) |
|-------------------------------|:---:|:---:|:---:|
| RandomForest (used here)      | 0.697 | 0.023 | 7,656 |
| XGBoost (default params)      | 0.642 | 0.026 | 8,483 |
| CatBoost                      | 0.698 | 0.022 | 7,683 |
| XGBoost (tuned)               | 0.697 | 0.021 | 7,831 |

All three algorithms converge to roughly the same R² (~0.70) once properly
configured, which indicates this is close to the practical ceiling for the
current feature set — not a limitation of the chosen algorithm.
RandomForest is used in production since it reaches this ceiling without
requiring hyperparameter tuning.

**Known limitations:**
- `injury_claim`, `property_claim`, and `total_claim_amount` are
  intentionally excluded from the feature set. `total_claim_amount` is
  highly correlated with `vehicle_claim` (r ≈ 0.98) and would constitute
  data leakage. `injury_claim`/`property_claim` are excluded because, per
  business logic, they are not known at prediction time.
- MAE of ~7,700 on a median `vehicle_claim` of ~42,000 corresponds to
  roughly 18% relative error — adequate for a supporting signal, but
  should be validated against business requirements before being used as
  a sole decision input.

## Data Setup

Training data is read from **PostgreSQL**, not directly from a CSV file.
`get_insurance_dataset()` (`src/utils/data_loader.py`) joins 5 normalized
tables (`kunden`, `policen`, `fahrzeuge`, `unfaelle`, `schaeden`) and maps
the German column names back to the original English schema.

To populate the database from the raw CSV (one-time / re-import), run the
ETL script:
```bash
python src/db/load_data.py
```
This reads the CSV configured at `INSURANCE_DATASET_PATH`
(`src/ml_config.py`, currently `data/raw/dataset.csv`), cleans it
(`"?"` → `NULL`, type conversions), and loads it into the 5 normalized
tables defined in `sql/schema.sql`. `INSURANCE_DATASET_PATH` is only used
by this one-time ETL step — the training pipeline itself
(`prepare_data.py` → `train_model.py`) always reads from the database via
`get_insurance_dataset()`, regardless of whether the original CSV file is
still present on disk.

Database connection settings are configured in `src/db/connection.py`
(`get_engine()`).

## Project Structure

```text
/InterGeeks-Agiles-Programmierprojekt
├── data/
│   └── raw/
│       └── dataset.csv             # Source CSV, used only by the ETL step
├── sql/
│   └── schema.sql                   # DB schema (5 normalized tables)
├── src/
│   ├── ml_config.py                  # Paths & field configuration
│   ├── db/
│   │   ├── connection.py              # get_engine() — DB connection
│   │   └── load_data.py               # One-time ETL: CSV -> PostgreSQL
│   ├── utils/
│   │   └── data_loader.py             # get_insurance_dataset() — DB -> DataFrame
│   └── models/
│       └── regression/
│           ├── prepare_data.py        # Data cleaning & filtering
│           ├── train_model.py         # Pipeline build, training & serialization
│           ├── predict_model.py       # Single-case inference
│           ├── test_model.py          # Smoke test (collision + theft cases)
│           └── regression.py          # Main execution pipeline
└
```

## Usage

Install dependencies:
```bash
pip install pandas scikit-learn joblib sqlalchemy
```

**Prerequisite:** a running, populated PostgreSQL database (see
**Data Setup** above) with connection settings configured in
`src/db/connection.py`. Training will fail if the database is unreachable
or empty.

Train and save the model:
```bash
python regression.py
```

Run the smoke test on the saved model:
```bash
python test_model.py
```

Predict a single case from Python:
```python
from predict_model import predict_new_claim

result = predict_new_claim({
    "auto_year": 2020,
    "auto_make": "Audi",
    "incident_type": "Multi-vehicle collision",
    "incident_severity": "Major Damage",
    "collision_type": "Rear Collision",
    "number_of_vehicles_involved": 2,
    "witnesses": 1,
    "police_report_available": 1,
    "bodily_injuries": 0,
    "property_damage": 1,
})
```