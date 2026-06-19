import joblib
<<<<<<< HEAD
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.ml_config import REGRESSION_MODEL_PATH, PREDICTION_FIELD
from src.models.regression.prepare_data import get_processed_data, REGRESSION_CATEGORICAL_COLS

def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), REGRESSION_CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])

def train_and_save_model():
    df = get_processed_data()
    X = df.drop(columns=[PREDICTION_FIELD])
    y = df[PREDICTION_FIELD]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print(f"📈 Test R2: {r2_score(y_test, pipeline.predict(X_test)):.4f}")

    REGRESSION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, REGRESSION_MODEL_PATH)
    print(f"✅ Model saved at: {REGRESSION_MODEL_PATH}")
=======
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from prepare_data import get_processed_data

def train_and_save_model():
    """
    Orchestrates the training process using Random Forest and saves the model.
    """
    print("Preparing data...")
    df = get_processed_data()
    
    # Split into features (X) and the target variable (y)
    X = df.drop(columns=["vehicle_claim"])
    y = df["vehicle_claim"]
    
    print("Training model (Random Forest Regressor)...")
    # Initialize the model with 100 trees
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Evaluate performance using R2 score
    predictions = model.predict(X)
    accuracy = r2_score(y, predictions)
    print(f"📈 Model Training Complete. R2 Accuracy Score: {accuracy:.4f}")
    
    # Save the trained model artifact
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Success! Model saved at: {model_path}")
>>>>>>> 3021e37f5e8ba506c913b8dd6540222ba7b85146

if __name__ == "__main__":
    train_and_save_model()