import joblib
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

if __name__ == "__main__":
    train_and_save_model()