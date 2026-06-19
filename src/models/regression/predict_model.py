import joblib
<<<<<<< HEAD
import pandas as pd
from src.ml_config import REGRESSION_MODEL_PATH


def predict_new_claim(input_data: dict) -> float:
    pipeline = joblib.load(REGRESSION_MODEL_PATH)
    df_input = pd.DataFrame([input_data])
    prediction = pipeline.predict(df_input)
    return float(prediction[0])
=======
import os
import pandas as pd
from prepare_data import get_processed_data

def predict_new_claim(input_data):
    # 1. Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = joblib.load(model_path)
    
    # 2. Get the structure of ALL columns from the training process
    # Ibi ni byo bizakemura kariya kosa kuko bizatuma duhuza inkingi (columns)
    train_df = get_processed_data()
    X_train = train_df.drop(columns=["vehicle_claim"])
    
    # 3. Create DataFrame from input and process it to match training structure
    df_input = pd.DataFrame([input_data])
    
    # Kora dummies nk'uko twabikoze muri prepare_data
    categorical_cols = ["auto_make", "incident_type", "incident_severity", "collision_type"]
    df_input = pd.get_dummies(df_input, columns=categorical_cols)
    
    # 4. Huza inkingi (Reindex)
    # Ibi byongeramo inkingi zose zabuze (zizajyaho 0) kandi bigakuraho izitakenewe
    df_input = df_input.reindex(columns=X_train.columns, fill_value=0)
    
    # 5. Predict
    prediction = model.predict(df_input)
    return prediction[0]

if __name__ == "__main__":
    # Test data - Hano ushobora gushyiramo data y'ukuri
    sample_data = {
        "auto_year": 2020,
        "auto_make": "Audi",
        "incident_type": "Multi-vehicle collision",
        "incident_severity": "Major Damage",
        "collision_type": "Rear Collision",
        "number_of_vehicles_involved": 2,
        "witnesses": 1,
        "police_report_available": 1,
        "bodily_injuries": 0,
        "property_damage": 1
    }
    
    try:
        result = predict_new_claim(sample_data)
        print(f"💰 Predicted Vehicle Claim Amount: {result:.2f} EUR")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
>>>>>>> 3021e37f5e8ba506c913b8dd6540222ba7b85146
