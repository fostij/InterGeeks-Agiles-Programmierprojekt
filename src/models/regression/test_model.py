import joblib
import pandas as pd
from prepare_data import get_processed_data
from src.ml_config import REGRESSION_MODEL_PATH

def test_model():
    model_path = REGRESSION_MODEL_PATH
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}.")
        return

    model = joblib.load(model_path)
    
    train_df = get_processed_data()
    X_train = train_df.drop(columns=["vehicle_claim"])
    
    sample_data = pd.DataFrame([{
        'auto_year': 2020,
        'auto_make': 'Audi',
        'incident_type': 'Multi-vehicle collision',
        'incident_severity': 'Major Damage',
        'collision_type': 'Rear Collision',
        'number_of_vehicles_involved': 2,
        'witnesses': 1,
        'police_report_available': 1,
        'bodily_injuries': 0,
        'property_damage': 1
    }])
    
    df_input = pd.get_dummies(sample_data)
    df_input = df_input.reindex(columns=X_train.columns, fill_value=0)
    
    prediction = model.predict(df_input)
    print(f"✅ Test prediction successful. Predicted value: {prediction[0]:,.2f}")

if __name__ == "__main__":
    test_model()