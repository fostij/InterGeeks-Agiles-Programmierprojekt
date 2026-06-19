import joblib
import os
import pandas as pd
from prepare_data import get_processed_data

def test_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}.")
        return

    model = joblib.load(model_path)
    
    # Koresha structure y'inkingi zose (Features)
    train_df = get_processed_data()
    X_train = train_df.drop(columns=["vehicle_claim"])
    
    # Sample input ikeneye inkingi zose 
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
    
    # Process inputs (One-Hot Encoding + Reindexing)
    df_input = pd.get_dummies(sample_data)
    df_input = df_input.reindex(columns=X_train.columns, fill_value=0)
    
    # Run prediction
    prediction = model.predict(df_input)
    print(f"✅ Test prediction successful. Predicted value: {prediction[0]:,.2f} EUR")

if __name__ == "__main__":
    test_model()