import joblib
import pandas as pd
from src.ml_config import REGRESSION_MODEL_PATH

def test_model():
    model_path = REGRESSION_MODEL_PATH
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}.")
        return

    model = joblib.load(model_path)
    
    collision_case = pd.DataFrame([{
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

    theft_case = pd.DataFrame([{
        'auto_year': 2018,
        'auto_make': 'Honda',
        'incident_type': 'Vehicle Theft',
        'incident_severity': 'Trivial Damage',
        'collision_type': 'NotApplicable',
        'number_of_vehicles_involved': 1,
        'witnesses': 0,
        'police_report_available': 1,
        'bodily_injuries': 0,
        'property_damage': 0
    }])


    collision_pred = model.predict(collision_case)[0]
    theft_pred = model.predict(theft_case)[0]

    print(f"✅ Kollisionsfall — vorhergesagter Wert: {collision_pred:,.2f}")
    print(f"✅ Diebstahl/Parked-Car-Fall — vorhergesagter Wert: {theft_pred:,.2f}")

    if abs(collision_pred - theft_pred) < 1000:
        print("⚠️ Warnung: Die Vorhersagen für beide Fälle liegen sehr nah beieinander.")


if __name__ == "__main__":
    test_model()