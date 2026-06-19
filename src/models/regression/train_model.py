import joblib
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

if __name__ == "__main__":
    train_and_save_model()