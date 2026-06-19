from train_model import train_and_save_model

def run_full_pipeline():
    """
    Executes the full regression pipeline.
    """
    print("--- Starting Regression Pipeline ---")
    try:
        train_and_save_model()
        print("--- Pipeline Completed Successfully ---")
    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")

if __name__ == "__main__":
    run_full_pipeline()