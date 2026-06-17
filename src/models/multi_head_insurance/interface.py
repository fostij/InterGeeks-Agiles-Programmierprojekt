import torch
from src.ml_config import BEST_CHECKPOINT
from src.models.multi_head_insurance.inference import (
    load_model,
    predict as predict_fn,
    save_predictions_to_csv,
)
from src.models.multi_head_insurance.train import TrainConfig, run_training


def train(config: TrainConfig):
    run_training(config)


def predict(*text: str):
    device = torch.device("cpu")

    model, network_config, numeric_stats, label_encoders = load_model(
        BEST_CHECKPOINT, device
    )

    result_rows = predict_fn(
        texts=list(text),
        model=model,
        network_config=network_config,
        numeric_stats=numeric_stats,
        label_encoders=label_encoders,
        device=device,
    )

    save_predictions_to_csv(result_rows, "reconstructed_table.csv")

    # Print results in a readable format
    if result_rows:
        print("\nPrediction Results:")
        print("-" * 80)
        for i, row in enumerate(result_rows, 1):
            print(f"\nSample {i}:")
            for field, value in row.items():
                print(f"  {field}: {value}")
