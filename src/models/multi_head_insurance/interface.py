import torch
from src.ml_config import BEST_CHECKPOINT_PATH
from src.models.multi_head_insurance.inference import load_model, predict as predict_fn
from src.models.multi_head_insurance.train import TrainConfig, run_training

def train(config: TrainConfig):
    run_training(config)

def predict(*text: str):
    device = torch.device("cpu")

    model, network_config, label_encoders = load_model(BEST_CHECKPOINT_PATH, device)

    result_df = predict_fn(
        texts=list(text),
        model=model,
        network_config=network_config,
        label_encoders=label_encoders,
        device=device,
    )

    result_df.to_csv("reconstructed_table.csv", index=False)
    print(result_df)
    return result_df