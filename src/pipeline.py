import torch
from src.ml_config import BEST_CHECKPOINT_PATH
from src.models.multi_head_insurance.inference import load_model, predict as predict_fn
from src.models.multi_head_insurance.train import TrainConfig, run_training

def train_multi_head(config: TrainConfig):
    run_training(config)

def predict_multi_head(*text: str):
    device = torch.device("cpu")

    model, network_config, label_encoders = load_model(BEST_CHECKPOINT_PATH, device)

    result_df = predict_fn(
        texts=list(text),
        model=model,
        label_encoders=label_encoders,
        device=device,
    )

    result_df.to_csv("reconstructed_table.csv", index=False)
    print(result_df)
    return result_df

def train_cnn():
    pass

def predict_cnn():
    pass

def train_regression():
    pass

def predict_regression():
    pass

def run_train_pipeline(config: TrainConfig):
    train_multi_head(config)
    train_cnn()
    train_regression()

def run_predict_pipeline(*text: str):
    predict_multi_head(*text)
    predict_cnn()
    predict_regression()