from machine_learning.multi_head import TrainConfig, run_training
from ml_config import DATASET_PATH, NETWORK_CONFIG_PATH, INPUT_FILE, OUTPUT_CHECKPOINT
def main() -> None:
    cfg = TrainConfig(
        dataset_path=DATASET_PATH,
        input_file=INPUT_FILE,
        network_config_path=NETWORK_CONFIG_PATH,
        output_checkpoint=OUTPUT_CHECKPOINT,
        epochs=1,
        dataset_count=100,
    )
    run_training(cfg)
    return

if __name__ == "__main__":
    main()