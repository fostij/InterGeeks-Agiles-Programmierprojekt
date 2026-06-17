from src.models.multi_head_insurance.train import TrainConfig
from src.ml_config import BEST_CHECKPOINT_PATH, INSURANCE_DATASET_PATH, MULTI_HEAD_DATASET_PATH, NETWORK_CONFIG_PATH, OUTPUT_CHECKPOINT_PATH
from src.models.multi_head_insurance.interface import train, predict
def main() -> None:
    
    cfg = TrainConfig(
        dataset_path=MULTI_HEAD_DATASET_PATH,
        network_config_path=NETWORK_CONFIG_PATH,
        best_checkpoint=BEST_CHECKPOINT_PATH,
        output_checkpoint=OUTPUT_CHECKPOINT_PATH,
        input_file=INSURANCE_DATASET_PATH,
        epochs=1,
        dataset_count=100,
        lr=2e-5,
        max_grad_norm=1.0,
    )

    train(cfg)

    predict("Ich war gestern Abend mit meinem Ford Escape aus dem Jahr 2006 unterwegs. Es hat geschneit und die Straßen waren ziemlich rutschig. An einer Kreuzung musste ich an einer roten Ampel anhalten. Während ich stand, ist ein anderes Fahrzeug von hinten auf mein Auto aufgefahren. Durch den Aufprall wurde die linke Seite beschädigt. Der linke Kotflügel ist eingedellt, der Scheinwerfer auf der Fahrerseite ist kaputt und die Türen lassen sich nur schwer öffnen. Außerdem wurde ein Teil der A-Säule beschädigt. Die Airbags haben ausgelöst und das Fahrzeug war danach nicht mehr fahrbereit. Die Polizei kam zum Unfallort und hat alles aufgenommen. Anschließend wurde das Auto abgeschleppt. Es gab zwei Augenzeugen. Verletzt wurde zum Glück niemand.")
    return

if __name__ == "__main__":
    main()