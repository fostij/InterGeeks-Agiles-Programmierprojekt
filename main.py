from src.models.multi_head_insurance.train import TrainConfig
from src.ml_config import BEST_CHECKPOINT_PATH, INSURANCE_DATASET_PATH, MULTI_HEAD_DATASET_PATH, NETWORK_CONFIG_PATH, OUTPUT_CHECKPOINT_PATH
from src.services.pipeline import PredictionResult, Pipeline
def main() -> None:
    pipeline = Pipeline()

    result = pipeline.run("Der Audi A3 (1999) hatte einen Crash. Am abend bei klarem Wetter kam es dazu: ein anderes Fahrzeug traf die Seitenpartie bei dichtem Verkehr. Ergebnis: der Kühler ist beschädigt, das Fahrwerk ist beschädigt, Kratzer an der Fahrertür, die Karosseriegeometrie ist verzogen, linker Kotflügel eingedellt. Die Frontairbags wurden aktiviert. Der Vorfall ist bei der Verkehrspolizei registriert. Ein Abschleppdienst war erforderlich.")
    print(result)
    return

if __name__ == "__main__":
    main()