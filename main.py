from src.models.multi_head_insurance.train import TrainConfig
from src.ml_config import BEST_CHECKPOINT_PATH, INSURANCE_DATASET_PATH, MULTI_HEAD_DATASET_PATH, NETWORK_CONFIG_PATH, OUTPUT_CHECKPOINT_PATH
from src.services.pipeline import PredictionResult, Pipeline

def main() -> None:
    pipeline = Pipeline()
    strings = ["Ich war gestern beim Supermarkt und wollte rückwärts ausparken. Hab mit meinem Golf komplett den Pfosten übersehen und ihn gerammt. Zum Glück nichts Wildes, nur ein fetter Kratzer an der Stoßstange. Trotzdem ärgerlich.",
               "Im Berufsverkehr hat der Dreier-BMW vor mir plötzlich voll gebremst. Ich konnte mit meinem Fiesta nicht mehr rechtzeitig reagieren und bin ihm hinten draufgefahren. Ist aber nur ein Blechschaden – seine Stoßstange ist Schrott, bei mir ist kaum was zu sehen.",
               "Ich habe heute Morgen beim Abbiegen die Kurve total geschnitten und bin mit dem Tesla voll gegen den Bordstein geknallt. Jetzt ist die Felge komplett zerkratzt und der Reifen hat ’ne Beule. Muss wohl in die Werkstatt.",
               "Beim Einparken in der Innenstadt habe ich einen parkenden Benz touchiert. War super eng. Ich habe direkt gewartet, bis der Besitzer kam. Wir haben die Daten ausgetauscht. Kein Riesending, nur etwas Lack ab bei beiden.",
               "Mir hat heute ein Audi die Vorfahrt genommen – klassisch Rechts vor Links missachtet. Ich habe noch versucht auszuweichen, aber es hat kurz gekracht. Mein Kotflügel ist jetzt eingedellt. Die Polizei war auch schon da."]
   
    for text in strings:
        result = pipeline.run(text, None)
        print(result)

    return

if __name__ == "__main__":
    main()