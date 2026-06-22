from src.services.pipeline import PredictionResult, Pipeline
from src.logging_config import setup_logging
from src.db.pipeline_repository import PipelineResultRepository

def main() -> None:
    setup_logging()
    repository = PipelineResultRepository()
    p = Pipeline(repository)
    result = p.run("Mit dem Subaru Legacy (1998) lief es heute leider schief. Zur hauptverkehrszeit bei klarem Wetter kam es zum Knall: das Fahrzeug fuhr in eine Straßenbegrenzung. Danach wurden diese Schaeden festgestellt: Kratzer an der Beifahrertür, vorderer Stoßfänger gebrochen. Es wurde keine Airbag-Ausloesung festgestellt.", 
          )
    print(result)
    return

if __name__ == "__main__":
    main()