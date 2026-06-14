from utils.labels_encoder import prepare_pipeline_and_save_jsonl

def main() -> None:
    prepare_pipeline_and_save_jsonl(1000, "data/output/output.jsonl")
    return

if __name__ == "__main__":
    main()