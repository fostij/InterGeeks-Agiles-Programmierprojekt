import json

from utils.orchester_data import get_description_and_labels
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq


GERMAN_MODEL_NAME = "bert-base-german-cased"

MODEL_NAME = "google/flan-t5-base"

def make_prediction() -> None:
    dataset, label2id, id2label = build_incident_type_dataset(20000)
    dataset = dataset.train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained(GERMAN_MODEL_NAME)
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
    
    dataset = dataset.map(tokenize, batched=True)

    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")


    model = AutoModelForSequenceClassification.from_pretrained(
        GERMAN_MODEL_NAME,
        num_labels=len(set(dataset["train"]["label"])),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="./incident_type_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    print(
        clf("Der Fahrer verlor die Kontrolle und prallte frontal gegen einen Baum.")
    )

    print(
        clf("Schwerer Frontalunfall in Berlin: Ein VW Golf (2020) prallte gegen eine Wand. Airbags haben ausgelöst, Totalschaden. Keine Verletzten")
    )

    print(
        clf("Hey, hab gerade beim Ausparken in München Mist gebaut. Bin mit meinem Audi A3 (2018) rückwärts gegen einen Pfosten gerollt. Ist zum Glück nur ein kleiner Kratzer an der Stoßstange.")
    )


def build_incident_type_dataset(size: int = 20000) -> tuple[Dataset, dict[str, int], dict[int, str]]:
    samples = get_description_and_labels(size)

    texts = []
    labels = []

    for sample in samples:
        texts.append(sample["text"])
        labels.append(sample["labels"]["incident_type"])

    unique_labels = sorted(set(labels))

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    label_ids = [label2id[label] for label in labels]

    dataset = Dataset.from_dict({
        "text": texts,
        "label": label_ids,
    })

    return dataset, label2id, id2label




def make_prediction_seq2seq() -> None:

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=256,
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            truncation=True,
            max_length=128,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    

    dataset = build_ie_dataset(20000)
    dataset = dataset.train_test_split(test_size=0.2)

    dataset = dataset.map(preprocess, batched=True)

    dataset = dataset.remove_columns(["input_text", "target_text"])

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir="./flan_ie_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=50,
        fp16=False,
        report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator
    )
    print(dataset["train"][0]["labels"])
    trainer.train()

    trainer.save_model("./flan_ie_model")
    tokenizer.save_pretrained("./flan_ie_model")
    
    model.eval()

    def predict(text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        output = model.generate(
            **inputs,
            max_length=128
        )

        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(
        predict("Schwerer Unfall mit Audi A4 (2023) nach Frontalaufprall gegen Baum. Airbags ausgelöst.")
    )
    print(
        predict("Schwerer Frontalunfall in Berlin: Ein VW Golf (2020) prallte gegen eine Wand. Airbags haben ausgelöst, Totalschaden. Keine Verletzten")
    )
    print(
        predict("Hey, hab gerade beim Ausparken in München Mist gebaut. Bin mit meinem Audi A3 (2018) rückwärts gegen einen Pfosten gerollt. Ist zum Glück nur ein kleiner Kratzer an der Stoßstange.")
    )



def build_ie_dataset(size: int = 20000) -> Dataset:
    samples = get_description_and_labels(size)

    inputs = []
    targets = []

    for s in samples:
        text = s["text"]
        labels = s["labels"]

        target = json.dumps({
            "incident_type": labels["incident_type"],
            "incident_severity": labels["incident_severity"],
            "auto_make": labels["auto_make"],
            "auto_model": labels["auto_model"],
            "auto_year": labels["auto_year"]
        }, ensure_ascii=False)

        inputs.append(text)
        targets.append(target)

    return Dataset.from_dict({
        "input_text": inputs,
        "target_text": targets
    })



