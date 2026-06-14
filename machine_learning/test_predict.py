import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "./flan_ie_model"

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Тестирование на устройстве: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    def predict(text: str) -> str:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        
        # Генерируем ответ
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=128,
                num_beams=4, 
                early_stopping=True
            )
        
        return tokenizer.decode(output[0], skip_special_tokens=True)

    test_cases = [
        "Schwerer Unfall mit Audi A4 (2023) nach Frontalaufprall gegen Baum. Airbags ausgelöst.",
        "Schwerer Frontalunfall in Berlin: Ein VW Golf (2020) prallte gegen eine Wand. Airbags haben ausgelöst, Totalschaden. Keine Verletzten",
        "Hey, hab gerade beim Ausparken in München Mist gebaut. Bin mit meinem Audi A3 (2018) rückwärts gegen einen Pfosten gerollt. Ist zum Glück nur ein kleiner Kratzer an der Stoßstange."
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"--- Test #{i} ---")
        print(f"Input text: {text}")
        
        raw_output = predict(text).strip()
        print(f"Generated string: {raw_output}")
        
        clean_output = raw_output
        if not clean_output.startswith("{"):
            clean_output = "{" + clean_output
        if not clean_output.endswith("}"):
            clean_output = clean_output + "}"
            
        try:
            parsed_json = json.loads(clean_output)
            print("Valid JSON (after auto-correction):")
            print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON is still invalid after correction: {e}")
            print(f"String after attempted correction: {clean_output}")
        print("\n")

def evaluate_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Оценка модели на устройстве: {device}\n")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
        model.eval()

        def predict(text: str) -> str:
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
            return tokenizer.decode(output[0], skip_special_tokens=True).strip()

        def fix_and_parse_json(raw_str: str) -> dict:
            """Пытается исправить синтаксис и спарсить JSON."""
            if not raw_str.startswith("{"):
                raw_str = "{" + raw_str
            if not raw_str.endswith("}"):
                raw_str = raw_str + "}"
            try:
                return json.loads(raw_str)
            except json.JSONDecodeError:
                return {}

        evaluation_cases = [
            {
                "text": "Schwerer Unfall mit Audi A4 (2023) nach Frontalaufprall gegen Baum. Airbags ausgelöst.",
                "expected": {
                    "incident_type": "Single Vehicle Collision",
                    "incident_severity": "Major Damage",
                    "auto_make": "Audi",
                    "auto_model": "A4",
                    "auto_year": 2023
                }
            },
            {
                "text": "Schwerer Frontalunfall in Berlin: Ein VW Golf (2020) prallte gegen eine Wand. Airbags haben ausgelöst, Totalschaden. Keine Verletzten",
                "expected": {
                    "incident_type": "Single Vehicle Collision",
                    "incident_severity": "Major Damage",
                    "auto_make": "VW",
                    "auto_model": "Golf",
                    "auto_year": 2020
                }
            },
            {
                "text": "Hey, hab gerade beim Ausparken in München Mist gebaut. Bin mit meinem Audi A3 (2018) rückwärts gegen einen Pfosten gerollt. Ist zum Glück nur ein kleiner Kratzer an der Stoßstange.",
                "expected": {
                    "incident_type": "Single Vehicle Collision",
                    "incident_severity": "Minor Damage",
                    "auto_make": "Audi",
                    "auto_model": "A3",
                    "auto_year": 2018
                }
            }
        ]

        keys_to_eval = ["incident_type", "incident_severity", "auto_make", "auto_model", "auto_year"]
        correct_counts = {key: 0 for key in keys_to_eval}
        total_cases = len(evaluation_cases)
        valid_json_count = 0

        print("--- СБОР ТЕСТОВЫХ ПРЕДСКАЗАНИЙ ---")
        for i, case in enumerate(evaluation_cases, 1):
            raw_output = predict(case["text"])
            predicted_json = fix_and_parse_json(raw_output)
            expected_json = case["expected"]

            if predicted_json:
                valid_json_count += 1

            print(f"Тест №{i}: Готово")

            for key in keys_to_eval:
                pred_val = str(predicted_json.get(key, "")).strip().lower()
                exp_val = str(expected_json.get(key, "")).strip().lower()
                
                if pred_val == exp_val:
                    correct_counts[key] += 1

        print("\n================ МЕТРИКИ ОЦЕНКИ ================")
        print(f"Всего тестовых примеров: {total_cases}")
        print(f"Успешно спарсено JSON: {valid_json_count} из {total_cases} ({valid_json_count / total_cases * 100:.1f}%)")
        print("------------------------------------------------")
        print("Точность (Accuracy) по каждому полю:")
        
        for key in keys_to_eval:
            accuracy = (correct_counts[key] / total_cases) * 100
            print(f"  - {key}: {accuracy:.1f}% ({correct_counts[key]}/{total_cases})")
        
        total_slots = total_cases * len(keys_to_eval)
        total_correct_slots = sum(correct_counts.values())
        total_accuracy = (total_correct_slots / total_slots) * 100
        print("------------------------------------------------")
        print(f"ОБЩАЯ ТОЧНОСТЬ СЛОТОВ (Slot Accuracy): {total_accuracy:.1f}%")
        print("================================================")

if __name__ == "__main__":
    run_test()