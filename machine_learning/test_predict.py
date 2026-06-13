import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Путь к сохраненной модели
MODEL_PATH = "./flan_ie_model"

def run_test():
    # 1. Автоматическое определение устройства (GPU если доступен, иначе CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Тестирование на устройстве: {device}\n")

    # 2. Загрузка токенизатора и обученной модели из папки
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # 3. Функция инференса
    def predict(text: str) -> str:
        # Токенизируем входной текст и отправляем на GPU/CPU
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        
        # Генерируем ответ
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=128,
                num_beams=4, # Использование Beam Search улучшает качество генерации JSON
                early_stopping=True
            )
        
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # 4. Тестовые примеры
    test_cases = [
        "Schwerer Unfall mit Audi A4 (2023) nach Frontalaufprall gegen Baum. Airbags ausgelöst.",
        "Schwerer Frontalunfall in Berlin: Ein VW Golf (2020) prallte gegen eine Wand. Airbags haben ausgelöst, Totalschaden. Keine Verletzten",
        "Hey, hab gerade beim Ausparken in München Mist gebaut. Bin mit meinem Audi A3 (2018) rückwärts gegen einen Pfosten gerollt. Ist zum Glück nur ein kleiner Kratzer an der Stoßstange."
    ]

    # 5. Вывод результатов
    for i, text in enumerate(test_cases, 1):
        print(f"--- Test #{i} ---")
        print(f"Input text: {text}")
        
        raw_output = predict(text).strip()
        print(f"Generated string: {raw_output}")
        
        # Correction: remove possible extra braces at the edges and forcibly wrap in {}
        clean_output = raw_output
        if not clean_output.startswith("{"):
            clean_output = "{" + clean_output
        if not clean_output.endswith("}"):
            clean_output = clean_output + "}"
            
        # Проверяем скорректированный JSON
        try:
            parsed_json = json.loads(clean_output)
            print("Valid JSON (after auto-correction):")
            print(json.dumps(parsed_json, indent=4, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON is still invalid after correction: {e}")
            print(f"String after attempted correction: {clean_output}")
        print("\n")

if __name__ == "__main__":
    run_test()