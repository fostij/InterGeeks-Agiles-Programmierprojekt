# Report: Vehicle Claim Prediction

## 1. Problem Statement
The goal is to predict insurance payout (vehicle_claim) using:
- text descriptions of incidents
- structured tabular features

This is a regression task.

---

## 2. NLP Models (Text-based)

### TF-IDF + Ridge Regression
- R2: -0.0307  
- MAE: 10244.99  

Conclusion: model performs worse than baseline.

### Dummy Regressor (Mean baseline)
- R2: ~0.0  
- MAE: 10431.48  

Conclusion: no predictive signal captured.

### Sentence Embeddings (MiniLM + Ridge)
- R2: -0.0724  
- MAE: 10293.08  

Conclusion: semantic embeddings do not improve performance.

---

## 3. Data Preprocessing

Steps:
- replacement of missing markers (?, NA, null, etc.)
- numeric imputation with median
- categorical imputation with mode
- date feature extraction (year/month/day)
- removal of leakage features:
  - injury_claim
  - property_claim
  - total_claim_amount

---

## 4. Models on Tabular Data

Models used:
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor

Note: final metrics not included in this report output.

---

## 5. Findings

- Text features are not predictive for target variable
- Tabular features contain the main signal
- Leakage features must be removed to avoid overfitting
- Ensemble models are expected to outperform linear baselines

---

## 6. Conclusion

Best performance is expected from gradient boosting models on structured data.
NLP approaches are not suitable for this dataset.

6. Выбор подхода к моделированию

В рамках задачи рассматривались два направления:

извлечение признаков из текста (NLP-подход)
моделирование на структурированных данных

Изначальная гипотеза заключалась в том, что текстовое описание инцидента может содержать информацию, достаточную для предсказания vehicle_claim.

Результаты проверки гипотезы (NLP)

Все протестированные NLP-подходы показали неудовлетворительное качество:

TF-IDF + Ridge: R² < 0
Sentence Embeddings + Ridge: R² < 0
Dummy baseline показывает сопоставимое качество
Вывод по гипотезе

Текстовые данные:

не содержат устойчивого предиктивного сигнала
не улучшают качество относительно базовой модели
Принятое решение

В связи с отсутствием улучшения качества на NLP-моделях, основным направлением был выбран:

моделинг на основе табличных данных с инженерией признаков

Причины выбора:

наличие структурированных и числовых признаков с высокой информативностью
существенный риск data leakage при неправильной обработке подтверждён и контролирован
ансамблевые модели (Random Forest, XGBoost, CatBoost) лучше подходят для табличных данных