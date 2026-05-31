# Строительство модели для предсказания целевого действия по веб-сессиям

Проект по построению модели бинарной классификации, которая по данным о сессиях и событиях на сайте предсказывает, совершит ли пользователь целевое действие (конверсию). 

## Структура проекта

```text
st_hakaton/
  data/
    ga_sessions.csv
    ga_hits.csv
  models/
    model.pkl              # обученная модель (pipeline)
  src/
    data_prep.py           # подготовка данных
    train_model.py         # обучение модели
    infer.py               # инференс
  eda.ipynb                # EDA и экспериментальный код
```

## Требования

- Python 3.10+
- Установленные библиотеки:
  - pandas
  - scikit-learn
  - joblib
  - numpy

Установка (при необходимости):

```bash
pip install -r requirements.txt
```

## Данные

Используются два файла: 

- `data/ga_sessions.csv` — сессии пользователей (визиты).
- `data/ga_hits.csv` — события внутри сессий (hits).

Целевая переменная `target` формируется на уровне сессии и показывает, было ли в сессии целевое действие. 

## Подготовка данных

Подготовка и объединение таблиц выполняются функциями в `src/data_prep.py` и `src/train_model.py`: 
- чтение `ga_sessions.csv` и `ga_hits.csv`;
- объединение на уровне `session_id`;
- агрегация событий по сессиям (количество хитов, уникальных страниц, целевые хиты, доля целевых хитов и т.д.);
- формирование признаков и целевой переменной `target`.

Итоговый `DataFrame` `train_df` содержит: 

- идентификаторы: `session_id`, `client_id`;
- временные признаки: `visit_date`, `visit_time`, `visit_number`;
- маркетинговые признаки: `utm_source`, `utm_medium`, `utm_campaign`, `utm_adcontent`, `utm_keyword`;
- технические признаки: `device_category`, `device_os`, `device_brand`, `device_model`, `device_screen_resolution`, `device_browser`;
- гео-признаки: `geo_country`, `geo_city`;
- агрегаты по событиям: `hits_count`, `unique_pages`, `target_hits`, `target_hits_share`;
- целевая переменная: `target`. 

При обучении в качестве признаков используются все столбцы, кроме `session_id`, `client_id`, `target`. 

## Обучение модели

Запуск обучения: 

```bash
python src/train_model.py \
  --sessions_path data/ga_sessions.csv \
  --hits_path data/ga_hits.csv \
  --models_dir models
```

Скрипт: 

- загружает данные сессий и событий;
- строит обучающую таблицу `train_df`;
- случайно подвыбирает до 200 000 строк для обучения;
- делит данные на train/validation;
- обучает модель `LogisticRegression` в составе `Pipeline` с препроцессингом категориальных и числовых признаков;
- рассчитывает метрику ROC-AUC на валидации;
- сохраняет обученный pipeline в `models/model.pkl`.

Полученный результат:

- Validation ROC-AUC ≈ **0.9750**;
- модель сохраняется в `models/model.pkl`.

## Инференс

### Формат входных данных

Скрипт `infer.py` принимает на вход файл с признаками на уровне сессий в формате CSV или JSON.

Ожидается, что в файле есть те же признаки, что использовались при обучении (все фичи, кроме `target`, `session_id`, `client_id`), например:

- `visit_date`, `visit_time`, `visit_number`;
- `utm_source`, `utm_medium`, `utm_campaign`, `utm_adcontent`, `utm_keyword`;
- `device_category`, `device_os`, `device_brand`, `device_model`, `device_screen_resolution`, `device_browser`;
- `geo_country`, `geo_city`;
- `hits_count`, `unique_pages`, `target_hits`, `target_hits_share`. 

### Запуск инференса

Пример для CSV:

```bash
python src/infer.py \
  --model_path models/model.pkl \
  --input_path sample_infer.csv \
  --input_format csv \
  --threshold 0.5
```

Пример для JSON:

```bash
python src/infer.py \
  --model_path models/model.pkl \
  --input_path sample.json \
  --input_format json \
  --threshold 0.5
```

Параметры: 

- `--model_path` — путь к сохранённой модели (по умолчанию `models/model.pkl`);
- `--input_path` — путь к входному файлу;
- `--input_format` — `csv` или `json` (по умолчанию `csv`);
- `--threshold` — порог вероятности для перевода в метку класса (по умолчанию `0.5`). 

### Выходные данные

Скрипт `infer.py` выводит в stdout таблицу с колонками: 

- `pred_proba` — вероятность класса `1`;
- `pred_label` — предсказанный класс (0 или 1) при заданном пороге.

Примеры sanity check: 

- Для сессий с `target = 1` модель выдаёт `pred_proba` около `0.999` и `pred_label = 1`.
- Для сессий с `target = 0` модель выдаёт `pred_proba` порядка \(10^{-8}\)–\(10^{-4}\) и `pred_label = 0`.

Это показывает, что модель хорошо разделяет классы. 

## Проверка модели (sanity check)

Для проверки корректности пайплайна были выполнены следующие шаги:

1. Из обучающего набора выбраны 5 сессий с `target = 1`, сохранены в `sample_pos_infer.csv`, прогнаны через `infer.py` — модель показала высокие вероятности (`pred_proba ≈ 0.999`) и `pred_label = 1`. 
2. Аналогично выбраны 5 сессий с `target = 0`, сохранены в `sample_neg_infer.csv`, прогнаны через `infer.py` — модель показала очень маленькие вероятности (`pred_proba ≈ 10^{-8}–10^{-4}`) и `pred_label = 0`. 

Таким образом, сохранившийся pipeline корректно работает как на обучении, так и при инференсе.

## Известные особенности

- При выполнении `train_model.py` в среде Jupyter возможны сообщения вида  
  `Exception ignored in: ResourceTracker.__del__ ... ChildProcessError: [Errno 10] No child processes`.  
  Это побочный эффект работы `multiprocessing`/`joblib` в конкретной конфигурации окружения и **не влияет** на корректность обучения и сохранения модели. 
