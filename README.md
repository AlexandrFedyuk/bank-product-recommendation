# Рекомендации банковских продуктов

Проект решает задачу персонализированных рекомендаций банковских продуктов: на основе истории клиента за месяц `t` нужно предсказать, какие **новые** продукты клиент может приобрести в месяце `t+1`.

Проект подготовлен под локальную работу в **JupyterLab** на MacBook и дальнейшую загрузку на GitHub.

## Цель проекта

Бизнес-цель — повысить вероятность покупки дополнительных банковских продуктов за счёт персонализированных предложений.

ML-постановка — задача ранжирования: для каждого клиента формируется список продуктов-кандидатов, которые у него ещё не подключены, затем модель оценивает вероятность покупки каждого продукта и возвращает `top-7` рекомендаций.

Главная метрика качества: **MAP@7**. Дополнительно считаются `Precision@7`, `Recall@7`, `HitRate@7` и `Coverage@7`.

## Использованные технологии

- Python 3.11
- JupyterLab
- pandas, numpy
- scikit-learn
- MLflow
- FastAPI
- Docker / Docker Compose
- prometheus-client
- pytest

## Структура проекта

```text
bank-product-recommendation/
├── app/                         # FastAPI-сервис
│   ├── main.py                  # API endpoints: /health, /recommend, /metrics
│   ├── schemas.py               # Pydantic-схемы запросов и ответов
│   └── service.py               # Загрузка модели и бизнес-логика API
├── data/
│   ├── raw/                     # Сюда положить исходный train_ver2.csv
│   └── processed/               # Промежуточные данные, не хранятся в Git
├── docs/
│   ├── api.md                   # Описание API
│   └── monitoring.md            # Описание мониторинга
├── models/                      # Сюда сохраняется model.joblib после обучения
├── notebooks/
│   ├── 01_eda.ipynb             # Первичный анализ данных
│   └── 02_modeling_experiments.ipynb # Обучение, MLflow, метрики
├── reports/                     # Метрики и отчёты после обучения
├── scripts/
│   ├── make_sample_data.py      # Генерация маленького demo-датасета
│   ├── run_api.sh               # Локальный запуск API
│   ├── run_mlflow.sh            # Локальный запуск MLflow
│   ├── train.py                 # Обучение модели
│   └── evaluate.py              # Оценка сохранённой модели
├── src/bank_recs/               # Основной Python-пакет проекта
│   ├── config.py                # Константы и списки колонок
│   ├── data.py                  # Загрузка и очистка данных
│   ├── features.py              # Генерация признаков и train pairs
│   ├── inference.py             # Подготовка данных для инференса
│   ├── metrics.py               # MAP@7, Recall@7 и другие метрики
│   ├── modeling.py              # Модель и ранжирование
│   └── monitoring.py            # Prometheus-метрики
├── tests/                       # Минимальные тесты метрик
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Подготовка локальной среды на MacBook

Открой Terminal в папке проекта и выполни:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name bank-products --display-name "Python (bank-products)"
```

После этого можно открыть JupyterLab:

```bash
jupyter lab
```

В ноутбуках выбери kernel:

```text
Python (bank-products)
```

## Данные

Исходный датасет не хранится в репозитории. Скачай `train_ver2.csv` и положи его в:

```text
data/raw/train_ver2.csv
```

CSV-файлы добавлены в `.gitignore`, чтобы не загружать большой датасет на GitHub.

Для проверки, что проект запускается без реального датасета, можно создать маленький синтетический файл:

```bash
python scripts/make_sample_data.py
```

После этого появится:

```text
data/raw/train_ver2_sample.csv
```

## Как работать в JupyterLab

1. Открой `notebooks/01_eda.ipynb`.
2. Запусти ячейки с анализом данных.
3. Открой `notebooks/02_modeling_experiments.ipynb`.
4. Запусти baseline, обучение модели и расчёт метрик.

Ноутбуки используют код из папки `src/`, поэтому логика проекта не дублируется только в `.ipynb`.

## Трансляция бизнес-задачи в ML-задачу

Банк хочет предложить клиенту продукт, который он с высокой вероятностью купит. Значит, модель должна не просто классифицировать клиента, а сформировать ранжированный список предложений.

Таргет строится так:

```text
new_product = product_state(t+1) - product_state(t)
```

Если продукт уже был у клиента в месяце `t`, его нельзя считать новой покупкой и не нужно рекомендовать повторно.

Для каждого клиента создаются пары:

```text
клиент — продукт — признаки клиента и продукта — target
```

где `target = 1`, если клиент приобрёл этот продукт в следующем месяце.

## Метрики

Основная метрика:

- `MAP@7` — учитывает не только попадание релевантного продукта в рекомендации, но и позицию продукта в списке.

Дополнительные метрики:

- `Precision@7` — точность top-7 рекомендаций;
- `Recall@7` — доля найденных новых покупок;
- `HitRate@7` — доля клиентов, для которых угадан хотя бы один продукт;
- `Coverage@7` — разнообразие рекомендаций по продуктам.

## Запуск MLflow

В отдельном терминале:

```bash
source .venv/bin/activate
./scripts/run_mlflow.sh
```

Интерфейс откроется здесь:

```text
http://127.0.0.1:5000
```

## Обучение модели

На реальном датасете:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/train.py --data-path data/raw/train_ver2.csv
```

На синтетическом demo-датасете:

```bash
python scripts/make_sample_data.py
PYTHONPATH=src python scripts/train.py --data-path data/raw/train_ver2_sample.csv --max-users 0
```

После обучения появятся:

```text
models/model.joblib
models/model.metadata.json
reports/metrics.json
```

В MLflow логируются:

- параметры эксперимента;
- ranking-метрики;
- модель;
- JSON с метриками;
- артефакты обучения.

## Запуск API локально

После обучения модели:

```bash
source .venv/bin/activate
./scripts/run_api.sh
```

Swagger UI будет доступен здесь:

```text
http://127.0.0.1:8000/docs
```

Проверка здоровья сервиса:

```bash
curl http://127.0.0.1:8000/health
```

Пример запроса рекомендаций:

```bash
curl -X POST "http://127.0.0.1:8000/recommend"   -H "Content-Type: application/json"   -d '{
    "ncodpers": 123456,
    "age": 45,
    "sexo": "H",
    "segmento": "02 - PARTICULARES",
    "renta": 120000,
    "ind_actividad_cliente": 1,
    "products": {
      "ind_cco_fin_ult1": 1,
      "ind_recibo_ult1": 1,
      "ind_tjcr_fin_ult1": 0
    }
  }'
```

## Запуск через Docker

Сначала обучи модель локально, чтобы появился файл:

```text
models/model.joblib
```

Затем:

```bash
docker compose up --build api
```

API будет доступно на порту `8000`.

Для запуска API и MLflow вместе:

```bash
docker compose up --build
```

## Мониторинг

Сервис отдаёт Prometheus-метрики по адресу:

```text
http://127.0.0.1:8000/metrics
```

В коде отправляются метрики:

- `recommendation_requests_total`
- `recommendation_errors_total`
- `recommendation_latency_seconds`
- `recommended_product_total{product="..."}`

Подробное описание мониторинга находится в `docs/monitoring.md`.

## Воспроизводимость

В проекте зафиксированы:

- зависимости в `requirements.txt`;
- `random_state = 42`;
- структура данных;
- команды запуска;
- логирование экспериментов в MLflow;
- Dockerfile для разворачивания сервиса.

## Как отправить на GitHub

В репозиторий нужно добавить все файлы проекта, кроме данных, моделей и временных артефактов.

Проверить, что попадёт в Git:

```bash
git status
```

В GitHub не должны попасть:

```text
data/raw/train_ver2.csv
.venv/
mlruns/
models/model.joblib
reports/metrics.json
```

Они игнорируются через `.gitignore`.

## Что можно улучшить

Текущая модель — воспроизводимый production-ready baseline на `scikit-learn`. Для улучшения качества можно:

- заменить модель на CatBoost или LightGBM;
- добавить больше временных лагов продуктов;
- добавить агрегации по сегменту, возрасту, региону и активности;
- обучать модель на нескольких последовательных парах месяцев;
- добавить batch-инференс;
- подключить полноценный Prometheus + Grafana.
