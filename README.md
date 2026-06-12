# Bank Product Recommendation

Проект посвящён задаче персональных рекомендаций банковских продуктов.  
Цель — определить, какие новые продукты банк может предложить клиенту в следующем месяце на основе его клиентских признаков и текущего продуктового портфеля.

В проекте реализован полный ML-контур: EDA, формирование таргета, обучение baseline и ML-модели, логирование экспериментов в MLflow, REST API для инференса, Docker-запуск и базовый мониторинг через Prometheus-метрики.

## Стек

- Python 3.11
- pandas, numpy
- scikit-learn
- MLflow
- FastAPI
- Uvicorn
- Prometheus Client
- Docker, Docker Compose
- JupyterLab

## Постановка задачи

Исходные данные содержат ежемесячные записи по клиентам банка за период с января 2015 года по май 2016 года. Для каждого клиента известны демографические и продуктовые признаки. Продуктовые колонки имеют формат `ind_*_ult1` и показывают, есть ли у клиента конкретный банковский продукт в данном месяце.

Задача формулируется не как предсказание всех продуктов клиента, а как рекомендация новых продуктов.  
Для клиента в месяце `t` нужно предсказать продукты, которые появятся у него в месяце `t+1`.

Таргет строится как переход продукта из состояния `0` в состояние `1` между соседними месяцами:

```text
product_t = 0
product_t+1 = 1
```

Такой подход ближе к бизнес-сценарию: банк не должен рекомендовать клиенту продукт, который у него уже есть.

## Метрики

Основная метрика проекта — `MAP@7`.

Она выбрана потому, что задача является рекомендательной: модель должна выдать ограниченный список продуктов, а качество зависит не только от попадания продукта в рекомендации, но и от его позиции в списке.

Дополнительно считаются:

- `Precision@7` — доля релевантных продуктов среди рекомендованных;
- `Recall@7` — доля найденных новых продуктов клиента;
- `HitRate@7` — доля клиентов, для которых модель угадала хотя бы один новый продукт;
- `Coverage@7` — доля продуктовой линейки, которая попадает в рекомендации.

## Структура проекта

```text
bank-product-recommendation/
├── app/                         # FastAPI-приложение
│   ├── main.py                  # API endpoints
│   ├── schemas.py               # Pydantic-схемы запросов и ответов
│   └── service.py               # Логика рекомендаций
│
├── docs/
│   ├── api.md                   # Описание API
│   └── monitoring.md            # Описание мониторинга
│
├── models/
│   └── .gitkeep                 # Модель создаётся после обучения
│
├── notebooks/
│   ├── 01_eda.ipynb             # Исследовательский анализ данных
│   └── 02_modeling_experiments.ipynb  # Моделирование и сравнение подходов
│
├── reports/
│   └── .gitkeep                 # Метрики создаются после обучения
│
├── scripts/
│   ├── make_sample_data.py      # Генерация тестового датасета
│   ├── run_api.sh               # Запуск API
│   ├── run_mlflow.sh            # Запуск MLflow
│   ├── train.py                 # Обучение модели
│   └── evaluate.py              # Оценка модели
│
├── src/bank_recs/
│   ├── config.py                # Конфигурация проекта
│   ├── data.py                  # Загрузка и подготовка данных
│   ├── features.py              # Генерация признаков
│   ├── inference.py             # Инференс и ранжирование продуктов
│   ├── metrics.py               # Ranking-метрики
│   ├── modeling.py              # ML-пайплайн
│   └── monitoring.py            # Prometheus-метрики
│
├── tests/
│   └── test_metrics.py          # Тесты для метрик
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .gitignore
└── README.md
```

## Данные

Данные не хранятся в репозитории, так как исходный CSV-файл имеет большой размер.  
После скачивания датасета его нужно положить в директорию:

```text
data/raw/train_ver2.csv
```

Файлы `.csv`, директории `data/raw/`, `data/processed/`, артефакты MLflow и обученные модели исключены из Git через `.gitignore`.

Для проверки работоспособности проекта без полного датасета можно создать небольшой синтетический пример:

```bash
python scripts/make_sample_data.py
```

## Установка окружения

Рекомендуемая версия Python — 3.11.

```bash
git clone https://github.com/AlexandrFedyuk/bank-product-recommendation.git
cd bank-product-recommendation
```

Создание окружения через conda:

```bash
conda create -n bank-products python=3.11 -y
conda activate bank-products
pip install -r requirements.txt
```

Подключение окружения к JupyterLab:

```bash
python -m ipykernel install --user --name bank-products --display-name "Python (bank-products)"
```

После этого в JupyterLab нужно выбрать kernel:

```text
Python (bank-products)
```

## EDA

Исследовательский анализ находится в ноутбуке:

```text
notebooks/01_eda.ipynb
```

В ходе EDA были проверены:

- размер данных и период наблюдений;
- количество уникальных клиентов;
- структура клиентских и продуктовых признаков;
- пропуски в данных;
- распределения возраста, дохода и стажа клиента;
- динамика по месяцам;
- популярность банковских продуктов;
- появление новых продуктов между соседними месяцами.

Основной вывод EDA: данные имеют временную структуру, поэтому случайное разбиение на train и validation не подходит. Для проверки качества используется временное разбиение, при котором признаки берутся из одного месяца, а целевые события — из следующего.

## Моделирование

Эксперименты с моделями находятся в ноутбуке:

```text
notebooks/02_modeling_experiments.ipynb
```

В проекте реализованы два подхода:

1. **Popularity baseline**  
   Простая стратегия, которая рекомендует наиболее популярные новые продукты.

2. **Sklearn model**  
   Бинарная модель в формате `клиент × продукт`, которая оценивает вероятность покупки конкретного продукта клиентом.

Обучающая выборка строится на временных парах месяцев:

```text
train history month: 2016-03
train target month:  2016-04

valid history month: 2016-04
valid target month:  2016-05
```

Текущий эксперимент был выполнен с ограничением по числу клиентов для воспроизводимого запуска на локальной машине.

## Результаты

Полученные метрики:

| Подход | MAP@7 | Precision@7 | Recall@7 | HitRate@7 | Coverage@7 |
|---|---:|---:|---:|---:|---:|
| Popularity baseline | 0.0177 | 0.1758 | 0.9501 | 0.9543 | 0.8750 |
| Sklearn model | 0.0011 | 0.0097 | 0.0531 | 0.0532 | 0.7917 |

В текущем эксперименте популярностный baseline оказался сильнее ML-модели. Это важный результат для рекомендательной задачи: простая популярностная стратегия может быть сильной точкой сравнения, особенно при выраженном дисбалансе классов и ограниченном наборе признаков.

Дальнейшие направления улучшения:

- добавить лаговые признаки по продуктам за несколько предыдущих месяцев;
- добавить агрегаты популярности продуктов по сегментам, возрастным группам и активности клиентов;
- увеличить обучающую выборку;
- подобрать стратегию negative sampling;
- протестировать CatBoost или LightGBM;
- отдельно проанализировать ошибки по продуктам и клиентским сегментам.

## Обучение модели

Запуск обучения:

```bash
python scripts/train.py
```

Скрипт выполняет:

- загрузку данных;
- построение train/validation выборок;
- обучение модели;
- расчёт ranking-метрик;
- логирование эксперимента в MLflow;
- сохранение модели в `models/`;
- сохранение метрик в `reports/`.

После успешного запуска создаются файлы:

```text
models/model.joblib
models/model.metadata.json
reports/metrics.json
```

Эти файлы не хранятся в GitHub и воспроизводятся командой обучения.

## MLflow

MLflow используется для логирования экспериментов: параметров, метрик, модели и артефактов.

Локальный запуск MLflow:

```bash
bash scripts/run_mlflow.sh
```

Интерфейс будет доступен по адресу:

```text
http://127.0.0.1:5000
```

После запуска `python scripts/train.py` в MLflow появляется эксперимент:

```text
bank_product_recommendation
```

В нём логируются:

- тип модели;
- месяцы train/validation;
- `negative_ratio`;
- `max_users`;
- `top_k`;
- `random_state`;
- `MAP@7`;
- `Precision@7`;
- `Recall@7`;
- `HitRate@7`;
- `Coverage@7`.

## API

Модель обёрнута в REST API на FastAPI.

Локальный запуск:

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Доступные endpoints:

```text
GET  /health
POST /recommend
GET  /metrics
```

### GET /health

Проверяет состояние сервиса и доступность модели.

Пример ответа:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/model.joblib"
}
```

### POST /recommend

Возвращает топ продуктовых рекомендаций для клиента.

Пример запроса:

```json
{
  "ncodpers": 123456,
  "age": 35,
  "antiguedad": 18,
  "renta": 120000,
  "ind_nuevo": 0,
  "indrel": 1,
  "ind_actividad_cliente": 1,
  "months_from_join": 18,
  "ind_empleado": "N",
  "pais_residencia": "ES",
  "sexo": "H",
  "indrel_1mes": "1",
  "tiprel_1mes": "A",
  "indresi": "S",
  "indext": "N",
  "conyuemp": "unknown",
  "canal_entrada": "KHE",
  "indfall": "N",
  "cod_prov": "28",
  "nomprov": "MADRID",
  "segmento": "02 - PARTICULARES",
  "products": {
    "ind_ahor_fin_ult1": 0,
    "ind_aval_fin_ult1": 0,
    "ind_cco_fin_ult1": 1,
    "ind_cder_fin_ult1": 0,
    "ind_cno_fin_ult1": 0,
    "ind_ctju_fin_ult1": 0,
    "ind_ctma_fin_ult1": 0,
    "ind_ctop_fin_ult1": 0,
    "ind_ctpp_fin_ult1": 0,
    "ind_deco_fin_ult1": 0,
    "ind_deme_fin_ult1": 0,
    "ind_dela_fin_ult1": 0,
    "ind_ecue_fin_ult1": 0,
    "ind_fond_fin_ult1": 0,
    "ind_hip_fin_ult1": 0,
    "ind_plan_fin_ult1": 0,
    "ind_pres_fin_ult1": 0,
    "ind_reca_fin_ult1": 0,
    "ind_tjcr_fin_ult1": 0,
    "ind_valo_fin_ult1": 0,
    "ind_viv_fin_ult1": 0,
    "ind_nomina_ult1": 0,
    "ind_nom_pens_ult1": 0,
    "ind_recibo_ult1": 1
  }
}
```

Пример ответа:

```json
{
  "client_id": 123456,
  "recommendations": [
    "ind_nom_pens_ult1",
    "ind_tjcr_fin_ult1",
    "ind_nomina_ult1",
    "ind_cno_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_ctma_fin_ult1"
  ],
  "top_k": 7,
  "model_loaded": true
}
```

## Docker

Проект можно поднять через Docker Compose:

```bash
docker compose up --build
```

После запуска доступны:

```text
API:    http://127.0.0.1:8000/docs
MLflow: http://127.0.0.1:5001
```

В Docker Compose MLflow проброшен на порт `5001`, так как на macOS порт `5000` может быть занят системными процессами.

Остановка контейнеров:

```bash
docker compose down
```

## Мониторинг

Сервис отдаёт Prometheus-compatible метрики через endpoint:

```text
GET /metrics
```

Мониторятся:

- количество запросов к сервису;
- latency рекомендаций;
- ошибки инференса;
- количество рекомендаций по каждому продукту;
- факт загрузки модели.

Эти метрики позволяют отслеживать техническое состояние сервиса и поведение модели после выкладки.

Дополнительное описание мониторинга находится в файле:

```text
docs/monitoring.md
```

## Воспроизводимость

Для воспроизводимости в проекте зафиксированы:

- зависимости в `requirements.txt`;
- `random_state`;
- структура train/validation разбиения;
- конфигурационные параметры в `src/bank_recs/config.py`;
- скрипт обучения `scripts/train.py`;
- Dockerfile и docker-compose.yml.

Основной сценарий воспроизведения:

```bash
conda create -n bank-products python=3.11 -y
conda activate bank-products
pip install -r requirements.txt

python scripts/train.py
uvicorn app.main:app --reload
```

Проверка API:

```text
http://127.0.0.1:8000/docs
```

Проверка MLflow:

```bash
bash scripts/run_mlflow.sh
```

```text
http://127.0.0.1:5000
```

## Что не хранится в репозитории

В репозиторий не добавляются:

- исходные CSV-файлы;
- обработанные датасеты;
- обученные модели;
- MLflow-артефакты;
- локальное виртуальное окружение;
- сгенерированные отчёты.

Эти файлы создаются локально при запуске соответствующих скриптов.
