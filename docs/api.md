# API рекомендательной системы

Сервис реализован на FastAPI.

## Запуск

```bash
./scripts/run_api.sh
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### `GET /health`

Проверяет состояние сервиса и наличие модели.

Пример ответа:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/model.joblib"
}
```

Если модель не обучена, сервис всё равно запускается, но возвращает fallback-рекомендации по популярным продуктам.

### `POST /recommend`

Возвращает список рекомендованных продуктов.

Пример запроса:

```json
{
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
}
```

Пример ответа:

```json
{
  "client_id": 123456,
  "recommendations": [
    "ind_ecue_fin_ult1",
    "ind_nom_pens_ult1",
    "ind_nomina_ult1",
    "ind_tjcr_fin_ult1",
    "ind_cno_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_dela_fin_ult1"
  ],
  "top_k": 7,
  "model_loaded": true
}
```

### `GET /metrics`

Возвращает Prometheus-метрики.
