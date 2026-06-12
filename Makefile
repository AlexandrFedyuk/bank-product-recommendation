.PHONY: sample train api mlflow test

sample:
	python scripts/make_sample_data.py

train:
	PYTHONPATH=src python scripts/train.py --data-path data/raw/train_ver2.csv

train-sample:
	PYTHONPATH=src python scripts/train.py --data-path data/raw/train_ver2_sample.csv --max-users 0

api:
	PYTHONPATH=src:. MODEL_PATH=models/model.joblib uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	./scripts/run_mlflow.sh

test:
	PYTHONPATH=src pytest -q
