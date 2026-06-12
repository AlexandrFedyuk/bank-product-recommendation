#!/bin/bash
set -e
export PYTHONPATH="$PWD/src:$PWD"
export MODEL_PATH="${MODEL_PATH:-models/model.joblib}"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
