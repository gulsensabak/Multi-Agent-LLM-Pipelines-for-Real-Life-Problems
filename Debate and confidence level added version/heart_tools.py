import json
from joblib import load
import pandas as pd

def ensure_dict(features):
    """Eğer string ise JSON olarak sözlüğe çevir, değilse aynen döndür"""
    if isinstance(features, str):
        try:
            return json.loads(features)
        except json.JSONDecodeError:
            return {}
    return features

def svm_heart_predictor(features) -> int:
    features = ensure_dict(features)
    df = pd.DataFrame([features])
    model = load("svm_model.joblib")
    return int(model.predict(df)[0])

def rf_heart_predictor(features) -> int:
    features = ensure_dict(features)
    df = pd.DataFrame([features])
    model = load("rf_heart_model.joblib")
    return int(model.predict(df)[0])
