import json
from joblib import load
import pandas as pd

def ensure_dict(features):
    """String gelirse JSON olarak yükle, değilse olduğu gibi döndür"""
    if isinstance(features, str):
        try:
            return json.loads(features)
        except json.JSONDecodeError:
            return {}
    return features

def svm_lung_predictor(features):
    features = ensure_dict(features)

    # GENDER dönüşümü
    if isinstance(features.get("gender"), str):
        gender_val = features["gender"].lower()
        features["gender"] = 1 if gender_val == "m" else 0 if gender_val == "f" else None

    df = pd.DataFrame([features])
    model = load("svm_lung_model.joblib")
    return int(model.predict(df)[0])


def rf_lung_predictor(features):
    features = ensure_dict(features)

    if isinstance(features.get("gender"), str):
        gender_val = features["gender"].lower()
        features["gender"] = 1 if gender_val == "m" else 0 if gender_val == "f" else None

    df = pd.DataFrame([features])
    model = load("rf_lung_model.joblib")
    return int(model.predict(df)[0])
