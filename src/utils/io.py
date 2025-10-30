import pandas as pd
import json
from pathlib import Path
import joblib

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f:
        json.dump(obj,f,indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_csv(path):
    return pd.read_csv(path)

def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
