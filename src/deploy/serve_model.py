from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import torch
import joblib
import json
import numpy as np

app = FastAPI(title="Medical AI Model Serving")

class PredictRequest(BaseModel):
    model_type: str
    input_data: dict

MODEL_DIR = Path(__file__).resolve().parent.parent / "experiments"

loaded_models = {}

def load_model(model_type):
    if model_type in loaded_models:
        return loaded_models[model_type]
    model_path = MODEL_DIR / model_type
    if not model_path.exists():
        raise FileNotFoundError(f"{model_type} model not found")
    if (model_path / "final_model").exists():  # TensorFlow model
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path / "final_model"))
    elif (model_path / "rf0.pkl").exists() or (model_path / "rf_ite_model.pkl").exists():  # RF / causal
        model = joblib.load(model_path / next(model_path.glob("*.pkl")))
    elif (model_path / "flan_t5_lora").exists():  # HuggingFace model
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(str(model_path / "flan_t5_lora"))
        model = T5ForConditionalGeneration.from_pretrained(str(model_path / "flan_t5_lora"))
        model.tokenizer = tokenizer
    else:
        raise ValueError("Unsupported model type")
    loaded_models[model_type] = model
    return model

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model = load_model(req.model_type)
        if req.model_type.startswith("unet3d"):  # imaging
            x = np.array(req.input_data["volume"])[None,...,None]
            y_pred = model.predict(x).tolist()
        elif req.model_type.startswith("t_learner") or req.model_type.startswith("s_learner"):  # causal
            X = np.array([list(req.input_data["features"].values())])
            y_pred = model.predict(X).tolist()
        elif req.model_type.startswith("flan_t5"):  # LLM
            input_text = req.input_data["text"]
            tokenizer = model.tokenizer
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(**inputs)
            y_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            y_pred = None
        return {"prediction": y_pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
