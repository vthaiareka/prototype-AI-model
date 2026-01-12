from fastapi import FastAPI, Query
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

app = FastAPI(title="Expense Category Inference API")

# ----------------------------
# Load model artifacts ONCE
# ----------------------------
model_path = Path(__file__).parent / "expense_category_model.joblib"
artifact = joblib.load(model_path)

MODEL_NAME = artifact["model_name"]
category_names = artifact["category_names"]
category_embeddings = artifact["category_embeddings"]

model = SentenceTransformer(MODEL_NAME)

# ----------------------------
# Prediction logic
# ----------------------------
def predict(text: str, top_k: int = 1):
    emb = model.encode([text], normalize_embeddings=True)
    scores = emb @ category_embeddings.T
    top_idx = np.argsort(scores[0])[::-1][:top_k]

    return [
        {
            "category": category_names[i],
            "score": float(scores[0][i])
        }
        for i in top_idx
    ]

# ----------------------------
# API endpoint
# ----------------------------
@app.post("/predict")
def predict_endpoint(
    text: str = Query(..., min_length=1),
    top_k: int = 1
):
    return {
        "input": text,
        "predictions": predict(text, top_k)
    }
