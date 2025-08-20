from fastapi import FastAPI
from typing import Optional, List
from pydantic import BaseModel
import joblib
import os




app = FastAPI(title="Scam Email Detector", version="0.0.1")
model = os.getenv("model", "models/phish_pipeline.joblib")

pipeline = joblib.load(model)

class Email(BaseModel):
    subject: Optional[str] = ""
    body: str
    sender: Optional[str] = ""

class Prediction(BaseModel):
    scam_probability: float
    label: str
    reasons: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)  
def predict(email: Email):
    text = f"{email.subject or ''}\n{email.body or ''}\n{email.sender or ''}"

    prob = float(pipeline.predict_proba([text])[0][1]) 

    if prob >= 0.5 and prob <= 0.8:
        label = "Possibly a scam"
    elif prob > 0.8:
        label = "scam"
    else:
        label = "legit"
   

    risky_terms = ["urgent", "verify", "gift card", "password", "wire", "limited time"]
    reasons = [f"found risky term: '{t}'" for t in risky_terms if t in text.lower()] or ["model prediction"]

    return Prediction(scam_probability=prob, label=label, reasons=reasons)