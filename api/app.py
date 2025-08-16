from fastapi import FastAPI
from typing import Optional, List
from pydantic import BaseModel



app = FastAPI(title="Scam Email Detector", version="0.0.1")

class Email(BaseModel):
    subject: Optional[str] = ""
    body: str
    send: Optional[str] = ""

class Prediction(BaseModel):
    scam_probability: float
    label: str
    reason: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)  
def predict(email: Email):
    text = f"{email.subject}\n{email.body}".lower()

    risky_terms = ["urgent", "verify", "gift card", "password", "wire", "limited time"]
    hits = [t for t in risky_terms if t in text]

    score = min(len(hits) * 0.2, 0.95)
    label = "scam" if score >= 0.5 else "legit"
    reasons = [f"found risky term: '{t}'" for t in hits] or ["no obvious risky terms"]

    return Prediction(scam_probability=score, label=label, reason=reasons)