from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests

app = FastAPI()

MODEL_API_URL = "https://punjab-yield-api-production-7fa3.up.railway.app/predict"

class ScenarioRequest(BaseModel):
    crop: str
    district: str
    sowing_date: Optional[str] = None
    fertilizer_protocol: Optional[str] = "Balanced NPK (15-15-15)"
    cultivar: Optional[str] = None

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/simulate")
def simulate(request: ScenarioRequest):
    try:
        response = requests.post(MODEL_API_URL, json=request.dict(), timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Model service timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Model service error: {e}")
