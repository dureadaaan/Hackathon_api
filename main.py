from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from predict import predict_yield

app = FastAPI()

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
    result = predict_yield(
        crop=request.crop,
        district=request.district,
        sowing_date=request.sowing_date,
        fertilizer_protocol=request.fertilizer_protocol,
        cultivar=request.cultivar
    )
    return result