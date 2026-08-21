from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Real input fields, confirmed with your teammates
class ScenarioRequest(BaseModel):
    crop_type: str
    district: str
    farm_area: float
    soil_moisture: float
    rainfall: float
    humidity: float
    temperature: float
    nitrogen: float
    fertilizer_amount: float

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/simulate")
def simulate(request: ScenarioRequest):

    # --- FAKE simulation result for now (swap for real model call later) ---
    fake_yield_result = {
        "predicted_yield_tonnes_per_ha": 4.2
    }

    # --- FAKE LLM advice for now ---
    fake_advice = "Based on current soil moisture and rainfall, consider reducing irrigation by 10%."

    return {
        "input_received": request.dict(),
        "simulation": fake_yield_result,
        "advice": fake_advice
    }