from fastapi import FastAPI
from pydantic import BaseModel

# This creates your "messenger" — the thing that will run on Railway later
app = FastAPI()

# This describes exactly what data Unity must send us.
# Think of it as a form with required fields.
class ScenarioRequest(BaseModel):
    crop_type: str
    field_size_acres: float
    irrigation_amount_mm: float
    fertilizer_kg_per_acre: float

# This is a simple "is it alive?" check, useful for testing
@app.get("/")
def home():
    return {"message": "API is running!"}

# THIS is the important one — Unity will send data here
@app.post("/simulate")
def simulate(request: ScenarioRequest):

    # --- FAKE simulation result (we'll swap this for the real thing later) ---
    fake_simulation_result = {
        "predicted_yield_kg": 2200,
        "water_used_liters": 450000,
        "water_efficiency_score": 0.72
    }

    # --- FAKE LLM advice (we'll swap this for the real thing later) ---
    fake_advice = "Reducing irrigation by 15% in week 2 could improve water efficiency without hurting yield."

    # Bundle both together and send back to Unity
    return {
        "input_received": request.dict(),
        "simulation": fake_simulation_result,
        "advice": fake_advice
    }