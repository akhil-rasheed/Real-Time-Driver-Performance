from fastapi import FastAPI
from pydantic import BaseModel
from model import *


class EventsPerKm(BaseModel):
    harshAcceleration: float
    harshBraking: float
    harshTurning: float

class ModelResult(BaseModel):
    score: float
    percentile: float

app = FastAPI()


@app.post("/getScore")
async def get_score(eventsPerKm: EventsPerKm):
    converted = [{
        'Harsh Acceleration': eventsPerKm.harshAcceleration,
        'Harsh Braking': eventsPerKm.harshBraking,
        'Harsh Turning': eventsPerKm.harshTurning
    }]
    result =  getRankAndMetric(converted)
    return ModelResult(score = result[1], percentile = result[0])
