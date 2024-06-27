from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import pandas as pd
from challenge.model import DelayModel

app = FastAPI(
    title="Flight Delay Prediction API",
    description="API to predict flight delays.",
    version="0.1",
)

delay_model = DelayModel()

airlines = [
    "Aerolineas Argentinas",
    "Aeromexico",
    "Air Canada",
    "Air France",
    "Alitalia",
    "American Airlines",
    "Austral",
    "Avianca",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Grupo LATAM",
    "Iberia",
    "JetSmart SPA",
    "K.L.M.",
    "Lacsa",
    "Latin American Wings",
    "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas",
    "Qantas Airways",
    "Sky Airline",
    "United Airlines",
]
flight_types = ["N", "I"]


class FlightData(BaseModel):
    MES: int = Field(..., description="Month of the flight.")
    OPERA: str = Field(..., description="Airline operating the flight.")
    TIPOVUELO: str = Field(
        ..., description="Type of flight (N: National, I: International)."
    )

    @validator("MES")
    def validate_mes(cls, value):
        if value not in range(1, 13):
            raise HTTPException(
                status_code=400, detail="MES must be between 1 and 12."
            )
        return value

    @validator("OPERA")
    def validate_opera(cls, value):
        if value not in airlines:
            raise HTTPException(
                status_code=400, detail=f"OPERA must be one of {airlines}."
            )
        return value

    @validator("TIPOVUELO")
    def validate_tipo_vuelo(cls, value):
        if value not in flight_types:
            raise HTTPException(
                status_code=400, detail="TIPOVUELO must be either 'N' or 'I'."
            )
        return value


class PredictionRequest(BaseModel):
    flights: List[FlightData]


class PredictionResponse(BaseModel):
    predict: List[int]


@app.get("/", status_code=200)
async def root() -> dict:
    """
    Root endpoint.
    """
    return {
        "message": "Welcome to the Flight Delay Prediction API. "
        "Use /docs to access the API documentation."
    }


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Health check endpoint.
    """
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> PredictionResponse:
    """
    Endpoint to predict flights delay.
    """
    try:
        df = pd.DataFrame([flight.dict() for flight in request.flights])
        features = delay_model.preprocess(df)
        predictions = delay_model.predict(features)
        return PredictionResponse(predict=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
