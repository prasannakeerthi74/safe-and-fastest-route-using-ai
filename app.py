```python
# main.py
# SafeRoute AI - FastAPI Backend

from pathlib import Path
from typing import Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Initialize FastAPI
# --------------------------------------------------
app = FastAPI(
    title="SafeRoute AI API",
    description=(
        "Accident analysis and candidate route risk assessment "
        "for a safe and fast route recommendation system."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# 2. Dataset paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

ACCIDENT_FILE = (
    BASE_DIR
    / "data"
    / "road-accidents-2024-states-road-accidents.csv"
)

# Optional traffic dataset. Configure this when you have the file.
TRAFFIC_FILE = BASE_DIR / "data" / "traffic_data.csv"


# --------------------------------------------------
# 3. Load accident dataset
# --------------------------------------------------
def load_accident_data():
    """Load the accident dataset and normalize column names."""

    if not ACCIDENT_FILE.exists():
        raise FileNotFoundError(
            f"Accident CSV not found: {ACCIDENT_FILE}"
        )

    df = pd.read_csv(ACCIDENT_FILE)

    # Normalize names to simplify matching columns.
    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    return df


try:
    accidents_df = load_accident_data()
except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    accidents_df = pd.DataFrame()
    DATA_LOAD_ERROR = str(exc)
else:
    DATA_LOAD_ERROR = None


# --------------------------------------------------
# 4. Helper functions
# --------------------------------------------------
def find_column(candidates):
    """Find a matching column in the loaded accident dataset."""
    for column in candidates:
        if column in accidents_df.columns:
            return column
    return None


def numeric_sum(column_name):
    """Return the numeric sum of a column, if available."""
    if not column_name:
        return None

    values = pd.to_numeric(
        accidents_df[column_name], errors="coerce"
    )
    return int(values.fillna(0).sum())


def require_data():
    """Stop data-dependent endpoints when the CSV is unavailable."""
    if accidents_df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"Accident dataset is unavailable. {DATA_LOAD_ERROR}"
        )


ACCIDENT_COUNT_COL = find_column([
    "accidents",
    "total_accidents",
    "accident_count",
    "number_of_accidents"
])

FATALITY_COL = find_column([
    "fatalities",
    "deaths",
    "total_deaths",
    "persons_killed"
])

INJURY_COL = find_column([
    "injuries",
    "total_injuries",
    "persons_injured"
])

LOCATION_COL = find_column([
    "location",
    "road_name",
    "road",
    "state",
    "district",
    "city",
    "area",
    "region"
])

SEVERITY_COL = find_column([
    "severity",
    "accident_severity",
    "severity_score"
])


# --------------------------------------------------
# 5. Root endpoint
# --------------------------------------------------
@app.get("/")
def home():
    return {
        "project": "SafeRoute AI",
        "status": "running",
        "data_available": not accidents_df.empty,
        "endpoints": [
            "/summary",
            "/high-risk-segments",
            "/accidents",
            "/predict-route-risk",
            "/docs"
        ]
    }


# --------------------------------------------------
# 6. Accident summary endpoint
# --------------------------------------------------
@app.get("/summary")
def accident_summary():
    require_data()

    return {
        "total_records": int(len(accidents_df)),
        "reported_accidents": numeric_sum(ACCIDENT_COUNT_COL),
        "total_fatalities": numeric_sum(FATALITY_COL),
        "total_injuries": numeric_sum(INJURY_COL),
        "unique_locations": (
            int(accidents_df[LOCATION_COL].nunique(dropna=True))
            if LOCATION_COL else None
        ),
        "available_columns": list(accidents_df.columns)
    }


# --------------------------------------------------
# 7. High-risk records endpoint
# --------------------------------------------------
@app.get("/high-risk-segments")
def high_risk_segments(
    limit: int = Query(default=10, ge=1, le=100)
):
    require_data()

    df = accidents_df.copy()

    # Use severity, fatality, injury or accident counts when available.
    if SEVERITY_COL:
        severity = pd.to_numeric(
            df[SEVERITY_COL], errors="coerce"
        )
        df["_risk_sort"] = severity

    elif FATALITY_COL:
        df["_risk_sort"] = pd.to_numeric(
            df[FATALITY_COL], errors="coerce"
        )

    elif INJURY_COL:
        df["_risk_sort"] = pd.to_numeric(
            df[INJURY_COL], errors="coerce"
        )

    elif ACCIDENT_COUNT_COL:
        df["_risk_sort"] = pd.to_numeric(
            df[ACCIDENT_COUNT_COL], errors="coerce"
        )

    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "Cannot rank high-risk records: the dataset needs a "
                "recognized severity, fatality, injury, or accident "
                "count column."
            )
        )

    df = df.sort_values(
        "_risk_sort", ascending=False, na_position="last"
    ).head(limit)

    # Replace NaN values so results can be returned as JSON.
    df = df.drop(columns=["_risk_sort"])
    df = df.astype(object).where(pd.notna(df), None)

    return {
        "count": len(df),
        "ranking_basis": (
            SEVERITY_COL or FATALITY_COL
            or INJURY_COL or ACCIDENT_COUNT_COL
        ),
        "records": df.to_dict(orient="records")
    }


# --------------------------------------------------
# 8. Accident records endpoint
# --------------------------------------------------
@app.get("/accidents")
def get_accidents(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0)
):
    require_data()

    page = accidents_df.iloc[offset:offset + limit].copy()
    page = page.astype(object).where(pd.notna(page), None)

    return {
        "total_records": int(len(accidents_df)),
        "offset": offset,
        "limit": limit,
        "records": page.to_dict(orient="records")
    }


# --------------------------------------------------
# 9. Route risk request model
# --------------------------------------------------
class RouteRiskInput(BaseModel):
    route_name: str = Field(
        default="Candidate Route",
        min_length=1,
        max_length=100
    )

    distance_km: float = Field(gt=0, le=5000)
    estimated_travel_time_min: float = Field(gt=0, le=100000)

    traffic_level: Literal["Low", "Medium", "High"] = "Medium"

    weather_condition: Literal[
        "Clear", "Rain", "Fog", "Unknown"
    ] = "Unknown"

    # A supplied estimate from a route/risk source, not a model output.
    base_risk_score: float = Field(default=30, ge=0, le=100)


# --------------------------------------------------
# 10. Candidate route risk assessment endpoint
# --------------------------------------------------
@app.post("/predict-route-risk")
def predict_route_risk(request: RouteRiskInput):
    # This is a transparent heuristic, not a trained ML prediction.
    traffic_penalties = {
        "Low": 0,
        "Medium": 10,
        "High": 25
    }

    weather_penalties = {
        "Clear": 0,
        "Rain": 15,
        "Fog": 20,
        "Unknown": 5
    }

    score = (
        request.base_risk_score * 0.75
        + traffic_penalties[request.traffic_level] * 0.15
        + weather_penalties[request.weather_condition] * 0.10
    )

    score = round(max(0, min(score, 100)), 2)

    if score < 30:
        risk_level = "Low"
    elif score < 60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "route_name": request.route_name,
        "distance_km": request.distance_km,
        "estimated_travel_time_min": (
            request.estimated_travel_time_min
        ),
        "traffic_level": request.traffic_level,
        "weather_condition": request.weather_condition,
        "risk_score": score,
        "risk_level": risk_level,
        "method": "heuristic",
        "warning": (
            "This score is not a trained ML prediction. "
            "It depends on the risk estimate supplied in the request. "
            "No route geometry or live traffic is fetched by this endpoint."
        )
    }


# --------------------------------------------------
# 11. Run the API server
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
