```python
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, field_validator


# ============================================================
# 1. FASTAPI APPLICATION SETUP
# ============================================================

app = FastAPI(
    title="SafeRoute AI — Road Accident Risk and Route Recommendation API",
    description="API for analyzing road accidents and supporting safe and fast routes",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent

# Change these paths if your files are stored elsewhere.
ACCIDENT_FILE = Path(
    r"C:\Users\DELL\Downloads\road-accidents-2024-states-road-accidents.csv"
)
TRAFFIC_FILE = Path(
    "C:\Users\DELL\Downloads\archive (1)\traffic.csv"
)


# ============================================================
# 2. DATA LOADING AND FEATURE ENGINEERING
# ============================================================

def load_data():
    """Load accident records and create features when source columns exist."""

    if not ACCIDENT_FILE.exists():
        print(f"WARNING: Accident dataset not found: {ACCIDENT_FILE}")
        return pd.DataFrame()

    try:
        data = pd.read_csv(ACCIDENT_FILE)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"Could not read accident dataset: {exc}")
        return pd.DataFrame()

    data.columns = [str(col).strip() for col in data.columns]

    # Identify columns without assuming every dataset uses identical names.
    columns = {col.lower(): col for col in data.columns}

    def find_col(*names):
        for name in names:
            if name.lower() in columns:
                return columns[name.lower()]
        return None

    # Create time-based features if date/time information is available.
    date_col = find_col("accident_date", "date", "crash_date", "datetime", "timestamp")

    if date_col:
        dt = pd.to_datetime(data[date_col], errors="coerce")
        data["accident_hour"] = dt.dt.hour.astype("Int64")
        data["day_of_week"] = dt.dt.dayofweek.astype("Int64")
        data["is_weekend"] = (
            dt.dt.dayofweek.isin([5, 6])
            .where(dt.notna(), pd.NA)
            .astype("boolean")
        )
        data["is_night"] = (
            ((dt.dt.hour >= 20) | (dt.dt.hour < 6))
            .where(dt.notna(), pd.NA)
            .astype("boolean")
        )

    else:
        time_col = find_col("accident_time", "time", "crash_time", "hour")
        if time_col:
            dt = pd.to_datetime(data[time_col].astype("string"), errors="coerce")
            data["accident_hour"] = dt.dt.hour.astype("Int64")
            data["is_night"] = (
                ((dt.dt.hour >= 20) | (dt.dt.hour < 6))
                .where(dt.notna(), pd.NA)
                .astype("boolean")
            )

    # Casualties: use a total column or sum available fatality/injury columns.
    casualty_col = find_col("casualty_count", "total_casualties", "casualties")
    if casualty_col:
        data["casualty_count"] = pd.to_numeric(
            data[casualty_col], errors="coerce"
        ).clip(lower=0)
    else:
        parts = []
        for names in (
            ("fatalities", "deaths", "number_of_deaths"),
            ("injuries", "number_of_injuries", "injured"),
        ):
            col = find_col(*names)
            if col:
                parts.append(pd.to_numeric(data[col], errors="coerce"))

        if parts:
            data["casualty_count"] = pd.concat(parts, axis=1).sum(
                axis=1, min_count=1
            ).clip(lower=0)

    # Vehicle count, if recorded.
    vehicle_col = find_col("vehicle_count", "number_of_vehicles", "total_vehicles")
    if vehicle_col:
        data["vehicle_count"] = pd.to_numeric(
            data[vehicle_col], errors="coerce"
        ).clip(lower=0)

    # Weather and rain features.
    weather_col = find_col("weather", "weather_condition", "weather_conditions")
    if weather_col:
        weather = data[weather_col].astype("string").str.lower()
        known = weather.notna() & (weather != "")
        data["has_rain"] = (
            weather.str.contains(r"rain|drizzle|shower", na=False)
            .where(known, pd.NA)
            .astype("boolean")
        )

    # Lighting features.
    lighting_col = find_col(
        "lighting", "light_condition", "lighting_condition", "road_lighting"
    )
    if lighting_col:
        lighting = data[lighting_col].astype("string").str.lower()
        known = lighting.notna() & (lighting != "")
        data["has_poor_lighting"] = (
            lighting.str.contains(
                r"poor|dark|unlit|no light|inadequate", na=False
            )
            .where(known, pd.NA)
            .astype("boolean")
        )

    # Wet-road features.
    surface_col = find_col("road_surface", "road_condition", "surface_condition")
    if surface_col:
        surface = data[surface_col].astype("string").str.lower()
        known = surface.notna() & (surface != "")
        data["has_wet_road"] = (
            surface.str.contains(r"wet|waterlogged|slippery|flood", na=False)
            .where(known, pd.NA)
            .astype("boolean")
        )

    # Severity score, only if a severity column exists.
    severity_col = find_col(
        "severity", "accident_severity", "severity_level", "crash_severity"
    )
    if severity_col:
        values = data[severity_col]
        numeric = pd.to_numeric(values, errors="coerce")
        severity_map = {
            "minor": 1, "slight": 1, "low": 1,
            "moderate": 2,
            "serious": 3, "severe": 3, "major": 3,
            "fatal": 4, "fatality": 4,
        }
        mapped = values.astype("string").str.strip().str.lower().map(severity_map)
        data["severity_score"] = numeric.where(numeric.notna(), mapped)

    print(f"Loaded {len(data)} accident records.")
    return data


# Load the dataset when this module starts.
df = load_data()


# Load traffic data separately; do not merge datasets without matching keys.
def load_traffic_data():
    if not TRAFFIC_FILE.exists():
        print(f"Traffic dataset not found: {TRAFFIC_FILE}")
        return pd.DataFrame()

    try:
        traffic = pd.read_csv(TRAFFIC_FILE)
        traffic.columns = [str(col).strip() for col in traffic.columns]
        print(f"Loaded {len(traffic)} traffic records.")
        return traffic
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"Could not read traffic dataset: {exc}")
        return pd.DataFrame()


traffic_df = load_traffic_data()


# ============================================================
# 3. GET /summary
# ============================================================

@app.get("/summary")
def summary():
    """Return a summary of the available historical accident data."""

    severity_col = next(
        (
            col for col in df.columns
            if col.lower() in ("severity", "accident_severity",
                               "severity_level", "crash_severity")
        ),
        None,
    )

    severity_distribution = {}
    if severity_col:
        severity_distribution = (
            df[severity_col].fillna("Unknown").astype(str)
            .value_counts().to_dict()
        )

    casualties = (
        pd.to_numeric(df["casualty_count"], errors="coerce")
        if "casualty_count" in df.columns
        else pd.Series(dtype=float)
    )

    rain_count = (
        int((df["has_rain"] == True).sum())
        if "has_rain" in df.columns else None
    )
    night_count = (
        int((df["is_night"] == True).sum())
        if "is_night" in df.columns else None
    )
    poor_lighting_count = (
        int((df["has_poor_lighting"] == True).sum())
        if "has_poor_lighting" in df.columns else None
    )

    return {
        "total_accidents": int(len(df)),
        "severity_distribution": severity_distribution,
        "average_casualties": (
            round(float(casualties.mean()), 2)
            if casualties.notna().any() else None
        ),
        "rain_accident_count": rain_count,
        "night_accident_count": night_count,
        "poor_lighting_accident_count": poor_lighting_count,
        "data_warning": (
            "Dataset is empty or unavailable."
            if df.empty else
            "Counts describe recorded historical accidents, not live risk probabilities."
        ),
    }


# ============================================================
# 4. GET /high-risk-segments
# ============================================================

@app.get("/high-risk-segments")
def high_risk_segments():
    """Group historical accidents by road, if a road identifier exists."""

    road_col = next(
        (col for col in df.columns
         if col.lower() in ("road_id", "road_name", "road", "street", "road_number")),
        None,
    )

    if df.empty:
        return {"message": "Accident dataset is empty or unavailable.", "segments": []}

    if road_col is None:
        return {
            "message": "No road identifier column exists in the dataset.",
            "segments": [],
        }

    grouped = df.dropna(subset=[road_col]).groupby(road_col, dropna=True)
    results = []

    for road, group in grouped:
        count = len(group)
        severity_col = "severity_score" if "severity_score" in group.columns else None
        avg_severity = (
            float(group[severity_col].mean())
            if severity_col and group[severity_col].notna().any()
            else None
        )

        # These are simple relative labels, not calibrated accident probabilities.
        risk_level = "High" if count >= 10 else "Medium" if count >= 4 else "Low"

        results.append({
            "road_name": str(road),
            "accident_count": int(count),
            "average_severity_score": (
                round(avg_severity, 2) if avg_severity is not None else None
            ),
            "risk_level": risk_level,
        })

    results.sort(key=lambda item: item["accident_count"], reverse=True)
    return {"segments": results}


# ============================================================
# 5. GET /accidents
# ============================================================

@app.get("/accidents")
def get_accidents(
    road_name: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    accident_hour: int | None = Query(default=None, ge=0, le=23),
):
    """Return accident rows with optional filters."""

    if df.empty:
        return {"message": "Accident dataset is empty or unavailable.", "accidents": []}

    result = df.copy()

    road_col = next(
        (col for col in result.columns
         if col.lower() in ("road_name", "road", "street", "road_number", "road_id")),
        None,
    )
    severity_col = next(
        (col for col in result.columns
         if col.lower() in ("severity", "accident_severity",
                            "severity_level", "crash_severity")),
        None,
    )

    if road_name and road_col:
        result = result[
            result[road_col].astype("string").str.contains(
                road_name, case=False, na=False
            )
        ]

    if severity and severity_col:
        result = result[
            result[severity_col].astype("string").str.lower() == severity.lower()
        ]

    if accident_hour is not None and "accident_hour" in result.columns:
        result = result[result["accident_hour"] == accident_hour]

    # Limit response size and convert missing values to JSON null.
    result = result.head(500).replace({np.nan: None, pd.NA: None})
    return {"count": len(result), "accidents": result.to_dict(orient="records")}


# ============================================================
# 6. POST /predict-route-risk — Pydantic input validation
# ============================================================

class RouteRiskInput(BaseModel):
    latitude: float = Field(ge=-90, le=90, description="Current latitude")
    longitude: float = Field(ge=-180, le=180, description="Current longitude")
    destination_latitude: float = Field(ge=-90, le=90)
    destination_longitude: float = Field(ge=-180, le=180)
    distance_km: float = Field(gt=0, le=1000)
    estimated_travel_time_min: float = Field(gt=0, le=1440)
    traffic_level: str
    weather_condition: str

    @field_validator("traffic_level")
    @classmethod
    def validate_traffic(cls, value):
        allowed = {"low", "medium", "high"}
        if value.strip().lower() not in allowed:
            raise ValueError("traffic_level must be Low, Medium, or High")
        return value.strip().title()

    @field_validator("weather_condition")
    @classmethod
    def validate_weather(cls, value):
        allowed = {"clear", "rain", "fog", "unknown"}
        if value.strip().lower() not in allowed:
            raise ValueError(
                "weather_condition must be Clear, Rain, Fog, or Unknown"
            )
        return value.strip().title()


@app.post("/predict-route-risk")
def predict_route_risk(route: RouteRiskInput):
    """
    Estimate a preliminary heuristic risk score.
    This is not a trained ML prediction or calibrated probability.
    """

    score = 20.0
    warnings = []

    if df.empty:
        warnings.append("Historical accident dataset is unavailable.")
    else:
        # Add a modest, documented penalty if the dataset has severity scores.
        if "severity_score" in df.columns and df["severity_score"].notna().any():
            average_severity = float(df["severity_score"].mean())
            score += min(max(average_severity, 0) * 5, 20)
        else:
            warnings.append("Historical severity information is unavailable.")

    traffic_penalty = {"Low": 0, "Medium": 15, "High": 30}
    weather_penalty = {"Clear": 0, "Rain": 15, "Fog": 20, "Unknown": 5}

    score += traffic_penalty[route.traffic_level]
    score += weather_penalty[route.weather_condition]
    score = round(min(max(score, 0), 100), 2)

    risk_level = "High" if score >= 60 else "Medium" if score >= 35 else "Low"

    return {
        "estimated_risk_score": score,
        "risk_level": risk_level,
        "distance_km": route.distance_km,
        "estimated_travel_time_min": route.estimated_travel_time_min,
        "traffic_level": route.traffic_level,
        "weather_condition": route.weather_condition,
        "method": "Heuristic estimate; not a trained ML model or accident probability.",
        "warnings": warnings,
    }


# ============================================================
# 7. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "SafeRoute AI — Road Accident Risk and Route Recommendation System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0",
    }


# ============================================================
# 8. RUN SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
