"""
SafeRoute AI — Road Accident Data Analysis
==========================================
This module prepares historical road-accident data, calculates descriptive
statistics, saves static charts, and displays interactive charts.

Expected input file:
    data/road_accidents.csv

Install dependencies:
    pip install pandas numpy matplotlib plotly

Important:
- Results describe the supplied historical dataset.
- Accident counts are not the same as accident probabilities.
- This file does not yet train an ML model or fetch live traffic data.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# ---------------------------------------------------------------------------
# PROMPT 1 — Load Road Accident Data and Feature Engineering
# ---------------------------------------------------------------------------
def load_and_prepare_data(filepath):
    """Load a road-accident CSV, prepare available fields, and return a DataFrame."""
    filepath = Path(filepath)

    # Fail clearly instead of silently creating or substituting a dataset.
    if not filepath.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {filepath.resolve()}\n"
            "Place your real CSV at data/road_accidents.csv or update the path."
        )

    # Load the CSV and normalize column names.
    df = pd.read_csv(filepath)
    original_count = len(df)
    df.columns = df.columns.astype(str).str.strip()

    # Prevent ambiguous behavior if trimming names caused duplicates.
    if df.columns.duplicated().any():
        duplicate_names = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names after cleaning: {duplicate_names}")

    # Standardize text values in likely categorical columns.
    # This changes formatting only; it does not invent missing labels.
    categorical_columns = [
        "city", "road_name", "road_type", "weather_condition",
        "road_surface_condition", "light_condition", "traffic_condition",
        "accident_severity",
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].map(
                lambda value: value.strip().lower() if isinstance(value, str) else value
            )

    # Convert available date columns to datetime; invalid values become NaT.
    if "accident_date" in df.columns:
        df["accident_date"] = pd.to_datetime(df["accident_date"], errors="coerce")

    # Convert coordinates and count fields to numeric; invalid values become NaN.
    numeric_sources = [
        "latitude", "longitude", "number_of_vehicles", "number_of_casualties"
    ]
    for col in numeric_sources:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicates only when an accident identifier exists.
    duplicates_removed = 0
    if "accident_id" in df.columns:
        duplicate_mask = df["accident_id"].notna() & df["accident_id"].duplicated()
        duplicates_removed = int(duplicate_mask.sum())
        df = df.loc[~duplicate_mask].copy()

    # Derive accident_hour from accident_time when available, otherwise from
    # a valid datetime in accident_date. No time is guessed when unavailable.
    if "accident_time" in df.columns:
        time_text = df["accident_time"].astype("string").str.strip()
        parsed_time = pd.to_datetime(time_text, errors="coerce")
        df["accident_hour"] = parsed_time.dt.hour
        # Support common time-only strings that pandas may parse inconsistently.
        missing_hour = df["accident_hour"].isna()
        extracted_hour = time_text.str.extract(
            r"^\s*(\d{1,2})(?::\d{2})?(?::\d{2})?\s*(?:AM|PM)?\s*$",
            expand=False,
            flags=0,
        )
        extracted_hour = pd.to_numeric(extracted_hour, errors="coerce")
        # The extraction fallback is safe for 24-hour values only; AM/PM is
        # left to the datetime parser so it is not misinterpreted.
        no_ampm = ~time_text.str.contains(r"\b(?:AM|PM)\b", case=False, na=False)
        valid_hour = extracted_hour.between(0, 23) & no_ampm
        df.loc[missing_hour & valid_hour, "accident_hour"] = extracted_hour[
            missing_hour & valid_hour
        ]
    elif "accident_date" in df.columns:
        df["accident_hour"] = df["accident_date"].dt.hour

    # Derive day of week and weekend status only from a valid date.
    if "accident_date" in df.columns:
        df["day_of_week"] = df["accident_date"].dt.day_name()
        df["is_weekend"] = df["accident_date"].dt.dayofweek.isin([5, 6]).where(
            df["accident_date"].notna(), pd.NA
        )

    # Define nighttime as 8 PM through 5:59 AM (documented convention).
    if "accident_hour" in df.columns:
        known_hour = df["accident_hour"].notna()
        night_values = ((df["accident_hour"] >= 20) | (df["accident_hour"] < 6))
        df["is_night"] = night_values.where(known_hour, pd.NA)

    # Create consistently named numeric feature columns when source fields exist.
    if "number_of_casualties" in df.columns:
        df["casualty_count"] = df["number_of_casualties"]
    if "number_of_vehicles" in df.columns:
        df["vehicle_count"] = df["number_of_vehicles"]

    # Create indicator columns only when the source column exists. Missing
    # source values remain missing rather than being classified as False.
    if "weather_condition" in df.columns:
        weather = df["weather_condition"].astype("string")
        known = weather.notna() & weather.str.strip().ne("")
        rain = weather.str.contains(r"\brain\b|rainy|drizzle|shower", case=False, na=False)
        df["has_rain"] = rain.where(known, pd.NA)

    if "light_condition" in df.columns:
        lighting = df["light_condition"].astype("string")
        known = lighting.notna() & lighting.str.strip().ne("")
        poor = lighting.str.contains(
            r"poor|dark|no light|unlit|inadequate|low visibility",
            case=False, na=False
        )
        df["has_poor_lighting"] = poor.where(known, pd.NA)

    if "road_surface_condition" in df.columns:
        surface = df["road_surface_condition"].astype("string")
        known = surface.notna() & surface.str.strip().ne("")
        wet = surface.str.contains(r"\bwet\b|slippery|waterlogged", case=False, na=False)
        df["has_wet_road"] = wet.where(known, pd.NA)

    # Count coordinates only when both values are within valid geographic ranges.
    if {"latitude", "longitude"}.issubset(df.columns):
        valid_coordinates = (
            df["latitude"].between(-90, 90)
            & df["longitude"].between(-180, 180)
        )
        valid_coordinate_count = int(valid_coordinates.sum())
    else:
        valid_coordinate_count = 0

    print("\nROAD ACCIDENT DATA PREPARATION")
    print("=" * 38)
    print(f"Records loaded: {original_count}")
    print(f"Duplicate records removed: {duplicates_removed}")
    print(f"Records after preparation: {len(df)}")
    print(f"Records with valid coordinates: {valid_coordinate_count}")
    print("Available columns:", ", ".join(map(str, df.columns)))
    print("=" * 38)

    return df


# ---------------------------------------------------------------------------
# PROMPT 2 — NumPy Statistical Analysis
# ---------------------------------------------------------------------------
def calculate_statistics(df):
    """Calculate descriptive statistics from available accident data."""
    stats = {}
    stats["total_accidents"] = int(len(df))

    # Count known severity categories only.
    if "accident_severity" in df.columns:
        stats["severity_distribution"] = (
            df["accident_severity"].dropna().astype(str).value_counts().to_dict()
        )
    else:
        stats["severity_distribution"] = {}

    # Calculate means from valid numeric observations only.
    for source_col, result_key in [
        ("casualty_count", "average_casualties"),
        ("vehicle_count", "average_vehicles"),
    ]:
        if source_col in df.columns:
            values = pd.to_numeric(df[source_col], errors="coerce").dropna()
            stats[result_key] = float(values.mean()) if not values.empty else None
        else:
            stats[result_key] = None

    # Find the hour with the most recorded accidents, if available.
    if "accident_hour" in df.columns:
        hours = pd.to_numeric(df["accident_hour"], errors="coerce").dropna()
        if not hours.empty:
            counts = hours.astype(int).value_counts()
            stats["peak_accident_hour"] = int(counts.idxmax())
        else:
            stats["peak_accident_hour"] = None
    else:
        stats["peak_accident_hour"] = None

    # Find the weekday with the most records, if available.
    if "day_of_week" in df.columns:
        days = df["day_of_week"].dropna().astype(str)
        stats["peak_accident_day"] = (
            str(days.value_counts().idxmax()) if not days.empty else None
        )
    else:
        stats["peak_accident_day"] = None

    # Percentages use only rows where the relevant indicator is known.
    percentage_fields = [
        ("is_night", "night_accident_percentage"),
        ("has_rain", "rain_accident_percentage"),
        ("has_poor_lighting", "poor_lighting_percentage"),
    ]
    for column, result_key in percentage_fields:
        if column in df.columns:
            values = df[column].dropna()
            if not values.empty:
                # Normalize common boolean-like strings if present.
                if values.dtype == object or str(values.dtype).startswith("string"):
                    normalized = values.astype(str).str.lower().str.strip()
                    true_values = normalized.isin(["true", "yes", "1"])
                    known_values = normalized.isin(
                        ["true", "false", "yes", "no", "1", "0"]
                    )
                    stats[result_key] = (
                        float(true_values[known_values].mean() * 100)
                        if known_values.any() else None
                    )
                else:
                    stats[result_key] = float(values.astype(bool).mean() * 100)
            else:
                stats[result_key] = None
        else:
            stats[result_key] = None

    # Build a NumPy correlation matrix from the available numeric fields.
    numeric_candidates = ["vehicle_count", "casualty_count", "accident_hour"]
    available = [col for col in numeric_candidates if col in df.columns]
    numeric_df = df[available].apply(pd.to_numeric, errors="coerce").dropna()

    if len(available) >= 2 and len(numeric_df) >= 2:
        matrix = np.corrcoef(numeric_df.to_numpy(), rowvar=False)
        # A constant column can produce NaN correlations; preserve that honestly.
        stats["correlation_matrix"] = {
            "columns": available,
            "matrix": matrix.tolist(),
        }
    elif len(available) == 1 and len(numeric_df) >= 2:
        stats["correlation_matrix"] = {
            "columns": available,
            "matrix": [[1.0]],
        }
    else:
        stats["correlation_matrix"] = None

    return stats


# ---------------------------------------------------------------------------
# PROMPT 3 — Matplotlib Static Charts
# ---------------------------------------------------------------------------
def generate_static_charts(df):
    """Save static accident-hour and severity charts; return success status."""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_saves_succeeded = True

    # Chart 1: number of recorded accidents for each hour.
    if "accident_hour" in df.columns:
        hours = pd.to_numeric(df["accident_hour"], errors="coerce").dropna()
        hours = hours[hours.between(0, 23)].astype(int)
        if not hours.empty:
            counts = hours.value_counts().reindex(range(24), fill_value=0).sort_index()
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(counts.index, counts.values)
            ax.set_title("Road Accident Distribution by Hour")
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Number of Recorded Accidents")
            ax.set_xticks(range(24))
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            try:
                fig.savefig(output_dir / "accidents_by_hour.png", dpi=150)
                print("Saved: output/accidents_by_hour.png")
            except OSError as exc:
                all_saves_succeeded = False
                print(f"Could not save accident-hour chart: {exc}")
            finally:
                plt.close(fig)
        else:
            print("Skipping hour chart: no valid accident-hour data.")
    else:
        print("Skipping hour chart: 'accident_hour' column is unavailable.")

    # Chart 2: pie chart of known accident-severity categories.
    if "accident_severity" in df.columns:
        severity = df["accident_severity"].dropna().astype(str)
        severity = severity[severity.str.strip().ne("")]
        counts = severity.value_counts()
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title("Road Accident Severity Distribution")
            ax.axis("equal")
            fig.tight_layout()
            try:
                fig.savefig(
                    output_dir / "accident_severity_distribution.png", dpi=150
                )
                print("Saved: output/accident_severity_distribution.png")
            except OSError as exc:
                all_saves_succeeded = False
                print(f"Could not save severity chart: {exc}")
            finally:
                plt.close(fig)
        else:
            print("Skipping severity chart: no known severity values.")
    else:
        print("Skipping severity chart: 'accident_severity' column is unavailable.")

    return all_saves_succeeded


# ---------------------------------------------------------------------------
# PROMPT 4 — Plotly Interactive Charts
# ---------------------------------------------------------------------------
def generate_interactive_charts(df):
    """Display an interactive accident-location map and weather-count chart."""
    # Chart 1: map only records with valid latitude and longitude.
    coordinate_columns = {"latitude", "longitude"}
    if coordinate_columns.issubset(df.columns):
        map_df = df.copy()
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
        map_df = map_df[
            map_df["latitude"].between(-90, 90)
            & map_df["longitude"].between(-180, 180)
        ].copy()

        if not map_df.empty:
            hover_candidates = [
                "road_name", "accident_date", "accident_hour",
                "number_of_casualties",
            ]
            hover_data = [col for col in hover_candidates if col in map_df.columns]
            color_column = (
                "accident_severity" if "accident_severity" in map_df.columns else None
            )

            try:
                fig = px.scatter_mapbox(
                    map_df,
                    lat="latitude",
                    lon="longitude",
                    color=color_column,
                    hover_name="road_name" if "road_name" in map_df.columns else None,
                    hover_data=hover_data,
                    zoom=10,
                    mapbox_style="open-street-map",
                    title="Interactive Road Accident Locations",
                )
                fig.show()
            except Exception as exc:
                print(f"Could not display the accident map: {exc}")
        else:
            print("Skipping accident map: no valid coordinates are available.")
    else:
        print("Skipping accident map: latitude/longitude columns are unavailable.")

    # Chart 2: count recorded cases by weather category. This is descriptive,
    # not evidence that weather caused the accidents.
    if "weather_condition" in df.columns:
        weather_df = df.dropna(subset=["weather_condition"]).copy()
        weather_df["weather_condition"] = (
            weather_df["weather_condition"].astype(str).str.strip()
        )
        weather_df = weather_df[weather_df["weather_condition"].ne("")]

        if not weather_df.empty:
            counts = (
                weather_df.groupby("weather_condition")
                .size()
                .reset_index(name="accident_count")
                .sort_values("accident_count", ascending=False)
            )
            fig = px.bar(
                counts,
                x="weather_condition",
                y="accident_count",
                color="weather_condition",
                title="Recorded Road Accidents by Weather Condition",
                labels={
                    "weather_condition": "Weather Condition",
                    "accident_count": "Number of Recorded Accidents",
                },
                hover_data={"accident_count": True},
            )
            fig.show()
        else:
            print("Skipping weather chart: no known weather values.")
    else:
        print("Skipping weather chart: 'weather_condition' column is unavailable.")


# ---------------------------------------------------------------------------
# PROMPT 5 — Summary Table and Main Block
# ---------------------------------------------------------------------------
def print_summary(stats):
    """Print a readable summary using calculated statistics only."""
    def display_value(value, decimals=2, suffix=""):
        if value is None:
            return "Not available"
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "Not available"
            return f"{value:.{decimals}f}{suffix}"
        return f"{value}{suffix}"

    line = "=" * 64
    print("\n" + line)
    print("SAFEROUTE AI")
    print("ROAD ACCIDENT DATA ANALYSIS SUMMARY")
    print(line)
    print(f"{'Total Recorded Accidents':30}: {display_value(stats.get('total_accidents'), 0)}")
    print(f"{'Average Casualties':30}: {display_value(stats.get('average_casualties'))}")
    print(f"{'Average Vehicles':30}: {display_value(stats.get('average_vehicles'))}")
    print(f"{'Peak Accident Hour':30}: {display_value(stats.get('peak_accident_hour'), 0)}")
    print(f"{'Peak Accident Day':30}: {display_value(stats.get('peak_accident_day'))}")
    print(
        f"{'Night Accident Percentage':30}: "
        f"{display_value(stats.get('night_accident_percentage'), suffix='%')}"
    )
    print(
        f"{'Rain Accident Percentage':30}: "
        f"{display_value(stats.get('rain_accident_percentage'), suffix='%')}"
    )
    print(
        f"{'Poor Lighting Percentage':30}: "
        f"{display_value(stats.get('poor_lighting_percentage'), suffix='%')}"
    )

    print("\nAccident Severity Distribution:")
    distribution = stats.get("severity_distribution") or {}
    if distribution:
        for severity, count in distribution.items():
            print(f"  {severity}: {count}")
    else:
        print("  Not available")

    correlation = stats.get("correlation_matrix")
    if correlation:
        print("\nCorrelation Matrix (columns:", ", ".join(correlation["columns"]) + ")")
        print(np.array2string(np.asarray(correlation["matrix"]), precision=3))
    else:
        print("\nCorrelation Matrix: Not available")

    print(line)


if __name__ == "__main__":
    # Run the complete historical-data analysis in the requested order.
    try:
        df = load_and_prepare_data("data/road_accidents.csv")
        stats = calculate_statistics(df)

        # Static chart function returns False if any chart save failed.
        charts_saved = generate_static_charts(df)

        # Interactive charts require a graphical/browser display environment.
        try:
            generate_interactive_charts(df)
        except Exception as exc:
            print(f"Interactive chart generation failed: {exc}")

        print_summary(stats)

        if charts_saved:
            print("Analysis complete. Charts saved to output/ folder")
        else:
            print(
                "Analysis finished, but one or more static charts could not be saved. "
                "Check the messages above."
            )

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except (pd.errors.ParserError, ValueError) as exc:
        print(f"ERROR: Could not process the dataset: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected analysis error: {exc}")
        sys.exit(1)
