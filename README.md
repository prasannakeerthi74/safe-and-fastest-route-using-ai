# safe-and-fastest-route-using-ai
# 🛣️ SafeRoute AI
### ML-Based Accident Risk Prediction and Safe & Fastest Route Recommendation System

## 📌 Project Overview
SafeRoute AI is a road safety and route recommendation system that analyzes historical road accident data to identify potentially high-risk locations. It compares candidate routes using estimated travel time, traffic conditions, weather conditions, and risk scores to help users make informed travel decisions.

## 🎯 Objectives
- Analyze historical road accident data.
- Identify potentially high-risk locations.
- Estimate route risk using available inputs.
- Compare candidate routes based on safety and travel time.
- Visualize accident statistics through an interactive dashboard.

## ✨ Features
- Accident data analysis and summary.
- High-risk accident record identification.
- Interactive road accident dashboard.
- Candidate route risk assessment.
- Safe, fast, and balanced route comparison.
- REST API using FastAPI.
- Data visualization using Plotly.

## 🛠️ Technologies Used
- **Programming Language:** Python
- **Backend:** FastAPI
- **Frontend Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Server:** Uvicorn

## 📁 Project Structure
```text
SafeRouteAI/
├── main.py
├── app.py
├── analysis.py
├── requirements.txt
└── data/
    ├── road-accidents-2024-states-road-accidents.csv
    └── traffic_data.csv
```

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone <your-repository-url>
cd SafeRouteAI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Prepare the dataset**

Place your historical accident CSV file inside the `data/` folder. Update the filename or dataset path in `main.py` and `app.py` if necessary.

**4. Run the FastAPI backend**
```bash
python main.py
```

Open the API documentation at:
http://127.0.0.1:8000/docs

**5. Run the Streamlit dashboard**

Open another terminal in the project folder and run:
```bash
streamlit run app.py
```

## 🔗 API Endpoints
- `GET /` — Check API status.
- `GET /summary` — View accident statistics.
- `GET /high-risk-segments` — Retrieve high-risk records.
- `GET /accidents` — Retrieve accident records.
- `POST /predict-route-risk` — Assess candidate route risk.

## 🚀 Future Enhancements
- Integration with live traffic APIs.
- Automatic traffic updates every 10 minutes.
- Map-based route generation.
- Trained machine learning model for accident risk prediction.
- Real-time weather integration.
- Interactive map displaying accident hotspots.

## ⚠️ Current Limitations
The current implementation analyzes historical accident data and provides preliminary, rule-based route risk estimates. Automatic route generation, live traffic integration, and trained machine learning predictions require additional development.

## 👩‍💻 Conclusion
SafeRoute AI aims to support safer and more efficient travel by combining accident data analysis, route risk assessment, and travel-time comparison. It demonstrates how data processing, APIs, and interactive dashboards can contribute to intelligent road safety solutions.
