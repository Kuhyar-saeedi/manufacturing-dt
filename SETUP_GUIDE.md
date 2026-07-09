# Manufacturing Digital Twin - Quick Setup Guide

## Project Structure

```
manufacturing-dt/
├── backend/
│   ├── app.py                 (FastAPI application)
│   ├── sensor_simulator.py    (Generate synthetic data)
│   └── requirements.txt
├── frontend/
│   ├── app.py                 (Main Streamlit dashboard)
│   ├── pages/
│   │   ├── 01_Dashboard.py
│   │   ├── 02_Predictions.py
│   │   ├── 03_Scheduler.py
│   │   └── 04_Reports.py
│   └── requirements.txt
└── README.md
```

---

## 3-Minute Quick Start

### Terminal 1: Setup & Generate Data
```bash
# Clone repo
git clone <repo-url>
cd manufacturing-dt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install backend + generate data
cd backend
pip install -r requirements.txt
python sensor_simulator.py
```

### Terminal 2: Start Backend (FastAPI)
```bash
cd manufacturing-dt/backend
python app.py
# ✅ API running at http://localhost:8000
# 📖 Docs at http://localhost:8000/docs
```

### Terminal 3: Start Frontend (Streamlit)
```bash
cd manufacturing-dt/frontend
pip install -r requirements.txt
streamlit run app.py
# ✅ Dashboard at http://localhost:8501
```

---

## What Happens Next

1. **FastAPI starts** → Loads SQLite database with synthetic sensor data
2. **Streamlit starts** → Loads data from SQLite, displays real-time dashboards
3. **You navigate to** `http://localhost:8501` → See your factory dashboard

---

## Project Files Explained

### Backend Files

**`app.py`** - FastAPI microservice
- Handles `/sensor-reading` ingestion (POST)
- Computes `/factory-status` KPIs (GET)
- Returns `/maintenance-alerts` (GET)
- Provides `/optimize-schedule` recommendations (POST)

**`sensor_simulator.py`** - Synthetic data generator
- Creates 24 hours of data for 5 machines
- Adds realistic anomalies (bearing wear, thermal issues)
- Saves to `factory.db` (SQLite)
- Run once to populate the database

**`requirements.txt`**
```
fastapi
uvicorn
sqlalchemy
pydantic
numpy
pandas
scikit-learn
xgboost
```

### Frontend Files

**`app.py`** - Main Streamlit dashboard
- Real-time KPI metrics (production, quality, downtime, utilization)
- 4 interactive Plotly charts (temperature, vibration, power, quality)
- Sensor data table with filtering
- Sidebar navigation to other pages

**`pages/01_Dashboard.py`** - Full factory floor overview
**`pages/02_Predictions.py`** - Maintenance alerts & failure predictions
**`pages/03_Scheduler.py`** - Job sequencing optimizer *(to build)*
**`pages/04_Reports.py`** - Historical analytics & PDF export *(to build)*

---

## Key Features to Explore

### 1. Real-Time Dashboard
- Filter by machine (M1–M5)
- Select time range (1h, 6h, 24h)
- See live temperature, vibration, power trends
- Check quality scores by machine

### 2. Maintenance Predictions
- Risk scores for each machine (0–100%)
- Specific failure warnings with estimated days to failure
- Cost-benefit analysis: planned vs unplanned downtime
- Recommended maintenance actions

### 3. Data Integrity
- All data persisted in SQLite (`factory.db`)
- Multi-machine data aggregation
- Historical trend analysis (24 hours available)

---

## Common Issues & Fixes

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
cd frontend
pip install -r requirements.txt
```

### "Cannot connect to http://localhost:8000"
- Make sure backend is running: `python app.py` in `backend/` folder
- Check if port 8000 is in use: `lsof -i :8000` (macOS/Linux)

### "Database is empty" or "No data"
```bash
cd backend
python sensor_simulator.py  # Regenerate synthetic data
```

### "Port 8501 already in use"
```bash
streamlit run app.py --server.port 8502
```

---

## Next Steps to Enhance the Project

1. **Train ML models** (Phase 2)
   - Use synthetic data to train XGBoost models for failure prediction
   - Replace heuristic scoring with real ML

2. **Build scheduler** (Phase 3)
   - Implement job sequencing optimizer
   - Add Gantt chart visualization

3. **Deploy to cloud** (Phase 4)
   - Push to GitHub
   - Deploy backend to Railway.app
   - Deploy frontend to Streamlit Cloud

4. **Add real data** (Phase 5)
   - Connect to actual SCADA systems (OPC-UA)
   - Replace synthetic data with live sensor streams

---

## File Paths Reference

| File | Purpose | Location |
|------|---------|----------|
| `factory.db` | SQLite database | `backend/` |
| `synthetic_data.csv` | CSV export of data | `backend/` |
| `app.py` (FastAPI) | Backend API | `backend/app.py` |
| `app.py` (Streamlit) | Dashboard UI | `frontend/app.py` |
| `sensor_simulator.py` | Data generator | `backend/sensor_simulator.py` |

---

## Commands Quick Reference

```bash
# Generate synthetic data
python backend/sensor_simulator.py

# Start FastAPI backend
cd backend && python app.py

# Start Streamlit frontend
cd frontend && streamlit run app.py

# View API documentation
# Open browser: http://localhost:8000/docs

# View Streamlit app
# Open browser: http://localhost:8501
```

---

## Video Demo Script (If Recording)

1. **Intro** (30s)
   - "This is a Manufacturing Plant Digital Twin"
   - "Shows real-time factory monitoring + predictive maintenance"

2. **Dashboard Demo** (1m)
   - Show KPI cards (production, quality, downtime)
   - Hover over charts to show details
   - Filter by machine, show different time ranges

3. **Maintenance Alerts** (1m)
   - Show risk scores for each machine
   - Explain what causes high risk (vibration, temperature)
   - Show cost savings calculation

4. **Architecture Explanation** (30s)
   - FastAPI backend ingests data
   - Streamlit visualizes it
   - SQLite persists everything

5. **Outro** (20s)
   - "Built with Python, FastAPI, Streamlit"
   - Link to GitHub repo

---

**Ready to go!** 🚀 Start with Terminal 1 setup, then run Terminal 2 & 3.
