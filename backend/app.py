"""
Manufacturing Plant Digital Twin - FastAPI Backend
Handles sensor data ingestion, ML predictions, and optimization
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import asyncio
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import joblib
from typing import List, Dict
from model_utils import compute_features
from opcua_server import ManufacturingOPCServer
from opcua_client import run_opcua_bridge, opcua_status

# ============================================================================
# DATABASE SETUP
# ============================================================================

DATABASE_URL = "sqlite:///./factory.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============================================================================
# DATABASE MODELS
# ============================================================================

class SensorReading(Base):
    """Store raw sensor data from factory machines"""
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)  # M1, M2, ..., M5
    timestamp = Column(DateTime, index=True)
    temperature = Column(Float)              # Celsius
    vibration = Column(Float)                # mm/s
    power_consumption = Column(Float)        # kW
    production_count = Column(Integer)       # Parts produced
    downtime_minutes = Column(Float)         # Equipment downtime
    quality_score = Column(Float)            # 0-100, defect rate

class MaintenanceAlert(Base):
    """Store predicted maintenance alerts"""
    __tablename__ = "maintenance_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    timestamp = Column(DateTime)
    failure_probability = Column(Float)      # 0-1, likelihood of failure
    days_to_failure = Column(Float)          # Predicted days until failure
    recommended_action = Column(String)      # e.g., "Schedule maintenance"
    is_active = Column(Integer, default=1)

class ProductionJob(Base):
    """Store production jobs and scheduling"""
    __tablename__ = "production_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    machine_id = Column(String, index=True)
    product_type = Column(String)            # e.g., "Part_A", "Part_B"
    scheduled_start = Column(DateTime)
    scheduled_end = Column(DateTime)
    changeover_time = Column(Float)          # Minutes
    quantity = Column(Integer)

# Create tables
Base.metadata.create_all(bind=engine)

# ============================================================================
# DEPENDENCY FUNCTION
# ============================================================================

def get_db():
    """Dependency: Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# PYDANTIC SCHEMAS (API Input/Output)
# ============================================================================

from pydantic import BaseModel

class SensorReadingIn(BaseModel):
    machine_id: str
    temperature: float
    vibration: float
    power_consumption: float
    production_count: int
    downtime_minutes: float
    quality_score: float

class SensorReadingOut(SensorReadingIn):
    id: int
    timestamp: datetime
    class Config:
        from_attributes = True

class MaintenanceAlertOut(BaseModel):
    machine_id: str
    failure_probability: float
    days_to_failure: float
    recommended_action: str
    is_active: int
    class Config:
        from_attributes = True

class FactoryStatusOut(BaseModel):
    total_machines: int
    total_production_today: int
    average_oee: float                       # Overall Equipment Effectiveness
    active_alerts: int
    predicted_downtime_cost: float           # USD
    timestamp: datetime

class SchedulingRecommendationOut(BaseModel):
    optimized_job_order: List[str]           # Reordered job IDs
    estimated_time_saved: float              # Minutes
    changeover_reduction: float              # Percentage

class MachineRiskOut(BaseModel):
    machine_id: str
    xgb_failure_probability: float           # XGBoost supervised score 0-1
    if_anomaly_score: float                  # Isolation Forest unsupervised score 0-1
    combined_risk: float                     # max(xgb, if) — conservative union
    temperature: float
    vibration: float
    power_consumption: float

class OPCNodeInfo(BaseModel):
    machine_id: str
    signal: str
    node_id: str
    value: float
    last_updated: str

class OPCStatusOut(BaseModel):
    connected: bool
    endpoint: str
    server_time: str | None
    nodes: List[OPCNodeInfo]
    readings_received: int

# ============================================================================
# ML MODELS (Stubs - you'll train these later)
# ============================================================================

class MaintenancePredictor:
    """Predict equipment failure probability using a trained XGBoost model."""

    def __init__(self):
        try:
            self.model = joblib.load("maintenance_model.joblib")
            print("✅ XGBoost maintenance model loaded")
        except FileNotFoundError:
            self.model = None
            print("⚠️  maintenance_model.joblib not found — using heuristic fallback")

    def predict_failure_probability(
        self,
        machine_id: str,
        temperature: float,
        vibration: float,
        power_consumption: float,
        db=None,
    ) -> tuple:
        """Returns (failure_probability: 0-1, days_to_failure: float)."""

        if self.model is not None and db is not None:
            recent_rows = (
                db.query(SensorReading)
                .filter(SensorReading.machine_id == machine_id)
                .order_by(SensorReading.timestamp.desc())
                .limit(5)
                .all()
            )
            # reverse so oldest-first for the rolling window
            recent = [(r.temperature, r.vibration, r.power_consumption)
                      for r in reversed(recent_rows)]
            feats = compute_features(machine_id, temperature, vibration, power_consumption, recent)
            failure_prob = float(self.model.predict_proba(feats)[0, 1])
        else:
            risk_score = (vibration / 10.0) * 0.5 + (temperature / 80.0) * 0.5
            failure_prob = min(risk_score, 1.0)

        if failure_prob > 0.8:
            days = np.random.uniform(1, 2)
        elif failure_prob > 0.6:
            days = np.random.uniform(2, 5)
        elif failure_prob > 0.4:
            days = np.random.uniform(5, 10)
        else:
            days = np.random.uniform(10, 30)

        return failure_prob, days

class AnomalyDetector:
    """
    Isolation Forest anomaly detector — unsupervised, trained on normal data only.
    Returns a score in [0, 1] where 1.0 = highly anomalous.
    """

    def __init__(self):
        try:
            artifact = joblib.load("anomaly_model.joblib")
            self.model      = artifact["model"]
            self.score_min  = artifact["score_min"]
            self.score_max  = artifact["score_max"]
            print("✅ Isolation Forest anomaly model loaded")
        except FileNotFoundError:
            self.model = None
            print("⚠️  anomaly_model.joblib not found — anomaly scoring disabled")

    def score(
        self,
        machine_id: str,
        temp: float,
        vib: float,
        power: float,
        recent: list,
    ) -> float:
        """Returns anomaly score in [0, 1]; higher means more anomalous."""
        if self.model is None:
            return 0.0
        feats = compute_features(machine_id, temp, vib, power, recent)
        raw = float(self.model.score_samples(feats)[0])
        spread = self.score_max - self.score_min
        if spread == 0:
            return 0.0
        # Lower raw score → more anomalous → higher output score
        normalized = (self.score_max - raw) / spread
        return float(np.clip(normalized, 0.0, 1.0))


class ProductionScheduler:
    """Optimize job sequence to minimize changeovers"""
    
    def __init__(self):
        # Changeover time matrix (minutes): Part_A → Part_B, etc.
        self.changeover_matrix = {
            ("Part_A", "Part_B"): 15,
            ("Part_B", "Part_A"): 15,
            ("Part_A", "Part_C"): 25,
            ("Part_C", "Part_A"): 25,
            ("Part_B", "Part_C"): 20,
            ("Part_C", "Part_B"): 20,
        }
    
    def optimize_job_order(self, jobs: List[Dict]) -> tuple:
        """
        Simple greedy scheduler: pick next job with minimal changeover
        Returns: (optimized_job_order, time_saved_minutes)
        """
        # Placeholder: implement nearest-neighbor or simulated annealing later
        optimized_order = sorted([j['id'] for j in jobs])
        time_saved = np.random.uniform(10, 120)
        return optimized_order, time_saved

# ============================================================================
# SHARED SENSOR INGEST (called by OPC-UA bridge and REST endpoint)
# ============================================================================

def ingest_sensor_data(
    db: Session,
    machine_id: str,
    temperature: float,
    vibration: float,
    power_consumption: float,
    production_count: int,
    downtime_minutes: float,
    quality_score: float,
) -> SensorReading:
    """Write one sensor reading to the DB, create a maintenance alert if needed,
    and prune rows older than 72 hours. Synchronous — safe to call from a thread."""
    now = datetime.now()
    reading = SensorReading(
        machine_id=machine_id,
        timestamp=now,
        temperature=round(temperature, 2),
        vibration=round(vibration, 3),
        power_consumption=round(power_consumption, 2),
        production_count=int(production_count),
        downtime_minutes=float(downtime_minutes),
        quality_score=round(quality_score, 2),
    )
    db.add(reading)

    prob, days = maintenance_predictor.predict_failure_probability(
        machine_id, temperature, vibration, power_consumption, db
    )
    if prob > 0.6:
        db.add(MaintenanceAlert(
            machine_id=machine_id,
            timestamp=now,
            failure_probability=prob,
            days_to_failure=days,
            recommended_action="Schedule preventive maintenance",
        ))

    db.commit()
    db.refresh(reading)

    cutoff = now - timedelta(hours=72)
    db.query(SensorReading).filter(SensorReading.timestamp < cutoff).delete()
    db.commit()

    return reading


def seed_database():
    """Seed 48 h of synthetic data if no recent readings exist."""
    db = SessionLocal()
    try:
        cutoff = datetime.now() - timedelta(hours=24)
        recent = db.query(SensorReading).filter(SensorReading.timestamp >= cutoff).count()
        if recent == 0:
            print("📊 No recent data — seeding 48 h of synthetic readings...")
            # Clear stale rows first
            db.query(SensorReading).delete()
            db.commit()
            from sensor_simulator import FactorySensorSimulator
            simulator = FactorySensorSimulator(n_machines=5, n_hours=48)
            df = simulator.generate_readings()
            df = simulator.add_anomalies(df)
            df.to_sql("sensor_readings", engine, if_exists="append", index=False)
            print(f"✅ Seeded {len(df)} readings")
        else:
            print(f"📊 Database ready: {recent} recent readings")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Manufacturing DT API starting up...")
    seed_database()
    print("🤖 ML models: Ready")

    opc_server = ManufacturingOPCServer()
    await opc_server.init()
    await opc_server.start()

    server_task = asyncio.create_task(opc_server.update_loop())
    bridge_task = asyncio.create_task(
        run_opcua_bridge(ingest_sensor_data, SessionLocal)
    )

    yield

    server_task.cancel()
    bridge_task.cancel()
    await opc_server.stop()


# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Manufacturing Digital Twin API",
    description="Real-time monitoring, predictive maintenance, and optimization",
    lifespan=lifespan,
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML models
maintenance_predictor = MaintenancePredictor()
anomaly_detector = AnomalyDetector()
scheduler = ProductionScheduler()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "timestamp": datetime.now()}

@app.post("/sensor-reading", response_model=SensorReadingOut)
async def ingest_sensor_reading(reading: SensorReadingIn, db: Session = Depends(get_db)):
    """Ingest sensor data from factory equipment (REST fallback; primary path is OPC-UA)"""
    return ingest_sensor_data(
        db=db,
        machine_id=reading.machine_id,
        temperature=reading.temperature,
        vibration=reading.vibration,
        power_consumption=reading.power_consumption,
        production_count=reading.production_count,
        downtime_minutes=reading.downtime_minutes,
        quality_score=reading.quality_score,
    )

@app.get("/opcua-status", response_model=OPCStatusOut)
async def get_opcua_status():
    """OPC-UA bridge status: connection state, node IDs, and live values."""
    return OPCStatusOut(**opcua_status)

@app.get("/factory-status", response_model=FactoryStatusOut)
async def get_factory_status(db: Session = Depends(get_db)):
    """Get real-time factory status and KPIs"""
    
    # Get latest readings from last 24 hours
    cutoff = datetime.now() - timedelta(hours=24)
    readings = db.query(SensorReading).filter(SensorReading.timestamp >= cutoff).all()
    
    if not readings:
        raise HTTPException(status_code=404, detail="No recent data available")
    
    df = pd.DataFrame([
        {
            'machine_id': r.machine_id,
            'production_count': r.production_count,
            'downtime_minutes': r.downtime_minutes,
            'quality_score': r.quality_score
        }
        for r in readings
    ])
    
    # Calculate OEE (Overall Equipment Effectiveness)
    # OEE = (Availability) × (Performance) × (Quality)
    total_time = 24 * 60  # minutes
    availability = 1 - (df['downtime_minutes'].sum() / (total_time * len(df['machine_id'].unique())))
    performance = df['production_count'].sum() / (len(df['machine_id'].unique()) * 100)  # Assume 100 parts/machine/day nominal
    quality = df['quality_score'].mean() / 100
    oee = availability * performance * quality
    
    # Count active maintenance alerts
    active_alerts = db.query(MaintenanceAlert).filter(MaintenanceAlert.is_active == 1).count()
    
    # Estimate downtime cost (rough: $50/minute per machine)
    downtime_cost = df['downtime_minutes'].sum() * 50
    
    return FactoryStatusOut(
        total_machines=len(df['machine_id'].unique()),
        total_production_today=int(df['production_count'].sum()),
        average_oee=float(oee),
        active_alerts=active_alerts,
        predicted_downtime_cost=downtime_cost,
        timestamp=datetime.now()
    )

@app.get("/sensor-readings", response_model=List[SensorReadingOut])
async def get_sensor_readings(hours: int = 24, machine_id: str = None, db: Session = Depends(get_db)):
    """Get raw sensor readings for the last N hours, optionally filtered by machine"""
    cutoff = datetime.now() - timedelta(hours=hours)
    query = db.query(SensorReading).filter(SensorReading.timestamp >= cutoff)
    if machine_id:
        query = query.filter(SensorReading.machine_id == machine_id)
    return query.order_by(SensorReading.timestamp.desc()).limit(1000).all()

@app.get("/maintenance-alerts", response_model=List[MaintenanceAlertOut])
async def get_maintenance_alerts(db: Session = Depends(get_db)):
    """Get all active maintenance alerts"""

    alerts = db.query(MaintenanceAlert).filter(MaintenanceAlert.is_active == 1).all()
    return alerts

@app.get("/risk-analysis", response_model=List[MachineRiskOut])
async def get_risk_analysis(db: Session = Depends(get_db)):
    """
    Per-machine dual-signal risk: XGBoost (supervised) + Isolation Forest (unsupervised).
    Uses the last 6 sensor readings per machine for rolling-window features.
    """
    machine_ids = ["M1", "M2", "M3", "M4", "M5"]
    results = []

    for machine_id in machine_ids:
        rows = (
            db.query(SensorReading)
            .filter(SensorReading.machine_id == machine_id)
            .order_by(SensorReading.timestamp.desc())
            .limit(6)
            .all()
        )
        if not rows:
            continue

        # Most recent reading
        latest = rows[0]
        # Older readings for rolling window (oldest-first)
        recent = [(r.temperature, r.vibration, r.power_consumption)
                  for r in reversed(rows[1:])]

        xgb_prob, _ = maintenance_predictor.predict_failure_probability(
            machine_id, latest.temperature, latest.vibration,
            latest.power_consumption, db,
        )
        if_score = anomaly_detector.score(
            machine_id, latest.temperature, latest.vibration,
            latest.power_consumption, recent,
        )

        results.append(MachineRiskOut(
            machine_id=machine_id,
            xgb_failure_probability=round(xgb_prob, 4),
            if_anomaly_score=round(if_score, 4),
            combined_risk=round(max(xgb_prob, if_score), 4),
            temperature=round(latest.temperature, 2),
            vibration=round(latest.vibration, 3),
            power_consumption=round(latest.power_consumption, 2),
        ))

    return results


@app.post("/optimize-schedule", response_model=SchedulingRecommendationOut)
async def optimize_production_schedule(job_ids: List[str], db: Session = Depends(get_db)):
    """Get optimized job schedule to minimize changeovers"""
    
    # Fetch jobs from database
    jobs = db.query(ProductionJob).filter(ProductionJob.job_id.in_(job_ids)).all()
    
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found")
    
    job_dicts = [{'id': j.job_id, 'product_type': j.product_type} for j in jobs]
    optimized_order, time_saved = scheduler.optimize_job_order(job_dicts)
    
    return SchedulingRecommendationOut(
        optimized_job_order=optimized_order,
        estimated_time_saved=time_saved,
        changeover_reduction=5.2  # Placeholder
    )

@app.get("/")
async def root():
    """Welcome message"""
    return {
        "message": "Manufacturing Plant Digital Twin API",
        "docs": "/docs",
        "version": "0.1.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
