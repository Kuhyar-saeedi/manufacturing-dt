"""
Manufacturing Plant Digital Twin - FastAPI Backend
Handles sensor data ingestion, ML predictions, and optimization
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import joblib
from typing import List, Dict

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

# ============================================================================
# ML MODELS (Stubs - you'll train these later)
# ============================================================================

class MaintenancePredictor:
    """Predict equipment failure probability"""
    
    def __init__(self):
        # Placeholder: in production, load a trained XGBoost/RandomForest
        self.model = None
    
    def predict_failure_probability(self, 
                                   temperature: float,
                                   vibration: float,
                                   power_consumption: float) -> tuple:
        """
        Returns: (failure_probability: 0-1, days_to_failure: float)
        """
        # Simple heuristic for now (replace with ML model)
        # High vibration + high temp = higher failure risk
        risk_score = (vibration / 10.0) * 0.5 + (temperature / 80.0) * 0.5
        failure_prob = min(risk_score, 1.0)
        
        # Estimate days to failure (rough inverse relationship)
        if failure_prob > 0.7:
            days = np.random.uniform(1, 3)
        elif failure_prob > 0.4:
            days = np.random.uniform(3, 7)
        else:
            days = np.random.uniform(7, 30)
        
        return failure_prob, days

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
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Manufacturing Digital Twin API",
    description="Real-time monitoring, predictive maintenance, and optimization"
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
    """Ingest sensor data from factory equipment"""
    
    # Store in database
    db_reading = SensorReading(
        machine_id=reading.machine_id,
        timestamp=datetime.now(),
        temperature=reading.temperature,
        vibration=reading.vibration,
        power_consumption=reading.power_consumption,
        production_count=reading.production_count,
        downtime_minutes=reading.downtime_minutes,
        quality_score=reading.quality_score
    )
    db.add(db_reading)
    
    # Predict maintenance if threshold exceeded
    failure_prob, days_to_failure = maintenance_predictor.predict_failure_probability(
        reading.temperature, reading.vibration, reading.power_consumption
    )
    
    if failure_prob > 0.6:  # Alert threshold
        alert = MaintenanceAlert(
            machine_id=reading.machine_id,
            timestamp=datetime.now(),
            failure_probability=failure_prob,
            days_to_failure=days_to_failure,
            recommended_action="Schedule preventive maintenance"
        )
        db.add(alert)
    
    db.commit()
    db.refresh(db_reading)
    return db_reading

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

@app.get("/maintenance-alerts", response_model=List[MaintenanceAlertOut])
async def get_maintenance_alerts(db: Session = Depends(get_db)):
    """Get all active maintenance alerts"""
    
    alerts = db.query(MaintenanceAlert).filter(MaintenanceAlert.is_active == 1).all()
    return alerts

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

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database and load models on startup"""
    print("✅ Manufacturing DT API starting up...")
    print("📊 Database: SQLite (factory.db)")
    print("🤖 ML models: Ready for predictions")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
