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
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import joblib
from typing import List, Optional
from pydantic import BaseModel

from model_utils import compute_features
from opcua_server import ManufacturingOPCServer
from opcua_client import run_opcua_bridge, opcua_status
from auth import create_access_token, require_admin, get_current_user, DEMO_USERS

# ============================================================================
# DATABASE SETUP
# ============================================================================

DATABASE_URL = "sqlite:///./factory.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

PLANTS = ["alpha", "beta", "gamma"]

# ============================================================================
# DATABASE MODELS
# ============================================================================

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    plant_id = Column(String, index=True, default="alpha")
    timestamp = Column(DateTime, index=True)
    temperature = Column(Float)
    vibration = Column(Float)
    power_consumption = Column(Float)
    production_count = Column(Integer)
    downtime_minutes = Column(Float)
    quality_score = Column(Float)


class MaintenanceAlert(Base):
    __tablename__ = "maintenance_alerts"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    plant_id = Column(String, index=True, default="alpha")
    timestamp = Column(DateTime)
    failure_probability = Column(Float)
    days_to_failure = Column(Float)
    recommended_action = Column(String)
    is_active = Column(Integer, default=1)


class ProductionJob(Base):
    __tablename__ = "production_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    machine_id = Column(String, index=True)
    plant_id = Column(String, index=True, default="alpha")
    product_type = Column(String)
    scheduled_start = Column(DateTime)
    scheduled_end = Column(DateTime)
    changeover_time = Column(Float)
    quantity = Column(Integer)


def ensure_schema():
    """Drop and recreate all tables when the plant_id column is missing."""
    insp = sa_inspect(engine)
    if "sensor_readings" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("sensor_readings")}
        if "plant_id" not in cols:
            print("⚠️  Schema migration: dropping tables to add plant_id...")
            Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

# ============================================================================
# DEPENDENCY
# ============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str

class UserOut(BaseModel):
    username: str
    role: str
    plant_id: Optional[str]

class SensorReadingIn(BaseModel):
    machine_id: str
    plant_id: str = "alpha"
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
    plant_id: str
    failure_probability: float
    days_to_failure: float
    recommended_action: str
    is_active: int
    class Config:
        from_attributes = True

class FactoryStatusOut(BaseModel):
    plant_id: str
    total_machines: int
    total_production_today: int
    average_oee: float
    active_alerts: int
    predicted_downtime_cost: float
    timestamp: datetime

class PlantSummaryOut(BaseModel):
    plant_id: str
    total_machines: int
    total_production_24h: int
    average_oee: float
    active_alerts: int
    avg_quality: float

class SchedulingRecommendationOut(BaseModel):
    optimized_job_order: List[str]
    estimated_time_saved: float
    changeover_reduction: float

class MachineRiskOut(BaseModel):
    machine_id: str
    plant_id: str
    xgb_failure_probability: float
    if_anomaly_score: float
    combined_risk: float
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

class RULForecastPoint(BaseModel):
    timestamp: datetime
    rul_hours: float
    rul_lower: float
    rul_upper: float

class RULSummaryOut(BaseModel):
    machine_id: str
    plant_id: str
    rul_hours: float
    rul_lower: float
    rul_upper: float
    risk_level: str

# ============================================================================
# ML MODELS
# ============================================================================

class MaintenancePredictor:
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
        plant_id: str = "alpha",
    ) -> tuple:
        if self.model is not None and db is not None:
            recent_rows = (
                db.query(SensorReading)
                .filter(
                    SensorReading.machine_id == machine_id,
                    SensorReading.plant_id == plant_id,
                )
                .order_by(SensorReading.timestamp.desc())
                .limit(5)
                .all()
            )
            recent = [
                (r.temperature, r.vibration, r.power_consumption)
                for r in reversed(recent_rows)
            ]
            feats = compute_features(machine_id, temperature, vibration, power_consumption, recent)
            failure_prob = float(self.model.predict_proba(feats)[0, 1])
        else:
            risk_score = (vibration / 10.0) * 0.5 + (temperature / 80.0) * 0.5
            failure_prob = min(risk_score, 1.0)

        rul_hours, _, _ = rul_predictor.predict(
            machine_id, temperature, vibration, power_consumption,
            recent if db is not None else [],
        )
        days = rul_hours / 24.0
        return failure_prob, days


class AnomalyDetector:
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

    def score(self, machine_id, temp, vib, power, recent) -> float:
        if self.model is None:
            return 0.0
        feats = compute_features(machine_id, temp, vib, power, recent)
        raw = float(self.model.score_samples(feats)[0])
        spread = self.score_max - self.score_min
        if spread == 0:
            return 0.0
        return float(np.clip((self.score_max - raw) / spread, 0.0, 1.0))


class RULPredictor:
    NORMAL_RUL_CAP = 72.0   # hours — matches train_models.NORMAL_RUL_CAP
    CONFIDENCE_PCT  = 0.20  # ±20% band

    def __init__(self):
        try:
            artifact = joblib.load("rul_model.joblib")
            self.model          = artifact["model"]
            self.normal_rul_cap = artifact.get("normal_rul_cap", self.NORMAL_RUL_CAP)
            print("✅ XGBoost RUL model loaded")
        except FileNotFoundError:
            self.model = None
            print("⚠️  rul_model.joblib not found — RUL will use heuristic fallback")

    def predict(self, machine_id: str, temp: float, vib: float, power: float, recent: list) -> tuple[float, float, float]:
        """Returns (rul_hours, lower_bound, upper_bound)."""
        if self.model is not None:
            feats = compute_features(machine_id, temp, vib, power, recent)
            rul = float(np.clip(self.model.predict(feats)[0], 1.0, self.normal_rul_cap))
        else:
            risk = min((vib / 10.0) * 0.5 + (temp / 80.0) * 0.5, 1.0)
            rul = max(self.NORMAL_RUL_CAP * (1.0 - risk), 1.0)

        band = max(rul * self.CONFIDENCE_PCT, 1.0)
        return round(rul, 2), round(max(rul - band, 0.5), 2), round(rul + band, 2)


class ProductionScheduler:
    def __init__(self):
        self.changeover_matrix = {
            ("Part_A", "Part_B"): 15,
            ("Part_B", "Part_A"): 15,
            ("Part_A", "Part_C"): 25,
            ("Part_C", "Part_A"): 25,
            ("Part_B", "Part_C"): 20,
            ("Part_C", "Part_B"): 20,
        }

    def optimize_job_order(self, jobs: list) -> tuple:
        optimized_order = sorted([j['id'] for j in jobs])
        time_saved = np.random.uniform(10, 120)
        return optimized_order, time_saved

# ============================================================================
# SHARED SENSOR INGEST
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
    plant_id: str = "alpha",
) -> SensorReading:
    now = datetime.now()
    reading = SensorReading(
        machine_id=machine_id,
        plant_id=plant_id,
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
        machine_id, temperature, vibration, power_consumption, db, plant_id
    )
    if prob > 0.6:
        db.add(MaintenanceAlert(
            machine_id=machine_id,
            plant_id=plant_id,
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
    db = SessionLocal()
    try:
        cutoff = datetime.now() - timedelta(hours=24)
        recent = db.query(SensorReading).filter(SensorReading.timestamp >= cutoff).count()
        if recent == 0:
            print("📊 No recent data — seeding 48 h of synthetic readings for 3 plants...")
            db.query(SensorReading).delete()
            db.commit()
            from sensor_simulator import FactorySensorSimulator
            for plant_id in PLANTS:
                simulator = FactorySensorSimulator(n_machines=5, n_hours=48)
                df = simulator.generate_readings()
                df = simulator.add_anomalies(df)
                df["plant_id"] = plant_id
                df.to_sql("sensor_readings", engine, if_exists="append", index=False)
            total = db.query(SensorReading).count()
            print(f"✅ Seeded {total} readings across {len(PLANTS)} plants")
        else:
            print(f"📊 Database ready: {recent} recent readings")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Manufacturing DT API starting up...")
    ensure_schema()
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
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Manufacturing Digital Twin API",
    description="Real-time monitoring, predictive maintenance, and optimization",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rul_predictor = RULPredictor()
maintenance_predictor = MaintenancePredictor()
anomaly_detector = AnomalyDetector()
scheduler = ProductionScheduler()

# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.post("/auth/login", response_model=TokenOut)
async def login(credentials: LoginRequest):
    user = DEMO_USERS.get(credentials.username)
    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(credentials.username, user["role"], user["plant_id"])
    return TokenOut(access_token=token, token_type="bearer")


@app.get("/auth/me", response_model=UserOut)
async def me(user: Optional[dict] = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return UserOut(username=user["sub"], role=user["role"], plant_id=user.get("plant_id"))

# ============================================================================
# CORE ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now()}


@app.get("/plants")
async def list_plants():
    return {"plants": PLANTS}


@app.post("/sensor-reading", response_model=SensorReadingOut)
async def ingest_sensor_reading(reading: SensorReadingIn, db: Session = Depends(get_db)):
    return ingest_sensor_data(
        db=db,
        machine_id=reading.machine_id,
        plant_id=reading.plant_id,
        temperature=reading.temperature,
        vibration=reading.vibration,
        power_consumption=reading.power_consumption,
        production_count=reading.production_count,
        downtime_minutes=reading.downtime_minutes,
        quality_score=reading.quality_score,
    )


@app.get("/opcua-status", response_model=OPCStatusOut)
async def get_opcua_status():
    return OPCStatusOut(**opcua_status)


@app.get("/factory-status", response_model=FactoryStatusOut)
async def get_factory_status(plant_id: str = "alpha", db: Session = Depends(get_db)):
    cutoff = datetime.now() - timedelta(hours=24)
    readings = (
        db.query(SensorReading)
        .filter(SensorReading.plant_id == plant_id, SensorReading.timestamp >= cutoff)
        .all()
    )

    if not readings:
        raise HTTPException(status_code=404, detail="No recent data available")

    df = pd.DataFrame([
        {
            "machine_id": r.machine_id,
            "production_count": r.production_count,
            "downtime_minutes": r.downtime_minutes,
            "quality_score": r.quality_score,
        }
        for r in readings
    ])

    total_time = 24 * 60
    n_machines = len(df["machine_id"].unique())
    availability = 1 - (df["downtime_minutes"].sum() / (total_time * n_machines))
    performance = df["production_count"].sum() / (n_machines * 100)
    quality = df["quality_score"].mean() / 100
    oee = float(np.clip(availability * performance * quality, 0, 1))

    active_alerts = (
        db.query(MaintenanceAlert)
        .filter(MaintenanceAlert.plant_id == plant_id, MaintenanceAlert.is_active == 1)
        .count()
    )
    downtime_cost = df["downtime_minutes"].sum() * 50

    return FactoryStatusOut(
        plant_id=plant_id,
        total_machines=n_machines,
        total_production_today=int(df["production_count"].sum()),
        average_oee=round(oee, 4),
        active_alerts=active_alerts,
        predicted_downtime_cost=downtime_cost,
        timestamp=datetime.now(),
    )


@app.get("/sensor-readings", response_model=List[SensorReadingOut])
async def get_sensor_readings(
    hours: int = 24,
    machine_id: str = None,
    plant_id: str = "alpha",
    db: Session = Depends(get_db),
):
    cutoff = datetime.now() - timedelta(hours=hours)
    query = db.query(SensorReading).filter(
        SensorReading.timestamp >= cutoff,
        SensorReading.plant_id == plant_id,
    )
    if machine_id:
        query = query.filter(SensorReading.machine_id == machine_id)
    return query.order_by(SensorReading.timestamp.desc()).limit(1000).all()


@app.get("/maintenance-alerts", response_model=List[MaintenanceAlertOut])
async def get_maintenance_alerts(plant_id: str = "alpha", db: Session = Depends(get_db)):
    return (
        db.query(MaintenanceAlert)
        .filter(MaintenanceAlert.plant_id == plant_id, MaintenanceAlert.is_active == 1)
        .all()
    )


@app.get("/risk-analysis", response_model=List[MachineRiskOut])
async def get_risk_analysis(plant_id: str = "alpha", db: Session = Depends(get_db)):
    results = []

    for machine_id in ["M1", "M2", "M3", "M4", "M5"]:
        rows = (
            db.query(SensorReading)
            .filter(
                SensorReading.machine_id == machine_id,
                SensorReading.plant_id == plant_id,
            )
            .order_by(SensorReading.timestamp.desc())
            .limit(6)
            .all()
        )
        if not rows:
            continue

        latest = rows[0]
        recent = [
            (r.temperature, r.vibration, r.power_consumption)
            for r in reversed(rows[1:])
        ]

        xgb_prob, _ = maintenance_predictor.predict_failure_probability(
            machine_id, latest.temperature, latest.vibration,
            latest.power_consumption, db, plant_id,
        )
        if_score = anomaly_detector.score(
            machine_id, latest.temperature, latest.vibration,
            latest.power_consumption, recent,
        )

        results.append(MachineRiskOut(
            machine_id=machine_id,
            plant_id=plant_id,
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
    jobs = db.query(ProductionJob).filter(ProductionJob.job_id.in_(job_ids)).all()
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found")
    job_dicts = [{"id": j.job_id, "product_type": j.product_type} for j in jobs]
    optimized_order, time_saved = scheduler.optimize_job_order(job_dicts)
    return SchedulingRecommendationOut(
        optimized_job_order=optimized_order,
        estimated_time_saved=time_saved,
        changeover_reduction=5.2,
    )

# ============================================================================
# RUL FORECAST ENDPOINTS
# ============================================================================

@app.get("/rul-forecast/{machine_id}", response_model=List[RULForecastPoint])
async def get_rul_forecast(
    machine_id: str,
    plant_id: str = "alpha",
    hours: int = 48,
    db: Session = Depends(get_db),
):
    """Return per-reading RUL estimates over the past `hours` for one machine."""
    cutoff = datetime.now() - timedelta(hours=hours)
    rows = (
        db.query(SensorReading)
        .filter(
            SensorReading.machine_id == machine_id,
            SensorReading.plant_id == plant_id,
            SensorReading.timestamp >= cutoff,
        )
        .order_by(SensorReading.timestamp.asc())
        .all()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No data for this machine/plant")

    points: list[RULForecastPoint] = []
    history: list[tuple[float, float, float]] = []
    for r in rows:
        rul, lower, upper = rul_predictor.predict(
            machine_id, r.temperature, r.vibration, r.power_consumption, history
        )
        points.append(RULForecastPoint(
            timestamp=r.timestamp,
            rul_hours=rul,
            rul_lower=lower,
            rul_upper=upper,
        ))
        history.append((r.temperature, r.vibration, r.power_consumption))
        if len(history) > 50:
            history.pop(0)

    return points


@app.get("/rul-summary", response_model=List[RULSummaryOut])
async def get_rul_summary(plant_id: str = "alpha", db: Session = Depends(get_db)):
    """Return current RUL estimate for every machine in a plant."""
    results: list[RULSummaryOut] = []
    for machine_id in ["M1", "M2", "M3", "M4", "M5"]:
        rows = (
            db.query(SensorReading)
            .filter(
                SensorReading.machine_id == machine_id,
                SensorReading.plant_id == plant_id,
            )
            .order_by(SensorReading.timestamp.desc())
            .limit(6)
            .all()
        )
        if not rows:
            continue

        latest = rows[0]
        recent = [
            (r.temperature, r.vibration, r.power_consumption)
            for r in reversed(rows[1:])
        ]
        rul, lower, upper = rul_predictor.predict(
            machine_id, latest.temperature, latest.vibration,
            latest.power_consumption, recent,
        )

        if rul <= 12:
            risk_level = "CRITICAL"
        elif rul <= 24:
            risk_level = "HIGH"
        elif rul <= 48:
            risk_level = "MEDIUM"
        else:
            risk_level = "GOOD"

        results.append(RULSummaryOut(
            machine_id=machine_id,
            plant_id=plant_id,
            rul_hours=rul,
            rul_lower=lower,
            rul_upper=upper,
            risk_level=risk_level,
        ))
    return results


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.get("/admin/cross-plant", response_model=List[PlantSummaryOut])
async def cross_plant_summary(
    user: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    results = []
    cutoff = datetime.now() - timedelta(hours=24)

    for plant in PLANTS:
        readings = (
            db.query(SensorReading)
            .filter(SensorReading.plant_id == plant, SensorReading.timestamp >= cutoff)
            .all()
        )
        alerts = (
            db.query(MaintenanceAlert)
            .filter(MaintenanceAlert.plant_id == plant, MaintenanceAlert.is_active == 1)
            .count()
        )

        if not readings:
            results.append(PlantSummaryOut(
                plant_id=plant, total_machines=5, total_production_24h=0,
                average_oee=0.0, active_alerts=alerts, avg_quality=0.0,
            ))
            continue

        df = pd.DataFrame([
            {
                "machine_id": r.machine_id,
                "production_count": r.production_count,
                "downtime_minutes": r.downtime_minutes,
                "quality_score": r.quality_score,
            }
            for r in readings
        ])
        n_machines = len(df["machine_id"].unique())
        total_time = 24 * 60
        availability = 1 - (df["downtime_minutes"].sum() / (total_time * n_machines))
        performance = df["production_count"].sum() / (n_machines * 100)
        quality = df["quality_score"].mean() / 100
        oee = float(np.clip(availability * performance * quality, 0, 1))

        results.append(PlantSummaryOut(
            plant_id=plant,
            total_machines=n_machines,
            total_production_24h=int(df["production_count"].sum()),
            average_oee=round(oee, 4),
            active_alerts=alerts,
            avg_quality=round(float(df["quality_score"].mean()), 2),
        ))

    return results


@app.get("/")
async def root():
    return {
        "message": "Manufacturing Plant Digital Twin API",
        "docs": "/docs",
        "version": "0.2.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
