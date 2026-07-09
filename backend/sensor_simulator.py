"""
Synthetic Sensor Data Generator for Manufacturing Plant Digital Twin
Generates realistic factory sensor data with anomalies and failure patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3

class FactorySensorSimulator:
    """Generate realistic sensor data for 5 manufacturing machines"""
    
    def __init__(self, n_machines=5, n_hours=24):
        self.n_machines = n_machines
        self.n_hours = n_hours
        self.machine_ids = [f"M{i+1}" for i in range(n_machines)]
        self.product_types = ["Part_A", "Part_B", "Part_C"]
        
        # Machine-specific baseline parameters
        self.machine_params = {
            "M1": {"temp_mean": 60, "temp_std": 3, "vib_mean": 2.5, "vib_std": 0.3},
            "M2": {"temp_mean": 65, "temp_std": 3, "vib_mean": 3.0, "vib_std": 0.4},
            "M3": {"temp_mean": 58, "temp_std": 3, "vib_mean": 2.2, "vib_std": 0.25},
            "M4": {"temp_mean": 70, "temp_std": 4, "vib_mean": 3.5, "vib_std": 0.5},  # Older machine
            "M5": {"temp_mean": 62, "temp_std": 3, "vib_mean": 2.8, "vib_std": 0.35},
        }
    
    def generate_readings(self):
        """Generate sensor readings for the past n_hours"""
        data = []
        now = datetime.now()
        
        for hour in range(self.n_hours):
            timestamp = now - timedelta(hours=self.n_hours - hour)
            
            for machine_id in self.machine_ids:
                # Get machine baseline
                params = self.machine_params[machine_id]
                
                # Temperature: normal distribution + periodic maintenance effect
                temp = np.random.normal(params["temp_mean"], params["temp_std"])
                temp = np.clip(temp, 40, 90)  # Realistic bounds
                
                # Vibration: with occasional spikes
                vib = np.random.normal(params["vib_mean"], params["vib_std"])
                if np.random.random() < 0.05:  # 5% chance of spike (bearing issue)
                    vib *= np.random.uniform(1.5, 3.0)
                vib = np.clip(vib, 0.5, 10)
                
                # Power consumption: proportional to production
                power = np.random.normal(15, 2) + (vib * 2)  # Higher vibration = more power
                power = np.clip(power, 5, 40)
                
                # Production count: 80-120 parts per hour
                production_count = np.random.randint(80, 120)
                
                # Downtime: most machines 0, occasional issues
                downtime = 0
                if np.random.random() < 0.02:  # 2% chance of downtime event
                    downtime = np.random.uniform(5, 45)  # 5-45 minutes
                
                # Quality: slight inverse relationship with vibration
                quality_base = 95 - (vib * 2)
                quality = np.random.normal(quality_base, 2)
                quality = np.clip(quality, 80, 100)
                
                data.append({
                    "timestamp": timestamp,
                    "machine_id": machine_id,
                    "temperature": round(temp, 2),
                    "vibration": round(vib, 3),
                    "power_consumption": round(power, 2),
                    "production_count": int(production_count),
                    "downtime_minutes": round(downtime, 1),
                    "quality_score": round(quality, 2)
                })
        
        return pd.DataFrame(data)
    
    def add_anomalies(self, df):
        """Inject realistic anomalies: bearing degradation, motor issues"""
        df = df.copy()
        
        # Simulate bearing degradation on M4 (older machine)
        m4_mask = df["machine_id"] == "M4"
        df.loc[m4_mask, "vibration"] *= np.linspace(1.0, 1.4, m4_mask.sum())
        df.loc[m4_mask, "temperature"] += 3
        
        # Simulate cooling system issue on M2 (last few hours)
        m2_recent = (df["machine_id"] == "M2") & (df["timestamp"] > datetime.now() - timedelta(hours=4))
        df.loc[m2_recent, "temperature"] += np.random.uniform(5, 15, m2_recent.sum())
        df.loc[m2_recent, "quality_score"] -= 5
        
        return df
    
    def save_to_csv(self, filepath="synthetic_data.csv"):
        """Generate data and save to CSV"""
        df = self.generate_readings()
        df = self.add_anomalies(df)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Generated {len(df)} sensor readings → {filepath}")
        return df
    
    def save_to_sqlite(self, db_path="factory.db"):
        """Generate data and save to SQLite database"""
        df = self.generate_readings()
        df = self.add_anomalies(df)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        conn = sqlite3.connect(db_path)
        df.to_sql("sensor_readings", conn, if_exists="append", index=False)
        conn.close()
        
        print(f"✅ Inserted {len(df)} readings into {db_path}")
        return df

if __name__ == "__main__":
    # Generate 24 hours of data for 5 machines
    simulator = FactorySensorSimulator(n_machines=5, n_hours=24)
    
    # Save as CSV
    print("📝 Saving to CSV...")
    df = simulator.save_to_csv("synthetic_data.csv")
    
    # Save to SQLite database
    print("💾 Saving to SQLite database...")
    simulator.save_to_sqlite("factory.db")
    
    print("\n✅ Data Preview:")
    print(df.head(10))
    print(f"\n✅ Shape: {df.shape}")
    print(f"✅ Machines: {df['machine_id'].unique()}")
    print(f"✅ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
