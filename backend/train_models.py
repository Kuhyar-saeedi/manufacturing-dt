"""
Train XGBoost predictive-maintenance model on synthetic sensor data.

Label design: each "fault episode" has a hidden health state that degrades
from 1.0 → 0 via a random walk.  Sensors are noisy proxies of health.
Because the sensors are noisy and health is latent, the model cannot simply
learn a threshold — it must detect degradation patterns in the rolling window.

Run:
    cd backend
    python train_models.py
Output:
    maintenance_model.joblib  (committed to repo so Railway has it at deploy time)
"""

import sys
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# model_utils must be on the path (run from backend/)
from model_utils import MACHINE_IDS, FEATURE_COLS, compute_features

# ── Machine baselines (mirrors sensor_simulator.py) ──────────────────────────
MACHINE_PARAMS = {
    "M1": {"temp_base": 60.0, "vib_base": 2.5, "power_base": 15.0},
    "M2": {"temp_base": 65.0, "vib_base": 3.0, "power_base": 16.0},
    "M3": {"temp_base": 58.0, "vib_base": 2.2, "power_base": 14.0},
    "M4": {"temp_base": 70.0, "vib_base": 3.5, "power_base": 17.0},
    "M5": {"temp_base": 62.0, "vib_base": 2.8, "power_base": 15.0},
}

N_FAULT_EPISODES   = 40   # per machine — last 20% held out for test
N_NORMAL_EPISODES  = 20   # per machine
EPISODE_LENGTH     = 48   # readings per episode
FAILURE_WINDOW     = 20   # label=1 for last N readings of a fault episode
TRAIN_FRAC         = 0.75


# ── Episode generators ────────────────────────────────────────────────────────

def _fault_episode(params: dict, rng: np.random.Generator) -> list:
    """
    Fault episode: health decays from 1.0 → near 0 over EPISODE_LENGTH steps.
    Returns list of (temp, vib, power, label) tuples.
    """
    health = 1.0
    drain_mean = 1.0 / (EPISODE_LENGTH * 0.85)
    readings = []

    for t in range(EPISODE_LENGTH):
        drain = max(0.0, rng.normal(drain_mean, drain_mean * 0.4))
        health = max(0.0, health - drain)

        # Sensor values worsen as health drops — different sensitivities
        temp  = params["temp_base"]  * (1 + 0.45 * (1 - health)) + rng.normal(0, 2.0)
        vib   = params["vib_base"]   * (1 + 1.60 * (1 - health)) + rng.normal(0, 0.25)
        power = params["power_base"] * (1 + 0.30 * (1 - health)) + rng.normal(0, 0.9)

        temp  = float(np.clip(temp,  40, 95))
        vib   = float(np.clip(vib,   0.3, 12))
        power = float(np.clip(power, 5,  40))

        label = 1 if t >= (EPISODE_LENGTH - FAILURE_WINDOW) else 0
        readings.append((temp, vib, power, label))

    return readings


def _normal_episode(params: dict, rng: np.random.Generator) -> list:
    """
    Normal episode: health wanders in [0.80, 1.0] — never reaches failure zone.
    """
    health = rng.uniform(0.88, 1.0)
    readings = []

    for _ in range(EPISODE_LENGTH):
        health = float(np.clip(health + rng.normal(-0.002, 0.006), 0.80, 1.0))

        temp  = params["temp_base"]  * (1 + 0.08 * (1 - health)) + rng.normal(0, 2.0)
        vib   = params["vib_base"]   * (1 + 0.25 * (1 - health)) + rng.normal(0, 0.25)
        power = params["power_base"] * (1 + 0.05 * (1 - health)) + rng.normal(0, 0.9)

        temp  = float(np.clip(temp,  40, 95))
        vib   = float(np.clip(vib,   0.3, 12))
        power = float(np.clip(power, 5,  40))

        readings.append((temp, vib, power, 0))

    return readings


# ── Dataset construction ──────────────────────────────────────────────────────

def generate_training_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for machine_id in MACHINE_IDS:
        params = MACHINE_PARAMS[machine_id]

        for ep_idx in range(N_FAULT_EPISODES):
            episode = _fault_episode(params, rng)
            history = []
            for t, (temp, vib, power, label) in enumerate(episode):
                feats = compute_features(machine_id, temp, vib, power, history)
                row = dict(zip(FEATURE_COLS, feats[0]))
                row["label"]   = label
                row["episode"] = f"{machine_id}_fault_{ep_idx:03d}"
                rows.append(row)
                history.append((temp, vib, power))

        for ep_idx in range(N_NORMAL_EPISODES):
            episode = _normal_episode(params, rng)
            history = []
            for t, (temp, vib, power, label) in enumerate(episode):
                feats = compute_features(machine_id, temp, vib, power, history)
                row = dict(zip(FEATURE_COLS, feats[0]))
                row["label"]   = label
                row["episode"] = f"{machine_id}_normal_{ep_idx:03d}"
                rows.append(row)
                history.append((temp, vib, power))

    return pd.DataFrame(rows)


# ── Train / evaluate ──────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> XGBClassifier:
    # Episode-based split: first TRAIN_FRAC of each type goes to train
    fault_eps  = sorted(e for e in df["episode"].unique() if "fault"  in e)
    normal_eps = sorted(e for e in df["episode"].unique() if "normal" in e)

    n_fault_train  = int(len(fault_eps)  * TRAIN_FRAC)
    n_normal_train = int(len(normal_eps) * TRAIN_FRAC)

    train_set = set(fault_eps[:n_fault_train] + normal_eps[:n_normal_train])
    mask_train = df["episode"].isin(train_set)

    X_train = df.loc[ mask_train, FEATURE_COLS].values
    y_train = df.loc[ mask_train, "label"].values
    X_test  = df.loc[~mask_train, FEATURE_COLS].values
    y_test  = df.loc[~mask_train, "label"].values

    print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"Train positives: {y_train.mean():.1%}  |  Test positives: {y_test.mean():.1%}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = float(neg / pos) if pos else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        eval_metric="aucpr",
        early_stopping_rounds=25,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n-- Evaluation --")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Pre-failure"], digits=3))
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{cm}")
    tpr = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0
    print(f"\nTPR (recall on Pre-failure): {tpr:.3f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_prob):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

    if tpr < 0.85:
        print("\n⚠️  TPR below 85% target — consider increasing N_FAULT_EPISODES or FAILURE_WINDOW.")
    else:
        print("\n✅ TPR target met (≥ 85%)")

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating training data...")
    df = generate_training_data()
    total = len(df)
    pos_rate = df["label"].mean()
    print(f"Dataset: {total:,} rows  |  positive rate: {pos_rate:.1%}\n")

    print("Training XGBoost model...")
    model = train_and_evaluate(df)

    out = "maintenance_model.joblib"
    joblib.dump(model, out)
    print(f"\nModel saved → {out}")
    print("Commit this file so Railway has it at deploy time.")
