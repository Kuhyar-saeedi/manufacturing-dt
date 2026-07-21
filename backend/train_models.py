"""
Train XGBoost predictive-maintenance model on synthetic sensor data.

Label design: each "fault episode" has a hidden health state that degrades
from 1.0 ->0 via a random walk.  Sensors are noisy proxies of health.
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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    mean_absolute_error,
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
    Fault episode: health decays from 1.0 ->near 0 over EPISODE_LENGTH steps.
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
        print("\nWARN: TPR below 85% target — consider increasing N_FAULT_EPISODES or FAILURE_WINDOW.")
    else:
        print("\nOK: TPR target met (>= 85%)")

    return model


# ── Isolation Forest (unsupervised anomaly detector) ─────────────────────────

def train_anomaly_detector(df: pd.DataFrame) -> dict:
    """
    Train IsolationForest on normal-episode data only.

    Returns a dict with the fitted model and the score range observed on training
    data so inference can normalize scores to [0, 1].
    """
    normal_mask = df["episode"].str.contains("normal")
    X_normal = df.loc[normal_mask, FEATURE_COLS].values

    print(f"Anomaly detector — training on {len(X_normal):,} normal rows")

    iforest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_normal)

    # Calibrate score range using the full dataset (includes fault episodes)
    X_all = df[FEATURE_COLS].values
    raw_scores = iforest.score_samples(X_all)
    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())

    # Quick evaluation: IF should assign lower scores to fault rows
    y_all = df["label"].values
    normal_mean = raw_scores[y_all == 0].mean()
    fault_mean  = raw_scores[y_all == 1].mean()
    print(f"Mean score - normal: {normal_mean:.4f}  |  pre-failure: {fault_mean:.4f}")
    if fault_mean < normal_mean:
        print("OK: IF correctly scores pre-failure readings as more anomalous")
    else:
        print("WARN: IF scores not clearly separated - check feature distribution")

    return {"model": iforest, "score_min": score_min, "score_max": score_max}


# ── RUL dataset construction ──────────────────────────────────────────────────

NORMAL_RUL_CAP = float(EPISODE_LENGTH + 24)  # ~72 h — "healthy, no near-term failure"


def generate_rul_data(seed: int = 42) -> pd.DataFrame:
    """
    Build a regression dataset where the target is hours-remaining-until-failure.

    Fault episodes  ->target declines linearly from EPISODE_LENGTH-1 ->0
    Normal episodes ->target is capped at NORMAL_RUL_CAP (machine is healthy)
    """
    rng = np.random.default_rng(seed)
    rows = []

    for machine_id in MACHINE_IDS:
        params = MACHINE_PARAMS[machine_id]

        for ep_idx in range(N_FAULT_EPISODES):
            episode = _fault_episode(params, rng)
            history = []
            for t, (temp, vib, power, _) in enumerate(episode):
                feats = compute_features(machine_id, temp, vib, power, history)
                row = dict(zip(FEATURE_COLS, feats[0]))
                row["rul_hours"] = float(EPISODE_LENGTH - 1 - t)
                rows.append(row)
                history.append((temp, vib, power))

        for ep_idx in range(N_NORMAL_EPISODES):
            episode = _normal_episode(params, rng)
            history = []
            for t, (temp, vib, power, _) in enumerate(episode):
                feats = compute_features(machine_id, temp, vib, power, history)
                row = dict(zip(FEATURE_COLS, feats[0]))
                row["rul_hours"] = NORMAL_RUL_CAP
                rows.append(row)
                history.append((temp, vib, power))

    return pd.DataFrame(rows)


def train_rul_model(df: pd.DataFrame) -> XGBRegressor:
    """Train XGBRegressor to predict remaining useful life (hours)."""
    X = df[FEATURE_COLS].values
    y = df["rul_hours"].values

    split = int(len(X) * TRAIN_FRAC)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=25,
        eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"RUL model — MAE on test set: {mae:.2f} hours")

    # sanity: predictions on normal data should be close to NORMAL_RUL_CAP
    normal_mask = y_test >= (NORMAL_RUL_CAP - 1)
    fault_mask  = y_test <= 5
    if normal_mask.any() and fault_mask.any():
        print(f"  Mean predicted RUL (healthy): {preds[normal_mask].mean():.1f} h  "
              f"(target ≈ {NORMAL_RUL_CAP:.0f} h)")
        print(f"  Mean predicted RUL (near-failure): {preds[fault_mask].mean():.1f} h  "
              f"(target ≈ 0–5 h)")

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating training data...")
    df = generate_training_data()
    total = len(df)
    pos_rate = df["label"].mean()
    print(f"Dataset: {total:,} rows  |  positive rate: {pos_rate:.1%}\n")

    print("Training XGBoost classifier (failure prediction)...")
    model = train_and_evaluate(df)

    xgb_out = "maintenance_model.joblib"
    joblib.dump(model, xgb_out)
    print(f"\nXGBoost classifier saved ->{xgb_out}")

    print("\nTraining Isolation Forest anomaly detector...")
    anomaly_artifact = train_anomaly_detector(df)

    if_out = "anomaly_model.joblib"
    joblib.dump(anomaly_artifact, if_out)
    print(f"Anomaly model saved ->{if_out}")

    print("\nGenerating RUL training data...")
    rul_df = generate_rul_data()
    print(f"RUL dataset: {len(rul_df):,} rows\n")

    print("Training XGBoost RUL regressor...")
    rul_model = train_rul_model(rul_df)

    rul_out = "rul_model.joblib"
    joblib.dump({"model": rul_model, "normal_rul_cap": NORMAL_RUL_CAP}, rul_out)
    print(f"RUL model saved ->{rul_out}")

    print("\nCommit all three .joblib files so Railway has them at deploy time.")
