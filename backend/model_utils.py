"""
Shared feature engineering for maintenance prediction.
Used by both train_models.py (offline) and app.py (serving) to guarantee
train/serve feature parity — do not split this logic.
"""

import numpy as np

MACHINE_IDS = ["M1", "M2", "M3", "M4", "M5"]
_MACHINE_ENC = {m: i for i, m in enumerate(MACHINE_IDS)}

# Column order that XGBoost was trained on — must not be reordered
FEATURE_COLS = [
    "machine_id_enc",
    "temp", "vib", "power",
    "temp_mean6", "vib_mean6", "power_mean6",
    "temp_std6",  "vib_std6",  "power_std6",
]

WINDOW = 6  # rolling window size


def compute_features(
    machine_id: str,
    temp: float,
    vib: float,
    power: float,
    recent: list,          # list of (temp, vib, power) tuples, oldest-first, up to WINDOW-1 items
) -> np.ndarray:
    """
    Returns shape (1, len(FEATURE_COLS)) array.

    `recent` should be the last WINDOW-1 readings before the current one so
    the full window includes the current reading.  If history is short, the
    window shrinks gracefully (std → 0 when only one point exists).
    """
    machine_enc = _MACHINE_ENC.get(machine_id, 0)

    # Build window: up to WINDOW-1 historical + current
    window = list(recent[-(WINDOW - 1):]) + [(temp, vib, power)]
    arr = np.array(window, dtype=float)

    t_mean = arr[:, 0].mean()
    v_mean = arr[:, 1].mean()
    p_mean = arr[:, 2].mean()
    t_std  = arr[:, 0].std() if len(arr) > 1 else 0.0
    v_std  = arr[:, 1].std() if len(arr) > 1 else 0.0
    p_std  = arr[:, 2].std() if len(arr) > 1 else 0.0

    return np.array([[
        machine_enc, temp, vib, power,
        t_mean, v_mean, p_mean,
        t_std,  v_std,  p_std,
    ]], dtype=float)
