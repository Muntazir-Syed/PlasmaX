import numpy as np
import json
import os
import uuid
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd

EXPERIMENT_DIR = os.environ.get("PLASMX_EXPERIMENT_DIR", "experiments")

# ------------------------------------------------------------------ #
# PHYSICAL SAFETY THRESHOLDS (documented)                             #
# ------------------------------------------------------------------ #
# Temperature: proxy units scaled so 1000 ~ upper safe operating limit
# Pressure:    normalised ATM-equivalent; >14 risks vessel integrity
# Mag field:   Tesla proxy; <1.0 T means confinement is insufficient
TEMP_CRITICAL  = 900.0
TEMP_WARNING   = 750.0
PRESSURE_CRIT  = 16.0
PRESSURE_WARN  = 12.0
MAG_CRIT       = 0.8
MAG_WARN       = 1.0

# Distressed thresholds used consistently in BOTH safety and prediction
# A signal is "distressed" if its component score >= this fraction of its max
_DISTRESSED_SCORE_MIN = 9   # matches lowest non-zero tier across all components


# ------------------------------------------------------------------ #
# SAFETY SYSTEM                                                        #
# ------------------------------------------------------------------ #
def check_plasma_safety(
    data: pd.DataFrame,
) -> Tuple[str, str, List[Tuple[str, str]]]:
    """
    Evaluate plasma safety from both the latest readings AND recent trends.

    Two-pass approach:
      1. Threshold check  — is a value already in the danger zone?
      2. Trend check      — is a value heading toward the danger zone fast enough
                            to be worth flagging before it crosses?

    Returns
    -------
    (status, message, alerts)
      status  : 'stable' | 'warning' | 'critical'
      message : human-readable summary
      alerts  : list of (severity, description) pairs
    """
    # Current (last-point) values
    temp     = float(data["temperature"].iloc[-1])
    pressure = float(data["pressure"].iloc[-1])
    mag      = float(data["magnetic_field"].iloc[-1])

    # Trend over the last 20 steps (mean pairwise difference = rate/step)
    # np.diff reduces length by 1, so window=20 yields 19 differences — this
    # is intentional and documented here to avoid confusion.
    window = min(20, len(data))
    t_rate = float(np.mean(np.diff(data["temperature"].values[-window:])))
    p_rate = float(np.mean(np.diff(data["pressure"].values[-window:])))
    m_rate = float(np.mean(np.diff(data["magnetic_field"].values[-window:])))

    alerts: List[Tuple[str, str]] = []

    # ---- Temperature ----
    if temp > TEMP_CRITICAL:
        alerts.append(("critical", f"Plasma overheating ({temp:.1f} > {TEMP_CRITICAL})"))
    elif temp > TEMP_WARNING:
        alerts.append(("warning", f"Elevated temperature ({temp:.1f})"))
    elif t_rate > 5.0 and temp > TEMP_WARNING * 0.85:
        alerts.append(("warning", f"Temperature rising rapidly (+{t_rate:.1f}/step, now {temp:.1f})"))

    # ---- Pressure ----
    if pressure > PRESSURE_CRIT:
        alerts.append(("critical", f"Pressure exceeds vessel limit ({pressure:.2f} > {PRESSURE_CRIT})"))
    elif pressure > PRESSURE_WARN:
        alerts.append(("warning", f"Pressure instability ({pressure:.2f})"))
    elif p_rate > 0.3 and pressure > PRESSURE_WARN * 0.80:
        alerts.append(("warning", f"Pressure trending upward (+{p_rate:.2f}/step, now {pressure:.2f})"))

    # ---- Magnetic confinement ----
    if mag < MAG_CRIT:
        alerts.append(("critical", f"Magnetic confinement failure ({mag:.3f} T)"))
    elif mag < MAG_WARN:
        alerts.append(("warning", f"Weakening magnetic confinement ({mag:.3f} T)"))
    elif m_rate < -0.02 and mag < MAG_WARN * 1.20:
        alerts.append(("warning", f"Magnetic field declining ({m_rate:.3f} T/step, now {mag:.3f} T)"))

    if any(a[0] == "critical" for a in alerts):
        return "critical", "CRITICAL SYSTEM STATE", alerts

    if alerts:
        return "warning", "System unstable — monitor closely", alerts

    return "stable", "All parameters within nominal range", []


# ------------------------------------------------------------------ #
# AI PREDICTION ENGINE  (multi-signal, weighted risk model)           #
# ------------------------------------------------------------------ #
def predict_instability(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Estimate plasma collapse risk from temperature, pressure, and
    magnetic-field signals combined.

    Requires data to have at least 60 rows (enforced by generate_fusion_data).

    Scoring (max 100):
    - Temperature level & volatility  : up to 40 pts
    - Pressure level & upward trend   : up to 30 pts
    - Magnetic field depression        : up to 30 pts
    - Multi-signal interaction bonus   : up to +15 pts when ≥2 signals are
                                         simultaneously distressed, because
                                         concurrent failures are disproportionately
                                         dangerous (compound instability).
    Final score is clamped to 100.

    A signal is "distressed" when its component score >= _DISTRESSED_SCORE_MIN,
    the same threshold used by check_plasma_safety's warning tier, ensuring the
    two subsystems agree on what "distressed" means.
    """
    n = len(data)
    temp = data["temperature"].values
    pres = data["pressure"].values
    mag  = data["magnetic_field"].values

    # Safe slice sizes — always use at most n//2 for trend windows so the
    # early and late windows never overlap, even on short runs.
    half = max(1, n // 2)
    trend_window = min(30, half)

    score = 0
    distressed: List[str] = []

    # ---- Temperature component (max 40) ----
    t_latest     = temp[-1]
    t_volatility = float(np.std(np.diff(temp)))
    t_trend      = float(np.mean(temp[-trend_window:]) - np.mean(temp[:trend_window]))

    t_score = 0
    if t_latest > TEMP_CRITICAL:
        t_score += 20
    elif t_latest > TEMP_WARNING:
        t_score += 10

    if t_volatility > 15:
        t_score += 12
    elif t_volatility > 9:
        t_score += 6

    if t_trend > 60:
        t_score += 8
    elif t_trend > 30:
        t_score += 4

    score += t_score
    if t_score >= _DISTRESSED_SCORE_MIN:
        distressed.append("temperature")

    # ---- Pressure component (max 30) ----
    p_latest = float(pres[-1])
    p_trend  = float(np.mean(pres[-trend_window:]) - np.mean(pres[:trend_window]))

    p_score = 0
    if p_latest > PRESSURE_CRIT:
        p_score += 18
    elif p_latest > PRESSURE_WARN:
        p_score += 9

    if p_trend > 2.0:
        p_score += 12
    elif p_trend > 0.8:
        p_score += 6

    score += p_score
    if p_score >= _DISTRESSED_SCORE_MIN:
        distressed.append("pressure")

    # ---- Magnetic field component (max 30) ----
    m_latest = float(mag[-1])
    m_trend  = float(np.mean(mag[-trend_window:]) - np.mean(mag[:trend_window]))

    m_score = 0
    if m_latest < MAG_CRIT:
        m_score += 18
    elif m_latest < MAG_WARN:
        m_score += 9

    if m_trend < -0.15:
        m_score += 12
    elif m_trend < -0.05:
        m_score += 6

    score += m_score
    if m_score >= _DISTRESSED_SCORE_MIN:
        distressed.append("magnetic_field")

    # ---- Multi-signal interaction bonus ----
    # Two signals distressed simultaneously: +10 pts
    # All three simultaneously: +15 pts
    n_distressed = len(distressed)
    if n_distressed == 3:
        score += 15
    elif n_distressed == 2:
        score += 10

    score = min(score, 100)

    if score >= 70:
        return {
            "status":  "critical",
            "score":   score,
            "message": "High collapse risk — initiate emergency protocol",
        }
    if score >= 40:
        return {
            "status":  "warning",
            "score":   score,
            "message": "Instability forming — reduce heating power",
        }

    return {
        "status":  "stable",
        "score":   score,
        "message": "Plasma stable — nominal operating conditions",
    }


# ------------------------------------------------------------------ #
# EXPERIMENT STORAGE                                                   #
# ------------------------------------------------------------------ #
def _unique_run_id(session_ts: str, cycle: int) -> str:
    """
    Generate a collision-resistant run ID.
    Appends a 6-char UUID hex fragment so concurrent sessions started in the
    same second can never overwrite each other's files.
    """
    uid = uuid.uuid4().hex[:6]
    return f"{session_ts}_cycle_{cycle:03d}_{uid}"


def save_experiment(
    run_id: str,
    data: pd.DataFrame,
    prediction: Dict[str, Any],
    status: str,
    experiment_dir: str = EXPERIMENT_DIR,
) -> str:
    """
    Persist a single experiment run to <experiment_dir>/<run_id>.json.

    Parameters
    ----------
    run_id         : unique identifier for this run
    data           : DataFrame with time, temperature, pressure, magnetic_field
    prediction     : dict returned by predict_instability
    status         : 'stable' | 'warning' | 'critical'
    experiment_dir : override the default storage directory (useful for tests)

    Returns
    -------
    Absolute path of the written JSON file.
    """
    os.makedirs(experiment_dir, exist_ok=True)

    file_path = os.path.join(experiment_dir, f"{run_id}.json")

    record = {
        "run_id":     run_id,
        "timestamp":  datetime.now().isoformat(),
        "status":     status,
        "prediction": prediction,
        "summary": {
            "temp_max":    float(data["temperature"].max()),
            "temp_mean":   float(data["temperature"].mean()),
            "pres_max":    float(data["pressure"].max()),
            "pres_mean":   float(data["pressure"].mean()),
            "mag_min":     float(data["magnetic_field"].min()),
            "mag_mean":    float(data["magnetic_field"].mean()),
        },
        "data": {
            "temperature":    data["temperature"].tolist(),
            "pressure":       data["pressure"].tolist(),
            "magnetic_field": data["magnetic_field"].tolist(),
        },
    }

    with open(file_path, "w") as f:
        json.dump(record, f, indent=2)

    return os.path.abspath(file_path)


# ------------------------------------------------------------------ #
# LOAD EXPERIMENTS                                                      #
# ------------------------------------------------------------------ #
def load_experiments(
    limit: Optional[int] = 10,
    include_data: bool = False,
    experiment_dir: str = EXPERIMENT_DIR,
) -> List[Dict[str, Any]]:
    """
    Load experiment records from experiment_dir.

    Parameters
    ----------
    limit          : max number of most-recent records to return.
                     Pass None to load all.
    include_data   : if False (default), the 'data' key (full signal arrays) is
                     stripped — saving significant memory for summary views.
    experiment_dir : override the default storage directory (useful for tests)

    Notes
    -----
    Records are sorted by the ISO 'timestamp' field inside the JSON, not by
    filesystem mtime, to avoid ordering anomalies from clock skew or file
    copies. Corrupt or unreadable JSON files are silently skipped.

    Returns
    -------
    List of experiment dicts in chronological order (oldest first).
    """
    if not os.path.exists(experiment_dir):
        return []

    # Cheap mtime pre-sort so we only parse the newest `limit` files —
    # avoids reading thousands of JSON blobs when limit is small.
    entries: List[Tuple[float, str]] = []
    for fname in os.listdir(experiment_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(experiment_dir, fname)
        try:
            entries.append((os.path.getmtime(fpath), fpath))
        except OSError:
            continue

    entries.sort(key=lambda x: x[0], reverse=True)
    if limit is not None:
        entries = entries[:limit]

    experiments = []
    for _, fpath in entries:
        try:
            with open(fpath, "r") as fh:
                record = json.load(fh)
            if not all(k in record for k in ("run_id", "timestamp", "status", "prediction")):
                continue
            if not include_data:
                record.pop("data", None)
            experiments.append(record)
        except (json.JSONDecodeError, OSError):
            continue

    # Final sort on the authoritative ISO timestamp inside the JSON record.
    # ISO-8601 strings sort lexicographically = chronologically when the
    # format is consistent (which datetime.now().isoformat() guarantees).
    return sorted(experiments, key=lambda x: x["timestamp"])


# ------------------------------------------------------------------ #
# DERIVED ANALYTICS HELPERS                                            #
# ------------------------------------------------------------------ #
def compute_session_stats(
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate statistics across a list of experiment records.

    Note: stats reflect only the records passed in. If load_experiments was
    called with limit=10, these stats cover only those 10 runs — callers
    should communicate this to the user (e.g. "Last 10 runs").
    """
    if not experiments:
        return {}

    statuses = [e["status"] for e in experiments]
    scores   = [e["prediction"]["score"] for e in experiments]

    return {
        "total_runs":     len(experiments),
        "stable_count":   statuses.count("stable"),
        "warning_count":  statuses.count("warning"),
        "critical_count": statuses.count("critical"),
        "avg_risk_score": round(float(np.mean(scores)), 1),
        "max_risk_score": int(max(scores)),
    }