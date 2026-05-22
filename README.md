# ⚛️ PlasmaX Control Room

> Real-time fusion reactor monitoring with AI-powered instability prediction and experiment logging.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

PlasmaX is a Streamlit dashboard that simulates and monitors a fusion reactor's plasma dynamics in real time. It models three physically coupled signals — temperature, pressure, and magnetic field — runs a two-tier safety checker on each cycle, and scores collapse risk through a weighted multi-signal AI prediction engine. All experiment runs are persisted locally for post-session review.

---

## Features

- **Physically realistic simulation** — temperature follows an Ornstein–Uhlenbeck mean-reverting process; pressure is coupled to temperature; magnetic field weakens under high pressure (β effect)
- **Two-tier safety system** — threshold checks (is a value in the danger zone?) combined with trend checks (is it heading there fast?)
- **AI risk scoring** — weighted 0–100 score across temperature, pressure, and magnetic field, with a compound-instability bonus when multiple signals are simultaneously distressed
- **Live Plotly charts** — three-panel subplot (temperature / pressure / magnetic field) with safety threshold lines, updated each cycle
- **Experiment archive** — every run is saved to disk as JSON with full signal arrays and a summary; the 10 most recent are shown in the UI
- **Reproducible simulations** — pass a `seed` to `generate_fusion_data` for deterministic output

---

## Project Structure

```
plasmx/
├── app.py               # Streamlit UI — controls, live charts, archive view
├── data_simulator.py    # Plasma physics simulation (Ornstein–Uhlenbeck + coupling)
├── utils.py             # Safety checker, AI predictor, storage, analytics
├── requirements.txt     # Pinned dependencies with upper bounds
└── experiments/         # Auto-created; stores one JSON file per run
```

---

## Quickstart

```bash
# 1. Clone and enter the repo
git clone https://github.com/your-org/plasmx.git
cd plasmx

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration

| Control | Default | Description |
|---|---|---|
| Simulation Cycles | 20 | Number of reactor cycles per session |
| Steps per Cycle | 200 | Time steps simulated per cycle (min 60) |
| Noise Level | 0.2 | Stochastic noise scaling (0 = deterministic) |
| Refresh Speed | 0.5 s | Pause between cycle renders |

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `PLASMX_EXPERIMENT_DIR` | `experiments/` | Directory for JSON experiment logs |

---

## Safety Thresholds

| Signal | Warning | Critical |
|---|---|---|
| Temperature (keV) | > 750 | > 900 |
| Pressure (atm) | > 12 | > 16 |
| Magnetic Field (T) | < 1.0 | < 0.8 |

Trend alerts are also raised when a signal is approaching a threshold at a rate that will breach it within the current cycle window.

---

## AI Risk Scoring

The predictor scores each signal independently, then adds a compound-instability bonus:

| Component | Max Points |
|---|---|
| Temperature level + volatility + trend | 40 |
| Pressure level + trend | 30 |
| Magnetic field level + trend | 30 |
| 2-signal interaction bonus | +10 |
| 3-signal interaction bonus | +15 |
| **Total (clamped)** | **100** |

| Score | Status |
|---|---|
| 0 – 39 | 🟢 Stable |
| 40 – 69 | 🟡 Warning |
| 70 – 100 | 🔴 Critical |

---

## Experiment Storage

Each cycle is saved to `experiments/<run_id>.json` with:

- Run metadata (ID, timestamp, status)
- AI prediction (score, status, message)
- Summary statistics (max, mean per signal)
- Full signal arrays (temperature, pressure, magnetic field)

Files are named `<session_timestamp>_cycle_<NNN>_<uid6>.json`. The 6-char UUID suffix prevents collisions between concurrent sessions started in the same second.

---

## Development

```bash
# Run with a fixed seed for reproducible testing
python - <<'EOF'
from data_simulator import generate_fusion_data
from utils import check_plasma_safety, predict_instability

data = generate_fusion_data(n_steps=200, noise_level=0.2, seed=42)
status, msg, alerts = check_plasma_safety(data)
prediction = predict_instability(data)
print(status, prediction)
EOF
```

To point experiment storage at a temp directory during tests:

```python
import os
os.environ["PLASMX_EXPERIMENT_DIR"] = "/tmp/plasmx_test"
```

---

## Requirements

- Python 3.10+
- streamlit `>=1.32.0,<2.0.0`
- pandas `>=2.0.0,<3.0.0`
- numpy `>=1.26.0,<2.0.0`
- plotly `>=5.20.0,<6.0.0`

---

## License

MIT
