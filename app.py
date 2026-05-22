import streamlit as st
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_simulator import generate_fusion_data
from utils import (
    check_plasma_safety,
    predict_instability,
    save_experiment,
    load_experiments,
    compute_session_stats,
    _unique_run_id,
    EXPERIMENT_DIR,
)

# ------------------------------------------------------------------ #
# PAGE CONFIG                                                          #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="PlasmaX Control Room",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0B0F1A; color: white; }
    .stMetric label { color: #8899AA !important; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# SESSION STATE INIT                                                   #
# ------------------------------------------------------------------ #
if "running" not in st.session_state:
    st.session_state.running = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

# ------------------------------------------------------------------ #
# SIDEBAR — REACTOR CONTROLS                                           #
# ------------------------------------------------------------------ #
st.sidebar.header("⚛️ Reactor Panel")

n_cycles = st.sidebar.slider("Simulation Cycles",   5,  50,  20)
# Minimum 60 steps enforced by generate_fusion_data; slider minimum matches.
steps    = st.sidebar.slider("Steps per Cycle",     60, 400, 200)
noise    = st.sidebar.slider("Noise Level",        0.0,  1.0,  0.2, step=0.05)
speed    = st.sidebar.slider("Refresh Speed (s)",  0.1,  2.0,  0.5, step=0.1)

st.sidebar.markdown("---")
col_start, col_stop = st.sidebar.columns(2)
start = col_start.button("▶ Start", use_container_width=True)
stop  = col_stop.button("⏹ Stop",  use_container_width=True)

if stop:
    st.session_state.stop_requested = True

# ------------------------------------------------------------------ #
# HEADER                                                               #
# ------------------------------------------------------------------ #
st.title("⚛️ PlasmaX CONTROL ROOM")
st.caption("Fusion Reactor Monitoring · AI Risk Prediction · Experiment Logging")
st.markdown("---")

# ------------------------------------------------------------------ #
# LIVE PANELS (placeholders)                                           #
# ------------------------------------------------------------------ #
status_bar = st.empty()

col1, col2, col3, col4 = st.columns(4)
m_temp  = col1.empty()
m_pres  = col2.empty()
m_mag   = col3.empty()
m_risk  = col4.empty()

chart_area = st.empty()
ai_panel   = st.empty()
alert_area = st.empty()

# ------------------------------------------------------------------ #
# HELPERS                                                              #
# ------------------------------------------------------------------ #
def _delta(prev: dict, key: str, current: float):
    """Return rounded delta vs previous cycle, or None on first cycle."""
    if prev[key] is None:
        return None
    return round(current - prev[key], 2)


def _temp_delta_color(temp: float, delta: float | None) -> str:
    """
    Temperature delta coloring is context-sensitive:
    - If we're approaching or above the warning threshold, rising is bad → 'inverse'
    - If we're well below the warning threshold, rising is fine  → 'off' (neutral)
    """
    if delta is None:
        return "off"
    from utils import TEMP_WARNING
    return "inverse" if temp > TEMP_WARNING * 0.80 else "off"


# ------------------------------------------------------------------ #
# MAIN LOOP                                                            #
# ------------------------------------------------------------------ #
if start:
    # Clear any leftover stop request from a previous run
    st.session_state.stop_requested = False
    st.session_state.running = True

    prev = {"temperature": None, "pressure": None, "magnetic_field": None}
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(n_cycles):
        # Re-read stop flag each iteration — set by the Stop button on a
        # separate Streamlit rerun that writes into shared session state.
        if st.session_state.stop_requested:
            status_bar.warning("⏹ Reactor manually halted.")
            break

        data       = generate_fusion_data(steps, noise)
        status, msg, alerts = check_plasma_safety(data)
        prediction = predict_instability(data)
        run_id     = _unique_run_id(session_ts, i)

        save_experiment(run_id, data, prediction, status)

        # ---- Status bar ----
        if status == "stable":
            status_bar.success(f"🟢 STABLE | Cycle {i + 1}/{n_cycles} | {run_id}")
        elif status == "warning":
            status_bar.warning(f"🟡 WARNING | Cycle {i + 1}/{n_cycles} | {run_id}")
        else:
            status_bar.error(f"🔴 CRITICAL | Cycle {i + 1}/{n_cycles} | {run_id}")

        # ---- Metrics with deltas + units ----
        t_now = data["temperature"].iloc[-1]
        p_now = data["pressure"].iloc[-1]
        b_now = data["magnetic_field"].iloc[-1]

        t_delta = _delta(prev, "temperature", t_now)
        m_temp.metric(
            "🌡 Temperature (keV)",
            f"{t_now:.1f}",
            delta=t_delta,
            delta_color=_temp_delta_color(t_now, t_delta),
        )
        m_pres.metric(
            "💨 Pressure (atm)",
            f"{p_now:.2f}",
            delta=_delta(prev, "pressure", p_now),
            delta_color="inverse",   # rising pressure is always bad
        )
        m_mag.metric(
            "🧲 Mag. Field (T)",
            f"{b_now:.3f}",
            delta=_delta(prev, "magnetic_field", b_now),
            delta_color="normal",    # rising field is always good (better confinement)
        )
        m_risk.metric(
            "🧠 AI Risk Score",
            f"{prediction['score']}/100",
        )

        prev.update({"temperature": t_now, "pressure": p_now, "magnetic_field": b_now})

        # ---- Chart ----
        # Pressure is plotted in its own subplot (row 1 col 1, secondary y-axis
        # approach is intentionally avoided: Plotly's make_subplots manages axes
        # internally and mixing manual yaxis= refs causes rendering conflicts).
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=("Temperature (keV)", "Pressure (atm)", "Magnetic Field (T)"),
            vertical_spacing=0.07,
        )

        t_axis = data["time"]

        fig.add_trace(
            go.Scatter(
                x=t_axis, y=data["temperature"],
                name="Temperature", line=dict(color="#FF6B35", width=1.5),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_axis, y=data["pressure"],
                name="Pressure", line=dict(color="#4ECDC4", width=1.5),
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_axis, y=data["magnetic_field"],
                name="Magnetic Field", line=dict(color="#A8E6CF", width=1.5),
                fill="tozeroy", fillcolor="rgba(168,230,207,0.08)",
            ),
            row=3, col=1,
        )

        # Safety threshold lines — each scoped to the correct subplot row
        fig.add_hline(
            y=900, line_dash="dot", line_color="red",
            annotation_text="Temp Critical", annotation_position="top right",
            row=1, col=1,
        )
        fig.add_hline(
            y=16, line_dash="dot", line_color="orange",
            annotation_text="Pres Critical", annotation_position="top right",
            row=2, col=1,
        )
        fig.add_hline(
            y=1.0, line_dash="dot", line_color="orange",
            annotation_text="Mag Warning", annotation_position="top right",
            row=3, col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=580,
            title=dict(text=f"Plasma Dynamics — {run_id}", font=dict(size=14)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=50, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,26,0.9)",
        )

        chart_area.plotly_chart(fig, use_container_width=True)

        # ---- AI panel ----
        risk_color = {"stable": "info", "warning": "warning", "critical": "error"}[
            prediction["status"]
        ]
        ai_text = (
            f"**🧠 AI Prediction: {prediction['status'].upper()}** — "
            f"Risk score {prediction['score']}/100  \n"
            f"{prediction['message']}"
        )
        getattr(ai_panel, risk_color)(ai_text)

        # ---- Per-alert warnings ----
        if alerts:
            with alert_area.container():
                for severity, detail in alerts:
                    icon = "🔴" if severity == "critical" else "🟡"
                    st.warning(f"{icon} **{severity.upper()}** — {detail}")
        else:
            alert_area.empty()

        time.sleep(speed)

    st.session_state.running = False
    st.session_state.stop_requested = False
    st.success("✅ Reactor session completed.")

# ------------------------------------------------------------------ #
# EXPERIMENT ARCHIVE                                                   #
# ------------------------------------------------------------------ #
st.markdown("---")
st.subheader("🧪 Experiment Archive")

ARCHIVE_LIMIT = 10
experiments = load_experiments(limit=ARCHIVE_LIMIT, include_data=False)

if experiments:
    stats = compute_session_stats(experiments)

    # Clarify that stats cover only the most recent N runs
    st.caption(f"Stats based on the {len(experiments)} most recent runs.")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total (shown)",   stats["total_runs"])
    s2.metric("✅ Stable",        stats["stable_count"])
    s3.metric("⚠️ Warnings",     stats["warning_count"])
    s4.metric("🔴 Critical",      stats["critical_count"])
    s5.metric("Avg Risk Score",  stats["avg_risk_score"])

    st.markdown("---")

    for exp in reversed(experiments[-ARCHIVE_LIMIT:]):
        summary = exp.get("summary", {})
        status_icon = {"stable": "🟢", "warning": "🟡", "critical": "🔴"}.get(
            exp["status"], "⚪"
        )
        with st.expander(
            f"{status_icon} {exp['run_id']} | {exp['status'].upper()} | "
            f"Risk: {exp['prediction']['score']}/100 | {exp['timestamp'][:19]}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Temp Max",  f"{summary.get('temp_max',  0):.1f}" if summary else "—")
            c1.metric("Temp Mean", f"{summary.get('temp_mean', 0):.1f}" if summary else "—")
            c2.metric("Pres Max",  f"{summary.get('pres_max',  0):.2f}" if summary else "—")
            c2.metric("Pres Mean", f"{summary.get('pres_mean', 0):.2f}" if summary else "—")
            c3.metric("Mag Min",   f"{summary.get('mag_min',   0):.3f}" if summary else "—")
            c3.metric("Mag Mean",  f"{summary.get('mag_mean',  0):.3f}" if summary else "—")
            st.info(
                f"**AI:** {exp['prediction']['message']}  "
                f"(score {exp['prediction']['score']}/100)"
            )
else:
    st.info("No experiments recorded yet. Start the reactor to begin.")