"""Streamlit dashboard for Predictive Equipment Failure — RUL prediction.

Connects to a live Vertex AI Endpoint and visualises:
  - Per-engine RUL + failure probability + maintenance alert
  - Sensor health time-series (last 30 cycles)
  - Fleet-wide RUL distribution and at-risk engine table
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.cloud import aiplatform, storage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "groovy-rope-496901-d2")
REGION = os.environ.get("GCP_REGION", "us-central1")
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID", "1041396390934085632")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "predictive-maintenance-artifacts")

_N_ENGINES = 100
_WINDOW = 30
_N_FEATURES = 15

SENSOR_NAMES = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

KEY_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_all_test_data() -> np.ndarray:
    """Load X_test.npy (100, 30, 15) — normalized — from GCS or local fallback."""
    local_path = Path(__file__).parent.parent / "data" / "processed" / "X_test.npy"

    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("data/processed/X_test.npy")
        if blob.exists():
            raw = blob.download_as_bytes()
            return np.load(io.BytesIO(raw))
    except Exception:
        pass

    if local_path.exists():
        return np.load(str(local_path))

    st.warning("Test data not found — using synthetic data for demonstration.")
    rng = np.random.default_rng(42)
    return rng.random((_N_ENGINES, _WINDOW, _N_FEATURES)).astype(np.float32)


@st.cache_data(show_spinner=False)
def load_scaler():
    """Load MinMaxScaler from GCS or local fallback."""
    local_path = Path(__file__).parent.parent / "model_artifacts" / "scaler.joblib"

    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("models/v1/scaler.joblib")
        if blob.exists():
            raw = blob.download_as_bytes()
            return joblib.load(io.BytesIO(raw))
    except Exception:
        pass

    if local_path.exists():
        return joblib.load(str(local_path))

    return None


def to_raw(window_scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform a (30, 15) scaled window to raw sensor values."""
    scaler = load_scaler()
    if scaler is None:
        return window_scaled
    return scaler.inverse_transform(window_scaled)


def load_test_data(engine_id: int, up_to_cycle: int) -> np.ndarray:
    """Return the normalized 30-cycle window for engine_id (1-indexed)."""
    X = load_all_test_data()
    window = X[engine_id - 1].copy()
    end = min(up_to_cycle, _WINDOW)
    if end < _WINDOW:
        window = np.vstack([np.zeros((_WINDOW - end, _N_FEATURES)), window[:end]])
    return window


@st.cache_data(ttl=60, show_spinner=False)
def call_vertex_endpoint(engine_id: int, up_to_cycle: int) -> dict[str, Any] | None:
    """Inverse-transform then call the Vertex AI Endpoint. Cached for 60 seconds."""
    window_scaled = load_test_data(engine_id, up_to_cycle)
    window_raw = to_raw(window_scaled)
    try:
        aiplatform.init(project=PROJECT_ID, location=REGION)
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)
        response = endpoint.predict(instances=[window_raw.tolist()])
        return response.predictions[0]
    except Exception as exc:
        st.warning(f"Endpoint unavailable: {exc}")
        return None


@st.cache_data(show_spinner="Running fleet predictions — this takes ~30 seconds…")
def get_fleet_predictions() -> pd.DataFrame:
    """Inverse-transform then predict all 100 engines. Cached for the session."""
    X = load_all_test_data()
    scaler = load_scaler()
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    records: list[dict] = []
    for i in range(0, _N_ENGINES, 10):
        batch_scaled = X[i : i + 10]  # (10, 30, 15)
        if scaler is not None:
            flat = scaler.inverse_transform(batch_scaled.reshape(-1, _N_FEATURES))
            batch_raw = flat.reshape(-1, _WINDOW, _N_FEATURES)
        else:
            batch_raw = batch_scaled
        try:
            response = endpoint.predict(instances=batch_raw.tolist())
            for j, pred in enumerate(response.predictions):
                records.append(
                    {
                        "engine_id": i + j + 1,
                        "rul": pred["rul"],
                        "failure_prob": pred["failure_prob"],
                        "alert": pred["alert"],
                        "confidence": pred["confidence"],
                    }
                )
        except Exception:
            for j in range(len(batch_raw)):
                records.append(
                    {
                        "engine_id": i + j + 1,
                        "rul": None,
                        "failure_prob": None,
                        "alert": False,
                        "confidence": "unknown",
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RUL Dashboard", page_icon="🔧", layout="wide")

# ---------------------------------------------------------------------------
# SECTION 1: Header
# ---------------------------------------------------------------------------

st.title("🔧 Predictive Equipment Failure — RUL Dashboard")
st.caption("NASA CMAPSS FD001 | PyTorch LSTM | Google Cloud Vertex AI")
st.divider()

# ---------------------------------------------------------------------------
# SECTION 2: Engine Selector + Live Prediction
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Engine Selector")
    engine_id = st.selectbox("Engine ID", list(range(1, _N_ENGINES + 1)), index=0)
    cycle = st.slider("Up to Cycle", min_value=1, max_value=_WINDOW, value=_WINDOW)
    st.caption("Selects the 30-cycle window ending at the chosen cycle.")

st.subheader(f"Engine {engine_id} — Live Prediction")

pred = call_vertex_endpoint(engine_id, cycle)

if pred is not None:
    rul: int = pred["rul"]
    failure_prob: float = pred["failure_prob"]
    alert: bool = pred["alert"]
    confidence: str = pred["confidence"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Remaining Useful Life", f"{rul} cycles")
    m2.metric("Failure Probability", f"{failure_prob:.1%}")
    m3.metric("Confidence", confidence.capitalize())

    st.progress(min(float(failure_prob), 1.0), text=f"Failure risk: {failure_prob:.1%}")

    if alert:
        st.error("⚠️ Schedule maintenance — failure risk is high")
    else:
        st.success("✅ No immediate maintenance required")
else:
    st.info("Waiting for endpoint response…")

# ---------------------------------------------------------------------------
# SECTION 3: Sensor Health Panel
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Sensor Health Panel")

engine_window = load_all_test_data()[engine_id - 1]  # normalized (30, 15)
cycles_x = list(range(1, _WINDOW + 1))

sensor_cols = st.columns(3)
for idx, sensor_name in enumerate(KEY_SENSORS):
    sensor_feature_idx = SENSOR_NAMES.index(sensor_name)
    values = engine_window[:, sensor_feature_idx]
    mean_val = float(values.mean())
    std_val = float(values.std())
    is_anomalous = std_val > 0 and abs(float(values[-1]) - mean_val) > 2 * std_val

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cycles_x,
            y=values.tolist(),
            mode="lines+markers",
            line=dict(color="crimson" if is_anomalous else "steelblue", width=2),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"{sensor_name}{'  ⚠️' if is_anomalous else ''}", font=dict(size=13)
        ),
        xaxis_title="Cycle",
        yaxis_title="Norm. value",
        height=220,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
    )
    sensor_cols[idx % 3].plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# SECTION 4: Fleet Overview
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Fleet Overview")

if st.button("Load Fleet Predictions", type="primary"):
    fleet_df = get_fleet_predictions()
    fleet_valid = fleet_df.dropna(subset=["rul"])

    n_alert = int(fleet_valid["alert"].sum())
    st.info(f"**{n_alert} engines** require maintenance attention")

    fig_fleet = go.Figure()
    fig_fleet.add_trace(
        go.Scatter(
            x=fleet_valid["engine_id"].tolist(),
            y=fleet_valid["rul"].tolist(),
            mode="markers",
            marker=dict(
                color=fleet_valid["failure_prob"].tolist(),
                colorscale="RdYlGn_r",
                size=10,
                showscale=True,
                colorbar=dict(title="Failure Prob"),
            ),
            text=fleet_valid.apply(
                lambda r: (
                    f"Engine {r.engine_id}<br>RUL: {r.rul}<br>Prob: {r.failure_prob:.2%}"
                ),
                axis=1,
            ).tolist(),
            hoverinfo="text",
        )
    )
    fig_fleet.update_layout(
        title="Fleet RUL Distribution",
        xaxis_title="Engine ID",
        yaxis_title="Predicted RUL (cycles)",
        height=400,
    )
    st.plotly_chart(fig_fleet, use_container_width=True)

    st.subheader("Top 10 Engines — Lowest Predicted RUL")
    top10 = (
        fleet_valid.sort_values("rul")
        .head(10)[["engine_id", "rul", "failure_prob", "alert", "confidence"]]
        .rename(
            columns={
                "engine_id": "Engine",
                "rul": "RUL (cycles)",
                "failure_prob": "Failure Prob",
                "alert": "Alert",
                "confidence": "Confidence",
            }
        )
        .reset_index(drop=True)
    )
    top10["Failure Prob"] = top10["Failure Prob"].map("{:.1%}".format)
    st.dataframe(top10, use_container_width=True)
