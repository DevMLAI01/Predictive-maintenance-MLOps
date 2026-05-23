"""Feature engineering for NASA CMAPSS FD001: RUL labeling, rolling windows, normalization."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 30
RUL_CAP = 125

# Confirmed by EDA: zero-variance + NaN-correlation sensors in FD001
DROP_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# 15 retained features after dropping the 6 above
SELECTED_SENSORS = [
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

N_FEATURES = len(SELECTED_SENSORS)  # 15


class FeatureEngineer:
    """Transform raw CMAPSS DataFrames into model-ready numpy arrays.

    Typical call order:
        fe = FeatureEngineer()
        train = fe.compute_rul(train_df)
        train = fe.select_sensors(train)
        test  = fe.select_sensors(test_df)
        train, test, scaler = fe.normalize(train, test)
        X_tr, y_tr = fe.create_sequences(train)
        X_te, y_te = fe.create_sequences(test, is_test=True, rul_series=rul_series)
        fe.save_processed(X_tr, y_tr, X_te, y_te)
    """

    def __init__(self, processed_dir: str = "data/processed") -> None:
        self.processed_dir = Path(processed_dir)
        self.scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------
    # Step 1 — RUL labeling (training set only)
    # ------------------------------------------------------------------

    def compute_rul(self, df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
        """Add a capped piecewise-linear RUL column to a training DataFrame.

        RUL[i] = min(max_cycle[engine] - current_cycle[i], rul_cap)
        """
        df = df.copy()
        max_cycles = df.groupby("engine_id")["cycle"].transform("max")
        rul = max_cycles - df["cycle"]
        df["rul"] = np.minimum(rul, rul_cap).astype(float)
        return df

    # ------------------------------------------------------------------
    # Step 2 — Sensor selection
    # ------------------------------------------------------------------

    def select_sensors(
        self,
        df: pd.DataFrame,
        drop_sensors: list[str] = DROP_SENSORS,
    ) -> pd.DataFrame:
        """Drop zero-variance and NaN-correlation sensors identified in EDA.

        Keeps: engine_id, cycle, rul (if present), and the 15 retained sensors.
        drop_sensors defaults to the 6 confirmed bad sensors for FD001.
        """
        df = df.copy()
        keep_cols = [c for c in df.columns if c not in drop_sensors]
        return df[keep_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 3 — Normalization (fit on train, transform both)
    # ------------------------------------------------------------------

    def normalize(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scaler_path: str = "data/processed/scaler.joblib",
    ) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
        """MinMaxScaler fitted on training sensors only (no data leakage).

        Saves the fitted scaler to *scaler_path* for use at inference time.
        """
        sensor_cols = [c for c in SELECTED_SENSORS if c in train_df.columns]

        scaler = MinMaxScaler()
        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
        test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

        self.scaler = scaler
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved → {scaler_path}")

        return train_df, test_df, scaler

    # ------------------------------------------------------------------
    # Step 4 — Sequence creation
    # ------------------------------------------------------------------

    def create_sequences(
        self,
        df: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        is_test: bool = False,
        rul_series: pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create rolling windows of shape (n_samples, window_size, n_features).

        Training mode (is_test=False):
            Slides a window of *window_size* cycles over each engine.
            y[i] = RUL at the last cycle of window i.

        Test mode (is_test=True):
            Takes only the LAST window per engine (final RUL prediction).
            Ground-truth y values come from *rul_series* (NASA-provided).

        Returns
        -------
        X : ndarray of shape (n_samples, window_size, n_features)
        y : ndarray of shape (n_samples,)
        """
        sensor_cols = [c for c in SELECTED_SENSORS if c in df.columns]
        sequences: list[np.ndarray] = []
        targets: list[float] = []

        if is_test:
            if rul_series is None:
                raise ValueError("rul_series required for test-set sequencing.")
            for engine_id, group in df.groupby("engine_id"):
                sensors = group[sensor_cols].values
                if len(sensors) < window_size:
                    # Pad with zeros on the left if engine has fewer cycles than window
                    pad = np.zeros((window_size - len(sensors), len(sensor_cols)))
                    sensors = np.vstack([pad, sensors])
                window = sensors[-window_size:]
                sequences.append(window)
                # rul_series is 0-indexed; engine_id is 1-indexed
                targets.append(float(rul_series.iloc[int(engine_id) - 1]))
        else:
            for _, group in df.groupby("engine_id"):
                sensors = group[sensor_cols].values
                ruls = group["rul"].values
                for i in range(len(sensors) - window_size + 1):
                    sequences.append(sensors[i : i + window_size])
                    targets.append(ruls[i + window_size - 1])

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        return X, y

    # ------------------------------------------------------------------
    # Step 5 — Persist to disk
    # ------------------------------------------------------------------

    def save_processed(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Save all arrays to data/processed/ as .npy files."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in [
            ("X_train", X_train),
            ("y_train", y_train),
            ("X_test", X_test),
            ("y_test", y_test),
        ]:
            path = self.processed_dir / f"{name}.npy"
            np.save(path, arr)
            print(f"  {name}: {arr.shape}  →  {path}")
        print("Processed arrays saved.")
