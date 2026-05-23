"""KFP v2 pipeline components for the RUL predictive-maintenance pipeline.

Each component is fully self-contained — all imports live inside the
function body because KFP serialises and runs each in its own container.

GCS multi-file artifact convention used throughout:
  artifact.uri  →  gs://bucket/pipeline_root/.../artifact_name
  Files are written/read as  artifact.uri + "/filename"
"""

from kfp.dsl import (
    Artifact,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
    component,
)
from typing import NamedTuple


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — Data Validation
# ═══════════════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-storage", "numpy<2.0", "pandas"],
)
def validate_data(
    gcs_data_uri: str,
    validation_report: Output[Metrics],
) -> NamedTuple("Outputs", [("is_valid", bool), ("n_engines", int)]):
    import json
    import os
    import tempfile
    from collections import namedtuple

    import pandas as pd
    from google.cloud import storage as gcs

    COLUMNS = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]
    EXPECTED_SENSORS = [f"sensor_{i}" for i in range(1, 22)]

    def _download(src: str, dst: str) -> None:
        without_scheme = src[5:]
        bucket_name, _, blob_path = without_scheme.partition("/")
        gcs.Client().bucket(bucket_name).blob(blob_path).download_to_filename(dst)

    with tempfile.TemporaryDirectory() as tmp:
        local_train = os.path.join(tmp, "train_FD001.txt")
        print(f"Downloading {gcs_data_uri}/train_FD001.txt")
        _download(f"{gcs_data_uri}/train_FD001.txt", local_train)

        df = pd.read_csv(
            local_train,
            sep=r"\s+",
            header=None,
            names=COLUMNS,
            engine="python",
        )[COLUMNS]
        df["engine_id"] = df["engine_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

    no_nulls = bool(df.isnull().sum().sum() == 0)
    sensors_ok = all(s in df.columns for s in EXPECTED_SENSORS)
    n_engines = int(df["engine_id"].nunique())
    engine_count_ok = n_engines > 50
    is_valid = no_nulls and sensors_ok and engine_count_ok

    print(
        json.dumps(
            {
                "is_valid": is_valid,
                "n_engines": n_engines,
                "no_nulls": no_nulls,
                "sensors_ok": sensors_ok,
                "rows": len(df),
            },
            indent=2,
        )
    )

    validation_report.log_metric("is_valid", int(is_valid))
    validation_report.log_metric("n_engines", n_engines)
    validation_report.log_metric("no_nulls", int(no_nulls))
    validation_report.log_metric("rows", len(df))

    Outputs = namedtuple("Outputs", ["is_valid", "n_engines"])
    return Outputs(is_valid=is_valid, n_engines=n_engines)


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-storage",
        "numpy<2.0",
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
def engineer_features(
    gcs_data_uri: str,
    window_size: int,
    rul_cap: int,
    processed_data: Output[Dataset],
) -> NamedTuple("Outputs", [("n_train_samples", int), ("n_features", int)]):
    import io
    import os
    import tempfile
    from collections import namedtuple

    import joblib
    import numpy as np
    import pandas as pd
    from google.cloud import storage
    from sklearn.preprocessing import MinMaxScaler

    COLUMNS = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]
    DROP_SENSORS = [
        "sensor_1",
        "sensor_5",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    ]
    SELECTED_SENSORS = [
        f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" not in DROP_SENSORS
    ]

    def _download(gcs_uri: str, local_path: str) -> None:
        uri = gcs_uri[5:]
        bucket_name, _, blob_path = uri.partition("/")
        storage.Client().bucket(bucket_name).blob(blob_path).download_to_filename(
            local_path
        )

    def _upload_bytes(data: bytes, gcs_uri: str) -> None:
        uri = gcs_uri[5:]
        bucket_name, _, blob_path = uri.partition("/")
        storage.Client().bucket(bucket_name).blob(blob_path).upload_from_string(data)

    with tempfile.TemporaryDirectory() as tmp:
        local_train = os.path.join(tmp, "train_FD001.txt")
        _download(f"{gcs_data_uri}/train_FD001.txt", local_train)

        df = pd.read_csv(
            local_train, sep=r"\s+", header=None, names=COLUMNS, engine="python"
        )[COLUMNS]
        df["engine_id"] = df["engine_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

    # RUL labeling (piecewise linear, capped)
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["rul"] = np.minimum(max_cycles - df["cycle"], rul_cap).astype(float)

    # Sensor selection
    keep_cols = ["engine_id", "cycle", "rul"] + SELECTED_SENSORS
    df = df[keep_cols].reset_index(drop=True)

    # Train / val split by engine_id (80/20, no leakage)
    engine_ids = sorted(df["engine_id"].unique())
    split_idx = int(len(engine_ids) * 0.8)
    train_ids = set(engine_ids[:split_idx])
    val_ids = set(engine_ids[split_idx:])

    train_df = df[df["engine_id"].isin(train_ids)].copy()
    val_df = df[df["engine_id"].isin(val_ids)].copy()

    # Normalize: fit on train, transform both
    scaler = MinMaxScaler()
    train_df[SELECTED_SENSORS] = scaler.fit_transform(train_df[SELECTED_SENSORS])
    val_df[SELECTED_SENSORS] = scaler.transform(val_df[SELECTED_SENSORS])

    def _make_sequences(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        seqs, targets = [], []
        for _, grp in frame.groupby("engine_id"):
            sensors = grp[SELECTED_SENSORS].values
            ruls = grp["rul"].values
            for i in range(len(sensors) - window_size + 1):
                seqs.append(sensors[i : i + window_size])
                targets.append(ruls[i + window_size - 1])
        return (
            np.array(seqs, dtype=np.float32),
            np.array(targets, dtype=np.float32),
        )

    X_train, y_train = _make_sequences(train_df)
    X_val, y_val = _make_sequences(val_df)
    n_features = X_train.shape[2]

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape}   y_val  : {y_val.shape}")
    print(f"n_features={n_features}")

    base_uri = processed_data.uri
    # Upload arrays as npy bytes
    for name, arr in [
        ("X_train.npy", X_train),
        ("y_train.npy", y_train),
        ("X_val.npy", X_val),
        ("y_val.npy", y_val),
    ]:
        buf = io.BytesIO()
        np.save(buf, arr)
        _upload_bytes(buf.getvalue(), f"{base_uri}/{name}")
        print(f"  Uploaded {name}")

    # Scaler
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        scaler_path = f.name
    joblib.dump(scaler, scaler_path)
    with open(scaler_path, "rb") as f:
        _upload_bytes(f.read(), f"{base_uri}/scaler.joblib")
    os.unlink(scaler_path)
    print("  Uploaded scaler.joblib")

    processed_data.metadata["n_train_samples"] = str(len(X_train))
    processed_data.metadata["n_features"] = str(n_features)

    Outputs = namedtuple("Outputs", ["n_train_samples", "n_features"])
    return Outputs(n_train_samples=int(len(X_train)), n_features=int(n_features))


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — Model Training  (PyTorch base image)
# ═══════════════════════════════════════════════════════════════════════════
@component(
    base_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
    packages_to_install=[
        "google-cloud-storage",
        "numpy<2.0",
        "scikit-learn",
        "joblib",
    ],
)
def train_model(
    processed_data: Input[Dataset],
    gcs_model_uri: str,
    epochs: int,
    batch_size: int,
    model_artifact: Output[Model],
) -> NamedTuple("Outputs", [("val_rmse", float), ("nasa_score", float)]):
    import io
    import math
    import os
    import tempfile
    from collections import namedtuple

    import joblib
    import numpy as np
    import torch
    import torch.nn as nn
    from google.cloud import storage
    from torch.utils.data import DataLoader, TensorDataset

    # ------------------------------------------------------------------
    # Inline model definition (mirrors src/model/lstm.py)
    # ------------------------------------------------------------------
    class RULPredictor(nn.Module):
        def __init__(self, n_features=15, hidden_size=128, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                n_features,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, 64)
            self.relu = nn.ReLU()
            self.rul_head = nn.Linear(64, 1)
            self.failure_head = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            out = self.relu(self.fc1(out))
            return {
                "rul": self.rul_head(out),
                "failure_prob": self.sigmoid(self.failure_head(out)),
            }

    # ------------------------------------------------------------------
    # GCS helpers
    # ------------------------------------------------------------------
    def _download_bytes(gcs_uri: str) -> bytes:
        uri = gcs_uri[5:]
        bucket_name, _, blob_path = uri.partition("/")
        return storage.Client().bucket(bucket_name).blob(blob_path).download_as_bytes()

    def _upload_file(local_path: str, gcs_uri: str) -> None:
        uri = gcs_uri[5:]
        bucket_name, _, blob_path = uri.partition("/")
        storage.Client().bucket(bucket_name).blob(blob_path).upload_from_filename(
            local_path
        )

    def _upload_bytes(data: bytes, gcs_uri: str) -> None:
        uri = gcs_uri[5:]
        bucket_name, _, blob_path = uri.partition("/")
        storage.Client().bucket(bucket_name).blob(blob_path).upload_from_string(data)

    # ------------------------------------------------------------------
    # Load processed data from artifact URI
    # ------------------------------------------------------------------
    base_uri = processed_data.uri
    print(f"Loading data from {base_uri}")

    def _load_npy(name: str) -> np.ndarray:
        raw = _download_bytes(f"{base_uri}/{name}")
        return np.load(io.BytesIO(raw))

    X_train = _load_npy("X_train.npy")
    y_train = _load_npy("y_train.npy")
    X_val = _load_npy("X_val.npy")
    y_val = _load_npy("y_val.npy")
    print(f"X_train {X_train.shape}  X_val {X_val.shape}")

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        scaler_local = f.name
    with open(scaler_local, "wb") as f:
        f.write(_download_bytes(f"{base_uri}/scaler.joblib"))
    scaler = joblib.load(scaler_local)
    os.unlink(scaler_local)

    # ------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    n_features = X_train.shape[2]
    model = RULPredictor(n_features=n_features).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    FAILURE_HORIZON = 30

    best_val_rmse = float("inf")
    patience_counter = 0
    EARLY_STOP = 15

    with tempfile.TemporaryDirectory() as tmp:
        best_ckpt = os.path.join(tmp, "best_model.pt")

        for epoch in range(1, epochs + 1):
            # — Train —
            model.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb)
                rul_loss = mse_loss(out["rul"].squeeze(), yb)
                failure_label = (yb <= FAILURE_HORIZON).float()
                cls_loss = bce_loss(out["failure_prob"].squeeze(), failure_label)
                loss = rul_loss + 0.3 * cls_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(Xb)
            train_loss /= len(train_ds)

            # — Validate —
            model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(device)
                    out = model(Xb)
                    preds.extend(out["rul"].squeeze().cpu().tolist())
                    actuals.extend(yb.tolist())

            preds_arr = np.array(preds)
            actuals_arr = np.array(actuals)
            val_rmse = math.sqrt(np.mean((preds_arr - actuals_arr) ** 2))
            scheduler.step(val_rmse)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_rmse={val_rmse:.4f}"
                )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), best_ckpt)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Reload best weights for NASA score
        model.load_state_dict(
            torch.load(best_ckpt, map_location=device, weights_only=True)
        )
        model.eval()
        preds_best, actuals_best = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                out = model(Xb.to(device))
                preds_best.extend(out["rul"].squeeze().cpu().tolist())
                actuals_best.extend(yb.tolist())

        p = np.array(preds_best)
        a = np.array(actuals_best)
        d = p - a
        nasa_score = float(
            np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
        )

        print(f"\nBest val RMSE : {best_val_rmse:.4f}")
        print(f"NASA score    : {nasa_score:.2f}")

        # Upload model to artifact URI
        art_uri = model_artifact.uri
        _upload_file(best_ckpt, f"{art_uri}/rul_predictor_v1.pt")

        # Upload scaler to artifact URI (needed by evaluate_model)
        scaler_upload_path = os.path.join(tmp, "scaler.joblib")
        joblib.dump(scaler, scaler_upload_path)
        _upload_file(scaler_upload_path, f"{art_uri}/scaler.joblib")

        # Also overwrite the primary GCS model location (used by TorchServe)
        _upload_file(best_ckpt, f"{gcs_model_uri}/rul_predictor_v1.pt")
        print(f"Saved model → gs artifact + {gcs_model_uri}/rul_predictor_v1.pt")

    model_artifact.metadata["val_rmse"] = str(round(best_val_rmse, 4))
    model_artifact.metadata["nasa_score"] = str(round(nasa_score, 2))

    Outputs = namedtuple("Outputs", ["val_rmse", "nasa_score"])
    return Outputs(val_rmse=round(best_val_rmse, 4), nasa_score=round(nasa_score, 2))


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 4 — Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10",
    packages_to_install=[
        "torch",
        "numpy<2.0",
        "scikit-learn",
        "joblib",
        "pandas",
        "google-cloud-storage",
    ],
)
def evaluate_model(
    model_artifact: Input[Model],
    test_data_uri: str,
    rmse_threshold: float,
    evaluation_report: Output[Metrics],
) -> NamedTuple("Outputs", [("test_rmse", float), ("passes_threshold", bool)]):
    import math
    import os
    import tempfile
    from collections import namedtuple

    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from google.cloud import storage

    # Inline model definition (must match train_model)
    class RULPredictor(nn.Module):
        def __init__(self, n_features=15, hidden_size=128, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                n_features,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, 64)
            self.relu = nn.ReLU()
            self.rul_head = nn.Linear(64, 1)
            self.failure_head = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            out = self.relu(self.fc1(out))
            return {
                "rul": self.rul_head(out),
                "failure_prob": self.sigmoid(self.failure_head(out)),
            }

    COLUMNS = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]
    DROP_SENSORS = [
        "sensor_1",
        "sensor_5",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    ]
    SELECTED_SENSORS = [
        f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" not in DROP_SENSORS
    ]
    WINDOW_SIZE = 30

    def _download(gcs_uri: str, local_path: str) -> None:
        uri = gcs_uri[5:]
        bucket_name, _, blob_path = uri.partition("/")
        storage.Client().bucket(bucket_name).blob(blob_path).download_to_filename(
            local_path
        )

    def _download_bytes(gcs_uri: str) -> bytes:
        uri = gcs_uri[5:]
        bucket_name, _, blob_path = uri.partition("/")
        return storage.Client().bucket(bucket_name).blob(blob_path).download_as_bytes()

    with tempfile.TemporaryDirectory() as tmp:
        # Download model and scaler from artifact URI
        art_uri = model_artifact.uri
        model_local = os.path.join(tmp, "model.pt")
        scaler_local = os.path.join(tmp, "scaler.joblib")
        _download(f"{art_uri}/rul_predictor_v1.pt", model_local)
        _download(f"{art_uri}/scaler.joblib", scaler_local)

        # Download test data
        test_local = os.path.join(tmp, "test_FD001.txt")
        rul_local = os.path.join(tmp, "RUL_FD001.txt")
        _download(f"{test_data_uri}/test_FD001.txt", test_local)
        _download(f"{test_data_uri}/RUL_FD001.txt", rul_local)

        scaler = joblib.load(scaler_local)

        # Parse test data
        test_df = pd.read_csv(
            test_local, sep=r"\s+", header=None, names=COLUMNS, engine="python"
        )[COLUMNS]
        test_df["engine_id"] = test_df["engine_id"].astype(int)
        rul_series = pd.read_csv(rul_local, header=None, names=["rul"])["rul"].astype(
            float
        )

        # Feature engineering (no RUL labeling for test)
        keep_cols = ["engine_id", "cycle"] + SELECTED_SENSORS
        test_df = test_df[keep_cols].reset_index(drop=True)
        test_df[SELECTED_SENSORS] = scaler.transform(test_df[SELECTED_SENSORS])

        # Create sequences (last window per engine)
        X_test_list, y_test_list = [], []
        for eng_id, grp in test_df.groupby("engine_id"):
            sensors = grp[SELECTED_SENSORS].values
            if len(sensors) < WINDOW_SIZE:
                pad = np.zeros((WINDOW_SIZE - len(sensors), len(SELECTED_SENSORS)))
                sensors = np.vstack([pad, sensors])
            X_test_list.append(sensors[-WINDOW_SIZE:])
            y_test_list.append(float(rul_series.iloc[int(eng_id) - 1]))

        X_test = np.array(X_test_list, dtype=np.float32)
        y_test = np.array(y_test_list, dtype=np.float32)

        # Load model
        n_features = X_test.shape[2]
        model = RULPredictor(n_features=n_features)
        state = torch.load(model_local, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Inference
        with torch.no_grad():
            preds = model(torch.tensor(X_test))["rul"].squeeze().numpy()

        # Metrics
        preds_clipped = np.clip(preds, 0, 125)
        test_rmse = float(math.sqrt(np.mean((preds_clipped - y_test) ** 2)))
        test_mae = float(np.mean(np.abs(preds_clipped - y_test)))
        d = preds_clipped - y_test
        nasa_score = float(
            np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
        )
        passes_threshold = test_rmse < rmse_threshold

        print(f"Test RMSE      : {test_rmse:.4f}")
        print(f"Test MAE       : {test_mae:.4f}")
        print(f"NASA score     : {nasa_score:.2f}")
        print(f"Passes ({rmse_threshold}): {passes_threshold}")

    evaluation_report.log_metric("test_rmse", test_rmse)
    evaluation_report.log_metric("test_mae", test_mae)
    evaluation_report.log_metric("nasa_score", nasa_score)
    evaluation_report.log_metric("rmse_threshold", rmse_threshold)
    evaluation_report.log_metric("passes_threshold", int(passes_threshold))

    Outputs = namedtuple("Outputs", ["test_rmse", "passes_threshold"])
    return Outputs(
        test_rmse=round(test_rmse, 4), passes_threshold=bool(passes_threshold)
    )


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 5 — Model Deployment
# ═══════════════════════════════════════════════════════════════════════════
@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model(
    model_resource_name: str,
    endpoint_display_name: str,
    machine_type: str,
    deployed_model_id: Output[Artifact],
) -> NamedTuple("Outputs", [("endpoint_uri", str)]):
    from collections import namedtuple

    from google.cloud import aiplatform

    # Derive project and location from the model resource name
    # format: projects/{project}/locations/{location}/models/{model_id}
    parts = model_resource_name.split("/")
    project = parts[1]
    location = parts[3]

    aiplatform.init(project=project, location=location)

    # Get or create endpoint
    existing = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
    )
    if existing:
        endpoint = existing[0]
        print(f"Reusing endpoint: {endpoint.resource_name}")
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            labels={"project": "predictive-maintenance", "managed-by": "kfp-pipeline"},
        )
        print(f"Created endpoint: {endpoint.resource_name}")

    # Deploy model
    model = aiplatform.Model(model_resource_name)
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=2,
        traffic_percentage=100,
        sync=True,
    )

    endpoint_uri = (
        f"https://{location}-aiplatform.googleapis.com/v1/{endpoint.resource_name}"
    )
    print(f"Endpoint URI: {endpoint_uri}")

    deployed_model_id.uri = endpoint.resource_name
    deployed_model_id.metadata["endpoint_uri"] = endpoint_uri
    deployed_model_id.metadata["endpoint_resource_name"] = endpoint.resource_name

    Outputs = namedtuple("Outputs", ["endpoint_uri"])
    return Outputs(endpoint_uri=endpoint_uri)
