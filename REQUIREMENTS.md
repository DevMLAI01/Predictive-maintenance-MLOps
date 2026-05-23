# REQUIREMENTS.md
# Predictive Equipment Failure — PyTorch LSTM + Google Cloud Vertex AI
# MLOps Portfolio Project | Claude Code Build Scaffold
# Author: Saurabh | Environment: Google Cloud Shell (gcloud CLI pre-installed)
# Execution: One phase at a time — confirm ✅ before proceeding to next phase

---

## PROJECT OVERVIEW

Build a production-grade predictive maintenance MLOps pipeline that predicts
Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS dataset.
The model is a PyTorch LSTM trained locally in Google Colab (free T4 GPU),
then deployed through a full Vertex AI pipeline on Google Cloud Platform.

**Target audience for this portfolio piece:**
- Google Cloud Forward Deployed Engineer roles
- Big 4 consulting firms (EY, Deloitte, Accenture) — Industry 4.0 / IoT practice
- AI Architect / Principal AI Engineer roles

**Key design decisions:**
- No local Docker installation required — use Vertex AI pre-built PyTorch containers
- All GCP infrastructure provisioned via Terraform (IaC signal for FDE roles)
- Training done on Google Colab free tier to preserve GCP trial credits
- Streamlit dashboard deployed on Cloud Run with min-instances=0 (zero idle cost)
- GCP billing alert set at $30 — stays well within $300 trial credit window

---

## REPOSITORY STRUCTURE

```
predictive-maintenance-vertex/
├── REQUIREMENTS.md                  # This file — Claude Code scaffold
├── README.md                        # GitHub portfolio README (Phase 10)
├── .env.example                     # Environment variable template
├── .gitignore
│
├── data/
│   ├── raw/                         # NASA CMAPSS raw files (not committed to git)
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   └── processed/                   # Engineered features, normalized arrays
│       ├── X_train.npy
│       ├── y_train.npy
│       ├── X_test.npy
│       └── y_test.npy
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis (Phase 2)
│   └── 02_model_training.ipynb      # PyTorch training on Colab GPU (Phase 3)
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                # CMAPSS download + parse
│   │   ├── features.py              # RUL labeling, rolling window, normalization
│   │   └── validate.py             # Data quality checks
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── lstm.py                  # PyTorch LSTM architecture
│   │   ├── train.py                 # Training loop, early stopping, checkpointing
│   │   ├── evaluate.py              # RMSE, MAE, NASA score function
│   │   └── predict.py               # Inference + RUL output formatting
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   └── predictor.py             # Vertex AI custom predictor class
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── components.py            # KFP v2 pipeline components
│       └── pipeline.py              # Vertex AI Pipeline definition
│
├── terraform/
│   ├── main.tf                      # GCP provider, project config
│   ├── variables.tf                 # Project ID, region, bucket name
│   ├── outputs.tf                   # Endpoint URL, bucket URI
│   ├── gcs.tf                       # Cloud Storage bucket
│   ├── artifact_registry.tf         # Container registry
│   ├── vertex.tf                    # Vertex AI resources
│   └── cloudrun.tf                  # Streamlit dashboard service
│
├── dashboard/
│   ├── app.py                       # Streamlit dashboard
│   ├── requirements.txt             # Dashboard-specific deps
│   └── Dockerfile                   # Cloud Run container (built via Cloud Build)
│
├── cloudbuild/
│   └── cloudbuild.yaml              # Serverless container build (no local Docker)
│
├── scripts/
│   ├── upload_to_gcs.sh             # Upload model artifact to GCS
│   ├── run_pipeline.py              # Trigger Vertex AI Pipeline
│   └── undeploy_endpoint.sh         # Cost guardrail: undeploy when not demoing
│
├── requirements.txt                 # Project-level Python deps
└── setup.py                         # Package installation
```

---

## ENVIRONMENT SETUP

```bash
# Python version
python >= 3.10

# GCP environment
# Execute ALL gcloud commands in Google Cloud Shell (browser-based)
# Cloud Shell URL: https://shell.cloud.google.com
# Cloud Shell has gcloud, docker, terraform, python pre-installed
# No local installation required

# Environment variables (.env — never commit)
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCS_BUCKET_NAME=predictive-maintenance-artifacts
VERTEX_ENDPOINT_ID=your-endpoint-id
STREAMLIT_ENDPOINT_URL=https://your-cloud-run-url
```

---

## PHASE EXECUTION INSTRUCTIONS FOR CLAUDE CODE

Each phase below is self-contained. At the end of each phase:
1. Claude Code will print: `✅ PHASE N COMPLETE — confirm to proceed`
2. You verify outputs manually before typing: `proceed to phase N+1`
3. Never auto-chain phases — each requires human confirmation

---

# ═══════════════════════════════════════════════════════════
# PHASE 1 — PROJECT SCAFFOLDING & GCP INITIALIZATION
# Estimated time: 30–45 minutes
# ═══════════════════════════════════════════════════════════

## Phase 1 Goal
Create the full repository structure, initialize GCP project,
enable required APIs, and set up Cloud Storage bucket.
All gcloud commands run in Google Cloud Shell.

## Phase 1 Tasks

### 1.1 Create repository structure
```
ACTION: Create all directories and empty __init__.py files
        as defined in REPOSITORY STRUCTURE above.
        Create .gitignore with entries for:
        - data/raw/
        - .env
        - __pycache__/
        - *.pyc
        - .pytest_cache/
        - model_artifacts/
        - *.pt
        Create .env.example with all required variable names (no values)
```

### 1.2 Create requirements.txt (project root)
```
FILE: requirements.txt
CONTENTS:
# Core ML
torch>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Google Cloud
google-cloud-aiplatform>=1.38.0
google-cloud-storage>=2.10.0
kfp>=2.4.0
google-cloud-pipeline-components>=2.0.0   # NOTE: package was renamed; kfp-google-cloud-pipeline-components is wrong

# Serving
fastapi>=0.104.0
uvicorn>=0.24.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0.0
joblib>=1.3.0
tqdm>=4.66.0
```

### 1.3 GCP project initialization (run in Cloud Shell)
```bash
# Set your project
gcloud config set project $GCP_PROJECT_ID

# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  compute.googleapis.com

# Create GCS bucket
gsutil mb -l us-central1 gs://$GCS_BUCKET_NAME

# Create subdirectory structure in bucket
gsutil cp /dev/null gs://$GCS_BUCKET_NAME/data/.keep
gsutil cp /dev/null gs://$GCS_BUCKET_NAME/models/.keep
gsutil cp /dev/null gs://$GCS_BUCKET_NAME/pipelines/.keep
gsutil cp /dev/null gs://$GCS_BUCKET_NAME/artifacts/.keep

# Set billing alert via gcloud (IMPORTANT — do this before any Vertex jobs)
gcloud billing budgets create \
  --billing-account=$BILLING_ACCOUNT_ID \
  --display-name="predictive-maintenance-budget" \
  --budget-amount=30USD \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9
```

### 1.4 Create Terraform variables file
```
FILE: terraform/variables.tf
Define variables:
- project_id (string)
- region (string, default "us-central1")
- bucket_name (string)
- artifact_registry_name (string, default "predictive-maintenance")
- vertex_endpoint_display_name (string)
```

### 1.5 Verify phase completion
```
VERIFICATION CHECKLIST:
[ ] All directories created per REPOSITORY STRUCTURE
[ ] requirements.txt exists at project root
[ ] .gitignore created
[ ] .env.example created
[ ] GCP APIs enabled (verify: gcloud services list --enabled)
[ ] GCS bucket created (verify: gsutil ls gs://$GCS_BUCKET_NAME)
[ ] Billing alert configured
```

```
✅ PHASE 1 COMPLETE — confirm to proceed to Phase 2
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 2 — DATA ACQUISITION & EXPLORATORY DATA ANALYSIS
# Estimated time: 2–3 hours
# ═══════════════════════════════════════════════════════════

## Phase 2 Goal
Download NASA CMAPSS dataset, parse it, run EDA to understand
sensor distributions, correlations, and RUL characteristics.
Produce a clean EDA notebook ready for stakeholder presentation.

## Phase 2 Tasks

### 2.1 Create data loader (src/data/loader.py)
```
FILE: src/data/loader.py

IMPLEMENT: CMAPSSLoader class with methods:
  - download(destination: str) -> None
      Downloads FD001 train/test/RUL files from:
      https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip
      Falls back to: direct file URLs from the NASA Prognostics Center
      Saves to data/raw/
  
  - parse(filepath: str) -> pd.DataFrame
      Parses space-delimited CMAPSS files.
      Column names: ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2',
                     'op_setting_3'] + ['sensor_{i}' for i in range(1, 22)]
      Returns clean DataFrame with proper dtypes.
  
  - load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]
      Returns (train_df, test_df, rul_series)
      rul_series: ground truth RUL values for test set final cycles

NOTES:
  - FD001: 100 training engines, 100 test engines
  - 21 sensor measurements per cycle
  - 3 operational settings per cycle
  - Engines run to failure in training set
  - Test set: engines stopped at unknown point before failure
```

### 2.2 Create EDA notebook (notebooks/01_eda.ipynb)
```
FILE: notebooks/01_eda.ipynb

SECTIONS:
  1. Dataset Overview
     - Shape, dtypes, missing values
     - Engine count, cycle range, sensor count
     - Print: "Training engines: 100 | Max cycles: {max} | Sensors: 21"

  2. RUL Distribution Analysis
     - Compute RUL for training set:
       RUL[i] = max_cycle[engine] - current_cycle[i]
     - Plot: RUL distribution histogram (seaborn)
     - Plot: RUL over cycles for 5 sample engines (line plot)
     - Note: apply piecewise linear RUL cap at 125 cycles (standard practice)

  3. Sensor Analysis
     - Plot: all 21 sensors over time for one engine (subplot grid)
     - Identify: which sensors show clear degradation trends
     - Drop: sensors with zero or near-zero variance:
             sensor_1, 10, 18, 19 → exactly zero variance (caught by std check)
             sensor_5, 16 → NaN correlation (near-constant, missed by std check)
             sensor_6 → RETAIN (low but non-NaN correlation 0.108)
             Result: 15 features kept (not 14 — sensor_6 stays)

  4. Correlation Analysis
     - Heatmap: sensor-to-RUL correlations
     - Identify top 8 sensors most correlated with RUL
     - These will be primary features in Phase 3

  5. Operational Settings
     - FD001: single operating condition
     - Verify op_settings are constant (FD001 characteristic)
     - Note for README: "FD001 chosen for single fault mode clarity"

  6. Summary Statistics Table
     - Min, max, mean, std per sensor
     - Flag any outliers (>3 std from mean)

OUTPUT:
  - Save EDA findings summary to data/eda_summary.json
  - Format: {"top_sensors": [...], "rul_cap": 125, "drop_sensors": [...],
             "n_train_engines": 100, "max_rul": ..., "mean_rul": ...}
```

### 2.3 Upload raw data to GCS
```bash
# Run in Cloud Shell after downloading data locally or to Cloud Shell
gsutil -m cp data/raw/*.txt gs://$GCS_BUCKET_NAME/data/raw/
```

### 2.4 Verify phase completion
```
VERIFICATION CHECKLIST:
[ ] data/raw/ contains train_FD001.txt, test_FD001.txt, RUL_FD001.txt
[ ] CMAPSSLoader.load_all() returns (train_df, test_df, rul_series)
[ ] EDA notebook runs top-to-bottom without errors
[ ] eda_summary.json saved with top_sensors list
[ ] Raw data uploaded to gs://$GCS_BUCKET_NAME/data/raw/
[ ] Identified which sensors to drop (zero variance)
[ ] RUL cap value confirmed (125 for FD001)
```

```
✅ PHASE 2 COMPLETE — confirm to proceed to Phase 3
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 3 — FEATURE ENGINEERING & PYTORCH MODEL TRAINING
# Estimated time: 3–4 hours (run notebook in Google Colab free T4 GPU)
# ═══════════════════════════════════════════════════════════

## Phase 3 Goal
Engineer rolling window features, normalize sensors, build PyTorch LSTM,
train on Colab T4 GPU, evaluate against NASA benchmark scores,
and export trained model artifact.

## Phase 3 Tasks

### 3.1 Create feature engineering module (src/data/features.py)
```
FILE: src/data/features.py

IMPLEMENT: FeatureEngineer class with methods:

  compute_rul(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame
    - Computes RUL per cycle per engine from training data
    - Applies piecewise linear cap: RUL = min(actual_rul, rul_cap)
    - Adds 'rul' column to DataFrame

  select_sensors(df: pd.DataFrame, top_sensors: list[str]) -> pd.DataFrame
    - Drops zero-variance sensors identified in EDA
    - Keeps only top_sensors + engine_id + cycle + rul
    - Returns filtered DataFrame

  normalize(train_df, test_df) -> tuple[pd.DataFrame, pd.DataFrame, object]
    - Fits MinMaxScaler on training data only (no data leakage)
    - Transforms both train and test
    - Returns (scaled_train, scaled_test, fitted_scaler)
    - Saves scaler to data/processed/scaler.joblib

  create_sequences(df: pd.DataFrame, window_size: int = 30) 
      -> tuple[np.ndarray, np.ndarray]
    - Creates rolling windows of shape (n_samples, window_size, n_features)
    - Target y shape: (n_samples,) — RUL value at last cycle of window
    - Returns (X, y) arrays
    - For test set: use only the LAST window per engine (final prediction)

  save_processed(X_train, y_train, X_test, y_test) -> None
    - Saves all arrays to data/processed/ as .npy files
    - Prints shapes for verification

CONSTANTS:
  WINDOW_SIZE = 30        # 30-cycle rolling window
  RUL_CAP = 125           # Piecewise linear cap
  SELECTED_SENSORS = []   # Populated from eda_summary.json
```

### 3.2 Create PyTorch LSTM model (src/model/lstm.py)
```
FILE: src/model/lstm.py

IMPLEMENT: RULPredictor class (nn.Module) with:

  Architecture:
    - Input: (batch, sequence_len=30, n_features)
    - LSTM layer 1: hidden_size=128, num_layers=2, dropout=0.2, batch_first=True
    - LSTM layer 2: (stacked, handled by num_layers=2)
    - Dropout layer: p=0.2
    - Fully connected layer 1: 128 → 64, ReLU activation
    - Fully connected layer 2: 64 → 1 (RUL regression output)
    - Optional classification head: 64 → 1, Sigmoid (failure within 30 cycles)

  forward(x: torch.Tensor) -> dict:
    - Returns {"rul": rul_tensor, "failure_prob": prob_tensor}
    - rul_tensor shape: (batch, 1)
    - prob_tensor shape: (batch, 1) — probability of failure in 30 cycles

  Hyperparameters (define as class defaults, overridable):
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    window_size: int = 30
    n_features: int = 15    # EDA confirmed: drop sensor_1,5,10,16,18,19 → 15 features remain
                             # sensor_6 retained (low but non-NaN correlation 0.108)
```

### 3.3 Create training module (src/model/train.py)
```
FILE: src/model/train.py

IMPLEMENT: Trainer class with:

  __init__(model, config: dict):
    config keys:
      - learning_rate: 0.001
      - batch_size: 256
      - epochs: 100
      - patience: 15        # Early stopping patience
      - device: "cuda" if available else "cpu"
      - checkpoint_dir: "model_artifacts/"
      - experiment_name: "rul-predictor-v1"

  train(X_train, y_train, X_val, y_val) -> dict:
    - DataLoader with batch_size, shuffle=True for train
    - Loss: MSELoss for RUL regression
           BCELoss for classification head (if enabled)
           Combined: total_loss = mse_loss + 0.3 * bce_loss
    - Optimizer: Adam(lr=0.001, weight_decay=1e-5)
    - Scheduler: ReduceLROnPlateau(patience=5, factor=0.5)
    - Early stopping: stop if val_loss no improvement for `patience` epochs
    - Save best checkpoint to model_artifacts/best_model.pt
    - Log per epoch: train_loss, val_loss, val_rmse, learning_rate
    - Returns training history dict

  save_model(path: str) -> None:
    - Saves model state_dict + config + scaler path
    - Format: torch.save({"state_dict": ..., "config": ..., "epoch": ...}, path)

TRAINING SPLIT:
  - 80% of training engines for training, 20% for validation
  - Split by engine_id (not by row) to prevent data leakage
```

### 3.4 Create evaluation module (src/model/evaluate.py)
```
FILE: src/model/evaluate.py

IMPLEMENT: Evaluator class with:

  compute_metrics(y_true, y_pred) -> dict:
    Returns:
      - rmse: Root Mean Squared Error
      - mae: Mean Absolute Error
      - nasa_score: NASA scoring function
          s = sum(exp(-d/13) - 1 for d < 0) + sum(exp(d/10) - 1 for d >= 0)
          where d = y_pred - y_true
          Lower is better. Asymmetric: late predictions penalized more.
      - within_10_pct: % predictions within 10% of true RUL

  BENCHMARK TARGETS for FD001 (from literature):
    RMSE: < 15.0 cycles (good), < 13.0 (excellent)
    NASA score: < 300 (good), < 200 (excellent)

  plot_predictions(y_true, y_pred, save_path: str) -> None:
    - Scatter plot: predicted vs actual RUL
    - Line plot: RUL over time for 5 sample engines
    - Save to model_artifacts/evaluation_plots/

  generate_report(metrics: dict, save_path: str) -> None:
    - Saves evaluation_report.json with all metrics + benchmark comparison
    - Prints formatted table to console
```

### 3.5 Create training notebook (notebooks/02_model_training.ipynb)
```
FILE: notebooks/02_model_training.ipynb
RUNTIME: Google Colab — Runtime > Change runtime type > T4 GPU

SECTIONS:

  0. Environment Setup
     !pip install torch numpy pandas scikit-learn matplotlib seaborn joblib tqdm
     Mount Google Drive or use Colab /content/ as working directory
     Download CMAPSS files (provide wget commands for NASA URLs)

  1. Feature Engineering
     from src.data.loader import CMAPSSLoader
     from src.data.features import FeatureEngineer
     Run full pipeline: load → compute_rul → select_sensors → normalize → sequences
     Print array shapes for verification

  2. Model Initialization
     from src.model.lstm import RULPredictor
     model = RULPredictor(n_features=len(SELECTED_SENSORS))
     Print model summary: total parameters (~450K expected)

  3. Training
     from src.model.train import Trainer
     trainer = Trainer(model, config={...})
     history = trainer.train(X_train, y_train, X_val, y_val)
     Plot: training and validation loss curves

  4. Evaluation
     from src.model.evaluate import Evaluator
     evaluator = Evaluator()
     metrics = evaluator.compute_metrics(y_test, y_pred)
     Print metrics table with benchmark comparison

  5. Export Artifact
     trainer.save_model("model_artifacts/rul_predictor_v1.pt")
     joblib.dump(scaler, "model_artifacts/scaler.joblib")
     Print: "Model saved. RMSE: {rmse:.2f} | NASA Score: {score:.1f}"

  6. Upload to GCS (run in Colab terminal or use gsutil)
     !gsutil cp model_artifacts/rul_predictor_v1.pt gs://$GCS_BUCKET_NAME/models/v1/
     !gsutil cp model_artifacts/scaler.joblib gs://$GCS_BUCKET_NAME/models/v1/
```

### 3.6 Verify phase completion
```
VERIFICATION CHECKLIST:
[ ] FeatureEngineer produces X_train shape: (n_samples, 30, n_features)
[ ] RULPredictor forward pass works: input (32, 30, 14) → output {"rul": (32,1)}
[ ] Training completes without OOM on Colab T4
[ ] Best model checkpoint saved: model_artifacts/best_model.pt
[ ] RMSE < 20 cycles on validation set (Phase 3 target — not final)
[ ] evaluation_report.json generated with all metrics
[ ] Model artifact uploaded to gs://$GCS_BUCKET_NAME/models/v1/
[ ] Scaler uploaded to gs://$GCS_BUCKET_NAME/models/v1/
```

```
✅ PHASE 3 COMPLETE — confirm to proceed to Phase 4
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 4 — VERTEX AI SETUP & MODEL REGISTRATION
# Estimated time: 2–3 hours
# Run all commands in Google Cloud Shell
# ═══════════════════════════════════════════════════════════

## Phase 4 Goal
Build a custom FastAPI serving container, push it to Artifact Registry via
Cloud Build, and register the model in Vertex AI Model Registry pointing at
that container. No local Docker required.

⚠️  LESSON LEARNED: The Vertex AI pre-built PyTorch/TorchServe container
    (pytorch-cpu.2-0:latest) crashes on startup with error code 9 — the Python
    worker exits within milliseconds before the handler is ever imported.
    Root cause: version mismatch between the container's TorchServe runtime and
    torch-model-archiver 0.12.0 / numpy 2.x ABI. DO NOT use the pre-built
    TorchServe container. Always use a custom FastAPI container.

## Phase 4 Tasks

### 4.1 Create FastAPI serving app (src/serving/serve.py)
```
FILE: src/serving/serve.py

IMPLEMENT: FastAPI application that Vertex AI calls via HTTP

  On startup:
    - Read AIP_STORAGE_URI env var (set automatically by Vertex AI to GCS artifact path)
    - Download rul_predictor_v1.pt and scaler_params.npz from GCS using
      google-cloud-storage Python client (not gsutil)
    - Load PyTorch model, reconstruct MinMaxScaler from npz arrays
    - Set model.eval()

  POST /predict endpoint:
    - Input: {"instances": [[30 x 15 float array]]}
    - Apply scaler, run model forward pass (torch.no_grad)
    - Return: {"predictions": [{"rul": int, "failure_prob": float,
                                "alert": bool, "confidence": str}]}
    - confidence: "high" rul>50, "medium" 20-50, "critical" <20

  GET /health endpoint:
    - Return: {"status": "ok"}

NOTE: Use scaler_params.npz (scale_ and min_ arrays), not scaler.joblib.
      joblib files cause import compatibility issues inside the container.
      Run scripts/extract_scaler_params.py locally to generate scaler_params.npz
      then upload: gsutil cp model_artifacts/scaler_params.npz gs://$GCS_BUCKET_NAME/models/v1/
```

### 4.2 Create serving Dockerfile (src/serving/Dockerfile)
```
FILE: src/serving/Dockerfile

FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    "numpy>=1.26,<2.0" fastapi uvicorn google-cloud-storage
COPY serve.py lstm.py ./
EXPOSE 8080
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]

NOTE: Copy src/model/lstm.py alongside serve.py — the container needs it.
      CPU-only torch wheel keeps the image under 1 GB.
```

### 4.3 Build and push container (run in Cloud Shell)
```bash
# ⚠️  IMPORTANT: gcloud builds submit does NOT support --dockerfile flag.
#    Copy Dockerfile to project root before submitting.

cp src/serving/Dockerfile Dockerfile

# Create Artifact Registry repo if it doesn't exist
gcloud artifacts repositories create predictive-maintenance \
  --repository-format=docker --location=us-central1 \
  --project=$GCP_PROJECT_ID 2>/dev/null || true

# Build and push (~2-3 minutes)
gcloud builds submit \
  --project=$GCP_PROJECT_ID \
  --tag=us-central1-docker.pkg.dev/$GCP_PROJECT_ID/predictive-maintenance/rul-predictor:v1 \
  --timeout=20m .

# Clean up root Dockerfile after build
rm Dockerfile
```

### 4.4 Register model in Vertex AI Model Registry
```python
# FILE: scripts/register_and_deploy_custom.py
# Run: python scripts/register_and_deploy_custom.py

from google.cloud import aiplatform

CUSTOM_CONTAINER = (
    f"us-central1-docker.pkg.dev/{GCP_PROJECT_ID}"
    f"/predictive-maintenance/rul-predictor:v1"
)

model = aiplatform.Model.upload(
    display_name="rul-predictor-v1-custom",
    artifact_uri=f"gs://{GCS_BUCKET_NAME}/models/v1/",
    serving_container_image_uri=CUSTOM_CONTAINER,
    serving_container_predict_route="/predict",
    serving_container_health_route="/health",
    serving_container_ports=[8080],
    labels={"project": "predictive-maintenance", "framework": "pytorch"},
)
print(f"Model registered: {model.resource_name}")
```

### 4.5 Verify phase completion
```
VERIFICATION CHECKLIST:
[ ] scaler_params.npz generated and uploaded to gs://$GCS_BUCKET_NAME/models/v1/
[ ] Cloud Build job succeeds (check Console > Cloud Build > History)
[ ] Container image visible in Artifact Registry
[ ] Model registered in Vertex AI Model Registry with custom container URI
[ ] model.resource_name saved to .env for Phase 5/6
```

```
✅ PHASE 4 COMPLETE — confirm to proceed to Phase 5
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 5 — VERTEX AI PIPELINE (KFP v2)
# Estimated time: 3–4 hours
# ═══════════════════════════════════════════════════════════

## Phase 5 Goal
Build a Kubeflow Pipelines v2 pipeline on Vertex AI that orchestrates
the full ML workflow: data validation → training → evaluation → 
conditional deployment. Pipeline is reusable for model retraining.

## Phase 5 Tasks

### 5.1 Create pipeline components (src/pipeline/components.py)
```
FILE: src/pipeline/components.py

IMPLEMENT 5 KFP v2 components using @component decorator:

⚠️  LESSON LEARNED — numpy<2.0 REQUIRED IN ALL COMPONENTS:
    Vertex AI Pipelines resolves numpy to the latest version (2.x) by default.
    PyTorch 2.0 was compiled against numpy 1.x headers — numpy 2.x breaks the
    C ABI causing "ModuleNotFoundError: No module named 'numpy._core'" at runtime.
    Add "numpy<2.0" to packages_to_install in EVERY component, even ones that
    don't directly import numpy (transitive deps pull it in).

COMPONENT 1: data_validation_component
  @component(base_image="python:3.10", packages_to_install=["numpy<2.0", "google-cloud-storage", "pandas"])
  def validate_data(
      gcs_data_uri: str,
      validation_report: Output[Artifact]
  ) -> NamedTuple("Outputs", [("is_valid", bool), ("n_engines", int)]):
    - Downloads data from GCS
    - Checks: no nulls, expected columns present, engine count > 50
    - Writes validation report JSON to output artifact
    - Returns (is_valid, n_engines)

COMPONENT 2: feature_engineering_component
  @component(base_image="python:3.10", packages_to_install=["numpy<2.0","pandas","scikit-learn","joblib"])
  def engineer_features(
      gcs_data_uri: str,
      window_size: int,
      rul_cap: int,
      processed_data: Output[Dataset]
  ) -> NamedTuple("Outputs", [("n_train_samples", int), ("n_features", int)]):
    - Runs full feature engineering pipeline (load → rul → normalize → sequence)
    - Saves processed arrays to output dataset artifact
    - Returns (n_train_samples, n_features)

COMPONENT 3: train_model_component
  @component(base_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
             packages_to_install=["numpy<2.0","google-cloud-storage","scikit-learn","joblib"])
  def train_model(
      processed_data: Input[Dataset],
      gcs_model_uri: str,
      epochs: int,
      batch_size: int,
      model_artifact: Output[Model]
  ) -> NamedTuple("Outputs", [("val_rmse", float), ("nasa_score", float)]):
    - Loads processed data from input artifact
    - Trains RULPredictor LSTM
    - Saves best checkpoint to output model artifact + GCS
    - Returns (val_rmse, nasa_score)

COMPONENT 4: evaluate_model_component
  @component(base_image="python:3.10", packages_to_install=["torch","numpy<2.0","scikit-learn"])
  def evaluate_model(
      model_artifact: Input[Model],
      test_data_uri: str,
      rmse_threshold: float,
      evaluation_report: Output[ClassificationMetrics]
  ) -> NamedTuple("Outputs", [("test_rmse", float), ("passes_threshold", bool)]):
    - Loads model from artifact
    - Runs inference on held-out test set
    - Computes RMSE, MAE, NASA score
    - Returns (test_rmse, passes_threshold) where threshold = 18.0 cycles (CPU-trained model baseline)

COMPONENT 5: deploy_model_component
  @component(base_image="python:3.10", packages_to_install=["numpy<2.0","google-cloud-aiplatform"])
  def deploy_model(
      model_resource_name: str,
      endpoint_display_name: str,
      machine_type: str,
      deployed_model_id: Output[Artifact]
  ) -> NamedTuple("Outputs", [("endpoint_uri", str)]):
    - Deploys registered model to Vertex AI Endpoint
    - machine_type: "n1-standard-2" (cost optimized)
    - min_replica_count: 1, max_replica_count: 2
    - Returns endpoint URI
```

### 5.2 Create pipeline definition (src/pipeline/pipeline.py)
```
FILE: src/pipeline/pipeline.py

IMPLEMENT: build_pipeline() function returning compiled pipeline

@pipeline(
    name="rul-predictor-pipeline",
    description="Predictive maintenance RUL pipeline — NASA CMAPSS FD001",
    pipeline_root=f"gs://{GCS_BUCKET_NAME}/pipelines/"
)
def rul_pipeline(
    gcs_data_uri: str,
    gcs_model_uri: str,
    window_size: int = 30,
    rul_cap: int = 125,
    epochs: int = 100,
    batch_size: int = 256,
    rmse_threshold: float = 18.0,   # NOTE: 15.0 is too tight for CPU-trained models (Colab T4 achieves ~15.0, CPU retraining ~16-17)
    machine_type: str = "n1-standard-2",
    endpoint_display_name: str = "rul-predictor-endpoint"
):
    # Step 1: Validate data
    validation = validate_data(gcs_data_uri=gcs_data_uri)

    # Step 2: Engineer features (only if data valid)
    with dsl.Condition(validation.outputs["is_valid"] == True, name="data-valid"):
        features = engineer_features(
            gcs_data_uri=gcs_data_uri,
            window_size=window_size,
            rul_cap=rul_cap
        )

        # Step 3: Train model
        training = train_model(
            processed_data=features.outputs["processed_data"],
            gcs_model_uri=gcs_model_uri,
            epochs=epochs,
            batch_size=batch_size
        )

        # Step 4: Evaluate model
        evaluation = evaluate_model(
            model_artifact=training.outputs["model_artifact"],
            test_data_uri=f"{gcs_data_uri}/test",
            rmse_threshold=rmse_threshold
        )

        # Step 5: Deploy only if evaluation passes threshold
        with dsl.Condition(evaluation.outputs["passes_threshold"] == True,
                           name="deploy-if-good"):
            deployment = deploy_model(
                model_resource_name=...,
                endpoint_display_name=endpoint_display_name,
                machine_type=machine_type
            )
```

### 5.3 Create pipeline trigger script (scripts/run_pipeline.py)
```
FILE: scripts/run_pipeline.py

IMPLEMENT:
  - Compile pipeline to rul_pipeline.yaml
  - Submit pipeline job to Vertex AI Pipelines
  - Print pipeline job URL for monitoring in GCP Console
  - Poll job status every 60 seconds until completion
  - Print final status: SUCCEEDED / FAILED
  - On success: print deployed endpoint URI
```

### 5.4 Verify phase completion
```
VERIFICATION CHECKLIST:
[ ] All 5 KFP components defined with correct Input/Output types
[ ] Pipeline compiles to rul_pipeline.yaml without errors
[ ] Pipeline run submitted to Vertex AI (check Console > Vertex AI > Pipelines)
[ ] All 5 steps show green (may take 30–60 min for full run)
[ ] Conditional deployment step executes only if RMSE < 18.0 (not 15.0 — see rmse_threshold note above)
[ ] Endpoint URI printed and saved to .env
[ ] Pipeline DAG visible in GCP Console (screenshot for README)
```

```
✅ PHASE 5 COMPLETE — confirm to proceed to Phase 6
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 6 — VERTEX AI ENDPOINT & MODEL MONITORING
# Estimated time: 1–2 hours
# ═══════════════════════════════════════════════════════════

## Phase 6 Goal
Build and deploy the custom FastAPI serving container to Vertex AI Endpoint,
validate it with live prediction calls, and configure Model Monitoring for
feature drift detection.

⚠️  LESSON LEARNED — SERVING CONTAINER:
    The Vertex AI pre-built TorchServe container fails at startup (error code 9).
    Use the custom FastAPI container built in Phase 4 instead.
    Endpoint deployment is done via scripts/register_and_deploy_custom.py,
    NOT via the KFP pipeline's deploy_model component.

⚠️  LESSON LEARNED — CLOUD BUILD:
    gcloud builds submit does NOT support the --dockerfile flag in Cloud Shell.
    Always copy the Dockerfile to the project root before submitting:
        cp src/serving/Dockerfile Dockerfile
        gcloud builds submit --tag IMAGE_URI --timeout=20m .
        rm Dockerfile   # clean up after build

## Phase 6 Tasks

### 6.1 Test endpoint predictions (scripts/test_endpoint.py)
```
FILE: scripts/test_endpoint.py

IMPLEMENT: EndpointTester class

  test_single_prediction(endpoint_id: str) -> dict:
    - Constructs synthetic payload: 30 cycles of sensor readings
    - Calls endpoint.predict(instances=[...])
    - Prints: {"rul": 45, "failure_prob": 0.23, "alert": false, "confidence": "medium"}

  test_batch_predictions(endpoint_id: str, n_samples: int = 10) -> pd.DataFrame:
    - Sends 10 prediction requests
    - Measures latency per request (target: < 200ms)
    - Returns DataFrame with predictions + latencies

  validate_response_schema(response: dict) -> bool:
    - Asserts all required keys present
    - Asserts RUL is positive integer
    - Asserts failure_prob in [0, 1]

  run_all_tests(endpoint_id: str) -> None:
    - Runs all tests above
    - Prints PASS/FAIL for each
    - Prints summary: "Endpoint healthy: {n_passed}/3 tests passed"
```

### 6.2 Configure Vertex AI Model Monitoring
```python
# FILE: scripts/setup_monitoring.py

from google.cloud.aiplatform import ModelDeploymentMonitoringJob

IMPLEMENT: setup_monitoring() function

  monitoring_job = ModelDeploymentMonitoringJob.create(
      display_name="rul-predictor-monitoring",
      endpoint=endpoint_resource_name,

      # Monitor for feature drift (sensor value distribution shift)
      feature_skew_thresholds={
          f"sensor_{i}": 0.3 for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
      },

      # Monitor for prediction drift (RUL distribution shift)
      prediction_drift_thresholds={
          "rul": 0.3,
          "failure_probability": 0.2
      },

      # Sampling rate and monitoring frequency
      logging_sampling_strategy={"random_sample_config": {"sample_rate": 0.1}},
      monitor_interval=Duration(seconds=3600),  # Check hourly

      # Alerting
      anomaly_cloud_logging=True,
  )

  # PORTFOLIO NOTE: This directly mirrors the LLM Drift Monitor project
  # narrative — "I build systems that detect when things go wrong,
  # whether ML models drifting or physical assets degrading"
```

### 6.3 Verify phase completion
```
VERIFICATION CHECKLIST:
[ ] Endpoint responds to single prediction in < 200ms
[ ] Response schema valid (all keys present, correct types)
[ ] 10/10 batch predictions succeed
[ ] Model monitoring job created (check Console > Vertex AI > Model Monitoring)
[ ] Monitoring configured for sensor drift + prediction drift
[ ] Endpoint URI confirmed functional: curl test passes
```

```
✅ PHASE 6 COMPLETE — confirm to proceed to Phase 7
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 7 — TERRAFORM INFRASTRUCTURE AS CODE
# Estimated time: 2–3 hours
# ═══════════════════════════════════════════════════════════

## Phase 7 Goal
Write Terraform configuration for all GCP resources provisioned
in Phases 1–6. This creates reproducible, version-controlled
infrastructure — a key FDE competency signal.

## Phase 7 Tasks

### 7.1 Create Terraform main configuration
```
FILE: terraform/main.tf

CONTENTS:
  terraform {
    required_providers {
      google = { source = "hashicorp/google", version = "~> 5.0" }
    }
    backend "gcs" {
      bucket = "predictive-maintenance-artifacts"
      prefix = "terraform/state"
    }
  }

  provider "google" {
    project = var.project_id
    region  = var.region
  }
```

### 7.2 Create GCS bucket resource
```
FILE: terraform/gcs.tf

RESOURCES:
  - google_storage_bucket "artifacts"
      name: var.bucket_name
      location: var.region
      force_destroy: false
      versioning: enabled
      lifecycle_rule: delete objects older than 90 days (cost control)

  - google_storage_bucket_iam_member "vertex_sa_access"
      Grants Vertex AI service account storage.objectAdmin on bucket
```

### 7.3 Create Artifact Registry resource
```
FILE: terraform/artifact_registry.tf

RESOURCES:
  - google_artifact_registry_repository "containers"
      repository_id: "predictive-maintenance"
      format: DOCKER
      location: var.region
      description: "Container images for predictive maintenance pipeline"
```

### 7.4 Create Vertex AI resources
```
FILE: terraform/vertex.tf

RESOURCES:
  - google_vertex_ai_endpoint "rul_endpoint"
      display_name: "rul-predictor-endpoint"
      location: var.region
      labels: {project: "predictive-maintenance"}

  NOTE: Model upload and deployment done via Python SDK in pipeline.
        Terraform manages endpoint resource only.
```

### 7.5 Create Cloud Run resource
```
FILE: terraform/cloudrun.tf

RESOURCES:
  - google_cloud_run_v2_service "streamlit_dashboard"
      name: "rul-dashboard"
      location: var.region
      ingress: INGRESS_TRAFFIC_ALL
      template:
        containers:
          image: "${var.region}-docker.pkg.dev/${var.project_id}/predictive-maintenance/dashboard:latest"
          resources: limits {cpu: "1", memory: "512Mi"}
          env: VERTEX_ENDPOINT_ID, GCP_PROJECT_ID, GCP_REGION
        scaling:
          min_instance_count: 0    # Scale to zero — zero idle cost
          max_instance_count: 3
```

### 7.6 Outputs file
```
FILE: terraform/outputs.tf

OUTPUTS:
  - bucket_uri: gs://${google_storage_bucket.artifacts.name}
  - artifact_registry_url: ${google_artifact_registry_repository.containers.name}
  - vertex_endpoint_id: ${google_vertex_ai_endpoint.rul_endpoint.name}
  - dashboard_url: ${google_cloud_run_v2_service.streamlit_dashboard.uri}
```

### 7.7 Verify Terraform plan
```bash
# Run in Cloud Shell
cd terraform
terraform init
terraform plan -var="project_id=$GCP_PROJECT_ID" -var="bucket_name=$GCS_BUCKET_NAME"
# Review plan output — should show ~6 resources to create
# DO NOT apply if this project already provisioned resources manually
# Terraform is for documentation + reproducibility signal in this portfolio
```

### 7.8 Verify phase completion
```
VERIFICATION CHECKLIST:
[ ] terraform init succeeds (backend configured)
[ ] terraform plan runs without errors
[ ] All resources from Phases 1–6 represented in .tf files
[ ] State backend configured (GCS bucket)
[ ] outputs.tf exports all key resource identifiers
[ ] terraform/ directory committed to GitHub
```

```
✅ PHASE 7 COMPLETE — confirm to proceed to Phase 8
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 8 — STREAMLIT DASHBOARD ON CLOUD RUN
# Estimated time: 2–3 hours
# ═══════════════════════════════════════════════════════════

## Phase 8 Goal
Build a Streamlit dashboard that calls the live Vertex AI Endpoint,
displays RUL predictions, sensor health gauges, and maintenance alerts.
Deploy to Cloud Run using Cloud Build (no local Docker).

## Phase 8 Tasks

### 8.1 Create Streamlit application (dashboard/app.py)
```
FILE: dashboard/app.py

IMPLEMENT: 4-section Streamlit dashboard

SECTION 1: Header
  st.title("🔧 Predictive Equipment Failure — RUL Dashboard")
  st.caption("NASA CMAPSS FD001 | PyTorch LSTM | Google Cloud Vertex AI")

SECTION 2: Engine Selector + Live Prediction
  - Sidebar: engine_id selector (1–100), cycle slider (1–max_cycle)
  - On selection: fetch sensor readings from GCS processed data
  - Call Vertex AI Endpoint with 30-cycle window ending at selected cycle
  - Display:
    * RUL Gauge (st.metric): "Remaining Useful Life: {rul} cycles"
    * Failure Probability (st.progress bar): color red if > 0.7
    * Maintenance Alert (st.error / st.success): "⚠️ Schedule maintenance"
    * Confidence Level (st.badge): critical / medium / high

SECTION 3: Sensor Health Panel
  - 3-column layout showing 6 key sensor time-series (plotly line charts)
  - Highlight anomalous sensor readings in red
  - X-axis: cycle number, Y-axis: normalized sensor value

SECTION 4: Fleet Overview
  - Run predictions for all test engines (cached with @st.cache_data)
  - Show scatter plot: predicted RUL distribution across fleet
  - Show table: top 10 engines with lowest predicted RUL
  - Show count: "{n} engines require maintenance attention"

HELPER FUNCTIONS:
  call_vertex_endpoint(instances: list) -> dict
    - Uses google.cloud.aiplatform Endpoint.predict()
    - Handles errors gracefully (show warning if endpoint unavailable)
    - Caches results for 60 seconds

  load_test_data(engine_id: int, up_to_cycle: int) -> np.ndarray
    - Loads from GCS or local processed data
    - Returns last 30 cycles of sensor readings
```

### 8.2 Create dashboard requirements
```
FILE: dashboard/requirements.txt

streamlit>=1.28.0
google-cloud-aiplatform>=1.38.0
google-cloud-storage>=2.10.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
torch>=2.1.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

### 8.3 Create Dockerfile for Cloud Run
```
FILE: dashboard/Dockerfile

FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY ../src/ ./src/
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

### 8.4 Create Cloud Build config (no local Docker)
```
FILE: cloudbuild/cloudbuild.yaml

steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/predictive-maintenance/dashboard:${SHORT_SHA}'
      - '-f'
      - 'dashboard/Dockerfile'
      - '.'

  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/predictive-maintenance/dashboard:${SHORT_SHA}'

  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - 'gcloud'
      - 'run'
      - 'deploy'
      - 'rul-dashboard'
      - '--image=${_REGION}-docker.pkg.dev/${PROJECT_ID}/predictive-maintenance/dashboard:${SHORT_SHA}'
      - '--region=${_REGION}'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--min-instances=0'
      - '--max-instances=3'
      - '--memory=512Mi'

substitutions:
  _REGION: us-central1
```

```bash
# Trigger build from Cloud Shell (no local Docker needed)
gcloud builds submit --config cloudbuild/cloudbuild.yaml \
  --substitutions=_REGION=us-central1 .
```

### 8.5 Verify phase completion
```
VERIFICATION CHECKLIST:
[ ] Streamlit app runs locally: streamlit run dashboard/app.py
[ ] Engine selector returns predictions from Vertex AI Endpoint
[ ] RUL gauge, failure probability, and maintenance alert display correctly
[ ] Fleet overview loads all test engine predictions
[ ] Cloud Build job succeeds (check Console > Cloud Build > History)
[ ] Cloud Run URL accessible in browser
[ ] Dashboard URL saved: $STREAMLIT_ENDPOINT_URL
[ ] min-instances=0 confirmed (check Cloud Run > rul-dashboard > Revisions)
```

```
✅ PHASE 8 COMPLETE — confirm to proceed to Phase 9
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 9 — VERTEX AI EXPERIMENTS & EXPERIMENT TRACKING
# Estimated time: 1 hour
# ═══════════════════════════════════════════════════════════

## Phase 9 Goal
Instrument training runs with Vertex AI Experiments for
hyperparameter tracking and run comparison — demonstrates
production MLOps discipline for FDE interviews.

## Phase 9 Tasks

### 9.1 Cloud Shell setup before running experiments

⚠️  LESSON LEARNED — Cloud Shell does not have PyTorch or python-dotenv installed.
    Run these before anything else:
```bash
# Install CPU-only torch (~200MB, not the 2GB CUDA wheel)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib scikit-learn

# Upload processed data to GCS if not already there (run from Mac/local)
# gsutil -m cp data/processed/*.npy gs://$GCS_BUCKET_NAME/data/processed/
```

⚠️  LESSON LEARNED — sys.path in scripts:
    Scripts that import from src/ must use Path(__file__).resolve() (not just
    Path(__file__)) to get an absolute path. Otherwise sys.path gets '.' which
    may not resolve correctly in all environments:
        _ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(_ROOT))

### 9.2 Run 3 experiment variants (scripts/run_experiments.py)
```bash
# Run in Cloud Shell — loads processed data from GCS, trains 3 variants,
# saves results to model_artifacts/experiment_summary.json
python scripts/run_experiments.py \
  --data-dir gs://predictive-maintenance-artifacts/data/processed \
  --vertex-experiments

# Quick smoke-test (5 epochs, no GCP logging):
python scripts/run_experiments.py --epochs 5
```

EXPERIMENT RUNS:
  Run 1 (baseline):   hidden_size=64,  num_layers=1, lr=0.001
  Run 2 (deeper):     hidden_size=128, num_layers=2, lr=0.001
  Run 3 (optimized):  hidden_size=128, num_layers=2, lr=0.0005, dropout=0.3

ACTUAL RESULTS (CPU training, Cloud Shell):
  | Run | Hidden | Layers | LR     | Dropout | RMSE  | NASA Score |
  |-----|--------|--------|--------|---------|-------|------------|
  | 1   | 64     | 1      | 0.001  | 0.2     | 15.26 | 496.9      |
  | 2   | 128    | 2      | 0.001  | 0.2     | 15.59 | 477.5      |
  | 3   | 128    | 2      | 0.0005 | 0.3     | 15.25 | 427.6 ✅   |

### 9.3 Log results to Vertex AI Experiments

⚠️  LESSON LEARNED — Vertex AI Experiments API:
    Do NOT use the context manager pattern (with aiplatform.start_run(...):).
    It fails silently in some SDK versions. Use explicit start/end instead:

```python
# CORRECT pattern — always use explicit start_run / end_run
from google.cloud import aiplatform

aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION,
                experiment="rul-predictor-experiments")

aiplatform.start_run("run-1-baseline")
aiplatform.log_params({"hidden_size": 64, "num_layers": 1, "learning_rate": 0.001})
aiplatform.log_metrics({"test_rmse": 15.26, "nasa_score": 496.9})
aiplatform.end_run()

# WRONG pattern — context manager silently fails in some SDK versions:
# with aiplatform.start_run("run-name"):   ← DO NOT USE
#     aiplatform.log_params({...})
```

If training already completed, use the standalone script to log results without retraining:
```bash
python scripts/log_to_vertex.py
```

### 9.3 Verify phase completion
```
VERIFICATION CHECKLIST:
[ ] Vertex AI Experiments created (Console > Vertex AI > Experiments)
[ ] 3 runs visible with logged parameters and metrics
[ ] Run comparison chart visible in Console
[ ] Best run identified and model from that run deployed to endpoint
[ ] Experiment table data saved to model_artifacts/experiment_summary.json
```

```
✅ PHASE 9 COMPLETE — confirm to proceed to Phase 10
```

---

# ═══════════════════════════════════════════════════════════
# PHASE 10 — PORTFOLIO DOCUMENTATION & GITHUB PUBLISH
# Estimated time: 2–3 hours
# ═══════════════════════════════════════════════════════════

## Phase 10 Goal
Create production-quality GitHub README, architecture diagram,
LinkedIn post draft, and resume one-liner. This is the
recruiter-facing layer — equal in importance to the code.

## Phase 10 Tasks

### 10.1 Create GitHub README (README.md)
```
FILE: README.md

SECTIONS (in order):

  1. Project Banner
     # 🔧 Predictive Equipment Failure — PyTorch LSTM + Google Cloud Vertex AI
     Badge row: Python 3.10 | PyTorch 2.1 | Vertex AI | Cloud Run | Terraform

  2. One-line Summary
     "End-to-end MLOps pipeline predicting turbofan engine Remaining Useful Life
     (RUL) from NASA CMAPSS sensor data — PyTorch LSTM deployed on Vertex AI
     with automated retraining pipeline, model drift monitoring, and live
     Streamlit dashboard."

  3. Live Demo
     🌐 Dashboard: [Cloud Run URL]
     📊 Vertex AI Pipeline: [screenshot]
     🔗 Model Registry: [screenshot]

  4. Architecture Diagram
     [Insert Lucidchart diagram — generated in Phase 10.2]

  5. Dataset
     - NASA CMAPSS FD001 description
     - 21 sensors, 100 training engines, RUL cap: 125 cycles
     - Link to NASA data source

  6. Model Performance
     | Metric     | This Model | Literature Benchmark |
     |------------|------------|----------------------|
     | RMSE       | xx.x       | ~13.0 (excellent)    |
     | MAE        | xx.x       | —                    |
     | NASA Score | xxx.x      | ~200 (excellent)     |

  7. Experiment Comparison Table
     [From Phase 9.2]

  8. Tech Stack Table
     Full stack: PyTorch, Vertex AI, KFP v2, GCS, Cloud Run, Terraform, etc.

  9. Project Structure
     [Abbreviated directory tree]

  10. Getting Started
      git clone → install deps → set .env → run phases

  11. Cost Notes
      "Entire project runs within GCP $300 free trial. Set billing alert at $30.
       Undeploy endpoint: bash scripts/undeploy_endpoint.sh"

  12. Portfolio Context
      "Part of AIOps portfolio trilogy alongside:
       - AIOps Data Lakehouse (MinIO, Iceberg, LangGraph)
       - LLM Eval Framework (Claude Sonnet judge)
       - LLM Drift & Behavioral Regression Monitor"

  13. Author
      Saurabh | Technology Lead → AI Architect
      LinkedIn | GitHub: github.com/DevMLAI01
```

### 10.2 Architecture diagram (Lucidchart)
```
DIAGRAM: Request generation via Lucid MCP in next Claude session

COMPONENTS TO INCLUDE:
  Left side (Data + Training):
    NASA CMAPSS → Google Colab (T4 GPU) → PyTorch LSTM → model.pt

  Middle (GCP MLOps):
    GCS Bucket → Vertex AI Model Registry
    Vertex AI Pipeline (KFP v2):
      [validate] → [feature engineering] → [train] → [evaluate] → [deploy]
    Vertex AI Experiments (hyperparameter tracking)
    Vertex AI Model Monitoring (drift detection)

  Right side (Serving):
    Vertex AI Endpoint → Streamlit on Cloud Run → End User

  Bottom (IaC):
    Terraform → [GCS] [Artifact Registry] [Cloud Run] [Vertex Endpoint]

  Color coding:
    Blue: GCP managed services
    Teal: PyTorch / ML components
    Amber: Data flow
    Gray: Infrastructure / IaC
```

### 10.3 Create undeploy script (cost guardrail)
```
FILE: scripts/undeploy_endpoint.sh

#!/bin/bash
# Run this after demos to stop endpoint billing
echo "Undeploying Vertex AI Endpoint..."
gcloud ai endpoints undeploy-model $VERTEX_ENDPOINT_ID \
  --project=$GCP_PROJECT_ID \
  --region=$GCP_REGION \
  --deployed-model-id=$DEPLOYED_MODEL_ID

echo "Endpoint undeployed. Re-deploy with: python scripts/run_pipeline.py"
echo "Cloud Run scales to zero automatically (min-instances=0)"
```

### 10.4 LinkedIn post draft
```
TEXT (for LinkedIn — adapt as needed):

🔧 New portfolio project: Predictive Equipment Failure on Google Cloud

Built an end-to-end MLOps pipeline predicting when turbofan engines will fail —
using NASA CMAPSS sensor data, PyTorch, and Google Cloud Vertex AI.

The pipeline:
→ PyTorch LSTM predicting Remaining Useful Life (RUL) from 21 IoT sensors
→ Automated Vertex AI Pipeline (KFP v2): validate → train → evaluate → deploy
→ Vertex AI Model Monitoring for feature drift detection
→ Live Streamlit dashboard on Cloud Run
→ Full infrastructure as code with Terraform

Why this matters for Industry 4.0 clients: unplanned equipment failure costs
manufacturers $50B+ annually. ML-driven predictive maintenance cuts downtime
by up to 30% while eliminating unnecessary early interventions.

The project is open source: [GitHub URL]
Live demo: [Cloud Run URL]

Tech: PyTorch · Vertex AI · KFP v2 · Cloud Run · Terraform · Google Cloud

#MLOps #PredictiveMaintenance #GoogleCloud #PyTorch #AIOps #MachineLearning
```

### 10.5 Resume bullet points
```
RESUME ADDITIONS (under Projects or Portfolio section):

Primary bullet:
"Built production MLOps pipeline on Google Cloud Vertex AI — PyTorch LSTM
predicting equipment Remaining Useful Life from NASA CMAPSS IoT sensor data;
deployed via automated KFP v2 pipeline with model drift monitoring and
Streamlit observability dashboard on Cloud Run (Terraform-provisioned infrastructure)"

Sub-bullets (for expanded project section):
• Achieved RMSE < 15 cycles on held-out test set against NASA benchmark dataset
• Implemented dual-head LSTM architecture: RUL regression + failure classification
• Configured Vertex AI Model Monitoring for feature skew and prediction drift
• Tracked 3 hyperparameter experiment runs via Vertex AI Experiments
• Deployed serverless dashboard on Cloud Run (min-instances=0, zero idle cost)
• Full IaC coverage with Terraform GCP provider across 6 resource types
```

### 10.6 Final verification
```
FINAL PROJECT VERIFICATION CHECKLIST:
[ ] GitHub repository public at github.com/DevMLAI01/predictive-maintenance-vertex
[ ] README.md complete with all 13 sections
[ ] Architecture diagram embedded in README
[ ] Live Cloud Run dashboard URL working
[ ] Vertex AI Endpoint responding (or clearly documented as undeployed for cost)
[ ] All 10 phases documented in commit history
[ ] Terraform directory present and functional
[ ] LinkedIn post drafted and ready to publish
[ ] Resume bullets finalized
[ ] evaluation_report.json committed with model metrics
[ ] No .env or raw data committed (verify .gitignore)
```

```
✅ PHASE 10 COMPLETE — PROJECT PORTFOLIO READY
🎉 Repository: github.com/DevMLAI01/predictive-maintenance-vertex
```

---

## COST SUMMARY

| Resource | Usage pattern | Estimated cost within $300 trial |
|---|---|---|
| Vertex AI Training | 1–3 custom training jobs | ~$2–5 |
| Vertex AI Endpoint | Deployed only during demos | ~$0.50/hour — undeploy after demos |
| Vertex AI Pipelines | 3–5 pipeline runs | ~$1–3 |
| Cloud Run (Streamlit) | min-instances=0 | ~$0 idle, ~$0.10/hour active |
| Cloud Storage | ~500MB total | ~$0.01/month |
| Cloud Build | 3–5 builds | ~$0 (free tier: 120 build-min/day) |
| Vertex AI Experiments | Metadata only | ~$0 |
| Vertex AI Monitoring | Sampling 10% | ~$1–2 |
| **Total estimate** | | **< $15 across full project** |

---

## QUICK REFERENCE — KEY COMMANDS

```bash
# Phase 1: Initialize GCP
gcloud services enable aiplatform.googleapis.com storage.googleapis.com ...

# Phase 3: Upload model to GCS
gsutil cp model_artifacts/rul_predictor_v1.pt gs://$GCS_BUCKET_NAME/models/v1/

# Phase 5: Run pipeline
python scripts/run_pipeline.py

# Phase 8: Trigger Cloud Build (no local Docker)
gcloud builds submit --config cloudbuild/cloudbuild.yaml .

# Cost guardrail: undeploy endpoint after demos
bash scripts/undeploy_endpoint.sh

# Redeploy endpoint for demo
python scripts/run_pipeline.py --deploy-only
```

---

*REQUIREMENTS.md generated for Claude Code execution in Google Cloud Shell*
*Project: Predictive Equipment Failure — PyTorch LSTM + Vertex AI*
*Portfolio target: Google Cloud Forward Deployed Engineer | Big 4 AI Architect*
