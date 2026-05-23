# Predictive Equipment Failure — PyTorch LSTM + Google Cloud Vertex AI

🎯 Test RMSE: **15.03 cycles**
🔢 Engines evaluated: **100 (NASA CMAPSS FD001)**
⚙️ Pipeline steps: **5-stage KFP v2 (validate → engineer → train → evaluate → deploy)**
☁️ Infrastructure: **Terraform-provisioned, fully on Google Cloud**

An end-to-end MLOps pipeline that predicts **Remaining Useful Life (RUL)** of turbofan engines from IoT sensor data — PyTorch LSTM trained on Google Colab, orchestrated through Vertex AI Pipelines, served via a custom FastAPI container, and monitored for feature drift. Full infrastructure as code with Terraform.

---

## Business Value

Unplanned industrial equipment failure costs manufacturers **$50B+ annually**. This pipeline demonstrates how ML-driven predictive maintenance:

- Shifts maintenance from **reactive (fail first)** to **predictive (act before failure)**
- Eliminates unnecessary early interventions driven by fixed-interval schedules
- Produces a **Remaining Useful Life score per engine** — giving operations teams a concrete decision signal rather than a binary alert
- Detects sensor drift in production, flagging when the live data distribution diverges from training

The same architecture — LSTM over rolling time-series windows, Vertex AI Pipelines for retraining orchestration, model monitoring for drift — applies directly to compressors, pumps, CNC machines, and any IoT-instrumented industrial asset.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  DATA & TRAINING                                                  │
│                                                                  │
│  NASA CMAPSS FD001          Google Colab (free T4 GPU)           │
│  (21 sensors, 100 engines)──► PyTorch LSTM ──► model.pt + scaler │
│                                  │                               │
│                                  ▼                               │
│                            GCS Bucket                            │
│                    gs://predictive-maintenance-artifacts/         │
└──────────────────────────────────────┬───────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────┐
│  VERTEX AI MLOPS PIPELINE (KFP v2)                               │
│                                                                  │
│  [1. validate_data] ──► [2. engineer_features]                   │
│        │                       │                                 │
│        ▼                       ▼                                 │
│  [3. train_model] ──► [4. evaluate_model] ──► (RMSE < threshold) │
│                                │                                 │
│                    ┌───────────▼──────────────┐                  │
│                    │  5. deploy_model          │                  │
│                    │  Vertex AI Endpoint       │                  │
│                    └───────────────────────────┘                 │
│                                                                  │
│  Vertex AI Experiments ── 3 hyperparameter runs tracked          │
│  Vertex AI Model Monitoring ── feature drift + prediction drift  │
└──────────────────────────────────────┬───────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────┐
│  SERVING & OBSERVABILITY                                         │
│                                                                  │
│  Custom FastAPI Container (Cloud Build → Artifact Registry)      │
│       │                                                          │
│       ▼                                                          │
│  Vertex AI Endpoint  ◄──  Streamlit Dashboard (Cloud Run)        │
│  (custom predictor)         min-instances=0, zero idle cost      │
│                                                                  │
│  Terraform ── GCS · Artifact Registry · Cloud Run · Endpoint    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Model Performance

Evaluated on 100 held-out NASA CMAPSS FD001 test engines (Colab T4 GPU training):

| Metric | This Model | Good Benchmark | Excellent Benchmark |
|---|---|---|---|
| **RMSE** | **15.03 cycles** | < 15.0 | < 13.0 |
| **MAE** | 11.36 cycles | — | — |
| **NASA Score** | 402.86 | < 300 | < 200 |
| **Within ±10%** | 39% predictions | — | — |

> NASA Score is asymmetric — late predictions (predicting longer life than actual) are penalized more heavily than early ones. Lower is better.

---

## Experiment Comparison (Vertex AI Experiments)

Three hyperparameter runs logged to `rul-predictor-experiments` on Vertex AI (CPU training in Cloud Shell):

| Run | Hidden Size | Layers | Learning Rate | Dropout | RMSE | NASA Score |
|---|---|---|---|---|---|---|
| run-1-baseline | 64 | 1 | 0.001 | 0.2 | 15.26 | 496.9 |
| run-2-deeper | 128 | 2 | 0.001 | 0.2 | 15.59 | 477.5 |
| **run-3-optimized** ✅ | **128** | **2** | **5e-4** | **0.3** | **15.25** | **427.6** |

Best run identified: `run-3-optimized` — lower learning rate + higher dropout improves generalization. GPU training (Colab T4) achieves RMSE 15.03.

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| **Model** | PyTorch LSTM (dual-head) | RUL regression + failure-within-30-cycles classification |
| **Dataset** | NASA CMAPSS FD001 | 21 sensors, 100 train engines, RUL cap 125 cycles |
| **Training** | Google Colab (free T4 GPU) | Preserves GCP $300 trial credits |
| **Feature Engineering** | Rolling window (30 cycles), MinMaxScaler | 15 features after zero-variance sensor drop |
| **Pipeline Orchestration** | Vertex AI Pipelines (KFP v2) | 5-stage: validate → engineer → train → evaluate → deploy |
| **Experiment Tracking** | Vertex AI Experiments | 3 runs, hyperparams + metrics, TensorBoard-backed |
| **Model Registry** | Vertex AI Model Registry | Labeled, versioned, linked to pipeline run |
| **Serving** | Custom FastAPI container (Vertex AI Endpoint) | Built via Cloud Build — no local Docker |
| **Drift Monitoring** | Vertex AI Model Monitoring | Feature skew + prediction drift, hourly, 10% sampling |
| **Dashboard** | Streamlit on Cloud Run | min-instances=0, zero idle cost |
| **Artifact Storage** | Google Cloud Storage | Model artifacts, processed data, pipeline YAMLs |
| **IaC** | Terraform (GCP provider ~5.0) | GCS, Artifact Registry, Cloud Run, Vertex Endpoint |
| **Container Build** | Cloud Build | No local Docker required |

---

## Project Structure

```
predictive-maintenance-vertex/
├── src/
│   ├── model/
│   │   ├── lstm.py              # PyTorch LSTM — dual-head RUL + failure classification
│   │   ├── train.py             # Trainer: early stopping, Vertex AI Experiments integration
│   │   └── evaluate.py          # RMSE, MAE, NASA score, evaluation report
│   ├── data/
│   │   ├── loader.py            # NASA CMAPSS parser
│   │   └── features.py          # RUL labeling, rolling window, MinMaxScaler
│   ├── serving/
│   │   └── predictor.py         # Vertex AI custom predictor interface
│   └── pipeline/
│       ├── components.py        # 5 KFP v2 @component definitions
│       └── pipeline.py          # Vertex AI Pipeline definition
│
├── notebooks/
│   ├── 01_eda.ipynb             # EDA: sensor distributions, RUL analysis, drop list
│   └── 02_model_training.ipynb  # Colab training notebook (T4 GPU)
│
├── dashboard/
│   ├── app.py                   # Streamlit: RUL gauge, sensor health, fleet overview
│   ├── requirements.txt
│   └── Dockerfile               # Built via Cloud Build
│
├── terraform/
│   ├── main.tf                  # GCP provider, GCS backend
│   ├── gcs.tf                   # Storage bucket + lifecycle rules
│   ├── artifact_registry.tf     # Container registry
│   ├── vertex.tf                # Vertex AI Endpoint
│   └── cloudrun.tf              # Streamlit dashboard service
│
├── scripts/
│   ├── run_pipeline.py          # Compile + submit KFP pipeline to Vertex AI
│   ├── run_experiments.py       # 3-variant hyperparameter search + Vertex AI logging
│   ├── log_to_vertex.py         # Standalone: log experiment results from JSON
│   ├── setup_monitoring.py      # Configure Vertex AI Model Monitoring
│   ├── test_endpoint.py         # Endpoint validation (latency + schema checks)
│   └── undeploy_endpoint.sh     # Cost guardrail: undeploy when not demoing
│
├── cloudbuild/
│   └── cloudbuild.yaml          # Build + push + deploy dashboard to Cloud Run
│
├── model_artifacts/
│   ├── evaluation_report.json   # Phase 3 metrics (RMSE 15.03, NASA 402.86)
│   └── experiment_summary.json  # Phase 9 — 3-run comparison table
│
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Google Cloud project with billing enabled
- `gcloud` CLI authenticated (`gcloud auth login`)

### 1. Clone and install

```bash
git clone https://github.com/DevMLAI01/Predictive-maintenance-MLOps.git
cd Predictive-maintenance-MLOps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: GCP_PROJECT_ID, GCP_REGION, GCS_BUCKET_NAME
```

### 3. Initialize GCP (run in Cloud Shell)

```bash
bash scripts/gcp_init.sh
```

Enables required APIs, creates GCS bucket, sets billing alert at $30.

### 4. Train model (Google Colab — free T4 GPU)

Open `notebooks/02_model_training.ipynb` in Google Colab. Set runtime to T4 GPU. Run all cells — uploads `rul_predictor_v1.pt` and `scaler.joblib` to GCS on completion.

### 5. Run the ML pipeline (Cloud Shell)

```bash
python scripts/run_pipeline.py
```

Compiles the KFP pipeline, submits to Vertex AI, polls until completion (~40 min). Deploys model to Vertex AI Endpoint if RMSE passes threshold.

### 6. Run hyperparameter experiments (Cloud Shell)

```bash
python scripts/run_experiments.py \
  --data-dir gs://predictive-maintenance-artifacts/data/processed \
  --vertex-experiments
```

### 7. Deploy Streamlit dashboard (Cloud Shell)

```bash
gcloud builds submit --config cloudbuild/cloudbuild.yaml .
```

### 8. Undeploy endpoint when done (cost guardrail)

```bash
bash scripts/undeploy_endpoint.sh
```

---

## Cost Profile

The entire project runs within GCP's **$300 free trial**. Billing alert set at $30.

| Resource | Pattern | Estimated cost |
|---|---|---|
| Vertex AI Endpoint | Deployed only during demos | ~$0.50/hr — undeploy after |
| Vertex AI Pipeline | 3–5 runs | ~$1–3 total |
| Cloud Run (Streamlit) | min-instances=0 | ~$0 idle |
| Cloud Storage | ~500 MB | ~$0.01/month |
| Cloud Build | 3–5 builds | ~$0 (free tier) |
| Vertex AI Experiments | Metadata only | ~$0 |
| **Total** | | **< $15 across full project** |

---

## Portfolio Context

Part of an AIOps portfolio trilogy targeting Google Cloud FDE and Big 4 AI Architect roles:

| Project | Focus | Stack |
|---|---|---|
| **Predictive Equipment Failure** ← this | IoT sensor → RUL prediction, MLOps pipeline | PyTorch · Vertex AI · KFP v2 · Terraform |
| [Telecom NOC Agent](https://github.com/DevMLAI01/telecom-noc-agent) | Autonomous alarm resolution, agentic RAG | LangGraph · GPT-4o · AWS Lambda |
| [Autonomous SRE & FinOps](https://github.com/DevMLAI01/autonomous-sre-finops) | Cloud cost optimization, HITL guardrails | LangGraph · Gemini · Qdrant · Terraform |

---

## Author

**Saurabh** — Technology Lead → AI Architect

19 years enterprise experience across telecom, cloud, and AI/data engineering. Currently building production-grade AIOps systems for Industry 4.0 and cloud-native ML platforms.

[GitHub: DevMLAI01](https://github.com/DevMLAI01)
