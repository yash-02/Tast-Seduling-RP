# CPU/GPU Job Scheduler Predictor

A machine learning system for intelligent CPU vs GPU job routing. This project trains binary classifiers and regression models to predict optimal processor allocation and job duration for distributed computing workloads across multiple platforms (Kubernetes, Slurm, Cloud Batch).

## Overview

The system addresses a core infrastructure problem: **How should incoming computational jobs be routed to CPU or GPU queues?** 

Using historical job execution data, the models learn patterns from resource requests, user behavior, and temporal features to:
- **Classify** jobs as CPU-bound or GPU-bound
- **Predict** job duration in log-scale seconds
- **Route** jobs to appropriate queues with configurable policies

## Key Features

- **Multi-platform submission extraction**: Kubernetes manifests, Slurm scripts, Cloud Batch jobs
- **Dual-model architecture**: LightGBM classifier + regressor
- **Policy-driven routing**: Explicit overrides, confidence thresholds, queue assignment
- **Interactive demo**: Streamlit web app for single-job and batch routing
- **Temporal & statistical features**: Hour-of-day, day-of-week, user/workload history rates
- **Production logging**: CSV decision logs for audit trails

## Project Structure

```
.
├── main.py                          # Quick data exploration script
├── train_pipeline.py                # Model training pipeline
├── predict_job.py                   # CLI for single-job prediction
├── production_submission_router.py  # Batch JSONL -> CSV routing
├── streamlit_app.py                 # Interactive web interface
├── content/                         # Training data (CSVs)
│   ├── pai_job_table.csv           # Job metadata
│   ├── pai_task_table.csv          # Task-level resource specs
│   └── pai_group_tag_table.csv     # User & workload classification
├── raw_submissions/                 # Production job inputs
│   └── production_raw_jobs.jsonl   # Sample raw submissions
├── decision_logs/                   # Output routing decisions
│   ├── routing_decisions.csv       # Batch routing results
│   └── streamlit_routing_decisions.csv
├── models/                          # Persisted artifacts
│   ├── classifier.joblib           # Latest classifier
│   ├── regressor.joblib            # Latest regressor
│   ├── latest.txt                  # Version pointer
│   └── [timestamp]/                # Versioned checkpoints
├── src/
│   ├── config.py                   # Constants & paths
│   ├── data_loader.py              # CSV ingestion
│   ├── features.py                 # Feature engineering
│   ├── model_io.py                 # Model save/load
│   ├── predictor.py                # Real-time inference
│   ├── policy.py                   # Routing logic
│   └── submission_extractor.py     # Platform-specific parsing
└── requirements.txt                # Dependencies

```

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`, `scipy`: ML utilities
- `lightgbm`: Gradient boosting classifiers/regressors
- `joblib`: Model serialization
- `streamlit`: Interactive web app
- `torch`, `pytorch-tabnet`: (optional) Advanced architectures

## Usage

### 1. Train the Models

Requires CSV files in `content/` directory:
- `pai_job_table.csv`
- `pai_task_table.csv`
- `pai_group_tag_table.csv`

```bash
python train_pipeline.py [--data-dir ./content] [--model-dir ./models]
```

**Output:**
- Trained classifier and regressor saved to `models/[timestamp]/`
- `metadata.json` containing feature list, thresholds, and metrics
- Evaluation metrics printed to console

### 2. Predict Single Job (CLI)

Route a single job with command-line arguments:

```bash
python predict_job.py \
  --user alice \
  --workload-type tensorflow \
  --cpu 16 \
  --mem 32 \
  --instances 2 \
  --tasks 4 \
  [--explicit-gpu] \
  [--model-dir ./models]
```

**Output:** JSON prediction with processor, queue, confidence, GPU probability.

### 3. Route Production Batch (CLI)

Process raw job submissions from Kubernetes/Slurm/Cloud Batch:

```bash
python production_submission_router.py \
  --input raw_submissions/production_raw_jobs.jsonl \
  --output decision_logs/routing_decisions.csv \
  [--model-dir ./models]
```

**Input format (JSONL):**
```json
{
  "platform": "kubernetes",
  "raw": { "apiVersion": "batch/v1", "metadata": { ... } }
}
```

**Output:** CSV with columns:
- `job_id`, `source_platform`, `user`, `workload_type`
- `total_plan_cpu`, `total_plan_mem`, `total_inst_num`, `num_tasks`
- `processor`, `queue`, `reason`, `confidence`, `gpu_probability`
- `predicted_duration_minutes`, `processor_threshold`

### 4. Interactive Web App (Streamlit)

Launch the demo interface for manual and batch routing:

```bash
streamlit run streamlit_app.py
```

Features:
- **Manual Job Tab**: Fill form fields, route single jobs, see predictions
- **Batch Tab**: Upload JSONL or use default submissions, download results
- **Model Info Tab**: View trained metrics, feature list, artifact metadata

Access at `http://localhost:8501`

## Model Architecture

### Classifier (CPU vs GPU)

**Model:** LightGBM Classification
- **Objective:** Binary (GPU=1, CPU=0)
- **Hyperparameters:**
  - `n_estimators=500`, `learning_rate=0.03`
  - `num_leaves=15`, `max_depth=5`
  - `class_weight='balanced'`
  - `subsample=0.7`, `colsample_bytree=0.7`
  - Early stopping on validation set

### Regressor (Job Duration)

**Model:** LightGBM Regression
- **Objective:** L2 (Squared Error)
- **Target:** Log-transformed duration: `log(1 + seconds)`
- **Hyperparameters:**
  - `n_estimators=1000`, `learning_rate=0.05`
  - `num_leaves=31`
  - `subsample=0.8`, `colsample_bytree=0.8`
  - Early stopping on validation set

### Feature Set (15 features)

**Resource Features:**
- `log_cpu`, `log_mem`, `log_inst`, `log_tasks` (log-transformed counts)
- `cpu_per_inst`, `mem_per_inst`, `tasks_per_inst` (per-instance ratios)
- `cpu_mem_ratio` (CPU to memory ratio)

**Temporal Features:**
- `hour`, `dow` (hour of day, day of week)
- `sin_hour`, `cos_hour` (cyclic encoding)
- `is_weekend` (binary weekend flag)

**Historical Features:**
- `user_gpu_rate` (past GPU allocation rate for user)
- `group_gpu_rate` (past GPU allocation rate for workload type)

### Data Split

Time-based split (no random shuffle to prevent leakage):
- **Train:** 70% of chronologically ordered jobs
- **Validation:** 15% (threshold optimization)
- **Test:** 15% (final evaluation)

## Platform Extraction

### Kubernetes

Extracts from Kubernetes Job manifest:
- Container CPU/memory requests
- GPU limits/requests
- Parallelism & image labels
- Workload type inference from labels & image names

### Slurm

Parses `sbatch` options:
- `--cpus-per-task`, `--ntasks`, `--nodes`
- `--mem`, `--gres=gpu`
- Job name for workload inference

### Cloud Batch

Parses Cloud Batch job spec:
- Resource requests (CPU, memory, GPU)
- Task & machine counts
- Container image & command for workload inference

## Routing Policy

Jobs are routed to CPU or GPU queues based on:

1. **Explicit Overrides:** `explicit_cpu`, `explicit_gpu`, `gpu_allowed`
2. **Capacity Constraints:** `cpu_capacity_ok` flag
3. **Model Predictions:** GPU probability vs threshold (default 0.70)
4. **Confidence Fallback:** If confidence < 0.75, optionally force CPU/GPU
5. **Small Job Policy:** Small jobs require explicit GPU request
6. **Workload Type Heuristics:** CPU-like workloads (data_processing, simulation) default to CPU

### Queue Assignment

Jobs are assigned to one of four queues based on processor + duration:
- `cpu-short`: CPU queue, predicted duration < 30 min
- `cpu-long`: CPU queue, predicted duration ≥ 30 min
- `gpu-short`: GPU queue, predicted duration < 30 min
- `gpu-long`: GPU queue, predicted duration ≥ 30 min

## Configuration

Key settings in `src/config.py`:

```python
FEATURE_COLUMNS = [...]        # Fixed 15-feature list
DEFAULT_CONFIDENCE_THRESHOLD = 0.75  # Low-confidence fallback threshold
SHORT_JOB_SECONDS = 30 * 60    # Duration threshold for queue assignment

# Small job policy
SMALL_CPU_LIKE_MAX_TOTAL_CPU = 400
SMALL_CPU_LIKE_MAX_MEM_PER_INSTANCE = 16
SMALL_CPU_LIKE_MAX_TASKS_PER_INSTANCE = 4
SMALL_CPU_LIKE_MAX_INSTANCES = 3
```

## Model Evaluation

The training pipeline reports:

- **Classification Metrics:**
  - Confusion matrix on test set
  - Precision, recall, F1-score per class
  - Macro F1 used to optimize threshold on validation set

- **Regression Metrics:**
  - RMSLE (Root Mean Squared Log Error) on test set
  - Spearman correlation between predicted and true log-durations

Example output:
```
--- Classifier Results ---
Decision threshold chosen on validation macro-F1: 0.14

Confusion matrix:
[[tn, fp], [fn, tp]]

Classification Report:
            precision  recall  f1-score
CPU             0.92     0.88     0.90
GPU             0.87     0.91     0.89

--- Duration Regressor Results ---
RMSLE        : 0.4231
Spearman rho : 0.6845
```

## Artifact Versioning

Models are versioned with ISO 8601 timestamps:

```
models/
  ├── latest.txt                 # Points to current version
  ├── 20260425T093744Z/          # Snapshot from Apr 25, 09:37:44 UTC
  │   ├── classifier.joblib
  │   ├── regressor.joblib
  │   └── metadata.json          # Feature list, thresholds, metrics
  └── 20260425T094300Z/          # Newer snapshot
      ├── classifier.joblib
      ├── regressor.joblib
      └── metadata.json
```

Metadata example:
```json
{
  "model_version": "20260425T093744Z",
  "trained_at": "2026-04-25T09:37:44.000Z",
  "processor_threshold": 0.14,
  "feature_columns": [...],
  "metrics": {
    "classification_report": {...},
    "confusion_matrix": [...],
    "duration_rmsle": 0.4231,
    "duration_spearman_rho": 0.6845
  },
  "history": {
    "global_gpu_rate": 0.65,
    "user_gpu_rates": {...},
    "group_gpu_rates": {...}
  }
}
```

## API Reference

### `SchedulerPredictor`

```python
from src.predictor import SchedulerPredictor

predictor = SchedulerPredictor(model_dir="./models")

prediction = predictor.predict(
    raw_job={
        "user": "alice",
        "workload_type": "tensorflow",
        "total_plan_cpu": 1600,  # Internal units
        "total_plan_mem": 32.0,  # GB
        "total_inst_num": 2,
        "num_tasks": 4,
        "start_time": time.time(),
        "explicit_gpu": False,
        "cpu_capacity_ok": True,
        "gpu_allowed": True,
    },
    confidence_threshold=0.75,
    processor_threshold=0.70,  # Override trained threshold
    low_confidence_fallback="MODEL"  # Or "CPU", "GPU"
)

# Returns:
# {
#   "processor": "GPU",
#   "queue": "gpu-long",
#   "reason": "model_gpu",
#   "confidence": 0.85,
#   "gpu_probability": 0.85,
#   "predicted_duration_seconds": 3600.0,
#   "predicted_duration_minutes": 60.0,
#   "processor_threshold": 0.70,
#   "low_confidence_fallback": "MODEL",
#   "feature_columns": [...]
# }
```

### `route_submissions`

```python
from src.production_submission_router import route_submissions

decisions, output_path = route_submissions(
    input_path="raw_submissions/production_raw_jobs.jsonl",
    output_path="decision_logs/routing_decisions.csv",
    model_dir="./models"
)
```

## Examples

### Example 1: Train models from raw data

```bash
python train_pipeline.py --data-dir ./content --model-dir ./models
```

### Example 2: Predict TensorFlow job

```bash
python predict_job.py \
  --user bob \
  --workload-type tensorflow \
  --cpu 8 --mem 16 --instances 1 --tasks 2
```

### Example 3: Route batch of Kubernetes jobs

```bash
# Ensure raw_submissions/production_raw_jobs.jsonl exists
python production_submission_router.py \
  --input raw_submissions/production_raw_jobs.jsonl \
  --output decision_logs/routing_decisions.csv
# View results in decision_logs/routing_decisions.csv
```

### Example 4: Use Streamlit demo with custom threshold

```bash
streamlit run streamlit_app.py
# In sidebar: Set "GPU decision threshold" to 0.50
# Upload custom JSONL or manually create jobs
```

## Common Issues & Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Missing required CSV file` | Place `pai_job_table.csv`, `pai_task_table.csv`, `pai_group_tag_table.csv` in `content/` directory |
| `Could not load model artifacts` | Run `python train_pipeline.py` first to create `models/latest.txt` |
| All jobs routed to GPU in demo | Raise the GPU decision threshold in Streamlit sidebar (try 0.50+) |
| Import errors for LightGBM | Run `pip install lightgbm --upgrade` |
| Streamlit port already in use | `streamlit run streamlit_app.py --server.port 8502` |

## Dependencies

See `requirements.txt`:
- pandas, numpy: Data processing
- scikit-learn, scipy: ML utilities & metrics
- lightgbm: Boosted tree models
- joblib: Model serialization
- streamlit: Web interface
- torch, pytorch-tabnet: Optional advanced models

## License

[Specify your license here]

## Contributing

[Contribution guidelines, if applicable]

## Contact

For questions or issues, contact the development team or submit an issue.
#   T a s t - S e d u l i n g - R P  
 