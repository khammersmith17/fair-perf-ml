# fair_perf_ml

A high-performance ML observability library written in Rust with Python bindings. Written to be easy to use and composable.

Provides a unified API for **pre-training data bias**, **post-training model bias**,
**model performance**, and **data drift** — in both batch and stateful streaming settings.

---

## Modules

| Module | Description |
|--------|-------------|
| [`fair_perf_ml.data_bias`](data_bias.md) | Pre-training bias analysis (CI, DPL, KL, JS, LpNorm, TVD, KS) |
| [`fair_perf_ml.model_bias`](model_bias.md) | Post-training model bias (DI, AccDiff, RecallDiff, CDDPL, GE, ...) |
| [`fair_perf_ml.model_perf`](model_perf.md) | Performance metrics for binary classification, linear and logistic regression |
| [`fair_perf_ml.drift`](drift.md) | Data drift detection — batch and streaming, continuous and categorical |
| [`fair_perf_ml.bias`](segmentation.md) | Segmentation primitives shared across bias modules |
| [`fair_perf_ml.models`](models.md) | Shared return types and metric enums |

---

## Installation

```bash
pip install fair_perf_ml
```

Requires Python ≥ 3.11.

---

## Quick example

```python
import numpy as np
from fair_perf_ml.data_bias import data_bias_perform_analysis, data_bias_runtime_comparison
from fair_perf_ml.drift import ContinuousDataDrift, DataDriftType

# --- Data bias ---
baseline = data_bias_perform_analysis(
    feature=[1, 0, 1, 1, 0, 0, 1],
    ground_truth=[1, 0, 1, 0, 1, 0, 1],
    feature_label_or_threshold=1,
    ground_truth_label_or_threshold=1,
)
latest = data_bias_perform_analysis(
    feature=[1, 1, 1, 0, 0, 0, 0],
    ground_truth=[1, 1, 0, 0, 0, 0, 1],
    feature_label_or_threshold=1,
    ground_truth_label_or_threshold=1,
)
report = data_bias_runtime_comparison(baseline, latest, threshold=0.10)

# --- Drift ---
baseline_data = np.random.normal(0, 1, 1000)
monitor = ContinuousDataDrift(baseline_data)
runtime_data = np.random.normal(0.5, 1, 500)
drift = monitor.compute_drift(runtime_data, DataDriftType.JensenShannon)
```
