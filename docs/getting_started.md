# Getting Started

## Installation

```bash
pip install fair_perf_ml
```

Requires Python ≥ 3.11 and numpy ≥ 2.0.

---

## Concepts

### Segmentation

Bias analysis requires dividing data into *advantaged* and *disadvantaged* groups.
`fair_perf_ml` handles this via **segmentation criteria** — either a label (equality)
or a threshold (ordering).

See [Segmentation](segmentation.md) for the full API.

### Analysis reports

Batch analysis functions (`data_bias_perform_analysis`, `model_bias_perform_analysis`,
`linear_regression_analysis`, etc.) return a plain `dict[str, float]` mapping metric
names to values. These dicts can be stored and passed back to runtime comparison
functions later.

### Runtime comparison / drift reports

Runtime check functions compare a latest analysis report against a stored baseline.
They return a [`DriftReport`](models.md) indicating whether any metric has drifted
beyond the configured threshold (default 10%).

```python
{
    "passed": True,           # False if any metric exceeded the threshold
    "failed_report": None,    # list of {"metric": ..., "drift": ...} when passed=False
}
```

### Streaming monitors

Streaming classes maintain a baseline and accumulate runtime examples incrementally.
They expose `drift_report()` and `performance_snapshot()` for point-in-time queries
without needing to collect and replay the full dataset.

---

## Data bias — walkthrough

```python
from fair_perf_ml.data_bias import (
    data_bias_perform_analysis,
    data_bias_runtime_comparison,
    data_bias_partial_runtime_comparison,
    DataBiasStreaming,
)
from fair_perf_ml.bias import LabeledBiasSegmentation
from fair_perf_ml.models import DataBiasMetric

# Batch analysis
baseline = data_bias_perform_analysis(
    feature=[0, 1, 1, 0, 1, 0],
    ground_truth=[1, 1, 0, 0, 1, 1],
    feature_label_or_threshold=1,
    ground_truth_label_or_threshold=1,
)
latest = data_bias_perform_analysis(
    feature=[1, 1, 1, 0, 0, 0],
    ground_truth=[1, 0, 1, 0, 0, 1],
    feature_label_or_threshold=1,
    ground_truth_label_or_threshold=1,
)

# Full runtime check
report = data_bias_runtime_comparison(baseline, latest, threshold=0.10)

# Partial (selected metrics only)
partial = data_bias_partial_runtime_comparison(
    baseline, latest,
    metrics=[DataBiasMetric.ClassImbalance, DataBiasMetric.KlDivergence],
)

# Streaming
feat_seg = LabeledBiasSegmentation(label=1)
gt_seg = LabeledBiasSegmentation(label=1)
monitor = DataBiasStreaming(feat_seg, gt_seg, feature_data=[0, 1, 1], ground_truth_data=[1, 1, 0])
monitor.push(feature_value=1, ground_truth_value=0)
snapshot = monitor.drift_report(drift_threshold=0.10)
```

---

## Model performance — walkthrough

```python
import numpy as np
from fair_perf_ml.model_perf import (
    binary_classification_analysis,
    logistic_regression_analysis,
    linear_regression_analysis,
    runtime_check_full,
    BinaryClassificationStreaming,
)

y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1])

baseline = binary_classification_analysis(y_true, y_pred)

# Later, compare against a new batch
y_true_new = np.array([1, 1, 0, 0, 1, 0, 0, 1])
y_pred_new = np.array([0, 1, 1, 0, 0, 0, 1, 1])
latest = binary_classification_analysis(y_true_new, y_pred_new)

report = runtime_check_full(latest, baseline, threshold=0.10)

# Streaming
monitor = BinaryClassificationStreaming(label=1, y_true=y_true, y_pred=y_pred)
monitor.update_stream_batch(y_true_new, y_pred_new)
drift = monitor.drift_report(drift_threshold=0.10)
```

---

## Data drift — walkthrough

```python
import numpy as np
from fair_perf_ml.drift import (
    ContinuousDataDrift,
    CategoricalDataDrift,
    DataDriftType,
    StreamingContinuousDataDriftFlush,
    StreamingCategoricalDataDriftDecay,
)

# Batch continuous
baseline = np.random.normal(0, 1, 2000)
monitor = ContinuousDataDrift(baseline, quantile_type="FreedmanDiaconis")
runtime = np.random.normal(0.3, 1, 500)
scores = monitor.compute_drift_multiple_criteria(
    runtime,
    [DataDriftType.JensenShannon, DataDriftType.PopulationStabilityIndex],
)

# Streaming continuous with flush
stream = StreamingContinuousDataDriftFlush(baseline, quantile_type="Scott", flush_rate=1000)
stream.update_stream_batch(runtime)
js = stream.compute_drift(DataDriftType.JensenShannon)

# Categorical
cat_baseline = ["cat", "dog", "cat", "bird", "dog", "cat"]
cat_monitor = CategoricalDataDrift(cat_baseline)
cat_runtime = ["cat", "cat", "cat", "dog", "bird"]
psi = cat_monitor.compute_drift(cat_runtime, DataDriftType.PopulationStabilityIndex)
```
