# fair-ml
Custom implementation of bias analysis for machine learning models. Based on the AWS SageMaker bias models, though this accommodates bias based on protected classes not used in model training.

Working on v0.2.0 to add some more runtime drift proxies, provide better runtime support, and make the crate available in both pure Rust and Python contexts.

This Python Library and Rust crate serves to treat ML Observability as a first class principle for ML systems, by making it accessible, and exploring more effective methods of implementing real time systems.

[Background](#background)
[Modules](#modules)
[Usage](#usage)
[Future](#future)

## Background
Governance in AI systems is becoming more important. Many large cloud providers and other vendors provide services for these analyses, but they are expensive and sometimes over-engineered. The overall goal of this project is to provide a lightweight monitoring framework for machine learning models, that is hopefully easy to use.

The core idea of this project is to provide composable pieces to create you own ML observavility service. The tools available on the market force users into a rather rigid pattern of use, and are built in such a way that they are not "editable" after definition, at least the tools I have used in my experience. For example, once the "schema" is defined for a model, you cannot later add a non model feature to the monitoring set for bias, where logically, this should not be a constraint as long as there is an ability to define a mapping between inference scores, ground truth, and feature values.

Bias analysis works by seperating a feature into two demographic groups, and predictions and outcomes into positive and negative outcomes. The goal in this analysis is to quantify the divergance between how the model and true outcomes favors one demographic.

To do so everything is segmented into two distinct groups representing favored and disfavord groups, though labeling one group as "favored" is a semantic detail. In reality, it does not matter which group is which, only that there is logical segmentation logic to distinctly define 2 groups.

Additionally, there is a module to monitor overall performance at runtime. The nature of ML makes it difficult to have unit tests, and to ensure performance at runtime. ML deployments are different from other software deployments given the inability to ensure accurate results. Our assertions need to be done after the fact. Though, most deploy the model and let it run. There often is not a consistent effort to ensure accuracy in the model predictions over the entire lifetime of the model. There are services available to do this with different vendors (ie AWS SageMaker), but this requires significant cost and compute; they also tend to be slow. 

The goal of this package is to offer that kind of ML observability at no cost, and limited resources needed. 

A newer goal of this package is to make an attempt at introducing utilities that can close the gap between how ML observability is done and how traditional system style observability is done. The gap is typically in the time it takes to get feedback, and how it is done. ML observability is often implement as batch data jobs on some cron cadence. It does not feel as smooth or as elegant as system observability, and the feedback time is tied to the cadence of the cron schedule. Version v0.2.0 introduces streaming types, that can accumulate data and create a smoother, real time observability cadence, more to come on this in future versions. The streaming types in v0.2.0 can be (hoepfully) easily composed to create a long running service. The internal representation in Rust has a very minimal memory footprint, so many different datasets can be used in a single process with relatively low resource constraint.

Other changes in v0.2.0 to exisiting logic is to refactor internal logic for performance and clean up the Python surface to have better annotations for static analysis.

The logic is written in Rust, with a python interface to let users pass in some different types (ie not only numpy arrays but also python lists, if for whatever reason someones likes to use python lists instead of numpy arrays), and an easy to use interface. The performance penalty there is minimal and makes use quite a bit easier. I generally feel that users do not need to pay someone a lot of money for services that do not require it.

This package would not be possbile without the great work done by the contributors of PYO3, that work is wonderdul.

## Modules

When I have more time, I will write a better wiki as documentation, but for now the documentation for the public api surface lives here.

### bias.data_bias

Batch pre-training bias analysis on feature and ground truth data.

- **`perform_analysis(feature, ground_truth, feature_label_or_threshold, ground_truth_label_or_threshold) -> dict[str, float]`**
    - `feature: list[str | float | int] | NDArray` — feature values
    - `ground_truth: list[str | float | int] | NDArray` — ground truth values
    - `feature_label_or_threshold: str | float | int` — segmentation parameter for the feature
    - `ground_truth_label_or_threshold: str | float | int` — segmentation parameter for the ground truth

- **`runtime_comparison(baseline, latest, threshold=0.10) -> DriftReport`**
    - `baseline: dict[str, float]` — result from `perform_analysis` on baseline data
    - `latest: dict[str, float]` — result from `perform_analysis` on runtime data
    - `threshold: float | None` — maximum allowable per-metric drift

- **`partial_runtime_comparison(baseline, latest, metrics, threshold=0.10) -> DriftReport`**
    - Same as `runtime_comparison` but scoped to `metrics: list[DataBiasDriftMetric]`

---

### bias.model_bias

Batch post-training bias analysis on feature, ground truth, and prediction data.

- **`perform_analysis(feature, ground_truth, predictions, feature_label_or_threshold, ground_truth_label_or_threshold, prediction_label_or_threshold) -> dict[str, float]`**
    - `feature: list[str | float | int] | NDArray`
    - `ground_truth: list[str | float | int] | NDArray`
    - `predictions: list[str | float | int] | NDArray`
    - `feature_label_or_threshold: str | float | int`
    - `ground_truth_label_or_threshold: str | float | int`
    - `prediction_label_or_threshold: str | float | int`

- **`runtime_comparison(baseline, comparison, threshold=0.10) -> DriftReport`**
    - `baseline: dict` — result from `perform_analysis` on baseline data
    - `comparison: dict` — result from `perform_analysis` on runtime data
    - `threshold: float | None` — maximum allowable per-metric drift

- **`partial_runtime_comparison(baseline, comparison, metrics, threshold=0.10) -> DriftReport`**
    - Same as `runtime_comparison` but scoped to `metrics: list[ModelBiasDriftMetric]`

---

### bias.streaming

Stateful streaming bias monitors for long-running services. Data is accumulated incrementally and evaluated against a fixed baseline.

**`BiasSegmentationCriteria[P]`** — defines how a value is assigned to the advantaged or disadvantaged group.
- `BiasSegmentationType.Label` — equality-based segmentation
- `BiasSegmentationType.Threshold` — comparison-based segmentation, requires a `BiasSegmentationThresholdType`

**`DataBiasStreaming[F, G]`** — streaming data bias monitor.
- `push(feature_value, ground_truth_value)` — accumulate a single example
- `push_batch(feature_data, ground_truth_data)` — accumulate a batch
- `reset_baseline(feature_data, ground_truth_data)` — replace baseline, keep segmentation criteria
- `reset_baseline_and_segmentation_criteria(...)` — replace baseline and segmentation criteria
- `flush()` — discard runtime state, preserve baseline
- `performance_snapshot() -> PerformanceSnapshot`
- `drift_snapshot() -> DriftSnapshot`
- `drift_report(drift_threshold) -> DriftReport`
- `drift_report_partial_metrics(drift_metrics, drift_threshold) -> DriftReport`

**`ModelBiasStreaming[F, P, G]`** — streaming model bias monitor. Same interface as `DataBiasStreaming` with an additional `prediction_segment_criteria` and `prediction_data` in the constructor and push methods.

---

### drift.base

Batch distributional drift detection. Computes a drift score by comparing a runtime dataset against a fixed baseline distribution.

**`ContinuousDataDrift`** — for floating-point features. The baseline is summarized as a histogram; bin count is derived from the data using the chosen quantile rule.
- `__init__(baseline_data, quantile_type=None)`
    - `quantile_type`: `"FreedmanDiaconis"` (default), `"Scott"`, or `"Sturges"`. Accepts `QuantileType` enum values.
- `compute_drift(runtime_data, drift_metric) -> float`
- `compute_drift_multiple_criteria(runtime_data, drift_metrics) -> list[float]`
- `reset_baseline(new_baseline)`
- `export_baseline() -> list[float]`
- `num_bins: int` — number of histogram bins

**`CategoricalDataDrift`** — for categorical features. The baseline is summarized as a label frequency distribution. Unseen runtime labels are collected in an overflow bin.
- `__init__(baseline_data)`
- `compute_drift(runtime_data, drift_metric) -> float`
- `compute_drift_multiple_criteria(runtime_data, drift_metrics) -> list[float]`
- `reset_baseline(new_baseline)`
- `export_baseline() -> list[float]`
- `num_bins: int`

Accepted `drift_metric` values (`DataDriftType` enum or string):
- `"JensenShannon"`
- `"PopulationStabilityIndex"`
- `"WassersteinDistance"`
- `"KullbackLeibler"`

---

### drift.streaming

Stateful streaming drift monitors. Data is accumulated incrementally; drift is computed against a fixed baseline at any point without needing to retain the raw stream.

Two flush strategies are provided for each data type:

**Flush** (`StreamingContinuousDataDriftFlush`, `StreamingCategoricalDataDriftFlush`) — accumulated data is periodically discarded. Accepts `flush_rate` (max samples before auto-flush) and `flush_cadence` (seconds between auto-flushes).

**Decay** (`StreamingContinuousDataDriftDecay`, `StreamingCategoricalDataDriftDecay`) — older samples are down-weighted over time using exponential decay. Accepts `decay_half_life` (in seconds).

All four classes share this interface:
- `update_stream(example)` — accumulate a single example
- `update_stream_batch(runtime_data)` — accumulate a batch
- `reset_baseline(new_baseline)` — replace the baseline
- `compute_drift(drift_metric) -> float`
- `compute_drift_multiple_criteria(drift_metrics) -> list[float]`
- `export_snapshot() -> dict`
- `export_baseline() -> dict`
- `total_samples: int`
- `num_bins: int` / `n_bins: int`

Flush variants additionally expose:
- `flush()` — manually discard accumulated runtime data
- `last_flush() -> int` — Unix timestamp of the last flush

---

### model_perf

Batch model performance analysis and runtime drift evaluation.

- **`linear_regression_analysis(y_true, y_pred) -> dict`**
- **`logistic_regression_analysis(y_true, y_pred, decision_threshold=0.5) -> dict`**
- **`binary_classification_analysis(y_true, y_pred) -> dict`**
    - `y_true: NDArray | list[int | float]`
    - `y_pred: NDArray | list[int | float]`
    - Returns a `ModelPerformanceReport`-shaped dict (see schemas below)

- **`runtime_check_full(latest, baseline, threshold=0.10) -> DriftReport`**
    - `latest: ModelPerformanceReport | dict`
    - `baseline: ModelPerformanceReport | dict`
    - `threshold: float`

- **`partial_runtime_check(latest, baseline, metrics, threshold=0.10) -> DriftReport`**
    - Same as `runtime_check_full` but scoped to `metrics: list[str]`
    - Accepted metric values by model type:
        - **LinearRegression**: `RootMeanSquaredError`, `MeanSquaredError`, `MeanAbsoluteError`, `RSquared`, `MaxError`, `MeanSquaredLogError`, `RootMeanSquaredLogError`, `MeanAbsolutePercentageError`
        - **BinaryClassification**: `BalancedAccuracy`, `PrecisionPositive`, `PrecisionNegative`, `RecallPositive`, `RecallNegative`, `Accuracy`, `F1Score`
        - **LogisticRegression**: `BalancedAccuracy`, `PrecisionPositive`, `PrecisionNegative`, `RecallPositive`, `RecallNegative`, `Accuracy`, `F1Score`, `LogLoss`

---

### model_perf.streaming

Stateful streaming model performance monitors. Maintains a baseline and accumulates runtime data incrementally.

**`BinaryClassificationStreaming[T: LabelBound]`**
- `__init__(label, y_true, y_pred)` — `label` is the positive class label; any type implementing `__eq__`
- `update_stream(y_true, y_pred)`
- `update_stream_batch(y_true, y_pred)`
- `reset_baseline(y_true, y_pred)`
- `reset_baseline_and_label(label, y_true, y_pred)` — only method that allows changing the label
- `flush()`
- `performance_snapshot() -> PerformanceSnapshot`
- `drift_snapshot() -> DriftSnapshot`
- `drift_report(drift_threshold) -> DriftReport`
- `drift_report_partial_metrics(metrics, drift_threshold) -> DriftReport`

**`LinearRegressionStreaming`** and **`LogisticRegressionStreaming`** — same interface. `LogisticRegressionStreaming` additionally accepts a `threshold: float | None = 0.5` in the constructor and exposes `reset_baseline_and_decision_threshold(y_true, y_pred, threshold)`.

## Usage

The intended usage is to monitor machine learning models for bias and performance degradation over time. The general flow is:

1. At training time, run analysis on a holdout set to establish baseline metrics.
2. Save those baseline results to persistent storage.
3. At runtime, collect inference data (features, predictions) and ground truth as it becomes available.
4. Periodically run analysis on the runtime data and compare it to the baseline.
5. Act on failures — alert, retrain, or escalate depending on severity.

Where this fits in a system architecture depends on deployment type:
- **API-served model**: run analysis as a background job on a cron schedule, triggered when ground truth is available.
- **Batch scoring**: run analysis alongside the batch inference job, evaluating the previous run's data once ground truth is collected.

### Bias Evaluations

Some pre-work is required to identify features to monitor for bias and define the logic to segment data into advantaged and disadvantaged groups. Feature data used at inference time should be persisted alongside predictions.

```python
from fair_perf_ml.bias import data_bias, model_bias

# --- At training time ---

data_bias_baseline = data_bias.perform_analysis(
    feature=[...],
    ground_truth=[...],
    feature_label_or_threshold=...,
    ground_truth_label_or_threshold=...
)

model_bias_baseline = model_bias.perform_analysis(
    feature=[...],
    ground_truth=[...],
    predictions=[...],
    feature_label_or_threshold=...,
    ground_truth_label_or_threshold=...,
    prediction_label_or_threshold=...
)

# Save data_bias_baseline and model_bias_baseline to persistent storage.

# --- At runtime ---

data_bias_latest = data_bias.perform_analysis(
    feature=[...],
    ground_truth=[...],
    feature_label_or_threshold=...,
    ground_truth_label_or_threshold=...
)

model_bias_latest = model_bias.perform_analysis(
    feature=[...],
    ground_truth=[...],
    predictions=[...],
    feature_label_or_threshold=...,
    ground_truth_label_or_threshold=...,
    prediction_label_or_threshold=...
)

# Load baselines from storage.
data_bias_baseline = ...
model_bias_baseline = ...

data_result = data_bias.runtime_comparison(
    baseline=data_bias_baseline,
    latest=data_bias_latest
)

if data_result.passed:
    print("Data bias check passed")
else:
    print("Data bias check failed", data_result.failed_report)

model_result = model_bias.runtime_comparison(
    baseline=model_bias_baseline,
    latest=model_bias_latest
)

if model_result.passed:
    print("Model bias check passed")
else:
    print("Model bias check failed", model_result.failed_report)
```

The output schema for `data_bias.perform_analysis`:
```json
{
    "ClassImbalance": float,
    "DifferenceInProportionOfLabels": float,
    "KlDivergence": float,
    "JsDivergence": float,
    "LpNorm": float,
    "TotalVarationDistance": float,
    "KolmogorovSmirnov": float
}
```

The output schema for `model_bias.perform_analysis`:
```json
{
    "DifferenceInPositivePredictedLabels": float,
    "DisparateImpact": float,
    "AccuracyDifference": float,
    "RecallDifference": float,
    "DifferenceInConditionalAcceptance": float,
    "DifferenceInAcceptanceRate": float,
    "SpecialityDifference": float,
    "DifferenceInConditionalRejection": float,
    "DifferenceInRejectionRate": float,
    "TreatmentEquity": float,
    "ConditionalDemographicDesparityPredictedLabels": float,
    "GeneralizedEntropy": float
}
```

### Bias Streaming

For long-running services where data arrives continuously, use the streaming monitors to avoid re-running full batch analysis every evaluation cycle.

```python
from fair_perf_ml.bias.streaming import DataBiasStreaming, ModelBiasStreaming
from fair_perf_ml.bias.streaming import BiasSegmentationCriteria, BiasSegmentationType

feature_criteria = BiasSegmentationCriteria(
    segmentation_type=BiasSegmentationType.Label,
    label="female"
)
ground_truth_criteria = BiasSegmentationCriteria(
    segmentation_type=BiasSegmentationType.Label,
    label=1
)

monitor = DataBiasStreaming(
    feature_segment_criteria=feature_criteria,
    ground_truth_segment_criteria=ground_truth_criteria,
    feature_data=[...],   # baseline feature data
    ground_truth_data=[...]  # baseline ground truth data
)

# Accumulate data as it arrives.
monitor.push(feature_value="female", ground_truth_value=1)
monitor.push_batch(feature_data=[...], ground_truth_data=[...])

# Evaluate at any point.
report = monitor.drift_report(drift_threshold=0.10)
if not report.passed:
    print("Bias drift detected", report.failed_report)

# Discard runtime data and start a new window, keeping the baseline.
monitor.flush()
```

### Model Performance

```python
from fair_perf_ml import model_perf

# --- At training time ---

# Choose one based on model type.
baseline = model_perf.linear_regression_analysis(y_true=bl_true, y_pred=bl_pred)
# baseline = model_perf.binary_classification_analysis(y_true=bl_true, y_pred=bl_pred)
# baseline = model_perf.logistic_regression_analysis(y_true=bl_true, y_pred=bl_pred, decision_threshold=0.5)

# Save baseline to persistent storage.

# --- At runtime ---

runtime = model_perf.linear_regression_analysis(y_true=rt_true, y_pred=rt_pred)

# Full check across all metrics.
result = model_perf.runtime_check_full(baseline=baseline, latest=runtime)

# Or check a specific subset of metrics.
result = model_perf.partial_runtime_check(
    baseline=baseline,
    latest=runtime,
    metrics=["RootMeanSquaredError", "MeanSquaredError", "RSquared"]
)

if result.passed:
    print("Performance check passed")
else:
    print("Performance degraded", result.failed_report)
```

The output schema for `linear_regression_analysis`:
```json
{
    "modelType": "LinearRegression",
    "performanceData": {
        "RootMeanSquaredError": float,
        "MeanSquaredError": float,
        "MeanAbsoluteError": float,
        "RSquared": float,
        "MaxError": float,
        "MeanSquaredLogError": float,
        "RootMeanSquaredLogError": float,
        "MeanAbsolutePercentageError": float
    }
}
```

The output schema for `binary_classification_analysis`:
```json
{
    "modelType": "BinaryClassification",
    "performanceData": {
        "BalancedAccuracy": float,
        "PrecisionPositive": float,
        "PrecisionNegative": float,
        "RecallPositive": float,
        "RecallNegative": float,
        "Accuracy": float,
        "F1Score": float
    }
}
```

The output schema for `logistic_regression_analysis`:
```json
{
    "modelType": "LogisticRegression",
    "performanceData": {
        "BalancedAccuracy": float,
        "PrecisionPositive": float,
        "PrecisionNegative": float,
        "RecallPositive": float,
        "RecallNegative": float,
        "Accuracy": float,
        "F1Score": float,
        "LogLoss": float
    }
}
```

All drift report results share this structure:
```json
{
    "passed": bool,
    "failed_report": [
        { "metric": "MetricName", "baseline_value": float, "latest_value": float, "delta": float }
    ]
}
```

### Model Performance Streaming

For API-served models where predictions and ground truth arrive continuously, use the streaming monitors to maintain a rolling window of performance metrics.

```python
from fair_perf_ml.model_perf.streaming import (
    BinaryClassificationStreaming,
    LinearRegressionStreaming,
    LogisticRegressionStreaming,
)

# Initialize with baseline data (e.g. from a holdout set).
monitor = BinaryClassificationStreaming(
    label=1,          # positive class label
    y_true=[...],     # baseline ground truth
    y_pred=[...]      # baseline predictions
)

# Accumulate predictions and ground truth as they arrive.
monitor.update_stream(y_true=1, y_pred=1)
monitor.update_stream_batch(y_true=[...], y_pred=[...])

# Snapshot current performance metrics.
snapshot = monitor.performance_snapshot()

# Evaluate drift against the baseline.
report = monitor.drift_report(drift_threshold=0.05)
if not report.passed:
    print("Performance drift detected", report.failed_report)

# Check only specific metrics.
from fair_perf_ml.models import ClassificationDriftMetric
partial = monitor.drift_report_partial_metrics(
    metrics=[ClassificationDriftMetric.F1Score, ClassificationDriftMetric.Accuracy],
    drift_threshold=0.05
)

# Reset the runtime window without touching the baseline.
monitor.flush()

# Replace the baseline entirely (e.g. after retraining).
monitor.reset_baseline(y_true=[...], y_pred=[...])
```

`LinearRegressionStreaming` and `LogisticRegressionStreaming` have the same interface. `LogisticRegressionStreaming` additionally accepts a `threshold: float | None = 0.5` decision threshold in the constructor and exposes `reset_baseline_and_decision_threshold(y_true, y_pred, threshold)`.

### Data Drift

```python
from fair_perf_ml.drift.base import ContinuousDataDrift, CategoricalDataDrift

# Continuous features.
drift = ContinuousDataDrift(baseline_data=[...], quantile_type="FreedmanDiaconis")
score = drift.compute_drift(runtime_data=[...], drift_metric="JensenShannon")
scores = drift.compute_drift_multiple_criteria(
    runtime_data=[...],
    drift_metrics=["JensenShannon", "WassersteinDistance"]
)

# Categorical features.
drift = CategoricalDataDrift(baseline_data=["cat", "dog", "cat", ...])
score = drift.compute_drift(runtime_data=[...], drift_metric="PopulationStabilityIndex")
```

### Data Drift Streaming

```python
from fair_perf_ml.drift.streaming import (
    StreamingContinuousDataDriftFlush,
    StreamingContinuousDataDriftDecay,
    StreamingCategoricalDataDriftFlush,
    StreamingCategoricalDataDriftDecay,
)

# Flush strategy: discard data after flush_rate samples or flush_cadence seconds.
monitor = StreamingContinuousDataDriftFlush(
    baseline_dataset=[...],
    quantile_type="FreedmanDiaconis",
    flush_rate=10_000,
    flush_cadence=3600,
)

monitor.update_stream(1.23)
monitor.update_stream_batch([1.1, 2.2, 3.3])

score = monitor.compute_drift("JensenShannon")
monitor.flush()  # manually reset the runtime window

# Decay strategy: older samples are down-weighted using exponential decay.
monitor = StreamingContinuousDataDriftDecay(
    baseline_dataset=[...],
    quantile_type=None,
    decay_half_life=3600,  # seconds
)
monitor.update_stream_batch([...])
score = monitor.compute_drift("WassersteinDistance")
```

## Future
- Multi-dimensional segmenter for bias analysis (currently requires one monitor per feature)
    - This can be done as-is with multiple monitors, but a unified multi-feature API would reduce boilerplate

