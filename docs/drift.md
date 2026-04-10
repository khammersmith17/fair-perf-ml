# Data Drift

Drift detection between continuous and categorical distributions — batch and streaming.
Import from `fair_perf_ml.drift`.

Supported metrics: Jensen–Shannon Divergence, Population Stability Index (PSI),
Wasserstein Distance, Kullback–Leibler Divergence.

---

## Enums

::: fair_perf_ml.drift.base.DataDriftType

::: fair_perf_ml.drift.base.QuantileType

---

## Batch functions

::: fair_perf_ml.drift.base.compute_drift_continuous_distribution

::: fair_perf_ml.drift.base.compute_drift_categorical_distribution

---

## Batch classes

::: fair_perf_ml.drift.base.ContinuousDataDrift

::: fair_perf_ml.drift.base.CategoricalDataDrift

---

## Streaming — continuous

::: fair_perf_ml.drift.streaming.StreamingContinuousDataDriftFlush

::: fair_perf_ml.drift.streaming.StreamingContinuousDataDriftDecay

---

## Streaming — categorical

::: fair_perf_ml.drift.streaming.StreamingCategoricalDataDriftFlush

::: fair_perf_ml.drift.streaming.StreamingCategoricalDataDriftDecay

---

## Exceptions

::: fair_perf_ml.drift.base.DataDriftParameterValidationError
