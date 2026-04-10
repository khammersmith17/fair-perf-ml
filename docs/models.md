# Types

Shared return types and metric enums. Import from `fair_perf_ml.models`.

---

## Return types

::: fair_perf_ml.models.DriftReport

::: fair_perf_ml.models.ModelPerformanceReport

::: fair_perf_ml.models.LinearRegressionReport

::: fair_perf_ml.models.LogisticRegressionReport

::: fair_perf_ml.models.BinaryClassificationReport

---

## Metric enums

::: fair_perf_ml.models.DataBiasMetric

::: fair_perf_ml.models.ModelBiasMetric

::: fair_perf_ml.models.ClassificationEvaluationMetric

::: fair_perf_ml.models.LinearRegressionEvaluationMetric

---

## Type aliases

| Alias | Definition |
|-------|-----------|
| `AnalysisReport` | `dict[str, float]` |
| `PerformanceSnapshot` | `dict[str, float]` |
| `DriftSnapshot` | `dict[str, float]` |
| `DataBiasDriftMetric` | `DataBiasMetric \| str` |
| `ModelBiasDriftMetric` | `ModelBiasMetric \| str` |
| `ClassificationDriftMetric` | `ClassificationEvaluationMetric \| str` |
| `LinearRegressionDriftMetric` | `LinearRegressionEvaluationMetric \| str` |
