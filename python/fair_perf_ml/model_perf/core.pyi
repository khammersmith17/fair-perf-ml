
from fair_perf_ml._internal import FloatingPointDataSlice
from fair_perf_ml.models import (DriftReport, ModelPerformanceDriftMetric,
                      ModelPerformanceReport)

class DifferentModelTypes(Exception): ...
class InvalidMetricsBody(Exception): ...

def linear_regression_analysis(
    y_true: FloatingPointDataSlice,
    y_pred: FloatingPointDataSlice,
) -> dict: ...

def logistic_regression_analysis(
    y_true: FloatingPointDataSlice,
    y_pred: FloatingPointDataSlice,
    decision_threshold: float | None = ...,
) -> dict: ...

def binary_classification_analysis(
    y_true: FloatingPointDataSlice,
    y_pred: FloatingPointDataSlice,
) -> dict: ...

def runtime_check_full(
    latest: ModelPerformanceReport | dict,
    baseline: ModelPerformanceReport | dict,
    threshold: float = ...,
) -> DriftReport: ...

def partial_runtime_check(
    latest: ModelPerformanceReport | dict,
    baseline: ModelPerformanceReport | dict,
    metrics: list[ModelPerformanceDriftMetric],
    threshold: float = 0.10
) -> DriftReport: ...
