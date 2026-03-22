from numpy.typing import NDArray
from ..models import (
    DriftReport,
    ModelPerformanceReport,
    ModelPerformanceDriftMetric,
)

class DifferentModelTypes(Exception): ...
class InvalidMetricsBody(Exception): ...

def linear_regression_analysis(
    y_true: NDArray | list[int | float],
    y_pred: NDArray | list[int | float],
) -> dict: ...

def logistic_regression_analysis(
    y_true: NDArray | list[int | float],
    y_pred: NDArray | list[int | float],
    decision_threshold: float | None = ...,
) -> dict: ...

def binary_classification_analysis(
    y_true: NDArray | list[int | float],
    y_pred: NDArray | list[int | float],
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
    threshold: float = ...,
) -> DriftReport: ...
