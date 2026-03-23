from numpy.typing import NDArray

from ..models import DriftReport, ModelBiasDriftMetric

def perform_analysis(
    feature: list[str | float | int] | NDArray,
    ground_truth: list[str | float | int] | NDArray,
    predictions: list[str | float | int] | NDArray,
    feature_label_or_threshold: str | float | int,
    ground_truth_label_or_threshold: str | float | int,
    prediction_label_or_threshold: str | float | int,
) -> dict[str, float]: ...

def runtime_comparison(
    baseline: dict,
    comparison: dict,
    threshold: float | None = ...,
) -> DriftReport: ...

def partial_runtime_comparison(
    baseline: dict,
    comparison: dict,
    metrics: list[ModelBiasDriftMetric],
    threshold: float | None = ...,
) -> DriftReport: ...
