from numpy.typing import NDArray

from ..models import DataBiasDriftMetric, DriftReport

def perform_analysis(
    feature: list[str | float | int] | NDArray,
    ground_truth: list[str | float | int] | NDArray,
    feature_label_or_threshold: str | float | int,
    ground_truth_label_or_threshold: str | float | int,
) -> dict[str, float]: ...

def runtime_comparison(
    baseline: dict[str, float],
    latest: dict[str, float],
    threshold: float | None = ...,
) -> DriftReport: ...

def partial_runtime_comparison(
    baseline: dict[str, float],
    latest: dict[str, float],
    metrics: list[DataBiasDriftMetric],
    threshold: float | None = ...,
) -> DriftReport: ...
