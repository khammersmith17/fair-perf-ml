from numpy.typing import NDArray

from ..models import DataBiasDriftMetric, DriftReport

def perform_analysis[F, G](
    feature: list[F] | NDArray,
    ground_truth: list[G] | NDArray,
    feature_label_or_threshold: F,
    ground_truth_label_or_threshold: G,
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
