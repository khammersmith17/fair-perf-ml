from numpy.typing import NDArray

from ..models import DriftReport, ModelBiasDriftMetric

def perform_analysis[F, G, P](
    feature: list[F] | NDArray,
    ground_truth: list[G] | NDArray,
    predictions: list[P] | NDArray,
    feature_label_or_threshold: F,
    ground_truth_label_or_threshold: G,
    prediction_label_or_threshold: P,
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
