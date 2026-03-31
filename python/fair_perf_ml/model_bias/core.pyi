from fair_perf_ml.bias.segmentation import (
    BiasDataPayload,
    DiscreteAnalysisSegmentationValueBounds,
)
from fair_perf_ml.models import AnalysisReport, DriftReport, ModelBiasDriftMetric
from numpy.typing import NDArray

def model_bias_perform_analysis_explicit_segmentation[
    F: DiscreteAnalysisSegmentationValueBounds,
    G: DiscreteAnalysisSegmentationValueBounds,
    P: DiscreteAnalysisSegmentationValueBounds,
](
    feature: BiasDataPayload[F],
    ground_truth: BiasDataPayload[G],
    prediction: BiasDataPayload[P],
) -> AnalysisReport: ...
def model_bias_perform_analysis[
    F, G, P
](
    feature: list[F] | NDArray,
    ground_truth: list[G] | NDArray,
    predictions: list[P] | NDArray,
    feature_label_or_threshold: F,
    ground_truth_label_or_threshold: G,
    prediction_label_or_threshold: P,
) -> AnalysisReport: ...
def model_bias_runtime_comparison(
    baseline: AnalysisReport, comparison: AnalysisReport, threshold: float | None = 0.10
) -> DriftReport: ...
def model_bias_partial_runtime_comparison(
    baseline: AnalysisReport,
    comparison: AnalysisReport,
    metrics: list[ModelBiasDriftMetric],
    threshold: float | None = None,
) -> DriftReport: ...
