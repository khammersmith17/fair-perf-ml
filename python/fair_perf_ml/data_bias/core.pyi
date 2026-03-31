from __future__ import annotations

from fair_perf_ml.bias.segmentation import (
    BiasDataPayload,
    DiscreteAnalysisSegmentationValueBounds,
)
from fair_perf_ml.models import AnalysisReport, DataBiasDriftMetric, DriftReport
from numpy.typing import NDArray

def data_bias_perform_analysis_explicit_segmentation[
    F: DiscreteAnalysisSegmentationValueBounds,
    G: DiscreteAnalysisSegmentationValueBounds,
](feature: BiasDataPayload[F], ground_truth: BiasDataPayload[G]) -> AnalysisReport: ...
def data_bias_perform_analysis[
    F, G
](
    feature: list[F] | NDArray,
    ground_truth: list[G] | NDArray,
    feature_label_or_threshold: F,
    ground_truth_label_or_threshold: G,
) -> AnalysisReport: ...
def data_bias_runtime_comparison(
    baseline: AnalysisReport,
    latest: AnalysisReport,
    threshold: float = 0.10,
) -> DriftReport: ...
def data_bias_partial_runtime_comparison(
    baseline: AnalysisReport,
    latest: AnalysisReport,
    metrics: list[DataBiasDriftMetric],
    threshold: float = 0.10,
) -> DriftReport: ...
