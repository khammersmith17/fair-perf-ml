from __future__ import annotations

from collections.abc import Sequence

from fair_perf_ml.bias.segmentation import (
    BiasSegmentationProtocol,
    SegmentationValueBounds,
)

from ..models import (
    DriftReport,
    DriftSnapshot,
    ModelBiasDriftMetric,
    PerformanceSnapshot,
)

class ModelBiasStreaming[
    F: SegmentationValueBounds, G: SegmentationValueBounds, P: SegmentationValueBounds
]:
    def __init__(
        self,
        feature_segment_criteria: BiasSegmentationProtocol[F],
        ground_truth_segment_criteria: BiasSegmentationProtocol[G],
        prediction_segment_criteria: BiasSegmentationProtocol[P],
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
        prediction_data: Sequence[P],
    ): ...
    def push(self, feature: F, prediction: P, ground_truth: G) -> None: ...
    def push_batch(
        self,
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
        prediction_data: Sequence[P],
    ) -> None: ...
    def reset_baseline(
        self,
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
        prediction_data: Sequence[P],
    ) -> None: ...
    def reset_baseline_and_segmentation_criteria(
        self,
        feature_segment_criteria: BiasSegmentationProtocol[F],
        ground_truth_segment_criteria: BiasSegmentationProtocol[G],
        prediction_segment_criteria: BiasSegmentationProtocol[P],
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
        prediction_data: Sequence[P],
    ) -> None: ...
    def flush(self) -> None: ...
    def performance_snapshot(self) -> PerformanceSnapshot: ...
    def drift_snapshot(self) -> DriftSnapshot: ...
    def drift_report(self, drift_threshold: float | None = 0.10) -> DriftReport: ...
    def drift_report_partial_metrics(
        self,
        drift_metrics: list[ModelBiasDriftMetric],
        drift_threshold: float | None = 0.10,
    ) -> DriftReport: ...
