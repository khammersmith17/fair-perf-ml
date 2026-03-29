from __future__ import annotations

from collections.abc import Sequence

from fair_perf_ml.bias.segmentation import (
    BiasSegmentationProtocol,
    SegmentationValueBounds,
)

from ..models import (
    DataBiasDriftMetric,
    DriftReport,
    DriftSnapshot,
    PerformanceSnapshot,
)

class DataBiasStreaming[F: SegmentationValueBounds, G: SegmentationValueBounds]:
    def push(self, feature_value: F, ground_truth_value: G) -> None: ...
    def push_batch(
        self, feature_data: Sequence[F], ground_truth_data: Sequence[G]
    ) -> None: ...
    def reset_baseline(
        self, feature_data: Sequence[F], ground_truth_data: Sequence[G]
    ) -> None: ...
    def reset_baseline_and_segmentation_criteria(
        self,
        updated_feature_segmentation: BiasSegmentationProtocol[F],
        updated_ground_truth_segmentation: BiasSegmentationProtocol[G],
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
    ) -> None: ...
    def flush(self) -> None: ...
    def performance_snapshot(self) -> PerformanceSnapshot: ...
    def drift_snapshot(self) -> DriftSnapshot: ...
    def drift_report(self, drift_threshold: float | None = 0.10) -> DriftReport: ...
    def drift_report_partial_metrics(
        self,
        drift_metrics: list[DataBiasDriftMetric],
        drift_threshold: float | None = 0.10,
    ) -> DriftReport: ...
