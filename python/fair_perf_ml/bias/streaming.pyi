from collections.abc import Iterable
from enum import Enum
from typing import Generic, Protocol, Self, TypeVar

from ..models import (DataBiasDriftMetric, DriftReport, DriftSnapshot,
                      ModelBiasDriftMetric, PerformanceSnapshot)

class BiasSegmentationType(str, Enum):
    Label: str
    Threshold: str

class BiasSegmentationThresholdType(str, Enum):
    GreaterThan: str
    GreaterThanEqaulTo: str
    LessThan: str
    LessThanEqaulTo: str

class SegementationValueBounds(Protocol):
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: Self) -> bool: ...

P = TypeVar("P", bound=SegementationValueBounds)
F = TypeVar("F", bound=SegementationValueBounds)
G = TypeVar("G", bound=SegementationValueBounds)

class BiasSegmentationCriteria(Generic[P]):
    def __init__(
        self,
        value: P,
        segmentation_type: BiasSegmentationType | str,
        threshold_type: BiasSegmentationThresholdType | str | None = ...,
    ) -> None: ...

class DataBiasStreaming(Generic[F, G]):
    def __init__(
        self,
        feature_segment_criteria: BiasSegmentationCriteria[F],
        ground_truth_segment_criteria: BiasSegmentationCriteria[G],
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
    ) -> None: ...
    def push(self, feature_value: F, ground_truth_value: G) -> None: ...
    def push_batch(self, feature_data: Iterable[F], ground_truth_data: Iterable[G]) -> None: ...
    def reset_baseline(self, feature_data: Iterable[F], ground_truth_data: Iterable[G]) -> None: ...
    def reset_baseline_and_segmentation_criteria(
        self,
        updated_feature_segmentation: BiasSegmentationCriteria[F],
        updated_ground_truth_segmentation: BiasSegmentationCriteria[G],
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
    ) -> None: ...
    def flush(self) -> None: ...
    def performance_snapshot(self) -> PerformanceSnapshot: ...
    def drift_snapshot(self) -> DriftSnapshot: ...
    def drift_report(self, drift_threshold: float | None = ...) -> DriftReport: ...
    def drift_report_partial_metrics(
        self,
        drift_metrics: list[DataBiasDriftMetric],
        drift_threshold: float | None = ...,
    ) -> DriftReport: ...

class ModelBiasStreaming(Generic[F, P, G]):
    def __init__(
        self,
        feature_segment_criteria: BiasSegmentationCriteria[F],
        ground_truth_segment_criteria: BiasSegmentationCriteria[G],
        prediction_segment_criteria: BiasSegmentationCriteria[P],
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
        prediction_data: Iterable[P],
    ) -> None: ...
    def push(self, feature: F, prediction: P, ground_truth: G) -> None: ...
    def push_batch(
        self,
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
        prediction_data: Iterable[P],
    ) -> None: ...
    def reset_baseline(
        self,
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
        prediction_data: Iterable[P],
    ) -> None: ...
    def reset_baseline_and_segmentation_criteria(
        self,
        feature_segment_criteria: BiasSegmentationCriteria[F],
        ground_truth_segment_criteria: BiasSegmentationCriteria[G],
        prediction_segment_criteria: BiasSegmentationCriteria[P],
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
        prediction_data: Iterable[P],
    ) -> None: ...
    def flush(self) -> None: ...
    def performance_snapshot(self) -> PerformanceSnapshot: ...
    def drift_snapshot(self) -> DriftSnapshot: ...
    def drift_report(self, drift_threshold: float | None = ...) -> DriftReport: ...
    def drift_report_partial_metrics(
        self,
        drift_metrics: list[ModelBiasDriftMetric],
        drift_threshold: float | None = ...,
    ) -> DriftReport: ...
