from __future__ import annotations
from enum import Enum
from typing import Union, Iterable, TypeVar, Generic, Dict, Protocol, Self
from .._fair_perf_ml import PyDataBiasStreaming, PyModelBiasStreaming
from ..models import DriftReport


class BiasSegmentationType(str, Enum):
    LABEL = "Label"
    THRESHOLD = "Threshold"


class ValueBounds(Protocol):
    """
    Protocol to enforces typing. The type used for segmentation should
    safely implement __eq__ and __ge__.
    """

    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: Self) -> bool: ...


F = TypeVar("F", bound=ValueBounds)
G = TypeVar("G", bound=ValueBounds)
P = TypeVar("P", bound=ValueBounds)


class BiasSegmentationCriteria(Generic[P]):
    __slots__ = ["_value", "_seg_type"]

    def __init__(self, value: P, seg_type: Union[BiasSegmentationType, str]):
        if isinstance(seg_type, str):
            try:
                seg_type = BiasSegmentationType(seg_type)
            except ValueError as exc:
                raise ValueError("Invalid segmentation type") from exc

        self._value = value
        self._seg_type = seg_type

    def label(self, value: P) -> int:
        """
        Assign a binary value to the value based on the internally defined
        segmentation logic.
        """
        if self._seg_type == BiasSegmentationType.LABEL:
            is_pos = value == self._value
        else:
            is_pos = value >= self._value

        return int(is_pos)


class DataBiasStreaming(Generic[F, G]):
    __slots__ = ["_inner", "_f_seg_criteria", "_gt_seg_criteria"]

    def __init__(
        self,
        feature_segment_criteria: BiasSegmentationCriteria[F],
        ground_truth_segment_criteria: BiasSegmentationCriteria[G],
        feautre_data: Iterable[F],
        ground_truth_data: Iterable[G],
    ):
        self._f_seg_criteria = feature_segment_criteria
        self._gt_seg_criteria = ground_truth_segment_criteria
        labeled_feats = list(map(self._f_seg_criteria.label, feautre_data))
        labeled_gt = list(map(self._gt_seg_criteria.label, ground_truth_data))
        self._inner: PyDataBiasStreaming = PyDataBiasStreaming(
            labeled_feats, labeled_gt
        )

    def push(self, feature_value: F, ground_truth_value: G) -> None:
        """
        Push a single feature and ground truth example into the stream. Types should be
        consistent with what is defined in the segmentation criteria.
        args:
            feature_value: F
            ground_truth_value: G
        returns:
            None
        """
        self._inner.push(
            self._f_seg_criteria.label(feature_value),
            self._gt_seg_criteria.label(ground_truth_value),
        )

    def push_batch(
        self, feautre_data: Iterable[F], ground_truth_data: Iterable[G]
    ) -> None:
        """
        Push a single feature and ground truth example into the stream. Types should be
        consistent with what is defined in the segmentation criteria, and the length of the
        2 arrays should be the same. If either invariant is broken, then an exception will be thrown.
        args:
            feature_value: Iterable[F]
            ground_truth_value: Iterable[G]
        returns:
            None
        """

        labeled_feats = list(map(self._f_seg_criteria.label, feautre_data))
        labeled_gt = list(map(self._gt_seg_criteria.label, ground_truth_data))

        self._inner.push_batch(labeled_feats, labeled_gt)

    def reset_baseline(
        self, feautre_data: Iterable[F], ground_truth_data: Iterable[G]
    ) -> None:
        """
        Reset the baseline state. The same segmentation criteria that was defined on
        object construction will be used.
        args:
            feature_value: Iterable[F]
            ground_truth_value: Iterable[G]
        returns:
            None
        """
        labeled_feats = list(map(self._f_seg_criteria.label, feautre_data))
        labeled_gt = list(map(self._gt_seg_criteria.label, ground_truth_data))
        self._inner.reset_baseline(labeled_feats, labeled_gt)

    def reset_baseline_and_segmentation_criteria(
        self,
        updated_feature_segmentation: BiasSegmentationCriteria[F],
        updated_ground_truth_segmentation: BiasSegmentationCriteria[G],
        feautre_data: Iterable[F],
        ground_truth_data: Iterable[G],
    ) -> None:
        """
        Reset the baseline state and update the segmentation criteria. This may be useful
        when there is a significant shift in the distribution of the data.
        args:
            feature_value: Iterable[F]
            ground_truth_value: Iterable[G]
        returns:
            None
        """
        self._f_seg_criteria = updated_feature_segmentation
        self._gt_seg_criteria = updated_ground_truth_segmentation
        labeled_feats = list(map(self._f_seg_criteria.label, feautre_data))
        labeled_gt = list(map(self._gt_seg_criteria.label, ground_truth_data))
        self._inner.reset_baseline(labeled_feats, labeled_gt)

    def flush(self) -> None:
        """
        Clear the runtime data state.
        """
        self._inner.flush()

    def performance_snapshot(self) -> Dict[str, float]:
        """
        Generate a performance snapshot, irrespective of the baseline data state.
        """
        report: Dict[str, float] = self._inner.performance_snapshot()
        return report

    def drift_snapshot(self) -> DriftReport:
        """
        Generate a drift report, detailing the drift from the baseline state observed
        in the runtime data stream.
        """
        report: DriftReport = self._inner.drift_report()
        return report
