from __future__ import annotations
from enum import Enum
from typing import Union, Iterable, TypeVar, Generic, Dict, Protocol, Self
from .._fair_perf_ml import PyDataBiasStreaming, PyModelBiasStreaming
from ..models import DriftReport


class BiasSegmentationType(str, Enum):
    LABEL = "Label"
    THRESHOLD = "Threshold"


class SegementationValueBounds(Protocol):
    """
    Protocol to enforces typing. The type used for segmentation should
    safely implement __eq__ and __ge__.
    """

    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: Self) -> bool: ...


# Generic for feature, prediction, and ground truth type
F = TypeVar("F", bound=SegementationValueBounds)
G = TypeVar("G", bound=SegementationValueBounds)
P = TypeVar("P", bound=SegementationValueBounds)


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
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
    ):
        self._f_seg_criteria = feature_segment_criteria
        self._gt_seg_criteria = ground_truth_segment_criteria
        labeled_feats = list(map(self._f_seg_criteria.label, feature_data))
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
        An exception will be thrown if no runtime data has been pushed into the stream.
        """
        report: Dict[str, float] = self._inner.performance_snapshot()
        return report

    def drift_snapshot(self) -> DriftReport:
        """
        Generate a drift report, detailing the drift from the baseline state observed
        in the runtime data stream.
        An exception will be thrown if no runtime data has been pushed into the stream.
        """
        report: DriftReport = self._inner.drift_report()
        return report


class ModelBiasStreaming(Generic[F, P, G]):
    __slots__ = ["_inner", "_f_seg_criteria", "_p_seg_criteria", "_gt_seg_criteria"]

    def __init__(
        self,
        feature_segment_criteria: BiasSegmentationCriteria[F],
        ground_truth_segment_criteria: BiasSegmentationCriteria[G],
        prediction_segment_criteria: BiasSegmentationCriteria[P],
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
        prediction_data: Iterable[P],
    ):
        self._f_seg_criteria = feature_segment_criteria
        self._p_seg_criteria = prediction_segment_criteria
        self._gt_seg_criteria = ground_truth_segment_criteria

        labeled_feats = list(map(self._f_seg_criteria.label, feature_data))
        labeled_gt = list(map(self._gt_seg_criteria.label, ground_truth_data))
        labeled_preds = list(map(self._p_seg_criteria.label, prediction_data))
        self._inner: PyModelBiasStreaming = PyModelBiasStreaming(
            labeled_feats, labeled_preds, labeled_gt
        )

    def push(self, feature: F, prediction: P, ground_truth: G) -> None:
        """
        Push a single feature, prediction, and ground truth example into the stream.
        Type checkers will enforce that the type passed for each value matches the type
        defined in the segmentation criteria.
        args:
            feature: F
            prediction: P
            ground_truth: G
        returns:
            None
        """
        self._inner.push(
            self._f_seg_criteria.label(feature),
            self._p_seg_criteria.label(prediction),
            self._gt_seg_criteria.label(ground_truth),
        )

    def push_batch(
        self,
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
        prediction_data: Iterable[P],
    ) -> None:
        """
        Push a single feature, prediction, and ground truth example into the stream.
        Type checkers will enforce that the type passed for each value matches the type
        defined in the segmentation criteria. The iterables/arrays passed should all be of
        the same length. If either invariant is violated, then an excpetion will be thrown.

        args:
            feature: Iterable[F]
            prediction: Iterable[P]
            ground_truth: Iterable[G]
        returns:
            None
        """
        labeled_feats = list(map(self._f_seg_criteria.label, feature_data))
        labeled_gt = list(map(self._gt_seg_criteria.label, ground_truth_data))
        labeled_preds = list(map(self._p_seg_criteria.label, prediction_data))

        self._inner.push(
            labeled_feats,
            labeled_preds,
            labeled_gt,
        )

    def reset_baseline(
        self,
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
        prediction_data: Iterable[P],
    ) -> None:
        """
        Resets the baseline state with using a new baseline dataset. This will leverage
        the same segmentation criteria defined at type construction.

        args:
            feature: Iterable[F]
            prediction: Iterable[P]
            ground_truth: Iterable[G]
        returns:
            None
        """
        labeled_feats = list(map(self._f_seg_criteria.label, feature_data))
        labeled_gt = list(map(self._gt_seg_criteria.label, ground_truth_data))
        labeled_preds = list(map(self._p_seg_criteria.label, prediction_data))
        self._inner.reset_baseline(labeled_feats, labeled_preds, labeled_gt)

    def reset_baseline_and_segmentation_criteria(
        self,
        feature_segment_criteria: BiasSegmentationCriteria[F],
        ground_truth_segment_criteria: BiasSegmentationCriteria[G],
        prediction_segment_criteria: BiasSegmentationCriteria[P],
        feature_data: Iterable[F],
        ground_truth_data: Iterable[G],
        prediction_data: Iterable[P],
    ) -> None:
        """
        Resets the baseline state with using a new baseline dataset. This is the only that allows
        a change in the segmentation criteria used for class segmentation. A reset in the
        baseline set is required, to avoid inconsistent state in runtime class bucketing.
        Type checkers will enforce the same type to be used in the segmentation criteria.

        args:
            feature: Iterable[F]
            prediction: Iterable[P]
            ground_truth: Iterable[G]
        returns:
            None
        """
        self._f_seg_criteria = feature_segment_criteria
        self._gt_seg_criteria = ground_truth_segment_criteria
        self._p_seg_criteria = prediction_segment_criteria

        labeled_feats = list(map(self._f_seg_criteria.label, feature_data))
        labeled_gt = list(map(self._gt_seg_criteria.label, ground_truth_data))
        labeled_preds = list(map(self._p_seg_criteria.label, prediction_data))
        self._inner.reset_baseline(labeled_feats, labeled_preds, labeled_gt)

    def flush(self) -> None:
        """
        Clear all runtime state.
        """
        self._inner.flush()

    def performance_snapshot(self) -> Dict[str, float]:
        """
        Generate a performance snapshot of runtime state, irrespective of the baseline state.
        An exception will be thrown if no runtime data has been pushed into the stream.
        """
        report: Dict[str, float] = self._inner.performance_snapshot()
        return report

    def drift_snapshot(self) -> DriftReport:
        """
        Generate a drift report, detailing the drift from the baseline state observed
        in the runtime data stream.
        An exception will be thrown if no runtime data has been pushed into the stream.
        """
        report: DriftReport = self._inner.drift_snapshot()
        return report
