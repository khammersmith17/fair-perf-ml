from __future__ import annotations

from collections.abc import Sequence

from .._fair_perf_ml import PyModelBiasStreaming
from ..models import (DriftReport, DriftSnapshot, ModelBiasDriftMetric,
                      PerformanceSnapshot)
from ..segmentation import BiasSegmentationProtocol, SegmentationValueBounds


class ModelBiasStreaming[
    F: SegmentationValueBounds, G: SegmentationValueBounds, P: SegmentationValueBounds
]:
    __slots__ = ["_inner", "_f_seg_criteria", "_p_seg_criteria", "_gt_seg_criteria"]

    def __init__(
        self,
        feature_segment_criteria: BiasSegmentationProtocol[F],
        ground_truth_segment_criteria: BiasSegmentationProtocol[G],
        prediction_segment_criteria: BiasSegmentationProtocol[P],
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
        prediction_data: Sequence[P],
    ):
        self._f_seg_criteria = feature_segment_criteria
        self._p_seg_criteria = prediction_segment_criteria
        self._gt_seg_criteria = ground_truth_segment_criteria

        labeled_feats = self._f_seg_criteria._label_batch(feature_data)
        labeled_gt = self._gt_seg_criteria._label_batch(ground_truth_data)
        labeled_preds = self._p_seg_criteria._label_batch(prediction_data)
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
            self._f_seg_criteria._label(feature),
            self._p_seg_criteria._label(prediction),
            self._gt_seg_criteria._label(ground_truth),
        )

    def push_batch(
        self,
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
        prediction_data: Sequence[P],
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
        labeled_feats = self._f_seg_criteria._label_batch(feature_data)
        labeled_gt = self._gt_seg_criteria._label_batch(ground_truth_data)
        labeled_preds = self._p_seg_criteria._label_batch(prediction_data)

        self._inner.push_batch(
            labeled_feats,
            labeled_preds,
            labeled_gt,
        )

    def reset_baseline(
        self,
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
        prediction_data: Sequence[P],
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
        labeled_feats = list(map(self._f_seg_criteria._label, feature_data))
        labeled_gt = list(map(self._gt_seg_criteria._label, ground_truth_data))
        labeled_preds = list(map(self._p_seg_criteria._label, prediction_data))
        self._inner.reset_baseline(labeled_feats, labeled_preds, labeled_gt)

    def reset_baseline_and_segmentation_criteria(
        self,
        feature_segment_criteria: BiasSegmentationProtocol[F],
        ground_truth_segment_criteria: BiasSegmentationProtocol[G],
        prediction_segment_criteria: BiasSegmentationProtocol[P],
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
        prediction_data: Sequence[P],
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

        labeled_feats = list(map(self._f_seg_criteria._label, feature_data))
        labeled_gt = list(map(self._gt_seg_criteria._label, ground_truth_data))
        labeled_preds = list(map(self._p_seg_criteria._label, prediction_data))
        self._inner.reset_baseline(labeled_feats, labeled_preds, labeled_gt)

    def flush(self) -> None:
        """
        Clear all runtime state.
        """
        self._inner.flush()

    def performance_snapshot(self) -> PerformanceSnapshot:
        """
        Generate a performance snapshot of runtime state, irrespective of the baseline state.
        An exception will be thrown if no runtime data has been pushed into the stream.
        """
        report: dict[str, float] = self._inner.performance_snapshot()
        return report

    def drift_snapshot(self) -> DriftSnapshot:
        """
        Generate a drift report, detailing the drift from the baseline state observed
        in the runtime data stream.
        An exception will be thrown if no runtime data has been pushed into the stream.
        """
        return self._inner.drift_snapshot()

    def drift_report(self, drift_threshold: float | None = 0.10) -> DriftReport:
        return self._inner.drift_report(drift_threshold)

    def drift_report_partial_metrics(
        self,
        drift_metrics: list[ModelBiasDriftMetric],
        drift_threshold: float | None = 0.10,
    ) -> DriftReport:
        return self._inner.drift_report_partial_metrics(drift_metrics, drift_threshold)
