from __future__ import annotations

from collections.abc import Sequence

from .._fair_perf_ml import PyDataBiasStreaming
from ..models import (DataBiasDriftMetric, DriftReport, DriftSnapshot,
                      PerformanceSnapshot)
from ..segmentation import BiasSegmentationProtocol, SegmentationValueBounds


class DataBiasStreaming[F: SegmentationValueBounds, G: SegmentationValueBounds]:
    __slots__ = ("_inner", "_f_seg_criteria", "_gt_seg_criteria")

    def __init__(
        self,
        feature_segment_criteria: BiasSegmentationProtocol[F],
        ground_truth_segment_criteria: BiasSegmentationProtocol[G],
        feature_data: Sequence[F],
        ground_truth_data: Sequence[G],
    ):
        self._f_seg_criteria = feature_segment_criteria
        self._gt_seg_criteria = ground_truth_segment_criteria
        labeled_feats = self._f_seg_criteria._label_batch(feature_data)
        labeled_gt = self._gt_seg_criteria._label_batch(ground_truth_data)
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
            self._f_seg_criteria._label(feature_value),
            self._gt_seg_criteria._label(ground_truth_value),
        )

    def push_batch(
        self, feature_data: Sequence[F], ground_truth_data: Sequence[G]
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

        labeled_feats = self._f_seg_criteria._label_batch(feature_data)
        labeled_gt = self._gt_seg_criteria._label_batch(ground_truth_data)

        self._inner.push_batch(labeled_feats, labeled_gt)

    def reset_baseline(
        self, feautre_data: Sequence[F], ground_truth_data: Sequence[G]
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
        labeled_feats = list(map(self._f_seg_criteria._label, feautre_data))
        labeled_gt = list(map(self._gt_seg_criteria._label, ground_truth_data))
        self._inner.reset_baseline(labeled_feats, labeled_gt)

    def reset_baseline_and_segmentation_criteria(
        self,
        updated_feature_segmentation: BiasSegmentationProtocol[F],
        updated_ground_truth_segmentation: BiasSegmentationProtocol[G],
        feautre_data: Sequence[F],
        ground_truth_data: Sequence[G],
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
        labeled_feats = list(map(self._f_seg_criteria._label, feautre_data))
        labeled_gt = list(map(self._gt_seg_criteria._label, ground_truth_data))
        self._inner.reset_baseline(labeled_feats, labeled_gt)

    def flush(self) -> None:
        """
        Clear the runtime data state.
        """
        self._inner.flush()

    def performance_snapshot(self) -> PerformanceSnapshot:
        """
        Generate a performance snapshot, irrespective of the baseline data state.
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
        drift_metrics: list[DataBiasDriftMetric],
        drift_threshold: float | None = 0.10,
    ) -> DriftReport:
        return self._inner.drift_report_partial_metrics(drift_metrics, drift_threshold)
