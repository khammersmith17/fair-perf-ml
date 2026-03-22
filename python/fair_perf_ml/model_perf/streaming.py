"""
This module contains streaming containers for the model performance
utilities found in this library. The streaming implemenations are implemented
in the rust crate and expose Python interfaces used here.

The streaming utilities are meant for long running services, and provide
a space and computationally efficient way to observe model performance for
an ML service. Snapshots are provided that perform snapshot analysis of model
performance at a point in time. There are utilities to update on the fly when
there is a model update, or data drifts, without needing to pull down the service.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Self
from collections.abc import Iterable
from .._fair_perf_ml import (
    PyBinaryClassificationStreaming,
    PyLinearRegressionStreaming,
    PyLogisticRegressionStreaming,
)
import numpy as np
from ..models import (
    DriftReport,
    PerformanceSnapshot,
    DriftSnapshot,
    ModelPerformanceDriftMetric,
    ClassificationDriftMetric,
    LinearRegressionDriftMetric,
)
from .._internal import cast_floating_point_slice


class LabelBound(Protocol):
    """
    Protocol that enforces a label type must implement explicit equality comparison.
    Any type that implements __eq__ satisfies this protocol.
    """

    def __eq__(self, other: Self, /) -> bool: ...


class ModelPerfStreamingBase[T](ABC):
    """
    Abstract base for stateful streaming model performance monitors.
    Defines the interface all streaming monitors implement.
    """

    @abstractmethod
    def reset_baseline(self, y_true: Iterable[T], y_pred: Iterable[T]) -> None: ...

    @abstractmethod
    def update_stream(self, y_true: T, y_pred: T) -> None: ...

    @abstractmethod
    def update_stream_batch(self, y_true: Iterable[T], y_pred: Iterable[T]) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def performance_snapshot(self) -> PerformanceSnapshot: ...

    @abstractmethod
    def drift_snapshot(self) -> DriftSnapshot: ...

    @abstractmethod
    def drift_report(self, drift_threshold: float) -> DriftReport: ...

    @abstractmethod
    def drift_report_partial_metrics(
        self, metrics: list[ModelPerformanceDriftMetric], drift_threshold: float
    ) -> DriftReport: ...


class LabeledStreamingBase[T: LabelBound](ModelPerfStreamingBase[T]): ...


class BinaryClassificationStreaming[T: LabelBound](LabeledStreamingBase[T]):
    """
    Stateful streaming monitor for binary classification models.

    Maintains a baseline computed from an initial dataset. As new predictions
    arrive they are accumulated in the stream and can be evaluated against the
    baseline at any point via snapshots or drift reports.

    The positive class label can be any type that supports equality comparison.
    Labels are applied in Python before being accumulated in the monitor.
    """

    __slots__ = ["_inner", "_label"]

    def __init__(
        self,
        label: T,
        y_true: Iterable[T],
        y_pred: Iterable[T],
    ):
        """
        Initialises the monitor with a positive class label and a baseline dataset.
        args:
            label: T - the positive class label
            y_true: Iterable[T] - baseline ground truth values
            y_pred: Iterable[T] - baseline prediction values
        """
        self._label = label
        labeled_y_true = self._apply_label_batch(y_true)
        labeled_y_pred = self._apply_label_batch(y_pred)

        self._inner = PyBinaryClassificationStreaming(labeled_y_true, labeled_y_pred)

    def _apply_label(self, value: T) -> int:
        return int(value == self._label)

    def _apply_label_batch(self, value_slice: Iterable[T]) -> list[int]:
        arr = np.array(value_slice)
        return (arr == self._label).astype(np.int8).tolist()

    def update_stream(self, y_true: T, y_pred: T) -> None:
        """
        Accumulate a single ground truth and prediction example into the stream.
        args:
            y_true: T - ground truth value
            y_pred: T - prediction value
        """
        true_label = int(self._label == y_true)
        pred_label = int(self._label == y_pred)
        self._inner.push(true_label, pred_label)

    def update_stream_batch(self, y_true: Iterable[T], y_pred: Iterable[T]) -> None:
        """
        Accumulate a batch of ground truth and prediction examples into the stream.
        args:
            y_true: Iterable[T] - ground truth values
            y_pred: Iterable[T] - prediction values
        """
        labeled_y_true = self._apply_label_batch(y_true)
        labeled_y_pred = self._apply_label_batch(y_pred)
        self._inner.push_batch(labeled_y_true, labeled_y_pred)

    def flush(self) -> None:
        """
        Discard all accumulated runtime data. The baseline is preserved.
        """
        self._inner.flush()

    def reset_baseline(self, y_true: Iterable[T], y_pred: Iterable[T]) -> None:
        """
        Replace the baseline with a new dataset. The positive label is unchanged.
        To change the label at the same time use reset_baseline_and_label.
        args:
            y_true: Iterable[T] - new baseline ground truth values
            y_pred: Iterable[T] - new baseline prediction values
        """
        labeled_y_true = self._apply_label_batch(y_true)
        labeled_y_pred = self._apply_label_batch(y_pred)
        self._inner.reset_baseline(labeled_y_true, labeled_y_pred)

    def reset_baseline_and_label(
        self,
        label: T,
        y_true: Iterable[T],
        y_pred: Iterable[T],
    ) -> None:
        """
        Replace the baseline and the positive class label simultaneously.
        Because changing the label invalidates all previously accumulated state,
        a new baseline dataset is required. This is the only method that allows
        the label to be changed.
        args:
            label: T - new positive class label
            y_true: Iterable[T] - new baseline ground truth values
            y_pred: Iterable[T] - new baseline prediction values
        """
        self._label = label
        labeled_y_true = self._apply_label_batch(y_true)
        labeled_y_pred = self._apply_label_batch(y_pred)
        self._inner.reset_baseline(labeled_y_true, labeled_y_pred)

    def performance_snapshot(self) -> PerformanceSnapshot:
        """
        Compute a point-in-time performance report over all accumulated runtime
        examples, independent of the baseline. Raises if no runtime data has been
        accumulated yet.
        returns:
            PerformanceSnapshot - metric name to value mapping
        """
        return self._inner.performance_snapshot()

    def drift_snapshot(self) -> DriftSnapshot:
        """
        Compute a point-in-time drift snapshot comparing accumulated runtime
        examples to the baseline using a default threshold. Raises if no runtime
        data has been accumulated yet.
        returns:
            DriftSnapshot - metric name to drift value mapping
        """
        return self._inner.drift_snapshot()

    def drift_report(self, drift_threshold: float) -> DriftReport:
        """
        Evaluate whether accumulated runtime performance has drifted beyond the
        given threshold relative to the baseline. Raises if no runtime data has
        been accumulated yet.
        args:
            drift_threshold: float - maximum allowable drift per metric
        returns:
            DriftReport - pass/fail result with details of any exceeded metrics
        """
        return self._inner.drift_report(drift_threshold)

    def drift_report_partial_metrics(
        self, metrics: list[ClassificationDriftMetric], drift_threshold: float
    ) -> DriftReport:
        """
        Same as drift_report but scoped to a specific subset of metrics.
        args:
            metrics: list[ClassificationDriftMetric] - metrics to evaluate
            drift_threshold: float - maximum allowable drift per metric
        returns:
            DriftReport - pass/fail result with details of any exceeded metrics
        """
        return self._inner.drift_report_partial_metrics(metrics, drift_threshold)


class LinearRegressionStreaming(ModelPerfStreamingBase[float]):
    """
    Stateful streaming monitor for linear regression models.

    Maintains a baseline computed from an initial dataset. As new predictions
    arrive they are accumulated in the stream and can be evaluated against the
    baseline at any point via snapshots or drift reports.
    """

    __slots__ = ["_inner"]

    def __init__(self, y_true: Iterable[float], y_pred: Iterable[float]):
        """
        Initialises the monitor with a baseline dataset.
        args:
            y_true: Iterable[float] - baseline ground truth values
            y_pred: Iterable[float] - baseline prediction values
        """
        self._inner = PyLinearRegressionStreaming(
            cast_floating_point_slice(y_true), cast_floating_point_slice(y_pred)
        )

    def update_stream(self, y_true: float, y_pred: float) -> None:
        """
        Accumulate a single ground truth and prediction example into the stream.
        args:
            y_true: float
            y_pred: float
        """
        self._inner.push(y_true, y_pred)

    def update_stream_batch(
        self, y_true: Iterable[float], y_pred: Iterable[float]
    ) -> None:
        """
        Accumulate a batch of ground truth and prediction examples into the stream.
        args:
            y_true: Iterable[float]
            y_pred: Iterable[float]
        """
        self._inner.push_batch(
            cast_floating_point_slice(y_true), cast_floating_point_slice(y_pred)
        )

    def reset_baseline(self, y_true: Iterable[float], y_pred: Iterable[float]) -> None:
        """
        Replace the baseline with a new dataset.
        args:
            y_true: Iterable[float] - new baseline ground truth values
            y_pred: Iterable[float] - new baseline prediction values
        """
        self._inner.reset_baseline(
            cast_floating_point_slice(y_true), cast_floating_point_slice(y_pred)
        )

    def flush(self) -> None:
        """
        Discard all accumulated runtime data. The baseline is preserved.
        """
        self._inner.flush()

    def performance_snapshot(self) -> PerformanceSnapshot:
        """
        Compute a point-in-time performance report over all accumulated runtime
        examples, independent of the baseline. Raises if no runtime data has been
        accumulated yet.
        returns:
            PerformanceSnapshot - metric name to value mapping
        """
        return self._inner.performance_snapshot()

    def drift_snapshot(self) -> DriftSnapshot:
        """
        Compute a point-in-time drift snapshot comparing accumulated runtime
        examples to the baseline using a default threshold. Raises if no runtime
        data has been accumulated yet.
        returns:
            DriftSnapshot - metric name to drift value mapping
        """
        return self._inner.drift_snapshot()

    def drift_report(self, drift_threshold: float) -> DriftReport:
        """
        Evaluate whether accumulated runtime performance has drifted beyond the
        given threshold relative to the baseline. Raises if no runtime data has
        been accumulated yet.
        args:
            drift_threshold: float - maximum allowable drift per metric
        returns:
            DriftReport - pass/fail result with details of any exceeded metrics
        """
        return self._inner.drift_report(drift_threshold)

    def drift_report_partial_metrics(
        self, metrics: list[LinearRegressionDriftMetric], drift_threshold: float
    ) -> DriftReport:
        """
        Same as drift_report but scoped to a specific subset of metrics.
        args:
            metrics: list[LinearRegressionDriftMetric] - metrics to evaluate
            drift_threshold: float - maximum allowable drift per metric
        returns:
            DriftReport - pass/fail result with details of any exceeded metrics
        """
        return self._inner.drift_report_partial_metrics(metrics, drift_threshold)


class LogisticRegressionStreaming(ModelPerfStreamingBase[float]):
    """
    Stateful streaming monitor for logistic regression models.

    Maintains a baseline computed from an initial dataset. As new predictions
    arrive they are accumulated in the stream and can be evaluated against the
    baseline at any point via snapshots or drift reports. A decision threshold
    is applied to predicted probabilities to produce binary labels.
    """

    __slots__ = ["_inner"]

    def __init__(
        self,
        y_true: Iterable[float],
        y_pred: Iterable[float],
        threshold: float | None = 0.5,
    ):
        """
        Initialises the monitor with a baseline dataset and a decision threshold.
        args:
            y_true: Iterable[float] - baseline ground truth values
            y_pred: Iterable[float] - baseline predicted probabilities
            threshold: float | None - decision threshold applied to probabilities,
                defaults to 0.5
        """
        self._inner: PyLogisticRegressionStreaming = PyLogisticRegressionStreaming(
            cast_floating_point_slice(y_true),
            cast_floating_point_slice(y_pred),
            threshold,
        )

    def update_stream(self, y_true: float, y_pred: float) -> None:
        """
        Accumulate a single ground truth and predicted probability into the stream.
        args:
            y_true: float
            y_pred: float - predicted probability
        """
        self._inner.push(y_true, y_pred)

    def update_stream_batch(
        self, y_true: Iterable[float], y_pred: Iterable[float]
    ) -> None:
        """
        Accumulate a batch of ground truth values and predicted probabilities
        into the stream.
        args:
            y_true: Iterable[float]
            y_pred: Iterable[float] - predicted probabilities
        """
        self._inner.push_batch(
            cast_floating_point_slice(y_true), cast_floating_point_slice(y_pred)
        )

    def flush(self) -> None:
        """
        Discard all accumulated runtime data. The baseline is preserved.
        """
        self._inner.flush()

    def reset_baseline(self, y_true: Iterable[float], y_pred: Iterable[float]) -> None:
        """
        Replace the baseline with a new dataset. The decision threshold is unchanged.
        To change the threshold at the same time use reset_baseline_and_decision_threshold.
        args:
            y_true: Iterable[float] - new baseline ground truth values
            y_pred: Iterable[float] - new baseline predicted probabilities
        """
        self._inner.reset_baseline(
            cast_floating_point_slice(y_true), cast_floating_point_slice(y_pred)
        )

    def reset_baseline_and_decision_threshold(
        self, y_true: Iterable[float], y_pred: Iterable[float], threshold: float
    ) -> None:
        """
        Replace the baseline and the decision threshold simultaneously. Because
        changing the threshold invalidates previously computed baseline statistics,
        a new baseline dataset is required. This is the only method that allows
        the threshold to be changed.
        args:
            y_true: Iterable[float] - new baseline ground truth values
            y_pred: Iterable[float] - new baseline predicted probabilities
            threshold: float - new decision threshold
        """
        self._inner.reset_baseline_and_decision_threshold(
            cast_floating_point_slice(y_true),
            cast_floating_point_slice(y_pred),
            threshold,
        )

    def performance_snapshot(self) -> PerformanceSnapshot:
        """
        Compute a point-in-time performance report over all accumulated runtime
        examples, independent of the baseline. Raises if no runtime data has been
        accumulated yet.
        returns:
            PerformanceSnapshot - metric name to value mapping
        """
        return self._inner.performance_snapshot()

    def drift_snapshot(self) -> DriftSnapshot:
        """
        Compute a point-in-time drift snapshot comparing accumulated runtime
        examples to the baseline using a default threshold. Raises if no runtime
        data has been accumulated yet.
        returns:
            DriftSnapshot - metric name to drift value mapping
        """
        return self._inner.drift_snapshot()

    def drift_report(self, drift_threshold: float) -> DriftReport:
        """
        Evaluate whether accumulated runtime performance has drifted beyond the
        given threshold relative to the baseline. Raises if no runtime data has
        been accumulated yet.
        args:
            drift_threshold: float - maximum allowable drift per metric
        returns:
            DriftReport - pass/fail result with details of any exceeded metrics
        """
        return self._inner.drift_report(drift_threshold)

    def drift_report_partial_metrics(
        self, metrics: list[ClassificationDriftMetric], drift_threshold: float
    ) -> DriftReport:
        """
        Same as drift_report but scoped to a specific subset of metrics.
        args:
            metrics: list[ClassificationDriftMetric] - metrics to evaluate
            drift_threshold: float - maximum allowable drift per metric
        returns:
            DriftReport - pass/fail result with details of any exceeded metrics
        """
        return self._inner.drift_report_partial_metrics(metrics, drift_threshold)
