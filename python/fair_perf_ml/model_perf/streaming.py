from typing import Protocol, Generic, TypeVar, Self, Iterable, Dict, Optional
from .._fair_perf_ml import (
    PyBinaryClassificationStreaming,
    PyLinearRegressionStreaming,
    PyLogisticRegressionStreaming,
)
from ..models import DriftReport


class LabelBounds(Protocol):
    """
    Type used for label in binary classification needs to implement an
    expclicit __eq__. Declaring a protocol to be enforced by type checkers.
    """

    def __eq__(self, other: Self, /) -> bool: ...


L = TypeVar("L", bound=LabelBounds)


class BinaryClassificationStreaming(Generic[L]):
    """
    Wrapper around core rust logic to provide a Pythonic api to users.
    Requires a label, which represents the positive class label. This
    label can be any type as long as there is an explicit __eq__ implementation.
    The label will be applied in Python before the labeled data is passed to the
    rust code.
    """

    __slots__ = ["_inner", "_label"]

    def __init__(self, label: L, y_true: Iterable[L], y_pred: Iterable[L]):
        """
        Define the positive label, and the baseline state.
        args:
            label: L - positive class label
            y_true: Iterable[L] - baseline ground truth dataset
            y_pred: Iterable[L] - baseline prediction dataset
        """
        self._label = label
        labeled_y_true = list(map(self._apply_label, y_true))
        labeled_y_pred = list(map(self._apply_label, y_pred))

        self._inner = PyBinaryClassificationStreaming(labeled_y_true, labeled_y_pred)

    def _apply_label(self, value: L) -> int:
        """
        Internal method to assign binary bit label based on user defined type.
        """
        return int(value == self._label)

    def flush(self) -> None:
        """
        Clear the baseline state.
        """
        self._inner.flush()

    def push(self, y_true: L, y_pred: L) -> None:
        """
        Push a single runtime example into the stream.
        args:
            y_true: L
            y_pred: L
        """
        self._inner.push(self._apply_label(y_true), self._apply_label(y_pred))

    def push_batch(self, y_true: Iterable[L], y_pred: Iterable[L]) -> None:
        """
        Push a batch of runtime examples into the stream.
        args:
            y_true: Iterable[L]
            y_pred: Iterable[L]
        """
        labeled_y_true = list(map(self._apply_label, y_true))
        labeled_y_pred = list(map(self._apply_label, y_pred))
        self._inner.push_batch(labeled_y_true, labeled_y_pred)

    def reset_baseline(self, y_true: Iterable[L], y_pred: Iterable[L]) -> None:
        """
        Reset the baseline state. This will recompute the baseline state on new data
        using the same label. To change the label, use the reset_baseline_and_label method.
        args:
            y_true: Iterable[L]
            y_pred: Iterable[L]
        """
        labeled_y_true = list(map(self._apply_label, y_true))
        labeled_y_pred = list(map(self._apply_label, y_pred))
        self._inner.reset_baseline(labeled_y_true, labeled_y_pred)

    def reset_baseline_and_label(
        self, label: L, y_true: Iterable[L], y_pred: Iterable[L]
    ) -> None:
        """
        Reset the baseline state and the positive label. This will recompute
        the baseline state on new data using the newly defined label. This is the
        only method that supports changing the label. When the label is changed, any
        previous state is invalidated, thus requiring clearing all previous state.
        args:
            y_true: Iterable[L]
            y_pred: Iterable[L]
        """
        self._label = label
        labeled_y_true = list(map(self._apply_label, y_true))
        labeled_y_pred = list(map(self._apply_label, y_pred))
        self._inner.reset_baseline(labeled_y_true, labeled_y_pred)

    def performance_snapshot(self) -> Dict[str, float]:
        """
        Export a performance snapshot on the runtime performance of all examples
        accumulated in the stream, irrespective of the baseline data. This method will
        throw an exception if no runtime data has been accumulated in the stream.
        """
        report: Dict[str, float] = self._inner.performance_snapshot()
        return report

    def drift_snapshot(self) -> DriftReport:
        """
        Export a drift snapshot on the runtime performance of all examples
        accumulated in the stream, with respect to the baseline state. This method will
        throw an exception if no runtime data has been accumulated in the stream.
        """
        report: DriftReport = self.drift_snapshot()
        return report


# TODO: doc strings
class LinearRegressionStreaming:
    __slots__ = ["_inner"]

    def __init__(self, y_true: Iterable[float], y_pred: Iterable[float]):
        self._inner = PyLinearRegressionStreaming(y_true, y_pred)

    def push(self, y_true: float, y_pred: float) -> None:
        self._inner.push(y_true, y_pred)

    def push_batch(self, y_true: Iterable[float], y_pred: Iterable[float]) -> None:
        self._inner.push_batch(y_true, y_pred)

    def reset_baseline(self, y_true: Iterable[float], y_pred: Iterable[float]) -> None:
        self._inner.reset_baseline(y_true, y_pred)

    def flush(self) -> None:
        self._inner.flush()

    def performance_snapshot(self) -> Dict[str, float]:
        report: Dict[str, float] = self._inner.performance_snapshot()
        return report

    def drift_snapshot(self) -> DriftReport:
        report: DriftReport = self._inner.drift_snpshot()
        return report


class LogisticRegressionStreaming:
    __slots__ = ["_inner"]

    def __init__(
        self,
        y_true: Iterable[float],
        y_pred: Iterable[float],
        threshold: Optional[float] = 0.5,
    ):
        self._inner: PyLogisticRegressionStreaming = PyLogisticRegressionStreaming(
            y_true, y_pred, threshold
        )

    def push(self, y_true: float, y_pred: float) -> None:
        self._inner.push(y_true, y_pred)

    def push_batch(self, y_true: Iterable[float], y_pred: Iterable[float]) -> None:
        self._inner.push_batch(y_true, y_pred)

    def flush(self) -> None:
        self._inner.flush()

    def reset_baseline(self, y_true: Iterable[float], y_pred: Iterable[float]):
        self._inner.reset_baseline(y_true, y_pred)

    def reset_baseline_and_decision_threshold(
        self, y_true: Iterable[float], y_pred: Iterable[float], threshold: float
    ) -> None:
        self._inner.reset_baseline_and_decision_threshold(y_true, y_pred, threshold)

    def performance_snapshot(self) -> Dict[str, float]:
        report: Dict[str, float] = self._inner.performance_report()
        return report

    def drift_snapshot(self) -> DriftReport:
        report: DriftReport = self._inner.drift_snapshot()
        return report
