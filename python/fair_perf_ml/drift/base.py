"""
Provides runtime drift monitoring utilites and managers for such utilities.
These drift techniques can be used for data drift or a proxy for model drift
when ground truth feedback loop is slow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Protocol

import numpy as np
from fair_perf_ml._internal import FloatingPointDataSlice
from numpy.typing import NDArray

from .._fair_perf_ml import (PyCategoricalDataDrift, PyContinuousDataDrift,
                             py_compute_drift_categorical_distribtuion,
                             py_compute_drift_continuous_distribtuion)


class QuantileType(str, Enum):
    """
    Supported method for deriving the number of bins to use when approximating
    a continuous distribution.
    """

    FreedmanDiaconis = "FreedmanDiaconis"
    Scott = "Scott"
    Sturges = "Sturges"


class DataDriftType(str, Enum):
    """
    Currently supported methods of deriving the divergence between two distributions.
    """

    JensenShannon = "JensenShannon"
    PopulationStabilityIndex = "PopulationStabilityIndex"
    WassersteinDistance = "WassersteinDistance"
    KullbackLeibler = "KullbackLeibler"


type DataDriftMetric = DataDriftType | str
type QuantileConfig = QuantileType | str | None


def _map_drift_metric_type(m: DataDriftMetric) -> str:
    if isinstance(m, DataDriftType):
        return m.value
    return m


class StringBound(Protocol):
    """
    Protocol to enforces typing. The type used for segmentation should
    implement __str__ so whatever is passed in can be safely casted into a string.
    """

    def __str__(self) -> str: ...


class DataDriftParameterValidationError(Exception):
    """
    Exception for when users pass invalid data in
    """


def _cast_to_numpy_float_arr(arr: FloatingPointDataSlice) -> NDArray:
    """
    Convert a non numpy arr to numpy array of float64.
    """
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array([float(item) for item in arr], dtype=np.float64)
        except ValueError:
            raise DataDriftParameterValidationError(
                "Data in iterable must be castable to float"
            )
    return arr


def _cast_to_string_iterable(arr: Sequence[StringBound]) -> Sequence[str]:
    """
    Iterable of something that can be casted to a string to a Iterable[str].
    """
    return list(map(str, arr))


def compute_drift_categorical_distribution(
    baseline_distribution: list[StringBound],
    candidate_distribution: list[StringBound],
    drift_metrics: list[DataDriftMetric],
) -> list[float]:
    """
    Ad hoc computation of drift between two distributions of cateogrical data.

    args:
        baseline_distribution: list[StringBound]
        candidate_distribution: list[StringBound]
        drift_metrics: list[DataDriftMetric]
    returns:
        list[float] - one entry for every drift method provided, element wise mapped.
    """
    return py_compute_drift_categorical_distribtuion(
        _cast_to_string_iterable(baseline_distribution),
        _cast_to_string_iterable(candidate_distribution),
        list(map(_map_drift_metric_type, drift_metrics)),
    )


def compute_drift_continuous_distribution(
    baseline_distribution: FloatingPointDataSlice,
    candidate_distribution: FloatingPointDataSlice,
    drift_metrics: list[DataDriftMetric],
    quantile_type: QuantileConfig = None,
) -> list[float]:
    """
    Ad hoc computation of drift between two distributions of continuous data.

    args:
        baseline_distribution: list[StringBound]
        candidate_distribution: list[StringBound]
        drift_metrics: list[DataDriftMetric]
        quantile_type: QuantileConfig = None - defaults to FreedmanDiaconis
    returns:
        list[float] - one entry for every drift method provided, element wise mapped.
    """
    if isinstance(quantile_type, QuantileType):
        quantile_type = quantile_type.value
    return py_compute_drift_continuous_distribtuion(
        _cast_to_numpy_float_arr(baseline_distribution),
        _cast_to_numpy_float_arr(candidate_distribution),
        list(map(_map_drift_metric_type, drift_metrics)),
        quantile_type,
    )


class DataDriftDiscreteBase[T, B](ABC):
    """
    Abtract class to define the streaming data drift api contract.
    More for correctness constraint rather than utility here.
    """

    @abstractmethod
    def reset_baseline(self, new_baseline: list[T]): ...

    @abstractmethod
    def compute_drift(
        self, runtime_data: list[T], drift_metric: DataDriftMetric
    ) -> float: ...

    @abstractmethod
    def compute_drift_multiple_criteria(
        self, runtime_data: list[T], drift_metrics: list[DataDriftMetric]
    ) -> list[float]: ...

    @property
    @abstractmethod
    def num_bins(self) -> int:
        """
        The number of bins to approximate the distribution.
        Derived from the dataset on construction.
        """

    @abstractmethod
    def export_baseline(self) -> B: ...


class ContinuousDataDrift(DataDriftDiscreteBase[float, list[float]]):
    """
    Detects distributional drift in continuous (floating-point) features between
    a fixed baseline dataset and a runtime dataset.

    Internally, the baseline is summarized as a histogram. The number of bins is
    derived automatically from the baseline data using the selected quantile rule.
    Drift is then measured by comparing the runtime data's distribution against
    the baseline histogram using the chosen divergence metric.

    This type is suited for batch analysis: you collect a runtime dataset and
    compare it against the baseline in one call. For long-running accumulation
    where data arrives incrementally, use the streaming variants instead.

    Considerations:
        1. Bin count is determined by the baseline data and the quantile rule.
           If the data does not support the target bin count, fewer bins will
           be used.
        2. Resetting the baseline recomputes the histogram from scratch using
           the same quantile rule.
    """

    __slots__ = "_inner"

    def __init__(
        self,
        baseline_data: FloatingPointDataSlice,
        quantile_type: str | None = None,
    ):
        """
        Initialize with a baseline dataset.

        Args:
            baseline_data: The reference distribution. Accepts a numpy array or
                any iterable of values castable to float.
            quantile_type: Controls how many histogram bins are derived from the
                baseline. Options: ``"FreedmanDiaconis"`` (default, IQR-based,
                robust to outliers), ``"Scott"`` (std-based, assumes roughly
                normal data), ``"Sturges"`` (log2-based, best for small datasets).
                Pass ``None`` to use the default. Also accepts a ``QuantileType``
                enum value.
        """
        typed_data = _cast_to_numpy_float_arr(baseline_data)
        if isinstance(quantile_type, QuantileType):
            quantile_type = quantile_type.value
        self._inner = PyContinuousDataDrift(typed_data, quantile_type)

    def reset_baseline(self, new_baseline: FloatingPointDataSlice):
        """
        Replace the baseline with a new dataset, recomputing the histogram.

        Args:
            new_baseline: The new reference distribution. Accepts a numpy array
                or any iterable of values castable to float.
        """
        new_baseline = _cast_to_numpy_float_arr(new_baseline)
        self._inner.reset_baseline(new_baseline)

    def compute_drift(
        self, runtime_data: FloatingPointDataSlice, drift_metric: DataDriftMetric
    ) -> float:
        """
        Compute a single drift score between ``runtime_data`` and the baseline.

        Args:
            runtime_data: The data collected at runtime. Accepts a numpy array
                or any iterable of values castable to float.
            drift_metric: The divergence measure to use. Accepts a
                ``DataDriftType`` enum value or one of the strings
                ``"JensenShannon"``, ``"PopulationStabilityIndex"``,
                ``"WassersteinDistance"``, ``"KullbackLeibler"``.

        Returns:
            The drift score as a float. Higher values indicate greater divergence
            from the baseline distribution.
        """
        typed_data = _cast_to_numpy_float_arr(runtime_data)
        return self._inner.compute_drift(
            typed_data, _map_drift_metric_type(drift_metric)
        )

    def compute_drift_multiple_criteria(
        self,
        runtime_data: FloatingPointDataSlice,
        drift_metrics: list[DataDriftMetric],
    ) -> list[float]:
        """
        Compute multiple drift scores against ``runtime_data`` in a single pass.

        Args:
            runtime_data: The data collected at runtime. Accepts a numpy array
                or any iterable of values castable to float.
            drift_metrics: A list of divergence measures to compute. Each entry
                accepts a ``DataDriftType`` enum value or a metric name string.

        Returns:
            A list of drift scores in the same order as ``drift_metrics``.
        """
        typed_data = _cast_to_numpy_float_arr(runtime_data)
        return self._inner.compute_drift_mutliple_criteria(
            typed_data, list(map(_map_drift_metric_type, drift_metrics))
        )

    def export_baseline(self) -> list[float]:
        """
        Export the baseline as a normalized probability distribution.

        Returns:
            A list of floats, one per bin, where each value is the fraction of
            baseline samples that fall into that bin. Values sum to 1.0.
        """
        return self._inner.export_baseline()

    @property
    def num_bins(self) -> int:
        """
        The number of histogram bins derived from the baseline dataset.
        """
        return self._inner.num_bins


class CategoricalDataDrift(DataDriftDiscreteBase[str, dict[str, float]]):
    __slots__ = "_inner"

    def __init__(self, baseline_data: Sequence[StringBound]):
        """
        Initialize with a baseline dataset.

        Args:
            baseline_data: The reference distribution. Any iterable whose
                elements implement ``__str__``.
        """
        typed_data = _cast_to_string_iterable(baseline_data)
        self._inner = PyCategoricalDataDrift(typed_data)

    def reset_baseline(self, new_baseline: Sequence[StringBound]):
        """
        Replace the baseline with a new dataset, recomputing the label distribution.

        Args:
            new_baseline: The new reference distribution. Any iterable whose
                elements implement ``__str__``.
        """
        typed_data = _cast_to_string_iterable(new_baseline)
        self._inner.reset_baseline(typed_data)

    def compute_drift(
        self, runtime_data: Sequence[StringBound], drift_metric: DataDriftMetric
    ) -> float:
        """
        Compute a single drift score between ``runtime_data`` and the baseline.

        Args:
            runtime_data: The data collected at runtime. Any iterable whose
                elements implement ``__str__``.
            drift_metric: The divergence measure to use. Accepts a
                ``DataDriftType`` enum value or one of the strings
                ``"JensenShannon"``, ``"PopulationStabilityIndex"``,
                ``"WassersteinDistance"``, ``"KullbackLeibler"``.

        Returns:
            The drift score as a float. Higher values indicate greater divergence
            from the baseline distribution.
        """
        typed_data = _cast_to_string_iterable(runtime_data)
        return self._inner.compute_drift(
            typed_data, _map_drift_metric_type(drift_metric)
        )

    def compute_drift_multiple_criteria(
        self,
        runtime_data: Sequence[StringBound],
        drift_metrics: list[DataDriftMetric],
    ) -> list[float]:
        """
        Compute multiple drift scores against ``runtime_data`` in a single pass.

        Args:
            runtime_data: The data collected at runtime. Any iterable whose
                elements implement ``__str__``.
            drift_metrics: A list of divergence measures to compute. Each entry
                accepts a ``DataDriftType`` enum value or a metric name string.

        Returns:
            A list of drift scores in the same order as ``drift_metrics``.
        """
        typed_data = _cast_to_string_iterable(runtime_data)
        return self._inner.compute_drift_mutliple_criteria(
            typed_data, list(map(_map_drift_metric_type, drift_metrics))
        )

    def export_baseline(self) -> dict[str, float]:
        """
        Export the baseline as a normalized label frequency distribution.

        Returns:
            A dict mapping each label (including the overflow bin) to its
            fraction of the baseline dataset. Values sum to 1.0.
        """
        return self._inner.export_baseline()

    @property
    def num_bins(self) -> int:
        """
        The number of histogram bins derived from the baseline dataset.
        """
        return self._inner.num_bins
