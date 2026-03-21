"""
Provides runtime drift monitoring utilites and managers for such utilities.
These drift techniques can be used for data drift or a proxy for model drift
when ground truth feedback loop is slow.
"""

from __future__ import annotations
from enum import Enum
from typing import Union, Iterable, Optional, Protocol, Dict, List
import numpy as np
from numpy.typing import NDArray
from .._fair_perf_ml import (
    PyContinuousDataDrift,
    PyCategoricalDataDrift,
)


class QuantileType(str, Enum):
    FreedmanDiaconis = "FreedmanDiaconis"
    Scott = "Scott "
    Sturges = "Sturges"


class DataDriftType(str, Enum):
    JensenShannon = "JensenShannon"
    PopulationStabilityIndex = "PopulationStabilityIndex"
    WassersteinDistance = "WassersteinDistance"
    KullbackLeibler = "KullbackLeibler"


type DataDriftMetric = DataDriftType | str
type QuantileConfig = QuantileType | str | None


def _map_drift_metric_type(m: DataDriftMetric) -> str:
    return str(m)


FloatingPointDataSlice = Union[Iterable[float], NDArray]


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
    assert isinstance(arr, np.ndarray)
    return arr


def _cast_to_string_iterable(arr: Iterable[StringBound]) -> Iterable[str]:
    """
    Iterable of something that can be casted to a string to a Iterable[str].
    """
    return list(map(lambda x: str(x), arr))


class ContinuousDataDrift:
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

    __slots__ = ["_inner"]

    def __init__(
        self,
        baseline_data: FloatingPointDataSlice,
        quantile_type: Optional[str] = None,
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
            quantile_type = str(quantile_type)
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
        drift_metrics: List[DataDriftMetric],
    ) -> List[float]:
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

    def export_baseline(self) -> List[float]:
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


class CategoricalDataDrift:
    """
    Detects distributional drift in categorical features between a fixed baseline
    dataset and a runtime dataset.

    The baseline is summarized as a label frequency distribution. Each unique
    value observed in the baseline becomes its own bin. An additional overflow
    bin captures any labels present in the runtime data that were not seen in
    the baseline. Drift is measured by comparing the runtime label distribution
    against the baseline using the chosen divergence metric.

    This type is suited for batch analysis: you collect a runtime dataset and
    compare it against the baseline in one call. For long-running accumulation
    where data arrives incrementally, use the streaming variants instead.

    Considerations:
        1. There will be n + 1 bins, where n is the number of unique labels in
           the baseline. The extra bin collects any unseen runtime labels.
        2. Resetting the baseline recomputes the distribution and bin set from
           the new data.
        3. The label used for the overflow bin can be configured via the
           ``FAIR_PERF_OTHER_BUCKET`` environment variable.
        4. Items passed in must be castable to ``str`` via ``__str__``. All
           bucketing is performed on string representations.
    """

    __slots__ = ["_inner"]

    def __init__(self, baseline_data: Iterable[StringBound]):
        """
        Initialize with a baseline dataset.

        Args:
            baseline_data: The reference distribution. Any iterable whose
                elements implement ``__str__``.
        """
        typed_data = _cast_to_string_iterable(baseline_data)
        self._inner = PyCategoricalDataDrift(typed_data)

    def reset_baseline(self, new_baseline: Iterable[StringBound]):
        """
        Replace the baseline with a new dataset, recomputing the label distribution.

        Args:
            new_baseline: The new reference distribution. Any iterable whose
                elements implement ``__str__``.
        """
        typed_data = _cast_to_string_iterable(new_baseline)
        self._inner.reset_baseline(typed_data)

    def compute_drift(
        self, runtime_data: Iterable[StringBound], drift_metric: DataDriftMetric
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
        runtime_data: Iterable[StringBound],
        drift_metrics: List[DataDriftMetric],
    ) -> List[float]:
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

    def export_baseline(self) -> Dict[str, float]:
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
