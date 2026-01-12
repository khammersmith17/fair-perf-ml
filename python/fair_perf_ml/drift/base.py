"""
Provides runtime drift monitoring utilites and managers for such utilities.
These drift techniques can be used for data drift or a proxy for model drift
when ground truth feedback loop is slow.
"""

from __future__ import annotations
from typing import Union, Iterable, Protocol, Dict, List
import numpy as np
from numpy.typing import NDArray
from .._fair_perf_ml import (
    PyContinuousDataDrift,
    PyCategoricalDataDrift,
)

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
    __slots__ = ["_inner"]

    def __init__(self, baseline_data: FloatingPointDataSlice, num_bins: int):
        typed_data = _cast_to_numpy_float_arr(baseline_data)
        self._inner = PyContinuousDataDrift(num_bins, typed_data)

    def reset_baseline(self, new_baseline: FloatingPointDataSlice):
        self._inner.reset_baseline(new_baseline)

    def compute_psi_drift(self, runtime_data: FloatingPointDataSlice) -> float:
        typed_data = _cast_to_numpy_float_arr(runtime_data)
        return self._inner.compute_psi_drift(typed_data)

    def compute_kl_divergence_drift(
        self, runtime_data: FloatingPointDataSlice
    ) -> float:
        typed_data = _cast_to_numpy_float_arr(runtime_data)
        return self._inner.compute_kl_divergence_drift(typed_data)

    def export_baseline(self) -> List[float]:
        return self._inner.export_baseline()

    @property
    def num_bins(self) -> int:
        return self._inner.num_bins()


class CategoricalDataDrift:
    __slots__ = ["_inner"]

    def __init__(self, baseline_data: Iterable[StringBound]):
        typed_data = _cast_to_string_iterable(baseline_data)
        self._inner = PyCategoricalDataDrift(typed_data)

    def reset_baseline(self, new_baseline: Iterable[StringBound]):
        self._inner.reset_baseline(new_baseline)

    def compute_psi_drift(self, runtime_data: Iterable[StringBound]) -> float:
        typed_data = _cast_to_string_iterable(runtime_data)
        return self._inner.compute_psi_drift(typed_data)

    def compute_kl_divergence_drift(self, runtime_data: Iterable[StringBound]) -> float:
        typed_data = _cast_to_string_iterable(runtime_data)
        return self._inner.compute_kl_divergence_drift(typed_data)

    @property
    def num_bins(self) -> int:
        return self._inner.num_bins()

    def export_baseline(self) -> Dict[str, float]:
        return self._inner.export_baseline()

    @property
    def other_bucket_label(self) -> str:
        return self._inner.other_bucket_label
