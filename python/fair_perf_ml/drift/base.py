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
    """
    Python wrapper type to expose the core functionality written in Rust.
    All apis here expose directly to the core methods provided by the
    Rust type, with some additional typing utilities. Internally, this type
    constructs a histogram and uses quantiles to construct a distribution.
    This type is meant to service continuous data.

    Considerations:
        1. The requested number of bins will determine the target number
            of bins allocated. This is a best effort attempt, if the
            dataset does not fulfill the number of bins, then less bins
            will be used. Look at the source to see how this is determined.
        2. When a baseline reset occurs, the same number of bins will be used,
            in the same manner as the initial bin construction.
        3. This type is better suited for the use for discrete datasets,
            for example when runtime data is collected in batches. For long
            running accumulation style support, use the streaming utilities.
    """

    __slots__ = ["_inner"]

    def __init__(self, baseline_data: FloatingPointDataSlice, num_bins: int):
        """
        args:
            baseline_data: Union[Iterable[float], NDArray]
        """
        typed_data = _cast_to_numpy_float_arr(baseline_data)
        self._inner = PyContinuousDataDrift(num_bins, typed_data)

    def reset_baseline(self, new_baseline: FloatingPointDataSlice):
        """
        Reset the internal baseline state with a new dataset.
        """
        self._inner.reset_baseline(new_baseline)

    def compute_psi_drift(self, runtime_data: FloatingPointDataSlice) -> float:
        """
        Compute the Population Stability Index drift between the runtime dataset
        and the baseline dataset.
        """
        typed_data = _cast_to_numpy_float_arr(runtime_data)
        return self._inner.compute_psi_drift(typed_data)

    def compute_kl_divergence_drift(
        self, runtime_data: FloatingPointDataSlice
    ) -> float:
        """
        Compute the Kullback-Leibler Divergence (KL) drift between the runtime dataset
        and the baseline dataset.
        """
        typed_data = _cast_to_numpy_float_arr(runtime_data)
        return self._inner.compute_kl_divergence_drift(typed_data)

    def compute_js_divergence_drift(
        self, runtime_data: FloatingPointDataSlice
    ) -> float:
        """
        Compute the Jensen-Shannon Divergence (JS) between the runtime dataset
        and the baseline dataset.
        """
        typed_data = _cast_to_numpy_float_arr(runtime_data)
        return self._inner.compute_js_divergence_drift(typed_data)

    def compute_wasserstein_distance_drift(
        self, runtime_data: FloatingPointDataSlice
    ) -> float:
        typed_data = _cast_to_numpy_float_arr(runtime_data)
        return self._inner.compute_wasserstein_distance_drift(typed_data)

    def export_baseline(self) -> List[float]:
        """
        Export the baseline bins, this will return ratio of the size of the
        histogram bar for a particular bin with respect the total number of
        examples in the baseline dataset.
        """
        return self._inner.export_baseline()

    @property
    def num_bins(self) -> int:
        """
        The number of bins.
        """
        return self._inner.num_bins()


class CategoricalDataDrift:
    """
    Python wrapper type to expose the core functionality written in Rust.
    All apis here expose directly to the core methods provided by the
    Rust type, with some additional typing utilities. Internally, this type
    constructs a histogram and uses quantiles to construct a distribution.
    This type is meant to service categorical data.
    Considerations:
        1. There will be n + 1 bins where n is the number of unique values
            observed in the baseline set, with an additional bin that will
            catch values observed in the runtime data, that are not in the
            baseline set.
        2. On a baseline reset, the number of bins will be reset to
            reflect the baseline dataset.
        3. The other bucket value can be configured using the
            "FAIR_PERF_OTHER_BUCKET" label.
        4. The type of items passed into this type need to be safely
            castable to a string. Bucket is done on strings in the core
            Rust implementation.
        5. This type is meant for analysis on discrete datasets.
    """

    __slots__ = ["_inner"]

    def __init__(self, baseline_data: Iterable[StringBound]):
        """
        args:
            baseline_data: Iterable[StringBound]
        """
        typed_data = _cast_to_string_iterable(baseline_data)
        self._inner = PyCategoricalDataDrift(typed_data)

    def reset_baseline(self, new_baseline: Iterable[StringBound]):
        """
        Reset the baseline using a new baseline dataset.
        args:
            baseline_data: Iterable[StringBound]
        """
        self._inner.reset_baseline(new_baseline)

    def compute_psi_drift(self, runtime_data: Iterable[StringBound]) -> float:
        """
        Compute the Population Stability Index drift between the runtime dataset
        and the baseline dataset.
        """
        typed_data = _cast_to_string_iterable(runtime_data)
        return self._inner.compute_psi_drift(typed_data)

    def compute_kl_divergence_drift(self, runtime_data: Iterable[StringBound]) -> float:
        """
        Compute the Kullback-Leibler Divergence (KL) drift between the runtime dataset
        and the baseline dataset.
        """
        typed_data = _cast_to_string_iterable(runtime_data)
        return self._inner.compute_kl_divergence_drift(typed_data)

    def compute_js_divergence_drift(self, runtime_data: Iterable[StringBound]) -> float:
        """
        Compute the Jensen-Shannon Divergence (JS) between the runtime dataset
        and the baseline dataset.
        """
        typed_data = _cast_to_string_iterable(runtime_data)
        return self._inner.compute_js_divergence_drift(typed_data)

    def compute_wasserstein_distance_drift(
        self, runtime_data: Iterable[StringBound]
    ) -> float:
        typed_data = _cast_to_string_iterable(runtime_data)
        return self._inner.compute_wasserstein_distance_drift(typed_data)

    @property
    def num_bins(self) -> int:
        """
        Returns the number of bins computed from the baseline dataset.
        """
        return self._inner.num_bins()

    def export_baseline(self) -> Dict[str, float]:
        """
        Export the baseline histogram. This will be the bucket label and a ratio of
        bin size to total items.
        """
        return self._inner.export_baseline()

    @property
    def other_bucket_label(self) -> str:
        """
        The label of the overflow bucket.
        """
        return self._inner.other_bucket_label
