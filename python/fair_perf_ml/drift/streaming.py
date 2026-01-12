from __future__ import annotations
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Iterable, Optional
from .._fair_perf_ml import (
    PyStreamingContinuousDataDrift,
    PyStreamingCategoricalDataDrift,
)
from .base import (
    StringBound,
    FloatingPointDataSlice,
    _cast_to_numpy_float_arr,
    _cast_to_string_iterable,
)


class DataDriftStreamingBase[T, R](ABC):
    """
    Abtract class to define the streaming data drift api contract.
    """

    @abstractmethod
    def reset_baseline(self, new_baseline: T): ...

    @abstractmethod
    def update_stream(self, example: T): ...

    @abstractmethod
    def update_stream_batch(self, runtime_data: T): ...

    @abstractmethod
    def compute_psi_drift(self) -> float: ...

    @abstractmethod
    def compute_kl_divergence_drift(self) -> float: ...

    @abstractmethod
    def compute_js_divergence_drift(self) -> float: ...

    @abstractmethod
    def flush(self): ...

    @abstractmethod
    def total_samples(self) -> int: ...

    @abstractmethod
    def last_flush(self) -> datetime: ...

    @abstractmethod
    def n_bins(self) -> int: ...

    @abstractmethod
    def export_snapshot(self) -> R: ...

    @abstractmethod
    def export_baseline(self) -> dict: ...


class StreamingContinuousDataDrift(DataDriftStreamingBase):
    __slots__ = ["_inner"]

    def __init__(
        self,
        baseline_dataset: FloatingPointDataSlice,
        n_bins: int,
        flush_cadence: Optional[int],
    ):
        baseline_dataset = _cast_to_numpy_float_arr(baseline_dataset)
        self._inner = PyStreamingContinuousDataDrift(
            n_bins, baseline_dataset, flush_cadence
        )

    def reset_baseline(self, new_baseline: FloatingPointDataSlice):
        new_baseline = _cast_to_numpy_float_arr(new_baseline)
        self._inner.reset_baseline(new_baseline)

    def update_stream(self, example: float):
        self._inner.update_stream(example)

    def update_stream_batch(self, runtime_data: FloatingPointDataSlice):
        runtime_data = _cast_to_numpy_float_arr(runtime_data)
        self._inner.update_stream_batch(runtime_data)

    def compute_psi_drift(self) -> float:
        return self._inner.compute_psi_drift()

    def compute_kl_divergence_drift(self) -> float:
        return self._inner.compute_kl_divergence_drift()

    def compute_js_divergence_drift(self) -> float:
        return self._inner.compute_js_divergence_drift()

    def flush(self):
        self._inner.flush()

    def total_samples(self) -> int:
        return self._inner.total_samples()

    def last_flush(self) -> datetime:
        ts: datetime = self._inner.last_flush()
        return ts

    def n_bins(self) -> int:
        return self._inner.n_bins()

    def export_snapshot(self) -> list[float]:
        return self._inner.export_snapshot()

    def export_baseline(self) -> dict:
        return self._inner.export_baseline()


class StreamingCategoricalDataDrift(DataDriftStreamingBase):
    __slots__ = ["_inner"]

    def __init__(
        self,
        baseline_dataset: Iterable[StringBound],
        flush_cadence: Optional[int],
    ):
        typed_data = _cast_to_string_iterable(baseline_dataset)
        self._inner = PyStreamingCategoricalDataDrift(typed_data, flush_cadence)

    def reset_baseline(self, new_baseline: Iterable[StringBound]):
        typed_data = _cast_to_string_iterable(new_baseline)
        self._inner.reset_baseline(typed_data)

    def update_stream(self, example: StringBound):
        self._inner.update_stream(str(example))

    def update_stream_batch(self, runtime_data: Iterable[StringBound]):
        typed_data = _cast_to_string_iterable(runtime_data)
        self._inner.update_stream_batch(typed_data)

    def compute_psi_drift(self) -> float:
        return self._inner.compute_psi_drift()

    def compute_kl_divergence_drift(self) -> float:
        return self._inner.compute_kl_divergence_drift()

    def compute_js_divergence_drift(self) -> float:
        return self._inner.compute_js_divergence_drift()

    def flush(self):
        self._inner.flush()

    def total_samples(self) -> int:
        return self._inner.total_samples()

    def last_flush(self) -> datetime:
        ts: datetime = self._inner.last_flush()
        return ts

    def n_bins(self) -> int:
        return self._inner.n_bins()

    def export_snapshot(self) -> list[float]:
        return self._inner.export_snapshot()

    def export_baseline(self) -> dict:
        return self._inner.export_baseline()
