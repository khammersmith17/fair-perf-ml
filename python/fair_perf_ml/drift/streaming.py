from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from .._fair_perf_ml import (PyStreamingCategoricalDataDriftDecay,
                             PyStreamingCategoricalDataDriftFlush,
                             PyStreamingContinuousDataDriftDecay,
                             PyStreamingContinuousDataDriftFlush)
from .base import (DataDriftMetric, FloatingPointDataSlice, QuantileConfig,
                   QuantileType, StringBound, _cast_to_numpy_float_arr,
                   _cast_to_string_iterable, _map_drift_metric_type)


class DataDriftStreamingBase[T, R](ABC):
    """
    Abtract class to define the streaming data drift api contract.
    More for correctness constraint rather than utility here.
    """

    @abstractmethod
    def reset_baseline(self, new_baseline: list[T]): ...

    @abstractmethod
    def update_stream(self, example: T): ...

    @abstractmethod
    def update_stream_batch(self, runtime_data: list[T]): ...

    @abstractmethod
    def compute_drift(self, drift_metric: DataDriftMetric) -> float: ...

    @abstractmethod
    def compute_drift_multiple_criteria(
        self, drift_metrics: List[DataDriftMetric]
    ) -> List[float]: ...

    @property
    @abstractmethod
    def total_samples(self) -> int: ...

    @property
    @abstractmethod
    def num_bins(self) -> int: ...

    @abstractmethod
    def export_snapshot(self) -> R: ...

    @abstractmethod
    def export_baseline(self) -> dict: ...


class StreamingContinuousDataDriftFlush(DataDriftStreamingBase):
    __slots__ = ["_inner"]

    def __init__(
        self,
        baseline_dataset: FloatingPointDataSlice,
        quantile_type: QuantileConfig,
        flush_rate: Optional[int],
        flush_cadence: Optional[int],
    ):
        baseline_dataset = _cast_to_numpy_float_arr(baseline_dataset)
        if isinstance(quantile_type, QuantileType):
            quantile_type = str(quantile_type)
        self._inner = PyStreamingContinuousDataDriftFlush(
            baseline_dataset, quantile_type, flush_rate, flush_cadence
        )

    def reset_baseline(self, new_baseline: FloatingPointDataSlice):
        new_baseline = _cast_to_numpy_float_arr(new_baseline)
        self._inner.reset_baseline(new_baseline)

    def update_stream(self, example: float):
        self._inner.update_stream(example)

    def update_stream_batch(self, runtime_data: FloatingPointDataSlice):
        runtime_data = _cast_to_numpy_float_arr(runtime_data)
        self._inner.update_stream_batch(runtime_data)

    def compute_drift(self, drift_metric: DataDriftMetric) -> float:
        return self._inner.compute_drift(_map_drift_metric_type(drift_metric))

    def compute_drift_multiple_criteria(
        self, drift_metrics: List[DataDriftMetric]
    ) -> List[float]:
        return self._inner.compute_drift_multiple_criteria(
            list(map(_map_drift_metric_type, drift_metrics))
        )

    @property
    def total_samples(self) -> int:
        return self._inner.total_samples

    @property
    def num_bins(self) -> int:
        return self._inner.n_bins

    def export_snapshot(self) -> dict[str, list[float]]:
        return self._inner.export_snapshot()

    def export_baseline(self) -> dict:
        return self._inner.export_baseline()

    def flush(self):
        self._inner.flush()

    def last_flush(self) -> int:
        return self._inner.last_flush


class StreamingContinuousDataDriftDecay(DataDriftStreamingBase):
    __slots__ = ["_inner"]

    def __init__(
        self,
        baseline_dataset: FloatingPointDataSlice,
        quantile_type: QuantileConfig,
        decay_half_life: Optional[int],
    ):
        baseline_dataset = _cast_to_numpy_float_arr(baseline_dataset)
        if isinstance(quantile_type, QuantileType):
            quantile_type = str(quantile_type)
        self._inner = PyStreamingContinuousDataDriftDecay(
            baseline_dataset, quantile_type, decay_half_life
        )

    def reset_baseline(self, new_baseline: FloatingPointDataSlice):
        new_baseline = _cast_to_numpy_float_arr(new_baseline)
        self._inner.reset_baseline(new_baseline)

    def update_stream(self, example: float):
        self._inner.update_stream(example)

    def update_stream_batch(self, runtime_data: FloatingPointDataSlice):
        runtime_data = _cast_to_numpy_float_arr(runtime_data)
        self._inner.update_stream_batch(runtime_data)

    def compute_drift(self, drift_metric: DataDriftMetric) -> float:
        return self._inner.compute_drift(_map_drift_metric_type(drift_metric))

    def compute_drift_multiple_criteria(
        self, drift_metrics: List[DataDriftMetric]
    ) -> List[float]:
        return self._inner.compute_drift_multiple_criteria(
            list(map(_map_drift_metric_type, drift_metrics))
        )

    @property
    def total_samples(self) -> int:
        return self._inner.total_samples

    @property
    def num_bins(self) -> int:
        return self._inner.n_bins

    def export_snapshot(self) -> dict[str, list[float]]:
        return self._inner.export_snapshot()

    def export_baseline(self) -> dict:
        return self._inner.export_baseline()


class StreamingCategoricalDataDriftFlush(DataDriftStreamingBase):
    __slots__ = ["_inner"]

    def __init__(
        self,
        baseline_dataset: Iterable[StringBound],
        flush_rate: Optional[int],
        flush_cadence: Optional[int],
    ):
        typed_data = _cast_to_string_iterable(baseline_dataset)
        self._inner = PyStreamingCategoricalDataDriftFlush(
            typed_data, flush_rate, flush_cadence
        )

    def reset_baseline(self, new_baseline: Iterable[StringBound]):
        typed_data = _cast_to_string_iterable(new_baseline)
        self._inner.reset_baseline(typed_data)

    def update_stream(self, example: StringBound):
        self._inner.update_stream(str(example))

    def update_stream_batch(self, runtime_data: Iterable[StringBound]):
        typed_data = _cast_to_string_iterable(runtime_data)
        self._inner.update_stream_batch(typed_data)

    def compute_drift(self, drift_metric: DataDriftMetric) -> float:
        return self._inner.compute_drift(_map_drift_metric_type(drift_metric))

    def compute_drift_multiple_criteria(
        self, drift_metrics: List[DataDriftMetric]
    ) -> List[float]:
        return self._inner.compute_drift_multiple_criteria(
            list(map(_map_drift_metric_type, drift_metrics))
        )

    @property
    def total_samples(self) -> int:
        return self._inner.total_samples

    @property
    def n_bins(self) -> int:
        return self._inner.n_bins

    def export_snapshot(self) -> dict[str, float]:
        return self._inner.export_snapshot()

    def export_baseline(self) -> dict:
        return self._inner.export_baseline()

    def flush(self):
        self._inner.flush()

    def last_flush(self) -> int:
        return self._inner.last_flush


class StreamingCategoricalDataDriftDecay(DataDriftStreamingBase):
    __slots__ = ["_inner"]

    def __init__(self, baseline_dataset: Iterable[StringBound], decay_half_life: int):
        typed_data = _cast_to_string_iterable(baseline_dataset)
        self._inner = PyStreamingCategoricalDataDriftDecay(typed_data, decay_half_life)

    def reset_baseline(self, new_baseline: Iterable[StringBound]):
        typed_data = _cast_to_string_iterable(new_baseline)
        self._inner.reset_baseline(typed_data)

    def update_stream(self, example: StringBound):
        self._inner.update_stream(str(example))

    def update_stream_batch(self, runtime_data: Iterable[StringBound]):
        typed_data = _cast_to_string_iterable(runtime_data)
        self._inner.update_stream_batch(typed_data)

    def compute_drift(self, drift_metric: DataDriftMetric) -> float:
        return self._inner.compute_drift(_map_drift_metric_type(drift_metric))

    def compute_drift_multiple_criteria(
        self, drift_metrics: List[DataDriftMetric]
    ) -> List[float]:
        return self._inner.compute_drift_multiple_criteria(
            list(map(_map_drift_metric_type, drift_metrics))
        )

    @property
    def total_samples(self) -> int:
        return self._inner.total_samples

    @property
    def n_bins(self) -> int:
        return self._inner.n_bins

    def export_snapshot(self) -> dict[str, float]:
        return self._inner.export_snapshot()

    def export_baseline(self) -> dict:
        return self._inner.export_baseline()
