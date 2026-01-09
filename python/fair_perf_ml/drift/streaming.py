from abc import ABC, abstractmethod
from typing import Iterable, Union, Optional
from numpy.typing import NDArray
import numpy as np
from .._fair_perf_ml import (
    PyStreamingContinuousDataDrift,
    PyStreamingCategoricalDataDrift,
)


class DataDriftStreamingBase(ABC):
    @abstractmethod
    def reset_baseline(self): ...

    @abstractmethod
    def update_stream(self): ...

    @abstractmethod
    def update_stream_batch(self): ...

    @abstractmethod
    def compute_psi_drift(self): ...

    @abstractmethod
    def compute_kl_divergence_drift(self): ...

    @abstractmethod
    def flush(self): ...

    @abstractmethod
    def total_samples(self): ...

    @abstractmethod
    def last_flush(self): ...

    @abstractmethod
    def n_bins(self): ...

    @abstractmethod
    def export_snapshot(self): ...

    @abstractmethod
    def export_baseline(self): ...


class StreamingContinuousDataDrift(DataDriftStreamingBase):
    __slots__ = ["_inner"]

    def __init__(
        self,
        baseline_dataset: Union[Iterable[float], NDArray],
        n_bins: int,
        flush_cadence: Optional[int],
    ):
        if not isinstance(baseline_dataset, np.ndarray):
            baseline_dataset = np.array(baseline_dataset)
        self._inner = PyStreamingContinuousDataDrift(
            n_bins, baseline_dataset, flush_cadence
        )

    def reset_baseline(self):
        raise NotImplementedError

    def update_stream(self):
        raise NotImplementedError

    def update_stream_batch(self):
        raise NotImplementedError

    def compute_psi_drift(self):
        raise NotImplementedError

    def compute_kl_divergence_drift(self):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def total_samples(self):
        raise NotImplementedError

    def last_flush(self):
        raise NotImplementedError

    def n_bins(self):
        raise NotImplementedError

    def export_snapshot(self):
        raise NotImplementedError

    def export_baseline(self):
        raise NotImplementedError
