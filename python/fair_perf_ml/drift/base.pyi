from abc import ABC
from collections.abc import Sequence
from enum import Enum
from typing import Protocol

from numpy.typing import NDArray

from .._internal import FloatingPointDataSlice

class QuantileType(str, Enum):
    FreedmanDiaconis: str
    Scott: str
    Sturges: str

class DataDriftType(str, Enum):
    JensenShannon: str
    PopulationStabilityIndex: str
    WassersteinDistance: str
    KullbackLeibler: str

type DataDriftMetric = DataDriftType | str
type QuantileConfig = QuantileType | str | None

class StringBound(Protocol):
    def __str__(self) -> str: ...

class DataDriftParameterValidationError(Exception): ...
def compute_drift_continuous_distribution(
    baseline_distribution: FloatingPointDataSlice,
    candidate_distribution: FloatingPointDataSlice,
    drift_metrics: list[DataDriftMetric],
    quantile_type: QuantileType
) -> list[float]: ...


def compute_drift_categorical_distribution(
    baseline_distribution: list[StringBound],
    candidate_distribution: list[StringBound],
    drift_metrics: list[DataDriftMetric],
) -> list[float]: ...

def _cast_to_numpy_float_arr(arr: FloatingPointDataSlice) -> NDArray: ...
def _map_drift_metric_type(m: DataDriftMetric) -> str: ...
def _cast_to_string_iterable(arr: Sequence[StringBound]) -> Sequence[str]: ...

class DataDriftDiscreteBase[T](ABC):
    def reset_baseline(self, new_baseline: list[T]) -> None: ...
    def compute_drift(self, runtime_data: list[T], drift_metric: DataDriftMetric) -> float: ...
    def compute_drift_multiple_criteria(
        self, runtime_data: list[T], drift_metrics: list[DataDriftMetric]
    ) -> list[float]: ...
    @property
    def total_samples(self) -> int: ...
    @property
    def num_bins(self) -> int: ...
    def export_baseline(self) -> list[float]: ...

class ContinuousDataDrift(DataDriftDiscreteBase[float]):
    def __init__(
        self,
        baseline_data: FloatingPointDataSlice,
        quantile_type: str | None = ...,
    ) -> None: ...
    def reset_baseline(self, new_baseline: FloatingPointDataSlice) -> None: ...
    def compute_drift(
        self, runtime_data: FloatingPointDataSlice, drift_metric: DataDriftMetric
    ) -> float: ...
    def compute_drift_multiple_criteria(
        self,
        runtime_data: FloatingPointDataSlice,
        drift_metrics: list[DataDriftMetric],
    ) -> list[float]: ...
    def export_baseline(self) -> list[float]: ...
    @property
    def num_bins(self) -> int: ...

class CategoricalDataDrift(DataDriftDiscreteBase[str]):
    def __init__(self, baseline_data: Sequence[StringBound]) -> None: ...
    def reset_baseline(self, new_baseline: Sequence[StringBound]) -> None: ...
    def compute_drift(
        self, runtime_data: Sequence[StringBound], drift_metric: DataDriftMetric
    ) -> float: ...
    def compute_drift_multiple_criteria(
        self,
        runtime_data: Sequence[StringBound],
        drift_metrics: list[DataDriftMetric],
    ) -> list[float]: ...
    def export_baseline(self) -> list[float]: ...
    @property
    def num_bins(self) -> int: ...
