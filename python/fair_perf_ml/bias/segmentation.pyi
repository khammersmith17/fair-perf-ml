from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, NamedTuple, Protocol, Self

from numpy.typing import NDArray
import numpy as np


class BiasPayloadTypeException(Exception): ...
class InvalidBiasSegmentationConfig(Exception): ...


class BiasSegmentationType(str, Enum):
    Label: str
    Threshold: str


class BiasSegmentationThresholdType(str, Enum):
    GreaterThan: str
    GreaterThanEqualTo: str
    LessThan: str
    LessThanEqualTo: str


class SegmentationValueBounds(Protocol):
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: Self) -> bool: ...


class DiscreteAnalysisSegmentationValueBounds(SegmentationValueBounds, Protocol):
    def __float__(self) -> float: ...
    def __str__(self) -> str: ...


class BiasSegmentationProtocol[T: SegmentationValueBounds](ABC):
    @abstractmethod
    def _label(self, value: T) -> int: ...
    @abstractmethod
    def _label_batch(self, values: Sequence[T]) -> list[int]: ...
    @abstractmethod
    def _seg_value_type(self) -> type: ...
    @property
    @abstractmethod
    def _seg_value(self) -> T: ...
    @property
    @abstractmethod
    def _seg_type(self) -> BiasSegmentationType: ...
    @property
    @abstractmethod
    def _thres_type(self) -> BiasSegmentationThresholdType | None: ...


class LabeledBiasSegmentation[T: SegmentationValueBounds](BiasSegmentationProtocol[T]):
    def __init__(self, value: T) -> None: ...
    def _seg_value_type(self) -> type: ...
    def _label(self, value: T) -> int: ...
    def _label_batch(self, values: Sequence[T]) -> list[int]: ...
    @property
    def _seg_value(self) -> T: ...
    @property
    def _seg_type(self) -> BiasSegmentationType: ...
    @property
    def _thres_type(self) -> None: ...


class ThresholdBiasSegmentation[T: SegmentationValueBounds](BiasSegmentationProtocol[T]):
    def __init__(self, value: T, threshold_type: BiasSegmentationThresholdType) -> None: ...
    def _seg_value_type(self) -> type: ...
    def _label(self, value: T) -> int: ...
    def _label_batch(self, values: Sequence[T]) -> list[int]: ...
    @property
    def _seg_value(self) -> T: ...
    @property
    def _seg_type(self) -> BiasSegmentationType: ...
    @property
    def _thres_type(self) -> BiasSegmentationThresholdType: ...


def _validate_factory_args(
    segmentation_type: BiasSegmentationType,
    segmentation_threshold_type: BiasSegmentationThresholdType | None,
) -> None: ...


def bias_segmentation_criteria_factory(
    value: Any,
    segmentation_type: BiasSegmentationType | str,
    segmentation_threshold_type: BiasSegmentationThresholdType | str | None = ...,
) -> BiasSegmentationProtocol: ...


class BiasDataPayload[T: SegmentationValueBounds]:
    def __init__(
        self,
        data_payload: Sequence[T],
        segmentation_criteria: BiasSegmentationProtocol[T],
    ) -> None: ...
    @classmethod
    def factory(
        cls,
        data_payload: Sequence[T],
        value: T,
        segmentation_type: BiasSegmentationType | str,
        segmentation_threshold_type: BiasSegmentationThresholdType | str | None = ...,
    ) -> Self: ...


class _ExplicitAnalysisArgs(NamedTuple):
    data: NDArray[np.float64] | NDArray[np.str_]
    threshold: float | None
    label: str | None
    threshold_type: BiasSegmentationThresholdType | None


def _construct_explicit_bias_args[
    T: DiscreteAnalysisSegmentationValueBounds
](payload: BiasDataPayload[T]) -> _ExplicitAnalysisArgs: ...

def _cast_sequence_and_label_thres[
    T: DiscreteAnalysisSegmentationValueBounds
](payload: BiasDataPayload[T]) -> _ExplicitAnalysisArgs: ...

def _cast_sequence_and_label_label[
    T: DiscreteAnalysisSegmentationValueBounds
](payload: BiasDataPayload[T]) -> _ExplicitAnalysisArgs: ...
