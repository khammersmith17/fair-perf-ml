from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, NamedTuple, Protocol, Self, cast

import numpy as np
from fair_perf_ml._internal import _extract_sequence_type
from numpy.typing import NDArray


class BiasPayloadTypeException(Exception):
    _default_message = "Segmentation value and sequence data type must be the same"

    def __init__(self, msg: str | None = None, *args):
        if msg is None:
            msg = self._default_message
        super().__init__(msg, *args)


class InvalidBiasSegmentationConfig(Exception):
    _default_message = (
        "Valid configurations:\n"
        "segmentation_type: BiasSegmentationType.Label and segmentation_threshold_type: None\n"
        "segmentation_type: BiasSegmentationType.Threshold and "
        "segmentation_threshold_type: BiasSegmentationThresholdType"
    )

    def __init__(self, msg: str | None = None, *args):
        if msg is None:
            msg = self._default_message
        super().__init__(msg, *args)


class BiasSegmentationType(str, Enum):
    Label = "Label"
    Threshold = "Threshold"


class BiasSegmentationThresholdType(str, Enum):
    GreaterThan = "GreaterThan"
    GreaterThanEqualTo = "GreaterThanEqualTo"
    LessThan = "LessThan"
    LessThanEqualTo = "LessThanEqualTo"


class SegmentationValueBounds(Protocol):
    """
    Protocol to enforces typing. The type used for segmentation should
    safely implement __eq__ and __ge__.
    """

    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, __other: Self) -> bool: ...


class DiscreteAnalysisSegmentationValueBounds(SegmentationValueBounds, Protocol):
    """
    Protocol to enforces typing. The type used for segmentation should
    safely implement __eq__ and __ge__.
    """

    def __float__(self) -> float: ...
    def __str__(self) -> str: ...


class BiasSegmentationProtocol[T: SegmentationValueBounds](ABC):
    """
    Interface to define the implementation contract for Segmentation implementations.
    """

    @abstractmethod
    def _label(self, value: T) -> int:
        """
        Label a single example.
        """

    @abstractmethod
    def _label_batch(self, values: Sequence[T]) -> list[int]:
        """
        Label a batch of examples.
        """

    @abstractmethod
    def _seg_value_type(self) -> type:
        """
        Resolve the type of the segmentation value.
        """

    @property
    @abstractmethod
    def _seg_value(self) -> T:
        """
        Property to expose the segmentation value.
        """

    @property
    @abstractmethod
    def _seg_type(self) -> BiasSegmentationType:
        """
        Property to expose the segmentation value.
        """

    @property
    @abstractmethod
    def _thres_type(self) -> BiasSegmentationThresholdType | None:
        """
        Property to expose the segmentation value.
        """


class LabeledBiasSegmentation[T: SegmentationValueBounds](BiasSegmentationProtocol):
    """
    Implements the BiasSegmentationProtocol for assigning segmentation values for
    equality types.
    """

    __slots__ = "_value"

    def __init__(
        self,
        value: T,
    ):
        self._value = value

    def _seg_value_type(self) -> type:
        return type(self._value)

    def _label(self, value: T) -> int:
        return int(value == self._value)

    def _label_batch(self, values: Sequence[T]) -> list[int]:
        arr = np.array(values)
        return (arr == self._value).astype(np.int8).tolist()

    @property
    def _seg_value(self) -> T:
        return self._value

    @property
    def _seg_type(self) -> BiasSegmentationType:
        return BiasSegmentationType.Label

    @property
    def _thres_type(self) -> BiasSegmentationThresholdType | None:
        return None


class ThresholdBiasSegmentation[T: SegmentationValueBounds](BiasSegmentationProtocol):
    """
    Implements the BiasSegmentationProtocol for assigning segmentation values for
    ordered types.
    """

    __slots__ = ("_value", "_threshold_type")

    def __init__(self, value: T, threshold_type: BiasSegmentationThresholdType | str):
        self._value = value
        self._threshold_type: BiasSegmentationThresholdType = (
            threshold_type
            if isinstance(threshold_type, BiasSegmentationThresholdType)
            else BiasSegmentationThresholdType(threshold_type)
        )

    def _seg_value_type(self) -> type:
        return type(self._value)

    def _label(self, value: T) -> int:
        match self._threshold_type:
            case BiasSegmentationThresholdType.GreaterThanEqualTo:
                return int(value >= self._value)
            case BiasSegmentationThresholdType.GreaterThan:
                return int(value >= self._value and value != self._value)
            case BiasSegmentationThresholdType.LessThanEqualTo:
                return int((not value >= self._value) or value == self._value)
            case BiasSegmentationThresholdType.LessThan:
                return int(not value >= self._value)
            case _:
                raise RuntimeError("State is invalid, please file an issue")

    def _label_batch(self, values: Sequence[T]) -> list[int]:
        arr = np.array(values)
        match self._threshold_type:
            case BiasSegmentationThresholdType.GreaterThanEqualTo:
                return (arr >= self._value).astype(np.int8).tolist()
            case BiasSegmentationThresholdType.GreaterThan:
                return (
                    ((arr >= self._value) & (arr != self._value))
                    .astype(np.int8)
                    .tolist()
                )
            case BiasSegmentationThresholdType.LessThanEqualTo:
                return (
                    ((~(arr >= self._value)) | (arr == self._value))
                    .astype(np.int8)
                    .tolist()
                )
            case BiasSegmentationThresholdType.LessThan:
                return (~(arr >= self._value)).astype(np.int8).tolist()
            case _:
                raise RuntimeError("State is invalid, please file an issue")

    @property
    def _seg_value(self) -> T:
        return self._value

    @property
    def _seg_type(self) -> BiasSegmentationType:
        return BiasSegmentationType.Threshold

    @property
    def _thres_type(self) -> BiasSegmentationThresholdType | None:
        return self._threshold_type


def _validate_factory_args(
    segmentation_type: BiasSegmentationType,
    segmentation_threshold_type: BiasSegmentationThresholdType | None,
):
    if (
        segmentation_type == BiasSegmentationType.Label
        and segmentation_threshold_type is not None
    ) or (
        segmentation_type == BiasSegmentationType.Threshold
        and segmentation_threshold_type is None
    ):
        raise InvalidBiasSegmentationConfig


def bias_segmentation_criteria_factory(
    value: Any,
    segmentation_type: BiasSegmentationType | str,
    segmentation_threshold_type: BiasSegmentationThresholdType | str | None = None,
) -> BiasSegmentationProtocol:
    if isinstance(segmentation_type, str):
        segmentation_type = BiasSegmentationType(segmentation_type)

    if segmentation_threshold_type is not None and isinstance(
        segmentation_threshold_type, str
    ):
        segmentation_threshold_type = BiasSegmentationThresholdType(
            segmentation_threshold_type
        )

    _validate_factory_args(segmentation_type, segmentation_threshold_type)

    match segmentation_type:
        case BiasSegmentationType.Label:
            return LabeledBiasSegmentation(value)
        case BiasSegmentationType.Threshold:
            segmentation_threshold_type = cast(
                BiasSegmentationThresholdType, segmentation_threshold_type
            )
            return ThresholdBiasSegmentation(value, segmentation_threshold_type)
        case _:
            raise ValueError(
                "Invalid value passed for segmentation_type. "
                "Valid arguments: BiasSegmentationType | str Label or Threshold"
            )


class BiasDataPayload[T: SegmentationValueBounds]:
    """
    Groups a dataset and the segmentation criteria as a single unit.
    """

    __slots__ = ("_seg_criteria", "_payload")

    def __init__(
        self,
        data_payload: Sequence[T],
        segmentation_criteria: BiasSegmentationProtocol[T],
    ):
        t = _extract_sequence_type(data_payload)
        if segmentation_criteria._seg_value_type() != t:
            raise BiasPayloadTypeException
        self._payload = data_payload
        self._seg_criteria = segmentation_criteria

    @classmethod
    def factory(
        cls,
        data_payload: Sequence[T],
        value: T,
        segmentation_type: BiasSegmentationType | str,
        segmentation_threshold_type: BiasSegmentationThresholdType | str | None = None,
    ) -> Self:
        """
        Factory method to derive the segmentation criteria.
        """
        if type(value) != _extract_sequence_type(data_payload):
            raise BiasPayloadTypeException
        segmentation_criteria = bias_segmentation_criteria_factory(
            value, segmentation_type, segmentation_threshold_type
        )
        return cls(data_payload, segmentation_criteria)


class _ExplicitAnalysisArgs(NamedTuple):
    data: NDArray[np.float64] | NDArray[np.str_]
    threshold: float | None
    label: str | None
    threshold_type: BiasSegmentationThresholdType | None


def _construct_explicit_bias_args[
    T: DiscreteAnalysisSegmentationValueBounds
](payload: BiasDataPayload[T]) -> _ExplicitAnalysisArgs:
    match payload._seg_criteria:
        case ThresholdBiasSegmentation():
            return _cast_sequence_and_label_thres(payload)
        case LabeledBiasSegmentation():
            return _cast_sequence_and_label_label(payload)
        case _:
            raise InvalidBiasSegmentationConfig


def _cast_sequence_and_label_thres[
    T: DiscreteAnalysisSegmentationValueBounds
](payload: BiasDataPayload[T]) -> _ExplicitAnalysisArgs:
    try:
        data = np.fromiter(
            (float(v) for v in payload._payload),
            count=len(payload._payload),
            dtype=np.float64,
        )
        thres = float(payload._seg_criteria._seg_value)
    except ValueError as exc:
        raise ValueError(
            "Data must be castable to float when using threshold segmentation"
        ) from exc

    return _ExplicitAnalysisArgs(
        data=data,
        threshold=thres,
        label=None,
        threshold_type=payload._seg_criteria._thres_type,
    )


def _cast_sequence_and_label_label[
    T: DiscreteAnalysisSegmentationValueBounds
](payload: BiasDataPayload[T]) -> _ExplicitAnalysisArgs:
    try:
        data = np.fromiter(
            (str(v) for v in payload._payload),
            count=len(payload._payload),
            dtype=object,
        )
    except ValueError as exc:
        raise ValueError(
            "Data must be castable to str when using label segmentation"
        ) from exc

    return _ExplicitAnalysisArgs(
        data=data,
        threshold=None,
        label=str(payload._seg_criteria._seg_value),
        threshold_type=payload._seg_criteria._thres_type,
    )
