"""
Provides runtime drift monitoring utilites and managers for such utilities.
These drift techniques can be used for data drift or a proxy for model drift
when ground truth feedback loop is slow.

TODO:
- Consider having super thin python wrapper to ensure type coercion, ie np arrays
    and string lists
- this is implement as on top of api wrapper
- streaming manager type then uses the thing wrapper
- my hunch is that the manager types present enough boundary separation to be implemented
    as a submodule
"""

from __future__ import annotations
from enum import Enum
from typing import Tuple, Union, List, Iterable, Any
import numpy as np
from numpy.typing import NDArray
from .._fair_perf_ml import (
    PyStreamingContinuousDataDrift as StreamingContinuousDataDrift,
    PyContinuousDataDrift as ContinuousDataDrift,
    PyStreamingCategoricalDataDrift as StreamingCategoricalDataDrift,
    PyCategoricalDataDrift as CategoricalDataDrift,
)


class DriftType(str, Enum):
    CONTINUOUS = "Continuous"
    CATEGORICAL = "Categoical"


DataDriftUtil = Union[
    StreamingContinuousDataDrift,
    ContinuousDataDrift,
    StreamingCategoricalDataDrift,
    CategoricalDataDrift,
]
DataDriftRegisterRequest = Tuple[str, Union[str, DriftType], Union[NDArray, List[str]]]
ContinuousDataDriftRegisterEntry = Tuple[str, NDArray]
CategoricalRegisterEntry = Tuple[str, List[str]]


class DataDriftParameterValidationError(Exception):
    """
    Exception for when users pass invalid data in
    """


def _coerce_data_to_np_float(data: Iterable[Any]) -> NDArray:
    """
    Utility to convert to np float64 array.
    Will throw an exception when data is not numeric and cannot be casted to float.
    """
    try:
        return np.array([float(item) for item in data], dtype=np.float64)
    except ValueError:
        raise TypeError("StreamingContinuousDataDrift data must be numeric")


def _coerce_data_to_string_list(data: Iterable[Any]) -> List[str]:
    """
    Utility to convert data into string type for categorical analysis.
    """
    return [str(item) for item in data]


def _coerce_data(
    agent: DataDriftUtil, data: Iterable[Any]
) -> Union[NDArray, List[str]]:
    if isinstance(agent, Union[StreamingContinuousDataDrift, ContinuousDataDrift]):
        return _coerce_data_to_np_float(data)
    else:
        return _coerce_data_to_string_list(data)


def smooth_continuous_register_entry(
    register_entry: DataDriftRegisterRequest,
) -> ContinuousDataDriftRegisterEntry:
    """
    Perform required data coersions.
    """
    if len(register_entry) != 3:
        raise DataDriftParameterValidationError(
            "Register entry must be length 3 (column name, DriftType | str, baseline data)"
        )

    col_name = register_entry[0]

    # coerce data into numpy float numpy array
    try:
        bl_data = _coerce_data_to_np_float(register_entry[2])
    except ValueError:
        raise DataDriftParameterValidationError(
            "Invalid data for continuous baseline data"
        )

    return (col_name, bl_data)


def smooth_categorical_register_entry(
    register_entry: DataDriftRegisterRequest,
) -> CategoricalRegisterEntry:
    """
    Perform required data coersions.
    """
    if len(register_entry) != 3:
        raise DataDriftParameterValidationError(
            "Register entry must be length 3 (column name, DriftType | str, baseline data)"
        )

    col_name = register_entry[0]
    bl_data = _coerce_data_to_string_list(register_entry[2])
    return (col_name, bl_data)


def resolve_drift_type(register_entry: DataDriftRegisterRequest) -> DriftType:
    try:
        return DriftType(register_entry[1])
    except ValueError:
        raise DataDriftParameterValidationError(
            "Register entry must be length 3 (column name, DriftType, baseline data)"
        )
