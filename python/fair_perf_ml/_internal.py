from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


class InvalidBaseline(Exception):
    """
    Exception for runtime utitlies when user passes
    invalid baseline shape
    """


class NonUniformTypeException(Exception):
    """
    Exception to be thrown when user passes in an Iterable
    that is not of uniform type
    """

    _default_message = "Datasets of hetergenous types are not supported"

    def __init__(self, msg: str | None = None, *args):
        if msg is None:
            msg = self._default_message
        super().__init__(msg, *args)


class EmptyDatasetException(Exception):
    """
    Exception to be thrown when user passes in an Iterable
    that is not of uniform type
    """

    _default_message = "Empty datasets are not supported"

    def __init__(self, msg: str | None = None, *args):
        if msg is None:
            msg = self._default_message
        super().__init__(msg, *args)


FloatingPointDataSlice: TypeAlias = Sequence[float] | list[float] | NDArray
UniformTypeDataSlice: TypeAlias = Sequence[int | float | str] | NDArray


def cast_floating_point_slice(arr: FloatingPointDataSlice) -> NDArray:
    if isinstance(arr, list) or isinstance(arr, Sequence):
        arr = check_and_convert_type(arr)

    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError("Requires data of floating point type")

    return arr


def check_and_convert_type[T](arr: list[T] | NDArray | Sequence[T]) -> NDArray:
    """
    Coerece container type into numpy array as the Rust function expects a
    numpy array.
    """
    if _is_numpy(arr):
        return arr  # pyright: ignore
    if not _is_uniform_type(arr):  # pyright: ignore
        raise NonUniformTypeException
    return _convert_obj_type(arr)  # pyright: ignore


def _is_numpy[T](arr: list[T] | NDArray | Sequence[T]) -> bool:
    """
    Utility to check to see if a container is a numpy array.
    """
    return isinstance(arr, np.ndarray)


def _extract_sequence_type(arr: Sequence[Any]) -> type:
    if not _is_uniform_type(arr):
        raise NonUniformTypeException

    if len(arr) == 0:
        raise EmptyDatasetException

    return type(arr[0])


def _is_uniform_type(arr: Sequence[Any]) -> bool:
    """
    Validate that all items in a python list are of the same type.
    """
    if len(arr) == 0:
        raise EmptyDatasetException
    T = type(arr[0])
    return all([True if isinstance(item, T) else False for item in arr])


def _convert_obj_type(arr: list[str | float | int]) -> NDArray:
    """
    Coerce to numpy array
    """
    return np.array(arr)
