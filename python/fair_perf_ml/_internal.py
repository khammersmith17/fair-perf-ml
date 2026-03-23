from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


class InvalidBaseline(Exception):
    """
    Exception for runtime utitlies when user passes
    invalid baseline shape
    """


FloatingPointDataSlice: TypeAlias = Iterable[float] | list[float] | NDArray


def cast_floating_point_slice(arr: FloatingPointDataSlice) -> NDArray:
    if isinstance(arr, list) or isinstance(arr, Iterable):
        arr = check_and_convert_type(arr)

    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError("Requires data of floating point type")

    return arr


def check_and_convert_type(
    arr: list[str | float | int] | NDArray | Iterable[float],
) -> NDArray:
    """
    Coerece container type into numpy array as the Rust function expects a
    numpy array.
    """
    if _is_numpy(arr):
        return arr  # pyright: ignore
    if not _is_uniform_type(arr):  # pyright: ignore
        raise ValueError(f"Array needs to be of uniform type when of type {list}")
    return _convert_obj_type(arr)  # pyright: ignore


def _is_numpy(arr: list[str | float | int] | NDArray | Iterable[float]) -> bool:
    """
    Utility to check to see if a container is a numpy array.
    """
    return isinstance(arr, np.ndarray)


def _is_uniform_type(arr: list[str | float | int]) -> bool:
    """
    Validate that all items in a python list are of the same type.
    """
    if not isinstance(arr, list):
        return False
    if len(arr) == 0:
        raise ValueError("Empty datasets are not supported")
    T = type(arr[0])
    return all([True if isinstance(item, T) else False for item in arr])


def _convert_obj_type(arr: list[str | float | int]) -> NDArray:
    """
    Coerce to numpy array
    """
    return np.array(arr)
