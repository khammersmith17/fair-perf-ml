import numpy as np
from numpy.typing import NDArray
from typing import List, Union


class InvalidBaseline(Exception):
    """
    Exception for runtime utitlies when user passes
    invalid baseline shape
    """


def check_and_convert_type(
    arr: Union[List[Union[str, float, int]], NDArray]
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


def _is_numpy(arr: Union[List[Union[str, float, int]], NDArray]) -> bool:
    """
    Utility to check to see if a container is a numpy array.
    """
    return isinstance(arr, np.ndarray)


def _is_uniform_type(arr: List[Union[str, float, int]]) -> bool:
    """
    Validate that all items in a python list are of the same type.
    """
    if not isinstance(arr, list):
        return False
    T = type(arr[0])
    return all([True if isinstance(item, T) else False for item in arr])


def _convert_obj_type(arr: List[Union[str, float, int]]) -> NDArray:
    """
    Coerce to numpy array
    """
    return np.array(arr)
