import numpy as np
from numpy.typing import NDArray
from typing import List, Union
from enum import Enum


class BiasArrayType(str, Enum):
    FEATURE = "feature"
    GROUND_TRUTH = "ground_truth"
    PREDICTIONS = "predictions"


def _check_and_convert_type(arr: Union[List[Union[str, float, int]], NDArray]) -> NDArray:
    if _is_numpy(arr):
        return arr #pyright: ignore
    if not _is_uniform_type(arr): #pyright: ignore
        raise ValueError(f"Array needs to be of uniform type when of type {list}")
    return _convert_obj_type(arr) #pyright: ignore


def _is_numpy(arr: Union[List[Union[str, float, int]], NDArray]) -> bool:
    return isinstance(arr, np.ndarray)


def _is_uniform_type(arr: List[Union[str, float, int]]) -> bool:
    T = type(arr[0])
    return all([True if type(item) == T else False for item in arr])


def _convert_obj_type(arr: List[Union[str, float, int]]) -> NDArray:
    return np.array(arr)
