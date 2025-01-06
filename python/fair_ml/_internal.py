import numpy as np
from numpy.typing import NDArray
from typing import List, Union
from enum import Enum


class ArrayType(str, Enum):
    FEATURE = "feature"
    GROUND_TRUTH = "ground_truth"
    PREDICTIONS = "predictions"


def _is_numpy(arr: Union[List[Union[str, float, int]], NDArray]) -> bool:
    return isinstance(arr, np.ndarray)


def _is_uniform_type(arr: List[Union[str, float, int]]) -> bool:
    T = type(arr[0])
    return all([True if type(item) == T else False for item in arr])


def _convert_obj_type(arr: List[Union[str, float, int]], t: ArrayType) -> NDArray:
    if not _is_uniform_type(arr):
        raise ValueError(f"Objects in {t.value} must all be of the same data type")

    return np.array(arr)

