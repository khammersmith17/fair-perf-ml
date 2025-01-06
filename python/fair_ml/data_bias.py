from ._fair_ml import data_bias_analyzer, data_bias_runtime_check
from typing import Union, Optional, List
from numpy.typing import NDArray
import orjson
from .models import BaseRuntimeReturn, DataBiasBaseline
from ._internal import (
    ArrayType,
    _is_numpy,
    _convert_obj_type
)


def perform_analysis(
    feature: Union[List[Union[str, float, int]], NDArray], # pyright: ignore
    ground_truth: Union[List[Union[str, float, int]], NDArray], # pyright: ignore
    feature_label_or_threshold: Union[str, float, int],
    ground_truth_label_or_threshold: Union[str, float, int]
    ) -> dict[str, float]:
    """
    interface into rust class
    makes sure we are passing numpy arrays to the rust function
    Args:
        feature: Union[List[Union[str, float, int]], NDArray] -> the feature data
            most efficient to pass as numpy array
        ground_truth: Union[List[Union[str, float, int]], NDArray] -> the ground truth data
            most efficient to pass as numpy array
        feature_label_or_threshold: Union[str, float, int] -> segmentation parameter for the feature
        ground_truth_label_or_threshold: Union[str, float, int] -> segmenation parameter for ground truth
    """
    # want to pass numpy arrays to rust
    # type resolution in rust mod depends on numpy arrays
    # ignoring lsp message because we are deliberately changing the type on these values
    if not _is_numpy(feature):
        assert(isinstance(feature, list))
        feature: NDArray = _convert_obj_type(feature, ArrayType.FEATURE)
    if not _is_numpy(ground_truth):
        assert(isinstance(ground_truth, list))
        ground_truth: NDArray = _convert_obj_type(ground_truth, ArrayType.GROUND_TRUTH)

    res: dict[str, float] = data_bias_analyzer(
        feature_array=feature,
        ground_truth_array=ground_truth,
        feature_label_or_threshold=feature_label_or_threshold,
        ground_truth_label_or_threshold=ground_truth_label_or_threshold
    )

    # simply for nice formatting
    return DataBiasBaseline(**res).model_dump()


def runtime_comparison(
    baseline: dict[str, float],
    latest: dict[str, float],
    threshold: Optional[float]=None
    ) -> dict[str, str]:
    """
    interface into rust module
    serves to nicely formats the return as dicts are ordered and hashmaps are not
    Args:
        baseline: dict -> the result from calling perform_analysis on the baseline data
        latest: dict -> the current data for comparison from calling perform_analysis
        threshold: Optionl[float]=None -> the comparison threshold, defaults to 0.10 in rust mod
    Returns:
        dict
    """
    res: str = data_bias_runtime_check(
        baseline=baseline,
        latest=latest,
        threshold=threshold
    )

    # for nicer formatting on the return
    return BaseRuntimeReturn(**orjson.loads(res)).model_dump()
