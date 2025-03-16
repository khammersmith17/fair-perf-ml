from ._fair_ml import (
    model_bias_analyzer,
    model_bias_runtime_check,
    model_bias_partial_check,
)
from ._internal import check_and_convert_type
from .models import ModelBiasBaseline
from numpy.typing import NDArray
from typing import List, Union, Optional
import orjson


def perform_analysis(
    feature: Union[List[Union[str, float, int]], NDArray],  # pyright: ignore
    ground_truth: Union[List[Union[str, float, int]], NDArray],  # pyright: ignore
    predictions: Union[List[Union[str, float, int]], NDArray],  # pyright: ignore
    feature_label_or_threshold: Union[str, float, int],
    ground_truth_label_or_threshold: Union[str, float, int],
    prediction_label_or_threshold: Union[str, float, int],
) -> dict[str, float]:
    """
    interface into rust class
    makes sure we are passing numpy arrays to the rust function
    Args:
        feature: Union[List[Union[str, float, int]], NDArray] -> the feature data
            most efficient to pass as numpy array
        ground_truth: Union[List[Union[str, float, int]], NDArray] -> the ground truth data
            most efficient to pass as numpy array
        predictions: Union[List[Union[str, float, int]], NDArray] -> the prediction data
            most efficient to pass as numpy array
        feature_label_or_threshold: Union[str, float, int] -> segmentation parameter for the feature
        ground_truth_label_or_threshold: Union[str, float, int] -> segmenation parameter for ground truth
        prediction_label_or_threshold: Union[str, float, int] -> segmenation parameter for predictions
    """
    feature: NDArray = check_and_convert_type(feature)
    ground_truth: NDArray = check_and_convert_type(ground_truth)
    predictions: NDArray = check_and_convert_type(predictions)

    res: dict[str, float] = model_bias_analyzer(
        feature_array=feature,
        ground_truth_array=ground_truth,
        prediction_array=predictions,
        feature_label_or_threshold=feature_label_or_threshold,
        ground_truth_label_or_threshold=ground_truth_label_or_threshold,
        prediction_label_or_threshold=prediction_label_or_threshold,
    )

    # for nice formatting
    return ModelBiasBaseline(**res).model_dump()


def runtime_comparison(
    baseline: dict, comparison: dict, threshold: Optional[float] = None
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
    res: str = model_bias_runtime_check(
        baseline=baseline, latest=comparison, threshold=threshold
    )

    # for nice formatting
    return orjson.loads(res)


def parital_runtime_comparison(
    baseline: dict,
    comparison: dict,
    metrics: List[str],
    threshold: Optional[float] = None,
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
    res: str = model_bias_partial_check(
        baseline=baseline, latest=comparison, metrics=metrics, threshold=threshold
    )

    # for nice formatting
    return orjson.loads(res)
