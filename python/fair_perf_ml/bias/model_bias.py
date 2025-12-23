from typing import List, Union, Optional, Dict
from numpy.typing import NDArray
from .._fair_perf_ml.py_model_bias import (
    py_model_bias_analyzer as model_bias_analyzer,
    py_model_bias_runtime_check as model_bias_runtime_check,
    py_model_bias_partial_check as model_bias_partial_check,
)
from .._internal import check_and_convert_type, _map_metric_enum
from ..models import DriftReport


def perform_analysis(
    feature: Union[List[Union[str, float, int]], NDArray],  # pyright: ignore
    ground_truth: Union[List[Union[str, float, int]], NDArray],  # pyright: ignore
    predictions: Union[List[Union[str, float, int]], NDArray],  # pyright: ignore
    feature_label_or_threshold: Union[str, float, int],
    ground_truth_label_or_threshold: Union[str, float, int],
    prediction_label_or_threshold: Union[str, float, int],
) -> Dict[str, float]:
    """
    Performs model bias analysis on the data passed. The arrays passed with the feature,
    prediction and ground truth data must all be of the same length. The collection type that
    is passed must be coercable to a numpy array.
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

    res: Dict[str, float] = model_bias_analyzer(
        feature_array=feature,
        ground_truth_array=ground_truth,
        prediction_array=predictions,
        feature_label_or_threshold=feature_label_or_threshold,
        ground_truth_label_or_threshold=ground_truth_label_or_threshold,
        prediction_label_or_threshold=prediction_label_or_threshold,
    )

    return res


def runtime_comparison(
    baseline: dict, comparison: dict, threshold: Optional[float] = 0.10
) -> DriftReport:
    """
    Evaluates a runtime analysis report for drift relative to the baseline, on all metrics
    define in the library suite. The criteria for a drift failure is the drift define by the user
    and from the baseline set.
    Args:
        baseline: dict -> the result from calling perform_analysis on the baseline data
        comparison: dict -> the current data for comparison from calling perform_analysis
        threshold: Optionl[float]=None -> the comparison threshold, defaults to 0.10 in rust mod
    Returns:
        dict
    """
    res: DriftReport = model_bias_runtime_check(
        baseline=baseline, latest=comparison, threshold=threshold
    )

    return res


def partial_runtime_comparison(
    baseline: dict,
    comparison: dict,
    metrics: List[str],
    threshold: Optional[float] = None,
) -> DriftReport:
    """
    Performs the same logic as the function above, but on a limited set of metrics defined by
    the user. This users to narrow the scope to what they are concerned about.
    Args:
        baseline: dict -> the result from calling perform_analysis on the baseline data
        latest: dict -> the current data for comparison from calling perform_analysis
        metrics: List[str] -> the list of metrics we want to evaluate on
        threshold: Optionl[float]=None -> the comparison threshold, defaults to 0.10 in rust mod
    Returns:
        dict
    """
    cleaned_metrics = list(map(_map_metric_enum, metrics))
    res: DriftReport = model_bias_partial_check(
        baseline=baseline,
        latest=comparison,
        metrics=cleaned_metrics,
        threshold=threshold,
    )

    return res
