from typing import Union, Optional, List, Dict
from .._fair_perf_ml import (
    py_data_bias_analyzer as data_bias_analyzer,
    py_data_bias_runtime_check as data_bias_runtime_check,
    py_data_bias_partial_check as data_bias_partial_check,
)
from numpy.typing import NDArray
from .._internal import check_and_convert_type, _map_metric_enum
from ..models import DriftReport, DataBiasMetric


def perform_analysis(
    feature: Union[List[Union[str, float, int]], NDArray],  # pyright: ignore
    ground_truth: Union[List[Union[str, float, int]], NDArray],  # pyright: ignore
    feature_label_or_threshold: Union[str, float, int],
    ground_truth_label_or_threshold: Union[str, float, int],
) -> Dict[str, float]:
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
    feature: NDArray = check_and_convert_type(feature)
    ground_truth: NDArray = check_and_convert_type(ground_truth)

    res: Dict[str, float] = data_bias_analyzer(
        feature_array=feature,
        ground_truth_array=ground_truth,
        feature_label_or_threshold=feature_label_or_threshold,
        ground_truth_label_or_threshold=ground_truth_label_or_threshold,
    )

    # simply for nice formatting
    return res


def runtime_comparison(
    baseline: dict[str, float],
    latest: dict[str, float],
    threshold: Optional[float] = 0.10,
) -> DriftReport:
    """
    Compare the current runtime analysis result to the baseline to determine the model
    drift from the baseline. Metrics the exceed the provided drift will be present in the
    drift report.
    Args:
        baseline: dict -> the result from calling perform_analysis on the baseline data
        latest: dict -> the current data for comparison from calling perform_analysis
        threshold: Optionl[float]=None -> the comparison threshold, defaults to 0.10 in rust mod
    Returns:
        dict - the drift report, detailing the metrics that have drifted and to what degree.
    """
    res: DriftReport = (
        data_bias_runtime_check(baseline=baseline, latest=latest, threshold=threshold)
        if threshold
        else data_bias_runtime_check(baseline=baseline, latest=latest)
    )
    return res


def partial_runtime_comparison(
    baseline: dict[str, float],
    latest: dict[str, float],
    metrics: List[Union[DataBiasMetric, str]],
    threshold: Optional[float] = 0.10,
) -> DriftReport:
    """
    Performs the same drift comparison as the above method, but allows the user to narrow the
    drift evaluation, explicitly specifying the metrics to evaluate for drift, rather than the
    full available suite.
    Args:
        baseline: dict -> the result from calling perform_analysis on the baseline data
        latest: dict -> the current data for comparison from calling perform_analysis
        metrics: List[str] -> the list of metrics we want to evaluate on
        threshold: Optionl[float]=None -> the comparison threshold, defaults to 0.10 in rust mod
    Returns:
        dict
    """
    cleaned_metrics = list(map(_map_metric_enum, metrics))
    res: DriftReport = data_bias_partial_check(
        baseline=baseline, latest=latest, metrics=cleaned_metrics, threshold=threshold
    )
    # for nicer formatting on the return
    return res
