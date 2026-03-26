from __future__ import annotations
from typing import cast
from numpy.typing import NDArray

from .._fair_perf_ml import py_data_bias_analyzer
from .._fair_perf_ml import py_data_bias_partial_check
from .._fair_perf_ml import py_data_bias_runtime_check
from .._internal import check_and_convert_type
from ..models import DataBiasDriftMetric, DriftReport


def perform_analysis[
    F, G
](
    feature: list[F] | NDArray,
    ground_truth: list[G] | NDArray,
    feature_label_or_threshold: F,
    ground_truth_label_or_threshold: G,
) -> dict[str, float]:
    """
    interface into rust class
    makes sure we are passing numpy arrays to the rust function
    Args:
        feature: list[str | float | int] | NDArray -> the feature data
            most efficient to pass as numpy array
        ground_truth: list[str | float | int] | NDArray -> the ground truth data
            most efficient to pass as numpy array
        feature_label_or_threshold: str | float | int -> segmentation parameter for the feature
        ground_truth_label_or_threshold: str | float | int -> segmenation parameter for ground truth
    """
    # want to pass numpy arrays to rust
    # type resolution in rust mod depends on numpy arrays
    feature = cast(NDArray, check_and_convert_type(feature))
    ground_truth = cast(NDArray, check_and_convert_type(ground_truth))

    res: dict[str, float] = py_data_bias_analyzer(
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
    threshold: float = 0.10,
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
    res: DriftReport = py_data_bias_runtime_check(
        baseline=baseline, latest=latest, threshold=threshold
    )
    return res


def partial_runtime_comparison(
    baseline: dict[str, float],
    latest: dict[str, float],
    metrics: list[DataBiasDriftMetric],
    threshold: float = 0.10,
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
    res: DriftReport = py_data_bias_partial_check(
        baseline=baseline, latest=latest, metrics=metrics, threshold=threshold
    )
    # for nicer formatting on the return
    return res
