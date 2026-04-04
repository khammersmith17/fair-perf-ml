from __future__ import annotations

from typing import cast

from fair_perf_ml._internal import check_and_convert_type
from fair_perf_ml.bias.segmentation import (
    BiasDataPayload, DiscreteAnalysisSegmentationValueBounds,
    _construct_explicit_bias_args)
from fair_perf_ml.models import (AnalysisReport, DriftReport,
                                 ModelBiasDriftMetric)
from numpy.typing import NDArray

from .._fair_perf_ml import (py_model_bias_analyzer,
                             py_model_bias_analyzer_explicit_seg,
                             py_model_bias_partial_check,
                             py_model_bias_runtime_check)


def model_bias_perform_analysis_explicit_segmentation[
    F: DiscreteAnalysisSegmentationValueBounds,
    G: DiscreteAnalysisSegmentationValueBounds,
    P: DiscreteAnalysisSegmentationValueBounds,
](
    feature: BiasDataPayload[F],
    ground_truth: BiasDataPayload[G],
    prediction: BiasDataPayload[P],
) -> AnalysisReport:
    """
    Method to provide explicit segmentation criteria for ad hoc data bias analysis as opposed
    to using the default derivation logic in Rust core to determine segmentation logic from
    heurisitcs.
    Segmentation and data are passed as a single unit in BiasDataPayload.

    args:
        feature: DataBiasPayload[F]
        ground_truth: DataBiasPayload[G]
    returns:
        AnalysisReport
    """
    f_args = _construct_explicit_bias_args(feature)
    gt_args = _construct_explicit_bias_args(ground_truth)
    p_args = _construct_explicit_bias_args(prediction)

    return py_model_bias_analyzer_explicit_seg(
        feature_array=f_args.data,
        feat_segmentation_threshold=f_args.threshold,
        feat_segmentation_label=f_args.label,
        feat_threshold_type=f_args.threshold_type,
        ground_truth_array=gt_args.data,
        gt_segmentation_threshold=gt_args.threshold,
        gt_segmentation_label=gt_args.label,
        gt_threshold_type=gt_args.threshold_type,
        prediction_array=p_args.data,
        pred_segmentation_threshold=p_args.threshold,
        pred_segmentation_label=p_args.label,
        pred_threshold_type=p_args.threshold_type,
    )


def model_bias_perform_analysis[
    F, G, P
](
    feature: list[F] | NDArray,
    ground_truth: list[G] | NDArray,
    predictions: list[P] | NDArray,
    feature_label_or_threshold: F,
    ground_truth_label_or_threshold: G,
    prediction_label_or_threshold: P,
) -> AnalysisReport:
    """
    Performs model bias analysis on the data passed. The arrays passed with the feature,
    prediction and ground truth data must all be of the same length. The collection type that
    is passed must be coercable to a numpy array.

    Type in the data container passed must the label or threshold value passed for each
    criteria type.
    Args:
        feature: list[F] | NDArray -> the feature data
            most efficient to pass as numpy array
        ground_truth: list[G] | NDArray -> the ground truth data
            most efficient to pass as numpy array
        predictions: list[P] | NDArray -> the prediction data
            most efficient to pass as numpy array
        feature_label_or_threshold: F -> segmentation parameter for the feature
        ground_truth_label_or_threshold: G -> segmenation parameter for ground truth
        prediction_label_or_threshold: P -> segmenation parameter for predictions
    """
    feature = cast(NDArray, check_and_convert_type(feature))
    ground_truth = cast(NDArray, check_and_convert_type(ground_truth))
    predictions = cast(NDArray, check_and_convert_type(predictions))

    return py_model_bias_analyzer(
        feature_array=feature,
        ground_truth_array=ground_truth,
        prediction_array=predictions,
        feature_label_or_threshold=feature_label_or_threshold,
        ground_truth_label_or_threshold=ground_truth_label_or_threshold,
        prediction_label_or_threshold=prediction_label_or_threshold,
    )


def model_bias_runtime_comparison(
    baseline: AnalysisReport, comparison: AnalysisReport, threshold: float | None = 0.10
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
    return py_model_bias_runtime_check(
        baseline=baseline, latest=comparison, threshold=threshold
    )


def model_bias_partial_runtime_comparison(
    baseline: AnalysisReport,
    comparison: AnalysisReport,
    metrics: list[ModelBiasDriftMetric],
    threshold: float = 0.10,
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
    return py_model_bias_partial_check(
        baseline=baseline,
        latest=comparison,
        metrics=metrics,
        threshold=threshold,
    )
