from ._fair_perf_ml.py_model_perf import (
    py_model_performance_regression as model_performance_regression,
    py_model_performance_classification as model_performance_classification,
    py_model_performance_logisitic_regression as model_performance_logisitic_regression,
    py_model_performance_runtime_entry_full as model_performance_runtime_entry_full,
    py_model_performance_runtime_entry_partial as model_performance_runtime_entry_partial,
)
from .models import (
    ModelType,
    ModelPerformance,
    LinearRegressionReport,
    LogisticRegressionReport,
    BinaryClassificationReport,
)
from ._internal import check_and_convert_type
from numpy.typing import NDArray
from typing import Union, List, Optional


class DifferentModelTypes(Exception):
    """
    Exception to handle when user passes in wrong model type
    """


class InvalidMetricsBody(Exception):
    """
    Exception to handle when the user passes an invalid metrics
    payload
    """


def linear_regression_analysis(
    y_true: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
    y_pred: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
) -> dict:
    """
    Analysis for a linear regression model type.
    Will apply labels in accordance with the given threshold
    args:
        y_true: numpy array/python list
        y_pred: numpy array/python list
    returns:
        dict: analysis output
    """

    y_true: NDArray = check_and_convert_type(y_true)  # pyright: ignore
    y_pred: NDArray = check_and_convert_type(y_pred)  # pyright: ignore
    res: dict = model_performance_regression(y_true=y_true, y_pred=y_pred)
    return ModelPerformance(
        modelType=ModelType.LinearRegression,
        performanceData=LinearRegressionReport(**res),
    ).model_dump()


def logistic_regression_analysis(
    y_true: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
    y_pred: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
    decision_threshold: Optional[float] = 0.5,
) -> dict:
    """
    Analysis for a logistic regression model type.
    Will apply labels in accordance with the given threshold
    args:
        y_true: numpy array/python list
        y_pred: numpy array/python list
        decision_threshold: threshold used to apply label
    returns:
        dict: analysis output
    """
    y_true: NDArray = check_and_convert_type(y_true)  # pyright: ignore
    y_pred: NDArray = check_and_convert_type(y_pred)  # pyright: ignore
    res: dict = model_performance_logisitic_regression(
        y_true=y_true, y_pred=y_pred, decision_threshold=decision_threshold
    )
    return ModelPerformance(
        modelType=ModelType.LogisticRegression,
        performanceData=LogisticRegressionReport(**res),
    ).model_dump()


def binary_classification_analysis(
    y_true: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
    y_pred: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
) -> dict:
    """
    Analysis for a classification model type.
    Will apply labels in accordance with the given threshold
    args:
        y_true: numpy array/python list
        y_pred: numpy array/python list
    returns:
        dict: analysis output
    """
    y_true: NDArray = check_and_convert_type(y_true)  # pyright: ignore
    y_pred: NDArray = check_and_convert_type(y_pred)  # pyright: ignore
    res: dict = model_performance_classification(y_true=y_true, y_pred=y_pred)
    return ModelPerformance(
        modelType=ModelType.BinaryClassification,
        performanceData=BinaryClassificationReport(**res),
    ).model_dump()


def runtime_check_full(
    latest: dict, baseline: dict, threshold: Optional[float] = 0.10
) -> dict:
    """
    Method to perform a full runtime performance monitoring job.
    args:
        latest: dict - latest analysis output, must match shape
        baseline: dict - baseline analysis output, must match shape
        threshold: Optional[float] - the allowable drift
    returns:
        dict - output analysis
    """
    model_type = baseline.get("modelType")
    if model_type != latest.get("modelType"):
        raise DifferentModelTypes("Models types do not match")
    latest_perf = latest.get("performanceData")
    baseline_perf = baseline.get("performanceData")
    if any([model_type is None, latest_perf is None, baseline_perf is None]):
        raise InvalidMetricsBody("Invalid metrics body")
    perf: dict = model_performance_runtime_entry_full(
        model_type=model_type,
        latest=latest_perf,
        baseline=baseline_perf,
        threshold=threshold,
    )
    return perf


def partial_runtime_check(
    latest: dict, baseline: dict, metrics: List[str], threshold: Optional[float] = 0.10
) -> dict:
    """
    Method to perform a runtime performance monitoring job on only selected metrics.
    args:
        latest: dict - latest analysis output, must match shape
        baseline: dict - baseline analysis output, must match shape
        threshold: Optional[float] - the allowable drift
    returns:
        dict - output analysis
    """
    model_type = baseline.get("modelType")
    latest_perf = latest.get("performanceData")
    baseline_perf = baseline.get("performanceData")
    if any([model_type is None, latest_perf is None, baseline_perf is None]):
        raise InvalidMetricsBody("Invalid metrics body")
    perf: dict = model_performance_runtime_entry_partial(
        model_type=model_type,
        latest=latest_perf,
        baseline=baseline_perf,
        evaluation_metrics=metrics,
        threshold=threshold,
    )
    return perf
