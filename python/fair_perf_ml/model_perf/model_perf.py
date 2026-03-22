from typing import NamedTuple
from numpy.typing import NDArray
from pydantic import ValidationError
from .._fair_perf_ml import (
    py_model_perf_classification,
    py_model_perf_logistic_regression,
    py_model_perf_linear_regression,
    py_model_perf_class_rt_full,
    py_model_perf_class_rt_partial,
    py_model_perf_lin_reg_rt_full,
    py_model_perf_lin_reg_rt_partial,
    py_model_perf_log_reg_rt_full,
    py_model_perf_log_reg_rt_partial,
)
from ..models import (
    DriftReport,
    ModelType,
    MachineLearningMetric,
    ModelPerformanceReport,
    LinearRegressionReport,
    LogisticRegressionReport,
    BinaryClassificationReport,
)
from .._internal import check_and_convert_type


class DifferentModelTypes(Exception):
    """
    Exception to handle when user passes in wrong model type.
    """


class InvalidMetricsBody(Exception):
    """
    Exception to handle when the user passes an invalid metrics
    payload.
    """


class CleanedRuntimeInput(NamedTuple):
    model_type: ModelType
    runtime: dict[str, float]
    baseline: dict[str, float]


def _serialize_runtime_input(
    runtime: ModelPerformanceReport | dict, baseline: ModelPerformanceReport | dict
) -> CleanedRuntimeInput:
    if isinstance(runtime, dict):
        try:
            runtime = ModelPerformanceReport(**runtime)
        except ValidationError as exc:
            raise InvalidMetricsBody("Invalid data passed for runtime report") from exc

    if isinstance(baseline, dict):
        try:
            baseline = ModelPerformanceReport(**baseline)
        except ValidationError as exc:
            raise InvalidMetricsBody("Invalid data passed for baseline report") from exc

    if runtime.model_type != baseline.model_type:
        raise DifferentModelTypes("Model type mismatch")

    return CleanedRuntimeInput(
        model_type=runtime.model_type,
        runtime=runtime.performance_data.model_dump(),
        baseline=baseline.performance_data.model_dump(),
    )


def linear_regression_analysis(
    y_true: NDArray | list[int | float],  # pyright: ignore
    y_pred: NDArray | list[int | float],  # pyright: ignore
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
    res: dict = py_model_perf_linear_regression(y_true, y_pred)
    return ModelPerformanceReport(
        model_type=ModelType.LinearRegression,
        performance_data=LinearRegressionReport(**res),
    ).model_dump()


def logistic_regression_analysis(
    y_true: NDArray | list[int | float],  # pyright: ignore
    y_pred: NDArray | list[int | float],  # pyright: ignore
    decision_threshold: float | None = 0.5,
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
    res: dict = py_model_perf_logistic_regression(y_true, y_pred, decision_threshold)
    return ModelPerformanceReport(
        model_type=ModelType.LogisticRegression,
        performance_data=LogisticRegressionReport(**res),
    ).model_dump()


def binary_classification_analysis(
    y_true: NDArray | list[int | float],  # pyright: ignore
    y_pred: NDArray | list[int | float],  # pyright: ignore
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
    res: dict = py_model_perf_classification(y_true, y_pred)
    return ModelPerformanceReport(
        model_type=ModelType.BinaryClassification,
        performance_data=BinaryClassificationReport(**res),
    ).model_dump()


def runtime_check_full(
    latest: ModelPerformanceReport | dict,
    baseline: ModelPerformanceReport | dict,
    threshold: float = 0.10,
) -> DriftReport:
    """
    Method to perform a full runtime performance monitoring job.
    args:
        latest: dict - latest analysis output, must match shape
        baseline: dict - baseline analysis output, must match shape
        threshold: Optional[float] - the allowable drift
    returns:
        dict - output analysis
    """
    cleaned_args = _serialize_runtime_input(latest, baseline)

    return _dispatch_runtime_check(cleaned_args, None, threshold)


def partial_runtime_check(
    latest: ModelPerformanceReport | dict,
    baseline: ModelPerformanceReport | dict,
    metrics: list[str],
    threshold: float = 0.10,
) -> DriftReport:
    """
    Method to perform a runtime performance monitoring job on only selected metrics.
    args:
        latest: dict - latest analysis output, must match shape
        baseline: dict - baseline analysis output, must match shape
        threshold: Optional[float] - the allowable drift
    returns:
        dict - output analysis
    """
    cleaned_args = _serialize_runtime_input(latest, baseline)
    return _dispatch_runtime_check(cleaned_args, metrics, threshold)


def _dispatch_runtime_check(
    cleaned_args: CleanedRuntimeInput,
    metrics: list[MachineLearningMetric] | None,
    threshold: float,
) -> DriftReport:

    match (cleaned_args.model_type, metrics):
        case (ModelType.BinaryClassification, None):
            return py_model_perf_class_rt_full(
                cleaned_args.baseline, cleaned_args.runtime, threshold
            )
        case (ModelType.LinearRegression, None):
            return py_model_perf_lin_reg_rt_full(
                cleaned_args.baseline, cleaned_args.runtime, threshold
            )
        case (ModelType.LogisticRegression, None):

            return py_model_perf_log_reg_rt_full(
                cleaned_args.baseline, cleaned_args.runtime, threshold
            )
        case (ModelType.BinaryClassification, list()):
            return py_model_perf_class_rt_partial(
                cleaned_args.baseline, cleaned_args.runtime, metrics, threshold
            )
        case (ModelType.LinearRegression, list()):
            return py_model_perf_lin_reg_rt_partial(
                cleaned_args.baseline, cleaned_args.runtime, metrics, threshold
            )
        case (ModelType.LogisticRegression, list()):
            return py_model_perf_log_reg_rt_partial(
                cleaned_args.baseline, cleaned_args.runtime, metrics, threshold
            )
        case _:
            raise RuntimeError("Invalid state please file an issue")
