from ._fair_ml import (
    model_performance_regression,
    model_performance_classification,
    model_performance_logisitic_regression,
    model_performance_runtime_entry_full,
    model_performance_runtime_entry_partial,
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
import orjson


class InvalidMetricsBody(Exception):
    pass


def linear_regression_analysis(
    y_true: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
    y_pred: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
) -> dict:
    y_true: NDArray = check_and_convert_type(y_true)  # pyright: ignore
    y_pred: NDArray = check_and_convert_type(y_pred)  # pyright: ignore
    res: dict = model_performance_regression(y_true=y_true, y_pred=y_pred)
    print(res)
    return ModelPerformance(
        modelType=ModelType.LinearRegression,
        performanceData=LinearRegressionReport(**res),
    ).model_dump()


def logisitc_regression_analysis(
    y_true: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
    y_pred: Union[NDArray, List[Union[int, float]]],  # pyright: ignore
    decision_threshold: Optional[float] = None,
) -> dict:
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
    model_type = baseline.get("modelType")
    latest_perf = latest.get("performanceData")
    baseline_perf = baseline.get("performanceData")
    if any([model_type is None, latest_perf is None, baseline_perf is None]):
        raise InvalidMetricsBody("Invalid metrics body")
    perf = model_performance_runtime_entry_full(
        model_type=model_type, latest=latest_perf, baseline=baseline_perf, threshold=threshold
    )
    return orjson.loads(perf)


def partial_runtime_check(
    latest: dict, baseline: dict, metrics: List[str], threshold: Optional[float] = 0.10
) -> dict:
    model_type = baseline.get("modelType")
    latest_perf = latest.get("performanceData")
    baseline_perf = baseline.get("performanceData")
    if any([model_type is None, latest_perf is None, baseline_perf is None]):
        raise InvalidMetricsBody("Invalid metrics body")
    perf = model_performance_runtime_entry_partial(
        model_type=model_type,
        latest=latest_perf,
        baseline=baseline_perf,
        evaluation_metrics=metrics,
        threshold=threshold,
    )
    return orjson.loads(perf)
