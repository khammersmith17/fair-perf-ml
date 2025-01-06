from ._fair_ml import model_bias_analyzer, model_bias_runtime
from ._internal import (
    ArrayType,
    _is_numpy,
    _convert_obj_type
)
from .models import BaseRuntimeReturn, ModelBiasBaseline
from numpy.typing import NDArray
from typing import List, Union, Optional
import orjson


def perform_analysis(
    feature: Union[List[Union[str, float, int]], NDArray],
    ground_truth: Union[List[Union[str, float, int]], NDArray],
    predictions: Union[List[Union[str, float, int]], NDArray],
    feature_label_or_threshold: Union[str, float, int],
    ground_truth_label_or_threshold: Union[str, float, int],
    prediction_label_or_threshold: Union[str, float, int]
    ) -> dict[str, float]:
    if not _is_numpy(feature):
        assert(isinstance(feature, list))
        feature: NDArray = _convert_obj_type(feature, ArrayType.FEATURE)
    if not _is_numpy(ground_truth):
        assert(isinstance(ground_truth, list))
        ground_truth: NDArray = _convert_obj_type(ground_truth, ArrayType.GROUND_TRUTH)
    if not _is_numpy(predictions):
        assert(isinstance(predictions, list))
        predictions: NDArray = _convert_obj_type(predictions, ArrayType.PREDICTIONS)

    res: dict[str, float] = model_bias_analyzer(
        feature_array=feature,
        ground_truth_array=ground_truth,
        prediction_array=predictions,
        feature_label_or_threshold=feature_label_or_threshold,
        ground_truth_label_or_threshold=ground_truth_label_or_threshold,
        prediction_label_or_threshold=prediction_label_or_threshold
    )

    # for nice formatting
    return ModelBiasBaseline(**res).model_dump()


def runtime_comparison(
    baseline: dict,
    comparison: dict,
    threshold: Optional[float]
    ) -> dict[str, str]:

    res: str = model_bias_runtime(
        baseline=baseline,
        latest=comparison,
        threshold=threshold
    )

    # for nice formatting
    return BaseRuntimeReturn(**orjson.loads(res)).model_dump()
