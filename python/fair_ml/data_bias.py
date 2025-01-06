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
    feature: Union[List[Union[str, float, int]], NDArray],
    ground_truth: Union[List[Union[str, float, int]], NDArray],
    feature_label_or_threshold: Union[str, float, int],
    ground_truth_label_or_threshold: Union[str, float, int]
    ) -> dict[str, float]:

    # want to pass numpy arrays to rust
    # type resolution in rust mod depends on numpy arrays
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
    threshold: Optional[float]
    ) -> dict[str, str]:
    res: str = data_bias_runtime_check(
        baseline=baseline,
        latest=latest,
        threshold=threshold
    )

    # for nicer formatting on the return
    return BaseRuntimeReturn(**orjson.loads(res)).model_dump()
