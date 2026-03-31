import pytest
from fair_perf_ml._internal import (EmptyDatasetException,
                                    NonUniformTypeException)
from fair_perf_ml.bias.segmentation import (BiasDataPayload,
                                            BiasPayloadTypeException,
                                            BiasSegmentationThresholdType,
                                            BiasSegmentationType,
                                            InvalidBiasSegmentationConfig,
                                            LabeledBiasSegmentation,
                                            ThresholdBiasSegmentation,
                                            _construct_explicit_bias_args,
                                            bias_segmentation_criteria_factory)


def test_labeled_label_match():
    seg = LabeledBiasSegmentation("male")  # pyright: ignore
    assert seg._label("male") == 1


def test_labeled_label_no_match():
    seg = LabeledBiasSegmentation("male")  # pyright: ignore
    assert seg._label("female") == 0


def test_labeled_label_batch():
    seg = LabeledBiasSegmentation("male")  # pyright: ignore
    result = seg._label_batch(["male", "female", "male", "other"])
    assert result == [1, 0, 1, 0]


def test_labeled_seg_value():
    seg = LabeledBiasSegmentation(1)  # pyright: ignore
    assert seg._seg_value == 1


def test_labeled_seg_type():
    seg = LabeledBiasSegmentation("x")  # pyright: ignore
    assert seg._seg_type == BiasSegmentationType.Label


def test_labeled_thres_type_is_none():
    seg = LabeledBiasSegmentation("x")  # pyright: ignore
    assert seg._thres_type is None


def test_labeled_seg_value_type():
    seg = LabeledBiasSegmentation("male")  # pyright: ignore
    assert seg._seg_value_type() == str


def test_threshold_greater_than_above():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.GreaterThan  # pyright: ignore
    )
    assert seg._label(0.6) == 1


def test_threshold_greater_than_equal():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.GreaterThan  # pyright: ignore
    )
    assert seg._label(0.5) == 0


def test_threshold_greater_than_below():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.GreaterThan  # pyright: ignore
    )
    assert seg._label(0.4) == 0


def test_threshold_greater_than_batch():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.GreaterThan  # pyright: ignore
    )
    result = seg._label_batch([0.4, 0.5, 0.6])
    assert result == [0, 0, 1]


def test_threshold_gte_above():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.GreaterThanEqualTo  # pyright: ignore
    )
    assert seg._label(0.6) == 1


def test_threshold_gte_equal():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.GreaterThanEqualTo  # pyright: ignore
    )
    assert seg._label(0.5) == 1


def test_threshold_gte_below():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.GreaterThanEqualTo  # pyright: ignore
    )
    assert seg._label(0.4) == 0


def test_threshold_gte_batch():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.GreaterThanEqualTo  # pyright: ignore
    )
    result = seg._label_batch([0.4, 0.5, 0.6])
    assert result == [0, 1, 1]


def test_threshold_less_than_below():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.LessThan  # pyright: ignore
    )
    assert seg._label(0.4) == 1


def test_threshold_less_than_equal():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.LessThan  # pyright: ignore
    )
    assert seg._label(0.5) == 0


def test_threshold_less_than_above():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.LessThan  # pyright: ignore
    )
    assert seg._label(0.6) == 0


def test_threshold_less_than_batch():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.LessThan  # pyright: ignore
    )
    result = seg._label_batch([0.4, 0.5, 0.6])
    assert result == [1, 0, 0]


def test_threshold_lte_below():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.LessThanEqualTo  # pyright: ignore
    )
    assert seg._label(0.4) == 1


def test_threshold_lte_equal():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.LessThanEqualTo  # pyright: ignore
    )
    assert seg._label(0.5) == 1


def test_threshold_lte_above():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.LessThanEqualTo  # pyright: ignore
    )
    assert seg._label(0.6) == 0


def test_threshold_lte_batch():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.LessThanEqualTo  # pyright: ignore
    )
    result = seg._label_batch([0.4, 0.5, 0.6])
    assert result == [1, 1, 0]


def test_threshold_seg_type():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.GreaterThan  # pyright: ignore
    )
    assert seg._seg_type == BiasSegmentationType.Threshold


def test_threshold_thres_type():
    seg = ThresholdBiasSegmentation(
        0.5, BiasSegmentationThresholdType.LessThan  # pyright: ignore
    )
    assert seg._thres_type == BiasSegmentationThresholdType.LessThan


def test_factory_label_returns_labeled():
    seg = bias_segmentation_criteria_factory("male", BiasSegmentationType.Label)
    assert isinstance(seg, LabeledBiasSegmentation)


def test_factory_threshold_returns_threshold():
    seg = bias_segmentation_criteria_factory(
        0.5, BiasSegmentationType.Threshold, BiasSegmentationThresholdType.GreaterThan
    )
    assert isinstance(seg, ThresholdBiasSegmentation)


def test_factory_accepts_string_seg_type():
    seg = bias_segmentation_criteria_factory("male", "Label")
    assert isinstance(seg, LabeledBiasSegmentation)


def test_factory_threshold_accepts_string_threshold_type():
    seg = bias_segmentation_criteria_factory(0.5, "Threshold", "GreaterThan")
    assert isinstance(seg, ThresholdBiasSegmentation)


def test_factory_label_with_threshold_type_raises():
    with pytest.raises(InvalidBiasSegmentationConfig):
        bias_segmentation_criteria_factory(
            "male",
            BiasSegmentationType.Label,
            BiasSegmentationThresholdType.GreaterThan,
        )


def test_factory_threshold_without_threshold_type_raises():
    with pytest.raises(InvalidBiasSegmentationConfig):
        bias_segmentation_criteria_factory(0.5, BiasSegmentationType.Threshold)


def test_factory_invalid_segmentation_type_raises():
    with pytest.raises(ValueError):
        bias_segmentation_criteria_factory("male", "Invalid")


def test_payload_init_valid():
    seg = LabeledBiasSegmentation("male")  # pyright: ignore
    payload = BiasDataPayload(["male", "female", "male"], seg)  # pyright: ignore
    assert payload._payload == ["male", "female", "male"]  # pyright: ignore


def test_payload_type_mismatch_raises():
    seg = LabeledBiasSegmentation("male")  # pyright: ignore
    with pytest.raises(BiasPayloadTypeException):
        BiasDataPayload([1, 2, 3], seg)  # pyright: ignore


def test_payload_empty_raises():
    seg = LabeledBiasSegmentation("male")  # pyright: ignore
    with pytest.raises(EmptyDatasetException):
        BiasDataPayload([], seg)


def test_payload_non_uniform_raises():
    seg = LabeledBiasSegmentation("male")  # pyright: ignore
    with pytest.raises(NonUniformTypeException):
        BiasDataPayload(["male", 1, "female"], seg)  # pyright: ignore


def test_payload_factory_label_valid():
    payload = BiasDataPayload.factory(
        ["male", "female", "male"], "male", BiasSegmentationType.Label
    )
    assert isinstance(payload._seg_criteria, LabeledBiasSegmentation)  # pyright: ignore


def test_payload_factory_threshold_valid():
    payload = BiasDataPayload.factory(
        [0.1, 0.5, 0.9],
        0.5,
        BiasSegmentationType.Threshold,
        BiasSegmentationThresholdType.GreaterThan,
    )
    assert isinstance(
        payload._seg_criteria, ThresholdBiasSegmentation  # pyright: ignore
    )


def test_payload_factory_type_mismatch_raises():
    with pytest.raises(BiasPayloadTypeException):
        BiasDataPayload.factory([1, 2, 3], "male", BiasSegmentationType.Label)


def test_explicit_args_threshold():
    payload = BiasDataPayload.factory(
        [0.1, 0.5, 0.9],
        0.5,
        BiasSegmentationType.Threshold,
        BiasSegmentationThresholdType.GreaterThan,
    )
    args = _construct_explicit_bias_args(payload)
    assert args.threshold == 0.5
    assert args.label is None
    assert args.threshold_type == BiasSegmentationThresholdType.GreaterThan
    assert len(args.data) == 3


def test_explicit_args_label():
    payload = BiasDataPayload.factory(
        ["male", "female", "male"], "male", BiasSegmentationType.Label
    )
    args = _construct_explicit_bias_args(payload)
    assert args.label == "male"
    assert args.threshold is None
    assert args.threshold_type is None
    assert len(args.data) == 3
