import numpy as np
import pytest
from fair_perf_ml.bias.segmentation import (BiasDataPayload,
                                            BiasSegmentationType,
                                            LabeledBiasSegmentation)
from fair_perf_ml.model_bias import (
    ModelBiasStreaming, model_bias_partial_runtime_comparison,
    model_bias_perform_analysis,
    model_bias_perform_analysis_explicit_segmentation,
    model_bias_runtime_comparison)
from fair_perf_ml.models import ModelBiasMetric

# ---------------------------------------------------------------------------
# Reference data
#
# Symmetric dataset: 4 advantaged (feat=1), 4 disadvantaged (feat=0).
# Each facet has identical pred=[1,1,0,0] and gt=[1,0,1,0].
# Confusion matrix per facet: TP=1, FP=1, FN=1, TN=1
# All difference metrics = 0, DisparateImpact = 1, DDPL = 0.
# ---------------------------------------------------------------------------
FEAT_SYM = [1, 1, 1, 1, 0, 0, 0, 0]
PRED_SYM = [1, 1, 0, 0, 1, 1, 0, 0]
GT_SYM = [1, 0, 1, 0, 1, 0, 1, 0]

ALL_METRIC_KEYS = {
    "DifferenceInPositivePredictedLabels",
    "DisparateImpact",
    "AccuracyDifference",
    "RecallDifference",
    "DifferenceInConditionalAcceptance",
    "DifferenceInAcceptanceRate",
    "SpecialityDifference",
    "DifferenceInConditionalRejection",
    "DifferenceInRejectionRate",
    "TreatmentEquity",
    "ConditionalDemographicDesparityPredictedLabels",
    "GeneralizedEntropy",
}

# ---------------------------------------------------------------------------
# Asymmetric dataset for targeted metric tests.
#
# facet_a (feat=1): pred=[1,1,1,0], gt=[1,1,0,1]
#   TP=2, FP=1, FN=1, TN=0; positive_pred=3, positive_gt=3, len=4
# facet_d (feat=0): pred=[0,1,0,0], gt=[1,0,1,0]
#   TP=0, FP=1, FN=2, TN=1; positive_pred=1, positive_gt=2, len=4
# ---------------------------------------------------------------------------
FEAT_ASYM = [1, 1, 1, 1, 0, 0, 0, 0]
PRED_ASYM = [1, 1, 1, 0, 0, 1, 0, 0]
GT_ASYM = [1, 1, 0, 1, 1, 0, 1, 0]

# Pre-computed expected values for ASYM:
_A_POS_PRED, _A_POS_GT, _A_LEN = 3, 3, 4
_D_POS_PRED, _D_POS_GT, _D_LEN = 1, 2, 4
_TP_A, _FP_A, _FN_A, _TN_A = 2, 1, 1, 0
_TP_D, _FP_D, _FN_D, _TN_D = 0, 1, 2, 1

EXPECTED_ASYM = {
    "DifferenceInPositivePredictedLabels": (_A_POS_PRED / _A_POS_GT)
    - (_D_POS_PRED / _D_POS_GT),
    "DisparateImpact": (_A_POS_PRED / _A_LEN) / (_D_POS_PRED / _D_LEN),
    "AccuracyDifference": (_TP_A + _TN_A) / _A_LEN - (_TP_D + _TN_D) / _D_LEN,
    "RecallDifference": _TP_A / (_TP_A + _FN_A) - _TP_D / (_TP_D + _FN_D),
    "DifferenceInConditionalAcceptance": _A_POS_PRED / _A_POS_GT
    - _D_POS_PRED / _D_POS_GT,
    "DifferenceInAcceptanceRate": _TP_A / (_TP_A + _FP_A) - _TP_D / (_TP_D + _FP_D),
    "SpecialityDifference": _TN_A / (_TN_A + _FP_A) - _TN_D / (_TN_D + _FP_D),
    "DifferenceInConditionalRejection": (_D_LEN - _D_POS_GT) / (_D_LEN - _D_POS_PRED)
    - (_A_LEN - _A_POS_GT) / (_A_LEN - _A_POS_PRED),
    "DifferenceInRejectionRate": _TN_D / (_TN_D + _FN_D) - _TN_A / (_TN_A + _FN_A),
    "TreatmentEquity": _FN_D / _FP_D - _FN_A / _FP_A,
}


# ---------------------------------------------------------------------------
# model_bias_perform_analysis (simple label-based segmentation)
# ---------------------------------------------------------------------------


def test_perform_analysis_returns_all_metrics():
    result = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    assert set(result.keys()) == ALL_METRIC_KEYS


def test_perform_analysis_sym_ddpl_zero():
    result = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    assert result["DifferenceInPositivePredictedLabels"] == pytest.approx(0.0, abs=1e-5)


def test_perform_analysis_sym_disparate_impact_one():
    result = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    assert result["DisparateImpact"] == pytest.approx(1.0, rel=1e-4)


def test_perform_analysis_sym_accuracy_difference_zero():
    result = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    assert result["AccuracyDifference"] == pytest.approx(0.0, abs=1e-5)


def test_perform_analysis_sym_recall_difference_zero():
    result = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    assert result["RecallDifference"] == pytest.approx(0.0, abs=1e-5)


def test_perform_analysis_asym_ddpl():
    result = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    assert result["DifferenceInPositivePredictedLabels"] == pytest.approx(
        EXPECTED_ASYM["DifferenceInPositivePredictedLabels"], rel=1e-4
    )


def test_perform_analysis_asym_disparate_impact():
    result = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    assert result["DisparateImpact"] == pytest.approx(
        EXPECTED_ASYM["DisparateImpact"], rel=1e-4
    )


def test_perform_analysis_asym_accuracy_difference():
    result = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    assert result["AccuracyDifference"] == pytest.approx(
        EXPECTED_ASYM["AccuracyDifference"], rel=1e-4
    )


def test_perform_analysis_asym_recall_difference():
    result = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    assert result["RecallDifference"] == pytest.approx(
        EXPECTED_ASYM["RecallDifference"], rel=1e-4
    )


def test_perform_analysis_asym_dca():
    result = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    assert result["DifferenceInConditionalAcceptance"] == pytest.approx(
        EXPECTED_ASYM["DifferenceInConditionalAcceptance"], rel=1e-4
    )


def test_perform_analysis_asym_treatment_equity():
    result = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    assert result["TreatmentEquity"] == pytest.approx(
        EXPECTED_ASYM["TreatmentEquity"], rel=1e-4
    )


def test_perform_analysis_numpy_input():
    result = model_bias_perform_analysis(
        np.array(FEAT_SYM), np.array(GT_SYM), np.array(PRED_SYM), 1, 1, 1
    )
    assert set(result.keys()) == ALL_METRIC_KEYS
    assert result["DisparateImpact"] == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# model_bias_perform_analysis_explicit_segmentation
# ---------------------------------------------------------------------------


def test_explicit_seg_label_returns_all_metrics():
    feat_payload = BiasDataPayload.factory(
        ["a", "a", "a", "a", "b", "b", "b", "b"],
        "a",
        BiasSegmentationType.Label,
    )
    gt_payload = BiasDataPayload.factory(
        ["y", "n", "y", "n", "y", "n", "y", "n"],
        "y",
        BiasSegmentationType.Label,
    )
    pred_payload = BiasDataPayload.factory(
        ["y", "y", "n", "n", "y", "y", "n", "n"],
        "y",
        BiasSegmentationType.Label,
    )
    result = model_bias_perform_analysis_explicit_segmentation(
        feat_payload, gt_payload, pred_payload
    )
    assert set(result.keys()) == ALL_METRIC_KEYS


def test_explicit_seg_label_sym_di_one():
    feat_payload = BiasDataPayload.factory(
        ["a", "a", "a", "a", "b", "b", "b", "b"],
        "a",
        BiasSegmentationType.Label,
    )
    gt_payload = BiasDataPayload.factory(
        ["y", "n", "y", "n", "y", "n", "y", "n"],
        "y",
        BiasSegmentationType.Label,
    )
    pred_payload = BiasDataPayload.factory(
        ["y", "y", "n", "n", "y", "y", "n", "n"],
        "y",
        BiasSegmentationType.Label,
    )
    result = model_bias_perform_analysis_explicit_segmentation(
        feat_payload, gt_payload, pred_payload
    )
    assert result["DisparateImpact"] == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# model_bias_runtime_comparison
# ---------------------------------------------------------------------------


def test_runtime_comparison_same_data_passes():
    report = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    drift = model_bias_runtime_comparison(report, report)
    assert drift["passed"] is True
    assert drift.get("failed_report", []) == []


def test_runtime_comparison_drifted_data_fails():
    baseline = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    current = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    drift = model_bias_runtime_comparison(baseline, current)
    assert drift["passed"] is False
    assert len(drift["failed_report"]) > 0


def test_runtime_comparison_drifted_data_fails_accuracy():
    baseline = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    current = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    drift = model_bias_runtime_comparison(baseline, current)
    assert any(
        d["metric"] == "AccuracyDifference"  # pyright: ignore
        for d in drift["failed_report"]
    )


def test_runtime_comparison_custom_threshold_relaxed():
    # With ASYM as both baseline and current, values are identical → always passes,
    # even with a strict threshold. Verifies the threshold parameter is accepted.
    baseline = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    current = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    drift = model_bias_runtime_comparison(baseline, current, threshold=2.0)
    assert drift["passed"] is True


# ---------------------------------------------------------------------------
# model_bias_partial_runtime_comparison
# ---------------------------------------------------------------------------


def test_partial_comparison_same_data_passes():
    report = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    drift = model_bias_partial_runtime_comparison(
        report,
        report,
        [ModelBiasMetric.DisparateImpact, ModelBiasMetric.AccuracyDifference],
    )
    assert drift["passed"] is True


def test_partial_comparison_drifted_fails_on_checked_metric():
    baseline = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    current = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    drift = model_bias_partial_runtime_comparison(
        baseline,
        current,
        [ModelBiasMetric.AccuracyDifference],
    )
    assert drift["passed"] is False
    assert any(
        d["metric"] == "AccuracyDifference"  # pyright: ignore
        for d in drift["failed_report"]
    )


def test_partial_comparison_only_checked_metric_in_failed_report():
    baseline = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    current = model_bias_perform_analysis(FEAT_ASYM, GT_ASYM, PRED_ASYM, 1, 1, 1)
    drift = model_bias_partial_runtime_comparison(
        baseline,
        current,
        [ModelBiasMetric.DisparateImpact],
    )
    if not drift["passed"]:
        assert all(
            d["metric"] == "DisparateImpact"  # pyright: ignore
            for d in drift["failed_report"]
        )


def test_partial_comparison_accepts_string_metrics():
    report = model_bias_perform_analysis(FEAT_SYM, GT_SYM, PRED_SYM, 1, 1, 1)
    drift = model_bias_partial_runtime_comparison(
        report, report, ["DisparateImpact", "AccuracyDifference"]
    )
    assert drift["passed"] is True


# ---------------------------------------------------------------------------
# ModelBiasStreaming
# ---------------------------------------------------------------------------


def _make_streaming() -> ModelBiasStreaming:
    feat_seg = LabeledBiasSegmentation(1)  # pyright: ignore
    gt_seg = LabeledBiasSegmentation(1)  # pyright: ignore
    pred_seg = LabeledBiasSegmentation(1)  # pyright: ignore
    return ModelBiasStreaming(feat_seg, gt_seg, pred_seg, FEAT_SYM, GT_SYM, PRED_SYM)


def test_streaming_init():
    stream = _make_streaming()
    assert stream is not None


def test_streaming_performance_snapshot_after_push_batch():
    stream = _make_streaming()
    stream.push_batch(FEAT_SYM, GT_SYM, PRED_SYM)
    snap = stream.performance_snapshot()
    assert set(snap.keys()) == ALL_METRIC_KEYS
    assert snap["DisparateImpact"] == pytest.approx(1.0, rel=1e-4)
    assert snap["AccuracyDifference"] == pytest.approx(0.0, abs=1e-5)


def test_streaming_drift_report_same_data_passes():
    stream = _make_streaming()
    stream.push_batch(FEAT_SYM, GT_SYM, PRED_SYM)
    drift = stream.drift_report()
    assert drift["passed"] is True


def test_streaming_drift_report_drifted_data_fails():
    stream = _make_streaming()
    stream.push_batch(FEAT_ASYM, GT_ASYM, PRED_ASYM)
    drift = stream.drift_report()
    assert drift["passed"] is False
    assert len(drift["failed_report"]) > 0


def test_streaming_push_single():
    stream = _make_streaming()
    for f, g, p in zip(FEAT_SYM, GT_SYM, PRED_SYM):
        stream.push(f, p, g)
    snap = stream.performance_snapshot()
    assert snap["DisparateImpact"] == pytest.approx(1.0, rel=1e-4)


def test_streaming_flush_clears_runtime():
    stream = _make_streaming()
    stream.push_batch(FEAT_SYM, GT_SYM, PRED_SYM)
    stream.flush()
    stream.push_batch(FEAT_ASYM, GT_ASYM, PRED_ASYM)
    drift = stream.drift_report()
    assert drift["passed"] is False


def test_streaming_reset_baseline():
    stream = _make_streaming()
    stream.reset_baseline(FEAT_ASYM, GT_ASYM, PRED_ASYM)
    stream.push_batch(FEAT_ASYM, GT_ASYM, PRED_ASYM)
    drift = stream.drift_report()
    assert drift["passed"] is True


def test_streaming_drift_report_partial_metrics_same_passes():
    stream = _make_streaming()
    stream.push_batch(FEAT_SYM, GT_SYM, PRED_SYM)
    drift = stream.drift_report_partial_metrics([ModelBiasMetric.DisparateImpact])
    assert drift["passed"] is True


def test_streaming_drift_report_partial_metrics_drifted_fails():
    stream = _make_streaming()
    stream.push_batch(FEAT_ASYM, GT_ASYM, PRED_ASYM)
    drift = stream.drift_report_partial_metrics([ModelBiasMetric.AccuracyDifference])
    assert drift["passed"] is False
    assert any(
        d["metric"] == "AccuracyDifference"  # pyright: ignore
        for d in drift["failed_report"]
    )
