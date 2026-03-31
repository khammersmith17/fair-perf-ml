import math

import numpy as np
import pytest
from fair_perf_ml.bias.segmentation import (BiasDataPayload,
                                            BiasSegmentationThresholdType,
                                            BiasSegmentationType,
                                            LabeledBiasSegmentation,
                                            ThresholdBiasSegmentation)
from fair_perf_ml.data_bias import (
    DataBiasStreaming, data_bias_partial_runtime_comparison,
    data_bias_perform_analysis,
    data_bias_perform_analysis_explicit_segmentation,
    data_bias_runtime_comparison)
from fair_perf_ml.models import DataBiasMetric

# ---------------------------------------------------------------------------
# Reference data — taken directly from the Rust core tests in statistics.rs
# feature=1 → advantaged (facet_a), feature=0 → disadvantaged (facet_d)
# gt=1 → positive outcome
# ---------------------------------------------------------------------------
FEATURE = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
GT = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1]
# facet_a: len=10, positive=7 → acceptance=0.7
# facet_d: len=10, positive=2 → acceptance=0.2
_A = 7 / 10
_D = 2 / 10


def _kl_bin(p: float, q: float) -> float:
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


EXPECTED = {
    "ClassImbalance": 0.0,
    "DifferenceInProportionOfLabels": _A - _D,
    "KlDivergence": _kl_bin(_A, _D),
    "JsDivergence": 0.5 * (_kl_bin(_A, 0.5 * (_A + _D)) + _kl_bin(_D, 0.5 * (_A + _D))),
    "LpNorm": math.sqrt((_A - _D) ** 2 + ((1 - _A) - (1 - _D)) ** 2),
    "TotalVariationDistance": 0.5 * (abs(_A - _D) + abs((1 - _A) - (1 - _D))),
    "KolmogorovSmirnov": 0.5,
}

# Dataset where facet_a acceptance=0.8, facet_d=0.1 → DPL=0.7 (> baseline |DPL|*1.1=0.55)
FEATURE_DRIFT = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
GT_DRIFT = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# facet_a (feature=1, indices 0-9):  gt = 1,1,1,1,1,1,1,1,0,0 → positive=8, acceptance=0.8
# facet_d (feature=0, indices 10-19): gt = 1,0,0,0,0,0,0,0,0,0 → positive=1, acceptance=0.1


# ---------------------------------------------------------------------------
# data_bias_perform_analysis (simple label-based segmentation)
# ---------------------------------------------------------------------------


def test_perform_analysis_returns_all_metrics():
    result = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    assert set(result.keys()) == set(EXPECTED.keys())


def test_perform_analysis_ci_balanced():
    result = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    assert result["ClassImbalance"] == pytest.approx(0.0, abs=1e-5)


def test_perform_analysis_ci_imbalanced():
    # 21-element dataset: 11 ones → CI = 1/21
    feat_21 = FEATURE + [1]
    gt_21 = GT + [1]
    result = data_bias_perform_analysis(feat_21, gt_21, 1, 1)
    assert result["ClassImbalance"] == pytest.approx(1 / 21, rel=1e-4)


def test_perform_analysis_dpl():
    result = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    assert result["DifferenceInProportionOfLabels"] == pytest.approx(
        EXPECTED["DifferenceInProportionOfLabels"], rel=1e-4
    )


def test_perform_analysis_kl():
    result = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    assert result["KlDivergence"] == pytest.approx(EXPECTED["KlDivergence"], rel=1e-4)


def test_perform_analysis_js():
    result = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    assert result["JsDivergence"] == pytest.approx(EXPECTED["JsDivergence"], rel=1e-4)


def test_perform_analysis_lp_norm():
    result = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    assert result["LpNorm"] == pytest.approx(EXPECTED["LpNorm"], rel=1e-4)


def test_perform_analysis_tvd():
    result = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    assert result["TotalVariationDistance"] == pytest.approx(
        EXPECTED["TotalVariationDistance"], rel=1e-4
    )


def test_perform_analysis_ks():
    result = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    assert result["KolmogorovSmirnov"] == pytest.approx(0.5, rel=1e-4)


def test_perform_analysis_numpy_input():
    result = data_bias_perform_analysis(np.array(FEATURE), np.array(GT), 1, 1)
    assert result["KolmogorovSmirnov"] == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# data_bias_perform_analysis_explicit_segmentation
# ---------------------------------------------------------------------------


def test_explicit_seg_label_returns_all_metrics():
    feat_payload = BiasDataPayload.factory(
        ["male", "female", "male", "male", "female", "male"],
        "male",
        BiasSegmentationType.Label,
    )
    gt_payload = BiasDataPayload.factory(
        ["yes", "no", "yes", "no", "yes", "no"],
        "yes",
        BiasSegmentationType.Label,
    )
    result = data_bias_perform_analysis_explicit_segmentation(feat_payload, gt_payload)
    assert set(result.keys()) == set(EXPECTED.keys())


def test_explicit_seg_label_dpl():
    # facet_a ("male"): indices 0,2,3,5 → len=4, gt=[yes,yes,no,no] → positive=2, acc=0.5
    # facet_d ("female"): indices 1,4 → len=2, gt=[no,yes] → positive=1, acc=0.5
    feat_payload = BiasDataPayload.factory(
        ["male", "female", "male", "male", "female", "male"],
        "male",
        BiasSegmentationType.Label,
    )
    gt_payload = BiasDataPayload.factory(
        ["yes", "no", "yes", "no", "yes", "no"],
        "yes",
        BiasSegmentationType.Label,
    )
    result = data_bias_perform_analysis_explicit_segmentation(feat_payload, gt_payload)
    assert result["DifferenceInProportionOfLabels"] == pytest.approx(
        0.5 - 0.5, abs=1e-5
    )


def test_explicit_seg_threshold_gte_dpl():
    # facet_a (feat >= 0.5): [0.7, 0.8, 0.6] indices 1,3,4 → gt=[0.3,0.4,0.8] → positive=1 (0.8>=0.5), acc=1/3
    # facet_d (feat < 0.5):  [0.3, 0.2, 0.4] indices 0,2,5 → gt=[0.6,0.7,0.2] → positive=2 (0.6,0.7>=0.5), acc=2/3
    feature_vals = [0.3, 0.7, 0.2, 0.8, 0.6, 0.4]
    gt_vals = [0.6, 0.3, 0.7, 0.4, 0.8, 0.2]
    feat_payload = BiasDataPayload.factory(
        feature_vals,
        0.5,
        BiasSegmentationType.Threshold,
        BiasSegmentationThresholdType.GreaterThanEqualTo,
    )
    gt_payload = BiasDataPayload.factory(
        gt_vals,
        0.5,
        BiasSegmentationType.Threshold,
        BiasSegmentationThresholdType.GreaterThanEqualTo,
    )
    result = data_bias_perform_analysis_explicit_segmentation(feat_payload, gt_payload)
    assert result["DifferenceInProportionOfLabels"] == pytest.approx(
        1 / 3 - 2 / 3, rel=1e-4
    )


# ---------------------------------------------------------------------------
# data_bias_runtime_comparison
# ---------------------------------------------------------------------------


def test_runtime_comparison_same_data_passes():
    report = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    drift = data_bias_runtime_comparison(report, report)
    assert drift["passed"] is True
    assert drift.get("failed_report", {}) == {}


def test_runtime_comparison_drifted_data_fails():
    baseline = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    current = data_bias_perform_analysis(FEATURE_DRIFT, GT_DRIFT, 1, 1)
    drift = data_bias_runtime_comparison(baseline, current)
    assert drift["passed"] is False
    assert len(drift["failed_report"]) > 0


def test_runtime_comparison_drifted_data_fails_dpl():
    baseline = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    current = data_bias_perform_analysis(FEATURE_DRIFT, GT_DRIFT, 1, 1)
    drift = data_bias_runtime_comparison(baseline, current)
    assert any(
        d["metric"] == "DifferenceInProportionOfLabels" for d in drift["failed_report"]
    )


def test_runtime_comparison_custom_threshold_relaxed():
    # Use a very wide threshold so even different data passes
    baseline = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    current = data_bias_perform_analysis(FEATURE_DRIFT, GT_DRIFT, 1, 1)
    drift = data_bias_runtime_comparison(baseline, current, threshold=2.0)
    print(drift)
    assert drift["passed"] is True


# ---------------------------------------------------------------------------
# data_bias_partial_runtime_comparison
# ---------------------------------------------------------------------------


def test_partial_comparison_same_data_passes():
    report = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    drift = data_bias_partial_runtime_comparison(
        report,
        report,
        [DataBiasMetric.ClassImbalance, DataBiasMetric.KolmogorovSmirnov],
    )
    assert drift["passed"] is True


def test_partial_comparison_drifted_only_checked_metrics():
    baseline = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    current = data_bias_perform_analysis(FEATURE_DRIFT, GT_DRIFT, 1, 1)
    # Only check ClassImbalance — both datasets are balanced (10/10), so should pass
    drift = data_bias_partial_runtime_comparison(
        baseline, current, [DataBiasMetric.ClassImbalance]
    )
    assert drift["passed"] is True


def test_partial_comparison_drifted_fails_on_checked_metric():
    baseline = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    current = data_bias_perform_analysis(FEATURE_DRIFT, GT_DRIFT, 1, 1)
    drift = data_bias_partial_runtime_comparison(
        baseline, current, [DataBiasMetric.DifferenceInProportionOfLabels]
    )
    assert drift["passed"] is False
    assert any(
        d["metric"] == "DifferenceInProportionOfLabels" for d in drift["failed_report"]
    )


def test_partial_comparison_accepts_string_metrics():
    report = data_bias_perform_analysis(FEATURE, GT, 1, 1)
    drift = data_bias_partial_runtime_comparison(
        report, report, ["ClassImbalance", "KolmogorovSmirnov"]
    )
    assert drift["passed"] is True


# ---------------------------------------------------------------------------
# DataBiasStreaming
# ---------------------------------------------------------------------------


def _make_streaming() -> DataBiasStreaming:
    feat_seg = LabeledBiasSegmentation(1)
    gt_seg = LabeledBiasSegmentation(1)
    return DataBiasStreaming(feat_seg, gt_seg, FEATURE, GT)


def test_streaming_init():
    stream = _make_streaming()
    assert stream is not None


def test_streaming_performance_snapshot_after_push_batch():
    stream = _make_streaming()
    stream.push_batch(FEATURE, GT)
    snap = stream.performance_snapshot()
    assert set(snap.keys()) == set(EXPECTED.keys())
    assert snap["KolmogorovSmirnov"] == pytest.approx(0.5, rel=1e-4)
    assert snap["DifferenceInProportionOfLabels"] == pytest.approx(0.5, rel=1e-4)


def test_streaming_drift_report_same_data_passes():
    stream = _make_streaming()
    stream.push_batch(FEATURE, GT)
    drift = stream.drift_report()
    assert drift["passed"] is True


def test_streaming_drift_report_drifted_data_fails():
    stream = _make_streaming()
    stream.push_batch(FEATURE_DRIFT, GT_DRIFT)
    drift = stream.drift_report()
    assert drift["passed"] is False
    assert len(drift["failed_report"]) > 0


def test_streaming_push_single():
    stream = _make_streaming()
    # Push items one by one to match push_batch result
    for f, g in zip(FEATURE, GT):
        stream.push(f, g)
    snap = stream.performance_snapshot()
    assert snap["KolmogorovSmirnov"] == pytest.approx(0.5, rel=1e-4)


def test_streaming_flush_clears_runtime():
    stream = _make_streaming()
    stream.push_batch(FEATURE, GT)
    stream.flush()
    # After flush, pushing drifted data should reflect only drifted data
    stream.push_batch(FEATURE_DRIFT, GT_DRIFT)
    drift = stream.drift_report()
    assert drift["passed"] is False


def test_streaming_reset_baseline():
    stream = _make_streaming()
    # Reset baseline to drifted data, then push the same drifted data as runtime
    stream.reset_baseline(FEATURE_DRIFT, GT_DRIFT)
    stream.push_batch(FEATURE_DRIFT, GT_DRIFT)
    drift = stream.drift_report()
    assert drift["passed"] is True


def test_streaming_drift_report_partial_metrics_passes():
    stream = _make_streaming()
    stream.push_batch(FEATURE, GT)
    drift = stream.drift_report_partial_metrics([DataBiasMetric.ClassImbalance])
    assert drift["passed"] is True


def test_streaming_drift_report_partial_metrics_drifted():
    stream = _make_streaming()
    stream.push_batch(FEATURE_DRIFT, GT_DRIFT)
    drift = stream.drift_report_partial_metrics(
        [DataBiasMetric.DifferenceInProportionOfLabels]
    )
    assert drift["passed"] is False
    assert "DifferenceInProportionOfLabels" in (
        d["metric"] for d in drift["failed_report"]
    )
