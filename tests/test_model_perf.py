import numpy as np
import pytest
from fair_perf_ml.model_perf import (BinaryClassificationStreaming,
                                     LinearRegressionStreaming,
                                     LogisticRegressionStreaming,
                                     binary_classification_analysis,
                                     linear_regression_analysis,
                                     logistic_regression_analysis,
                                     partial_runtime_check, runtime_check_full)
from fair_perf_ml.model_perf.core import DifferentModelTypes
from fair_perf_ml.models import (ClassificationEvaluationMetric,
                                 LinearRegressionEvaluationMetric, ModelType)

# ---------------------------------------------------------------------------
# Reference data — taken from Rust core tests in statistics.rs
# ---------------------------------------------------------------------------
LR_TRUE = [11.0, 12.5, 14.0, 11.7, 15.1, 15.4, 13.2, 11.5, 11.6]
LR_PRED = [11.1, 12.2, 13.4, 10.7, 15.8, 16.3, 14.5, 12.3, 11.0]

# Poor predictions → RMSE / MAE >> 10% above baseline
LR_PRED_BAD = [14.0, 9.0, 17.0, 8.0, 19.0, 10.0, 17.0, 9.0, 15.0]

BC_TRUE = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
BC_PRED = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
# All-wrong predictions → accuracy ≈ 0, clearly > 10% drift
BC_PRED_BAD = [1 - x for x in BC_TRUE]

LOG_TRUE = [
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    1.0,
    0.0,
    1.0,
    1.0,
]
LOG_PRED = [
    0.7,
    0.3,
    0.65,
    0.55,
    0.1,
    0.2,
    0.25,
    0.66,
    0.12,
    0.98,
    0.23,
    0.34,
    0.67,
    0.77,
    0.45,
    0.88,
]
# Inverted probabilities → accuracy near-zero, log-loss much higher
LOG_PRED_BAD = [1.0 - p for p in LOG_PRED]

LR_METRICS = set(LinearRegressionEvaluationMetric.__members__)
BC_METRICS = {
    "BalancedAccuracy",
    "PrecisionPositive",
    "PrecisionNegative",
    "RecallPositive",
    "RecallNegative",
    "Accuracy",
    "F1Score",
}
LOG_METRICS = BC_METRICS | {"LogLoss"}


# ---------------------------------------------------------------------------
# linear_regression_analysis
# ---------------------------------------------------------------------------


def test_lr_analysis_returns_eight_metrics():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    assert set(report.performance_data.__class__.model_fields) == LR_METRICS


def test_lr_analysis_model_type():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    assert report.model_type == ModelType.LinearRegression


def test_lr_analysis_rmse():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    assert report.performance_data.RootMeanSquaredError == pytest.approx(
        0.77817, rel=1e-3
    )


def test_lr_analysis_mse():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    assert report.performance_data.MeanSquaredError == pytest.approx(0.60556, rel=1e-3)


def test_lr_analysis_mae():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    assert report.performance_data.MeanAbsoluteError == pytest.approx(0.7000, rel=1e-3)


def test_lr_analysis_r2():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    assert report.performance_data.RSquared == pytest.approx(0.74352, rel=1e-3)


def test_lr_analysis_max_error():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    assert report.performance_data.MaxError == pytest.approx(1.3, rel=1e-3)


def test_lr_analysis_rmsle():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    assert report.performance_data.RootMeanSquaredLogError == pytest.approx(
        0.055311, rel=1e-3
    )


def test_lr_analysis_mape():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    assert report.performance_data.MeanAbsolutePercentageError == pytest.approx(
        0.05399, rel=1e-3
    )


def test_lr_analysis_numpy_input():
    report = linear_regression_analysis(np.array(LR_TRUE), np.array(LR_PRED))
    assert report.performance_data.RootMeanSquaredError == pytest.approx(
        0.77817, rel=1e-3
    )


# ---------------------------------------------------------------------------
# binary_classification_analysis
# ---------------------------------------------------------------------------


def test_bc_analysis_returns_seven_metrics():
    report = binary_classification_analysis(BC_TRUE, BC_PRED)
    assert set(report.performance_data.__class__.model_fields) == BC_METRICS


def test_bc_analysis_model_type():
    report = binary_classification_analysis(BC_TRUE, BC_PRED)
    assert report.model_type == ModelType.BinaryClassification


def test_bc_analysis_accuracy():
    report = binary_classification_analysis(BC_TRUE, BC_PRED)
    assert report.performance_data.Accuracy == pytest.approx(0.6875, rel=1e-4)


def test_bc_analysis_precision():
    report = binary_classification_analysis(BC_TRUE, BC_PRED)
    assert report.performance_data.PrecisionPositive == pytest.approx(0.75, rel=1e-4)


def test_bc_analysis_recall():
    report = binary_classification_analysis(BC_TRUE, BC_PRED)
    assert report.performance_data.RecallPositive == pytest.approx(2 / 3, rel=1e-4)


def test_bc_analysis_f1():
    report = binary_classification_analysis(BC_TRUE, BC_PRED)
    assert report.performance_data.F1Score == pytest.approx(0.70588, rel=1e-3)


def test_bc_analysis_balanced_accuracy():
    report = binary_classification_analysis(BC_TRUE, BC_PRED)
    assert report.performance_data.BalancedAccuracy == pytest.approx(0.69048, rel=1e-3)


# ---------------------------------------------------------------------------
# logistic_regression_analysis
# ---------------------------------------------------------------------------


def test_log_analysis_returns_eight_metrics():
    report = logistic_regression_analysis(LOG_TRUE, LOG_PRED)
    assert set(report.performance_data.__class__.model_fields) == LOG_METRICS


def test_log_analysis_model_type():
    report = logistic_regression_analysis(LOG_TRUE, LOG_PRED)
    assert report.model_type == ModelType.LogisticRegression


def test_log_analysis_accuracy():
    report = logistic_regression_analysis(LOG_TRUE, LOG_PRED)
    assert report.performance_data.Accuracy == pytest.approx(0.6875, rel=1e-4)


def test_log_analysis_log_loss():
    report = logistic_regression_analysis(LOG_TRUE, LOG_PRED)
    assert report.performance_data.LogLoss == pytest.approx(0.71450, rel=1e-3)


def test_log_analysis_custom_threshold():
    # With threshold=0.9, only index 9 (0.98) is predicted positive
    # ground truth positive at index 9 → TP=1, FP=0 → precision=1.0
    report = logistic_regression_analysis(LOG_TRUE, LOG_PRED, decision_threshold=0.9)
    assert report.performance_data.PrecisionPositive == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# runtime_check_full
# ---------------------------------------------------------------------------


def test_runtime_check_full_lr_same_data_passes():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    drift = runtime_check_full(report, report)
    assert drift["passed"] is True


def test_runtime_check_full_bc_same_data_passes():
    report = binary_classification_analysis(BC_TRUE, BC_PRED)
    drift = runtime_check_full(report, report)
    assert drift["passed"] is True


def test_runtime_check_full_log_same_data_passes():
    report = logistic_regression_analysis(LOG_TRUE, LOG_PRED)
    drift = runtime_check_full(report, report)
    assert drift["passed"] is True


def test_runtime_check_full_lr_drifted_fails():
    baseline = linear_regression_analysis(LR_TRUE, LR_PRED)
    current = linear_regression_analysis(LR_TRUE, LR_PRED_BAD)
    drift = runtime_check_full(current, baseline)
    assert drift["passed"] is False
    assert len(drift["failed_report"]) > 0


def test_runtime_check_full_bc_drifted_fails():
    baseline = binary_classification_analysis(BC_TRUE, BC_PRED)
    current = binary_classification_analysis(BC_TRUE, BC_PRED_BAD)
    drift = runtime_check_full(current, baseline)
    assert drift["passed"] is False
    assert len(drift["failed_report"]) > 0


def test_runtime_check_full_accepts_dict():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    report_dict = report.model_dump()
    drift = runtime_check_full(report_dict, report_dict)
    assert drift["passed"] is True


def test_runtime_check_full_mismatched_types_raises():
    lr_report = linear_regression_analysis(LR_TRUE, LR_PRED)
    bc_report = binary_classification_analysis(BC_TRUE, BC_PRED)
    with pytest.raises(DifferentModelTypes):
        runtime_check_full(lr_report, bc_report)


def test_runtime_check_full_relaxed_threshold_passes():
    baseline = linear_regression_analysis(LR_TRUE, LR_PRED)
    current = linear_regression_analysis(LR_TRUE, LR_PRED_BAD)
    drift = runtime_check_full(current, baseline, threshold=100.0)
    assert drift["passed"] is True


# ---------------------------------------------------------------------------
# partial_runtime_check
# ---------------------------------------------------------------------------


def test_partial_runtime_check_lr_same_data_passes():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    drift = partial_runtime_check(
        report, report, [LinearRegressionEvaluationMetric.RootMeanSquaredError]
    )
    assert drift["passed"] is True


def test_partial_runtime_check_bc_same_data_passes():
    report = binary_classification_analysis(BC_TRUE, BC_PRED)
    drift = partial_runtime_check(
        report, report, [ClassificationEvaluationMetric.Accuracy]
    )
    assert drift["passed"] is True


def test_partial_runtime_check_lr_drifted_fails():
    baseline = linear_regression_analysis(LR_TRUE, LR_PRED)
    current = linear_regression_analysis(LR_TRUE, LR_PRED_BAD)
    drift = partial_runtime_check(
        current, baseline, [LinearRegressionEvaluationMetric.RootMeanSquaredError]
    )
    assert drift["passed"] is False
    assert any(d["metric"] == "RootMeanSquaredError" for d in drift["failed_report"])


def test_partial_runtime_check_relaxed_threshold_passes():
    baseline = linear_regression_analysis(LR_TRUE, LR_PRED)
    current = linear_regression_analysis(LR_TRUE, LR_PRED_BAD)
    drift = partial_runtime_check(
        current,
        baseline,
        [LinearRegressionEvaluationMetric.RootMeanSquaredError],
        threshold=100.0,
    )
    assert drift["passed"] is True


def test_partial_runtime_check_accepts_string_metrics():
    report = linear_regression_analysis(LR_TRUE, LR_PRED)
    drift = partial_runtime_check(
        report, report, ["RootMeanSquaredError", "MeanAbsoluteError"]
    )
    assert drift["passed"] is True


def test_partial_runtime_check_mismatched_types_raises():
    lr_report = linear_regression_analysis(LR_TRUE, LR_PRED)
    bc_report = binary_classification_analysis(BC_TRUE, BC_PRED)
    with pytest.raises(DifferentModelTypes):
        partial_runtime_check(
            lr_report,
            bc_report,
            [LinearRegressionEvaluationMetric.RootMeanSquaredError],
        )


# ---------------------------------------------------------------------------
# BinaryClassificationStreaming
# ---------------------------------------------------------------------------


def _make_bc_streaming() -> BinaryClassificationStreaming:
    return BinaryClassificationStreaming(1, BC_TRUE, BC_PRED)


def test_bc_streaming_init():
    stream = _make_bc_streaming()
    assert stream is not None


def test_bc_streaming_performance_snapshot_after_push():
    stream = _make_bc_streaming()
    stream.update_stream_batch(BC_TRUE, BC_PRED)
    snap = stream.performance_snapshot()
    assert set(snap.keys()) == BC_METRICS
    assert snap["Accuracy"] == pytest.approx(0.6875, rel=1e-4)


def test_bc_streaming_drift_report_same_data_passes():
    stream = _make_bc_streaming()
    stream.update_stream_batch(BC_TRUE, BC_PRED)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is True


def test_bc_streaming_drift_report_drifted_fails():
    stream = _make_bc_streaming()
    stream.update_stream_batch(BC_TRUE, BC_PRED_BAD)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is False
    assert len(drift["failed_report"]) > 0


def test_bc_streaming_update_stream_single():
    stream = _make_bc_streaming()
    for t, p in zip(BC_TRUE, BC_PRED):
        stream.update_stream(t, p)
    snap = stream.performance_snapshot()
    assert snap["Accuracy"] == pytest.approx(0.6875, rel=1e-4)


def test_bc_streaming_flush_clears_runtime():
    stream = _make_bc_streaming()
    stream.update_stream_batch(BC_TRUE, BC_PRED)
    stream.flush()
    stream.update_stream_batch(BC_TRUE, BC_PRED_BAD)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is False


def test_bc_streaming_reset_baseline():
    stream = _make_bc_streaming()
    stream.reset_baseline(BC_TRUE, BC_PRED_BAD)
    stream.update_stream_batch(BC_TRUE, BC_PRED_BAD)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is True


def test_bc_streaming_reset_baseline_and_label():
    stream = _make_bc_streaming()
    # Reset label to 0 (negative class becomes "positive") — same data still passes vs itself
    stream.reset_baseline_and_label(0, BC_TRUE, BC_PRED)
    stream.update_stream_batch(BC_TRUE, BC_PRED)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is True


def test_bc_streaming_drift_snapshot():
    stream = _make_bc_streaming()
    stream.update_stream_batch(BC_TRUE, BC_PRED)
    snap = stream.drift_snapshot()
    assert isinstance(snap, dict)
    assert len(snap) > 0


def test_bc_streaming_drift_report_partial_metrics_passes():
    stream = _make_bc_streaming()
    stream.update_stream_batch(BC_TRUE, BC_PRED)
    drift = stream.drift_report_partial_metrics(
        [ClassificationEvaluationMetric.Accuracy], 0.10
    )
    assert drift["passed"] is True


def test_bc_streaming_drift_report_partial_metrics_drifted():
    stream = _make_bc_streaming()
    stream.update_stream_batch(BC_TRUE, BC_PRED_BAD)
    drift = stream.drift_report_partial_metrics(
        [ClassificationEvaluationMetric.Accuracy], 0.10
    )
    assert drift["passed"] is False
    assert any(d["metric"] == "Accuracy" for d in drift["failed_report"])


def test_bc_streaming_accepts_string_labels():
    stream = BinaryClassificationStreaming(
        "yes", ["yes", "no", "yes"], ["yes", "yes", "no"]
    )
    stream.update_stream_batch(["yes", "no", "yes"], ["yes", "yes", "no"])
    snap = stream.performance_snapshot()
    assert "Accuracy" in snap


# ---------------------------------------------------------------------------
# LinearRegressionStreaming
# ---------------------------------------------------------------------------


def _make_lr_streaming() -> LinearRegressionStreaming:
    return LinearRegressionStreaming(LR_TRUE, LR_PRED)


def test_lr_streaming_init():
    stream = _make_lr_streaming()
    assert stream is not None


def test_lr_streaming_performance_snapshot_after_push():
    stream = _make_lr_streaming()
    stream.update_stream_batch(LR_TRUE, LR_PRED)
    snap = stream.performance_snapshot()
    assert set(snap.keys()) == LR_METRICS
    assert snap["RootMeanSquaredError"] == pytest.approx(0.77817, rel=1e-3)


def test_lr_streaming_drift_report_same_data_passes():
    stream = _make_lr_streaming()
    stream.update_stream_batch(LR_TRUE, LR_PRED)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is True


def test_lr_streaming_drift_report_drifted_fails():
    stream = _make_lr_streaming()
    stream.update_stream_batch(LR_TRUE, LR_PRED_BAD)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is False
    assert len(drift["failed_report"]) > 0


def test_lr_streaming_update_stream_single():
    stream = _make_lr_streaming()
    for t, p in zip(LR_TRUE, LR_PRED):
        stream.update_stream(t, p)
    snap = stream.performance_snapshot()
    assert snap["RootMeanSquaredError"] == pytest.approx(0.77817, rel=1e-3)


def test_lr_streaming_flush_clears_runtime():
    stream = _make_lr_streaming()
    stream.update_stream_batch(LR_TRUE, LR_PRED)
    stream.flush()
    stream.update_stream_batch(LR_TRUE, LR_PRED_BAD)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is False


def test_lr_streaming_reset_baseline():
    stream = _make_lr_streaming()
    stream.reset_baseline(LR_TRUE, LR_PRED_BAD)
    stream.update_stream_batch(LR_TRUE, LR_PRED_BAD)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is True


def test_lr_streaming_drift_snapshot():
    stream = _make_lr_streaming()
    stream.update_stream_batch(LR_TRUE, LR_PRED)
    snap = stream.drift_snapshot()
    assert isinstance(snap, dict)
    assert len(snap) > 0


def test_lr_streaming_drift_report_partial_metrics_passes():
    stream = _make_lr_streaming()
    stream.update_stream_batch(LR_TRUE, LR_PRED)
    drift = stream.drift_report_partial_metrics(
        [LinearRegressionEvaluationMetric.RootMeanSquaredError], 0.10
    )
    assert drift["passed"] is True


def test_lr_streaming_drift_report_partial_metrics_drifted():
    stream = _make_lr_streaming()
    stream.update_stream_batch(LR_TRUE, LR_PRED_BAD)
    drift = stream.drift_report_partial_metrics(
        [LinearRegressionEvaluationMetric.RootMeanSquaredError], 0.10
    )
    assert drift["passed"] is False
    assert any(d["metric"] == "RootMeanSquaredError" for d in drift["failed_report"])


# ---------------------------------------------------------------------------
# LogisticRegressionStreaming
# ---------------------------------------------------------------------------


def _make_log_streaming() -> LogisticRegressionStreaming:
    return LogisticRegressionStreaming(LOG_TRUE, LOG_PRED, 0.5)


def test_log_streaming_init():
    stream = _make_log_streaming()
    assert stream is not None


def test_log_streaming_performance_snapshot_after_push():
    stream = _make_log_streaming()
    stream.update_stream_batch(LOG_TRUE, LOG_PRED)
    snap = stream.performance_snapshot()
    assert set(snap.keys()) == LOG_METRICS
    assert snap["Accuracy"] == pytest.approx(0.6875, rel=1e-4)
    assert snap["LogLoss"] == pytest.approx(0.71450, rel=1e-3)


def test_log_streaming_drift_report_same_data_passes():
    stream = _make_log_streaming()
    stream.update_stream_batch(LOG_TRUE, LOG_PRED)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is True


def test_log_streaming_drift_report_drifted_fails():
    stream = _make_log_streaming()
    stream.update_stream_batch(LOG_TRUE, LOG_PRED_BAD)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is False
    assert len(drift["failed_report"]) > 0


def test_log_streaming_update_stream_single():
    stream = _make_log_streaming()
    for t, p in zip(LOG_TRUE, LOG_PRED):
        stream.update_stream(t, p)
    snap = stream.performance_snapshot()
    assert snap["Accuracy"] == pytest.approx(0.6875, rel=1e-4)


def test_log_streaming_flush_clears_runtime():
    stream = _make_log_streaming()
    stream.update_stream_batch(LOG_TRUE, LOG_PRED)
    stream.flush()
    stream.update_stream_batch(LOG_TRUE, LOG_PRED_BAD)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is False


def test_log_streaming_reset_baseline():
    stream = _make_log_streaming()
    stream.reset_baseline(LOG_TRUE, LOG_PRED_BAD)
    stream.update_stream_batch(LOG_TRUE, LOG_PRED_BAD)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is True


def test_log_streaming_reset_baseline_and_decision_threshold():
    stream = _make_log_streaming()
    stream.reset_baseline_and_decision_threshold(LOG_TRUE, LOG_PRED, 0.9)
    stream.update_stream_batch(LOG_TRUE, LOG_PRED)
    drift = stream.drift_report(0.10)
    assert drift["passed"] is True


def test_log_streaming_drift_snapshot():
    stream = _make_log_streaming()
    stream.update_stream_batch(LOG_TRUE, LOG_PRED)
    snap = stream.drift_snapshot()
    assert isinstance(snap, dict)
    assert len(snap) > 0


def test_log_streaming_drift_report_partial_metrics_passes():
    stream = _make_log_streaming()
    stream.update_stream_batch(LOG_TRUE, LOG_PRED)
    drift = stream.drift_report_partial_metrics(
        [ClassificationEvaluationMetric.Accuracy], 0.10
    )
    assert drift["passed"] is True


def test_log_streaming_drift_report_partial_metrics_drifted():
    stream = _make_log_streaming()
    stream.update_stream_batch(LOG_TRUE, LOG_PRED_BAD)
    drift = stream.drift_report_partial_metrics(
        [ClassificationEvaluationMetric.Accuracy], 0.10
    )
    assert drift["passed"] is False
    assert any(d["metric"] == "Accuracy" for d in drift["failed_report"])
