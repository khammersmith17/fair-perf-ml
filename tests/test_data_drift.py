import numpy as np
import pytest
from fair_perf_ml.drift import (CategoricalDataDrift, ContinuousDataDrift,
                                DataDriftType, QuantileType,
                                compute_drift_categorical_distribution,
                                compute_drift_continuous_distribution)
from fair_perf_ml.drift.streaming import (StreamingCategoricalDataDriftDecay,
                                          StreamingCategoricalDataDriftFlush,
                                          StreamingContinuousDataDriftDecay,
                                          StreamingContinuousDataDriftFlush)

# ---------------------------------------------------------------------------
# Reference data — mirrored from Rust core tests in drift/mod.rs
# ---------------------------------------------------------------------------
CONT_BASELINE = [1.0, 2.0, 3.0, 4.0, 5.0]
CONT_SAME = [1.0, 2.0, 3.0, 4.0, 5.0]
CONT_SHIFTED = [20.0, 21.0, 22.0, 23.0, 24.0]  # completely outside baseline range

CAT_BASELINE = ["a", "b", "a", "c", "b", "a"]
CAT_SAME = ["a", "b", "a", "c", "b", "a"]
CAT_SHIFTED = ["c", "c", "c", "c", "c", "c"]  # all mass on single label

# Large datasets for decay mode — apply_decay() calls floor() on bin counts, which
# zeroes any bin with count < 1.  Small baselines (≤5 items) have bins with count=1,
# so the first compute_drift call zeros them and produces spuriously large PSI.
CONT_DECAY_BASELINE = [float(i) for i in range(1, 101)]   # 100 items, ~12/bin
CONT_DECAY_SAME = [float(i) for i in range(1, 101)]
CONT_DECAY_SHIFTED = [float(i + 200) for i in range(1, 101)]  # outside baseline range
CAT_DECAY_BASELINE = ["a"] * 50 + ["b"] * 50 + ["c"] * 50  # 150 items, 50/category
CAT_DECAY_SAME = ["a"] * 50 + ["b"] * 50 + ["c"] * 50

ALL_METRICS = [
    DataDriftType.JensenShannon,
    DataDriftType.PopulationStabilityIndex,
    DataDriftType.WassersteinDistance,
    DataDriftType.KullbackLeibler,
]


# ---------------------------------------------------------------------------
# compute_drift_continuous_distribution (batch)
# ---------------------------------------------------------------------------


def test_cont_batch_same_distribution_near_zero():
    result = compute_drift_continuous_distribution(
        CONT_BASELINE, CONT_SAME, [DataDriftType.PopulationStabilityIndex], None
    )
    assert len(result) == 1
    assert result[0] == pytest.approx(0.0, abs=1e-9)


def test_cont_batch_shifted_distribution_detects_drift():
    result = compute_drift_continuous_distribution(
        CONT_BASELINE, CONT_SHIFTED, [DataDriftType.PopulationStabilityIndex], None
    )
    assert result[0] > 0.5


def test_cont_batch_all_metrics_returns_correct_length():
    result = compute_drift_continuous_distribution(
        CONT_BASELINE, CONT_SAME, ALL_METRICS, None
    )
    assert len(result) == len(ALL_METRICS)
    for score in result:
        assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_batch_string_metric_accepted():
    result = compute_drift_continuous_distribution(
        CONT_BASELINE, CONT_SAME, ["JensenShannon"], None
    )
    assert len(result) == 1
    assert result[0] == pytest.approx(0.0, abs=1e-9)


def test_cont_batch_explicit_quantile_type():
    result = compute_drift_continuous_distribution(
        CONT_BASELINE, CONT_SAME, [DataDriftType.JensenShannon], QuantileType.Sturges
    )
    assert len(result) == 1
    assert result[0] == pytest.approx(0.0, abs=1e-9)


def test_cont_batch_numpy_input():
    result = compute_drift_continuous_distribution(
        np.array(CONT_BASELINE),
        np.array(CONT_SAME),
        [DataDriftType.JensenShannon],
        None,
    )
    assert result[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_drift_categorical_distribution (batch)
# ---------------------------------------------------------------------------


def test_cat_batch_same_distribution_near_zero():
    result = compute_drift_categorical_distribution(
        CAT_BASELINE, CAT_SAME, [DataDriftType.PopulationStabilityIndex]
    )
    assert len(result) == 1
    assert result[0] == pytest.approx(0.0, abs=1e-9)


def test_cat_batch_shifted_distribution_detects_drift():
    result = compute_drift_categorical_distribution(
        ["a", "a", "a", "a", "b"],
        ["b", "b", "b", "b", "a"],
        [DataDriftType.PopulationStabilityIndex],
    )
    assert result[0] > 0.1


def test_cat_batch_all_metrics_returns_correct_length():
    result = compute_drift_categorical_distribution(CAT_BASELINE, CAT_SAME, ALL_METRICS)
    assert len(result) == len(ALL_METRICS)
    for score in result:
        assert score == pytest.approx(0.0, abs=1e-9)


def test_cat_batch_unseen_label_handled():
    result = compute_drift_categorical_distribution(
        ["a", "b", "a", "b"],
        ["a", "b", "c", "d"],  # c and d unseen in baseline
        [DataDriftType.JensenShannon],
    )
    assert len(result) == 1
    assert result[0] >= 0.0


def test_cat_batch_string_metric_accepted():
    result = compute_drift_categorical_distribution(
        CAT_BASELINE, CAT_SAME, ["KullbackLeibler"]
    )
    assert len(result) == 1
    assert result[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# ContinuousDataDrift (stateful batch class)
# ---------------------------------------------------------------------------


def test_cont_drift_init():
    d = ContinuousDataDrift(CONT_BASELINE)
    assert d is not None
    assert d.num_bins > 0


def test_cont_drift_same_data_near_zero():
    d = ContinuousDataDrift(CONT_BASELINE)
    score = d.compute_drift(CONT_SAME, DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_drift_shifted_detects_drift():
    d = ContinuousDataDrift(CONT_BASELINE)
    score = d.compute_drift(CONT_SHIFTED, DataDriftType.PopulationStabilityIndex)
    assert score > 0.5


def test_cont_drift_multiple_criteria_correct_length():
    d = ContinuousDataDrift(CONT_BASELINE)
    scores = d.compute_drift_multiple_criteria(CONT_SAME, ALL_METRICS)
    assert len(scores) == len(ALL_METRICS)


def test_cont_drift_multiple_criteria_same_data_near_zero():
    d = ContinuousDataDrift(CONT_BASELINE)
    scores = d.compute_drift_multiple_criteria(CONT_SAME, ALL_METRICS)
    for score in scores:
        assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_drift_quantile_type_sturges():
    d = ContinuousDataDrift(CONT_BASELINE, quantile_type=QuantileType.Sturges)
    score = d.compute_drift(CONT_SAME, DataDriftType.JensenShannon)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_drift_quantile_type_string():
    d = ContinuousDataDrift(CONT_BASELINE, quantile_type="Scott")
    score = d.compute_drift(CONT_SAME, DataDriftType.JensenShannon)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_drift_export_baseline_sums_to_one():
    d = ContinuousDataDrift(CONT_BASELINE)
    bl = d.export_baseline()
    assert abs(sum(bl) - 1.0) < 1e-9


def test_cont_drift_export_baseline_length_matches_num_bins():
    d = ContinuousDataDrift(CONT_BASELINE)
    bl = d.export_baseline()
    assert len(bl) == d.num_bins


def test_cont_drift_reset_baseline_changes_distribution():
    d = ContinuousDataDrift(CONT_BASELINE)
    score_before = d.compute_drift(CONT_SHIFTED, DataDriftType.PopulationStabilityIndex)
    d.reset_baseline(CONT_SHIFTED)
    score_after = d.compute_drift(CONT_SHIFTED, DataDriftType.PopulationStabilityIndex)
    assert score_before > score_after
    assert score_after == pytest.approx(0.0, abs=1e-9)


def test_cont_drift_string_metric_accepted():
    d = ContinuousDataDrift(CONT_BASELINE)
    score = d.compute_drift(CONT_SAME, "WassersteinDistance")
    assert score == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# CategoricalDataDrift (stateful batch class)
# ---------------------------------------------------------------------------


def test_cat_drift_init():
    d = CategoricalDataDrift(CAT_BASELINE)
    assert d is not None
    assert d.num_bins > 0


def test_cat_drift_same_data_near_zero():
    d = CategoricalDataDrift(CAT_BASELINE)
    score = d.compute_drift(CAT_SAME, DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cat_drift_shifted_detects_drift():
    d = CategoricalDataDrift(["a", "a", "a", "a", "b"])
    score = d.compute_drift(
        ["b", "b", "b", "b", "a"], DataDriftType.PopulationStabilityIndex
    )
    assert score > 0.1


def test_cat_drift_multiple_criteria_correct_length():
    d = CategoricalDataDrift(CAT_BASELINE)
    scores = d.compute_drift_multiple_criteria(CAT_SAME, ALL_METRICS)
    assert len(scores) == len(ALL_METRICS)


def test_cat_drift_multiple_criteria_same_data_near_zero():
    d = CategoricalDataDrift(CAT_BASELINE)
    scores = d.compute_drift_multiple_criteria(CAT_SAME, ALL_METRICS)
    for score in scores:
        assert score == pytest.approx(0.0, abs=1e-9)


def test_cat_drift_num_bins_is_unique_labels_plus_one():
    # baseline has 3 unique labels → 4 bins (3 + overflow)
    d = CategoricalDataDrift(["a", "b", "c", "a", "b"])
    assert d.num_bins == 4


def test_cat_drift_export_baseline_sums_to_one():
    d = CategoricalDataDrift(CAT_BASELINE)
    bl = d.export_baseline()
    # export_baseline returns a list of floats (one per bin including overflow)
    assert abs(sum(bl.values()) - 1.0) < 1e-9


def test_cat_drift_reset_baseline_changes_distribution():
    d = CategoricalDataDrift(["a", "a", "a", "a", "b"])
    score_before = d.compute_drift(
        ["b", "b", "b", "b", "a"], DataDriftType.PopulationStabilityIndex
    )
    d.reset_baseline(["b", "b", "b", "b", "a"])
    score_after = d.compute_drift(
        ["b", "b", "b", "b", "a"], DataDriftType.PopulationStabilityIndex
    )
    assert score_before > score_after
    assert score_after == pytest.approx(0.0, abs=1e-9)


def test_cat_drift_unseen_label_does_not_raise():
    d = CategoricalDataDrift(["a", "b"])
    score = d.compute_drift(["a", "z"], DataDriftType.JensenShannon)
    assert score >= 0.0


def test_cat_drift_string_metric_accepted():
    d = CategoricalDataDrift(CAT_BASELINE)
    score = d.compute_drift(CAT_SAME, "JensenShannon")
    assert score == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# StreamingContinuousDataDriftFlush
# ---------------------------------------------------------------------------


def _make_cont_flush() -> StreamingContinuousDataDriftFlush:
    return StreamingContinuousDataDriftFlush(CONT_BASELINE, None)


def test_cont_flush_init():
    s = _make_cont_flush()
    assert s is not None
    assert s.num_bins > 0


def test_cont_flush_total_samples_after_batch():
    s = _make_cont_flush()
    s.update_stream_batch(CONT_SAME)
    assert s.total_samples == len(CONT_SAME)


def test_cont_flush_total_samples_after_single():
    s = _make_cont_flush()
    for v in CONT_SAME:
        s.update_stream(v)
    assert s.total_samples == len(CONT_SAME)


def test_cont_flush_same_data_near_zero():
    s = _make_cont_flush()
    s.update_stream_batch(CONT_SAME)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_flush_shifted_detects_drift():
    s = _make_cont_flush()
    s.update_stream_batch(CONT_SHIFTED)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score > 0.5


def test_cont_flush_multiple_criteria_correct_length():
    s = _make_cont_flush()
    s.update_stream_batch(CONT_SAME)
    scores = s.compute_drift_multiple_criteria(ALL_METRICS)
    assert len(scores) == len(ALL_METRICS)


def test_cont_flush_export_snapshot_returns_dict():
    s = _make_cont_flush()
    s.update_stream_batch(CONT_SAME)
    snap = s.export_snapshot()
    assert isinstance(snap, dict)
    assert len(snap) > 0


def test_cont_flush_export_baseline_sums_to_one():
    s = _make_cont_flush()
    bl = s.export_baseline()
    assert isinstance(bl, list)


def test_cont_flush_resets_stream():
    s = _make_cont_flush()
    s.update_stream_batch(CONT_SHIFTED)
    s.flush()
    assert s.total_samples == 0


def test_cont_flush_after_flush_same_data_near_zero():
    s = _make_cont_flush()
    s.update_stream_batch(CONT_SHIFTED)
    s.flush()
    s.update_stream_batch(CONT_SAME)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_flush_reset_baseline():
    s = _make_cont_flush()
    s.reset_baseline(CONT_SHIFTED)
    s.update_stream_batch(CONT_SHIFTED)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_flush_string_metric_accepted():
    s = _make_cont_flush()
    s.update_stream_batch(CONT_SAME)
    score = s.compute_drift("JensenShannon")
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_flush_quantile_type_enum():
    s = StreamingContinuousDataDriftFlush(CONT_BASELINE, QuantileType.Sturges)
    s.update_stream_batch(CONT_SAME)
    score = s.compute_drift(DataDriftType.JensenShannon)
    assert score == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# StreamingContinuousDataDriftDecay
# ---------------------------------------------------------------------------


def _make_cont_decay() -> StreamingContinuousDataDriftDecay:
    return StreamingContinuousDataDriftDecay(CONT_BASELINE, None)


def test_cont_decay_init():
    s = _make_cont_decay()
    assert s is not None
    assert s.num_bins > 0


def test_cont_decay_same_data_near_zero():
    s = _make_cont_decay()
    s.update_stream_batch(CONT_SAME)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_decay_shifted_detects_drift():
    s = _make_cont_decay()
    s.update_stream_batch(CONT_SHIFTED)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score > 0.5


def test_cont_decay_total_samples_accumulates():
    s = _make_cont_decay()
    s.update_stream_batch(CONT_SAME)
    assert s.total_samples > 0


def test_cont_decay_multiple_criteria_correct_length():
    s = _make_cont_decay()
    s.update_stream_batch(CONT_SAME)
    scores = s.compute_drift_multiple_criteria(ALL_METRICS)
    assert len(scores) == len(ALL_METRICS)


def test_cont_decay_reset_baseline():
    s = _make_cont_decay()
    s.reset_baseline(CONT_SHIFTED)
    s.update_stream_batch(CONT_SHIFTED)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cont_decay_custom_half_life():
    s = StreamingContinuousDataDriftDecay(CONT_BASELINE, None, decay_half_life=100)
    s.update_stream_batch(CONT_SAME)
    score = s.compute_drift(DataDriftType.JensenShannon)
    assert score == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# StreamingCategoricalDataDriftFlush
# ---------------------------------------------------------------------------


def _make_cat_flush() -> StreamingCategoricalDataDriftFlush:
    return StreamingCategoricalDataDriftFlush(CAT_BASELINE)


def test_cat_flush_init():
    s = _make_cat_flush()
    assert s is not None
    assert s.num_bins > 0


def test_cat_flush_total_samples_after_batch():
    s = _make_cat_flush()
    s.update_stream_batch(CAT_SAME)
    assert s.total_samples == len(CAT_SAME)


def test_cat_flush_total_samples_after_single():
    s = _make_cat_flush()
    for v in CAT_SAME:
        s.update_stream(v)
    assert s.total_samples == len(CAT_SAME)


def test_cat_flush_same_data_near_zero():
    s = _make_cat_flush()
    s.update_stream_batch(CAT_SAME)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cat_flush_shifted_detects_drift():
    s = StreamingCategoricalDataDriftFlush(["a", "a", "a", "a", "b"])
    s.update_stream_batch(["b", "b", "b", "b", "a"])
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score > 0.1


def test_cat_flush_multiple_criteria_correct_length():
    s = _make_cat_flush()
    s.update_stream_batch(CAT_SAME)
    scores = s.compute_drift_multiple_criteria(ALL_METRICS)
    assert len(scores) == len(ALL_METRICS)


def test_cat_flush_export_snapshot_returns_dict():
    s = _make_cat_flush()
    s.update_stream_batch(CAT_SAME)
    snap = s.export_snapshot()
    assert isinstance(snap, dict)


def test_cat_flush_resets_stream():
    s = _make_cat_flush()
    s.update_stream_batch(CAT_SHIFTED)
    s.flush()
    assert s.total_samples == 0


def test_cat_flush_after_flush_same_data_near_zero():
    s = _make_cat_flush()
    s.update_stream_batch(CAT_SHIFTED)
    s.flush()
    s.update_stream_batch(CAT_SAME)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cat_flush_reset_baseline():
    s = _make_cat_flush()
    s.reset_baseline(CAT_SHIFTED)
    s.update_stream_batch(CAT_SHIFTED)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cat_flush_unseen_label_does_not_raise():
    s = _make_cat_flush()
    s.update_stream_batch(["a", "z", "unknown"])
    score = s.compute_drift(DataDriftType.JensenShannon)
    assert score >= 0.0


def test_cat_flush_string_metric_accepted():
    s = _make_cat_flush()
    s.update_stream_batch(CAT_SAME)
    score = s.compute_drift("PopulationStabilityIndex")
    assert score == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# StreamingCategoricalDataDriftDecay
# ---------------------------------------------------------------------------


def _make_cat_decay() -> StreamingCategoricalDataDriftDecay:
    return StreamingCategoricalDataDriftDecay(CAT_BASELINE, decay_half_life=3600)


def test_cat_decay_init():
    s = _make_cat_decay()
    assert s is not None
    assert s.num_bins > 0


def test_cat_decay_same_data_near_zero():
    s = _make_cat_decay()
    s.update_stream_batch(CAT_SAME)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cat_decay_shifted_detects_drift():
    s = StreamingCategoricalDataDriftDecay(
        ["a", "a", "a", "a", "b"], decay_half_life=3600
    )
    s.update_stream_batch(["b", "b", "b", "b", "a"])
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score > 0.1


def test_cat_decay_total_samples_accumulates():
    s = _make_cat_decay()
    s.update_stream_batch(CAT_SAME)
    assert s.total_samples > 0


def test_cat_decay_multiple_criteria_correct_length():
    s = _make_cat_decay()
    s.update_stream_batch(CAT_SAME)
    scores = s.compute_drift_multiple_criteria(ALL_METRICS)
    assert len(scores) == len(ALL_METRICS)


def test_cat_decay_reset_baseline():
    s = _make_cat_decay()
    s.reset_baseline(CAT_SHIFTED)
    s.update_stream_batch(CAT_SHIFTED)
    score = s.compute_drift(DataDriftType.PopulationStabilityIndex)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_cat_decay_export_snapshot_returns_dict():
    s = _make_cat_decay()
    s.update_stream_batch(CAT_SAME)
    snap = s.export_snapshot()
    assert isinstance(snap, dict)
