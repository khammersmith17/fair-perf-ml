"""
Type stubs for the compiled Rust extension module ``_fair_perf_ml``.

This module is the PyO3 extension backend.  End-users should import from the
public sub-packages (``fair_perf_ml.data_bias``, ``fair_perf_ml.model_bias``,
``fair_perf_ml.model_perf``, ``fair_perf_ml.drift``) rather than from this
module directly.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray
from fair_perf_ml.models import LinearRegressionReport, LogisticRegressionReport, BinaryClassificationReport, DriftReport

# ---------------------------------------------------------------------------
# Return-type aliases
# The raw Rust functions return plain Python dicts.  These aliases document
# the expected shapes without coupling the stub to the higher-level models.
# ---------------------------------------------------------------------------

# {"metric_name": value, ...}
_AnalysisReport = dict[str, float]

# {"metric_name": delta, ...}
_DriftSnapshot = dict[str, float]

# {"metric_name": value, ...}
_PerformanceSnapshot = dict[str, float]

# ============================================================================
# Data bias
# ============================================================================


def py_data_bias_analyzer(
    feature_array: NDArray,
    ground_truth_array: NDArray,
    feature_label_or_threshold: Any,
    ground_truth_label_or_threshold: Any,
) -> _AnalysisReport:
    """
    Perform pre-training data bias analysis, deriving segmentation from a
    single label or threshold value for each of the feature and ground-truth
    arrays.

    Returns a dict mapping each data bias metric name to its computed value.
    """
    ...


def py_data_bias_analyzer_explicit_seg(
    feature_array: NDArray,
    feat_segmentation_threshold: float | None,
    feat_segmentation_label: str | None,
    feat_threshold_type: str | None,
    ground_truth_array: NDArray,
    gt_segmentation_threshold: float | None,
    gt_segmentation_label: str | None,
    gt_threshold_type: str | None,
) -> _AnalysisReport:
    """
    Perform pre-training data bias analysis with explicit, fine-grained
    segmentation criteria supplied separately for the feature and ground-truth
    arrays.

    Returns a dict mapping each data bias metric name to its computed value.
    """
    ...


def py_data_bias_runtime_check(
    baseline: _AnalysisReport,
    latest: _AnalysisReport,
    threshold: float = 0.10,
) -> DriftReport:
    """
    Compare a runtime data bias analysis report against the baseline across
    all supported metrics.

    Returns a drift report indicating which metrics exceeded ``threshold``.
    """
    ...


def py_data_bias_partial_check(
    baseline: _AnalysisReport,
    latest: _AnalysisReport,
    metrics: list[str],
    threshold: float = 0.10,
) -> DriftReport:
    """
    Compare a runtime data bias analysis report against the baseline for a
    specific subset of metrics.

    Returns a drift report indicating which of the requested metrics exceeded
    ``threshold``.
    """
    ...


# ============================================================================
# Model bias
# ============================================================================


def py_model_bias_analyzer(
    feature_array: NDArray,
    ground_truth_array: NDArray,
    prediction_array: NDArray,
    feature_label_or_threshold: Any,
    ground_truth_label_or_threshold: Any,
    prediction_label_or_threshold: Any,
) -> _AnalysisReport:
    """
    Perform post-training model bias analysis, deriving segmentation from a
    single label or threshold value for each of the feature, ground-truth, and
    prediction arrays.

    Returns a dict mapping each model bias metric name to its computed value.
    """
    ...


def py_model_bias_analyzer_explicit_seg(
    feature_array: NDArray,
    feat_segmentation_threshold: float | None,
    feat_segmentation_label: str | None,
    feat_threshold_type: str | None,
    ground_truth_array: NDArray,
    gt_segmentation_threshold: float | None,
    gt_segmentation_label: str | None,
    gt_threshold_type: str | None,
    prediction_array: NDArray,
    pred_segmentation_threshold: float | None,
    pred_segmentation_label: str | None,
    pred_threshold_type: str | None,
) -> _AnalysisReport:
    """
    Perform post-training model bias analysis with explicit, fine-grained
    segmentation criteria supplied separately for all three arrays.

    Returns a dict mapping each model bias metric name to its computed value.
    """
    ...


def py_model_bias_runtime_check(
    baseline: _AnalysisReport,
    latest: _AnalysisReport,
    threshold: float = 0.10,
) -> DriftReport:
    """
    Compare a runtime model bias report against the baseline across all
    supported metrics.

    Returns a drift report indicating which metrics exceeded ``threshold``.
    """
    ...


def py_model_bias_partial_check(
    baseline: _AnalysisReport,
    latest: _AnalysisReport,
    metrics: list[str],
    threshold: float = 0.10,
) -> DriftReport:
    """
    Compare a runtime model bias report against the baseline for a specific
    subset of metrics.

    Returns a drift report indicating which of the requested metrics exceeded
    ``threshold``.
    """
    ...


# ============================================================================
# Model performance
# ============================================================================


def py_model_perf_linear_regression(
    y_true_src: NDArray,
    y_pred_src: NDArray,
) -> LinearRegressionReport:
    """
    Compute linear regression performance metrics.

    Note: argument order is ``y_pred`` first, then ``y_true``.

    Returns a dict of metric name → value for all linear regression metrics.
    """
    ...


def py_model_perf_classification(
    y_true_src: NDArray,
    y_pred_src: NDArray,
) -> BinaryClassificationReport:
    """
    Compute binary classification performance metrics.

    Returns a dict of metric name → value for all binary classification
    metrics.
    """
    ...


def py_model_perf_logistic_regression(
    y_true_src: NDArray,
    y_pred_src: NDArray,
    threshold: float,
) -> LogisticRegressionReport:
    """
    Compute logistic regression performance metrics.  Predicted probabilities
    are binarised using ``threshold`` before metric computation.

    Returns a dict of metric name → value for all logistic regression metrics.
    """
    ...


def py_model_perf_lin_reg_rt_full(
    baseline: dict[str, float],
    latest: dict[str, float],
    threshold: float = 0.10,
) -> DriftReport:
    """
    Runtime comparison for a linear regression model across all supported
    metrics.

    Returns a drift report indicating which metrics exceeded ``threshold``.
    """
    ...


def py_model_perf_lin_reg_rt_partial(
    baseline: dict[str, float],
    latest: dict[str, float],
    evaluation_metrics: list[str],
    threshold: float = 0.10,
) -> DriftReport:
    """
    Runtime comparison for a linear regression model restricted to the
    provided subset of metric names.

    Returns a drift report indicating which of the requested metrics exceeded
    ``threshold``.
    """
    ...


def py_model_perf_log_reg_rt_full(
    baseline: dict[str, float],
    latest: dict[str, float],
    threshold: float = 0.10,
) -> DriftReport:
    """
    Runtime comparison for a logistic regression model across all supported
    metrics.

    Returns a drift report indicating which metrics exceeded ``threshold``.
    """
    ...


def py_model_perf_log_reg_rt_partial(
    baseline: dict[str, float],
    latest: dict[str, float],
    evaluation_metrics: list[str],
    threshold: float = 0.10,
) -> DriftReport:
    """
    Runtime comparison for a logistic regression model restricted to the
    provided subset of metric names.

    Returns a drift report indicating which of the requested metrics exceeded
    ``threshold``.
    """
    ...


def py_model_perf_class_rt_full(
    baseline: dict[str, float],
    latest: dict[str, float],
    threshold: float = 0.10,
) -> DriftReport:
    """
    Runtime comparison for a binary classification model across all supported
    metrics.

    Returns a drift report indicating which metrics exceeded ``threshold``.
    """
    ...


def py_model_perf_class_rt_partial(
    baseline: dict[str, float],
    latest: dict[str, float],
    evaluation_metrics: list[str],
    threshold: float = 0.10,
) -> DriftReport:
    """
    Runtime comparison for a binary classification model restricted to the
    provided subset of metric names.

    Returns a drift report indicating which of the requested metrics exceeded
    ``threshold``.
    """
    ...


# ============================================================================
# Data drift — batch / discrete
# ============================================================================


def py_compute_drift_continuous_distribtuion(
    baseline: NDArray,
    candidate: NDArray,
    drift_metrics: list[str],
    quantile_type: str | None = None,
) -> list[float]:
    """
    Compute one or more drift scores between two continuous distributions in a
    single pass.

    ``drift_metrics`` must be a list of strings drawn from
    ``{"JensenShannon", "PopulationStabilityIndex", "WassersteinDistance",
    "KullbackLeibler"}``.

    Returns one score per metric in the same order as ``drift_metrics``.

    Note: the function name contains a deliberate typo (``distribtuion``) that
    mirrors the Rust export name.
    """
    ...


def py_compute_drift_categorical_distribtuion(
    baseline: list[str],
    candidate: list[str],
    drift_metrics: list[str],
) -> list[float]:
    """
    Compute one or more drift scores between two categorical distributions in a
    single pass.

    ``drift_metrics`` must be a list of strings drawn from
    ``{"JensenShannon", "PopulationStabilityIndex", "WassersteinDistance",
    "KullbackLeibler"}``.

    Returns one score per metric in the same order as ``drift_metrics``.

    Note: the function name contains a deliberate typo (``distribtuion``) that
    mirrors the Rust export name.
    """
    ...


class PyContinuousDataDrift:
    """
    Stateless batch drift monitor for continuous (floating-point) features.

    The baseline distribution is summarised as a histogram whose bin count is
    derived from the baseline data using the selected quantile rule.  Drift is
    measured by comparing a runtime dataset against that histogram.
    """

    def __init__(
        self,
        baseline_data: NDArray,
        quantile_type: str | None = None,
    ) -> None:
        """
        Construct with a baseline dataset.

        Args:
            baseline_data: Numpy array of float64 values.
            quantile_type: Bin-count rule — one of ``"FreedmanDiaconis"``,
                ``"Scott"``, ``"Sturges"``, or ``None`` (defaults to
                FreedmanDiaconis).
        """
        ...

    def reset_baseline(self, new_baseline: NDArray) -> None:
        """Replace the baseline, recomputing the histogram from scratch."""
        ...

    def compute_drift(self, runtime_data: NDArray, drift_metric: str) -> float:
        """
        Compute a single drift score between ``runtime_data`` and the baseline.

        ``drift_metric`` must be one of ``"JensenShannon"``,
        ``"PopulationStabilityIndex"``, ``"WassersteinDistance"``,
        ``"KullbackLeibler"``.
        """
        ...

    def compute_drift_mutliple_criteria(
        self, runtime_data: NDArray, drift_metrics: list[str]
    ) -> list[float]:
        """
        Compute multiple drift scores in a single pass.

        Returns one score per metric in the same order as ``drift_metrics``.

        Note: the method name contains a deliberate typo (``mutliple``) that
        mirrors the Rust export name.
        """
        ...

    def export_baseline(self) -> list[float]:
        """
        Export the baseline as a normalised probability distribution.

        Returns a list of floats (one per bin) that sum to 1.0.
        """
        ...

    @property
    def num_bins(self) -> int:
        """Number of histogram bins derived from the baseline dataset."""
        ...


class PyCategoricalDataDrift:
    """
    Stateless batch drift monitor for categorical (string) features.

    The baseline distribution is summarised as a label-frequency map.  Drift
    is measured by comparing a runtime dataset against that map.
    """

    def __init__(self, baseline_data: list[str]) -> None:
        """
        Construct with a baseline dataset.

        Args:
            baseline_data: List of category label strings.
        """
        ...

    def reset_baseline(self, new_baseline: list[str]) -> None:
        """Replace the baseline, recomputing the label frequency map."""
        ...

    def compute_drift(self, runtime_data: list[str], drift_metric: str) -> float:
        """
        Compute a single drift score between ``runtime_data`` and the baseline.

        ``drift_metric`` must be one of ``"JensenShannon"``,
        ``"PopulationStabilityIndex"``, ``"WassersteinDistance"``,
        ``"KullbackLeibler"``.
        """
        ...

    def compute_drift_mutliple_criteria(
        self, runtime_data: list[str], drift_metrics: list[str]
    ) -> list[float]:
        """
        Compute multiple drift scores in a single pass.

        Returns one score per metric in the same order as ``drift_metrics``.

        Note: the method name contains a deliberate typo (``mutliple``) that
        mirrors the Rust export name.
        """
        ...

    def export_baseline(self) -> dict[str, float]:
        """
        Export the baseline as a normalised label frequency map.

        Returns a dict mapping each label to its fraction of the baseline
        dataset.  Values sum to 1.0.
        """
        ...

    @property
    def num_bins(self) -> int:
        """Number of distinct category bins in the baseline dataset."""
        ...


# ============================================================================
# Data drift — streaming
# ============================================================================


class PyStreamingContinuousDataDriftFlush:
    """
    Stateful streaming drift monitor for continuous features with periodic
    flush semantics.

    Accumulates runtime examples incrementally.  The runtime distribution is
    reset to empty either manually via ``flush()`` or automatically when
    ``flush_rate`` samples have been received or ``flush_cadence`` seconds
    have elapsed since the last flush.
    """

    def __init__(
        self,
        baseline_dataset: NDArray,
        quantile_type: str | None,
        flush_rate: int | None = None,
        flush_cadence: int | None = None,
    ) -> None:
        """
        Construct with a baseline dataset and flush policy.

        Args:
            baseline_dataset: Numpy array of float64 values.
            quantile_type: Bin-count rule — one of ``"FreedmanDiaconis"``,
                ``"Scott"``, ``"Sturges"``, or ``None`` (defaults to
                FreedmanDiaconis).
            flush_rate: Flush after this many accumulated samples. ``None``
                disables sample-count flushing.
            flush_cadence: Flush after this many seconds since the last flush.
                ``None`` uses the library default (86 400 s / 24 h).
        """
        ...

    def reset_baseline(self, new_baseline: NDArray) -> None:
        """Replace the baseline, recomputing the histogram from scratch."""
        ...

    def update_stream(self, example: float) -> None:
        """Accumulate a single runtime example into the stream."""
        ...

    def update_stream_batch(self, runtime_data: NDArray) -> None:
        """Accumulate a batch of runtime examples into the stream."""
        ...

    def compute_drift(self, drift_metric: str) -> float:
        """
        Compute a single drift score between the accumulated runtime data and
        the baseline.
        """
        ...

    def compute_drift_multiple_criteria(self, drift_metrics: list[str]) -> list[float]:
        """
        Compute multiple drift scores in a single pass.

        Returns one score per metric in the same order as ``drift_metrics``.
        """
        ...

    @property
    def total_samples(self) -> int:
        """Total number of runtime examples accumulated since the last flush."""
        ...

    @property
    def n_bins(self) -> int:
        """Number of histogram bins derived from the baseline dataset."""
        ...

    def export_snapshot(self) -> dict[str, list[float]]:
        """
        Export a point-in-time snapshot of the runtime distribution.

        Returns a dict mapping bin labels to per-bin sample counts.
        """
        ...

    def export_baseline(self) -> list[float]:
        """Export the baseline histogram as a plain dict."""
        ...

    def flush(self) -> None:
        """Discard all accumulated runtime data.  The baseline is preserved."""
        ...

    @property
    def last_flush(self) -> int:
        """Unix timestamp (seconds) of the most recent flush."""
        ...


class PyStreamingContinuousDataDriftDecay:
    """
    Stateful streaming drift monitor for continuous features with exponential
    decay semantics.

    Recent examples are weighted more heavily than older ones.  The effective
    window is controlled by ``decay_half_life``.
    """

    def __init__(
        self,
        baseline_dataset: NDArray,
        quantile_type: str | None,
        decay_half_life: int | None = None,
    ) -> None:
        """
        Construct with a baseline dataset and decay policy.

        Args:
            baseline_dataset: Numpy array of float64 values.
            quantile_type: Bin-count rule — one of ``"FreedmanDiaconis"``,
                ``"Scott"``, ``"Sturges"``, or ``None`` (defaults to
                FreedmanDiaconis).
            decay_half_life: Number of examples after which an observation's
                weight is halved.  ``None`` uses the library default.
        """
        ...

    def reset_baseline(self, new_baseline: NDArray) -> None:
        """Replace the baseline, recomputing the histogram from scratch."""
        ...

    def update_stream(self, example: float) -> None:
        """Accumulate a single runtime example into the stream."""
        ...

    def update_stream_batch(self, runtime_data: NDArray) -> None:
        """Accumulate a batch of runtime examples into the stream."""
        ...

    def compute_drift(self, drift_metric: str) -> float:
        """
        Compute a single drift score between the accumulated (decayed) runtime
        data and the baseline.
        """
        ...

    def compute_drift_multiple_criteria(self, drift_metrics: list[str]) -> list[float]:
        """
        Compute multiple drift scores in a single pass.

        Returns one score per metric in the same order as ``drift_metrics``.
        """
        ...

    @property
    def total_samples(self) -> int:
        """Total number of runtime examples accumulated."""
        ...

    @property
    def n_bins(self) -> int:
        """Number of histogram bins derived from the baseline dataset."""
        ...

    def export_snapshot(self) -> dict[str, list[float]]:
        """
        Export a point-in-time snapshot of the decayed runtime distribution.

        Returns a dict mapping bin labels to per-bin effective weights.
        """
        ...

    def export_baseline(self) -> list[float]:
        """Export the baseline histogram as a plain dict."""
        ...


class PyStreamingCategoricalDataDriftFlush:
    """
    Stateful streaming drift monitor for categorical features with periodic
    flush semantics.

    Accumulates runtime examples incrementally.  The runtime distribution is
    reset to empty either manually via ``flush()`` or automatically when
    ``flush_rate`` samples have been received or ``flush_cadence`` seconds
    have elapsed since the last flush.
    """

    def __init__(
        self,
        baseline_data: list[str],
        flush_rate: int | None = None,
        flush_cadence: int | None = None,
    ) -> None:
        """
        Construct with a baseline dataset and flush policy.

        Args:
            baseline_data: List of category label strings.
            flush_rate: Flush after this many accumulated samples.  ``None``
                disables sample-count flushing.
            flush_cadence: Flush after this many seconds since the last flush.
                ``None`` uses the library default (86 400 s / 24 h).
        """
        ...

    def reset_baseline(self, new_baseline: list[str]) -> None:
        """Replace the baseline, recomputing the label frequency map."""
        ...

    def update_stream(self, example: str) -> None:
        """Accumulate a single runtime example into the stream."""
        ...

    def update_stream_batch(self, runtime_data: list[str]) -> None:
        """Accumulate a batch of runtime examples into the stream."""
        ...

    def compute_drift(self, drift_metric: str) -> float:
        """
        Compute a single drift score between the accumulated runtime data and
        the baseline.
        """
        ...

    def compute_drift_multiple_criteria(self, drift_metrics: list[str]) -> list[float]:
        """
        Compute multiple drift scores in a single pass.

        Returns one score per metric in the same order as ``drift_metrics``.
        """
        ...

    @property
    def total_samples(self) -> int:
        """Total number of runtime examples accumulated since the last flush."""
        ...

    @property
    def n_bins(self) -> int:
        """Number of distinct category bins in the baseline dataset."""
        ...

    def export_snapshot(self) -> dict[str, float]:
        """
        Export a point-in-time snapshot of the runtime label distribution.

        Returns a dict mapping each label to its accumulated count since the
        last flush.
        """
        ...

    def export_baseline(self) -> dict:
        """Export the baseline label frequency map as a plain dict."""
        ...

    def flush(self) -> None:
        """Discard all accumulated runtime data.  The baseline is preserved."""
        ...

    @property
    def last_flush(self) -> int:
        """Unix timestamp (seconds) of the most recent flush."""
        ...


class PyStreamingCategoricalDataDriftDecay:
    """
    Stateful streaming drift monitor for categorical features with exponential
    decay semantics.

    Recent examples are weighted more heavily than older ones.  The effective
    window is controlled by ``decay_half_life``.
    """

    def __init__(
        self,
        baseline_data: list[str],
        decay_half_life: int | None = None,
    ) -> None:
        """
        Construct with a baseline dataset and decay policy.

        Args:
            baseline_data: List of category label strings.
            decay_half_life: Number of examples after which an observation's
                weight is halved.  ``None`` uses the library default.
        """
        ...

    def reset_baseline(self, new_baseline: list[str]) -> None:
        """Replace the baseline, recomputing the label frequency map."""
        ...

    def update_stream(self, example: str) -> None:
        """Accumulate a single runtime example into the stream."""
        ...

    def update_stream_batch(self, runtime_data: list[str]) -> None:
        """Accumulate a batch of runtime examples into the stream."""
        ...

    def compute_drift(self, drift_metric: str) -> float:
        """
        Compute a single drift score between the accumulated (decayed) runtime
        data and the baseline.
        """
        ...

    def compute_drift_multiple_criteria(self, drift_metrics: list[str]) -> list[float]:
        """
        Compute multiple drift scores in a single pass.

        Returns one score per metric in the same order as ``drift_metrics``.
        """
        ...

    @property
    def total_samples(self) -> int:
        """Total number of runtime examples accumulated."""
        ...

    @property
    def n_bins(self) -> int:
        """Number of distinct category bins in the baseline dataset."""
        ...

    def export_snapshot(self) -> dict[str, float]:
        """
        Export a point-in-time snapshot of the decayed runtime label
        distribution.

        Returns a dict mapping each label to its effective accumulated weight.
        """
        ...

    def export_baseline(self) -> dict:
        """Export the baseline label frequency map as a plain dict."""
        ...


# ============================================================================
# Bias streaming
# ============================================================================


class PyDataBiasStreaming:
    """
    Stateful streaming monitor for pre-training data bias metrics.

    The Python layer handles segmentation (converting raw feature/ground-truth
    values to 0/1 labels) before passing data into this class.  All arrays
    passed here must already be integer-labelled.
    """

    def __init__(
        self,
        labeled_features: list[int],
        labeled_ground_truth: list[int],
    ) -> None:
        """
        Construct with a labelled baseline dataset.

        Args:
            labeled_features: Pre-labelled feature values (0 = disadvantaged,
                1 = advantaged).
            labeled_ground_truth: Pre-labelled ground truth values (0/1).
        """
        ...

    def push(self, feature_label: int, ground_truth_label: int) -> None:
        """Accumulate a single pre-labelled example into the stream."""
        ...

    def push_batch(
        self, labeled_features: list[int], labeled_ground_truth: list[int]
    ) -> None:
        """Accumulate a batch of pre-labelled examples into the stream."""
        ...

    def reset_baseline(
        self, labeled_features: list[int], labeled_ground_truth: list[int]
    ) -> None:
        """Replace the baseline with a new pre-labelled dataset."""
        ...

    def flush(self) -> None:
        """Discard all accumulated runtime data.  The baseline is preserved."""
        ...

    def performance_snapshot(self) -> _PerformanceSnapshot:
        """
        Compute a point-in-time performance report over all accumulated runtime
        examples, independent of the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_snapshot(self) -> _DriftSnapshot:
        """
        Compute a point-in-time drift snapshot comparing accumulated runtime
        examples to the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report(self, threshold: float | None = 0.10) -> DriftReport:
        """
        Evaluate whether accumulated data bias has drifted beyond ``threshold``
        relative to the baseline across all supported metrics.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report_partial_metrics(
        self, metrics: list[str], threshold: float | None = 0.10
    ) -> DriftReport:
        """
        Same as ``drift_report`` but scoped to the provided subset of metric
        name strings.
        """
        ...


class PyModelBiasStreaming:
    """
    Stateful streaming monitor for post-training model bias metrics.

    The Python layer handles segmentation before passing data into this class.
    All arrays passed here must already be integer-labelled.
    """

    def __init__(
        self,
        labeled_features: list[int],
        labeled_predictions: list[int],
        labeled_ground_truth: list[int],
    ) -> None:
        """
        Construct with a labelled baseline dataset.

        Args:
            labeled_features: Pre-labelled feature values (0/1).
            labeled_predictions: Pre-labelled prediction values (0/1).
            labeled_ground_truth: Pre-labelled ground truth values (0/1).
        """
        ...

    def push(
        self, feature_label: int, prediction_label: int, ground_truth_label: int
    ) -> None:
        """Accumulate a single pre-labelled example into the stream."""
        ...

    def push_batch(
        self,
        labeled_features: list[int],
        labeled_predictions: list[int],
        labeled_ground_truth: list[int],
    ) -> None:
        """Accumulate a batch of pre-labelled examples into the stream."""
        ...

    def reset_baseline(
        self,
        labeled_features: list[int],
        labeled_predictions: list[int],
        labeled_ground_truth: list[int],
    ) -> None:
        """Replace the baseline with a new pre-labelled dataset."""
        ...

    def flush(self) -> None:
        """Discard all accumulated runtime data.  The baseline is preserved."""
        ...

    def performance_snapshot(self) -> _PerformanceSnapshot:
        """
        Compute a point-in-time performance report over all accumulated runtime
        examples, independent of the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_snapshot(self) -> _DriftSnapshot:
        """
        Compute a point-in-time drift snapshot comparing accumulated runtime
        examples to the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report(self, threshold: float | None = 0.10) -> DriftReport:
        """
        Evaluate whether accumulated model bias has drifted beyond
        ``threshold`` relative to the baseline across all supported metrics.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report_partial_metrics(
        self, metrics: list[str], threshold: float | None = 0.10
    ) -> DriftReport:
        """
        Same as ``drift_report`` but scoped to the provided subset of metric
        name strings.
        """
        ...


# ============================================================================
# Model performance streaming
# ============================================================================


class PyBinaryClassificationStreaming:
    """
    Stateful streaming monitor for binary classification model performance.

    The Python layer applies the positive-class label before passing data into
    this class.  All arrays passed here must already be integer-labelled
    (0 = negative, 1 = positive).
    """

    def __init__(
        self,
        labeled_y_true: list[int],
        labeled_y_pred: list[int],
    ) -> None:
        """
        Construct with a labelled baseline dataset.

        Args:
            labeled_y_true: Pre-labelled ground truth values (0/1).
            labeled_y_pred: Pre-labelled prediction values (0/1).
        """
        ...

    def push(self, y_true: int, y_pred: int) -> None:
        """Accumulate a single pre-labelled example into the stream."""
        ...

    def push_batch(
        self, labeled_y_true: list[int], labeled_y_pred: list[int]
    ) -> None:
        """Accumulate a batch of pre-labelled examples into the stream."""
        ...

    def reset_baseline(
        self, labeled_y_true: list[int], labeled_y_pred: list[int]
    ) -> None:
        """Replace the baseline with a new pre-labelled dataset."""
        ...

    def flush(self) -> None:
        """Discard all accumulated runtime data.  The baseline is preserved."""
        ...

    def performance_snapshot(self) -> _PerformanceSnapshot:
        """
        Compute a point-in-time performance report over all accumulated runtime
        examples, independent of the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_snapshot(self) -> _DriftSnapshot:
        """
        Compute a point-in-time drift snapshot comparing accumulated runtime
        examples to the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report(self, drift_threshold: float) -> DriftReport:
        """
        Evaluate whether classification performance has drifted beyond
        ``drift_threshold`` relative to the baseline across all supported
        metrics.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report_partial_metrics(
        self, metrics: list[str], drift_threshold: float
    ) -> DriftReport:
        """
        Same as ``drift_report`` but scoped to the provided subset of metric
        name strings.
        """
        ...


class PyLinearRegressionStreaming:
    """
    Stateful streaming monitor for linear regression model performance.
    """

    def __init__(self, y_true: NDArray, y_pred: NDArray) -> None:
        """
        Construct with a baseline dataset.

        Args:
            y_true: Numpy array of float ground truth values.
            y_pred: Numpy array of float predicted values.
        """
        ...

    def push(self, y_true: float, y_pred: float) -> None:
        """Accumulate a single example into the stream."""
        ...

    def push_batch(self, y_true: NDArray, y_pred: NDArray) -> None:
        """Accumulate a batch of examples into the stream."""
        ...

    def reset_baseline(self, y_true: NDArray, y_pred: NDArray) -> None:
        """Replace the baseline with a new dataset."""
        ...

    def flush(self) -> None:
        """Discard all accumulated runtime data.  The baseline is preserved."""
        ...

    def performance_snapshot(self) -> _PerformanceSnapshot:
        """
        Compute a point-in-time performance report over all accumulated runtime
        examples, independent of the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_snapshot(self) -> _DriftSnapshot:
        """
        Compute a point-in-time drift snapshot comparing accumulated runtime
        examples to the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report(self, drift_threshold: float) -> DriftReport:
        """
        Evaluate whether regression performance has drifted beyond
        ``drift_threshold`` relative to the baseline across all supported
        metrics.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report_partial_metrics(
        self, metrics: list[str], drift_threshold: float
    ) -> DriftReport:
        """
        Same as ``drift_report`` but scoped to the provided subset of metric
        name strings.
        """
        ...


class PyLogisticRegressionStreaming:
    """
    Stateful streaming monitor for logistic regression model performance.

    A decision threshold is applied to predicted probabilities to produce
    binary labels for metric computation.
    """

    def __init__(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        threshold: float | None = None,
    ) -> None:
        """
        Construct with a baseline dataset and a decision threshold.

        Args:
            y_true: Numpy array of float ground truth values.
            y_pred: Numpy array of float predicted probabilities.
            threshold: Decision threshold applied to probabilities.  ``None``
                defaults to 0.5.
        """
        ...

    def push(self, y_true: float, y_pred: float) -> None:
        """Accumulate a single example into the stream."""
        ...

    def push_batch(self, y_true: NDArray, y_pred: NDArray) -> None:
        """Accumulate a batch of examples into the stream."""
        ...

    def reset_baseline(self, y_true: NDArray, y_pred: NDArray) -> None:
        """
        Replace the baseline with a new dataset.  The decision threshold is
        unchanged.
        """
        ...

    def reset_baseline_and_decision_threshold(
        self, y_true: NDArray, y_pred: NDArray, threshold: float
    ) -> None:
        """
        Replace the baseline and update the decision threshold simultaneously.

        Changing the threshold invalidates previously computed baseline
        statistics, so a new baseline dataset is required.
        """
        ...

    def flush(self) -> None:
        """Discard all accumulated runtime data.  The baseline is preserved."""
        ...

    def performance_snapshot(self) -> _PerformanceSnapshot:
        """
        Compute a point-in-time performance report over all accumulated runtime
        examples, independent of the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_snapshot(self) -> _DriftSnapshot:
        """
        Compute a point-in-time drift snapshot comparing accumulated runtime
        examples to the baseline.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report(self, drift_threshold: float) -> DriftReport:
        """
        Evaluate whether logistic regression performance has drifted beyond
        ``drift_threshold`` relative to the baseline across all supported
        metrics.

        Raises if no runtime data has been accumulated.
        """
        ...

    def drift_report_partial_metrics(
        self, metrics: list[str], drift_threshold: float
    ) -> DriftReport:
        """
        Same as ``drift_report`` but scoped to the provided subset of metric
        name strings.
        """
        ...
