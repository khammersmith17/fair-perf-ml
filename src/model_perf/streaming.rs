use crate::{
    data_handler::{ConfusionMatrix, ConfusionPushPayload},
    errors::{ModelPerfResult, ModelPerformanceError},
    metrics::{
        get_stability_eps, ClassificationEvaluationMetric, LinearRegressionEvaluationMetric,
    },
    reporting::{
        BinaryClassificationAnalysisReport, BinaryClassificationDriftSnapshot, DriftReport,
        LinearRegressionAnalysisReport, LinearRegressionDriftSnapshot,
        LogisticRegressionAnalysisReport, LogisticRegressionDriftSnapshot, DEFAULT_DRIFT_THRESHOLD,
    },
    runtime::{BinaryClassificationRuntime, LinearRegressionRuntime, LogisticRegressionRuntime},
    zip_iters,
};

#[cfg(feature = "python")]
pub(crate) mod py_api {
    use super::{
        BinaryClassificationStreaming, LinearRegressionStreaming, LogisticRegressionStreaming,
    };
    use crate::data_handler::py_types_handler::{
        report_to_py_dict as perf_report_to_py_dict, PyDictResult,
    };
    use crate::errors::ModelPerformanceError;
    use crate::metrics::{
        ClassificationMetricVec, LinearRegressionMetricVec, LogisticRegressionMetricVec,
    };
    use pyo3::prelude::*;
    use pyo3::types::IntoPyDict;

    // All types here are simply logical wrappers around core types, simply to expose the apis to
    // python through FFI.

    // requires label to be applied in Python wrapper, for now at least
    // the generics are easier to define in this case
    #[pyclass]
    pub(crate) struct PyBinaryClassificationStreaming {
        inner: BinaryClassificationStreaming<i32>,
    }

    #[pymethods]
    impl PyBinaryClassificationStreaming {
        #[new]
        fn new(y_true: Vec<i32>, y_pred: Vec<i32>) -> PyResult<PyBinaryClassificationStreaming> {
            let inner = BinaryClassificationStreaming::new(1_i32, &y_true, &y_pred)?;
            Ok(PyBinaryClassificationStreaming { inner })
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn push_batch(&mut self, y_true: Vec<i32>, y_pred: Vec<i32>) -> PyResult<()> {
            self.inner.push_batch(&y_true, &y_pred)?;
            Ok(())
        }

        fn push(&mut self, y_true: i32, y_pred: i32) {
            self.inner.push(&y_true, &y_pred)
        }

        fn reset_baseline(&mut self, y_true: Vec<i32>, y_pred: Vec<i32>) -> PyResult<()> {
            self.inner.reset_baseline(&y_true, &y_pred)?;
            Ok(())
        }

        fn performance_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.performance_snapshot()?;
            Ok(perf_report_to_py_dict(py, report))
        }

        fn drift_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.drift_snapshot()?;
            Ok(perf_report_to_py_dict(py, report))
        }

        fn drift_report<'py>(&self, py: Python<'py>, drift_threshold: f32) -> PyDictResult<'py> {
            let report = self.inner.drift_report(Some(drift_threshold))?;
            Ok(report.into_py_dict(py)?)
        }

        fn drift_report_partial_metrics<'py>(
            &self,
            py: Python<'py>,
            metrics: Vec<String>,
            drift_threshold: f32,
        ) -> PyDictResult<'py> {
            let m_vec = ClassificationMetricVec::try_from(metrics.as_ref())?;
            let report = self
                .inner
                .drift_report_partial_metrics(m_vec.as_ref(), Some(drift_threshold))?;
            Ok(report.into_py_dict(py)?)
        }
    }

    #[pyclass]
    pub(crate) struct PyLinearRegressionStreaming {
        inner: LinearRegressionStreaming,
    }

    #[pymethods]
    impl PyLinearRegressionStreaming {
        #[new]
        fn new(y_true: Vec<f32>, y_pred: Vec<f32>) -> PyResult<PyLinearRegressionStreaming> {
            let inner = LinearRegressionStreaming::new(&y_true, &y_pred)?;

            Ok(PyLinearRegressionStreaming { inner })
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn push(&mut self, t: f32, p: f32) {
            self.inner.push(t, p)
        }

        fn push_batch(&mut self, y_true: Vec<f32>, y_pred: Vec<f32>) -> PyResult<()> {
            self.inner.push_batch(&y_true, &y_pred)?;
            Ok(())
        }

        fn reset_baseline(&mut self, y_true: Vec<f32>, y_pred: Vec<f32>) -> PyResult<()> {
            self.inner.reset_baseline(&y_true, &y_pred)?;
            Ok(())
        }

        fn performance_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.performance_snapshot()?;
            Ok(perf_report_to_py_dict(py, report))
        }

        fn drift_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.drift_snapshot()?;
            Ok(perf_report_to_py_dict(py, report))
        }

        fn drift_report<'py>(&self, py: Python<'py>, drift_threshold: f32) -> PyDictResult<'py> {
            let report = self.inner.drift_report(Some(drift_threshold))?;
            Ok(report.into_py_dict(py)?)
        }

        fn drift_report_partial_metrics<'py>(
            &self,
            py: Python<'py>,
            metrics: Vec<String>,
            drift_threshold: f32,
        ) -> PyDictResult<'py> {
            let m_vec = LinearRegressionMetricVec::try_from(metrics.as_ref())?;
            let report = self
                .inner
                .drift_report_partial_metrics(m_vec.as_ref(), Some(drift_threshold))?;
            Ok(report.into_py_dict(py)?)
        }
    }

    #[pyclass]
    pub(crate) struct PyLogisticRegressionStreaming {
        inner: LogisticRegressionStreaming,
    }

    #[pymethods]
    impl PyLogisticRegressionStreaming {
        #[new]
        fn new(
            y_true: Vec<f32>,
            y_pred: Vec<f32>,
            threshold: f32,
        ) -> PyResult<PyLogisticRegressionStreaming> {
            let inner = LogisticRegressionStreaming::new(&y_true, &y_pred, Some(threshold))?;

            Ok(PyLogisticRegressionStreaming { inner })
        }

        fn push(&mut self, y_true: f32, y_pred: f32) {
            self.inner.push(y_true, y_pred);
        }

        fn push_batch(&mut self, y_true: Vec<f32>, y_pred: Vec<f32>) -> PyResult<()> {
            self.inner.push_batch(&y_true, &y_pred)?;
            Ok(())
        }

        fn performance_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.performance_snapshot()?;
            Ok(perf_report_to_py_dict(py, report))
        }

        fn drift_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self.inner.drift_snapshot()?;
            Ok(perf_report_to_py_dict(py, report))
        }

        fn drift_report<'py>(&self, py: Python<'py>, drift_threshold: f32) -> PyDictResult<'py> {
            let report = self.inner.drift_report(Some(drift_threshold))?;
            Ok(report.into_py_dict(py)?)
        }

        fn drift_report_partial_metrics<'py>(
            &self,
            py: Python<'py>,
            metrics: Vec<String>,
            drift_threshold: f32,
        ) -> PyDictResult<'py> {
            let m_vec = LogisticRegressionMetricVec::try_from(metrics.as_ref())?;
            let report = self
                .inner
                .drift_report_partial_metrics(m_vec.as_ref(), Some(drift_threshold))?;
            Ok(report.into_py_dict(py)?)
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn reset_baseline(&mut self, y_true: Vec<f32>, y_pred: Vec<f32>) -> PyResult<()> {
            self.inner.reset_baseline(&y_true, &y_pred)?;
            Ok(())
        }

        fn reset_baseline_and_decision_threshold(
            &mut self,
            y_true: Vec<f32>,
            y_pred: Vec<f32>,
            threshold: f32,
        ) -> PyResult<()> {
            self.inner
                .reset_baseline_and_decision_threshold(&y_true, &y_pred, threshold)?;
            Ok(())
        }
    }
}

#[derive(Default)]
pub(crate) struct RSquaredSupplement {
    sum_y_true2: f64,       // sum of y true ^ 2 across all examples
    sum_y_pred2: f64,       // sum of y pred across all examples
    sum_y_true_y_pred: f64, // sum of y pred ^ 2 across all examples
}

impl RSquaredSupplement {
    /// Using the state bucket members in the type to compute a snapshot R^2.
    #[inline]
    fn snapshot(&self, y_true_sum: f64, n: f64) -> f64 {
        let sse = self.sum_y_true2 - 2.0 * self.sum_y_true_y_pred + self.sum_y_pred2;
        let sst = self.sum_y_true2 - (y_true_sum.powi(2)) / n;

        if sst == 0_f64 {
            return if sse == 0_f64 { 1_f64 } else { 0_f64 };
        }

        return 1_f64 - (sse / sst);
    }

    /// Push a single example. Takes in the true value and the predicted value.
    fn update(&mut self, y_true: f64, y_pred: f64) {
        self.sum_y_true2 += y_true.powi(2);
        self.sum_y_pred2 += y_pred.powi(2);
        self.sum_y_true_y_pred += y_true * y_pred;
    }
}

/// Contianer to hold the error state for a linear regression model. Store all the different error
/// values that are required to compute the different linear regression error metrics with simple
/// arithmetic instructions.
#[derive(Default)]
pub(crate) struct LinearRegressionErrorBuckets {
    pub(crate) squared_error_sum: f64,
    pub(crate) abs_error_sum: f64,
    pub(crate) max_error: f64,
    pub(crate) squared_log_error_sum: f64,
    pub(crate) abs_percent_error_sum: f64,
    pub(crate) y_true_sum: f64,
    pub(crate) len: f64,
    pub(crate) r2: RSquaredSupplement,
}

impl LinearRegressionErrorBuckets {
    /// Accumulate the error buckets with a single example.
    #[inline]
    fn update(&mut self, y_true: f64, y_pred: f64) {
        self.len += 1_f64;
        let error = y_true - y_pred;
        let abs_error = error.abs();

        self.r2.update(y_true, y_pred);
        self.squared_error_sum += error.powi(2);
        self.abs_error_sum += abs_error;
        self.max_error = self.max_error.max(abs_error);
        self.squared_log_error_sum += ((1_f64 + y_true).ln() - (1_f64 + y_pred).ln()).powi(2);
        self.y_true_sum += y_true;
        self.abs_percent_error_sum += (abs_error / y_true).abs();
    }

    /// Compute R^2 from partial state.
    #[inline]
    pub(crate) fn r2_snapshot(&self) -> f64 {
        self.r2.snapshot(self.y_true_sum, self.len)
    }
}

/// Streaming variant of the Linear Regression monitoring tools included in the crate. This follows
/// a similar bucketing algorithm as other streaming implementations in this crate for compact
/// storage of all the information needed to describe the runtime performance. This type provides
/// the ability to accumulate runtime inference examples, compute performance snapshots, compute
/// drift snapshots relative to the baseline dataset, and reset the baseline state through the
/// lifetime of the type instance.
pub struct LinearRegressionStreaming {
    bl: LinearRegressionRuntime,              // Baseline computations
    rt_buckets: LinearRegressionErrorBuckets, // Stream runtime state
}

impl LinearRegressionStreaming {
    /// Construct initial baseline state with a baseline dataset of predictions and ground truth.
    /// This will error in the case where the baseline dataset is empty, or the baseline datasets
    /// has an inconsistent length.
    pub fn new(
        baseline_y_true: &[f32],
        baseline_y_pred: &[f32],
    ) -> ModelPerfResult<LinearRegressionStreaming> {
        if baseline_y_true.len() != baseline_y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        if baseline_y_pred.is_empty() {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let bl = LinearRegressionRuntime::new(baseline_y_true, baseline_y_pred)?;
        let rt_buckets = LinearRegressionErrorBuckets::default();
        Ok(LinearRegressionStreaming { bl, rt_buckets })
    }

    /// Push a batch of runtime data examples. The same error invariants as the constructor
    /// applies.
    pub fn push_batch(&mut self, y_true: &[f32], y_pred: &[f32]) -> ModelPerfResult<()> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        if y_pred.is_empty() {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        for (y_true, y_pred) in zip_iters!(y_true, y_pred) {
            self.push((*y_true).into(), (*y_pred).into())
        }
        Ok(())
    }

    /// Push a single runtime prediction and ground truth pair.
    #[inline]
    pub fn push(&mut self, y_true: f32, y_pred: f32) {
        self.rt_buckets.update(y_true.into(), y_pred.into())
    }

    /// Clear the runtime state that has been accumulated to this point.
    pub fn flush(&mut self) {
        self.rt_buckets = LinearRegressionErrorBuckets::default();
    }

    /// Reset the baseline state with new baseline predictions and ground truth datasets.
    pub fn reset_baseline(
        &mut self,
        baseline_y_true: &[f32],
        baseline_y_pred: &[f32],
    ) -> ModelPerfResult<()> {
        if baseline_y_true.len() != baseline_y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        if baseline_y_pred.is_empty() {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        self.bl = LinearRegressionRuntime::new(baseline_y_true, baseline_y_pred)?;
        Ok(())
    }

    /// Generate a snapshot of performance of the inference examples accumulate since the last
    /// flush. Will error when there is no runtime data accumulated since construction of last
    /// flush or, in other words, when runtime state is empty.
    pub fn performance_snapshot(&self) -> ModelPerfResult<LinearRegressionAnalysisReport> {
        let rt = LinearRegressionRuntime::runtime_from_parts(&self.rt_buckets)?;
        Ok(rt.generate_report())
    }

    /// Compute a point in time snapshot, describing the drift across all built in metrics. Returns
    /// a 'DriftReport<LinearRegressionEvaluationMetric>'. Will error when there is no runtime data
    /// accumulated since construction of last flush or, in other words, when runtime state is empty.
    /// This method returns the absoulte drift from the baseline state.
    pub fn drift_snapshot(&self) -> ModelPerfResult<LinearRegressionDriftSnapshot> {
        let rt = LinearRegressionRuntime::runtime_from_parts(&self.rt_buckets)?;
        let drift_report = rt.runtime_drift_report(&self.bl);
        Ok(drift_report)
    }
    pub fn drift_report(
        &self,
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<LinearRegressionEvaluationMetric>> {
        let rt = LinearRegressionRuntime::runtime_from_parts(&self.rt_buckets)?;
        let drift_report = rt.runtime_drift_report(&self.bl);
        let drift_threshold = drift_threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
        Ok(DriftReport::from_runtime(
            drift_report
                .into_iter()
                .filter(|(_, v)| *v >= drift_threshold)
                .collect(),
        ))
    }

    pub fn drift_report_partial_metrics(
        &self,
        metrics: &[LinearRegressionEvaluationMetric],
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<LinearRegressionEvaluationMetric>> {
        let rt = LinearRegressionRuntime::runtime_from_parts(&self.rt_buckets)?;
        let drift_report = rt.runtime_drift_report(&self.bl);
        let drift_threshold = drift_threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
        Ok(DriftReport::from_runtime(
            drift_report
                .into_iter()
                .filter(|(m, v)| *v >= drift_threshold && metrics.contains(&m))
                .collect(),
        ))
    }
}

/// Streaming variant of Binary Classification monitoring tools offered in the crate. This type
/// allows for flexible types as inference scores to account for non numeric labels that may be
/// applied on inference class labels. To account for this, a positive label is required on type
/// construction to properly determine the positive outcome case. Like other streaming types,
/// storage is compact using a bucketing algorithm to store the information needed to compute the
/// runtime performance and drift with bounded storage space. Runtime data passed into the stream
/// are bound to the type of the label.
pub struct BinaryClassificationStreaming<T>
where
    T: PartialOrd,
{
    label: T,                        // Label to evaluate true/false prediction
    bl: BinaryClassificationRuntime, // Baseline computations
    confusion_rt: ConfusionMatrix,   // Runtime Confusion matrix buckets
}

impl<T> BinaryClassificationStreaming<T>
where
    T: PartialOrd,
{
    /// Construct a new instance with a baseline dataset or predicted labels, ground truth, and a
    /// positive label. The type of the positive label will determine the generic type T for the
    /// type. The constructor will error in the case that the baseline dataset is empty, or the
    /// predicted class dataset does not have the same length as the ground truth class dataset.
    pub fn new(
        positive_label: T,
        baseline_true: &[T],
        baseline_pred: &[T],
    ) -> ModelPerfResult<BinaryClassificationStreaming<T>> {
        if baseline_true.len() != baseline_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        if baseline_true.is_empty() {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let bl = BinaryClassificationRuntime::new(baseline_true, baseline_pred, &positive_label)?;
        let confusion_rt = ConfusionMatrix::default();

        Ok(BinaryClassificationStreaming {
            label: positive_label,
            bl,
            confusion_rt,
        })
    }

    fn len(&self) -> f32 {
        self.confusion_rt.len()
    }

    /// Push a single observed runtime example to the stream.
    #[inline]
    pub fn push(&mut self, y_true: &T, y_pred: &T) {
        let gt_is_true = self.label.eq(y_true);
        let pred_is_true = self.label.eq(y_pred);

        self.confusion_rt.push(ConfusionPushPayload {
            true_gt: gt_is_true,
            true_pred: pred_is_true,
        });
    }

    /// Push a batch prediction and ground truth observed example set into the stream.
    pub fn push_batch(&mut self, runtime_true: &[T], runtime_pred: &[T]) -> ModelPerfResult<()> {
        if runtime_true.len() != runtime_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        for (t, p) in zip_iters!(runtime_true, runtime_pred) {
            self.push(t, p)
        }

        Ok(())
    }

    /// Reset the baseline stream.
    pub fn reset_baseline(
        &mut self,
        baseline_true: &[T],
        baseline_pred: &[T],
    ) -> ModelPerfResult<()> {
        self.bl = BinaryClassificationRuntime::new(baseline_true, baseline_pred, &self.label)?;
        self.flush();
        Ok(())
    }

    /// Generate a point in time drift report, detailing the drift from the baseline state. This
    /// will the drift in the metric magnitude from the baseline state. Thus a negative drift value
    /// for accuracy indicates that the accuracy computed at the snapshot if performing better than
    /// what was computed in the baseline state. This will error when no data has been pushed into
    /// the stream. This method returns the absoulte drift from the baseline state.
    pub fn drift_snapshot(&self) -> ModelPerfResult<BinaryClassificationDriftSnapshot> {
        if self.len() == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let rt = BinaryClassificationRuntime::runtime_from_parts(&self.confusion_rt);
        let report = rt.runtime_drift_report(&self.bl);
        Ok(report)
    }

    pub fn drift_report(
        &self,
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<ClassificationEvaluationMetric>> {
        if self.len() == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let rt = BinaryClassificationRuntime::runtime_from_parts(&self.confusion_rt);
        let report = rt.runtime_drift_report(&self.bl);
        let drift_threshold = drift_threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
        Ok(DriftReport::from_runtime(
            report
                .into_iter()
                .filter(|(_, v)| *v >= drift_threshold)
                .collect(),
        ))
    }
    pub fn drift_report_partial_metrics(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<ClassificationEvaluationMetric>> {
        if self.len() == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let rt = BinaryClassificationRuntime::runtime_from_parts(&self.confusion_rt);
        let report = rt.runtime_drift_report(&self.bl);
        let drift_threshold = drift_threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
        Ok(DriftReport::from_runtime(
            report
                .into_iter()
                .filter(|(m, v)| *v >= drift_threshold && metrics.contains(&m))
                .collect(),
        ))
    }

    /// Performance snapshot of all runtime examples accumulated in the stream irrelevant of the
    /// baseline state. This will error when no data has been pushed into
    /// the stream.

    pub fn performance_snapshot(&self) -> ModelPerfResult<BinaryClassificationAnalysisReport> {
        if self.len() == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let rt = BinaryClassificationRuntime::runtime_from_parts(&self.confusion_rt);
        Ok(rt.generate_report())
    }

    /// Clear the runtime stream.
    pub fn flush(&mut self) {
        self.confusion_rt = ConfusionMatrix::default();
    }
}

/// Small utitity to hold the log loss penalties with some numeric smoothing.
struct LogPenalty {
    p: f32,
    eps: f32, // storing eps for the sake of looking from closer memory
}

impl Default for LogPenalty {
    fn default() -> LogPenalty {
        LogPenalty {
            p: 0_f32,
            eps: get_stability_eps() as f32,
        }
    }
}

impl LogPenalty {
    #[inline]
    fn push(&mut self, gt: f32, mut pred: f32) {
        pred = pred.clamp(self.eps, 1_f32 - self.eps);
        self.p += gt * pred.ln() + (1_f32 - gt) * (1_f32 - pred).ln();
    }

    #[inline]
    fn compute(&self, n: f32) -> f32 {
        return (-1_f32 * self.p) / n;
    }
}

/// Streaming style variant for LogisticRegression models. Like the other streaming variants of the
/// monitors, this type leverages a bucketing algorithm for compact space.
pub struct LogisticRegressionStreaming {
    decision_threshold: f32,       // Logisitic decision threshold
    confusion_rt: ConfusionMatrix, // Runtime confusion matrix buckets of label
    log_penalties: LogPenalty,     // Accumulated runtime log penalties
    bl: LogisticRegressionRuntime, // Baseline computations
}

impl LogisticRegressionStreaming {
    /// Construct a new instance by passing a labeled ground truth dataset, and an inference
    /// dataset containing the associated logistic values. A decision threshold can be optionally passed.
    /// When the threshold is not passed, it will default to 0.5.
    pub fn new(
        y_true: &[f32],
        y_pred: &[f32],
        threshold_opt: Option<f32>,
    ) -> ModelPerfResult<LogisticRegressionStreaming> {
        let decision_threshold = threshold_opt.unwrap_or(0.5_f32);
        let bl = LogisticRegressionRuntime::new(y_true, y_pred, decision_threshold)?;
        let confusion_rt = ConfusionMatrix::default();
        let log_penalties = LogPenalty::default();
        Ok(LogisticRegressionStreaming {
            bl,
            confusion_rt,
            log_penalties,
            decision_threshold,
        })
    }

    fn len(&self) -> f32 {
        self.confusion_rt.len()
    }

    /// Push a single record into the stream and update runtime stream state.
    #[inline]
    pub fn push(&mut self, gt: f32, pred: f32) {
        self.log_penalties.push(gt, pred);
        let true_gt = gt == 1_f32;
        let true_pred = pred >= self.decision_threshold;

        self.confusion_rt
            .push(ConfusionPushPayload { true_gt, true_pred });
    }

    /// Push records into the stream from a batched dataset.
    pub fn push_batch(&mut self, y_true: &[f32], y_pred: &[f32]) -> ModelPerfResult<()> {
        if y_true.len() != y_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        if y_true.is_empty() {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        for (gt, p) in zip_iters!(y_true, y_pred) {
            self.push(*gt, *p);
        }
        Ok(())
    }

    /// Clear and reset the runtime state.
    pub fn flush(&mut self) {
        self.confusion_rt = ConfusionMatrix::default();
        self.log_penalties = LogPenalty::default();
    }

    /// Compute a snapshot drift report. The drift scalar values are compute by taking the
    /// difference in baseline metric score and current runtime snapshot score. Thus for instance
    /// an accuracy drift that is negative will indicate the model if performing above expected
    /// value, by the absolute value of the drift score, indicating positive performance. In this
    /// sense, log loss would be the inverse. This will error when no data has been pushed into the
    /// stream. This method returns the absoulte drift from the baseline state.
    pub fn drift_snapshot(&self) -> ModelPerfResult<LogisticRegressionDriftSnapshot> {
        // compute log loss
        let n = self.len() as f32;
        if n == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let log_loss = self.log_penalties.compute(n);
        let rt = LogisticRegressionRuntime::runtime_from_parts(&self.confusion_rt, log_loss)?;
        let report = rt.runtime_drift_report(&self.bl);
        Ok(report)
    }
    pub fn drift_report(
        &self,
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<ClassificationEvaluationMetric>> {
        // compute log loss
        let n = self.len() as f32;
        if n == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let log_loss = self.log_penalties.compute(n);
        let rt = LogisticRegressionRuntime::runtime_from_parts(&self.confusion_rt, log_loss)?;
        let report = rt.runtime_drift_report(&self.bl);
        let drift_threshold = drift_threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
        Ok(DriftReport::from_runtime(
            report
                .into_iter()
                .filter(|(_, v)| *v >= drift_threshold)
                .collect(),
        ))
    }
    pub fn drift_report_partial_metrics(
        &self,
        metrics: &[ClassificationEvaluationMetric],
        drift_threshold_opt: Option<f32>,
    ) -> ModelPerfResult<DriftReport<ClassificationEvaluationMetric>> {
        // compute log loss
        let n = self.len() as f32;
        if n == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let log_loss = self.log_penalties.compute(n);
        let rt = LogisticRegressionRuntime::runtime_from_parts(&self.confusion_rt, log_loss)?;
        let report = rt.runtime_drift_report(&self.bl);
        let drift_threshold = drift_threshold_opt.unwrap_or(DEFAULT_DRIFT_THRESHOLD);
        Ok(DriftReport::from_runtime(
            report
                .into_iter()
                .filter(|(m, v)| *v >= drift_threshold && metrics.contains(&m))
                .collect(),
        ))
    }
    /// Compute a snapshot of runtime model performance accumulated in the stream, irrelevant of
    /// the baseline state. This will error when no data has been pushed into the stream.
    pub fn performance_snapshot(&self) -> ModelPerfResult<LogisticRegressionAnalysisReport> {
        let n = self.len() as f32;
        if n == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let log_loss = self.log_penalties.compute(n);
        let rt = LogisticRegressionRuntime::runtime_from_parts(&self.confusion_rt, log_loss)?;
        Ok(rt.generate_report())
    }

    /// Reset the baseline state in the stream with new baseline examples. This will leverage the same
    /// decision threshold. To also update the decision threshold, use `reset_baseline_and_decision_threshold`.
    pub fn reset_baseline(&mut self, y_true: &[f32], y_pred: &[f32]) -> ModelPerfResult<()> {
        self.bl = LogisticRegressionRuntime::new(&y_true, &y_pred, self.decision_threshold)?;
        Ok(())
    }

    /// Reset the baseline state with new baseline examples, and also update the decision
    /// threshold. Useful when a change is made to the decision threshold dynamically at runtime.
    /// This is the only place where the decision threshold can be updated, and it requires a reset
    /// of the baseline to maintain consistency.
    pub fn reset_baseline_and_decision_threshold(
        &mut self,
        y_true: &[f32],
        y_pred: &[f32],
        threshold: f32,
    ) -> ModelPerfResult<()> {
        self.decision_threshold = threshold;
        self.bl = LogisticRegressionRuntime::new(&y_true, &y_pred, self.decision_threshold)?;
        Ok(())
    }
}

#[cfg(test)]
mod analysis_report_test_containers {
    use crate::metrics::LinearRegressionEvaluationMetric as LRM;
    use crate::runtime::EQUALITY_ERROR_ALLOWANCE;
    #[derive(Debug)]
    pub struct TestLinearRegressionReport(
        pub(super) crate::reporting::LinearRegressionAnalysisReport,
    );

    impl PartialEq for TestLinearRegressionReport {
        fn eq(&self, other: &Self) -> bool {
            if (self.0.get(&LRM::RootMeanSquaredError).unwrap()
                - other.0.get(&LRM::RootMeanSquaredError).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&LRM::MeanSquaredError).unwrap()
                - other.0.get(&LRM::MeanSquaredError).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&LRM::MeanAbsoluteError).unwrap()
                - other.0.get(&LRM::MeanAbsoluteError).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&LRM::RSquared).unwrap() - other.0.get(&LRM::RSquared).unwrap()).abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&LRM::MaxError).unwrap() - other.0.get(&LRM::MaxError).unwrap()).abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&LRM::MeanSquaredLogError).unwrap()
                - other.0.get(&LRM::MeanSquaredLogError).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&LRM::RootMeanSquaredLogError).unwrap()
                - other.0.get(&LRM::RootMeanSquaredLogError).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&LRM::MeanAbsolutePercentageError).unwrap()
                - other.0.get(&LRM::MeanAbsolutePercentageError).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            true
        }
    }
    use crate::metrics::ClassificationEvaluationMetric as CM;

    #[derive(Debug)]
    pub(super) struct TestBinaryClassificationAnalysisReport(
        pub(super) crate::reporting::BinaryClassificationAnalysisReport,
    );

    impl PartialEq for TestBinaryClassificationAnalysisReport {
        fn eq(&self, other: &Self) -> bool {
            if (self.0.get(&CM::BalancedAccuracy).unwrap()
                - other.0.get(&CM::BalancedAccuracy).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::PrecisionPositive).unwrap()
                - other.0.get(&CM::PrecisionPositive).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::PrecisionNegative).unwrap()
                - other.0.get(&CM::PrecisionNegative).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::RecallPositive).unwrap()
                - other.0.get(&CM::RecallPositive).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::RecallNegative).unwrap()
                - other.0.get(&CM::RecallNegative).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::Accuracy).unwrap() - other.0.get(&CM::Accuracy).unwrap()).abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::F1Score).unwrap() - other.0.get(&CM::F1Score).unwrap()).abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::BalancedAccuracy).unwrap()
                - other.0.get(&CM::BalancedAccuracy).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            true
        }
    }
    #[derive(Debug)]
    pub(super) struct TestLogisticRegressionAnalysisReport(
        pub(super) crate::reporting::LogisticRegressionAnalysisReport,
    );

    impl PartialEq for TestLogisticRegressionAnalysisReport {
        fn eq(&self, other: &Self) -> bool {
            if (self.0.get(&CM::BalancedAccuracy).unwrap()
                - other.0.get(&CM::BalancedAccuracy).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::PrecisionPositive).unwrap()
                - other.0.get(&CM::PrecisionPositive).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::PrecisionNegative).unwrap()
                - other.0.get(&CM::PrecisionNegative).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::RecallPositive).unwrap()
                - other.0.get(&CM::RecallPositive).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::RecallNegative).unwrap()
                - other.0.get(&CM::RecallNegative).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::Accuracy).unwrap() - other.0.get(&CM::Accuracy).unwrap()).abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::F1Score).unwrap() - other.0.get(&CM::F1Score).unwrap()).abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::BalancedAccuracy).unwrap()
                - other.0.get(&CM::BalancedAccuracy).unwrap())
            .abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            if (self.0.get(&CM::LogLoss).unwrap() - other.0.get(&CM::LogLoss).unwrap()).abs()
                > EQUALITY_ERROR_ALLOWANCE
            {
                return false;
            }
            true
        }
    }
}

#[cfg(test)]
mod test_perf_streaming {
    use super::*;
    use crate::runtime::{BinaryClassificationRuntime, LinearRegressionRuntime};
    use analysis_report_test_containers::{
        TestBinaryClassificationAnalysisReport, TestLinearRegressionReport,
        TestLogisticRegressionAnalysisReport,
    };

    /*
     * 1. Test the accumulation of the error buckets from LinearRegressionErrorBuckets/
     *    RSquaredSupplement
     * 2. Test accumulate of the BinaryClassificationAccuracyBuckets
     * 3. Test push batch records into each stream, this will implicitly test the push single
     *    implementation.
     *
     * */

    #[test]
    fn test_linear_regression_baseline() {
        /*
         * {
         *       'rmse': 0.7781745019952505,
         *       'mse': 0.6055555555555561,
         *       'msle': 0.0030593723018136412,
         *       'rmsle': 0.055311592833814156,
         *       'r2': 0.7435160008366448,
         *       'mape': 0.053999057284547014,
         *       'max_error': 1.3000000000000007,
         *       'mae': 0.7000000000000003
         *   }
         * */
        let y_pred = vec![11.1, 12.2, 13.4, 10.7, 15.8, 16.3, 14.5, 12.3, 11.0];
        let y_true = vec![11.0, 12.5, 14.0, 11.7, 15.1, 15.4, 13.2, 11.5, 11.6];

        let true_bl = LinearRegressionRuntime {
            rmse: 0.7781745019952505,
            mse: 0.6055555555555561,
            msle: 0.0030593723018136412,
            rmsle: 0.055311592833814156,
            r_squared: 0.7435160008366448,
            mape: 0.053999057284547014,
            max_error: 1.3000000000000007,
            mae: 0.7000000000000003,
        };
        let streaming = LinearRegressionStreaming::new(&y_true, &y_pred).unwrap();
        assert!((streaming.bl.rmse - true_bl.rmse).abs() < 1e5_f32);
        assert!((streaming.bl.mse - true_bl.mse).abs() < 1e5_f32);
        assert!((streaming.bl.mae - true_bl.mae).abs() < 1e5_f32);
        assert!((streaming.bl.r_squared - true_bl.r_squared).abs() < 1e5_f32);
        assert!((streaming.bl.max_error - true_bl.max_error).abs() < 1e5_f32);
        assert!((streaming.bl.msle - true_bl.msle).abs() < 1e5_f32);
        assert!((streaming.bl.rmsle - true_bl.rmsle).abs() < 1e5_f32);
        assert!((streaming.bl.mape - true_bl.mape).abs() < 1e5_f32);
    }

    #[test]
    fn test_linear_regression_streaming_accumulation() {
        let y_pred = vec![11.1, 12.2, 13.4, 10.7, 15.8, 16.3, 14.5, 12.3, 11.0];
        let y_true = vec![11.0, 12.5, 14.0, 11.7, 15.1, 15.4, 13.2, 11.5, 11.6];

        let true_bl = LinearRegressionRuntime {
            rmse: 0.7781745019952505,
            mse: 0.6055555555555561,
            msle: 0.0030593723018136412,
            rmsle: 0.055311592833814156,
            r_squared: 0.7435160008366448,
            mape: 0.053999057284547014,
            max_error: 1.3000000000000007,
            mae: 0.7000000000000003,
        };

        let mut streaming = LinearRegressionStreaming::new(&y_true, &y_pred).unwrap();
        streaming.push_batch(&y_true, &y_pred).unwrap();
        let base = TestLinearRegressionReport(true_bl.generate_report());
        let test = TestLinearRegressionReport(streaming.performance_snapshot().unwrap());
        assert_eq!(base, test);
    }

    #[test]
    fn test_classification_baseline() {
        /*
         * {
         *       'precision': 0.75,
         *       'recall': 0.6666666666666666,
         *       'f1': 0.7058823529411765,
         *       'accuracy': 0.6875,
         *       'balanced_accuracy': 0.6904761904761905
         *   }
         * */

        let y_pred = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let y_true = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1];
        let streaming = super::BinaryClassificationStreaming::new(1, &y_true, &y_pred);
        let true_bl = BinaryClassificationRuntime {
            balanced_accuracy: 0.690476190,
            precision_positive: 0.75,
            precision_negative: 0.625,
            recall_positive: 0.666666,
            recall_negative: 0.71428,
            accuracy: 0.6875,
            f1_score: 0.70588235,
        };
        assert_eq!(true_bl, streaming.unwrap().bl);
    }

    #[test]
    fn test_binary_classification_streaming_accumulation() {
        let y_pred = [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let y_true = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1];
        let mut streaming = super::BinaryClassificationStreaming::new(1, &y_true, &y_pred).unwrap();
        streaming.push_batch(&y_true, &y_pred).unwrap();
        let true_bl = BinaryClassificationRuntime {
            balanced_accuracy: 0.690476190,
            precision_positive: 0.75,
            precision_negative: 0.625,
            recall_positive: 0.666666,
            recall_negative: 0.71428,
            accuracy: 0.6875,
            f1_score: 0.70588235,
        };
        let base = TestBinaryClassificationAnalysisReport(true_bl.generate_report());
        let test =
            TestBinaryClassificationAnalysisReport(streaming.performance_snapshot().unwrap());
        assert_eq!(base, test);
    }

    #[test]
    fn test_logistic_regression_streaming_baseline() {
        let y_pred = [
            0.7, 0.3, 0.65, 0.55, 0.1, 0.2, 0.25, 0.66, 0.12, 0.98, 0.23, 0.34, 0.67, 0.77, 0.45,
            0.88,
        ];
        let y_true = [
            0_f32, 0_f32, 1_f32, 1_f32, 1_f32, 0_f32, 0_f32, 1_f32, 1_f32, 1_f32, 0_f32, 0_f32,
            1_f32, 0_f32, 1_f32, 1_f32,
        ];
        let streaming =
            super::LogisticRegressionStreaming::new(&y_true, &y_pred, Some(0.5_f32)).unwrap();
        let true_bl = LogisticRegressionRuntime {
            balanced_accuracy: 0.690476190,
            precision_positive: 0.75,
            precision_negative: 0.625,
            recall_positive: 0.666666,
            recall_negative: 0.71428,
            accuracy: 0.6875,
            f1_score: 0.70588235,
            log_loss: 0.7145021801144907,
        };
        assert_eq!(true_bl, streaming.bl);
    }

    #[test]
    fn test_logistic_regression_accumulation() {
        let y_pred = [
            0.7, 0.3, 0.65, 0.55, 0.1, 0.2, 0.25, 0.66, 0.12, 0.98, 0.23, 0.34, 0.67, 0.77, 0.45,
            0.88,
        ];
        let y_true = [
            0_f32, 0_f32, 1_f32, 1_f32, 1_f32, 0_f32, 0_f32, 1_f32, 1_f32, 1_f32, 0_f32, 0_f32,
            1_f32, 0_f32, 1_f32, 1_f32,
        ];
        let mut streaming =
            super::LogisticRegressionStreaming::new(&y_true, &y_pred, Some(0.5_f32)).unwrap();
        let true_bl = LogisticRegressionRuntime {
            balanced_accuracy: 0.690476190,
            precision_positive: 0.75,
            precision_negative: 0.625,
            recall_positive: 0.666666,
            recall_negative: 0.71428,
            accuracy: 0.6875,
            f1_score: 0.70588235,
            log_loss: 0.7145021801144907,
        };

        streaming.push_batch(&y_true, &y_pred).unwrap();
        let base = TestLogisticRegressionAnalysisReport(true_bl.generate_report());
        let test = TestLogisticRegressionAnalysisReport(streaming.performance_snapshot().unwrap());
        assert_eq!(base, test);
    }
}
