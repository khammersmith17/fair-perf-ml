use crate::{
    data_handler::ConfusionMatrix,
    errors::{ModelPerfResult, ModelPerformanceError},
    metrics::{ClassificationEvaluationMetric, LinearRegressionEvaluationMetric},
    reporting::{
        BinaryClassificationAnalysisReport, DriftReport, LinearRegressionAnalysisReport,
        LogisticRegressionAnalysisReport,
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
    use pyo3::prelude::*;
    use pyo3::types::IntoPyDict;

    // All types here are simply logic wrappers around core types, simply to expose the apis to
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
            let inner = BinaryClassificationStreaming::new(1_i32, &y_true, &y_pred)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(PyBinaryClassificationStreaming { inner })
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn push_batch(&mut self, y_true: Vec<i32>, y_pred: Vec<i32>) -> PyResult<()> {
            self.inner
                .push_batch(&y_true, &y_pred)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn push(&mut self, y_true: i32, y_pred: i32) {
            self.inner.push(&y_true, &y_pred)
        }

        fn reset_baseline(&mut self, y_true: Vec<i32>, y_pred: Vec<i32>) -> PyResult<()> {
            self.inner
                .reset_baseline(&y_true, &y_pred)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn performance_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self
                .inner
                .performance_snapshot()
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(perf_report_to_py_dict(py, report))
        }

        fn drift_report<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self
                .inner
                .drift_snapshot()
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
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
            let inner = LinearRegressionStreaming::new(&y_true, &y_pred)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;

            Ok(PyLinearRegressionStreaming { inner })
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn push(&mut self, t: f32, p: f32) {
            self.inner.push(t, p)
        }

        fn push_batch(&mut self, y_true: Vec<f32>, y_pred: Vec<f32>) -> PyResult<()> {
            self.inner
                .push_batch(&y_true, &y_pred)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn reset_baseline(&mut self, y_true: Vec<f32>, y_pred: Vec<f32>) -> PyResult<()> {
            self.inner
                .reset_baseline(&y_true, &y_pred)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn performance_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self
                .inner
                .performance_snapshot()
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;

            Ok(perf_report_to_py_dict(py, report))
        }

        fn drift_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self
                .inner
                .drift_snapshot()
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
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
            let inner = LogisticRegressionStreaming::new(&y_true, &y_pred, Some(threshold))
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;

            Ok(PyLogisticRegressionStreaming { inner })
        }

        fn push(&mut self, y_true: f32, y_pred: f32) {
            self.inner.push(y_true, y_pred);
        }

        fn push_batch(&mut self, y_true: Vec<f32>, y_pred: Vec<f32>) -> PyResult<()> {
            self.inner
                .push_batch(&y_true, &y_pred)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn performance_snapshot<'py>(&self, py: Python<'py>) -> PyDictResult<'py> {
            let report = self
                .inner
                .performance_snapshot()
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(perf_report_to_py_dict(py, report))
        }

        fn drift_snapshot<'py>(&mut self, py: Python<'py>) -> PyDictResult<'py> {
            let drift_report = self
                .inner
                .drift_snapshot()
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;

            Ok(drift_report.into_py_dict(py)?)
        }

        fn flush(&mut self) {
            self.inner.flush()
        }

        fn reset_baseline(&mut self, y_true: Vec<f32>, y_pred: Vec<f32>) -> PyResult<()> {
            self.inner
                .reset_baseline(&y_true, &y_pred)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }

        fn reset_baseline_and_decision_threshold(
            &mut self,
            y_true: Vec<f32>,
            y_pred: Vec<f32>,
            threshold: f32,
        ) -> PyResult<()> {
            self.inner
                .reset_baseline_and_decision_threshold(&y_true, &y_pred, threshold)
                .map_err(|e| <ModelPerformanceError as Into<PyErr>>::into(e))?;
            Ok(())
        }
    }
}

#[derive(Default)]
pub(crate) struct RSquaredSupplement {
    sum_y_true2: f64, // sum of y true ^ 2 across all examples
    sum_y_pred: f64,  // sum of y pred across all examples
    sum_y_pred2: f64, // sum of y pred ^ 2 across all examples
}

impl RSquaredSupplement {
    fn snapshot(&self, y_true_sum: f64, n: f64) -> f64 {
        let sse = self.sum_y_true2 - 2_f64 * self.sum_y_pred + self.sum_y_pred2;
        let sst = self.sum_y_true2 - (y_true_sum.powi(2)) / n;

        return 1_f64 - (sse / sst);
    }

    fn update(&mut self, t: f64, p: f64) {
        self.sum_y_true2 += t.powi(2);
        self.sum_y_pred += p;
        self.sum_y_pred2 += p.powi(2);
    }
}

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
    // accumulate the error buckets with a single example.
    #[inline]
    fn update(&mut self, t: f64, p: f64) {
        self.len += 1_f64;
        let error = t - p;
        let abs_error = error.abs();
        self.r2.update(t, p);
        self.squared_error_sum += error.powi(2);
        self.abs_error_sum += abs_error;
        self.max_error = self.max_error.max(error);
        self.squared_log_error_sum += ((1_f64 + t).log10() - (1_f64 + p).log10()).powi(2);
        self.y_true_sum += t;
        self.abs_percent_error_sum = abs_error / t;
    }

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
    bl: LinearRegressionRuntime,
    rt_buckets: LinearRegressionErrorBuckets,
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
        for (t, p) in zip_iters!(y_true, y_pred) {
            self.push((*t).into(), (*p).into())
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
    pub fn drift_snapshot(&self) -> ModelPerfResult<DriftReport<LinearRegressionEvaluationMetric>> {
        let rt = LinearRegressionRuntime::runtime_from_parts(&self.rt_buckets)?;
        let drift_report = rt.runtime_drift_report(&self.bl);
        Ok(DriftReport::from_runtime(drift_report))
    }
}

// Buckets for cheap storage and computation of classication inference scores,
// This is essentially just a named tuple.
#[derive(Default)]
struct BinaryClassificationAccuracyBucket {
    len: u64,
    true_preds: u64,
}

impl BinaryClassificationAccuracyBucket {
    // Point in time snapshot
    fn snapshot(&self) -> f32 {
        self.true_preds as f32 / self.len as f32
    }

    #[inline]
    fn push(&mut self, true_pred: bool) {
        self.true_preds += true_pred as u64;
        self.len += 1;
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
    T: PartialOrd + Clone,
{
    label: T,
    bl: BinaryClassificationRuntime,
    confusion_rt: ConfusionMatrix,
    accuracy_rt: BinaryClassificationAccuracyBucket,
}

impl<T> BinaryClassificationStreaming<T>
where
    T: PartialOrd + Clone,
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

        let bl =
            BinaryClassificationRuntime::new(baseline_true, baseline_pred, positive_label.clone())?;
        let confusion_rt = ConfusionMatrix::default();
        let accuracy_rt = BinaryClassificationAccuracyBucket::default();

        Ok(BinaryClassificationStreaming {
            label: positive_label,
            bl,
            confusion_rt,
            accuracy_rt,
        })
    }

    /// Push a single observed runtime example to the stream.
    #[inline]
    pub fn push(&mut self, y_true: &T, y_pred: &T) {
        let gt_is_true = self.label.eq(y_true);
        let pred_is_true = self.label.eq(y_pred);
        let true_pred = gt_is_true == pred_is_true;

        self.accuracy_rt.push(gt_is_true == pred_is_true);
        self.confusion_rt.push(gt_is_true, true_pred);
    }

    /// Push a batch prediction and ground truth observed example set into the stream.
    pub fn push_batch(&mut self, runtime_true: &[T], runtime_pred: &[T]) -> ModelPerfResult<()> {
        if runtime_true.len() != runtime_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        self.accuracy_rt.len += runtime_pred.len() as u64;

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
        self.bl =
            BinaryClassificationRuntime::new(baseline_true, baseline_pred, self.label.clone())?;
        self.flush();
        Ok(())
    }

    /// Generate a point in time drift report, detailing the drift from the baseline state. This
    /// will the drift in the metric magnitude from the baseline state. Thus a negative drift value
    /// for accuracy indicates that the accuracy computed at the snapshot if performing better than
    /// what was computed in the baseline state. This will error when no data has been pushed into
    /// the stream.
    pub fn drift_snapshot(&self) -> ModelPerfResult<DriftReport<ClassificationEvaluationMetric>> {
        if self.accuracy_rt.len == 0 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let rt = BinaryClassificationRuntime::runtime_from_parts(
            &self.confusion_rt,
            self.accuracy_rt.snapshot(),
        );
        let report = rt.runtime_drift_report(&self.bl);
        Ok(DriftReport::from_runtime(report))
    }

    /// Performance snapshot of all runtime examples accumulated in the stream irrelevant of the
    /// baseline state. This will error when no data has been pushed into
    /// the stream.

    pub fn performance_snapshot(&self) -> ModelPerfResult<BinaryClassificationAnalysisReport> {
        if self.accuracy_rt.len == 0 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let rt = BinaryClassificationRuntime::runtime_from_parts(
            &self.confusion_rt,
            self.accuracy_rt.snapshot(),
        );
        Ok(rt.generate_report())
    }

    /// Clear the runtime stream.
    pub fn flush(&mut self) {
        self.confusion_rt = ConfusionMatrix::default();
        self.accuracy_rt = BinaryClassificationAccuracyBucket::default();
    }
}

/// Streaming style variant for LogisticRegression models. Like the other streaming variants of the
/// monitors, this type leverages a bucketing algorithm for compact space.
pub struct LogisticRegressionStreaming {
    decision_threshold: f32,
    accuracy_bucket: BinaryClassificationAccuracyBucket,
    confusion_rt: ConfusionMatrix,
    log_penalties: f32,
    bl: LogisticRegressionRuntime,
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
        let log_penalties = 0_f32;
        let accuracy_bucket = BinaryClassificationAccuracyBucket::default();
        Ok(LogisticRegressionStreaming {
            bl,
            confusion_rt,
            log_penalties,
            decision_threshold,
            accuracy_bucket,
        })
    }

    /// Push a single record into the stream and update runtime stream state.
    #[inline]
    pub fn push(&mut self, gt: f32, pred: f32) {
        self.log_penalties += gt * f32::log10(pred) + (1_f32 - gt) * f32::log10(1_f32 - pred);

        let true_gt = gt == 1_f32;
        let true_pred = pred >= self.decision_threshold;

        self.confusion_rt.push(true_gt, true_pred);
        self.accuracy_bucket.push(true_gt == true_pred);
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
        self.accuracy_bucket = BinaryClassificationAccuracyBucket::default();
        self.log_penalties = 0_f32;
    }

    /// Compute a snapshot drift report. The drift scalar values are compute by taking the
    /// difference in baseline metric score and current runtime snapshot score. Thus for instance
    /// an accuracy drift that is negative will indicate the model if performing above expected
    /// value, by the absolute value of the drift score, indicating positive performance. In this
    /// sense, log loss would be the inverse. This will error when no data has been pushed into the
    /// stream.
    pub fn drift_snapshot(&self) -> ModelPerfResult<DriftReport<ClassificationEvaluationMetric>> {
        // compute log loss
        let n = self.accuracy_bucket.len as f32;
        if n == 0_f32 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }
        let log_loss = (-1_f32 * self.log_penalties) / n;
        let accuracy = self.accuracy_bucket.snapshot();
        let rt =
            LogisticRegressionRuntime::runtime_from_parts(&self.confusion_rt, accuracy, log_loss)?;
        let report = rt.runtime_drift_report(&self.bl);
        Ok(DriftReport::from_runtime(report))
    }

    /// Compute a snapshot of runtime model performance accumulated in the stream, irrelevant of
    /// the baseline state. This will error when no data has been pushed into the stream.
    pub fn performance_snapshot(&self) -> ModelPerfResult<LogisticRegressionAnalysisReport> {
        let n = self.accuracy_bucket.len;
        if n == 0 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let log_loss = (-1_f32 * self.log_penalties) / n as f32;
        let accuracy = self.accuracy_bucket.snapshot();
        let rt =
            LogisticRegressionRuntime::runtime_from_parts(&self.confusion_rt, accuracy, log_loss)?;
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
