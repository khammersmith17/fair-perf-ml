use crate::{
    data_handler::ConfusionMatrix,
    errors::ModelPerformanceError,
    metrics::{ClassificationEvaluationMetric, LinearRegressionEvaluationMetric},
    reporting::{DriftReport, LinearRegressionAnalysisReport},
    runtime::{BinaryClassificationRuntime, LinearRegressionRuntime},
    zip_iters,
};

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
    ) -> Result<LinearRegressionStreaming, ModelPerformanceError> {
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
    pub fn push_batch(
        &mut self,
        y_true: &[f32],
        y_pred: &[f32],
    ) -> Result<(), ModelPerformanceError> {
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
    ) -> Result<(), ModelPerformanceError> {
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
    pub fn performance_snapshot(
        &self,
    ) -> Result<LinearRegressionAnalysisReport, ModelPerformanceError> {
        let rt = LinearRegressionRuntime::from_parts(&self.rt_buckets)?;
        Ok(rt.generate_report())
    }

    /// Compute a point in time snapshot, describing the drift across all built in metrics. Returns
    /// a 'DriftReport<LinearRegressionEvaluationMetric>'. Will error when there is no runtime data
    /// accumulated since construction of last flush or, in other words, when runtime state is empty.
    pub fn compute_drft(
        &self,
    ) -> Result<DriftReport<LinearRegressionEvaluationMetric>, ModelPerformanceError> {
        let rt = LinearRegressionRuntime::from_parts(&self.rt_buckets)?;
        let drift_report = rt.runtime_drift_report(&self.bl);
        Ok(DriftReport::from_runtime(drift_report))
    }
}
pub struct LogisticRegressionStreaming {}

// Buckets for cheap storage and computation of classication inference scores,
// This is essentially just a named tuple.
#[derive(Default)]
struct BinaryClassificationAccuracyBucket {
    len: u64,
    true_preds: u64,
}

impl BinaryClassificationAccuracyBucket {
    fn snapshot(&self) -> f32 {
        self.true_preds as f32 / self.len as f32
    }
}

/// Streaming variant of Binary Classification monitoring tools offered in the crate. This type
/// allows for flexible types as inference scores to account for non numeric labels that may be
/// applied on inference class labels. To account for this, a positive label is required on type
/// construction to properly determine the positive outcome case. Like other streaming types,
/// storage is compact using a bucketing algorithm to store the information needed to compute the
/// runtime performance and drift with bounded storage space.
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
    ) -> Result<BinaryClassificationStreaming<T>, ModelPerformanceError> {
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

    pub fn push(
        &mut self,
        runtime_true: &[T],
        runtime_pred: &[T],
    ) -> Result<(), ModelPerformanceError> {
        if runtime_true.len() != runtime_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }

        self.accuracy_rt.len += runtime_pred.len() as u64;

        for (t, p) in zip_iters!(runtime_true, runtime_pred) {
            let gt_is_true = *t == self.label;
            let pred_is_true = *p == self.label;
            let true_pred = gt_is_true == pred_is_true;

            self.accuracy_rt.true_preds += true_pred as u64;

            self.confusion_rt.true_p = ((gt_is_true && true_pred) as usize) as f32;
            self.confusion_rt.false_p = ((gt_is_true && !true_pred) as usize) as f32;
            self.confusion_rt.true_n = ((!gt_is_true && true_pred) as usize) as f32;
            self.confusion_rt.false_n = ((!gt_is_true && !true_pred) as usize) as f32;
        }

        Ok(())
    }

    pub fn reset_baseline(
        &mut self,
        baseline_true: &[T],
        baseline_pred: &[T],
    ) -> Result<(), ModelPerformanceError> {
        self.bl =
            BinaryClassificationRuntime::new(baseline_true, baseline_pred, self.label.clone())?;
        self.flush();
        Ok(())
    }

    pub fn drift_report(
        &self,
    ) -> Result<DriftReport<ClassificationEvaluationMetric>, ModelPerformanceError> {
        if self.accuracy_rt.len == 0 {
            return Err(ModelPerformanceError::EmptyDataVector);
        }

        let rt = BinaryClassificationRuntime::from_confusion_and_accuracy(
            &self.confusion_rt,
            self.accuracy_rt.snapshot(),
        );
        let report = rt.runtime_drift_report(&self.bl);
        Ok(DriftReport::from_runtime(report))
    }

    pub fn flush(&mut self) {
        self.confusion_rt = ConfusionMatrix::default();
        self.accuracy_rt = BinaryClassificationAccuracyBucket::default();
    }
}
