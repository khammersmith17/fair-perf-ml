use crate::{
    data_handler::ConfusionMatrix,
    errors::ModelPerformanceError,
    metrics::ClassificationEvaluationMetric,
    reporting::DriftReport,
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
    pub(crate) fn snapshot(&self, n: f64, y_true_sum: f64) -> f64 {
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
struct LinearRegressionErrorBuckets {
    squared_error_sum: f64,
    abs_error_sum: f64,
    max_error: f64,
    log_error_sum: f64,
    abs_percent_error_sum: f64,
    y_true_sum: f64,
    len: f64,
    r2: RSquaredSupplement,
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
        self.log_error_sum += error.ln();
        self.y_true_sum += t;
        self.abs_percent_error_sum = abs_error / t;
    }
}

pub struct LinearRegressionStreaming {
    bl: LinearRegressionRuntime,
    rt_buckets: LinearRegressionErrorBuckets,
}

impl LinearRegressionStreaming {
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

    pub fn push(&mut self, y_true: &[f32], y_pred: &[f32]) {
        for (t, p) in zip_iters!(y_true, y_pred) {
            self.rt_buckets.update(*t as f64, *p as f64)
        }
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

// this should accpet artibrary label I think
// accepting arbitrary label for now, must implement partial eq
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
    pub fn new(
        label: T,
        baseline_true: &[T],
        baseline_pred: &[T],
    ) -> Result<BinaryClassificationStreaming<T>, ModelPerformanceError> {
        if baseline_true.len() != baseline_pred.len() {
            return Err(ModelPerformanceError::DataVectorLengthMismatch);
        }
        let bl = BinaryClassificationRuntime::new(baseline_true, baseline_pred, label.clone())?;
        let confusion_rt = ConfusionMatrix::default();
        let accuracy_rt = BinaryClassificationAccuracyBucket::default();
        Ok(BinaryClassificationStreaming {
            label,
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
