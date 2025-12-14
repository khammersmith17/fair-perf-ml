use crate::{
    data_handler::ConfusionMatrix, errors::ModelPerformanceError,
    metrics::ClassificationEvaluationMetric, reporting::DriftReport,
    runtime::BinaryClassificationRuntime, zip_iters,
};
pub struct LinearRegressionStreaming {}
pub struct LogisticRegressionStreaming {}

// Buckets for cheap storage and computation of classication inference scores,
// This is essentially just a named tuple.
#[derive(Default)]
struct BinaryClassificationAccuracyBucket {
    len: u64,
    true_preds: u64,
}

impl BinaryClassificationAccuracyBucket {
    fn snapshot(&mut self) -> f32 {
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
        todo!()
    }

    pub fn drift_report(
        &self,
    ) -> Result<DriftReport<ClassificationEvaluationMetric>, ModelPerformanceError> {
        todo!()
    }

    pub fn flush(&mut self) {
        todo!()
    }
}
